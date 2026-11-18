"""codehygiene v0.7.0 — Zero-vocabulary target byte-grid projection and slice renderer.

Projects a flagged `(L x 128)` byte grid `G` onto a `_forward`-verified green grid `G*`
via pure 2D byte-grid operations (zero keywords, ASTs, or English rule templates).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# Block 0 per-cell channel indices synchronized with compile.py (N_HC = 68)
_HC_COMM = 1
_HC_STR = 46
_WINDOW_PAD = 36
_MAX_DIRECT_LINES = 140


@dataclass(frozen=True)
class TargetProjection:
    """Projected green byte-grid state and compact changed-slice rendering."""

    target_score: float
    target_squiggles: int
    target_slice: str


def _row_indent(row: np.ndarray) -> int:
    nonspace = np.flatnonzero((row > 32) & (row < 128))
    return int(nonspace[0]) if len(nonspace) else 0


def _row_len(row: np.ndarray) -> int:
    nonspace = np.flatnonzero(row > 32)
    return int(nonspace[-1]) + 1 if len(nonspace) else 0


def _blank_row(width: int = 128) -> np.ndarray:
    return np.zeros((width,), dtype=np.uint8)


def _synth_row(indent: int, pattern: bytes, width: int = 128) -> np.ndarray:
    out = np.zeros((width,), dtype=np.uint8)
    ind = min(max(0, indent), width - 4)
    out[:ind] = 32
    raw = np.frombuffer(pattern, dtype=np.uint8)
    take = min(len(raw), width - ind)
    out[ind : ind + take] = raw[:take]
    return out


def _shift_indent(row: np.ndarray, new_indent: int) -> np.ndarray:
    rlen = _row_len(row)
    if rlen == 0:
        return _blank_row(len(row))
    cur = _row_indent(row)
    return _synth_row(new_indent, row[cur:rlen].tobytes(), len(row))


def _kind_spans(
    lprobs: np.ndarray, kind_idx: int, thresh: float, max_gap: int = 1
) -> list[tuple[int, int]]:
    if len(lprobs) == 0:
        return []
    best_k = np.argmax(lprobs, axis=1)
    best_p = lprobs[np.arange(len(lprobs)), best_k]
    rows = [int(r) for r in np.flatnonzero((best_k == kind_idx) & (best_p >= thresh))]
    if not rows:
        return []
    spans: list[tuple[int, int]] = []
    s0 = p0 = rows[0]
    for row in rows[1:]:
        if row - p0 <= max_gap:
            p0 = row
        else:
            spans.append((s0, p0))
            s0 = p0 = row
    spans.append((s0, p0))
    return spans


def _extract_grid_features(
    hygiene: Any, params: dict[str, np.ndarray], grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_lines, width = grid.shape
    hc_dim = params["col_fuse_w"].shape[1]
    cells = np.zeros((num_lines, width, hc_dim), dtype=np.float32)
    spans = hygiene._extract_active_spans(grid)
    if spans is not None:
        act = hygiene._fuse_block0_cells(params, grid, spans)
        for idx, row in enumerate(spans.nonblank):
            start = int(spans.line_starts[idx])
            take = min(int(spans.span_lens[idx]), width)
            cells[row, :take] = act[start : start + take]
    _, _, _, raw_sk = hygiene._block01_active_lines(params, grid)
    sk = hygiene._relu(raw_sk) - hygiene._relu(raw_sk - 1.0)
    return cells, hygiene._l2_norm(sk[:, :40]), hygiene._l2_norm(sk[:, 80:120])


def _row_pair_sim(
    grid: np.ndarray, sk_code: np.ndarray, sk_skel: np.ndarray, r_a: int, r_b: int
) -> float:
    la, lb = _row_len(grid[r_a]), _row_len(grid[r_b])
    if la == 0 or lb == 0:
        return 0.0

    ia, ib = _row_indent(grid[r_a]), _row_indent(grid[r_b])
    if ia == 0 or ib == 0:
        return 0.0

    sa = bytes(grid[r_a, ia:la]).strip()
    sb = bytes(grid[r_b, ib:lb]).strip()
    if sa == sb and sa in (b"{", b"}"):
        return 1.0

    c_sim = float(np.dot(sk_code[r_a], sk_code[r_b]))
    k_sim = float(np.dot(sk_skel[r_a], sk_skel[r_b]))
    return max(c_sim, 0.45 * c_sim + 0.55 * k_sim)


def _best_clone_offset(
    grid: np.ndarray, sk_code: np.ndarray, sk_skel: np.ndarray, seed_row: int
) -> int:
    best_off, best_sim = 0, 0.0
    for cand in range(len(grid)):
        if abs(cand - seed_row) >= 4:
            sim = _row_pair_sim(grid, sk_code, sk_skel, seed_row, cand)
            if sim > best_sim:
                best_sim, best_off = sim, cand - seed_row
    return best_off if best_sim >= 0.52 else 0


def _expand_clone_bounds(
    grid: np.ndarray,
    sk_code: np.ndarray,
    sk_skel: np.ndarray,
    seed_row: int,
    offset: int,
) -> tuple[int, int]:
    n_rows = len(grid)
    max_span = abs(offset) - 1
    lo, hi = seed_row, seed_row
    for step in (-1, 1):
        while (hi - lo + 1) < max_span:
            nxt = (lo - 1) if step < 0 else (hi + 1)
            if not (0 <= nxt < n_rows and 0 <= nxt + offset < n_rows):
                break
            if _row_pair_sim(grid, sk_code, sk_skel, nxt, nxt + offset) < 0.50:
                break
            if step < 0:
                lo = nxt
            else:
                hi = nxt
    return lo, hi


def _span_sim(
    grid: np.ndarray, sk_code: np.ndarray, sk_skel: np.ndarray, lo: int, cur: int, slen: int
) -> float:
    total = 0.0
    for k in range(slen):
        total += _row_pair_sim(grid, sk_code, sk_skel, lo + k, cur + k)
    return total / slen


def _grow_clone_peers(
    grid: np.ndarray, sk_code: np.ndarray, sk_skel: np.ndarray, seed_row: int
) -> list[tuple[int, int]]:
    best_off = _best_clone_offset(grid, sk_code, sk_skel, seed_row)
    if best_off == 0:
        return []

    lo, hi = _expand_clone_bounds(grid, sk_code, sk_skel, seed_row, best_off)
    slen = hi - lo + 1
    if slen < 3:
        return []

    peers = [(lo, hi)]
    cur = 0
    while cur <= len(grid) - slen:
        overlaps = any(abs(cur - pa) < slen for pa, _ in peers)
        if not overlaps and _span_sim(grid, sk_code, sk_skel, lo, cur, slen) >= 0.55:
            peers.append((cur, cur + slen - 1))
            cur += slen
        else:
            cur += 1
    return sorted(peers) if len(peers) >= 2 else []


def _apply_clone_factor(
    hygiene: Any,
    params: dict[str, np.ndarray],
    grid: np.ndarray,
    prov: list[tuple[bool, str]],
    lprobs: np.ndarray,
    thresh: float,
) -> tuple[np.ndarray, list[tuple[bool, str]], np.ndarray]:
    dup_spans = _kind_spans(lprobs, 3, thresh, max_gap=2)
    if not dup_spans:
        return grid, prov, lprobs
    _, sk_code, sk_skel = _extract_grid_features(hygiene, params, grid)
    peers = _grow_clone_peers(grid, sk_code, sk_skel, dup_spans[0][0])
    if not peers:
        return grid, prov, lprobs

    span_tag = " & ".join(f"{prov[pa][1]}..{prov[pb][1]}" for pa, pb in peers)
    rows_out: list[np.ndarray] = [_synth_row(0, b"fn f(a, b, c):")]
    prov_out: list[tuple[bool, str]] = [(True, span_tag)]
    base_p0 = _row_indent(grid[peers[0][0]])
    for r_i in range(peers[0][0], peers[0][1] + 1):
        rel_ind = max(4, _row_indent(grid[r_i]) - base_p0 + 4)
        rows_out.append(_shift_indent(grid[r_i], rel_ind))
        prov_out.append((True, prov[r_i][1]))
    rows_out.append(_blank_row())
    prov_out.append((True, ""))

    starts = {pa: (pa, pb) for pa, pb in peers}
    interior = {r_i for pa, pb in peers for r_i in range(pa + 1, pb + 1)}
    for idx in range(len(grid)):
        if idx in starts:
            pa, pb = starts[idx]
            rows_out.append(_synth_row(max(4, _row_indent(grid[pa])), b"v = f(a, b, c)"))
            prov_out.append((True, f"{prov[pa][1]}..{prov[pb][1]}"))
        elif idx not in interior:
            rows_out.append(grid[idx])
            prov_out.append(prov[idx])
    cand = np.stack(rows_out, axis=0)
    lp_c, _ = hygiene._forward(params, cand)
    if int(np.sum(lp_c[:, 3] >= thresh)) < int(np.sum(lprobs[:, 3] >= thresh)):
        return cand, prov_out, lp_c
    return grid, prov, lprobs


def _apply_dense_unpack(
    hygiene: Any,
    params: dict[str, np.ndarray],
    grid: np.ndarray,
    prov: list[tuple[bool, str]],
    lprobs: np.ndarray,
    thresh: float,
) -> tuple[np.ndarray, list[tuple[bool, str]], np.ndarray]:
    spans = _kind_spans(lprobs, 1, thresh)
    if not spans:
        return grid, prov, lprobs

    cells, _, _ = _extract_grid_features(hygiene, params, grid)
    flagged = {r_i for a, b in spans for r_i in range(a, b + 1)}
    rows_out: list[np.ndarray] = []
    prov_out: list[tuple[bool, str]] = []

    for idx in range(len(grid)):
        if idx not in flagged:
            rows_out.append(grid[idx])
            prov_out.append(prov[idx])
            continue

        ind = _row_indent(grid[idx])
        rlen = _row_len(grid[idx])
        semis = [
            col
            for col in range(ind, rlen - 1)
            if grid[idx, col] == 59
            and cells[idx, col, _HC_STR] < 0.5
            and cells[idx, col, _HC_COMM] < 0.5
        ]
        if semis:
            cuts = [ind, *[col + 1 for col in semis], rlen]
            for seg_i in range(len(cuts) - 1):
                seg = bytes(grid[idx, cuts[seg_i] : cuts[seg_i + 1]]).strip()
                if seg:
                    rows_out.append(_synth_row(ind, seg))
                    prov_out.append((True, prov[idx][1]))
        else:
            rows_out.extend([
                _synth_row(ind, b"v = f("),
                _synth_row(ind + 4, b"a=x,"),
                _synth_row(ind + 4, b"b=y,"),
                _synth_row(ind, b")"),
            ])
            prov_out.extend([(True, prov[idx][1]), (True, ""), (True, ""), (True, "")])

    cand = np.stack(rows_out, axis=0)
    lp_c, _ = hygiene._forward(params, cand)
    return cand, prov_out, lp_c


def _apply_ladder(
    hygiene: Any,
    params: dict[str, np.ndarray],
    grid: np.ndarray,
    prov: list[tuple[bool, str]],
    lprobs: np.ndarray,
    thresh: float,
) -> tuple[np.ndarray, list[tuple[bool, str]], np.ndarray]:
    lad_spans = _kind_spans(lprobs, 4, thresh, max_gap=2)
    if not lad_spans:
        return grid, prov, lprobs

    smap = {a: (a, b) for a, b in lad_spans}
    skip = {r_i for a, b in lad_spans for r_i in range(a + 1, b + 1)}
    rows_out, prov_out = [], []

    for idx in range(len(grid)):
        if idx in skip:
            continue
        if idx in smap:
            a, b = smap[idx]
            ind = _row_indent(grid[a])
            tag = f"{prov[a][1]}..{prov[b][1]}"
            rows_out.append(_synth_row(ind, b"m = {"))
            prov_out.append((True, tag))
            for k in range(4):
                rows_out.append(_synth_row(ind + 4, f"k{k}: v{k},".encode()))
                prov_out.append((True, ""))
            rows_out.extend([_synth_row(ind, b"}"), _synth_row(ind, b"r = m[k]")])
            prov_out.extend([(True, ""), (True, tag)])
        else:
            rows_out.append(grid[idx])
            prov_out.append(prov[idx])

    cand = np.stack(rows_out, axis=0)
    lp_c, _ = hygiene._forward(params, cand)
    return cand, prov_out, lp_c


def _is_deep_nonblank(grid: np.ndarray, row_idx: int) -> bool:
    return _row_len(grid[row_idx]) > 0 and _row_indent(grid[row_idx]) >= 8


def _apply_pyramid(
    hygiene: Any,
    params: dict[str, np.ndarray],
    grid: np.ndarray,
    prov: list[tuple[bool, str]],
    lprobs: np.ndarray,
    thresh: float,
) -> tuple[np.ndarray, list[tuple[bool, str]], np.ndarray]:
    nest_spans = _kind_spans(lprobs, 0, thresh, max_gap=2)
    if not nest_spans:
        return grid, prov, lprobs

    a, b = nest_spans[0]
    top, bot = a, b
    while top > 0 and _is_deep_nonblank(grid, top - 1):
        top -= 1
    while bot + 1 < len(grid) and _is_deep_nonblank(grid, bot + 1):
        bot += 1

    base = min(8, max(4, _row_indent(grid[top])))
    guard_tag = f"{prov[top][1]}..{prov[max(top, a - 1)][1]}"
    rows_out, prov_out = [], []

    for idx in range(len(grid)):
        if idx == top:
            rows_out.extend([
                _synth_row(base, b"if !ok: return"),
                _synth_row(base, b"if !rdy: return"),
                _blank_row(),
            ])
            prov_out.extend([(True, guard_tag), (True, guard_tag), (True, "")])
            for core_r in range(a, b + 1):
                rows_out.append(_shift_indent(grid[core_r], base + (core_r - a) * 2))
                prov_out.append((True, prov[core_r][1]))
        elif not (top < idx <= bot):
            rows_out.append(grid[idx])
            prov_out.append(prov[idx])

    cand = np.stack(rows_out, axis=0)
    lp_c, _ = hygiene._forward(params, cand)
    return cand, prov_out, lp_c


def _apply_layout_fixes(
    hygiene: Any,
    params: dict[str, np.ndarray],
    grid: np.ndarray,
    prov: list[tuple[bool, str]],
    lprobs: np.ndarray,
    thresh: float,
) -> tuple[np.ndarray, list[tuple[bool, str]], np.ndarray]:
    for a, b in _kind_spans(lprobs, 2, thresh, max_gap=1):
        cur = _row_indent(grid[a])
        if a == b and cur % 2 == 1:
            best_g, best_sc = grid, -1.0
            for cand_i in sorted({max(0, (cur // 4) * 4), ((cur + 3) // 4) * 4, (cur // 2) * 2}):
                if cand_i == cur:
                    continue
                cand = grid.copy()
                cand[a] = _shift_indent(grid[a], cand_i)
                _, sc_c = hygiene._forward(params, cand)
                if sc_c > best_sc:
                    best_g, best_sc = cand, sc_c
            grid = best_g
            prov[a] = (True, prov[a][1])

    for _ in range(3):
        lprobs, _ = hygiene._forward(params, grid)
        walls = _kind_spans(lprobs, 2, thresh, max_gap=2)
        if not walls:
            break
        wa, wb = walls[0]
        _, sk_code, sk_skel = _extract_grid_features(hygiene, params, grid)
        best_cut, min_cos = (wa + wb) // 2 + 1, 99.0
        for row in range(max(1, wa - 3), min(len(grid) - 2, wb + 3) + 1):
            if _row_len(grid[row]) == 0 or _row_len(grid[row + 1]) == 0:
                continue
            cos = float(
                np.dot(sk_code[row], sk_code[row + 1])
                + np.dot(sk_skel[row], sk_skel[row + 1])
            )
            if cos < min_cos:
                min_cos, best_cut = cos, row + 1
        grid = np.insert(grid, best_cut, _blank_row(), axis=0)
        prov.insert(best_cut, (True, ""))
    lprobs, _ = hygiene._forward(params, grid)
    return grid, prov, lprobs


def _silhouette_row(row: np.ndarray, line_num: int, arrow_src: str) -> str:
    rlen = _row_len(row)
    if rlen == 0:
        return f"  {line_num:3d} |"
    ind = _row_indent(row)
    pad = " " * (ind // 2) + ("·" if ind % 2 == 1 else "")
    glyphs: list[str] = []
    for col in range(ind, min(rlen, 100), 2):
        b0 = int(row[col])
        b1 = int(row[col + 1]) if col + 1 < rlen else 32
        pair = (b0, b1)
        if b0 == 32 and b1 == 32:
            glyphs.append(" ")
        elif any(32 < x < 127 and chr(x) in "{}()[]" for x in pair):
            glyphs.append(next(chr(x) for x in pair if 32 < x < 127 and chr(x) in "{}()[]"))
        elif any(32 < x < 127 and chr(x) in ";:,=" for x in pair):
            glyphs.append(next(chr(x) for x in pair if 32 < x < 127 and chr(x) in ";:,="))
        else:
            glyphs.append("■")
    suffix = f"  <-{arrow_src}" if arrow_src else ""
    return f"  {line_num:3d} | {pad}{''.join(glyphs)}{suffix}"


_STEPS = (
    _apply_clone_factor,
    _apply_dense_unpack,
    _apply_ladder,
    _apply_pyramid,
    _apply_layout_fixes,
)


def _rank(
    hygiene: Any, params: dict[str, np.ndarray], grid: np.ndarray, thresh: float
) -> tuple[int, float]:
    """Order grids by flagged-row count, then by higher document score."""
    lprobs, sc = hygiene._forward(params, grid)
    return int(np.sum(lprobs.max(axis=1) >= thresh)), -float(sc)


def _project_window(
    hygiene: Any,
    params: dict[str, np.ndarray],
    sub_grid: np.ndarray,
    line_offset: int,
    thresh: float,
) -> tuple[np.ndarray, list[tuple[bool, str]]]:
    grid = sub_grid.copy()
    prov: list[tuple[bool, str]] = [(False, str(line_offset + i + 1)) for i in range(len(grid))]
    lprobs, sc0 = hygiene._forward(params, grid)

    comm_spans = _kind_spans(lprobs, 5, thresh)
    if comm_spans:
        drops = {r_i for a, b in comm_spans for r_i in range(a, b + 1)}
        if len(drops) < len(grid):
            cand = np.delete(grid, sorted(drops), axis=0)
            lp_c, sc_c = hygiene._forward(params, cand)
            sq_before = int(np.sum(lprobs.max(axis=1) >= thresh))
            sq_after = int(np.sum(lp_c.max(axis=1) >= thresh))
            if sq_after < sq_before or sc_c > sc0:
                keep_nb = max(0, min(drops) - 1)
                prov = [p for i, p in enumerate(prov) if i not in drops]
                if keep_nb < len(prov):
                    prov[keep_nb] = (True, prov[keep_nb][1])
                grid, lprobs = cand, lp_c

    best_rank = _rank(hygiene, params, grid, thresh)
    for step in _STEPS:
        cand, cand_prov, cand_lp = step(hygiene, params, grid, list(prov), lprobs, thresh)
        cand_rank = _rank(hygiene, params, cand, thresh)
        if cand_rank < best_rank:
            grid, prov, lprobs, best_rank = cand, cand_prov, cand_lp, cand_rank
    return grid, prov


def project_target_grid(
    code: str,
    *,
    threshold: float = 0.55,
    weights: dict[str, np.ndarray] | None = None,
    hygiene: Any = None,
) -> TargetProjection:
    """Project `code`'s flagged byte-grid regions onto a verified green target slice."""
    if hygiene is None:
        import codehygiene as hygiene

    params = weights or hygiene.load_weights()
    full_grid, _ = hygiene.encode_byte_grid(code)
    if len(full_grid) == 0:
        return TargetProjection(target_score=1.0, target_squiggles=0, target_slice="")

    lprobs0, sc0 = hygiene._forward(params, full_grid)
    flagged = np.flatnonzero(lprobs0.max(axis=1) >= threshold)
    if len(flagged) == 0:
        return TargetProjection(target_score=sc0, target_squiggles=0, target_slice="")

    if len(full_grid) <= _MAX_DIRECT_LINES:
        win_lo, win_hi = 0, len(full_grid)
    else:
        win_lo = max(0, int(flagged[0]) - _WINDOW_PAD)
        win_hi = min(len(full_grid), int(flagged[-1]) + _WINDOW_PAD + 1)
        if win_hi - win_lo > _MAX_DIRECT_LINES:
            win_hi = min(len(full_grid), win_lo + _MAX_DIRECT_LINES)

    opt_g, opt_prov = _project_window(
        hygiene, params, full_grid[win_lo:win_hi], win_lo, threshold
    )
    spliced = np.concatenate([full_grid[:win_lo], opt_g, full_grid[win_hi:]], axis=0)
    t_sq, neg_score = _rank(hygiene, params, spliced, threshold)
    t_score = -neg_score

    unchanged = TargetProjection(target_score=sc0, target_squiggles=len(flagged), target_slice="")
    if t_score < sc0 or (t_sq >= len(flagged) and t_score <= sc0 + 1e-4):
        return unchanged

    changed = [i for i, (is_mod, _) in enumerate(opt_prov) if is_mod]
    keep: set[int] = set()
    for idx in changed:
        keep.update(range(max(0, idx - 1), min(len(opt_g), idx + 2)))
    if not keep:
        return unchanged
    selected = sorted(keep)

    lines_out: list[str] = []
    prev_i = -2
    for idx in selected[:24]:
        if prev_i >= 0 and idx > prev_i + 1:
            lines_out.append("    ...")
        is_mod, tag = opt_prov[idx]
        lines_out.append(_silhouette_row(opt_g[idx], win_lo + idx + 1, tag if is_mod else ""))
        prev_i = idx
    return TargetProjection(
        target_score=t_score,
        target_squiggles=t_sq,
        target_slice="\n".join(lines_out),
    )
