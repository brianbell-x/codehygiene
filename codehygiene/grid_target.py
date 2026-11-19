"""codehygiene v0.7.0 — Target byte-grid projection and slice renderer.

Projects a flagged `(L x 128)` byte grid `G` onto a `_forward`-verified green grid `G*`. Every
rewritten row is rebuilt from the file's own rows (their bytes, indents, brackets and code cells);
a few control-flow words keep the file's own block, exit and table syntax. A fix that cannot be
derived from the file is skipped. Nothing here runs inside `_forward`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Block 0 per-cell channel indices synchronized with compile.py (N_HC = 68)
_HC_COMM = 1
_HC_STR = 46
_WINDOW_PAD = 36
_MAX_DIRECT_LINES = 140
_SCOPE_REACH = 200
_SLICE_ROWS = 40

# Byte classes for clone factoring, which reads row bytes outside `_forward`
_WORD = frozenset(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_")
_DIGITS = frozenset(b"0123456789")
_OPEN = frozenset(b"([{")
_CLOSE = frozenset(b")]}")
_QUOTES = frozenset(b"\"'`")
_PREFIX = _WORD | frozenset(b" *&(")
_WORD_BYTES = {bytes([byte]) for byte in _WORD}
_HELPER = b"f"
_RETURN = b"return"

# Control-flow words, read only here (outside `_forward`), so suggested rows keep the file's own
# block, exit and table syntax.
_IF = b"if"
_LET = b"let "
_CONTINUE = b"continue"
_LOOPS = frozenset((b"for", b"while", b"loop", b"foreach", b"do"))
_ELSE = frozenset((b"else", b"elif"))
_CONTINUES = frozenset((b"else", b"elif", b"except", b"catch", b"finally"))
_EXITS = (b"return", b"throw", b"break", b"continue")
_HEADS = frozenset((b"if", b"for", b"while", b"foreach"))
_DEFS = {b"def": "py", b"fn": "rust", b"func": "go", b"function": "js"}
_CONTROL = frozenset(
    (b"if", b"else", b"for", b"while", b"do", b"switch", b"match", b"case", b"catch", b"try")
    + (b"using", b"lock", b"unsafe", b"loop", b"foreach", b"return")
)
_LOGIC = frozenset(
    (b"&&", b"||", b"and", b"or", b"not", b"?", b"??", b":", b"=", b"=>", b"&", b"|", b"^")
    + (b"if", b"else", b"lambda")
)
_SWAP = {b"===": b"!==", b"!==": b"===", b"==": b"!=", b"!=": b"==", b"<=": b">", b">=": b"<", b"<": b">=", b">": b"<="}
_WORD_SWAP = {(b"is", b"not"): b"is", (b"not", b"in"): b"in", (b"is",): b"is not", (b"in",): b"not in"}
_EQUALS = frozenset((b"==", b"==="))



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
) -> tuple[int, list[tuple[int, int]]]:
    """Return the reference clone's first row and every peer span, sorted (one length for all)."""
    best_off = _best_clone_offset(grid, sk_code, sk_skel, seed_row)
    if best_off == 0:
        return 0, []

    lo, hi = _expand_clone_bounds(grid, sk_code, sk_skel, seed_row, best_off)
    slen = hi - lo + 1
    if slen < 3:
        return 0, []

    peers = [(lo, hi)]
    cur = 0
    while cur <= len(grid) - slen:
        overlaps = any(abs(cur - pa) < slen for pa, _ in peers)
        if not overlaps and _span_sim(grid, sk_code, sk_skel, lo, cur, slen) >= 0.55:
            # Take the best-aligned start among the next few rows, not the first one past the bar.
            nearby = [
                c for c in range(cur, min(cur + 3, len(grid) - slen + 1))
                if not any(abs(c - pa) < slen for pa, _ in peers)
            ]
            cur = max(nearby, key=lambda c: _span_sim(grid, sk_code, sk_skel, lo, c, slen))
            peers.append((cur, cur + slen - 1))
            cur += slen
        else:
            cur += 1
    return (lo, sorted(peers)) if len(peers) >= 2 else (0, [])


def _shared_run(
    grid: np.ndarray, sk_code: np.ndarray, sk_skel: np.ndarray, ref: int, start: int, slen: int
) -> int:
    """Count the leading rows of the peer at `start` that match the reference clone row for row."""
    ref_ind, peer_ind = _row_indent(grid[ref]), _row_indent(grid[start])
    for k in range(slen):
        r_a, r_b = ref + k, start + k
        if _row_len(grid[r_a]) == 0 or _row_len(grid[r_b]) == 0:
            return k
        if _row_indent(grid[r_a]) - ref_ind != _row_indent(grid[r_b]) - peer_ind:
            return k
        if r_a != r_b and _row_pair_sim(grid, sk_code, sk_skel, r_a, r_b) < 0.50:
            return k
    return slen


# Clone factoring from the file's own bytes: rows, brackets, names and helper plans.


@dataclass(frozen=True)
class _Lines:
    """Row bytes (trailing blanks cut), code-cell masks (False in strings and comments), rows the
    grid's width cut short, and the full text and mask of clipped rows a step read from the source."""

    text: list[bytes]
    code: np.ndarray
    clipped: np.ndarray
    tails: dict[int, tuple[bytes, np.ndarray]] = field(default_factory=dict)

    @classmethod
    def of(cls, grid: np.ndarray, cells: np.ndarray, tails: dict | None = None) -> _Lines:
        code = (cells[:, :, _HC_STR] <= 0.5) & (cells[:, :, _HC_COMM] <= 0.5)
        return cls([bytes(row[: _row_len(row)]) for row in grid], code, grid[:, -1] != 0, tails or {})

    def indent(self, row: int) -> int:
        return next((col for col, byte in enumerate(self.text[row]) if 32 < byte < 128), 0)

    def blank(self, row: int) -> bool:
        return not self.text[row].strip()

    def opens_block(self, row: int) -> bool:
        return not self.blank(row) and bool(self.code[row][self.indent(row)])

    def next_nonblank(self, row: int) -> int:
        return next((cur for cur in range(row + 1, len(self.text)) if not self.blank(cur)), len(self.text))

    def starts_with(self, row: int, chars: frozenset[int]) -> bool:
        text = self.text[row]
        return bool(text.strip()) and text[self.indent(row)] in chars

    def is_opener(self, row: int) -> bool:
        """True when the next nonblank row is deeper, or is a `{` at the same indent."""
        nxt = self.next_nonblank(row)
        if nxt >= len(self.text) or self.blank(row):
            return False
        return self.indent(nxt) > self.indent(row) or (
            self.indent(nxt) == self.indent(row) and self.text[nxt].strip()[:1] == b"{"
        )


@dataclass(frozen=True)
class _HelperPlan:
    """One shared helper: its rows, where it goes, and the call that replaces each clone."""

    insert_at: int
    rows: list[bytes]
    blank_after: int
    calls: dict[int, tuple[int, list[int], bytes]]


def _depths(text: bytes, code: np.ndarray, depth: int = 0) -> tuple[list[int], int]:
    """Bracket depth before each byte, ignoring strings and comments, and the depth after the row."""
    before: list[int] = []
    for col, byte in enumerate(text):
        before.append(depth)
        if code[col] and byte in _OPEN:
            depth += 1
        elif code[col] and byte in _CLOSE:
            depth -= 1
    return before, depth


def _words(text: bytes, code: np.ndarray, lo: int = 0, hi: int | None = None) -> list[tuple[int, int]]:
    """Name spans in the code cells of text[lo:hi]; numbers and attributes after `.` are skipped."""
    hi = len(text) if hi is None else hi
    spans: list[tuple[int, int]] = []
    col = lo
    while col < hi:
        if text[col] not in _WORD or not code[col]:
            col += 1
            continue
        end = col + 1
        while end < hi and text[end] in _WORD and code[end]:
            end += 1
        if text[col] not in _DIGITS and (col == 0 or text[col - 1] != 46):
            spans.append((col, end))
        col = end
    return spans


def _bound_names(text: bytes, code: np.ndarray, depth_at: list[int], eq_col: int) -> list[tuple[int, int]]:
    """Last name of each depth-0, comma-split target left of `=`, before any `:` annotation."""
    commas = [c for c in range(eq_col) if text[c] == 44 and code[c] and depth_at[c] == 0]
    names: list[tuple[int, int]] = []
    lo = 0
    for cut in (*commas, eq_col):
        colons = (c for c in range(lo, cut) if text[c] == 58 and code[c] and depth_at[c] == 0)
        hi = next(colons, cut)
        spans = [span for span in _words(text, code, lo, hi) if depth_at[span[0]] == 0]
        if spans and not text[spans[-1][1] : hi].strip():
            names.append(spans[-1])
        lo = cut + 1
    return names


def _binding(text: bytes, code: np.ndarray) -> tuple[list[tuple[int, int]], bool]:
    """Name spans bound by the row's first depth-0 `=` or `:=`, and whether it is `:=`."""
    depth_at, _ = _depths(text, code)
    for col, byte in enumerate(text):
        if byte != 61 or not code[col] or depth_at[col] != 0:
            continue
        prev = text[col - 1] if col else 0
        nxt = text[col + 1] if col + 1 < len(text) else 0
        if nxt in b"=>" or prev in b"=!<>":
            continue
        return _bound_names(text, code, depth_at, col), prev == 58
    return [], False


def _keyword_arg(text: bytes, depth_at: list[int], end: int) -> bool:
    """True for a name inside brackets followed by a single `=` (`type=str`): a label, not a read."""
    col = end
    while col < len(text) and text[col] == 32:
        col += 1
    if col >= len(text) or text[col] != 61 or depth_at[col] == 0:
        return False
    return col + 1 >= len(text) or text[col + 1] != 61


def _row_flow(
    lines: _Lines, row: int, skip: tuple[int, int] | None = None
) -> tuple[list[bytes], list[bytes], bool]:
    """Names a row binds, names it reads (outside `skip`), and whether it binds with `:=`."""
    text, code = lines.text[row], lines.code[row]
    bound, walrus = _binding(text, code)
    starts = {start for start, _ in bound}
    lo, hi = skip or (0, 0)
    depth_at, _ = _depths(text, code)
    reads = [
        text[s:e]
        for s, e in _words(text, code)
        if s not in starts and not lo <= s < hi and not _keyword_arg(text, depth_at, e)
    ]
    return [text[s:e] for s, e in bound], reads, walrus


def _open_row(lines: _Lines, row: int) -> int:
    """Walk up from `row` to the row that opens its still-open brackets (multi-line headers)."""
    need = 0
    for cur in range(row, -1, -1):
        need -= _depths(lines.text[cur], lines.code[cur])[1]
        if need <= 0:
            return cur
    return 0


def _enclosing(lines: _Lines, start: int) -> list[tuple[int, int]]:
    """First and last row of each block header around `start`, innermost first."""
    blocks: list[tuple[int, int]] = []
    cur, row = lines.indent(start), start - 1
    while row >= 0 and cur > 0:
        if lines.opens_block(row) and lines.indent(row) < cur:
            head = row
            prev = next((r for r in range(row - 1, -1, -1) if not lines.blank(r)), -1)
            if lines.text[row].strip()[:1] == b"{" and prev >= 0 and lines.indent(prev) == lines.indent(row):
                head = prev
            first = _open_row(lines, head)
            blocks.append((first, row))
            cur, row = lines.indent(first), first - 1
        else:
            row -= 1
    return blocks


def _header_prefix(text: bytes, code: np.ndarray) -> bytes | None:
    """Bytes before `NAME(` on a definition-like header row, such as `def ` or `func (s *S) `."""
    ind = next((col for col, byte in enumerate(text) if 32 < byte < 128), 0)
    depth_at, _ = _depths(text, code)
    for col in range(ind, len(text)):
        if text[col] != 40 or not code[col] or depth_at[col] != 0:
            continue
        start = col
        while start > ind and text[start - 1] in _WORD:
            start -= 1
        if start == col or text[start] in _DIGITS:
            continue
        head = range(ind, start)
        keyword = any(text[c] in _WORD and depth_at[c] == 0 for c in head)
        plain = all(depth_at[c] > 0 or text[c] in _PREFIX for c in head)
        return text[ind:start] if keyword and plain else None
    return None


def _param_decl(text: bytes, code: np.ndarray, depth_at: list[int], span: tuple[int, int]) -> bytes:
    """The parameter item around `span` on its own row (`User user`, `drive: Drive`), else the name."""
    lo, hi = span[0] - 1, span[1]
    while lo >= 0 and not (code[lo] and (text[lo] == 44 and depth_at[lo] == 1 or text[lo] == 40 and depth_at[lo] == 0)):
        lo -= 1
    while hi < len(text) and not (code[hi] and depth_at[hi] == 1 and text[hi] in b",=)"):
        hi += 1
    if lo < 0 or hi >= len(text):
        return text[span[0] : span[1]]
    return text[lo + 1 : hi].strip()


def _header_params(lines: _Lines, first: int, last: int) -> list[tuple[bytes, bytes]]:
    """Parameters in header rows as (name, declaration): depth-1 names before any `:` or `=`."""
    params: list[tuple[bytes, bytes]] = []
    depth, typed = 0, False
    for row in range(first, last + 1):
        text, code = lines.text[row], lines.code[row]
        depth_at, _ = _depths(text, code, depth)
        ends = dict(_words(text, code))
        for col, byte in enumerate(text):
            if not code[col]:
                continue
            if byte in _OPEN:
                depth += 1
                typed = typed and depth > 1
            elif byte in _CLOSE:
                depth -= 1
            elif depth == 1 and byte in b",:=":
                typed = byte != 44
            elif depth == 1 and not typed and col in ends:
                params.append((text[col : ends[col]], _param_decl(text, code, depth_at, (col, ends[col]))))
    return params


def _header_names(lines: _Lines, first: int, last: int) -> list[bytes]:
    return [name for name, _ in _header_params(lines, first, last)]


def _uses_scope(lines: _Lines, block: tuple[int, int], outer: tuple[int, int]) -> bool:
    """True when a header's bracket names are already bound in the block around it (`if f(x):`)."""
    scope = set(_names_before(lines, block[0], (outer[0], outer[1], b""), lines.indent(block[0])))
    return bool(scope.intersection(_header_names(lines, block[0], block[1])))


def _pick_header(lines: _Lines, start: int, reads: set[bytes]) -> tuple[int, int, bytes] | None:
    """Header of the definition around `start`: the outermost whose parameters the clone reads.

    Headers that only use names already in scope are control flow, not definitions."""
    blocks = _enclosing(lines, start)
    if not blocks or lines.indent(blocks[-1][0]) != 0:
        return None
    forms = []
    for idx, (first, last) in enumerate(blocks):
        prefix = _header_prefix(lines.text[first], lines.code[first])
        if prefix is None or (idx + 1 < len(blocks) and _uses_scope(lines, (first, last), blocks[idx + 1])):
            continue
        forms.append((first, last, prefix))
    for first, last, prefix in reversed(forms):
        if reads.intersection(_header_names(lines, first, last)):
            return first, last, prefix
    return forms[0] if forms else None


def _names_before(
    lines: _Lines, start: int, header: tuple[int, int, bytes] | None, deepest: int = 128
) -> list[bytes]:
    """Names bound before `start` in its definition: parameters, then assignments indented <= `deepest`."""
    first, last = (header[0], header[1]) if header else (0, -1)
    names = _header_names(lines, first, last) if header else []
    depth = 0
    for row in range(last + 1, start):
        if depth == 0 and lines.indent(row) <= deepest:
            names.extend(_row_flow(lines, row)[0])
        depth = max(0, _depths(lines.text[row], lines.code[row], depth)[1])
    return names


def _reads_after(lines: _Lines, row: int, header: tuple[int, int, bytes] | None) -> set[bytes]:
    """Names read from `row` to the end of the definition that holds it."""
    stop = lines.indent(header[0]) if header else lines.indent(row) - 1
    reads: set[bytes] = set()
    for cur in range(row, len(lines.text)):
        if lines.opens_block(cur) and lines.indent(cur) <= stop:
            break
        reads.update(_row_flow(lines, cur)[1])
    return reads


def _atom(mid: bytes, code: np.ndarray) -> bool:
    """Check for one name, number, dotted name or plain string literal."""
    if len(mid) >= 2 and mid[0] in _QUOTES:
        return mid[-1] == mid[0] and mid[0] not in mid[1:-1] and 92 not in mid
    dotted = bool(mid) and all(byte in _WORD or byte == 46 for byte in mid)
    return dotted and mid[:1] != b"." and mid[-1:] != b"." and bool(code.all())


def _slot(rows: list[bytes], codes: list[np.ndarray]) -> tuple[int, int] | None:
    """Common prefix and suffix lengths when the rows differ in exactly one atom."""
    ref = rows[0]
    short = min(len(row) for row in rows)
    pre = suf = 0
    while pre < short and all(row[pre] == ref[pre] for row in rows):
        pre += 1
    while suf < short - pre and all(row[-1 - suf] == ref[-1 - suf] for row in rows):
        suf += 1
    while pre and (ref[pre - 1] in _WORD or ref[pre - 1] == 46):
        pre -= 1
    while suf and (ref[len(ref) - suf] in _WORD or ref[len(ref) - suf] == 46):
        suf -= 1
    if pre and suf and ref[pre - 1] in _QUOTES and ref[len(ref) - suf] == ref[pre - 1]:
        pre, suf = pre - 1, suf - 1
    for row, code in zip(rows, codes):
        if not _atom(row[pre : len(row) - suf], code[pre : len(row) - suf]):
            return None
    return pre, suf


def _fresh(stem: bytes, taken: set[bytes]) -> bytes:
    name, k = stem, 2
    while name in taken:
        name, k = stem + str(k).encode(), k + 1
    return name


def _last_code_byte(lines: _Lines, row: int) -> int:
    text, code = lines.text[row], lines.code[row]
    return next((text[c] for c in range(len(text) - 1, -1, -1) if code[c] and text[c] > 32), 0)


def _clone_body(lines: _Lines, members: list[int], length: int) -> tuple[list[int], int]:
    """Drop a header (and its closing row) the clones start with, so the helper takes the body."""
    ref = members[0]
    if length < 3 or not lines.is_opener(ref):
        return members, length
    skip = 2 if lines.text[ref + 1].strip()[:1] == b"{" else 1
    last = ref + length - 1
    closes = lines.indent(last) == lines.indent(ref) and lines.starts_with(last, _CLOSE)
    return [m + skip for m in members], length - skip - int(closes)


def _starts_statement(lines: _Lines, row: int) -> bool:
    """False when the row before `row` leaves an expression open (`(`, `[`, `,`, `\\`)."""
    prev = next((r for r in range(row - 1, -1, -1) if not lines.blank(r)), -1)
    return prev < 0 or _last_code_byte(lines, prev) not in b"([,\\"


def _row_kinds(lines: _Lines, members: list[int], length: int, width: int) -> list[Any] | None:
    """Per clone row: "same" text, a `(prefix, suffix)` slot, or None for a row each clone keeps.

    Rows stop at the first one above the clone's indent, and at a kept row that is nested or opens
    a block (moving it would change control flow)."""
    kinds: list[Any] = []
    for k in range(length):
        rows = [m + k for m in members]
        if any(len(lines.text[row]) >= width for row in rows):
            return None
        rel = lines.indent(rows[0]) - lines.indent(members[0])
        if rel < 0:
            break
        bodies = [lines.text[row][lines.indent(row) :] for row in rows]
        if all(body == bodies[0] for body in bodies):
            kinds.append("same")
            continue
        codes = [lines.code[row][lines.indent(row) : len(lines.text[row])] for row in rows]
        slot = _slot(bodies, codes)
        if slot is None and (rel > 0 or any(lines.is_opener(row) for row in rows)):
            break
        kinds.append(slot)
    return kinds


def _balanced_length(lines: _Lines, members: list[int], kinds: list[Any]) -> int:
    """Longest prefix of the clone rows that closes its brackets and blocks in every clone."""
    limit = len(kinds)
    for m in members:
        base, depth, good = lines.indent(m), 0, 0
        for k in range(limit):
            depth_at, after = _depths(lines.text[m + k], lines.code[m + k], depth)
            low = min(depth_at, default=depth)
            if low < 0 or after < 0 or (kinds[k] is None and (depth or after)):
                break
            depth = after
            nxt = lines.next_nonblank(m + k)
            ends_block = nxt >= len(lines.text) or (lines.indent(nxt) <= base and not lines.is_opener(m + k))
            good = k + 1 if depth == 0 and ends_block else good
        limit = min(limit, good)
    return limit


def _slot_span(lines: _Lines, row: int, slot: tuple[int, int]) -> tuple[int, int]:
    lo = lines.indent(row) + slot[0]
    return lo, len(lines.text[row]) - slot[1]


def _flow_cut(lines: _Lines, members: list[int], kinds: list[Any]) -> tuple[int, list[bytes], list[bytes], bool]:
    """Keep rows while data flow allows: kept rows must not read or rebind helper names."""
    bound: list[bytes] = []
    free: list[bytes] = []
    walrus = False
    for k, kind in enumerate(kinds):
        if kind is None:
            for m in members:
                names, reads, _ = _row_flow(lines, m + k)
                if set(reads) & set(bound) or set(names) & set(bound + free):
                    return k, bound, free, walrus
            continue
        skip = _slot_span(lines, members[0] + k, kind) if isinstance(kind, tuple) else None
        if skip and any(lines.text[m + k][slice(*_slot_span(lines, m + k, kind))] in bound for m in members):
            return k, bound, free, walrus
        spans, _ = _binding(lines.text[members[0] + k], lines.code[members[0] + k])
        if skip and any(s < skip[1] and skip[0] < e for s, e in spans):
            return k, bound, free, walrus
        names, reads, binds_walrus = _row_flow(lines, members[0] + k, skip)
        free.extend(name for name in reads if name not in bound and name not in free)
        bound.extend(name for name in names if name not in bound)
        walrus = walrus or binds_walrus
    return len(kinds), bound, free, walrus


def _slot_names(lines: _Lines, first: int, kinds: list[Any], params: list[bytes], bound: list[bytes]) -> dict[int, bytes]:
    """Name each slot after the reference text when that name is free, else `n`, `s` or `x`."""
    taken = {text[s:e] for text, code in zip(lines.text, lines.code) for s, e in _words(text, code)}
    names: dict[int, bytes] = {}
    for k, kind in enumerate(kinds):
        if not isinstance(kind, tuple):
            continue
        row = first + k
        lo, hi = _slot_span(lines, row, kind)
        mid = lines.text[row][lo:hi]
        others = {
            lines.text[first + j][s:e]
            for j, other in enumerate(kinds)
            if other is not None
            for s, e in _words(lines.text[first + j], lines.code[first + j])
            if j != k or not lo <= s < hi
        }
        plain = mid[0] not in _DIGITS and mid[0] not in _QUOTES and 46 not in mid
        if plain and mid not in others and mid not in params and mid not in bound:
            names[k] = mid
        else:
            stem = b"n" if mid[0] in _DIGITS else b"s" if mid[0] in _QUOTES else b"x"
            names[k] = _fresh(stem, taken | set(params) | set(names.values()))
        params.append(names[k])
    return names


def _named_block(lines: _Lines, row: int) -> bool:
    """A `:` header made of two names and an optional bracket group, like `class Name(Base):`."""
    text, code = lines.text[row], lines.code[row]
    if _last_code_byte(lines, row) != 58:
        return False
    depth_at, _ = _depths(text, code)
    top = [c for c in range(lines.indent(row), len(text)) if code[c] and depth_at[c] == 0]
    if any(text[c] not in _WORD and text[c] not in b" (:" for c in top):
        return False
    return len([s for s in _words(text, code) if depth_at[s[0]] == 0]) == 2


def _helper_site(lines: _Lines, header: tuple[int, int, bytes], first: int) -> tuple[int, int, bytes, bytes, int]:
    """Placement and syntax copied from the enclosing header: (row, indent, prefix, tail, body indent).

    A `:`-block helper leaves named blocks such as a class body, so it needs no receiver."""
    top, last, prefix = header
    ind = lines.indent(top)
    end = _last_code_byte(lines, last)
    tail = b":" if end == 58 else b" {" if end == 123 else b""
    body = next((lines.indent(r) for r in range(last + 1, first + 1) if not lines.blank(r)), ind + 4)
    body = body - ind if body > ind else 4
    blocks = _enclosing(lines, top) if tail == b":" else []
    while blocks and _named_block(lines, blocks[0][0]):
        top = blocks[0][0]
        blocks = _enclosing(lines, top)
    ind = lines.indent(top)
    while top > 0 and not lines.blank(top - 1) and lines.indent(top - 1) == ind:
        if lines.text[top - 1][ind] in _CLOSE:
            break
        top -= 1
    return top, ind, prefix, tail, ind + body


def _prose(lines: _Lines, row: int) -> bool:
    """Six or more names in a row joined by single spaces: comment text the masks missed."""
    text, run, best = lines.text[row], 0, 0
    spans = _words(text, lines.code[row])
    for (_, end), (start, _) in zip(spans, spans[1:]):
        run = run + 1 if text[end:start] == b" " else 0
        best = max(best, run)
    return best + 1 >= 6


def _movable_body(lines: _Lines, rows: list[int]) -> bool:
    """At least two rows start with code (not comments), and none is a `return` the helper would capture."""
    starts = [lines.text[row][lines.indent(row) :] for row in rows if lines.opens_block(row) and not _prose(lines, row)]
    returns = any(text[: len(_RETURN)] == _RETURN and text[len(_RETURN) : len(_RETURN) + 1] not in _WORD_BYTES for text in starts)
    return len(starts) >= 2 and not returns


def _plan_helper(lines: _Lines, members: list[int], length: int, width: int) -> _HelperPlan | None:
    """Plan a helper for `length` matching rows at each clone start in `members`, from their text."""
    members, length = _clone_body(lines, members, length)
    if not all(_starts_statement(lines, m) for m in members):
        return None
    kinds = _row_kinds(lines, members, length, width)
    if kinds is None:
        return None
    kinds = kinds[: _balanced_length(lines, members, kinds)]
    cut, bound, free, walrus = _flow_cut(lines, members, kinds)
    kinds = kinds[:cut]
    while kinds and kinds[-1] is None:
        kinds.pop()
    body_rows = [k for k, kind in enumerate(kinds) if kind is not None]
    if not _movable_body(lines, [members[0] + k for k in body_rows]):
        return None

    first = members[0]
    headers = [_pick_header(lines, m, set(free)) for m in members]
    if any(header is None for header in headers):
        return None
    avail: list[bytes] = []
    for m, header in zip(members, headers):
        kept = [n for k, kind in enumerate(kinds) if kind is None for n in _row_flow(lines, m + k)[0]]
        avail.extend(n for n in _names_before(lines, m, header) + kept if n not in avail)
    inputs = sorted((n for n in free if n in avail), key=avail.index)
    after: set[bytes] = set()
    for m, header in zip(members, headers):
        after |= _reads_after(lines, m + len(kinds), header)
    outputs = [n for n in bound if n in after]
    params = list(inputs)
    slots = _slot_names(lines, first, kinds, params, bound)

    taken = {text[s:e] for text, code in zip(lines.text, lines.code) for s, e in _words(text, code)}
    name = _fresh(_HELPER, taken)
    at, ind, prefix, tail, body_ind = _helper_site(lines, headers[0], first)
    semi = b";" if all(_last_code_byte(lines, first + k) == 59 for k in body_rows) else b""
    decls = dict(reversed(_header_params(lines, headers[0][0], headers[0][1])))
    rows = [b" " * ind + prefix + name + b"(" + b", ".join(decls.get(p, p) for p in params) + b")" + tail]
    for k in body_rows:
        row = first + k
        text = lines.text[row][lines.indent(row) :]
        if k in slots:
            text = text[: kinds[k][0]] + slots[k] + text[len(text) - kinds[k][1] :]
        rows.append(b" " * (body_ind + lines.indent(row) - lines.indent(first)) + text)
    if outputs:
        rows.append(b" " * body_ind + _RETURN + b" " + b", ".join(outputs) + semi)
    if tail == b" {":
        rows.append(b" " * ind + b"}")

    bind = (b", ".join(outputs) + (b" := " if walrus else b" = ")) if outputs else b""
    calls: dict[int, tuple[int, list[int], bytes]] = {}
    for m in members:
        args = inputs + [lines.text[m + k][slice(*_slot_span(lines, m + k, kinds[k]))] for k in slots]
        kept = [m + k for k, kind in enumerate(kinds) if kind is None]
        call = b" " * lines.indent(m) + bind + name + b"(" + b", ".join(args) + b")" + semi
        calls[m] = (m + len(kinds) - 1, kept, call)
    if max(len(text) for text in rows + [call for _, _, call in calls.values()]) > width:
        return None
    blanks = 0
    while blanks < 2 and at - blanks > 0 and lines.blank(at - blanks - 1):
        blanks += 1
    return _HelperPlan(at, rows, max(1, blanks), calls)


def _splice_helper(
    grid: np.ndarray, prov: list[tuple[bool, str]], plan: _HelperPlan
) -> tuple[np.ndarray, list[tuple[bool, str]]]:
    width = grid.shape[1]
    inner = {row for start, (last, _, _) in plan.calls.items() for row in range(start, last + 1)}
    rows_out: list[np.ndarray] = []
    prov_out: list[tuple[bool, str]] = []
    for idx in range(len(grid)):
        if idx == plan.insert_at:
            rows_out.extend(_synth_row(0, text, width) for text in plan.rows)
            prov_out.extend((True, "") for _ in plan.rows)
            rows_out.extend(_blank_row(width) for _ in range(plan.blank_after))
            prov_out.extend((True, "") for _ in range(plan.blank_after))
        if idx in plan.calls:
            last, kept, call = plan.calls[idx]
            rows_out.extend(grid[row] for row in kept)
            prov_out.extend((True, prov[row][1]) for row in kept)
            rows_out.append(_synth_row(0, call, width))
            prov_out.append((True, f"{prov[idx][1]}..{prov[last][1]}"))
        elif idx not in inner:
            rows_out.append(grid[idx])
            prov_out.append(prov[idx])
    return np.stack(rows_out, axis=0), prov_out


def _apply_clone_factor(
    hygiene: Any,
    params: dict[str, np.ndarray],
    grid: np.ndarray,
    prov: list[tuple[bool, str]],
    lprobs: np.ndarray,
    thresh: float,
    source: list[bytes],
) -> tuple[np.ndarray, list[tuple[bool, str]], np.ndarray]:
    dup_spans = _kind_spans(lprobs, 3, thresh, max_gap=2)
    if not dup_spans:
        return grid, prov, lprobs
    cells, sk_code, sk_skel = _extract_grid_features(hygiene, params, grid)
    ref, peers = _grow_clone_peers(grid, sk_code, sk_skel, dup_spans[0][0])
    if not peers:
        return grid, prov, lprobs

    lines = _Lines.of(grid, cells)
    slen = peers[0][1] - peers[0][0] + 1
    runs = [_shared_run(grid, sk_code, sk_skel, ref, start, slen) for start, _ in peers]
    dup_before = int(np.sum(lprobs[:, 3] >= thresh))
    best = None
    for length in sorted({n for n in runs if n >= 2}, reverse=True):
        members = [start for (start, _), n in zip(peers, runs) if n >= length]
        plan = _plan_helper(lines, members, length, grid.shape[1]) if len(members) >= 2 else None
        if plan is None:
            continue
        cand, cand_prov = _splice_helper(grid, prov, plan)
        lp_c, sc_c = hygiene._forward(params, cand)
        rank = (int(np.sum(lp_c.max(axis=1) >= thresh)), -float(sc_c))
        if int(np.sum(lp_c[:, 3] >= thresh)) < dup_before and (best is None or rank < best[0]):
            best = (rank, cand, cand_prov, lp_c)
    if best is None:
        return grid, prov, lprobs
    return best[1], best[2], best[3]


# Syntax read from the file's own rows, for the guard, dense-line and ladder steps.


@dataclass(frozen=True)
class _Block:
    """A block: header rows `head..open`, body rows `lo..hi` (last nonblank), closing row or -1."""

    head: int
    open: int
    lo: int
    hi: int
    close: int

    @property
    def end(self) -> int:
        return self.close if self.close >= 0 else self.hi


@dataclass(frozen=True)
class _Syntax:
    """File-wide conventions: statement `;` and the indent step."""

    semi: bytes
    unit: int


def _first_word(text: bytes, col: int) -> bytes:
    end = col
    while end < len(text) and text[end] in _WORD:
        end += 1
    return text[col:end]


def _starts_word(text: bytes, word: bytes) -> bool:
    return text[: len(word)] == word and text[len(word) : len(word) + 1] not in _WORD_BYTES


def _code_tail(lines: _Lines, row: int, col: int) -> bytes:
    """Code bytes of `row` from `col` on, without spaces."""
    text, code = lines.text[row], lines.code[row]
    return bytes(text[c] for c in range(col, len(text)) if code[c] and text[c] > 32)


def _trim(text: bytes, lo: int, hi: int) -> tuple[int, int]:
    while lo < hi and text[lo] == 32:
        lo += 1
    while hi > lo and text[hi - 1] == 32:
        hi -= 1
    return lo, hi


def _match(text: bytes, code: np.ndarray, col: int) -> int:
    """Column of the bracket that closes the one at `col` on the same row, else -1."""
    depth = 0
    for cur in range(col, len(text)):
        if code[cur] and text[cur] in _OPEN:
            depth += 1
        elif code[cur] and text[cur] in _CLOSE:
            depth -= 1
            if depth == 0:
                return cur
    return -1


def _block(lines: _Lines, head: int, open_row: int) -> _Block | None:
    """The block opened by a header ending in `:` or `{`, with its body and closing row."""
    tail = _last_code_byte(lines, open_row)
    if tail not in (58, 123):
        return None
    ind, row, last = lines.indent(head), open_row + 1, -1
    while row < len(lines.text) and (lines.blank(row) or lines.indent(row) > ind):
        last = row if not lines.blank(row) else last
        row += 1
    if last < 0 or (tail == 123 and not (row < len(lines.text) and lines.starts_with(row, _CLOSE))):
        return None
    return _Block(head, open_row, open_row + 1, last, row if tail == 123 else -1)


def _block_at(lines: _Lines, row: int) -> _Block | None:
    nxt = lines.next_nonblank(row)
    brace = nxt < len(lines.text) and lines.text[nxt].strip() == b"{"
    allman = brace and lines.indent(nxt) == lines.indent(row)
    return _block(lines, row, nxt if allman else row)


def _body_indent(lines: _Lines, blk: _Block) -> int:
    return next(lines.indent(r) for r in range(blk.lo, blk.hi + 1) if not lines.blank(r))


def _def_word(lines: _Lines, row: int) -> str | None:
    """Family of a definition header: `def`, `fn`, `func`, `function` or an arrow `=>`."""
    text, code = lines.text[row], lines.code[row]
    depth_at, _ = _depths(text, code)
    paren = next((c for c in range(len(text)) if text[c] == 40 and code[c] and depth_at[c] == 0), len(text))
    for start, end in _words(text, code, 0, paren):
        if depth_at[start] == 0 and text[start:end] in _DEFS:
            return _DEFS[text[start:end]]
    arrow = any(text[c : c + 2] == b"=>" and code[c] for c in range(len(text)))
    return "js" if arrow and _last_code_byte(lines, row) == 123 else None


def _family(lines: _Lines, row: int) -> str:
    """`py`, `rust`, `go`, `js` or `c`, from the blocks around `row`, else the first definition in the window."""
    for first, opener in _enclosing(lines, row):
        if _last_code_byte(lines, opener) == 58:
            return "py"
        word = _def_word(lines, first)
        if word:
            return word
    for cur in range(len(lines.text)):
        if not lines.blank(cur) and lines.is_opener(cur):
            word = _def_word(lines, cur) or ("py" if _last_code_byte(lines, cur) == 58 else None)
            if word:
                return word
    return "c"


def _syntax(lines: _Lines) -> _Syntax:
    """Statement `;` when code rows end with one, and the most common opener-to-body indent step."""
    semi = any(_last_code_byte(lines, row) == 59 for row in range(len(lines.text)))
    steps: dict[int, int] = {}
    for row in range(len(lines.text)):
        nxt = lines.next_nonblank(row)
        if nxt < len(lines.text) and lines.is_opener(row) and lines.indent(nxt) > lines.indent(row):
            step = lines.indent(nxt) - lines.indent(row)
            steps[step] = steps.get(step, 0) + 1
    unit = min(steps, key=lambda step: (-steps[step], step)) if steps else 4
    return _Syntax(b";" if semi else b"", unit)


def _is_loop(lines: _Lines, blk: _Block) -> bool:
    return _first_word(lines.text[blk.head], lines.indent(blk.head)) in _LOOPS


def _is_def(lines: _Lines, blk: _Block) -> bool:
    text, code = lines.text[blk.head], lines.code[blk.head]
    if _def_word(lines, blk.head) or _def_word(lines, blk.open):
        return True
    plain = _first_word(text, lines.indent(blk.head)) not in _CONTROL
    return plain and _last_code_byte(lines, blk.open) == 123 and _header_prefix(text, code) is not None


# Conditions: split at spaces outside brackets, negated by the file's own operators.


def _top_tokens(text: bytes, code: np.ndarray) -> list[tuple[int, int]]:
    """Spans of `text` between spaces that sit in code cells outside brackets."""
    depth_at, _ = _depths(text, code)
    spans: list[tuple[int, int]] = []
    start = -1
    for col, byte in enumerate(text):
        cut = byte == 32 and bool(code[col]) and depth_at[col] == 0
        if cut and start >= 0:
            spans.append((start, col))
        start = -1 if cut else (col if start < 0 else start)
    if start >= 0:
        spans.append((start, len(text)))
    return spans


def _operand(text: bytes, code: np.ndarray) -> bool:
    """One name, call, index or path: only names, `.`, `::` and brackets outside brackets."""
    depth_at, _ = _depths(text, code)
    top = [text[c] for c in range(len(text)) if code[c] and depth_at[c] == 0]
    return bool(text) and text[0] not in b"!-" and all(b in _WORD or b in b".:()[]" for b in top)


def _unwrap(text: bytes, code: np.ndarray) -> bytes:
    whole = len(text) > 2 and text[0] == 40 and _match(text, code, 0) == len(text) - 1
    return text[1:-1] if whole else text


def _compare_ops(toks: list[bytes], colon: bool) -> list[tuple[int, int, bytes]]:
    """(first token, token count, negated operator) for each comparison among `toks`."""
    table = {(op,): swap for op, swap in _SWAP.items()}
    table.update(_WORD_SWAP if colon else {})
    ops: list[tuple[int, int, bytes]] = []
    idx = 0
    while idx < len(toks):
        size = next((n for n in (2, 1) if tuple(toks[idx : idx + n]) in table and idx + n <= len(toks)), 0)
        if size:
            ops.append((idx, size, table[tuple(toks[idx : idx + size])]))
        idx += max(size, 1)
    return ops


def _swap_compare(text: bytes, code: np.ndarray, colon: bool) -> bytes | None:
    """`text` with its one comparison negated (`>` to `<=`, `is` to `is not`), when nothing else joins it."""
    spans = _top_tokens(text, code)
    toks = [text[s:e] for s, e in spans]
    ops = _compare_ops(toks, colon)
    used = {i for idx, size, _ in ops for i in range(idx, idx + size)}
    if len(ops) != 1 or any(tok in _LOGIC for i, tok in enumerate(toks) if i not in used):
        return None
    idx, size, swap = ops[0]
    return text[: spans[idx][0]] + swap + text[spans[idx + size - 1][1] :]


def _negate(text: bytes, code: np.ndarray, colon: bool) -> bytes:
    """The condition's negation in the file's operators: `!`/`not` dropped, a comparison swapped, or added."""
    bang = b"not " if colon else b"!"
    rest, rest_code = text[len(bang) :], code[len(bang) :]
    if text.startswith(bang) and _operand(rest, rest_code):
        return _unwrap(rest, rest_code)
    swapped = _swap_compare(text, code, colon)
    if swapped is not None:
        return swapped
    return bang + text if _operand(text, code) else bang + b"(" + text + b")"


def _holds_statement(text: bytes, code: np.ndarray) -> bool:
    """True when a condition carries its own statement: a `;` or a lone `=` outside brackets."""
    depth_at, _ = _depths(text, code)
    for col, byte in enumerate(text):
        if not code[col] or depth_at[col] or byte not in (59, 61):
            continue
        if byte == 59 or (text[col - 1 : col] not in b"=!<>:" and text[col + 1 : col + 2] not in b"=>"):
            return True
    return False


def _condition(lines: _Lines, row: int, colon: bool) -> tuple[int, int, int] | None:
    """Condition columns of an `if` row, and the column where the rest of the row starts."""
    text, code = lines.text[row], lines.code[row]
    ind = lines.indent(row)
    if lines.clipped[row] or _first_word(text, ind) != _IF or text[ind + 2 : ind + 3] not in (b" ", b"("):
        return None
    depth_at, _ = _depths(text, code)
    start, _ = _trim(text, ind + 2, len(text))
    top = [c for c in range(start, len(text)) if code[c] and depth_at[c] == 0]
    if colon:
        end = next((c for c in top if text[c] == 58 and text[c + 1 : c + 2] != b"="), -1)
        return (start, end, end + 1) if end > start else None
    close = _match(text, code, start) if text[start : start + 1] == b"(" else -1
    after = text[close + 1 :].strip()[:1] if close > 0 else b""
    if close > 0 and (not after or after not in b"&|=<>!+-*/%^.?"):
        return start + 1, close, close + 1
    brace = next((c for c in top if text[c] == 123), -1)
    return (start, brace, brace) if brace > start else None


# Pyramid flattening: each nested `if` header becomes an early exit, and the innermost body moves out.


def _guard_rows(lines: _Lines, blk: _Block, family: str) -> tuple[list[bytes], bytes] | None:
    """Header rows of the early-exit form of an `if` block, and its closing row (empty for `:` blocks)."""
    head, ind = blk.head, lines.indent(blk.head)
    colon = _last_code_byte(lines, blk.open) == 58
    allman = blk.open != head
    if (blk.close >= 0 and lines.text[blk.close].strip() != b"}") or (allman and blk.open != lines.next_nonblank(head)):
        return None
    span = _condition(lines, head, colon)
    if span is None or _code_tail(lines, head, span[2]) not in (b"", b":", b"{"):
        return None
    text, code = lines.text[head], lines.code[head]
    lo, hi = _trim(text, span[0], span[1])
    cond = text[lo:hi]
    if not colon and cond.startswith(_LET):
        return _let_else(text, code, ind) if family == "rust" and not allman else None
    if _holds_statement(cond, code[lo:hi]):
        return None
    header = text[ind:lo] + _negate(cond, code[lo:hi], colon) + text[hi:]
    return [header, b"{"] if allman else [header], b"" if colon else b"}"


def _last_brace(text: bytes, code: np.ndarray) -> int:
    """Column of the row's last `{` in code cells, else -1."""
    return next((c for c in range(len(text) - 1, -1, -1) if code[c] and text[c] == 123), -1)


def _let_else(text: bytes, code: np.ndarray, ind: int) -> tuple[list[bytes], bytes]:
    """Rust `if let P = E {` as the guard `let P = E else {`, closed by `};`."""
    brace = _last_brace(text, code)
    return [text[ind + 3 : brace].rstrip() + b" else " + text[brace:]], b"};"


def _lead_word(lines: _Lines, row: int) -> bytes:
    """First word of a row, after a leading `}` (as in `} else {`)."""
    text = lines.text[row]
    skip = int(text[lines.indent(row) : lines.indent(row) + 1] == b"}")
    return _first_word(text, _trim(text, lines.indent(row) + skip, len(text))[0])


def _statement_end(lines: _Lines, blk: _Block) -> int:
    """Last row of the statement `blk` opens, through its `else`, `elif`, `except`, `catch` and `finally` blocks."""
    ind = lines.indent(blk.head)
    while True:
        joined = blk.close >= 0 and lines.text[blk.close].strip() != b"}"
        row = blk.close if joined else lines.next_nonblank(blk.end)
        level = row < len(lines.text) and lines.indent(row) == ind
        if not level or _lead_word(lines, row) not in _CONTINUES:
            return blk.end
        nxt = _block_at(lines, row)
        if nxt is None or nxt.end <= blk.end:
            return blk.end
        blk = nxt


def _guardable(lines: _Lines, blk: _Block, family: str) -> bool:
    """An `if` with no `else` that can take the early-exit form."""
    return _guard_rows(lines, blk, family) is not None and _statement_end(lines, blk) == blk.end


def _opaque(lines: _Lines, blk: _Block) -> bool:
    """A block an early exit must not leave through: `finally`, or a closure (`|x| {`, `{` inside brackets)."""
    if _lead_word(lines, blk.head) == b"finally":
        return True
    text, code = lines.text[blk.open], lines.code[blk.open]
    brace = _last_brace(text, code)
    if brace < 0:
        return False
    depth_at, _ = _depths(text, code)
    return depth_at[brace] > 0 or text[:brace].rstrip()[-1:] == b"|"


@dataclass(frozen=True)
class _Site:
    """An `if` block to flatten, the exit row its guards use, and the definition row it copied (or -1)."""

    blk: _Block
    leave: bytes
    family: str
    follow: int = -1


def _copied_exit(lines: _Lines, blk: _Block, parent: _Block, family: str, direct: bool) -> tuple[bytes, int] | None:
    """The definition's last row after `blk` as a guard exit: a `return` with a plain value, or a Rust tail.

    The row is dropped later only when `blk` is the guarded `if` itself (`direct`)."""
    after = lines.next_nonblank(_statement_end(lines, blk))
    if after != parent.hi or lines.indent(after) != lines.indent(blk.head) or lines.clipped[after]:
        return None
    text = lines.text[after][lines.indent(after) :]
    if _depths(lines.text[after], lines.code[after])[1] != 0 or lines.is_opener(after):
        return None
    if any(text[c] == 46 and _calls_at(text, c) for c in range(len(text))):
        return None
    follow = after if direct else -1
    if _starts_word(text, _RETURN):
        return text, follow
    tail = family == "rust" and _last_code_byte(lines, after) not in (59, 125)
    return (_RETURN + b" " + text + b";", follow) if tail else None


def _exit_row(
    lines: _Lines, chain: list[_Block | None], idx: int, syntax: _Syntax, family: str
) -> tuple[bytes, int] | None:
    """How a guard for `chain[idx]` leaves early: `continue` in a loop; in a definition, `return` or its last row.

    Blocks between (`try`, `with`, `else`, match arms) are passed when each one ends its parent's body."""
    blk = chain[idx]
    for parent in chain[idx + 1 :]:
        if blk is None or parent is None:
            return None
        ends = _statement_end(lines, blk) == parent.hi
        if ends and _is_loop(lines, parent):
            return _CONTINUE + syntax.semi, -1
        if _is_def(lines, parent):
            direct = blk is chain[idx]
            return (_RETURN + syntax.semi, -1) if ends else _copied_exit(lines, blk, parent, family, direct)
        if not ends or _is_loop(lines, parent) or _opaque(lines, parent):
            return None
        blk = parent
    return None


def _last_if(lines: _Lines, blk: _Block, family: str) -> _Block | None:
    """The `if` block that ends `blk`'s body, when it can become a guard as well."""
    ind = _body_indent(lines, blk)
    for row in range(blk.hi, blk.lo - 1, -1):
        if lines.blank(row) or lines.indent(row) != ind or lines.text[row].strip() in (b"}", b"{"):
            continue
        inner = _block_at(lines, row)
        good = inner is not None and inner.end == blk.hi and _guardable(lines, inner, family)
        return inner if good else None
    return None


def _guard_sites(lines: _Lines, rows: list[int], syntax: _Syntax) -> dict[int, _Site]:
    """Outermost `if` of each run of last-statement `if` blocks above `rows`, with how it exits."""
    sites: dict[int, _Site] = {}
    for row in rows:
        chain = [_block(lines, first, opener) for first, opener in _enclosing(lines, row)]
        family = _family(lines, row)
        for idx, (blk, parent) in enumerate(zip(chain, chain[1:])):
            if blk is None or parent is None or blk.head in sites or not _guardable(lines, blk, family):
                continue
            if blk.end == parent.hi and _guardable(lines, parent, family):
                continue
            leave = _exit_row(lines, chain, idx, syntax, family)
            if leave is not None:
                sites[blk.head] = _Site(blk, leave[0], family, leave[1])
    return sites


_Entry = tuple[int, bytes, int, int, bool]


def _emit(lines: _Lines, span: tuple[int, int], shift: int, sites: dict[int, _Site], out: list[_Entry]) -> None:
    """Rows `span` moved left by `shift`, each guard site flattened: (indent, text, source rows, changed).

    A definition's last row that the guards copied is dropped once the moved body ends in an exit."""
    row, hi = span
    dead: set[int] = set()
    while row <= hi:
        site = sites.get(row)
        if site is not None:
            if _flatten(lines, site, shift, sites, out):
                dead.add(site.follow)
            row = site.blk.end + 1
            continue
        ind = 0 if lines.blank(row) else lines.indent(row) - shift
        if row not in dead:
            out.append((ind, lines.text[row][lines.indent(row) :], row, row, shift > 0))
        row += 1


def _flatten(lines: _Lines, site: _Site, shift: int, sites: dict[int, _Site], out: list[_Entry]) -> bool:
    """Guards for the site's `if` and each `if` that ends its body, then the body; True if it ends in an exit."""
    cur: _Block | None = site.blk
    base = lines.indent(site.blk.head) - shift
    while cur is not None:
        header, close = _guard_rows(lines, cur, site.family) or ([], b"")
        body = _body_indent(lines, cur)
        out.extend((base, text, cur.head, cur.head, True) for text in header)
        out.append((base + body - lines.indent(cur.head), site.leave, cur.head, cur.head, True))
        if close:
            out.append((base, close, cur.close, cur.close, True))
        inner = _last_if(lines, cur, site.family)
        _emit(lines, (cur.lo, inner.head - 1 if inner else cur.hi), body - base, sites, out)
        cur = inner
    last = next((i for i in range(len(out) - 1, -1, -1) if out[i][1] and out[i][0] == base), -1)
    if site.follow < 0 or last < 0 or not any(_starts_word(out[last][1], word) for word in _EXITS[:2]):
        return False
    ind, text, lo, _, _ = out[last]
    out[last] = (ind, text, lo, site.follow, True)
    return True


def _assemble(
    grid: np.ndarray, prov: list[tuple[bool, str]], out: list[_Entry]
) -> tuple[np.ndarray, list[tuple[bool, str]]] | None:
    """Rows and provenance for entries; changed rows are tagged with their source rows."""
    width = grid.shape[1]
    if any(changed and ind + len(text) >= width for ind, text, _, _, changed in out):
        return None
    rows: list[np.ndarray] = []
    tags: list[tuple[bool, str]] = []
    for ind, text, lo, hi, changed in out:
        rows.append(_synth_row(ind, text, width) if changed else grid[lo])
        tag = "" if lo < 0 else prov[lo][1] if lo == hi else f"{prov[lo][1]}..{prov[hi][1]}"
        tags.append((True, tag) if changed else prov[lo])
    return np.stack(rows, axis=0), tags


def _apply_pyramid(
    hygiene: Any,
    params: dict[str, np.ndarray],
    grid: np.ndarray,
    prov: list[tuple[bool, str]],
    lprobs: np.ndarray,
    thresh: float,
    source: list[bytes],
) -> tuple[np.ndarray, list[tuple[bool, str]], np.ndarray]:
    nest_spans = _kind_spans(lprobs, 0, thresh, max_gap=2)
    if not nest_spans:
        return grid, prov, lprobs
    cells, _, _ = _extract_grid_features(hygiene, params, grid)
    lines = _Lines.of(grid, cells)
    rows = [row for a, b in nest_spans for row in range(a, b + 1)]
    sites = _guard_sites(lines, rows, _syntax(lines))
    out: list[_Entry] = []
    _emit(lines, (0, len(grid) - 1), 0, sites, out)
    built = _assemble(grid, prov, out) if sites else None
    if built is None:
        return grid, prov, lprobs
    lp_c, _ = hygiene._forward(params, built[0])
    return built[0], built[1], lp_c


# Dense lines: the row split at its own separators, each piece keeping its bytes.


def _pieces(body: bytes, cuts: list[int], steps: list[int]) -> list[tuple[int, bytes]]:
    bounds = [0, *cuts, len(body)]
    out = [(step, body[lo:hi].strip()) for lo, hi, step in zip(bounds, bounds[1:], steps)]
    return [(step, text) for step, text in out if text]


def _open_brace(seg: bytes, mask: np.ndarray) -> list[tuple[int, bytes]] | None:
    """A one-row `{ a; b; }` block opened onto rows: header, its statements one step in, closing row."""
    depth_at, _ = _depths(seg, mask)
    col = next((c for c in range(len(seg)) if seg[c] == 123 and mask[c] and depth_at[c] == 0), -1)
    end = _match(seg, mask, col) if col >= 0 else -1
    inner = _split_statements(seg[col + 1 : end], mask[col + 1 : end], "") if end > col else None
    if inner is None:
        return None
    return [(0, seg[: col + 1].rstrip()), *((step + 1, text) for step, text in inner), (0, seg[end:].strip())]


def _split_statements(body: bytes, mask: np.ndarray, family: str) -> list[tuple[int, bytes]] | None:
    """One row per statement at each `;` outside brackets (dropped in Python and Go); one-row braces are opened."""
    depth_at, _ = _depths(body, mask)
    cuts = [c + 1 for c in range(len(body)) if body[c] == 59 and mask[c] and depth_at[c] == 0 and body[c + 1 :].strip()]
    pieces: list[tuple[int, bytes]] = []
    for lo, hi in zip([0, *cuts], [*cuts, len(body)]):
        lo, hi = _trim(body, lo, hi)
        hi = hi - 1 if family in ("py", "go") and hi > lo and body[hi - 1] == 59 else hi
        if lo < hi:
            pieces.extend(_open_brace(body[lo:hi], mask[lo:hi]) or [(0, body[lo:hi])])
    return pieces if len(pieces) > 1 else None


def _spaced(body: bytes, mask: np.ndarray, depth_at: list[int], col: int, byte: int) -> bool:
    return body[col] == byte and bool(mask[col]) and depth_at[col] == 0 and body[col - 1 : col + 2 : 2] == b"  "


def _split_header(body: bytes, mask: np.ndarray, family: str) -> list[tuple[int, bytes]] | None:
    """A one-row `if (c) stmt`, `for (...) stmt` or `if c: stmt` cut after its header, the rest one step in."""
    word = _first_word(body, 0)
    if word not in _HEADS:
        return None
    depth_at, _ = _depths(body, mask)
    colons = [c + 1 for c in range(len(word), len(body)) if body[c] == 58 and mask[c] and depth_at[c] == 0]
    start, _ = _trim(body, len(word), len(body))
    paren = _match(body, mask, start) + 1 if body[start : start + 1] == b"(" else 0
    cut = next((c for c in colons if body[c : c + 1] != b"="), 0) if family == "py" else paren
    lo, hi = _trim(body, cut, len(body)) if cut > 0 else (0, 0)
    if lo >= hi or body[lo] in b"{;":
        return None
    inner = _split_header(body[lo:hi], mask[lo:hi], family) or [(0, body[lo:hi])]
    return [(0, body[:cut].rstrip()), *((step + 1, text) for step, text in inner)]


def _split_ternary(body: bytes, mask: np.ndarray, family: str) -> list[tuple[int, bytes]] | None:
    """A `? :` chain cut before each `:` (one `cond ? value` per row), or before `?` and `:` when single."""
    depth_at, _ = _depths(body, mask)
    asks = [c for c in range(1, len(body) - 1) if _spaced(body, mask, depth_at, c, 63)]
    elses = [c for c in range(asks[0], len(body) - 1) if _spaced(body, mask, depth_at, c, 58)] if asks else []
    if family == "py" or not asks or len(elses) != len(asks):
        return None
    cuts = elses if len(asks) > 1 else [asks[0], elses[0]]
    return _pieces(body, cuts, [0] + [1] * len(cuts))


def _calls_at(text: bytes, col: int) -> bool:
    """True for `.name(` or `.name::<T>(` at `col`: a method call in a chain."""
    end = col + 1
    if end >= len(text) or text[end] not in _WORD or text[end] in _DIGITS:
        return False
    while end < len(text) and text[end] in _WORD:
        end += 1
    if text[end : end + 3] == b"::<":
        end = text.find(b">", end) + 1 or end
    return text[end : end + 1] == b"("


def _split_chain(body: bytes, mask: np.ndarray, family: str) -> list[tuple[int, bytes]] | None:
    """A call chain cut at each `.name(`: before the dot, after it in Go, inside `( )` in Python."""
    depth_at, _ = _depths(body, mask)
    dots = [c for c in range(1, len(body) - 1) if body[c] == 46 and mask[c] and depth_at[c] == 0 and _calls_at(body, c)]
    if len(dots) < 2 or any(body[c] == 32 and mask[c] and depth_at[c] == 0 for c in range(dots[0], len(body))):
        return None
    if family == "go":
        cuts = [dot + 1 for dot in dots[1:]]
        return _pieces(body, cuts, [0] + [1] * len(cuts))
    pieces = _pieces(body, dots, [0] + [1] * len(dots))
    if family != "py":
        return pieces
    spaces = [c + 1 for c in range(dots[0]) if body[c] == 32 and mask[c] and depth_at[c] == 0]
    lead = body[: spaces[-1]] if spaces else b""
    root = (1, pieces[0][1][len(lead) :])
    return [(0, lead + b"("), root, *((1, text) for _, text in pieces[1:]), (0, b")")]


def _split_args(body: bytes, mask: np.ndarray, family: str) -> list[tuple[int, bytes]] | None:
    """The bracket group with the most top-level commas, one item per row: three or more items, or a
    group that ends the statement."""
    depth_at, _ = _depths(body, mask)
    best: tuple[int, int, list[int]] | None = None
    for col in range(len(body)):
        end = _match(body, mask, col) if body[col] in _OPEN and mask[col] and depth_at[col] == 0 else -1
        commas = [c for c in range(col + 1, end) if body[c] == 44 and mask[c] and depth_at[c] == 1]
        if commas and (best is None or len(commas) > len(best[2])):
            best = (col, end, commas)
    ends = best is not None and body[best[1] + 1 :].strip(b" ;:{,)]") == b""
    if best is None or (len(best[2]) < 2 and not ends):
        return None
    col, end, commas = best
    items = [body[lo + 1 : hi].strip() for lo, hi in zip([col, *commas], [*commas, end])]
    trail = b"," if family == "go" or not items[-1] else b""
    items = [item for item in items if item]
    if not items:
        return None
    rows = [(1, item + b",") for item in items[:-1]] + [(1, items[-1] + trail)]
    return [(0, body[: col + 1].rstrip()), *rows, (0, body[end:].strip())]


def _dense_pieces(lines: _Lines, row: int, width: int) -> list[tuple[int, bytes]] | None:
    """The row's pieces as (indent steps, text): split at `;`, a control header, `? :`, a call chain or commas.

    A row the grid clipped is split only when its source line's rest was read into `lines.tails`."""
    text, code = lines.tails.get(row, (lines.text[row], lines.code[row]))
    if lines.blank(row) or (lines.clipped[row] and row not in lines.tails):
        return None
    ind = lines.indent(row)
    body, mask = text[ind:], code[ind : len(text)]
    family = _family(lines, row)
    for split in (_split_statements, _split_header, _split_ternary, _split_chain, _split_args):
        pieces = split(body, mask, family)
        if pieces:
            return pieces
    return None


def _clipped_tails(
    hygiene: Any,
    params: dict[str, np.ndarray],
    lines: _Lines,
    prov: list[tuple[bool, str]],
    rows: list[int],
    source: list[bytes],
) -> dict[int, tuple[bytes, np.ndarray]]:
    """Full text and code mask of unchanged rows the grid clipped, read from their source lines.

    A row is read only when its last grid cell is code; the rest is masked as a row of its own."""
    width = lines.code.shape[1]
    picks = {}
    for row in rows:
        rest = source[int(prov[row][1]) - 1][width:] if lines.clipped[row] and not prov[row][0] else b""
        if 0 < len(rest) < width and lines.code[row][-1]:
            picks[row] = rest
    if not picks:
        return {}
    cells, _, _ = _extract_grid_features(hygiene, params, np.stack([_synth_row(0, rest, width) for rest in picks.values()]))
    rest_code = (cells[:, :, _HC_STR] <= 0.5) & (cells[:, :, _HC_COMM] <= 0.5)
    tails: dict[int, tuple[bytes, np.ndarray]] = {}
    for k, (row, rest) in enumerate(picks.items()):
        head = bytes(lines.text[row]).ljust(width)
        tails[row] = (head + rest, np.concatenate([lines.code[row], rest_code[k, : len(rest)]]))
    return tails


def _apply_dense_unpack(
    hygiene: Any,
    params: dict[str, np.ndarray],
    grid: np.ndarray,
    prov: list[tuple[bool, str]],
    lprobs: np.ndarray,
    thresh: float,
    source: list[bytes],
) -> tuple[np.ndarray, list[tuple[bool, str]], np.ndarray]:
    spans = _kind_spans(lprobs, 1, thresh)
    if not spans:
        return grid, prov, lprobs
    cells, _, _ = _extract_grid_features(hygiene, params, grid)
    flagged = {r_i for a, b in spans for r_i in range(a, b + 1)}
    lines = _Lines.of(grid, cells)
    lines = _Lines.of(grid, cells, _clipped_tails(hygiene, params, lines, prov, sorted(flagged), source))
    unit = _syntax(lines).unit
    out: list[_Entry] = []
    for row in range(len(grid)):
        pieces = _dense_pieces(lines, row, grid.shape[1]) if row in flagged else None
        ind = lines.indent(row)
        if pieces and max(ind + step * unit + len(text) for step, text in pieces) >= grid.shape[1]:
            pieces = None
        out.extend((ind + step * unit, text, row, row, True) for step, text in pieces or [(0, b"")])
        if not pieces:
            out[-1] = (ind, b"", row, row, False)
    built = _assemble(grid, prov, out) if len(out) > len(grid) else None
    if built is None:
        return grid, prov, lprobs
    lp_c, _ = hygiene._forward(params, built[0])
    return built[0], built[1], lp_c


# Ladders: adjacent `if SUBJECT == KEY` rows become the language's switch, match or lookup table.


@dataclass(frozen=True)
class _Rung:
    """One `if SUBJECT == KEY` branch holding one statement: its rows and parts."""

    lo: int
    hi: int
    subject: bytes
    key: bytes
    body: bytes


@dataclass(frozen=True)
class _Tail:
    """What follows a ladder: `end` of the block, a `return` or Rust `tail` default, or `other` code."""

    row: int
    kind: str
    value: bytes = b""


_Row = tuple[int, bytes, int, int]  # indent steps, text, first and last source row


@dataclass(frozen=True)
class _LadderPlan:
    """Replacement rows for ladder rows `lo..hi` (each tagged with the rung it comes from), and an
    optional table inserted at row `at`."""

    lo: int
    hi: int
    rows: list[_Row]
    at: int = -1
    table: tuple[tuple[int, bytes], ...] = ()
    blanks: int = 0


def _single_statement(text: bytes, code: np.ndarray) -> bool:
    """One statement: no braces, and no `;` before its end, outside strings and comments."""
    depth_at, _ = _depths(text, code)
    last = len(text.rstrip()) - 1
    braces = any(code[c] and text[c] in b"{}" for c in range(len(text)))
    semis = any(code[c] and text[c] == 59 and depth_at[c] == 0 for c in range(last))
    return last >= 0 and not braces and not semis


def _equality(cond: bytes, code: np.ndarray) -> tuple[bytes, bytes] | None:
    """(subject, key) of `SUBJECT == KEY` when that test is the whole condition and KEY is one atom."""
    spans = _top_tokens(cond, code)
    toks = [cond[s:e] for s, e in spans]
    ops = [i for i, tok in enumerate(toks) if tok in _EQUALS]
    others = [tok for i, tok in enumerate(toks) if i not in ops and (tok in _LOGIC or tok in _SWAP)]
    if len(ops) != 1 or others:
        return None
    lo, hi = spans[ops[0]]
    k_lo, k_hi = _trim(cond, hi, len(cond))
    subject, key = cond[:lo].strip(), cond[k_lo:k_hi]
    return (subject, key) if subject and key and _atom(key, code[k_lo:k_hi]) else None


def _next_row_statement(lines: _Lines, row: int, closing: bool) -> tuple[int, bytes] | None:
    """The one deeper statement row after `row`, followed by a lone `}` when `closing`."""
    nxt = row + 1
    deeper = nxt < len(lines.text) and not lines.blank(nxt) and lines.indent(nxt) > lines.indent(row)
    if not deeper or lines.clipped[nxt]:
        return None
    end = nxt + 1 if closing else nxt
    after = lines.next_nonblank(end)
    shut = end < len(lines.text) and lines.text[end].strip() == b"}" and lines.indent(end) == lines.indent(row)
    if (closing and not shut) or (after < len(lines.text) and lines.indent(after) > lines.indent(row)):
        return None
    body = lines.text[nxt][lines.indent(nxt) :]
    single = _single_statement(body, lines.code[nxt][lines.indent(nxt) : len(lines.text[nxt])])
    return (end, body) if single else None


def _rung_body(lines: _Lines, row: int, rest: int, colon: bool) -> tuple[int, bytes] | None:
    """Last row and statement of a rung whose condition ends at column `rest`."""
    text, code = lines.text[row], lines.code[row]
    tail = _code_tail(lines, row, rest)
    lo, hi = _trim(text, rest, len(text))
    if not tail:
        return _next_row_statement(lines, row, False) if colon else None
    if not colon and tail == b"{":
        return _next_row_statement(lines, row, True)
    if not colon and text[lo] == 123:
        if _match(text, code, lo) != hi - 1:
            return None
        lo, hi = _trim(text, lo + 1, hi - 1)
    return (row, text[lo:hi]) if _single_statement(text[lo:hi], code[lo:hi]) else None


def _rung(lines: _Lines, row: int, colon: bool) -> _Rung | None:
    if not 0 <= row < len(lines.text) or lines.blank(row):
        return None
    span = _condition(lines, row, colon)
    if span is None:
        return None
    text, code = lines.text[row], lines.code[row]
    lo, hi = _trim(text, span[0], span[1])
    test = _equality(text[lo:hi], code[lo:hi])
    found = _rung_body(lines, row, span[2], colon) if test else None
    return _Rung(row, found[0], test[0], test[1], found[1]) if test and found else None


def _ladder(lines: _Lines, row: int, colon: bool) -> list[_Rung]:
    """The adjacent rungs around `row` that test one subject against distinct keys."""
    first = _rung(lines, row, colon)
    if first is None:
        return []
    start = row
    while True:
        prev = [r for r in (_rung(lines, start - size, colon) for size in (1, 2, 3)) if r and r.hi == start - 1]
        if not prev or prev[0].subject != first.subject:
            break
        start = prev[0].lo
    rungs: list[_Rung] = []
    while not rungs or rungs[-1].hi + 1 < len(lines.text):
        rung = _rung(lines, rungs[-1].hi + 1 if rungs else start, colon)
        if rung is None or rung.subject != first.subject or lines.indent(rung.lo) != lines.indent(start):
            break
        rungs.append(rung)
    keys = {rung.key for rung in rungs}
    return rungs if len(rungs) >= 2 and len(keys) == len(rungs) else []


def _ladder_tail(lines: _Lines, rungs: list[_Rung]) -> _Tail | None:
    """The row after the ladder: the block's end, a `return D`, a Rust tail `D`, or other code."""
    ind = lines.indent(rungs[0].lo)
    after = lines.next_nonblank(rungs[-1].hi)
    if after >= len(lines.text) or lines.indent(after) < ind:
        return _Tail(after, "end")
    text, code = lines.text[after], lines.code[after]
    single = _single_statement(text[ind:], code[ind : len(text)]) and not lines.clipped[after]
    if lines.indent(after) > ind or _first_word(text, ind) in _ELSE or not single:
        return None
    value = text[ind:].rstrip(b"; ")
    if _starts_word(value, _RETURN):
        return _Tail(after, "return", value[len(_RETURN) :].strip())
    closes = lines.next_nonblank(after)
    tail = _last_code_byte(lines, after) != 59 and (closes >= len(lines.text) or lines.indent(closes) < ind)
    return _Tail(after, "tail" if tail else "other", value)


def _literal(text: bytes) -> bool:
    """A number or a plain quoted string, safe to evaluate once in a table."""
    if len(text) >= 2 and text[0] in _QUOTES:
        return text[-1] == text[0] and text[0] not in text[1:-1] and 92 not in text and b"${" not in text
    return bool(text) and text[0] in _DIGITS and all(b in _WORD or b == 46 for b in text)


def _returned(body: bytes) -> bytes | None:
    """The value of a `return VALUE` statement."""
    text = body.rstrip(b"; ")
    value = text[len(_RETURN) :].strip()
    return value if _starts_word(text, _RETURN) and value else None


def _switch_rows(rungs: list[_Rung], family: str) -> list[_Row]:
    """`switch` with one `case KEY:` per rung and the rung's own statement, in Go or C-family form."""
    subject, lo, hi = rungs[0].subject, rungs[0].lo, rungs[-1].hi
    if family == "go":
        rows = [(0, b"switch " + subject + b" {", lo, hi)]
        for rung in rungs:
            rows += [(0, b"case " + rung.key + b":", rung.lo, rung.hi), (1, rung.body, rung.lo, rung.hi)]
        return [*rows, (0, b"}", lo, hi)]
    head = b"switch " + (subject if subject[:1] == b"(" else b"(" + subject + b")") + b" {"
    cases = []
    for rung in rungs:
        stop = any(_starts_word(rung.body, word) for word in _EXITS)
        cases.append((1, b"case " + rung.key + b": " + rung.body + (b"" if stop else b" break;"), rung.lo, rung.hi))
    return [(0, head, lo, hi), *cases, (0, b"}", lo, hi)]


def _case_rows(rungs: list[_Rung]) -> list[_Row] | None:
    """Python `match` with one `case KEY:` per rung and its own statement; keys must be values, not names."""
    if not all(_literal(rung.key) or 46 in rung.key for rung in rungs):
        return None
    cases = [row for r in rungs for row in ((1, b"case " + r.key + b":", r.lo, r.hi), (2, r.body, r.lo, r.hi))]
    return [(0, b"match " + rungs[0].subject + b":", rungs[0].lo, rungs[-1].hi), *cases]


def _match_rows(rungs: list[_Rung], tail: _Tail) -> list[_Row] | None:
    """Rust `match` arms `KEY => VALUE,` with the default as `_ => D,`, as the function's value."""
    values = [_returned(rung.body) for rung in rungs]
    if None in values or tail.kind not in ("return", "tail"):
        return None
    lead = b"return " if tail.kind == "return" else b""
    lo, hi = rungs[0].lo, tail.row
    arms = [(1, rung.key + b" => " + value + b",", rung.lo, rung.hi) for rung, value in zip(rungs, values)]
    head = (0, lead + b"match " + rungs[0].subject + b" {", lo, hi)
    return [head, *arms, (1, b"_ => " + tail.value + b",", hi, hi), (0, b"};" if lead else b"}", lo, hi)]


def _upper_snake(name: bytes) -> bytes:
    out = bytearray()
    for idx, byte in enumerate(name):
        if 65 <= byte <= 90 and idx and (97 <= name[idx - 1] <= 122 or name[idx - 1] in _DIGITS):
            out.append(95)
        out.append(byte - 32 if 97 <= byte <= 122 else byte)
    return bytes(out)


def _def_name(lines: _Lines, row: int) -> bytes | None:
    """The defined name: the word before the header's first `(`, else the name it binds."""
    text, code = lines.text[row], lines.code[row]
    depth_at, _ = _depths(text, code)
    paren = next((c for c in range(len(text)) if text[c] == 40 and code[c] and depth_at[c] == 0), -1)
    start = paren
    while start > 0 and text[start - 1] in _WORD:
        start -= 1
    if 0 <= start < paren:
        return text[start:paren]
    bound, _ = _binding(text, code)
    return text[bound[0][0] : bound[0][1]] if bound else None


def _gap_above(lines: _Lines, row: int) -> int:
    """Blank rows directly above `row`, when code sits above them."""
    gap = 0
    while row - gap > 0 and lines.blank(row - gap - 1):
        gap += 1
    return gap if row - gap > 0 else 0


def _table_site(lines: _Lines, row: int) -> tuple[int, int, bytes] | None:
    """Row of the top-level statement around `row`, blank rows to leave, and the function's name."""
    blocks = [(first, _block(lines, first, opener)) for first, opener in _enclosing(lines, row)]
    defs = [first for first, blk in blocks if blk is not None and _is_def(lines, blk)]
    if not defs or lines.indent(blocks[-1][0]) != 0:
        return None
    name = _def_name(lines, defs[0])
    top = blocks[-1][0]
    while top > 0 and not lines.blank(top - 1) and lines.indent(top - 1) == 0 and not lines.starts_with(top - 1, _CLOSE):
        top -= 1
    gaps = [_gap_above(lines, r) for r in range(1, len(lines.text)) if lines.indent(r) == 0 and not lines.blank(r)]
    blanks = _gap_above(lines, top) or max([g for g in gaps if g] or [1])
    return (top, min(blanks, 2), name) if name else None


def _table_plan(
    lines: _Lines, rungs: list[_Rung], tail: _Tail, family: str, syntax: _Syntax
) -> _LadderPlan | None:
    """A module-level table named after the function, and one lookup row that keeps the default."""
    values = [_returned(rung.body) for rung in rungs]
    plain = all(v and _literal(v) for v in values) and all(_literal(rung.key) for rung in rungs)
    default = tail.value if tail.kind == "return" else b""
    simple = not default or _literal(default) or all(b in _WORD for b in default) or family == "js"
    site = _table_site(lines, rungs[0].lo) if plain and simple and tail.kind in ("return", "end") else None
    if site is None:
        return None
    top, blanks, name = site
    taken = {text[s:e] for text, code in zip(lines.text, lines.code) for s, e in _words(text, code)}
    name = _fresh(_upper_snake(name), taken)
    get = name + b".get(" + rungs[0].subject
    if family == "py":
        entries = tuple((1, rung.key + b": " + value + b",") for rung, value in zip(rungs, values))
        table = ((0, name + b" = {"), *entries, (0, b"}"))
        lookup = _RETURN + b" " + get + (b", " + default if default else b"") + b")"
    else:
        entries = tuple((1, b"[" + rung.key + b", " + value + b"],") for rung, value in zip(rungs, values))
        table = ((0, b"const " + name + b" = new Map(["), *entries, (0, b"])" + syntax.semi))
        lookup = _RETURN + b" " + get + b")" + (b" ?? " + default if default else b"") + syntax.semi
    hi = tail.row if tail.kind == "return" else rungs[-1].hi
    return _LadderPlan(rungs[0].lo, hi, [(0, lookup, rungs[0].lo, hi)], top, table, blanks)


def _plan_ladder(lines: _Lines, row: int, syntax: _Syntax) -> _LadderPlan | None:
    family = _family(lines, row)
    rungs = _ladder(lines, row, family == "py")
    tail = _ladder_tail(lines, rungs) if rungs else None
    if tail is None:
        return None
    plan = _table_plan(lines, rungs, tail, family, syntax) if family in ("py", "js") else None
    if plan is not None:
        return plan
    if family == "py":
        cases = _case_rows(rungs)
        return _LadderPlan(rungs[0].lo, rungs[-1].hi, cases) if cases else None
    if family == "rust":
        arms = _match_rows(rungs, tail)
        return _LadderPlan(rungs[0].lo, tail.row, arms) if arms else None
    return _LadderPlan(rungs[0].lo, rungs[-1].hi, _switch_rows(rungs, family))


def _rung_unit(lines: _Lines, plan: _LadderPlan, syntax: _Syntax) -> int:
    """Indent step inside the ladder (a rung's body row), else the file's step."""
    deeper = [lines.indent(r) - lines.indent(plan.lo) for r in range(plan.lo, plan.hi + 1) if not lines.blank(r)]
    return min([d for d in deeper if d > 0] or [syntax.unit])


def _apply_ladder(
    hygiene: Any,
    params: dict[str, np.ndarray],
    grid: np.ndarray,
    prov: list[tuple[bool, str]],
    lprobs: np.ndarray,
    thresh: float,
    source: list[bytes],
) -> tuple[np.ndarray, list[tuple[bool, str]], np.ndarray]:
    lad_spans = _kind_spans(lprobs, 4, thresh, max_gap=2)
    if not lad_spans:
        return grid, prov, lprobs
    cells, _, _ = _extract_grid_features(hygiene, params, grid)
    lines = _Lines.of(grid, cells)
    syntax = _syntax(lines)
    plans: dict[int, _LadderPlan] = {}
    for a, _ in lad_spans:
        plan = _plan_ladder(lines, a, syntax)
        if plan is not None and not any(p.lo <= plan.hi and plan.lo <= p.hi for p in plans.values()):
            plans[plan.lo] = plan
    out: list[_Entry] = []
    row = 0
    while row < len(grid):
        for plan in (p for p in plans.values() if p.at == row):
            out.extend((step * syntax.unit, text, -1, -1, True) for step, text in plan.table)
            out.extend((0, b"", -1, -1, True) for _ in range(plan.blanks))
        plan = plans.get(row)
        ind, unit = lines.indent(row), _rung_unit(lines, plan, syntax) if plan else 0
        rows = plan.rows if plan else [(0, b"", row, row)]
        out.extend((ind + step * unit, text, lo, hi, bool(plan)) for step, text, lo, hi in rows)
        row = plan.hi + 1 if plan else row + 1
    built = _assemble(grid, prov, out) if plans else None
    if built is None:
        return grid, prov, lprobs
    lp_c, _ = hygiene._forward(params, built[0])
    return built[0], built[1], lp_c


def _apply_layout_fixes(
    hygiene: Any,
    params: dict[str, np.ndarray],
    grid: np.ndarray,
    prov: list[tuple[bool, str]],
    lprobs: np.ndarray,
    thresh: float,
    source: list[bytes],
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
    source: list[bytes],
) -> tuple[np.ndarray, list[tuple[bool, str]]]:
    """Run the rewrite steps on a window; `source` is the file's full lines, for rows the grid clipped."""
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
        cand, cand_prov, cand_lp = step(hygiene, params, grid, list(prov), lprobs, thresh, source)
        cand_rank = _rank(hygiene, params, cand, thresh)
        if cand_rank < best_rank:
            grid, prov, lprobs, best_rank = cand, cand_prov, cand_lp, cand_rank
    return grid, prov


def _window(grid: np.ndarray, flagged: np.ndarray) -> tuple[int, int]:
    """Rows to project: around the flags, reaching back to the top-level statement they sit in."""
    if len(grid) <= _MAX_DIRECT_LINES:
        return 0, len(grid)
    win_lo = max(0, int(flagged[0]) - _WINDOW_PAD)
    win_hi = min(len(grid), int(flagged[-1]) + _WINDOW_PAD + 1)
    if win_hi - win_lo > _MAX_DIRECT_LINES:
        win_hi = min(len(grid), win_lo + _MAX_DIRECT_LINES)
    for row in range(win_lo, max(-1, win_lo - _SCOPE_REACH - 1), -1):
        if _row_len(grid[row]) and grid[row, 0] > 32:
            while row > 0 and _row_len(grid[row - 1]) and grid[row - 1, 0] > 32:
                row -= 1  # decorators and comments that belong to the statement
            return row, win_hi
    return win_lo, win_hi


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
    full_grid, full_lines = hygiene.encode_byte_grid(code)
    if len(full_grid) == 0:
        return TargetProjection(target_score=1.0, target_squiggles=0, target_slice="")

    lprobs0, sc0 = hygiene._forward(params, full_grid)
    flagged = np.flatnonzero(lprobs0.max(axis=1) >= threshold)
    if len(flagged) == 0:
        return TargetProjection(target_score=sc0, target_squiggles=0, target_slice="")

    win_lo, win_hi = _window(full_grid, flagged)
    source = [bytes(min(ord(ch), 127) or 32 for ch in line) for line in full_lines]
    opt_g, opt_prov = _project_window(
        hygiene, params, full_grid[win_lo:win_hi], win_lo, threshold, source
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
    for idx in selected[:_SLICE_ROWS]:
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
