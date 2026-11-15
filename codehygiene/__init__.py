"""Weight-Space Programmed code hygiene decision model and squiggle linter."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from functools import lru_cache
import json
import os
from pathlib import Path
import sys
from typing import Any
import numpy as np

__version__ = "0.1.0"
__all__ = [
    "ARCHETYPE_NAMES", "MAX_COLS", "MAX_REL_DIST", "Result", "SCORE_NAMES",
    "SQUIGGLE_KINDS", "SQUIGGLE_MESSAGES", "Squiggle", "encode_byte_grid",
    "lint", "load_weights", "main", "render_report", "rl_reward", "score",
]

MAX_COLS = 128
MAX_REL_DIST = 32
_CODE_EXTS = frozenset({
    ".py", ".pyi", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".c", ".h",
    ".cc", ".cpp", ".hpp", ".rs", ".go", ".zig", ".java", ".cs", ".lua", ".rb",
    ".php", ".swift", ".kt", ".scala", ".sh",
})
_SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn", "__pycache__", "node_modules", "target",
    "build", "dist", "venv", ".venv", ".tox", ".mypy_cache", ".pytest_cache",
})

_SPEC = (
    ("deep_nesting", "Deep nesting; flatten with guard clauses"),
    ("poor_naming", "Over-abstracted (*Factory/*Manager) or cryptic name"),
    ("noisy_comment", "Comment restates code or leaves TODO/lint noise"),
    ("swallowed_error", "Ignored error or silent fallback value"),
    ("dense_line", "Long line, chained calls, or nested ternary"),
    ("special_casing", "Stacked hardcoded special-case branches"),
    ("duplicate_block", "Duplicated logic block"),
    ("cramped_layout", "Unbroken statement wall or inconsistent indent"),
)
SQUIGGLE_KINDS = tuple(k for k, _ in _SPEC)
SQUIGGLE_MESSAGES = dict(_SPEC)
SCORE_NAMES = (
    "flatness", "naming", "simplicity",
    "comments", "error_handling", "layout",
)
ARCHETYPE_NAMES = (
    "minimal", "clean", "enterprise",
    "dense", "slop",
)
_LEVELS = np.array([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0], dtype=np.float32)
_ANSI = ("\033[1m", "\033[2m", "\033[33m", "\033[36m", "\033[32m", "\033[31m", "\033[0m")


@dataclass
class Squiggle:
    line: int
    col_start: int
    col_end: int
    probability: float
    kind: str
    message: str


@dataclass
class Result:
    score: float
    rl_hygiene: float
    scores: dict[str, float]
    archetype: str
    archetype_probs: dict[str, float]
    squiggles: list[Squiggle]
    line_probs: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        sqs = [{**asdict(s), "probability": round(s.probability, 4)} for s in self.squiggles]
        return {
            "score": round(self.score, 4),
            "rl_hygiene": round(self.rl_hygiene, 4),
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
            "archetype": self.archetype,
            "archetype_probs": {k: round(v, 4) for k, v in self.archetype_probs.items()},
            "squiggles": sqs,
        }


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _softmax(x: np.ndarray) -> np.ndarray:
    ex = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return ex / np.sum(ex, axis=-1, keepdims=True)


def _l2_norm(x: np.ndarray, eps: float = 2.5) -> np.ndarray:
    return x / np.sqrt(np.sum(x * x, axis=-1, keepdims=True) + eps)


def _conv1d(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    l_len, w_len, _ = x.shape
    k, _, c_out = w.shape
    pad = (k - 1) // 2
    xp = np.pad(x, ((0, 0), (pad, k - 1 - pad), (0, 0))) if k > 1 else x
    out = np.empty((l_len, w_len, c_out), dtype=np.float32)
    out[:] = b
    for ki in range(k):
        out += xp[:, ki : ki + w_len, :] @ w[ki]
    return out


def _sink_attn(
    z: np.ndarray, wq: np.ndarray, wk: np.ndarray,
    wv: np.ndarray, wo: np.ndarray, b_rel: np.ndarray, b_sink: np.ndarray,
) -> np.ndarray:
    """Multi-head relative-offset attention across lines with an unmatched-query sink logit."""
    n_l, n_h = z.shape[0], b_sink.shape[0]
    dk = wq.shape[1] // n_h
    q, k, v = ((z @ w).reshape(n_l, n_h, -1).transpose(1, 0, 2) for w in (wq, wk, wv))

    idx = np.arange(n_l)
    rel = np.clip(idx[None, :] - idx[:, None], -MAX_REL_DIST, MAX_REL_DIST) + MAX_REL_DIST
    scores = (q @ k.transpose(0, 2, 1)) / np.sqrt(float(dk)) + b_rel[:, rel]
    sink = np.broadcast_to(b_sink[:, None, None], (n_h, n_l, 1))
    attn = _softmax(np.concatenate([scores, sink], axis=-1))[:, :, :n_l]
    return (attn @ v).transpose(1, 0, 2).reshape(n_l, -1) @ wo


def encode_byte_grid(code: str, max_lines: int = 512) -> tuple[np.ndarray, list[str]]:
    """Convert raw source text into a (lines, 128) uint8 grid of 7-bit ASCII codes."""
    lines = (code.expandtabs(2).splitlines() or [""])[:max_lines]
    grid = np.zeros((len(lines), MAX_COLS), dtype=np.uint8)
    for i, ln in enumerate(lines):
        if ln:
            arr = np.frombuffer(ln.encode("ascii", "replace")[:MAX_COLS], dtype=np.uint8) & 0x7F
            grid[i, : len(arr)] = np.where(arr == 0, 32, arr)
    return grid, lines


def _forward(p: dict[str, np.ndarray], x: np.ndarray) -> tuple[np.ndarray, ...]:
    # Block 0: character + column embeddings and left-to-right string/comment state masks -> hc (L, W, 52)
    wc = x.shape[1]
    e = p["tok_emb"][x] + p["col_emb"][None, :wc, :]
    p0 = _relu(_conv1d(e, p["pre_conv3_w"], p["pre_conv3_b"]))
    c1 = p["W_causal1"][:wc, :wc] @ (np.concatenate([e, p0], axis=-1) @ p["pref1_proj"])
    g1 = _relu(np.concatenate([e, p0, c1], axis=-1) @ p["gate1_w"] + p["gate1_b"])
    c2 = p["W_causal2"][:wc, :wc] @ (g1 @ p["pref2_proj"])
    hc = _relu(np.concatenate([e, p0, g1, c2], axis=-1) @ p["col_fuse_w"] + p["col_fuse_b"])

    # Block 1: k=3/7/13 lexical convolutions + 40-dim code/comment SimHash sketches -> z (L, 192)
    k3 = _relu(_conv1d(hc, p["conv3_w"], p["conv3_b"])).sum(axis=1)
    k7 = _relu(_conv1d(hc, p["conv7_w"], p["conv7_b"])).sum(axis=1)
    k13 = _relu(_conv1d(hc, p["conv13_w"], p["conv13_b"])).sum(axis=1)
    csum = hc.sum(axis=1) @ p["col_sum_proj"]

    sk = _relu(_conv1d(hc, p["sketch_w"], p["sketch_b"])).sum(axis=1)
    skb = _relu(sk) - _relu(sk - 1.0)
    u = np.concatenate([csum, k3, k7, k13], axis=-1)
    ml = _relu(u @ p["lmlp1_w"] + p["lmlp1_b"]) @ p["lmlp2_w"] + p["lmlp2_b"]
    z = np.concatenate([ml, _l2_norm(skb[:, :40]), _l2_norm(skb[:, 40:])], axis=-1)

    # Block 2: two cross-line sink-attention + context-MLP passes over [-32..+32] line offsets
    for k in ("1", "2"):
        z = z + _sink_attn(
            z, p[f"attn{k}_wq"], p[f"attn{k}_wk"], p[f"attn{k}_wv"],
            p[f"attn{k}_wo"], p[f"attn{k}_brel"], p[f"attn{k}_bsink"],
        )
        z = z + _relu(z @ p[f"cmlp{k}_w"] + p[f"cmlp{k}_b"]) @ p[f"cmlp{k}_2_w"] + p[f"cmlp{k}_2_b"]

    # Block 3: per-line squiggle probabilities + active-line pooling into System One heads
    line_p = _sigmoid(z @ p["sq_w"] + p["sq_b"])
    gate = _sigmoid(z @ p["pool_gate_w"] + p["pool_gate_b"])
    z_mean = (z * gate).sum(axis=0) / (gate.sum(axis=0) + 1e-5)
    z_doc = np.concatenate([z_mean, (_relu(z) * gate).max(axis=0)], axis=-1)
    logits = _relu(z_doc @ p["doc1_w"] + p["doc1_b"]) @ p["doc2_w"] + p["doc2_b"]

    nouls = _sigmoid(logits[:2])
    s_vals = (_softmax(logits[2:26].reshape(6, 4)) * _LEVELS[None, :]).sum(axis=-1)
    return line_p, nouls, s_vals, _softmax(logits[26:31])


@lru_cache(maxsize=2)
def load_weights(path: str | Path | None = None) -> dict[str, np.ndarray]:
    target = Path(path) if path else Path(__file__).with_name("weights.npz")
    with np.load(target) as d:
        return {k: d[k].astype(np.float32) for k in d.files}


def score(code: str, *, threshold: float = 0.55, weights: dict[str, np.ndarray] | None = None) -> Result:
    x, lines = encode_byte_grid(code)
    line_p, nouls, s_vals, c_probs = _forward(weights or load_weights(), x)
    sqs: list[Squiggle] = []
    for i, raw in enumerate(lines):
        bk = int(np.argmax(line_p[i]))
        pr = float(line_p[i, bk])
        if pr < threshold:
            continue
        st = raw.lstrip(" ")
        c0 = len(raw) - len(st) + 1 if st else 1
        kind = SQUIGGLE_KINDS[bk]
        sqs.append(Squiggle(i + 1, c0, max(c0, len(raw)), pr, kind, SQUIGGLE_MESSAGES[kind]))

    return Result(
        score=float(nouls[0]),
        rl_hygiene=float(nouls[1]),
        scores=dict(zip(SCORE_NAMES, map(float, s_vals))),
        archetype=ARCHETYPE_NAMES[int(np.argmax(c_probs))],
        archetype_probs=dict(zip(ARCHETYPE_NAMES, map(float, c_probs))),
        squiggles=sqs,
        line_probs=line_p,
    )


def lint(code: str, *, threshold: float = 0.55) -> list[Squiggle]:
    return score(code, threshold=threshold).squiggles


def rl_reward(
    code: str, tests_passed: bool | float = True, *,
    weight: float = 0.25, hygiene_blend: float = 0.5,
) -> float:
    gate = float(tests_passed)
    if gate <= 0.0:
        return 0.0
    res = score(code)
    return gate * (1.0 + weight * ((1.0 - hygiene_blend) * res.score + hygiene_blend * res.rl_hygiene))


def render_report(
    label: str, src: str, thresh: float = 0.55,
    color: bool = True, res: Result | None = None,
) -> str:
    _, lines = encode_byte_grid(src, max_lines=100_000)
    res = res or score(src, threshold=thresh)
    b, d, y, c, g, r, z = _ANSI if color else ("",) * 7
    sc = (r, y, g)[int(res.score >= 0.40) + int(res.score >= 0.70)]
    pct = res.archetype_probs[res.archetype] * 100
    out = [
        f"{b}{label}{z}",
        f"  score: {sc}{res.score:.3f}{z}  rl_hygiene: {c}{res.rl_hygiene:.3f}{z}  "
        f"style: {b}{res.archetype}{z} ({pct:.0f}%)",
    ]
    for k, v in res.scores.items():
        n = int(round(max(0.0, min(1.0, v)) * 16))
        out.append(f"    {k:15s} {'#' * n + '.' * (16 - n)} {v:.2f}")
    if not res.squiggles:
        return "\n".join([*out, f"\n  {g}ok - no squiggles (threshold {thresh:.2f}){z}"])

    out.append(f"\n  {y}{len(res.squiggles)} squiggle(s):{z}")
    for sq in res.squiggles:
        ln = lines[sq.line - 1]
        pad = " " * (sq.col_start - 1)
        wavy = "~" * max(1, min(len(ln) - sq.col_start + 1, 72))
        tag = f"[{sq.kind} {sq.probability:.2f}]"
        out.append(f"  {d}{sq.line:4d} |{z} {ln}")
        out.append(f"       {d}|{z} {pad}{y}{wavy} {tag}{z} {d}{sq.message}{z}")
    return "\n".join(out)


def _walk_dir(root: Path) -> list[Path]:
    found: list[Path] = []
    for cur, dirs, files in os.walk(root):
        dirs[:] = sorted(
            d for d in dirs
            if not d.startswith(".") and d not in _SKIP_DIRS and not d.endswith(".egg-info")
        )
        base = Path(cur)
        found.extend(base / f for f in sorted(files) if Path(f).suffix.lower() in _CODE_EXTS)
    return found


def _expand_paths(specs: list[str]) -> list[Path]:
    out: list[Path] = []
    for s in specs:
        pts = sorted(Path().glob(s)) if any(c in s for c in "*?[") else [Path(s)]
        for p in pts:
            out.extend(_walk_dir(p) if p.is_dir() else [p])
    return out


def main(argv: list[str] | None = None) -> int:
    desc = "33 KB weight-space programmed System One model for linting code aesthetics."
    ap = argparse.ArgumentParser(prog="codehygiene", description=desc)
    ap.add_argument("files", nargs="*", help="Files, directories, or globs (stdin if piped)")
    ap.add_argument("-t", "--threshold", type=float, default=0.55, help="Squiggle threshold")
    ap.add_argument("--check", action="store_true", help="Exit 1 when squiggles are found")
    ap.add_argument("--json", action="store_true", help="Output structured JSON")
    ap.add_argument("--no-color", action="store_true", help="Disable ANSI colors")
    ap.add_argument("-V", "--version", action="version", version=f"%(prog)s {__version__}")
    args = ap.parse_args(argv)

    if not args.files and sys.stdin.isatty():
        ap.print_help()
        return 0
    paths = _expand_paths(args.files)
    targets = [(p.as_posix(), p.read_text("utf-8", "replace")) for p in paths]
    if not args.files:
        targets = [("<stdin>", sys.stdin.read())]
    if not targets:
        return 0

    runs = [(k, s, score(s, threshold=args.threshold)) for k, s in targets]
    exit_code = 1 if (args.check and any(r.squiggles for _, _, r in runs)) else 0
    if args.json:
        out = [r.to_dict() for _, _, r in runs]
        single = len(out) == 1 and (not args.files or not Path(args.files[0]).is_dir())
        body = out[0] if single else dict(zip((k for k, _, _ in runs), out))
        print(json.dumps(body, indent=2))
        return exit_code

    reports = [render_report(k, s, args.threshold, not args.no_color, r) for k, s, r in runs]
    print("\n\n".join(reports))
    return exit_code
