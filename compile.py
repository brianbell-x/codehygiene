"""Closed-form Weight-Space Program compiling polyglot circuits into weights.npz.

Architecture & Residual Layout (Z has 192 channels per line):
  - Block 0 (_build_block0): Maps (L, 128) ASCII bytes + column positions into 52 per-cell
    channels (HC_*), using causal prefix matrices to mask string literals (16-quote sawtooth)
    and polyglot comments (//, /* */, # , -- ) left-to-right across each line.
  - Block 1 (_build_block1): Runs k=3, 7, 13 1D convolutions for operators, control-flow
    keywords, and ceremonial suffixes, plus 40 Rademacher 4-gram SimHash filters (_sketch_bank)
    for live code (Z[112:152]) and inline comments (Z[152:192]), projecting line stats into Z[0:40].
  - Block 2 (_build_block2_and_heads): Two cross-line sink-attention + MLP passes populate
    relational context in Z[40:59], bounded [0, 1] issue flags in Z[64:77], and neighborhood-
    smoothed flags in Z[80:86].
  - Block 3 (_build_block2_and_heads): Projects Z into 8 per-line squiggle probabilities (sq_w)
    and pools active lines into 31 document logits (2 scores, 6x4 sub-scores, 5 style classes).
"""

from __future__ import annotations

from pathlib import Path
import numpy as np

from codehygiene import MAX_COLS, MAX_REL_DIST

_B7 = np.arange(7)
_STOP = (
    "this that with from have will when then else true fals null cons onst func unct "
    "ctio tion retu etur turn thro hrow type ypeo peof numb stri trin ring unde ndef "
    "defi efin fine ined only also more some into over must empt leng ngth erro rror "
    "expo xpor modu odul data chan self none elif pass"
).split()

# HC channels (52 dims)
(
    HC_ACT, HC_COMM, HC_CODE, HC_CNS, HC_MNS, HC_IND, HC_FNS, HC_ODD,
    HC_M42, HC_M40, HC_D8, HC_D10, HC_D12, HC_G80, HC_G96, HC_ID,
    HC_LO, HC_UP, HC_DIG, HC_UND, HC_DOT, HC_QM, HC_COL, HC_EQ,
    HC_LB, HC_RB, HC_LP, HC_RP, HC_SEM, HC_PA, HC_QT, HC_LET,
) = range(32)
HC_BP, HC_BN, HC_G110, HC_SL_C = 32, 39, 46, 47
HC_STR, HC_LEAD, HC_D16, HC_D20 = 48, 49, 50, 51
N_HC = 52


def _sub_filter(
    w: np.ndarray, b: np.ndarray, ch: int, pat: str, *,
    code: bool = True, comm: bool = False, lead: int = 0, no_str: bool = True,
    wl: int = 0, wr: int = 0, cs: int = 0, gain: float = 12.5,
) -> None:
    """Write an exact ASCII bit-template into conv kernel channel `ch` for substring `pat`."""
    s0 = (w.shape[0] - (len(pat) + wl + wr)) // 2 + wl
    w[:, :, ch] = 0.0
    th = float(len(pat))
    if wl:
        w[s0 - 1, HC_ID, ch] = -2.0
    if wr:
        w[s0 + len(pat), HC_ID, ch] = -2.0
    if lead:
        w[s0, HC_LEAD, ch] += 1.0
        th += 1.0

    for i, c in enumerate(pat):
        on = (((ord(c.lower()) & 0x7F) >> _B7) & 1) == 1
        w[s0 + i, HC_BP + _B7[on], ch] += 1.0 / 7.0
        w[s0 + i, HC_BN + _B7[~on], ch] += 1.0 / 7.0
        if cs and c.isalpha():
            hi, lo = (HC_UP, HC_LO) if c.isupper() else (HC_LO, HC_UP)
            w[s0 + i, [hi, lo], ch] += [0.25, -0.50]
            th += 0.25

    mid = s0 + len(pat) // 2
    if code:
        w[mid, HC_COMM, ch] -= 5.0
        if no_str:
            w[mid, HC_STR, ch] -= 5.0
    if comm:
        w[mid, HC_CODE, ch] -= 5.0
    b[ch] = -(th - 0.08) * gain
    w[:, :, ch] *= gain


def _encode_4g(word: str) -> np.ndarray:
    raw = np.array([ord(c) & 0x1F for c in word[:4]], dtype=np.int32)[:, None]
    return np.where(((raw >> np.arange(5)) & 1) == 1, 1.0, -1.0).astype(np.float32)


def _sketch_bank() -> tuple[np.ndarray, np.ndarray]:
    """Build 40 near-orthogonal 4-char SimHash filters (0..39: live code, 40..79: comments)."""
    sw, sb = np.zeros((5, N_HC, 80), dtype=np.float32), np.zeros((80,), dtype=np.float32)
    stop_v = [_encode_4g(w) for w in _STOP]
    rng = np.random.RandomState(20261115)
    chosen: list[np.ndarray] = []

    while len(chosen) < 40:
        cand = rng.choice([-1.0, 1.0], size=(4, 5)).astype(np.float32)
        hits = sum(float(np.sum((sv * cand + 1.0) * 0.5)) >= 14.0 for sv in stop_v)
        far = all(float(np.abs(np.sum(cand * p))) <= 8.0 for p in chosen)
        if hits <= 1 and far:
            chosen.append(cand)

    for idx, filt in enumerate(chosen):
        pos = (filt > 0).astype(np.float32) / 20.0
        neg = (filt < 0).astype(np.float32) / 20.0
        sw[:4, HC_LET, [idx, 40 + idx]] = 2.0
        sw[:4, HC_UND, [idx, 40 + idx]] = 2.0
        for ch in (idx, 40 + idx):
            sw[:4, HC_BP : HC_BP + 5, ch] = pos
            sw[:4, HC_BN : HC_BN + 5, ch] = neg
        sw[1, [HC_COMM, HC_STR], idx] = -6.0
        sw[1, HC_SL_C, 40 + idx] = 2.0
        sb[[idx, 40 + idx]] = [-(8.0 + 13.7 / 20.0) * 6.0, -(10.0 + 13.7 / 20.0) * 6.0]
        sw[:, :, [idx, 40 + idx]] *= 6.0
    return sw, sb


def _build_block0(p: dict[str, np.ndarray]) -> None:
    """Compile byte/column embeddings and left-to-right quote & comment masking circuits."""
    tok = np.zeros((128, 46), dtype=np.float32)
    tok[1:, 0] = 1.0
    tok[33:, 1] = 1.0
    tok[32, 2] = 1.0
    tok[97:123, 3] = 1.0
    tok[65:91, 4] = 1.0
    tok[48:58, 5] = 1.0

    for b in range(1, 128):
        ch = chr(b)
        tok[b, 6] = float(ch.isalnum() or ch in "_$#")
        tok[b, 21:28] = ((ord(ch.lower()) & 0x7F) >> _B7) & 1
    for ch, col_i in zip("_/*:?.={}();", range(7, 19)):
        tok[ord(ch), col_i] = 1.0
    tok[[ord("|"), ord("&")], 19] = 1.0
    tok[[ord("'"), ord('"'), ord("`")], 20] = 1.0
    for ch, col_i in zip('"\'\\#-', range(40, 45)):
        tok[ord(ch), col_i] = 1.0
    tok[[ord("&"), ord("<")], 45] = 1.0

    col = np.zeros((MAX_COLS, 46), dtype=np.float32)
    ci = np.arange(MAX_COLS)
    col[0, 28] = 1.0
    col[ci % 2 == 1, 29] = 1.0
    col[ci % 4 == 2, 30] = 1.0
    col[(ci > 0) & (ci % 4 == 0) & (ci <= 28), 31] = 1.0
    for thr, dst in zip((8, 10, 12, 80, 96, 110, 16, 20), range(32, 40)):
        col[ci >= thr, dst] = 1.0
    p["tok_emb"], p["col_emb"] = tok, col

    pre_w, pre_b = np.zeros((3, 46, 16), dtype=np.float32), np.zeros((16,), dtype=np.float32)
    pre_w[[1, 2], 8, 0] = 2.0
    pre_w[0, [10, 9, 20], 0] = -4.0
    pre_w[[1, 2], [8, 9], 1] = 2.0
    pre_w[[0, 1, 0], [9, 8, 8], 2] = [2.0, 2.0, -4.0]
    pre_b[:3] = -3.0

    # ch 3: '# ' comment; ch 4: '-- ' comment; ch 5: '"'; ch 6: "'"; ch 7: a'b contraction
    pre_w[0, 1, 3] = -4.0
    pre_w[[1, 2], [43, 2], 3] = 2.0
    pre_b[3] = -3.0

    pre_w[[0, 1, 2], [44, 44, 2], 4] = 2.0
    pre_b[4] = -5.0

    pre_w[1, 40, 5] = 1.0
    pre_w[0, [42, 41], 5] = -2.0
    pre_w[2, 41, 5] = -2.0

    pre_w[1, 41, 6] = 1.0
    pre_w[0, [42, 45, 40], 6] = -2.0
    pre_w[2, 40, 6] = -2.0

    pre_w[0, [3, 4], 7] = 1.0
    pre_w[1, 41, 7] = 1.0
    pre_w[2, [3, 4], 7] = 1.0
    pre_b[7] = -2.0
    p["pre_conv3_w"], p["pre_conv3_b"] = pre_w, pre_b

    # [e(46), p0(16)] -> 62 dims; p0 starts at 46
    pf1 = np.zeros((62, 4), dtype=np.float32)
    pf1[[1, 51, 52, 53, 47], [0, 1, 2, 2, 3]] = [1.0, 1.0, 1.0, -1.0, 1.0]
    p["pref1_proj"] = pf1
    p["W_causal1"] = np.tril(np.ones((MAX_COLS, MAX_COLS), dtype=np.float32), k=-1)
    p["W_causal2"] = np.tril(np.ones((MAX_COLS, MAX_COLS), dtype=np.float32), k=0)

    # [e(46), p0(16), c1(4)] -> 66 dims; p0 at 46..61, c1 at 62..65
    g1_w, g1_b = np.zeros((66, 50), dtype=np.float32), np.zeros((50,), dtype=np.float32)
    g1_w[[2, 1], [0, 1]] = 1.0
    g1_w[62, [0, 1]] = -2.0
    g1_w[[1, 9, 62], 2] = [2.0, 2.0, -4.0]
    g1_b[2] = -3.0
    g1_w[[46, 47, 49, 50], 3] = 1.0
    g1_w[48, [4, 48]] = 10.0
    g1_w[65, [4, 48]] = 1.0
    g1_b[[4, 48]] = [-10.0, -11.0]
    g1_w[[46, 49, 50], 13] = 1.0
    for gi, c_ch in enumerate((29, 30, 31, 32, 33, 34, 38, 39), 5):
        g1_w[[1, c_ch, 9, 62], gi] = [2.0, 2.0, -4.0, -4.0]
        g1_b[gi] = -3.0

    g1_w[62, [14, 15]] = -1.0
    g1_b[[14, 15]] = [2.0, 1.0]
    for m in range(16):
        g1_w[63, 16 + m] = 1.0
        g1_b[16 + m] = -float(m)
        g1_w[64, 32 + m] = 1.0
        g1_b[32 + m] = -float(m)
    p["gate1_w"], p["gate1_b"] = g1_w, g1_b

    pf2 = np.zeros((50, 4), dtype=np.float32)
    pf2[[2, 3, 4, 48, 13], [0, 0, 0, 0, 1]] = [1.0, 1.0, -1.0, 1.0, 1.0]
    p["pref2_proj"] = pf2

    # [e(46), p0(16), g1(50), c2(4)] -> 116 dims; g1 at 62..111, c2 at 112..115
    cf_w, cf_b = np.zeros((116, N_HC), dtype=np.float32), np.zeros((N_HC,), dtype=np.float32)
    cf_w[0, [HC_ACT, HC_COMM, HC_SL_C, HC_CODE]] = 1.0
    cf_w[112, [HC_COMM, HC_CODE, HC_CNS]] = [1.0, -2.0, -2.0]
    cf_w[113, [HC_SL_C, HC_MNS]] = 1.0
    cf_w[1, [HC_CNS, HC_MNS]] = 1.0
    cf_w[8, HC_MNS] = -1.0
    cf_b[[HC_COMM, HC_SL_C, HC_MNS]] = -1.0
    cf_w[[62, 63], [HC_IND, HC_FNS]] = 1.0

    saw = np.array([1.0] + [-2.0, 2.0] * 7 + [-2.0], dtype=np.float32)
    cf_w[78:94, HC_STR] = saw
    cf_w[94:110, HC_STR] = saw
    cf_w[112, HC_STR] = -3.0
    cf_w[[1, 76, 77, 112], HC_LEAD] = [1.0, 1.0, -1.0, -2.0]
    cf_b[HC_LEAD] = -1.0

    ind_dsts = [HC_ODD, HC_M42, HC_M40, HC_D8, HC_D10, HC_D12, HC_D16, HC_D20]
    for gs, hd in zip(range(67, 75), ind_dsts):
        cf_w[[gs, 112], hd] = [1.0, -2.0]
    for cs_c, hd in ((35, HC_G80), (36, HC_G96), (37, HC_G110)):
        cf_w[[1, cs_c, 112], hd] = [1.0, 1.0, -2.0]
        cf_b[hd] = -1.0

    e_cols = (6, 3, 4, 5, 7, 12, 11, 10, 13, 14, 15, 16, 17, 18, 19, 20)
    cf_w[list(e_cols), list(range(HC_ID, HC_QT + 1))] = 1.0
    cf_w[[3, 4], HC_LET] = 1.0
    cf_w[21 + _B7, HC_BP + _B7] = 1.0
    cf_w[0, HC_BN + _B7] = 1.0
    cf_w[21 + _B7, HC_BN + _B7] = -1.0
    p["col_fuse_w"], p["col_fuse_b"] = cf_w, cf_b


_K7_CODE = (
    (0, "if", 1, 1, 1, 1), (1, "else", 1, 1, 1, 1), (2, "for", 1, 1, 1, 1),
    (3, "while", 1, 1, 1, 1), (4, "switch", 0, 1, 1, 1), (5, "return", 0, 1, 1, 0),
    (6, "throw", 1, 1, 1, 0), (7, "break", 1, 1, 1, 0), (8, "catch", 1, 1, 1, 0),
    (9, "try", 1, 1, 1, 0), (10, "typeof", 0, 1, 1, 0), (11, ".map(", 0, 0, 1, 0),
    (12, ".filter", 0, 0, 1, 0), (13, ".reduce", 0, 0, 1, 0), (14, ".flatMa", 0, 0, 1, 0),
    (15, ".code", 0, 1, 1, 0), (16, "reject(", 0, 0, 1, 0), (17, "cb(", 1, 0, 1, 0),
    (18, "emit(", 0, 0, 1, 0), (19, "destroy", 0, 0, 1, 0), (20, "|| {}", 0, 0, 0, 0),
    (21, "|| []", 0, 0, 0, 0), (22, "|| '')", 0, 0, 0, 0), (23, "console", 0, 0, 1, 0),
    (24, "null;", 1, 0, 1, 0), (25, "== '", 0, 0, 0, 0), (26, '== "', 0, 0, 0, 0),
    (27, "case ", 1, 0, 1, 1), (28, "false;", 1, 0, 1, 0), (29, "catch (", 0, 0, 1, 0),
    (30, "== -1", 0, 0, 0, 0), (31, "Manager", 0, 0, 1, 0), (32, "Factory", 0, 0, 1, 0),
    (33, "Service", 0, 0, 1, 0), (34, "Builder", 0, 0, 1, 0), (35, "Wrapper", 0, 0, 1, 0),
    (36, "Helper", 0, 0, 1, 0), (37, "Impl", 0, 1, 1, 0), (38, "Default", 0, 0, 1, 0),
    (39, "Context", 0, 0, 1, 0), (40, "ERR_", 1, 0, 1, 0), (41, "validat", 0, 0, 0, 0),
    (42, "require", 0, 0, 1, 0), (43, "exports", 0, 0, 1, 0), (44, "strict'", 0, 0, 1, 0),
    (45, "SAFE_IN", 0, 0, 1, 0), (60, "elif", 1, 1, 1, 1), (61, "else if", 0, 0, 1, 1),
    (62, "match", 1, 1, 1, 1), (63, "except:", 0, 0, 1, 1), (64, "raise", 1, 1, 1, 0),
    (65, "pass", 1, 1, 1, 1), (66, "None", 1, 1, 1, 0), (67, "0.0", 1, 1, 0, 0),
    (68, " or []", 0, 0, 1, 0), (69, " or {}", 0, 0, 1, 0), (70, ".unwrap", 0, 0, 1, 0),
    (71, "type(", 1, 0, 1, 0), (72, "from ", 1, 0, 1, 1), (73, "use ", 1, 0, 1, 1),
    (74, "assert(", 0, 0, 0, 0),
)

_K13_CODE = (
    (1, "Abstract", 1), (2, "Repository", 1), (3, "Configuration", 1),
    (4, "Controller", 1), (5, "Strategy", 1), (6, "Provider", 1), (7, "Processor", 1),
    (8, "Prototype", 1), (9, "primordials", 1), (10, "HideStackFram", 1),
    (11, "internal/", 0), (12, "continue", 1), (13, "callback(", 0),
    (14, "=== undefined", 0), (15, "!== undefined", 0), (16, "EventEmitter", 1),
    (17, "#[allow(", 1), (18, "import ", 1), (19, "package ", 1), (22, "#include", 1),
    (23, "except Excep", 1), (24, "except BaseE", 1),
)


def _build_block1(p: dict[str, np.ndarray]) -> None:
    """Compile k=3, 7, 13 lexical convolutions, SimHash bank, and line-level feature MLP."""
    # k=3: snake_case/camelCase transitions, isolated single-letter identifiers, ternaries, operators
    c3_w, c3_b = np.zeros((3, N_HC, 64), dtype=np.float32), np.zeros((64,), dtype=np.float32)
    c3_w[[0, 1, 2], [HC_LO, HC_UND, HC_LO], 0] = 2.0
    c3_w[[0, 1], [HC_LO, HC_UP], 1] = 2.0
    c3_w[0, [HC_ID, HC_DOT, HC_QT], 2] = -4.0
    c3_w[1, HC_LO, 2] = 2.0
    c3_w[2, [HC_ID, HC_QT], 2] = -4.0
    c3_w[[0, 1, 2], [HC_ID, HC_LO, HC_ID], 3] = 2.0
    c3_w[0, [HC_ID, HC_QM], 4] = [-2.0, -4.0]
    c3_w[1, HC_QM, 4] = 2.0
    c3_w[2, [HC_DOT, HC_QM, HC_EQ, HC_SEM, HC_RP], 4] = -4.0
    c3_w[1, HC_COMM, :5] = -6.0
    c3_w[1, HC_STR, :5] = -6.0
    c3_b[:5] = [-5.0, -3.0, -1.0, -5.0, -1.0]

    for ci_3, pat in enumerate(("?.", "=>", "===", "{}", "||", "#", "??="), 5):
        _sub_filter(c3_w, c3_b, ci_3, pat)
    c3_w[1, [HC_SEM, HC_EQ, HC_COMM, HC_STR], 12] = [1.0, 1.0, -5.0, -5.0]
    c3_w[1, HC_D16, 13] = 1.0
    c3_w[1, HC_D20, 14] = 1.0
    c3_w[[0, 1], [HC_ID, HC_LP], 15] = 2.0
    c3_w[1, [HC_COMM, HC_STR], 15] = -6.0
    c3_b[15] = -3.0
    p["conv3_w"], p["conv3_b"] = c3_w, c3_b

    c7_w, c7_b = np.zeros((7, N_HC, 96), dtype=np.float32), np.zeros((96,), dtype=np.float32)
    for idx, pat, wl, wr, cs, ld in _K7_CODE:
        ns = idx not in (22, 44)
        _sub_filter(c7_w, c7_b, idx, pat, wl=wl, wr=wr, cs=cs, lead=ld, no_str=ns)
    c7_w[[1, 2, 3, 5], [HC_EQ, HC_EQ, HC_EQ, HC_DIG], 46] = 2.0
    c7_w[2, [HC_COMM, HC_STR], 46] = -8.0
    c7_b[46] = -7.0

    k7_comm = (
        "step 1", "step 2", "step 3", "todo:", "fixme:",
        "eslint-", "@ts-ign", "initial", "loop th", "check i",
    )
    for idx, pat in enumerate(k7_comm, 50):
        _sub_filter(c7_w, c7_b, idx, pat, code=False, comm=True)
    for idx, pat in enumerate(("type: i", "noqa", "nolint"), 78):
        _sub_filter(c7_w, c7_b, idx, pat, code=False, comm=True)
    p["conv7_w"], p["conv7_b"] = c7_w, c7_b

    c13_w, c13_b = np.zeros((13, N_HC, 64), dtype=np.float32), np.zeros((64,), dtype=np.float32)
    c13_w[:, HC_ID, 0] = 2.0
    c13_w[6, [HC_COMM, HC_STR], 0] = -30.0
    c13_b[0] = -25.0
    for idx, pat, cs in _K13_CODE:
        ld = int(idx in (18, 19, 22, 23, 24))
        _sub_filter(c13_w, c13_b, idx, pat, cs=cs, lead=ld, no_str=(idx != 11))
    _sub_filter(c13_w, c13_b, 20, "eslint-disabl", code=False, comm=True)
    _sub_filter(c13_w, c13_b, 21, "@ts-ignore", code=False, comm=True)
    p["conv13_w"], p["conv13_b"] = c13_w, c13_b
    p["sketch_w"], p["sketch_b"] = _sketch_bank()

    csp = np.zeros((N_HC, 16), dtype=np.float32)
    rows = [
        HC_CNS, HC_MNS, HC_IND, HC_ODD, HC_M42, HC_M40, HC_D8, HC_D10,
        HC_D12, HC_G80, HC_G96, HC_G110, HC_LB, HC_RB, HC_COMM, HC_FNS,
    ]
    csp[rows, range(16)] = [1, 1, 0.5, 1, 1, 1, 1, 1, 1, 1 / 16, 1 / 12, 1 / 8, 1, 1, 1, 1]
    p["col_sum_proj"] = csp

    lm1_w, lm1_b = np.zeros((240, 96), dtype=np.float32), np.zeros((96,), dtype=np.float32)
    lm2_w, lm2_b = np.zeros((96, 112), dtype=np.float32), np.zeros((112,), dtype=np.float32)
    decl = [(152, -3.0), (153, -3.0), (194, -3.0), (195, -3.0), (198, -3.0)]
    wt = [
        (80, 1.5), (85, 1.0), (105, 1.5), (106, 1.5), (126, 1.5), (110, 1.5),
        (25, -2.5), (90, -2.5), (151, -2.5), (95, -3.0), (190, -3.0),
    ]
    ct = [
        (111, 1.0), (112, 1.2), (113, 0.9), (114, 0.8), (115, 1.0), (116, 0.9), (117, 1.1),
        (177, 1.3), (178, 1.1), (179, 0.8), (180, 0.9), (181, 1.0), (182, 0.9), (183, 0.9),
    ]
    it = [(184, 12.0), (120, 10.0), (125, 10.0), (186, 12.0), (185, 8.0), (192, 12.0), (154, 8.0)]
    st = [(28, 1.0), (31, 1.0), (85, 1.0), (86, 1.0), (144, 1.0)] + decl
    im = [
        (0, 0.05), (28, 0.65), (31, 0.65), (2, 0.25), (120, -3.0), (184, -3.0),
        (122, -3.0), (123, -3.0), (85, -0.8), (86, -0.8), (144, -0.8),
    ] + decl

    rules = (
        (0, [(15, 1.0)], 0.0, 0, 1.0),
        (1, [(0, 1.0)], 0.0, 1, 1.0), (2, [(0, 1.0)], -1.0, 1, -1.0),
        (3, [(1, 1.0), (0, -5.0)], -2.0, 2, 1.0), (4, [(1, 1.0), (0, -5.0)], -3.0, 2, -1.0),
        (43, [(1, -0.25), (0, -5.0)], 11.0, 30, 1.0),
        (44, [(1, -0.25), (0, -5.0)], 10.0, 30, -1.0),
        (5, [(14, 1.0), (0, -5.0)], -2.0, 34, 1.0),
        (6, [(14, 1.0), (0, -5.0)], -3.0, 34, -1.0),
        (48, [(15, -1.0)], 1.0, 34, 1.0),
        (7, [(12, 1.0), (13, 1.0), (0, -0.4)], 0.2, 3, 1.0),
        (8, [(12, 1.0), (13, 1.0), (0, -0.4)], -0.8, 3, -1.0),

        # Indentation & control flow
        (9, [(2, 1.0)], 0.0, 4, 1.0), (10, [(6, 1.0)], 0.0, 5, 1.0),
        (11, [(7, 1.0)], 0.0, 6, 1.0), (12, [(8, 1.0)], 0.0, 7, 1.0),
        (13, [(3, 1.0)], 0.0, 8, 1.0), (14, [(4, 1.0)], 0.0, 9, 1.0),
        (15, [(5, 1.0)], 0.0, 10, 1.0),
        (45, [(29, 0.5), (30, 1.2)], 0.0, 36, 1.0),
        (46, [(152, 1.0), (153, 1.0), (194, 1.0), (195, 1.0), (198, 1.0)], 0.0, 37, 1.0),
        (16, [(80, 1.0), (81, 0.8), (82, 0.8), (83, 0.8),
              (140, 0.9), (141, 0.5), (142, 0.8)], 0.0, 11, 1.0),
        (17, [(81, 1.0), (140, 0.8)], 0.0, 12, 1.0),
        (18, [(85, 1.0), (86, 1.2), (144, 1.2), (87, 0.8), (188, 0.8), (29, -2.5)], 0.0, 13, 1.0),
        (19, [(80, 1.0), (85, 1.0), (86, 1.0), (144, 1.0), (29, -2.5)], -1.5, 14, 2.0),
        (20, wt, -3.2, 15, 1.0), (21, wt, -4.2, 15, -1.0),
        (22, [(84, 2.0), (142, 2.0), (107, 1.5), (87, 1.5)], 0.0, 35, 1.0),

        # Naming, error handling & slop cues
        (23, ct, 0.0, 16, 1.0), (24, it, 0.0, 33, 1.0),
        (25, [(176, 0.11)] + [(c, -w * 0.25) for c, w in it], -1.05, 17, 1.0),
        (26, [(18, 1.0)], 0.0, 18, 1.0), (27, [(19, 0.25)], 0.0, 19, 0.9),
        (28, [(109, 1.0), (143, 1.2), (199, 1.2), (200, 1.2)], 0.0, 20, 1.0),
        (29, [(88, 1.0), (24, 1.0)], -1.5, 20, 2.0),
        (30, [(86, 1.5), (144, 1.5), (95, 1.5), (96, 1.5), (97, 1.5),
              (98, 1.0), (99, 1.2), (189, 1.5)], 0.0, 21, 1.0),
        (31, [(24, 1.5), (103, 1.2), (104, 1.4), (108, 1.2), (145, 1.4)], 0.0, 22, 1.0),
        (47, [(85, 1.0), (146, 1.0), (147, 1.0)], -1.5, 22, 1.8),
        (32, [(100, 1.4), (101, 1.4), (102, 1.0), (148, 1.4), (149, 1.4),
              (158, 1.8), (159, 1.6), (160, 1.6), (193, 0.7),
              (196, 2.0), (197, 2.0), (21, 0.45)], -0.95, 23, 1.2),

        # Chaining, narrator tags & imperative rhythm
        (33, [(91, 1.0), (92, 1.0), (93, 1.0), (94, 1.0), (150, 0.7)], -1.2, 24, 1.2),
        (34, [(20, 1.0)], -1.2, 25, 1.5), (35, [(9, 1.0)], -0.45, 26, 1.0),
        (36, [(10, 1.2), (11, 1.5)], -0.35, 27, 1.2),
        (37, [(130, 3.2), (131, 3.2), (132, 3.2), (133, 2.2), (134, 2.2),
              (137, 1.3), (138, 1.5), (139, 1.1), (1, -0.045)], -0.8, 28, 1.2),
        (38, st, 0.0, 29, 1.0), (39, st, -1.0, 29, -1.0),
        (40, im, -1.8, 31, 2.0), (41, im, -2.3, 31, -2.0),
        (42, [(124, 0.8), (185, 1.0), (120, 0.6), (121, 0.6), (26, 0.4), (154, 0.8)], 0.0, 32, 1.0),
    )
    for hi, terms, bias, dst, sc in rules:
        for uch, wv in terms:
            lm1_w[uch, hi] += wv
        lm1_b[hi] = bias
        lm2_w[hi, dst] += sc
    p["lmlp1_w"], p["lmlp1_b"], p["lmlp2_w"], p["lmlp2_b"] = lm1_w, lm1_b, lm2_w, lm2_b


def _build_block2_and_heads(p: dict[str, np.ndarray]) -> None:
    """Compile cross-line sink-attention heads, [0, 1] issue clamps, and System One readout."""
    d0 = MAX_REL_DIST
    a1_wq, a1_wk = np.zeros((192, 128), dtype=np.float32), np.zeros((192, 128), dtype=np.float32)
    a1_wv, a1_wo = np.zeros((192, 96), dtype=np.float32), np.zeros((96, 192), dtype=np.float32)
    a1_br = np.full((8, 2 * d0 + 1), -1e4, dtype=np.float32)
    a1_bs = np.array([4.5, 1.2, 1.5, 2.5, 8.9, 2.0, -12.0, -12.0], dtype=np.float32)

    # Head 0: comment-to-next-line code SimHash overlap (detects comments restating code on L+1..L+2)
    q_mat, _ = np.linalg.qr(np.random.RandomState(777).randn(40, 40))
    proj = (q_mat[:, :16] * np.sqrt(40.0 / 16.0)).astype(np.float32)
    a1_wq[152:192, :16] = proj * float(np.sqrt(60.0))
    a1_wk[112:152, :16] = proj * float(np.sqrt(60.0))
    a1_br[0, [d0 + 1, d0 + 2]] = [0.0, -1.2]
    a1_wv[1, 0] = a1_wo[0, 40] = 1.0

    # Head 1: try/catch/except opener matching nearby real handler vs. empty/fallback body (L..L+3)
    a1_wq[20, [16, 17]] = [float(np.sqrt(32.0)), float(np.sqrt(16.0))]
    a1_wk[[21, 22], [16, 17]] = [float(np.sqrt(32.0)), float(np.sqrt(16.0))]
    a1_br[1, d0 : d0 + 4] = 0.0
    a1_wv[[21, 22], [12, 13]] = 1.0
    a1_wo[[12, 13], [41, 42]] = 1.0

    # Head 2: causal lookback (L-8..L) counting enclosing branch openers vs. guard-clause exits
    a1_br[2, d0 - 8 : d0 + 1] = 0.0
    a1_wv[[6, 7, 11, 13, 14, 32, 36], range(24, 31)] = [1, 1.5, 1, 1, 1, 1, 1]
    a1_wo[[24, 25], 43] = [0.8, 1.0]
    a1_wo[26, 44] = 1.0
    a1_wo[[27, 28, 29], 45] = [1.0, 0.8, 0.8]
    a1_wo[30, 51] = 1.0

    # Head 3: local window (L-9..L+9) measuring statement density vs. blank/comment paragraph breaks
    a1_br[3, d0 - 9 : d0 + 10] = 0.0
    a1_wv[[29, 34, 8], [36, 37, 38]] = 1.0
    a1_wo[[36, 37, 38], [46, 47, 50]] = 1.0

    # Head 4: distant code-to-code SimHash matching (|offset| >= 6) for duplicate logic blocks
    a1_wq[112:152, 64:80] = proj * 8.0
    a1_wk[112:152, 64:80] = proj * 8.0
    a1_br[4, np.abs(np.arange(-d0, d0 + 1)) >= 6] = 0.0
    a1_wv[31, 48] = a1_wo[48, 48] = 1.0

    # Head 5: local window (L-6..L+6) pooling stacked hardcoded equality edge-case branches
    a1_br[5, d0 - 6 : d0 + 7] = 0.0
    a1_wv[[15, 35], 60] = [1.0, -2.5]
    a1_wo[60, 49] = 1.0

    # Head 6: file-wide mean pooling to infer naming conventions and 2- vs 3/4-space indent scale
    a1_br[6, :] = 0.0
    a1_wv[[18, 19, 10, 9, 8], [72, 73, 74, 75, 76]] = 1.0
    a1_wo[[72, 73, 74, 75, 76], [52, 53, 56, 57, 58]] = 1.0

    # Head 7: local window (L-3..L+3) pooling dense chaining and naming bloat clusters
    a1_br[7, d0 - 3 : d0 + 4] = 0.0
    a1_wv[[24, 25, 16], [84, 84, 85]] = 1.0
    a1_wo[[84, 85], [54, 55]] = 1.0
    p["attn1_wq"], p["attn1_wk"], p["attn1_wv"], p["attn1_wo"] = a1_wq, a1_wk, a1_wv, a1_wo
    p["attn1_brel"], p["attn1_bsink"] = a1_br, a1_bs

    # Paired ReLU clamps ReLU(x) - ReLU(x - 1) write bounded [0, 1] issue flags into Z[64..76]
    cm1_w, cm1_b = np.zeros((192, 96), dtype=np.float32), np.zeros((96,), dtype=np.float32)
    cm1_2w, cm1_2b = np.zeros((96, 192), dtype=np.float32), np.zeros((192,), dtype=np.float32)
    c1_rules = (
        (0, [(1, 0.4), (7, 1.1), (6, 0.6), (44, 5.2),
             (12, 0.5), (45, -3.0), (3, -1.2)], -2.65, 76, 2.5),
        (26, [(1, 0.4), (36, 1.1), (51, 1.6), (44, 5.2),
              (12, 0.5), (45, -3.0), (3, -1.2)], -3.25, 64, 2.5),
        (28, [(56, 20.0), (57, -50.0), (58, -40.0)], -0.5, 75, 2.0),
        (30, [(58, 20.0), (57, -30.0)], -4.0, 75, 2.0),
        (2, [(16, 0.85), (17, 0.75), (55, 0.65), (33, -2.5)], -1.05, 65, 2.2),
        (4, [(18, 0.55), (52, 1.5), (53, -2.0), (19, -0.8), (1, 0.4)], -1.45, 72, 2.2),
        (6, [(2, 1.0), (30, 1.0), (40, 2.4)], -3.35, 66, 2.5),
        (8, [(28, 1.5)], -0.45, 66, 2.2),
        (10, [(20, 1.8), (42, 1.4), (41, -4.0)], -2.1, 67, 2.5),
        (12, [(23, 1.4)], -0.35, 67, 2.2),

        (14, [(24, 1.2), (25, 1.25), (27, 1.1), (26, 0.25),
              (33, -0.9), (37, -3.0)], -1.0, 68, 2.2),
        (16, [(15, 1.5), (49, 8.5), (35, -3.0)], -2.75, 69, 2.5),
        (18, [(31, 1.5), (48, 2.0)], -2.45, 70, 2.5),
        (20, [(29, 1.0), (46, 14.2), (47, -18.0)], -8.55, 71, 3.0),
        (22, [(29, 1.0), (8, 1.8), (58, -7.5)], -2.0, 71, 2.5),
        (24, [(13, 0.6), (14, 0.7), (32, 0.6), (9, 0.35), (19, 0.25)], -0.25, 73, 1.2),
    )
    for h0, terms, bias, dst, gn in c1_rules:
        for zs, wv in terms:
            cm1_w[zs, [h0, h0 + 1]] += wv * gn
        cm1_b[[h0, h0 + 1]] = [bias * gn, bias * gn - 1.0]
        cm1_2w[[h0, h0 + 1], dst] += [1.0, -1.0]
    p["cmlp1_w"], p["cmlp1_b"], p["cmlp1_2_w"], p["cmlp1_2_b"] = cm1_w, cm1_b, cm1_2w, cm1_2b

    a2_wq, a2_wk = np.zeros((192, 64), dtype=np.float32), np.zeros((192, 64), dtype=np.float32)
    a2_wv, a2_wo = np.zeros((192, 64), dtype=np.float32), np.zeros((64, 192), dtype=np.float32)
    a2_br = np.full((4, 2 * d0 + 1), -1e4, dtype=np.float32)
    a2_br[0, d0 - 3 : d0 + 4] = 0.0
    a2_br[1, d0 - 2 : d0 + 3] = 0.0
    a2_br[2, d0 - 3 : d0 + 4] = 0.0
    a2_br[3, d0 - 2 : d0 + 3] = 0.0
    a2_wv[[70, 64, 69, 72, 71, 76], [0, 16, 17, 32, 33, 48]] = 1.0
    a2_wo[[0, 16, 17, 32, 33, 48], [80, 81, 83, 82, 84, 85]] = 1.0
    p["attn2_wq"], p["attn2_wk"], p["attn2_wv"], p["attn2_wo"] = a2_wq, a2_wk, a2_wv, a2_wo
    p["attn2_brel"] = a2_br
    p["attn2_bsink"] = np.array([0.7, -12.0, -12.0, -12.0], dtype=np.float32)

    cm2_w, cm2_b = np.zeros((192, 64), dtype=np.float32), np.zeros((64,), dtype=np.float32)
    cm2_2w, cm2_2b = np.zeros((64, 192), dtype=np.float32), np.zeros((192,), dtype=np.float32)
    cm2_w[70, [0, 1]] = 2.0
    cm2_w[80, [0, 1]] = 5.0
    cm2_w[72, [2, 3]] = 1.6
    cm2_w[82, [2, 3]] = 3.0
    cm2_b[:4] = [-3.25, -4.25, -2.0, -3.0]
    cm2_2w[[0, 1], 74] = [1.0, -1.0]
    cm2_2w[[2, 3], 65] = [1.0, -1.0]

    # Gate 2-space pyramid (Z[76], Z[85]) off when file uses 3/4-space indent (Z[75])
    cm2_w[76, [4, 5]] = 1.0
    cm2_w[75, [4, 5]] = -3.0
    cm2_b[[4, 5]] = [0.0, -1.0]
    cm2_2w[[4, 5], 64] = [1.0, -1.0]

    cm2_w[85, [6, 7]] = 1.0
    cm2_w[75, [6, 7]] = -3.0
    cm2_b[[6, 7]] = [0.0, -1.0]
    cm2_2w[[6, 7], 81] = [1.0, -1.0]
    p["cmlp2_w"], p["cmlp2_b"], p["cmlp2_2_w"], p["cmlp2_2_b"] = cm2_w, cm2_b, cm2_2w, cm2_2b

    # Block 3 readout: 8 squiggle heads (sq_w) + gated document pooling into 31 System One logits
    regs = [64, 65, 66, 67, 68, 69, 74, 71]
    sq_w, sq_b = np.zeros((192, 8), dtype=np.float32), np.full((8,), -2.7, dtype=np.float32)
    sq_w[regs, range(8)] = [4.2, 4.6, 4.8, 4.7, 4.7, 4.3, 4.8, 4.3]
    sq_w[[81, 83, 84], [0, 5, 7]] = [1.2, 1.2, 1.0]
    p["sq_w"], p["sq_b"] = sq_w, sq_b

    pg_w = np.zeros((192, 1), dtype=np.float32)
    pg_w[0, 0] = 12.0
    p["pool_gate_w"], p["pool_gate_b"] = pg_w, np.array([-6.0], dtype=np.float32)

    d1_w, d1_b = np.zeros((384, 64), dtype=np.float32), np.zeros((64,), dtype=np.float32)
    d2_w, d2_b = np.zeros((64, 31), dtype=np.float32), np.zeros((31,), dtype=np.float32)
    for hi, reg in enumerate(regs):
        d1_w[[reg, 192 + reg], hi] = [4.5, 0.65]
        d1_b[hi] = -0.20
    d1_w[[73, 9], 8] = [1.4, 0.4]
    d1_w[[72, 264], 9] = [3.5, 0.6]
    d1_w[[16, 17], 10] = [2.0, 1.5]
    d1_w[[32, 13, 14], 11] = [2.0, 1.2, 1.2]

    d2_b[:2] = [1.45, 1.85]
    d2_w[:8, 0] = [-2.1, -2.1, -1.9, -2.2, -2.2, -2.2, -2.1, -2.0]
    d2_w[:8, 1] = [-1.2, -1.0, -2.2, -2.5, -1.1, -2.6, -2.3, -1.3]
    d2_w[8, :2] = [0.85, 0.45]

    s_heads = (
        [(0, 1.25)], [(1, 1.25)], [(4, 1.2), (5, 0.9), (6, 0.9)],
        [(2, 1.35)], [(3, 1.35), (5, 0.6)], [(7, 1.25), (0, 0.35)],
    )
    for si, units in enumerate(s_heads):
        o = 2 + 4 * si
        d2_b[o : o + 4] = [-2.3, -0.9, 0.6, 2.0]
        for ui, wm in units:
            d2_w[ui, o : o + 4] += [2.1 * wm, 0.8 * wm, -0.7 * wm, -2.2 * wm]
        d2_w[8, [o, o + 3]] += [-0.45, 0.45]

    d2_b[26:31] = [1.25, 1.35, -2.10, -2.10, -2.20]
    d2_w[:8, 26] -= 1.50
    d2_w[:8, 27] -= 1.25
    d2_w[[11, 8], 26] += [1.3, 0.5]
    d2_w[8, 27] += 0.3
    d2_w[10, 28] += 3.5
    d2_w[[4, 9, 0], 29] += [3.3, 3.3, 1.1]
    d2_w[[2, 3, 5, 6, 7, 0], 30] += [2.9, 3.1, 3.3, 3.0, 2.2, 1.1]
    p["doc1_w"], p["doc1_b"], p["doc2_w"], p["doc2_b"] = d1_w, d1_b, d2_w, d2_b


def compile_weights(output_path: Path | None = None) -> dict[str, np.ndarray]:
    p: dict[str, np.ndarray] = {}
    _build_block0(p)
    _build_block1(p)
    _build_block2_and_heads(p)
    out = output_path or (Path(__file__).with_name("codehygiene") / "weights.npz")
    np.savez_compressed(out, **p)
    return p


if __name__ == "__main__":
    target = Path(__file__).with_name("codehygiene") / "weights.npz"
    res = compile_weights(target)
    kb = target.stat().st_size / 1024
    print(f"Compiled {len(res)} tensors ({sum(v.size for v in res.values()):,} params, {kb:.1f} KB)")
