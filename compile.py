"""Closed-form 1.58-bit Ternary Weight-Space Program compiling circuits into weights.npz.

Every 2D/3D weight, convolution, projection, causal, and embedding tensor in the model
has entries strictly in {-1, 0, +1} (packed at 5 trits per uint8 byte), paired only with
1D per-output-channel scales and biases:
  - Block 0 (_build_block0): Maps (L, 128) ASCII bytes + column positions into 54 per-cell
    channels (HC_*), using causal prefix matrices and De Morgan closer gates to mask strings
    (16-quote sawtooth) and single-line comments (//, /* */, # , -- ) left-to-right.
  - Block 1 (_build_block1): Runs k=3, 7, 13 {-1, 0, +1} integer-margin convolutions,
    cross-line docstring/comment sawtooth + Dyck paren-balance and previous-line filters
    (ml_proj, ml_gate_w, ml_paren_w), and 40 SimHash stem filters (_sketch_bank).
  - Block 2 (_build_block2_and_heads): Two cross-line sink-attention + MLP passes using
    Paley Hadamard and Gram-fitted ternary projections populate context and issue registers.
  - Block 3 (_build_block2_and_heads): Projects Z into 8 per-line squiggle probabilities
    and balanced-ternary System One document heads (31 logits).
"""

from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
import numpy as np

from codehygiene import MAX_COLS, MAX_REL_DIST, unpack_ternary_archive

_BITS7 = np.arange(7)
_POW3 = 3 ** np.arange(5, dtype=np.uint8)
_STOPWORDS = (
    "this that with from have will when then else true fals null cons onst func unct "
    "ctio tion retu etur turn thro hrow type ypeo peof numb stri trin ring unde ndef "
    "defi efin fine ined only also more some into over must empt leng ngth erro rror "
    "expo xpor modu odul data chan self none elif pass"
).split()

# Per-cell feature channels emitted by Block 0 (54 channels per grid cell)
(
    HC_ACT, HC_COMM, HC_CODE, HC_CNS, HC_MNS, HC_IND, HC_FNS, HC_ODD,
    HC_M42, HC_M40, HC_D8, HC_D10, HC_D12, HC_G80, HC_G96, HC_ID,
    HC_LO, HC_UP, HC_DIG, HC_UND, HC_DOT, HC_QM, HC_NOT_WORD, HC_EQ,
    HC_LB, HC_RB, HC_LP, HC_RP, HC_SEM, HC_SK_VETO, HC_QT, HC_SL_BOOST,
) = range(32)
HC_BP, HC_BN, HC_G110, HC_SL_C = 32, 39, 46, 47
HC_STR, HC_LEAD, HC_D16, HC_D20, HC_D24, HC_RAW_STR = range(48, 54)
N_HC = 54

# Byte and column embedding layout inside Block 0 (51 channels)
EMB_ACTIVE, EMB_NONSPACE, EMB_SPACE = 0, 1, 2
EMB_LOWER, EMB_UPPER, EMB_DIGIT, EMB_IDENT = 3, 4, 5, 6
EMB_UND, EMB_SLASH, EMB_STAR, EMB_COLON, EMB_QMARK, EMB_DOT, EMB_EQ = range(7, 14)
EMB_LBRACE, EMB_RBRACE, EMB_LPAREN, EMB_RPAREN = 14, 15, 16, 17
EMB_SEMI, EMB_LOGIC, EMB_ANY_QUOTE, EMB_ASCII_BITS = 18, 19, 20, 21

EMB_COL0, EMB_COL_ODD, EMB_COL_MOD4_2, EMB_COL_IND4 = 28, 29, 30, 31
EMB_COL_GE8, EMB_COL_GE10, EMB_COL_GE12 = 32, 33, 34
EMB_COL_GE80, EMB_COL_GE96, EMB_COL_GE110 = 35, 36, 37
EMB_COL_GE16, EMB_COL_GE20 = 38, 39
EMB_DQUOTE, EMB_SQUOTE, EMB_BACKSLASH, EMB_HASH, EMB_DASH, EMB_LIFE_PFX = range(40, 46)
EMB_COL_GE24 = 46
EMB_BQUOTE, EMB_BANG, EMB_ANGLE, EMB_PERCENT = range(47, 51)
N_EMB = 51

# Block 0 lexical pre-convolution, causal prefixes, and quote/comment gates
(
    PRE_SLASH_COMM, PRE_BLOCK_OPEN, PRE_BLOCK_CLOSE, PRE_HASH_COMM,
    PRE_DASH_COMM, PRE_DQUOTE, PRE_SQUOTE, PRE_APOSTROPHE, PRE_HASH2_COMM,
    PRE_BANG_DOC, PRE_BANG_LEAD, PRE_SP_BANG2, PRE_BANG2_CONT, PRE_BANG2_ABORT,
    PRE_VB_COMM, PRE_ZIG_RAW, PRE_BQUOTE, PRE_QUOTED_BQUOTE, PRE_EOL,
) = range(19)
N_PRE = 19
(
    C1_NONSPACE, C1_DQUOTES, C1_SQUOTES, C1_BLOCK_OPENS,
    C1_BQUOTES, C1_SP_BANG2, C1_MID_COMM, C1_IN_BCOMM,
) = range(8)
N_PREF1 = 8

N_QSAW = 32
G1_INDENT_SPACE, G1_FIRST_NONSPACE, G1_DOC_STAR, G1_ANY_COMM_START = 0, 1, 2, 3
G1_CLOSE_HI, G1_INDENT_BASE, G1_LINE_COMM_START = 4, 5, 13
G1_LEAD_TWO, G1_LEAD_ONE, G1_CLOSE_LO, G1_IND_GE24 = 14, 15, 16, 17
G1_DQUOTE_SAW = 18
G1_SQUOTE_SAW = G1_DQUOTE_SAW + N_QSAW
G1_BQUOTE_SAW = G1_SQUOTE_SAW + N_QSAW
(
    G1_BANG_LEAD, G1_BANG2_COMM, G1_PCT_COMM, G1_VB_COMM, G1_ZIG_RAW,
    G1_ACTIVE, G1_ACT_B, G1_MC_HI, G1_MC_LO, G1_BC_HI, G1_BC_LO, G1_B2_HI, G1_B2_LO,
) = range(G1_BQUOTE_SAW + N_QSAW, G1_BQUOTE_SAW + N_QSAW + 13)
N_GATE1 = G1_B2_LO + 1
(
    C2_IN_COMMENT, C2_IN_LINE_COMM, C2_IN_COMM_B, C2_IN_COMM_C,
    C2_IN_RAW_LINE, C2_EFF_LEN, C2_LEAD_VETO,
) = range(7)
N_PREF2 = 7

# Cross-line delimiter sawtooth & Dyck parenthesis balance dimensions
N_DQ_SAW = 384
N_SQ_SAW = 128
N_ML_GATE = N_DQ_SAW + N_SQ_SAW + 2
N_PAREN_STEPS = 16
N_PAREN_SUF = 2 * N_PAREN_STEPS
N_PREV_LINE = 4
N_ML_IN = 7

# Line-level residual stream Z registers (202 line stats + 3 * 40 sketches = 322 channels)
N_LMLP2 = 202
SKETCH_DIM = 40
N_Z = N_LMLP2 + 3 * SKETCH_DIM
(
    Z_ACTIVE, Z_HAS_CODE, Z_PURE_COMMENT, Z_OPEN_NET, Z_INDENT,
    Z_DEPTH8, Z_DEPTH10, Z_DEPTH12, Z_ODD_INDENT, Z_MOD4_EQ2, Z_MOD4_EQ0,
    Z_BRANCH_OPEN, Z_ELSE_BRANCH, Z_EARLY_EXIT, Z_GUARD_CLAUSE,
    Z_HARDCODED_CASE, Z_CEREMONY, Z_LONG_IDENT, Z_ONE_CHAR_ID, Z_WORD_COUNT,
) = range(20)

(
    Z_TRY_CATCH, Z_REAL_HANDLER, Z_EMPTY_HANDLER, Z_FALLBACK_SWALLOW,
    Z_CHAINED_CALLS, Z_NESTED_TERNARY, Z_OVER_80, Z_OVER_96, Z_NARRATOR_TAG,
    Z_IS_STMT, Z_SHORT_COMMENT, Z_IMPERATIVE_LINE, Z_CLEAN_IDIOM, Z_STDLIB_IDIOM,
    Z_PARAGRAPH_BREAK, Z_SWITCH_BLOCK, Z_DEEP_PYRAMID, Z_MODULE_HEADER,
    Z_SHORT_GOLF, Z_MAGIC_NUM,
) = range(20, 40)

# Cross-line attention context registers (Z[40..63])
(
    Z_RESTATES_CODE, Z_NEAR_REAL_HANDLER, Z_NEAR_EMPTY_HANDLER,
    Z_ENCLOSING_DEPTH, Z_ENCLOSING_BRANCHES, Z_ENCLOSING_GUARDS,
    Z_LOCAL_STMT_DENSITY, Z_LOCAL_BREAK_DENSITY, Z_DISTANT_CLONE,
    Z_LOCAL_SPECIAL_CASES, Z_LOCAL_ODD_INDENT, Z_ENCLOSING_DEEP16,
    Z_FILE_ONE_CHAR, Z_FILE_WORDS, Z_LOCAL_CHAINING, Z_LOCAL_CEREMONY,
    Z_FILE_MOD4_0, Z_FILE_MOD4_2, Z_FILE_ODD_IND,
    Z_LOCAL_SHORT_GOLF, Z_LOCAL_MAGIC_NUM,
    Z_FMOD42_DUP, Z_FODD_DUP, Z_FODD_SCALED,
) = range(40, 64)

# Bounded [0, 1] per-line issue registers (Z[64..79]) and smoothed registers (Z[80..86])
(
    Z_ISS_NESTING, Z_ISS_NAMING, Z_ISS_COMMENT, Z_ISS_ERROR,
    Z_ISS_DENSE, Z_ISS_SPECIAL, Z_RAW_CLONE, Z_ISS_LAYOUT,
    Z_RAW_CRYPTIC, Z_CLEAN_BONUS, Z_ISS_CLONE,
    Z_USES_WIDE_INDENT, Z_RAW_NESTING_2SP,
    Z_WIDE_DUP_A, Z_WIDE_DUP_B, Z_NEXT_COAL,
) = range(64, 80)
(
    Z_SMOOTH_CLONE, Z_SMOOTH_NEST, Z_SMOOTH_CRYPTIC,
    Z_SMOOTH_SPECIAL, Z_SMOOTH_LAYOUT, Z_SMOOTH_NEST_2SP,
    Z_IS_INDENTED,
) = range(80, 87)

# Static slop, paranoia, and v0.5 multiline/container registers (Z[113..157]; Z[87..112] helpers)
(
    Z_SIG_OPEN, Z_STATIC_TYPE, Z_SIG_VETO, Z_SEMI_ONLY,
    Z_IF_LINE, Z_TYPE_HEDGE, Z_FAIL_FAST, Z_NEUTRAL_RET,
    Z_HAS_ENV, Z_TWO_ENVS, Z_HAS_COALESCE, Z_DEV_DEFAULT,
    Z_ENV_VETO, Z_HAS_ASSIGN, Z_ELISION_RAW, Z_LINE_COMM_VETO,
    Z_COMM_FENCE, Z_COMM_STRONG_CUE, Z_COMM_ASSIGN, Z_COMM_FCALL,
) = range(113, 133)
(
    Z_NEAR_COMM_FENCE, Z_TYPED_INTERNAL_SIG, Z_BOUNDARY_SIG_LINE,
    Z_TYPE_HEDGE_IF, Z_ENV_CHAIN_1L, Z_ENV_DEV_1L,
    Z_CONT_ENV, Z_CONT_DEV_DEF, Z_COMM_CODE_LINE,
    Z_SMOOTH_COMM_CODE, Z_HAS_RECENT_TYPED_SIG, Z_NEAR_NEUTRAL_RET,
    Z_NEXT_CONT_ENV, Z_NEXT_CONT_DEV, Z_NEXT_FAIL_FAST,
) = range(133, 148)
(
    Z_IN_DOC, Z_IN_PAREN, Z_CONTAINER_HDR, Z_PYR_CLASS,
    Z_FILE_CONTAINERS, Z_HAS_CONTAINER, Z_ENCLOSING_DEEP24,
    Z_RAW_NEST_TOP, Z_SHALLOW_LINE, Z_CRYPTIC_GATE,
) = range(148, 158)
Z_PREV_COMMENT, Z_NIL_SWALLOW = 158, 159

# Track 2 polyglot registers (Z[160..167]) and Track 3 structural-friction registers (Z[168..200])
(
    Z_MATH_SIG, Z_KT_RUNCATCH, Z_KT_GETNULL, Z_SWIFT_TRYQ,
    Z_SWIFT_NIL, Z_TRY_BOOL_SIG, Z_NARROW_CATCH, Z_LUA_PCALL,
) = range(160, 168)
(
    Z_CURSOR_MUT, Z_CURSOR_GENERIC, Z_CURSOR_INDENT, Z_LOOP_HDR,
    Z_COND_LOOP_HDR, Z_COND_BRANCH, Z_SIG_CAND, Z_SIG_CONT,
    Z_SPAN_COMMA, Z_SPAN_BOOL, Z_SPAN_OUTREF, Z_SKEL_IMP,
    Z_MUT_STMT, Z_ENC_LOOP, Z_ENC_COND_LOOP, Z_ENC_BRANCH,
    Z_HAS_ENC_LOOP, Z_HAS_COND_LOOP, Z_HAS_ENC_BR1, Z_HAS_ENC_BR2,
    Z_ARITY_6PLUS, Z_HAS_BOOL_SPAN, Z_HAS_OUTREF_SPAN, Z_SKEL_DISTANT,
    Z_RAW_SKEL_CLONE, Z_SMOOTH_SKEL_CLONE, Z_LOCAL_MUT_DENS, Z_LOCAL_CTRL_DENS,
    Z_HAS_MUT_DENS, Z_HAS_CTRL_DENS, Z_SCOPE_ENTANGLE, Z_BRANCH_CLOSE,
    Z_HAS_CTRL_MOD, Z_SIG_OPEN_PAREN,
) = range(168, 202)
Z_CODE_SKETCH = N_LMLP2
Z_COMM_SKETCH = Z_CODE_SKETCH + SKETCH_DIM
Z_SKEL_SKETCH = Z_COMM_SKETCH + SKETCH_DIM

# Named attention head slices
H0_QK = slice(0, 16)
H4_QK = slice(64, 80)
H5_QK = slice(80, 96)
H45_QK = slice(64, 96)
H10_QK = slice(160, 176)
H11_QK = slice(176, 192)
H1011_QK = slice(160, 192)


def _balanced_trits(val: float, step: float, num_trits: int = 3) -> list[int]:
    """Balanced-ternary digits in {-1, 0, +1} representing round(val / step)."""
    max_int = (3**num_trits - 1) // 2
    target_int = max(-max_int, min(max_int, int(round(val / step))))
    digits: list[int] = []
    for _ in range(num_trits):
        rem = target_int % 3
        if rem == 2:
            rem = -1
            target_int = (target_int + 1) // 3
        else:
            target_int = (target_int - rem) // 3
        digits.append(int(rem))
    return digits


def _pack_trits(arr: np.ndarray) -> np.ndarray:
    """Pack {-1, 0, +1} array at 5 trits per uint8 byte (mapping 0 -> 0x00)."""
    flat = (arr.ravel() % 3).astype(np.uint8)
    rem = (-len(flat)) % 5
    if rem:
        flat = np.pad(flat, (0, rem))
    return (flat.reshape(-1, 5) * _POW3).sum(axis=1).astype(np.uint8)


def _stamp_char(kernel: np.ndarray, pos: int, channel: int, char: str) -> None:
    """Stamp 7-bit case-folded ASCII bit match for `char` into `(pos, channel)`."""
    bits_on = (((ord(char.lower()) & 0x7F) >> _BITS7) & 1) == 1
    kernel[pos, HC_BP + _BITS7[bits_on], channel] = 1
    kernel[pos, HC_BN + _BITS7[~bits_on], channel] = 1


def _sub_filter(
    kernel: np.ndarray,
    scale: np.ndarray,
    bias: np.ndarray,
    channel: int,
    pattern: str,
    *,
    code: bool = True,
    comm: bool = False,
    lead: int = 0,
    no_str: bool = True,
    wl: int = 0,
    wr: int = 0,
    cs: int = 0,
    req_left_id: int = 0,
    eol: int = 0,
    out_scale: float = 1.0,
) -> None:
    """Write a {-1, 0, +1} ASCII bit-template into `channel` using integer max-bound margin."""
    left_pad = max(wl, req_left_id)
    span = len(pattern) + left_pad + max(wr, eol)
    start = (kernel.shape[0] - span) // 2 + left_pad
    kernel[:, :, channel] = 0
    thresh = 7 * len(pattern)

    if wl:
        kernel[start - 1, HC_ID, channel] = -1
    elif req_left_id:
        kernel[start - 1, HC_ID, channel] = 1
        thresh += 1
    if wr:
        kernel[start + len(pattern), HC_ID, channel] = -1
    if eol:
        kernel[start + len(pattern), HC_ACT, channel] = -1
    if lead:
        kernel[start, HC_LEAD, channel] = 1
        thresh += 1

    for offset, char in enumerate(pattern):
        _stamp_char(kernel, start + offset, channel, char)
        if cs and char.isalpha():
            hi_ch, lo_ch = (HC_UP, HC_LO) if char.isupper() else (HC_LO, HC_UP)
            kernel[start + offset, [hi_ch, lo_ch], channel] = [1, -1]
            thresh += 1

    mid = start + len(pattern) // 2
    if code:
        kernel[mid, HC_COMM, channel] = -1
        if no_str:
            kernel[mid, HC_STR, channel] = -1
    if comm:
        kernel[mid, HC_CODE, channel] = -1

    scale[channel] = 2.0 * out_scale
    bias[channel] = -(2.0 * thresh - 1.0) * out_scale


def _encode_4gram(word: str) -> np.ndarray:
    codes = np.array([ord(ch) & 0x1F for ch in word[:4]], dtype=np.int32)[:, None]
    return np.where(((codes >> np.arange(5)) & 1) == 1, 1.0, -1.0).astype(np.float32)


_SKEL_BASIS_PATS = (
    "if (", "if ", "elif ", "else ", "} els", ") {", "; }", "then", "case ", "yiel",
    " < ", " > ", "<= ", ">= ", "< -", "|| ", "&& ", ".or.", ".not", " != ", " == ", " or ",
    " += ", " -= ", " = 0", " = -", "let ", " * ", " / ", ".X", ".Y", "%",
    "call ", "500.", "_dp", "NaN(", "fin", "Sqrt", "Atan", "= nil",
)


def _fill_skeleton_sketch(
    sketch_w: np.ndarray, sketch_s: np.ndarray, sketch_b: np.ndarray, base_ch: int
) -> None:
    """Populate 40 alpha-invariant operator and control-skeleton detectors in `base_ch..`."""
    for idx, pat in enumerate(_SKEL_BASIS_PATS):
        _sub_filter(
            sketch_w, sketch_s, sketch_b, base_ch + idx, pat,
            cs=False, code=True, no_str=True,
        )


def _sketch_bank() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build 120 {-1, 0, +1} sketch filters (code: 0..39, comments: 40..79, skel: 80..119)."""
    total_ch = 3 * SKETCH_DIM
    sketch_w = np.zeros((5, N_HC, total_ch), dtype=np.int8)
    sketch_s = np.full((total_ch,), 0.3, dtype=np.float32)
    sketch_b = np.zeros((total_ch,), dtype=np.float32)

    stop_vecs = [_encode_4gram(word) for word in _STOPWORDS]
    rng = np.random.RandomState(20261115)
    chosen: list[np.ndarray] = []
    while len(chosen) < SKETCH_DIM:
        cand = rng.choice([-1.0, 1.0], size=(4, 5)).astype(np.float32)
        stop_hits = sum(float(np.sum((svec * cand + 1.0) * 0.5)) >= 14.0 for svec in stop_vecs)
        orthogonal = all(float(np.abs(np.sum(cand * prev))) <= 8.0 for prev in chosen)
        if stop_hits <= 1 and orthogonal:
            chosen.append(cand)

    for idx, filt in enumerate(chosen):
        pos_mask = (filt > 0).astype(np.int8)
        neg_mask = (filt < 0).astype(np.int8)
        pair = [idx, SKETCH_DIM + idx]
        sketch_w[:4, HC_NOT_WORD, pair] = -1
        for ch_idx in pair:
            sketch_w[:4, HC_BP : HC_BP + 5, ch_idx] = pos_mask
            sketch_w[:4, HC_BN : HC_BN + 5, ch_idx] = neg_mask

        sketch_w[1, HC_SK_VETO, idx] = -1
        sketch_w[1, HC_SL_BOOST, SKETCH_DIM + idx] = 1
        sketch_b[pair] = [-13.7 * 0.3, -(40.0 + 13.7) * 0.3]

    _fill_skeleton_sketch(sketch_w, sketch_s, sketch_b, 2 * SKETCH_DIM)
    return sketch_w, sketch_s, sketch_b


def _paley_uniform_proj(seed: int = 0) -> np.ndarray:
    """Uniform-diagonal orthogonal {-1, +1} (40, 16) projection via disjoint Paley H_20 halves."""
    prime_q = 19
    quad_res = {(val * val) % prime_q for val in range(1, prime_q)}
    jacobi = np.zeros((prime_q, prime_q), dtype=np.int8)
    for row in range(prime_q):
        for col in range(prime_q):
            if row != col:
                jacobi[row, col] = 1 if ((row - col) % prime_q) in quad_res else -1

    had20 = np.ones((20, 20), dtype=np.int8)
    had20[1:, 0] = -1
    had20[1:, 1:] = jacobi + np.eye(prime_q, dtype=np.int8)
    had40 = np.block([[had20, had20], [had20, -had20]]).astype(np.int8)

    rng = np.random.RandomState(seed)
    row_signs = rng.choice(np.array([-1, 1], dtype=np.int8), size=(SKETCH_DIM, 1))
    perm = rng.permutation(SKETCH_DIM)
    disjoint_cols = [*range(1, 9), *range(29, 37)]
    return (had40 * row_signs)[perm][:, disjoint_cols]


def _fit_sketch_column(
    col: int, trits: np.ndarray, scales: np.ndarray, gram_target: np.ndarray
) -> None:
    """Update column `col` of `trits` and `scales` by coordinate descent on Gram error."""
    w_other = trits.astype(np.float32) * scales
    w_other[:, col] = 0.0
    res_gram = gram_target - (w_other @ w_other.T)
    col_s = float(scales[col])
    cur_vec = trits[:, col].astype(np.float32) * col_s
    cand_trits = (-1, 0, 1)
    cand_vals = np.array([-col_s, 0.0, col_s], dtype=np.float32)

    for row in range(SKETCH_DIM):
        mask = np.ones(SKETCH_DIM, dtype=bool)
        mask[row] = False
        diffs = cur_vec[mask, None] * cand_vals[None, :] - res_gram[row, mask, None]
        errs = 2.0 * np.sum(diffs * diffs, axis=0) + (cand_vals * cand_vals - res_gram[row, row]) ** 2
        best_trit = cand_trits[int(np.argmin(errs))]
        trits[row, col] = best_trit
        cur_vec[row] = best_trit * col_s

    t_col = trits[:, col].astype(np.float32)
    norm_sq = float(np.dot(t_col, t_col))
    quad_num = float(t_col @ res_gram @ t_col) if norm_sq > 0 else 0.0
    if quad_num > 0:
        scales[col] = math.sqrt(quad_num) / norm_sq


def _ternary_sketch_proj() -> tuple[np.ndarray, np.ndarray]:
    """Compile {-1, 0, +1} (40, 16) projection + 1D scales matching QR Gram geometry."""
    qr_mat, _ = np.linalg.qr(np.random.RandomState(777).randn(SKETCH_DIM, SKETCH_DIM))
    target = (qr_mat[:, H0_QK] * np.sqrt(SKETCH_DIM / 16.0)).astype(np.float32)
    gram_target = target @ target.T

    col_std = np.sqrt(np.mean(target * target, axis=0, keepdims=True))
    trits = np.zeros((SKETCH_DIM, 16), dtype=np.int8)
    trits[target > 0.61 * col_std] = 1
    trits[target < -0.61 * col_std] = -1
    scales = np.array([
        float(np.sum(target[:, col] * trits[:, col]) / max(int(np.sum(trits[:, col] ** 2)), 1))
        for col in range(16)
    ], dtype=np.float32)

    for _ in range(12):
        for col in range(16):
            _fit_sketch_column(col, trits, scales, gram_target)
    return trits, scales


def _build_block0_preconv() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compile local 3-char {-1, 0, +1} detectors for polyglot comments, quotes, and EOL."""
    pre_w = np.zeros((3, N_EMB, N_PRE), dtype=np.int8)
    pre_s = np.ones((N_PRE,), dtype=np.float32)
    pre_b = np.zeros((N_PRE,), dtype=np.float32)

    pre_w[[1, 2], EMB_SLASH, PRE_SLASH_COMM] = 1
    pre_w[0, [EMB_COLON, EMB_STAR, EMB_ANY_QUOTE], PRE_SLASH_COMM] = -1
    pre_w[[1, 2], [EMB_SLASH, EMB_STAR], PRE_BLOCK_OPEN] = 1
    pre_w[[0, 1], [EMB_STAR, EMB_SLASH], PRE_BLOCK_CLOSE] = 1
    pre_w[0, EMB_SLASH, PRE_BLOCK_CLOSE] = -1
    pre_s[:3] = 2.0
    pre_b[:3] = -3.0

    pre_w[0, EMB_NONSPACE, [PRE_HASH_COMM, PRE_HASH2_COMM]] = -1
    pre_w[[1, 2], [EMB_HASH, EMB_SPACE], PRE_HASH_COMM] = 1
    pre_w[[1, 2], EMB_HASH, PRE_HASH2_COMM] = 1
    pre_s[[PRE_HASH_COMM, PRE_HASH2_COMM]] = 2.0
    pre_b[[PRE_HASH_COMM, PRE_HASH2_COMM]] = -3.0

    pre_w[[0, 1, 2], [EMB_DASH, EMB_DASH, EMB_SPACE], PRE_DASH_COMM] = 1
    pre_s[PRE_DASH_COMM], pre_b[PRE_DASH_COMM] = 2.0, -5.0

    pre_w[0, [EMB_ANY_QUOTE, EMB_HASH], PRE_BANG_DOC] = -1
    pre_w[[1, 2], [EMB_BANG, EMB_ANGLE], PRE_BANG_DOC] = 1
    pre_w[[1, 2], [EMB_BANG, EMB_NONSPACE], PRE_BANG_LEAD] = [1, -1]
    pre_s[[PRE_BANG_DOC, PRE_BANG_LEAD]] = 2.0
    pre_b[[PRE_BANG_DOC, PRE_BANG_LEAD]] = [-3.0, -1.0]

    pre_w[0, EMB_NONSPACE, PRE_SP_BANG2] = -1
    pre_w[[1, 2], EMB_BANG, PRE_SP_BANG2] = 1
    pre_w[[0, 1], EMB_BANG, [PRE_BANG2_CONT, PRE_BANG2_ABORT]] = 1
    pre_w[2, [EMB_NONSPACE, EMB_BANG], PRE_BANG2_CONT] = [-1, 1]
    pre_w[2, [EMB_NONSPACE, EMB_BANG], PRE_BANG2_ABORT] = [1, -1]
    pre_s[[PRE_SP_BANG2, PRE_BANG2_CONT, PRE_BANG2_ABORT]] = 2.0
    pre_b[[PRE_SP_BANG2, PRE_BANG2_CONT, PRE_BANG2_ABORT]] = [-3.0, -3.0, -5.0]

    pre_w[[0, 1], [EMB_SQUOTE, EMB_SPACE], PRE_VB_COMM] = 1
    pre_w[2, [EMB_NONSPACE, EMB_SQUOTE], PRE_VB_COMM] = [1, -1]
    pre_s[PRE_VB_COMM], pre_b[PRE_VB_COMM] = 4.0, -10.0

    pre_w[[1, 2], EMB_BACKSLASH, PRE_ZIG_RAW] = 1
    pre_s[PRE_ZIG_RAW], pre_b[PRE_ZIG_RAW] = 2.0, -3.0

    pre_w[1, EMB_DQUOTE, PRE_DQUOTE] = 1
    pre_w[0, [EMB_BACKSLASH, EMB_SQUOTE], PRE_DQUOTE] = -1
    pre_w[2, EMB_SQUOTE, PRE_DQUOTE] = -1
    pre_w[1, EMB_SQUOTE, PRE_SQUOTE] = 1
    pre_w[0, [EMB_BACKSLASH, EMB_LIFE_PFX, EMB_DQUOTE], PRE_SQUOTE] = -1
    pre_w[2, EMB_DQUOTE, PRE_SQUOTE] = -1
    pre_w[1, EMB_BQUOTE, PRE_BQUOTE] = 1
    pre_w[0, [EMB_BACKSLASH, EMB_BQUOTE], PRE_BQUOTE] = -1
    pre_w[2, EMB_BQUOTE, PRE_BQUOTE] = -1

    pre_w[0, [EMB_LOWER, EMB_UPPER], PRE_APOSTROPHE] = 1
    pre_w[1, EMB_SQUOTE, PRE_APOSTROPHE] = 1
    pre_w[2, [EMB_LOWER, EMB_UPPER], PRE_APOSTROPHE] = 1
    pre_w[0, [EMB_DQUOTE, EMB_SQUOTE], PRE_QUOTED_BQUOTE] = 1
    pre_w[1, EMB_BQUOTE, PRE_QUOTED_BQUOTE] = 1
    pre_w[2, [EMB_DQUOTE, EMB_SQUOTE], PRE_QUOTED_BQUOTE] = 1
    pre_b[[PRE_APOSTROPHE, PRE_QUOTED_BQUOTE]] = -2.0

    pre_w[[1, 2], EMB_ACTIVE, PRE_EOL] = [1, -1]
    pre_s[PRE_EOL], pre_b[PRE_EOL] = 512.0, -256.0
    return pre_w, pre_s, pre_b


def _build_block0(trit_mats: dict[str, np.ndarray], vecs_1d: dict[str, np.ndarray]) -> None:
    """Compile {-1, 0, +1} byte/column embeddings and quote/comment masking circuits."""
    tok_emb = np.zeros((128, N_EMB), dtype=np.int8)
    tok_emb[1:, EMB_ACTIVE] = 1
    tok_emb[33:, EMB_NONSPACE] = 1
    tok_emb[32, EMB_SPACE] = 1
    tok_emb[ord("a") : ord("z") + 1, EMB_LOWER] = 1
    tok_emb[ord("A") : ord("Z") + 1, EMB_UPPER] = 1
    tok_emb[ord("0") : ord("9") + 1, EMB_DIGIT] = 1

    for byte_val in range(1, 128):
        char = chr(byte_val)
        tok_emb[byte_val, EMB_IDENT] = int(char.isalnum() or char in "_$")
        bits = ((ord(char.lower()) & 0x7F) >> _BITS7) & 1
        tok_emb[byte_val, EMB_ASCII_BITS : EMB_ASCII_BITS + 7] = bits.astype(np.int8)

    for char, dst_col in zip("_/*:?.={}();", range(EMB_UND, EMB_SEMI + 1)):
        tok_emb[ord(char), dst_col] = 1
    tok_emb[[ord("|"), ord("&")], EMB_LOGIC] = 1
    tok_emb[[ord("'"), ord('"'), ord("`")], EMB_ANY_QUOTE] = 1
    for char, dst_col in zip('"\'\\#-', range(EMB_DQUOTE, EMB_DASH + 1)):
        tok_emb[ord(char), dst_col] = 1
    tok_emb[[ord("&"), ord("<")], EMB_LIFE_PFX] = 1
    tok_emb[[ord("`"), ord("!"), ord("%")], [EMB_BQUOTE, EMB_BANG, EMB_PERCENT]] = 1
    tok_emb[[ord("<"), ord(">")], EMB_ANGLE] = 1

    col_emb = np.zeros((MAX_COLS, N_EMB), dtype=np.int8)
    col_idx = np.arange(MAX_COLS)
    col_emb[0, EMB_COL0] = 1
    col_emb[col_idx % 2 == 1, EMB_COL_ODD] = 1
    col_emb[col_idx % 4 == 2, EMB_COL_MOD4_2] = 1
    col_emb[(col_idx > 0) & (col_idx % 4 == 0) & (col_idx <= 28), EMB_COL_IND4] = 1
    col_thresholds = (
        (8, EMB_COL_GE8), (10, EMB_COL_GE10), (12, EMB_COL_GE12),
        (80, EMB_COL_GE80), (96, EMB_COL_GE96), (110, EMB_COL_GE110),
        (16, EMB_COL_GE16), (20, EMB_COL_GE20), (24, EMB_COL_GE24),
    )
    for min_col, dst_col in col_thresholds:
        col_emb[col_idx >= min_col, dst_col] = 1
    trit_mats["tok_emb"], trit_mats["col_emb"] = tok_emb, col_emb

    pre_w, pre_s, pre_b = _build_block0_preconv()
    trit_mats["pre_conv3_w"] = pre_w
    vecs_1d["pre_conv3_s"], vecs_1d["pre_conv3_b"] = pre_s, pre_b

    off_pre = N_EMB
    off_pref1 = N_EMB + N_PRE
    pref1 = np.zeros((off_pref1, N_PREF1), dtype=np.int8)
    pref1[EMB_NONSPACE, C1_NONSPACE] = 1
    pref1[off_pre + PRE_DQUOTE, C1_DQUOTES] = 1
    pref1[[off_pre + PRE_SQUOTE, off_pre + PRE_APOSTROPHE], C1_SQUOTES] = [1, -1]
    pref1[off_pre + PRE_BLOCK_OPEN, C1_BLOCK_OPENS] = 1
    pref1[[off_pre + PRE_BQUOTE, off_pre + PRE_QUOTED_BQUOTE], C1_BQUOTES] = [1, -1]
    pref1[[off_pre + PRE_SP_BANG2, off_pre + PRE_BANG2_ABORT], C1_SP_BANG2] = [1, -1]
    line_comm = [
        off_pre + PRE_SLASH_COMM, off_pre + PRE_HASH_COMM,
        off_pre + PRE_HASH2_COMM, off_pre + PRE_DASH_COMM,
    ]
    mid_comm = [*line_comm, off_pre + PRE_BANG_DOC]
    pref1[mid_comm, C1_MID_COMM] = 1
    pref1[[off_pre + PRE_BLOCK_OPEN, off_pre + PRE_BLOCK_CLOSE], C1_IN_BCOMM] = [1, -1]
    trit_mats["pref1_proj"] = pref1
    trit_mats["W_causal1"] = np.tril(np.ones((MAX_COLS, MAX_COLS), dtype=np.int8), k=-1)
    trit_mats["W_causal2"] = np.tril(np.ones((MAX_COLS, MAX_COLS), dtype=np.int8), k=0)

    # Gate layer with De Morgan /*-paired */ closer, polyglot comments, and 32-quote sawtooths
    gate1_w = np.zeros((off_pref1 + N_PREF1, N_GATE1), dtype=np.int8)
    gate1_s = np.ones((N_GATE1,), dtype=np.float32)
    gate1_b = np.zeros((N_GATE1,), dtype=np.float32)

    gate1_w[[EMB_SPACE, EMB_NONSPACE], [G1_INDENT_SPACE, G1_FIRST_NONSPACE]] = 1
    gate1_w[off_pref1 + C1_NONSPACE, [G1_INDENT_SPACE, G1_FIRST_NONSPACE]] = -1
    gate1_w[[EMB_STAR, off_pref1 + C1_NONSPACE], G1_DOC_STAR] = [1, -1]

    gate1_w[[*mid_comm, off_pre + PRE_BLOCK_OPEN], G1_ANY_COMM_START] = 1
    gate1_w[line_comm, G1_LINE_COMM_START] = 1
    gate1_w[off_pre + PRE_BLOCK_CLOSE, [G1_CLOSE_HI, G1_CLOSE_LO]] = 1
    gate1_w[off_pref1 + C1_BLOCK_OPENS, G1_CLOSE_LO] = -1

    indent_cols = (
        EMB_COL_ODD, EMB_COL_MOD4_2, EMB_COL_IND4, EMB_COL_GE8,
        EMB_COL_GE10, EMB_COL_GE12, EMB_COL_GE16, EMB_COL_GE20,
    )
    for offset, emb_col in enumerate(indent_cols):
        dst_gate = G1_INDENT_BASE + offset
        gate1_w[[EMB_NONSPACE, emb_col, EMB_STAR, off_pref1 + C1_NONSPACE], dst_gate] = [
            1, 1, -1, -1,
        ]
        gate1_s[dst_gate], gate1_b[dst_gate] = 2.0, -3.0
    gate1_w[[EMB_NONSPACE, EMB_COL_GE24, EMB_STAR, off_pref1 + C1_NONSPACE], G1_IND_GE24] = [
        1, 1, -1, -1,
    ]
    gate1_s[G1_IND_GE24], gate1_b[G1_IND_GE24] = 2.0, -3.0

    gate1_w[off_pref1 + C1_NONSPACE, [G1_LEAD_TWO, G1_LEAD_ONE]] = -1
    gate1_b[[G1_LEAD_TWO, G1_LEAD_ONE]] = [2.0, 1.0]
    for step in range(N_QSAW):
        saw_scale = 2.0 if step > 0 else 1.0
        saw_gates = [G1_DQUOTE_SAW + step, G1_SQUOTE_SAW + step, G1_BQUOTE_SAW + step]
        saw_srcs = [off_pref1 + C1_DQUOTES, off_pref1 + C1_SQUOTES, off_pref1 + C1_BQUOTES]
        gate1_w[saw_srcs, saw_gates] = 1
        gate1_s[saw_gates] = saw_scale
        gate1_b[saw_gates] = -float(step) * saw_scale

    lead_openers = (
        (off_pre + PRE_BANG_LEAD, G1_BANG_LEAD),
        (off_pre + PRE_VB_COMM, G1_VB_COMM),
        (off_pre + PRE_ZIG_RAW, G1_ZIG_RAW),
    )
    for src_row, dst_gate in lead_openers:
        gate1_w[[src_row, off_pref1 + C1_NONSPACE], dst_gate] = [1, -1]
        gate1_s[dst_gate], gate1_b[dst_gate] = 2.0, -1.0

    gate1_w[[EMB_PERCENT, off_pref1 + C1_NONSPACE, EMB_COL_GE24], G1_PCT_COMM] = [1, -1, -1]
    gate1_s[G1_PCT_COMM], gate1_b[G1_PCT_COMM] = 2.0, -1.0
    gate1_w[[off_pre + PRE_BANG2_CONT, off_pref1 + C1_SP_BANG2], G1_BANG2_COMM] = 1
    gate1_s[G1_BANG2_COMM], gate1_b[G1_BANG2_COMM] = 2.0, -3.0

    gate1_w[EMB_ACTIVE, [G1_ACTIVE, G1_ACT_B]] = 1
    for c1_src, hi_g, lo_g in (
        (C1_MID_COMM, G1_MC_HI, G1_MC_LO),
        (C1_IN_BCOMM, G1_BC_HI, G1_BC_LO),
        (C1_SP_BANG2, G1_B2_HI, G1_B2_LO),
    ):
        gate1_w[off_pref1 + c1_src, [hi_g, lo_g]] = 1
        gate1_s[[hi_g, lo_g]] = 2.0
        gate1_b[[hi_g, lo_g]] = [0.0, -2.0]
    trit_mats["gate1_w"] = gate1_w
    vecs_1d["gate1_s"], vecs_1d["gate1_b"] = gate1_s, gate1_b

    saw_trits = np.array([1] + [-1, 1] * (N_QSAW // 2 - 1) + [-1], dtype=np.int8)
    pref2 = np.zeros((N_GATE1, N_PREF2), dtype=np.int8)
    pref2_s = np.ones((N_PREF2,), dtype=np.float32)
    extra_sl = [G1_BANG_LEAD, G1_PCT_COMM, G1_VB_COMM]
    for dst_c2 in (C2_IN_COMMENT, C2_IN_COMM_B, C2_IN_COMM_C):
        pref2[[G1_DOC_STAR, G1_ANY_COMM_START, G1_CLOSE_HI, G1_CLOSE_LO], dst_c2] = [1, 1, -1, 1]
        pref2[[*extra_sl, G1_BANG2_COMM], dst_c2] = 1
    pref2[[G1_LINE_COMM_START, *extra_sl], C2_IN_LINE_COMM] = 1
    pref2[G1_ZIG_RAW, C2_IN_RAW_LINE] = 1

    pref2[[G1_ACTIVE, G1_ACT_B], C2_EFF_LEN] = 1
    for saw_base in (G1_DQUOTE_SAW, G1_SQUOTE_SAW, G1_BQUOTE_SAW):
        pref2[saw_base : saw_base + N_QSAW, C2_EFF_LEN] = -saw_trits
    pref2[[G1_MC_HI, G1_MC_LO, G1_BC_HI, G1_BC_LO, G1_B2_HI, G1_B2_LO], C2_EFF_LEN] = [
        -1, 1, -1, 1, -1, 1,
    ]
    pref2[[G1_DOC_STAR, *extra_sl, G1_ZIG_RAW], C2_LEAD_VETO] = 1
    pref2_s[C2_LEAD_VETO] = float(2 * MAX_COLS)
    trit_mats["pref2_proj"], vecs_1d["pref2_proj_s"] = pref2, pref2_s

    # Fuse [emb, pre_conv3, gate1, c2] into 54 per-cell HC_* channels in {-1, 0, +1}
    off_gate1 = off_pref1
    off_pref2 = off_gate1 + N_GATE1
    fuse_w = np.zeros((off_pref2 + N_PREF2, N_HC), dtype=np.int8)
    fuse_s = np.ones((N_HC,), dtype=np.float32)
    fuse_b = np.zeros((N_HC,), dtype=np.float32)

    in_comm_all = [off_pref2 + C2_IN_COMMENT, off_pref2 + C2_IN_COMM_B, off_pref2 + C2_IN_COMM_C]
    in_comm_pair = in_comm_all[:2]
    in_line_comm = off_pref2 + C2_IN_LINE_COMM
    in_raw_line = off_pref2 + C2_IN_RAW_LINE

    fuse_w[EMB_ACTIVE, [HC_ACT, HC_COMM, HC_SL_C, HC_CODE]] = 1
    fuse_w[in_comm_all[0], HC_COMM] = 1
    fuse_w[in_comm_pair, HC_CODE] = -1
    fuse_w[[*in_comm_pair, in_raw_line], HC_CNS] = -1
    fuse_w[in_line_comm, [HC_SL_C, HC_MNS]] = 1
    fuse_w[EMB_NONSPACE, [HC_CNS, HC_MNS]] = 1
    fuse_w[EMB_SLASH, HC_MNS] = -1
    fuse_b[[HC_COMM, HC_SL_C, HC_MNS]] = -1.0
    fuse_w[[off_gate1 + G1_INDENT_SPACE, off_gate1 + G1_FIRST_NONSPACE], [HC_IND, HC_FNS]] = 1

    dquote_slice = slice(off_gate1 + G1_DQUOTE_SAW, off_gate1 + G1_DQUOTE_SAW + N_QSAW)
    squote_slice = slice(off_gate1 + G1_SQUOTE_SAW, off_gate1 + G1_SQUOTE_SAW + N_QSAW)
    bquote_slice = slice(off_gate1 + G1_BQUOTE_SAW, off_gate1 + G1_BQUOTE_SAW + N_QSAW)
    for q_slice in (dquote_slice, squote_slice, bquote_slice):
        fuse_w[q_slice, [HC_STR, HC_RAW_STR]] = saw_trits[:, None]
        fuse_w[q_slice, HC_SK_VETO] = saw_trits
    fuse_w[in_raw_line, [HC_STR, HC_RAW_STR]] = 1
    fuse_w[in_comm_all, HC_STR] = -1
    fuse_w[in_line_comm, HC_RAW_STR] = -1

    lead_srcs = [
        EMB_NONSPACE, off_gate1 + G1_LEAD_TWO, off_gate1 + G1_LEAD_ONE, *in_comm_pair, in_raw_line,
    ]
    fuse_w[lead_srcs, HC_LEAD] = [1, 1, -1, -1, -1, -1]
    fuse_b[HC_LEAD] = -1.0

    indent_hc_dsts = [HC_ODD, HC_M42, HC_M40, HC_D8, HC_D10, HC_D12, HC_D16, HC_D20]
    for offset, hc_dst in enumerate(indent_hc_dsts):
        fuse_w[[off_gate1 + G1_INDENT_BASE + offset, *in_comm_pair, in_raw_line], hc_dst] = [
            1, -1, -1, -1,
        ]
    fuse_w[[off_gate1 + G1_IND_GE24, *in_comm_pair, in_raw_line], HC_D24] = [1, -1, -1, -1]

    eff_src = [off_pre + PRE_EOL, off_pref2 + C2_EFF_LEN, off_pref2 + C2_LEAD_VETO]
    for thresh, hc_dst in ((80.0, HC_G80), (96.0, HC_G96), (110.0, HC_G110)):
        fuse_w[eff_src, hc_dst] = [1, 1, -1]
        fuse_s[hc_dst] = 0.425
        fuse_b[hc_dst] = -(2.0 * float(MAX_COLS) + 2.0 * thresh) * 0.425

    char_pairs = (
        (EMB_IDENT, HC_ID), (EMB_LOWER, HC_LO), (EMB_UPPER, HC_UP), (EMB_DIGIT, HC_DIG),
        (EMB_UND, HC_UND), (EMB_DOT, HC_DOT), (EMB_QMARK, HC_QM), (EMB_EQ, HC_EQ),
        (EMB_LBRACE, HC_LB), (EMB_RBRACE, HC_RB), (EMB_LPAREN, HC_LP), (EMB_RPAREN, HC_RP),
        (EMB_SEMI, HC_SEM), (EMB_ANY_QUOTE, HC_QT),
    )
    for emb_src, hc_dst in char_pairs:
        fuse_w[emb_src, hc_dst] = 1
    fuse_w[EMB_ASCII_BITS + _BITS7, HC_BP + _BITS7] = 1
    fuse_w[EMB_ACTIVE, HC_BN + _BITS7] = 1
    fuse_w[EMB_ASCII_BITS + _BITS7, HC_BN + _BITS7] = -1

    fuse_w[[EMB_LOWER, EMB_UPPER, EMB_UND], HC_NOT_WORD] = -1
    fuse_s[HC_NOT_WORD], fuse_b[HC_NOT_WORD] = 40.0, 40.0
    fuse_w[[in_comm_all[0], in_raw_line], HC_SK_VETO] = 1
    fuse_s[HC_SK_VETO] = 120.0
    fuse_w[[EMB_ACTIVE, in_line_comm], HC_SL_BOOST] = 1
    fuse_s[HC_SL_BOOST], fuse_b[HC_SL_BOOST] = 40.0, -40.0
    trit_mats["col_fuse_w"] = fuse_w
    vecs_1d["col_fuse_s"], vecs_1d["col_fuse_b"] = fuse_s, fuse_b


# k=7 lexical code patterns: (pattern, left_word_bound, right_word_bound, case_sens, line_lead)
_K7_PRIMARY = (
    ("if", 1, 1, 1, 1), ("else", 1, 1, 1, 1), ("for", 1, 1, 1, 1),
    ("while", 1, 1, 1, 1), ("switch", 0, 1, 1, 1), ("return", 0, 1, 1, 0),
    ("throw", 1, 1, 1, 0), ("break", 1, 1, 1, 0), ("catch", 1, 1, 1, 0),
    ("try", 1, 1, 1, 0), ("typeof", 0, 1, 1, 0), (".map(", 0, 0, 1, 0),
    (".filter", 0, 0, 1, 0), (".reduce", 0, 0, 1, 0), (".flatMa", 0, 0, 1, 0),
    (".code", 0, 1, 1, 0), ("reject(", 0, 0, 1, 0), ("cb(", 1, 0, 1, 0),
    ("emit(", 0, 0, 1, 0), ("destroy", 0, 0, 1, 0), ("|| {}", 0, 0, 0, 0),
    ("|| []", 0, 0, 0, 0), ("|| '')", 0, 0, 0, 0), ("console", 0, 0, 1, 0),
    ("null;", 1, 0, 1, 0), ("== '", 0, 0, 0, 0), ('== "', 0, 0, 0, 0),
    ("case ", 1, 0, 1, 1), ("false;", 1, 0, 1, 0), ("catch (", 0, 0, 1, 0),
    ("== -1", 0, 0, 0, 0), ("Manager", 0, 0, 1, 0), ("Factory", 0, 0, 1, 0),
    ("Service", 0, 0, 1, 0), ("Builder", 0, 0, 1, 0), ("Wrapper", 0, 0, 1, 0),
    ("Helper", 0, 0, 1, 0), ("Impl", 0, 1, 1, 0), ("Default", 0, 0, 1, 0),
    ("Context", 0, 0, 1, 0), ("ERR_", 1, 0, 1, 0), ("validat", 0, 0, 0, 0),
    ("require", 0, 0, 1, 0), ("exports", 0, 0, 1, 0), ("strict'", 0, 0, 1, 0),
    ("SAFE_IN", 0, 0, 1, 0),
)

_K7_EXTENDED = (
    ("elif", 1, 1, 1, 1), ("else if", 0, 0, 1, 1), ("match", 1, 1, 1, 1),
    ("except:", 0, 0, 1, 1), ("raise", 1, 1, 1, 0), ("pass", 1, 1, 1, 1),
    ("None", 1, 1, 1, 0), ("0.0", 1, 1, 0, 0), (" or []", 0, 0, 1, 0),
    (" or {}", 0, 0, 1, 0), (".unwrap", 0, 0, 1, 0), ("type(", 1, 0, 1, 0),
    ("from ", 1, 0, 1, 1), ("use ", 1, 0, 1, 1), ("assert(", 0, 0, 0, 0),
    ("...", 0, 1, 0, 1), ("class ", 1, 0, 1, 1), ("impl ", 1, 0, 1, 1),
    ("impl<", 1, 0, 1, 1),
)

_K7_V04_CODE = (
    ("def ", 1, 0, 1, 0), (") ->", 0, 0, 0, 0), (": str", 0, 1, 1, 0),
    (": int", 0, 1, 1, 0), (": bool", 0, 1, 1, 0), (": list", 0, 1, 0, 0),
    (": dict", 0, 1, 0, 0), ("pub ", 1, 0, 1, 0), ("any", 1, 1, 1, 0),
    ("Any", 1, 1, 1, 0), ("null", 1, 1, 1, 0), ("= None", 0, 1, 1, 0),
    (" | ", 0, 0, 0, 0), (" or ", 0, 0, 1, 0), ("return;", 1, 0, 1, 1),
    ("_old", 0, 1, 0, 0), ("_bak", 0, 1, 0, 0),
)

_K7_V04_COMM = (
    ("// ...", 0), ("# ...", 0), ("```", 0),
    (">>>", 0), ("// }", 0), ("# def ", 1),
    ("  def ", 1), ("# if ", 1), ("  if ", 1),
    ("# for ", 1), ("  for ", 1),
)

_K13_PRIMARY = (
    ("Abstract", 1), ("Repository", 1), ("Configuration", 1), ("Controller", 1),
    ("Strategy", 1), ("Provider", 1), ("Processor", 1), ("Prototype", 1),
    ("primordials", 1), ("HideStackFram", 1), ("internal/", 0), ("continue", 1),
    ("callback(", 0), ("=== undefined", 0), ("!== undefined", 0), ("EventEmitter", 1),
    ("#[allow(", 1), ("import ", 1), ("package ", 1),
)

_K13_EXTENDED = (("#include", 1), ("except Excep", 1), ("except BaseE", 1))

_K13_V04_CODE = (
    ("function ", 1, 0, 1), (": string", 0, 1, 1), (": number", 0, 1, 1),
    (": boolean", 0, 1, 1), (": float", 0, 1, 1), (": Record<", 0, 0, 1),
    ("export ", 1, 0, 1), ("public ", 1, 0, 1), ("unknown", 1, 1, 1),
    (": object", 0, 1, 0), ("Optional", 1, 1, 1), ("undefined", 1, 1, 1),
    ("Array.isArray", 0, 0, 1), ("isinstance(", 1, 0, 1), ("assert ", 1, 0, 1),
    ("return 0", 1, 1, 1), ("return []", 1, 0, 1), ("return {}", 1, 0, 1),
    ('return ""', 1, 0, 1), ("return ''", 1, 0, 1), ("return null", 1, 1, 1),
    ("return None", 1, 1, 1), ("process.env", 1, 0, 1), (".meta.env", 0, 0, 1),
    ("os.getenv(", 1, 0, 1), ("os.environ", 1, 0, 1), ("secret-key", 0, 0, 0),
    ("password123", 0, 0, 0), ("namespace ", 1, 0, 1), ("dev-secret", 0, 0, 0),
    ("changeme", 0, 0, 0), (" as any", 0, 1, 1), ("as unknown", 1, 1, 1),
    ("cast(Any", 1, 1, 1), ("_backup", 0, 1, 0), ("_deprecated", 0, 1, 0),
    ("Deprecated", 0, 1, 1),
)

_K13_V04_COMM = (
    ("pyright: ign", 0), ("biome-ignore", 0), ("rest of ", 0),
    ("existing cod", 0), ("placeholder", 0), ("remain uncha", 0),
    ("// const ", 1), ("  const ", 1), ("// let ", 1),
    ("  let ", 1), ("// return ", 1), ("# return ", 1),
    ("  return ", 1),
)

# Go directive comments (//go:embed, //go:build, // +kubebuilder markers) are code, not prose
_K7_DIRECTIVE_COMM = ("//go:", "// +")

# Test declarations whose long descriptive names are idiomatic (Go testing, testify, pytest)
_K7_TEST_METHOD = ") Test"
_K13_TEST_DECLS = ("func Test", "func Benchmar", "func Example", "func Fuzz", "def test_")

# contextlib.suppress(Exception | BaseException) swallows every error like `except Exception: pass`;
# narrow suppress(FileNotFoundError) stays clean
_K13_BROAD_SUPPRESS = ("suppress(Exce", "suppress(Base")

_K3_TERNARY_VETO = ("<?", "?>", ", ?")

_K7_V06_CODE = (
    ("for (", 1, 0, 1, 0), ("For ", 1, 0, 1, 1), (" :: ", 0, 0, 0, 0),
    (": f64", 0, 1, 1, 0), (": f32", 0, 1, 1, 0), (": i32", 0, 1, 1, 0),
    (": i64", 0, 1, 1, 0), ("Vec", 1, 0, 1, 0), ("fn ", 1, 0, 1, 0),
    (" or {})", 0, 0, 1, 0), (" or [])", 0, 0, 1, 0), ("|| []}", 0, 0, 0, 0),
    ("|| {}}", 0, 0, 0, 0), ("try? ", 1, 0, 1, 1), ("= try? ", 0, 0, 1, 0),
    (" ?? ", 0, 0, 0, 0), ("Catch ", 1, 0, 1, 1), ("Throw", 1, 1, 1, 0),
    ("pcall(", 0, 0, 1, 0), ("#[test]", 0, 0, 1, 1), (" or 0", 0, 1, 1, 0),
    (",_:=", 0, 0, 0, 0), ("type ", 1, 0, 1, 1), ("error(", 1, 0, 1, 0),
    (", err", 0, 1, 1, 0),
)

_K13_V06_CODE = (
    ("intent(", 1, 0, 0), ("subroutine ", 1, 0, 0), (": usize", 0, 1, 1),
    ("double ", 1, 0, 1), ("float ", 1, 0, 1), ("real(", 1, 0, 0),
    ("process.exit", 1, 0, 1), ("exitCode", 1, 1, 1), ("os.Exit(", 1, 0, 1),
    ("sys.exit(", 1, 0, 1), ("os.exit(", 1, 0, 1), (".error =", 0, 0, 1),
    (".Log", 0, 0, 1), ("stderr", 0, 0, 1), ("bool Try", 1, 0, 1),
    ("catch (Excep", 0, 0, 1), (") when (", 0, 0, 1), ("export type ", 1, 0, 1),
    ("interface ", 1, 0, 1), ("create table", 1, 0, 0), ("create index", 1, 0, 0),
    ("create uniqu", 1, 0, 0), ("create trigg", 1, 0, 0), ("alter table ", 1, 0, 0),
    ("elseif ", 1, 0, 1), ("} else if", 0, 0, 1), ("runCatching", 0, 1, 1),
    (".getOrNull()", 0, 0, 1), (".getOrDefaul", 0, 0, 1), (".onFailure", 0, 0, 1),
    ("_ = try?", 0, 0, 1), ("return nil", 1, 1, 1), ("return false", 0, 1, 0),
    ("As Exception", 0, 1, 1), ("End Try", 1, 1, 1), ("String.Empty", 0, 1, 1),
    (" or nil", 0, 1, 1), (" or false", 0, 1, 1), ("if not ok", 1, 1, 1),
    ("ok and ", 1, 0, 1), ("Return ", 1, 0, 1),
    ("public stati", 1, 0, 1), ("private stat", 1, 0, 1),
)

_DECL_TYPE_DDL = (
    "export type ", "interface ", "create table", "create index",
    "create uniqu", "create trigg", "alter table ",
    "public stati", "private stat",
)

# Track 3 structural-friction k=7 & k=13 code patterns
_K7_FRICTION = (
    ("do ", 1, 0, 0, 1), ("until ", 1, 0, 1, 1), (".each ", 0, 0, 1, 0),
    (".each{", 0, 0, 1, 0), ("_each ", 0, 0, 1, 0), ("elsif ", 1, 0, 1, 0),
    ("when ", 1, 0, 1, 1), ("case(", 1, 0, 0, 1), (" if ", 0, 0, 1, 0),
    ("sub ", 1, 0, 1, 1), ("func ", 1, 0, 1, 0), ("= @_;", 0, 0, 0, 0),
    ("void ", 1, 0, 1, 0), ("bool ", 1, 0, 1, 0), ("int ", 1, 0, 1, 0),
    ("&mut ", 0, 0, 1, 0), ("out ", 1, 0, 1, 0), ("ref ", 1, 0, 1, 0),
    (r"\%", 0, 0, 0, 0), (r"\@", 0, 0, 0, 0), ("out_", 1, 0, 0, 0),
    ("const ", 1, 0, 1, 0), ("my ", 1, 0, 1, 1), ("let ", 1, 0, 1, 1),
    ("var ", 1, 0, 1, 1), ("local ", 1, 0, 1, 1), ("call ", 1, 0, 0, 0),
    ("yield ", 1, 0, 1, 0), ("cycle", 1, 1, 0, 0), ("next ", 1, 0, 1, 0),
    (" += ", 0, 0, 0, 0), (" -= ", 0, 0, 0, 0), (" << ", 0, 0, 0, 0),
    ("push ", 1, 0, 1, 0), ("idx++", 1, 0, 0, 0), ("++idx", 0, 1, 0, 0),
    ("= true", 0, 1, 0, 0), ("$skip", 0, 1, 0, 0), ("$omit", 0, 1, 0, 0),
    ("+= 2", 0, 1, 0, 0), ("+= 4", 0, 1, 0, 0), ("{end+1}", 0, 0, 0, 0),
)

_K13_FRICTION = (
    ("do while", 1, 0, 0), ("select case", 1, 0, 0), ("unless ", 1, 0, 1),
    ("end if", 1, 1, 0), ("end select", 1, 1, 0), ("intent(out)", 0, 0, 0),
    ("intent(inou", 0, 0, 0), ("logical", 1, 1, 0), ("= false", 0, 1, 0),
    ("= .true.", 0, 0, 0), ("ByRef ", 1, 0, 0),
    ("$warnings", 0, 1, 0), ("__init__", 0, 0, 1),
    ("constructor", 1, 0, 1), ("sub new", 1, 0, 1), ("func New", 1, 0, 1),
    ("i = i + 1", 1, 1, 0), ("j = j + 1", 1, 1, 0), ("idx = idx +", 1, 0, 0),
    ("pos = pos +", 1, 0, 0), ("pos = last ", 1, 0, 0), ("cursor +=", 1, 0, 0),
    ("$cursor +=", 0, 0, 0), ("cursor -=", 0, 0, 0), ("cursor++", 0, 0, 0),
    ("++cursor", 0, 0, 0), ("$cursor = ", 0, 0, 0), ("idx += ", 1, 0, 0),
    ("idx -= ", 1, 0, 0), ("pos += ", 1, 0, 0), ("pos -= ", 1, 0, 0),
    ("stop -= 1", 1, 1, 0), ("i += 1", 1, 1, 1),
    ("i += 2", 1, 1, 1), (".append(", 0, 0, 1),
)

_SUFFIX_GRAVEYARD = frozenset({"_old", "_bak", "_backup", "_deprecated", "Deprecated"})

_CSUM_SPECS = (
    ("code_chars", HC_CNS, 1.0),
    ("comment_chars", HC_MNS, 1.0),
    ("indent_cols", HC_IND, 0.5),
    ("odd_indent", HC_ODD, 1.0),
    ("mod4_eq2_indent", HC_M42, 1.0),
    ("mod4_eq0_indent", HC_M40, 1.0),
    ("depth_ge8", HC_D8, 1.0),
    ("depth_ge10", HC_D10, 1.0),
    ("depth_ge12", HC_D12, 1.0),
    ("cols_ge80", HC_G80, 1.0 / 16.0),
    ("cols_ge96", HC_G96, 1.0 / 12.0),
    ("cols_ge110", HC_G110, 1.0 / 8.0),
    ("open_braces", HC_LB, 1.0),
    ("close_braces", HC_RB, 1.0),
    ("comment_mask", HC_COMM, 1.0),
    ("has_nonspace", HC_FNS, 1.0),
    ("depth_ge24", HC_D24, 1.0),
)


def _stamp_triple_quote(
    conv7_w: np.ndarray,
    conv7_s: np.ndarray,
    conv7_b: np.ndarray,
    channel: int,
    quote_char: str,
    require_lead: bool,
) -> None:
    for pos in (2, 3, 4):
        _stamp_char(conv7_w, pos, channel, quote_char)
    conv7_w[[1, 5], HC_QT, channel] = -1
    conv7_w[3, HC_SL_C, channel] = -1
    thresh = 21
    if require_lead:
        conv7_w[2, HC_LEAD, channel] = 1
        thresh += 1
    conv7_s[channel] = 2.0
    conv7_b[channel] = -float(2 * thresh - 1)


def _build_multiline_gates(
    trit_mats: dict[str, np.ndarray],
    vecs_1d: dict[str, np.ndarray],
    feat: dict[str, int],
    num_raw_u: int,
) -> None:
    """Compile {-1, 0, +1} docstring/comment sawtooth, Dyck paren, and previous-line filters."""
    ml_proj = np.zeros((num_raw_u, N_ML_IN), dtype=np.int8)
    ml_proj_s = np.ones((N_ML_IN,), dtype=np.float32)
    ml_proj[feat["triple_dq"], 0] = 1
    ml_proj[feat["triple_sq"], 1] = 1
    ml_proj[[feat["ml_bcomm_open"], feat["ml_bcomm_close"]], 2] = [1, -1]
    ml_proj[[feat["code_lparen"], feat["code_rparen"]], 3] = [1, -1]
    ml_proj[feat["comment_mask"], 4] = 1
    ml_proj[feat["code_chars"], 5] = 1
    ml_proj[feat["err_check"], 6] = 1
    ml_proj_s[5] = float(MAX_COLS)
    trit_mats["ml_proj"] = ml_proj
    vecs_1d["ml_proj_s"] = ml_proj_s

    ml_gate_w = np.zeros((N_ML_IN, N_ML_GATE), dtype=np.int8)
    ml_gate_s = np.ones((N_ML_GATE,), dtype=np.float32)
    ml_gate_b = np.zeros((N_ML_GATE,), dtype=np.float32)
    for step in range(N_DQ_SAW):
        saw_s = 2.0 if step > 0 else 1.0
        ml_gate_w[0, step] = 1
        ml_gate_s[step] = saw_s
        ml_gate_b[step] = -float(step) * saw_s
    for step in range(N_SQ_SAW):
        col = N_DQ_SAW + step
        saw_s = 2.0 if step > 0 else 1.0
        ml_gate_w[1, col] = 1
        ml_gate_s[col] = saw_s
        ml_gate_b[col] = -float(step) * saw_s

    bc_hi, bc_lo = N_DQ_SAW + N_SQ_SAW, N_DQ_SAW + N_SQ_SAW + 1
    ml_gate_w[2, [bc_hi, bc_lo]] = 1
    ml_gate_b[[bc_hi, bc_lo]] = [0.0, -1.0]
    trit_mats["ml_gate_w"] = ml_gate_w
    vecs_1d["ml_gate_s"], vecs_1d["ml_gate_b"] = ml_gate_s, ml_gate_b

    n_line_conv = N_PAREN_SUF + N_PREV_LINE
    ml_paren_w = np.zeros((2 * N_PAREN_STEPS + 1, N_ML_IN, n_line_conv), dtype=np.int8)
    ml_paren_s = np.ones((n_line_conv,), dtype=np.float32)
    ml_paren_b = np.zeros((n_line_conv,), dtype=np.float32)
    for step_k in range(N_PAREN_STEPS):
        span_k = step_k + 1
        u_hi, u_lo = 2 * step_k, 2 * step_k + 1
        ml_paren_w[N_PAREN_STEPS - span_k : N_PAREN_STEPS, 3, [u_hi, u_lo]] = 1
        ml_paren_b[[u_hi, u_lo]] = [0.0, -1.0]

    # Tap L-1 only: clamp(comment cells - 128 * code chars) and clamp(err != nil count)
    prev_tap = N_PAREN_STEPS - 1
    prev_comm, prev_err = N_PAREN_SUF, N_PAREN_SUF + 2
    ml_paren_w[prev_tap, [4, 5], prev_comm] = [1, -1]
    ml_paren_w[prev_tap, [4, 5], prev_comm + 1] = [1, -1]
    ml_paren_w[prev_tap, 6, [prev_err, prev_err + 1]] = 1
    ml_paren_b[N_PAREN_SUF:] = [0.0, -1.0, 0.0, -1.0]
    trit_mats["ml_paren_w"] = ml_paren_w
    vecs_1d["ml_paren_s"], vecs_1d["ml_paren_b"] = ml_paren_s, ml_paren_b


def _stamp_v06_member_vetoes(
    conv7_w: np.ndarray,
    conv7_s: np.ndarray,
    conv7_b: np.ndarray,
    start_ch: int,
    off_c7: int,
    feat: dict[str, int],
) -> int:
    """Stamp k=7 vetoes for `->x`, Fortran `s%x` / `c%field`, and regex `\\d` escapes."""
    ch_arr, ch_pr, ch_pl, ch_esc = range(start_ch, start_ch + 4)
    _stamp_char(conv7_w, 1, ch_arr, "-")
    _stamp_char(conv7_w, 2, ch_arr, ">")
    conv7_w[3, [HC_LO, HC_COMM, HC_STR], ch_arr] = [1, -1, -1]
    conv7_w[4, [HC_ID, HC_DOT, HC_QT], ch_arr] = -1
    conv7_s[ch_arr], conv7_b[ch_arr] = 2.0, -29.0

    conv7_w[[0, 1], HC_ID, ch_pr] = 1
    _stamp_char(conv7_w, 2, ch_pr, "%")
    conv7_w[3, [HC_LO, HC_COMM, HC_STR], ch_pr] = [1, -1, -1]
    conv7_w[4, [HC_ID, HC_DOT, HC_QT], ch_pr] = -1
    conv7_s[ch_pr], conv7_b[ch_pr] = 2.0, -19.0

    conv7_w[1, [HC_ID, HC_DOT, HC_QT], ch_pl] = -1
    conv7_w[2, [HC_LO, HC_COMM, HC_STR], ch_pl] = [1, -1, -1]
    _stamp_char(conv7_w, 3, ch_pl, "%")
    conv7_w[[4, 5], HC_ID, ch_pl] = 1
    conv7_s[ch_pl], conv7_b[ch_pl] = 2.0, -19.0

    _stamp_char(conv7_w, 2, ch_esc, "\\")
    conv7_w[3, [HC_LO, HC_COMM, HC_STR], ch_esc] = [1, -1, -1]
    conv7_w[4, [HC_ID, HC_DOT, HC_QT], ch_esc] = -1
    conv7_s[ch_esc], conv7_b[ch_esc] = 2.0, -15.0

    names = ("arrow_1ch", "pct_rhs_1ch", "pct_lhs_1ch", "esc_1ch")
    for offset, name in enumerate(names):
        feat[name] = off_c7 + start_ch + offset
    return start_ch + 4


def _stamp_friction_convs(
    conv3_w: np.ndarray,
    conv3_s: np.ndarray,
    conv3_b: np.ndarray,
    conv7_w: np.ndarray,
    conv7_s: np.ndarray,
    conv7_b: np.ndarray,
    conv13_w: np.ndarray,
    conv13_s: np.ndarray,
    conv13_b: np.ndarray,
    feat: dict[str, int],
    offsets: tuple[int, int, int],
    starts: tuple[int, int, int],
) -> None:
    """Stamp k=3/7/13 detectors for cursor mutation, out-arg/flag signatures, and state soup."""
    off_c3, off_c7, off_c13 = offsets
    c3_start, c7_start, c13_start = starts
    _sub_filter(conv3_w, conv3_s, conv3_b, c3_start, ",")
    feat["code_comma"] = off_c3 + c3_start
    for offset, pat in enumerate(("++", "^=", "|="), 1):
        ch_i = c3_start + offset
        _sub_filter(conv3_w, conv3_s, conv3_b, ch_i, pat)
        feat[pat] = off_c3 + ch_i

    for offset, (pat, wl, wr, cs, ld) in enumerate(_K7_FRICTION):
        ch_i = c7_start + offset
        _sub_filter(conv7_w, conv7_s, conv7_b, ch_i, pat, wl=wl, wr=wr, cs=cs, lead=ld)
        if pat == "out_":
            conv7_w[1, HC_DOT, ch_i] = -1
        feat[pat] = off_c7 + ch_i

    ch_bdo = c7_start + len(_K7_FRICTION)
    _sub_filter(conv7_w, conv7_s, conv7_b, ch_bdo, "do", wl=1, eol=1, cs=0, lead=1)
    feat["bare_do"] = off_c7 + ch_bdo

    # Tuple/stack-top index increment: `1] += 1` or `0] += 1` (never matches `map["k"] += 1`)
    ch_tup = ch_bdo + 1
    conv7_w[0, HC_DIG, ch_tup] = 1
    for pos, ch_str in enumerate("] += 1", 1):
        _stamp_char(conv7_w, pos, ch_tup, ch_str)
    conv7_w[3, [HC_COMM, HC_STR], ch_tup] = -1
    conv7_s[ch_tup], conv7_b[ch_tup] = 2.0, -85.0
    feat["tup_idx_inc"] = off_c7 + ch_tup

    # C/C++/Zig/PHP pointer & reference parameter sigils (` *id`, `type* id`, ` &id`, `type& id`)
    ch_pl, ch_pt, ch_rl, ch_rt = ch_tup + 1, ch_tup + 2, ch_tup + 3, ch_tup + 4
    conv7_w[2, HC_CNS, ch_pl] = -1
    _stamp_char(conv7_w, 3, ch_pl, "*")
    conv7_w[4, [HC_ID, HC_DIG], ch_pl] = [1, -1]
    conv7_w[3, [HC_COMM, HC_STR], ch_pl] = -1
    conv7_s[ch_pl], conv7_b[ch_pl] = 2.0, -15.0

    conv7_w[2, [HC_ID, HC_DIG], ch_pt] = [1, -1]
    _stamp_char(conv7_w, 3, ch_pt, "*")
    conv7_w[4, HC_CNS, ch_pt] = -1
    conv7_w[5, HC_LO, ch_pt] = 1
    conv7_w[3, [HC_COMM, HC_STR], ch_pt] = -1
    conv7_s[ch_pt], conv7_b[ch_pt] = 2.0, -17.0

    conv7_w[2, HC_CNS, ch_rl] = -1
    _stamp_char(conv7_w, 3, ch_rl, "&")
    conv7_w[4, HC_LO, ch_rl] = 1
    conv7_w[3, [HC_COMM, HC_STR], ch_rl] = -1
    conv7_s[ch_rl], conv7_b[ch_rl] = 2.0, -15.0

    conv7_w[[1, 2], HC_CNS, ch_rt] = 1
    _stamp_char(conv7_w, 3, ch_rt, "&")
    conv7_w[4, HC_CNS, ch_rt] = -1
    conv7_w[5, HC_LO, ch_rt] = 1
    conv7_w[3, [HC_COMM, HC_STR], ch_rt] = -1
    conv7_s[ch_rt], conv7_b[ch_rt] = 2.0, -19.0

    for ch_i, name in (
        (ch_pl, "ptr_lead_id"), (ch_pt, "ptr_trail_id"),
        (ch_rl, "ref_lead_id"), (ch_rt, "ref_trail_id"),
    ):
        feat[name] = off_c7 + ch_i

    for offset, (pat, wl, wr, cs) in enumerate(_K13_FRICTION):
        ch_i = c13_start + offset
        _sub_filter(conv13_w, conv13_s, conv13_b, ch_i, pat, wl=wl, wr=wr, cs=cs)
        feat[pat] = off_c13 + ch_i


def _build_block1(
    trit_mats: dict[str, np.ndarray],
    vecs_1d: dict[str, np.ndarray],
) -> dict[str, int]:
    """Compile {-1, 0, +1} k=3/7/13 convolutions, SimHash bank, and line-feature MLP."""
    feat: dict[str, int] = {}
    n_csum, n_conv3, n_conv7, n_conv13 = len(_CSUM_SPECS), 84, 268, 236
    off_c3 = n_csum
    off_c7 = off_c3 + n_conv3
    off_c13 = off_c7 + n_conv7

    csum_w = np.zeros((N_HC, n_csum), dtype=np.int8)
    csum_s = np.ones((n_csum,), dtype=np.float32)
    for idx, (name, hc_col, scale) in enumerate(_CSUM_SPECS):
        csum_w[hc_col, idx] = 1
        csum_s[idx] = scale
        feat[name] = idx
    trit_mats["col_sum_proj"], vecs_1d["col_sum_s"] = csum_w, csum_s

    conv3_w = np.zeros((3, N_HC, n_conv3), dtype=np.int8)
    conv3_s = np.ones((n_conv3,), dtype=np.float32)
    conv3_b = np.zeros((n_conv3,), dtype=np.float32)
    conv3_w[[0, 1, 2], [HC_LO, HC_UND, HC_LO], 0] = 1
    conv3_w[[0, 1], [HC_LO, HC_UP], 1] = 1
    conv3_w[0, [HC_ID, HC_DOT, HC_QT], 2] = -1
    conv3_w[1, HC_LO, 2] = 1
    conv3_w[2, [HC_ID, HC_DOT, HC_QT], 2] = -1

    conv3_w[[0, 1, 2], [HC_ID, HC_LO, HC_ID], 3] = 1
    conv3_w[0, [HC_ID, HC_QM, HC_LP, HC_RP, HC_RB, HC_DOT], 4] = -1
    conv3_w[1, HC_QM, 4] = 1
    conv3_w[2, [HC_DOT, HC_QM, HC_EQ, HC_SEM, HC_RP], 4] = -1
    conv3_w[1, HC_COMM, :5] = -1
    conv3_w[1, HC_STR, :5] = -1
    conv3_s[:5] = 2.0
    conv3_b[:5] = [-5.0, -3.0, -1.0, -5.0, -1.0]
    for ch_i, name in enumerate(
        ("snake_case", "camel_case", "one_char_id", "three_char_id", "ternary_op")
    ):
        feat[name] = off_c3 + ch_i

    k3_code_pats = ("?.", "=>", "===", "{}", "||", "#", "??=")
    for ch_i, pat in enumerate(k3_code_pats, 5):
        _sub_filter(conv3_w, conv3_s, conv3_b, ch_i, pat)
        feat[pat] = off_c3 + ch_i

    ch_semi_eq, ch_d16, ch_d20, ch_fcall = 12, 13, 14, 15
    conv3_w[1, [HC_SEM, HC_EQ, HC_COMM, HC_STR], ch_semi_eq] = [1, 1, -1, -1]
    conv3_s[ch_semi_eq], conv3_b[ch_semi_eq] = 2.0, -1.0
    conv3_w[1, HC_D16, ch_d16] = 1
    conv3_w[1, HC_D20, ch_d20] = 1
    conv3_w[[0, 1], [HC_ID, HC_LP], ch_fcall] = 1
    conv3_w[1, [HC_COMM, HC_STR], ch_fcall] = -1
    conv3_s[ch_fcall], conv3_b[ch_fcall] = 2.0, -3.0
    for ch_i, name in enumerate(("semi_or_eq", "depth_ge16", "depth_ge20", "func_call"), 12):
        feat[name] = off_c3 + ch_i

    for ch_i, pat in enumerate(("??", "?:", "): ", " = "), 16):
        _sub_filter(conv3_w, conv3_s, conv3_b, ch_i, pat)
        feat[pat] = off_c3 + ch_i

    ch_doc_blk, ch_semi_only, ch_typed_arr = 20, 21, 22
    ch_comm_semi, ch_comm_fcall = 23, 24
    conv3_w[1, [HC_FNS, HC_COMM, HC_SL_C], ch_doc_blk] = [1, 1, -1]
    conv3_s[ch_doc_blk], conv3_b[ch_doc_blk] = 2.0, -3.0
    conv3_w[1, [HC_SEM, HC_COMM, HC_STR], ch_semi_only] = [1, -1, -1]
    conv3_s[ch_semi_only], conv3_b[ch_semi_only] = 2.0, -1.0

    conv3_w[0, HC_ID, ch_typed_arr] = 1
    _stamp_char(conv3_w, 1, ch_typed_arr, "[")
    _stamp_char(conv3_w, 2, ch_typed_arr, "]")
    conv3_w[1, [HC_COMM, HC_STR], ch_typed_arr] = -1
    conv3_s[ch_typed_arr], conv3_b[ch_typed_arr] = 2.0, -29.0

    conv3_w[1, [HC_SEM, HC_SL_C, HC_CODE], ch_comm_semi] = [1, 1, -1]
    conv3_s[ch_comm_semi], conv3_b[ch_comm_semi] = 2.0, -3.0
    conv3_w[[0, 1], [HC_ID, HC_LP], ch_comm_fcall] = 1
    conv3_w[1, [HC_SL_C, HC_CODE], ch_comm_fcall] = [1, -1]
    conv3_s[ch_comm_fcall], conv3_b[ch_comm_fcall] = 2.0, -5.0
    custom_c3_names = (
        "doc_or_block_comm", "semi_only", "typed_arr", "comm_semi", "comm_fcall",
    )
    for ch_i, name in enumerate(custom_c3_names, 20):
        feat[name] = off_c3 + ch_i

    k3_comm_pats = ("///", "//!", "===", "=>", "){", ") {", " = ")
    for ch_i, pat in enumerate(k3_comm_pats, 25):
        _sub_filter(conv3_w, conv3_s, conv3_b, ch_i, pat, code=False, comm=True)
        feat[f"c:{pat}"] = off_c3 + ch_i

    ch_brk1, ch_lp, ch_rp, ch_bopen, ch_bclose = 32, 33, 34, 35, 36
    _stamp_char(conv3_w, 0, ch_brk1, "[")
    conv3_w[1, HC_LO, ch_brk1] = 1
    _stamp_char(conv3_w, 2, ch_brk1, "]")
    conv3_w[1, [HC_COMM, HC_STR], ch_brk1] = -1
    conv3_s[ch_brk1], conv3_b[ch_brk1] = 2.0, -29.0

    conv3_w[1, [HC_LP, HC_COMM, HC_STR], ch_lp] = [1, -1, -1]
    conv3_w[1, [HC_RP, HC_COMM, HC_STR], ch_rp] = [1, -1, -1]
    conv3_s[[ch_lp, ch_rp]], conv3_b[[ch_lp, ch_rp]] = 2.0, -1.0

    _stamp_char(conv3_w, 1, ch_bopen, "/")
    _stamp_char(conv3_w, 2, ch_bopen, "*")
    conv3_w[0, [HC_ID, HC_DOT, HC_QT], ch_bopen] = -1
    conv3_w[1, [HC_RAW_STR, HC_SL_C], ch_bopen] = -1
    _stamp_char(conv3_w, 1, ch_bclose, "*")
    _stamp_char(conv3_w, 2, ch_bclose, "/")
    conv3_w[0, HC_QT, ch_bclose] = -1
    conv3_w[1, [HC_RAW_STR, HC_SL_C], ch_bclose] = -1
    conv3_s[[ch_bopen, ch_bclose]], conv3_b[[ch_bopen, ch_bclose]] = 2.0, -27.0

    v05_c3_names = ("bracket_1ch", "code_lparen", "code_rparen", "ml_bcomm_open", "ml_bcomm_close")
    for ch_i, name in enumerate(v05_c3_names, 32):
        feat[name] = off_c3 + ch_i
    for ch_i, pat in enumerate(_K3_TERNARY_VETO, 37):
        _sub_filter(conv3_w, conv3_s, conv3_b, ch_i, pat)
        feat[pat] = off_c3 + ch_i
    c3_next = 37 + len(_K3_TERNARY_VETO)

    conv7_w = np.zeros((7, N_HC, n_conv7), dtype=np.int8)
    conv7_s = np.ones((n_conv7,), dtype=np.float32)
    conv7_b = np.zeros((n_conv7,), dtype=np.float32)
    str_bearing = {"|| '')", "strict'"}
    k7_groups = ((0, _K7_PRIMARY), (60, _K7_EXTENDED), (85, _K7_V04_CODE))
    for start_ch, group in k7_groups:
        for offset, (pat, wl, wr, cs, ld) in enumerate(group):
            ch_i = start_ch + offset
            _sub_filter(
                conv7_w, conv7_s, conv7_b, ch_i, pat,
                wl=wl, wr=wr, cs=cs, lead=ld,
                req_left_id=int(pat in _SUFFIX_GRAVEYARD),
                no_str=(pat not in str_bearing),
            )
            feat[pat] = off_c7 + ch_i

    ch_eq_digit = 46
    conv7_w[[1, 2, 3, 5], [HC_EQ, HC_EQ, HC_EQ, HC_DIG], ch_eq_digit] = 1
    conv7_w[2, [HC_COMM, HC_STR], ch_eq_digit] = -1
    conv7_s[ch_eq_digit], conv7_b[ch_eq_digit] = 2.0, -7.0
    feat["===_digit"] = off_c7 + ch_eq_digit

    comm_cues_a = (
        "step 1", "step 2", "step 3", "todo:", "fixme:",
        "eslint-", "@ts-ign", "initial", "loop th", "check i",
    )
    comm_cues_b = ("type: i", "noqa", "nolint")
    for start_ch, group in ((50, comm_cues_a), (79, comm_cues_b)):
        for offset, pat in enumerate(group):
            ch_i = start_ch + offset
            _sub_filter(conv7_w, conv7_s, conv7_b, ch_i, pat, code=False, comm=True)
            feat[pat] = off_c7 + ch_i

    ch_short_lhs, ch_multi_dig, ch_magic_idx, ch_pascal = 113, 114, 115, 116
    conv7_w[1, [HC_LO, HC_LEAD], ch_short_lhs] = 1
    conv7_w[2, [HC_LO, HC_LEAD, HC_COMM, HC_STR], ch_short_lhs] = [1, 1, -1, -1]
    conv7_w[3, [HC_ACT, HC_CNS], ch_short_lhs] = [1, -1]
    conv7_w[4, HC_EQ, ch_short_lhs] = 1
    conv7_w[5, HC_EQ, ch_short_lhs] = -1
    conv7_s[ch_short_lhs], conv7_b[ch_short_lhs] = 2.0, -11.0

    conv7_w[1, [HC_ID, HC_DOT, HC_EQ], ch_multi_dig] = -1
    conv7_w[2, [HC_DIG, HC_COMM, HC_STR], ch_multi_dig] = [1, -1, -1]
    conv7_w[3, HC_DIG, ch_multi_dig] = 1
    conv7_s[ch_multi_dig], conv7_b[ch_multi_dig] = 2.0, -3.0

    conv7_w[0, HC_ID, ch_magic_idx] = 1
    _stamp_char(conv7_w, 1, ch_magic_idx, "[")
    conv7_w[2, [HC_DIG, HC_COMM, HC_STR], ch_magic_idx] = [1, -1, -1]
    conv7_w[3, HC_DIG, ch_magic_idx] = 1
    conv7_s[ch_magic_idx], conv7_b[ch_magic_idx] = 2.0, -19.0

    ch_decl_short = 121
    conv7_w[0, [HC_ACT, HC_CNS], ch_decl_short] = [1, -1]
    conv7_w[1, HC_LO, ch_decl_short] = 1
    conv7_w[2, [HC_LO, HC_COMM, HC_STR], ch_decl_short] = [1, -1, -1]
    conv7_w[3, [HC_ACT, HC_CNS], ch_decl_short] = [1, -1]
    conv7_w[4, HC_EQ, ch_decl_short] = 1
    conv7_w[5, HC_EQ, ch_decl_short] = -1
    conv7_s[ch_decl_short], conv7_b[ch_decl_short] = 2.0, -9.0
    feat["decl_short_lhs"] = off_c7 + ch_decl_short

    conv7_w[1, HC_ID, ch_pascal] = 1
    _stamp_char(conv7_w, 2, ch_pascal, ":")
    _stamp_char(conv7_w, 3, ch_pascal, " ")
    conv7_w[4, HC_UP, ch_pascal] = 1
    conv7_w[5, HC_LO, ch_pascal] = 1
    conv7_w[2, [HC_COMM, HC_STR], ch_pascal] = -1
    conv7_s[ch_pascal], conv7_b[ch_pascal] = 2.0, -33.0
    feat["short_lhs"] = off_c7 + ch_short_lhs
    feat["multi_digit"] = off_c7 + ch_multi_dig
    feat["magic_index"] = off_c7 + ch_magic_idx
    feat["param_pascal"] = off_c7 + ch_pascal

    for offset, (pat, cs) in enumerate(_K7_V04_COMM):
        ch_i = 102 + offset
        _sub_filter(conv7_w, conv7_s, conv7_b, ch_i, pat, code=False, comm=True, cs=cs)
        feat[f"c:{pat}"] = off_c7 + ch_i

    for offset, pat in enumerate(_K7_DIRECTIVE_COMM):
        ch_i = 122 + offset
        _sub_filter(conv7_w, conv7_s, conv7_b, ch_i, pat, code=False, comm=True)
        feat[f"c:{pat}"] = off_c7 + ch_i
    _sub_filter(conv7_w, conv7_s, conv7_b, 124, _K7_TEST_METHOD, cs=1)
    feat[_K7_TEST_METHOD] = off_c7 + 124
    _sub_filter(conv7_w, conv7_s, conv7_b, 125, ", nil", eol=1)
    feat["nil_eol"] = off_c7 + 125

    _stamp_triple_quote(conv7_w, conv7_s, conv7_b, 117, '"', require_lead=False)
    _stamp_triple_quote(conv7_w, conv7_s, conv7_b, 118, "'", require_lead=False)
    _stamp_triple_quote(conv7_w, conv7_s, conv7_b, 119, '"', require_lead=True)
    _stamp_triple_quote(conv7_w, conv7_s, conv7_b, 120, "'", require_lead=True)
    for ch_i, name in enumerate(("triple_dq", "triple_sq", "lead_tdq", "lead_tsq"), 117):
        feat[name] = off_c7 + ch_i

    c7_v06 = _stamp_v06_member_vetoes(conv7_w, conv7_s, conv7_b, 126, off_c7, feat)
    for offset, (pat, wl, wr, cs, ld) in enumerate(_K7_V06_CODE):
        ch_i = c7_v06 + offset
        _sub_filter(conv7_w, conv7_s, conv7_b, ch_i, pat, wl=wl, wr=wr, cs=cs, lead=ld)
        feat[pat] = off_c7 + ch_i
    ch_pcall_stmt = c7_v06 + len(_K7_V06_CODE)
    _sub_filter(conv7_w, conv7_s, conv7_b, ch_pcall_stmt, "pcall(", wl=1, cs=1, lead=1)
    feat["pcall_stmt"] = off_c7 + ch_pcall_stmt
    c7_next = ch_pcall_stmt + 1

    conv13_w = np.zeros((13, N_HC, n_conv13), dtype=np.int8)
    conv13_s = np.ones((n_conv13,), dtype=np.float32)
    conv13_b = np.zeros((n_conv13,), dtype=np.float32)
    conv13_w[:, HC_ID, 0] = 1
    conv13_w[6, [HC_COMM, HC_STR], 0] = -1
    conv13_s[0], conv13_b[0] = 2.0, -25.0
    feat["ident_13ch"] = off_c13

    lead_k13 = {"import ", "package ", "#include", "except Excep", "except BaseE"}
    for start_ch, group in ((1, _K13_PRIMARY), (22, _K13_EXTENDED)):
        for offset, (pat, cs) in enumerate(group):
            ch_i = start_ch + offset
            _sub_filter(
                conv13_w, conv13_s, conv13_b, ch_i, pat,
                cs=cs, lead=int(pat in lead_k13), no_str=(pat != "internal/"),
            )
            feat[pat] = off_c13 + ch_i

    for ch_i, pat in ((20, "eslint-disabl"), (21, "@ts-ignore")):
        _sub_filter(conv13_w, conv13_s, conv13_b, ch_i, pat, code=False, comm=True)
        feat[pat] = off_c13 + ch_i

    str_k13 = {
        'return ""', "return ''", "secret-key",
        "password123", "dev-secret", "changeme",
    }
    for offset, (pat, wl, wr, cs) in enumerate(_K13_V04_CODE):
        ch_i = 25 + offset
        _sub_filter(
            conv13_w, conv13_s, conv13_b, ch_i, pat,
            wl=wl, wr=wr, cs=cs,
            req_left_id=int(pat in _SUFFIX_GRAVEYARD),
            no_str=(pat not in str_k13),
        )
        feat[pat] = off_c13 + ch_i

    for offset, (pat, cs) in enumerate(_K13_V04_COMM):
        ch_i = 62 + offset
        _sub_filter(conv13_w, conv13_s, conv13_b, ch_i, pat, code=False, comm=True, cs=cs)
        feat[f"c:{pat}"] = off_c13 + ch_i

    for offset, pat in enumerate(_K13_TEST_DECLS):
        ch_i = 75 + offset
        _sub_filter(conv13_w, conv13_s, conv13_b, ch_i, pat, cs=1, lead=1)
        feat[pat] = off_c13 + ch_i
    _sub_filter(conv13_w, conv13_s, conv13_b, 80, "err != nil")
    feat["err_check"] = off_c13 + 80
    for offset, pat in enumerate(_K13_BROAD_SUPPRESS):
        ch_i = 81 + offset
        _sub_filter(conv13_w, conv13_s, conv13_b, ch_i, pat, cs=1)
        feat[pat] = off_c13 + ch_i
    for offset, (pat, wl, wr, cs) in enumerate(_K13_V06_CODE):
        ch_i = 83 + offset
        _sub_filter(conv13_w, conv13_s, conv13_b, ch_i, pat, wl=wl, wr=wr, cs=cs)
        if pat == "return false":
            conv13_w[12, HC_SEM, ch_i] = -1
        feat[pat] = off_c13 + ch_i
    c13_next = 83 + len(_K13_V06_CODE)

    _stamp_friction_convs(
        conv3_w, conv3_s, conv3_b, conv7_w, conv7_s, conv7_b,
        conv13_w, conv13_s, conv13_b, feat,
        (off_c3, off_c7, off_c13), (c3_next, c7_next, c13_next),
    )

    num_raw_u = n_csum + n_conv3 + n_conv7 + n_conv13
    _build_multiline_gates(trit_mats, vecs_1d, feat, num_raw_u)

    # Declarative line-level rules compiled into strictly {-1, 0, +1} lmlp1_w & lmlp2_w
    module_headers = [
        ("from ", -3.0), ("use ", -3.0), ("import ", -3.0),
        ("package ", -3.0), ("#include", -3.0),
    ]
    decl_hdr_terms = [(pat, 3.0) for pat in _DECL_TYPE_DDL]
    type_field_veto = [
        (": string", -1.0), (": number", -1.0), (": boolean", -1.0),
        ("typed_arr", -1.0), ("param_pascal", -1.0),
    ] + [(pat, -2.0) for pat in _DECL_TYPE_DDL]
    hardcoded_branch_terms = [
        ("if", 1.5), ("elseif ", 1.5), ("return", 1.0), ("== '", 1.5), ('== "', 1.5),
        ("===_digit", 1.5), ("== -1", 1.5), ("||", -2.5), ("typeof", -2.5),
        ("type(", -2.5), (".code", -3.0), ("=== undefined", -3.0),
    ]
    # Service and Builder are domain and stdlib type names (strings.Builder, StringBuilder);
    # Impl is clamped separately via paired ReLU units so repeating `opts.fetchImpl` stays clean
    ceremony_terms = [
        ("Manager", 1.0), ("Factory", 1.2), ("Service", 0.3), ("Builder", 0.3),
        ("Wrapper", 1.0), ("Helper", 0.9), ("Abstract", 1.3),
        ("Repository", 1.1), ("Configuration", 0.8), ("Controller", 0.9),
        ("Strategy", 1.0), ("Provider", 0.9), ("Processor", 0.9),
        ("_old", 1.5), ("_bak", 1.5), ("_backup", 1.5),
        ("_deprecated", 1.5), ("Deprecated", 1.5),
    ]
    stdlib_idiom_terms = [
        ("Prototype", 12.0), ("ERR_", 10.0), ("SAFE_IN", 10.0),
        ("HideStackFram", 12.0), ("primordials", 8.0), ("EventEmitter", 12.0),
        ("assert(", 8.0), ("#[test]", 12.0),
    ]
    stmt_terms = [
        ("semi_or_eq", 1.0), ("func_call", 1.0), ("return", 1.0),
        ("throw", 1.0), ("raise", 1.0),
    ] + module_headers + type_field_veto
    imperative_terms = [
        ("code_chars", 0.05), ("semi_or_eq", 0.65), ("func_call", 0.65),
        ("indent_cols", 0.25), ("ERR_", -3.0), ("Prototype", -3.0),
        ("require", -3.0), ("exports", -3.0), ("return", -0.8),
        ("throw", -0.8), ("raise", -0.8),
    ] + module_headers

    member_1ch_vetoes = [
        ("arrow_1ch", -1.0), ("pct_rhs_1ch", -1.0),
        ("pct_lhs_1ch", -1.0), ("esc_1ch", -1.0),
    ]
    one_char_terms = [
        ("one_char_id", 1.0), ("bracket_1ch", -1.0),
        ("for", -2.0), ("for (", -1.5), ("For ", -2.0),
    ] + member_1ch_vetoes
    short_golf_terms = [
        ("short_lhs", 1.6), ("decl_short_lhs", 1.6), ("one_char_id", 0.5),
        ("bracket_1ch", -0.5), ("for", -1.0), ("for (", -0.8), ("For ", -1.0),
        ("three_char_id", -0.16), ("arrow_1ch", -0.5), ("pct_rhs_1ch", -0.5),
        ("pct_lhs_1ch", -0.5), ("esc_1ch", -0.5),
    ]
    magic_num_terms = [("magic_index", 1.0), ("multi_digit", 0.30)]
    fallback_swallow_terms = [
        ("|| {}", 1.4), ("|| []", 1.4), ("|| []}", -2.0), ("|| {}}", -2.0),
        ("|| '')", 1.0), ("type: i", 1.8), ("noqa", 1.6), ("nolint", 1.6),
        ("#[allow(", 0.7), ("eslint-disabl", 2.0), ("@ts-ignore", 2.0), ("?.", 0.45),
        (" as any", 1.8), ("as unknown", 1.8), ("cast(Any", 1.8),
        ("c:pyright: ign", 1.8), ("c:biome-ignore", 1.8),
        ("_ = try?", 2.0), ("try? ", 2.0), ("pcall_stmt", 2.0),
    ] + [(pat, 1.8) for pat in _K13_BROAD_SUPPRESS]
    or_coll_guard = float(MAX_COLS // 2)
    or_coll_terms = [
        (" or []", or_coll_guard), (" or {}", or_coll_guard),
        (" or [])", -or_coll_guard), (" or {})", -or_coll_guard),
        ("func_call", 1.0), ("return", 1.0), ("type(", -1.5),
    ]
    real_handler_terms = [
        ("throw", 1.5), ("raise", 1.5), (".code", 1.5), ("reject(", 1.5),
        ("cb(", 1.5), ("emit(", 1.0), ("destroy", 1.2), ("callback(", 1.5),
        ("process.exit", 2.0), ("exitCode", 2.0), ("os.Exit(", 2.0),
        ("sys.exit(", 2.0), ("os.exit(", 2.0), (".error =", 2.0),
        ("Throw", 2.0), (".Log", 2.0), ("stderr", 2.0), ("error(", 2.0), (", err", 2.0),
    ]
    ternary_terms = [
        ("ternary_op", 1.0), ("?:", -1.0), ("<?", -1.0), ("?>", -1.0), (", ?", -1.0),
    ]
    container_hdr_terms = [
        ("class ", 2.0), ("impl ", 2.0), ("impl<", 2.0), ("namespace ", 2.0),
    ]
    # Test names describe the unit under test: allow length and the SUT's own type names
    test_decl_terms = [(pat, -3.0) for pat in (*_K13_TEST_DECLS, _K7_TEST_METHOD)]
    # Empty `{}` only signals a swallowed error beside `catch`; a line holds at most 64 pairs
    catch_gate = float(MAX_COLS // 2)
    pure_comment_terms = [("comment_chars", 1.0), ("code_chars", -5.0)] + [
        (f"c:{pat}", -float(MAX_COLS)) for pat in _K7_DIRECTIVE_COMM
    ]

    line_rules: list[tuple[list[tuple[str, float]], float, int, float]] = [
        ([("has_nonspace", 1.0)], 0.0, Z_ACTIVE, 1.0),
        ([("code_chars", 1.0)], 0.0, Z_HAS_CODE, 1.0),
        ([("code_chars", 1.0)], -1.0, Z_HAS_CODE, -1.0),
        (pure_comment_terms, -2.0, Z_PURE_COMMENT, 1.0),
        (pure_comment_terms, -3.0, Z_PURE_COMMENT, -1.0),
        ([("comment_chars", -0.25), ("code_chars", -5.0)], 11.0, Z_SHORT_COMMENT, 1.0),
        ([("comment_chars", -0.25), ("code_chars", -5.0)], 10.0, Z_SHORT_COMMENT, -1.0),

        ([("comment_mask", 1.0), ("code_chars", -5.0)], -2.0, Z_PARAGRAPH_BREAK, 1.0),
        ([("comment_mask", 1.0), ("code_chars", -5.0)], -3.0, Z_PARAGRAPH_BREAK, -1.0),
        ([("has_nonspace", -1.0)], 1.0, Z_PARAGRAPH_BREAK, 1.0),
        ([(pat, 1.0) for pat in _DECL_TYPE_DDL], 0.0, Z_PARAGRAPH_BREAK, 1.0),
        ([("open_braces", 1.0), ("close_braces", 1.0), ("code_chars", -0.4)], 0.2, Z_OPEN_NET, 1.0),
        ([("open_braces", 1.0), ("close_braces", 1.0), ("code_chars", -0.4)], -0.8, Z_OPEN_NET, -1.0),

        ([("indent_cols", 1.0)], 0.0, Z_INDENT, 1.0),
        ([("depth_ge8", 1.0)], 0.0, Z_DEPTH8, 1.0),
        ([("depth_ge10", 1.0)], 0.0, Z_DEPTH10, 1.0),
        ([("depth_ge12", 1.0)], 0.0, Z_DEPTH12, 1.0),
        ([("odd_indent", 1.0)], 0.0, Z_ODD_INDENT, 1.0),
        ([("mod4_eq2_indent", 1.0)], 0.0, Z_MOD4_EQ2, 1.0),
        ([("mod4_eq0_indent", 1.0)], 0.0, Z_MOD4_EQ0, 1.0),
        ([("depth_ge16", 0.35), ("depth_ge20", 1.25), ("depth_ge24", 0.8)], 0.0, Z_DEEP_PYRAMID, 1.0),
        ([("depth_ge24", 1.7)], 0.0, Z_PYR_CLASS, 1.0),
        ([("depth_ge16", -1.0)], 1.0, Z_SHALLOW_LINE, 1.0),
        ([(name, 1.0) for name, _ in module_headers] + decl_hdr_terms, 0.0, Z_MODULE_HEADER, 1.0),

        ([("if", 1.0), ("else if", -1.0), ("} else if", -1.0), ("elseif ", -1.0),
          ("for", 0.8), ("For ", 0.8), ("while", 0.8), ("match", 0.8)],
         0.0, Z_BRANCH_OPEN, 1.0),
        ([("else", 1.0), ("else if", -1.0)], 0.0, Z_ELSE_BRANCH, 1.0),
        ([("return", 1.0), ("throw", 1.2), ("raise", 1.2), ("break", 0.8),
          ("continue", 0.8), ("depth_ge16", -2.5)], 0.0, Z_EARLY_EXIT, 1.0),
        ([("if", 1.0), ("return", 1.0), ("throw", 1.0),
          ("raise", 1.0), ("depth_ge16", -2.5)], -1.5, Z_GUARD_CLAUSE, 2.0),
        (hardcoded_branch_terms, -3.2, Z_HARDCODED_CASE, 1.0),
        (hardcoded_branch_terms, -4.2, Z_HARDCODED_CASE, -1.0),
        ([("switch", 2.0), ("match", 2.0), ("case ", 1.5), ("break", 1.5)], 0.0, Z_SWITCH_BLOCK, 1.0),

        (ceremony_terms + test_decl_terms, 0.0, Z_CEREMONY, 1.0),
        ([("Impl", 1.0)] + test_decl_terms, 0.0, Z_CEREMONY, 1.1),
        ([("Impl", 1.0)] + test_decl_terms, -1.0, Z_CEREMONY, -1.1),
        (stdlib_idiom_terms, 0.0, Z_STDLIB_IDIOM, 1.0),
        ([("ident_13ch", 0.11)] + [(key, -wt * 0.25) for key, wt in stdlib_idiom_terms]
         + test_decl_terms + module_headers, -1.05, Z_LONG_IDENT, 1.0),
        (one_char_terms, 0.0, Z_ONE_CHAR_ID, 1.0),
        (one_char_terms, -1.0, Z_CRYPTIC_GATE, 1.0),
        (one_char_terms, -2.0, Z_CRYPTIC_GATE, -1.0),
        ([("three_char_id", 0.25)], 0.0, Z_WORD_COUNT, 0.9),

        ([("catch (", 1.0), (") when (", -2.0), ("except:", 1.2), ("except Excep", 1.2),
          ("except BaseE", 1.2), ("if not ok", 1.0)],
         0.0, Z_TRY_CATCH, 1.0),
        ([("Catch ", 2.0), ("As Exception", 2.0)], -3.0, Z_TRY_CATCH, 1.0),
        ([("catch", catch_gate), ("{}", 1.0)], -catch_gate - 0.5, Z_TRY_CATCH, 2.0),
        (real_handler_terms, 0.0, Z_REAL_HANDLER, 1.0),
        ([("{}", 1.5), ("console", 1.2), ("pass", 1.4), ("...", 1.5),
          ("End Try", 1.5), ("String.Empty", 1.5)],
         0.0, Z_EMPTY_HANDLER, 1.0),
        ([("return", 1.0), ("Return ", 1.0), ("None", 1.0), ("0.0", 1.0),
          ("null;", 1.0), ("false;", 1.0), ("return nil", 1.0), ("return false", 1.0)],
         -1.5, Z_EMPTY_HANDLER, 3.4),
        (fallback_swallow_terms, -0.95, Z_FALLBACK_SWALLOW, 1.2),
        (or_coll_terms, -or_coll_guard - 0.5, Z_FALLBACK_SWALLOW, 1.6),
        ([(",_:=", 2.0), ("for", -3.0), ("for (", -3.0)], -1.5, Z_FALLBACK_SWALLOW, 1.6),

        ([(".map(", 1.0), (".filter", 1.0), (".reduce", 1.0), (".flatMa", 1.0), (".unwrap", 0.7)],
         -1.45, Z_CHAINED_CALLS, 1.2),
        (ternary_terms, -1.2, Z_NESTED_TERNARY, 1.5),
        ([("cols_ge80", 1.0)], -0.45, Z_OVER_80, 1.0),
        ([("cols_ge96", 1.2), ("cols_ge110", 1.5)], -0.35, Z_OVER_96, 1.2),
        ([("step 1", 3.2), ("step 2", 3.2), ("step 3", 3.2), ("todo:", 2.2), ("fixme:", 2.2),
          ("initial", 1.3), ("loop th", 1.5), ("check i", 1.1), ("comment_chars", -0.045)],
         -0.8, Z_NARRATOR_TAG, 1.2),
        (stmt_terms, 0.0, Z_IS_STMT, 1.0),
        (stmt_terms, -1.0, Z_IS_STMT, -1.0),
        (imperative_terms, -1.8, Z_IMPERATIVE_LINE, 2.0),
        (imperative_terms, -2.3, Z_IMPERATIVE_LINE, -2.0),
        ([("strict'", 0.8), ("primordials", 1.0), ("ERR_", 0.6),
          ("validat", 0.6), ("#", 0.4), ("assert(", 0.8)], 0.0, Z_CLEAN_IDIOM, 1.0),

        (short_golf_terms, -0.5, Z_SHORT_GOLF, 1.0),
        (short_golf_terms, -1.5, Z_SHORT_GOLF, -1.0),
        (magic_num_terms, -0.95, Z_MAGIC_NUM, 1.0),
        (magic_num_terms, -1.95, Z_MAGIC_NUM, -1.0),
        ([("indent_cols", 2.0)], 0.0, Z_IS_INDENTED, 1.0),
        ([("indent_cols", 2.0)], -1.0, Z_IS_INDENTED, -1.0),
        (container_hdr_terms, 0.0, Z_CONTAINER_HDR, 1.0),
        (container_hdr_terms, -1.0, Z_CONTAINER_HDR, -1.0),
    ]

    sig_open_pats = (
        "function ", "def ", "fn ", "subroutine ", "=>", "func_call", "): ", ") ->",
    )
    static_type_pats = (
        ": string", ": number", ": boolean", ": str",
        ": int", ": float", ": bool", ": list",
        ": dict", ": Record<", "typed_arr", "param_pascal",
        " :: ", "intent(", ": f64", ": f32",
        ": i32", ": i64", ": usize", "double ", "float ",
    )
    math_sig_pats = (
        " :: ", "intent(", "real(", ": f64", ": f32",
        ": i32", ": i64", ": usize", "Vec",
    )
    sig_veto_pats = (
        "export ", "pub ", "public ", "any",
        "Any", "unknown", ": object", "Optional",
        "null", "undefined", "= None", "?:", " | ",
    )
    neutral_ret_pats = (
        "return 0", "return []", "return {}", 'return ""',
        "return ''", "return null", "return None",
    )
    env_lookup_pats = ("process.env", ".meta.env", "os.getenv(", "os.environ")
    dev_default_pats = ("secret-key", "password123", "dev-secret", "changeme")
    env_veto_pats = (
        "if", "elif", "else if", "while",
        "throw", "raise", "ERR_", "assert(", "assert ",
    )
    elision_pats = (
        "c:rest of ", "c:existing cod", "c:placeholder",
        "c:remain uncha", "c:// ...", "c:# ...",
    )
    comm_code_pats = (
        "comm_semi", "c:===", "c:=>", "c:){",
        "c:) {", "c:// }", "c:# def ", "c:  def ",
        "c:# if ", "c:  if ", "c:# for ", "c:  for ",
        "c:// const ", "c:  const ", "c:// let ", "c:  let ",
        "c:// return ", "c:# return ", "c:  return ",
    )
    clamped_indicators = (
        (sig_open_pats, Z_SIG_OPEN),
        (static_type_pats, Z_STATIC_TYPE),
        (sig_veto_pats, Z_SIG_VETO),
        (("semi_only",), Z_SEMI_ONLY),
        (("if",), Z_IF_LINE),
        (("Array.isArray", "typeof", "isinstance("), Z_TYPE_HEDGE),
        (("throw", "raise", "ERR_", "assert(", "assert "), Z_FAIL_FAST),
        (neutral_ret_pats, Z_NEUTRAL_RET),
        (env_lookup_pats, Z_HAS_ENV),
        (("||", "??", " or "), Z_HAS_COALESCE),
        (dev_default_pats, Z_DEV_DEFAULT),
        (env_veto_pats, Z_ENV_VETO),
        ((" = ",), Z_HAS_ASSIGN),
        (elision_pats, Z_ELISION_RAW),
        (("doc_or_block_comm", "c:///", "c://!", "code_chars"), Z_LINE_COMM_VETO),
        (("c:```", "c:>>>"), Z_COMM_FENCE),
        (comm_code_pats, Z_COMM_STRONG_CUE),
        (("c: = ",), Z_COMM_ASSIGN),
        (("comm_fcall",), Z_COMM_FCALL),
    )
    for pat_names, dst_reg in clamped_indicators:
        unit_terms = [(key, 1.0) for key in pat_names]
        line_rules.append((unit_terms, 0.0, dst_reg, 1.0))
        line_rules.append((unit_terms, -1.0, dst_reg, -1.0))

    two_env_terms = [(key, 1.0) for key in env_lookup_pats]
    line_rules.append((two_env_terms, -1.0, Z_TWO_ENVS, 1.0))
    line_rules.append((two_env_terms, -2.0, Z_TWO_ENVS, -1.0))

    # Track 2 polyglot line-level rules (Z[160..167])
    math_terms = [(key, 1.0) for key in math_sig_pats]
    lua_ok_gate = 16.0
    lua_and_or_terms = [
        ("ok and ", lua_ok_gate), (" or 0", 1.0), (" or nil", 1.0), (" or false", 1.0),
        (" or {}", 1.0), (" or []", 1.0), (" or {})", -2.0), ("type(", -2.0),
    ]
    line_rules.extend([
        (math_terms, 0.0, Z_MATH_SIG, 0.36),
        (math_terms, -1.0, Z_MATH_SIG, -0.36),
        ([("runCatching", 1.0)], 0.0, Z_KT_RUNCATCH, 1.0),
        ([("runCatching", 1.0)], -1.0, Z_KT_RUNCATCH, -1.0),
        ([(".onFailure", 1.0)], 0.0, Z_KT_RUNCATCH, -3.0),
        ([(".getOrNull()", 1.0), (".getOrDefaul", 1.0), (".onFailure", -2.0)],
         0.0, Z_KT_GETNULL, 8.0),
        ([(".getOrNull()", 1.0), (".getOrDefaul", 1.0), (".onFailure", -2.0)],
         -1.0, Z_KT_GETNULL, -8.0),
        (lua_and_or_terms, -lua_ok_gate - 0.5, Z_KT_GETNULL, 16.0),
        (lua_and_or_terms, -lua_ok_gate - 1.0, Z_KT_GETNULL, -16.0),
        ([("= try? ", 1.0), (" ?? ", -2.0)], 0.0, Z_SWIFT_TRYQ, 8.0),
        ([("= try? ", 1.0), (" ?? ", -2.0)], -1.0, Z_SWIFT_TRYQ, -8.0),
        ([("return nil", 1.0), ("return false", 1.0)], 0.0, Z_SWIFT_NIL, 1.0),
        ([("return nil", 1.0), ("return false", 1.0)], -1.0, Z_SWIFT_NIL, -1.0),
        ([("bool Try", 1.0)], 0.0, Z_TRY_BOOL_SIG, 1.0),
        ([("bool Try", 1.0)], -1.0, Z_TRY_BOOL_SIG, -1.0),
        ([("catch (", 1.0), ("catch (Excep", -2.0)], 0.0, Z_NARROW_CATCH, 1.0),
        ([("catch (", 1.0), ("catch (Excep", -2.0)], -1.0, Z_NARROW_CATCH, -1.0),
        ([("pcall(", 1.0)], 0.0, Z_LUA_PCALL, 1.0),
        ([("pcall(", 1.0)], -1.0, Z_LUA_PCALL, -1.0),
    ])

    # Track 3 structural-friction line-level rules (Z[168..179])
    decl_veto = [
        ("my ", -3.0), ("let ", -3.0), ("var ", -3.0),
        ("local ", -3.0), ("const ", -3.0), ("for", -5.0),
    ]
    # Non-monotonic / multi-step / embedded cursor mutations vs unit +1 steps
    cursor_mut_pats = (
        "pos = last ", "cursor -=", "cursor++", "++cursor", "$cursor = ",
        "idx -= ", "idx++", "++idx", "pos -= ", "stop -= 1", "tup_idx_inc",
    )
    cursor_terms = [(pat, 2.0) for pat in cursor_mut_pats] + decl_veto
    # Also fire when a named cursor/idx/pos jumps by += 2 or += 4 on the same line
    jump_step_terms = [
        ("cursor +=", 1.0), ("$cursor +=", 1.0), ("idx += ", 1.0), ("pos += ", 1.0),
        ("i += 2", 2.0), ("+= 2", 1.0), ("+= 4", 1.0),
    ] + decl_veto
    generic_cur_pats = (
        "i = i + 1", "j = j + 1", "idx = idx +", "pos = pos +",
        "cursor +=", "$cursor +=", "idx += ", "pos += ", "i += 1",
    )
    generic_cur_terms = [(pat, 2.0) for pat in generic_cur_pats] + decl_veto
    loop_hdr_pats = (
        "for", "while", "do while", "do ", "bare_do",
        "until ", ".each ", ".each{", "_each ",
    )
    cond_loop_pats = ("while", "do while", "until ", "bare_do")
    branch_hdr_pats = (
        "if", " if ", "else", "elif", "else if", "elsif ",
        "select case", "case ", "case(", "when ", "unless ",
    )
    sig_pos_terms = [
        ("def ", 2.0), ("function ", 2.0), ("func ", 2.0),
        ("fn ", 2.0), ("sub ", 2.0), ("subroutine ", 2.0),
        ("void ", 1.0), ("bool ", 1.0), ("int ", 1.0), ("func_call", 1.0),
    ]
    sig_neg_pats = (
        "__init__", "constructor", "sub new", "func New",
        *_K13_TEST_DECLS, _K7_TEST_METHOD,
        "if", "elif", "else if", "while", "for", "switch", "match",
        "return", "call ", "throw", "raise", "=>",
        "let ", "var ", "const ", "local ", "depth_ge12",
    )
    sig_cand_terms = sig_pos_terms + [(pat, -5.0) for pat in sig_neg_pats]
    bool_flag_pats = (
        "bool ", ": bool", ": boolean", "= true",
        "= false", "logical", "$skip", "$omit",
    )
    out_ref_pats = (
        "&mut ", "out ", "ref ", r"\%", r"\@", "out_",
        "intent(out)", "intent(inou", "ByRef ", "$warnings",
        "ptr_lead_id", "ptr_trail_id", "ref_lead_id", "ref_trail_id",
    )
    out_ref_terms = [(pat, 1.0) for pat in out_ref_pats] + [("const ", -1.0)]
    skel_imp_terms = [
        ("code_chars", 0.05), ("semi_or_eq", 0.75), ("func_call", 0.75),
        ("indent_cols", 0.30), ("yield ", 1.5), ("return", -1.2),
        ("throw", -1.5), ("raise", -1.5), ("def ", -3.0),
        ("function ", -3.0), ("func ", -3.0), ("sub ", -3.0), ("subroutine ", -3.0),
        ("const ", -2.5), ("let ", -2.5), ("var ", -2.5), ("for", -2.5),
    ] + module_headers
    mut_stmt_pats = (
        " += ", " -= ", " << ", "push ", ".append(", "{end+1}",
        "= .true.", "= true", "break", "continue", "cycle", "next ",
        "++", "^=", "|=",
    )
    deep_reassign_terms = [(" = ", 1.0), ("indent_cols", 0.20)] + decl_veto
    branch_close_terms = [
        ("close_braces", 1.0), ("open_braces", -2.0),
        ("end if", 1.0), ("end select", 1.0),
        ("continue", 1.0), ("break", 1.0), ("return", 1.0),
    ]
    sig_open_paren_terms = sig_cand_terms + [("code_lparen", 10.0), ("code_rparen", -10.0)]

    line_rules.extend([
        (cursor_terms, -1.0, Z_CURSOR_MUT, 1.0),
        (cursor_terms, -2.0, Z_CURSOR_MUT, -1.0),
        (jump_step_terms, -1.5, Z_CURSOR_MUT, 2.0),
        (jump_step_terms, -2.0, Z_CURSOR_MUT, -2.0),
        (generic_cur_terms, -1.0, Z_CURSOR_GENERIC, 1.0),
        (generic_cur_terms, -2.0, Z_CURSOR_GENERIC, -1.0),
        ([("indent_cols", 2.0)], -5.0, Z_CURSOR_INDENT, 1.0),
        ([("indent_cols", 2.0)], -6.0, Z_CURSOR_INDENT, -1.0),
        ([(pat, 1.0) for pat in loop_hdr_pats], 0.0, Z_LOOP_HDR, 1.0),
        ([(pat, 1.0) for pat in loop_hdr_pats], -1.0, Z_LOOP_HDR, -1.0),
        ([(pat, 1.0) for pat in cond_loop_pats], 0.0, Z_COND_LOOP_HDR, 1.0),
        ([(pat, 1.0) for pat in cond_loop_pats], -1.0, Z_COND_LOOP_HDR, -1.0),
        ([(pat, 1.0) for pat in branch_hdr_pats], -0.5, Z_COND_BRANCH, 1.0),
        ([(pat, 1.0) for pat in branch_hdr_pats], -1.5, Z_COND_BRANCH, -1.0),
        (branch_close_terms, -0.5, Z_BRANCH_CLOSE, 1.0),
        (branch_close_terms, -1.5, Z_BRANCH_CLOSE, -1.0),
        (sig_cand_terms, -1.25, Z_SIG_CAND, 2.0),
        (sig_cand_terms, -1.75, Z_SIG_CAND, -2.0),
        (sig_open_paren_terms, -11.25, Z_SIG_OPEN_PAREN, 2.0),
        (sig_open_paren_terms, -11.75, Z_SIG_OPEN_PAREN, -2.0),
        ([("code_comma", 1.0)], 0.0, Z_SPAN_COMMA, 1.0),
        ([(pat, 1.0) for pat in bool_flag_pats], 0.0, Z_SPAN_BOOL, 1.0),
        (out_ref_terms, 0.0, Z_SPAN_OUTREF, 1.0),
        (skel_imp_terms, -1.6, Z_SKEL_IMP, 2.0),
        (skel_imp_terms, -2.1, Z_SKEL_IMP, -2.0),
        ([(pat, 1.0) for pat in mut_stmt_pats], 0.0, Z_MUT_STMT, 1.0),
        ([(pat, 1.0) for pat in mut_stmt_pats], -1.0, Z_MUT_STMT, -1.0),
        (deep_reassign_terms, -1.3, Z_MUT_STMT, 2.0),
        (deep_reassign_terms, -1.8, Z_MUT_STMT, -2.0),
    ])

    free_narrow = [slot for slot in range(n_conv3) if not np.any(conv3_w[:, :, slot])]
    free_medium = [slot for slot in range(n_conv7) if not np.any(conv7_w[:, :, slot])]
    free_wide = [slot for slot in range(n_conv13) if not np.any(conv13_w[:, :, slot])]
    scaled_cache: dict[tuple[int, float], int] = {}
    conv_banks = (
        (off_c7, off_c3, conv3_w, conv3_s, conv3_b, free_narrow),
        (off_c13, off_c7, conv7_w, conv7_s, conv7_b, free_medium),
        (10**9, off_c13, conv13_w, conv13_s, conv13_b, free_wide),
    )

    def rescale_conv_in_place(base_idx: int, mult: float) -> None:
        for upper_off, bank_off, _, s_arr, b_arr, _ in conv_banks:
            if base_idx < upper_off:
                s_arr[base_idx - bank_off] *= mult
                b_arr[base_idx - bank_off] *= mult
                return

    def clone_scaled_feat(base_idx: int, mult: float) -> int:
        if abs(mult - 1.0) < 1e-7:
            return base_idx
        cache_key = (base_idx, round(mult, 6))
        if cache_key in scaled_cache:
            return scaled_cache[cache_key]
        if base_idx < off_c3:
            dst = free_narrow.pop(0)
            hc_col = int(np.argmax(csum_w[:, base_idx]))
            conv3_w[1, hc_col, dst] = 1
            conv3_s[dst] = csum_s[base_idx] * mult
            out_idx = off_c3 + dst
        else:
            out_idx = -1
            for upper_off, bank_off, w_arr, s_arr, b_arr, pool in conv_banks:
                if base_idx >= upper_off:
                    continue
                src = base_idx - bank_off
                if pool:
                    dst = pool.pop(0)
                    w_arr[:, :, dst] = w_arr[:, :, src]
                    s_arr[dst], b_arr[dst] = s_arr[src] * mult, b_arr[src] * mult
                    out_idx = bank_off + dst
                else:
                    dst = free_wide.pop(0)
                    conv13_w[3:10, :, dst] = conv7_w[:, :, src]
                    conv13_s[dst], conv13_b[dst] = s_arr[src] * mult, b_arr[src] * mult
                    out_idx = off_c13 + dst
                break
        scaled_cache[cache_key] = out_idx
        return out_idx

    num_u = num_raw_u + N_ML_GATE + N_PAREN_SUF + N_PREV_LINE
    n_lm1 = 224
    lm1_w = np.zeros((num_u, n_lm1), dtype=np.int8)
    lm1_s = np.ones((n_lm1,), dtype=np.float32)
    lm1_b = np.zeros((n_lm1,), dtype=np.float32)
    lm2_w = np.zeros((n_lm1, N_LMLP2), dtype=np.int8)
    lm2_b = np.zeros((N_LMLP2,), dtype=np.float32)

    ml_protected = {
        feat["triple_dq"], feat["triple_sq"],
        feat["ml_bcomm_open"], feat["ml_bcomm_close"],
        feat["code_lparen"], feat["code_rparen"],
        feat["= @_;"], feat["intent("],
    }
    ref_counts: Counter[int] = Counter()
    for terms, _, _, _ in line_rules:
        for name, _ in terms:
            ref_counts[feat[name]] += 1
    rescaled_once: set[int] = set()

    for unit, (terms, bias, dst_z, out_scale) in enumerate(line_rules):
        merged: dict[int, float] = {}
        for name, weight in terms:
            idx = feat[name]
            merged[idx] = merged.get(idx, 0.0) + weight

        abs_out = abs(out_scale)
        mag_counts = Counter(round(abs(wt), 6) for wt in merged.values())
        base_mag = max(mag_counts.items(), key=lambda pair: (pair[1], pair[0]))[0]
        lm1_s[unit] = base_mag * abs_out
        lm1_b[unit] = bias * abs_out
        lm2_w[unit, dst_z] = 1 if out_scale >= 0 else -1

        for base_idx, wt_val in merged.items():
            sign_in = 1 if wt_val >= 0 else -1
            mult = abs(wt_val) / base_mag
            can_rescale = (
                ref_counts[base_idx] == 1
                and base_idx >= off_c3
                and base_idx not in rescaled_once
                and base_idx not in ml_protected
            )
            if abs(mult - 1.0) < 1e-6:
                src_col = base_idx
            elif can_rescale:
                rescaled_once.add(base_idx)
                rescale_conv_in_place(base_idx, mult)
                src_col = base_idx
            else:
                src_col = clone_scaled_feat(base_idx, mult)
            lm1_w[src_col, unit] = sign_in

    # Wire cross-line docstring/comment sawtooth -> Z_IN_DOC and Dyck suffixes -> Z_IN_PAREN
    dq_trits = np.array([1] + [-1, 1] * (N_DQ_SAW // 2 - 1) + [-1], dtype=np.int8)
    sq_trits = np.array([1] + [-1, 1] * (N_SQ_SAW // 2 - 1) + [-1], dtype=np.int8)
    off_dq = num_raw_u
    off_sq = off_dq + N_DQ_SAW
    off_bc = off_sq + N_SQ_SAW
    off_ps = num_raw_u + N_ML_GATE

    u_doc_hi, u_doc_lo = len(line_rules), len(line_rules) + 1
    for u_doc in (u_doc_hi, u_doc_lo):
        lm1_w[off_dq : off_dq + N_DQ_SAW, u_doc] = dq_trits
        lm1_w[off_sq : off_sq + N_SQ_SAW, u_doc] = sq_trits
        lm1_w[[off_bc, off_bc + 1], u_doc] = [1, -1]
        lm1_w[[feat["lead_tdq"], feat["lead_tsq"]], u_doc] = 1
    lm1_b[[u_doc_hi, u_doc_lo]] = [0.0, -1.0]
    lm2_w[[u_doc_hi, u_doc_lo], Z_IN_DOC] = [1, -1]

    u_par_hi, u_par_lo = len(line_rules) + 2, len(line_rules) + 3
    par_signs = np.tile(np.array([1, -1], dtype=np.int8), N_PAREN_STEPS)
    for u_par in (u_par_hi, u_par_lo):
        lm1_w[off_ps : off_ps + N_PAREN_SUF, u_par] = par_signs
    lm1_b[[u_par_hi, u_par_lo]] = [0.0, -1.0]
    lm2_w[[u_par_hi, u_par_lo], Z_IN_PAREN] = [1, -1]

    # Indented function/method signatures also indicate class/module-indented containers
    u_ind_fn_hi, u_ind_fn_lo = len(line_rules) + 4, len(line_rules) + 5
    ind_fn_cols = [feat["def "], feat["function "], feat["pub "], feat["public "], feat["indent_cols"]]
    for u_ifn in (u_ind_fn_hi, u_ind_fn_lo):
        lm1_w[ind_fn_cols, u_ifn] = 1
    lm1_s[[u_ind_fn_hi, u_ind_fn_lo]] = 2.0
    lm1_b[[u_ind_fn_hi, u_ind_fn_lo]] = [-3.5, -4.5]
    lm2_w[[u_ind_fn_hi, u_ind_fn_lo], Z_CONTAINER_HDR] = [1, -1]

    # Previous-line registers: comment-block continuation and `return ..., nil` under `err != nil`
    off_prev = off_ps + N_PAREN_SUF
    u_prev_comm = len(line_rules) + 6
    lm1_w[[off_prev, off_prev + 1], u_prev_comm] = [1, -1]
    lm2_w[u_prev_comm, Z_PREV_COMMENT] = 1

    u_nil_hi, u_nil_lo = len(line_rules) + 7, len(line_rules) + 8
    nil_cols = [off_prev + 2, off_prev + 3, feat["return"], feat["nil_eol"]]
    for u_nil in (u_nil_hi, u_nil_lo):
        lm1_w[nil_cols, u_nil] = [1, -1, 1, 1]
    lm1_s[[u_nil_hi, u_nil_lo]] = 2.0
    lm1_b[[u_nil_hi, u_nil_lo]] = [-5.0, -6.0]
    lm2_w[[u_nil_hi, u_nil_lo], Z_NIL_SWALLOW] = [1, -1]

    # Clamped [0, 1] declaration-based signature continuation (`= @_;` OR `intent(`)
    u_sc_hi, u_sc_lo = len(line_rules) + 9, len(line_rules) + 10
    for u_sc in (u_sc_hi, u_sc_lo):
        lm1_w[[feat["= @_;"], feat["intent("]], u_sc] = 1
    lm1_b[[u_sc_hi, u_sc_lo]] = [0.0, -1.0]
    lm2_w[[u_sc_hi, u_sc_lo], Z_SIG_CONT] = [1, -1]

    # Emit helper scaled Z[87..112] registers for exact {-1, 0, +1} cmlp1_w & doc1_w
    active_units = list(range(len(line_rules) + 11))
    next_unit = len(line_rules) + 11
    next_z = 87
    z_helpers: dict[str, int] = {}
    helper_specs = (
        ("has_code_08", Z_HAS_CODE, 0.8),
        ("d12_22", Z_DEPTH12, 2.2),
        ("d10_08", Z_DEPTH10, 0.8),
        ("open_24", Z_OPEN_NET, 2.4),
        ("pyr_22", Z_DEEP_PYRAMID, 2.2),
        ("pyr_cls_22", Z_PYR_CLASS, 2.2),
        ("long_id_r4", Z_LONG_IDENT, 0.75 / 0.85),
        ("stdlib_r4", Z_STDLIB_IDIOM, 2.5 / 0.85),
        ("one_r5", Z_ONE_CHAR_ID, 0.55 / 0.4),
        ("words_r5", Z_WORD_COUNT, 0.8 / 0.4),
        ("chain_12", Z_CHAINED_CALLS, 1.2),
        ("tern_125", Z_NESTED_TERNARY, 1.25),
        ("o96_11", Z_OVER_96, 1.1),
        ("o80_025", Z_OVER_80, 0.25),
        ("stdlib_09", Z_STDLIB_IDIOM, 0.9),
        ("modhdr_30", Z_MODULE_HEADER, 3.0),
        ("sw_dup", Z_SWITCH_BLOCK, 1.0),
        ("imp_075", Z_IMPERATIVE_LINE, 0.75),
        ("odd_18", Z_ODD_INDENT, 1.8),
        ("gc_09", Z_GUARD_CLAUSE, 0.9),
        ("ci_06", Z_CLEAN_IDIOM, 0.6),
        ("m42_025", Z_MOD4_EQ2, 0.25),
        ("wc_020", Z_WORD_COUNT, 0.20),
        ("m42_doc", Z_MOD4_EQ2, 0.3 / 1.4),
        ("in_doc_8", Z_IN_DOC, 8.0),
        ("in_par_8", Z_IN_PAREN, 8.0),
    )
    for key_name, src_z, mult in helper_specs:
        dst_z = next_z
        next_z += 1
        for src_u in active_units:
            if lm2_w[src_u, src_z] != 0:
                lm1_w[:, next_unit] = lm1_w[:, src_u]
                lm1_s[next_unit] = lm1_s[src_u] * mult
                lm1_b[next_unit] = lm1_b[src_u] * mult
                lm2_w[next_unit, dst_z] = lm2_w[src_u, src_z]
                next_unit += 1
        z_helpers[key_name] = dst_z

    trit_mats["conv3_w"], vecs_1d["conv3_s"], vecs_1d["conv3_b"] = conv3_w, conv3_s, conv3_b
    trit_mats["conv7_w"], vecs_1d["conv7_s"], vecs_1d["conv7_b"] = conv7_w, conv7_s, conv7_b
    trit_mats["conv13_w"], vecs_1d["conv13_s"], vecs_1d["conv13_b"] = conv13_w, conv13_s, conv13_b
    trit_mats["sketch_w"], vecs_1d["sketch_s"], vecs_1d["sketch_b"] = _sketch_bank()
    trit_mats["lmlp1_w"] = lm1_w[:, :next_unit]
    vecs_1d["lmlp1_s"], vecs_1d["lmlp1_b"] = lm1_s[:next_unit], lm1_b[:next_unit]
    trit_mats["lmlp2_w"], vecs_1d["lmlp2_b"] = lm2_w[:next_unit, :], lm2_b
    return z_helpers


def _build_block2_and_heads(
    trit_mats: dict[str, np.ndarray],
    vecs_1d: dict[str, np.ndarray],
    zh: dict[str, int],
) -> None:
    """Compile {-1, 0, +1} sink-attention heads, [0, 1] issue clamps, and System One readout."""
    center = MAX_REL_DIST
    num_rel = 2 * center + 1
    a1_wq = np.zeros((N_Z, 192), dtype=np.int8)
    a1_wq_s = np.ones((192,), dtype=np.float32)
    a1_wk = np.zeros((N_Z, 192), dtype=np.int8)
    a1_wk_s = np.ones((192,), dtype=np.float32)
    a1_wv = np.zeros((N_Z, 144), dtype=np.int8)
    a1_wv_s = np.ones((144,), dtype=np.float32)
    a1_wo = np.zeros((144, N_Z), dtype=np.int8)
    a1_br = np.full((12, num_rel), -1e4, dtype=np.float32)
    a1_bs = np.array(
        [
            4.5, 1.2, 1.5, 2.5, 9.05, 11.1, math.log(16.0), -12.0,
            math.log(9.0), 6.5, 8.8, 10.8,
        ],
        dtype=np.float32,
    )

    # Head 0: uniform-diagonal Paley Hadamard projection for comment-to-code overlap
    h0_trits = _paley_uniform_proj()
    sk_trits, sk_scales = _ternary_sketch_proj()
    comm_slice = slice(Z_COMM_SKETCH, Z_COMM_SKETCH + SKETCH_DIM)
    code_slice = slice(Z_CODE_SKETCH, Z_CODE_SKETCH + SKETCH_DIM)

    a1_wq[comm_slice, H0_QK] = h0_trits
    a1_wk[code_slice, H0_QK] = h0_trits
    a1_wq_s[H0_QK] = a1_wk_s[H0_QK] = float(np.sqrt(68.0 / 16.0))
    a1_br[0, [center + 1, center + 2]] = [0.0, -1.2]
    a1_wv[Z_HAS_CODE, 0] = a1_wo[0, Z_RESTATES_CODE] = 1

    # Head 1: try/catch/except opener matching real vs. empty/fallback handler on L..L+6,
    # plus disjoint QK dim matching Swift `= try?` to `return nil` / `return false`
    h1_qk, swift_qk, swift_slot = [16, 17], 18, 14
    a1_wq[Z_TRY_CATCH, h1_qk] = 1
    a1_wk[[Z_REAL_HANDLER, Z_EMPTY_HANDLER], h1_qk] = 1
    a1_wq_s[h1_qk] = a1_wk_s[h1_qk] = [float(np.sqrt(32.0)), float(np.sqrt(16.0))]
    a1_wq[Z_SWIFT_TRYQ, swift_qk] = a1_wk[Z_SWIFT_NIL, swift_qk] = 1
    a1_wq_s[swift_qk] = float(np.sqrt(32.0)) / 8.0
    a1_wk_s[swift_qk] = float(np.sqrt(32.0))
    a1_wv[Z_SWIFT_NIL, swift_slot] = a1_wo[swift_slot, Z_SWIFT_NIL] = 1
    a1_wv_s[swift_slot] = 1.5
    # Handler lines only outbid the sink when a try/catch opener queries them; a line with no
    # opener (zero query) attends to the sink instead of averaging stray `{}` literals
    a1_br[1, center : center + 7] = -3.5

    # Head 2 (L-8..L), Head 3 (L-9..L+9), Head 6 (file-wide with sink prior), Head 7 (L-3..L+3)
    a1_br[2, center - 8 : center + 1] = 0.0
    a1_br[3, center - 9 : center + 10] = 0.0
    a1_br[6, :] = 0.0
    a1_br[7, center - 3 : center + 4] = 0.0

    h2_srcs = [
        Z_DEPTH10, Z_DEPTH12, Z_BRANCH_OPEN, Z_EARLY_EXIT,
        Z_GUARD_CLAUSE, Z_CLEAN_IDIOM, Z_DEEP_PYRAMID, Z_PYR_CLASS,
    ]
    h2_slots = list(range(24, 32))
    a1_wv[h2_srcs, h2_slots] = 1
    a1_wv_s[slice(24, 32)] = [0.8, 1.5, 1.0, 1.0, 0.8, 0.8, 1.0, 1.0]
    a1_wo[[24, 25], Z_ENCLOSING_DEPTH] = 1
    a1_wo[26, Z_ENCLOSING_BRANCHES] = 1
    a1_wo[[27, 28, 29], Z_ENCLOSING_GUARDS] = 1
    a1_wo[30, Z_ENCLOSING_DEEP16] = 1
    a1_wo[31, Z_ENCLOSING_DEEP24] = 1

    spec_slots = [39, 40]
    a1_wv[[Z_HARDCODED_CASE, Z_SWITCH_BLOCK], spec_slots] = 1
    a1_wv_s[spec_slots] = [1.53, 2.5 * 1.53]
    a1_wo[spec_slots, Z_LOCAL_SPECIAL_CASES] = [1, -1]
    a1_wv[[Z_CHAINED_CALLS, Z_NESTED_TERNARY, Z_CEREMONY], [84, 84, 85]] = 1
    a1_wo[[84, 85], [Z_LOCAL_CHAINING, Z_LOCAL_CEREMONY]] = 1

    direct_copies = (
        ([Z_REAL_HANDLER, Z_EMPTY_HANDLER], [12, 13], [Z_NEAR_REAL_HANDLER, Z_NEAR_EMPTY_HANDLER]),
        ([Z_IS_STMT, Z_PARAGRAPH_BREAK, Z_ODD_INDENT], [36, 37, 38],
         [Z_LOCAL_STMT_DENSITY, Z_LOCAL_BREAK_DENSITY, Z_LOCAL_ODD_INDENT]),
        ([Z_ONE_CHAR_ID, Z_WORD_COUNT, Z_MOD4_EQ0, Z_MOD4_EQ2, Z_ODD_INDENT], list(range(72, 77)),
         [Z_FILE_ONE_CHAR, Z_FILE_WORDS, Z_FILE_MOD4_0, Z_FILE_MOD4_2, Z_FILE_ODD_IND]),
        ([Z_SHORT_GOLF, Z_MAGIC_NUM], [86, 87], [Z_LOCAL_SHORT_GOLF, Z_LOCAL_MAGIC_NUM]),
    )
    for srcs, slots, dsts in direct_copies:
        a1_wv[srcs, slots] = 1
        a1_wo[slots, dsts] = 1

    # Heads 4 & 5: Gram-fitted ternary clone match (8 <= |d| < 32) minus adjacent table match
    abs_offset = np.abs(np.arange(-center, center + 1))
    a1_wq[code_slice, H4_QK] = sk_trits
    a1_wk[code_slice, H4_QK] = sk_trits
    a1_wq[code_slice, H5_QK] = sk_trits
    a1_wk[code_slice, H5_QK] = sk_trits
    a1_wq_s[H45_QK] = a1_wk_s[H45_QK] = np.tile(sk_scales * 8.0, 2)
    a1_br[4, (abs_offset >= 8) & (abs_offset < center)] = 0.0
    a1_br[5, (abs_offset >= 1) & (abs_offset <= 3)] = 0.0
    a1_wv[Z_IMPERATIVE_LINE, [48, 60]] = 1
    a1_wv_s[60] = 0.80
    a1_wo[[48, 60], Z_DISTANT_CLONE] = [1, -1]

    # Pre-scale single-consumer context channels via a1_wv_s for exact {-1, 0, +1} cmlp1_w
    a1_wv_s[0] *= 2.4
    a1_wv_s[[12, 13]] *= [4.0 / 1.8, 1.4 / 1.8]
    a1_wv_s[26] *= 5.2 / 0.5
    a1_wv_s[[27, 28, 29]] *= 3.0 / 0.5
    a1_wv_s[[30, 31]] *= 1.6 / 0.5
    a1_wv_s[[36, 37]] *= [14.2, 18.0]
    a1_wv_s[spec_slots] *= 8.5 / 1.5
    a1_wv_s[[72, 73]] *= [1.5 / 0.4, 2.0 / 0.4]
    a1_wv_s[75] *= 1.5
    a1_wv_s[85] *= 0.65 / 0.85
    a1_wv_s[86] *= 1.8 / 0.6
    a1_wv_s[87] *= 2.2 / 0.9

    extra_f_slots = [77, 78, 79, 80]
    a1_wv[[Z_MOD4_EQ2, Z_ODD_INDENT, Z_ODD_INDENT, Z_CONTAINER_HDR], extra_f_slots] = 1
    a1_wv_s[extra_f_slots] = [1.0, 1.0, 7.5, 48.0]
    a1_wo[extra_f_slots, [Z_FMOD42_DUP, Z_FODD_DUP, Z_FODD_SCALED, Z_FILE_CONTAINERS]] = 1

    fence_slot, kt_slot, math_slot, test_slot = 41, 32, 46, 88
    a1_wv[Z_COMM_FENCE, fence_slot] = 1
    a1_wv_s[fence_slot] = 40.0
    a1_wo[fence_slot, Z_NEAR_COMM_FENCE] = 1
    a1_wv[Z_KT_RUNCATCH, kt_slot] = a1_wo[kt_slot, Z_KT_RUNCATCH] = 1
    a1_wv_s[kt_slot] = 4.5
    a1_wv[Z_MATH_SIG, math_slot] = a1_wo[math_slot, Z_MATH_SIG] = 1
    a1_wv_s[math_slot] = 6.0
    a1_wv[Z_STDLIB_IDIOM, test_slot] = a1_wo[test_slot, zh["stdlib_r4"]] = 1
    a1_wv_s[test_slot] = 1.5

    # Track 3 attention heads: scope entanglement (H3), enclosing loops (H8),
    # multiline signature span (H9), and alpha-invariant skeleton clones (H10, H11)
    a1_wv[[Z_MUT_STMT, Z_SIG_CAND], [42, 43]] = 1
    a1_wv_s[[42, 43]] = [7.8, 22.0]
    a1_wo[[42, 43], Z_LOCAL_MUT_DENS] = [1, -1]
    a1_wv[[Z_COND_BRANCH, Z_LOOP_HDR], 44] = 1
    a1_wv[Z_SIG_CAND, 45] = 1
    a1_wv_s[[44, 45]] = [6.2, 22.0]
    a1_wo[[44, 45], Z_LOCAL_CTRL_DENS] = [1, -1]

    a1_br[8, center - 31 : center] = 0.0
    h8_slots = [96, 97, 98, 99]
    a1_wv[[Z_LOOP_HDR, Z_COND_LOOP_HDR, Z_COND_BRANCH, Z_BRANCH_CLOSE], h8_slots] = 1
    a1_wv_s[h8_slots] = 40.0
    a1_wo[[96, 97, 98], [Z_ENC_LOOP, Z_ENC_COND_LOOP, Z_ENC_BRANCH]] = 1
    a1_wo[99, Z_ENC_BRANCH] = -1

    a1_br[9, center + 1 : center + 9] = -12.0
    h9_qk, h9_decl = 9 * 16, 9 * 16 + 1
    h9_slots = [108, 109, 110]
    a1_wq[Z_SIG_OPEN_PAREN, h9_qk] = 1
    a1_wk[Z_IN_PAREN, h9_qk] = 1
    a1_wq[Z_SIG_CAND, h9_decl] = 1
    a1_wk[Z_SIG_CONT, h9_decl] = 1
    a1_wq_s[[h9_qk, h9_decl]] = a1_wk_s[[h9_qk, h9_decl]] = float(np.sqrt(60.0))
    span_regs = [Z_SPAN_COMMA, Z_SPAN_BOOL, Z_SPAN_OUTREF]
    a1_wv[span_regs, h9_slots] = 1
    a1_wv_s[h9_slots] = float(np.exp(3.5))
    a1_wo[h9_slots, span_regs] = 1

    # Joint 4-dim lexical stem + 12-dim Paley operator-skeleton QK kernel
    skel_proj = _paley_uniform_proj(seed=7)
    skel_slice = slice(Z_SKEL_SKETCH, Z_SKEL_SKETCH + SKETCH_DIM)
    for qk_base in (160, 176):
        code_cols = slice(qk_base, qk_base + 4)
        skel_cols = slice(qk_base + 4, qk_base + 16)
        a1_wq[code_slice, code_cols] = sk_trits[:, :4]
        a1_wk[code_slice, code_cols] = sk_trits[:, :4]
        a1_wq[skel_slice, skel_cols] = skel_proj[:, :12]
        a1_wk[skel_slice, skel_cols] = skel_proj[:, :12]
    joint_scales = np.concatenate([sk_scales[:4] * 7.5, np.full((12,), 2.65, dtype=np.float32)])
    a1_wq_s[H1011_QK] = a1_wk_s[H1011_QK] = np.tile(joint_scales, 2)
    a1_br[10, (abs_offset >= 4) & (abs_offset < center)] = 0.0
    a1_br[11, (abs_offset >= 1) & (abs_offset <= 3)] = 0.0
    skel_v_slots = [120, 132]
    a1_wv[Z_SKEL_IMP, skel_v_slots] = 1
    a1_wv_s[skel_v_slots] = [1.0, 0.60]
    a1_wo[skel_v_slots, Z_SKEL_DISTANT] = [1, -1]

    trit_mats["attn1_wq"], vecs_1d["attn1_wq_s"] = a1_wq, a1_wq_s
    trit_mats["attn1_wk"], vecs_1d["attn1_wk_s"] = a1_wk, a1_wk_s
    trit_mats["attn1_wv"], vecs_1d["attn1_wv_s"] = a1_wv, a1_wv_s
    trit_mats["attn1_wo"] = a1_wo
    vecs_1d["attn1_brel"], vecs_1d["attn1_bsink"] = a1_br, a1_bs

    # Context MLP 1: paired {-1, 0, +1} ReLU clamps writing [0, 1] issue flags
    wide_dsts = [Z_USES_WIDE_INDENT, Z_WIDE_DUP_A, Z_WIDE_DUP_B]
    doc_veto = (zh["in_doc_8"], -1)
    par_veto = (zh["in_par_8"], -1)

    ctx1_rules: list[tuple[list[tuple[int, int]], float, list[int], float, float]] = [
        ([(zh["has_code_08"], 1), (zh["d12_22"], 1), (zh["d10_08"], 1), (Z_ENCLOSING_BRANCHES, 1),
          (Z_ELSE_BRANCH, 1), (Z_ENCLOSING_GUARDS, -1), (zh["open_24"], -1), doc_veto, par_veto],
         -3.05, [Z_RAW_NESTING_2SP], 0.5, 2.5),
        ([(zh["has_code_08"], 1), (zh["pyr_22"], 1), (Z_ENCLOSING_DEEP16, 1),
          (Z_ENCLOSING_BRANCHES, 1), (Z_ELSE_BRANCH, 1),
          (Z_ENCLOSING_GUARDS, -1), (zh["open_24"], -1), doc_veto, par_veto],
         -3.25, [Z_RAW_NEST_TOP], 0.5, 2.5),
        ([(zh["has_code_08"], 1), (zh["pyr_cls_22"], 1), (Z_ENCLOSING_DEEP24, 1),
          (Z_ENCLOSING_BRANCHES, 1), (Z_ELSE_BRANCH, 1),
          (Z_ENCLOSING_GUARDS, -1), (zh["open_24"], -1), doc_veto, par_veto],
         -3.25, [Z_ISS_NESTING], 0.5, 2.5),
        ([(Z_FILE_CONTAINERS, 1)], -0.4, [Z_HAS_CONTAINER], 4.0, 2.0),

        ([(Z_FILE_MOD4_0, 1), (Z_FILE_MOD4_2, -1), (Z_FMOD42_DUP, -1),
          (Z_FILE_ODD_IND, -1), (Z_FODD_DUP, -1)],
         -0.25, wide_dsts, 20.0, 2.0),
        ([(Z_FILE_ODD_IND, 1), (Z_FILE_MOD4_2, -1)], -2.5, wide_dsts, 20.0, 2.0),

        ([(Z_CEREMONY, 1), (zh["long_id_r4"], 1), (Z_LOCAL_CEREMONY, 1),
          (zh["stdlib_r4"], -1), doc_veto],
         -1.05, [Z_ISS_NAMING], 0.85, 2.2),
        ([(zh["one_r5"], 1), (Z_FILE_ONE_CHAR, 1), (Z_FILE_WORDS, -1),
          (zh["words_r5"], -1), (Z_CRYPTIC_GATE, 1), (Z_MATH_SIG, -1), doc_veto],
         -2.15, [Z_RAW_CRYPTIC], 0.4, 2.2),
        ([(Z_PURE_COMMENT, 1), (Z_SHORT_COMMENT, 1), (Z_RESTATES_CODE, 1),
          (Z_PREV_COMMENT, -1), doc_veto],
         -3.35, [Z_ISS_COMMENT], 1.0, 2.5),
        ([(Z_NARRATOR_TAG, 1), doc_veto], -0.45, [Z_ISS_COMMENT], 1.5, 2.2),
        ([(Z_TRY_CATCH, 1), (Z_NEAR_EMPTY_HANDLER, 1), (Z_NEAR_REAL_HANDLER, -1), doc_veto],
         -2.1, [Z_ISS_ERROR], 1.8, 2.5),
        ([(Z_FALLBACK_SWALLOW, 1), doc_veto], -0.35, [Z_ISS_ERROR], 1.4, 2.2),
        ([(Z_NIL_SWALLOW, 1), doc_veto], -0.5, [Z_ISS_ERROR], 1.0, 4.0),
        ([(Z_KT_GETNULL, 1), (Z_KT_RUNCATCH, 1), doc_veto], -8.15, [Z_ISS_ERROR], 1.0, 6.0),
        ([(Z_SWIFT_TRYQ, 1), (Z_SWIFT_NIL, 1), doc_veto], -8.5, [Z_ISS_ERROR], 1.0, 4.0),

        ([(zh["chain_12"], 1), (zh["tern_125"], 1), (zh["o96_11"], 1), (zh["o80_025"], 1),
          (zh["stdlib_09"], -1), (zh["modhdr_30"], -1), doc_veto],
         -1.0, [Z_ISS_DENSE], 1.0, 2.2),
        ([(Z_HARDCODED_CASE, 1), (Z_LOCAL_SPECIAL_CASES, 1),
          (Z_SWITCH_BLOCK, -1), (zh["sw_dup"], -1), doc_veto],
         -2.75, [Z_ISS_SPECIAL], 1.5, 2.5),
        ([(zh["imp_075"], 1), (Z_DISTANT_CLONE, 1), (Z_IS_INDENTED, 1),
          (Z_IS_STMT, 1), doc_veto, par_veto],
         -6.45, [Z_RAW_CLONE], 2.0, 2.5),
        ([(Z_IS_STMT, 1), (Z_LOCAL_STMT_DENSITY, 1), (Z_LOCAL_BREAK_DENSITY, -1),
          doc_veto, par_veto],
         -9.15, [Z_ISS_LAYOUT], 1.0, 3.0),
        ([(Z_IS_STMT, 1), (zh["odd_18"], 1), (Z_FODD_SCALED, -1), doc_veto, par_veto],
         -2.0, [Z_ISS_LAYOUT], 1.0, 2.5),
        ([(zh["gc_09"], 1), (zh["ci_06"], 1), (zh["m42_025"], 1), (zh["wc_020"], 1), doc_veto],
         -0.25, [Z_CLEAN_BONUS], 1.0, 1.2),
        ([(Z_SHORT_GOLF, 1), (Z_LOCAL_SHORT_GOLF, 1), (Z_MATH_SIG, -1), doc_veto],
         -1.35, [Z_ISS_NAMING], 0.6, 2.5),
        ([(Z_MAGIC_NUM, 1), (Z_LOCAL_MAGIC_NUM, 1), doc_veto], -0.78, [Z_ISS_NAMING], 0.9, 2.5),

        ([(Z_SIG_OPEN, 1), (Z_STATIC_TYPE, 1), (Z_SIG_VETO, -1), (Z_SEMI_ONLY, -1), doc_veto],
         -1.5, [Z_TYPED_INTERNAL_SIG], 1.0, 4.0),
        ([(Z_SIG_VETO, 1), (Z_SEMI_ONLY, -1), doc_veto],
         -0.5, [Z_BOUNDARY_SIG_LINE], 1.0, 4.0),
        ([(Z_IF_LINE, 1), (Z_TYPE_HEDGE, 1), (Z_FAIL_FAST, -1), doc_veto],
         -1.5, [Z_TYPE_HEDGE_IF], 1.0, 4.0),
        ([(Z_TWO_ENVS, 1), (Z_HAS_COALESCE, 1), (Z_ENV_VETO, -1), doc_veto],
         -1.5, [Z_ENV_CHAIN_1L], 1.0, 4.0),
        ([(Z_HAS_ENV, 1), (Z_DEV_DEFAULT, 1), (Z_ENV_VETO, -1), doc_veto],
         -1.5, [Z_ENV_DEV_1L], 1.0, 4.0),

        ([(Z_HAS_ENV, 1), (Z_HAS_ASSIGN, -1), (Z_ENV_VETO, -1), doc_veto],
         -0.5, [Z_CONT_ENV], 1.0, 4.0),
        ([(Z_DEV_DEFAULT, 1), (Z_HAS_ASSIGN, -1), (Z_FAIL_FAST, -1), doc_veto],
         -0.5, [Z_CONT_DEV_DEF], 1.0, 4.0),
        ([(Z_ELISION_RAW, 1), (Z_LINE_COMM_VETO, -1), (Z_NEAR_COMM_FENCE, -1), doc_veto],
         -0.5, [Z_ISS_COMMENT], 1.0, 4.0),
        ([(Z_PURE_COMMENT, 1), (Z_COMM_STRONG_CUE, 1),
          (Z_LINE_COMM_VETO, -1), (Z_NEAR_COMM_FENCE, -1), doc_veto],
         -1.5, [Z_COMM_CODE_LINE], 1.0, 4.0),
        ([(Z_PURE_COMMENT, 1), (Z_COMM_ASSIGN, 1), (Z_COMM_FCALL, 1),
          (Z_COMM_STRONG_CUE, -1), (Z_LINE_COMM_VETO, -1), (Z_NEAR_COMM_FENCE, -1), doc_veto],
         -2.5, [Z_COMM_CODE_LINE], 1.0, 4.0),

        # Track 3 per-clause [0, 1] clamps & skeleton-clone candidate
        ([(Z_ENC_LOOP, 1), doc_veto], -0.75, [Z_HAS_ENC_LOOP], 1.0, 4.0),
        ([(Z_ENC_COND_LOOP, 1), doc_veto], -0.75, [Z_HAS_COND_LOOP], 1.0, 4.0),
        ([(Z_ENC_BRANCH, 1), (Z_COND_BRANCH, 1), doc_veto], -0.75, [Z_HAS_ENC_BR1], 1.0, 4.0),
        ([(Z_ENC_BRANCH, 1), (Z_COND_BRANCH, 1), doc_veto], -1.75, [Z_HAS_ENC_BR2], 1.0, 4.0),
        ([(Z_SPAN_COMMA, 1), doc_veto], -4.15, [Z_ARITY_6PLUS], 1.0, 4.0),
        ([(Z_SPAN_BOOL, 1), doc_veto], -0.65, [Z_HAS_BOOL_SPAN], 1.0, 4.0),
        ([(Z_SPAN_OUTREF, 1), doc_veto], -0.65, [Z_HAS_OUTREF_SPAN], 1.0, 4.0),
        ([(Z_SKEL_IMP, 1), (Z_SKEL_DISTANT, 1), (Z_IS_INDENTED, 1),
          (Z_SWITCH_BLOCK, -1), doc_veto, par_veto],
         -2.48, [Z_RAW_SKEL_CLONE], 1.0, 4.0),
        ([(Z_LOCAL_MUT_DENS, 1), doc_veto], -0.95, [Z_HAS_MUT_DENS], 1.0, 4.0),
        ([(Z_LOCAL_CTRL_DENS, 1), doc_veto], -0.75, [Z_HAS_CTRL_DENS], 1.0, 8.0),
        ([(Z_LOCAL_CTRL_DENS, 1), doc_veto], -0.35, [Z_HAS_CTRL_MOD], 1.0, 16.0),
    ]

    n_cm1 = 2 * len(ctx1_rules)
    cm1_w = np.zeros((N_Z, n_cm1), dtype=np.int8)
    cm1_s = np.ones((n_cm1,), dtype=np.float32)
    cm1_b = np.zeros((n_cm1,), dtype=np.float32)
    cm1_2w = np.zeros((n_cm1, N_Z), dtype=np.int8)
    cm1_2b = np.zeros((N_Z,), dtype=np.float32)

    for idx, (terms, bias, dst_list, base_sc, gain) in enumerate(ctx1_rules):
        unit = 2 * idx
        for src_z, trit in terms:
            cm1_w[src_z, [unit, unit + 1]] = trit
        cm1_s[[unit, unit + 1]] = base_sc * gain
        cm1_b[[unit, unit + 1]] = [bias * gain, bias * gain - 1.0]
        for dst_z in dst_list:
            cm1_2w[[unit, unit + 1], dst_z] = [1, -1]

    trit_mats["cmlp1_w"], vecs_1d["cmlp1_s"], vecs_1d["cmlp1_b"] = cm1_w, cm1_s, cm1_b
    trit_mats["cmlp1_2_w"], vecs_1d["cmlp1_2_b"] = cm1_2w, cm1_2b

    # Pass 2 neighborhood smoothing attention (9 heads) & context MLP 2 in {-1, 0, +1}
    a2_wq = np.zeros((N_Z, 144), dtype=np.int8)
    a2_wq_s = np.ones((144,), dtype=np.float32)
    a2_wk = np.zeros((N_Z, 144), dtype=np.int8)
    a2_wk_s = np.ones((144,), dtype=np.float32)
    a2_wv = np.zeros((N_Z, 144), dtype=np.int8)
    a2_wv_s = np.ones((144,), dtype=np.float32)
    a2_wo = np.zeros((144, N_Z), dtype=np.int8)
    a2_br = np.full((9, num_rel), -1e4, dtype=np.float32)
    a2_br[0, center - 3 : center + 4] = 0.0
    a2_br[1, center - 2 : center + 3] = 0.0
    a2_br[2, center - 3 : center + 4] = 0.0
    a2_br[3, center - 24 : center + 1] = -0.15 * np.arange(24, -1, -1, dtype=np.float32)
    a2_br[4, center - 2 : center + 3] = 0.0
    a2_br[5, center - 4 : center] = 0.0
    a2_br[6, center : center + 2] = 0.0
    a2_br[7, center + 1] = 0.0
    a2_br[8, center - 5 : center + 6] = 0.0

    smooth_srcs = [
        Z_RAW_CLONE, Z_ISS_NESTING, Z_ISS_SPECIAL,
        Z_RAW_CRYPTIC, Z_ISS_LAYOUT, Z_RAW_NESTING_2SP,
    ]
    smooth_dsts = [
        Z_SMOOTH_CLONE, Z_SMOOTH_NEST, Z_SMOOTH_SPECIAL,
        Z_SMOOTH_CRYPTIC, Z_SMOOTH_LAYOUT, Z_SMOOTH_NEST_2SP,
    ]
    smooth_slots = [0, 16, 17, 32, 33, 18]
    a2_wv[smooth_srcs, smooth_slots] = 1
    a2_wo[smooth_slots, smooth_dsts] = 1
    a2_wv_s[smooth_slots] = [5.0 / 2.0, 1.2 / 4.2, 1.2 / 4.3, 3.0 / 1.6, 1.0 / 4.3, 1.0]

    h8_a2_slot = 8 * 16
    a2_wv[Z_RAW_SKEL_CLONE, h8_a2_slot] = 1
    a2_wv_s[h8_a2_slot] = 1.54
    a2_wo[h8_a2_slot, Z_SMOOTH_SKEL_CLONE] = 1

    # Top-level (non-class) nesting smoothing in Head 1 -> Z_ENCLOSING_DEPTH (reused post-ctx1)
    Z_SMOOTH_NEST_TOP = Z_ENCLOSING_DEPTH
    h1_top_slot, h3_slots = 19, [48, 49]
    a2_wv[Z_RAW_NEST_TOP, h1_top_slot] = 1
    a2_wv_s[h1_top_slot] = 1.0
    a2_wo[h1_top_slot, Z_SMOOTH_NEST_TOP] = 1

    # Head 3: [-24..0] causal lookback for C# `bool Try*` narrow catches & Lua `pcall(` fallbacks
    a2_wq[[Z_NARROW_CATCH, Z_KT_GETNULL], h3_slots] = 1
    a2_wk[[Z_TRY_BOOL_SIG, Z_LUA_PCALL], h3_slots] = 1
    a2_wq_s[h3_slots] = [6.0, 6.0 / 8.0]
    a2_wk_s[h3_slots] = 6.0
    a2_wv[[Z_TRY_BOOL_SIG, Z_LUA_PCALL], h3_slots] = 1
    a2_wv_s[h3_slots] = [1.3, 1.5]
    a2_wo[h3_slots, [Z_TRY_BOOL_SIG, Z_LUA_PCALL]] = 1

    h4_slot = 64
    a2_wv[Z_COMM_CODE_LINE, h4_slot] = 1
    a2_wo[h4_slot, Z_SMOOTH_COMM_CODE] = 1

    h5_qk, h5_pos, h5_neg = 80, 80, 81
    a2_wq[Z_ACTIVE, h5_qk] = 1
    a2_wk[[Z_TYPED_INTERNAL_SIG, Z_BOUNDARY_SIG_LINE], h5_qk] = 1
    a2_wq_s[h5_qk] = a2_wk_s[h5_qk] = 8.0
    a2_wv[[Z_TYPED_INTERNAL_SIG, Z_BOUNDARY_SIG_LINE], [h5_pos, h5_neg]] = 1
    a2_wv_s[[h5_pos, h5_neg]] = [1.0, 4.0]
    a2_wo[[h5_pos, h5_neg], Z_HAS_RECENT_TYPED_SIG] = [1, -1]

    h6_qk, h6_pos, h6_neg = 96, 96, 97
    a2_wq[Z_ACTIVE, h6_qk] = 1
    a2_wk[[Z_NEUTRAL_RET, Z_FAIL_FAST], h6_qk] = 1
    a2_wq_s[h6_qk] = a2_wk_s[h6_qk] = 8.0
    a2_wv[[Z_NEUTRAL_RET, Z_FAIL_FAST], [h6_pos, h6_neg]] = 1
    a2_wv_s[[h6_pos, h6_neg]] = [1.0, 3.0]
    a2_wo[[h6_pos, h6_neg], Z_NEAR_NEUTRAL_RET] = [1, -1]

    h7_slots = [112, 113, 114, 115]
    h7_srcs = [Z_CONT_ENV, Z_CONT_DEV_DEF, Z_FAIL_FAST, Z_HAS_COALESCE]
    h7_dsts = [Z_NEXT_CONT_ENV, Z_NEXT_CONT_DEV, Z_NEXT_FAIL_FAST, Z_NEXT_COAL]
    a2_wv[h7_srcs, h7_slots] = 1
    a2_wv_s[h7_slots] = 2.0
    a2_wo[h7_slots, h7_dsts] = 1

    trit_mats["attn2_wq"], vecs_1d["attn2_wq_s"] = a2_wq, a2_wq_s
    trit_mats["attn2_wk"], vecs_1d["attn2_wk_s"] = a2_wk, a2_wk_s
    trit_mats["attn2_wv"], vecs_1d["attn2_wv_s"], trit_mats["attn2_wo"] = a2_wv, a2_wv_s, a2_wo
    vecs_1d["attn2_brel"] = a2_br
    vecs_1d["attn2_bsink"] = np.array(
        [0.7, -12.0, -12.0, 1.5, math.log(5.0), 4.0, 4.0, 0.0, 1.1], dtype=np.float32
    )

    n_cm2 = 64
    cm2_w = np.zeros((N_Z, n_cm2), dtype=np.int8)
    cm2_s = np.ones((n_cm2,), dtype=np.float32)
    cm2_b = np.zeros((n_cm2,), dtype=np.float32)
    cm2_2w = np.zeros((n_cm2, N_Z), dtype=np.int8)
    cm2_2b = np.zeros((N_Z,), dtype=np.float32)

    for pair_u in (0, 1):
        cm2_w[[Z_RAW_CLONE, Z_SMOOTH_CLONE], pair_u] = 1
        cm2_w[[Z_IN_DOC, Z_IN_PAREN], pair_u] = -1
    for pair_u in (2, 3):
        cm2_w[[Z_RAW_CRYPTIC, Z_SMOOTH_CRYPTIC, Z_CRYPTIC_GATE], pair_u] = 1
        cm2_w[Z_IN_DOC, pair_u] = -1
    cm2_s[:4] = [2.0, 2.0, 1.6, 1.6]
    cm2_b[:4] = [-3.35, -4.35, -3.6, -4.6]
    cm2_2w[[0, 1], Z_ISS_CLONE] = [1, -1]
    cm2_2w[[2, 3], Z_ISS_NAMING] = [1, -1]

    cm2_w[Z_RAW_NESTING_2SP, [4, 5]] = 1
    cm2_w[Z_SMOOTH_NEST_2SP, [6, 7]] = 1
    for wide_u in range(4, 8):
        cm2_w[[*wide_dsts, Z_IN_DOC, Z_IN_PAREN], wide_u] = -1
    nest_sm_ratio = 1.2 / 4.2
    cm2_s[[6, 7]] = nest_sm_ratio
    cm2_b[4:8] = [0.0, -1.0, 0.0, -nest_sm_ratio]
    cm2_2w[[4, 5], Z_ISS_NESTING] = [1, -1]
    cm2_2w[[6, 7], Z_SMOOTH_NEST] = [1, -1]

    # Add top-level 4-space nesting only when file does NOT use class/impl containers
    cm2_w[Z_RAW_NEST_TOP, [8, 9]] = 1
    cm2_w[Z_SMOOTH_NEST_TOP, [10, 11]] = 1
    for top_u in range(8, 12):
        cm2_w[[Z_HAS_CONTAINER, Z_ISS_NESTING, Z_IN_DOC, Z_IN_PAREN, Z_SHALLOW_LINE], top_u] = -1
    cm2_s[[10, 11]] = nest_sm_ratio
    cm2_b[8:12] = [0.0, -1.0, 0.0, -nest_sm_ratio]
    cm2_2w[[8, 9], Z_ISS_NESTING] = [1, -1]
    cm2_2w[[10, 11], Z_SMOOTH_NEST] = [1, -1]

    # Cancel Z_SMOOTH_NEST smearing onto shallow / docstring / paren-continuation lines
    for cancel_u in (12, 13):
        cm2_w[[Z_SMOOTH_NEST, Z_SHALLOW_LINE, Z_IN_DOC, Z_IN_PAREN], cancel_u] = 1
    cm2_s[[12, 13]] = 1.0
    cm2_b[[12, 13]] = [-1.0, -2.0]
    cm2_2w[[12, 13], Z_SMOOTH_NEST] = [-1, 1]

    paranoid_dsts = [
        (Z_ISS_ERROR, 1), (Z_CLEAN_BONUS, -1),
        (Z_GUARD_CLAUSE, -1), (Z_EARLY_EXIT, -1),
    ]
    ctx2_rules: list[tuple[list[tuple[int, int]], float, list[tuple[int, int]], float]] = [
        ([(Z_COMM_CODE_LINE, 1), (Z_SMOOTH_COMM_CODE, 1), (Z_IN_DOC, -1)],
         -1.293, [(Z_ISS_COMMENT, 1)], 150.0),
        ([(Z_TYPE_HEDGE_IF, 1), (Z_HAS_RECENT_TYPED_SIG, 1), (Z_NEAR_NEUTRAL_RET, 1), (Z_IN_DOC, -1)],
         -2.5, paranoid_dsts, 4.0),
        ([(Z_ENV_CHAIN_1L, 1), (Z_NEXT_FAIL_FAST, -1), (Z_IN_DOC, -1)],
         -0.5, [(Z_ISS_ERROR, 1)], 4.0),
        ([(Z_ENV_DEV_1L, 1), (Z_ENV_CHAIN_1L, -1), (Z_NEXT_FAIL_FAST, -1), (Z_IN_DOC, -1)],
         -0.5, [(Z_ISS_ERROR, 1)], 4.0),

        ([(Z_HAS_ENV, 1), (Z_NEXT_CONT_DEV, 1), (Z_ENV_VETO, -1),
          (Z_SEMI_ONLY, -1), (Z_ENV_CHAIN_1L, -1), (Z_ENV_DEV_1L, -1), (Z_IN_DOC, -1)],
         -1.5, [(Z_ISS_ERROR, 1)], 4.0),
        ([(Z_HAS_ENV, 1), (Z_NEXT_CONT_ENV, 1), (Z_NEXT_COAL, 1),
          (Z_ENV_VETO, -1), (Z_SEMI_ONLY, -1), (Z_NEXT_FAIL_FAST, -1),
          (Z_ENV_CHAIN_1L, -1), (Z_ENV_DEV_1L, -1), (Z_NEXT_CONT_DEV, -1), (Z_IN_DOC, -1)],
         -2.5, [(Z_ISS_ERROR, 1)], 4.0),
        ([(Z_HAS_ENV, 1), (Z_NEXT_CONT_ENV, 1), (Z_HAS_COALESCE, 1), (Z_NEXT_COAL, -1),
          (Z_ENV_VETO, -1), (Z_SEMI_ONLY, -1), (Z_NEXT_FAIL_FAST, -1),
          (Z_ENV_CHAIN_1L, -1), (Z_ENV_DEV_1L, -1), (Z_NEXT_CONT_DEV, -1), (Z_IN_DOC, -1)],
         -2.5, [(Z_ISS_ERROR, 1)], 4.0),
        ([(Z_NARROW_CATCH, 1), (Z_TRY_BOOL_SIG, 1), (Z_ISS_ERROR, 1), (Z_IN_DOC, -1)],
         -2.5, [(Z_ISS_ERROR, -1)], 4.0),
        ([(Z_KT_GETNULL, 1), (Z_LUA_PCALL, 1), (Z_IN_DOC, -1)],
         -8.5, [(Z_ISS_ERROR, 1)], 4.0),

        # Track 3 Boolean-AND readouts: cursor mutation, flag+out-ref threading,
        # multi-line alpha-invariant skeleton clones, and intra-scope state entanglement
        ([(Z_CURSOR_MUT, 1), (Z_CURSOR_INDENT, 1), (Z_HAS_ENC_LOOP, 1),
          (Z_HAS_ENC_BR1, 1), (Z_ISS_NESTING, -1), (Z_IN_DOC, -1), (Z_IN_PAREN, -1)],
         -3.6, [(Z_ISS_NESTING, 1)], 4.0),
        ([(Z_CURSOR_GENERIC, 1), (Z_CURSOR_INDENT, 1), (Z_HAS_COND_LOOP, 1),
          (Z_HAS_ENC_BR2, 1), (Z_ISS_NESTING, -1), (Z_IN_DOC, -1), (Z_IN_PAREN, -1)],
         -3.6, [(Z_ISS_NESTING, 1)], 4.0),
        ([(Z_SIG_CAND, 1), (Z_ARITY_6PLUS, 1), (Z_HAS_BOOL_SPAN, 1),
          (Z_HAS_OUTREF_SPAN, 1), (Z_ISS_SPECIAL, -1), (Z_IN_DOC, -1)],
         -3.6, [(Z_ISS_SPECIAL, 1)], 4.0),
        ([(Z_RAW_SKEL_CLONE, 1), (Z_SMOOTH_SKEL_CLONE, 1), (Z_HAS_CTRL_MOD, 1),
          (Z_ISS_CLONE, -1), (Z_IN_DOC, -1), (Z_IN_PAREN, -1)],
         -2.38, [(Z_ISS_CLONE, 1)], 8.0),
        ([(Z_HAS_MUT_DENS, 1), (Z_HAS_CTRL_DENS, 1), (Z_IN_DOC, -1)],
         -1.6, [(Z_SCOPE_ENTANGLE, 1)], 4.0),
    ]
    used_cm2 = 14 + 2 * len(ctx2_rules)
    for idx, (terms, bias, dst_signs, gain) in enumerate(ctx2_rules):
        unit = 14 + 2 * idx
        for src_z, trit in terms:
            cm2_w[src_z, [unit, unit + 1]] = trit
        cm2_s[[unit, unit + 1]] = gain
        cm2_b[[unit, unit + 1]] = [bias * gain, bias * gain - 1.0]
        for dst_z, sign_out in dst_signs:
            cm2_2w[[unit, unit + 1], dst_z] = [sign_out, -sign_out]

    trit_mats["cmlp2_w"] = cm2_w[:, :used_cm2]
    vecs_1d["cmlp2_s"], vecs_1d["cmlp2_b"] = cm2_s[:used_cm2], cm2_b[:used_cm2]
    trit_mats["cmlp2_2_w"], vecs_1d["cmlp2_2_b"] = cm2_2w[:used_cm2, :], cm2_2b

    # Block 3 readout: {-1, 0, +1} squiggle head & balanced-ternary System One heads
    issue_regs = [
        Z_ISS_NESTING, Z_ISS_NAMING, Z_ISS_COMMENT, Z_ISS_ERROR,
        Z_ISS_DENSE, Z_ISS_SPECIAL, Z_ISS_CLONE, Z_ISS_LAYOUT,
    ]
    sq_w = np.zeros((N_Z, 8), dtype=np.int8)
    sq_w[issue_regs, range(8)] = 1
    sq_w[[Z_SMOOTH_NEST, Z_SMOOTH_SPECIAL, Z_SMOOTH_LAYOUT], [0, 5, 7]] = 1
    sq_s = np.array([4.2, 4.6, 4.8, 4.7, 4.7, 4.3, 4.8, 4.3], dtype=np.float32)
    sq_b = np.full((8,), -2.7, dtype=np.float32)
    trit_mats["sq_w"], vecs_1d["sq_s"], vecs_1d["sq_b"] = sq_w, sq_s, sq_b

    pool_gate_w = np.zeros((N_Z, 1), dtype=np.int8)
    pool_gate_w[[Z_ACTIVE, Z_IN_DOC], 0] = [1, -1]
    trit_mats["pool_gate_w"] = pool_gate_w
    vecs_1d["pool_gate_s"] = np.array([12.0], dtype=np.float32)
    vecs_1d["pool_gate_b"] = np.array([-6.0], dtype=np.float32)

    # Build target readout weights (9 logical rows -> 31 logits), then decompose into trits
    target_doc2 = np.zeros((9, 31), dtype=np.float32)
    doc2_b = np.zeros((31,), dtype=np.float32)
    doc2_b[:2] = [1.88, 1.98]
    target_doc2[:8, 0] = [-2.64, -0.66, -1.32, -2.86, -0.66, -2.20, -2.86, -0.22]
    target_doc2[:8, 1] = [-2.42, -0.66, -1.32, -2.86, -0.66, -2.20, -2.86, -0.22]
    target_doc2[8, :2] = [0.30, 0.25]

    subscore_heads = (
        [(0, 1.25)],
        [(1, 1.25)],
        [(4, 0.85), (5, 1.0), (6, 1.1)],
        [(2, 1.35)],
        [(3, 1.40), (5, 0.6)],
        [(7, 1.10), (0, 0.35)],
    )
    for score_i, units in enumerate(subscore_heads):
        base = 2 + 4 * score_i
        doc2_b[base : base + 4] = [-2.3, -0.9, 0.6, 2.0]
        for unit_i, mult in units:
            target_doc2[unit_i, base : base + 4] += [
                2.1 * mult, 0.8 * mult, -0.7 * mult, -2.2 * mult,
            ]
        target_doc2[8, [base, base + 3]] += [-0.30, 0.30]

    style_base = 26
    doc2_b[style_base : style_base + 5] = [1.25, 1.35, -2.10, -2.10, -2.20]
    target_doc2[:8, style_base] -= 1.20
    target_doc2[:8, style_base + 1] -= 1.05
    target_doc2[8, [style_base, style_base + 1]] += [0.35, 0.25]
    target_doc2[[4, 0], style_base + 3] += [3.0, 1.1]
    target_doc2[[2, 3, 5, 6, 7, 0], style_base + 4] += [2.9, 3.1, 3.3, 3.0, 1.8, 1.1]

    n_doc1 = 140
    doc1_w = np.zeros((2 * N_Z, n_doc1), dtype=np.int8)
    doc1_s = np.ones((n_doc1,), dtype=np.float32)
    doc1_b = np.zeros((n_doc1,), dtype=np.float32)
    doc2_w = np.zeros((n_doc1, 31), dtype=np.int8)

    caps_mean = (1.25, 0.45, 0.60, 1.55, 0.40, 1.15, 1.55, 0.35)
    caps_max = (1.85, 0.55, 0.70, 1.85, 0.45, 1.35, 1.85, 0.40)
    mean_gain, max_gain, max_deadzone = 2.8, 12.0, 0.030
    step_issue = 0.22
    doc_cursor = 0

    for head_i, reg in enumerate(issue_regs):
        cap_m = caps_mean[head_i]
        cap_p = caps_max[head_i]
        for trit_i, pow3 in enumerate((1.0, 3.0, 9.0)):
            step_c = step_issue * pow3
            u_pos, u_neg = doc_cursor, doc_cursor + 1
            v_pos, v_neg = doc_cursor + 2, doc_cursor + 3
            doc_cursor += 4

            doc1_w[reg, [u_pos, u_neg]] = 1
            doc1_s[[u_pos, u_neg]] = mean_gain * step_c
            doc1_b[[u_pos, u_neg]] = [0.0, -cap_m * step_c]

            doc1_w[N_Z + reg, [v_pos, v_neg]] = 1
            doc1_s[[v_pos, v_neg]] = max_gain * step_c
            peak_off = max_gain * max_deadzone
            doc1_b[[v_pos, v_neg]] = [-peak_off * step_c, (-peak_off - cap_p) * step_c]

            for col_j in range(31):
                w_val = float(target_doc2[head_i, col_j])
                if abs(w_val) <= 1e-6:
                    continue
                trit = _balanced_trits(w_val, step_issue, num_trits=3)[trit_i]
                if trit != 0:
                    doc2_w[[u_pos, u_neg], col_j] = [trit, -trit]
                    doc2_w[[v_pos, v_neg], col_j] = [trit, -trit]

    step_bonus = 0.05
    for trit_i, pow3 in enumerate((1.0, 3.0, 9.0)):
        step_c = step_bonus * pow3
        u_pos, u_neg = doc_cursor, doc_cursor + 1
        doc_cursor += 2
        doc1_w[[Z_GUARD_CLAUSE, Z_CLEAN_IDIOM], [u_pos, u_neg]] = 1
        doc1_s[[u_pos, u_neg]] = 2.0 * step_c
        doc1_b[[u_pos, u_neg]] = [0.0, -1.0 * step_c]
        for col_j in range(31):
            w_val = float(target_doc2[8, col_j])
            if abs(w_val) <= 1e-6:
                continue
            trit = _balanced_trits(w_val, step_bonus, num_trits=3)[trit_i]
            if trit != 0:
                doc2_w[[u_pos, u_neg], col_j] = [trit, -trit]

    struct_clamps = (
        (Z_ENCLOSING_DEPTH, 2.2, 0.22, 0.35),
        (Z_INDENT, 0.55, 3.10, 0.25),
        (Z_ENCLOSING_BRANCHES, 1.6, 0.80, 0.65),
        (Z_IMPERATIVE_LINE, 2.8, 0.45, 0.45),
        (N_Z + Z_DISTANT_CLONE, 6.0, 0.112, 0.40),
    )
    for src_reg, s_gain, s_offset, w_mag in struct_clamps:
        u_pos, u_neg = doc_cursor, doc_cursor + 1
        doc_cursor += 2
        doc1_w[src_reg, [u_pos, u_neg]] = 1
        doc1_s[[u_pos, u_neg]] = s_gain * w_mag
        off_scaled = s_offset * s_gain * w_mag
        doc1_b[[u_pos, u_neg]] = [-off_scaled, -off_scaled - w_mag]
        doc2_w[[u_pos, u_neg], 0] = [-1, 1]
        doc2_w[[u_pos, u_neg], 1] = [-1, 1]

    style_exact = (
        ([Z_RAW_CRYPTIC], 3.5 * 3.3, style_base + 3),
        ([N_Z + Z_RAW_CRYPTIC], 0.6 * 3.3, style_base + 3),
        ([Z_CEREMONY], 2.0 * 3.5, style_base + 2),
        ([Z_LONG_IDENT], 1.5 * 3.5, style_base + 2),
        ([Z_CLEAN_IDIOM], 2.0 * 1.3, style_base),
        ([Z_EARLY_EXIT, Z_GUARD_CLAUSE], 1.2 * 1.3, style_base),
    )
    for srcs, sc_val, dst_col in style_exact:
        slot = doc_cursor
        doc_cursor += 1
        doc1_w[srcs, slot] = 1
        doc1_s[slot] = sc_val
        doc2_w[slot, dst_col] = 1

    # Track 3 mutable-scope entanglement saturating readout (mean + span-damped 32-line peak)
    ent_row = np.zeros((31,), dtype=np.float32)
    ent_row[:2] = [-1.45, -1.30]
    for score_i, mult in ((0, 0.75), (2, 0.85)):
        base = 2 + 4 * score_i
        ent_row[base : base + 4] += [2.1 * mult, 0.8 * mult, -0.7 * mult, -2.2 * mult]
    ent_row[style_base : style_base + 2] -= [1.10, 0.90]
    ent_row[style_base + 3] += 1.40

    for trit_i, pow3 in enumerate((1.0, 3.0, 9.0)):
        step_c = step_issue * pow3
        u_pos, u_neg = doc_cursor, doc_cursor + 1
        v_pos, v_neg = doc_cursor + 2, doc_cursor + 3
        doc_cursor += 4

        doc1_w[Z_SCOPE_ENTANGLE, [u_pos, u_neg]] = 1
        doc1_s[[u_pos, u_neg]] = 2.6 * step_c
        doc1_b[[u_pos, u_neg]] = [-0.08 * step_c, (-0.08 - 0.85) * step_c]

        doc1_w[N_Z + Z_SCOPE_ENTANGLE, [v_pos, v_neg]] = 1
        doc1_s[[v_pos, v_neg]] = 3.2 * step_c
        doc1_b[[v_pos, v_neg]] = [-0.45 * step_c, (-0.45 - 1.10) * step_c]

        for col_j in range(31):
            w_val = float(ent_row[col_j])
            if abs(w_val) <= 1e-6:
                continue
            trit = _balanced_trits(w_val, step_issue, num_trits=3)[trit_i]
            if trit != 0:
                doc2_w[[u_pos, u_neg], col_j] = [trit, -trit]
                doc2_w[[v_pos, v_neg], col_j] = [trit, -trit]

    trit_mats["doc1_w"] = doc1_w[:, :doc_cursor]
    vecs_1d["doc1_s"], vecs_1d["doc1_b"] = doc1_s[:doc_cursor], doc1_b[:doc_cursor]
    trit_mats["doc2_w"], vecs_1d["doc2_b"] = doc2_w[:doc_cursor, :], doc2_b


def compile_ternary_tensors() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return raw {-1, 0, +1} int8 weight matrices and 1D float32 scales/biases."""
    trit_mats: dict[str, np.ndarray] = {}
    vecs_1d: dict[str, np.ndarray] = {}
    _build_block0(trit_mats, vecs_1d)
    z_helpers = _build_block1(trit_mats, vecs_1d)
    _build_block2_and_heads(trit_mats, vecs_1d, z_helpers)

    for name, arr in trit_mats.items():
        unique_vals = set(np.unique(arr).tolist())
        assert arr.ndim >= 2 and unique_vals.issubset({-1, 0, 1}), f"{name}: {unique_vals}"
    return trit_mats, vecs_1d


def compile_weights(output_path: Path | None = None) -> dict[str, np.ndarray]:
    """Compile and pack 5-trit-per-byte ternary weights into `weights.npz`."""
    trit_mats, vecs_1d = compile_ternary_tensors()
    trit_chunks: list[np.ndarray] = []
    trit_meta: list[str] = []
    for name, arr in trit_mats.items():
        packed = _pack_trits(arr)
        trit_chunks.append(packed)
        dims = ",".join(map(str, arr.shape))
        trit_meta.append(f"{name}:{dims}:{len(packed)}")

    vec_chunks: list[np.ndarray] = []
    vec_meta: list[str] = []
    for name, arr in vecs_1d.items():
        flat = arr.astype(np.float32).ravel()
        vec_chunks.append(flat)
        dims = ",".join(map(str, arr.shape))
        vec_meta.append(f"{name}:{dims}:{len(flat)}")

    trit_buf = np.concatenate(trit_chunks)
    vec_buf = np.concatenate(vec_chunks)
    meta_str = "|".join([" ; ".join(trit_meta), " ; ".join(vec_meta)])
    target = output_path or (Path(__file__).with_name("codehygiene") / "weights.npz")
    np.savez_compressed(target, trits=trit_buf, vecs=vec_buf, meta=np.array(meta_str))
    return unpack_ternary_archive(trit_buf, vec_buf, meta_str)


if __name__ == "__main__":
    out_file = Path(__file__).with_name("codehygiene") / "weights.npz"
    trits_raw, vecs_raw = compile_ternary_tensors()
    compile_weights(out_file)
    docs_weights = Path(__file__).with_name("docs") / "weights.npz"
    if docs_weights.parent.is_dir():
        docs_weights.write_bytes(out_file.read_bytes())
    size_kb = out_file.stat().st_size / 1024
    num_trits = sum(arr.size for arr in trits_raw.values())
    num_vecs = sum(arr.size for arr in vecs_raw.values())
    print(
        f"Compiled {len(trits_raw)} ternary matrices ({num_trits:,} trits in {{-1,0,+1}}, "
        f"{num_vecs:,} scale/bias scalars, {size_kb:.2f} KiB)"
    )
