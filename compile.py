"""Closed-form 1.58-bit Ternary Weight-Space Program compiling circuits into weights.npz.

Every 2D/3D weight, convolution, projection, causal, and embedding tensor in the model
has entries strictly in {-1, 0, +1} (425,044 trits packed at 5 trits per uint8 byte,
1.60 bits/weight), paired only with 1D per-output-channel scales and biases:
  - Block 0 (_build_block0): Maps (L, 128) ASCII bytes + column positions into 52 per-cell
    channels (HC_*), using causal prefix matrices and De Morgan closer gates to mask strings
    (16-quote sawtooth) and polyglot comments (//, /* */, # , -- ) left-to-right.
  - Block 1 (_build_block1): Runs k=3, 7, 13 {-1, 0, +1} integer-margin convolutions and
    40 SimHash stem filters (_sketch_bank) for live code and inline comments.
  - Block 2 (_build_block2_and_heads): Two cross-line sink-attention + MLP passes using
    Paley Hadamard and Gram-fitted ternary projections populate Z[40:86].
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

# Per-cell feature channels emitted by Block 0 (52 channels per grid cell)
(
    HC_ACT, HC_COMM, HC_CODE, HC_CNS, HC_MNS, HC_IND, HC_FNS, HC_ODD,
    HC_M42, HC_M40, HC_D8, HC_D10, HC_D12, HC_G80, HC_G96, HC_ID,
    HC_LO, HC_UP, HC_DIG, HC_UND, HC_DOT, HC_QM, HC_NOT_WORD, HC_EQ,
    HC_LB, HC_RB, HC_LP, HC_RP, HC_SEM, HC_SK_VETO, HC_QT, HC_SL_BOOST,
) = range(32)
HC_BP, HC_BN, HC_G110, HC_SL_C = 32, 39, 46, 47
HC_STR, HC_LEAD, HC_D16, HC_D20 = 48, 49, 50, 51
N_HC = 52

# Byte and column embedding layout inside Block 0 (46 channels)
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
N_EMB = 46

# Block 0 lexical pre-convolution, causal prefixes, and quote/comment gates
(
    PRE_SLASH_COMM, PRE_BLOCK_OPEN, PRE_BLOCK_CLOSE, PRE_HASH_COMM,
    PRE_DASH_COMM, PRE_DQUOTE, PRE_SQUOTE, PRE_APOSTROPHE,
) = range(8)
N_PRE = 16
C1_NONSPACE, C1_DQUOTES, C1_SQUOTES, C1_BLOCK_OPENS = range(4)
N_PREF1 = 4

G1_INDENT_SPACE, G1_FIRST_NONSPACE, G1_DOC_STAR, G1_ANY_COMM_START = 0, 1, 2, 3
G1_CLOSE_HI, G1_INDENT_BASE, G1_LINE_COMM_START = 4, 5, 13
G1_LEAD_TWO, G1_LEAD_ONE, G1_DQUOTE_SAW, G1_SQUOTE_SAW, G1_CLOSE_LO = 14, 15, 16, 32, 48
N_GATE1 = 50
C2_IN_COMMENT, C2_IN_LINE_COMM, C2_IN_COMM_B, C2_IN_COMM_C = 0, 1, 2, 3
N_PREF2 = 4

# Line-level residual stream Z registers (192 channels per line)
N_Z = 192
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

# Cross-line attention context registers (Z[40..63]) and SimHash slices (Z[112..192])
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

# Bounded [0, 1] per-line issue registers (Z[64..78]) and smoothed registers (Z[80..86])
(
    Z_ISS_NESTING, Z_ISS_NAMING, Z_ISS_COMMENT, Z_ISS_ERROR,
    Z_ISS_DENSE, Z_ISS_SPECIAL, Z_RAW_CLONE, Z_ISS_LAYOUT,
    Z_RAW_CRYPTIC, Z_CLEAN_BONUS, Z_ISS_CLONE,
    Z_USES_WIDE_INDENT, Z_RAW_NESTING_2SP,
    Z_WIDE_DUP_A, Z_WIDE_DUP_B,
) = range(64, 79)
(
    Z_SMOOTH_CLONE, Z_SMOOTH_NEST, Z_SMOOTH_CRYPTIC,
    Z_SMOOTH_SPECIAL, Z_SMOOTH_LAYOUT, Z_SMOOTH_NEST_2SP,
    Z_IS_INDENTED,
) = range(80, 87)
Z_CODE_SKETCH, Z_COMM_SKETCH, SKETCH_DIM = 112, 152, 40

# Named attention head slices
H0_QK = slice(0, 16)
H4_QK = slice(64, 80)
H5_QK = slice(80, 96)
H45_QK = slice(64, 96)


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
    out_scale: float = 1.0,
) -> None:
    """Write a {-1, 0, +1} ASCII bit-template into `channel` using integer max-bound margin."""
    span = len(pattern) + wl + wr
    start = (kernel.shape[0] - span) // 2 + wl
    kernel[:, :, channel] = 0
    thresh = 7 * len(pattern)

    if wl:
        kernel[start - 1, HC_ID, channel] = -1
    if wr:
        kernel[start + len(pattern), HC_ID, channel] = -1
    if lead:
        kernel[start, HC_LEAD, channel] = 1
        thresh += 1

    for offset, char in enumerate(pattern):
        bits_on = (((ord(char.lower()) & 0x7F) >> _BITS7) & 1) == 1
        kernel[start + offset, HC_BP + _BITS7[bits_on], channel] = 1
        kernel[start + offset, HC_BN + _BITS7[~bits_on], channel] = 1
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


def _sketch_bank() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build 40 {-1, 0, +1} 4-char SimHash filters (code: 0..39, comments: 40..79)."""
    total_ch = 2 * SKETCH_DIM
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
    return sketch_w, sketch_s, sketch_b


def _paley_uniform_proj() -> np.ndarray:
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

    rng = np.random.RandomState(0)
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


def _build_block0(trit_mats: dict[str, np.ndarray], vecs_1d: dict[str, np.ndarray]) -> None:
    """Compile {-1, 0, +1} byte/column embeddings and quote/comment masking circuits."""
    tok_emb = np.zeros((128, N_EMB), dtype=np.int8)
    tok_emb[1:, EMB_ACTIVE] = 1
    tok_emb[33:, EMB_NONSPACE] = 1
    tok_emb[32, EMB_SPACE] = 1
    tok_emb[97:123, EMB_LOWER] = 1
    tok_emb[65:91, EMB_UPPER] = 1
    tok_emb[48:58, EMB_DIGIT] = 1

    for byte_val in range(1, 128):
        char = chr(byte_val)
        tok_emb[byte_val, EMB_IDENT] = int(char.isalnum() or char in "_$#")
        bits = ((ord(char.lower()) & 0x7F) >> _BITS7) & 1
        tok_emb[byte_val, EMB_ASCII_BITS : EMB_ASCII_BITS + 7] = bits.astype(np.int8)

    for char, dst_col in zip("_/*:?.={}();", range(EMB_UND, EMB_SEMI + 1)):
        tok_emb[ord(char), dst_col] = 1
    tok_emb[[ord("|"), ord("&")], EMB_LOGIC] = 1
    tok_emb[[ord("'"), ord('"'), ord("`")], EMB_ANY_QUOTE] = 1
    for char, dst_col in zip('"\'\\#-', range(EMB_DQUOTE, EMB_DASH + 1)):
        tok_emb[ord(char), dst_col] = 1
    tok_emb[[ord("&"), ord("<")], EMB_LIFE_PFX] = 1

    col_emb = np.zeros((MAX_COLS, N_EMB), dtype=np.int8)
    col_idx = np.arange(MAX_COLS)
    col_emb[0, EMB_COL0] = 1
    col_emb[col_idx % 2 == 1, EMB_COL_ODD] = 1
    col_emb[col_idx % 4 == 2, EMB_COL_MOD4_2] = 1
    col_emb[(col_idx > 0) & (col_idx % 4 == 0) & (col_idx <= 28), EMB_COL_IND4] = 1
    col_thresholds = (
        (8, EMB_COL_GE8), (10, EMB_COL_GE10), (12, EMB_COL_GE12),
        (80, EMB_COL_GE80), (96, EMB_COL_GE96), (110, EMB_COL_GE110),
        (16, EMB_COL_GE16), (20, EMB_COL_GE20),
    )
    for min_col, dst_col in col_thresholds:
        col_emb[col_idx >= min_col, dst_col] = 1
    trit_mats["tok_emb"], trit_mats["col_emb"] = tok_emb, col_emb

    # Local 3-char {-1, 0, +1} detectors for comment openers and unescaped quotes
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

    pre_w[0, EMB_NONSPACE, PRE_HASH_COMM] = -1
    pre_w[[1, 2], [EMB_HASH, EMB_SPACE], PRE_HASH_COMM] = 1
    pre_s[PRE_HASH_COMM], pre_b[PRE_HASH_COMM] = 2.0, -3.0

    pre_w[[0, 1, 2], [EMB_DASH, EMB_DASH, EMB_SPACE], PRE_DASH_COMM] = 1
    pre_s[PRE_DASH_COMM], pre_b[PRE_DASH_COMM] = 2.0, -5.0

    pre_w[1, EMB_DQUOTE, PRE_DQUOTE] = 1
    pre_w[0, [EMB_BACKSLASH, EMB_SQUOTE], PRE_DQUOTE] = -1
    pre_w[2, EMB_SQUOTE, PRE_DQUOTE] = -1
    pre_w[1, EMB_SQUOTE, PRE_SQUOTE] = 1
    pre_w[0, [EMB_BACKSLASH, EMB_LIFE_PFX, EMB_DQUOTE], PRE_SQUOTE] = -1
    pre_w[2, EMB_DQUOTE, PRE_SQUOTE] = -1

    pre_w[0, [EMB_LOWER, EMB_UPPER], PRE_APOSTROPHE] = 1
    pre_w[1, EMB_SQUOTE, PRE_APOSTROPHE] = 1
    pre_w[2, [EMB_LOWER, EMB_UPPER], PRE_APOSTROPHE] = 1
    pre_b[PRE_APOSTROPHE] = -2.0
    trit_mats["pre_conv3_w"] = pre_w
    vecs_1d["pre_conv3_s"], vecs_1d["pre_conv3_b"] = pre_s, pre_b

    off_pre = N_EMB
    off_pref1 = N_EMB + N_PRE
    pref1 = np.zeros((off_pref1, N_PREF1), dtype=np.int8)
    pref1_rows = [
        EMB_NONSPACE, off_pre + PRE_DQUOTE, off_pre + PRE_SQUOTE,
        off_pre + PRE_APOSTROPHE, off_pre + PRE_BLOCK_OPEN,
    ]
    pref1_cols = [C1_NONSPACE, C1_DQUOTES, C1_SQUOTES, C1_SQUOTES, C1_BLOCK_OPENS]
    pref1[pref1_rows, pref1_cols] = [1, 1, 1, -1, 1]
    trit_mats["pref1_proj"] = pref1
    trit_mats["W_causal1"] = np.tril(np.ones((MAX_COLS, MAX_COLS), dtype=np.int8), k=-1)
    trit_mats["W_causal2"] = np.tril(np.ones((MAX_COLS, MAX_COLS), dtype=np.int8), k=0)

    # Gate layer with De Morgan /*-paired */ closer: ReLU(close) - ReLU(close - opens)
    gate1_w = np.zeros((off_pref1 + N_PREF1, N_GATE1), dtype=np.int8)
    gate1_s = np.ones((N_GATE1,), dtype=np.float32)
    gate1_b = np.zeros((N_GATE1,), dtype=np.float32)

    gate1_w[[EMB_SPACE, EMB_NONSPACE], [G1_INDENT_SPACE, G1_FIRST_NONSPACE]] = 1
    gate1_w[off_pref1 + C1_NONSPACE, [G1_INDENT_SPACE, G1_FIRST_NONSPACE]] = -1
    gate1_w[[EMB_STAR, off_pref1 + C1_NONSPACE], G1_DOC_STAR] = [1, -1]

    any_comm = [
        off_pre + PRE_SLASH_COMM, off_pre + PRE_BLOCK_OPEN,
        off_pre + PRE_HASH_COMM, off_pre + PRE_DASH_COMM,
    ]
    line_comm = [off_pre + PRE_SLASH_COMM, off_pre + PRE_HASH_COMM, off_pre + PRE_DASH_COMM]
    gate1_w[any_comm, G1_ANY_COMM_START] = 1
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

    gate1_w[off_pref1 + C1_NONSPACE, [G1_LEAD_TWO, G1_LEAD_ONE]] = -1
    gate1_b[[G1_LEAD_TWO, G1_LEAD_ONE]] = [2.0, 1.0]
    for step in range(16):
        saw_scale = 2.0 if step > 0 else 1.0
        dq_gate, sq_gate = G1_DQUOTE_SAW + step, G1_SQUOTE_SAW + step
        gate1_w[off_pref1 + C1_DQUOTES, dq_gate] = 1
        gate1_w[off_pref1 + C1_SQUOTES, sq_gate] = 1
        gate1_s[[dq_gate, sq_gate]] = saw_scale
        gate1_b[[dq_gate, sq_gate]] = -float(step) * saw_scale
    trit_mats["gate1_w"] = gate1_w
    vecs_1d["gate1_s"], vecs_1d["gate1_b"] = gate1_s, gate1_b

    pref2 = np.zeros((N_GATE1, N_PREF2), dtype=np.int8)
    for dst_c2 in (C2_IN_COMMENT, C2_IN_COMM_B, C2_IN_COMM_C):
        pref2[[G1_DOC_STAR, G1_ANY_COMM_START, G1_CLOSE_HI, G1_CLOSE_LO], dst_c2] = [1, 1, -1, 1]
    pref2[G1_LINE_COMM_START, C2_IN_LINE_COMM] = 1
    trit_mats["pref2_proj"] = pref2

    # Fuse [emb, pre_conv3, gate1, c2] into 52 per-cell HC_* channels in {-1, 0, +1}
    off_gate1 = off_pref1
    off_pref2 = off_gate1 + N_GATE1
    fuse_w = np.zeros((off_pref2 + N_PREF2, N_HC), dtype=np.int8)
    fuse_s = np.ones((N_HC,), dtype=np.float32)
    fuse_b = np.zeros((N_HC,), dtype=np.float32)

    in_comm_all = [off_pref2 + C2_IN_COMMENT, off_pref2 + C2_IN_COMM_B, off_pref2 + C2_IN_COMM_C]
    in_comm_pair = in_comm_all[:2]
    in_line_comm = off_pref2 + C2_IN_LINE_COMM

    fuse_w[EMB_ACTIVE, [HC_ACT, HC_COMM, HC_SL_C, HC_CODE]] = 1
    fuse_w[in_comm_all[0], HC_COMM] = 1
    fuse_w[in_comm_pair, HC_CODE] = -1
    fuse_w[in_comm_pair, HC_CNS] = -1
    fuse_w[in_line_comm, [HC_SL_C, HC_MNS]] = 1
    fuse_w[EMB_NONSPACE, [HC_CNS, HC_MNS]] = 1
    fuse_w[EMB_SLASH, HC_MNS] = -1
    fuse_b[[HC_COMM, HC_SL_C, HC_MNS]] = -1.0
    fuse_w[[off_gate1 + G1_INDENT_SPACE, off_gate1 + G1_FIRST_NONSPACE], [HC_IND, HC_FNS]] = 1

    saw_trits = np.array([1] + [-1, 1] * 7 + [-1], dtype=np.int8)
    dquote_slice = slice(off_gate1 + G1_DQUOTE_SAW, off_gate1 + G1_DQUOTE_SAW + 16)
    squote_slice = slice(off_gate1 + G1_SQUOTE_SAW, off_gate1 + G1_SQUOTE_SAW + 16)
    fuse_w[dquote_slice, HC_STR] = saw_trits
    fuse_w[squote_slice, HC_STR] = saw_trits
    fuse_w[in_comm_all, HC_STR] = -1

    lead_srcs = [EMB_NONSPACE, off_gate1 + G1_LEAD_TWO, off_gate1 + G1_LEAD_ONE, *in_comm_pair]
    fuse_w[lead_srcs, HC_LEAD] = [1, 1, -1, -1, -1]
    fuse_b[HC_LEAD] = -1.0

    indent_hc_dsts = [HC_ODD, HC_M42, HC_M40, HC_D8, HC_D10, HC_D12, HC_D16, HC_D20]
    for offset, hc_dst in enumerate(indent_hc_dsts):
        fuse_w[[off_gate1 + G1_INDENT_BASE + offset, *in_comm_pair], hc_dst] = [1, -1, -1]
    width_gates = ((EMB_COL_GE80, HC_G80), (EMB_COL_GE96, HC_G96), (EMB_COL_GE110, HC_G110))
    for col_src, hc_dst in width_gates:
        fuse_w[[EMB_NONSPACE, col_src, *in_comm_pair], hc_dst] = [1, 1, -1, -1]
        fuse_b[hc_dst] = -1.0

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
    fuse_w[dquote_slice, HC_SK_VETO] = saw_trits
    fuse_w[squote_slice, HC_SK_VETO] = saw_trits
    fuse_w[in_comm_all[0], HC_SK_VETO] = 1
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
)

_K13_PRIMARY = (
    ("Abstract", 1), ("Repository", 1), ("Configuration", 1), ("Controller", 1),
    ("Strategy", 1), ("Provider", 1), ("Processor", 1), ("Prototype", 1),
    ("primordials", 1), ("HideStackFram", 1), ("internal/", 0), ("continue", 1),
    ("callback(", 0), ("=== undefined", 0), ("!== undefined", 0), ("EventEmitter", 1),
    ("#[allow(", 1), ("import ", 1), ("package ", 1),
)

_K13_EXTENDED = (("#include", 1), ("except Excep", 1), ("except BaseE", 1))

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
)


def _build_block1(
    trit_mats: dict[str, np.ndarray],
    vecs_1d: dict[str, np.ndarray],
) -> dict[str, int]:
    """Compile {-1, 0, +1} k=3/7/13 convolutions, SimHash bank, and line-feature MLP."""
    feat: dict[str, int] = {}
    n_csum, n_conv3, n_conv7, n_conv13 = 16, 64, 96, 64
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
    conv3_w[2, [HC_ID, HC_QT], 2] = -1

    conv3_w[[0, 1, 2], [HC_ID, HC_LO, HC_ID], 3] = 1
    conv3_w[0, [HC_ID, HC_QM], 4] = -1
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

    for ch_i, pat in enumerate(("?.", "=>", "===", "{}", "||", "#", "??="), 5):
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

    conv7_w = np.zeros((7, N_HC, n_conv7), dtype=np.int8)
    conv7_s = np.ones((n_conv7,), dtype=np.float32)
    conv7_b = np.zeros((n_conv7,), dtype=np.float32)
    str_bearing = {"|| '')", "strict'"}
    for start_ch, group in ((0, _K7_PRIMARY), (60, _K7_EXTENDED)):
        for offset, (pat, wl, wr, cs, ld) in enumerate(group):
            ch_i = start_ch + offset
            _sub_filter(
                conv7_w, conv7_s, conv7_b, ch_i, pat,
                wl=wl, wr=wr, cs=cs, lead=ld, no_str=(pat not in str_bearing),
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
    for start_ch, group in ((50, comm_cues_a), (78, comm_cues_b)):
        for offset, pat in enumerate(group):
            ch_i = start_ch + offset
            _sub_filter(conv7_w, conv7_s, conv7_b, ch_i, pat, code=False, comm=True)
            feat[pat] = off_c7 + ch_i

    ch_short_lhs, ch_multi_dig, ch_magic_idx = 81, 82, 83
    conv7_w[1, [HC_LO, HC_LEAD], ch_short_lhs] = 1
    conv7_w[2, [HC_ID, HC_LEAD, HC_COMM, HC_STR], ch_short_lhs] = [1, 1, -1, -1]
    conv7_w[3, [HC_ACT, HC_CNS], ch_short_lhs] = [1, -1]
    conv7_w[4, HC_EQ, ch_short_lhs] = 1
    conv7_w[5, HC_EQ, ch_short_lhs] = -1
    conv7_s[ch_short_lhs], conv7_b[ch_short_lhs] = 2.0, -11.0

    conv7_w[1, [HC_ID, HC_DOT], ch_multi_dig] = -1
    conv7_w[2, [HC_DIG, HC_COMM, HC_STR], ch_multi_dig] = [1, -1, -1]
    conv7_w[3, HC_DIG, ch_multi_dig] = 1
    conv7_s[ch_multi_dig], conv7_b[ch_multi_dig] = 2.0, -3.0

    conv7_w[1, HC_LB, ch_magic_idx] = 1
    conv7_w[2, [HC_DIG, HC_COMM, HC_STR], ch_magic_idx] = [1, -1, -1]
    conv7_w[3, HC_DIG, ch_magic_idx] = 1
    conv7_s[ch_magic_idx], conv7_b[ch_magic_idx] = 2.0, -5.0
    feat["short_lhs"] = off_c7 + ch_short_lhs
    feat["multi_digit"] = off_c7 + ch_multi_dig
    feat["magic_index"] = off_c7 + ch_magic_idx

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

    # Declarative line-level rules compiled into strictly {-1, 0, +1} lmlp1_w & lmlp2_w
    module_headers = [
        ("from ", -3.0), ("use ", -3.0), ("import ", -3.0),
        ("package ", -3.0), ("#include", -3.0),
    ]
    hardcoded_branch_terms = [
        ("if", 1.5), ("return", 1.0), ("== '", 1.5), ('== "', 1.5),
        ("===_digit", 1.5), ("== -1", 1.5), ("||", -2.5), ("typeof", -2.5),
        ("type(", -2.5), (".code", -3.0), ("=== undefined", -3.0),
    ]
    ceremony_terms = [
        ("Manager", 1.0), ("Factory", 1.2), ("Service", 0.9), ("Builder", 0.8),
        ("Wrapper", 1.0), ("Helper", 0.9), ("Impl", 1.1), ("Abstract", 1.3),
        ("Repository", 1.1), ("Configuration", 0.8), ("Controller", 0.9),
        ("Strategy", 1.0), ("Provider", 0.9), ("Processor", 0.9),
    ]
    stdlib_idiom_terms = [
        ("Prototype", 12.0), ("ERR_", 10.0), ("SAFE_IN", 10.0),
        ("HideStackFram", 12.0), ("primordials", 8.0), ("EventEmitter", 12.0),
        ("assert(", 8.0),
    ]
    stmt_terms = [
        ("semi_or_eq", 1.0), ("func_call", 1.0), ("return", 1.0),
        ("throw", 1.0), ("raise", 1.0),
    ] + module_headers
    imperative_terms = [
        ("code_chars", 0.05), ("semi_or_eq", 0.65), ("func_call", 0.65),
        ("indent_cols", 0.25), ("ERR_", -3.0), ("Prototype", -3.0),
        ("require", -3.0), ("exports", -3.0), ("return", -0.8),
        ("throw", -0.8), ("raise", -0.8),
    ] + module_headers

    short_golf_terms = [("short_lhs", 1.5), ("one_char_id", 0.5)]
    magic_num_terms = [("magic_index", 1.0), ("multi_digit", 0.30)]
    line_rules: list[tuple[list[tuple[str, float]], float, int, float]] = [
        ([("has_nonspace", 1.0)], 0.0, Z_ACTIVE, 1.0),
        ([("code_chars", 1.0)], 0.0, Z_HAS_CODE, 1.0),
        ([("code_chars", 1.0)], -1.0, Z_HAS_CODE, -1.0),
        ([("comment_chars", 1.0), ("code_chars", -5.0)], -2.0, Z_PURE_COMMENT, 1.0),
        ([("comment_chars", 1.0), ("code_chars", -5.0)], -3.0, Z_PURE_COMMENT, -1.0),
        ([("comment_chars", -0.25), ("code_chars", -5.0)], 11.0, Z_SHORT_COMMENT, 1.0),
        ([("comment_chars", -0.25), ("code_chars", -5.0)], 10.0, Z_SHORT_COMMENT, -1.0),

        ([("comment_mask", 1.0), ("code_chars", -5.0)], -2.0, Z_PARAGRAPH_BREAK, 1.0),
        ([("comment_mask", 1.0), ("code_chars", -5.0)], -3.0, Z_PARAGRAPH_BREAK, -1.0),
        ([("has_nonspace", -1.0)], 1.0, Z_PARAGRAPH_BREAK, 1.0),
        ([("open_braces", 1.0), ("close_braces", 1.0), ("code_chars", -0.4)], 0.2, Z_OPEN_NET, 1.0),
        ([("open_braces", 1.0), ("close_braces", 1.0), ("code_chars", -0.4)], -0.8, Z_OPEN_NET, -1.0),

        ([("indent_cols", 1.0)], 0.0, Z_INDENT, 1.0),
        ([("depth_ge8", 1.0)], 0.0, Z_DEPTH8, 1.0),
        ([("depth_ge10", 1.0)], 0.0, Z_DEPTH10, 1.0),
        ([("depth_ge12", 1.0)], 0.0, Z_DEPTH12, 1.0),
        ([("odd_indent", 1.0)], 0.0, Z_ODD_INDENT, 1.0),
        ([("mod4_eq2_indent", 1.0)], 0.0, Z_MOD4_EQ2, 1.0),
        ([("mod4_eq0_indent", 1.0)], 0.0, Z_MOD4_EQ0, 1.0),
        ([("depth_ge16", 0.5), ("depth_ge20", 1.2)], 0.0, Z_DEEP_PYRAMID, 1.0),
        ([(name, 1.0) for name, _ in module_headers], 0.0, Z_MODULE_HEADER, 1.0),

        ([("if", 1.0), ("else", 0.8), ("for", 0.8), ("while", 0.8),
          ("elif", 0.9), ("else if", 0.5), ("match", 0.8)], 0.0, Z_BRANCH_OPEN, 1.0),
        ([("else", 1.0), ("elif", 0.8)], 0.0, Z_ELSE_BRANCH, 1.0),
        ([("return", 1.0), ("throw", 1.2), ("raise", 1.2), ("break", 0.8),
          ("continue", 0.8), ("depth_ge16", -2.5)], 0.0, Z_EARLY_EXIT, 1.0),
        ([("if", 1.0), ("return", 1.0), ("throw", 1.0),
          ("raise", 1.0), ("depth_ge16", -2.5)], -1.5, Z_GUARD_CLAUSE, 2.0),
        (hardcoded_branch_terms, -3.2, Z_HARDCODED_CASE, 1.0),
        (hardcoded_branch_terms, -4.2, Z_HARDCODED_CASE, -1.0),
        ([("switch", 2.0), ("match", 2.0), ("case ", 1.5), ("break", 1.5)], 0.0, Z_SWITCH_BLOCK, 1.0),

        (ceremony_terms, 0.0, Z_CEREMONY, 1.0),
        (stdlib_idiom_terms, 0.0, Z_STDLIB_IDIOM, 1.0),
        ([("ident_13ch", 0.11)] + [(key, -wt * 0.25) for key, wt in stdlib_idiom_terms],
         -1.05, Z_LONG_IDENT, 1.0),
        ([("one_char_id", 1.0)], 0.0, Z_ONE_CHAR_ID, 1.0),
        ([("three_char_id", 0.25)], 0.0, Z_WORD_COUNT, 0.9),

        ([("catch (", 1.0), ("except:", 1.2), ("except Excep", 1.2), ("except BaseE", 1.2)],
         0.0, Z_TRY_CATCH, 1.0),
        ([("catch", 1.0), ("{}", 1.0)], -1.5, Z_TRY_CATCH, 2.0),
        ([("throw", 1.5), ("raise", 1.5), (".code", 1.5), ("reject(", 1.5),
          ("cb(", 1.5), ("emit(", 1.0), ("destroy", 1.2), ("callback(", 1.5)],
         0.0, Z_REAL_HANDLER, 1.0),
        ([("{}", 1.5), ("console", 1.2), ("null;", 1.4), ("false;", 1.2), ("pass", 1.4)],
         0.0, Z_EMPTY_HANDLER, 1.0),
        ([("return", 1.0), ("None", 1.0), ("0.0", 1.0)], -1.5, Z_EMPTY_HANDLER, 1.8),
        ([("|| {}", 1.4), ("|| []", 1.4), ("|| '')", 1.0), (" or []", 1.4),
          (" or {}", 1.4), ("type: i", 1.8), ("noqa", 1.6), ("nolint", 1.6),
          ("#[allow(", 0.7), ("eslint-disabl", 2.0), ("@ts-ignore", 2.0), ("?.", 0.45)],
         -0.95, Z_FALLBACK_SWALLOW, 1.2),

        ([(".map(", 1.0), (".filter", 1.0), (".reduce", 1.0), (".flatMa", 1.0), (".unwrap", 0.7)],
         -1.2, Z_CHAINED_CALLS, 1.2),
        ([("ternary_op", 1.0)], -1.2, Z_NESTED_TERNARY, 1.5),
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
    ]

    free_narrow = [slot for slot in range(n_conv3) if not np.any(conv3_w[:, :, slot])]
    free_medium = [slot for slot in range(n_conv7) if not np.any(conv7_w[:, :, slot])]
    free_wide = [slot for slot in range(n_conv13) if not np.any(conv13_w[:, :, slot])]
    scaled_cache: dict[tuple[int, float], int] = {}

    def rescale_conv_in_place(base_idx: int, mult: float) -> None:
        if base_idx < off_c7:
            conv3_s[base_idx - off_c3] *= mult
            conv3_b[base_idx - off_c3] *= mult
        elif base_idx < off_c13:
            conv7_s[base_idx - off_c7] *= mult
            conv7_b[base_idx - off_c7] *= mult
        else:
            conv13_s[base_idx - off_c13] *= mult
            conv13_b[base_idx - off_c13] *= mult

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
        elif base_idx < off_c7:
            src = base_idx - off_c3
            dst = free_narrow.pop(0)
            conv3_w[:, :, dst] = conv3_w[:, :, src]
            conv3_s[dst] = conv3_s[src] * mult
            conv3_b[dst] = conv3_b[src] * mult
            out_idx = off_c3 + dst
        elif base_idx < off_c13:
            src = base_idx - off_c7
            if free_medium:
                dst = free_medium.pop(0)
                conv7_w[:, :, dst] = conv7_w[:, :, src]
                conv7_s[dst] = conv7_s[src] * mult
                conv7_b[dst] = conv7_b[src] * mult
                out_idx = off_c7 + dst
            else:
                dst = free_wide.pop(0)
                conv13_w[3:10, :, dst] = conv7_w[:, :, src]
                conv13_s[dst] = conv7_s[src] * mult
                conv13_b[dst] = conv7_b[src] * mult
                out_idx = off_c13 + dst
        else:
            src = base_idx - off_c13
            dst = free_wide.pop(0)
            conv13_w[:, :, dst] = conv13_w[:, :, src]
            conv13_s[dst] = conv13_s[src] * mult
            conv13_b[dst] = conv13_b[src] * mult
            out_idx = off_c13 + dst
        scaled_cache[cache_key] = out_idx
        return out_idx

    num_u = n_csum + n_conv3 + n_conv7 + n_conv13
    lm1_w = np.zeros((num_u, 96), dtype=np.int8)
    lm1_s = np.ones((96,), dtype=np.float32)
    lm1_b = np.zeros((96,), dtype=np.float32)
    lm2_w = np.zeros((96, 112), dtype=np.int8)
    lm2_b = np.zeros((112,), dtype=np.float32)

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

    # Emit helper scaled Z[87..110] registers for exact {-1, 0, +1} cmlp1_w & doc1_w
    active_units = list(range(len(line_rules)))
    next_unit = len(line_rules)
    next_z = 87
    z_helpers: dict[str, int] = {}
    helper_specs = (
        ("has_code_08", Z_HAS_CODE, 0.8),
        ("d12_22", Z_DEPTH12, 2.2),
        ("d10_12", Z_DEPTH10, 1.2),
        ("open_24", Z_OPEN_NET, 2.4),
        ("pyr_22", Z_DEEP_PYRAMID, 2.2),
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
        ("ee_06", Z_EARLY_EXIT, 0.6),
        ("gc_07", Z_GUARD_CLAUSE, 0.7),
        ("ci_06", Z_CLEAN_IDIOM, 0.6),
        ("m42_035", Z_MOD4_EQ2, 0.35),
        ("wc_025", Z_WORD_COUNT, 0.25),
        ("m42_doc", Z_MOD4_EQ2, 0.4 / 1.4),
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
    trit_mats["lmlp1_w"], vecs_1d["lmlp1_s"], vecs_1d["lmlp1_b"] = lm1_w, lm1_s, lm1_b
    trit_mats["lmlp2_w"], vecs_1d["lmlp2_b"] = lm2_w, lm2_b
    return z_helpers


def _build_block2_and_heads(
    trit_mats: dict[str, np.ndarray],
    vecs_1d: dict[str, np.ndarray],
    zh: dict[str, int],
) -> None:
    """Compile {-1, 0, +1} sink-attention heads, [0, 1] issue clamps, and System One readout."""
    center = MAX_REL_DIST
    num_rel = 2 * center + 1
    a1_wq = np.zeros((N_Z, 128), dtype=np.int8)
    a1_wq_s = np.ones((128,), dtype=np.float32)
    a1_wk = np.zeros((N_Z, 128), dtype=np.int8)
    a1_wk_s = np.ones((128,), dtype=np.float32)
    a1_wv = np.zeros((N_Z, 96), dtype=np.int8)
    a1_wv_s = np.ones((96,), dtype=np.float32)
    a1_wo = np.zeros((96, N_Z), dtype=np.int8)
    a1_br = np.full((8, num_rel), -1e4, dtype=np.float32)
    a1_bs = np.array([4.5, 1.2, 1.5, 2.5, 9.05, 11.1, -12.0, -12.0], dtype=np.float32)

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

    # Head 1: try/catch/except opener matching real vs. empty/fallback handler on L..L+3
    h1_qk = [16, 17]
    a1_wq[Z_TRY_CATCH, h1_qk] = 1
    a1_wk[[Z_REAL_HANDLER, Z_EMPTY_HANDLER], h1_qk] = 1
    a1_wq_s[h1_qk] = a1_wk_s[h1_qk] = [float(np.sqrt(32.0)), float(np.sqrt(16.0))]
    a1_br[1, center : center + 4] = 0.0

    # Head 2 (L-8..L), Head 3 (L-9..L+9), Head 6 (file-wide), and Head 7 (L-3..L+3)
    a1_br[2, center - 8 : center + 1] = 0.0
    a1_br[3, center - 9 : center + 10] = 0.0
    a1_br[6, :] = 0.0
    a1_br[7, center - 3 : center + 4] = 0.0

    h2_srcs = [
        Z_DEPTH10, Z_DEPTH12, Z_BRANCH_OPEN, Z_EARLY_EXIT,
        Z_GUARD_CLAUSE, Z_CLEAN_IDIOM, Z_DEEP_PYRAMID,
    ]
    h2_slots = list(range(24, 31))
    a1_wv[h2_srcs, h2_slots] = 1
    a1_wv_s[slice(24, 31)] = [0.8, 1.5, 1.0, 1.0, 0.8, 0.8, 1.0]
    a1_wo[[24, 25], Z_ENCLOSING_DEPTH] = 1
    a1_wo[26, Z_ENCLOSING_BRANCHES] = 1
    a1_wo[[27, 28, 29], Z_ENCLOSING_GUARDS] = 1
    a1_wo[30, Z_ENCLOSING_DEEP16] = 1

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
    a1_wv_s[30] *= 1.6 / 0.5
    a1_wv_s[[36, 37]] *= [14.2, 18.0]
    a1_wv_s[spec_slots] *= 8.5 / 1.5
    a1_wv_s[[72, 73]] *= [1.5 / 0.4, 2.0 / 0.4]
    a1_wv_s[75] *= 1.5
    a1_wv_s[85] *= 0.65 / 0.85
    a1_wv_s[86] *= 1.8 / 0.6
    a1_wv_s[87] *= 2.2 / 0.9

    extra_f_slots = [77, 78, 79]
    a1_wv[[Z_MOD4_EQ2, Z_ODD_INDENT, Z_ODD_INDENT], extra_f_slots] = 1
    a1_wv_s[extra_f_slots] = [1.0, 1.0, 7.5]
    a1_wo[extra_f_slots, [Z_FMOD42_DUP, Z_FODD_DUP, Z_FODD_SCALED]] = 1

    trit_mats["attn1_wq"], vecs_1d["attn1_wq_s"] = a1_wq, a1_wq_s
    trit_mats["attn1_wk"], vecs_1d["attn1_wk_s"] = a1_wk, a1_wk_s
    trit_mats["attn1_wv"], vecs_1d["attn1_wv_s"] = a1_wv, a1_wv_s
    trit_mats["attn1_wo"] = a1_wo
    vecs_1d["attn1_brel"], vecs_1d["attn1_bsink"] = a1_br, a1_bs

    # Context MLP 1: paired {-1, 0, +1} ReLU clamps writing [0, 1] issue flags
    cm1_w = np.zeros((N_Z, 96), dtype=np.int8)
    cm1_s = np.ones((96,), dtype=np.float32)
    cm1_b = np.zeros((96,), dtype=np.float32)
    cm1_2w = np.zeros((96, N_Z), dtype=np.int8)
    cm1_2b = np.zeros((N_Z,), dtype=np.float32)

    wide_dsts = [Z_USES_WIDE_INDENT, Z_WIDE_DUP_A, Z_WIDE_DUP_B]
    ctx1_rules: list[tuple[list[tuple[int, int]], float, list[int], float, float]] = [
        ([(zh["has_code_08"], 1), (zh["d12_22"], 1), (zh["d10_12"], 1), (Z_ENCLOSING_BRANCHES, 1),
          (Z_ELSE_BRANCH, 1), (Z_ENCLOSING_GUARDS, -1), (zh["open_24"], -1)],
         -2.65, [Z_RAW_NESTING_2SP], 0.5, 2.5),
        ([(zh["has_code_08"], 1), (zh["pyr_22"], 1), (Z_ENCLOSING_DEEP16, 1),
          (Z_ENCLOSING_BRANCHES, 1), (Z_ELSE_BRANCH, 1),
          (Z_ENCLOSING_GUARDS, -1), (zh["open_24"], -1)],
         -3.25, [Z_ISS_NESTING], 0.5, 2.5),
        ([(Z_FILE_MOD4_0, 1), (Z_FILE_MOD4_2, -1), (Z_FMOD42_DUP, -1),
          (Z_FILE_ODD_IND, -1), (Z_FODD_DUP, -1)],
         -0.5, wide_dsts, 20.0, 2.0),
        ([(Z_FILE_ODD_IND, 1), (Z_FILE_MOD4_2, -1)], -4.0, wide_dsts, 20.0, 2.0),

        ([(Z_CEREMONY, 1), (zh["long_id_r4"], 1), (Z_LOCAL_CEREMONY, 1), (zh["stdlib_r4"], -1)],
         -1.05, [Z_ISS_NAMING], 0.85, 2.2),
        ([(zh["one_r5"], 1), (Z_FILE_ONE_CHAR, 1), (Z_FILE_WORDS, -1),
          (zh["words_r5"], -1), (Z_HAS_CODE, 1)],
         -1.45, [Z_RAW_CRYPTIC], 0.4, 2.2),
        ([(Z_PURE_COMMENT, 1), (Z_SHORT_COMMENT, 1), (Z_RESTATES_CODE, 1)],
         -3.35, [Z_ISS_COMMENT], 1.0, 2.5),
        ([(Z_NARRATOR_TAG, 1)], -0.45, [Z_ISS_COMMENT], 1.5, 2.2),
        ([(Z_TRY_CATCH, 1), (Z_NEAR_EMPTY_HANDLER, 1), (Z_NEAR_REAL_HANDLER, -1)],
         -2.1, [Z_ISS_ERROR], 1.8, 2.5),
        ([(Z_FALLBACK_SWALLOW, 1)], -0.35, [Z_ISS_ERROR], 1.4, 2.2),

        ([(zh["chain_12"], 1), (zh["tern_125"], 1), (zh["o96_11"], 1), (zh["o80_025"], 1),
          (zh["stdlib_09"], -1), (zh["modhdr_30"], -1)],
         -1.0, [Z_ISS_DENSE], 1.0, 2.2),
        ([(Z_HARDCODED_CASE, 1), (Z_LOCAL_SPECIAL_CASES, 1),
          (Z_SWITCH_BLOCK, -1), (zh["sw_dup"], -1)],
         -2.75, [Z_ISS_SPECIAL], 1.5, 2.5),
        ([(zh["imp_075"], 1), (Z_DISTANT_CLONE, 1), (Z_IS_INDENTED, 1), (Z_IS_STMT, 1)],
         -6.45, [Z_RAW_CLONE], 2.0, 2.5),
        ([(Z_IS_STMT, 1), (Z_LOCAL_STMT_DENSITY, 1), (Z_LOCAL_BREAK_DENSITY, -1)],
         -8.55, [Z_ISS_LAYOUT], 1.0, 3.0),
        ([(Z_IS_STMT, 1), (zh["odd_18"], 1), (Z_FODD_SCALED, -1)],
         -2.0, [Z_ISS_LAYOUT], 1.0, 2.5),
        ([(zh["ee_06"], 1), (zh["gc_07"], 1), (zh["ci_06"], 1),
          (zh["m42_035"], 1), (zh["wc_025"], 1)],
         -0.25, [Z_CLEAN_BONUS], 1.0, 1.2),
        ([(Z_SHORT_GOLF, 1), (Z_LOCAL_SHORT_GOLF, 1)], -1.35, [Z_ISS_NAMING], 0.6, 2.5),
        ([(Z_MAGIC_NUM, 1), (Z_LOCAL_MAGIC_NUM, 1)], -0.78, [Z_ISS_NAMING], 0.9, 2.5),
    ]

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

    # Pass 2 neighborhood smoothing attention & context MLP 2 in {-1, 0, +1}
    trit_mats["attn2_wq"] = np.zeros((N_Z, 64), dtype=np.int8)
    trit_mats["attn2_wk"] = np.zeros((N_Z, 64), dtype=np.int8)
    a2_wv = np.zeros((N_Z, 64), dtype=np.int8)
    a2_wv_s = np.ones((64,), dtype=np.float32)
    a2_wo = np.zeros((64, N_Z), dtype=np.int8)
    a2_br = np.full((4, num_rel), -1e4, dtype=np.float32)
    a2_br[0, center - 3 : center + 4] = 0.0
    a2_br[1, center - 2 : center + 3] = 0.0
    a2_br[2, center - 3 : center + 4] = 0.0
    a2_br[3, center - 2 : center + 3] = 0.0

    smooth_srcs = [
        Z_RAW_CLONE, Z_ISS_NESTING, Z_ISS_SPECIAL,
        Z_RAW_CRYPTIC, Z_ISS_LAYOUT, Z_RAW_NESTING_2SP,
    ]
    smooth_dsts = [
        Z_SMOOTH_CLONE, Z_SMOOTH_NEST, Z_SMOOTH_SPECIAL,
        Z_SMOOTH_CRYPTIC, Z_SMOOTH_LAYOUT, Z_SMOOTH_NEST_2SP,
    ]
    smooth_slots = [0, 16, 17, 32, 33, 48]
    a2_wv[smooth_srcs, smooth_slots] = 1
    a2_wo[smooth_slots, smooth_dsts] = 1
    a2_wv_s[smooth_slots] = [5.0 / 2.0, 1.2 / 4.2, 1.2 / 4.3, 3.0 / 1.6, 1.0 / 4.3, 1.0]

    trit_mats["attn2_wv"], vecs_1d["attn2_wv_s"], trit_mats["attn2_wo"] = a2_wv, a2_wv_s, a2_wo
    vecs_1d["attn2_brel"] = a2_br
    vecs_1d["attn2_bsink"] = np.array([0.7, -12.0, -12.0, -12.0], dtype=np.float32)

    cm2_w = np.zeros((N_Z, 64), dtype=np.int8)
    cm2_s = np.ones((64,), dtype=np.float32)
    cm2_b = np.zeros((64,), dtype=np.float32)
    cm2_2w = np.zeros((64, N_Z), dtype=np.int8)
    cm2_2b = np.zeros((N_Z,), dtype=np.float32)

    for pair_u in (0, 1):
        cm2_w[[Z_RAW_CLONE, Z_SMOOTH_CLONE], pair_u] = 1
    for pair_u in (2, 3):
        cm2_w[[Z_RAW_CRYPTIC, Z_SMOOTH_CRYPTIC], pair_u] = 1
    cm2_s[:4] = [2.0, 2.0, 1.6, 1.6]
    cm2_b[:4] = [-3.35, -4.35, -2.0, -3.0]
    cm2_2w[[0, 1], Z_ISS_CLONE] = [1, -1]
    cm2_2w[[2, 3], Z_ISS_NAMING] = [1, -1]

    cm2_w[Z_RAW_NESTING_2SP, [4, 5]] = 1
    cm2_w[Z_SMOOTH_NEST_2SP, [6, 7]] = 1
    for wide_u in range(4, 8):
        cm2_w[wide_dsts, wide_u] = -1
    nest_sm_ratio = 1.2 / 4.2
    cm2_s[[6, 7]] = nest_sm_ratio
    cm2_b[4:8] = [0.0, -1.0, 0.0, -nest_sm_ratio]
    cm2_2w[[4, 5], Z_ISS_NESTING] = [1, -1]
    cm2_2w[[6, 7], Z_SMOOTH_NEST] = [1, -1]
    trit_mats["cmlp2_w"], vecs_1d["cmlp2_s"], vecs_1d["cmlp2_b"] = cm2_w, cm2_s, cm2_b
    trit_mats["cmlp2_2_w"], vecs_1d["cmlp2_2_b"] = cm2_2w, cm2_2b

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
    pool_gate_w[Z_ACTIVE, 0] = 1
    trit_mats["pool_gate_w"] = pool_gate_w
    vecs_1d["pool_gate_s"] = np.array([12.0], dtype=np.float32)
    vecs_1d["pool_gate_b"] = np.array([-6.0], dtype=np.float32)

    # Build target readout weights (9 logical rows -> 31 logits), then decompose into trits
    target_doc2 = np.zeros((9, 31), dtype=np.float32)
    doc2_b = np.zeros((31,), dtype=np.float32)
    doc2_b[:2] = [1.45, 1.85]
    target_doc2[:8, 0] = [-2.1, -2.1, -1.9, -2.2, -2.2, -2.2, -2.1, -2.0]
    target_doc2[:8, 1] = [-1.2, -1.0, -2.2, -2.5, -1.1, -2.6, -2.3, -1.3]
    target_doc2[8, :2] = [0.85, 0.45]

    subscore_heads = (
        [(0, 1.25)],
        [(1, 1.25)],
        [(4, 1.2), (5, 0.9), (6, 0.9)],
        [(2, 1.35)],
        [(3, 1.35), (5, 0.6)],
        [(7, 1.25), (0, 0.35)],
    )
    for score_i, units in enumerate(subscore_heads):
        base = 2 + 4 * score_i
        doc2_b[base : base + 4] = [-2.3, -0.9, 0.6, 2.0]
        for unit_i, mult in units:
            target_doc2[unit_i, base : base + 4] += [
                2.1 * mult, 0.8 * mult, -0.7 * mult, -2.2 * mult,
            ]
        target_doc2[8, [base, base + 3]] += [-0.45, 0.45]

    style_base = 26
    doc2_b[style_base : style_base + 5] = [1.25, 1.35, -2.10, -2.10, -2.20]
    target_doc2[:8, style_base] -= 1.50
    target_doc2[:8, style_base + 1] -= 1.25
    target_doc2[8, [style_base, style_base + 1]] += [0.5, 0.3]
    target_doc2[[4, 0], style_base + 3] += [3.3, 1.1]
    target_doc2[[2, 3, 5, 6, 7, 0], style_base + 4] += [2.9, 3.1, 3.3, 3.0, 2.2, 1.1]

    doc1_w = np.zeros((2 * N_Z, 64), dtype=np.int8)
    doc1_s = np.ones((64,), dtype=np.float32)
    doc1_b = np.zeros((64,), dtype=np.float32)
    doc2_w = np.zeros((64, 31), dtype=np.int8)

    step_issue = 0.26
    for head_i, reg in enumerate(issue_regs):
        mean_slot = 6 * head_i
        max_slot = mean_slot + 3
        for trit_i, pow3 in enumerate((1.0, 3.0, 9.0)):
            doc1_w[reg, mean_slot + trit_i] = 1
            doc1_s[mean_slot + trit_i] = 4.5 * step_issue * pow3
            doc1_b[mean_slot + trit_i] = -0.03 * step_issue * pow3

            doc1_w[N_Z + reg, max_slot + trit_i] = 1
            doc1_s[max_slot + trit_i] = 0.65 * step_issue * pow3
            doc1_b[max_slot + trit_i] = -0.17 * step_issue * pow3

        for col_j in range(31):
            w_val = float(target_doc2[head_i, col_j])
            if abs(w_val) <= 1e-6:
                continue
            trits = _balanced_trits(w_val, step_issue, num_trits=3)
            doc2_w[mean_slot : mean_slot + 3, col_j] = trits
            doc2_w[max_slot : max_slot + 3, col_j] = trits

    bonus_base = 48
    step_bonus = 0.05
    for trit_i, pow3 in enumerate((1.0, 3.0, 9.0, 27.0)):
        slot = bonus_base + trit_i
        doc1_w[[Z_CLEAN_BONUS, zh["m42_doc"]], slot] = 1
        doc1_s[slot] = 1.4 * step_bonus * pow3
    for col_j in range(31):
        w_val = float(target_doc2[8, col_j])
        if abs(w_val) <= 1e-6:
            continue
        doc2_w[bonus_base : bonus_base + 4, col_j] = _balanced_trits(w_val, step_bonus, num_trits=4)

    style_exact = (
        ([Z_RAW_CRYPTIC], 3.5 * 3.3, style_base + 3),
        ([N_Z + Z_RAW_CRYPTIC], 0.6 * 3.3, style_base + 3),
        ([Z_CEREMONY], 2.0 * 3.5, style_base + 2),
        ([Z_LONG_IDENT], 1.5 * 3.5, style_base + 2),
        ([Z_CLEAN_IDIOM], 2.0 * 1.3, style_base),
        ([Z_EARLY_EXIT, Z_GUARD_CLAUSE], 1.2 * 1.3, style_base),
    )
    exact_base = 52
    for offset, (srcs, sc_val, dst_col) in enumerate(style_exact):
        slot = exact_base + offset
        doc1_w[srcs, slot] = 1
        doc1_s[slot] = sc_val
        doc2_w[slot, dst_col] = 1

    trit_mats["doc1_w"], vecs_1d["doc1_s"], vecs_1d["doc1_b"] = doc1_w, doc1_s, doc1_b
    trit_mats["doc2_w"], vecs_1d["doc2_b"] = doc2_w, doc2_b


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
    size_kb = out_file.stat().st_size / 1024
    num_trits = sum(arr.size for arr in trits_raw.values())
    num_vecs = sum(arr.size for arr in vecs_raw.values())
    print(
        f"Compiled {len(trits_raw)} ternary matrices ({num_trits:,} trits in {{-1,0,+1}}, "
        f"{num_vecs:,} scale/bias scalars, {size_kb:.2f} KiB)"
    )
