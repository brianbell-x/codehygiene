"""codehygiene v0.7.0 — Zero-vocabulary 1.58-bit ternary weight-space compiler for weights.npz.

Every 2D/3D weight, convolution, projection, and embedding tensor has entries strictly in
{-1, 0, +1} (packed at 5 trits per uint8 byte), paired only with 1D channel scales/biases.
Zero alphabetic or numeric substring filters exist anywhere in the network: letters and digits
enter only uniform character-class masks and symmetric cross-line SimHash self-similarity.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from codehygiene import MAX_COLS, MAX_REL_DIST, score, unpack_ternary_archive

_BITS5 = np.arange(5)
_POW3 = 3 ** np.arange(5, dtype=np.uint8)
_LO_A, _LO_Z = 97, 122
_UP_A, _UP_Z = 65, 90
_DIG_0, _DIG_9 = 48, 57

# Per-cell feature channels emitted by Block 0 (68 channels; synchronized with engine.js)
(
    HC_ACT, HC_COMM, HC_CODE, HC_CNS, HC_MNS, HC_IND, HC_FNS, HC_ODD,
    HC_M42, HC_M40, HC_D8, HC_D10, HC_D12, HC_G96, HC_G110, HC_ID,
    HC_LO, HC_UP, HC_DIG, HC_DOT, HC_QM, HC_NOT_WORD, HC_EQ,
    HC_LB, HC_RB, HC_LP, HC_RP, HC_SEM, HC_SK_VETO, HC_QT, HC_SL_BOOST,
) = range(31)
HC_BP = 31
HC_BN = 36
(
    HC_LBRK, HC_RBRK, HC_COM, HC_COL, HC_SL_C, HC_STR, HC_LEAD,
    HC_D16, HC_D20, HC_D24, HC_RAW_STR, HC_SPACE, HC_BANG, HC_LT,
    HC_GT, HC_PLUS, HC_MINUS, HC_STAR, HC_SLASH, HC_PCT, HC_IN_GRP,
    HC_DQ, HC_SQ, HC_BQ, HC_AMP, HC_PIPE, HC_DEEP_GRP,
) = range(41, 68)
N_HC = 68
_SH_HC_COLS = [*range(HC_BP, HC_BP + 5), *range(HC_BN, HC_BN + 5)]

# Byte and column embedding layout inside Block 0 (52 channels)
EMB_ACTIVE, EMB_NONSPACE, EMB_SPACE = 0, 1, 2
EMB_LOWER, EMB_UPPER, EMB_DIGIT, EMB_IDENT, EMB_UND = 3, 4, 5, 6, 7
(
    EMB_SLASH, EMB_STAR, EMB_COLON, EMB_QMARK, EMB_DOT, EMB_EQ,
    EMB_LBRACE, EMB_RBRACE, EMB_LPAREN, EMB_RPAREN, EMB_SEMI,
    EMB_AMP, EMB_PIPE, EMB_ANY_QUOTE, EMB_DQUOTE, EMB_SQUOTE,
    EMB_BQUOTE, EMB_BACKSLASH, EMB_HASH, EMB_DASH, EMB_BANG,
    EMB_PERCENT, EMB_LBRK, EMB_RBRK, EMB_COMMA, EMB_PLUS,
    EMB_LT, EMB_GT, EMB_LIFE_PFX, EMB_ANGLE,
) = range(8, 38)
(
    EMB_COL_ODD, EMB_COL_MOD4_2, EMB_COL_IND4, EMB_COL_GE8,
    EMB_COL_GE10, EMB_COL_GE12, EMB_COL_GE16, EMB_COL_GE20, EMB_COL_GE24,
) = range(38, 47)
EMB_SH_BITS = 47
N_EMB = 52

# Block 0 lexical pre-convolution, causal prefixes, and quote/comment/group gates
(
    PRE_SLASH_COMM, PRE_BLOCK_OPEN, PRE_BLOCK_CLOSE, PRE_HASH_COMM,
    PRE_DASH_COMM, PRE_DQUOTE, PRE_SQUOTE, PRE_APOSTROPHE, PRE_HASH2_COMM,
    PRE_BANG_DOC, PRE_BANG_LEAD, PRE_SP_BANG2, PRE_BANG2_CONT, PRE_BANG2_ABORT,
    PRE_VB_COMM, PRE_ZIG_RAW, PRE_BQUOTE, PRE_QUOTED_BQUOTE, PRE_EOL,
) = range(19)
N_PRE = 19
(
    C1_NONSPACE, C1_DQUOTES, C1_SQUOTES, C1_BLOCK_OPENS,
    C1_BQUOTES, C1_SP_BANG2, C1_MID_COMM, C1_IN_BCOMM, C1_GROUP_DEPTH,
) = range(9)
N_PREF1 = 9

N_QSAW = 32
G1_INDENT_SPACE, G1_FIRST_NONSPACE, G1_DOC_STAR, G1_ANY_COMM_START = 0, 1, 2, 3
G1_CLOSE_HI, G1_INDENT_BASE = 4, 5
G1_LINE_COMM_START, G1_LEAD_TWO, G1_LEAD_ONE, G1_CLOSE_LO = 14, 15, 16, 17
G1_DQUOTE_SAW = 18
G1_SQUOTE_SAW = G1_DQUOTE_SAW + N_QSAW
G1_BQUOTE_SAW = G1_SQUOTE_SAW + N_QSAW
(
    G1_BANG_LEAD, G1_BANG2_COMM, G1_PCT_COMM, G1_VB_COMM, G1_ZIG_RAW,
    G1_ACTIVE, G1_ACT_B, G1_MC_HI, G1_MC_LO, G1_BC_HI, G1_BC_LO,
    G1_B2_HI, G1_B2_LO, G1_GRP_HI, G1_GRP_LO, G1_DEEP_GRP,
) = range(G1_BQUOTE_SAW + N_QSAW, G1_BQUOTE_SAW + N_QSAW + 16)
N_GATE1 = G1_DEEP_GRP + 1
(
    C2_IN_COMMENT, C2_IN_LINE_COMM, C2_IN_COMM_B, C2_IN_COMM_C,
    C2_IN_RAW_LINE, C2_EFF_LEN, C2_LEAD_VETO,
) = range(7)
N_PREF2 = 7

# Cross-line delimiter sawtooth, Dyck group balance, and step-in contour dimensions
N_DQ_SAW = 384
N_SQ_SAW = 128
N_ML_GATE = N_DQ_SAW + N_SQ_SAW + 2
N_PAREN_STEPS = 32
N_PAREN_SUF = 2 * N_PAREN_STEPS
N_EXTRA_LINE_CONV = 4
N_ML_IN = 8

# Named geometric residual stream Z registers (89 line stats + 120 SimHash dims = 209 channels)
(
    Z_ACTIVE, Z_HAS_CODE, Z_PURE_COMMENT, Z_SHORT_COMMENT, Z_PARAGRAPH_BREAK,
    Z_INDENT, Z_DEPTH10, Z_DEPTH12, Z_DEPTH20, Z_DEPTH24,
    Z_ODD_INDENT, Z_MOD4_EQ2, Z_MOD4_EQ0, Z_SHALLOW_LINE, Z_SHALLOW_2SP, Z_IS_INDENTED,
    Z_AT_COL0_BLOCK, Z_SIG_LINE, Z_CHAIN_CALLS, Z_NESTED_TERNARY, Z_PACKED_STMTS,
    Z_OVER_96, Z_OVER_110, Z_DEEP_INLINE_GRP,
    Z_IS_STMT, Z_IS_DATA_ROW, Z_CLOSE_DELIM_ONLY, Z_EQ_LADDER_RUNG, Z_GUARD_LINE,
    Z_CLONE_ELIGIBLE, Z_SKEL_ELIGIBLE, Z_ELISION_COMM, Z_COMM_FENCE,
    Z_DOC_OR_CODE_VETO, Z_COMM_CODE_CUE, Z_MUT_LINE, Z_BLOCK_OPEN_LINE,
    Z_INDENTED_STMT, Z_PROSE_COMMENT,
) = range(39)
Z_IN_DOC, Z_IN_PAREN, Z_PREV_COMMENT, Z_STEP_IN = 39, 40, 41, 42
(
    Z_ECHO_SIM, Z_NEAR_FENCE, Z_LOCAL_COMM_CODE, Z_ADJ_COMMENT, Z_LOCAL_DATA_ROWS,
    Z_LOCAL_DEEP20, Z_LOCAL_DEEP24, Z_LOCAL_DEEP2SP, Z_STAIRCASE_STEPS,
    Z_LOCAL_STMT_DENS, Z_LOCAL_BREAK_DENS, Z_LOCAL_ODD_IND, Z_LOCAL_BRANCH_DENS,
    Z_LOCAL_LADDER, Z_LOCAL_PROSE, Z_DISTANT_CODE_CLONE, Z_DISTANT_SKEL_CLONE,
    Z_FILE_MOD4_0, Z_FILE_MOD4_2, Z_FILE_ODD, Z_FILE_CONTAINERS,
) = range(43, 64)
(
    Z_HAS_CONTAINER, Z_USES_2SP, Z_STRICT_EVEN_IND,
    Z_RAW_NEST_4SP, Z_RAW_NEST_2SP, Z_RAW_LAYOUT_WALL, Z_RAW_ODD_GLITCH,
    Z_RAW_CODE_CLONE, Z_RAW_SKEL_CLONE, Z_RAW_LADDER,
    Z_RAW_COMM_ECHO, Z_RAW_COMM_CODE, Z_ISS_DENSE,
) = range(64, 77)
(
    Z_SMOOTH_CODE_CLONE, Z_SMOOTH_SKEL_CLONE, Z_SMOOTH_NEST_CLS,
    Z_SMOOTH_NEST_2SP, Z_SMOOTH_LADDER, Z_SMOOTH_WALL, Z_SMOOTH_COMM_CODE,
) = range(77, 84)
(
    Z_ISS_NESTING, Z_ISS_LAYOUT, Z_ISS_CLONE, Z_ISS_LADDER, Z_ISS_COMMENT,
) = range(84, 89)
N_LMLP2 = 89
SKETCH_DIM = 40
N_Z = N_LMLP2 + 3 * SKETCH_DIM
Z_CODE_SKETCH = N_LMLP2
Z_COMM_SKETCH = Z_CODE_SKETCH + SKETCH_DIM
Z_SKEL_SKETCH = Z_COMM_SKETCH + SKETCH_DIM

FeatSlot = tuple[int, int]
LineRule = tuple[list[tuple[str, float]], float, int, float]
BankBuffer = tuple[
    np.ndarray | None,
    np.ndarray,
    np.ndarray | None,
    np.ndarray,
    np.ndarray | None,
    list[int],
]

# Head 0 exact 40-dim comment-echo cosine gate
ECHO_COS_GAIN, ECHO_COS_THRESH = 22.0, 0.79
N_CODE_PROJ = 36


def _pack_trits(arr: np.ndarray) -> np.ndarray:
    """Pack {-1, 0, +1} array at 5 trits per uint8 byte (mapping 0 -> 0x00)."""
    flat = (arr.ravel() % 3).astype(np.uint8)
    rem = (-len(flat)) % 5
    if rem:
        flat = np.pad(flat, (0, rem))
    return (flat.reshape(-1, 5) * _POW3).sum(axis=1).astype(np.uint8)


def _punct_filter(
    kernel: np.ndarray,
    scale: np.ndarray,
    bias: np.ndarray,
    channel: int,
    pos_specs: list[tuple[int, int]],
    neg_specs: list[tuple[int, int]] | None = None,
    *,
    code: bool = True,
    comm: bool = False,
    no_str: bool = True,
    out_scale: float = 1.0,
) -> None:
    """Stamp a {-1, 0, +1} character-class / punctuation motif without touching SimHash bits."""
    kernel[:, :, channel] = 0
    for pos, hc_col in pos_specs:
        kernel[pos, hc_col, channel] = 1
    for pos, hc_col in neg_specs or []:
        kernel[pos, hc_col, channel] = -1

    mid = kernel.shape[0] // 2
    thresh = len(pos_specs)
    if code:
        mask_cols = [HC_COMM, HC_STR] if no_str else [HC_COMM]
        for mc in mask_cols:
            kernel[mid, mc, channel] = -1
    if comm:
        kernel[mid, HC_SL_C, channel] = 1
        kernel[mid, HC_CODE, channel] = -1
        thresh += 1

    scale[channel] = 2.0 * out_scale
    bias[channel] = -(2.0 * thresh - 1.0) * out_scale


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
    perm = rng.permutation(SKETCH_DIM)
    col_perm = rng.permutation(20) + 20
    return had40[perm][:, col_perm[:16]]


def _fit_sketch_column(
    col: int, trits: np.ndarray, scales: np.ndarray, gram_target: np.ndarray
) -> None:
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
        diag_err = (cand_vals * cand_vals - res_gram[row, row]) ** 2
        errs = 2.0 * np.sum(diffs * diffs, axis=0) + diag_err
        best_trit = cand_trits[int(np.argmin(errs))]
        trits[row, col] = best_trit
        cur_vec[row] = best_trit * col_s

    t_col = trits[:, col].astype(np.float32)
    norm_sq = float(np.dot(t_col, t_col))
    quad_num = float(t_col @ res_gram @ t_col) if norm_sq > 0 else 0.0
    if quad_num > 0:
        scales[col] = math.sqrt(quad_num) / norm_sq


def _ternary_sketch_proj(n_cols: int = N_CODE_PROJ) -> tuple[np.ndarray, np.ndarray]:
    """Compile {-1, 0, +1} (40, n_cols) QR projection matching target Gram geometry."""
    qr_mat, _ = np.linalg.qr(np.random.RandomState(777).randn(SKETCH_DIM, SKETCH_DIM))
    target = (qr_mat[:, :n_cols] * np.sqrt(SKETCH_DIM / float(n_cols))).astype(np.float32)
    gram_target = target @ target.T

    col_std = np.sqrt(np.mean(target * target, axis=0, keepdims=True))
    trits = np.zeros((SKETCH_DIM, n_cols), dtype=np.int8)
    trits[target > 0.61 * col_std] = 1
    trits[target < -0.61 * col_std] = -1
    scales = np.array([
        float(np.sum(target[:, col] * trits[:, col]) / max(int(np.sum(trits[:, col] ** 2)), 1))
        for col in range(n_cols)
    ], dtype=np.float32)

    for _ in range(12):
        for col in range(n_cols):
            _fit_sketch_column(col, trits, scales, gram_target)
    return trits, scales


# 40 zero-word operator & punctuation skeleton motifs: (pos_specs, neg_specs) on k=5 window
_SKEL_MOTIFS: tuple[tuple[list[tuple[int, int]], list[tuple[int, int]]], ...] = (
    ([(1, HC_EQ), (2, HC_EQ)], [(0, HC_EQ), (0, HC_BANG), (3, HC_EQ)]),
    ([(1, HC_EQ), (2, HC_EQ), (3, HC_EQ)], []),
    ([(1, HC_BANG), (2, HC_EQ)], [(3, HC_EQ)]),
    ([(1, HC_BANG), (2, HC_EQ), (3, HC_EQ)], []),
    ([(1, HC_LT), (2, HC_EQ)], []),
    ([(1, HC_GT), (2, HC_EQ)], []),
    ([(1, HC_SPACE), (2, HC_LT), (3, HC_SPACE)], []),
    ([(1, HC_SPACE), (2, HC_GT), (3, HC_SPACE)], []),
    ([(1, HC_LT), (2, HC_SPACE), (3, HC_MINUS)], []),
    ([(1, HC_AMP), (2, HC_AMP)], []),
    ([(1, HC_PIPE), (2, HC_PIPE)], []),
    ([(1, HC_BANG), (2, HC_ID)], []),
    ([(1, HC_SPACE), (2, HC_EQ), (3, HC_SPACE)], []),
    ([(1, HC_COL), (2, HC_EQ)], []),
    ([(1, HC_PLUS), (2, HC_EQ)], []),
    ([(1, HC_MINUS), (2, HC_EQ)], []),
    ([(1, HC_EQ), (2, HC_SPACE), (3, HC_DIG)], []),
    ([(1, HC_EQ), (2, HC_SPACE), (3, HC_MINUS), (4, HC_DIG)], []),
    ([(1, HC_PLUS), (2, HC_PLUS)], []),
    ([(1, HC_MINUS), (2, HC_MINUS)], []),
    ([(1, HC_ID), (2, HC_LP)], []),
    ([(1, HC_RP), (2, HC_DOT), (3, HC_ID)], []),
    ([(1, HC_ID), (2, HC_DOT), (3, HC_ID)], []),
    ([(1, HC_ID), (2, HC_PCT), (3, HC_ID)], []),
    ([(1, HC_MINUS), (2, HC_GT), (3, HC_ID)], []),
    ([(1, HC_ID), (2, HC_LBRK), (3, HC_ID)], []),
    ([(1, HC_ID), (2, HC_LBRK), (3, HC_DIG)], []),
    ([(1, HC_LP), (2, HC_ID), (3, HC_COM)], []),
    ([(1, HC_SPACE), (2, HC_PLUS), (3, HC_SPACE)], []),
    ([(1, HC_SPACE), (2, HC_MINUS), (3, HC_SPACE)], []),
    ([(1, HC_SPACE), (2, HC_STAR), (3, HC_SPACE)], []),
    ([(1, HC_SPACE), (2, HC_SLASH), (3, HC_SPACE)], []),
    ([(1, HC_SPACE), (2, HC_QM), (3, HC_SPACE)], []),
    ([(1, HC_SPACE), (2, HC_COL), (3, HC_SPACE)], []),
    ([(1, HC_EQ), (2, HC_GT)], []),
    ([(1, HC_COL), (2, HC_COL)], []),
    ([(1, HC_RP), (2, HC_SPACE), (3, HC_LB)], []),
    ([(1, HC_SEM), (2, HC_SPACE), (3, HC_RB)], []),
    ([(1, HC_DOT), (2, HC_LO), (3, HC_LO), (4, HC_DOT)], []),
    ([(1, HC_COM), (2, HC_SPACE), (3, HC_QT)], []),
)


def _sketch_bank() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build 120 {-1, 0, +1} zero-word sketch filters (code: 0..39, comm: 40..79, skel: 80..119)."""
    total_ch = 3 * SKETCH_DIM
    sketch_w = np.zeros((5, N_HC, total_ch), dtype=np.int8)
    sketch_s = np.full((total_ch,), 0.3, dtype=np.float32)
    sketch_b = np.zeros((total_ch,), dtype=np.float32)

    rng = np.random.RandomState(20260700)
    chosen: list[np.ndarray] = []
    while len(chosen) < SKETCH_DIM:
        cand = rng.choice([-1, 1], size=(4, 5)).astype(np.int8)
        if all(int(np.abs(np.sum(cand * prev))) <= 6 for prev in chosen):
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
        sketch_b[pair] = [-14.2 * 0.3, -(40.0 + 14.2) * 0.3]

    base_skel = 2 * SKETCH_DIM
    for idx, (pos_sp, neg_sp) in enumerate(_SKEL_MOTIFS):
        _punct_filter(sketch_w, sketch_s, sketch_b, base_skel + idx, pos_sp, neg_sp)
    return sketch_w, sketch_s, sketch_b


def _build_embeddings() -> tuple[np.ndarray, np.ndarray]:
    """Compile {-1, 0, +1} class/punctuation byte embeddings and column-index embeddings."""
    tok_emb = np.zeros((128, N_EMB), dtype=np.int8)
    tok_emb[1:, EMB_ACTIVE] = 1
    tok_emb[33:, EMB_NONSPACE] = 1
    tok_emb[32, EMB_SPACE] = 1
    tok_emb[_LO_A : _LO_Z + 1, EMB_LOWER] = 1
    tok_emb[_UP_A : _UP_Z + 1, EMB_UPPER] = 1
    tok_emb[_DIG_0 : _DIG_9 + 1, EMB_DIGIT] = 1

    tok_emb[_LO_A : _LO_Z + 1, EMB_IDENT] = 1
    tok_emb[_UP_A : _UP_Z + 1, EMB_IDENT] = 1
    tok_emb[_DIG_0 : _DIG_9 + 1, EMB_IDENT] = 1
    tok_emb[[ord("_"), ord("$")], EMB_IDENT] = 1

    for byte_val in range(1, 128):
        folded = (byte_val | 0x20) if (_UP_A <= byte_val <= _UP_Z) else byte_val
        bits5 = ((folded & 0x1F) >> _BITS5) & 1
        tok_emb[byte_val, EMB_SH_BITS : EMB_SH_BITS + 5] = bits5.astype(np.int8)

    punct_map = (
        ("_", EMB_UND), ("/", EMB_SLASH), ("*", EMB_STAR), (":", EMB_COLON),
        ("?", EMB_QMARK), (".", EMB_DOT), ("=", EMB_EQ), ("{", EMB_LBRACE),
        ("}", EMB_RBRACE), ("(", EMB_LPAREN), (")", EMB_RPAREN), (";", EMB_SEMI),
        ("&", EMB_AMP), ("|", EMB_PIPE), ('"', EMB_DQUOTE), ("'", EMB_SQUOTE),
        ("`", EMB_BQUOTE), ("\\", EMB_BACKSLASH), ("#", EMB_HASH), ("-", EMB_DASH),
        ("!", EMB_BANG), ("%", EMB_PERCENT), ("[", EMB_LBRK), ("]", EMB_RBRK),
        (",", EMB_COMMA), ("+", EMB_PLUS), ("<", EMB_LT), (">", EMB_GT),
    )
    for sym, dst_col in punct_map:
        tok_emb[ord(sym), dst_col] = 1
    tok_emb[[ord("'"), ord('"'), ord("`")], EMB_ANY_QUOTE] = 1
    tok_emb[[ord("&"), ord("<")], EMB_LIFE_PFX] = 1
    tok_emb[[ord("<"), ord(">")], EMB_ANGLE] = 1

    col_emb = np.zeros((MAX_COLS, N_EMB), dtype=np.int8)
    col_idx = np.arange(MAX_COLS)
    col_emb[col_idx % 2 == 1, EMB_COL_ODD] = 1
    col_emb[col_idx % 4 == 2, EMB_COL_MOD4_2] = 1
    col_emb[(col_idx > 0) & (col_idx % 4 == 0) & (col_idx <= 28), EMB_COL_IND4] = 1
    col_thresholds = (
        (8, EMB_COL_GE8), (10, EMB_COL_GE10), (12, EMB_COL_GE12),
        (16, EMB_COL_GE16), (20, EMB_COL_GE20), (24, EMB_COL_GE24),
    )
    for min_col, dst_col in col_thresholds:
        col_emb[col_idx >= min_col, dst_col] = 1
    return tok_emb, col_emb


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
    pre_s[:3], pre_b[:3] = 2.0, -3.0

    pre_w[0, EMB_NONSPACE, [PRE_HASH_COMM, PRE_HASH2_COMM]] = -1
    pre_w[[1, 2], [EMB_HASH, EMB_SPACE], PRE_HASH_COMM] = 1
    pre_w[[1, 2], EMB_HASH, PRE_HASH2_COMM] = 1
    pre_s[[PRE_HASH_COMM, PRE_HASH2_COMM]] = 2.0
    pre_b[[PRE_HASH_COMM, PRE_HASH2_COMM]] = -3.0

    pre_w[0, EMB_NONSPACE, PRE_DASH_COMM] = -1
    pre_w[[1, 2], EMB_DASH, PRE_DASH_COMM] = 1
    pre_s[PRE_DASH_COMM], pre_b[PRE_DASH_COMM] = 2.0, -3.0

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


def _build_block0_gates(
    off_pre: int,
    off_pref1: int,
    line_comm: list[int],
    mid_comm: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compile De Morgan /*-paired */ closer, polyglot comment, quote, and group-depth gates."""
    gate1_w = np.zeros((off_pref1 + N_PREF1, N_GATE1), dtype=np.int8)
    gate1_s = np.ones((N_GATE1,), dtype=np.float32)
    gate1_b = np.zeros((N_GATE1,), dtype=np.float32)

    gate1_w[[EMB_SPACE, EMB_NONSPACE], [G1_INDENT_SPACE, G1_FIRST_NONSPACE]] = 1
    gate1_w[off_pref1 + C1_NONSPACE, [G1_INDENT_SPACE, G1_FIRST_NONSPACE]] = -1
    gate1_w[[EMB_STAR, off_pref1 + C1_NONSPACE], G1_DOC_STAR] = [1, -1]

    gate1_w[[*mid_comm, off_pre + PRE_BLOCK_OPEN], G1_ANY_COMM_START] = 1
    gate1_w[line_comm, G1_LINE_COMM_START] = 1
    gate1_w[off_pref1 + C1_MID_COMM, [G1_ANY_COMM_START, G1_LINE_COMM_START]] = -1
    gate1_w[off_pre + PRE_BLOCK_CLOSE, [G1_CLOSE_HI, G1_CLOSE_LO]] = 1
    gate1_w[off_pref1 + C1_BLOCK_OPENS, G1_CLOSE_LO] = -1

    indent_cols = (
        EMB_COL_ODD, EMB_COL_MOD4_2, EMB_COL_IND4, EMB_COL_GE8,
        EMB_COL_GE10, EMB_COL_GE12, EMB_COL_GE16, EMB_COL_GE20, EMB_COL_GE24,
    )
    for offset, emb_col in enumerate(indent_cols):
        dst_gate = G1_INDENT_BASE + offset
        gate1_w[[EMB_NONSPACE, emb_col, EMB_STAR, off_pref1 + C1_NONSPACE], dst_gate] = [
            1, 1, -1, -1,
        ]
        gate1_s[dst_gate], gate1_b[dst_gate] = 2.0, -3.0

    gate1_w[off_pref1 + C1_NONSPACE, [G1_LEAD_TWO, G1_LEAD_ONE]] = -1
    gate1_b[[G1_LEAD_TWO, G1_LEAD_ONE]] = [2.0, 1.0]
    for step in range(N_QSAW):
        saw_scale = 2.0 if step > 0 else 1.0
        saw_gates = [G1_DQUOTE_SAW + step, G1_SQUOTE_SAW + step, G1_BQUOTE_SAW + step]
        saw_srcs = [off_pref1 + C1_DQUOTES, off_pref1 + C1_SQUOTES, off_pref1 + C1_BQUOTES]
        gate1_w[saw_srcs, saw_gates] = 1
        gate1_s[saw_gates] = saw_scale
        gate1_b[saw_gates] = -float(step) * saw_scale

    for src_row, dst_gate in (
        (off_pre + PRE_BANG_LEAD, G1_BANG_LEAD),
        (off_pre + PRE_VB_COMM, G1_VB_COMM),
        (off_pre + PRE_ZIG_RAW, G1_ZIG_RAW),
    ):
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
        (C1_GROUP_DEPTH, G1_GRP_HI, G1_GRP_LO),
    ):
        gate1_w[off_pref1 + c1_src, [hi_g, lo_g]] = 1
        gate1_s[[hi_g, lo_g]] = 2.0
        gate1_b[[hi_g, lo_g]] = [0.0, -2.0]

    gate1_w[off_pref1 + C1_GROUP_DEPTH, G1_DEEP_GRP] = 1
    gate1_s[G1_DEEP_GRP], gate1_b[G1_DEEP_GRP] = 0.5, -1.5
    return gate1_w, gate1_s, gate1_b


def _build_col_fuse(
    off_pre: int,
    off_gate1: int,
    off_pref2: int,
    saw_trits: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fuse [emb, pre_conv3, gate1, c2] into N_HC per-cell channels in {-1, 0, +1}."""
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

    indent_hc_dsts = (
        HC_ODD, HC_M42, HC_M40, HC_D8, HC_D10, HC_D12, HC_D16, HC_D20, HC_D24,
    )
    for offset, hc_dst in enumerate(indent_hc_dsts):
        fuse_w[[off_gate1 + G1_INDENT_BASE + offset, *in_comm_pair, in_raw_line], hc_dst] = [
            1, -1, -1, -1,
        ]

    eff_src = [off_pre + PRE_EOL, off_pref2 + C2_EFF_LEN, off_pref2 + C2_LEAD_VETO]
    for thresh, hc_dst in ((96.0, HC_G96), (110.0, HC_G110)):
        fuse_w[eff_src, hc_dst] = [1, 1, -1]
        fuse_s[hc_dst] = 0.425
        fuse_b[hc_dst] = -(2.0 * float(MAX_COLS) + 2.0 * thresh) * 0.425

    char_pairs = (
        (EMB_IDENT, HC_ID), (EMB_LOWER, HC_LO), (EMB_UPPER, HC_UP), (EMB_DIGIT, HC_DIG),
        (EMB_DOT, HC_DOT), (EMB_QMARK, HC_QM), (EMB_EQ, HC_EQ),
        (EMB_LBRACE, HC_LB), (EMB_RBRACE, HC_RB), (EMB_LPAREN, HC_LP), (EMB_RPAREN, HC_RP),
        (EMB_SEMI, HC_SEM), (EMB_ANY_QUOTE, HC_QT), (EMB_LBRK, HC_LBRK), (EMB_RBRK, HC_RBRK),
        (EMB_COMMA, HC_COM), (EMB_COLON, HC_COL), (EMB_SPACE, HC_SPACE), (EMB_BANG, HC_BANG),
        (EMB_LT, HC_LT), (EMB_GT, HC_GT), (EMB_PLUS, HC_PLUS), (EMB_DASH, HC_MINUS),
        (EMB_STAR, HC_STAR), (EMB_SLASH, HC_SLASH), (EMB_PERCENT, HC_PCT),
        (EMB_DQUOTE, HC_DQ), (EMB_SQUOTE, HC_SQ), (EMB_BQUOTE, HC_BQ),
        (EMB_AMP, HC_AMP), (EMB_PIPE, HC_PIPE),
    )
    for emb_src, hc_dst in char_pairs:
        fuse_w[emb_src, hc_dst] = 1

    fuse_w[EMB_SH_BITS + _BITS5, HC_BP + _BITS5] = 1
    fuse_w[EMB_ACTIVE, HC_BN + _BITS5] = 1
    fuse_w[EMB_SH_BITS + _BITS5, HC_BN + _BITS5] = -1

    fuse_w[[off_gate1 + G1_GRP_HI, off_gate1 + G1_GRP_LO], HC_IN_GRP] = [1, -1]
    fuse_s[HC_IN_GRP] = 0.5
    fuse_w[[EMB_NONSPACE, off_gate1 + G1_DEEP_GRP, *in_comm_pair, in_raw_line], HC_DEEP_GRP] = [
        1, 1, -1, -1, -1,
    ]
    fuse_b[HC_DEEP_GRP] = -1.0

    fuse_w[[EMB_LOWER, EMB_UPPER, EMB_UND], HC_NOT_WORD] = -1
    fuse_s[HC_NOT_WORD], fuse_b[HC_NOT_WORD] = 40.0, 40.0
    fuse_w[[in_comm_all[0], in_raw_line], HC_SK_VETO] = 1
    fuse_s[HC_SK_VETO] = 120.0
    fuse_w[[EMB_ACTIVE, in_line_comm], HC_SL_BOOST] = 1
    fuse_s[HC_SL_BOOST], fuse_b[HC_SL_BOOST] = 40.0, -40.0
    return fuse_w, fuse_s, fuse_b


def _build_block0(trit_mats: dict[str, np.ndarray], vecs_1d: dict[str, np.ndarray]) -> None:
    """Compile {-1, 0, +1} byte/column embeddings and quote/comment/group masking circuits."""
    trit_mats["tok_emb"], trit_mats["col_emb"] = _build_embeddings()
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
    pref1[[EMB_LPAREN, EMB_LBRK, EMB_RPAREN, EMB_RBRK], C1_GROUP_DEPTH] = [1, 1, -1, -1]
    trit_mats["pref1_proj"] = pref1

    gate1_w, gate1_s, gate1_b = _build_block0_gates(off_pre, off_pref1, line_comm, mid_comm)
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

    off_pref2 = off_pref1 + N_GATE1
    fuse_w, fuse_s, fuse_b = _build_col_fuse(off_pre, off_pref1, off_pref2, saw_trits)
    trit_mats["col_fuse_w"] = fuse_w
    vecs_1d["col_fuse_s"], vecs_1d["col_fuse_b"] = fuse_s, fuse_b


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
    ("depth_ge16", HC_D16, 1.0),
    ("depth_ge20", HC_D20, 1.0),
    ("depth_ge24", HC_D24, 1.0),
    ("cols_ge96", HC_G96, 1.0 / 12.0),
    ("cols_ge110", HC_G110, 1.0 / 8.0),
    ("deep_group", HC_DEEP_GRP, 1.0 / 8.0),
    ("comment_mask", HC_COMM, 1.0),
    ("has_nonspace", HC_FNS, 1.0),
    ("lower_chars", HC_LO, 1.0),
    ("upper_chars", HC_UP, 1.0),
)


def _build_conv3_bank(
    cap_c3: int, feat: dict[str, FeatSlot]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Compile k=3 zero-word punctuation, delimiter, and comment-structure filters."""
    conv3_w = np.zeros((3, N_HC, cap_c3), dtype=np.int8)
    conv3_s = np.ones((cap_c3,), dtype=np.float32)
    conv3_b = np.zeros((cap_c3,), dtype=np.float32)

    c3_specs: tuple[tuple[str, list[tuple[int, int]], list[tuple[int, int]], bool, bool], ...] = (
        ("code_lparen", [(1, HC_LP)], [], True, False),
        ("code_rparen", [(1, HC_RP)], [], True, False),
        ("code_lbrk", [(1, HC_LBRK)], [], True, False),
        ("code_rbrk", [(1, HC_RBRK)], [], True, False),
        ("func_call", [(0, HC_ID), (1, HC_LP)], [], True, False),
        ("chain_call", [(0, HC_RP), (1, HC_DOT), (2, HC_ID)], [], True, False),
        ("chain_idx", [(0, HC_RBRK), (1, HC_DOT), (2, HC_ID)], [], True, False),
        ("spaced_ternary", [(0, HC_SPACE), (1, HC_QM), (2, HC_SPACE)], [], True, False),
        ("spaced_assign", [(0, HC_SPACE), (1, HC_EQ), (2, HC_SPACE)], [], True, False),
        ("colon_assign", [(0, HC_COL), (1, HC_EQ)], [], True, False),
        ("plus_assign", [(0, HC_PLUS), (1, HC_EQ)], [], True, False),
        ("minus_assign", [(0, HC_MINUS), (1, HC_EQ)], [], True, False),
        ("eq_op", [(0, HC_EQ), (1, HC_EQ)], [], True, False),
        ("neq_op", [(0, HC_BANG), (1, HC_EQ)], [], True, False),
        ("diamond_neq", [(0, HC_LT), (1, HC_GT)], [], True, False),
        ("le_op", [(0, HC_LT), (1, HC_EQ)], [], True, False),
        ("ge_op", [(0, HC_GT), (1, HC_EQ)], [], True, False),
        ("spaced_lt", [(0, HC_SPACE), (1, HC_LT), (2, HC_SPACE)], [], True, False),
        ("spaced_gt", [(0, HC_SPACE), (1, HC_GT), (2, HC_SPACE)], [], True, False),
        ("arrow_op", [(0, HC_EQ), (1, HC_GT)], [], True, False),
        ("pipe_or", [(0, HC_PIPE), (1, HC_PIPE)], [], True, False),
        ("amp_and", [(0, HC_AMP), (1, HC_AMP)], [], True, False),
        ("inc_op", [(0, HC_PLUS), (1, HC_PLUS)], [], True, False),
        ("dec_op", [(0, HC_MINUS), (1, HC_MINUS)], [], True, False),
        ("comma_eol", [(1, HC_COM)], [(2, HC_ACT)], True, False),
        ("colon_eol", [(1, HC_COL)], [(0, HC_COL), (2, HC_ACT)], True, False),
        ("lbrace_eol", [(1, HC_LB)], [(2, HC_ACT)], True, False),
        ("lead_rbrace", [(1, HC_LEAD), (1, HC_RB)], [], True, False),
        ("col0_nonspace", [(1, HC_FNS)], [(0, HC_ACT)], False, False),
        ("top_pascal", [(0, HC_SPACE), (1, HC_UP)], [(1, HC_IN_GRP)], True, False),
        ("doc_slash3", [(0, HC_SLASH), (1, HC_SLASH), (2, HC_SLASH)], [], False, True),
        ("doc_bang", [(0, HC_SLASH), (1, HC_SLASH), (2, HC_BANG)], [], False, True),
        ("comm_semi_eol", [(1, HC_SEM)], [(2, HC_ACT)], False, True),
        ("comm_lbrace_eol", [(1, HC_LB)], [(2, HC_ACT)], False, True),
        ("comm_rbrace_eol", [(1, HC_RB)], [(2, HC_ACT)], False, True),
        ("comm_fcall", [(0, HC_ID), (1, HC_LP)], [], False, True),
        ("comm_assign", [(0, HC_SPACE), (1, HC_EQ), (2, HC_SPACE)], [], False, True),
        ("comm_colon_assign", [(0, HC_COL), (1, HC_EQ)], [], False, True),
        ("comm_plus_assign", [(0, HC_PLUS), (1, HC_EQ)], [], False, True),
    )
    for ch_i, (name, pos_sp, neg_sp, is_code, is_comm) in enumerate(c3_specs):
        _punct_filter(conv3_w, conv3_s, conv3_b, ch_i, pos_sp, neg_sp, code=is_code, comm=is_comm)
        feat[name] = (1, ch_i)

    cur = len(c3_specs)
    ch_bopen, ch_bclose, ch_doc_blk = cur, cur + 1, cur + 2
    _punct_filter(
        conv3_w, conv3_s, conv3_b, ch_bopen,
        [(1, HC_SLASH), (2, HC_STAR)],
        [(0, HC_ID), (0, HC_DOT), (0, HC_QT), (1, HC_RAW_STR), (1, HC_SL_C)],
        code=False, comm=False,
    )
    _punct_filter(
        conv3_w, conv3_s, conv3_b, ch_bclose,
        [(1, HC_STAR), (2, HC_SLASH)],
        [(0, HC_QT), (1, HC_RAW_STR), (1, HC_SL_C)],
        code=False, comm=False,
    )
    conv3_w[1, [HC_FNS, HC_COMM, HC_SL_C], ch_doc_blk] = [1, 1, -1]
    conv3_s[ch_doc_blk], conv3_b[ch_doc_blk] = 2.0, -3.0
    feat["ml_bcomm_open"] = (1, ch_bopen)
    feat["ml_bcomm_close"] = (1, ch_bclose)
    feat["doc_block_line"] = (1, ch_doc_blk)
    return conv3_w, conv3_s, conv3_b, cur + 3


def _build_conv7_bank(
    cap_c7: int, feat: dict[str, FeatSlot]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Compile k=7 zero-word multi-token punctuation, equality-ladder, and comment filters."""
    conv7_w = np.zeros((7, N_HC, cap_c7), dtype=np.int8)
    conv7_s = np.ones((cap_c7,), dtype=np.float32)
    conv7_b = np.zeros((cap_c7,), dtype=np.float32)

    _punct_filter(
        conv7_w, conv7_s, conv7_b, 0,
        [(2, HC_DQ), (3, HC_DQ), (4, HC_DQ)], [(1, HC_QT), (5, HC_QT), (3, HC_SL_C)],
        code=False, comm=False,
    )
    _punct_filter(
        conv7_w, conv7_s, conv7_b, 1,
        [(2, HC_SQ), (3, HC_SQ), (4, HC_SQ)], [(1, HC_QT), (5, HC_QT), (3, HC_SL_C)],
        code=False, comm=False,
    )
    feat["triple_dq"], feat["triple_sq"] = (2, 0), (2, 1)

    c7_specs: tuple[tuple[str, list[tuple[int, int]], list[tuple[int, int]], bool, bool], ...] = (
        ("mid_top_semi", [(2, HC_SEM), (4, HC_CNS)], [(2, HC_IN_GRP)], True, False),
        ("mid_top_semi_sp", [(2, HC_SEM), (3, HC_CNS)], [(2, HC_IN_GRP)], True, False),
        ("rparen_stmt", [(2, HC_RP), (3, HC_SPACE), (4, HC_LO)], [(3, HC_IN_GRP)], True, False),
        ("eq_quote_lit", [(1, HC_EQ), (2, HC_EQ), (3, HC_SPACE), (4, HC_QT)], [], True, False),
        (
            "eq3_quote_lit",
            [(0, HC_EQ), (1, HC_EQ), (2, HC_EQ), (3, HC_SPACE), (4, HC_QT)],
            [], True, False,
        ),
        ("eq_num_lit", [(1, HC_EQ), (2, HC_EQ), (3, HC_SPACE), (4, HC_DIG)], [], True, False),
        (
            "eq3_num_lit",
            [(0, HC_EQ), (1, HC_EQ), (2, HC_EQ), (3, HC_SPACE), (4, HC_DIG)],
            [], True, False,
        ),
        (
            "eq_neg_num",
            [(1, HC_EQ), (2, HC_EQ), (3, HC_SPACE), (4, HC_MINUS), (5, HC_DIG)],
            [], True, False,
        ),
        (
            "assign_num",
            [(1, HC_SPACE), (2, HC_EQ), (3, HC_SPACE), (4, HC_DIG)],
            [(0, HC_RBRK)], True, False,
        ),
        (
            "assign_neg_num",
            [(1, HC_SPACE), (2, HC_EQ), (3, HC_SPACE), (4, HC_MINUS), (5, HC_DIG)],
            [(0, HC_RBRK)], True, False,
        ),
        ("rbrk_assign", [(0, HC_RBRK), (1, HC_SPACE), (2, HC_EQ), (3, HC_SPACE)], [], True, False),
        (
            "kv_id_row",
            [(1, HC_ID), (2, HC_COL), (3, HC_SPACE), (4, HC_CNS)],
            [(1, HC_COL)], True, False,
        ),
        ("kv_qt_row", [(1, HC_QT), (2, HC_COL), (3, HC_SPACE), (4, HC_CNS)], [], True, False),
        ("sig_colon", [(2, HC_RP), (3, HC_COL)], [(4, HC_ACT)], True, False),
        ("sig_thin_arr", [(1, HC_RP), (2, HC_SPACE), (3, HC_MINUS), (4, HC_GT)], [], True, False),
        ("sig_fat_arr", [(1, HC_RP), (2, HC_SPACE), (3, HC_EQ), (4, HC_GT)], [], True, False),
        ("sig_brace", [(2, HC_RP), (3, HC_SPACE), (4, HC_LB)], [(5, HC_ACT)], True, False),
        (
            "elision_slash",
            [
                (0, HC_IND), (1, HC_SLASH), (2, HC_SLASH), (3, HC_SPACE),
                (4, HC_DOT), (5, HC_DOT), (6, HC_DOT),
            ],
            [], False, True,
        ),
        (
            "elision_hash",
            [(0, HC_IND), (1, HC_MNS), (2, HC_SPACE), (3, HC_DOT), (4, HC_DOT), (5, HC_DOT)],
            [(6, HC_DOT)], False, True,
        ),
        ("fence_bt", [(2, HC_BQ), (3, HC_BQ), (4, HC_BQ)], [], False, True),
        ("fence_gt", [(1, HC_GT), (2, HC_GT), (3, HC_GT), (4, HC_SPACE)], [], False, True),
        (
            "directive_nosp",
            [(1, HC_SLASH), (2, HC_SLASH), (3, HC_LO), (4, HC_LO), (5, HC_COL)],
            [], False, True,
        ),
        (
            "directive_plus",
            [(1, HC_SLASH), (2, HC_SLASH), (3, HC_SPACE), (4, HC_PLUS)],
            [], False, True,
        ),
        ("lbrace_inline", [(1, HC_RP), (2, HC_SPACE), (3, HC_LB), (4, HC_SPACE)], [], True, False),
        ("brace_inline", [(1, HC_SPACE), (2, HC_LB), (3, HC_SPACE), (4, HC_CNS)], [], True, False),
        ("comm_rparen_col", [(3, HC_RP), (4, HC_COL)], [(5, HC_ACT)], False, True),
        ("comm_brk_eol", [(3, HC_LBRK), (4, HC_RBRK)], [(5, HC_ACT)], False, True),
        ("comm_arrow", [(2, HC_EQ), (3, HC_GT)], [], False, True),
    )
    for offset, (name, pos_sp, neg_sp, is_code, is_comm) in enumerate(c7_specs, 2):
        _punct_filter(conv7_w, conv7_s, conv7_b, offset, pos_sp, neg_sp, code=is_code, comm=is_comm)
        feat[name] = (2, offset)
    return conv7_w, conv7_s, conv7_b, 2 + len(c7_specs)


def _build_multiline_gates(
    trit_mats: dict[str, np.ndarray],
    vecs_1d: dict[str, np.ndarray],
    raw_idx: dict[str, int],
    num_raw_u: int,
) -> None:
    """Compile {-1, 0, +1} docstring/comment sawtooth, Dyck ()[] balance, and step-in taps."""
    ml_proj = np.zeros((num_raw_u, N_ML_IN), dtype=np.int8)
    ml_proj_s = np.ones((N_ML_IN,), dtype=np.float32)
    ml_proj[raw_idx["triple_dq"], 0] = 1
    ml_proj[raw_idx["triple_sq"], 1] = 1
    ml_proj[[raw_idx["ml_bcomm_open"], raw_idx["ml_bcomm_close"]], 2] = [1, -1]
    ml_proj[[raw_idx["code_lparen"], raw_idx["code_lbrk"]], 3] = 1
    ml_proj[[raw_idx["code_rparen"], raw_idx["code_rbrk"]], 3] = -1
    ml_proj[raw_idx["comment_mask"], 4] = 1
    ml_proj[raw_idx["code_chars"], 5] = 1
    ml_proj[raw_idx["indent_cols"], 6] = 1
    ml_proj[raw_idx["has_nonspace"], 7] = 1
    ml_proj_s[[5, 7]] = float(MAX_COLS)
    trit_mats["ml_proj"], vecs_1d["ml_proj_s"] = ml_proj, ml_proj_s

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

    n_line_conv = N_PAREN_SUF + N_EXTRA_LINE_CONV
    ml_paren_w = np.zeros((2 * N_PAREN_STEPS + 1, N_ML_IN, n_line_conv), dtype=np.int8)
    ml_paren_s = np.ones((n_line_conv,), dtype=np.float32)
    ml_paren_b = np.zeros((n_line_conv,), dtype=np.float32)
    for step_k in range(N_PAREN_STEPS):
        span_k = step_k + 1
        u_hi, u_lo = 2 * step_k, 2 * step_k + 1
        ml_paren_w[N_PAREN_STEPS - span_k : N_PAREN_STEPS, 3, [u_hi, u_lo]] = 1
        ml_paren_b[[u_hi, u_lo]] = [0.0, -1.0]

    prev_tap, cur_tap = N_PAREN_STEPS - 1, N_PAREN_STEPS
    p_comm, step_in = N_PAREN_SUF, N_PAREN_SUF + 2
    ml_paren_w[prev_tap, [4, 5], p_comm] = [1, -1]
    ml_paren_w[prev_tap, [4, 5], p_comm + 1] = [1, -1]
    ml_paren_b[[p_comm, p_comm + 1]] = [0.0, -1.0]

    ml_paren_w[cur_tap, 6, [step_in, step_in + 1]] = 1
    ml_paren_w[prev_tap, 6, [step_in, step_in + 1]] = -1
    ml_paren_w[prev_tap, 7, [step_in, step_in + 1]] = 1
    ml_paren_s[[step_in, step_in + 1]] = 2.0
    nonblank_off = 2.0 * float(MAX_COLS)
    ml_paren_b[[step_in, step_in + 1]] = [-nonblank_off, -nonblank_off - 1.0]
    trit_mats["ml_paren_w"] = ml_paren_w
    vecs_1d["ml_paren_s"], vecs_1d["ml_paren_b"] = ml_paren_s, ml_paren_b


def _build_line_rules() -> list[LineRule]:
    """Build declarative zero-word geometric line rules populating Z[0..38]."""
    dir_veto = [("directive_nosp", -float(MAX_COLS)), ("directive_plus", -float(MAX_COLS))]
    doc_veto = [("doc_slash3", -4.0), ("doc_bang", -4.0), ("doc_block_line", -4.0), *dir_veto]
    pure_comm = [("lower_chars", 0.5), ("upper_chars", 0.5), ("code_chars", -5.0), *dir_veto]
    short_comm_hi = [("comment_chars", 1.0), ("code_chars", -5.0), *doc_veto]
    short_comm_lo = [("comment_chars", -0.25), ("code_chars", -5.0), *doc_veto]

    col0_block = [
        ("lbrace_eol", 2.0), ("colon_eol", 2.0),
        ("indent_cols", -8.0), ("code_lparen", -4.0), ("kv_id_row", -4.0), ("kv_qt_row", -4.0),
    ]
    col0_pascal = [
        ("col0_nonspace", 2.0), ("top_pascal", 2.0),
        ("spaced_assign", -6.0), ("kv_id_row", -4.0), ("kv_qt_row", -4.0),
    ]
    sig_terms = [
        ("sig_colon", 2.0), ("sig_thin_arr", 2.0), ("sig_fat_arr", 2.0), ("sig_brace", 2.0),
        ("depth_ge8", -6.0),
    ]
    packed_terms = [
        ("mid_top_semi", 1.2), ("mid_top_semi_sp", 1.2),
        ("rparen_stmt", 1.10), ("cols_ge96", 0.45), ("kv_id_row", -1.5),
    ]
    stmt_ops = [
        ("spaced_assign", 1.2), ("colon_assign", 1.2), ("plus_assign", 1.2),
        ("minus_assign", 1.2), ("func_call", 1.0), ("inc_op", 1.0), ("dec_op", 1.0),
    ]

    stmt_terms = [*stmt_ops, ("comma_eol", -1.5), ("kv_id_row", -2.0), ("kv_qt_row", -2.0)]
    ind_stmt_terms = [
        *stmt_ops, ("col0_nonspace", -6.0),
        ("comma_eol", -4.0), ("kv_id_row", -4.0), ("kv_qt_row", -4.0),
    ]
    data_row_terms = [
        ("comma_eol", 1.5), ("kv_id_row", 1.5), ("kv_qt_row", 1.5), ("func_call", -2.0),
    ]
    close_only_terms = [("lead_rbrace", 3.0), ("lbrace_eol", 3.0), ("code_chars", -2.0)]

    eq_lits = [
        ("eq_quote_lit", 4.0), ("eq3_quote_lit", 4.0),
        ("eq_num_lit", 4.0), ("eq3_num_lit", 4.0), ("eq_neg_num", 4.0),
    ]
    branch_punc = [
        ("lbrace_eol", 2.0), ("brace_inline", 2.0), ("lbrace_inline", 2.0),
        ("colon_eol", 2.0), ("rparen_stmt", 2.0),
    ]
    ladder_terms = [
        *eq_lits, *branch_punc,
        ("neq_op", -8.0), ("spaced_assign", -6.0), ("colon_assign", -6.0), ("arrow_op", -6.0),
        ("spaced_lt", -6.0), ("spaced_gt", -6.0), ("le_op", -6.0), ("ge_op", -6.0),
        ("amp_and", -6.0), ("pipe_or", -6.0),
    ]
    guard_terms = [
        ("neq_op", 1.5), ("diamond_neq", 1.5), ("le_op", 1.2), ("ge_op", 1.2),
        ("spaced_lt", 1.2), ("spaced_gt", 1.2),
    ]
    clone_elig = [
        ("code_chars", 0.12), *stmt_ops, ("rbrk_assign", -2.5),
        ("comma_eol", -4.0), ("kv_id_row", -4.0), ("kv_qt_row", -4.0),
    ]
    skel_elig = [
        ("code_chars", 0.04),
        ("eq_op", 1.8), ("neq_op", 1.8), ("diamond_neq", 1.8),
        ("le_op", 1.8), ("ge_op", 1.8), ("spaced_lt", 1.8), ("spaced_gt", 1.8),
        ("pipe_or", 1.8), ("amp_and", 1.8), ("assign_num", 1.85), ("assign_neg_num", 1.85),
        ("plus_assign", 1.6), ("minus_assign", 1.6), ("spaced_assign", 0.60), ("func_call", 2.15),
        ("rbrk_assign", -3.0), ("comma_eol", -4.0), ("kv_id_row", -4.0), ("kv_qt_row", -4.0),
    ]
    comm_vetoes = [*doc_veto, ("code_chars", -5.0)]
    strong_comm_code = [
        ("comm_semi_eol", 2.0), ("comm_lbrace_eol", 2.0), ("comm_rbrace_eol", 2.0),
        ("comm_rparen_col", 2.0), ("comm_brk_eol", 2.0),
        ("comm_colon_assign", 2.0), ("comm_plus_assign", 2.0), ("comm_arrow", 2.0),
        *comm_vetoes,
    ]
    weak_comm_fcall = [("comm_fcall", 2.0), *comm_vetoes]
    weak_comm_assign = [("comm_assign", 2.0), *comm_vetoes]
    prose_terms = [
        ("lower_chars", 0.1), ("upper_chars", 0.1), ("code_chars", -5.0), *dir_veto,
        *[
            (name, -3.0)
            for name in (
                "comm_semi_eol", "comm_lbrace_eol", "comm_rbrace_eol", "comm_rparen_col",
                "comm_brk_eol", "comm_colon_assign", "comm_plus_assign", "comm_arrow",
                "comm_assign",
            )
        ],
    ]
    mut_terms = [
        ("plus_assign", 1.5), ("minus_assign", 1.5), ("inc_op", 1.5), ("dec_op", 1.5),
        ("spaced_assign", 0.8), ("depth_ge8", 0.8), ("comma_eol", -2.0),
    ]
    blk_open_terms = [
        ("lbrace_eol", 1.5), ("colon_eol", 1.5), ("indent_cols", 0.5),
        ("kv_id_row", -3.0), ("kv_qt_row", -3.0), ("comma_eol", -3.0),
    ]
    doc_code_veto = [
        ("doc_slash3", 2.0), ("doc_bang", 2.0), ("doc_block_line", 2.0),
        ("directive_nosp", 2.0), ("directive_plus", 2.0), ("code_chars", 2.0),
    ]

    def clamp01(
        terms: list[tuple[str, float]],
        thresh: float,
        dst: int,
        gain: float = 1.0,
        cap: float = 1.0,
    ) -> list[LineRule]:
        return [(terms, -thresh, dst, gain), (terms, -thresh - cap / gain, dst, -gain)]

    return [
        ([("has_nonspace", 1.0)], 0.0, Z_ACTIVE, 1.0),
        *clamp01([("code_chars", 1.0)], 0.0, Z_HAS_CODE),
        *clamp01(pure_comm, 0.5, Z_PURE_COMMENT, 2.0),
        *clamp01(short_comm_hi, 5.0, Z_SHORT_COMMENT),
        *clamp01(short_comm_lo, -15.0, Z_SHORT_COMMENT),
        ([("has_nonspace", -1.0)], 1.0, Z_PARAGRAPH_BREAK, 1.0),
        *clamp01([("comment_mask", 1.0), ("code_chars", -5.0)], 2.0, Z_PARAGRAPH_BREAK),
        *clamp01(sig_terms, 1.0, Z_PARAGRAPH_BREAK),

        ([("indent_cols", 1.0)], 0.0, Z_INDENT, 1.0),
        ([("depth_ge10", 1.0)], 0.0, Z_DEPTH10, 1.0),
        ([("depth_ge12", 1.0)], 0.0, Z_DEPTH12, 1.0),
        ([("depth_ge20", 1.0)], 0.0, Z_DEPTH20, 1.0),
        ([("depth_ge24", 1.0)], 0.0, Z_DEPTH24, 1.0),
        ([("odd_indent", 1.0)], 0.0, Z_ODD_INDENT, 1.0),
        ([("mod4_eq2_indent", 1.0)], 0.0, Z_MOD4_EQ2, 1.0),
        ([("mod4_eq0_indent", 1.0)], 0.0, Z_MOD4_EQ0, 1.0),
        ([("depth_ge16", -1.0)], 1.0, Z_SHALLOW_LINE, 1.0),
        ([("depth_ge10", -1.0)], 1.0, Z_SHALLOW_2SP, 1.0),
        *clamp01([("indent_cols", 2.0)], 0.5, Z_IS_INDENTED),

        *clamp01(col0_block, 1.0, Z_AT_COL0_BLOCK),
        *clamp01(col0_pascal, 3.0, Z_AT_COL0_BLOCK),
        *clamp01(sig_terms, 1.0, Z_SIG_LINE),
        ([("chain_call", 1.0), ("chain_idx", 1.0)], -0.45, Z_CHAIN_CALLS, 0.55),
        ([("spaced_ternary", 1.0)], -1.2, Z_NESTED_TERNARY, 1.6),
        (packed_terms, -1.65, Z_PACKED_STMTS, 1.5),
        ([("cols_ge96", 1.0)], -0.55, Z_OVER_96, 0.35),
        *clamp01([("cols_ge110", 1.0)], 0.80, Z_OVER_110, 0.55),
        ([("deep_group", 1.0)], -9.0, Z_DEEP_INLINE_GRP, 0.5),

        *clamp01(stmt_terms, 0.4, Z_IS_STMT),
        *clamp01(data_row_terms, 2.2, Z_IS_DATA_ROW),
        *clamp01(close_only_terms, 0.0, Z_CLOSE_DELIM_ONLY),
        *clamp01(ladder_terms, 5.2, Z_EQ_LADDER_RUNG, 2.0),
        *clamp01(guard_terms, 0.5, Z_GUARD_LINE, 2.0),
        *clamp01(clone_elig, 4.35, Z_CLONE_ELIGIBLE, 2.0),
        *clamp01(skel_elig, 2.65, Z_SKEL_ELIGIBLE, 2.0),
        *clamp01(
            [("elision_slash", 2.0), ("elision_hash", 2.0), ("code_chars", -5.0)],
            1.0, Z_ELISION_COMM,
        ),
        *clamp01([("fence_bt", 2.0), ("fence_gt", 2.0), ("code_chars", -5.0)], 1.0, Z_COMM_FENCE),
        *clamp01(doc_code_veto, 1.0, Z_DOC_OR_CODE_VETO),
        *clamp01(strong_comm_code, 0.5, Z_COMM_CODE_CUE, gain=4.0, cap=2.0),
        *clamp01(weak_comm_fcall, 0.5, Z_COMM_CODE_CUE, 2.0),
        *clamp01(weak_comm_assign, 0.5, Z_COMM_CODE_CUE, 2.0),
        *clamp01(mut_terms, 1.2, Z_MUT_LINE, 2.0),
        *clamp01(blk_open_terms, 1.6, Z_BLOCK_OPEN_LINE, 2.0),
        *clamp01(ind_stmt_terms, 0.35, Z_INDENTED_STMT, 2.0),
        *clamp01(prose_terms, 0.8, Z_PROSE_COMMENT, 2.0),
    ]


_ML_UNIT_NAMES = (
    "triple_dq", "triple_sq", "ml_bcomm_open", "ml_bcomm_close",
    "code_lparen", "code_lbrk", "code_rparen", "code_rbrk",
    "comment_mask", "code_chars", "indent_cols", "has_nonspace",
)


def _prepare_rule_merges(
    line_rules: list[LineRule], feat: dict[str, FeatSlot]
) -> tuple[list[tuple[dict[FeatSlot, float], float, float, int, float]], set[FeatSlot]]:
    unit_scale: set[FeatSlot] = {feat[name] for name in _ML_UNIT_NAMES}
    prepared: list[tuple[dict[FeatSlot, float], float, float, int, float]] = []
    for terms, bias, dst_z, out_scale in line_rules:
        merged: dict[FeatSlot, float] = {}
        for name, weight in terms:
            slot = feat[name]
            merged[slot] = merged.get(slot, 0.0) + weight
        mag_counts = Counter(round(abs(wt), 6) for wt in merged.values())
        base_mag = max(mag_counts.items(), key=lambda pair: (pair[1], pair[0]))[0]
        for slot, wt_val in merged.items():
            if abs(abs(wt_val) / base_mag - 1.0) < 1e-6:
                unit_scale.add(slot)
        prepared.append((merged, base_mag, bias, dst_z, out_scale))
    return prepared, unit_scale


def _resolve_feat_slot(
    slot: FeatSlot,
    mult: float,
    banks: tuple[BankBuffer, ...],
    unit_scale: set[FeatSlot],
    rescaled_once: set[FeatSlot],
    scaled_cache: dict[tuple[FeatSlot, float], FeatSlot],
) -> FeatSlot:
    if abs(mult - 1.0) < 1e-6:
        return slot
    cache_key = (slot, round(mult, 6))
    if cache_key in scaled_cache:
        return scaled_cache[cache_key]

    bank_id, src_c = slot
    w_arr, s_arr, b_arr, orig_s, orig_b, cur_box = banks[bank_id]
    if slot not in unit_scale and slot not in rescaled_once:
        rescaled_once.add(slot)
        s_arr[src_c] = orig_s[src_c] * mult
        if b_arr is not None and orig_b is not None:
            b_arr[src_c] = orig_b[src_c] * mult
        scaled_cache[cache_key] = slot
        return slot

    dst_c = cur_box[0]
    cur_box[0] += 1
    if w_arr is not None:
        if w_arr.ndim == 2:
            w_arr[:, dst_c] = w_arr[:, src_c]
        else:
            w_arr[:, :, dst_c] = w_arr[:, :, src_c]
    s_arr[dst_c] = orig_s[src_c] * mult
    if b_arr is not None and orig_b is not None:
        b_arr[dst_c] = orig_b[src_c] * mult
    cloned = (bank_id, dst_c)
    scaled_cache[cache_key] = cloned
    return cloned


def _wire_multiline_units(
    lm1_w: np.ndarray,
    lm1_b: np.ndarray,
    lm2_w: np.ndarray,
    cur_u: int,
    num_raw_u: int,
    raw_idx: dict[str, int],
) -> int:
    dq_trits = np.array([1] + [-1, 1] * (N_DQ_SAW // 2 - 1) + [-1], dtype=np.int8)
    sq_trits = np.array([1] + [-1, 1] * (N_SQ_SAW // 2 - 1) + [-1], dtype=np.int8)
    off_saw = num_raw_u
    off_par = off_saw + N_ML_GATE

    for u_off, b_val, w2_sign in ((0, 0.0, 1), (1, -1.0, -1)):
        u_i = cur_u + u_off
        lm1_w[off_saw : off_saw + N_DQ_SAW, u_i] = dq_trits
        lm1_w[off_saw + N_DQ_SAW : off_saw + N_DQ_SAW + N_SQ_SAW, u_i] = sq_trits
        lm1_w[off_saw + N_DQ_SAW + N_SQ_SAW : off_saw + N_ML_GATE, u_i] = [1, -1]
        lm1_w[[raw_idx["triple_dq"], raw_idx["triple_sq"], raw_idx["ml_bcomm_close"]], u_i] = -1
        lm1_b[u_i] = b_val
        lm2_w[u_i, Z_IN_DOC] = w2_sign

    par_trits = np.array([1, -1] * N_PAREN_STEPS, dtype=np.int8)
    for u_off, b_val, w2_sign in ((2, 0.0, 1), (3, -1.0, -1)):
        u_i = cur_u + u_off
        lm1_w[off_par : off_par + N_PAREN_SUF, u_i] = par_trits
        lm1_b[u_i] = b_val
        lm2_w[u_i, Z_IN_PAREN] = w2_sign

    u_pc, u_si = cur_u + 4, cur_u + 5
    lm1_w[[off_par + N_PAREN_SUF, off_par + N_PAREN_SUF + 1], u_pc] = [1, -1]
    lm2_w[u_pc, Z_PREV_COMMENT] = 1
    lm1_w[[off_par + N_PAREN_SUF + 2, off_par + N_PAREN_SUF + 3], u_si] = [1, -1]
    lm2_w[u_si, Z_STEP_IN] = 1
    return cur_u + 6


def _build_block1(trit_mats: dict[str, np.ndarray], vecs_1d: dict[str, np.ndarray]) -> None:
    """Compile zero-word k=3/7 line convolutions, multiline state, and 3-slice SimHash."""
    feat: dict[str, FeatSlot] = {}
    csum_w = np.zeros((N_HC, 48), dtype=np.int8)
    csum_s = np.ones((48,), dtype=np.float32)
    for col_i, (name, hc_src, scale_val) in enumerate(_CSUM_SPECS):
        csum_w[hc_src, col_i] = 1
        csum_s[col_i] = scale_val
        feat[name] = (0, col_i)

    conv3_w, conv3_s, conv3_b, c3_init = _build_conv3_bank(72, feat)
    conv7_w, conv7_s, conv7_b, c7_init = _build_conv7_bank(64, feat)
    csum_box, c3_box, c7_box = [len(_CSUM_SPECS)], [c3_init], [c7_init]
    banks: tuple[BankBuffer, ...] = (
        (csum_w, csum_s, None, csum_s.copy(), None, csum_box),
        (conv3_w, conv3_s, conv3_b, conv3_s.copy(), conv3_b.copy(), c3_box),
        (conv7_w, conv7_s, conv7_b, conv7_s.copy(), conv7_b.copy(), c7_box),
    )

    line_rules = _build_line_rules()
    prepared, unit_scale = _prepare_rule_merges(line_rules, feat)
    n_lm1 = len(prepared) + 6
    lm1_s = np.ones((n_lm1,), dtype=np.float32)
    lm1_b = np.zeros((n_lm1,), dtype=np.float32)
    lm2_w = np.zeros((n_lm1, N_LMLP2), dtype=np.int8)
    lm2_b = np.zeros((N_LMLP2,), dtype=np.float32)
    lm2_b[Z_SHORT_COMMENT] = -1.0

    rescaled_once: set[FeatSlot] = set()
    scaled_cache: dict[tuple[FeatSlot, float], FeatSlot] = {}
    rule_links: list[tuple[FeatSlot, int, int]] = []

    for unit, (merged, base_mag, bias, dst_z, out_scale) in enumerate(prepared):
        abs_out = abs(out_scale)
        lm1_s[unit] = base_mag * abs_out
        lm1_b[unit] = bias * abs_out
        lm2_w[unit, dst_z] = 1 if out_scale >= 0 else -1
        for slot, wt_val in merged.items():
            resolved = _resolve_feat_slot(
                slot, abs(wt_val) / base_mag, banks, unit_scale, rescaled_once, scaled_cache
            )
            rule_links.append((resolved, unit, 1 if wt_val >= 0 else -1))

    n_csum, n_c3, n_c7 = csum_box[0], c3_box[0], c7_box[0]
    bank_offs = (0, n_csum, n_csum + n_c3)
    num_raw_u = n_csum + n_c3 + n_c7
    raw_idx = {name: bank_offs[b_id] + col for name, (b_id, col) in feat.items()}
    _build_multiline_gates(trit_mats, vecs_1d, raw_idx, num_raw_u)

    num_line_u = num_raw_u + N_ML_GATE + N_PAREN_SUF + N_EXTRA_LINE_CONV
    lm1_w = np.zeros((num_line_u, n_lm1), dtype=np.int8)
    for (b_id, col), unit, sign in rule_links:
        lm1_w[bank_offs[b_id] + col, unit] = sign
    _wire_multiline_units(lm1_w, lm1_b, lm2_w, len(prepared), num_raw_u, raw_idx)

    lm2_s = np.ones((N_LMLP2,), dtype=np.float32)
    lm2_s[[
        Z_IN_DOC, Z_IN_PAREN, Z_PREV_COMMENT, Z_SIG_LINE,
        Z_DOC_OR_CODE_VETO, Z_IS_DATA_ROW, Z_CLOSE_DELIM_ONLY,
    ]] = 4.0
    lm2_s[Z_COMM_CODE_CUE] = 0.5

    trit_mats["col_sum_proj"], vecs_1d["col_sum_s"] = csum_w[:, :n_csum], csum_s[:n_csum]
    trit_mats["conv3_w"] = conv3_w[:, :, :n_c3]
    vecs_1d["conv3_s"], vecs_1d["conv3_b"] = conv3_s[:n_c3], conv3_b[:n_c3]
    trit_mats["conv7_w"] = conv7_w[:, :, :n_c7]
    vecs_1d["conv7_s"], vecs_1d["conv7_b"] = conv7_s[:n_c7], conv7_b[:n_c7]
    trit_mats["sketch_w"], vecs_1d["sketch_s"], vecs_1d["sketch_b"] = _sketch_bank()
    trit_mats["lmlp1_w"], vecs_1d["lmlp1_s"], vecs_1d["lmlp1_b"] = lm1_w, lm1_s, lm1_b
    trit_mats["lmlp2_w"], vecs_1d["lmlp2_s"], vecs_1d["lmlp2_b"] = lm2_w, lm2_s, lm2_b


def _build_attn1(trit_mats: dict[str, np.ndarray], vecs_1d: dict[str, np.ndarray]) -> None:
    """Compile Pass 1 10-head cross-line geometric & SimHash attention."""
    num_heads, head_dim, val_dim = 10, SKETCH_DIM, 4
    qk_comp = (head_dim / 16.0) ** 0.25
    center = MAX_REL_DIST
    num_rel = 2 * center + 1
    qk_cols, val_cols = num_heads * head_dim, num_heads * val_dim

    a1_wq = np.zeros((N_Z, qk_cols), dtype=np.int8)
    a1_wq_s = np.ones((qk_cols,), dtype=np.float32)
    a1_wk = np.zeros((N_Z, qk_cols), dtype=np.int8)
    a1_wk_s = np.ones((qk_cols,), dtype=np.float32)
    a1_wv = np.zeros((N_Z, val_cols), dtype=np.int8)
    a1_wv_s = np.ones((val_cols,), dtype=np.float32)
    a1_wo = np.zeros((val_cols, N_Z), dtype=np.int8)
    a1_br = np.full((num_heads, num_rel), -1e4, dtype=np.float32)
    a1_bs = np.array([
        ECHO_COS_GAIN * ECHO_COS_THRESH, 1.5, 1.5, math.log(8.0), math.log(4.0),
        10.9, 14.4, 8.75, 11.1, math.log(16.0),
    ], dtype=np.float32)

    comm_slice = slice(Z_COMM_SKETCH, Z_COMM_SKETCH + SKETCH_DIM)
    code_slice = slice(Z_CODE_SKETCH, Z_CODE_SKETCH + SKETCH_DIM)
    skel_slice = slice(Z_SKEL_SKETCH, Z_SKEL_SKETCH + SKETCH_DIM)

    # Head 0: Exact 40-dim cosine between comment SimHash (L) and statement code SimHash (L+1)
    eye_sk = np.eye(SKETCH_DIM, dtype=np.int8)
    a1_wq[comm_slice, 0:head_dim] = eye_sk
    a1_wk[code_slice, 0:head_dim] = eye_sk
    a1_wq_s[0:head_dim] = a1_wk_s[0:head_dim] = math.sqrt(ECHO_COS_GAIN * math.sqrt(head_dim))
    a1_br[0, center + 1] = 0.0
    a1_wv[[Z_IS_STMT, Z_SIG_LINE], 0] = [1, -1]
    a1_wo[0, Z_ECHO_SIM] = 1
    a1_wv_s[0] = 2.4

    # Head 1: Local comment/fence/data-row context [-3..+3] (excluding self offset 0)
    a1_br[1, [center - 3, center - 2, center - 1, center + 1, center + 2, center + 3]] = 0.0
    h1_v = [4, 5, 6, 7]
    a1_wv[[Z_COMM_FENCE, Z_COMM_CODE_CUE, Z_PURE_COMMENT, Z_IS_DATA_ROW], h1_v] = 1
    a1_wv_s[h1_v] = [20.0, 6.5, 6.0, 8.0]
    a1_wo[h1_v, [Z_NEAR_FENCE, Z_LOCAL_COMM_CODE, Z_ADJ_COMMENT, Z_LOCAL_DATA_ROWS]] = 1

    # Head 2: Enclosing pyramid indentation depth and step-in contour window
    a1_br[2, center - 5 : center + 4] = 0.0
    h2_v = [8, 9, 10, 11]
    a1_wv[[Z_DEPTH20, Z_DEPTH24, Z_DEPTH12, Z_STEP_IN], h2_v] = 1
    a1_wv[Z_DEPTH10, 10] = 1
    a1_wv_s[h2_v] = [1.8, 2.2, 1.4, 1.8]
    a1_wo[h2_v, [Z_LOCAL_DEEP20, Z_LOCAL_DEEP24, Z_LOCAL_DEEP2SP, Z_STAIRCASE_STEPS]] = 1

    # Head 3: Vertical rhythm & paragraph break density [-10..+10]
    a1_br[3, center - 10 : center + 11] = 0.0
    h3_v = [12, 13, 14, 15]
    a1_wv[[Z_INDENTED_STMT, Z_PARAGRAPH_BREAK, Z_ODD_INDENT, Z_BLOCK_OPEN_LINE], h3_v] = 1
    a1_wv_s[h3_v] = [14.5, 20.0, 5.0, 6.0]
    a1_wo[h3_v, [Z_LOCAL_STMT_DENS, Z_LOCAL_BREAK_DENS, Z_LOCAL_ODD_IND, Z_LOCAL_BRANCH_DENS]] = 1

    # Head 4: Parallel equality-branch ladder & prose-essay veto [-6..+6] (excluding self offset 0)
    abs_off = np.abs(np.arange(-center, center + 1))
    a1_br[4, (abs_off >= 1) & (abs_off <= 6)] = 0.0
    a1_wv[Z_EQ_LADDER_RUNG, 16] = a1_wo[16, Z_LOCAL_LADDER] = 1
    a1_wv_s[16] = 16.0
    a1_wv[Z_PROSE_COMMENT, 17] = a1_wo[17, Z_LOCAL_PROSE] = 1
    a1_wv_s[17] = 6.0

    # Heads 5..8: Zero-mean QR/Paley far-minus-near SimHash clone pairs (code 5/6 & skel 7/8)
    sk_trits, sk_scales = _ternary_sketch_proj(N_CODE_PROJ)
    sk16_trits, sk16_scales = _ternary_sketch_proj(16)
    skel_proj = _paley_uniform_proj(seed=7)
    joint_proj = np.zeros((N_Z, 16), dtype=np.int8)
    joint_proj[code_slice, :4] = sk16_trits[:, :4]
    joint_proj[skel_slice, 4:] = skel_proj[:, :12]
    code_proj = np.zeros((N_Z, N_CODE_PROJ), dtype=np.int8)
    code_proj[code_slice, :] = sk_trits
    joint_sc = np.concatenate([sk16_scales[:4] * 7.5, np.full((12,), 2.45, dtype=np.float32)])

    clone_head_specs = (
        (5, 8, Z_CLONE_ELIGIBLE, 0.72, Z_DISTANT_CODE_CLONE, code_proj, sk_scales * 8.0),
        (7, 4, Z_SKEL_ELIGIBLE, 0.60, Z_DISTANT_SKEL_CLONE, joint_proj, joint_sc),
    )
    for far_h, min_d, elig_z, veto_w, dst_z, proj_m, sc_vec in clone_head_specs:
        near_h = far_h + 1
        for head in (far_h, near_h):
            qk_span = slice(head * head_dim, head * head_dim + proj_m.shape[1])
            a1_wq[:, qk_span] = a1_wk[:, qk_span] = proj_m
            a1_wq_s[qk_span] = a1_wk_s[qk_span] = sc_vec * qk_comp
        a1_br[far_h, (abs_off >= min_d) & (abs_off < center)] = 0.0
        a1_br[near_h, (abs_off >= 1) & (abs_off <= 3)] = 0.0
        v_pair = [far_h * val_dim, near_h * val_dim]
        a1_wv[elig_z, v_pair] = 1
        a1_wv_s[v_pair] = [1.0, veto_w]
        a1_wo[v_pair, dst_z] = [1, -1]

    # Head 9: Global O(L) file-wide indentation & col-0 container baseline (b_rel == 0)
    a1_br[9, :] = 0.0
    h9_v = [36, 37, 38, 39]
    a1_wv[[Z_MOD4_EQ0, Z_MOD4_EQ2, Z_ODD_INDENT, Z_AT_COL0_BLOCK], h9_v] = 1
    a1_wv_s[h9_v] = [4.0, 6.0, 8.0, 48.0]
    a1_wo[h9_v, [Z_FILE_MOD4_0, Z_FILE_MOD4_2, Z_FILE_ODD, Z_FILE_CONTAINERS]] = 1

    trit_mats["attn1_wq"], vecs_1d["attn1_wq_s"] = a1_wq, a1_wq_s
    trit_mats["attn1_wk"], vecs_1d["attn1_wk_s"] = a1_wk, a1_wk_s
    trit_mats["attn1_wv"], vecs_1d["attn1_wv_s"] = a1_wv, a1_wv_s
    trit_mats["attn1_wo"] = a1_wo
    vecs_1d["attn1_brel"], vecs_1d["attn1_bsink"] = a1_br, a1_bs


def _build_ctx_mlp1(trit_mats: dict[str, np.ndarray], vecs_1d: dict[str, np.ndarray]) -> None:
    """Compile Context MLP 1 paired {-1, 0, +1} ReLU clamps writing candidate issue flags."""
    veto_common = [(Z_IN_DOC, -1), (Z_IN_PAREN, -1), (Z_CLOSE_DELIM_ONLY, -1)]
    nest_vetoes = [*veto_common, (Z_LOCAL_DATA_ROWS, -1), (Z_IS_DATA_ROW, -1)]

    ctx1_rules: list[tuple[list[tuple[int, int]], float, int, float]] = [
        ([(Z_FILE_CONTAINERS, 1)], 0.35, Z_HAS_CONTAINER, 8.0),
        ([(Z_FILE_MOD4_2, 1), (Z_FILE_MOD4_0, -1), (Z_FILE_ODD, -1)], 0.15, Z_USES_2SP, 12.0),
        ([(Z_FILE_MOD4_0, 1), (Z_FILE_MOD4_2, 1), (Z_FILE_ODD, -1)], 0.45, Z_STRICT_EVEN_IND, 8.0),

        ([(Z_HAS_CODE, 1), (Z_DEPTH24, 1), (Z_LOCAL_DEEP24, 1), (Z_STAIRCASE_STEPS, 1),
          (Z_GUARD_LINE, -1), *nest_vetoes], 2.55, Z_RAW_NEST_4SP, 3.0),
        ([(Z_HAS_CODE, 1), (Z_DEPTH12, 1), (Z_LOCAL_DEEP2SP, 1), (Z_STAIRCASE_STEPS, 1),
          (Z_GUARD_LINE, -1), *nest_vetoes], 2.55, Z_RAW_NEST_2SP, 3.0),

        ([(Z_INDENTED_STMT, 1), (Z_LOCAL_STMT_DENS, 1), (Z_LOCAL_BREAK_DENS, -1),
          *veto_common, (Z_IS_DATA_ROW, -1)], 9.65, Z_RAW_LAYOUT_WALL, 3.0),
        ([(Z_INDENTED_STMT, 1), (Z_ODD_INDENT, 1), (Z_LOCAL_ODD_IND, -1), *veto_common],
         1.35, Z_RAW_ODD_GLITCH, 4.0),

        ([(Z_IS_INDENTED, 1), (Z_CLONE_ELIGIBLE, 1), (Z_DISTANT_CODE_CLONE, 1),
          (Z_SIG_LINE, -1), *veto_common],
         2.56, Z_RAW_CODE_CLONE, 4.5),
        ([(Z_IS_INDENTED, 1), (Z_SKEL_ELIGIBLE, 1), (Z_DISTANT_SKEL_CLONE, 1),
          (Z_SIG_LINE, -1), *veto_common],
         2.52, Z_RAW_SKEL_CLONE, 4.5),

        ([(Z_EQ_LADDER_RUNG, 1), (Z_LOCAL_LADDER, 1), (Z_IN_DOC, -1), (Z_IN_PAREN, -1)],
         3.55, Z_RAW_LADDER, 3.5),

        ([(Z_PURE_COMMENT, 1), (Z_SHORT_COMMENT, 1), (Z_ECHO_SIM, 1),
          (Z_PREV_COMMENT, -1), (Z_ADJ_COMMENT, -1), (Z_COMM_FENCE, -1),
          (Z_DOC_OR_CODE_VETO, -1), (Z_IN_DOC, -1)],
         3.15, Z_RAW_COMM_ECHO, 4.0),
        ([(Z_PURE_COMMENT, 1), (Z_COMM_CODE_CUE, 1), (Z_LOCAL_COMM_CODE, 1), (Z_LOCAL_PROSE, -1),
          (Z_COMM_FENCE, -1), (Z_NEAR_FENCE, -1), (Z_DOC_OR_CODE_VETO, -1), (Z_IN_DOC, -1)],
         2.15, Z_RAW_COMM_CODE, 3.5),

        ([(Z_CHAIN_CALLS, 1), (Z_NESTED_TERNARY, 1), (Z_PACKED_STMTS, 1),
          (Z_OVER_96, 1), (Z_OVER_110, 1), (Z_DEEP_INLINE_GRP, 1),
          (Z_IS_DATA_ROW, -1), (Z_IN_DOC, -1)],
         1.05, Z_ISS_DENSE, 2.2),
    ]

    n_cm1 = 2 * len(ctx1_rules)
    cm1_w = np.zeros((N_Z, n_cm1), dtype=np.int8)
    cm1_s = np.ones((n_cm1,), dtype=np.float32)
    cm1_b = np.zeros((n_cm1,), dtype=np.float32)
    cm1_2w = np.zeros((n_cm1, N_Z), dtype=np.int8)
    cm1_2b = np.zeros((N_Z,), dtype=np.float32)

    for idx, (terms, thresh, dst_z, slope) in enumerate(ctx1_rules):
        unit = 2 * idx
        for src_z, trit in terms:
            cm1_w[src_z, [unit, unit + 1]] = trit
        cm1_s[[unit, unit + 1]] = slope
        cm1_b[[unit, unit + 1]] = [-thresh * slope, -thresh * slope - 1.0]
        cm1_2w[[unit, unit + 1], dst_z] = [1, -1]

    trit_mats["cmlp1_w"], vecs_1d["cmlp1_s"], vecs_1d["cmlp1_b"] = cm1_w, cm1_s, cm1_b
    trit_mats["cmlp1_2_w"], vecs_1d["cmlp1_2_b"] = cm1_2w, cm1_2b


def _build_attn2(trit_mats: dict[str, np.ndarray], vecs_1d: dict[str, np.ndarray]) -> None:
    """Compile Pass 2 4-head pure relative-offset neighborhood smoothing attention."""
    num_heads, val_dim = 4, 2
    center = MAX_REL_DIST
    num_rel = 2 * center + 1
    val_cols = num_heads * val_dim

    a2_wv = np.zeros((N_Z, val_cols), dtype=np.int8)
    a2_wv_s = np.ones((val_cols,), dtype=np.float32)
    a2_wo = np.zeros((val_cols, N_Z), dtype=np.int8)
    a2_br = np.full((num_heads, num_rel), -1e4, dtype=np.float32)
    a2_bs = np.array([0.7, 1.0, 1.2, 0.8], dtype=np.float32)

    a2_br[0, center - 2 : center + 3] = 0.0
    a2_br[1, center - 2 : center + 3] = 0.0
    a2_br[2, center - 3 : center + 4] = 0.0
    a2_br[3, center - 2 : center + 3] = 0.0

    # Head 0: Consecutive [-2..+2] clone stanza verification (>= 3 of 5 lines must match)
    a2_wv[[Z_RAW_CODE_CLONE, Z_RAW_SKEL_CLONE], [0, 1]] = 1
    a2_wv_s[[0, 1]] = 2.5
    a2_wo[[0, 1], [Z_SMOOTH_CODE_CLONE, Z_SMOOTH_SKEL_CLONE]] = 1

    # Head 1: Pyramid apex smoothing across 4-space and 2-space
    a2_wv[[Z_RAW_NEST_4SP, Z_RAW_NEST_2SP], [2, 3]] = 1
    a2_wv_s[[2, 3]] = 1.4
    a2_wo[[2, 3], [Z_SMOOTH_NEST_CLS, Z_SMOOTH_NEST_2SP]] = 1

    # Head 2: Equality ladder & statement-wall core smoothing
    a2_wv[[Z_RAW_LADDER, Z_RAW_LAYOUT_WALL], [4, 5]] = 1
    a2_wv_s[[4, 5]] = [1.8, 1.6]
    a2_wo[[4, 5], [Z_SMOOTH_LADDER, Z_SMOOTH_WALL]] = 1

    # Head 3: Consecutive commented-out code confirmation
    a2_wv[Z_RAW_COMM_CODE, 6] = 1
    a2_wv_s[6] = 3.0
    a2_wo[6, Z_SMOOTH_COMM_CODE] = 1

    trit_mats["attn2_wv"], vecs_1d["attn2_wv_s"] = a2_wv, a2_wv_s
    trit_mats["attn2_wo"] = a2_wo
    vecs_1d["attn2_brel"], vecs_1d["attn2_bsink"] = a2_br, a2_bs


def _build_ctx_mlp2(trit_mats: dict[str, np.ndarray], vecs_1d: dict[str, np.ndarray]) -> None:
    """Compile Context MLP 2 assembling the final 6 pure-geometry issue registers."""
    doc_par_veto = [(Z_IN_DOC, -1), (Z_IN_PAREN, -1)]
    ctx2_rules: list[tuple[list[tuple[int, int]], float, int, float]] = [
        ([(Z_RAW_NEST_4SP, 1), (Z_SMOOTH_NEST_CLS, 1), (Z_SHALLOW_LINE, -1), *doc_par_veto],
         0.95, Z_ISS_NESTING, 4.0),
        ([(Z_RAW_NEST_4SP, 1), (Z_SMOOTH_NEST_CLS, 1), (Z_HAS_CONTAINER, -1),
          (Z_SHALLOW_LINE, -1), *doc_par_veto],
         0.95, Z_ISS_NESTING, 4.0),
        ([(Z_RAW_NEST_2SP, 1), (Z_SMOOTH_NEST_2SP, 1), (Z_USES_2SP, 1),
          (Z_SHALLOW_2SP, -1), *doc_par_veto],
         1.95, Z_ISS_NESTING, 4.0),

        ([(Z_RAW_LAYOUT_WALL, 1), (Z_SMOOTH_WALL, 1), *doc_par_veto],
         1.10, Z_ISS_LAYOUT, 4.0),
        ([(Z_RAW_ODD_GLITCH, 1), (Z_STRICT_EVEN_IND, 1), *doc_par_veto],
         1.35, Z_ISS_LAYOUT, 4.0),

        ([(Z_CLONE_ELIGIBLE, 1), (Z_RAW_CODE_CLONE, 1), (Z_SMOOTH_CODE_CLONE, 1), *doc_par_veto],
         2.96, Z_ISS_CLONE, 4.5),
        ([(Z_SKEL_ELIGIBLE, 1), (Z_RAW_SKEL_CLONE, 1), (Z_SMOOTH_SKEL_CLONE, 1), *doc_par_veto],
         2.86, Z_ISS_CLONE, 4.5),

        ([(Z_EQ_LADDER_RUNG, 1), (Z_RAW_LADDER, 1), (Z_SMOOTH_LADDER, 1), *doc_par_veto],
         2.15, Z_ISS_LADDER, 4.0),

        ([(Z_RAW_COMM_ECHO, 1), (Z_COMM_FENCE, -1), (Z_IN_DOC, -1)], 0.55, Z_ISS_COMMENT, 4.0),
        ([(Z_ELISION_COMM, 1), (Z_COMM_FENCE, -1), (Z_NEAR_FENCE, -1),
          (Z_DOC_OR_CODE_VETO, -1), (Z_IN_DOC, -1)],
         0.50, Z_ISS_COMMENT, 4.0),
        ([(Z_RAW_COMM_CODE, 1), (Z_SMOOTH_COMM_CODE, 1), (Z_COMM_FENCE, -1),
          (Z_NEAR_FENCE, -1), (Z_IN_DOC, -1)],
         1.30, Z_ISS_COMMENT, 9.0),
    ]

    n_cm2 = 2 * len(ctx2_rules)
    cm2_w = np.zeros((N_Z, n_cm2), dtype=np.int8)
    cm2_s = np.ones((n_cm2,), dtype=np.float32)
    cm2_b = np.zeros((n_cm2,), dtype=np.float32)
    cm2_2w = np.zeros((n_cm2, N_Z), dtype=np.int8)
    cm2_2b = np.zeros((N_Z,), dtype=np.float32)

    for idx, (terms, thresh, dst_z, slope) in enumerate(ctx2_rules):
        unit = 2 * idx
        for src_z, trit in terms:
            cm2_w[src_z, [unit, unit + 1]] = trit
        cm2_s[[unit, unit + 1]] = slope
        cm2_b[[unit, unit + 1]] = [-thresh * slope, -thresh * slope - 1.0]
        cm2_2w[[unit, unit + 1], dst_z] = [1, -1]

    trit_mats["cmlp2_w"], vecs_1d["cmlp2_s"], vecs_1d["cmlp2_b"] = cm2_w, cm2_s, cm2_b
    trit_mats["cmlp2_2_w"], vecs_1d["cmlp2_2_b"] = cm2_2w, cm2_2b


def _build_readout_heads(trit_mats: dict[str, np.ndarray], vecs_1d: dict[str, np.ndarray]) -> None:
    """Compile 6-kind geometric squiggle head and 1-column saturating document score head."""
    issue_regs = [
        Z_ISS_NESTING, Z_ISS_DENSE, Z_ISS_LAYOUT,
        Z_ISS_CLONE, Z_ISS_LADDER, Z_ISS_COMMENT,
    ]
    sq_w = np.zeros((N_Z, 6), dtype=np.int8)
    sq_w[issue_regs, range(6)] = 1
    sq_s = np.full((6,), 4.6, dtype=np.float32)
    sq_b = np.full((6,), -2.2, dtype=np.float32)
    trit_mats["sq_w"], vecs_1d["sq_s"], vecs_1d["sq_b"] = sq_w, sq_s, sq_b

    pool_gate_w = np.zeros((N_Z, 1), dtype=np.int8)
    pool_gate_w[[Z_ACTIVE, Z_IN_DOC], 0] = [1, -1]
    trit_mats["pool_gate_w"] = pool_gate_w
    vecs_1d["pool_gate_s"] = np.array([12.0], dtype=np.float32)
    vecs_1d["pool_gate_b"] = np.array([-6.0], dtype=np.float32)

    issue_clamp_specs = (
        (Z_ISS_NESTING, 2.15, 1.15, 1.35),
        (Z_ISS_DENSE, 0.80, 0.40, 0.50),
        (Z_ISS_LAYOUT, 0.75, 0.40, 0.48),
        (Z_ISS_CLONE, 2.10, 1.15, 1.30),
        (Z_ISS_LADDER, 1.75, 0.90, 1.10),
        (Z_ISS_COMMENT, 1.15, 0.55, 0.65),
    )
    struct_clamps = (
        ([Z_INDENT, Z_LOCAL_DEEP20], [], 0.55, 3.60, 0.45),
        ([Z_LOCAL_BRANCH_DENS], [], 1.35, 0.90, 0.48),
        ([Z_LOCAL_STMT_DENS], [Z_LOCAL_BREAK_DENS], 1.15, 4.80, 0.65),
        ([Z_MUT_LINE], [], 5.00, 0.26, 0.45),
    )

    n_doc1 = 4 * len(issue_clamp_specs) + 2 * len(struct_clamps) + 2
    doc1_w = np.zeros((2 * N_Z, n_doc1), dtype=np.int8)
    doc1_s = np.ones((n_doc1,), dtype=np.float32)
    doc1_b = np.zeros((n_doc1,), dtype=np.float32)
    doc2_w = np.zeros((n_doc1, 1), dtype=np.int8)
    doc2_b = np.array([2.00], dtype=np.float32)

    cursor = 0
    mean_gain, max_gain, max_deadzone = 2.4, 8.2, 0.036
    for reg, weight_mag, cap_m, cap_p in issue_clamp_specs:
        u_m, v_m, u_p, v_p = cursor, cursor + 1, cursor + 2, cursor + 3
        cursor += 4
        doc1_w[reg, [u_m, v_m]] = 1
        doc1_s[[u_m, v_m]] = mean_gain * weight_mag
        doc1_b[[u_m, v_m]] = [0.0, -cap_m * weight_mag]
        doc2_w[[u_m, v_m], 0] = [-1, 1]

        doc1_w[N_Z + reg, [u_p, v_p]] = 1
        doc1_s[[u_p, v_p]] = max_gain * weight_mag
        p_off = max_gain * max_deadzone * weight_mag
        doc1_b[[u_p, v_p]] = [-p_off, -p_off - cap_p * weight_mag]
        doc2_w[[u_p, v_p], 0] = [-1, 1]

    for pos_regs, neg_regs, s_gain, s_offset, w_mag in struct_clamps:
        u_pos, u_neg = cursor, cursor + 1
        cursor += 2
        if pos_regs:
            doc1_w[pos_regs, [u_pos, u_neg]] = 1
        if neg_regs:
            doc1_w[neg_regs, [u_pos, u_neg]] = -1
        eff = s_gain * w_mag
        off_sc = s_offset * eff
        doc1_s[[u_pos, u_neg]] = eff
        doc1_b[[u_pos, u_neg]] = [-off_sc, -off_sc - w_mag]
        doc2_w[[u_pos, u_neg], 0] = [-1, 1]

    u_g, v_g = cursor, cursor + 1
    doc1_w[Z_GUARD_LINE, [u_g, v_g]] = 1
    doc1_s[[u_g, v_g]] = 0.60
    doc1_b[[u_g, v_g]] = [0.0, -0.25]
    doc2_w[[u_g, v_g], 0] = [1, -1]

    trit_mats["doc1_w"] = doc1_w
    vecs_1d["doc1_s"], vecs_1d["doc1_b"] = doc1_s, doc1_b
    trit_mats["doc2_w"], vecs_1d["doc2_b"] = doc2_w, doc2_b


def verify_zero_vocabulary_invariance(trit_mats: dict[str, np.ndarray]) -> None:
    """Prove mathematically that zero alphabetic/digit substring filters exist in the weights."""
    tok_geom = trit_mats["tok_emb"][:, :EMB_SH_BITS]
    if not np.all(tok_geom[_LO_A : _LO_Z + 1] == tok_geom[_LO_A]):
        raise AssertionError("Lowercase letters [a-z] are not identical in geometric tok_emb")
    if not np.all(tok_geom[_UP_A : _UP_Z + 1] == tok_geom[_UP_A]):
        raise AssertionError("Uppercase letters [A-Z] are not identical in geometric tok_emb")
    if not np.all(tok_geom[_DIG_0 : _DIG_9 + 1] == tok_geom[_DIG_0]):
        raise AssertionError("Digits [0-9] are not identical in geometric tok_emb")

    if np.any(trit_mats["pre_conv3_w"][:, EMB_SH_BITS:, :] != 0):
        raise AssertionError("pre_conv3_w leaked SimHash character-identity bits")
    if np.any(trit_mats["col_sum_proj"][_SH_HC_COLS, :] != 0):
        raise AssertionError("col_sum_proj leaked SimHash character-identity bits")
    for bank in ("conv3_w", "conv7_w"):
        if np.any(trit_mats[bank][:, _SH_HC_COLS, :] != 0):
            raise AssertionError(f"{bank} leaked SimHash character-identity bits")
    if np.any(trit_mats["sketch_w"][:, _SH_HC_COLS, 2 * SKETCH_DIM :] != 0):
        raise AssertionError("skeleton SimHash leaked character-identity bits")


_BUILTIN_SMOKE_PAIRS: tuple[tuple[str, str, str], ...] = (
    (
        "deep_nesting",
        "def scan(items):\n    if not items:\n        return 0\n    return items[0]\n",
        (
            "def scan(items, cfg):\n"
            "    if items:\n"
            "        for it in items:\n"
            "            if it.ok:\n"
            "                if cfg.strict:\n"
            "                    if it.val > 0:\n"
            "                        return it.val\n"
            "    return 0\n"
        ),
    ),
    (
        "dense_line",
        "def step(x):\n    y = x + 1\n    return y * 2\n",
        "def step(x):\n    a = x + 1; b = a * 2; c = b - 3; return c + 4\n",
    ),
    (
        "cramped_layout",
        "def norm(x):\n    a = x + 1\n    b = a * 2\n    return b - 1\n",
        (
            "def norm(x, rate, fee):\n"
            "    sub = x + 1\n"
            "    disc = sub * rate\n"
            "    net = sub - disc\n"
            "     ship = fee + 2\n"
            "    tot = net + ship\n"
            "    return tot - 1\n"
        ),
    ),
    (
        "duplicate_block",
        "def apply_pair(a, b, fn):\n    return fn(a), fn(b)\n",
        (
            "def send_invoice(client, region, event):\n"
            "    endpoint = resolve_endpoint(region)\n"
            "    payload = encode_payload(event)\n"
            "    digest = compute_sha256_hex(payload)\n"
            "    token = sign_payload_digest(digest, client.key)\n"
            "    status = post_signed_frame(client, endpoint, payload, token)\n"
            "    return status\n\n"
            "def send_refund(client, region, event):\n"
            "    endpoint = resolve_endpoint(region)\n"
            "    payload = encode_payload(event)\n"
            "    digest = compute_sha256_hex(payload)\n"
            "    token = sign_payload_digest(digest, client.key)\n"
            "    status = post_signed_frame(client, endpoint, payload, token)\n"
            "    return status\n"
        ),
    ),
    (
        "repetitive_structure",
        "CODES = {'a': 1, 'b': 2, 'c': 3, 'd': 4}\ndef lookup(k):\n    return CODES.get(k, 0)\n",
        (
            "def lookup(mode):\n"
            "    if mode == 'alpha':\n        return 1\n"
            "    if mode == 'bravo':\n        return 2\n"
            "    if mode == 'charlie':\n        return 3\n"
            "    if mode == 'delta':\n        return 4\n"
            "    if mode == 'omega':\n        return 5\n"
            "    return 0\n"
        ),
    ),
    (
        "comment_echo",
        "def send(req):\n    # Retry idempotent payloads before circuit breaker trips\n"
        "    dispatch_payload(req)\n",
        (
            "def clean_email(record):\n"
            "    # Strip and lower user_email from record\n"
            "    user_email = record.user_email.strip().lower()\n"
            "    return user_email\n"
        ),
    ),
)

_DEFAULT_BLUEPRINT_PATH = Path(__file__).with_name(".codex-notes") / "evals" / "evals.json"


def _load_precompile_grid_blueprint(
    evals_path: Path | None = None,
) -> tuple[dict[str, Any], ...]:
    target = evals_path if evals_path is not None else _DEFAULT_BLUEPRINT_PATH
    if not target.is_file():
        return ()
    dataset = json.loads(target.read_text("utf-8"))
    return tuple(dataset.get("precompile_grid_blueprint", ()))


PRECOMPILE_GRID_BLUEPRINT: tuple[dict[str, Any], ...] = _load_precompile_grid_blueprint()


def verify_builtin_smoke_grids(params: dict[str, np.ndarray]) -> int:
    """Verify built-in clean/messy geometric smoke pairs across all 6 squiggle kinds."""
    for target_kind, clean_code, messy_code in _BUILTIN_SMOKE_PAIRS:
        clean_sq = score(clean_code, weights=params).squiggles
        if clean_sq:
            raise AssertionError(f"Smoke clean FP for {target_kind}: {clean_sq}")
        messy_kinds = [sq.kind for sq in score(messy_code, weights=params).squiggles]
        if target_kind not in messy_kinds:
            raise AssertionError(
                f"Smoke recall miss for {target_kind}: got {messy_kinds}"
            )
    return len(_BUILTIN_SMOKE_PAIRS)


def verify_precompile_grid_blueprint(
    params: dict[str, np.ndarray],
    blueprint: tuple[dict[str, Any], ...] | None = None,
) -> int:
    """Verify every paired green/red (L x 128) byte-grid boundary before saving weights.npz."""
    active_bp = PRECOMPILE_GRID_BLUEPRINT if blueprint is None else blueprint
    if len(active_bp) < 24:
        raise AssertionError("PRECOMPILE_GRID_BLUEPRINT must define at least 24 boundary pairs")
    for pair in active_bp:
        name = pair["name"]
        green_sq = score(pair["green"]["code"], weights=params).squiggles
        if green_sq:
            raise AssertionError(f"Blueprint green FP in {name}: {green_sq}")
        target_kind = pair["red"]["target"]
        red_kinds = [sq.kind for sq in score(pair["red"]["code"], weights=params).squiggles]
        if target_kind not in red_kinds or any(kind != target_kind for kind in red_kinds):
            raise AssertionError(
                f"Blueprint red mismatch in {name}: expected {target_kind}, got {red_kinds}"
            )
    return len(active_bp)


def _format_entry_meta(name: str, shape: tuple[int, ...], length: int) -> str:
    dims = ",".join(str(dim) for dim in shape)
    return f"{name}:{dims}:{length}"


def compile_weights(
    output_path: Path | None = None,
    *,
    blueprint_path: Path | None = None,
) -> tuple[int, int, int]:
    """Compile all 30 ternary matrices and 1D scale/bias vectors into `weights.npz`."""
    trit_mats: dict[str, np.ndarray] = {}
    vecs_1d: dict[str, np.ndarray] = {}
    _build_block0(trit_mats, vecs_1d)
    _build_block1(trit_mats, vecs_1d)
    _build_attn1(trit_mats, vecs_1d)
    _build_ctx_mlp1(trit_mats, vecs_1d)
    _build_attn2(trit_mats, vecs_1d)
    _build_ctx_mlp2(trit_mats, vecs_1d)
    _build_readout_heads(trit_mats, vecs_1d)
    verify_zero_vocabulary_invariance(trit_mats)

    packed_chunks: list[np.ndarray] = []
    trit_meta: list[str] = []
    for name, mat in trit_mats.items():
        if not np.isin(mat, (-1, 0, 1)).all():
            raise AssertionError(f"Non-ternary entry in {name}")
        packed = _pack_trits(mat)
        packed_chunks.append(packed)
        trit_meta.append(_format_entry_meta(name, mat.shape, len(packed)))

    vec_chunks = [np.asarray(vec, dtype=np.float32).ravel() for vec in vecs_1d.values()]
    vec_meta = [
        _format_entry_meta(name, np.asarray(vec).shape, len(flat_v))
        for (name, vec), flat_v in zip(vecs_1d.items(), vec_chunks, strict=True)
    ]

    total_trits = sum(int(mat.size) for mat in trit_mats.values())
    total_scalars = sum(len(chunk) for chunk in vec_chunks)
    trit_buf = np.concatenate(packed_chunks)
    vec_buf = np.concatenate(vec_chunks)
    meta_str = " ; ".join(trit_meta) + "|" + " ; ".join(vec_meta)
    params = unpack_ternary_archive(trit_buf, vec_buf, meta_str)

    verify_builtin_smoke_grids(params)
    blueprint = (
        PRECOMPILE_GRID_BLUEPRINT
        if blueprint_path is None
        else _load_precompile_grid_blueprint(blueprint_path)
    )
    if blueprint:
        verify_precompile_grid_blueprint(params, blueprint)

    target = output_path or (Path(__file__).with_name("codehygiene") / "weights.npz")
    np.savez_compressed(target, trits=trit_buf, vecs=vec_buf, meta=np.array(meta_str))
    return len(trit_mats), total_trits, total_scalars


def main() -> int:
    num_mats, num_trits, num_scalars = compile_weights()
    npz_path = Path(__file__).with_name("codehygiene") / "weights.npz"
    kib = npz_path.stat().st_size / 1024.0
    n_bp = len(PRECOMPILE_GRID_BLUEPRINT)
    gate_msg = (
        f"{n_bp}/{n_bp} blueprint pairs verified"
        if n_bp
        else f"{len(_BUILTIN_SMOKE_PAIRS)}/{len(_BUILTIN_SMOKE_PAIRS)} smoke pairs verified"
    )
    print(
        f"Compiled {num_mats} zero-vocabulary ternary matrices "
        f"({num_trits:,} trits in {{-1,0,+1}}, {num_scalars:,} scalars, "
        f"{gate_msg}, {kib:.2f} KiB)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
