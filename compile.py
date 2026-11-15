"""Closed-form Weight-Space Program compiling polyglot circuits into weights.npz.

Architecture & Residual Layout (Z has 192 channels per line):
  - Block 0 (_build_block0): Maps (L, 128) ASCII bytes + column positions into 52 per-cell
    channels (HC_*), using causal prefix matrices to mask string literals (16-quote sawtooth)
    and polyglot comments (//, /* */, # , -- ) left-to-right across each line.
  - Block 1 (_build_block1): Runs k=3, 7, 13 1D convolutions for operators, control-flow
    keywords, ceremonial suffixes, and golfed short-name/magic-index patterns, plus 40
    SimHash stem filters (_sketch_bank) for live code and inline comments.
  - Block 2 (_build_block2_and_heads): Two cross-line sink-attention + MLP passes populate
    relational context in Z[40:61], bounded [0, 1] issue flags in Z[64:77], and smoothed
    neighborhood flags in Z[80:86].
  - Block 3 (_build_block2_and_heads): Projects Z into 8 per-line squiggle probabilities
    and pools active lines into 31 System One logits (2 scores, 6x4 sub-scores, 5 styles).
"""

from __future__ import annotations

from pathlib import Path
import numpy as np

from codehygiene import MAX_COLS, MAX_REL_DIST

_BITS7 = np.arange(7)
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
    HC_LO, HC_UP, HC_DIG, HC_UND, HC_DOT, HC_QM, HC_COL, HC_EQ,
    HC_LB, HC_RB, HC_LP, HC_RP, HC_SEM, HC_PA, HC_QT, HC_LET,
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
C2_IN_COMMENT, C2_IN_LINE_COMM = 0, 1
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

# Cross-line attention context registers (Z[40..60]) and SimHash slices (Z[112..192])
(
    Z_RESTATES_CODE, Z_NEAR_REAL_HANDLER, Z_NEAR_EMPTY_HANDLER,
    Z_ENCLOSING_DEPTH, Z_ENCLOSING_BRANCHES, Z_ENCLOSING_GUARDS,
    Z_LOCAL_STMT_DENSITY, Z_LOCAL_BREAK_DENSITY, Z_DISTANT_CLONE,
    Z_LOCAL_SPECIAL_CASES, Z_LOCAL_ODD_INDENT, Z_ENCLOSING_DEEP16,
    Z_FILE_ONE_CHAR, Z_FILE_WORDS, Z_LOCAL_CHAINING, Z_LOCAL_CEREMONY,
    Z_FILE_MOD4_0, Z_FILE_MOD4_2, Z_FILE_ODD_IND,
    Z_LOCAL_SHORT_GOLF, Z_LOCAL_MAGIC_NUM,
) = range(40, 61)

# Bounded [0, 1] per-line issue registers (Z[64..76]) and smoothed registers (Z[80..86])
(
    Z_ISS_NESTING, Z_ISS_NAMING, Z_ISS_COMMENT, Z_ISS_ERROR,
    Z_ISS_DENSE, Z_ISS_SPECIAL, Z_RAW_CLONE, Z_ISS_LAYOUT,
    Z_RAW_CRYPTIC, Z_CLEAN_BONUS, Z_ISS_CLONE,
    Z_USES_WIDE_INDENT, Z_RAW_NESTING_2SP,
) = range(64, 77)
(
    Z_SMOOTH_CLONE, Z_SMOOTH_NEST, Z_SMOOTH_CRYPTIC,
    Z_SMOOTH_SPECIAL, Z_SMOOTH_LAYOUT, Z_SMOOTH_NEST_2SP,
    Z_IS_INDENTED,
) = range(80, 87)
Z_CODE_SKETCH, Z_COMM_SKETCH, SKETCH_DIM = 112, 152, 40


def _sub_filter(
    kernel: np.ndarray,
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
    gain: float = 12.5,
) -> None:
    """Write an exact ASCII bit-template into conv kernel `channel` for substring `pattern`."""
    span = len(pattern) + wl + wr
    start = (kernel.shape[0] - span) // 2 + wl
    kernel[:, :, channel] = 0.0
    thresh = float(len(pattern))

    if wl:
        kernel[start - 1, HC_ID, channel] = -2.0
    if wr:
        kernel[start + len(pattern), HC_ID, channel] = -2.0
    if lead:
        kernel[start, HC_LEAD, channel] += 1.0
        thresh += 1.0

    for offset, char in enumerate(pattern):
        bits_on = (((ord(char.lower()) & 0x7F) >> _BITS7) & 1) == 1
        kernel[start + offset, HC_BP + _BITS7[bits_on], channel] += 1.0 / 7.0
        kernel[start + offset, HC_BN + _BITS7[~bits_on], channel] += 1.0 / 7.0
        if cs and char.isalpha():
            hi_ch, lo_ch = (HC_UP, HC_LO) if char.isupper() else (HC_LO, HC_UP)
            kernel[start + offset, [hi_ch, lo_ch], channel] += [0.25, -0.50]
            thresh += 0.25

    mid = start + len(pattern) // 2
    if code:
        kernel[mid, HC_COMM, channel] -= 5.0
        if no_str:
            kernel[mid, HC_STR, channel] -= 5.0
    if comm:
        kernel[mid, HC_CODE, channel] -= 5.0
    bias[channel] = -(thresh - 0.08) * gain
    kernel[:, :, channel] *= gain


def _encode_4gram(word: str) -> np.ndarray:
    codes = np.array([ord(ch) & 0x1F for ch in word[:4]], dtype=np.int32)[:, None]
    return np.where(((codes >> np.arange(5)) & 1) == 1, 1.0, -1.0).astype(np.float32)


def _sketch_bank() -> tuple[np.ndarray, np.ndarray]:
    """Build 40 near-orthogonal 4-char SimHash filters (code: 0..39, comments: 40..79)."""
    total_ch = 2 * SKETCH_DIM
    sketch_w = np.zeros((5, N_HC, total_ch), dtype=np.float32)
    sketch_b = np.zeros((total_ch,), dtype=np.float32)
    stop_vecs = [_encode_4gram(word) for word in _STOPWORDS]
    rng = np.random.RandomState(20261115)
    chosen: list[np.ndarray] = []

    while len(chosen) < SKETCH_DIM:
        cand = rng.choice([-1.0, 1.0], size=(4, 5)).astype(np.float32)
        stop_hits = sum(float(np.sum((sv * cand + 1.0) * 0.5)) >= 14.0 for sv in stop_vecs)
        orthogonal = all(float(np.abs(np.sum(cand * prev))) <= 8.0 for prev in chosen)
        if stop_hits <= 1 and orthogonal:
            chosen.append(cand)

    for idx, filt in enumerate(chosen):
        pos_mask = (filt > 0).astype(np.float32) / 20.0
        neg_mask = (filt < 0).astype(np.float32) / 20.0
        pair = [idx, SKETCH_DIM + idx]
        sketch_w[:4, HC_LET, pair] = 2.0
        sketch_w[:4, HC_UND, pair] = 2.0
        for ch_idx in pair:
            sketch_w[:4, HC_BP : HC_BP + 5, ch_idx] = pos_mask
            sketch_w[:4, HC_BN : HC_BN + 5, ch_idx] = neg_mask

        sketch_w[1, [HC_COMM, HC_STR], idx] = -6.0
        sketch_w[1, HC_SL_C, SKETCH_DIM + idx] = 2.0
        sketch_b[pair] = [-(8.0 + 13.7 / 20.0) * 6.0, -(10.0 + 13.7 / 20.0) * 6.0]
        sketch_w[:, :, pair] *= 6.0
    return sketch_w, sketch_b


def _build_block0(params: dict[str, np.ndarray]) -> None:
    """Compile byte/column embeddings and left-to-right quote & comment masking circuits."""
    tok_emb = np.zeros((128, N_EMB), dtype=np.float32)
    tok_emb[1:, EMB_ACTIVE] = 1.0
    tok_emb[33:, EMB_NONSPACE] = 1.0
    tok_emb[32, EMB_SPACE] = 1.0
    tok_emb[97:123, EMB_LOWER] = 1.0
    tok_emb[65:91, EMB_UPPER] = 1.0
    tok_emb[48:58, EMB_DIGIT] = 1.0

    for byte_val in range(1, 128):
        char = chr(byte_val)
        tok_emb[byte_val, EMB_IDENT] = float(char.isalnum() or char in "_$#")
        bits = ((ord(char.lower()) & 0x7F) >> _BITS7) & 1
        tok_emb[byte_val, EMB_ASCII_BITS : EMB_ASCII_BITS + 7] = bits

    for char, dst_col in zip("_/*:?.={}();", range(EMB_UND, EMB_SEMI + 1)):
        tok_emb[ord(char), dst_col] = 1.0
    tok_emb[[ord("|"), ord("&")], EMB_LOGIC] = 1.0
    tok_emb[[ord("'"), ord('"'), ord("`")], EMB_ANY_QUOTE] = 1.0
    for char, dst_col in zip('"\'\\#-', range(EMB_DQUOTE, EMB_DASH + 1)):
        tok_emb[ord(char), dst_col] = 1.0
    tok_emb[[ord("&"), ord("<")], EMB_LIFE_PFX] = 1.0

    col_emb = np.zeros((MAX_COLS, N_EMB), dtype=np.float32)
    col_idx = np.arange(MAX_COLS)
    col_emb[0, EMB_COL0] = 1.0
    col_emb[col_idx % 2 == 1, EMB_COL_ODD] = 1.0
    col_emb[col_idx % 4 == 2, EMB_COL_MOD4_2] = 1.0
    col_emb[(col_idx > 0) & (col_idx % 4 == 0) & (col_idx <= 28), EMB_COL_IND4] = 1.0
    col_thresholds = (
        (8, EMB_COL_GE8), (10, EMB_COL_GE10), (12, EMB_COL_GE12),
        (80, EMB_COL_GE80), (96, EMB_COL_GE96), (110, EMB_COL_GE110),
        (16, EMB_COL_GE16), (20, EMB_COL_GE20),
    )
    for min_col, dst_col in col_thresholds:
        col_emb[col_idx >= min_col, dst_col] = 1.0
    params["tok_emb"], params["col_emb"] = tok_emb, col_emb

    # Local 3-char detectors for comment openers (//, /*, */, # , -- ) and unescaped quotes
    pre_w = np.zeros((3, N_EMB, N_PRE), dtype=np.float32)
    pre_b = np.zeros((N_PRE,), dtype=np.float32)
    pre_w[[1, 2], EMB_SLASH, PRE_SLASH_COMM] = 2.0
    pre_w[0, [EMB_COLON, EMB_STAR, EMB_ANY_QUOTE], PRE_SLASH_COMM] = -4.0
    pre_w[[1, 2], [EMB_SLASH, EMB_STAR], PRE_BLOCK_OPEN] = 2.0
    pre_w[[0, 1, 0], [EMB_STAR, EMB_SLASH, EMB_SLASH], PRE_BLOCK_CLOSE] = [2.0, 2.0, -4.0]
    pre_b[:3] = -3.0

    pre_w[0, EMB_NONSPACE, PRE_HASH_COMM] = -4.0
    pre_w[[1, 2], [EMB_HASH, EMB_SPACE], PRE_HASH_COMM] = 2.0
    pre_b[PRE_HASH_COMM] = -3.0
    pre_w[[0, 1, 2], [EMB_DASH, EMB_DASH, EMB_SPACE], PRE_DASH_COMM] = 2.0
    pre_b[PRE_DASH_COMM] = -5.0

    pre_w[1, EMB_DQUOTE, PRE_DQUOTE] = 1.0
    pre_w[0, [EMB_BACKSLASH, EMB_SQUOTE], PRE_DQUOTE] = -2.0
    pre_w[2, EMB_SQUOTE, PRE_DQUOTE] = -2.0
    pre_w[1, EMB_SQUOTE, PRE_SQUOTE] = 1.0
    pre_w[0, [EMB_BACKSLASH, EMB_LIFE_PFX, EMB_DQUOTE], PRE_SQUOTE] = -2.0
    pre_w[2, EMB_DQUOTE, PRE_SQUOTE] = -2.0

    pre_w[0, [EMB_LOWER, EMB_UPPER], PRE_APOSTROPHE] = 1.0
    pre_w[1, EMB_SQUOTE, PRE_APOSTROPHE] = 1.0
    pre_w[2, [EMB_LOWER, EMB_UPPER], PRE_APOSTROPHE] = 1.0
    pre_b[PRE_APOSTROPHE] = -2.0
    params["pre_conv3_w"], params["pre_conv3_b"] = pre_w, pre_b

    # Strict-left prefix sum c1 tracks leading spaces, quote counts, and inline /* openers
    off_pre = N_EMB
    off_pref1 = N_EMB + N_PRE
    pref1 = np.zeros((off_pref1, N_PREF1), dtype=np.float32)
    pref1_rows = [
        EMB_NONSPACE,
        off_pre + PRE_DQUOTE,
        off_pre + PRE_SQUOTE,
        off_pre + PRE_APOSTROPHE,
        off_pre + PRE_BLOCK_OPEN,
    ]
    pref1_cols = [C1_NONSPACE, C1_DQUOTES, C1_SQUOTES, C1_SQUOTES, C1_BLOCK_OPENS]
    pref1[pref1_rows, pref1_cols] = [1.0, 1.0, 1.0, -1.0, 1.0]
    params["pref1_proj"] = pref1
    params["W_causal1"] = np.tril(np.ones((MAX_COLS, MAX_COLS), dtype=np.float32), k=-1)
    params["W_causal2"] = np.tril(np.ones((MAX_COLS, MAX_COLS), dtype=np.float32), k=0)

    # Gate layer combines [emb, pre_conv3, c1] into indent detectors and quote-parity steps
    gate1_in = off_pref1 + N_PREF1
    gate1_w = np.zeros((gate1_in, N_GATE1), dtype=np.float32)
    gate1_b = np.zeros((N_GATE1,), dtype=np.float32)
    gate1_w[[EMB_SPACE, EMB_NONSPACE], [G1_INDENT_SPACE, G1_FIRST_NONSPACE]] = 1.0
    gate1_w[off_pref1 + C1_NONSPACE, [G1_INDENT_SPACE, G1_FIRST_NONSPACE]] = -2.0
    gate1_w[[EMB_NONSPACE, EMB_STAR, off_pref1 + C1_NONSPACE], G1_DOC_STAR] = [2.0, 2.0, -4.0]
    gate1_b[G1_DOC_STAR] = -3.0

    any_comm = [
        off_pre + PRE_SLASH_COMM, off_pre + PRE_BLOCK_OPEN,
        off_pre + PRE_HASH_COMM, off_pre + PRE_DASH_COMM,
    ]
    line_comm = [off_pre + PRE_SLASH_COMM, off_pre + PRE_HASH_COMM, off_pre + PRE_DASH_COMM]
    gate1_w[any_comm, G1_ANY_COMM_START] = 1.0
    gate1_w[line_comm, G1_LINE_COMM_START] = 1.0

    # Paired ReLU on */ subtracts 1 from comment state only when /* opened earlier on this line
    close_pair = [G1_CLOSE_HI, G1_CLOSE_LO]
    gate1_w[off_pre + PRE_BLOCK_CLOSE, close_pair] = 10.0
    gate1_w[off_pref1 + C1_BLOCK_OPENS, close_pair] = 1.0
    gate1_b[close_pair] = [-10.0, -11.0]

    indent_cols = (
        EMB_COL_ODD, EMB_COL_MOD4_2, EMB_COL_IND4, EMB_COL_GE8,
        EMB_COL_GE10, EMB_COL_GE12, EMB_COL_GE16, EMB_COL_GE20,
    )
    for offset, emb_col in enumerate(indent_cols):
        dst_gate = G1_INDENT_BASE + offset
        gate1_w[[EMB_NONSPACE, emb_col, EMB_STAR, off_pref1 + C1_NONSPACE], dst_gate] = [
            2.0, 2.0, -4.0, -4.0,
        ]
        gate1_b[dst_gate] = -3.0

    gate1_w[off_pref1 + C1_NONSPACE, [G1_LEAD_TWO, G1_LEAD_ONE]] = -1.0
    gate1_b[[G1_LEAD_TWO, G1_LEAD_ONE]] = [2.0, 1.0]
    for step in range(16):
        gate1_w[off_pref1 + C1_DQUOTES, G1_DQUOTE_SAW + step] = 1.0
        gate1_b[G1_DQUOTE_SAW + step] = -float(step)
        gate1_w[off_pref1 + C1_SQUOTES, G1_SQUOTE_SAW + step] = 1.0
        gate1_b[G1_SQUOTE_SAW + step] = -float(step)
    params["gate1_w"], params["gate1_b"] = gate1_w, gate1_b

    pref2 = np.zeros((N_GATE1, N_PREF2), dtype=np.float32)
    pref2_rows = [G1_DOC_STAR, G1_ANY_COMM_START, G1_CLOSE_HI, G1_CLOSE_LO, G1_LINE_COMM_START]
    pref2_cols = [C2_IN_COMMENT, C2_IN_COMMENT, C2_IN_COMMENT, C2_IN_COMMENT, C2_IN_LINE_COMM]
    pref2[pref2_rows, pref2_cols] = [1.0, 1.0, -1.0, 1.0, 1.0]
    params["pref2_proj"] = pref2

    # Fuse [emb, pre_conv3, gate1, c2] into the 52 per-cell HC_* channels
    off_gate1 = off_pref1
    off_pref2 = off_gate1 + N_GATE1
    fuse_w = np.zeros((off_pref2 + N_PREF2, N_HC), dtype=np.float32)
    fuse_b = np.zeros((N_HC,), dtype=np.float32)

    in_comm = off_pref2 + C2_IN_COMMENT
    in_line_comm = off_pref2 + C2_IN_LINE_COMM
    fuse_w[EMB_ACTIVE, [HC_ACT, HC_COMM, HC_SL_C, HC_CODE]] = 1.0
    fuse_w[in_comm, [HC_COMM, HC_CODE, HC_CNS]] = [1.0, -2.0, -2.0]
    fuse_w[in_line_comm, [HC_SL_C, HC_MNS]] = 1.0
    fuse_w[EMB_NONSPACE, [HC_CNS, HC_MNS]] = 1.0
    fuse_w[EMB_SLASH, HC_MNS] = -1.0
    fuse_b[[HC_COMM, HC_SL_C, HC_MNS]] = -1.0
    fuse_w[[off_gate1 + G1_INDENT_SPACE, off_gate1 + G1_FIRST_NONSPACE], [HC_IND, HC_FNS]] = 1.0

    sawtooth = np.array([1.0] + [-2.0, 2.0] * 7 + [-2.0], dtype=np.float32)
    dquote_slice = slice(off_gate1 + G1_DQUOTE_SAW, off_gate1 + G1_DQUOTE_SAW + 16)
    squote_slice = slice(off_gate1 + G1_SQUOTE_SAW, off_gate1 + G1_SQUOTE_SAW + 16)
    fuse_w[dquote_slice, HC_STR] = sawtooth
    fuse_w[squote_slice, HC_STR] = sawtooth
    fuse_w[in_comm, HC_STR] = -3.0

    lead_srcs = [EMB_NONSPACE, off_gate1 + G1_LEAD_TWO, off_gate1 + G1_LEAD_ONE, in_comm]
    fuse_w[lead_srcs, HC_LEAD] = [1.0, 1.0, -1.0, -2.0]
    fuse_b[HC_LEAD] = -1.0

    indent_hc_dsts = [HC_ODD, HC_M42, HC_M40, HC_D8, HC_D10, HC_D12, HC_D16, HC_D20]
    for offset, hc_dst in enumerate(indent_hc_dsts):
        fuse_w[[off_gate1 + G1_INDENT_BASE + offset, in_comm], hc_dst] = [1.0, -2.0]
    width_gates = ((EMB_COL_GE80, HC_G80), (EMB_COL_GE96, HC_G96), (EMB_COL_GE110, HC_G110))
    for col_src, hc_dst in width_gates:
        fuse_w[[EMB_NONSPACE, col_src, in_comm], hc_dst] = [1.0, 1.0, -2.0]
        fuse_b[hc_dst] = -1.0

    char_srcs = [
        EMB_IDENT, EMB_LOWER, EMB_UPPER, EMB_DIGIT, EMB_UND, EMB_DOT, EMB_QMARK, EMB_COLON,
        EMB_EQ, EMB_LBRACE, EMB_RBRACE, EMB_LPAREN, EMB_RPAREN, EMB_SEMI, EMB_LOGIC, EMB_ANY_QUOTE,
    ]
    fuse_w[char_srcs, list(range(HC_ID, HC_QT + 1))] = 1.0
    fuse_w[[EMB_LOWER, EMB_UPPER], HC_LET] = 1.0
    fuse_w[EMB_ASCII_BITS + _BITS7, HC_BP + _BITS7] = 1.0
    fuse_w[EMB_ACTIVE, HC_BN + _BITS7] = 1.0
    fuse_w[EMB_ASCII_BITS + _BITS7, HC_BN + _BITS7] = -1.0
    params["col_fuse_w"], params["col_fuse_b"] = fuse_w, fuse_b


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


def _build_block1(params: dict[str, np.ndarray]) -> None:
    """Compile k=3, 7, 13 lexical convolutions, SimHash bank, and line-level feature MLP."""
    feat: dict[str, int] = {}
    n_csum, n_conv3, n_conv7, n_conv13 = 16, 64, 96, 64
    off_c3 = n_csum
    off_c7 = off_c3 + n_conv3
    off_c13 = off_c7 + n_conv7

    col_sum_proj = np.zeros((N_HC, n_csum), dtype=np.float32)
    for idx, (name, hc_col, scale) in enumerate(_CSUM_SPECS):
        col_sum_proj[hc_col, idx] = scale
        feat[name] = idx
    params["col_sum_proj"] = col_sum_proj

    # k=3: naming case transitions, 1-char / 3+-char identifiers, ternaries, short operators
    conv3_w = np.zeros((3, N_HC, n_conv3), dtype=np.float32)
    conv3_b = np.zeros((n_conv3,), dtype=np.float32)
    conv3_w[[0, 1, 2], [HC_LO, HC_UND, HC_LO], 0] = 2.0
    conv3_w[[0, 1], [HC_LO, HC_UP], 1] = 2.0
    conv3_w[0, [HC_ID, HC_DOT, HC_QT], 2] = -4.0
    conv3_w[1, HC_LO, 2] = 2.0
    conv3_w[2, [HC_ID, HC_QT], 2] = -4.0

    conv3_w[[0, 1, 2], [HC_ID, HC_LO, HC_ID], 3] = 2.0
    conv3_w[0, [HC_ID, HC_QM], 4] = [-2.0, -4.0]
    conv3_w[1, HC_QM, 4] = 2.0
    conv3_w[2, [HC_DOT, HC_QM, HC_EQ, HC_SEM, HC_RP], 4] = -4.0
    conv3_w[1, HC_COMM, :5] = -6.0
    conv3_w[1, HC_STR, :5] = -6.0
    conv3_b[:5] = [-5.0, -3.0, -1.0, -5.0, -1.0]
    for ch_i, name in enumerate(
        ("snake_case", "camel_case", "one_char_id", "three_char_id", "ternary_op")
    ):
        feat[name] = off_c3 + ch_i

    for ch_i, pat in enumerate(("?.", "=>", "===", "{}", "||", "#", "??="), 5):
        _sub_filter(conv3_w, conv3_b, ch_i, pat)
        feat[pat] = off_c3 + ch_i

    conv3_w[1, [HC_SEM, HC_EQ, HC_COMM, HC_STR], 12] = [1.0, 1.0, -5.0, -5.0]
    conv3_w[1, HC_D16, 13] = 1.0
    conv3_w[1, HC_D20, 14] = 1.0
    conv3_w[[0, 1], [HC_ID, HC_LP], 15] = 2.0
    conv3_w[1, [HC_COMM, HC_STR], 15] = -6.0
    conv3_b[15] = -3.0
    for ch_i, name in enumerate(("semi_or_eq", "depth_ge16", "depth_ge20", "func_call"), 12):
        feat[name] = off_c3 + ch_i
    params["conv3_w"], params["conv3_b"] = conv3_w, conv3_b

    # k=7: control flow, error/ceremony tokens, narrator comment cues, and v5 anti-golf filters
    conv7_w = np.zeros((7, N_HC, n_conv7), dtype=np.float32)
    conv7_b = np.zeros((n_conv7,), dtype=np.float32)
    str_bearing = {"|| '')", "strict'"}
    for start_ch, group in ((0, _K7_PRIMARY), (60, _K7_EXTENDED)):
        for offset, (pat, wl, wr, cs, ld) in enumerate(group):
            ch_i = start_ch + offset
            _sub_filter(
                conv7_w, conv7_b, ch_i, pat,
                wl=wl, wr=wr, cs=cs, lead=ld, no_str=(pat not in str_bearing),
            )
            feat[pat] = off_c7 + ch_i

    ch_eq_digit = 46
    conv7_w[[1, 2, 3, 5], [HC_EQ, HC_EQ, HC_EQ, HC_DIG], ch_eq_digit] = 2.0
    conv7_w[2, [HC_COMM, HC_STR], ch_eq_digit] = -8.0
    conv7_b[ch_eq_digit] = -7.0
    feat["===_digit"] = off_c7 + ch_eq_digit

    comm_cues_a = (
        "step 1", "step 2", "step 3", "todo:", "fixme:",
        "eslint-", "@ts-ign", "initial", "loop th", "check i",
    )
    comm_cues_b = ("type: i", "noqa", "nolint")
    for start_ch, group in ((50, comm_cues_a), (78, comm_cues_b)):
        for offset, pat in enumerate(group):
            ch_i = start_ch + offset
            _sub_filter(conv7_w, conv7_b, ch_i, pat, code=False, comm=True)
            feat[pat] = off_c7 + ch_i

    # 2-char leading LHS variable assignment (`xx = ...`) outside strings and comments
    ch_short_lhs, ch_multi_dig, ch_magic_idx = 81, 82, 83
    conv7_w[1, [HC_LO, HC_LEAD], ch_short_lhs] = 2.0
    conv7_w[2, [HC_ID, HC_LEAD, HC_COMM, HC_STR], ch_short_lhs] = [2.0, 2.0, -6.0, -6.0]
    conv7_w[3, [HC_ACT, HC_CNS], ch_short_lhs] = [1.0, -4.0]
    conv7_w[4, HC_EQ, ch_short_lhs] = 2.0
    conv7_w[5, HC_EQ, ch_short_lhs] = -4.0
    conv7_b[ch_short_lhs] = -10.5

    # Standalone 2+-digit integer literal (excludes identifier suffixes like float32 / k13)
    conv7_w[1, [HC_ID, HC_DOT], ch_multi_dig] = -4.0
    conv7_w[2, [HC_DIG, HC_COMM, HC_STR], ch_multi_dig] = [2.0, -6.0, -6.0]
    conv7_w[3, HC_DIG, ch_multi_dig] = 2.0
    conv7_b[ch_multi_dig] = -3.5

    # Raw multi-digit bracket index (`[46`, `[112`, `[78:94`)
    conv7_w[1, HC_LB, ch_magic_idx] = 2.0
    conv7_w[2, [HC_DIG, HC_COMM, HC_STR], ch_magic_idx] = [2.0, -6.0, -6.0]
    conv7_w[3, HC_DIG, ch_magic_idx] = 2.0
    conv7_b[ch_magic_idx] = -5.5

    golf_chs = [ch_short_lhs, ch_multi_dig, ch_magic_idx]
    conv7_w[:, :, golf_chs] *= 2.0
    conv7_b[golf_chs] *= 2.0
    feat["short_lhs"] = off_c7 + ch_short_lhs
    feat["multi_digit"] = off_c7 + ch_multi_dig
    feat["magic_index"] = off_c7 + ch_magic_idx
    params["conv7_w"], params["conv7_b"] = conv7_w, conv7_b

    # k=13: long identifiers, enterprise suffixes, module headers, and linter suppressions
    conv13_w = np.zeros((13, N_HC, n_conv13), dtype=np.float32)
    conv13_b = np.zeros((n_conv13,), dtype=np.float32)
    conv13_w[:, HC_ID, 0] = 2.0
    conv13_w[6, [HC_COMM, HC_STR], 0] = -30.0
    conv13_b[0] = -25.0
    feat["ident_13ch"] = off_c13

    lead_k13 = {"import ", "package ", "#include", "except Excep", "except BaseE"}
    for start_ch, group in ((1, _K13_PRIMARY), (22, _K13_EXTENDED)):
        for offset, (pat, cs) in enumerate(group):
            ch_i = start_ch + offset
            _sub_filter(
                conv13_w, conv13_b, ch_i, pat,
                cs=cs, lead=int(pat in lead_k13), no_str=(pat != "internal/"),
            )
            feat[pat] = off_c13 + ch_i

    for ch_i, pat in ((20, "eslint-disabl"), (21, "@ts-ignore")):
        _sub_filter(conv13_w, conv13_b, ch_i, pat, code=False, comm=True)
        feat[pat] = off_c13 + ch_i
    params["conv13_w"], params["conv13_b"] = conv13_w, conv13_b
    params["sketch_w"], params["sketch_b"] = _sketch_bank()

    # Project 240 pooled line features into named Z[0..39] registers via lmlp
    n_u = n_csum + n_conv3 + n_conv7 + n_conv13
    lm1_w, lm1_b = np.zeros((n_u, 96), dtype=np.float32), np.zeros((96,), dtype=np.float32)
    lm2_w, lm2_b = np.zeros((96, 112), dtype=np.float32), np.zeros((112,), dtype=np.float32)

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

        # Indentation geometry, module headers, and control-flow structure
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

        # Naming bloat, error-handling signals, and lint/fallback suppressions
        (ceremony_terms, 0.0, Z_CEREMONY, 1.0),
        (stdlib_idiom_terms, 0.0, Z_STDLIB_IDIOM, 1.0),
        ([("ident_13ch", 0.11)] + [(k, -w * 0.25) for k, w in stdlib_idiom_terms],
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

        # Chaining, line width, narrator comments, and statement rhythm
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
    ]

    short_golf_terms = [("short_lhs", 1.5), ("one_char_id", 0.5)]
    magic_num_terms = [("magic_index", 1.0), ("multi_digit", 0.30)]
    line_rules.extend([
        (short_golf_terms, -0.5, Z_SHORT_GOLF, 1.0),
        (short_golf_terms, -1.5, Z_SHORT_GOLF, -1.0),
        (magic_num_terms, -0.95, Z_MAGIC_NUM, 1.0),
        (magic_num_terms, -1.95, Z_MAGIC_NUM, -1.0),
        ([("indent_cols", 2.0)], 0.0, Z_IS_INDENTED, 1.0),
        ([("indent_cols", 2.0)], -1.0, Z_IS_INDENTED, -1.0),
    ])

    for unit, (terms, bias, dst_z, scale) in enumerate(line_rules):
        for name, weight in terms:
            lm1_w[feat[name], unit] += weight
        lm1_b[unit] = bias
        lm2_w[unit, dst_z] += scale
    params["lmlp1_w"], params["lmlp1_b"], params["lmlp2_w"], params["lmlp2_b"] = (
        lm1_w, lm1_b, lm2_w, lm2_b,
    )


def _build_block2_and_heads(params: dict[str, np.ndarray]) -> None:
    """Compile cross-line sink-attention heads, [0, 1] issue clamps, and System One readout."""
    center = MAX_REL_DIST
    n_rel = 2 * center + 1
    a1_wq = np.zeros((N_Z, 128), dtype=np.float32)
    a1_wk = np.zeros((N_Z, 128), dtype=np.float32)
    a1_wv = np.zeros((N_Z, 96), dtype=np.float32)
    a1_wo = np.zeros((96, N_Z), dtype=np.float32)
    a1_br = np.full((8, n_rel), -1e4, dtype=np.float32)
    a1_bs = np.array([4.5, 1.2, 1.5, 2.5, 9.05, 11.1, -12.0, -12.0], dtype=np.float32)

    # Head 0: comment-to-next-line code SimHash overlap (comments restating L+1..L+2)
    qr_mat, _ = np.linalg.qr(np.random.RandomState(777).randn(SKETCH_DIM, SKETCH_DIM))
    sketch_proj = (qr_mat[:, :16] * np.sqrt(SKETCH_DIM / 16.0)).astype(np.float32)
    comm_slice = slice(Z_COMM_SKETCH, Z_COMM_SKETCH + SKETCH_DIM)
    code_slice = slice(Z_CODE_SKETCH, Z_CODE_SKETCH + SKETCH_DIM)

    a1_wq[comm_slice, :16] = sketch_proj * float(np.sqrt(60.0))
    a1_wk[code_slice, :16] = sketch_proj * float(np.sqrt(60.0))
    a1_br[0, [center + 1, center + 2]] = [0.0, -1.2]
    a1_wv[Z_HAS_CODE, 0] = a1_wo[0, Z_RESTATES_CODE] = 1.0

    # Head 1: try/catch/except opener matching real vs. empty/fallback handler on L..L+3
    a1_wq[Z_TRY_CATCH, [16, 17]] = [float(np.sqrt(32.0)), float(np.sqrt(16.0))]
    a1_wk[[Z_REAL_HANDLER, Z_EMPTY_HANDLER], [16, 17]] = [
        float(np.sqrt(32.0)), float(np.sqrt(16.0)),
    ]
    a1_br[1, center : center + 4] = 0.0

    # Head 2: causal lookback (L-8..L) counting enclosing branches vs. guard-clause exits
    a1_br[2, center - 8 : center + 1] = 0.0
    h2_srcs = [
        Z_DEPTH10, Z_DEPTH12, Z_BRANCH_OPEN, Z_EARLY_EXIT,
        Z_GUARD_CLAUSE, Z_CLEAN_IDIOM, Z_DEEP_PYRAMID,
    ]
    a1_wv[h2_srcs, range(24, 31)] = [1.0, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0]
    a1_wo[[24, 25], Z_ENCLOSING_DEPTH] = [0.8, 1.0]
    a1_wo[26, Z_ENCLOSING_BRANCHES] = 1.0
    a1_wo[[27, 28, 29], Z_ENCLOSING_GUARDS] = [1.0, 0.8, 0.8]
    a1_wo[30, Z_ENCLOSING_DEEP16] = 1.0

    # Head 3 (L-9..L+9), Head 6 (file-wide), and Head 7 (L-3..L+3) window masks
    a1_br[3, center - 9 : center + 10] = 0.0
    a1_br[6, :] = 0.0
    a1_br[7, center - 3 : center + 4] = 0.0
    a1_wv[[Z_HARDCODED_CASE, Z_SWITCH_BLOCK], 39] = [1.53, -2.5 * 1.53]
    a1_wo[39, Z_LOCAL_SPECIAL_CASES] = 1.0
    a1_wv[[Z_CHAINED_CALLS, Z_NESTED_TERNARY, Z_CEREMONY], [84, 84, 85]] = 1.0
    a1_wo[[84, 85], [Z_LOCAL_CHAINING, Z_LOCAL_CEREMONY]] = 1.0

    direct_copies = (
        (
            [Z_REAL_HANDLER, Z_EMPTY_HANDLER],
            [12, 13],
            [Z_NEAR_REAL_HANDLER, Z_NEAR_EMPTY_HANDLER],
        ),
        (
            [Z_IS_STMT, Z_PARAGRAPH_BREAK, Z_ODD_INDENT],
            [36, 37, 38],
            [Z_LOCAL_STMT_DENSITY, Z_LOCAL_BREAK_DENSITY, Z_LOCAL_ODD_INDENT],
        ),
        (
            [Z_ONE_CHAR_ID, Z_WORD_COUNT, Z_MOD4_EQ0, Z_MOD4_EQ2, Z_ODD_INDENT],
            list(range(72, 77)),
            [Z_FILE_ONE_CHAR, Z_FILE_WORDS, Z_FILE_MOD4_0, Z_FILE_MOD4_2, Z_FILE_ODD_IND],
        ),
        (
            [Z_SHORT_GOLF, Z_MAGIC_NUM],
            [86, 87],
            [Z_LOCAL_SHORT_GOLF, Z_LOCAL_MAGIC_NUM],
        ),
    )
    for srcs, slots, dsts in direct_copies:
        a1_wv[srcs, slots] = 1.0
        a1_wo[slots, dsts] = 1.0

    # Heads 4 & 5: distant clone match (8 <= |d| < 32) minus adjacent table match (1 <= |d| <= 3)
    abs_offset = np.abs(np.arange(-center, center + 1))
    a1_wq[code_slice, 64:80] = sketch_proj * 8.0
    a1_wk[code_slice, 64:80] = sketch_proj * 8.0
    a1_br[4, (abs_offset >= 8) & (abs_offset < center)] = 0.0
    a1_wv[Z_IMPERATIVE_LINE, 48] = a1_wo[48, Z_DISTANT_CLONE] = 1.0

    a1_wq[code_slice, 80:96] = sketch_proj * 8.0
    a1_wk[code_slice, 80:96] = sketch_proj * 8.0
    a1_br[5, (abs_offset >= 1) & (abs_offset <= 3)] = 0.0
    a1_wv[Z_IMPERATIVE_LINE, 60] = 1.0
    a1_wo[60, Z_DISTANT_CLONE] = -0.80
    params["attn1_wq"], params["attn1_wk"], params["attn1_wv"], params["attn1_wo"] = (
        a1_wq, a1_wk, a1_wv, a1_wo,
    )
    params["attn1_brel"], params["attn1_bsink"] = a1_br, a1_bs

    # Context MLP 1: paired ReLU clamps ReLU(x) - ReLU(x - 1) write [0, 1] issue flags
    cm1_w, cm1_b = np.zeros((N_Z, 96), dtype=np.float32), np.zeros((96,), dtype=np.float32)
    cm1_2w, cm1_2b = np.zeros((96, N_Z), dtype=np.float32), np.zeros((N_Z,), dtype=np.float32)
    ctx1_rules: list[tuple[list[tuple[int, float]], float, int, float]] = [
        ([(Z_HAS_CODE, 0.4), (Z_DEPTH12, 1.1), (Z_DEPTH10, 0.6), (Z_ENCLOSING_BRANCHES, 5.2),
          (Z_ELSE_BRANCH, 0.5), (Z_ENCLOSING_GUARDS, -3.0), (Z_OPEN_NET, -1.2)],
         -2.65, Z_RAW_NESTING_2SP, 2.5),
        ([(Z_HAS_CODE, 0.4), (Z_DEEP_PYRAMID, 1.1), (Z_ENCLOSING_DEEP16, 1.6),
          (Z_ENCLOSING_BRANCHES, 5.2), (Z_ELSE_BRANCH, 0.5),
          (Z_ENCLOSING_GUARDS, -3.0), (Z_OPEN_NET, -1.2)],
         -3.25, Z_ISS_NESTING, 2.5),
        ([(Z_FILE_MOD4_0, 20.0), (Z_FILE_MOD4_2, -50.0), (Z_FILE_ODD_IND, -40.0)],
         -0.5, Z_USES_WIDE_INDENT, 2.0),
        ([(Z_FILE_ODD_IND, 20.0), (Z_FILE_MOD4_2, -30.0)], -4.0, Z_USES_WIDE_INDENT, 2.0),

        ([(Z_CEREMONY, 0.85), (Z_LONG_IDENT, 0.75), (Z_LOCAL_CEREMONY, 0.65),
          (Z_STDLIB_IDIOM, -2.5)], -1.05, Z_ISS_NAMING, 2.2),
        ([(Z_ONE_CHAR_ID, 0.55), (Z_FILE_ONE_CHAR, 1.5), (Z_FILE_WORDS, -2.0),
          (Z_WORD_COUNT, -0.8), (Z_HAS_CODE, 0.4)], -1.45, Z_RAW_CRYPTIC, 2.2),
        ([(Z_PURE_COMMENT, 1.0), (Z_SHORT_COMMENT, 1.0), (Z_RESTATES_CODE, 2.4)],
         -3.35, Z_ISS_COMMENT, 2.5),
        ([(Z_NARRATOR_TAG, 1.5)], -0.45, Z_ISS_COMMENT, 2.2),
        ([(Z_TRY_CATCH, 1.8), (Z_NEAR_EMPTY_HANDLER, 1.4), (Z_NEAR_REAL_HANDLER, -4.0)],
         -2.1, Z_ISS_ERROR, 2.5),
        ([(Z_FALLBACK_SWALLOW, 1.4)], -0.35, Z_ISS_ERROR, 2.2),

        ([(Z_CHAINED_CALLS, 1.2), (Z_NESTED_TERNARY, 1.25), (Z_OVER_96, 1.1),
          (Z_OVER_80, 0.25), (Z_STDLIB_IDIOM, -0.9), (Z_MODULE_HEADER, -3.0)],
         -1.0, Z_ISS_DENSE, 2.2),
        ([(Z_HARDCODED_CASE, 1.5), (Z_LOCAL_SPECIAL_CASES, 8.5), (Z_SWITCH_BLOCK, -3.0)],
         -2.75, Z_ISS_SPECIAL, 2.5),
        ([(Z_IMPERATIVE_LINE, 1.5), (Z_DISTANT_CLONE, 2.0),
          (Z_IS_INDENTED, 2.0), (Z_IS_STMT, 2.0)], -6.45, Z_RAW_CLONE, 2.5),
        ([(Z_IS_STMT, 1.0), (Z_LOCAL_STMT_DENSITY, 14.2), (Z_LOCAL_BREAK_DENSITY, -18.0)],
         -8.55, Z_ISS_LAYOUT, 3.0),
        ([(Z_IS_STMT, 1.0), (Z_ODD_INDENT, 1.8), (Z_FILE_ODD_IND, -7.5)],
         -2.0, Z_ISS_LAYOUT, 2.5),
        ([(Z_EARLY_EXIT, 0.6), (Z_GUARD_CLAUSE, 0.7), (Z_CLEAN_IDIOM, 0.6),
          (Z_MOD4_EQ2, 0.35), (Z_WORD_COUNT, 0.25)], -0.25, Z_CLEAN_BONUS, 1.2),
        ([(Z_SHORT_GOLF, 0.6), (Z_LOCAL_SHORT_GOLF, 1.8)], -1.35, Z_ISS_NAMING, 2.5),
        ([(Z_MAGIC_NUM, 0.9), (Z_LOCAL_MAGIC_NUM, 2.2)], -0.78, Z_ISS_NAMING, 2.5),
    ]

    for idx, (terms, bias, dst_z, gain) in enumerate(ctx1_rules):
        unit = 2 * idx
        for src_z, weight in terms:
            cm1_w[src_z, [unit, unit + 1]] += weight * gain
        cm1_b[[unit, unit + 1]] = [bias * gain, bias * gain - 1.0]
        cm1_2w[[unit, unit + 1], dst_z] += [1.0, -1.0]
    params["cmlp1_w"], params["cmlp1_b"], params["cmlp1_2_w"], params["cmlp1_2_b"] = (
        cm1_w, cm1_b, cm1_2w, cm1_2b,
    )

    # Pass 2 attention smooths multi-line issues across immediate neighbor lines
    a2_wq = np.zeros((N_Z, 64), dtype=np.float32)
    a2_wk = np.zeros((N_Z, 64), dtype=np.float32)
    a2_wv = np.zeros((N_Z, 64), dtype=np.float32)
    a2_wo = np.zeros((64, N_Z), dtype=np.float32)
    a2_br = np.full((4, n_rel), -1e4, dtype=np.float32)
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
    a2_wv[smooth_srcs, smooth_slots] = 1.0
    a2_wo[smooth_slots, smooth_dsts] = 1.0
    params["attn2_wq"], params["attn2_wk"], params["attn2_wv"], params["attn2_wo"] = (
        a2_wq, a2_wk, a2_wv, a2_wo,
    )
    params["attn2_brel"] = a2_br
    params["attn2_bsink"] = np.array([0.7, -12.0, -12.0, -12.0], dtype=np.float32)

    cm2_w, cm2_b = np.zeros((N_Z, 64), dtype=np.float32), np.zeros((64,), dtype=np.float32)
    cm2_2w, cm2_2b = np.zeros((64, N_Z), dtype=np.float32), np.zeros((N_Z,), dtype=np.float32)
    cm2_w[Z_RAW_CLONE, [0, 1]] = 2.0
    cm2_w[Z_SMOOTH_CLONE, [0, 1]] = 5.0
    cm2_w[Z_RAW_CRYPTIC, [2, 3]] = 1.6
    cm2_w[Z_SMOOTH_CRYPTIC, [2, 3]] = 3.0
    cm2_b[:4] = [-3.35, -4.35, -2.0, -3.0]
    cm2_2w[[0, 1], Z_ISS_CLONE] = [1.0, -1.0]
    cm2_2w[[2, 3], Z_ISS_NAMING] = [1.0, -1.0]

    # Gate 2-space pyramid flags off whenever file-wide statistics show 3- or 4-space indent
    cm2_w[Z_RAW_NESTING_2SP, [4, 5]] = 1.0
    cm2_w[Z_USES_WIDE_INDENT, [4, 5]] = -3.0
    cm2_b[[4, 5]] = [0.0, -1.0]
    cm2_2w[[4, 5], Z_ISS_NESTING] = [1.0, -1.0]

    cm2_w[Z_SMOOTH_NEST_2SP, [6, 7]] = 1.0
    cm2_w[Z_USES_WIDE_INDENT, [6, 7]] = -3.0
    cm2_b[[6, 7]] = [0.0, -1.0]
    cm2_2w[[6, 7], Z_SMOOTH_NEST] = [1.0, -1.0]
    params["cmlp2_w"], params["cmlp2_b"], params["cmlp2_2_w"], params["cmlp2_2_b"] = (
        cm2_w, cm2_b, cm2_2w, cm2_2b,
    )

    # Block 3 readout: 8 squiggle heads + gated document pooling into 31 System One logits
    issue_regs = [
        Z_ISS_NESTING, Z_ISS_NAMING, Z_ISS_COMMENT, Z_ISS_ERROR,
        Z_ISS_DENSE, Z_ISS_SPECIAL, Z_ISS_CLONE, Z_ISS_LAYOUT,
    ]
    sq_w = np.zeros((N_Z, 8), dtype=np.float32)
    sq_b = np.full((8,), -2.7, dtype=np.float32)
    sq_w[issue_regs, range(8)] = [4.2, 4.6, 4.8, 4.7, 4.7, 4.3, 4.8, 4.3]
    sq_w[[Z_SMOOTH_NEST, Z_SMOOTH_SPECIAL, Z_SMOOTH_LAYOUT], [0, 5, 7]] = [1.2, 1.2, 1.0]
    params["sq_w"], params["sq_b"] = sq_w, sq_b

    pool_gate_w = np.zeros((N_Z, 1), dtype=np.float32)
    pool_gate_w[Z_ACTIVE, 0] = 12.0
    params["pool_gate_w"] = pool_gate_w
    params["pool_gate_b"] = np.array([-6.0], dtype=np.float32)

    doc1_w = np.zeros((2 * N_Z, 64), dtype=np.float32)
    doc1_b = np.zeros((64,), dtype=np.float32)
    doc2_w = np.zeros((64, 31), dtype=np.float32)
    doc2_b = np.zeros((31,), dtype=np.float32)

    for head_i, reg in enumerate(issue_regs):
        doc1_w[[reg, N_Z + reg], head_i] = [4.5, 0.65]
        doc1_b[head_i] = -0.20
    doc1_w[[Z_CLEAN_BONUS, Z_MOD4_EQ2], 8] = [1.4, 0.4]
    doc1_w[[Z_RAW_CRYPTIC, N_Z + Z_RAW_CRYPTIC], 9] = [3.5, 0.6]
    doc1_w[[Z_CEREMONY, Z_LONG_IDENT], 10] = [2.0, 1.5]
    doc1_w[[Z_CLEAN_IDIOM, Z_EARLY_EXIT, Z_GUARD_CLAUSE], 11] = [2.0, 1.2, 1.2]

    doc2_b[:2] = [1.45, 1.85]
    doc2_w[:8, 0] = [-2.1, -2.1, -1.9, -2.2, -2.2, -2.2, -2.1, -2.0]
    doc2_w[:8, 1] = [-1.2, -1.0, -2.2, -2.5, -1.1, -2.6, -2.3, -1.3]
    doc2_w[8, :2] = [0.85, 0.45]

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
            doc2_w[unit_i, base : base + 4] += [
                2.1 * mult, 0.8 * mult, -0.7 * mult, -2.2 * mult,
            ]
        doc2_w[8, [base, base + 3]] += [-0.45, 0.45]

    style_base = 26
    doc2_b[style_base : style_base + 5] = [1.25, 1.35, -2.10, -2.10, -2.20]
    doc2_w[:8, style_base] -= 1.50
    doc2_w[:8, style_base + 1] -= 1.25
    doc2_w[[11, 8], style_base] += [1.3, 0.5]
    doc2_w[8, style_base + 1] += 0.3
    doc2_w[10, style_base + 2] += 3.5
    doc2_w[[4, 9, 0], style_base + 3] += [3.3, 3.3, 1.1]
    doc2_w[[2, 3, 5, 6, 7, 0], style_base + 4] += [2.9, 3.1, 3.3, 3.0, 2.2, 1.1]
    params["doc1_w"], params["doc1_b"], params["doc2_w"], params["doc2_b"] = (
        doc1_w, doc1_b, doc2_w, doc2_b,
    )


def compile_weights(output_path: Path | None = None) -> dict[str, np.ndarray]:
    params: dict[str, np.ndarray] = {}
    _build_block0(params)
    _build_block1(params)
    _build_block2_and_heads(params)
    target = output_path or (Path(__file__).with_name("codehygiene") / "weights.npz")
    np.savez_compressed(target, **params)
    return params


if __name__ == "__main__":
    out_file = Path(__file__).with_name("codehygiene") / "weights.npz"
    tensors = compile_weights(out_file)
    size_kb = out_file.stat().st_size / 1024
    total_params = sum(arr.size for arr in tensors.values())
    print(f"Compiled {len(tensors)} tensors ({total_params:,} params, {size_kb:.1f} KB)")
