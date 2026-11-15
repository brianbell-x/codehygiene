# AGENTS.md

## Using `codehygiene` as a Coding Agent

1. **Install as an offline agent skill (macOS, Linux, Windows):**
   ```bash
   npx skills add brianbell-x/codehygiene -g
   ```
   Or run directly without installing via `uvx`:
   ```bash
   uvx --from git+https://github.com/brianbell-x/codehygiene codehygiene . --json
   ```
2. **Gate edits in agent loops or CI:**
   ```bash
   codehygiene path/to/changed_files --check --no-color
   ```
   Exit code `0` indicates zero squiggles at or above `--threshold` (default `0.55`); exit code `1` indicates at least one squiggle requires refactoring.
3. **Consume structured diagnostics:**
   Pass `--json` to parse `score`, `rl_hygiene`, `scores`, `archetype`, and `squiggles` (`line`, `col_start`, `col_end`, `probability`, `kind`, `message`). Passing a single file or `stdin` emits one JSON object, whereas passing a directory or multiple files emits a `{path: Result}` map.
4. **Shape RL coding rewards:**
   Call `codehygiene.rl_reward(code, tests_passed=passed, weight=0.25)` inside GRPO / PPO / verifier environments. Functional test failure always yields `0.0`.
5. **Report impact and edge cases:**
   Capture baseline scores and squiggle counts before editing and re-score after refactoring. Summarize the before/after `score` and squiggle deltas in the handoff, noting which diagnostics genuinely improved the code and flagging any false positives or missed patterns.

## Modifying or Porting This Repository

- **Repository map (OS-independent pure Python + NumPy across macOS, Linux, and Windows):**
  - `codehygiene/__init__.py` — Pure-NumPy neural inference (`_forward`), 5-trit archive unpacker (`unpack_ternary_archive`), byte-grid encoder (`encode_byte_grid`), Python API (`score`, `lint`, `rl_reward`), and CLI (`main`).
  - `codehygiene/__main__.py` — Supports both `python3 -m codehygiene` and direct PEP 338 skill-directory execution (`python3 "<SKILL_DIR>"`).
  - `codehygiene/SKILL.md` — Agent skill definition discovered directly by `npx skills add brianbell-x/codehygiene`.
  - `compile.py` — Deterministic Weight-Space Program that compiles all `32` `{-1, 0, +1}` matrices (`569,662` trits packed at 5 trits/byte) and `4,103` 1D scale/bias scalars into `codehygiene/weights.npz`.
  - `codehygiene/weights.npz` — Compiled `12.97 KiB` (`1.58-bit` ternary) checkpoint loaded at runtime.
  - `docs/` — Zero-backend client-side browser playground unpacking `weights.npz` directly in JavaScript.
  - `corpus/` — Four-language `GPT-OSS 120B` `before/` vs. `after/` agent benchmark suites (`billing_python`, `gateway_js`, `checkout_ts`, `matching_rust`).
  - `corpus/evaluate.py` — Cross-platform benchmark runner computing `codehygiene` scores, `rl_reward` deltas, and executing all `64` `before`/`after` unit tests.
- **Architecture pipeline (`_forward`):**
  1. **Block 0 (Column Prefix & Masking):** `tok_emb (128, 32)` + `col_emb (128, 32)` -> `pre_conv3` -> causal prefix `W_causal1` -> `gate1` (16-quote sawtooth string parity + comment openers) -> `W_causal2` -> `col_fuse` (`52` channels).
  2. **Block 1 (Line Convolutions & Sketches):** Multi-scale column `Conv1D` (`k=3, 7, 13`) + 4-gram SimHash `sketch` (`40` code + `40` comment dims, L2-normalized) + `lmlp1/2` -> `Z` (`L, 226`).
  3. **Block 2 (Cross-Line Sink Attention):** Two residual blocks of 8-head relative-position attention (`[-32..+32]` + sink logits `bsink`) and context MLPs (`cmlp1`, `cmlp2`) computing narrator-comment overlap, swallowed-error spans, scale-invariant indentation depth, golfed short-name/magic-index clusters, rhythm walls, and copy-paste clones.
  4. **Block 3 (System One Heads):** Per-line squiggle head (`sq_w`, 8 kinds) + gated mean/max pooling into `doc1/doc2` (`score`, `rl_hygiene`, 6 four-level sub-scores, 5 style archetypes).
- **Hard invariants:**
  - Never hand-edit `codehygiene/weights.npz`; express all behavior changes in `compile.py` and run `python3 compile.py`.
  - Keep `_forward` strictly neural (matrix multiplies, 1D convolutions, elementwise activations, attention). Do not introduce regexes, tokenizers, or AST parsers into inference.
  - Keep runtime dependencies limited to `numpy` with zero OS-specific or native extensions.
  - Verify self-hygiene and corpus gates before committing (`corpus/*/before/` intentionally retains unrefactored Phase 1 baselines):
    ```bash
    python3 compile.py
    python3 -m codehygiene codehygiene compile.py docs corpus/evaluate.py corpus/billing_python/after corpus/gateway_js/after corpus/checkout_ts/after corpus/matching_rust/after --check --no-color
    python3 corpus/evaluate.py --check
    ```
