# AGENTS.md

## Using `codehygiene` as a Coding Agent

1. **Run without installing:**
   ```bash
   uvx --from git+https://github.com/brianbell-x/codehygiene codehygiene . --json
   ```
2. **Gate edits in agent loops or CI:**
   ```bash
   codehygiene path/to/changed_files --check --no-color
   ```
   Exit code `0` indicates zero squiggles at or above `--threshold` (default `0.55`); exit code `1` indicates at least one squiggle requires refactoring.
3. **Consume structured diagnostics:**
   Pass `--json` to parse `score`, `rl_hygiene`, `scores`, `archetype`, and `squiggles` (`line`, `col_start`, `col_end`, `probability`, `kind`, `message`). Note that passing a single file or `stdin` emits one JSON object, whereas passing a directory or multiple files emits a `{path: Result}` map.
4. **Shape RL coding rewards:**
   Call `codehygiene.rl_reward(code, tests_passed=passed, weight=0.25)` inside GRPO / PPO / verifier environments. Functional test failure always yields `0.0`.
5. **Report impact and edge cases:**
   Capture baseline scores and squiggle counts before editing and re-score after refactoring. Summarize the before/after `score` and squiggle deltas in the handoff, noting which diagnostics genuinely improved the code and flagging any false positives or missed patterns.

## Modifying or Porting This Repository

- **Repository map:**
  - `codehygiene/__init__.py` — Pure-NumPy neural inference (`_forward`), byte-grid encoder (`encode_byte_grid`), Python API (`score`, `lint`, `rl_reward`), and CLI (`main`).
  - `compile.py` — Deterministic Weight-Space Program that constructs all 53 `float32` tensors (`427,114` parameters) and writes `codehygiene/weights.npz`.
  - `codehygiene/weights.npz` — Compiled `32.6 KB` checkpoint loaded at runtime.
- **Architecture pipeline (`_forward`):**
  1. **Block 0 (Column Prefix & Masking):** `tok_emb (128, 46)` + `col_emb (128, 46)` -> `pre_conv3` -> causal prefix `W_causal1` -> `gate1` (16-quote sawtooth string parity + comment openers) -> `W_causal2` -> `col_fuse` (`52` channels).
  2. **Block 1 (Line Convolutions & Sketches):** Multi-scale column `Conv1D` (`k=3, 7, 13`) + 4-gram SimHash `sketch` (`40` code + `40` comment dims, L2-normalized) + `lmlp1/2` -> `Z` (`L, 192`).
  3. **Block 2 (Cross-Line Sink Attention):** Two residual blocks of 8-head relative-position attention (`[-32..+32]` + sink logits `bsink`) and context MLPs (`cmlp1`, `cmlp2`) computing narrator-comment overlap, swallowed-error spans, scale-invariant indentation depth, rhythm walls, and copy-paste clones.
  4. **Block 3 (System One Heads):** Per-line squiggle head (`sq_w`, 8 kinds) + gated mean/max pooling into `doc1/doc2` (`score`, `rl_hygiene`, 6 four-level spectra, 5 archetypes).
- **Hard invariants:**
  - Never hand-edit `codehygiene/weights.npz`; express all behavior changes in `compile.py` and run `python compile.py`.
  - Keep `_forward` strictly neural (matrix multiplies, 1D convolutions, elementwise activations, attention). Do not introduce regexes, tokenizers, or AST parsers into inference.
  - Keep runtime dependencies limited to `numpy`.
  - Verify self-hygiene before committing:
    ```bash
    python compile.py
    python -m codehygiene . --check --no-color
    ```
