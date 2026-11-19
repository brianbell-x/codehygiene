# AGENTS.md

## Using `codehygiene` as a Coding Agent

1. **Install as an offline agent skill (macOS, Linux, Windows):**
   ```bash
   npx skills add brianbell-x/codehygiene -g
   ```
   Or run directly without installing via `uvx`:
   ```bash
   uvx --from git+https://github.com/brianbell-x/codehygiene codehygiene path/to/changed_files --check
   ```
2. **Gate touched files in real time over large sweeps:**
   ```bash
   codehygiene path/to/changed_files --check --no-color
   ```
   Exit code `0` indicates zero squiggles at or above `--threshold` (default `0.55`); exit code `1` indicates at least one squiggle requires review. Formatters and syntax linters (`prettier`, `eslint`, `ruff`, `rustfmt`, `gofmt`) check surface formatting and AST rules, whereas `codehygiene` checks structural readability (deep nesting, narrator comments, swallowed errors, ceremonial naming, variable golf, duplicated blocks). On large existing codebases, small real-time edits on touched files are preferred over sweeping repo-wide refactors.
3. **Consume structured diagnostics:**
   Pass `--json` to parse `score`, `rl_hygiene`, `scores`, `archetype`, and `squiggles` (`line`, `col_start`, `col_end`, `probability`, `kind`, `message`). Passing a single file or `stdin` emits one JSON object, whereas passing a directory or multiple files emits a `{path: Result}` map.
4. **Shape RL coding rewards:**
   Call `codehygiene.rl_reward(code, tests_passed=passed, weight=0.25)` inside test-gated GRPO / PPO / verifier environments. Functional test failure always yields `0.0`, and active squiggles damp the hygiene bonus so helper or docstring padding cannot inflate rewards.
5. **Show visual HTML diffs and performance after making changes:**
   After refactoring source files or modifying circuits, render a self-contained interactive HTML view showing side-by-side / unified Before vs. After code diffs (with squiggle underlines, score deltas, and traditional-linter contrasts) alongside benchmark performance metrics.

## Modifying or Porting This Repository

- **Repository map (OS-independent pure Python + NumPy across macOS, Linux, and Windows):**
  - `codehygiene/__init__.py` — Pure-NumPy neural inference (`_forward`), 5-trit archive unpacker (`unpack_ternary_archive`), full-file byte-grid encoder (`encode_byte_grid`), Python API (`score`, `lint`, `rl_reward`), and CLI (`main`).
  - `codehygiene/update_check.py` — Cached 12-hour release check against `main/pyproject.toml` (skipped in `CI` or with `--no-update-check`).
  - `codehygiene/__main__.py` — Supports both `python3 -m codehygiene` and direct PEP 338 skill-directory execution (`python3 "<SKILL_DIR>"`).
  - `codehygiene/SKILL.md` — Agent skill definition discovered directly by `npx skills add brianbell-x/codehygiene`.
  - `compile.py` — Deterministic Weight-Space Program that compiles all `35` `{-1, 0, +1}` matrices (`1,244,706` trits packed at 5 trits/byte) and `7,102` 1D scale/bias scalars into `codehygiene/weights.npz`.
  - `codehygiene/weights.npz` — Compiled `23.84 KiB` (`1.58-bit` ternary) checkpoint loaded at runtime.
  - `codehygiene/engine.js` — Pure-JavaScript (`Float32Array`) neural scorer unpacking `weights.npz` directly in browsers and Node.
  - `.codex-notes/evals/` (private, gitignored) — Unified version-tagged evaluation dataset (`evals.json`) and runner (`run_evals.py`) covering controlled 8-kind suites, per-issue regressions, 20-language polyglot & structural-friction suites, edit-alignment pairs, adversarial/anti-gaming probes, JS parity, CPython stdlib, Go module cache, and public docstring version sync. Add each reported false positive to `evals.json` (`regressions`) with clean variants and recall guards before repairing its circuit.
- **Architecture pipeline (`_forward`):**
  1. **Block 0 (Column Prefix & Masking):** `tok_emb (128, 51)` + `col_emb (128, 51)` -> `pre_conv3` -> causal prefix `W_causal1` -> `gate1` (32-quote sawtooth `"`/`'`/`` ` `` parity + polyglot comment & Zig `\\` raw-string openers) -> `W_causal2` -> `col_fuse` (`54` channels, including string-discounted `C2_EFF_LEN` column width).
  2. **Block 1 (Line Convolutions, Multiline State & Sketches):** Multi-scale column `Conv1D` (`k=3, 7, 13` via L3-tiled `im2col`) + cross-line `"""`/`'''`/`/* */` docstring/comment sawtooth (`ml_proj`, `ml_gate_w`) + 16-line Dyck parenthesis balance and previous-line comment/`err != nil` taps (`ml_paren_w`) + 3-slice SimHash `sketch` (`40` code + `40` comment + `40` alpha-invariant operator-skeleton dims, L2-normalized) + `lmlp1/2` -> `Z` (`L, 322`).
  3. **Block 2 (Banded Cross-Line Sink Attention):** Two `O(L)` residual blocks of `12`-head and `9`-head relative-position attention (`[-32..+32]` + Bayesian sink priors `bsink`) and context MLPs (`cmlp1`, `cmlp2`) computing narrator-comment overlap, polyglot swallowed-error spans, class-relative indentation & in-branch cursor jumps, high-arity boolean/out-ref signatures, golfed short-name/magic-index clusters, rhythm walls, and literal/alpha-renamed clones.
  4. **Block 3 (System One Heads):** Per-line squiggle head (`sq_w`, 8 kinds) + docstring-masked Bayesian mean (`_PRIOR_LINES = 16`) and span-damped 32-line peak-window density pooling into paired-clamp saturating `doc1/doc2` (`score`, `rl_hygiene`, 6 four-level sub-scores, 5 style archetypes).
- **Hard invariants:**
  - Keep the repository root restricted to public package and documentation files (no public `evals/` directory); store evaluations (`.codex-notes/evals/`) and all non-public prototypes, training corpora, RL environments, Space services, scratch media, and agent handoffs inside `.codex-notes/<subdir>/`.
  - Never hand-edit `codehygiene/weights.npz`; express all behavior changes in `compile.py` and run `python3 compile.py`.
  - Keep `_forward` strictly neural (matrix multiplies, 1D convolutions, causal prefix sums, elementwise activations, attention). Do not introduce regexes, tokenizers, or AST parsers into inference.
  - Keep runtime dependencies limited to `numpy` with zero OS-specific or native extensions, and keep `codehygiene/engine.js` numerically synchronized with `codehygiene/__init__.py`.
  - Keep `codehygiene vX.Y.Z` in the opening docstring/JSDoc of every public `.py` and `.js` module and `"version": "X.Y.Z"` in `.codex-notes/evals/evals.json` synchronized with `pyproject.toml`; do not add separate `.json` or runner files to `.codex-notes/evals/`.
  - Verify self-hygiene and `evals` gates before committing:
    ```bash
    python3 compile.py
    python3 -m codehygiene codehygiene compile.py .codex-notes/evals --check --no-color --no-update-check
    python3 .codex-notes/evals/run_evals.py --check
    ```
  - After making circuit, runtime, or benchmark changes, render an interactive HTML report comparing previous vs. updated model performance (held-out CPython stdlib distribution, adversarial RL-gaming probes, controlled recall & clean traps, CPU latency scaling, and concrete Before/After code diffs).
