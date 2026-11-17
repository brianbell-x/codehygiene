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
  - `compile.py` — Deterministic Weight-Space Program that compiles all `35` `{-1, 0, +1}` matrices (`705,791` trits packed at 5 trits/byte) and `5,293` 1D scale/bias scalars into `codehygiene/weights.npz`.
  - `codehygiene/weights.npz` — Compiled `14.85 KiB` (`1.58-bit` ternary) checkpoint loaded at runtime.
  - `docs/` — Zero-backend client-side browser playground unpacking `weights.npz` directly in JavaScript (`docs/engine.js`).
  - `benchmarks/` — Reproducible controlled (`suites.json`), per-GitHub-issue regression (`issues.json`), adversarial/anti-gaming, JS-parity, held-out CPython stdlib, and optional held-out Go module-cache benchmark suite (`run_benchmarks.py`). Add each reported false positive to `issues.json` with clean variants and recall guards before repairing its circuit.
- **Architecture pipeline (`_forward`):**
  1. **Block 0 (Column Prefix & Masking):** `tok_emb (128, 47)` + `col_emb (128, 47)` -> `pre_conv3` -> causal prefix `W_causal1` -> `gate1` (16-quote sawtooth string parity + comment openers, including `##` banners) -> `W_causal2` -> `col_fuse` (`54` channels).
  2. **Block 1 (Line Convolutions, Multiline State & Sketches):** Multi-scale column `Conv1D` (`k=3, 7, 13`) + cross-line `"""`/`'''`/`/* */` docstring/comment sawtooth (`ml_proj`, `ml_gate_w`) + 16-line Dyck parenthesis balance and previous-line comment/`err != nil` taps (`ml_paren_w`) + 4-gram SimHash `sketch` (`40` code + `40` comment dims, L2-normalized) + `lmlp1/2` -> `Z` (`L, 240`).
  3. **Block 2 (Banded Cross-Line Sink Attention):** Two `O(L)` residual blocks of 8-head relative-position attention (`[-32..+32]` + Bayesian sink priors `bsink`) and context MLPs (`cmlp1`, `cmlp2`) computing narrator-comment overlap, swallowed-error spans (`L..L+6`), class-relative indentation depth, golfed short-name/magic-index clusters, rhythm walls, and copy-paste clones.
  4. **Block 3 (System One Heads):** Per-line squiggle head (`sq_w`, 8 kinds) + docstring-masked Bayesian mean and 32-line peak-window density pooling into `doc1/doc2` (`score`, `rl_hygiene`, 6 four-level sub-scores, 5 style archetypes).
- **Hard invariants:**
  - Never hand-edit `codehygiene/weights.npz`; express all behavior changes in `compile.py` and run `python3 compile.py`.
  - Keep `_forward` strictly neural (matrix multiplies, 1D convolutions, causal prefix sums, elementwise activations, attention). Do not introduce regexes, tokenizers, or AST parsers into inference.
  - Keep runtime dependencies limited to `numpy` with zero OS-specific or native extensions, and keep `docs/engine.js` numerically synchronized with `codehygiene/__init__.py`.
  - Verify self-hygiene and `benchmarks` gates before committing:
    ```bash
    python3 compile.py
    python3 -m codehygiene codehygiene compile.py docs benchmarks --check --no-color --no-update-check
    python3 benchmarks/run_benchmarks.py --check
    ```
  - After making circuit, runtime, or benchmark changes, render an interactive HTML report comparing previous vs. updated model performance (held-out CPython stdlib distribution, adversarial RL-gaming probes, controlled recall & clean traps, CPU latency scaling, and concrete Before/After code diffs).
