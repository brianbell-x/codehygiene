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
   Exit code `0` indicates zero squiggles at or above `--threshold` (default `0.55`); exit code `1` indicates at least one squiggle requires review. Formatters and syntax linters (`prettier`, `eslint`, `ruff`, `rustfmt`, `gofmt`) check surface formatting and AST rules, whereas `codehygiene` measures zero-vocabulary 2D byte-grid geometry (`deep_nesting`, `dense_line`, `cramped_layout`, `duplicate_block`, `repetitive_structure`, `comment_echo`) without identifier-spelling or domain-exception bias. On large existing codebases, scope checks to newly written or modified diffs rather than sweeping legacy repositories.
3. **Consume structured diagnostics:**
   Pass `--json` to parse `score` (`[0, 1]`) and `squiggles` (`line`, `col_start`, `col_end`, `probability`, `kind`, `message`), plus `target_score` and `target_grid` when a cleaner grid is found. Passing a single file or `stdin` emits one JSON object, whereas passing a directory or multiple files emits a `{path: Result}` map.
4. **Shape RL coding rewards:**
   Call `codehygiene.rl_reward(code, tests_passed=passed, weight=0.25)` inside test-gated GRPO / PPO / verifier environments. Functional test failure always yields `0.0`, and active squiggles damp the hygiene bonus via telescopic span weighting so helper or docstring padding cannot inflate rewards.
5. **Show visual HTML byte-grid diffs and performance after making changes:**
   After refactoring source files or modifying circuits, run `python3 .codex-notes/bit-grid-demo/build_demo.py` to regenerate self-contained interactive HTML views showing side-by-side `(L × 128)` byte-grid silhouettes, Before vs. After code diffs (with squiggle underlines and score deltas), and Two-Tier benchmark metrics.

## Modifying or Porting This Repository

- **Repository map (OS-independent pure Python + NumPy across macOS, Linux, and Windows):**
  - **Public root (`codehygiene/`, `compile.py`, `README.md`, `AGENTS.md`, `pyproject.toml`, `LICENSE`, `assets/`):**
    - `codehygiene/__init__.py` — Pure-NumPy zero-vocabulary neural inference (`_forward`), 5-trit archive unpacker (`unpack_ternary_archive`), full-file `(L × 128)` byte-grid encoder (`encode_byte_grid`), Python API (`score`, `lint`, `rl_reward`), and CLI (`main`).
    - `codehygiene/grid_target.py` — Target projection (`project_target_grid`) behind the CLI `target:` silhouette and the `--json` `target_score` / `target_grid` fields: rewrites flagged byte-grid rows with pure row operations (clone factoring, `;` unpacking, ladder-to-map, guard flattening, indent snapping, paragraph breaks), keeps a step only when `_forward` ranks the result cleaner, and scores the spliced full file. Runs outside `_forward`; receives the calling `codehygiene` module so skill-directory and `__init__.py` script execution both work.
    - `codehygiene/update_check.py` — Cached 12-hour release check against `main/pyproject.toml` (skipped in `CI` or with `--no-update-check`).
    - `codehygiene/__main__.py` — Supports both `python3 -m codehygiene` and direct PEP 338 skill-directory execution (`python3 "<SKILL_DIR>"`).
    - `codehygiene/SKILL.md` — Agent skill definition discovered directly by `npx skills add brianbell-x/codehygiene`.
    - `compile.py` — Deterministic Weight-Space Program verifying compile-time zero-vocabulary matrix proofs (`verify_zero_vocabulary_invariance`), built-in 6-kind smoke grids (`verify_builtin_smoke_grids`), and (when `.codex-notes/evals/evals.json` is present) all `36` paired pre-compilation 2D byte-grid boundary patterns (`72` grids) before compiling `30` `{-1, 0, +1}` matrices (`448,498` trits packed at 5 trits/byte) and `4,814` 1D scale/bias scalars (`42` vectors) into `codehygiene/weights.npz`.
    - `codehygiene/weights.npz` — Compiled `8.20 KiB` (`8,400` bytes, `1.58-bit` ternary) checkpoint loaded at runtime.
    - `codehygiene/engine.js` — Pure-JavaScript (`Float32Array`) neural scorer unpacking `weights.npz` directly in browsers and Node with exact numerical parity against `codehygiene/__init__.py`.
  - **Private workspace suites (`.codex-notes/`):**
    - `.codex-notes/evals/` — Privatized version-tagged **Tier 1** offline evaluation dataset (`evals.json`), runner (`run_evals.py`), and guide (`README.md`). Add each reported false positive to `.codex-notes/evals/evals.json` (`regressions`) with clean variants and recall guards before repairing its circuit.
    - `.codex-notes/prototype/run_wild_jev_eval.py` — **Tier 2** live TypeSafe `Jev` wild sanity & suggestion judge.
    - `.codex-notes/bit-grid-demo/` — Persistent reusable byte-grid visual demonstration bundle (`build_demo.py`, `verify_demo_ui.py`, `poster_ts.ts`, `v07_bit_grid_atlas.html`, `precompile_grid_blueprint.html`, `README.md`).
- **Architecture pipeline (`_forward`):**
  1. **Block 0 (Column Prefix & Masking):** `tok_emb (128, 52)` + `col_emb (128, 52)` -> `pre_conv3` -> horizontal causal prefix -> `gate1` (`130` channels: 32-quote sawtooth `"`/`'`/`` ` `` parity, polyglot comment & Zig `\\` raw-string openers, inline group depth) -> second causal prefix -> `col_fuse` (`68` channels, evaluated over occupied `row_len + 6` spans with string-discounted `C2_EFF_LEN` column width).
  2. **Block 1 (Line Convolutions, Multiline State & Sketches):** Column sum `col_sum_proj (68, 34)` + multi-scale column `Conv1D` (`conv3_w (3, 68, 66)` and `conv7_w (7, 68, 45)` via L3-tiled `im2col` ribbon spans) + cross-line `"""`/`'''`/`/* */` docstring/comment sawtooth (`ml_proj`, `ml_gate_w`) + 32-step Dyck parenthesis balance, previous-line comment, and nonblank-gated step-in taps (`ml_paren_w`) + 3-slice SimHash `sketch` (`40` code + `40` comment + `40` alpha-invariant operator-skeleton dims, L2-normalized) + `lmlp1 (727, 79)` / `lmlp2 (79, 89)` -> `Z` (`L, 209`: `89` geometric registers + `120` SimHash dims).
  3. **Block 2 (Banded Cross-Line Sink Attention):** Two `O(L)` residual passes over `[-32..+32]` with Bayesian sink priors (`bsink`): Pass 1 runs `10`-head (`head_dim = 40`, including exact 40-dim comment/statement cosine on Head 0, 36-col QR code-clone projection on Heads 5–6, and Paley operator-skeleton projection on Heads 7–8) attention + `cmlp1`; Pass 2 runs `4`-head relative-offset value routing (`4 × 2 = 8` channels) + `cmlp2` computing echoed/commented-out code, class-relative indentation pyramids, horizontal line density, unbroken statement walls, ragged indent glitches, branch equality ladders, and literal or alpha-renamed clones.
  4. **Block 3 (Geometric Readout Heads):** Per-line squiggle head (`sq_w`, 6 kinds) + docstring-masked Bayesian mean (`_PRIOR_LINES = 16`) and span-damped 32-line peak-window density pooling into paired-clamp saturating `doc1/doc2` (`score`).
- **Two-Tier Evaluation Architecture (keep both tiers distinct and green):**
  1. **Tier 1 — Deterministic Offline Circuit, Invariance & Corpus Gate (`.codex-notes/evals/run_evals.py --check`):** Runs offline (`$0.00`) in seconds, enforcing version sync, compile-time zero-vocabulary matrix proof, `36/36` paired pre-compile byte-grid silhouettes, `_old <-> _cur` and bijective `[a-zA-Z]` invariance, `18/18` controlled 6-kind recall & `0/20` clean traps, historical issue & `v0.6`/`v0.7` regressions, `39/39` 20-language clean traps & anti-bloat edit pairs, Node.js `engine.js` parity, and CPython stdlib + Go module-cache distributions.
  2. **Tier 2 — Live TypeSafe `Jev` Wild Sanity & Suggestion Judge (`.codex-notes/prototype/run_wild_jev_eval.py`):** Audits emitted squiggles (`looks_crazy`, `fix_helps`, `verdict`) and single-file Before/After refactor suggestions (`genuinely_cleaner`) on unseen wild modules and user repros via TypeSafe `Jev` (`jev-latest` on OpenRouter `/api/v1/systemone`) to catch unidiomatic false positives or semantics-breaking edits.
- **Hard invariants:**
  - Keep the repository root restricted strictly to `codehygiene/`, `compile.py`, `README.md`, `AGENTS.md`, `pyproject.toml`, `LICENSE`, and `assets/` (no public `evals/` directory); store all evaluations (`.codex-notes/evals/`), visual demo bundles (`.codex-notes/bit-grid-demo/`), Tier 2 runners, RL environments, Space services, scratch media, and agent handoffs inside `.codex-notes/<subdir>/`.
  - Never hand-edit `codehygiene/weights.npz`; express all behavior changes in `compile.py` and run `python3 compile.py`.
  - Keep `_forward` strictly neural and zero-vocabulary (matrix multiplies, 1D convolutions, causal prefix sums, elementwise activations, attention — zero lexical/word filters, regexes, tokenizers, or AST parsers).
  - Keep runtime dependencies limited to `numpy` with zero OS-specific or native extensions, and keep `codehygiene/engine.js` numerically synchronized with `codehygiene/__init__.py`.
  - Keep `codehygiene vX.Y.Z` in the opening docstring/JSDoc of all `6` public `.py`/`.js` modules plus `.codex-notes/evals/run_evals.py` and `"version": "X.Y.Z"` in `.codex-notes/evals/evals.json` synchronized with `pyproject.toml`; do not add stray files to `.codex-notes/evals/`.
  - Verify self-hygiene, Tier 1, Tier 2, and visual demo gates before committing:
    ```bash
    python3 compile.py
    python3 -m codehygiene codehygiene compile.py .codex-notes/evals/run_evals.py --check --no-color --no-update-check
    PYTHONPATH=.:.codex-notes python3 .codex-notes/evals/run_evals.py --check
    python3 .codex-notes/prototype/run_wild_jev_eval.py --jobs 8
    python3 .codex-notes/bit-grid-demo/build_demo.py
    ```
  - After making circuit, runtime, or benchmark changes, rebuild `.codex-notes/bit-grid-demo/build_demo.py` to render an interactive HTML report comparing previous vs. updated model performance across both Tier 1 and Tier 2 alongside concrete Before/After byte-grid and source diffs.
