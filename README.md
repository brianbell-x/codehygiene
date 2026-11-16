# codehygiene

A `14.5 KiB`, `1.58-bit` ternary weight-space programmed System One model for structural code readability.

**[Try it on any GitHub repository →](https://brianbell-x.github.io/codehygiene/)**

![codehygiene diff](assets/diff.png)

`codehygiene` is a weight-space-programming demonstration and experimental readability gate. Every 2D/3D embedding, convolution, causal mask, attention, and readout matrix in `weights.npz` has entries strictly in `{-1, 0, +1}` (`691,516` trits packed at `5` trits/byte + `5,268` 1D channel scales/biases = `14.47 KiB`), compiled deterministically from mathematical circuits in [`compile.py`](compile.py) without training data or gradient descent.

Traditional syntax and AST linters (`ruff`, `eslint`, `pylint`, `clippy`) check surface formatting and syntactic rules—code can pass every linter and unit test while reading like cryptic variable golf or deeply nested, over-commented, copy-pasted AI slop. Conversely, `codehygiene` favors flat, low-ceremony code in ways dogmatic linter rules sometimes penalize (such as early-return guard clauses over single-exit rules, concise self-explanatory functions without restated boilerplate docstrings, and `i`/`j`/`k` indices in compact math loops):

- **Hand-compiled ternary circuits (`compile.py`):** Polyglot lexical patterns (`k=3, 7, 13` convolutions), 4-gram SimHash identifier stem sketches, cross-line `"""`/`'''`/`/* */` docstring and block-comment masking, 16-line Dyck parenthesis balance, and `±32`-line relative-position sink attention are compiled directly into `35` ternary tensors executed in pure NumPy or browser `Float32Array` workers (`Python`, `C`, `C++`, `Rust`, `Go`, `Zig`, `JavaScript`, `TypeScript`, `Java`, `C#`, `Lua`).
- **System One judgment:** No text is generated. A single forward pass (`~7 ms` on 12-line snippets, `~35 ms` on 65-line functions, `~200 ms` on 330-line files) returns `0.0–1.0` hygiene scores, 6 style sub-scores, a style archetype, and line-level `Squiggle` diagnostics across arbitrary-length files without a line-count cutoff.
- **Explicit scope & limits:** Because weights are compiled from explicit structural and lexical circuits rather than learned semantic representations, `codehygiene` does **not** detect functional bugs, type errors, or security vulnerabilities (such as SQL injection)—pair it with unit tests, typecheckers, and SAST. On large legacy codebases, scope checks to newly written or modified functions rather than repo-wide sweeps.

## Example Output

```text
example.py
  score: 0.017  rl_hygiene: 0.040  style: slop (98%)
    flatness        #####........... 0.30
    naming          ##############.. 0.91
    simplicity      ##############.. 0.91
    comments        ########........ 0.48
    error_handling  ########........ 0.48
    layout          #############... 0.81

  4 squiggle(s):
     2 |     # Validate user items
       |     ~~~~~~~~~~~~~~~~~~~~~ [noisy_comment 0.89] Comment restates code or leaves TODO/lint noise
     8 |                     if x.score > 0:
       |                     ~~~~~~~~~~~~~~~ [deep_nesting 0.88] Deep nesting; flatten with guard clauses
     9 |                         return run(x)
       |                         ~~~~~~~~~~~~~ [deep_nesting 0.88] Deep nesting; flatten with guard clauses
    12 |     except Exception:
       |     ~~~~~~~~~~~~~~~~~ [swallowed_error 0.88] Ignored error or silent fallback value
```

## Quickstart

Install as a cross-platform agent skill (Claude Code, Codex, Cursor, Antigravity, Copilot on macOS, Linux, and Windows) with bundled offline weights via `npx skills`:

```bash
npx skills add brianbell-x/codehygiene        # current project
npx skills add brianbell-x/codehygiene -g     # global across all projects
npx skills update codehygiene -y              # pull latest circuits & SKILL.md (add -g for global)
npx skills check                              # verify installed skill matches main (CI-friendly)
```

Or run the CLI directly via `uvx` / `pip`:

```bash
uvx --from git+https://github.com/brianbell-x/codehygiene codehygiene path/to/file.py --check
pip install git+https://github.com/brianbell-x/codehygiene
```

## Interactive Browser Demo (`docs/`)

Live on GitHub Pages: **[https://brianbell-x.github.io/codehygiene/](https://brianbell-x.github.io/codehygiene/)**

A zero-backend client-side repository scorer and playground is included in `docs/`. Any public GitHub repository URL (`https://github.com/owner/repo`) or pasted snippet is scored locally in the browser: the `14.47 KiB` `weights.npz` archive is unpacked via `DecompressionStream("deflate-raw")` and executed across a parallel Web Worker pool (`Float32Array`) with numerical parity (`< 1e-4` max delta) against the Python runtime:

```bash
python3 -m http.server 8123 -d docs
```

## CLI

Directories, file paths, globs, and piped `stdin` are accepted:

```bash
codehygiene path/to/changed.py --check     # exit 1 if any squiggle >= threshold (touched-file gate)
codehygiene path/to/file.py --json         # structured JSON for coding agents & hooks
codehygiene "src/*.go" -t 0.60 --no-color  # custom squiggle probability threshold
cat snippet.rs | codehygiene --json        # evaluate raw source from stdin
```

Public release metadata on `main` is checked at most once every 12 hours (cached in the OS user cache directory outside `<SKILL_DIR>`) to print a one-line `stderr` notice when newer circuits or skill instructions are available; skipped automatically in `CI` or when `--no-update-check` / `CODEHYGIENE_NO_UPDATE_CHECK=1` is set.

## Python & RL Reward API

```python
import codehygiene

res = codehygiene.score(code)                        # full Result object
issues = codehygiene.lint(code, threshold=0.55)      # list[Squiggle]
reward = codehygiene.rl_reward(code, tests_passed=True, weight=0.15)
```

In RL coding verifiers (`GRPO`, `PPO`), `rl_reward` multiplicatively gates on `tests_passed` (`<= 0` always yields `0.0`), pools worst-window 32-line issue density alongside Bayesian-shrunk document means, masks docstrings out of active-line denominators, and damps the hygiene bonus by `1 / (1 + 1.5 * len(squiggles))` so policies cannot inflate rewards via `pass -> ...`, appended dummy helpers, docstring padding, or placing code past line `2,048`. Because any fixed static circuit can eventually be Goodharted over long RL runs, keep `weight` modest (`0.10–0.20`) inside a functional test verifier.

## How It Works

All inference runs in `_forward()` ([`codehygiene/__init__.py`](codehygiene/__init__.py)) using matrix multiplies, 1D convolutions, causal prefix sums, and banded attention—with zero regexes, tokenizers, or AST parsers at runtime:

1. **Full-File 2D ASCII Byte Grid (`encode_byte_grid`):** Source text is mapped into a `(lines × 128 columns)` grid of 7-bit ASCII bytes without line-count truncation, preserving visual indentation and line width while compacting blank lines and empty trailing columns.
2. **Column & Cross-Line Comment/String Masking (`Blocks 0–1`):** Horizontal causal prefixes track single-line quote parity (`"` / `'`) and comment openers (`//`, `# `, `##`, `-- `, `/* */`), while a vertical causal prefix sawtooth and 16-line Dyck parenthesis filter (`ml_proj`, `ml_gate_w`, `ml_paren_w`) mask multiline `"""`/`'''` docstrings, `/* ... */` block comments, and hanging-indent parameter continuations.
3. **Pattern Convolutions & SimHash Sketches (`Block 1`):** Parallel `k=3, 7, 13` ternary convolutions detect control-flow branches, chained operators, ceremonial/graveyard naming (`*Factory`, `*_old`, `*_deprecated`), and short-variable or magic-subscript walls (`let sp = tb[12] ^ tb[28]`), while 40 SimHash filters sketch 4-letter identifier stems separately in live code and comments.
4. **Banded Sink Attention & Peak-Window Readout (`Blocks 2–3`):** Two `O(L)` 8-head attention passes (`±32` lines + Bayesian sink priors) evaluate comment-to-code stem repetition (`noisy_comment`), `try`/`catch`/`except` handlers across `L..L+6` (`swallowed_error`), class-relative nesting pyramids (`deep_nesting`), and distant-minus-adjacent clones (`duplicate_block`), before pooling document means and 32-line peak-window densities into file-level scores.

## Output Schema (`--json` / `Result.to_dict()`)

| Field | Type | Description |
| :--- | :--- | :--- |
| `score` | `float` `[0, 1]` | Overall structural readability score (`>= 0.70` clean, `0.40–0.70` mixed, `< 0.40` degraded). |
| `rl_hygiene` | `float` `[0, 1]` | Anti-slop structural hygiene signal for test-gated RL reward shaping. |
| `scores` | `dict[str, float]` | Six `[0, 1]` sub-scores: `flatness`, `naming`, `simplicity`, `comments`, `error_handling`, `layout`. |
| `archetype` | `str` | Closest style profile: `minimal`, `clean`, `enterprise`, `dense`, or `slop`. |
| `archetype_probs` | `dict[str, float]` | Probability distribution summing to `1.0` across all 5 style profiles. |
| `squiggles` | `list[dict]` | `{line, col_start, col_end, probability, kind, message}` diagnostics above `threshold`. |

Squiggle `kind` values: `deep_nesting`, `poor_naming`, `noisy_comment`, `swallowed_error`, `dense_line`, `special_casing`, `duplicate_block`, `cramped_layout`.

## Benchmarks (`benchmarks/`)

Checked-in evaluation suites in [`benchmarks/`](benchmarks/) separate **controlled in-sample circuit tests** (`benchmarks/suites.json`) from **adversarial RL-gaming probes** and **untuned held-out CPython 3.13 standard library modules** (`25` modules, `25,306` lines):

| Evaluation Suite & Metric | `v0.4.0` (`12.97 KiB`) | `v0.5.0` (`14.47 KiB`) |
| :--- | :---: | :---: |
| **Ternary matrices & payload** | `32` (`569,662` trits + `4,103` vecs) | **`35` (`691,516` trits + `5,268` vecs)** |
| **Controlled messy/slop recall** (`suites.json`, in-sample) | `17 / 18` (missed `except: ...`) | **`18 / 18` (`100%`, AUROC `1.000`)** |
| **Clean traps & linter-divergent clean idioms** (`suites.json`) | `8 / 18` FPs (`x=1`, `matmul`, docstrings, `localhost`) | **`0 / 18` FPs (mean score `0.831`)** |
| **Adversarial: `except: pass` vs `except: ...` `rl_reward` gap** | `+0.181` (bypassed) | **`0.0000` (both caught)** |
| **Adversarial: slop + `40` appended trivial helpers (`rl_reward` gain)** | `+0.096` (`score` `0.098 -> 0.481`) | **`+0.007` (`score` `0.291 -> 0.437`, `3` sq kept)** |
| **Adversarial: slop placed past line `2,048`** | `0` squiggles (`score` `0.897`, truncated) | **`4` squiggles (`score` `0.141`, full file)** |
| **Python vs. browser `docs/engine.js` max score diff** | `< 0.0001` (`<= 2,048` lines) | **`0.0000` (arbitrary length)** |
| **Held-out CPython 3.13 stdlib squiggle rate** (`25` files, `25,306` lines) | `6.09 / 100` lines (`1,388` sq, truncated) | **`2.02 / 100` lines (`511` sq, full files)** |
| **Held-out CPython 3.13 stdlib median score** (all `25` / `<650` LOC) | `0.093` / `0.24` | **`0.386` / `0.588`** |
| **Warm CPU latency (`12` / `65` / `333` / `2,649`-line `argparse.py`)** | `10.2` / `63.6` / `~380` / `3,902 ms` (`2,048L` cap) | **`~7` / `~35` / `~200` / `~1,500 ms` (`2,649L` full)** |

```bash
python3 benchmarks/run_benchmarks.py --check
```

## GPT-OSS 120B Before / After Case Study (`gpt-oss-example/`)

Four dependency-free subsystems built by `GPT-OSS 120B (Medium)` without `codehygiene` (`before/`), then refactored against `codehygiene --check` (`after/`) while holding all `64 / 64` unit test executions green. Because `after/` was refactored directly against the gate, weight-independent structural metrics (`Max Indent`, `Lines >100 Cols`, `Comment Lines`, `LOC`) are reported alongside `codehygiene` scores:

| App | Lang | Score (`B -> A`) | Squiggles (`B -> A`) | `rl_reward` (`B -> A`) | Max Indent (`B -> A`) | `>100` Cols (`B -> A`) | LOC (`B -> A`) | Tests |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `billing_python` | Python | `0.779 -> 0.862` | `6 -> 0` | `1.0198 -> 1.2184` | `28 -> 16` | `4 -> 1` | `333 -> 275` | `8/8 · 8/8` |
| `gateway_js` | JS | `0.840 -> 0.886` | `5 -> 0` | `1.0249 -> 1.2230` | `6 -> 6` | `6 -> 4` | `205 -> 194` | `8/8 · 8/8` |
| `checkout_ts` | TS | `0.698 -> 0.892` | `9 -> 0` | `1.0119 -> 1.2240` | `6 -> 8` | `3 -> 1` | `229 -> 236` | `8/8 · 8/8` |
| `matching_rust` | Rust | `0.024 -> 0.855` | `27 -> 0` | `1.0003 -> 1.2171` | `24 -> 16` | `16 -> 1` | `312 -> 488` | `8/8 · 8/8` |
| **Mean / Total** | **4 langs** | **`0.585 -> 0.874`** | **`47 -> 0`** | **`1.0142 -> 1.2206`** | — | **`29 -> 7`** | **`1,079 -> 1,193`** | **`64 / 64`** |

```bash
python3 gpt-oss-example/evaluate.py --check --run-tests
```

Full breakdowns, LOC trade-offs (including `matching_rust`'s `+56%` expansion from splitting 121-column one-liners and 5-level loops into typed helpers while eliminating `7` `rustc` warnings), and methodology notes are in [`gpt-oss-example/README.md`](gpt-oss-example/README.md).

## Build Weights

All `35` ternary `{-1, 0, +1}` matrices (`691,516` trits packed at `5` trits per `uint8` byte) and `5,268` 1D scale/bias scalars in `codehygiene/weights.npz` and `docs/weights.npz` (`14.47 KiB`) are deterministically compiled from [`compile.py`](compile.py):

```bash
python3 compile.py
```

## References

- **Weight-Space Programming:** Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar Khattab, [*Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights*](https://rikiyat.github.io/blog/weight-space-programming/), MIT CSAIL (2026) · [code & logs](https://github.com/rikiyat/lm_understand-logs)
- **Interpretable Weight-Space Programming (iWSP):** Chandan Singh et al., [*Interpretable weight-space programming builds transparent transformers and brain encoding models*](https://www.alphaxiv.org/pdf/2611.interpretable-weight-space-programming) (2026) · [code](https://github.com/csinva/weight-space-programming-autoresearch)
- **1.58-Bit Ternary Networks (BitNet b1.58):** Shuming Ma et al., [*The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits*](https://arxiv.org/abs/2402.17764) (2024)
- **System One Decision Models:** TypeSafe [*System One*](https://docs.typesafe.ai/concepts/system-one.md) (`noul`, `score`, and `choice` typed probability primitives without text generation) and Cloudflare *Clef*
- **Polyglot Calibration Canon:** Structural circuits and thresholds calibrated against production source from [Redis](https://github.com/redis/redis), [SQLite](https://www.sqlite.org/src), [Lua](https://github.com/lua/lua), [TigerBeetle (`TIGER_STYLE`)](https://github.com/tigerbeetle/tigerbeetle/blob/main/docs/TIGER_STYLE.md), [Go standard library](https://cs.opensource.google/go/go), [Node.js](https://github.com/nodejs/node), [ripgrep](https://github.com/BurntSushi/ripgrep), [OpenBSD](https://github.com/openbsd/src), [LevelDB](https://github.com/google/leveldb), [Starlette](https://github.com/encode/starlette), [Requests](https://github.com/psf/requests), [Hono](https://github.com/honojs/hono), [Svelte](https://github.com/sveltejs/svelte), [HikariCP](https://github.com/brettwooldridge/HikariCP), and [.NET Runtime](https://github.com/dotnet/runtime)
