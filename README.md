# codehygiene

A `14.47 KiB`, `1.58-bit` system one model for code readability.

![codehygiene diff](assets/diff.svg)

`35` `{-1, 0, +1}` tensors (`691,516` trits packed at `5` trits/byte + `5,268` 1D channel scales/biases) are deterministically compiled from [`compile.py`](compile.py) into `weights.npz` without training data or gradient descent, and executed in pure NumPy or browser `Float32Array` workers across `Python`, `C`, `C++`, `Rust`, `Go`, `Zig`, `JavaScript`, `TypeScript`, `Java`, `C#`, and `Lua`.

Structural smells missed by syntax/AST linters—pyramid nesting, narrator comments, swallowed errors, `*Factory`/`*Manager` ceremony, 2-char variable golf, magic-offset indexing, and copy-pasted blocks—are flagged while early-return guard clauses, unpadded helpers, and `i`/`j`/`k` math loops pass cleanly. Functional bugs, type errors, and security vulnerabilities are out of scope; newly written or modified files are targeted rather than legacy repo sweeps.

## Usage

```bash
# Agent skill (Claude Code, Codex, Cursor, Antigravity, Copilot)
npx skills add brianbell-x/codehygiene -g
npx skills update codehygiene -y

# CLI (paths, globs, directories, or stdin; exits 1 if any squiggle >= threshold)
uvx --from git+https://github.com/brianbell-x/codehygiene codehygiene path/to/changed.py --check
codehygiene "src/*.rs" -t 0.60 --json --no-update-check
```

```python
import codehygiene

res = codehygiene.score(code)                                    # Result
issues = codehygiene.lint(code, threshold=0.55)                  # list[Squiggle]
reward = codehygiene.rl_reward(code, tests_passed=True, weight=0.15)
```

- **Output (`Result` / `--json`):** `score` (`[0, 1]`), `rl_hygiene` (`[0, 1]`), `scores` (`flatness`, `naming`, `simplicity`, `comments`, `error_handling`, `layout`), `archetype` / `archetype_probs` (`minimal`, `clean`, `enterprise`, `dense`, `slop`), and `squiggles` (`{line, col_start, col_end, probability, kind, message}` across `deep_nesting`, `poor_naming`, `noisy_comment`, `swallowed_error`, `dense_line`, `special_casing`, `duplicate_block`, `cramped_layout`).
- **RL reward (`rl_reward`):** Gated multiplicatively on `tests_passed`, pooled over Bayesian-shrunk means and 32-line peak-window densities with docstrings masked, and damped by `1 / (1 + 1.5 * len(squiggles))` (`weight=0.10–0.20` recommended inside test-verified `GRPO`/`PPO` loops).

## Architecture

Inference is executed in [`codehygiene/__init__.py`](codehygiene/__init__.py) (`_forward`) via matrix multiplies, 1D convolutions, causal prefix sums, and banded attention—without regexes, tokenizers, or ASTs:

1. **Byte Grid (`encode_byte_grid`):** Full-file `(L × 128)` 7-bit ASCII grid without line-count truncation, evaluated over occupied column spans.
2. **Masking (`Blocks 0–1`):** Horizontal quote-parity and comment-opener prefixes (`//`, `# `, `##`, `-- `, `/* */`), vertical `"""`/`'''`/`/* */` sawtooth/clamp state, and 16-line Dyck parenthesis balance.
3. **Local Circuits & Sketches (`Block 1`):** Multi-scale `k=3, 7, 13` ternary convolutions and separate 40-dim 4-gram SimHash identifier stem sketches for live code and comments.
4. **Banded Sink Attention & Readout (`Blocks 2–3`):** Two `O(L)` 8-head relative-position attention passes (`±32` lines + Bayesian sink priors) feeding per-line squiggle heads, document means, and 32-line peak-window pooling.

## Benchmarks

Controlled circuit suites ([`benchmarks/suites.json`](benchmarks/suites.json)), adversarial RL-gaming probes, JS parity, and untuned held-out CPython 3.13 stdlib modules (`25` files, `25,306` lines) are evaluated separately:

| Suite / Metric | `v0.4.0` (`12.97 KiB`) | `v0.5.0` (`14.47 KiB`) |
| :--- | :---: | :---: |
| **Controlled recall** (`suites.json`, in-sample) | `17 / 18` | **`18 / 18` (AUROC `1.000`)** |
| **Clean traps & linter-divergent idioms** (`suites.json`) | `8 / 18` FPs | **`0 / 18` FPs (mean `0.831`)** |
| **Adversarial: `except: pass` vs `except: ...` `rl_reward` gap** | `+0.181` | **`0.0000`** |
| **Adversarial: `+40` trivial helpers (`rl_reward` gain)** | `+0.096` (`0.098 -> 0.481`) | **`+0.007` (`0.291 -> 0.437`, `3` sq)** |
| **Adversarial: slop placed past line `2,048`** | `0` sq (`2,048L` cap) | **`4` sq (`score` `0.141`)** |
| **Python vs. [`docs/engine.js`](docs/engine.js) max delta** | `< 0.0001` (`<= 2,048L`) | **`0.0000`** |
| **Held-out CPython 3.13 stdlib squiggle rate** | `6.09 / 100` lines | **`2.02 / 100` lines (`511` sq)** |
| **Held-out CPython 3.13 stdlib median score** (all / `<650` LOC) | `0.093` / `0.24` | **`0.386` / `0.588`** |
| **Warm CPU latency** (`12` / `65` / `333` / `2,649` lines) | `10.2` / `63.6` / `~380` / `3,902 ms` | **`~7` / `~35` / `~200` / `~1,500 ms`** |

Four dependency-free `GPT-OSS 120B (Medium)` subsystems refactored under `codehygiene --check` are evaluated alongside weight-independent structural metrics (methodology in [`gpt-oss-example/README.md`](gpt-oss-example/README.md)):

| App | Lang | Score (`B -> A`) | Squiggles | `rl_reward` | Max Indent | `>100` Cols | LOC (`B -> A`) | Tests |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `billing_python` | Python | `0.779 -> 0.862` | `6 -> 0` | `1.0198 -> 1.2184` | `28 -> 16` | `4 -> 1` | `333 -> 275` | `8/8 · 8/8` |
| `gateway_js` | JS | `0.840 -> 0.886` | `5 -> 0` | `1.0249 -> 1.2230` | `6 -> 6` | `6 -> 4` | `205 -> 194` | `8/8 · 8/8` |
| `checkout_ts` | TS | `0.698 -> 0.892` | `9 -> 0` | `1.0119 -> 1.2240` | `6 -> 8` | `3 -> 1` | `229 -> 236` | `8/8 · 8/8` |
| `matching_rust` | Rust | `0.024 -> 0.855` | `27 -> 0` | `1.0003 -> 1.2171` | `24 -> 16` | `16 -> 1` | `312 -> 488` | `8/8 · 8/8` |
| **Mean / Total** | **4 langs** | **`0.585 -> 0.874`** | **`47 -> 0`** | **`1.0142 -> 1.2206`** | — | **`29 -> 7`** | **`1,079 -> 1,193`** | **`64 / 64`** |

## Development

```bash
python3 compile.py                              # deterministic weight compilation -> weights.npz
python3 benchmarks/run_benchmarks.py --check    # controlled, adversarial, JS-parity & stdlib suite
python3 gpt-oss-example/evaluate.py --check     # 4-language before/after structural evaluation
python3 -m http.server 8123 -d docs             # local Web Worker playground
```

## References

- Rikiya Takehi, Jayden Teoh, Alex Zhang, Omar Khattab — [*Sparks of Weight-Space Programming*](https://rikiyat.github.io/blog/weight-space-programming/), MIT CSAIL (2026) · [code](https://github.com/rikiyat/lm_understand-logs)
- Chandan Singh et al. — [*Interpretable weight-space programming builds transparent transformers and brain encoding models*](https://www.alphaxiv.org/pdf/2611.interpretable-weight-space-programming) (2026) · [code](https://github.com/csinva/weight-space-programming-autoresearch)
- Shuming Ma et al. — [*The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits*](https://arxiv.org/abs/2402.17764) (2024)
- TypeSafe [*System One*](https://docs.typesafe.ai/concepts/system-one.md) and Cloudflare *Clef*
- Calibration canon: [Redis](https://github.com/redis/redis), [SQLite](https://www.sqlite.org/src), [Lua](https://github.com/lua/lua), [TigerBeetle](https://github.com/tigerbeetle/tigerbeetle/blob/main/docs/TIGER_STYLE.md), [Go stdlib](https://cs.opensource.google/go/go), [Node.js](https://github.com/nodejs/node), [ripgrep](https://github.com/BurntSushi/ripgrep), [OpenBSD](https://github.com/openbsd/src), [LevelDB](https://github.com/google/leveldb), [Starlette](https://github.com/encode/starlette), [Requests](https://github.com/psf/requests), [Hono](https://github.com/honojs/hono), [Svelte](https://github.com/sveltejs/svelte), [HikariCP](https://github.com/brettwooldridge/HikariCP), [.NET Runtime](https://github.com/dotnet/runtime)
