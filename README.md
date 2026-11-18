# codehygiene

A `23.84 KiB`, `1.58-bit` system one model for code readability.

![codehygiene diff](assets/diff.svg)

`35` `{-1, 0, +1}` tensors (`1,244,706` trits packed at `5` trits/byte + `7,102` 1D channel scales/biases) are deterministically compiled from [`compile.py`](compile.py) into `weights.npz` without training data or gradient descent, and executed in pure NumPy or browser `Float32Array` workers across `20+` languages (`Python`, `TypeScript`, `JavaScript`, `Rust`, `Go`, `C`, `C++`, `C#`, `Java`, `Ruby`, `Swift`, `Kotlin`, `Zig`, `SQL`, `Perl`, `Visual Basic`, `Fortran`, `MATLAB`, `Lua`, and `PHP`).

Structural smells missed by syntax/AST linters—pyramid nesting, in-branch loop-cursor mutations, narrator comments, swallowed errors, `*Factory`/`*Manager` ceremony, 2-char variable golf, magic-offset indexing, boolean/out-reference parameter traps, and copy-pasted or alpha-renamed blocks—are flagged while early-return guard clauses, unpadded helpers, string-heavy format lines, and typed math/ODE kernels pass cleanly. Functional bugs, type errors, and security vulnerabilities are out of scope; newly written or modified files are targeted rather than legacy repo sweeps.

## Usage

```bash
# Agent skill (Claude Code, Codex, Cursor, Antigravity, Copilot)
npx skills add brianbell-x/codehygiene -g
npx skills update codehygiene -y

# CLI (paths, globs, directories, or stdin; exits 1 if any squiggle >= threshold)
uvx --from git+https://github.com/brianbell-x/codehygiene codehygiene path/to/changed.py --check
codehygiene "src/*.rs" -t 0.60 --json --no-update-check

# prek / pre-commit hook
prek add git+https://github.com/brianbell-x/codehygiene
prek install
```

```python
import codehygiene

res = codehygiene.score(code)                                    # Result
issues = codehygiene.lint(code, threshold=0.55)                  # list[Squiggle]
reward = codehygiene.rl_reward(code, tests_passed=True, weight=0.15)
```

```yaml
# .pre-commit-config.yaml
repos:
  - repo: git+https://github.com/brianbell-x/codehygiene
    rev: v0.6.1
    hooks:
      - id: codehygiene
        # Optional: use -t arg to adjust squiggle confidence cutoff, higher = fewer flags.
```

- **Output (`Result` / `--json`):** `score` (`[0, 1]`), `rl_hygiene` (`[0, 1]`), `scores` (`flatness`, `naming`, `simplicity`, `comments`, `error_handling`, `layout`), `archetype` / `archetype_probs` (`minimal`, `clean`, `enterprise`, `dense`, `slop`), and `squiggles` (`{line, col_start, col_end, probability, kind, message}` across `deep_nesting`, `poor_naming`, `noisy_comment`, `swallowed_error`, `dense_line`, `special_casing`, `duplicate_block`, `cramped_layout`).
- **RL reward (`rl_reward`):** Gated multiplicatively on `tests_passed`, pooled over Bayesian-shrunk means (`_PRIOR_LINES = 16`) and span-damped 32-line peak-window densities with docstrings masked, and damped by `1 / (1 + 1.5 * load^0.85)` over telescopic-span, severity-weighted squiggle events (`weight=0.10–0.20` recommended inside test-verified `GRPO`/`PPO` loops).

## Architecture

Inference is executed in [`codehygiene/__init__.py`](codehygiene/__init__.py) (`_forward`) via matrix multiplies, 1D convolutions, causal prefix sums, and banded attention—without regexes, tokenizers, or ASTs:

1. **Byte Grid (`encode_byte_grid`):** Full-file `(L × 128)` 7-bit ASCII grid without line-count truncation, evaluated over occupied column spans with L3-tiled `im2col` ribbon convolutions.
2. **Masking (`Blocks 0–1`):** Horizontal 32-quote parity (`"` / `'` / `` ` ``), polyglot comment and raw-string gates (`//`, `# `, `##`, `-- `, `/* */`, `!>`/`!!`/`!`, `%`, `' `, Zig `\\`), string-discounted effective column width (`C2_EFF_LEN`), vertical `"""`/`'''`/`/* */` sawtooth state, 16-line Dyck parenthesis balance, and previous-line comment and `err != nil` taps.
3. **Local Circuits & Sketches (`Block 1`):** Multi-scale `k=3, 7, 13` ternary convolutions and three 40-dim SimHash sketches (`code`, `comment`, and alpha-invariant operator/branch `skeleton`).
4. **Banded Sink Attention & Readout (`Blocks 2–3`):** Two `O(L)` relative-position attention passes (`12` and `9` heads over `±32` lines + Bayesian sink priors) feeding per-line squiggle heads and paired-clamp saturating document readouts.

## Benchmarks

Unified, version-tagged evaluation suites in [`evals/evals.json`](evals/evals.json)—covering controlled 8-kind circuit gates, per-issue regression suites, deduplicated 20-language polyglot traps and structural-friction recall, before/after edit-alignment pairs, adversarial RL-gaming probes, JS parity, untuned held-out CPython 3.13 stdlib modules (`25` files, `25,306` lines), and, when present, a held-out sample of the local Go module cache (`352` files, `60,302` lines)—are evaluated via [`evals/run_evals.py`](evals/run_evals.py):

| Suite / Metric | `v0.4.0` (`12.97 KiB`) | `v0.5.1` (`14.85 KiB`) | `v0.6.1` (`23.84 KiB`) |
| :--- | :---: | :---: | :---: |
| **Controlled recall** (`evals.json`, in-sample) | `17 / 18` | `18 / 18` (AUROC `1.000`) | **`18 / 18` (AUROC `1.000`)** |
| **Clean traps & linter-divergent idioms** (`evals.json`) | `8 / 18` FPs | `0 / 18` FPs (mean `0.831`) | **`0 / 18` FPs (mean `0.867`)** |
| **20-language clean traps** (`evals.json`, all / held-out) | — | `8 / 39` · `4 / 16` (`86` FP sq) | **`35 / 39` · `14 / 16` (`10` FP sq)** |
| **196-file 20-language clarity Pearson `r`** (`score` / `rl_reward`) | — | `0.271` / `0.257` | **`0.452` / `0.497`** |
| **[Issue #3](https://github.com/brianbell-x/codehygiene/issues/3) + `v0.6` regression suites** (`evals.json`) | — | `2 / 2` suites | **`6 / 6` suites (`53 / 53` guards)** |
| **Anti-gaming & bloat-loophole edit pairs** (`evals.json`) | — | `2 / 4` | **`4 / 4` (max jump `0.000`)** |
| **Adversarial: `except: pass` vs `except: ...` `rl_reward` gap** | `+0.181` | `0.0000` | **`0.0000`** |
| **Adversarial: `+40` trivial helpers (`rl_reward` gain)** | `+0.096` (`0.098 -> 0.481`) | `+0.007` (`0.291 -> 0.437`) | **`+0.012` (`0.172 -> 0.280`, `3` sq)** |
| **Adversarial: slop placed past line `2,048`** | `0` sq (`2,048L` cap) | `4` sq (`score` `0.141`) | **`4` sq (`score` `0.134`)** |
| **Python vs. [`codehygiene/engine.js`](codehygiene/engine.js) max delta** | `< 0.0001` (`<= 2,048L`) | `0.0000` | **`0.0000`** |
| **Held-out CPython 3.13 stdlib squiggle rate** | `6.09 / 100` lines | `1.90 / 100` lines (`480` sq) | **`1.77 / 100` lines (`447` sq)** |
| **Held-out CPython 3.13 stdlib median score** (all / `<650` LOC) | `0.093` / `0.24` | `0.391` / `0.588` | **`0.370` / `0.459`** |
| **Held-out Go module cache squiggle rate** | — | `1.66 / 100` lines (`1,003` sq) | **`1.22 / 100` lines (`737` sq)** |
| **Self-hygiene across all `6` public repo modules** | — | `0` sq | **`0` sq (`0.868` across `6 / 6` files)** |
| **Warm CPU latency** (`12` / `65` / `333` / `2,649` lines) | `10.2` / `63.6` / `~380` / `3,902 ms` | `~7` / `~35` / `~200` / `~1,500 ms` | **`~6` / `~23` / `~109` / `~975 ms`** |

## Development

```bash
python3 compile.py                              # deterministic weight compilation -> weights.npz
python3 evals/run_evals.py --check              # version-sync, controlled, regression, polyglot, anti-gaming, JS-parity & stdlib gates
python3 evals/run_evals.py --compare a.npz b.npz
```

## References

- Rikiya Takehi, Jayden Teoh, Alex Zhang, Omar Khattab — [*Sparks of Weight-Space Programming*](https://rikiyat.github.io/blog/weight-space-programming/), MIT CSAIL (2026) · [code](https://github.com/rikiyat/lm_understand-logs)
- Chandan Singh et al. — [*Interpretable weight-space programming builds transparent transformers and brain encoding models*](https://www.alphaxiv.org/pdf/2611.interpretable-weight-space-programming) (2026) · [code](https://github.com/csinva/weight-space-programming-autoresearch)
- Shuming Ma et al. — [*The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits*](https://arxiv.org/abs/2402.17764) (2024)
- TypeSafe [*System One*](https://docs.typesafe.ai/concepts/system-one.md) and Cloudflare *Clef*
- Calibration canon: [Redis](https://github.com/redis/redis), [SQLite](https://www.sqlite.org/src), [Lua](https://github.com/lua/lua), [TigerBeetle](https://github.com/tigerbeetle/tigerbeetle/blob/main/docs/TIGER_STYLE.md), [Go stdlib](https://cs.opensource.google/go/go), [Node.js](https://github.com/nodejs/node), [ripgrep](https://github.com/BurntSushi/ripgrep), [OpenBSD](https://github.com/openbsd/src), [LevelDB](https://github.com/google/leveldb), [Starlette](https://github.com/encode/starlette), [Requests](https://github.com/psf/requests), [Hono](https://github.com/honojs/hono), [Svelte](https://github.com/sveltejs/svelte), [HikariCP](https://github.com/brettwooldridge/HikariCP), [.NET Runtime](https://github.com/dotnet/runtime)
