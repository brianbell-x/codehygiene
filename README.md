# codehygiene

An 11 KB, 1.58-bit ternary weight-space programmed System One model for code readability and hygiene.

![codehygiene diff](assets/diff.png)

Traditional linters check syntax and formatting rules—code can pass every linter and unit test while still reading like cryptic golf or deeply nested, over-commented, copy-pasted AI slop. Instead of maintaining separate AST rules per language or calling a slow LLM judge, `codehygiene` runs a compact `428K`-parameter `1.58-bit` ternary neural network in pure NumPy (`~10 ms` on CPU) across 11 languages (`Python`, `C`, `C++`, `Rust`, `Go`, `Zig`, `JavaScript`, `TypeScript`, `Java`, `C#`, `Lua`):

- **1.58-bit ternary weight-space programmed:** No training data or gradient descent is used. Every 2D/3D weight, convolution, causal, attention, and embedding matrix in `weights.npz` has entries strictly in `{-1, 0, +1}` (`425,044` trits packed at `5` trits/byte + `3,253` 1D channel scales/biases = `10.5 KiB`), compiled directly from explicit mathematical circuits in `compile.py`.
- **System One judgment:** No text is generated. A single forward pass returns calibrated `0.0–1.0` hygiene scores, 6 style sub-scores, a style archetype, and line-level `Squiggle` diagnostics.

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

Install as an agent skill (Claude Code, Codex, Cursor, Antigravity, Copilot) with bundled offline weights via `npx skills`:

```bash
npx skills add brianbell-x/codehygiene     # current project
npx skills add brianbell-x/codehygiene -g  # global across all projects
```

Or run the CLI directly via `uvx` / `pip`:

```bash
uvx --from git+https://github.com/brianbell-x/codehygiene codehygiene .
pip install git+https://github.com/brianbell-x/codehygiene
```

## CLI

Directories, file paths, globs, and piped `stdin` are accepted:

```bash
codehygiene .                              # recursively lint supported source files
codehygiene src/ --check                   # exit 1 if any squiggle >= threshold (CI / agent gate)
codehygiene path/to/file.py --json         # structured JSON for coding agents & tooling
codehygiene "src/*.go" -t 0.60 --no-color  # custom squiggle probability threshold
cat snippet.rs | codehygiene --json        # evaluate raw source from stdin
```

## Python & RL Reward API

```python
import codehygiene

res = codehygiene.score(code)                        # full Result object (~10 ms on CPU)
issues = codehygiene.lint(code, threshold=0.55)      # list[Squiggle]
reward = codehygiene.rl_reward(code, tests_passed=True, weight=0.25)
```

In RL coding loops (`GRPO`, `PPO`, verifiers), `rl_reward` is gated by test pass rate: broken solutions (`tests_passed <= 0`) always receive `0.0` so models cannot reward-hack by emitting clean-looking non-functional code, while passing solutions receive `1.0` to `1.25` based on code hygiene.

## How It Works

All inference happens in `_forward()` (`codehygiene/__init__.py`) using standard matrix multiplies, 1D convolutions, and attention—with zero regexes, tokenizers, or AST parsers:

1. **2D ASCII Byte Grid (`encode_byte_grid`):** Source text is placed into a `(lines × 128 columns)` grid of 7-bit ASCII bytes, preserving visual indentation, alignment, and line width.
2. **Left-to-Right String & Comment Masking (`Block 0`):** Causal prefix matrices scan across each line to track quote parity (`"` / `'`) and comment openers (`//`, `# `, `-- `, `/* */`), preventing words inside strings or comments from misfiring code detectors.
3. **Pattern Convolutions & Word-Stem Sketches (`Block 1`):** Parallel `k=3, 7, 13` convolutions detect control-flow branches, chained operators, ceremonial naming (`*Factory`, `*Manager`), and golfed short-variable or magic-index soup (`xx = ...`, `[112]`), while 40 SimHash filters sketch 4-letter identifier stems separately in code and comments.
4. **Cross-Line Sink Attention & Readout (`Blocks 2–3`):** Multi-head attention compares each line against neighbors (`±32` lines) and file-wide indentation statistics—catching comments whose stems repeat the next line's code (`noisy_comment`), swallowed error handlers (`swallowed_error`), duplicated blocks (`duplicate_block`), and deep pyramids (`deep_nesting`)—before pooling into file-level scores.

## Output Schema (`--json` / `Result.to_dict()`)

| Field | Type | Description |
| :--- | :--- | :--- |
| `score` | `float` `[0, 1]` | Overall aesthetic score (`>= 0.70` clean, `0.40–0.70` mixed, `< 0.40` degraded). |
| `rl_hygiene` | `float` `[0, 1]` | Anti-slop structural hygiene signal tailored for RL reward shaping. |
| `scores` | `dict[str, float]` | Six `[0, 1]` sub-scores: `flatness`, `naming`, `simplicity`, `comments`, `error_handling`, `layout`. |
| `archetype` | `str` | Closest style profile: `minimal`, `clean`, `enterprise`, `dense`, or `slop`. |
| `archetype_probs` | `dict[str, float]` | Probability distribution summing to `1.0` across all 5 style profiles. |
| `squiggles` | `list[dict]` | `{line, col_start, col_end, probability, kind, message}` diagnostics above `threshold`. |

Squiggle `kind` values: `deep_nesting`, `poor_naming`, `noisy_comment`, `swallowed_error`, `dense_line`, `special_casing`, `duplicate_block`, `cramped_layout`.

## Benchmarks

Evaluated with zero training across a 54-slice production canon (`27` files, `11` languages) and annotated polyglot/anti-golf diagnostic suites, comparing the `32-bit float32` (`v0.2`) and `1.58-bit ternary` (`v0.3`) weight compilations:

| Metric | `float32` (`v0.2`) | `1.58-bit` Ternary (`v0.3`) | Delta |
| :--- | :---: | :---: | :---: |
| **Matrix weight domain (`425,044` weights)** | `float32` (`32.00` b/w) | **`{-1, 0, +1}` (`1.60` b/w, 5 trits/byte)** | **`20.0x` denser** |
| **Uncompressed payload (`trits` + `vecs`)** | `1,668.41 KiB` | **`95.74 KiB`** (`83.03 KiB` trits + `12.71 KiB` scales/biases) | **`17.4x` smaller** |
| **Compressed `weights.npz` on disk** | `33.03 KiB` (`33,821 B`) | **`10.52 KiB` (`10,772 B`)** | **`3.14x` smaller** |
| **Cold `.npz` load + unpack latency** | `8.03 ms` | **`5.06 ms`** | **`1.59x` faster** |
| **Warm CPU latency (`12` / `65` / `391` lines)** | `9.68` / `69.2` / `464.6 ms` | **`9.70` / `50.4` / `346.1 ms`** | **up to `1.37x` faster** |
| **Polyglot canon vs. slop AUROC** (`11` languages) | `0.986` | **`0.976`** | `-0.010` |
| **Controlled diagnostic AUROC** (clean vs. messy) | `1.000` | **`1.000`** | `0.000` |
| **Annotated squiggle & anti-golf recall** | `100%` (`14 / 14`) | **`100%` (`14 / 14`)** | `0.0%` |
| **Production canon squiggle rate** (`3,513` lines) | `1.99` / 100 lines | **`2.05` / 100 lines** | `+0.06` / 100L |
| **Clean trap & repository self-lint false positives** | `0` | **`0`** | `0` |

## Build Weights

All `32` ternary `{-1, 0, +1}` matrices (`425,044` trits packed at `5` trits per `uint8` byte, `92.6%` zero) and `3,253` 1D scale/bias scalars in `codehygiene/weights.npz` (`10.52 KiB`) are deterministically generated from `compile.py`:

```bash
python compile.py
```

## References

- **Weight-Space Programming:** Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar Khattab, [*Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights*](https://rikiyat.github.io/blog/weight-space-programming/), MIT CSAIL (2026) · [code & logs](https://github.com/rikiyat/lm_understand-logs)
- **Interpretable Weight-Space Programming (iWSP):** Chandan Singh et al., [*Interpretable weight-space programming builds transparent transformers and brain encoding models*](https://www.alphaxiv.org/pdf/2611.interpretable-weight-space-programming) (2026) · [code](https://github.com/csinva/weight-space-programming-autoresearch)
- **1.58-Bit Ternary Networks (BitNet b1.58):** Shuming Ma et al., [*The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits*](https://arxiv.org/abs/2402.17764) (2024)
- **System One Decision Models:** TypeSafe [*System One*](https://docs.typesafe.ai/concepts/system-one.md) (`noul`, `score`, and `choice` typed probability primitives without text generation) and Cloudflare *Clef*
- **Polyglot Calibration Canon:** Structural circuits and thresholds calibrated against production source from [Redis](https://github.com/redis/redis), [SQLite](https://www.sqlite.org/src), [Lua](https://github.com/lua/lua), [TigerBeetle (`TIGER_STYLE`)](https://github.com/tigerbeetle/tigerbeetle/blob/main/docs/TIGER_STYLE.md), [Go standard library](https://cs.opensource.google/go/go), [Node.js](https://github.com/nodejs/node), [ripgrep](https://github.com/BurntSushi/ripgrep), [OpenBSD](https://github.com/openbsd/src), [LevelDB](https://github.com/google/leveldb), [Starlette](https://github.com/encode/starlette), [Requests](https://github.com/psf/requests), [Hono](https://github.com/honojs/hono), [Svelte](https://github.com/sveltejs/svelte), [HikariCP](https://github.com/brettwooldridge/HikariCP), and [.NET Runtime](https://github.com/dotnet/runtime)
