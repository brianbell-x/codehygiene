# codehygiene

A 33 KB, weight-space programmed System One model that lints and scores code for aesthetics and understandability.

Traditional linters check syntax and formatting rules—code can pass every linter and unit test while still reading like cryptic golf or deeply nested, over-commented, copy-pasted AI slop. Instead of maintaining separate AST rules per language or calling a slow LLM judge, `codehygiene` runs a compact `427K`-parameter neural network in pure NumPy (`~15 ms` on CPU) across 11 languages (`Python`, `C`, `C++`, `Rust`, `Go`, `Zig`, `JavaScript`, `TypeScript`, `Java`, `C#`, `Lua`):

- **Weight-space programmed:** No training data or gradient descent is used. Every weight in `weights.npz` is compiled directly from explicit mathematical circuits in `compile.py`.
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

Run directly without installing via `uvx`, or install with `pip`:

```bash
uvx --from git+https://github.com/brianbell-x/codehygiene codehygiene .
```

```bash
pip install git+https://github.com/brianbell-x/codehygiene
# or from a local clone:
pip install .
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

res = codehygiene.score(code)                        # full Result object (~15 ms on CPU)
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

Evaluated with zero training across a 54-slice production canon (`27` files, `11` languages) and an 18-case annotated diagnostic suite:

| Evaluation | Result |
| :--- | :--- |
| **Polyglot canon vs. slop AUROC** (`27` production files, `11` languages) | **`0.985`** |
| **Controlled diagnostic AUROC** (`18` cases: clean mean `0.903` vs. slop mean `0.060`) | **`1.000`** |
| **Annotated squiggle recall** (nesting, ceremony, narrator comments, swallowed errors, clones) | **`100%`** (`14 / 14`) |
| **Production canon squiggle rate** (`0.00` Rust/C#, `0.51` Go, `2.06` Zig, `2.21` C, `2.31` JS) | **`2.64` / 100 lines** |
| **Clean trap false positives** (Python comprehensions, `TigerStyle` Zig, terse C, Java builders) | **`0`** |
| **Runtime footprint & latency** (pure NumPy on CPU, no GPU) | **`33.0 KB` · `~15 ms`** |

## Build Weights

All 53 float32 tensors in `codehygiene/weights.npz` are deterministically generated from `compile.py`:

```bash
python compile.py
```

## References

- **Weight-Space Programming:** Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar Khattab, [*Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights*](https://rikiyat.github.io/blog/weight-space-programming/), MIT CSAIL (2026) · [code & logs](https://github.com/rikiyat/lm_understand-logs)
- **Interpretable Weight-Space Programming (iWSP):** Chandan Singh et al., [*Interpretable weight-space programming builds transparent transformers and brain encoding models*](https://www.alphaxiv.org/pdf/2611.interpretable-weight-space-programming) (2026) · [code](https://github.com/csinva/weight-space-programming-autoresearch)
- **System One Decision Models:** TypeSafe [*System One*](https://docs.typesafe.ai/concepts/system-one.md) (`noul`, `score`, and `choice` typed probability primitives without text generation) and Cloudflare *Clef*
- **Polyglot Calibration Canon:** Structural circuits and thresholds calibrated against production source from [Redis](https://github.com/redis/redis), [SQLite](https://www.sqlite.org/src), [Lua](https://github.com/lua/lua), [TigerBeetle (`TIGER_STYLE`)](https://github.com/tigerbeetle/tigerbeetle/blob/main/docs/TIGER_STYLE.md), [Go standard library](https://cs.opensource.google/go/go), [Node.js](https://github.com/nodejs/node), [ripgrep](https://github.com/BurntSushi/ripgrep), [OpenBSD](https://github.com/openbsd/src), [LevelDB](https://github.com/google/leveldb), [Starlette](https://github.com/encode/starlette), [Requests](https://github.com/psf/requests), [Hono](https://github.com/honojs/hono), [Svelte](https://github.com/sveltejs/svelte), [HikariCP](https://github.com/brettwooldridge/HikariCP), and [.NET Runtime](https://github.com/dotnet/runtime)
