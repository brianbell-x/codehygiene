# codehygiene

A 33 KB, weight-space programmed System One model for linting and scoring code aesthetics. All 427,114 float32 parameters (`53` tensors) are compiled analytically in pure NumPy over raw 7-bit ASCII `(L, 128)` byte grids—with no gradient descent, training corpus, tokenizers, regular expressions, or AST parsers.

Calibrated across 11 languages (`Python`, `C`, `C++`, `Rust`, `Go`, `Zig`, `JavaScript`, `TypeScript`, `Java`, `C#`, `Lua`), a single forward pass returns calibrated System One judgments (`noul` probabilities, 6 structural `score` spectra, 5-way `choice` archetype) alongside line-level `Squiggle` diagnostics in ~15 ms on CPU.

## Quickstart

Run directly without installation via `uvx`, or install with `pip`:

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

`rl_reward` is multiplicatively gated by functional correctness: `0.0` is returned whenever `tests_passed <= 0`, preventing reward hacking on broken but well-formatted rollouts; passing rollouts receive `tests_passed * (1.0 + weight * blend(score, rl_hygiene))` in `[1.0, 1.25]`.

## Output Schema (`--json` / `Result.to_dict()`)

| Field | Type | Description |
| :--- | :--- | :--- |
| `score` | `float` `[0, 1]` | Overall aesthetic judgment (`>= 0.70` clean, `0.40–0.70` mixed, `< 0.40` degraded). |
| `rl_hygiene` | `float` `[0, 1]` | Anti-slop and structural hygiene signal tailored for RL rollout shaping. |
| `scores` | `dict[str, float]` | Six `[0, 1]` spectra: `flat_flow`, `terse_naming`, `direct_idiom`, `comment_signal`, `error_discipline`, `visual_rhythm`. |
| `archetype` | `str` | Top style class: `austere_systems`, `clean_modern`, `enterprise_ceremony`, `clever_dense`, or `rl_ai_slop`. |
| `archetype_probs` | `dict[str, float]` | Calibrated probability distribution summing to `1.0` across all 5 archetypes. |
| `squiggles` | `list[dict]` | `{line, col_start, col_end, probability, kind, message}` spans exceeding `threshold`. |

Diagnostic `kind` values: `nesting_pyramid`, `naming_ceremony`, `narrator_comment`, `defensive_slop`, `clever_chaining`, `special_case_wart`, `copy_paste_clone`, `rhythm_clash`.

## Build Weights

All 53 float32 tensors in `codehygiene/weights.npz` are deterministically generated from closed-form circuits in `compile.py`:

```bash
python compile.py
```

## References

- **Weight-Space Programming:** Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar Khattab, [*Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights*](https://rikiyat.github.io/blog/weight-space-programming/), MIT CSAIL (2026) · [code & logs](https://github.com/rikiyat/lm_understand-logs)
- **Interpretable Weight-Space Programming (iWSP):** Chandan Singh et al., [*Interpretable weight-space programming builds transparent transformers and brain encoding models*](https://www.alphaxiv.org/pdf/2611.interpretable-weight-space-programming) (2026) · [code](https://github.com/csinva/weight-space-programming-autoresearch)
- **System One Decision Models:** TypeSafe [*System One*](https://docs.typesafe.ai/concepts/system-one.md) (`noul`, `score`, and `choice` typed probability primitives without text generation) and Cloudflare *Clef*
- **Polyglot Calibration Canon:** Structural circuits and thresholds calibrated against production source from [Redis](https://github.com/redis/redis), [SQLite](https://www.sqlite.org/src), [Lua](https://github.com/lua/lua), [TigerBeetle (`TIGER_STYLE`)](https://github.com/tigerbeetle/tigerbeetle/blob/main/docs/TIGER_STYLE.md), [Go standard library](https://cs.opensource.google/go/go), [Node.js](https://github.com/nodejs/node), [ripgrep](https://github.com/BurntSushi/ripgrep), [OpenBSD](https://github.com/openbsd/src), [LevelDB](https://github.com/google/leveldb), [Starlette](https://github.com/encode/starlette), [Requests](https://github.com/psf/requests), [Hono](https://github.com/honojs/hono), [Svelte](https://github.com/sveltejs/svelte), [HikariCP](https://github.com/brettwooldridge/HikariCP), and [.NET Runtime](https://github.com/dotnet/runtime)
