# GPT-OSS 120B Before / After Example (`gpt-oss-example/`)

End-to-end polyglot case study measuring structural changes and test preservation when an open-weights coding agent (`GPT-OSS 120B (Medium)`) refactors four dependency-free subsystems under `codehygiene`.

## Experimental Design & Scope Note

Four subsystems were implemented in Python, JavaScript (ESM), TypeScript, and Rust with zero external package dependencies using a two-phase protocol:

1. **Phase 1 — Blind Functional Baseline (`before/`):** `GPT-OSS 120B (Medium)` implemented each subsystem (`205–333` LOC) and an 8-scenario unit test suite while `codehygiene` (`SKILL.md` and CLI) was withheld. Once all `8/8` tests passed, `before/<module>` and the shared test harness were frozen.
2. **Phase 2 — Gate-Guided Refactor (`after/`):** Each baseline was copied to `after/<module>`, `codehygiene/SKILL.md` was provided, and refactoring ran against `codehygiene <after-file> --check` (`0` squiggles at `threshold = 0.55`, `score >= 0.80`) alongside the unmodified 8-test suite (`32` baseline + `32` refactored = `64/64` total test executions).
3. **Interpreting Circularity & Independent Metrics:** Because Phase 2 refactored directly against `codehygiene --check`, the `score` and squiggle deltas demonstrate gate compliance without breaking behavior rather than independent human preference. To provide non-circular evidence alongside `codehygiene` scores, [`evaluate.py`](evaluate.py) also measures weight-independent structural metrics (maximum code indentation depth, lines exceeding `100` columns, comment line counts, compiler warnings, and total LOC).

## Quantitative Summary (`v0.5.0`)

| App | Lang | Before Score | After Score | Delta | Before `rl_hygiene` | After `rl_hygiene` | Before `rl_reward` | After `rl_reward` | Squiggles (`B -> A`) | Style (`B -> A`) | LOC (`B -> A`) | Tests (`B / A`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [`billing_python`](billing_python/) | Python | `0.779` | **`0.862`** | `+0.083` | `0.804` | `0.885` | `1.0198` | `1.2184` | `6 -> 0` | `clean (46%) -> minimal (50%)` | `333 -> 275` | `8/8 · 8/8` |
| [`gateway_js`](gateway_js/) | JavaScript | `0.840` | **`0.886`** | `+0.046` | `0.853` | `0.897` | `1.0249` | `1.2230` | `5 -> 0` | `minimal (52%) -> minimal (55%)` | `205 -> 194` | `8/8 · 8/8` |
| [`checkout_ts`](checkout_ts/) | TypeScript | `0.698` | **`0.892`** | `+0.194` | `0.683` | `0.899` | `1.0119` | `1.2240` | `9 -> 0` | `minimal (41%) -> minimal (54%)` | `229 -> 236` | `8/8 · 8/8` |
| [`matching_rust`](matching_rust/) | Rust | `0.024` | **`0.855`** | **`+0.831`** | `0.060` | `0.882` | `1.0003` | `1.2171` | `27 -> 0` | `slop (89%) -> minimal (48%)` | `312 -> 488` | `8/8 · 8/8` |
| **Mean / Total** | **4 langs** | **`0.585`** | **`0.874`** | **`+0.288`** | **`0.600`** | **`0.891`** | **`1.0142`** | **`1.2206`** | **`47 -> 0`** | — | **`1,079 -> 1,193`** | **`64 / 64`** |

## Weight-Independent Structural Metrics (`Before -> After`)

| App | Max Indent Cols (`B -> A`) | Lines `>100` Cols (`B -> A`) | Comment Lines (`B -> A`) | Compiler Warnings (`B -> A`) | LOC (`B -> A`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `billing_python` | `28 -> 16` | `4 -> 1` | `55 -> 24` | `0 -> 0` | `333 -> 275` (`-17%`) |
| `gateway_js` | `6 -> 6` | `6 -> 4` | `21 -> 9` | `0 -> 0` | `205 -> 194` (`-5%`) |
| `checkout_ts` | `6 -> 8` | `3 -> 1` | `30 -> 8` | `0 -> 0` | `229 -> 236` (`+3%`) |
| `matching_rust` | `24 -> 16` | `16 -> 1` | `30 -> 15` | `7 -> 0` (`rustc`) | `312 -> 488` (`+56%`) |

Note on LOC trade-offs: `billing_python` shrank by `17%` after deleting an unreachable 19-line duplicate block and narrator comments, whereas `matching_rust` grew by `56%` (`312 -> 488` lines) because `16` dense single-line `if let` chains (`>100` cols) and 5-level nested loops (`24`-space indent) were expanded into explicit multi-line structs and focused two-level helper methods while resolving `7` `rustc` dead-code warnings.

## Concrete Issues Caught in Ungated Agent Output

1. **Nested Order-Book Pyramids & One-Line Chains (`matching_rust`):**
   - `matching_rust/before/matching.rs` passed all 8 order-book scenarios while carrying `7` `rustc` dead-code warnings, 5-level `for / for / if / match` pyramids inside `match_order` and `can_fully_fill`, and `121`-column single-line nested `if let` cleanup chains.
   - Extracting `match_against_book`, `match_at_price`, `match_maker`, `reject_self_trade`, `available_at_price`, and `cleanup_price_level` reduced max indentation from `24` to `16` columns and lines `>100` cols from `16` to `1`.
2. **Unreachable Copy-Pasted Dead Code (`billing_python`):**
   - `billing_python/before/billing.py` passed `8/8` tests while leaving a 19-line unreachable duplicate body directly below `return sub` inside `BillingSystem.create_subscription` (`lines 191–209`) alongside restating comments.
3. **Silent Fallback Masking (`gateway_js`):**
   - `gateway_js/before/gateway.mjs` masked missing inputs via `|| []` / `|| {}` (`swallowed_error`) and repeated `113`-character inline `logStructured(...)` calls (`dense_line`). Normalizing defaults at parameter destructuring boundaries and validating unknown tenant tiers explicitly via `Array.isArray(perms)` caught a latent `TypeError` path.
4. **Step-Numbered Narration & Nested Ternaries (`checkout_ts`):**
   - `checkout_ts/before/checkout.ts` annotated `processOrder` with `// 1.` through `// 7.` step narration (`noisy_comment`), a nested carrier-rate ternary, and a `133`-column return literal.

## Reproducing

```bash
python3 gpt-oss-example/evaluate.py --check --run-tests
python3 gpt-oss-example/evaluate.py --json
```

On Windows where `python` is the active interpreter:

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
python -B gpt-oss-example/evaluate.py --check --run-tests
```
