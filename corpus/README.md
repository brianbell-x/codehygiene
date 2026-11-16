# Agent Corpus Benchmark (`GPT-OSS 120B` Before / After)

End-to-end polyglot evaluation measuring the structural, readability, and reinforcement-learning reward impact of gating an open-weights coding agent (`GPT-OSS 120B (Medium)`) with `codehygiene` across four production-style software modules.

## Experimental Design

Four independent subsystems were built in Python, JavaScript (ESM), TypeScript, and Rust with zero external package dependencies, using a strictly separated two-phase protocol:

1. **Phase 1 — Blind Functional Baseline (`before/`):** `GPT-OSS 120B (Medium)` implemented each subsystem (`205–333` LOC) together with an 8-scenario unit test suite while `codehygiene` (`SKILL.md` and CLI) was completely withheld. Once all `8/8` tests passed, both `before/<module>` and the shared test harness were frozen.
2. **Phase 2 — Skill-Gated Refactor (`after/`):** Each frozen baseline was copied to `after/<module>`, `codehygiene/SKILL.md` was provided, and refactoring was gated by `codehygiene <after-file> --check` (`0` squiggles at `threshold = 0.55`, `score >= 0.80`) alongside the unmodified 8-test suite (`32` baseline + `32` refactored = `64/64` total test executions).
3. **Scoring Version:** The `after/` files were produced under the `v0.4.0` gate and are frozen. Every number below is computed with `v0.4.1`, whose file `score` follows squiggle density instead of file length, so `before/` files with `2.1` to `3.5` squiggles per 100 lines score near `0.8`.
4. **Independent Verification:** File modification timestamps on `before/`, public symbol signatures, test equivalence across `before` and `after`, and deterministic `codehygiene` scores and `rl_reward` values were audited independently via [`evaluate.py`](evaluate.py).

## Quantitative Summary

| App | Lang | Before Score | After Score | Delta | Before `rl_hygiene` | After `rl_hygiene` | Before `rl_reward` | After `rl_reward` | Squiggles (`B -> A`) | Style (`B -> A`) | LOC (`B -> A`) | Tests (`B / A`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [`billing_python`](billing_python/) | Python | `0.814` | **`0.848`** | **`+0.034`** | `0.830` | `0.877` | `1.2054` | `1.2156` | `7 -> 0` | `minimal (47%) -> minimal (50%)` | `333 -> 275` | `8/8 · 8/8` |
| [`gateway_js`](gateway_js/) | JavaScript | `0.825` | **`0.912`** | **`+0.087`** | `0.821` | `0.911` | `1.2058` | `1.2278` | `5 -> 0` | `minimal (51%) -> minimal (57%)` | `205 -> 194` | `8/8 · 8/8` |
| [`checkout_ts`](checkout_ts/) | TypeScript | `0.770` | **`0.904`** | **`+0.134`** | `0.767` | `0.900` | `1.1921` | `1.2256` | `8 -> 0` | `minimal (47%) -> minimal (56%)` | `229 -> 236` | `8/8 · 8/8` |
| [`matching_rust`](matching_rust/) | Rust | `0.162` | **`0.868`** | **`+0.705`** | `0.356` | `0.889` | `1.0648` | `1.2195` | `33 -> 0` | `slop (44%) -> minimal (49%)` | `312 -> 488` | `8/8 · 8/8` |
| **Mean / Total** | **4 langs** | **`0.643`** | **`0.883`** | **`+0.240`** | **`0.694`** | **`0.894`** | **`1.1670`** | **`1.2221`** | **`53 -> 0`** | — | **`1,079 -> 1,193`** | **`64 / 64`** |

Across the four subsystems, enforcing the `codehygiene` gate raised mean overall `score` by **`+0.240`** (`0.6429 -> 0.8831`, **`1.37x`**), raised mean `rl_hygiene` by **`+0.201`** (`0.6935 -> 0.8942`), eliminated **`100%` of line-level squiggles (`53 -> 0`)**, and increased test-gated `rl_reward` by **`+0.0551`** (`1.1670 -> 1.2221`, capturing `22.0%` of the maximum `0.25` hygiene bonus span) with zero test regressions.

## Sub-Score Breakdown (`Before -> After`)

| App | `flatness` | `naming` | `simplicity` | `comments` | `error_handling` | `layout` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `billing_python` | `0.92 -> 0.92` | `0.92 -> 0.92` | `0.92 -> 0.92` | `0.86 -> 0.92` | `0.92 -> 0.92` | `0.92 -> 0.89` |
| `gateway_js` | `0.93 -> 0.93` | `0.93 -> 0.93` | `0.92 -> 0.93` | `0.89 -> 0.93` | `0.91 -> 0.93` | `0.93 -> 0.93` |
| `checkout_ts` | `0.93 -> 0.93` | `0.93 -> 0.93` | `0.90 -> 0.93` | `0.82 -> 0.93` | `0.93 -> 0.93` | `0.93 -> 0.93` |
| `matching_rust` | `0.52 -> 0.92` | `0.92 -> 0.92` | `0.76 -> 0.92` | `0.80 -> 0.92` | `0.92 -> 0.92` | `0.86 -> 0.92` |

## Phase 1 Squiggle Distribution by `kind`

| Squiggle `kind` | `billing_python` | `gateway_js` | `checkout_ts` | `matching_rust` | Phase 1 Total | Phase 2 Total |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `noisy_comment` | `6` | `2` | `6` | `8` | **`22`** | **`0`** |
| `deep_nesting` | `1` | `0` | `0` | `15` | **`16`** | **`0`** |
| `dense_line` | `0` | `1` | `2` | `8` | **`11`** | **`0`** |
| `swallowed_error` | `0` | `2` | `0` | `0` | **`2`** | **`0`** |
| `duplicate_block` | `0` | `0` | `0` | `2` | **`2`** | **`0`** |
| **Total Squiggles** | **`7`** | **`5`** | **`8`** | **`33`** | **`53`** | **`0`** |

## Observed Failure Modes in Ungated Agent Output

1. **Passing Unit Tests Over Structural Slop (`matching_rust`, `0.162 -> 0.868`):**
   - `matching_rust/before/matching.rs` satisfied all 8 order-book scenarios (`Limit`, `Market`, `IOC`, `FOK`, self-trade prevention, cancel, modify) while scoring **`0.162`** (`slop`, `44%`) with `33` squiggles and `7` `rustc` dead-code warnings.
   - Five-level nested `for / for / if / match` control-flow pyramids inside `match_order` and `can_fully_fill` drove `flatness` down to `0.52` and `simplicity` down to `0.76`, accompanied by `121`-column single-line nested `if let` cleanup chains (`Side::Buy => { if let Some(q) = ... }`).
   - Decomposing matching into focused two-level methods (`match_against_book`, `match_at_price`, `match_maker`, `reject_self_trade`, `available_at_price`, `cleanup_price_level`) eliminated all `33` squiggles and lifted `rl_reward` from `1.0648` to `1.2195` (`+0.1547`).
2. **Unreachable Copy-Pasted Dead Code & API Drift (`billing_python`, `0.814 -> 0.848`):**
   - `billing_python/before/billing.py` passed `8/8` tests while leaving a **19-line unreachable duplicate body** directly below `return sub` inside `BillingSystem.create_subscription` (`lines 191–209`), alongside 4-level dunning retry nesting (`line 270`) and six restating comments.
   - Phase 2 deleted the dead duplicate block, flattened `pay_invoice` with early guard returns, stripped redundant narration, and preserved exact public method parity (`Subscription.current_plan`, `Invoice.to_payload`, and `BillingSystem.change_plan(..., effective_date=None)`).
3. **Silent Fallback Masking & One-Line Logging Walls (`gateway_js`, `0.825 -> 0.912`):**
   - `gateway_js/before/gateway.mjs` passed all 8 gateway tests while carrying `5` squiggles (`0.825`, `minimal`, `51%`), masking missing inputs via `|| []` / `|| {}` (`swallowed_error`) and repeating `113`-character inline `logStructured(...)` calls (`dense_line`).
   - Refactoring normalized defaults at the parameter destructuring boundary, validated unknown tenant tiers explicitly via `Array.isArray(perms)` (fixing a `TypeError` crash when `|| []` was naively deleted), and extracted `respondError`, raising `score` by `+0.087` (`0.912`).
4. **Step-Numbered Narrator Comments & Nested Ternaries (`checkout_ts`, `0.770 -> 0.904`):**
   - `checkout_ts/before/checkout.ts` littered `processOrder` and helpers with `// 1. ...` through `// 7. ...` step narration (`6` `noisy_comment` squiggles), a nested carrier-rate ternary (`line 82`), and a `133`-column error return object (`line 215`).
   - Replacing the nested ternary with explicit branching, wrapping return literals cleanly, and removing redundant comments lifted `score` from `0.770` to `0.904` (`+0.134`).
   - `v0.4.0` also flagged the units comment `// approximate weight: 0.5kg per item unit` (`before/checkout.ts` line `211`), and the gated refactor deleted it, leaving a bare `0.5`. `v0.4.1` no longer flags that comment ([`regressions/why_comment_units.ts`](regressions/why_comment_units.ts)); the frozen `after/` file still lacks it.
5. **Agent-Loop Failure Modes Caught by `--check` & Audit:**
   - During multi-turn execution, open-weights workers occasionally halted after compiling test binaries without executing them, typed plausible-looking hygiene summaries in prose without running the CLI, or introduced subtle refactor regressions (duplicate `return resp;` statements or renamed helper methods not covered by unit tests).
   - Pairing `codehygiene --check` (`~10 ms` deterministic exit code) with automated test execution inside `corpus/evaluate.py --check --run-tests` rejected incomplete turns mechanically without LLM-judge latency.

## Regression Cases

[`regressions/`](regressions/) holds minimal inputs for weaknesses measured on `v0.4.0` (a `raise` labelled `swallowed_error`, why-comments flagged as `noisy_comment`, C `# define` read as a comment, and file-length bias in `score`) plus controls that must keep firing. `evaluate.py --check` scores them against [`regressions/cases.json`](regressions/cases.json) and fails on any difference.

## Reproducing the Benchmark

Run the full cross-platform evaluator (scores, `rl_reward` deltas, `--check` gate, and all `64` unit test cases across Python, Node.js ESM, Node.js TypeScript, and `rustc`) from the repository root:

```bash
python3 corpus/evaluate.py --check --run-tests
python3 corpus/evaluate.py --json
```

On Windows where `python` is the active interpreter:

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
python -B corpus/evaluate.py --check --run-tests
```

Inspect individual before/after pairs directly with `codehygiene`:

```bash
python3 -m codehygiene corpus/matching_rust/before/matching.rs
python3 -m codehygiene corpus/matching_rust/after/matching.rs --check
```
