# Benchmarks (`benchmarks/`)

Reproducible evaluation suites separating controlled in-sample circuit tests from adversarial reward-gaming probes and untuned held-out standard-library modules.

## Suites Included

1. **Controlled Diagnostic & Clean-Trap Suite (`suites.json`, in-sample):**
   - **Messy & static-slop recall (`18` cases):** Covers all eight squiggle categories across Python, TypeScript, JavaScript, Rust, and Go, including syntactic AI slop that passes default AST/syntax linters (`ruff`, `tsc`, `node --check`, `rustc`) such as `except Exception: ...`, step narration, placeholder elisions, commented-out code, redundant internal runtime type hedges, and 2-letter variable golf.
   - **Clean traps & linter-divergent clean idioms (`18` cases):** Verifies zero false positives on flat early-return guard clauses (`5` returns that strict single-exit `PLR0911` rules penalize), concise self-explanatory functions without boilerplate docstrings (`C0116` / `D103`), 9-line `matmul` using `i/j/k`, `x = 1`, `os.environ.get("HOST", "localhost")`, multiline `"""..."""` docstrings and `/* ... */` block comments containing keywords (`Deprecated`, `Factory`, `existing`), odd-column PEP 8 hanging parameter lists, 3-level class methods, `####` comment banners, and `AbortError` `catch` resets.
2. **Adversarial & RL Anti-Gaming Probes (`run_benchmarks.py`):**
   - Verifies invariance under `except: pass` -> `except: ...`, resistance to 40 appended trivial helper functions and 200-line docstring padding, full-file evaluation past line `2,048`, byte-identical `weights.npz` compilation, and numerical parity between Python (`codehygiene/__init__.py`) and Web Worker JavaScript (`docs/engine.js`).
3. **Held-Out CPython Standard Library Evaluation (`25` unmodified modules):**
   - Evaluates `25` CPython standard library modules (`~25,300` lines) from the active Python installation without line-count truncation, reporting overall median/mean scores, compact (`<650` LOC) vs. large legacy (`>=650` LOC) module medians, squiggles per 100 lines, and warm CPU latencies across file sizes.

## Running

```bash
python3 benchmarks/run_benchmarks.py --check
python3 benchmarks/run_benchmarks.py --json
```
