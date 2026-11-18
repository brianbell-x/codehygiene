# Evals (`evals/`)

Unified, version-gated evaluation dataset ([`evals.json`](evals.json)) and runner ([`run_evals.py`](run_evals.py)) separating controlled circuit tests, issue regressions, 20-language polyglot suites, edit-alignment checks, adversarial reward-gaming probes, JavaScript engine parity, and held-out standard-library modules.

## Structure of `evals.json` (`codehygiene-evals-v2`)

Every top-level suite and test case carries a `"since"` version tag, and `"version"` at the root of `evals.json` is enforced against `pyproject.toml` and all public module docstrings by `run_evals.py --check`:

1. **Version & Directory Hygiene Gate (`run_version_sync_check`):**
   - Verifies `pyproject.toml`, `codehygiene.__version__`, `evals/evals.json["version"]`, and the opening `codehygiene vX.Y.Z` docstring/JSDoc header across all six public `.py` / `.js` modules stay synchronized, and rejects stray unmerged files inside `evals/`.
2. **Controlled Diagnostic & Clean-Trap Suite (`controlled`):**
   - `18` messy recall cases across all eight squiggle kinds and `18` clean-trap idioms (`0` false positives, `AUROC = 1.0`).
3. **Adversarial & RL Anti-Gaming Probes (`probe_seed` + `run_adversarial_probes`):**
   - Verifies invariance under `except: pass` -> `except: ...`, `contextlib.suppress(Exception)` anti-evasion (while narrow `suppress(FileNotFoundError)` stays clean), `+40` helper-padding resistance, `+200`-line docstring-padding invariance, unbounded detection past line `2,048`, byte-identical `weights.npz` compilation, and exact numerical parity between `codehygiene/__init__.py` and `codehygiene/engine.js`.
4. **Issue & Circuit Regression Suites (`regressions`, gated by `--check`):**
   - Keyed by GitHub issue number or circuit slug (`"3"`, `"broad-suppress"`, `"v06-block0-lexical"`, `"v06-polyglot-circuits"`, `"v06-structural-friction"`, `"v06-clean-concise-calibration"`), each holding `repro` files (`0` squiggles), `clean_variants` (`0` squiggles), and line-and-kind `recall_guards`.
5. **20-Language Polyglot & Edit-Alignment Suite (`polyglot`):**
   - Deduplicated 20-language `clean_traps` (`dev` / `held_out`), `messy_recall` (including `structural_friction`), and `18` `edit_pairs` (`after_better` directionality + gated `no_cosmetic_jump` and `no_bloat_gain` anti-hack pairs). When `.codex-notes/polyglot-diag/corpus_index.json` is present locally, `196`-file TypeSafe Jev `cognitive_clarity` correlation is also evaluated.
6. **Held-Out Corpora & CPU Latency (`stdlib_modules` + optional `GOMODCACHE`):**
   - Evaluates `25` unmodified CPython standard-library modules (`~25,300` lines), optional third-party Go module-cache files (`--skip-go` to omit), and warm pure-NumPy CPU latencies.

## Running

```bash
python3 evals/run_evals.py --check
python3 evals/run_evals.py --json
python3 evals/run_evals.py --check --skip-go --skip-js
python3 evals/run_evals.py --compare baseline.npz codehygiene/weights.npz
```
