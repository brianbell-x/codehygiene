# codehygiene

A 33 KB, weight-space programmed System One model for linting and scoring code aesthetics. All 427K float32 parameters are compiled analytically in pure NumPy over raw ASCII byte grids with no gradient descent, tokenizers, regexes, or AST parsers.

## Install

```bash
pip install .
```

## CLI

```bash
codehygiene path/to/file.py
codehygiene src/*.go --threshold 0.55 --json
```

## Python API

```python
import codehygiene

res = codehygiene.score(code)          # score, rl_hygiene, scores, archetype, squiggles
issues = codehygiene.lint(code)        # line/col Squiggle diagnostics
reward = codehygiene.rl_reward(code, tests_passed=True)
```

## Build Weights

All 53 float32 tensors in `codehygiene/weights.npz` are generated from `compile.py`:

```bash
python compile.py
```
