# codehygiene

An offline structural linter for 20+ languages in one 8 KB model.

- **No parsers or config:** runs in NumPy, Node, or a browser, even on partial snippets.
- **Beyond formatters:** catches structural clutter formatters leave intact, including unbroken code walls and comments that restate code, without policing names or domain conventions.
- **Built for coding agents:** prints a score, flagged lines, and a cleaner target shape instead of a bare rule code.

Source is read as a grid of bytes, not words. The ternary weights are programmed in [`compile.py`](compile.py), not trained.

## Install

```bash
# As an agent skill (Claude Code, Codex, Cursor, Copilot, ...)
npx skills add brianbell-x/codehygiene -g

# Or run once without installing
uvx --from git+https://github.com/brianbell-x/codehygiene codehygiene path/to/file.py --check
```

## Use

```bash
codehygiene changed.py --check   # exit 1 if anything is flagged
codehygiene src/ --json          # machine-readable results
```

Clean files print nothing. A flagged file prints its score, the flagged lines, and a target shape that scores cleaner:

```text
status.js  0.555/1 -> 0.881/1
    2!|   if (code === 200) return "ok";
    3!|   if (code === 301) return "moved";
    4!|   if (code === 404) return "not found";
    5!|   if (code === 500) return "server error";
  target:
    1 | ■■■■■■■■■=■■■([
    2 |  [■,■■]
    3 |  [■,■■■],
    4 |  [■,■■■■■],
    5 |  [■,■■■■■■■]
    6 | ];
    7 |
    8 | ■■■■■■■■■(■■){
    9 |  ■■■■■■■■■■■(■)■■■■■■;  <-2..6
   10 | }
```

Each `■` is two bytes of code, and `<-N` marks the source line a row came from.

## What gets flagged

| Kind | Looks like |
| :--- | :--- |
| `deep_nesting` | deeply nested control flow inside a function |
| `dense_line` | long call chains, nested ternaries, several statements on one line |
| `cramped_layout` | 20+ lines with no blank line, or ragged indentation |
| `duplicate_block` | similar 4+ line blocks pasted nearby |
| `repetitive_structure` | 3+ `if x == ...` branches in a row |
| `comment_echo` | comments that restate the next line, or commented-out code |

Bugs, type errors, and security issues are out of scope. Newly written or changed code is the intended target, not whole legacy repositories.

## Python and JavaScript

```python
import codehygiene

codehygiene.score(code)  # score, squiggles, per-line probabilities
```

The same model runs in browsers and Node through [`codehygiene/engine.js`](codehygiene/engine.js).

## Background

- [*Sparks of Weight-Space Programming*](https://rikiyat.github.io/blog/weight-space-programming/)
- [*The Era of 1-bit LLMs*](https://arxiv.org/abs/2402.17764)
