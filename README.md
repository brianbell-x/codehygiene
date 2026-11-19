# codehygiene

An offline linter that flags hard-to-read code by its shape.

Source is read as a grid of bytes, not words, so names and spelling are ignored. The model is an 8 KB file of ternary weights that are programmed in [`compile.py`](compile.py), not trained. It runs on NumPy and covers 20+ languages.

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
sync.go  0.233/1 -> 0.881/1
    9!|                             if sink.Send(item.ID, item.Payload) {
   10!|                                 sent += 1
  target:
    3 |   ■■,■■■:=■■■■■■■{
    4 |     ■■■:■■■  <-4..8
    5 |     ■■■:■■■■  <-4..8
    6 |
    7 |     ■■■■■■(■■■,■■■■■■){  <-9
    8 |      ■■■=■  <-10
    9 |   }
```

Each `■` is two bytes of code, and `<-N` marks the source line a row came from.

## What gets flagged

| Kind | Looks like |
| :--- | :--- |
| `deep_nesting` | 4+ levels of indentation inside a function |
| `dense_line` | long call chains, nested ternaries, several statements on one line |
| `cramped_layout` | 20+ lines with no blank line, or ragged indentation |
| `duplicate_block` | the same 4+ lines pasted nearby, even with renamed variables |
| `repetitive_structure` | 4+ `if x == ...` branches in a row |
| `comment_echo` | comments that restate the next line, or commented-out code |

Bugs, type errors, and security issues are out of scope. Newly written or changed code is the intended target, not whole legacy repositories.

## Python and JavaScript

```python
import codehygiene

codehygiene.score(code)  # score, squiggles, per-line probabilities
```

The same model runs in browsers and Node through [`codehygiene/engine.js`](codehygiene/engine.js).

## More

- [`AGENTS.md`](AGENTS.md): architecture, compilation, and evals
- [`codehygiene/SKILL.md`](codehygiene/SKILL.md): the agent workflow
- Background: [*Sparks of Weight-Space Programming*](https://rikiyat.github.io/blog/weight-space-programming/) and [*The Era of 1-bit LLMs*](https://arxiv.org/abs/2402.17764)
