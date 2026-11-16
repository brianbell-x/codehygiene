# Regression Cases

Minimal inputs for weaknesses measured on `v0.4.0` (commit `423c7e1`) across 15 public repositories. Each input is scored by `python3 corpus/evaluate.py --check`, which fails when the squiggles found differ from the `expect` list in [`cases.json`](cases.json). The `v040` field records what `v0.4.0` reported for the same input.

| Input | Weakness on `v0.4.0` | Expected now |
| :--- | :--- | :--- |
| [`raise_noqa.py`](raise_noqa.py) | `raise ValueError(msg)  # noqa: TRY004` labelled `swallowed_error` ("Ignored error") | no squiggle |
| [`why_comment_units.ts`](why_comment_units.ts) | Units comment `0.5kg per item unit` flagged `noisy_comment` | no squiggle |
| [`why_comment_prose.js`](why_comment_prose.js) | Why-comment containing `existing` flagged at `0.93` by the elision-placeholder cue | no squiggle |
| [`why_comment_block.zig`](why_comment_block.zig) | Last line of a three-line explanation flagged as restating the next line | no squiggle |
| [`doc_comment_initial.c`](doc_comment_initial.c) | Block-comment line containing `initialized` flagged by the narrator cue | no squiggle |
| [`preproc_define.c`](preproc_define.c) | `# define` lines inside `#ifndef` blocks read as comments restating code | no squiggle |
| [`length_bias_issues.py`](length_bias_issues.py) + [`length_bias_filler.py`](length_bias_filler.py) | Six squiggles pinned the score at `0.06` / `0.14` / `0.16` with the clean filler repeated `1` / `5` / `30` times | score rises with each repeat and ends `>= 0.80` |
| [`readme_example.py`](readme_example.py), [`noise_controls.py`](noise_controls.py), [`noise_controls.c`](noise_controls.c) | Controls: restating one-liners, `TODO`, step narration, elision placeholders, a short block-comment narrator, deep nesting and swallowed errors | every listed squiggle still fires |

These files hold deliberate smells, so they are excluded from the self-hygiene gate in the same way as `corpus/*/before/`.
