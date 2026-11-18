"""codehygiene v0.6.1 — Unified controlled, regression, polyglot, and parity runner."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import tomllib

EVALS_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVALS_DIR.parent
GO_CORPUS_LINES = 60_000
GO_CORPUS_MAX_FILE_LINES = 1_500

ALLOWED_EVAL_FILES = frozenset({"evals.json", "run_evals.py", "README.md"})
VERSIONED_SOURCE_FILES = (
    "codehygiene/__init__.py",
    "codehygiene/__main__.py",
    "codehygiene/update_check.py",
    "codehygiene/engine.js",
    "compile.py",
    "evals/run_evals.py",
)

_GO_TEST_FUNC = re.compile(r"^func (\([^)]*\) )?(Test|Benchmark|Example|Fuzz)\w*\(")
_GO_DIRECTIVE = re.compile(r"^//(go:|\s*\+)")
_NARRATION = re.compile(r"(?i)todo|fixme|step \d")
_STRONG_CEREMONY = re.compile(r"Manager|Factory|Impl\b|Wrapper|Helper|Abstract")
_FALLBACK_LITERAL = re.compile(r"\|\||\?\?| or |catch|except")
_ERR_CHECK = re.compile(r"\berr != nil\b")

_NODE_PARITY_SCRIPT = "\n".join([
    "import fs from 'node:fs';",
    "import { unpackNpzBuffer, scoreCode, computeRlReward } from './codehygiene/engine.js';",
    "const buf = fs.readFileSync('./codehygiene/weights.npz');",
    "const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);",
    "const w = await unpackNpzBuffer(ab);",
    "const paths = ['compile.py', 'codehygiene/engine.js', 'codehygiene/__init__.py'];",
    "const out = {};",
    "for (const p of paths) {",
    "  const src = fs.readFileSync(p, 'utf8');",
    "  const r = scoreCode(src, w);",
    "  const rew = computeRlReward(r, true);",
    "  out[p] = { score: r.score, rl: r.rlHygiene, rew, sq: r.squiggles.length };",
    "}",
    "console.log(JSON.stringify(out));",
])


def _is_doc_tail(text: str, prev: str) -> bool:
    return text.startswith("//") and prev.startswith("//") and not _NARRATION.search(text)


def _is_service_noun(text: str, _prev: str) -> bool:
    return bool(re.search(r"Service|Builder", text)) and not _STRONG_CEREMONY.search(text)


def _is_empty_literal(text: str, prev: str) -> bool:
    if "{}" not in text or _FALLBACK_LITERAL.search(text):
        return False
    swallows_err = (
        text.startswith("return ") and text.endswith(", nil") and bool(_ERR_CHECK.search(prev))
    )
    return not swallows_err


_GO_FP_SHAPES = (
    ("directive", "noisy_comment", lambda text, _prev: bool(_GO_DIRECTIVE.match(text))),
    ("doc_tail", "noisy_comment", _is_doc_tail),
    ("test_name", "poor_naming", lambda text, _prev: bool(_GO_TEST_FUNC.match(text))),
    ("service_naming", "poor_naming", _is_service_noun),
    ("empty_literal", "swallowed_error", _is_empty_literal),
)


@lru_cache(maxsize=1)
def load_modules() -> tuple[ModuleType, ModuleType]:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    return importlib.import_module("codehygiene"), importlib.import_module("compile")


def _safe_median(values: list[float]) -> float:
    return round(float(np.median(values)), 4) if values else 0.0


def _safe_mean(values: list[float]) -> float:
    return round(float(np.mean(values)), 4) if values else 0.0


def _squiggle_pairs(
    hygiene: ModuleType, code: str, weights: dict[str, np.ndarray]
) -> list[tuple[int, str]]:
    return [(sq.line, sq.kind) for sq in hygiene.score(code, weights=weights).squiggles]


def run_version_sync_check(dataset: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    hygiene, _ = load_modules()
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text("utf-8"))
    pkg_ver = str(pyproject["project"]["version"])
    expected_tag = f"codehygiene v{pkg_ver}"
    failures: list[str] = []

    if hygiene.__version__ != pkg_ver:
        failures.append(
            f"version_sync/__init__: __version__={hygiene.__version__!r} != {pkg_ver!r}"
        )
    if dataset.get("version") != pkg_ver:
        failures.append(
            f"version_sync/evals.json: version={dataset.get('version')!r} != {pkg_ver!r}"
        )

    for rel_path in VERSIONED_SOURCE_FILES:
        head = (REPO_ROOT / rel_path).read_text("utf-8")[:400]
        if expected_tag not in head:
            failures.append(
                f"version_sync/{rel_path}: missing {expected_tag!r} in header docstring"
            )

    stray = sorted(
        entry.name
        for entry in EVALS_DIR.iterdir()
        if entry.name != "__pycache__" and entry.name not in ALLOWED_EVAL_FILES
    )
    if stray:
        failures.append(f"version_sync/evals_dir: unexpected files {stray}")

    summary = {
        "package_version": pkg_ver,
        "dataset_version": dataset.get("version"),
        "sources_checked": len(VERSIONED_SOURCE_FILES),
        "stray_eval_files": stray,
        "synced": not failures,
    }
    return summary, failures


def compute_auroc(clean_scores: list[float], messy_scores: list[float]) -> float:
    if not clean_scores or not messy_scores:
        return 0.0
    clean_arr = np.asarray(clean_scores, dtype=np.float64)[:, None]
    messy_arr = np.asarray(messy_scores, dtype=np.float64)[None, :]
    wins = float(np.sum(clean_arr > messy_arr) + 0.5 * np.sum(clean_arr == messy_arr))
    return wins / float(clean_arr.size * messy_arr.size)


def _pearson(x_vals: list[float], y_vals: list[float]) -> float:
    if len(x_vals) < 3:
        return 0.0
    x_arr = np.asarray(x_vals, dtype=np.float64)
    y_arr = np.asarray(y_vals, dtype=np.float64)
    denom = float(np.std(x_arr) * np.std(y_arr))
    if denom < 1e-16:
        return 0.0
    cov = float(np.mean((x_arr - np.mean(x_arr)) * (y_arr - np.mean(y_arr))))
    return round(cov / denom, 4)


def _spearman(x_vals: list[float], y_vals: list[float]) -> float:
    if len(x_vals) < 3:
        return 0.0
    x_ranks = np.argsort(np.argsort(np.asarray(x_vals, dtype=np.float64))).astype(np.float64)
    y_ranks = np.argsort(np.argsort(np.asarray(y_vals, dtype=np.float64))).astype(np.float64)
    return _pearson(x_ranks.tolist(), y_ranks.tolist())


def _score_with_rl(code: str, weights: dict[str, np.ndarray]) -> tuple[Any, float]:
    hygiene, _ = load_modules()
    res = hygiene.score(code, weights=weights)
    if hasattr(hygiene, "_rl_from_result"):
        return res, float(hygiene._rl_from_result(res))
    if weights is hygiene.load_weights():
        return res, float(hygiene.rl_reward(code, tests_passed=True))
    blended = 0.5 * res.score + 0.5 * res.rl_hygiene
    damp = 1.0 / (1.0 + 1.5 * len(res.squiggles))
    return res, 1.0 + 0.25 * blended * damp


def run_controlled_suites(
    ctrl: dict[str, Any], weights: dict[str, np.ndarray]
) -> tuple[dict[str, Any], list[str]]:
    hygiene, _ = load_modules()
    failures: list[str] = []
    messy_scores: list[float] = []
    recalled = 0
    for item in ctrl["messy_cases"]:
        res = hygiene.score(item["code"], weights=weights)
        messy_scores.append(res.score)
        kinds = {sq.kind for sq in res.squiggles}
        if item["expected_kind"] in kinds:
            recalled += 1
        else:
            failures.append(
                f"messy/{item['id']}: expected {item['expected_kind']}, got {sorted(kinds)}"
            )

    clean_scores: list[float] = []
    false_positives = 0
    for item in ctrl["clean_traps"]:
        res = hygiene.score(item["code"], weights=weights)
        clean_scores.append(res.score)
        if res.squiggles:
            false_positives += 1
            hits = [(sq.line, sq.kind, round(sq.probability, 2)) for sq in res.squiggles]
            failures.append(f"clean_trap/{item['id']}: unexpected squiggles {hits}")
        elif res.score < 0.72:
            failures.append(f"clean_trap/{item['id']}: score {res.score:.3f} < 0.72")

    summary = {
        "messy_recall": f"{recalled}/{len(ctrl['messy_cases'])}",
        "recalled_count": recalled,
        "messy_total": len(ctrl["messy_cases"]),
        "clean_trap_fps": false_positives,
        "clean_trap_total": len(ctrl["clean_traps"]),
        "auroc": round(compute_auroc(clean_scores, messy_scores), 4),
        "mean_clean_score": _safe_mean(clean_scores),
        "mean_messy_score": _safe_mean(messy_scores),
    }
    return summary, failures


def _probe_ellipsis_and_suppress(hygiene: ModuleType) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    pass_code = "def run():\n    try:\n        work()\n    except Exception:\n        pass\n"
    ell_code = "def run():\n    try:\n        work()\n    except Exception:\n        ...\n"
    rew_pass = hygiene.rl_reward(pass_code, tests_passed=True)
    rew_ell = hygiene.rl_reward(ell_code, tests_passed=True)
    if not hygiene.lint(ell_code) or abs(rew_pass - rew_ell) > 1e-4:
        failures.append(f"probe/ellipsis_bypass: pass={rew_pass:.4f} vs ellipsis={rew_ell:.4f}")

    swallow_base = (
        "import os\n\ndef cleanup(target):\n    try:\n        os.remove(target)\n"
        "    except Exception:\n        pass\n"
    )
    broad_suppress = (
        "import contextlib\nimport os\n\ndef cleanup(target):\n"
        "    with contextlib.suppress(Exception):\n        os.remove(target)\n"
    )
    narrow_suppress = (
        "import contextlib\nimport os\n\ndef cleanup(target):\n"
        "    with contextlib.suppress(FileNotFoundError):\n        os.remove(target)\n"
    )
    rew_swallow = hygiene.rl_reward(swallow_base, tests_passed=True)
    rew_suppress = hygiene.rl_reward(broad_suppress, tests_passed=True)
    broad_kinds = {sq.kind for sq in hygiene.lint(broad_suppress)}
    narrow_hits = hygiene.lint(narrow_suppress)
    if "swallowed_error" not in broad_kinds or (rew_suppress - rew_swallow) > 0.005:
        failures.append(
            f"probe/broad_suppress: kinds={sorted(broad_kinds)}, "
            f"reward {rew_swallow:.4f}->{rew_suppress:.4f}"
        )
    if narrow_hits:
        failures.append(f"probe/narrow_suppress_fp: {[(s.line, s.kind) for s in narrow_hits]}")

    metrics = {
        "ellipsis_rl_delta": round(abs(rew_pass - rew_ell), 6),
        "broad_suppress_rl_delta": round(rew_suppress - rew_swallow, 4),
        "narrow_suppress_squiggles": len(narrow_hits),
    }
    return metrics, failures


def _probe_padding_and_horizon(
    hygiene: ModuleType, slop_sample: str
) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    helpers = "\n\n".join(f"def helper_fn_{idx}(val):\n    return val + {idx}" for idx in range(40))
    padded_helpers = f"{slop_sample}\n\n{helpers}\n"
    base_res = hygiene.score(slop_sample)
    pad_res = hygiene.score(padded_helpers)
    base_rew = hygiene.rl_reward(slop_sample, tests_passed=True)
    pad_rew = hygiene.rl_reward(padded_helpers, tests_passed=True)
    if pad_res.score >= 0.45 or (pad_rew - base_rew) > 0.025:
        failures.append(
            f"probe/helper_padding: score {base_res.score:.3f}->{pad_res.score:.3f}, "
            f"reward {base_rew:.4f}->{pad_rew:.4f}"
        )

    doc_lines = "\n".join(['"""', *(f"Documentation line {idx}." for idx in range(200)), '"""'])
    doc_pad_res = hygiene.score(f"{doc_lines}\n\n{slop_sample}\n")
    if abs(doc_pad_res.score - base_res.score) > 0.02:
        failures.append(f"probe/docstring_padding: {base_res.score:.3f} -> {doc_pad_res.score:.3f}")

    long_prefix = "\n".join(f"ITEM_{idx} = {idx}" if idx % 4 == 0 else "" for idx in range(2060))
    tail_res = hygiene.score(f"{long_prefix}\n\n{slop_sample}\n")
    tail_lines = [sq.line for sq in tail_res.squiggles if sq.line > 2048]
    if len(tail_lines) < 3 or tail_res.score >= 0.20:
        failures.append(
            f"probe/unbounded_lines: score={tail_res.score:.3f}, tail_squiggles={tail_lines}"
        )

    metrics = {
        "base_slop_score": round(base_res.score, 4),
        "padded_40_helpers_score": round(pad_res.score, 4),
        "padded_40_helpers_rl_delta": round(pad_rew - base_rew, 4),
        "padded_200L_docstring_score": round(doc_pad_res.score, 4),
        "after_line_2048_score": round(tail_res.score, 4),
        "after_line_2048_squiggles": len(tail_lines),
    }
    return metrics, failures


def run_adversarial_probes(slop_sample: str) -> tuple[dict[str, Any], list[str]]:
    hygiene, weight_compiler = load_modules()
    ell_metrics, ell_fail = _probe_ellipsis_and_suppress(hygiene)
    pad_metrics, pad_fail = _probe_padding_and_horizon(hygiene, slop_sample)

    with tempfile.TemporaryDirectory() as tmp_dir:
        rebuilt = Path(tmp_dir) / "weights.npz"
        weight_compiler.compile_weights(rebuilt)
        live_bytes = (REPO_ROOT / "codehygiene" / "weights.npz").read_bytes()
        deterministic = live_bytes == rebuilt.read_bytes()

    failures = [*ell_fail, *pad_fail]
    if not deterministic:
        failures.append("probe/weights_determinism: weights.npz drift detected")
    summary = {**ell_metrics, **pad_metrics, "byte_identical_weights": deterministic}
    return summary, failures


def run_js_parity_check() -> tuple[dict[str, Any], list[str]]:
    if shutil.which("node") is None:
        return {"available": False, "max_score_diff": None}, []

    hygiene, _ = load_modules()
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", _NODE_PARITY_SCRIPT],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        err = proc.stderr.strip()
        return {"available": True, "error": err}, [f"js_parity: {err}"]

    js_map = json.loads(proc.stdout)
    max_diff = 0.0
    failures: list[str] = []
    for rel_path, js_vals in js_map.items():
        src = (REPO_ROOT / rel_path).read_text("utf-8")
        py_res = hygiene.score(src)
        py_rew = hygiene.rl_reward(src, tests_passed=True)
        diff = max(
            abs(py_res.score - js_vals["score"]),
            abs(py_res.rl_hygiene - js_vals["rl"]),
            abs(py_rew - js_vals["rew"]),
        )
        max_diff = max(max_diff, diff)
        if diff > 1e-3 or len(py_res.squiggles) != js_vals["sq"]:
            failures.append(
                f"js_parity/{rel_path}: diff={diff:.6f}, sq={len(py_res.squiggles)}"
            )

    summary = {"available": True, "max_abs_diff": round(max_diff, 6), "files_checked": len(js_map)}
    return summary, failures


def _evaluate_regression_entry(
    reg_id: str,
    spec: dict[str, Any],
    hygiene: ModuleType,
    weights: dict[str, np.ndarray],
) -> tuple[dict[str, Any], list[str]]:
    tag = f"regression/{reg_id}"
    failures: list[str] = []
    reported_total = 0
    still_flagged = 0
    for case in spec["repro"]:
        hits = _squiggle_pairs(hygiene, case["code"], weights)
        reported = set(case["reported_lines"])
        reported_total += len(reported)
        still_flagged += len(reported & {line for line, _ in hits})
        if hits:
            failures.append(f"{tag}/repro/{case['id']}: squiggles {hits}")

    clean_fps = 0
    for case in spec["clean_variants"]:
        hits = _squiggle_pairs(hygiene, case["code"], weights)
        if hits:
            clean_fps += 1
            failures.append(f"{tag}/clean/{case['id']}: squiggles {hits}")

    recalled = 0
    for case in spec["recall_guards"]:
        hits = _squiggle_pairs(hygiene, case["code"], weights)
        if (case["expected_line"], case["expected_kind"]) in hits:
            recalled += 1
        else:
            want = f"{case['expected_kind']}@{case['expected_line']}"
            failures.append(f"{tag}/guard/{case['id']}: expected {want}, got {hits}")

    entry = {
        "since": spec.get("since", "0.6.0"),
        "reported_still_flagged": f"{still_flagged}/{reported_total}",
        "clean_variant_fps": f"{clean_fps}/{len(spec['clean_variants'])}",
        "recall_guards": f"{recalled}/{len(spec['recall_guards'])}",
    }
    return entry, failures


def run_regressions(
    regressions: dict[str, Any], weights: dict[str, np.ndarray]
) -> tuple[dict[str, Any], list[str]]:
    hygiene, _ = load_modules()
    failures: list[str] = []
    summary: dict[str, Any] = {}
    for reg_id, spec in regressions.items():
        entry, errs = _evaluate_regression_entry(reg_id, spec, hygiene, weights)
        summary[reg_id] = entry
        failures.extend(errs)
    return summary, failures


def _resolve_corpus_file(corpus_root: Path, rel_path: str) -> Path | None:
    prefixes = (
        corpus_root,
        corpus_root / ".codex-notes",
        corpus_root / ".codex-notes" / "slop-corpus",
        corpus_root / "slop-corpus",
    )
    for base in prefixes:
        candidate = base / rel_path
        if candidate.is_file():
            return candidate
    return None


def _eval_corpus_calibration(
    weights: dict[str, np.ndarray], corpus_root: Path
) -> dict[str, Any]:
    idx_path = corpus_root / ".codex-notes" / "polyglot-diag" / "corpus_index.json"
    if not idx_path.is_file():
        return {"available_files": 0}
    index_rows = json.loads(idx_path.read_text("utf-8"))["corpus_index"]

    splits: dict[str, dict[str, list[float]]] = {
        name: {"ch": [], "rl": [], "clar": []} for name in ("all", "dev", "held_out")
    }
    avail = cliff_collapses = blind_spot_escapes = fps_rem = fps_tot = 0
    for row in index_rows:
        fpath = _resolve_corpus_file(corpus_root, str(row["rel_path"]))
        if fpath is None:
            continue
        avail += 1
        res, rl_val = _score_with_rl(fpath.read_text("utf-8", "replace"), weights)
        clar = float(row["cognitive_clarity"])
        fric = float(row.get("structural_friction", 0.0))
        for key in ("all", str(row["split"])):
            splits[key]["ch"].append(res.score)
            splits[key]["rl"].append(rl_val)
            splits[key]["clar"].append(clar)

        cliff_collapses += int(clar >= 0.65 and res.score < 0.25)
        blind_spot_escapes += int((fric >= 0.60 or clar <= 0.42) and res.score >= 0.80)
        flagged = {sq.line for sq in res.squiggles}
        fp_lines = [int(ln) for ln in row.get("confirmed_fp_lines", [])]
        fps_tot += len(fp_lines)
        fps_rem += sum(1 for ln in fp_lines if ln in flagged)

    if avail == 0:
        return {"available_files": 0}

    metrics: dict[str, Any] = {
        "available_files": avail,
        "cliff_collapses": cliff_collapses,
        "blind_spot_escapes": blind_spot_escapes,
        "confirmed_fp_lines_remaining": fps_rem,
        "confirmed_fp_lines_total": fps_tot,
    }
    for sp_name, vecs in splits.items():
        metrics[f"{sp_name}_pearson_score_vs_clarity"] = _pearson(vecs["ch"], vecs["clar"])
        metrics[f"{sp_name}_spearman_score_vs_clarity"] = _spearman(vecs["ch"], vecs["clar"])
        metrics[f"{sp_name}_pearson_rl_vs_clarity"] = _pearson(vecs["rl"], vecs["clar"])
    return metrics


def _eval_polyglot_traps(
    cases: list[dict[str, Any]], hygiene: ModuleType, weights: dict[str, np.ndarray]
) -> dict[str, Any]:
    scored = [
        (case["split"], len(hygiene.score(case["code"], weights=weights).squiggles))
        for case in cases
    ]
    dev = [n_sq == 0 for split, n_sq in scored if split == "dev"]
    held_out = [n_sq == 0 for split, n_sq in scored if split != "dev"]
    return {
        "passed": sum(dev) + sum(held_out),
        "total": len(cases),
        "dev_passed": sum(dev),
        "dev_total": len(dev),
        "held_out_passed": sum(held_out),
        "held_out_total": len(held_out),
        "total_fp_squiggles": sum(n_sq for _, n_sq in scored),
    }


def _check_anti_hack_pair(
    pair: dict[str, Any], sc_delta: float, rl_delta: float, after_sq: int
) -> tuple[bool, float, str | None]:
    expect = pair["expect"]
    if expect == "no_reward_gain":
        ok = rl_delta <= 0.005 and after_sq >= 1
        err = None if ok else f"polyglot/anti_hack/{pair['id']}: rl_delta={rl_delta:+.4f}"
        return ok, 0.0, err
    if expect == "no_bloat_gain":
        ok = sc_delta <= 0.005 and rl_delta <= 0.005
        err = (
            None
            if ok
            else f"polyglot/no_bloat_gain/{pair['id']}: sc_delta={sc_delta:+.4f}"
        )
        return ok, max(0.0, sc_delta), err
    ok = sc_delta <= 0.45
    err = None if ok else f"polyglot/cosmetic_jump/{pair['id']}: sc_delta={sc_delta:+.4f}"
    return ok, sc_delta, err


def _eval_edit_pairs(
    pairs: list[dict[str, Any]], weights: dict[str, np.ndarray]
) -> tuple[dict[str, Any], list[str]]:
    cleanup_ok = cleanup_total = anti_hack_ok = anti_hack_total = 0
    max_cosmetic_jump = 0.0
    failures: list[str] = []

    for pair in pairs:
        b_res, b_rl = _score_with_rl(pair["before_code"], weights)
        a_res, a_rl = _score_with_rl(pair["after_code"], weights)
        sc_delta = a_res.score - b_res.score
        rl_delta = a_rl - b_rl
        if pair["expect"] == "after_better":
            cleanup_total += 1
            cleanup_ok += int(sc_delta > 0.0 and rl_delta > 0.0)
            continue
        anti_hack_total += 1
        ok, jump, err = _check_anti_hack_pair(pair, sc_delta, rl_delta, len(a_res.squiggles))
        anti_hack_ok += int(ok)
        max_cosmetic_jump = max(max_cosmetic_jump, jump)
        if err is not None:
            failures.append(err)

    summary = {
        "cleanup_passed": cleanup_ok,
        "cleanup_total": cleanup_total,
        "anti_hack_passed": anti_hack_ok,
        "anti_hack_total": anti_hack_total,
        "max_cosmetic_score_jump": round(max_cosmetic_jump, 4),
    }
    return summary, failures


def run_polyglot_suite(
    poly: dict[str, Any], weights: dict[str, np.ndarray], corpus_root: Path = REPO_ROOT
) -> tuple[dict[str, Any], list[str]]:
    hygiene, _ = load_modules()
    by_cat: dict[str, dict[str, int]] = {}
    for case in poly["messy_recall"]:
        cat = str(case.get("category", "other"))
        bucket = by_cat.setdefault(cat, {"hits": 0, "total": 0})
        bucket["total"] += 1
        res = hygiene.score(case["code"], weights=weights)
        if res.squiggles or (cat == "structural_friction" and res.score < 0.72):
            bucket["hits"] += 1

    edit_summary, failures = _eval_edit_pairs(poly["edit_pairs"], weights)
    summary = {
        "clean_traps": _eval_polyglot_traps(poly["clean_traps"], hygiene, weights),
        "messy_recall": {
            "hits": sum(b["hits"] for b in by_cat.values()),
            "total": sum(b["total"] for b in by_cat.values()),
            "by_category": by_cat,
        },
        "edit_pairs": edit_summary,
        "corpus_calibration": _eval_corpus_calibration(weights, corpus_root),
    }
    return summary, failures


def run_heldout_stdlib_suite(
    module_names: list[str], weights: dict[str, np.ndarray]
) -> dict[str, Any]:
    hygiene, _ = load_modules()
    stdlib_dir = Path(os.__file__).resolve().parent
    all_scores: list[float] = []
    compact_scores: list[float] = []
    large_scores: list[float] = []
    kind_counts: Counter[str] = Counter()
    total_lines = found_modules = 0

    for rel_name in module_names:
        mod_path = stdlib_dir / rel_name
        if not mod_path.is_file():
            continue
        found_modules += 1
        src = mod_path.read_text("utf-8", "replace")
        loc = len(src.splitlines())
        res = hygiene.score(src, weights=weights)
        all_scores.append(res.score)
        (compact_scores if loc < 650 else large_scores).append(res.score)
        total_lines += loc
        kind_counts.update(sq.kind for sq in res.squiggles)

    total_sq = sum(kind_counts.values())
    return {
        "python_version": sys.version.split()[0],
        "modules_evaluated": found_modules,
        "total_lines": total_lines,
        "median_score": _safe_median(all_scores),
        "mean_score": _safe_mean(all_scores),
        "compact_lt650_median": _safe_median(compact_scores),
        "large_ge650_median": _safe_median(large_scores),
        "total_squiggles": total_sq,
        "squiggles_per_100_lines": round(total_sq / max(total_lines, 1) * 100.0, 2),
        "by_kind": dict(sorted(kind_counts.items())),
    }


def classify_go_issue_signature(lines: list[str], line_no: int, kind: str) -> str | None:
    text = lines[line_no - 1].strip()
    prev = lines[line_no - 2].strip() if line_no > 1 else ""
    for label, shape_kind, matches in _GO_FP_SHAPES:
        if kind == shape_kind and matches(text, prev):
            return label
    return None


def sample_go_corpus(root: Path) -> list[tuple[str, str]]:
    ranked: list[tuple[str, str, Path]] = []
    for path in root.rglob("*.go"):
        rel = path.relative_to(root).as_posix()
        if rel.startswith("cache/") or "/vendor/" in rel or "/testdata/" in rel:
            continue
        ranked.append((hashlib.sha1(rel.encode("utf-8")).hexdigest(), rel, path))

    picked: list[tuple[str, str]] = []
    total = 0
    for _, rel, path in sorted(ranked):
        src = path.read_text("utf-8", "replace")
        head = src[:1000]
        loc = len(src.splitlines())
        if ("Code generated" in head and "DO NOT EDIT" in head) or loc > GO_CORPUS_MAX_FILE_LINES:
            continue
        picked.append((rel, src))
        total += loc
        if total >= GO_CORPUS_LINES:
            break
    return picked


def run_go_module_corpus(weights: dict[str, np.ndarray]) -> dict[str, Any]:
    root = Path(os.environ.get("GOMODCACHE") or Path.home() / "go" / "pkg" / "mod")
    if not root.is_dir():
        return {"available": False}

    hygiene, _ = load_modules()
    kind_counts: Counter[str] = Counter()
    signatures: Counter[str] = Counter()
    files = sample_go_corpus(root)
    total_lines = 0
    for _, src in files:
        lines = src.splitlines()
        total_lines += len(lines)
        for sq in hygiene.score(src, weights=weights).squiggles:
            kind_counts[sq.kind] += 1
            signature = classify_go_issue_signature(lines, sq.line, sq.kind)
            if signature:
                signatures[signature] += 1

    total_sq = sum(kind_counts.values())
    return {
        "available": True,
        "files": len(files),
        "total_lines": total_lines,
        "total_squiggles": total_sq,
        "squiggles_per_100_lines": round(total_sq / max(total_lines, 1) * 100.0, 2),
        "by_kind": dict(sorted(kind_counts.items())),
        "issue3_signatures": dict(sorted(signatures.items())),
        "issue3_signature_total": sum(signatures.values()),
    }


def measure_latencies(weights: dict[str, np.ndarray]) -> dict[str, float]:
    hygiene, _ = load_modules()
    lines = (REPO_ROOT / "codehygiene" / "__init__.py").read_text("utf-8").splitlines()
    samples = {
        "warm_12_lines_ms": "\n".join(lines[:12]),
        "warm_65_lines_ms": "\n".join(lines[:65]),
        "warm_333_lines_ms": "\n".join(lines[:333]),
    }
    argparse_path = Path(os.__file__).resolve().parent / "argparse.py"
    if argparse_path.is_file():
        samples["warm_argparse_full_ms"] = argparse_path.read_text("utf-8", "replace")

    timings: dict[str, float] = {}
    for key, code in samples.items():
        hygiene.score(code, weights=weights)
        runs = 3 if len(code.splitlines()) < 1000 else 1
        start = time.perf_counter()
        for _ in range(runs):
            hygiene.score(code, weights=weights)
        timings[key] = round((time.perf_counter() - start) * 1000.0 / runs, 2)
    return timings


def format_report(payload: dict[str, Any]) -> str:
    vsync = payload["version_sync"]
    ctrl = payload["controlled"]
    adv = payload["adversarial"]
    poly = payload["polyglot"]
    traps = poly["clean_traps"]
    recall = poly["messy_recall"]
    edits = poly["edit_pairs"]
    calib = poly["corpus_calibration"]
    std = payload["held_out_stdlib"]
    go_c = payload["go_corpus"]

    cats = ", ".join(f"{k}={v['hits']}/{v['total']}" for k, v in recall["by_category"].items())
    lines = [
        f"## 0. Version & Schema Sync (`v{vsync['package_version']}`)",
        (
            f"- Synced: `{vsync['synced']}` "
            f"(`{vsync['sources_checked']}` public headers + `evals/evals.json`)"
        ),
        "",
        "## 1. Controlled Diagnostic & Clean-Trap Suite (`evals/evals.json`)",
        (
            f"- Messy & static-slop recall: **{ctrl['messy_recall']}** "
            f"(mean `{ctrl['mean_messy_score']:.3f}`)"
        ),
        (
            f"- Clean traps & linter-divergent idioms: "
            f"**{ctrl['clean_trap_fps']} / {ctrl['clean_trap_total']} FPs** "
            f"(mean `{ctrl['mean_clean_score']:.3f}`)"
        ),
        f"- Clean vs. messy AUROC: **{ctrl['auroc']:.4f}**",
        "",
        "## 2. Adversarial & RL Anti-Gaming Probes",
        f"- `except: pass` vs `except: ...` `rl_reward` delta: `{adv['ellipsis_rl_delta']:.6f}`",
        (
            f"- Slop + 40 trivial helpers: "
            f"`{adv['base_slop_score']:.3f} -> {adv['padded_40_helpers_score']:.3f}`, "
            f"`rl_reward` delta `{adv['padded_40_helpers_rl_delta']:+.4f}`"
        ),
        (
            f"- Slop + 200-line docstring: "
            f"`{adv['base_slop_score']:.3f} -> {adv['padded_200L_docstring_score']:.3f}`"
        ),
        (
            f"- Slop past line 2,048: `{adv['after_line_2048_score']:.3f}` "
            f"(`{adv['after_line_2048_squiggles']}` sq)"
        ),
        (
            f"- `except Exception: pass` -> `contextlib.suppress(Exception)` `rl_reward` delta: "
            f"`{adv['broad_suppress_rl_delta']:+.4f}` · narrow `suppress(FileNotFoundError)` "
            f"squiggles: `{adv['narrow_suppress_squiggles']}`"
        ),
        (
            f"- Byte-identical `weights.npz`: `{adv['byte_identical_weights']}` · "
            f"JS parity: `{payload['js_parity']}`"
        ),
        "",
        "## 3. Issue & Circuit Regression Suites (`evals/evals.json`)",
    ]
    for reg_id, res in payload["issues"].items():
        label = f"Issue #{reg_id}" if reg_id.isdigit() else f"`{reg_id}`"
        lines.append(
            f"- {label} (`v{res['since']}`): reported **{res['reported_still_flagged']}** · "
            f"clean FPs **{res['clean_variant_fps']}** · recall guards **{res['recall_guards']}**"
        )

    lines.extend([
        "",
        "## 4. 20-Language Polyglot & Edit-Alignment Suite (`evals/evals.json`)",
        (
            f"- Clean traps (deduplicated): **{traps['passed']}/{traps['total']}** "
            f"(dev `{traps['dev_passed']}/{traps['dev_total']}`, "
            f"held-out `{traps['held_out_passed']}/{traps['held_out_total']}`, "
            f"FP sq `{traps['total_fp_squiggles']}`)"
        ),
        f"- Messy & structural-friction recall: **{recall['hits']}/{recall['total']}** ({cats})",
        (
            f"- Edit-pair alignment: cleanup "
            f"**{edits['cleanup_passed']}/{edits['cleanup_total']}** · "
            f"anti-hack **{edits['anti_hack_passed']}/{edits['anti_hack_total']}** "
            f"(max cosmetic jump `{edits['max_cosmetic_score_jump']:+.3f}`)"
        ),
    ])
    if calib.get("available_files", 0) > 0:
        fp_rem = calib["confirmed_fp_lines_remaining"]
        fp_tot = calib["confirmed_fp_lines_total"]
        lines.append(
            f"- 196-file clarity correlation (`.codex-notes` present): "
            f"overall `r={calib['all_pearson_score_vs_clarity']:.4f}` "
            f"(held-out `r={calib['held_out_pearson_score_vs_clarity']:.4f}`, "
            f"`rl_reward r={calib['all_pearson_rl_vs_clarity']:.4f}`, "
            f"confirmed FPs `{fp_rem}/{fp_tot}`)"
        )

    if std.get("modules_evaluated", 0) > 0:
        lines.extend([
            "",
            (
                f"## 5. Held-Out CPython {std['python_version']} Stdlib "
                f"(`{std['modules_evaluated']}` modules, `{std['total_lines']:,}` lines)"
            ),
            (
                f"- Overall median score: **{std['median_score']:.3f}** "
                f"(mean `{std['mean_score']:.3f}`)"
            ),
            (
                f"- Compact (`<650` LOC) median: **{std['compact_lt650_median']:.3f}** · "
                f"Large legacy (`>=650` LOC) median: **{std['large_ge650_median']:.3f}**"
            ),
            (
                f"- Held-out squiggle rate: **{std['squiggles_per_100_lines']:.2f} / 100 lines** "
                f"(`{std['total_squiggles']}` total)"
            ),
        ])

    lines.extend(["", "## 6. Held-Out Go Module Cache (report only)"])
    if go_c.get("available"):
        go_kinds = ", ".join(f"{kind} {count}" for kind, count in go_c["by_kind"].items())
        lines.append(
            f"- `{go_c['files']}` files, `{go_c['total_lines']:,}` lines: "
            f"**{go_c['squiggles_per_100_lines']:.2f} / 100 lines** ({go_kinds})"
        )
    else:
        lines.append("- Skipped: no Go module cache found (set `GOMODCACHE`).")

    lat_str = " · ".join(f"`{key}`: **{val} ms**" for key, val in payload["latency_ms"].items())
    lines.extend(["", "## 7. Warm CPU Latency (Pure NumPy)", f"- {lat_str}"])
    return "\n".join(lines)


def run_compare(dataset: dict[str, Any], base_path: str, cand_path: str) -> int:
    hygiene, _ = load_modules()
    b_w = hygiene.load_weights(base_path)
    c_w = hygiene.load_weights(cand_path)
    b_ctrl, _ = run_controlled_suites(dataset["controlled"], b_w)
    c_ctrl, _ = run_controlled_suites(dataset["controlled"], c_w)
    b_poly, _ = run_polyglot_suite(dataset["polyglot"], b_w)
    c_poly, _ = run_polyglot_suite(dataset["polyglot"], c_w)

    rows = [
        ("Controlled Recall", b_ctrl["messy_recall"], c_ctrl["messy_recall"]),
        ("Controlled Clean-Trap FPs", str(b_ctrl["clean_trap_fps"]), str(c_ctrl["clean_trap_fps"])),
        (
            "Polyglot Clean Traps",
            f"{b_poly['clean_traps']['passed']}/{b_poly['clean_traps']['total']}",
            f"{c_poly['clean_traps']['passed']}/{c_poly['clean_traps']['total']}",
        ),
        (
            "Polyglot Messy & Friction",
            f"{b_poly['messy_recall']['hits']}/{b_poly['messy_recall']['total']}",
            f"{c_poly['messy_recall']['hits']}/{c_poly['messy_recall']['total']}",
        ),
        (
            "Edit Anti-Hack Gate",
            f"{b_poly['edit_pairs']['anti_hack_passed']}/{b_poly['edit_pairs']['anti_hack_total']}",
            f"{c_poly['edit_pairs']['anti_hack_passed']}/{c_poly['edit_pairs']['anti_hack_total']}",
        ),
    ]
    b_cc, c_cc = b_poly["corpus_calibration"], c_poly["corpus_calibration"]
    if b_cc.get("available_files", 0) > 0 and c_cc.get("available_files", 0) > 0:
        rows.append((
            "196-File Clarity Pearson r",
            f"{b_cc['all_pearson_score_vs_clarity']:.4f}",
            f"{c_cc['all_pearson_score_vs_clarity']:.4f}",
        ))
    print(f"=== Checkpoint Comparison ({base_path} -> {cand_path}) ===")
    for label, left, right in rows:
        print(f"  {label:28s} | {left:12s} -> {right}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    desc = "Run unified codehygiene v0.6.1 evaluation suites."
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--suite", default=str(EVALS_DIR / "evals.json"), help="Suite JSON path")
    parser.add_argument("--weights", help="Optional custom weights.npz path")
    parser.add_argument(
        "--compare", nargs=2, metavar=("BASE", "CAND"), help="Compare two checkpoints"
    )
    parser.add_argument("--json", action="store_true", help="Output structured JSON")
    parser.add_argument(
        "--check", action="store_true", help="Exit 1 on regression, probe, or version drift"
    )
    parser.add_argument("--skip-js", action="store_true", help="Skip Node.js parity check")
    parser.add_argument("--skip-go", action="store_true", help="Skip Go module-cache corpus")
    parser.add_argument("--skip-stdlib", action="store_true", help="Skip CPython stdlib evaluation")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    hygiene, _ = load_modules()
    dataset = json.loads(Path(args.suite).read_text("utf-8"))
    if args.compare:
        return run_compare(dataset, args.compare[0], args.compare[1])

    weights = hygiene.load_weights(args.weights)
    vsync_summary, vsync_fail = run_version_sync_check(dataset)
    ctrl_summary, ctrl_fail = run_controlled_suites(dataset["controlled"], weights)
    adv_summary, adv_fail = run_adversarial_probes(dataset["probe_seed"])
    js_summary, js_fail = ({"available": False}, []) if args.skip_js else run_js_parity_check()
    reg_summary, reg_fail = run_regressions(dataset["regressions"], weights)
    poly_summary, poly_fail = run_polyglot_suite(dataset["polyglot"], weights)
    std_summary = (
        {"modules_evaluated": 0}
        if args.skip_stdlib
        else run_heldout_stdlib_suite(dataset["stdlib_modules"], weights)
    )
    go_summary = {"available": False} if args.skip_go else run_go_module_corpus(weights)
    latencies = measure_latencies(weights)

    all_failures = [*vsync_fail, *ctrl_fail, *adv_fail, *js_fail, *reg_fail, *poly_fail]
    default_weights = REPO_ROOT / "codehygiene" / "weights.npz"
    weights_file = Path(args.weights) if args.weights else default_weights
    payload = {
        "version": hygiene.__version__,
        "weights_kib": round(weights_file.stat().st_size / 1024.0, 2),
        "version_sync": vsync_summary,
        "controlled": ctrl_summary,
        "adversarial": adv_summary,
        "js_parity": js_summary,
        "issues": reg_summary,
        "polyglot": poly_summary,
        "held_out_stdlib": std_summary,
        "go_corpus": go_summary,
        "latency_ms": latencies,
        "passed": not all_failures,
        "failures": all_failures,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(format_report(payload))
        for failure in all_failures:
            print(f"FAIL: {failure}", file=sys.stderr)
    return 1 if (args.check and all_failures) else 0


if __name__ == "__main__":
    raise SystemExit(main())
