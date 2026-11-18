"""Reproducible in-sample, adversarial, parity, and held-out stdlib benchmark runner."""

from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from types import ModuleType
from typing import Any
import numpy as np

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent
GO_CORPUS_LINES = 60_000
GO_CORPUS_MAX_FILE_LINES = 1_500

_GO_TEST_FUNC = re.compile(r"^func (\([^)]*\) )?(Test|Benchmark|Example|Fuzz)\w*\(")
_GO_DIRECTIVE = re.compile(r"^//(go:|\s*\+)")
_NARRATION = re.compile(r"(?i)todo|fixme|step \d")
_STRONG_CEREMONY = re.compile(r"Manager|Factory|Impl\b|Wrapper|Helper|Abstract")
_FALLBACK_LITERAL = re.compile(r"\|\||\?\?| or |catch|except")
_ERR_CHECK = re.compile(r"\berr != nil\b")


def _is_doc_tail(text: str, prev: str) -> bool:
    return text.startswith("//") and prev.startswith("//") and not _NARRATION.search(text)


def _is_service_noun(text: str, _prev: str) -> bool:
    return bool(re.search(r"Service|Builder", text)) and not _STRONG_CEREMONY.search(text)


def _is_empty_literal(text: str, prev: str) -> bool:
    if "{}" not in text or _FALLBACK_LITERAL.search(text):
        return False
    swallows_err = text.startswith("return ") and text.endswith(", nil") and _ERR_CHECK.search(prev)
    return not swallows_err


# Issue #3 false-positive shapes, checked in order: (label, squiggle kind, line predicate)
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


def compute_auroc(clean_scores: list[float], messy_scores: list[float]) -> float:
    wins = 0.0
    total = len(clean_scores) * len(messy_scores)
    for clean_val in clean_scores:
        for messy_val in messy_scores:
            if clean_val > messy_val:
                wins += 1.0
            elif clean_val == messy_val:
                wins += 0.5
    return wins / max(total, 1)


def run_controlled_suites(suites: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    hygiene, _ = load_modules()
    failures: list[str] = []
    messy_scores: list[float] = []
    clean_scores: list[float] = []
    recalled = 0
    false_positives = 0

    for item in suites["messy_cases"]:
        res = hygiene.score(item["code"])
        messy_scores.append(res.score)
        kinds = {sq.kind for sq in res.squiggles}
        if item["expected_kind"] in kinds:
            recalled += 1
        else:
            failures.append(
                f"messy/{item['id']}: expected {item['expected_kind']}, got {sorted(kinds)}"
            )

    for item in suites["clean_traps"]:
        res = hygiene.score(item["code"])
        clean_scores.append(res.score)
        if res.squiggles:
            false_positives += 1
            kinds = [(sq.line, sq.kind, round(sq.probability, 2)) for sq in res.squiggles]
            failures.append(f"clean_trap/{item['id']}: unexpected squiggles {kinds}")
        elif res.score < 0.72:
            failures.append(f"clean_trap/{item['id']}: score {res.score:.3f} < 0.72")

    auroc = compute_auroc(clean_scores, messy_scores)
    summary = {
        "messy_recall": f"{recalled}/{len(suites['messy_cases'])}",
        "recalled_count": recalled,
        "messy_total": len(suites["messy_cases"]),
        "clean_trap_fps": false_positives,
        "clean_trap_total": len(suites["clean_traps"]),
        "auroc": round(auroc, 4),
        "mean_clean_score": round(float(np.mean(clean_scores)), 4),
        "mean_messy_score": round(float(np.mean(messy_scores)), 4),
    }
    return summary, failures


def run_adversarial_probes() -> tuple[dict[str, Any], list[str]]:
    hygiene, weight_compiler = load_modules()
    failures: list[str] = []
    presets = json.loads((REPO_ROOT / "docs" / "presets.json").read_text("utf-8"))
    slop_sample = presets[0]["before"]

    # 1. pass -> ... invariance
    pass_code = "def run():\n    try:\n        work()\n    except Exception:\n        pass\n"
    ell_code = "def run():\n    try:\n        work()\n    except Exception:\n        ...\n"
    rew_pass = hygiene.rl_reward(pass_code, tests_passed=True)
    rew_ell = hygiene.rl_reward(ell_code, tests_passed=True)
    sq_ell = hygiene.lint(ell_code)
    if not sq_ell or abs(rew_pass - rew_ell) > 1e-4:
        failures.append(f"probe/ellipsis_bypass: pass={rew_pass:.4f} vs ellipsis={rew_ell:.4f}")

    # 2. Helper-padding resistance (40 appended trivial helpers)
    helpers = "\n\n".join(f"def helper_{idx}(val):\n    return val + {idx}" for idx in range(40))
    base_res = hygiene.score(slop_sample)
    pad_code = f"{slop_sample}\n\n{helpers}\n"
    pad_res = hygiene.score(pad_code)
    base_rew = hygiene.rl_reward(slop_sample, tests_passed=True)
    pad_rew = hygiene.rl_reward(pad_code, tests_passed=True)
    kept_squiggles = len(pad_res.squiggles) >= len(base_res.squiggles)
    if pad_res.score >= 0.45 or (pad_rew - base_rew) > 0.015 or not kept_squiggles:
        failures.append(
            f"probe/helper_padding: score {base_res.score:.3f}->{pad_res.score:.3f}, "
            f"reward {base_rew:.4f}->{pad_rew:.4f}"
        )

    # 3. Docstring-padding resistance (200-line prepended docstring)
    doc_body = [f"Documentation line {idx}." for idx in range(200)]
    doc_lines = "\n".join(['"""', *doc_body, '"""'])
    doc_pad_res = hygiene.score(f"{doc_lines}\n\n{slop_sample}\n")
    if abs(doc_pad_res.score - base_res.score) > 0.02:
        failures.append(
            f"probe/docstring_padding: score {base_res.score:.3f}->{doc_pad_res.score:.3f}"
        )

    # 4. >2048-line full-file coverage
    long_lines = [f"ITEM_{idx} = {idx}" if idx % 4 == 0 else "" for idx in range(2060)]
    long_prefix = "\n".join(long_lines)
    tail_res = hygiene.score(f"{long_prefix}\n\n{slop_sample}\n")
    tail_lines = [sq.line for sq in tail_res.squiggles if sq.line > 2048]
    if len(tail_lines) < 3 or tail_res.score >= 0.20:
        failures.append(
            f"probe/after_line_2048: score={tail_res.score:.3f}, tail_squiggles={tail_lines}"
        )

    # 5. Rewriting `except Exception: pass` as a broad contextlib.suppress must not gain reward
    send_call = "        mailer.send(order.email, order.id)\n"
    swallow_code = f"def send_receipt(order):\n    try:\n{send_call}    except Exception:\n        pass\n"
    suppress_code = f"def send_receipt(order):\n    with contextlib.suppress(Exception):\n{send_call}"
    narrow_code = "with contextlib.suppress(FileNotFoundError):\n    os.remove(path)\n"
    rew_swallow = hygiene.rl_reward(swallow_code, tests_passed=True)
    rew_suppress = hygiene.rl_reward(suppress_code, tests_passed=True)
    suppress_kinds = {sq.kind for sq in hygiene.lint(suppress_code)}
    narrow_hits = hygiene.lint(narrow_code)
    if "swallowed_error" not in suppress_kinds or rew_suppress > rew_swallow + 0.005:
        failures.append(
            f"probe/broad_suppress_bypass: except-pass={rew_swallow:.4f} vs "
            f"suppress(Exception)={rew_suppress:.4f}, kinds={sorted(suppress_kinds)}"
        )
    if narrow_hits:
        failures.append(f"probe/narrow_suppress_clean: {[sq.kind for sq in narrow_hits]}")

    # 6. Deterministic byte-identical weight compilation
    with tempfile.TemporaryDirectory() as tmp_dir:
        rebuilt = Path(tmp_dir) / "weights.npz"
        weight_compiler.compile_weights(rebuilt)
        live_bytes = (REPO_ROOT / "codehygiene" / "weights.npz").read_bytes()
        docs_bytes = (REPO_ROOT / "docs" / "weights.npz").read_bytes()
        deterministic = (live_bytes == rebuilt.read_bytes()) and (live_bytes == docs_bytes)
    if not deterministic:
        failures.append("probe/weights_determinism: weights.npz drift detected")

    summary = {
        "ellipsis_rl_delta": round(abs(rew_pass - rew_ell), 6),
        "base_slop_score": round(base_res.score, 4),
        "padded_40_helpers_score": round(pad_res.score, 4),
        "padded_40_helpers_rl_delta": round(pad_rew - base_rew, 4),
        "padded_200L_docstring_score": round(doc_pad_res.score, 4),
        "after_line_2048_score": round(tail_res.score, 4),
        "after_line_2048_squiggles": len(tail_lines),
        "broad_suppress_rl_delta": round(rew_suppress - rew_swallow, 4),
        "narrow_suppress_squiggles": len(narrow_hits),
        "byte_identical_weights": deterministic,
    }
    return summary, failures


def run_js_parity_check() -> tuple[dict[str, Any], list[str]]:
    if shutil.which("node") is None:
        return {"available": False, "max_score_diff": None}, []

    hygiene, _ = load_modules()
    script_parts = [
        "import fs from 'node:fs';",
        "import { unpackNpzBuffer, scoreCode, computeRlReward } from './docs/engine.js';",
        "const buf = fs.readFileSync('./docs/weights.npz');",
        "const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);",
        "const w = await unpackNpzBuffer(ab);",
        "const paths = [",
        "  'compile.py',",
        "  'docs/app.js',",
        "  'codehygiene/__init__.py'",
        "];",
        "const out = {};",
        "for (const p of paths) {",
        "  const src = fs.readFileSync(p, 'utf8');",
        "  const r = scoreCode(src, w);",
        "  const rew = computeRlReward(r, true);",
        "  out[p] = { score: r.score, rl: r.rlHygiene, rew, sq: r.squiggles.length };",
        "}",
        "console.log(JSON.stringify(out));",
    ]
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", "\n".join(script_parts)],
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
            msg = f"js_parity/{rel_path}: diff={diff:.6f}, sq={len(py_res.squiggles)}"
            failures.append(msg)

    summary = {"available": True, "max_abs_diff": round(max_diff, 6), "files_checked": len(js_map)}
    return summary, failures


def run_heldout_stdlib_suite(module_names: list[str]) -> dict[str, Any]:
    hygiene, _ = load_modules()
    stdlib_dir = Path(os.__file__).resolve().parent
    all_scores: list[float] = []
    compact_scores: list[float] = []
    large_scores: list[float] = []
    kind_counts: Counter[str] = Counter()
    total_lines = 0
    found_modules = 0

    for rel_name in module_names:
        mod_path = stdlib_dir / rel_name
        if not mod_path.is_file():
            continue
        found_modules += 1
        src = mod_path.read_text("utf-8", "replace")
        loc = len(src.splitlines())
        res = hygiene.score(src)
        all_scores.append(res.score)
        if loc < 650:
            compact_scores.append(res.score)
        else:
            large_scores.append(res.score)
        total_lines += loc
        for sq in res.squiggles:
            kind_counts[sq.kind] += 1

    total_sq = sum(kind_counts.values())
    return {
        "python_version": sys.version.split()[0],
        "modules_evaluated": found_modules,
        "total_lines": total_lines,
        "median_score": round(float(np.median(all_scores)), 4) if all_scores else 0.0,
        "mean_score": round(float(np.mean(all_scores)), 4) if all_scores else 0.0,
        "compact_lt650_median": round(float(np.median(compact_scores)), 4) if compact_scores else 0.0,
        "large_ge650_median": round(float(np.median(large_scores)), 4) if large_scores else 0.0,
        "total_squiggles": total_sq,
        "squiggles_per_100_lines": round(total_sq / max(total_lines, 1) * 100.0, 2),
        "by_kind": dict(sorted(kind_counts.items())),
    }


def run_issue_suites(issues: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Gate reported false positives, held-out clean variants, and recall guards per issue."""
    hygiene, _ = load_modules()
    failures: list[str] = []
    summary: dict[str, Any] = {}

    for issue_id, spec in issues.items():
        tag = f"issue{issue_id}"
        reported_total = 0
        still_flagged = 0
        for case in spec["repro"]:
            hits = [(sq.line, sq.kind) for sq in hygiene.score(case["code"]).squiggles]
            reported = set(case["reported_lines"])
            reported_total += len(reported)
            still_flagged += len(reported & {line for line, _ in hits})
            if hits:
                failures.append(f"{tag}/repro/{case['id']}: squiggles {hits}")

        clean_fps = 0
        for case in spec["clean_variants"]:
            hits = [(sq.line, sq.kind) for sq in hygiene.score(case["code"]).squiggles]
            if hits:
                clean_fps += 1
                failures.append(f"{tag}/clean/{case['id']}: squiggles {hits}")

        recalled = 0
        for case in spec["recall_guards"]:
            hits = [(sq.line, sq.kind) for sq in hygiene.score(case["code"]).squiggles]
            if (case["expected_line"], case["expected_kind"]) in hits:
                recalled += 1
            else:
                want = f"{case['expected_kind']}@{case['expected_line']}"
                failures.append(f"{tag}/guard/{case['id']}: expected {want}, got {hits}")

        summary[issue_id] = {
            "reported_still_flagged": f"{still_flagged}/{reported_total}",
            "clean_variant_fps": f"{clean_fps}/{len(spec['clean_variants'])}",
            "recall_guards": f"{recalled}/{len(spec['recall_guards'])}",
        }
    return summary, failures


def classify_go_issue_signature(lines: list[str], line_no: int, kind: str) -> str | None:
    """Map a squiggle on real Go code to one of the issue #3 false-positive shapes."""
    text = lines[line_no - 1].strip()
    prev = lines[line_no - 2].strip() if line_no > 1 else ""
    for label, shape_kind, matches in _GO_FP_SHAPES:
        if kind == shape_kind and matches(text, prev):
            return label
    return None


def sample_go_corpus(root: Path) -> list[tuple[str, str]]:
    """Pick non-generated module-cache Go files in stable path-hash order up to a line budget."""
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
        generated = "Code generated" in head and "DO NOT EDIT" in head
        if generated or loc > GO_CORPUS_MAX_FILE_LINES:
            continue
        picked.append((rel, src))
        total += loc
        if total >= GO_CORPUS_LINES:
            break
    return picked


def run_go_module_corpus() -> dict[str, Any]:
    """Score a held-out sample of real third-party Go from the local module cache."""
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
        for sq in hygiene.score(src).squiggles:
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


def measure_latencies() -> dict[str, float]:
    hygiene, _ = load_modules()
    init_path = REPO_ROOT / "codehygiene" / "__init__.py"
    lines = init_path.read_text("utf-8").splitlines()
    samples = {
        "warm_12_lines_ms": "\n".join(lines[:12]),
        "warm_65_lines_ms": "\n".join(lines[:65]),
        "warm_333_lines_ms": "\n".join(lines[:333]),
    }
    argparse_path = Path(os.__file__).resolve().parent / "argparse.py"
    if argparse_path.is_file():
        samples["warm_argparse_full_ms"] = argparse_path.read_text("utf-8", "replace")

    hygiene.load_weights()
    timings: dict[str, float] = {}
    for key, code in samples.items():
        hygiene.score(code)
        runs = 3 if len(code.splitlines()) < 1000 else 1
        start = time.perf_counter()
        for _ in range(runs):
            hygiene.score(code)
        timings[key] = round((time.perf_counter() - start) * 1000.0 / runs, 2)
    return timings


def format_report(payload: dict[str, Any]) -> str:
    ctrl = payload["controlled"]
    adv = payload["adversarial"]
    js_p = payload["js_parity"]
    std = payload["held_out_stdlib"]
    lat = payload["latency_ms"]

    fps_str = f"{ctrl['clean_trap_fps']} / {ctrl['clean_trap_total']}"
    pad_str = f"{adv['base_slop_score']:.3f} -> {adv['padded_40_helpers_score']:.3f}"
    doc_str = f"{adv['base_slop_score']:.3f} -> {adv['padded_200L_docstring_score']:.3f}"
    std_hdr = (
        f"## 3. Held-Out CPython {std['python_version']} Stdlib "
        f"(`{std['modules_evaluated']}` modules, `{std['total_lines']:,}` lines)"
    )
    split_med = (
        f"- Compact (`<650` LOC) median: **{std['compact_lt650_median']:.3f}** · "
        f"Large legacy (`>=650` LOC) median: **{std['large_ge650_median']:.3f}**"
    )
    rate_str = (
        f"- Held-out squiggle rate: **{std['squiggles_per_100_lines']:.2f} / 100 lines** "
        f"(`{std['total_squiggles']}` total)"
    )

    recall_str = (
        f"- Messy & static-slop recall: **{ctrl['messy_recall']}** "
        f"(mean `{ctrl['mean_messy_score']:.3f}`)"
    )
    trap_str = (
        f"- Clean traps & linter-divergent clean idioms: **{fps_str} FPs** "
        f"(mean `{ctrl['mean_clean_score']:.3f}`)"
    )
    pad_line = (
        f"- Slop + 40 trivial helpers: `{pad_str}`, "
        f"`rl_reward` delta `{adv['padded_40_helpers_rl_delta']:+.4f}`"
    )
    tail_line = (
        f"- Slop past line 2,048: `{adv['after_line_2048_score']:.3f}` "
        f"(`{adv['after_line_2048_squiggles']}` sq)"
    )
    suppress_line = (
        f"- `except Exception: pass` -> `contextlib.suppress(Exception)` `rl_reward` delta: "
        f"`{adv['broad_suppress_rl_delta']:+.4f}` · narrow `suppress(FileNotFoundError)` "
        f"squiggles: `{adv['narrow_suppress_squiggles']}`"
    )

    lines = [
        "## 1. Controlled Diagnostic & Clean-Trap Suite (`benchmarks/suites.json`, in-sample)",
        recall_str,
        trap_str,
        f"- Clean vs. messy AUROC: **{ctrl['auroc']:.4f}**",
        "",
        "## 2. Adversarial & RL Anti-Gaming Probes",
        f"- `except: pass` vs `except: ...` `rl_reward` delta: `{adv['ellipsis_rl_delta']:.6f}`",
        pad_line,
        f"- Slop + 200-line docstring: `{doc_str}`",
        tail_line,
        suppress_line,
        f"- Byte-identical `weights.npz`: `{adv['byte_identical_weights']}` · JS parity: `{js_p}`",
        "",
        std_hdr,
        f"- Overall median score: **{std['median_score']:.3f}** (mean `{std['mean_score']:.3f}`)",
        split_med,
        rate_str,
        "",
        "## 4. Issue Regression Suites (`benchmarks/issues.json`)",
    ]
    for issue_id, res in payload["issues"].items():
        label = f"Issue #{issue_id}" if issue_id.isdigit() else f"`{issue_id}`"
        lines.append(
            f"- {label}: reported lines still flagged **{res['reported_still_flagged']}** · "
            f"clean-variant FPs **{res['clean_variant_fps']}** · "
            f"recall guards **{res['recall_guards']}**"
        )

    go_c = payload["go_corpus"]
    lines += ["", "## 5. Held-Out Go Module Cache (report only)"]
    if go_c.get("available"):
        go_kinds = ", ".join(f"{kind} {count}" for kind, count in go_c["by_kind"].items())
        go_sigs = ", ".join(f"{sig} {count}" for sig, count in go_c["issue3_signatures"].items())
        lines += [
            f"- `{go_c['files']}` files, `{go_c['total_lines']:,}` lines: "
            f"**{go_c['squiggles_per_100_lines']:.2f} / 100 lines** ({go_kinds})",
            f"- Issue #3 false-positive shapes: **{go_c['issue3_signature_total']}** ({go_sigs or 'none'})",
        ]
    else:
        lines.append("- Skipped: no Go module cache found (set `GOMODCACHE`).")

    lines += [
        "",
        "## 6. Warm CPU Latency (Pure NumPy)",
        "- " + " · ".join(f"`{key}`: **{val} ms**" for key, val in lat.items()),
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run codehygiene benchmark suites.")
    parser.add_argument("--json", action="store_true", help="Output structured JSON")
    parser.add_argument("--check", action="store_true", help="Exit 1 on regression or probe failure")
    parser.add_argument("--skip-js", action="store_true", help="Skip Node.js browser parity check")
    parser.add_argument("--skip-go", action="store_true", help="Skip the Go module-cache corpus")
    args = parser.parse_args(argv)

    hygiene, _ = load_modules()
    suites = json.loads((BENCH_DIR / "suites.json").read_text("utf-8"))
    issues = json.loads((BENCH_DIR / "issues.json").read_text("utf-8"))
    ctrl_summary, ctrl_fail = run_controlled_suites(suites)
    adv_summary, adv_fail = run_adversarial_probes()
    js_summary, js_fail = ({"available": False}, []) if args.skip_js else run_js_parity_check()
    std_summary = run_heldout_stdlib_suite(suites["stdlib_modules"])
    issue_summary, issue_fail = run_issue_suites(issues)
    go_summary = {"available": False} if args.skip_go else run_go_module_corpus()
    latencies = measure_latencies()

    all_failures = [*ctrl_fail, *adv_fail, *js_fail, *issue_fail]
    weights_kib = round((REPO_ROOT / "codehygiene" / "weights.npz").stat().st_size / 1024.0, 2)
    payload = {
        "version": hygiene.__version__,
        "weights_kib": weights_kib,
        "controlled": ctrl_summary,
        "adversarial": adv_summary,
        "js_parity": js_summary,
        "held_out_stdlib": std_summary,
        "issues": issue_summary,
        "go_corpus": go_summary,
        "latency_ms": latencies,
        "passed": len(all_failures) == 0,
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
