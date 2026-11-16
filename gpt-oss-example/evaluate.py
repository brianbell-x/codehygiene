"""Evaluate before/after codehygiene metrics and test suites across gpt-oss-example."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from types import ModuleType
from typing import Any

sys.dont_write_bytecode = True

EXAMPLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_DIR.parent


@lru_cache(maxsize=1)
def load_hygiene_module() -> ModuleType:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    return importlib.import_module("codehygiene")


@dataclass(frozen=True)
class AppSpec:
    name: str
    language: str
    before_rel: str
    after_rel: str
    test_rel: str
    runner: str
    expected_tests: int = 8


BILLING_SPEC = AppSpec(
    "billing_python", "Python",
    "billing_python/before/billing.py", "billing_python/after/billing.py",
    "billing_python/test_billing.py", "python_unittest",
)

GATEWAY_SPEC = AppSpec(
    "gateway_js", "JavaScript",
    "gateway_js/before/gateway.mjs", "gateway_js/after/gateway.mjs",
    "gateway_js/gateway.test.mjs", "node_gateway",
)

CHECKOUT_SPEC = AppSpec(
    "checkout_ts", "TypeScript",
    "checkout_ts/before/checkout.ts", "checkout_ts/after/checkout.ts",
    "checkout_ts/checkout.test.mts", "node_checkout",
)

MATCHING_SPEC = AppSpec(
    "matching_rust", "Rust",
    "matching_rust/before/matching.rs", "matching_rust/after/matching.rs",
    "matching_rust/tests_harness.rs", "rustc_test",
)

APPS: tuple[AppSpec, ...] = (BILLING_SPEC, GATEWAY_SPEC, CHECKOUT_SPEC, MATCHING_SPEC)


def measure_independent_stats(source: str) -> dict[str, int]:
    """Compute non-neural structural metrics independent of codehygiene weights."""
    max_indent = 0
    over_100 = 0
    comments = 0
    for raw_line in source.expandtabs(2).splitlines():
        stripped = raw_line.lstrip(" ")
        if not stripped:
            continue
        if len(raw_line) > 100:
            over_100 += 1
        if stripped.startswith(("#", "//", "/*", "*")):
            comments += 1
            continue
        indent = len(raw_line) - len(stripped)
        if indent > max_indent:
            max_indent = indent
    return {
        "max_indent_cols": max_indent,
        "lines_over_100_cols": over_100,
        "comment_lines": comments,
    }


def evaluate_file(path: Path) -> dict[str, Any]:
    hygiene = load_hygiene_module()
    source = path.read_text(encoding="utf-8")
    result = hygiene.score(source)
    reward = hygiene.rl_reward(source, tests_passed=True)
    indep = measure_independent_stats(source)

    kind_counts = Counter(sq.kind for sq in result.squiggles)
    style_pct = round(result.archetype_probs[result.archetype] * 100)

    return {
        "path": path.relative_to(REPO_ROOT).as_posix(),
        "loc": len(source.splitlines()),
        **indep,
        "score": round(result.score, 4),
        "rl_hygiene": round(result.rl_hygiene, 4),
        "rl_reward": round(reward, 4),
        "squiggles": len(result.squiggles),
        "squiggle_kinds": dict(sorted(kind_counts.items())),
        "archetype": result.archetype,
        "style_label": f"{result.archetype} ({style_pct}%)",
        "scores": {key: round(val, 4) for key, val in result.scores.items()},
    }


def run_process(cmd: list[str], env: dict[str, str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def run_python_suite(test_path: Path, variant: str) -> tuple[bool, int, str]:
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "BILLING_IMPL": variant}
    proc = run_process([sys.executable, "-B", str(test_path)], env, test_path.parent)
    combined = f"{proc.stdout}\n{proc.stderr}"

    match = re.search(r"Ran\s+(\d+)\s+tests", combined)
    passed = int(match.group(1)) if (proc.returncode == 0 and match) else 0
    return proc.returncode == 0 and passed > 0, passed, combined.strip()


def run_node_suite(test_path: Path, env_key: str, variant: str, expected: int) -> tuple[bool, int, str]:
    env = {**os.environ, env_key: variant}
    proc = run_process(["node", "--test", str(test_path)], env, test_path.parent)
    combined = f"{proc.stdout}\n{proc.stderr}"

    if env_key == "CHECKOUT_IMPL":
        ok = proc.returncode == 0 and "All checkout tests passed." in combined
        return ok, (expected if ok else 0), combined.strip()

    match = re.search(r"# pass\s+(\d+)", combined)
    passed = int(match.group(1)) if (proc.returncode == 0 and match) else 0
    return proc.returncode == 0 and passed == expected, passed, combined.strip()


def run_rust_suite(test_path: Path, variant: str) -> tuple[bool, int, str]:
    env = dict(os.environ)
    suffix = ".exe" if os.name == "nt" else ""

    with tempfile.TemporaryDirectory(prefix="codehygiene_rust_") as tmp_dir:
        bin_path = Path(tmp_dir) / f"tests_{variant}{suffix}"
        compile_cmd = ["rustc", "--test"]
        if variant == "after":
            compile_cmd.extend(["--cfg", "after_impl"])
        compile_cmd.extend([str(test_path), "-o", str(bin_path)])

        comp = run_process(compile_cmd, env, test_path.parent)
        if comp.returncode != 0:
            return False, 0, f"{comp.stdout}\n{comp.stderr}".strip()

        run = run_process([str(bin_path)], env, test_path.parent)
        combined = f"{run.stdout}\n{run.stderr}"
        match = re.search(r"test result: ok\.\s+(\d+)\s+passed;\s+0\s+failed", combined)
        passed = int(match.group(1)) if (run.returncode == 0 and match) else 0
        return run.returncode == 0 and passed > 0, passed, combined.strip()


def execute_variant_tests(spec: AppSpec, variant: str) -> dict[str, Any]:
    test_path = EXAMPLE_DIR / spec.test_rel
    if spec.runner == "python_unittest":
        ok, count, detail = run_python_suite(test_path, variant)
    elif spec.runner == "node_gateway":
        ok, count, detail = run_node_suite(test_path, "GATEWAY_IMPL", variant, spec.expected_tests)
    elif spec.runner == "node_checkout":
        ok, count, detail = run_node_suite(test_path, "CHECKOUT_IMPL", variant, spec.expected_tests)
    elif spec.runner == "rustc_test":
        ok, count, detail = run_rust_suite(test_path, variant)
    else:
        raise ValueError(f"Unsupported test runner: {spec.runner}")

    passed_gate = ok and count == spec.expected_tests
    return {
        "passed": passed_gate,
        "cases_passed": count,
        "cases_expected": spec.expected_tests,
        "error": "" if passed_gate else detail[-400:],
    }


def evaluate_app(spec: AppSpec, include_tests: bool) -> dict[str, Any]:
    before = evaluate_file(EXAMPLE_DIR / spec.before_rel)
    after = evaluate_file(EXAMPLE_DIR / spec.after_rel)

    entry: dict[str, Any] = {
        "app": spec.name,
        "language": spec.language,
        "before": before,
        "after": after,
        "delta": {
            "score": round(after["score"] - before["score"], 4),
            "rl_hygiene": round(after["rl_hygiene"] - before["rl_hygiene"], 4),
            "rl_reward": round(after["rl_reward"] - before["rl_reward"], 4),
            "squiggles": after["squiggles"] - before["squiggles"],
            "loc": after["loc"] - before["loc"],
        },
    }

    if include_tests:
        entry["tests"] = {
            "before": execute_variant_tests(spec, "before"),
            "after": execute_variant_tests(spec, "after"),
        }
    return entry


def summarize_apps(apps: list[dict[str, Any]], include_tests: bool) -> dict[str, Any]:
    count = len(apps)
    before_score = sum(item["before"]["score"] for item in apps) / count
    after_score = sum(item["after"]["score"] for item in apps) / count
    before_hyg = sum(item["before"]["rl_hygiene"] for item in apps) / count
    after_hyg = sum(item["after"]["rl_hygiene"] for item in apps) / count

    before_rew = sum(item["before"]["rl_reward"] for item in apps) / count
    after_rew = sum(item["after"]["rl_reward"] for item in apps) / count
    before_sq = sum(item["before"]["squiggles"] for item in apps)
    after_sq = sum(item["after"]["squiggles"] for item in apps)
    before_loc = sum(item["before"]["loc"] for item in apps)
    after_loc = sum(item["after"]["loc"] for item in apps)

    summary: dict[str, Any] = {
        "app_count": count,
        "mean_before_score": round(before_score, 4),
        "mean_after_score": round(after_score, 4),
        "mean_score_delta": round(after_score - before_score, 4),
        "mean_before_rl_hygiene": round(before_hyg, 4),
        "mean_after_rl_hygiene": round(after_hyg, 4),
        "mean_rl_hygiene_delta": round(after_hyg - before_hyg, 4),
        "mean_before_rl_reward": round(before_rew, 4),
        "mean_after_rl_reward": round(after_rew, 4),
        "mean_rl_reward_delta": round(after_rew - before_rew, 4),
        "total_before_squiggles": before_sq,
        "total_after_squiggles": after_sq,
        "total_squiggles_delta": after_sq - before_sq,
        "total_before_loc": before_loc,
        "total_after_loc": after_loc,
    }

    if include_tests:
        passed_cases = sum(
            item["tests"]["before"]["cases_passed"] + item["tests"]["after"]["cases_passed"]
            for item in apps
        )
        expected_cases = sum(
            item["tests"]["before"]["cases_expected"] + item["tests"]["after"]["cases_expected"]
            for item in apps
        )
        summary["total_tests_passed"] = passed_cases
        summary["total_tests_expected"] = expected_cases
    return summary


def check_gate(apps: list[dict[str, Any]], include_tests: bool) -> list[str]:
    failures: list[str] = []
    for item in apps:
        name = item["app"]
        after = item["after"]
        before = item["before"]

        if after["squiggles"] != 0:
            failures.append(f"{name}: after squiggles={after['squiggles']} (expected 0)")
        if after["score"] < 0.80:
            failures.append(f"{name}: after score={after['score']:.4f} (< 0.80)")
        if after["score"] <= before["score"]:
            failures.append(f"{name}: after score did not improve over before")
        if include_tests and not item["tests"]["before"]["passed"]:
            failures.append(f"{name}: before test suite failed")
        if include_tests and not item["tests"]["after"]["passed"]:
            failures.append(f"{name}: after test suite failed")
    return failures


def build_app_row(item: dict[str, Any], include_tests: bool) -> str:
    bef = item["before"]
    aft = item["after"]
    dlt = item["delta"]

    cells = [
        f"`{item['app']}`",
        item["language"],
        f"{bef['score']:.3f}",
        f"{aft['score']:.3f}",
        f"{dlt['score']:+.3f}",
        f"{bef['rl_hygiene']:.3f}",
        f"{aft['rl_hygiene']:.3f}",
        f"{bef['rl_reward']:.4f}",
        f"{aft['rl_reward']:.4f}",
        str(bef["squiggles"]),
        str(aft["squiggles"]),
        bef["style_label"],
        aft["style_label"],
        f"{bef['loc']}->{aft['loc']}",
    ]

    if include_tests:
        tb = item["tests"]["before"]
        ta = item["tests"]["after"]
        cells.append(
            f"{tb['cases_passed']}/{tb['cases_expected']} · "
            f"{ta['cases_passed']}/{ta['cases_expected']}"
        )
    return "| " + " | ".join(cells) + " |"


def format_markdown_report(
    apps: list[dict[str, Any]],
    summary: dict[str, Any],
    include_tests: bool,
) -> str:
    headers = [
        "App", "Lang", "Before Score", "After Score", "Delta",
        "Before RL Hygiene", "After RL Hygiene", "Before RL Reward", "After RL Reward",
        "Before Squiggles", "After Squiggles", "Before Style", "After Style",
        "LOC Before->After",
    ]
    if include_tests:
        headers.append("Tests (B/A)")

    align = [":---", ":---"] + [":---:"] * (len(headers) - 2)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(align) + " |",
        *(build_app_row(item, include_tests) for item in apps),
    ]

    mean_cells = [
        "**Mean / Total**",
        "4 langs",
        f"**{summary['mean_before_score']:.3f}**",
        f"**{summary['mean_after_score']:.3f}**",
        f"**{summary['mean_score_delta']:+.3f}**",
        f"**{summary['mean_before_rl_hygiene']:.3f}**",
        f"**{summary['mean_after_rl_hygiene']:.3f}**",
        f"**{summary['mean_before_rl_reward']:.4f}**",
        f"**{summary['mean_after_rl_reward']:.4f}**",
        f"**{summary['total_before_squiggles']}**",
        f"**{summary['total_after_squiggles']}**",
        "—",
        "—",
        f"**{summary['total_before_loc']}->{summary['total_after_loc']}**",
    ]

    if include_tests:
        mean_cells.append(
            f"**{summary['total_tests_passed']}/{summary['total_tests_expected']}**"
        )
    lines.append("| " + " | ".join(mean_cells) + " |")

    lines.extend([
        "",
        "| App | Max Indent (`B->A`) | Lines >100 Cols (`B->A`) | Comment Lines (`B->A`) | LOC (`B->A`) |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ])
    for item in apps:
        bef = item["before"]
        aft = item["after"]
        lines.append(
            f"| `{item['app']}` | {bef['max_indent_cols']}->{aft['max_indent_cols']} "
            f"| {bef['lines_over_100_cols']}->{aft['lines_over_100_cols']} "
            f"| {bef['comment_lines']}->{aft['comment_lines']} "
            f"| {bef['loc']}->{aft['loc']} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate gpt-oss-example before/after scores, RL rewards, and tests."
    )
    parser.add_argument("--json", action="store_true", help="Emit structured JSON output")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 unless all after files pass 0 squiggles, score >= 0.80, and improve",
    )
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="Execute before and after test suites across Python, JS, TS, and Rust",
    )
    args = parser.parse_args(argv)

    apps = [evaluate_app(spec, args.run_tests) for spec in APPS]
    summary = summarize_apps(apps, args.run_tests)
    failures = check_gate(apps, args.run_tests)

    if args.json:
        payload = {
            "apps": apps,
            "summary": summary,
            "passed_check": len(failures) == 0,
            "failures": failures,
        }
        print(json.dumps(payload, indent=2))
    else:
        print(format_markdown_report(apps, summary, args.run_tests))
        for msg in failures:
            print(f"FAIL: {msg}", file=sys.stderr)

    return 1 if (args.check and failures) else 0


if __name__ == "__main__":
    raise SystemExit(main())
