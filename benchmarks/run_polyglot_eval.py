"""Evaluate codehygiene on the 20-language polyglot & cognitive-clarity suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import codehygiene as ch


def _pearson(x_vals: list[float], y_vals: list[float]) -> float:
    if len(x_vals) < 3:
        return 0.0
    x_arr = np.asarray(x_vals, dtype=np.float64)
    y_arr = np.asarray(y_vals, dtype=np.float64)
    x_std = float(np.std(x_arr))
    y_std = float(np.std(y_arr))
    if x_std < 1e-8 or y_std < 1e-8:
        return 0.0
    cov = float(np.mean((x_arr - np.mean(x_arr)) * (y_arr - np.mean(y_arr))))
    return round(cov / (x_std * y_std), 4)


def _spearman(x_vals: list[float], y_vals: list[float]) -> float:
    if len(x_vals) < 3:
        return 0.0
    x_ranks = np.argsort(np.argsort(np.asarray(x_vals, dtype=np.float64))).astype(
        np.float64
    )
    y_ranks = np.argsort(np.argsort(np.asarray(y_vals, dtype=np.float64))).astype(
        np.float64
    )
    return _pearson(x_ranks.tolist(), y_ranks.tolist())


def _rl_for_weights(
    code: str,
    weights: dict[str, np.ndarray],
    threshold: float = 0.55,
) -> tuple[ch.Result, float]:
    res = ch.score(code, threshold=threshold, weights=weights)
    if hasattr(ch, "_rl_from_result"):
        rl_val = float(ch._rl_from_result(res))
    elif weights is ch.load_weights():
        rl_val = float(ch.rl_reward(code, tests_passed=True))
    else:
        blended = 0.5 * res.score + 0.5 * res.rl_hygiene
        damp = 1.0 / (1.0 + 1.5 * len(res.squiggles))
        rl_val = 1.0 + 0.25 * blended * damp
    return res, rl_val


def _eval_clean_traps(
    cases: list[dict[str, Any]],
    weights: dict[str, np.ndarray],
) -> dict[str, Any]:
    passed_all = 0
    passed_dev = 0
    total_dev = 0
    passed_ho = 0
    total_ho = 0
    total_fp_sq = 0
    failures: list[dict[str, Any]] = []

    for case in cases:
        res = ch.score(case["code"], weights=weights)
        n_sq = len(res.squiggles)
        total_fp_sq += n_sq
        is_clean = n_sq == 0
        if is_clean:
            passed_all += 1
        else:
            failures.append(
                {
                    "id": case["id"],
                    "language": case["language"],
                    "split": case["split"],
                    "squiggles": n_sq,
                    "kinds": sorted({sq.kind for sq in res.squiggles}),
                }
            )
        if case["split"] == "dev":
            total_dev += 1
            passed_dev += int(is_clean)
        else:
            total_ho += 1
            passed_ho += int(is_clean)

    return {
        "passed": passed_all,
        "total": len(cases),
        "dev_passed": passed_dev,
        "dev_total": total_dev,
        "held_out_passed": passed_ho,
        "held_out_total": total_ho,
        "total_fp_squiggles": total_fp_sq,
        "failures": failures,
    }


def _eval_messy_recall(
    cases: list[dict[str, Any]],
    weights: dict[str, np.ndarray],
) -> dict[str, Any]:
    by_cat: dict[str, dict[str, int]] = {}
    misses: list[dict[str, Any]] = []

    for case in cases:
        cat = str(case.get("category", "other"))
        bucket = by_cat.setdefault(cat, {"hits": 0, "total": 0})
        bucket["total"] += 1
        res = ch.score(case["code"], weights=weights)
        if res.squiggles or (cat == "structural_friction" and res.score < 0.72):
            bucket["hits"] += 1
        else:
            misses.append(
                {
                    "id": case["id"],
                    "language": case["language"],
                    "category": cat,
                    "score": round(res.score, 4),
                }
            )

    total_hits = sum(b["hits"] for b in by_cat.values())
    total_cases = sum(b["total"] for b in by_cat.values())
    return {
        "hits": total_hits,
        "total": total_cases,
        "by_category": by_cat,
        "misses": misses,
    }


def _eval_edit_pairs(
    pairs: list[dict[str, Any]],
    weights: dict[str, np.ndarray],
) -> dict[str, Any]:
    cleanup_ok = 0
    cleanup_total = 0
    anti_hack_ok = 0
    anti_hack_total = 0
    max_cosmetic_jump = 0.0
    details: list[dict[str, Any]] = []

    for pair in pairs:
        b_res, b_rl = _rl_for_weights(pair["before_code"], weights)
        a_res, a_rl = _rl_for_weights(pair["after_code"], weights)
        sc_delta = a_res.score - b_res.score
        rl_delta = a_rl - b_rl
        expect = pair["expect"]

        if expect == "after_better":
            cleanup_total += 1
            passed = sc_delta > 0.0 and rl_delta > 0.0
            cleanup_ok += int(passed)
        elif expect == "no_reward_gain":
            anti_hack_total += 1
            passed = rl_delta <= 0.005 and len(a_res.squiggles) >= 1
            anti_hack_ok += int(passed)
        else:
            anti_hack_total += 1
            max_cosmetic_jump = max(max_cosmetic_jump, sc_delta)
            passed = sc_delta <= 0.45
            anti_hack_ok += int(passed)

        details.append(
            {
                "id": pair["id"],
                "language": pair["language"],
                "expect": expect,
                "passed": passed,
                "score_delta": round(sc_delta, 4),
                "rl_delta": round(rl_delta, 4),
            }
        )

    return {
        "cleanup_passed": cleanup_ok,
        "cleanup_total": cleanup_total,
        "anti_hack_passed": anti_hack_ok,
        "anti_hack_total": anti_hack_total,
        "max_cosmetic_score_jump": round(max_cosmetic_jump, 4),
        "details": details,
    }


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


def _eval_full_corpus(
    index_rows: list[dict[str, Any]],
    weights: dict[str, np.ndarray],
    corpus_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    avail = 0
    splits: dict[str, dict[str, list[float]]] = {
        "all": {"ch": [], "rl": [], "clar": []},
        "dev": {"ch": [], "rl": [], "clar": []},
        "held_out": {"ch": [], "rl": [], "clar": []},
    }
    cliff_collapses = 0
    blind_spot_escapes = 0
    confirmed_fps_remaining = 0
    confirmed_fps_total = 0

    for row in index_rows:
        fpath = _resolve_corpus_file(corpus_root, str(row["rel_path"]))
        if fpath is None:
            continue
        avail += 1
        code = fpath.read_text(encoding="utf-8", errors="replace")
        res, rl_val = _rl_for_weights(code, weights)
        clar = float(row["cognitive_clarity"])
        fric = float(row.get("structural_friction", 0.0))
        sp = str(row["split"])

        for key in ("all", sp):
            splits[key]["ch"].append(res.score)
            splits[key]["rl"].append(rl_val)
            splits[key]["clar"].append(clar)

        if clar >= 0.65 and res.score < 0.25:
            cliff_collapses += 1
        if (fric >= 0.60 or clar <= 0.42) and res.score >= 0.80:
            blind_spot_escapes += 1

        flagged_lines = {sq.line for sq in res.squiggles}
        fp_lines = [int(ln) for ln in row.get("confirmed_fp_lines", [])]
        confirmed_fps_total += len(fp_lines)
        confirmed_fps_remaining += sum(1 for ln in fp_lines if ln in flagged_lines)

    if avail == 0:
        return {"available_files": 0}

    metrics: dict[str, Any] = {
        "available_files": avail,
        "cliff_collapses": cliff_collapses,
        "blind_spot_escapes": blind_spot_escapes,
        "confirmed_fp_lines_remaining": confirmed_fps_remaining,
        "confirmed_fp_lines_total": confirmed_fps_total,
    }
    for sp_name, vecs in splits.items():
        metrics[f"{sp_name}_pearson_score_vs_clarity"] = _pearson(
            vecs["ch"], vecs["clar"]
        )
        metrics[f"{sp_name}_spearman_score_vs_clarity"] = _spearman(
            vecs["ch"], vecs["clar"]
        )
        metrics[f"{sp_name}_pearson_rl_vs_clarity"] = _pearson(
            vecs["rl"], vecs["clar"]
        )
    return metrics


def evaluate_suite(
    suite_path: Path,
    weights_path: str | None = None,
    corpus_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    weights = ch.load_weights(weights_path)
    return {
        "weights": weights_path or "active_default",
        "clean_traps": _eval_clean_traps(suite["clean_traps"], weights),
        "messy_recall": _eval_messy_recall(suite["messy_recall"], weights),
        "edit_pairs": _eval_edit_pairs(suite["edit_pairs"], weights),
        "corpus_calibration": _eval_full_corpus(
            suite["corpus_index"], weights, corpus_root=corpus_root
        ),
    }


def _print_single_report(rep: dict[str, Any]) -> None:
    traps = rep["clean_traps"]
    recall = rep["messy_recall"]
    edits = rep["edit_pairs"]
    calib = rep["corpus_calibration"]

    print(f"=== Polyglot & Clarity Eval ({rep['weights']}) ===")
    print(
        f"  Clean Traps (0 sq):     {traps['passed']}/{traps['total']} "
        f"(dev {traps['dev_passed']}/{traps['dev_total']}, "
        f"held_out {traps['held_out_passed']}/{traps['held_out_total']}, "
        f"total FP sq={traps['total_fp_squiggles']})"
    )
    cats = ", ".join(
        f"{k}={v['hits']}/{v['total']}" for k, v in recall["by_category"].items()
    )
    print(f"  Messy & Friction Recall:{recall['hits']:3d}/{recall['total']} ({cats})")
    print(
        f"  Edit Pair Alignment:    cleanup {edits['cleanup_passed']}/{edits['cleanup_total']} | "
        f"anti-hack {edits['anti_hack_passed']}/{edits['anti_hack_total']} "
        f"(max cosmetic jump={edits['max_cosmetic_score_jump']:+.3f})"
    )
    if calib.get("available_files", 0) > 0:
        print(
            f"  196-File Clarity Corr:  overall r={calib['all_pearson_score_vs_clarity']:.4f} "
            f"(dev r={calib['dev_pearson_score_vs_clarity']:.4f}, "
            f"held_out r={calib['held_out_pearson_score_vs_clarity']:.4f}) | "
            f"rl_reward r={calib['all_pearson_rl_vs_clarity']:.4f}"
        )
        fp_rem = calib["confirmed_fp_lines_remaining"]
        fp_tot = calib["confirmed_fp_lines_total"]
        print(
            f"  Calibration Errors:     cliff_collapses={calib['cliff_collapses']} | "
            f"blind_spot_escapes={calib['blind_spot_escapes']} | "
            f"confirmed_fp_lines={fp_rem}/{fp_tot}"
        )


def _print_comparison(base: dict[str, Any], cand: dict[str, Any]) -> None:
    b_ct, c_ct = base["clean_traps"], cand["clean_traps"]
    b_mr, c_mr = base["messy_recall"], cand["messy_recall"]
    b_ep, c_ep = base["edit_pairs"], cand["edit_pairs"]
    b_cc, c_cc = base["corpus_calibration"], cand["corpus_calibration"]

    print("=== Polyglot & Cognitive-Clarity A/B Comparison ===")
    print(f"  Baseline:  {base['weights']}")
    print(f"  Candidate: {cand['weights']}\n")
    rows = [
        (
            "Clean Traps Passed (all)",
            f"{b_ct['passed']}/{b_ct['total']}",
            f"{c_ct['passed']}/{c_ct['total']}",
        ),
        (
            "Clean Traps Passed (held_out)",
            f"{b_ct['held_out_passed']}/{b_ct['held_out_total']}",
            f"{c_ct['held_out_passed']}/{c_ct['held_out_total']}",
        ),
        (
            "Total Clean-Trap FP Squiggles",
            str(b_ct["total_fp_squiggles"]),
            str(c_ct["total_fp_squiggles"]),
        ),
        (
            "Messy & Friction Recall",
            f"{b_mr['hits']}/{b_mr['total']}",
            f"{c_mr['hits']}/{c_mr['total']}",
        ),
        (
            "Edit Cleanup Directionality",
            f"{b_ep['cleanup_passed']}/{b_ep['cleanup_total']}",
            f"{c_ep['cleanup_passed']}/{c_ep['cleanup_total']}",
        ),
        (
            "Edit Anti-Hack / Cosmetic Gate",
            f"{b_ep['anti_hack_passed']}/{b_ep['anti_hack_total']}",
            f"{c_ep['anti_hack_passed']}/{c_ep['anti_hack_total']}",
        ),
    ]
    if b_cc.get("available_files", 0) > 0 and c_cc.get("available_files", 0) > 0:
        rows.extend(
            [
                (
                    "Clarity r (all 196 files)",
                    f"{b_cc['all_pearson_score_vs_clarity']:.4f}",
                    f"{c_cc['all_pearson_score_vs_clarity']:.4f}",
                ),
                (
                    "Clarity r (held_out split)",
                    f"{b_cc['held_out_pearson_score_vs_clarity']:.4f}",
                    f"{c_cc['held_out_pearson_score_vs_clarity']:.4f}",
                ),
                (
                    "rl_reward Clarity r (all)",
                    f"{b_cc['all_pearson_rl_vs_clarity']:.4f}",
                    f"{c_cc['all_pearson_rl_vs_clarity']:.4f}",
                ),
                (
                    "Cliff Collapses (clar>=0.65)",
                    str(b_cc["cliff_collapses"]),
                    str(c_cc["cliff_collapses"]),
                ),
                (
                    "Blind-Spot Escapes (score>=0.8)",
                    str(b_cc["blind_spot_escapes"]),
                    str(c_cc["blind_spot_escapes"]),
                ),
                (
                    "Confirmed FP Lines Remaining",
                    f"{b_cc['confirmed_fp_lines_remaining']}/{b_cc['confirmed_fp_lines_total']}",
                    f"{c_cc['confirmed_fp_lines_remaining']}/{c_cc['confirmed_fp_lines_total']}",
                ),
            ]
        )
    for metric, b_val, c_val in rows:
        print(f"  {metric:32s} | {b_val:14s} -> {c_val}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the 20-language polyglot & cognitive-clarity benchmark."
    )
    parser.add_argument(
        "--suite",
        default=str(REPO_ROOT / "benchmarks" / "polyglot_eval.json"),
    )
    parser.add_argument("--weights", help="Optional weights.npz path")
    parser.add_argument(
        "--corpus-root",
        default=str(REPO_ROOT),
        help="Optional root directory for full-file corpus calibration",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("BASELINE_NPZ", "CANDIDATE_NPZ"),
        help="Compare baseline vs candidate weights side-by-side",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    suite_path = Path(args.suite)
    corpus_root = Path(args.corpus_root)
    if args.compare:
        base_rep = evaluate_suite(suite_path, args.compare[0], corpus_root=corpus_root)
        cand_rep = evaluate_suite(suite_path, args.compare[1], corpus_root=corpus_root)
        if args.json:
            print(json.dumps({"baseline": base_rep, "candidate": cand_rep}, indent=2))
        else:
            _print_comparison(base_rep, cand_rep)
        return 0

    rep = evaluate_suite(suite_path, args.weights, corpus_root=corpus_root)
    if args.json:
        print(json.dumps(rep, indent=2))
    else:
        _print_single_report(rep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
