#!/usr/bin/env python3

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from alignment_peak_metrics import attach_peak_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run per-seed aligned-vs-noalign comparison and aggregate alignment metrics."
    )
    parser.add_argument("--aligned-root", type=Path, required=True, help="Aligned multiseed output root.")
    parser.add_argument("--noalign-root", type=Path, required=True, help="No-align multiseed output root.")
    parser.add_argument("--raw-dataset", type=Path, required=True, help="Raw-gap dataset CSV used for evaluation.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for per-seed and aggregated outputs.")
    parser.add_argument("--bump-plan", type=Path, default=None, help="Optional framework local_bump_plan.json.")
    parser.add_argument("--edge", default="stage1_to_stage2", help="Edge name passed through to comparison scripts.")
    parser.add_argument("--force-compare", action="store_true", help="Regenerate per-seed comparison files even if they exist.")
    parser.add_argument("--skip-visuals", action="store_true", help="Skip per-seed visualize_alignment_comparison.py.")
    return parser.parse_args()


def _seed_dirs(root: Path) -> Dict[str, Path]:
    if not root.exists():
        raise FileNotFoundError("Missing multiseed root: %s" % root)
    return {path.name: path for path in sorted(root.iterdir()) if path.is_dir() and path.name.startswith("seed_")}


def _common_seeds(aligned_root: Path, noalign_root: Path) -> List[str]:
    aligned = _seed_dirs(aligned_root)
    noalign = _seed_dirs(noalign_root)
    common = sorted(set(aligned).intersection(noalign), key=lambda item: int(item.split("_")[-1]))
    if not common:
        raise ValueError("No common seed directories found between %s and %s" % (aligned_root, noalign_root))
    return common


def _run(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True)


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _get_path(payload: Dict[str, Any], keys: Iterable[str]) -> Any:
    current: Any = payload
    path: List[str] = []
    for key in keys:
        path.append(key)
        if not isinstance(current, dict) or key not in current:
            raise KeyError("Missing key path: %s" % ".".join(path))
        current = current[key]
    return current


def _summary_block(values: List[float]) -> Dict[str, Any]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return {"n": 0, "mean": None, "std": None, "formatted": "NA"}
    value_mean = mean(clean)
    value_std = pstdev(clean) if len(clean) > 1 else 0.0
    return {
        "n": len(clean),
        "mean": value_mean,
        "std": value_std,
        "formatted": "%.6f +/- %.6f" % (value_mean, value_std),
    }


def _aggregate_metric(rows: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    aligned_values = [row.get("%s_aligned" % key) for row in rows]
    noalign_values = [row.get("%s_noalign" % key) for row in rows]
    diff_values = [
        row.get("%s_noalign" % key) - row.get("%s_aligned" % key)
        for row in rows
        if row.get("%s_aligned" % key) is not None and row.get("%s_noalign" % key) is not None
    ]
    return {
        "aligned": _summary_block(aligned_values),
        "noalign": _summary_block(noalign_values),
        "diff_noalign_minus_aligned": _summary_block(diff_values),
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No rows provided for CSV output.")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _find_zero_lag_row(per_lag: pd.DataFrame) -> Dict[str, Any]:
    rows = per_lag[per_lag["lag_gt"].astype(int) == 0]
    if rows.empty:
        raise ValueError("Could not find lag_gt=0 row in per_lag_comparison.")
    return rows.iloc[0].to_dict()


def _ensure_peak(summary: Dict[str, Any], seed_output: Path) -> pd.DataFrame:
    joined_path = seed_output / "alignment_test_joined.csv"
    if not joined_path.exists():
        raise FileNotFoundError("Missing joined comparison CSV: %s" % joined_path)
    joined = pd.read_csv(joined_path)
    peak_blocks = attach_peak_metrics(summary, joined)
    peak_blocks.to_csv(seed_output / "alignment_peak_summary.csv", index=False)
    _write_json(seed_output / "alignment_comparison_summary.json", summary)
    return peak_blocks


def _run_compare_if_needed(args: argparse.Namespace, seed_name: str, seed_output: Path) -> None:
    summary_path = seed_output / "alignment_comparison_summary.json"
    joined_path = seed_output / "alignment_test_joined.csv"
    if summary_path.exists() and joined_path.exists() and not args.force_compare:
        return

    project_root = Path(__file__).resolve().parents[1]
    compare_script = project_root / "scripts" / "compare_alignment_lag_effect.py"
    aligned_seed = args.aligned_root / seed_name
    noalign_seed = args.noalign_root / seed_name
    seed_output.mkdir(parents=True, exist_ok=True)
    compare_cmd = [
        sys.executable,
        str(compare_script),
        "--aligned-estimates",
        str(aligned_seed / "test_delay_estimates.csv"),
        "--noalign-estimates",
        str(noalign_seed / "test_delay_estimates.csv"),
        "--aligned-metrics",
        str(aligned_seed / "test_metrics.json"),
        "--noalign-metrics",
        str(noalign_seed / "test_metrics.json"),
        "--raw-dataset",
        str(args.raw_dataset),
        "--output-dir",
        str(seed_output),
        "--edge",
        args.edge,
    ]
    _run(compare_cmd)


def _seed_metric_row(seed_name: str, summary: Dict[str, Any], per_lag: pd.DataFrame) -> Dict[str, Any]:
    zero_lag = _find_zero_lag_row(per_lag)
    peak = summary["benchmark"]["peak"]
    row: Dict[str, Any] = {
        "seed": seed_name,
        "forecast_mae_aligned": _get_path(summary, ["forecast_metrics", "aligned", "MAE"]),
        "forecast_mae_noalign": _get_path(summary, ["forecast_metrics", "noalign", "MAE"]),
        "forecast_rmse_aligned": _get_path(summary, ["forecast_metrics", "aligned", "RMSE"]),
        "forecast_rmse_noalign": _get_path(summary, ["forecast_metrics", "noalign", "RMSE"]),
        "forecast_r2_aligned": _get_path(summary, ["forecast_metrics", "aligned", "R2"]),
        "forecast_r2_noalign": _get_path(summary, ["forecast_metrics", "noalign", "R2"]),
        "block_in_expected_lag_mae_aligned": _get_path(summary, ["benchmark", "block_in_expected_lag_mae", "aligned"]),
        "block_in_expected_lag_mae_noalign": _get_path(summary, ["benchmark", "block_in_expected_lag_mae", "noalign"]),
        "localization_auprc_aligned": _get_path(summary, ["benchmark", "localization", "aligned", "auprc"]),
        "localization_auprc_noalign": _get_path(summary, ["benchmark", "localization", "noalign", "auprc"]),
        "localization_best_f1_aligned": _get_path(summary, ["benchmark", "localization", "aligned", "best_f1"]),
        "localization_best_f1_noalign": _get_path(summary, ["benchmark", "localization", "noalign", "best_f1"]),
        "block_out_false_alarm_rate_aligned": _get_path(summary, ["benchmark", "block_out_false_alarm_rate", "aligned"]),
        "block_out_false_alarm_rate_noalign": _get_path(summary, ["benchmark", "block_out_false_alarm_rate", "noalign"]),
        "d0_mean_pred_expected_lag_aligned": zero_lag["aligned_mean_pred_expected_lag"],
        "d0_mean_pred_expected_lag_noalign": zero_lag["noalign_mean_pred_expected_lag"],
        "lagged_expected_lag_mae_aligned": _get_path(summary, ["lag_recovery", "aligned", "lagged_only", "expected_lag_mae"]),
        "lagged_expected_lag_mae_noalign": _get_path(summary, ["lag_recovery", "noalign", "lagged_only", "expected_lag_mae"]),
        "lagged_argmax_accuracy_aligned": _get_path(summary, ["lag_recovery", "aligned", "lagged_only", "argmax_lag_accuracy"]),
        "lagged_argmax_accuracy_noalign": _get_path(summary, ["lag_recovery", "noalign", "lagged_only", "argmax_lag_accuracy"]),
        "peak_error_aligned": peak["aligned"]["peak_error"],
        "peak_error_noalign": peak["noalign"]["peak_error"],
        "peak_hit_at_0_aligned": peak["aligned"]["peak_hit_at_0"],
        "peak_hit_at_0_noalign": peak["noalign"]["peak_hit_at_0"],
        "peak_hit_at_pm1_aligned": peak["aligned"]["peak_hit_at_pm1"],
        "peak_hit_at_pm1_noalign": peak["noalign"]["peak_hit_at_pm1"],
        "peak_pred_mean_aligned": peak["aligned"]["mean_pred_peak_expected_lag"],
        "peak_pred_mean_noalign": peak["noalign"]["mean_pred_peak_expected_lag"],
    }
    return row


def _aggregate_by_dmax(seed_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_dmax: Dict[str, Dict[str, Any]] = {}
    for summary in seed_summaries:
        for dmax_key, payload in summary.get("benchmark_by_dmax", {}).items():
            item = by_dmax.setdefault(dmax_key, {"n_samples": [], "metrics": {}})
            item["n_samples"].append(payload.get("n_samples"))
            metric_map = {
                "block_in_expected_lag_mae": (
                    payload["aligned"].get("block_in_expected_lag_mae"),
                    payload["noalign"].get("block_in_expected_lag_mae"),
                ),
                "localization_auprc": (
                    payload["aligned"]["localization"].get("auprc"),
                    payload["noalign"]["localization"].get("auprc"),
                ),
                "localization_best_f1": (
                    payload["aligned"]["localization"].get("best_f1"),
                    payload["noalign"]["localization"].get("best_f1"),
                ),
                "block_out_false_alarm_rate": (
                    payload["aligned"].get("block_out_false_alarm_rate"),
                    payload["noalign"].get("block_out_false_alarm_rate"),
                ),
                "peak_error": (
                    payload["aligned"]["peak"].get("peak_error"),
                    payload["noalign"]["peak"].get("peak_error"),
                ),
                "peak_hit_at_0": (
                    payload["aligned"]["peak"].get("peak_hit_at_0"),
                    payload["noalign"]["peak"].get("peak_hit_at_0"),
                ),
                "peak_hit_at_pm1": (
                    payload["aligned"]["peak"].get("peak_hit_at_pm1"),
                    payload["noalign"]["peak"].get("peak_hit_at_pm1"),
                ),
            }
            for metric_key, (aligned_value, noalign_value) in metric_map.items():
                metric_bucket = item["metrics"].setdefault(metric_key, {"aligned": [], "noalign": []})
                metric_bucket["aligned"].append(aligned_value)
                metric_bucket["noalign"].append(noalign_value)

    output: Dict[str, Any] = {}
    for dmax_key, payload in by_dmax.items():
        metric_summary: Dict[str, Any] = {}
        for metric_key, values in payload["metrics"].items():
            rows = [
                {"%s_aligned" % metric_key: a, "%s_noalign" % metric_key: n}
                for a, n in zip(values["aligned"], values["noalign"])
            ]
            metric_summary[metric_key] = _aggregate_metric(rows, metric_key)
        output[dmax_key] = {
            "n_samples": _summary_block(payload["n_samples"]),
            "metrics": metric_summary,
        }
    return output


def _render_markdown(seeds: List[str], metrics: Dict[str, Any], by_dmax: Dict[str, Any]) -> str:
    metric_order = [
        ("lagged_expected_lag_mae", "Lagged-only Expected-Lag MAE"),
        ("localization_auprc", "Localization AUPRC"),
        ("localization_best_f1", "Localization best-F1"),
        ("block_out_false_alarm_rate", "Block-out False Alarm Rate"),
        ("d0_mean_pred_expected_lag", "E[d_hat | d=0]"),
        ("lagged_argmax_accuracy", "Lagged-only Argmax Accuracy"),
        ("peak_error", "Peak Error"),
        ("peak_hit_at_0", "Peak Hit@0"),
        ("peak_hit_at_pm1", "Peak Hit@+/-1"),
        ("peak_pred_mean", "Mean Predicted Peak"),
    ]
    lines = [
        "# Multiseed Alignment Comparison",
        "",
        "Seeds: %s" % ", ".join(seeds),
        "",
        "## Core Metrics",
        "",
        "| metric | aligned | noalign | diff (noalign - aligned) |",
        "| --- | --- | --- | --- |",
    ]
    for metric_key, label in metric_order:
        metric = metrics[metric_key]
        lines.append(
            "| %s | %s | %s | %s |"
            % (
                label,
                metric["aligned"]["formatted"],
                metric["noalign"]["formatted"],
                metric["diff_noalign_minus_aligned"]["formatted"],
            )
        )

    lines.extend(
        [
            "",
            "## By True dmax",
            "",
            "| dmax | n_samples | metric | aligned | noalign | diff (noalign - aligned) |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    by_dmax_order = [
        ("block_in_expected_lag_mae", "Block-in Expected-Lag MAE"),
        ("localization_auprc", "Localization AUPRC"),
        ("localization_best_f1", "Localization best-F1"),
        ("block_out_false_alarm_rate", "Block-out False Alarm Rate"),
        ("peak_error", "Peak Error"),
        ("peak_hit_at_0", "Peak Hit@0"),
        ("peak_hit_at_pm1", "Peak Hit@+/-1"),
    ]
    for dmax_key in sorted(by_dmax.keys(), key=lambda item: int(item)):
        item = by_dmax[dmax_key]
        for metric_key, label in by_dmax_order:
            metric = item["metrics"][metric_key]
            lines.append(
                "| %s | %s | %s | %s | %s | %s |"
                % (
                    dmax_key,
                    item["n_samples"]["formatted"],
                    label,
                    metric["aligned"]["formatted"],
                    metric["noalign"]["formatted"],
                    metric["diff_noalign_minus_aligned"]["formatted"],
                )
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = _common_seeds(args.aligned_root, args.noalign_root)

    seed_rows: List[Dict[str, Any]] = []
    seed_summaries: List[Dict[str, Any]] = []
    for seed_name in seeds:
        seed_output = args.output_dir / seed_name
        _run_compare_if_needed(args, seed_name, seed_output)
        summary = _read_json(seed_output / "alignment_comparison_summary.json")
        per_lag = pd.read_csv(seed_output / "alignment_per_lag_comparison.csv")
        _ensure_peak(summary, seed_output)
        summary = _read_json(seed_output / "alignment_comparison_summary.json")
        seed_summaries.append(summary)
        seed_rows.append(_seed_metric_row(seed_name, summary, per_lag))

    metric_keys = [
        "forecast_mae",
        "forecast_rmse",
        "forecast_r2",
        "block_in_expected_lag_mae",
        "localization_auprc",
        "localization_best_f1",
        "block_out_false_alarm_rate",
        "d0_mean_pred_expected_lag",
        "lagged_expected_lag_mae",
        "lagged_argmax_accuracy",
        "peak_error",
        "peak_hit_at_0",
        "peak_hit_at_pm1",
        "peak_pred_mean",
    ]
    metrics = {metric_key: _aggregate_metric(seed_rows, metric_key) for metric_key in metric_keys}
    by_dmax = _aggregate_by_dmax(seed_summaries)

    summary = {
        "aligned_root": str(args.aligned_root.resolve()),
        "noalign_root": str(args.noalign_root.resolve()),
        "raw_dataset": str(args.raw_dataset.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "seeds": seeds,
        "metrics": metrics,
        "benchmark_by_dmax": by_dmax,
    }

    seed_csv = args.output_dir / "multiseed_alignment_metrics.csv"
    summary_json = args.output_dir / "multiseed_alignment_summary.json"
    report_md = args.output_dir / "multiseed_alignment_report.md"
    by_dmax_csv = args.output_dir / "multiseed_alignment_by_dmax.csv"

    _write_csv(seed_csv, seed_rows)
    _write_json(summary_json, summary)
    report_md.write_text(_render_markdown(seeds, metrics, by_dmax), encoding="utf-8")

    flat_dmax_rows: List[Dict[str, Any]] = []
    for dmax_key in sorted(by_dmax.keys(), key=lambda item: int(item)):
        item = by_dmax[dmax_key]
        for metric_key, metric in item["metrics"].items():
            flat_dmax_rows.append(
                {
                    "dmax": dmax_key,
                    "n_samples": item["n_samples"]["formatted"],
                    "metric": metric_key,
                    "aligned": metric["aligned"]["formatted"],
                    "noalign": metric["noalign"]["formatted"],
                    "diff_noalign_minus_aligned": metric["diff_noalign_minus_aligned"]["formatted"],
                }
            )
    _write_csv(by_dmax_csv, flat_dmax_rows)

    print("Wrote per-seed metrics to %s" % seed_csv)
    print("Wrote aggregate summary to %s" % summary_json)
    print("Wrote markdown report to %s" % report_md)


if __name__ == "__main__":
    main()
