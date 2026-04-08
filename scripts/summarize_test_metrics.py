#!/usr/bin/env python3

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Tuple


def _load_pairs(csv_path: Path) -> Tuple[List[float], List[float]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        # 第 1 点修改：汇总脚本固定读取单点预测列。
        if "y_true" not in fieldnames or "y_pred" not in fieldnames:
            raise ValueError(f"Missing y_true/y_pred columns in {csv_path}")

        all_true: List[float] = []
        all_pred: List[float] = []

        for row in reader:
            all_true.append(float(row["y_true"]))
            all_pred.append(float(row["y_pred"]))

    return all_true, all_pred


def _metrics(y_true: List[float], y_pred: List[float]) -> Dict[str, float]:
    n = len(y_true)
    if n == 0:
        raise ValueError("No rows found while computing metrics")

    abs_err = [abs(a - b) for a, b in zip(y_true, y_pred)]
    sq_err = [(a - b) ** 2 for a, b in zip(y_true, y_pred)]
    y_bar = sum(y_true) / n
    ss_res = sum(sq_err)
    ss_tot = sum((a - y_bar) ** 2 for a in y_true)
    mse = ss_res / n

    return {
        "MSE": mse,
        "MAE": sum(abs_err) / n,
        "RMSE": math.sqrt(mse),
        "R2": 1.0 - ss_res / (ss_tot + 1e-12),
        "n": float(n),
    }


def summarize_dir(output_dir: Path) -> Dict[str, object]:
    raw_csv = output_dir / "test_pred_vs_true.csv"
    scaled_csv = output_dir / "test_pred_vs_true_scaled.csv"
    if not raw_csv.exists():
        raise FileNotFoundError(f"Missing file: {raw_csv}")

    raw_true, raw_pred = _load_pairs(raw_csv)
    scaled_true, scaled_pred = _load_pairs(scaled_csv) if scaled_csv.exists() else ([], [])

    raw_metrics = _metrics(raw_true, raw_pred)
    scaled_metrics = _metrics(scaled_true, scaled_pred) if scaled_true else None

    return {
        "run": output_dir.as_posix(),
        "task": "single_point_y_t_plus_H",
        "raw": raw_metrics,
        "scaled": scaled_metrics,
    }


def aggregate(runs: List[Dict[str, object]]) -> Dict[str, Dict[str, str]]:
    def collect(section: str, metric: str) -> List[float]:
        values = []
        for run in runs:
            part = run.get(section)
            if isinstance(part, dict) and metric in part:
                values.append(float(part[metric]))
        return values

    summary: Dict[str, Dict[str, str]] = {}
    for section in ("raw", "scaled"):
        if not isinstance(runs[0].get(section), dict):
            continue
        summary[section] = {}
        for metric in ("MSE", "MAE", "RMSE", "R2"):
            vals = collect(section, metric)
            if not vals:
                continue
            if len(vals) == 1:
                summary[section][metric] = f"{vals[0]:.6f}"
            else:
                summary[section][metric] = f"{mean(vals):.6f} ± {pstdev(vals):.6f}"
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize DIMF test metrics from output directories.")
    parser.add_argument("output_dirs", nargs="+", help="One or more output directories, e.g. outputs/multistage_aligned")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a readable summary")
    args = parser.parse_args()

    runs = [summarize_dir(Path(p)) for p in args.output_dirs]
    result = {"runs": runs, "summary": aggregate(runs)}

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    for run in runs:
        print(f"Run: {run['run']}")
        print(f"  task: {run['task']}")
        for section in ("raw", "scaled"):
            part = run.get(section)
            if not isinstance(part, dict):
                continue
            print(f"  {section}:")
            for metric in ("MSE", "MAE", "RMSE", "R2"):
                print(f"    {metric}: {part[metric]:.6f}")

    if len(runs) > 1:
        print("Summary:")
        for section, metrics in result["summary"].items():
            print(f"  {section}:")
            for metric, value in metrics.items():
                print(f"    {metric}: {value}")


if __name__ == "__main__":
    main()
