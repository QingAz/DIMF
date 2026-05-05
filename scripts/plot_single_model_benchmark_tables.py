#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render overall and by-dmax benchmark tables for a single model summary."
    )
    parser.add_argument("--summary", type=Path, required=True, help="Path to benchmark_summary.json")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    return parser.parse_args()


def _load_summary(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (int,)) and not isinstance(value, bool):
        return str(value)
    return f"{float(value):.{digits}f}"


def _render_table(df: pd.DataFrame, title: str, output_path: Path) -> None:
    fig_height = max(2.2, 0.55 * (len(df) + 1))
    fig, ax = plt.subplots(figsize=(10.5, fig_height))
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.35)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d1d5db")
        if row == 0:
            cell.set_facecolor("#e5e7eb")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("#ffffff" if row % 2 else "#f9fafb")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _overall_table(summary: Dict[str, Any]) -> pd.DataFrame:
    benchmark = summary.get("benchmark", {})
    forecast = summary.get("forecast_metrics", {})
    rows: List[Dict[str, str]] = [
        {"Metric": "Forecast MAE", "Value": _fmt(forecast.get("MAE"))},
        {"Metric": "Forecast RMSE", "Value": _fmt(forecast.get("RMSE"))},
        {"Metric": "Forecast R2", "Value": _fmt(forecast.get("R2"))},
        {"Metric": "Block-in Expected-Lag MAE", "Value": _fmt(benchmark.get("block_in_expected_lag_mae"))},
        {"Metric": "Localization AUPRC", "Value": _fmt(benchmark.get("localization", {}).get("auprc"))},
        {"Metric": "Localization best-F1", "Value": _fmt(benchmark.get("localization", {}).get("best_f1"))},
        {"Metric": "Block-out False Alarm Rate", "Value": _fmt(benchmark.get("block_out_false_alarm_rate"))},
        {
            "Metric": "E[d_hat | d_true = 0]",
            "Value": _fmt(benchmark.get("mean_pred_expected_lag_when_true_zero")),
        },
    ]
    return pd.DataFrame(rows)


def _by_dmax_table(summary: Dict[str, Any]) -> pd.DataFrame:
    by_dmax = summary.get("benchmark_by_dmax", {})
    rows: List[Dict[str, str]] = []
    for dmax_str in sorted(by_dmax.keys(), key=lambda x: int(x)):
        item = by_dmax[dmax_str]
        rows.append(
            {
                "true_dmax": dmax_str,
                "n_segments": _fmt(item.get("n_segments"), 0),
                "n_rows": _fmt(item.get("n_rows"), 0),
                "n_positive_rows": _fmt(item.get("n_positive_rows"), 0),
                "Block-in MAE": _fmt(item.get("block_in_expected_lag_mae")),
                "AUPRC": _fmt(item.get("localization", {}).get("auprc")),
                "best-F1": _fmt(item.get("localization", {}).get("best_f1")),
                "FAR": _fmt(item.get("block_out_false_alarm_rate")),
                "E[d_hat|d=0]": _fmt(item.get("mean_pred_expected_lag_when_true_zero")),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    summary = _load_summary(args.summary)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    overall_df = _overall_table(summary)
    by_dmax_df = _by_dmax_table(summary)

    overall_df.to_csv(output_dir / "benchmark_overall.csv", index=False)
    by_dmax_df.to_csv(output_dir / "benchmark_by_dmax.csv", index=False)

    _render_table(overall_df, "Single-Model Benchmark Summary", output_dir / "benchmark_overall.png")
    _render_table(by_dmax_df, "Conditioned on true dmax", output_dir / "benchmark_by_dmax.png")


if __name__ == "__main__":
    main()
