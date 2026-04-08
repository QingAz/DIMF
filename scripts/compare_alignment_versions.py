#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare two alignment-comparison experiment summaries, e.g. strict vs legacy."
    )
    parser.add_argument("--current-summary", type=Path, required=True, help="Current experiment summary JSON")
    parser.add_argument("--legacy-summary", type=Path, required=True, help="Legacy experiment summary JSON")
    parser.add_argument("--current-per-lag", type=Path, required=True, help="Current per-lag CSV")
    parser.add_argument("--legacy-per-lag", type=Path, required=True, help="Legacy per-lag CSV")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for figures and markdown")
    parser.add_argument("--current-label", default="strict", help="Label used for current experiment")
    parser.add_argument("--legacy-label", default="legacy", help="Label used for legacy experiment")
    return parser.parse_args()


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(str(path)))


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_forecast_plot(current_summary, legacy_summary, current_label, legacy_label, output_path: Path) -> None:
    metrics = ["MAE", "RMSE", "R2"]
    models = ["aligned", "noalign"]
    colors = {legacy_label: "#94a3b8", current_label: "#2563eb"}
    x = np.arange(len(models), dtype=np.float64)
    width = 0.34

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for ax, metric in zip(axes, metrics):
        legacy_values = [legacy_summary["forecast_metrics"][model][metric] for model in models]
        current_values = [current_summary["forecast_metrics"][model][metric] for model in models]
        ax.bar(x - width / 2.0, legacy_values, width=width, label=legacy_label, color=colors[legacy_label])
        ax.bar(x + width / 2.0, current_values, width=width, label=current_label, color=colors[current_label])
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_title(metric)
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("Forecast metric")
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle("Forecast Metrics: Strict vs Legacy", y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_lagged_plot(current_summary, legacy_summary, current_label, legacy_label, output_path: Path) -> None:
    models = ["aligned", "noalign"]
    metrics = [
        ("expected_lag_mae", "Lagged-only expected lag MAE"),
        ("argmax_lag_accuracy", "Lagged-only argmax accuracy"),
        ("mean_pred_expected_lag", "Lagged-only mean predicted lag"),
    ]
    colors = {legacy_label: "#94a3b8", current_label: "#2563eb"}
    x = np.arange(len(models), dtype=np.float64)
    width = 0.34

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
    for ax, (metric_key, title) in zip(axes, metrics):
        legacy_values = [legacy_summary["lag_recovery"][model]["lagged_only"][metric_key] for model in models]
        current_values = [current_summary["lag_recovery"][model]["lagged_only"][metric_key] for model in models]
        ax.bar(x - width / 2.0, legacy_values, width=width, label=legacy_label, color=colors[legacy_label])
        ax.bar(x + width / 2.0, current_values, width=width, label=current_label, color=colors[current_label])
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        if metric_key == "argmax_lag_accuracy":
            ax.set_ylim(0.0, 1.05)

    axes[0].set_ylabel("Lag-recovery metric")
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle("Lag-Recovery on Lagged Samples: Strict vs Legacy", y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_per_lag_plot(current_per_lag: pd.DataFrame, legacy_per_lag: pd.DataFrame, model: str, current_label: str, legacy_label: str, output_path: Path) -> None:
    value_col = f"{model}_expected_lag_mae"
    current = current_per_lag[["lag_gt", value_col]].rename(columns={value_col: current_label})
    legacy = legacy_per_lag[["lag_gt", value_col]].rename(columns={value_col: legacy_label})
    merged = legacy.merge(current, on="lag_gt", how="outer").sort_values("lag_gt").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(merged["lag_gt"], merged[legacy_label], marker="o", linewidth=1.8, color="#94a3b8", label=legacy_label)
    ax.plot(merged["lag_gt"], merged[current_label], marker="o", linewidth=1.8, color="#2563eb", label=current_label)
    ax.set_xlabel("True lag")
    ax.set_ylabel("Expected lag MAE")
    ax.set_title(f"{model}: per-lag expected-lag MAE")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_summary_markdown(current_summary, legacy_summary, current_label, legacy_label, output_path: Path) -> None:
    lines = [
        "# Strict vs Legacy Alignment Comparison",
        "",
        "## Scope",
        "",
        f"- Legacy summary: `{legacy_summary['raw_dataset']}`",
        f"- Current summary: `{current_summary['raw_dataset']}`",
        f"- Compared labels: `{legacy_label}` vs `{current_label}`",
        "",
        "## Forecast",
        "",
        "| model | legacy MAE | strict MAE | legacy RMSE | strict RMSE | legacy R2 | strict R2 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    for model in ["aligned", "noalign"]:
        legacy_metrics = legacy_summary["forecast_metrics"][model]
        current_metrics = current_summary["forecast_metrics"][model]
        lines.append(
            "| %s | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |"
            % (
                model,
                float(legacy_metrics["MAE"]),
                float(current_metrics["MAE"]),
                float(legacy_metrics["RMSE"]),
                float(current_metrics["RMSE"]),
                float(legacy_metrics["R2"]),
                float(current_metrics["R2"]),
            )
        )

    lines.extend(
        [
            "",
            "## Lagged-Only Recovery",
            "",
            "| model | legacy exp-lag MAE | strict exp-lag MAE | legacy argmax acc | strict argmax acc | legacy pred mean | strict pred mean |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )

    for model in ["aligned", "noalign"]:
        legacy_metrics = legacy_summary["lag_recovery"][model]["lagged_only"]
        current_metrics = current_summary["lag_recovery"][model]["lagged_only"]
        lines.append(
            "| %s | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |"
            % (
                model,
                float(legacy_metrics["expected_lag_mae"]),
                float(current_metrics["expected_lag_mae"]),
                float(legacy_metrics["argmax_lag_accuracy"]),
                float(current_metrics["argmax_lag_accuracy"]),
                float(legacy_metrics["mean_pred_expected_lag"]),
                float(current_metrics["mean_pred_expected_lag"]),
            )
        )

    aligned_legacy = legacy_summary["lag_recovery"]["aligned"]["lagged_only"]["expected_lag_mae"]
    aligned_current = current_summary["lag_recovery"]["aligned"]["lagged_only"]["expected_lag_mae"]
    noalign_legacy = legacy_summary["lag_recovery"]["noalign"]["lagged_only"]["expected_lag_mae"]
    noalign_current = current_summary["lag_recovery"]["noalign"]["lagged_only"]["expected_lag_mae"]

    lines.extend(
        [
            "",
            "## Takeaways",
            "",
            "- `aligned` 的 lagged-only expected lag MAE: `%.3f -> %.3f`。" % (aligned_legacy, aligned_current),
            "- `noalign` 的 lagged-only expected lag MAE: `%.3f -> %.3f`。" % (noalign_legacy, noalign_current),
            "- 重点看图：`forecast_by_version.png`、`lagged_recovery_by_version.png`、`aligned_per_lag_mae.png`、`noalign_per_lag_mae.png`。",
            "",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    current_summary_path = _absolute_path(args.current_summary)
    legacy_summary_path = _absolute_path(args.legacy_summary)
    current_per_lag_path = _absolute_path(args.current_per_lag)
    legacy_per_lag_path = _absolute_path(args.legacy_per_lag)
    output_dir = _absolute_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    current_summary = _load_json(current_summary_path)
    legacy_summary = _load_json(legacy_summary_path)
    current_per_lag = pd.read_csv(current_per_lag_path)
    legacy_per_lag = pd.read_csv(legacy_per_lag_path)

    forecast_plot = output_dir / "forecast_by_version.png"
    lagged_plot = output_dir / "lagged_recovery_by_version.png"
    aligned_plot = output_dir / "aligned_per_lag_mae.png"
    noalign_plot = output_dir / "noalign_per_lag_mae.png"
    summary_md = output_dir / "strict_vs_legacy_summary.md"

    _write_forecast_plot(current_summary, legacy_summary, args.current_label, args.legacy_label, forecast_plot)
    _write_lagged_plot(current_summary, legacy_summary, args.current_label, args.legacy_label, lagged_plot)
    _write_per_lag_plot(current_per_lag, legacy_per_lag, "aligned", args.current_label, args.legacy_label, aligned_plot)
    _write_per_lag_plot(current_per_lag, legacy_per_lag, "noalign", args.current_label, args.legacy_label, noalign_plot)
    _write_summary_markdown(current_summary, legacy_summary, args.current_label, args.legacy_label, summary_md)

    print("Wrote: %s" % forecast_plot)
    print("Wrote: %s" % lagged_plot)
    print("Wrote: %s" % aligned_plot)
    print("Wrote: %s" % noalign_plot)
    print("Wrote: %s" % summary_md)


if __name__ == "__main__":
    main()
