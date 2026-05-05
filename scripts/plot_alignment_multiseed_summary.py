#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

if not os.environ.get("MPLCONFIGDIR"):
    _mpl_dir = Path.cwd() / ".matplotlib-codex"
    _mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(_mpl_dir)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MetricSpec = Tuple[str, str]

CORE_METRICS: Sequence[MetricSpec] = (
    ("Lagged-only Expected-Lag MAE", "lagged_expected_lag_mae"),
    ("Localization AUPRC", "localization_auprc"),
    ("Localization best-F1", "localization_best_f1"),
    ("Block-out False Alarm Rate", "block_out_false_alarm_rate"),
    ("E[d_hat | d=0]", "d0_mean_pred_expected_lag"),
    ("Lagged-only Argmax Accuracy", "lagged_argmax_accuracy"),
    ("Peak Error", "peak_error"),
    ("Peak Hit@0", "peak_hit_at_0"),
    ("Peak Hit@+/-1", "peak_hit_at_pm1"),
)

RECOVERY_METRICS: Sequence[MetricSpec] = (
    ("Lagged MAE", "lagged_expected_lag_mae"),
    ("AUPRC", "localization_auprc"),
    ("best-F1", "localization_best_f1"),
    ("FAR", "block_out_false_alarm_rate"),
)

BIAS_METRICS: Sequence[MetricSpec] = (
    ("E[d_hat|d=0]", "d0_mean_pred_expected_lag"),
    ("Lagged argmax acc", "lagged_argmax_accuracy"),
)

PEAK_METRICS: Sequence[MetricSpec] = (
    ("Peak Error", "peak_error"),
    ("Hit@0", "peak_hit_at_0"),
    ("Hit@+/-1", "peak_hit_at_pm1"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render 5-seed alignment comparison dashboard figures."
    )
    parser.add_argument("--summary", type=Path, required=True, help="Path to multiseed_alignment_summary.json")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for 5-seed visualization outputs")
    return parser.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _metric_payload(summary: Dict[str, Any], metric_key: str) -> Dict[str, Any]:
    metrics = summary.get("metrics", {})
    if metric_key not in metrics:
        raise KeyError(f"Missing metric in multiseed summary: {metric_key}")
    return metrics[metric_key]


def _metric_triplet(summary: Dict[str, Any], metric_key: str) -> Tuple[float, float, float, float]:
    metric = _metric_payload(summary, metric_key)
    aligned = metric["aligned"]
    noalign = metric["noalign"]
    return (
        float(aligned["mean"]),
        float(aligned["std"]),
        float(noalign["mean"]),
        float(noalign["std"]),
    )


def _formatted_triplet(summary: Dict[str, Any], metric_key: str) -> Tuple[str, str, str]:
    metric = _metric_payload(summary, metric_key)
    return (
        str(metric["aligned"]["formatted"]),
        str(metric["noalign"]["formatted"]),
        str(metric["diff_noalign_minus_aligned"]["formatted"]),
    )


def _render_table(df: pd.DataFrame, title: str, subtitle: str, output_path: Path) -> None:
    n_rows, n_cols = df.shape
    fig_w = max(10.5, 2.6 * n_cols)
    fig_h = max(3.0, 1.2 + 0.65 * n_rows)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    col_widths = None
    if list(df.columns) == ["Metric", "aligned", "noalign", "diff (noalign - aligned)"]:
        col_widths = [0.32, 0.22, 0.22, 0.24]

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        colWidths=col_widths,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.45)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#dbeafe")
            cell.set_text_props(weight="bold")
        elif row % 2 == 1:
            cell.set_facecolor("#f8fafc")
        else:
            cell.set_facecolor("#ffffff")
        cell.set_edgecolor("#94a3b8")

    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.97)
    ax.set_title(subtitle, fontsize=10, pad=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_grouped_bar(
    ax: plt.Axes,
    labels: Sequence[str],
    aligned_means: Sequence[float],
    aligned_stds: Sequence[float],
    noalign_means: Sequence[float],
    noalign_stds: Sequence[float],
    ylabel: str,
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    x = np.arange(len(labels), dtype=np.float64)
    width = 0.36

    ax.bar(
        x - width / 2,
        aligned_means,
        width=width,
        yerr=aligned_stds,
        capsize=4,
        label="aligned",
        color="#2563eb",
        edgecolor="#1d4ed8",
        linewidth=0.8,
    )
    ax.bar(
        x + width / 2,
        noalign_means,
        width=width,
        yerr=noalign_stds,
        capsize=4,
        label="noalign",
        color="#ef4444",
        edgecolor="#b91c1c",
        linewidth=0.8,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    if ylim is not None:
        ax.set_ylim(ylim)


def _collect_metric_values(
    summary: Dict[str, Any], metric_specs: Sequence[MetricSpec]
) -> Tuple[List[str], List[float], List[float], List[float], List[float]]:
    labels: List[str] = []
    aligned_means: List[float] = []
    aligned_stds: List[float] = []
    noalign_means: List[float] = []
    noalign_stds: List[float] = []
    for label, metric_key in metric_specs:
        a_mean, a_std, n_mean, n_std = _metric_triplet(summary, metric_key)
        labels.append(label)
        aligned_means.append(a_mean)
        aligned_stds.append(a_std)
        noalign_means.append(n_mean)
        noalign_stds.append(n_std)
    return labels, aligned_means, aligned_stds, noalign_means, noalign_stds


def _write_core_dashboard(summary: Dict[str, Any], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.4))
    fig.patch.set_facecolor("white")

    labels, a_mean, a_std, n_mean, n_std = _collect_metric_values(summary, RECOVERY_METRICS)
    _plot_grouped_bar(axes[0], labels, a_mean, a_std, n_mean, n_std, ylabel="Value")
    axes[0].set_title("Lag Recovery And Detection", fontsize=13, fontweight="bold")
    axes[0].legend(loc="upper right", frameon=False)

    labels, a_mean, a_std, n_mean, n_std = _collect_metric_values(summary, PEAK_METRICS)
    _plot_grouped_bar(axes[1], labels, a_mean, a_std, n_mean, n_std, ylabel="Value")
    axes[1].set_title("Peak Accuracy", fontsize=13, fontweight="bold")
    axes[1].legend(loc="upper right", frameon=False)

    labels, a_mean, a_std, n_mean, n_std = _collect_metric_values(summary, BIAS_METRICS)
    _plot_grouped_bar(axes[2], labels, a_mean, a_std, n_mean, n_std, ylabel="Value")
    axes[2].set_title("Zero-Region Bias And Exact-Hit Accuracy", fontsize=13, fontweight="bold")
    axes[2].legend(loc="upper right", frameon=False)

    lines = [
        "5-seed alignment comparison",
        "Lag behavior: aligned improves lagged-only MAE and localization metrics.",
        "Peak behavior: peak error and hit rates separate detection from amplitude accuracy.",
        "Tradeoff: aligned also introduces a positive bias when true lag = 0.",
        "See the table and by-dmax chart for exact mean +/- std values.",
    ]
    fig.text(
        0.012,
        0.02,
        "\n".join(lines),
        ha="left",
        va="bottom",
        fontsize=9.5,
        color="#333333",
        linespacing=1.35,
        bbox={"facecolor": "#f8fafc", "edgecolor": "#cbd5e1", "boxstyle": "round,pad=0.6"},
    )

    fig.suptitle("DIMF Alignment 5-Seed Dashboard", fontsize=16, fontweight="bold", y=0.98)
    output_path = output_dir / "alignment_5seed_dashboard.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.16, 1.0, 0.94))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_core_table(summary: Dict[str, Any], output_dir: Path) -> None:
    rows: List[Dict[str, str]] = []
    for label, metric_key in CORE_METRICS:
        aligned_fmt, noalign_fmt, diff_fmt = _formatted_triplet(summary, metric_key)
        rows.append(
            {
                "Metric": label,
                "aligned": aligned_fmt,
                "noalign": noalign_fmt,
                "diff (noalign - aligned)": diff_fmt,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "alignment_5seed_core_metrics.csv", index=False)
    _render_table(
        df,
        title="Alignment 5-Seed Core Metrics",
        subtitle="Cells show mean +/- std across seeds.",
        output_path=output_dir / "alignment_5seed_core_metrics.png",
    )


def _dmax_metric(summary: Dict[str, Any], dmax_key: str, metric_key: str) -> Dict[str, Any]:
    return summary["benchmark_by_dmax"][dmax_key]["metrics"][metric_key]


def _write_dmax_dashboard(summary: Dict[str, Any], output_dir: Path) -> None:
    by_dmax = summary.get("benchmark_by_dmax", {})
    dmax_keys = sorted(by_dmax.keys(), key=lambda item: int(item))
    labels = [f"d{key}" for key in dmax_keys]
    x = np.arange(len(labels), dtype=np.float64)
    width = 0.36

    fig, axes = plt.subplots(2, 4, figsize=(20.0, 8.5))
    axes_flat = axes.ravel()
    metric_specs: Sequence[MetricSpec] = (
        ("Block-in Expected-Lag MAE", "block_in_expected_lag_mae"),
        ("Localization AUPRC", "localization_auprc"),
        ("Localization best-F1", "localization_best_f1"),
        ("Block-out False Alarm Rate", "block_out_false_alarm_rate"),
        ("Peak Error", "peak_error"),
        ("Peak Hit@0", "peak_hit_at_0"),
        ("Peak Hit@+/-1", "peak_hit_at_pm1"),
    )

    for ax, (label, metric_key) in zip(axes_flat, metric_specs):
        aligned_mean = [
            float(_dmax_metric(summary, dmax_key, metric_key)["aligned"]["mean"]) for dmax_key in dmax_keys
        ]
        aligned_std = [
            float(_dmax_metric(summary, dmax_key, metric_key)["aligned"]["std"]) for dmax_key in dmax_keys
        ]
        noalign_mean = [
            float(_dmax_metric(summary, dmax_key, metric_key)["noalign"]["mean"]) for dmax_key in dmax_keys
        ]
        noalign_std = [
            float(_dmax_metric(summary, dmax_key, metric_key)["noalign"]["std"]) for dmax_key in dmax_keys
        ]

        ax.bar(
            x - width / 2,
            aligned_mean,
            width=width,
            yerr=aligned_std,
            capsize=4,
            label="aligned",
            color="#2563eb",
            edgecolor="#1d4ed8",
            linewidth=0.8,
        )
        ax.bar(
            x + width / 2,
            noalign_mean,
            width=width,
            yerr=noalign_std,
            capsize=4,
            label="noalign",
            color="#ef4444",
            edgecolor="#b91c1c",
            linewidth=0.8,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        ax.legend(loc="upper left", frameon=False)

    for ax in axes_flat[len(metric_specs) :]:
        ax.axis("off")

    fig.suptitle("Alignment 5-Seed Metrics By True dmax", fontsize=16, fontweight="bold", y=0.98)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(output_dir / "alignment_5seed_by_dmax_dashboard.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_dmax_table(summary: Dict[str, Any], output_dir: Path) -> None:
    rows: List[Dict[str, str]] = []
    by_dmax = summary.get("benchmark_by_dmax", {})
    for dmax_key in sorted(by_dmax.keys(), key=lambda item: int(item)):
        item = by_dmax[dmax_key]
        metrics = item["metrics"]
        rows.append(
            {
                "true dmax": f"d{dmax_key}",
                "n_samples": str(item["n_samples"]["formatted"]),
                "Lagged MAE (aligned)": str(metrics["block_in_expected_lag_mae"]["aligned"]["formatted"]),
                "Lagged MAE (noalign)": str(metrics["block_in_expected_lag_mae"]["noalign"]["formatted"]),
                "AUPRC (aligned)": str(metrics["localization_auprc"]["aligned"]["formatted"]),
                "FAR (aligned)": str(metrics["block_out_false_alarm_rate"]["aligned"]["formatted"]),
                "Peak Error (aligned)": str(metrics["peak_error"]["aligned"]["formatted"]),
                "Peak Hit@0 (aligned)": str(metrics["peak_hit_at_0"]["aligned"]["formatted"]),
                "Peak Hit@+/-1 (aligned)": str(metrics["peak_hit_at_pm1"]["aligned"]["formatted"]),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "alignment_5seed_by_dmax.csv", index=False)
    _render_table(
        df,
        title="Alignment 5-Seed By True dmax",
        subtitle="Rows summarize per-dmax means +/- std across seeds.",
        output_path=output_dir / "alignment_5seed_by_dmax.png",
    )


def main() -> None:
    args = parse_args()
    summary = _load_json(args.summary)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_core_dashboard(summary, output_dir)
    _write_core_table(summary, output_dir)
    _write_dmax_dashboard(summary, output_dir)
    _write_dmax_table(summary, output_dir)
    print(f"Wrote 5-seed visuals to {output_dir}")


if __name__ == "__main__":
    main()
