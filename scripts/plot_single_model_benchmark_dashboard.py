#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

if not os.environ.get("MPLCONFIGDIR"):
    _mpl_dir = Path.cwd() / ".matplotlib-codex"
    _mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(_mpl_dir)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd


BG = "#fbfaf8"
TEXT = "#111827"
MUTED = "#6b7280"
GRID = "#d1d5db"
CARD_BG = "#ffffff"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a chart-first dashboard for a single-model benchmark summary."
    )
    parser.add_argument("--summary", type=Path, required=True, help="Path to benchmark_summary.json")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write dashboard files")
    parser.add_argument(
        "--title",
        default="Single-Model Benchmark Dashboard",
        help="Figure title",
    )
    return parser.parse_args()


def _load_summary(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _fmt_value(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (int,)) and not isinstance(value, bool):
        return str(value)
    return f"{float(value):.{digits}f}"


def _metric_cards(summary: Dict[str, Any]) -> List[Dict[str, str]]:
    benchmark = summary.get("benchmark", {})
    localization = benchmark.get("localization", {})
    forecast = summary.get("forecast_metrics", {})
    return [
        {
            "group": "Forecast",
            "label": "Forecast MAE",
            "value": _fmt_value(forecast.get("MAE")),
            "note": "Lower is better",
            "color": "#2563eb",
        },
        {
            "group": "Forecast",
            "label": "Forecast RMSE",
            "value": _fmt_value(forecast.get("RMSE")),
            "note": "Lower is better",
            "color": "#2563eb",
        },
        {
            "group": "Forecast",
            "label": "Forecast R2",
            "value": _fmt_value(forecast.get("R2")),
            "note": "Higher is better",
            "color": "#2563eb",
        },
        {
            "group": "Lag Accuracy",
            "label": "Block-in MAE",
            "value": _fmt_value(benchmark.get("block_in_expected_lag_mae")),
            "note": "Lower is better",
            "color": "#ea580c",
        },
        {
            "group": "Localization",
            "label": "AUPRC",
            "value": _fmt_value(localization.get("auprc")),
            "note": "Higher is better",
            "color": "#ca8a04",
        },
        {
            "group": "Localization",
            "label": "best-F1",
            "value": _fmt_value(localization.get("best_f1")),
            "note": "Higher is better",
            "color": "#ca8a04",
        },
        {
            "group": "Robustness",
            "label": "False Alarm Rate",
            "value": _fmt_value(benchmark.get("block_out_false_alarm_rate")),
            "note": "Lower is better",
            "color": "#059669",
        },
        {
            "group": "Bias",
            "label": "E[d_hat | d=0]",
            "value": _fmt_value(benchmark.get("mean_pred_expected_lag_when_true_zero")),
            "note": "Closer to 0 is better",
            "color": "#7c3aed",
        },
    ]


def _by_dmax_frame(summary: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for dmax_str, item in sorted(summary.get("benchmark_by_dmax", {}).items(), key=lambda x: int(x[0])):
        localization = item.get("localization", {})
        rows.append(
            {
                "true_dmax": int(dmax_str),
                "n_segments": int(item.get("n_segments", 0) or 0),
                "n_rows": int(item.get("n_rows", 0) or 0),
                "n_positive_rows": int(item.get("n_positive_rows", 0) or 0),
                "block_in_mae": float(item.get("block_in_expected_lag_mae", np.nan)),
                "auprc": float(localization.get("auprc", np.nan)),
                "best_f1": float(localization.get("best_f1", np.nan)),
                "far": float(item.get("block_out_false_alarm_rate", np.nan)),
                "zero_bias": float(item.get("mean_pred_expected_lag_when_true_zero", np.nan)),
            }
        )
    return pd.DataFrame(rows)


def _style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(CARD_BG)
    ax.grid(axis="y", color=GRID, alpha=0.6, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9ca3af")
    ax.spines["bottom"].set_color("#9ca3af")
    ax.tick_params(colors=TEXT, labelsize=10)


def _draw_card(ax: plt.Axes, payload: Dict[str, str]) -> None:
    color = payload["color"]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    card = FancyBboxPatch(
        (0.02, 0.05),
        0.96,
        0.90,
        boxstyle="round,pad=0.015,rounding_size=0.045",
        linewidth=1.5,
        edgecolor=color,
        facecolor=CARD_BG,
    )
    ax.add_patch(card)
    ax.add_patch(
        FancyBboxPatch(
            (0.05, 0.81),
            0.28,
            0.09,
            boxstyle="round,pad=0.01,rounding_size=0.03",
            linewidth=0.0,
            facecolor=color,
            alpha=0.14,
        )
    )
    ax.text(0.07, 0.845, payload["group"], fontsize=8.5, fontweight="bold", color=color, va="center")
    ax.text(0.07, 0.61, payload["value"], fontsize=19, fontweight="bold", color=TEXT, va="center")
    ax.text(0.07, 0.36, payload["label"], fontsize=10.5, color=TEXT, va="center")
    ax.text(0.07, 0.16, payload["note"], fontsize=8.5, color=MUTED, va="center")


def _plot_support(ax: plt.Axes, by_dmax: pd.DataFrame) -> None:
    _style_axis(ax)
    x = np.arange(len(by_dmax), dtype=np.float64)
    labels = [f"d{int(v)}" for v in by_dmax["true_dmax"]]
    rows = by_dmax["n_rows"].to_numpy(dtype=np.float64)
    positives = by_dmax["n_positive_rows"].to_numpy(dtype=np.float64)

    ax.bar(x, rows, width=0.62, color="#cbd5e1", edgecolor="#94a3b8")
    ax.plot(x, positives, color="#dc2626", marker="o", linewidth=2.2, label="positive rows")

    for idx, value in enumerate(rows):
        ax.text(idx, value + max(rows) * 0.03, f"{int(value)}", ha="center", va="bottom", fontsize=9, color=TEXT)
    for idx, value in enumerate(positives):
        ax.text(idx, value + max(rows) * 0.01, f"{int(value)}", ha="center", va="bottom", fontsize=8.5, color="#dc2626")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("rows")
    ax.set_title("Support by True dmax", fontsize=12, fontweight="bold", color=TEXT)
    ax.legend(frameon=False, loc="upper right")


def _plot_block_in(ax: plt.Axes, by_dmax: pd.DataFrame) -> None:
    _style_axis(ax)
    x = np.arange(len(by_dmax), dtype=np.float64)
    labels = [f"d{int(v)}" for v in by_dmax["true_dmax"]]
    values = by_dmax["block_in_mae"].to_numpy(dtype=np.float64)

    ax.bar(x, values, width=0.58, color="#fb923c", edgecolor="#ea580c")
    ax.plot(x, values, color="#9a3412", linewidth=2.0, alpha=0.7)

    for idx, value in enumerate(values):
        ax.text(idx, value + max(values) * 0.04, f"{value:.3f}", ha="center", va="bottom", fontsize=9, color=TEXT)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("MAE")
    ax.set_title("Block-in Expected-Lag MAE", fontsize=12, fontweight="bold", color=TEXT)


def _plot_localization(ax: plt.Axes, by_dmax: pd.DataFrame) -> None:
    _style_axis(ax)
    x = np.arange(len(by_dmax), dtype=np.float64)
    labels = [f"d{int(v)}" for v in by_dmax["true_dmax"]]
    width = 0.34
    auprc = by_dmax["auprc"].to_numpy(dtype=np.float64)
    best_f1 = by_dmax["best_f1"].to_numpy(dtype=np.float64)

    ax.bar(x - width / 2, auprc, width=width, color="#fde68a", edgecolor="#d97706", label="AUPRC")
    ax.bar(x + width / 2, best_f1, width=width, color="#fbbf24", edgecolor="#b45309", label="best-F1")

    for idx, value in enumerate(auprc):
        ax.text(idx - width / 2, value + max(best_f1) * 0.04, f"{value:.3f}", ha="center", va="bottom", fontsize=8.5)
    for idx, value in enumerate(best_f1):
        ax.text(idx + width / 2, value + max(best_f1) * 0.04, f"{value:.3f}", ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("score")
    ax.set_title("Localization by True dmax", fontsize=12, fontweight="bold", color=TEXT)
    ax.legend(frameon=False, loc="upper left")


def _plot_robustness(ax: plt.Axes, by_dmax: pd.DataFrame) -> None:
    _style_axis(ax)
    x = np.arange(len(by_dmax), dtype=np.float64)
    labels = [f"d{int(v)}" for v in by_dmax["true_dmax"]]
    far = by_dmax["far"].to_numpy(dtype=np.float64)
    bias = by_dmax["zero_bias"].to_numpy(dtype=np.float64)

    ax.bar(x, far, width=0.56, color="#86efac", edgecolor="#059669", label="FAR")
    for idx, value in enumerate(far):
        ax.text(idx, value + max(far) * 0.04, f"{value:.3f}", ha="center", va="bottom", fontsize=8.5, color=TEXT)

    ax2 = ax.twinx()
    ax2.plot(x, bias, color="#7c3aed", marker="o", linewidth=2.2, label="E[d_hat|d=0]")
    for idx, value in enumerate(bias):
        ax2.text(idx, value + max(bias) * 0.025, f"{value:.3f}", ha="center", va="bottom", fontsize=8.5, color="#6d28d9")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("FAR")
    ax2.set_ylabel("E[d_hat|d=0]", color="#6d28d9")
    ax2.tick_params(axis="y", colors="#6d28d9", labelsize=10)
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["right"].set_color("#a78bfa")
    ax.set_title("False Alarms and Zero-Lag Bias", fontsize=12, fontweight="bold", color=TEXT)

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, frameon=False, loc="upper left")


def _render_empty_note(ax: plt.Axes, message: str) -> None:
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=11, color=MUTED)


def main() -> None:
    args = parse_args()
    summary = _load_summary(args.summary)
    by_dmax = _by_dmax_frame(summary)
    cards = _metric_cards(summary)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(15.5, 11.0), facecolor=BG)
    outer = GridSpec(3, 4, figure=fig, height_ratios=[1.35, 1.0, 1.0], hspace=0.52, wspace=0.32)
    card_grid = outer[0, :].subgridspec(2, 4, hspace=0.25, wspace=0.22)

    for idx, payload in enumerate(cards):
        ax = fig.add_subplot(card_grid[idx // 4, idx % 4])
        _draw_card(ax, payload)

    if by_dmax.empty:
        ax = fig.add_subplot(outer[1:, :])
        _render_empty_note(ax, "No by-dmax benchmark data found in summary.")
    else:
        ax_support = fig.add_subplot(outer[1, :2])
        ax_block = fig.add_subplot(outer[1, 2:])
        ax_local = fig.add_subplot(outer[2, :2])
        ax_robust = fig.add_subplot(outer[2, 2:])
        _plot_support(ax_support, by_dmax)
        _plot_block_in(ax_block, by_dmax)
        _plot_localization(ax_local, by_dmax)
        _plot_robustness(ax_robust, by_dmax)

    fig.suptitle(args.title, fontsize=20, fontweight="bold", color=TEXT, y=0.985)
    fig.text(
        0.012,
        0.015,
        f"Source: {args.summary}",
        fontsize=8.5,
        color=MUTED,
        ha="left",
    )

    png_path = output_dir / "benchmark_dashboard.png"
    svg_path = output_dir / "benchmark_dashboard.svg"
    fig.savefig(png_path, dpi=220, bbox_inches="tight", facecolor=BG)
    fig.savefig(svg_path, bbox_inches="tight", facecolor=BG)
    plt.close(fig)

    print(f"Wrote {png_path}")
    print(f"Wrote {svg_path}")


if __name__ == "__main__":
    main()
