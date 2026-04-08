#!/usr/bin/env python3

import argparse
import csv
import json
import math
import os
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


METHODS = [
    {
        "name": "piecewise",
        "label": "Piecewise Constant",
        "category": "01_piecewise_constant",
        "category_label": "Piecewise Constant Delay",
        "output_dir": "outputs/stage12_piecewise_delay",
    },
    {
        "name": "linear",
        "label": "Linear Drift",
        "category": "02_smoothly_varying",
        "category_label": "Smoothly Varying Delay",
        "output_dir": "outputs/stage12_linear_delay",
    },
    {
        "name": "sinusoidal",
        "label": "Sinusoidal",
        "category": "02_smoothly_varying",
        "category_label": "Smoothly Varying Delay",
        "output_dir": "outputs/stage12_sinusoidal_delay",
    },
    {
        "name": "bimodal",
        "label": "Bimodal",
        "category": "03_multimodal",
        "category_label": "Multimodal / Distributional Delay",
        "output_dir": "outputs/stage12_bimodal_delay",
    },
]

PLOT_FILES = [
    "delay_overlay_expected.png",
    "delay_overlay_argmax.png",
    "delay_error_over_time.png",
    "delay_entropy_over_time.png",
    "delay_pi_heatmap_true_vs_pred.png",
    "delay_js_divergence_over_time.png",
]

PLOT_TITLES = {
    "delay_overlay_expected.png": "Expected Lag",
    "delay_overlay_argmax.png": "Argmax Lag",
    "delay_error_over_time.png": "Lag Error",
    "delay_entropy_over_time.png": "Alignment Uncertainty",
    "delay_pi_heatmap_true_vs_pred.png": "Delay Distribution",
    "delay_js_divergence_over_time.png": "Distribution Mismatch",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Organize and summarize the four synthetic-delay visualization outputs."
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("outputs/delay_recovery_suite/delay_recovery_suite_summary.csv"),
        help="Suite-level CSV with delay recovery metrics.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/data_visualizations/delay_methods"),
        help="Destination directory for organized delay visualizations.",
    )
    return parser.parse_args()


def _absolute_path(path):
    return Path(os.path.abspath(str(path)))


def _read_summary(summary_csv):
    with summary_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    by_method = {}
    for method in METHODS:
        matching = [
            row for row in rows if row["config"].endswith("multistage_aligned_stage12_%s_delay.yaml" % method["name"])
        ]
        if not matching:
            raise ValueError("Missing summary row for method %s" % method["name"])
        by_method[method["name"]] = matching[0]
    return by_method


def _float(row, key):
    value = row.get(key)
    if value in (None, ""):
        return None
    return float(value)


def _copy_plot_set(project_root, method, target_dir):
    src_plot_dir = project_root / method["output_dir"] / "plots"
    copied = []
    for plot_name in PLOT_FILES:
        src = src_plot_dir / plot_name
        if not src.exists():
            raise FileNotFoundError("Missing source plot: %s" % src)
        dst = target_dir / plot_name
        shutil.copy2(src, dst)
        copied.append(dst)
    return copied


def _metric_lines(row):
    outside_mae = _float(row, "outside_long_gap_expected_lag_mae")
    return [
        "Expected lag MAE: %.3f" % _float(row, "expected_lag_mae"),
        "Argmax accuracy: %.3f" % _float(row, "argmax_lag_accuracy"),
        "Mean JS divergence: %.3f" % _float(row, "mean_js_divergence"),
        "Prediction MAE: %.3f" % _float(row, "MAE"),
        "Outside long-gap MAE: %.3f" % outside_mae if outside_mae is not None else "Outside long-gap MAE: n/a",
        "Mean pred entropy: %.3f" % _float(row, "mean_pred_entropy"),
    ]


def _save_method_panel(target_dir, method_label, row):
    fig = plt.figure(figsize=(16, 13))
    gs = fig.add_gridspec(
        4,
        2,
        height_ratios=[0.18, 1.0, 1.0, 1.0],
        hspace=0.16,
        wspace=0.06,
    )

    header_ax = fig.add_subplot(gs[0, :])
    header_ax.axis("off")
    metric_text = "\n".join(_metric_lines(row))
    header_ax.text(0.01, 0.88, method_label, fontsize=18, fontweight="bold", va="top")
    header_ax.text(
        0.99,
        0.88,
        metric_text,
        fontsize=10.5,
        va="top",
        ha="right",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#f6f6f6", "edgecolor": "#d0d0d0"},
    )

    for idx, plot_name in enumerate(PLOT_FILES):
        row_idx = idx // 2 + 1
        col_idx = idx % 2
        ax = fig.add_subplot(gs[row_idx, col_idx])
        image = mpimg.imread(target_dir / plot_name)
        ax.imshow(image)
        ax.set_title(PLOT_TITLES[plot_name], fontsize=11)
        ax.axis("off")

    fig.tight_layout()
    out_path = target_dir / "delay_method_panel.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _save_suite_metric_comparison(output_dir, rows_by_method):
    ordered_rows = [rows_by_method[method["name"]] for method in METHODS]
    labels = [method["label"] for method in METHODS]
    x = np.arange(len(labels))

    expected_mae = np.asarray([_float(row, "expected_lag_mae") for row in ordered_rows], dtype=np.float64)
    outside_mae = np.asarray([_float(row, "outside_long_gap_expected_lag_mae") for row in ordered_rows], dtype=np.float64)
    argmax_acc = np.asarray([_float(row, "argmax_lag_accuracy") for row in ordered_rows], dtype=np.float64)
    mean_js = np.asarray([_float(row, "mean_js_divergence") for row in ordered_rows], dtype=np.float64)
    pred_mae = np.asarray([_float(row, "MAE") for row in ordered_rows], dtype=np.float64)
    entropy = np.asarray([_float(row, "mean_pred_entropy") for row in ordered_rows], dtype=np.float64)

    fig, axes = plt.subplots(3, 2, figsize=(14, 11))
    axes = axes.reshape(-1)
    bar_color = "#4c78a8"

    metric_specs = [
        ("Expected Lag MAE", expected_mae, True),
        ("Outside Long-Gap MAE", outside_mae, True),
        ("Argmax Accuracy", argmax_acc, False),
        ("Mean JS Divergence", mean_js, True),
        ("Prediction MAE", pred_mae, True),
        ("Mean Pred Entropy", entropy, True),
    ]

    for ax, (title, values, lower_is_better) in zip(axes, metric_specs):
        colors = [bar_color] * len(values)
        best_idx = int(np.argmin(values) if lower_is_better else np.argmax(values))
        colors[best_idx] = "#f58518"
        ax.bar(x, values, color=colors, alpha=0.9)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=18, ha="right")
        ax.grid(axis="y", alpha=0.22)
        for idx, value in enumerate(values):
            ax.text(idx, value, "%.3f" % value, ha="center", va="bottom", fontsize=9)

    fig.suptitle("Synthetic Delay Recovery Comparison", fontsize=16, fontweight="bold")
    fig.tight_layout()
    out_path = output_dir / "delay_metric_comparison.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _save_category_overviews(output_dir, manifests):
    saved = []
    category_to_items = {}
    for item in manifests:
        category_to_items.setdefault(item["category"], []).append(item)

    for category, items in category_to_items.items():
        items = sorted(items, key=lambda item: item["method_name"])
        fig, axes = plt.subplots(len(items), 1, figsize=(16, 7.6 * len(items)))
        if len(items) == 1:
            axes = [axes]

        for ax, item in zip(axes, items):
            image = mpimg.imread(Path(item["panel_path"]))
            ax.imshow(image)
            ax.set_title(item["method_label"], fontsize=13)
            ax.axis("off")

        category_label = items[0]["category_label"]
        fig.suptitle(category_label, fontsize=17, fontweight="bold")
        fig.tight_layout()
        out_path = output_dir / category / "category_overview.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        saved.append(out_path)

    return saved


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    summary_csv = _absolute_path(args.summary_csv)
    output_dir = _absolute_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_by_method = _read_summary(summary_csv)
    manifest_items = []

    for method in METHODS:
        row = rows_by_method[method["name"]]
        method_dir = output_dir / method["category"] / method["name"]
        method_dir.mkdir(parents=True, exist_ok=True)
        copied_paths = _copy_plot_set(project_root, method, method_dir)
        panel_path = _save_method_panel(method_dir, method["label"], row)
        manifest_items.append(
            {
                "method_name": method["name"],
                "method_label": method["label"],
                "category": method["category"],
                "category_label": method["category_label"],
                "source_output_dir": str(project_root / method["output_dir"]),
                "copied_plot_paths": [str(path) for path in copied_paths],
                "panel_path": str(panel_path),
                "metrics": {
                    "expected_lag_mae": _float(row, "expected_lag_mae"),
                    "outside_long_gap_expected_lag_mae": _float(row, "outside_long_gap_expected_lag_mae"),
                    "argmax_lag_accuracy": _float(row, "argmax_lag_accuracy"),
                    "mean_js_divergence": _float(row, "mean_js_divergence"),
                    "mean_pred_entropy": _float(row, "mean_pred_entropy"),
                    "prediction_mae": _float(row, "MAE"),
                },
            }
        )

    suite_plot = _save_suite_metric_comparison(output_dir, rows_by_method)
    category_overviews = _save_category_overviews(output_dir, manifest_items)

    manifest = {
        "summary_csv": summary_csv.as_posix(),
        "output_dir": output_dir.as_posix(),
        "methods": manifest_items,
        "suite_metric_plot": suite_plot.as_posix(),
        "category_overviews": [path.as_posix() for path in category_overviews],
    }
    manifest_path = output_dir / "delay_method_gallery_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("Organized methods: %d" % len(manifest_items))
    print("Saved manifest: %s" % manifest_path)
    print("Saved suite metric comparison: %s" % suite_plot)
    for path in category_overviews:
        print("Saved: %s" % path)


if __name__ == "__main__":
    main()
