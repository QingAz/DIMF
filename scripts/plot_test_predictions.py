#!/usr/bin/env python3

import argparse
import csv
import os
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np


TIME_FORMAT = "%Y-%m-%d %H:%M"


def parse_args():
    parser = argparse.ArgumentParser(description="Plot DIMF test predictions against ground truth.")
    parser.add_argument("--predictions", type=Path, required=True, help="Path to test_pred_vs_true.csv")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for prediction plots. Defaults to <predictions parent>/plots",
    )
    parser.add_argument(
        "--title-prefix",
        type=str,
        default="",
        help="Optional text prefix added to plot titles.",
    )
    return parser.parse_args()


def _absolute_path(path):
    return Path(os.path.abspath(str(path)))


def _title(prefix, base_title):
    return ("%s | %s" % (prefix, base_title)) if prefix else base_title


def _read_prediction_csv(csv_path):
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = reader.fieldnames or []
    if not rows:
        raise ValueError("Prediction CSV is empty: %s" % csv_path)
    # 第 5 节修改：预测结果必须显式区分输入时刻 t 和目标时刻 t+H。
    required_cols = ["InputTimeStamp", "TargetTimeStamp", "y_true", "y_pred"]
    missing_cols = [name for name in required_cols if name not in fields]
    if missing_cols:
        raise ValueError("Missing columns %s in %s" % (missing_cols, csv_path))
    return rows


def _extract_arrays(rows):
    input_timestamps = [datetime.strptime(row["InputTimeStamp"], TIME_FORMAT) for row in rows]
    target_timestamps = [datetime.strptime(row["TargetTimeStamp"], TIME_FORMAT) for row in rows]
    y_true = np.asarray([float(row["y_true"]) for row in rows], dtype=np.float64)
    y_pred = np.asarray([float(row["y_pred"]) for row in rows], dtype=np.float64)
    return input_timestamps, target_timestamps, y_true, y_pred


def _setup_time_axis(ax, timestamps):
    locator = mdates.AutoDateLocator(minticks=4, maxticks=9)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.set_xlim(timestamps[0], timestamps[-1])


def _save_overlay_plot(output_dir, target_timestamps, y_true, y_pred, title_prefix):
    fig, ax = plt.subplots(1, 1, figsize=(14, 4.8))
    x_axis = target_timestamps

    ax.plot(x_axis, y_true, label="True", linewidth=2.0, color="#1f77b4")
    ax.plot(x_axis, y_pred, label="Pred", linewidth=1.6, color="#d62728", alpha=0.9)
    ax.set_ylabel("y(t+H)")
    ax.grid(alpha=0.25)
    ax.set_title(_title(title_prefix, "Test prediction over time"))
    ax.legend(loc="upper right")

    _setup_time_axis(ax, target_timestamps)
    ax.set_xlabel("Target time t+H")

    fig.tight_layout()
    out_path = output_dir / "test_prediction_overlay.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _save_scatter_plot(output_dir, y_true, y_pred, title_prefix):
    fig, ax = plt.subplots(1, 1, figsize=(5.6, 4.8))
    low = min(y_true.min(), y_pred.min())
    high = max(y_true.max(), y_pred.max())
    ax.scatter(y_true, y_pred, s=12, alpha=0.3, color="#2ca02c", edgecolors="none")
    ax.plot([low, high], [low, high], color="black", linestyle="--", linewidth=1.1)
    ax.set_xlabel("True y(t+H)")
    ax.set_ylabel("Pred y(t+H)")
    ax.grid(alpha=0.2)
    ax.set_title(_title(title_prefix, "Prediction scatter"))

    fig.tight_layout()
    out_path = output_dir / "test_prediction_scatter.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _save_residual_plot(output_dir, target_timestamps, y_true, y_pred, title_prefix):
    residual = y_pred - y_true
    fig, ax = plt.subplots(1, 1, figsize=(14, 4.8))
    x_axis = target_timestamps

    ax.plot(x_axis, residual, linewidth=1.3, color="#9467bd")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_ylabel("Residual")
    ax.grid(alpha=0.25)
    ax.set_title(_title(title_prefix, "Prediction residual over time"))

    _setup_time_axis(ax, target_timestamps)
    ax.set_xlabel("Target time t+H")

    fig.tight_layout()
    out_path = output_dir / "test_prediction_residual.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main():
    args = parse_args()
    predictions_path = _absolute_path(args.predictions)
    output_dir = _absolute_path(args.output_dir) if args.output_dir is not None else predictions_path.parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_prediction_csv(predictions_path)
    input_timestamps, target_timestamps, y_true, y_pred = _extract_arrays(rows)

    saved_paths = [
        _save_overlay_plot(output_dir, target_timestamps, y_true, y_pred, args.title_prefix),
        _save_scatter_plot(output_dir, y_true, y_pred, args.title_prefix),
        _save_residual_plot(output_dir, target_timestamps, y_true, y_pred, args.title_prefix),
    ]

    print("Matched samples: %d" % y_true.shape[0])
    print(
        "Input window range: %s -> %s | Target range: %s -> %s"
        % (
            input_timestamps[0].strftime(TIME_FORMAT),
            input_timestamps[-1].strftime(TIME_FORMAT),
            target_timestamps[0].strftime(TIME_FORMAT),
            target_timestamps[-1].strftime(TIME_FORMAT),
        )
    )
    for path in saved_paths:
        print("Saved: %s" % path)


if __name__ == "__main__":
    main()
