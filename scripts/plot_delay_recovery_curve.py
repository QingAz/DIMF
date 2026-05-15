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
    parser = argparse.ArgumentParser(description="Plot predicted vs injected lag curves.")
    parser.add_argument("--comparison", type=Path, required=True, help="Path to delay_recovery_comparison.csv")
    parser.add_argument("--edge", type=str, default="stage1_to_stage2", help="Edge to plot")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for plots. Defaults to <comparison parent>/plots",
    )
    parser.add_argument("--max-points", type=int, default=2000, help="Downsample to at most this many points")
    return parser.parse_args()


def _absolute_path(path):
    return Path(os.path.abspath(str(path)))


def _read_rows(path, edge):
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = [row for row in csv.DictReader(f) if row.get("edge") == edge]
    if not rows:
        raise ValueError("No rows found for edge %s in %s" % (edge, path))
    return rows


def _downsample(values, max_points):
    if len(values[0]) <= max_points:
        return values
    idx = np.linspace(0, len(values[0]) - 1, max_points, dtype=np.int64)
    return [np.asarray(value)[idx] for value in values]


def main():
    args = parse_args()
    comparison_path = _absolute_path(args.comparison)
    output_dir = _absolute_path(args.output_dir) if args.output_dir is not None else comparison_path.parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_rows(comparison_path, args.edge)
    timestamps = [datetime.strptime(row["TimeStamp"], TIME_FORMAT) for row in rows]
    true_expected = np.asarray([float(row["true_expected_lag"]) for row in rows], dtype=np.float64)
    pred_expected = np.asarray([float(row["pred_expected_lag"]) for row in rows], dtype=np.float64)
    true_argmax = np.asarray([float(row["true_argmax_lag"]) for row in rows], dtype=np.float64)
    pred_argmax = np.asarray([float(row["pred_argmax_lag"]) for row in rows], dtype=np.float64)

    timestamps, true_expected, pred_expected, true_argmax, pred_argmax = _downsample(
        [timestamps, true_expected, pred_expected, true_argmax, pred_argmax],
        args.max_points,
    )

    fig, ax = plt.subplots(1, 1, figsize=(14, 5.2))
    ax.plot(timestamps, true_expected, label="Injected expected lag", color="#1f77b4", linewidth=1.8)
    ax.plot(timestamps, pred_expected, label="Predicted expected lag", color="#ff7f0e", linewidth=1.6)
    ax.plot(timestamps, true_argmax, label="Injected argmax lag", color="#1f77b4", linewidth=0.9, linestyle="--", alpha=0.55)
    ax.plot(timestamps, pred_argmax, label="Predicted argmax lag", color="#ff7f0e", linewidth=0.9, linestyle="--", alpha=0.55)
    ax.set_title("%s lag recovery" % args.edge)
    ax.set_xlabel("Time")
    ax.set_ylabel("Lag step")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=9))
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.set_xlim(timestamps[0], timestamps[-1])
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.legend(loc="upper right")
    fig.tight_layout()

    out_path = output_dir / ("%s_delay_recovery_curve.png" % args.edge)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print("Saved: %s" % out_path)


if __name__ == "__main__":
    main()
