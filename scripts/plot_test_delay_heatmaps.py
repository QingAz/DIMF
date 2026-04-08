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
    parser = argparse.ArgumentParser(description="Plot DIMF test delay heatmaps from test_delay_estimates.csv")
    parser.add_argument("--estimates", type=Path, required=True, help="Path to test_delay_estimates.csv")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for delay heatmaps. Defaults to <estimates parent>/plots",
    )
    parser.add_argument(
        "--edge",
        type=str,
        default=None,
        help="Optional edge name, such as stage1_to_stage2. If omitted, plot all edges.",
    )
    parser.add_argument(
        "--interval-min",
        type=int,
        default=15,
        help="Sampling interval in minutes used to label lag units.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=1200,
        help="Downsample to at most this many time points for readability.",
    )
    parser.add_argument(
        "--title-prefix",
        type=str,
        default="",
        help="Optional text prefix added to figure titles.",
    )
    return parser.parse_args()


def _absolute_path(path):
    return Path(os.path.abspath(str(path)))


def _read_csv(csv_path):
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = reader.fieldnames or []
    if not rows:
        raise ValueError("Delay estimate CSV is empty: %s" % csv_path)
    if "TimeStamp" not in fields:
        raise ValueError("Delay estimate CSV must contain TimeStamp column: %s" % csv_path)
    return fields, rows


def _discover_edges(fields):
    suffix = "_pred_expected_lag"
    return sorted(name[: -len(suffix)] for name in fields if name.endswith(suffix))


def _extract_pi_matrix(rows, edge):
    prefix = "%s_pred_pi_lag" % edge
    pi_cols = sorted(
        [name for name in rows[0].keys() if name.startswith(prefix)],
        key=lambda name: int(name.split("lag")[-1]),
    )
    if not pi_cols:
        raise ValueError("No lag distribution columns found for edge %s" % edge)

    timestamps = [datetime.strptime(row["TimeStamp"], TIME_FORMAT) for row in rows]
    pi = np.asarray([[float(row[col]) for col in pi_cols] for row in rows], dtype=np.float64)
    expected_lag = np.asarray([float(row["%s_pred_expected_lag" % edge]) for row in rows], dtype=np.float64)
    argmax_lag = np.asarray([float(row["%s_pred_argmax_lag" % edge]) for row in rows], dtype=np.float64)
    return timestamps, pi, expected_lag, argmax_lag


def _downsample_series(timestamps, pi, expected_lag, argmax_lag, max_points):
    if len(timestamps) <= max_points:
        return timestamps, pi, expected_lag, argmax_lag
    idx = np.linspace(0, len(timestamps) - 1, max_points, dtype=np.int64)
    timestamps_ds = [timestamps[i] for i in idx]
    return timestamps_ds, pi[idx], expected_lag[idx], argmax_lag[idx]


def _title(prefix, base_title):
    return ("%s | %s" % (prefix, base_title)) if prefix else base_title


def _setup_time_axis(ax, timestamps):
    locator = mdates.AutoDateLocator(minticks=4, maxticks=9)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.set_xlim(timestamps[0], timestamps[-1])


def _save_edge_heatmap(output_dir, edge, timestamps, pi, expected_lag, argmax_lag, interval_min, title_prefix):
    # 当前热力图严格对应 pi(l | t)，横轴是当前时刻 t，纵轴是离散 lag 步数。
    timestamps_num = mdates.date2num(timestamps)
    fig, ax = plt.subplots(1, 1, figsize=(14, 5.5))
    image = ax.imshow(
        pi.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[timestamps_num[0], timestamps_num[-1], -0.5, pi.shape[1] - 0.5],
        cmap="viridis",
    )
    ax.plot(timestamps, expected_lag, color="#ff7f0e", linewidth=1.8, label="Expected lag")
    ax.plot(timestamps, argmax_lag, color="#d62728", linewidth=1.2, linestyle="--", label="Argmax lag")
    ax.set_ylabel("Lag step (%d min each)" % interval_min)
    ax.set_xlabel("Current time t")
    ax.set_title(_title(title_prefix, "%s delay distribution heatmap" % edge))
    _setup_time_axis(ax, timestamps)
    ax.legend(loc="upper right")

    cbar = fig.colorbar(image, ax=ax, pad=0.02)
    cbar.set_label("pi(l | t)")

    fig.tight_layout()
    out_path = output_dir / ("%s_delay_heatmap.png" % edge)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main():
    args = parse_args()
    estimates_path = _absolute_path(args.estimates)
    output_dir = _absolute_path(args.output_dir) if args.output_dir is not None else estimates_path.parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    fields, rows = _read_csv(estimates_path)
    edges = _discover_edges(fields)
    if args.edge is not None:
        if args.edge not in edges:
            raise ValueError("Unknown edge %s; available edges: %s" % (args.edge, ", ".join(edges)))
        edges = [args.edge]

    saved_paths = []
    for edge in edges:
        timestamps, pi, expected_lag, argmax_lag = _extract_pi_matrix(rows, edge)
        timestamps, pi, expected_lag, argmax_lag = _downsample_series(
            timestamps,
            pi,
            expected_lag,
            argmax_lag,
            args.max_points,
        )
        saved_paths.append(
            _save_edge_heatmap(
                output_dir,
                edge,
                timestamps,
                pi,
                expected_lag,
                argmax_lag,
                args.interval_min,
                args.title_prefix,
            )
        )

    print("Plotted edges: %s" % ", ".join(edges))
    for path in saved_paths:
        print("Saved: %s" % path)


if __name__ == "__main__":
    main()
