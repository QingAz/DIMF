#!/usr/bin/env python3

import argparse
import csv
import math
import os
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np


TIME_FORMAT = "%Y-%m-%d %H:%M"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot synthetic-delay recovery results from truth and DIMF delay estimates."
    )
    parser.add_argument("--estimates", type=Path, required=True, help="Path to test_delay_estimates.csv")
    parser.add_argument("--truth", type=Path, required=True, help="Path to delay_truth.csv")
    parser.add_argument(
        "--edge",
        type=str,
        default=None,
        help="Target edge, e.g. stage1_to_stage2. If omitted, infer the common edge from the two CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated figures. Defaults to <estimates parent>/plots",
    )
    parser.add_argument(
        "--max-heatmap-points",
        type=int,
        default=1200,
        help="Downsample heatmaps to at most this many time points for readability.",
    )
    parser.add_argument(
        "--title-prefix",
        type=str,
        default="",
        help="Optional text prefix added to plot titles.",
    )
    parser.add_argument(
        "--raw-reference",
        type=Path,
        default=None,
        help="Optional raw pre-interpolation CSV used to mark long interpolation spans in gray.",
    )
    parser.add_argument(
        "--interval-min",
        type=int,
        default=15,
        help="Sampling interval in minutes used when deriving raw gap spans.",
    )
    parser.add_argument(
        "--long-gap-min-slots",
        type=int,
        default=8,
        help="Shade raw gaps with at least this many missing slots.",
    )
    return parser.parse_args()


def _absolute_path(path):
    return Path(os.path.abspath(str(path)))


def _read_csv(csv_path):
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames or [], list(reader)


def _parse_timestamp(raw_value):
    return datetime.strptime(raw_value, TIME_FORMAT)


def _discover_edges(fieldnames, marker):
    suffix = "_%s_expected_lag" % marker
    return sorted([name[:-len(suffix)] for name in fieldnames if name.endswith(suffix)])


def _infer_edge(estimate_fields, truth_fields, requested_edge):
    common_edges = sorted(set(_discover_edges(estimate_fields, "pred")) & set(_discover_edges(truth_fields, "true")))
    if requested_edge is not None:
        if requested_edge not in common_edges:
            raise ValueError("Requested edge %s is not shared by the estimate/truth CSVs" % requested_edge)
        return requested_edge
    if len(common_edges) != 1:
        raise ValueError("Expected exactly one common edge, found: %s" % ", ".join(common_edges))
    return common_edges[0]


def _distribution_from_row(row, edge, marker):
    prefix = "%s_%s_pi_lag" % (edge, marker)
    cols = sorted(
        [key for key in row.keys() if key.startswith(prefix)],
        key=lambda name: int(name.split("lag")[-1]),
    )
    return np.asarray([float(row[col]) for col in cols], dtype=np.float64)


def _kl_divergence(p, q):
    total = 0.0
    for p_i, q_i in zip(p, q):
        if p_i <= 0.0:
            continue
        total += p_i * math.log(p_i / max(q_i, 1e-12))
    return total


def _js_divergence(p, q):
    mid = 0.5 * (p + q)
    return 0.5 * _kl_divergence(p, mid) + 0.5 * _kl_divergence(q, mid)


def _entropy(values):
    return -sum(value * math.log(max(value, 1e-12)) for value in values)


def _load_long_gap_spans(raw_reference_path, interval_min, long_gap_min_slots):
    if raw_reference_path is None:
        return []

    _, raw_rows = _read_csv(raw_reference_path)
    timestamps = [_parse_timestamp(row["TimeStamp"]) for row in raw_rows]
    interval = timedelta(minutes=interval_min)
    spans = []

    for prev_ts, cur_ts in zip(timestamps, timestamps[1:]):
        delta_minutes = int((cur_ts - prev_ts).total_seconds() // 60)
        if delta_minutes <= interval_min or delta_minutes % interval_min != 0:
            continue
        missing_slots = delta_minutes // interval_min - 1
        if missing_slots < long_gap_min_slots:
            continue
        spans.append(
            {
                "start": prev_ts + interval,
                "end": cur_ts - interval,
                "missing_slots": missing_slots,
            }
        )
    return spans


def _prepare_joined_series(estimate_rows, truth_rows, edge):
    truth_by_time = {row["TimeStamp"]: row for row in truth_rows}
    joined = []
    for est_row in estimate_rows:
        timestamp = est_row["TimeStamp"]
        truth_row = truth_by_time.get(timestamp)
        if truth_row is None:
            continue
        true_pi = _distribution_from_row(truth_row, edge, "true")
        pred_pi = _distribution_from_row(est_row, edge, "pred")
        if true_pi.shape != pred_pi.shape:
            raise ValueError("Lag support mismatch for %s at %s" % (edge, timestamp))

        joined.append(
            {
                "timestamp_str": timestamp,
                "timestamp": datetime.strptime(timestamp, TIME_FORMAT),
                "true_expected": float(truth_row["%s_true_expected_lag" % edge]),
                "pred_expected": float(est_row["%s_pred_expected_lag" % edge]),
                "true_argmax": int(float(truth_row["%s_true_argmax_lag" % edge])),
                "pred_argmax": int(float(est_row["%s_pred_argmax_lag" % edge])),
                "true_pi": true_pi,
                "pred_pi": pred_pi,
                "js_divergence": _js_divergence(true_pi, pred_pi),
                "pred_entropy": _entropy(pred_pi),
            }
        )

    if not joined:
        raise ValueError("No overlapping timestamps were found between the estimate and truth CSVs")
    return joined


def _as_arrays(joined_rows):
    timestamps = [row["timestamp"] for row in joined_rows]
    true_expected = np.asarray([row["true_expected"] for row in joined_rows], dtype=np.float64)
    pred_expected = np.asarray([row["pred_expected"] for row in joined_rows], dtype=np.float64)
    true_argmax = np.asarray([row["true_argmax"] for row in joined_rows], dtype=np.float64)
    pred_argmax = np.asarray([row["pred_argmax"] for row in joined_rows], dtype=np.float64)
    true_pi = np.stack([row["true_pi"] for row in joined_rows], axis=0)
    pred_pi = np.stack([row["pred_pi"] for row in joined_rows], axis=0)
    js_values = np.asarray([row["js_divergence"] for row in joined_rows], dtype=np.float64)
    entropy_values = np.asarray([row["pred_entropy"] for row in joined_rows], dtype=np.float64)
    return timestamps, true_expected, pred_expected, true_argmax, pred_argmax, true_pi, pred_pi, js_values, entropy_values


def _title(prefix, base_title):
    return ("%s | %s" % (prefix, base_title)) if prefix else base_title


def _setup_time_axis(ax, timestamps):
    locator = mdates.AutoDateLocator(minticks=4, maxticks=9)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.set_xlim(timestamps[0], timestamps[-1])


def _shade_long_gap_spans(ax, long_gap_spans):
    if not long_gap_spans:
        return
    first = True
    for span in long_gap_spans:
        ax.axvspan(
            span["start"],
            span["end"],
            color="#9e9e9e",
            alpha=0.18,
            lw=0.0,
            label="Long interpolated span" if first else None,
        )
        first = False


def _save_expected_overlay(output_dir, timestamps, true_expected, pred_expected, title_prefix, edge, long_gap_spans):
    fig, ax = plt.subplots(figsize=(14, 5))
    _shade_long_gap_spans(ax, long_gap_spans)
    ax.plot(timestamps, true_expected, label="True expected lag", linewidth=2.2, color="#1f77b4")
    ax.plot(timestamps, pred_expected, label="Pred expected lag", linewidth=1.8, color="#d62728", alpha=0.9)
    ax.set_ylabel("Lag")
    ax.set_title(_title(title_prefix, "%s expected lag over time" % edge))
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    _setup_time_axis(ax, timestamps)
    fig.tight_layout()
    out_path = output_dir / "delay_overlay_expected.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _save_argmax_overlay(output_dir, timestamps, true_argmax, pred_argmax, title_prefix, edge, long_gap_spans):
    fig, ax = plt.subplots(figsize=(14, 5))
    _shade_long_gap_spans(ax, long_gap_spans)
    ax.step(timestamps, true_argmax, where="post", label="True argmax lag", linewidth=2.0, color="#1f77b4")
    ax.step(
        timestamps,
        pred_argmax,
        where="post",
        label="Pred argmax lag",
        linewidth=1.5,
        color="#d62728",
        alpha=0.9,
    )
    ax.set_ylabel("Lag")
    ax.set_title(_title(title_prefix, "%s argmax lag over time" % edge))
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    _setup_time_axis(ax, timestamps)
    fig.tight_layout()
    out_path = output_dir / "delay_overlay_argmax.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _save_error_plot(output_dir, timestamps, true_expected, pred_expected, true_argmax, pred_argmax, title_prefix, edge, long_gap_spans):
    expected_error = pred_expected - true_expected
    argmax_error = pred_argmax - true_argmax

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    _shade_long_gap_spans(axes[0], long_gap_spans)
    axes[0].plot(timestamps, expected_error, color="#2ca02c", linewidth=1.6)
    axes[0].axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
    axes[0].set_ylabel("Pred - True")
    axes[0].set_title(_title(title_prefix, "%s lag error over time" % edge))
    axes[0].grid(alpha=0.25)

    _shade_long_gap_spans(axes[1], long_gap_spans)
    axes[1].step(timestamps, argmax_error, where="post", color="#9467bd", linewidth=1.4)
    axes[1].axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
    axes[1].set_ylabel("Argmax error")
    axes[1].grid(alpha=0.25)
    _setup_time_axis(axes[1], timestamps)

    fig.tight_layout()
    out_path = output_dir / "delay_error_over_time.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _save_scatter_plot(output_dir, true_expected, pred_expected, true_argmax, pred_argmax, title_prefix, edge):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    exp_min = min(true_expected.min(), pred_expected.min())
    exp_max = max(true_expected.max(), pred_expected.max())
    axes[0].scatter(true_expected, pred_expected, s=10, alpha=0.25, color="#d62728", edgecolors="none")
    axes[0].plot([exp_min, exp_max], [exp_min, exp_max], color="black", linestyle="--", linewidth=1.2)
    axes[0].set_xlabel("True expected lag")
    axes[0].set_ylabel("Pred expected lag")
    axes[0].set_title("Expected lag")
    axes[0].grid(alpha=0.2)

    arg_min = min(true_argmax.min(), pred_argmax.min())
    arg_max = max(true_argmax.max(), pred_argmax.max())
    axes[1].scatter(true_argmax, pred_argmax, s=10, alpha=0.25, color="#1f77b4", edgecolors="none")
    axes[1].plot([arg_min, arg_max], [arg_min, arg_max], color="black", linestyle="--", linewidth=1.2)
    axes[1].set_xlabel("True argmax lag")
    axes[1].set_ylabel("Pred argmax lag")
    axes[1].set_title("Argmax lag")
    axes[1].grid(alpha=0.2)

    fig.suptitle(_title(title_prefix, "%s lag scatter" % edge))
    fig.tight_layout()
    out_path = output_dir / "delay_scatter.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _downsample_for_heatmap(timestamps, true_pi, pred_pi, max_points):
    n_points = len(timestamps)
    if n_points <= max_points:
        return timestamps, true_pi, pred_pi
    stride = int(math.ceil(float(n_points) / float(max_points)))
    idx = np.arange(0, n_points, stride)
    timestamps_ds = [timestamps[i] for i in idx]
    return timestamps_ds, true_pi[idx], pred_pi[idx]


def _save_heatmap(output_dir, timestamps, true_pi, pred_pi, title_prefix, edge, max_heatmap_points, long_gap_spans):
    timestamps_ds, true_pi_ds, pred_pi_ds = _downsample_for_heatmap(timestamps, true_pi, pred_pi, max_heatmap_points)
    x0 = mdates.date2num(timestamps_ds[0])
    x1 = mdates.date2num(timestamps_ds[-1])
    n_lags = true_pi_ds.shape[1]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    vmax = max(float(true_pi_ds.max()), float(pred_pi_ds.max()), 1e-6)
    for ax, matrix, subtitle in [
        (axes[0], true_pi_ds, "True delay distribution"),
        (axes[1], pred_pi_ds, "Predicted delay distribution"),
    ]:
        im = ax.imshow(
            matrix.T,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            extent=[x0, x1, 0, n_lags - 1],
            vmin=0.0,
            vmax=vmax,
            cmap="viridis",
        )
        ax.set_ylabel("Lag")
        ax.set_title(subtitle)
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.015)
        _shade_long_gap_spans(ax, long_gap_spans)

    axes[0].set_title(_title(title_prefix, "%s true delay distribution" % edge))
    axes[1].set_title("Predicted delay distribution")

    locator = mdates.AutoDateLocator(minticks=4, maxticks=9)
    formatter = mdates.ConciseDateFormatter(locator)
    axes[1].xaxis.set_major_locator(locator)
    axes[1].xaxis.set_major_formatter(formatter)
    axes[1].set_xlim(timestamps_ds[0], timestamps_ds[-1])

    fig.tight_layout()
    out_path = output_dir / "delay_pi_heatmap_true_vs_pred.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _save_js_plot(output_dir, timestamps, js_values, title_prefix, edge, long_gap_spans):
    fig, ax = plt.subplots(figsize=(14, 4.5))
    _shade_long_gap_spans(ax, long_gap_spans)
    ax.plot(timestamps, js_values, color="#ff7f0e", linewidth=1.4)
    ax.set_ylabel("JS divergence")
    ax.set_title(_title(title_prefix, "%s distribution mismatch over time" % edge))
    ax.grid(alpha=0.25)
    _setup_time_axis(ax, timestamps)
    fig.tight_layout()
    out_path = output_dir / "delay_js_divergence_over_time.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _save_entropy_plot(output_dir, timestamps, entropy_values, title_prefix, edge, n_lags, long_gap_spans):
    fig, ax = plt.subplots(figsize=(14, 4.5))
    _shade_long_gap_spans(ax, long_gap_spans)
    ax.plot(timestamps, entropy_values, color="#8c564b", linewidth=1.4, label="Predicted pi entropy")
    ax.axhline(math.log(float(n_lags)), color="black", linestyle="--", linewidth=1.0, alpha=0.6, label="Max entropy")
    ax.set_ylabel("Entropy")
    ax.set_title(_title(title_prefix, "%s alignment uncertainty over time" % edge))
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    _setup_time_axis(ax, timestamps)
    fig.tight_layout()
    out_path = output_dir / "delay_entropy_over_time.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main():
    args = parse_args()
    estimates_path = _absolute_path(args.estimates)
    truth_path = _absolute_path(args.truth)
    raw_reference_path = _absolute_path(args.raw_reference) if args.raw_reference is not None else None
    output_dir = _absolute_path(args.output_dir) if args.output_dir is not None else estimates_path.parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    estimate_fields, estimate_rows = _read_csv(estimates_path)
    truth_fields, truth_rows = _read_csv(truth_path)
    edge = _infer_edge(estimate_fields, truth_fields, args.edge)
    long_gap_spans = _load_long_gap_spans(raw_reference_path, args.interval_min, args.long_gap_min_slots)

    joined_rows = _prepare_joined_series(estimate_rows, truth_rows, edge)
    (
        timestamps,
        true_expected,
        pred_expected,
        true_argmax,
        pred_argmax,
        true_pi,
        pred_pi,
        js_values,
        entropy_values,
    ) = _as_arrays(joined_rows)

    saved_paths = [
        _save_expected_overlay(output_dir, timestamps, true_expected, pred_expected, args.title_prefix, edge, long_gap_spans),
        _save_argmax_overlay(output_dir, timestamps, true_argmax, pred_argmax, args.title_prefix, edge, long_gap_spans),
        _save_error_plot(output_dir, timestamps, true_expected, pred_expected, true_argmax, pred_argmax, args.title_prefix, edge, long_gap_spans),
        _save_scatter_plot(output_dir, true_expected, pred_expected, true_argmax, pred_argmax, args.title_prefix, edge),
        _save_heatmap(output_dir, timestamps, true_pi, pred_pi, args.title_prefix, edge, args.max_heatmap_points, long_gap_spans),
        _save_js_plot(output_dir, timestamps, js_values, args.title_prefix, edge, long_gap_spans),
        _save_entropy_plot(output_dir, timestamps, entropy_values, args.title_prefix, edge, pred_pi.shape[1], long_gap_spans),
    ]

    print("Edge: %s" % edge)
    print("Matched timestamps: %d" % len(joined_rows))
    print("Long gap spans shaded: %d" % len(long_gap_spans))
    for path in saved_paths:
        print("Saved: %s" % path)


if __name__ == "__main__":
    main()
