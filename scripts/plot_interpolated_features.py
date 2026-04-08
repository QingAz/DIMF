#!/usr/bin/env python3

import argparse
import csv
import json
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
        description="Plot every LiquidSugar feature and highlight timestamps filled by linear interpolation."
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=Path("data/LiquidSugar.csv"),
        help="Original irregular CSV used as the observation reference.",
    )
    parser.add_argument(
        "--interpolated",
        type=Path,
        default=Path("data/processed/LiquidSugar_linear_interpolated.csv"),
        help="Regularized CSV produced by linear interpolation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/data_visualizations/linear_interpolated_features"),
        help="Directory for generated figures and summary metadata.",
    )
    parser.add_argument("--time-col", default="TimeStamp", help="Timestamp column name.")
    parser.add_argument(
        "--interval-min",
        type=int,
        default=15,
        help="Expected regular sampling interval in minutes.",
    )
    parser.add_argument(
        "--overview-per-page",
        type=int,
        default=6,
        help="Number of features included in each overview page.",
    )
    return parser.parse_args()


def _absolute_path(path):
    return Path(os.path.abspath(str(path)))


def _parse_timestamp(raw_value):
    return datetime.strptime(raw_value, TIME_FORMAT)


def _read_csv(csv_path, time_col):
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if not fieldnames:
            raise ValueError("Missing header in %s" % csv_path)
        if time_col not in fieldnames:
            raise ValueError("Missing timestamp column %s in %s" % (time_col, csv_path))
        rows = list(reader)
    if not rows:
        raise ValueError("CSV is empty: %s" % csv_path)
    return fieldnames, rows


def _load_series(csv_path, time_col):
    fieldnames, rows = _read_csv(csv_path, time_col)
    feature_names = [name for name in fieldnames if name != time_col]
    timestamps = [_parse_timestamp(row[time_col]) for row in rows]
    values = {
        feature: np.asarray([float(row[feature]) for row in rows], dtype=np.float64)
        for feature in feature_names
    }
    return feature_names, timestamps, values


def _setup_time_axis(ax, timestamps):
    locator = mdates.AutoDateLocator(minticks=4, maxticks=9)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.set_xlim(timestamps[0], timestamps[-1])


def _contiguous_spans(mask, timestamps):
    spans = []
    start_idx = None
    for idx, is_interpolated in enumerate(mask):
        if is_interpolated and start_idx is None:
            start_idx = idx
        elif not is_interpolated and start_idx is not None:
            spans.append((start_idx, idx - 1))
            start_idx = None
    if start_idx is not None:
        spans.append((start_idx, len(mask) - 1))
    return [
        {
            "start_index": start_idx,
            "end_index": end_idx,
            "start_timestamp": timestamps[start_idx].strftime(TIME_FORMAT),
            "end_timestamp": timestamps[end_idx].strftime(TIME_FORMAT),
            "n_points": end_idx - start_idx + 1,
        }
        for start_idx, end_idx in spans
    ]


def _shade_spans(ax, spans, timestamps, half_interval):
    if not spans:
        return
    first = True
    for span in spans:
        start = timestamps[span["start_index"]] - half_interval
        end = timestamps[span["end_index"]] + half_interval
        ax.axvspan(
            start,
            end,
            color="#ffb74d",
            alpha=0.22,
            lw=0.0,
            label="Interpolated span" if first else None,
        )
        first = False


def _feature_group(feature_name):
    if feature_name.startswith("feed_"):
        return "feed"
    if feature_name.startswith("stage1_"):
        return "stage1"
    if feature_name.startswith("stage2_"):
        return "stage2"
    if feature_name.startswith("stage3_"):
        return "stage3"
    if feature_name.startswith("yield"):
        return "target"
    return "other"


def _save_feature_plot(output_dir, feature_name, timestamps, values, raw_mask, spans, title_prefix, half_interval):
    fig, ax = plt.subplots(figsize=(14, 5))
    _shade_spans(ax, spans, timestamps, half_interval)

    raw_values = np.where(raw_mask, values, np.nan)
    interp_values = np.where(raw_mask, np.nan, values)

    ax.plot(timestamps, values, color="#1f77b4", linewidth=1.35, alpha=0.9, label="Regularized series")
    ax.plot(
        timestamps,
        interp_values,
        color="#ff7f0e",
        linewidth=2.0,
        alpha=0.95,
        label="Interpolated points",
    )
    ax.scatter(
        np.asarray(timestamps)[raw_mask],
        raw_values[raw_mask],
        s=7,
        color="#0d47a1",
        alpha=0.6,
        label="Original observations",
        zorder=3,
    )

    ax.set_ylabel(feature_name)
    if title_prefix:
        ax.set_title("%s | %s" % (title_prefix, feature_name))
    else:
        ax.set_title(feature_name)
    ax.grid(alpha=0.24)
    ax.legend(loc="upper right")
    _setup_time_axis(ax, timestamps)

    fig.tight_layout()
    out_path = output_dir / ("%s.png" % feature_name)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _save_overview_pages(output_dir, feature_names, timestamps, interpolated_values, raw_mask, spans, per_page, title_prefix, half_interval):
    saved_paths = []
    total_pages = int(math.ceil(len(feature_names) / float(per_page)))
    x_raw = np.asarray(timestamps)[raw_mask]

    for page_idx in range(total_pages):
        page_features = feature_names[page_idx * per_page:(page_idx + 1) * per_page]
        fig, axes = plt.subplots(len(page_features), 1, figsize=(16, max(4.8, 3.1 * len(page_features))), sharex=True)
        if len(page_features) == 1:
            axes = [axes]

        for ax, feature_name in zip(axes, page_features):
            _shade_spans(ax, spans, timestamps, half_interval)
            values = interpolated_values[feature_name]
            raw_values = np.where(raw_mask, values, np.nan)
            interp_values = np.where(raw_mask, np.nan, values)

            ax.plot(timestamps, values, color="#1f77b4", linewidth=1.0, alpha=0.85)
            ax.plot(timestamps, interp_values, color="#ff7f0e", linewidth=1.5, alpha=0.95)
            ax.scatter(x_raw, raw_values[raw_mask], s=5, color="#0d47a1", alpha=0.45, zorder=3)
            ax.set_ylabel(feature_name, fontsize=9)
            ax.grid(alpha=0.18)

        if title_prefix:
            axes[0].set_title("%s | Feature Overview %d/%d" % (title_prefix, page_idx + 1, total_pages))
        else:
            axes[0].set_title("Feature Overview %d/%d" % (page_idx + 1, total_pages))

        _setup_time_axis(axes[-1], timestamps)
        fig.tight_layout()
        out_path = output_dir / ("overview_page_%02d.png" % (page_idx + 1))
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        saved_paths.append(out_path)

    return saved_paths


def main():
    args = parse_args()
    raw_path = _absolute_path(args.raw)
    interpolated_path = _absolute_path(args.interpolated)
    output_dir = _absolute_path(args.output_dir)
    by_feature_dir = output_dir / "by_feature"
    output_dir.mkdir(parents=True, exist_ok=True)
    by_feature_dir.mkdir(parents=True, exist_ok=True)

    raw_features, raw_timestamps, _ = _load_series(raw_path, args.time_col)
    interp_features, interp_timestamps, interp_values = _load_series(interpolated_path, args.time_col)
    if raw_features != interp_features:
        raise ValueError("Raw/interpolated feature columns do not match")

    raw_timestamp_set = set(raw_timestamps)
    raw_mask = np.asarray([timestamp in raw_timestamp_set for timestamp in interp_timestamps], dtype=bool)
    interpolated_mask = ~raw_mask
    spans = _contiguous_spans(interpolated_mask.tolist(), interp_timestamps)
    half_interval = timedelta(minutes=args.interval_min / 2.0)

    feature_paths = []
    for feature_name in interp_features:
        group_dir = by_feature_dir / _feature_group(feature_name)
        group_dir.mkdir(parents=True, exist_ok=True)
        feature_paths.append(
            _save_feature_plot(
                group_dir,
                feature_name,
                interp_timestamps,
                interp_values[feature_name],
                raw_mask,
                spans,
                "Linear interpolation markers",
                half_interval,
            )
        )

    overview_paths = _save_overview_pages(
        output_dir,
        interp_features,
        interp_timestamps,
        interp_values,
        raw_mask,
        spans,
        args.overview_per_page,
        "Linear interpolation markers",
        half_interval,
    )

    summary = {
        "raw_path": raw_path.as_posix(),
        "interpolated_path": interpolated_path.as_posix(),
        "n_features": len(interp_features),
        "n_total_timestamps": len(interp_timestamps),
        "n_raw_timestamps": int(raw_mask.sum()),
        "n_interpolated_timestamps": int(interpolated_mask.sum()),
        "n_interpolated_spans": len(spans),
        "feature_names": interp_features,
        "spans": spans,
        "feature_plot_paths": [path.as_posix() for path in feature_paths],
        "overview_paths": [path.as_posix() for path in overview_paths],
    }
    summary_path = output_dir / "feature_plot_manifest.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("Features plotted: %d" % len(interp_features))
    print("Interpolated timestamps highlighted: %d" % int(interpolated_mask.sum()))
    print("Interpolated spans highlighted: %d" % len(spans))
    print("Saved manifest: %s" % summary_path)
    for path in overview_paths:
        print("Saved: %s" % path)


if __name__ == "__main__":
    main()
