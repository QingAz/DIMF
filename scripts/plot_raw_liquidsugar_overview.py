#!/usr/bin/env python3

import argparse
import csv
import json
import math
import os
from collections import OrderedDict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, NamedTuple, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


TIME_FORMAT = "%Y-%m-%d %H:%M"


class Gap(NamedTuple):
    prev_timestamp: datetime
    next_timestamp: datetime
    gap_minutes: int
    missing_slots: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize the original irregular LiquidSugar dataset without interpolation."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/LiquidSugar.csv"),
        help="Path to the original LiquidSugar CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/data_visualizations/raw_liquidsugar"),
        help="Directory for generated figures and summary metadata.",
    )
    parser.add_argument(
        "--time-col",
        default="TimeStamp",
        help="Timestamp column name.",
    )
    parser.add_argument(
        "--interval-min",
        type=int,
        default=15,
        help="Expected sampling interval in minutes.",
    )
    parser.add_argument(
        "--max-features-per-figure",
        type=int,
        default=6,
        help="Maximum number of feature subplots per figure.",
    )
    return parser.parse_args()


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(str(path)))


def _parse_timestamp(raw_value: str) -> datetime:
    return datetime.strptime(raw_value, TIME_FORMAT)


def _format_timestamp(value: datetime) -> str:
    return value.strftime(TIME_FORMAT)


def _read_csv_series(
    csv_path: Path,
    time_col: str,
) -> Tuple[List[str], List[datetime], Dict[str, np.ndarray]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if not fieldnames:
            raise ValueError("Missing CSV header in %s" % csv_path)
        if time_col not in fieldnames:
            raise ValueError("Missing timestamp column %s in %s" % (time_col, csv_path))
        rows = list(reader)

    if not rows:
        raise ValueError("CSV is empty: %s" % csv_path)

    feature_names = [name for name in fieldnames if name != time_col]
    parsed_rows = []
    for line_no, row in enumerate(rows, start=2):
        timestamp = _parse_timestamp(row[time_col])
        values = {}
        for feature_name in feature_names:
            raw_value = row[feature_name]
            if raw_value == "":
                raise ValueError(
                    "Found empty value in column %s at line %d; expected irregularity to come from missing timestamps."
                    % (feature_name, line_no)
                )
            values[feature_name] = float(raw_value)
        parsed_rows.append((timestamp, values))

    parsed_rows.sort(key=lambda item: item[0])
    for previous, current in zip(parsed_rows, parsed_rows[1:]):
        if previous[0] == current[0]:
            raise ValueError("Duplicate timestamp found: %s" % _format_timestamp(current[0]))

    timestamps = [timestamp for timestamp, _ in parsed_rows]
    values = {
        feature_name: np.asarray(
            [row_values[feature_name] for _, row_values in parsed_rows],
            dtype=np.float64,
        )
        for feature_name in feature_names
    }
    return feature_names, timestamps, values


def _collect_gaps(timestamps: Sequence[datetime], interval_min: int) -> List[Gap]:
    interval = timedelta(minutes=interval_min)
    gaps = []
    for prev_timestamp, next_timestamp in zip(timestamps, timestamps[1:]):
        delta = next_timestamp - prev_timestamp
        if delta <= timedelta(0):
            raise ValueError("Timestamps must be strictly increasing")
        gap_minutes = int(delta.total_seconds() // 60)
        if gap_minutes % interval_min != 0:
            raise ValueError(
                "Gap %d minutes is not divisible by %d minutes: %s -> %s"
                % (
                    gap_minutes,
                    interval_min,
                    _format_timestamp(prev_timestamp),
                    _format_timestamp(next_timestamp),
                )
            )
        missing_slots = max(0, int(delta / interval) - 1)
        if missing_slots > 0:
            gaps.append(
                Gap(
                    prev_timestamp=prev_timestamp,
                    next_timestamp=next_timestamp,
                    gap_minutes=gap_minutes,
                    missing_slots=missing_slots,
                )
            )
    return gaps


def _observation_segments(timestamps: Sequence[datetime], interval_min: int) -> List[Tuple[int, int]]:
    interval = timedelta(minutes=interval_min)
    segments: List[Tuple[int, int]] = []
    start_idx = 0
    for idx in range(1, len(timestamps)):
        if timestamps[idx] - timestamps[idx - 1] > interval:
            segments.append((start_idx, idx - 1))
            start_idx = idx
    segments.append((start_idx, len(timestamps) - 1))
    return segments


def _feature_group(feature_name: str) -> str:
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


def _group_features(feature_names: Sequence[str]) -> "OrderedDict[str, List[str]]":
    grouped: "OrderedDict[str, List[str]]" = OrderedDict()
    for feature_name in feature_names:
        group_name = _feature_group(feature_name)
        grouped.setdefault(group_name, []).append(feature_name)
    return grouped


def _chunked(items: Sequence[str], chunk_size: int) -> Iterable[Sequence[str]]:
    for idx in range(0, len(items), chunk_size):
        yield items[idx:idx + chunk_size]


def _setup_time_axis(ax, timestamps: Sequence[datetime]) -> None:
    locator = mdates.AutoDateLocator(minticks=4, maxticks=9)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.set_xlim(timestamps[0], timestamps[-1])


def _shade_gaps(ax, gaps: Sequence[Gap], half_interval: timedelta) -> None:
    first = True
    for gap in gaps:
        start = gap.prev_timestamp + half_interval
        end = gap.next_timestamp - half_interval
        if end <= start:
            continue
        ax.axvspan(
            start,
            end,
            color="#f59e0b",
            alpha=0.18,
            lw=0.0,
            label="Missing timestamps" if first else None,
        )
        first = False


def _plot_feature_chunk(
    output_dir: Path,
    group_name: str,
    page_idx: int,
    page_count: int,
    feature_names: Sequence[str],
    timestamps: Sequence[datetime],
    values: Dict[str, np.ndarray],
    segments: Sequence[Tuple[int, int]],
    gaps: Sequence[Gap],
    interval_min: int,
) -> Path:
    n_features = len(feature_names)
    fig_height = max(4.5, 2.5 * n_features)
    fig, axes = plt.subplots(n_features, 1, figsize=(16, fig_height), sharex=True)
    if n_features == 1:
        axes = [axes]

    x_values = np.asarray(timestamps)
    half_interval = timedelta(minutes=interval_min / 2.0)

    for ax, feature_name in zip(axes, feature_names):
        _shade_gaps(ax, gaps, half_interval)
        feature_values = values[feature_name]
        for segment_start, segment_end in segments:
            ax.plot(
                x_values[segment_start:segment_end + 1],
                feature_values[segment_start:segment_end + 1],
                color="#2563eb",
                linewidth=0.95,
                alpha=0.88,
            )
        ax.scatter(
            x_values,
            feature_values,
            s=4,
            color="#0f172a",
            alpha=0.26,
            linewidths=0.0,
            zorder=3,
        )
        ax.set_ylabel(feature_name, fontsize=9)
        ax.grid(alpha=0.18)

    axes[0].set_title(
        "Original LiquidSugar raw series | group=%s | page %d/%d" % (group_name, page_idx, page_count)
    )
    _setup_time_axis(axes[-1], timestamps)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, loc="upper right", frameon=False)

    fig.tight_layout()
    output_path = output_dir / ("%s_page_%02d.png" % (group_name, page_idx))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _month_tick_positions(start_day: datetime, end_day: datetime) -> Tuple[List[int], List[str]]:
    positions = []
    labels = []
    cursor = datetime(start_day.year, start_day.month, 1)
    while cursor <= end_day:
        if cursor >= start_day:
            positions.append((cursor.date() - start_day.date()).days)
            labels.append(cursor.strftime("%Y-%m"))
        if cursor.month == 12:
            cursor = datetime(cursor.year + 1, 1, 1)
        else:
            cursor = datetime(cursor.year, cursor.month + 1, 1)
    return positions, labels


def _plot_observation_coverage(
    output_path: Path,
    timestamps: Sequence[datetime],
    gaps: Sequence[Gap],
    interval_min: int,
) -> Path:
    start_day = datetime(timestamps[0].year, timestamps[0].month, timestamps[0].day)
    end_day = datetime(timestamps[-1].year, timestamps[-1].month, timestamps[-1].day)
    n_days = (end_day.date() - start_day.date()).days + 1
    slots_per_day = int((24 * 60) // interval_min)

    grid = np.zeros((n_days, slots_per_day), dtype=np.int8)
    for timestamp in timestamps:
        day_idx = (timestamp.date() - start_day.date()).days
        slot_idx = (timestamp.hour * 60 + timestamp.minute) // interval_min
        grid[day_idx, slot_idx] = 1

    delta_minutes = np.asarray(
        [
            (next_timestamp - prev_timestamp).total_seconds() / 60.0
            for prev_timestamp, next_timestamp in zip(timestamps, timestamps[1:])
        ],
        dtype=np.float64,
    )
    delta_times = np.asarray(timestamps[1:])

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(16, 9),
        gridspec_kw={"height_ratios": [1.8, 1.0]},
    )

    cmap = ListedColormap(["#f8fafc", "#2563eb"])
    im = axes[0].imshow(
        grid,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=0,
        vmax=1,
        origin="upper",
    )
    axes[0].set_title("Raw observation coverage on the 15-minute grid")
    axes[0].set_ylabel("Day index")
    axes[0].set_xlabel("Hour of day")
    hour_ticks = np.arange(0, slots_per_day + 1, max(1, 60 // interval_min * 3))
    hour_labels = [str(int((tick * interval_min) // 60)) for tick in hour_ticks]
    axes[0].set_xticks(hour_ticks)
    axes[0].set_xticklabels(hour_labels)
    month_positions, month_labels = _month_tick_positions(start_day, end_day)
    axes[0].set_yticks(month_positions)
    axes[0].set_yticklabels(month_labels)
    cbar = fig.colorbar(im, ax=axes[0], pad=0.01, fraction=0.025)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["missing", "observed"])

    axes[1].plot(delta_times, delta_minutes, color="#0f766e", linewidth=1.0, alpha=0.9)
    axes[1].scatter(delta_times, delta_minutes, s=8, color="#0f766e", alpha=0.45, linewidths=0.0)
    axes[1].axhline(interval_min, color="#dc2626", linestyle="--", linewidth=1.0, alpha=0.8)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Gap to next sample (minutes, log)")
    axes[1].set_title("Observed sampling interval over time")
    axes[1].grid(alpha=0.2)
    _setup_time_axis(axes[1], delta_times)

    annotation_lines = [
        "rows=%d" % len(timestamps),
        "gap_spans=%d" % len(gaps),
        "missing_15min_slots=%d" % sum(gap.missing_slots for gap in gaps),
        "max_gap_hours=%.2f" % (
            max((gap.gap_minutes for gap in gaps), default=interval_min) / 60.0
        ),
    ]
    axes[1].text(
        0.995,
        0.98,
        "\n".join(annotation_lines),
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.92},
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _write_summary(
    output_path: Path,
    input_path: Path,
    feature_names: Sequence[str],
    timestamps: Sequence[datetime],
    gaps: Sequence[Gap],
    interval_min: int,
    figure_paths: Sequence[Path],
    coverage_path: Path,
) -> None:
    deltas = [
        int((next_timestamp - prev_timestamp).total_seconds() // 60)
        for prev_timestamp, next_timestamp in zip(timestamps, timestamps[1:])
    ]
    gap_examples = [
        {
            "prev_timestamp": _format_timestamp(gap.prev_timestamp),
            "next_timestamp": _format_timestamp(gap.next_timestamp),
            "gap_minutes": gap.gap_minutes,
            "missing_slots": gap.missing_slots,
        }
        for gap in sorted(gaps, key=lambda item: item.missing_slots, reverse=True)[:15]
    ]

    summary = {
        "input_path": input_path.as_posix(),
        "n_rows": len(timestamps),
        "n_features": len(feature_names),
        "feature_names": list(feature_names),
        "start_timestamp": _format_timestamp(timestamps[0]),
        "end_timestamp": _format_timestamp(timestamps[-1]),
        "expected_interval_minutes": interval_min,
        "observed_interval_minutes": {
            str(delta): deltas.count(delta)
            for delta in sorted(set(deltas))
        },
        "n_gap_spans": len(gaps),
        "n_missing_expected_timestamps": int(sum(gap.missing_slots for gap in gaps)),
        "largest_gaps": gap_examples,
        "figure_paths": [path.as_posix() for path in figure_paths],
        "coverage_path": coverage_path.as_posix(),
    }
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_path = _absolute_path(args.input)
    output_dir = _absolute_path(args.output_dir)
    groups_dir = output_dir / "groups"
    output_dir.mkdir(parents=True, exist_ok=True)
    groups_dir.mkdir(parents=True, exist_ok=True)

    feature_names, timestamps, values = _read_csv_series(input_path, args.time_col)
    gaps = _collect_gaps(timestamps, args.interval_min)
    segments = _observation_segments(timestamps, args.interval_min)
    grouped = _group_features(feature_names)

    figure_paths: List[Path] = []
    for group_name, group_features in grouped.items():
        page_count = int(math.ceil(len(group_features) / float(args.max_features_per_figure)))
        for page_idx, feature_chunk in enumerate(
            _chunked(group_features, args.max_features_per_figure),
            start=1,
        ):
            figure_paths.append(
                _plot_feature_chunk(
                    groups_dir,
                    group_name,
                    page_idx,
                    page_count,
                    feature_chunk,
                    timestamps,
                    values,
                    segments,
                    gaps,
                    args.interval_min,
                )
            )

    coverage_path = _plot_observation_coverage(
        output_dir / "raw_observation_pattern.png",
        timestamps,
        gaps,
        args.interval_min,
    )
    summary_path = output_dir / "raw_liquidsugar_summary.json"
    _write_summary(
        summary_path,
        input_path,
        feature_names,
        timestamps,
        gaps,
        args.interval_min,
        figure_paths,
        coverage_path,
    )

    print("Loaded rows: %d" % len(timestamps))
    print("Loaded features: %d" % len(feature_names))
    print("Detected gap spans: %d" % len(gaps))
    print("Generated group figures: %d" % len(figure_paths))
    print("Saved coverage figure: %s" % coverage_path)
    print("Saved summary: %s" % summary_path)


if __name__ == "__main__":
    main()
