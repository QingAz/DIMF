#!/usr/bin/env python3

import argparse
import csv
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, NamedTuple, Sequence, Tuple


TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M"


class Gap(NamedTuple):
    prev_timestamp: datetime
    next_timestamp: datetime
    gap_minutes: int
    missing_slots: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a regular 15-minute LiquidSugar copy by linearly interpolating missing timestamps."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/O_data/LiquidSugar.csv"),
        help="Source CSV path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/LiquidSugar_linear_interpolated.csv"),
        help="Destination CSV path for the interpolated copy.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Optional summary path. Defaults to <output>.summary.txt",
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
    return parser.parse_args()


def _parse_timestamp(raw_value: str) -> datetime:
    return datetime.strptime(raw_value, TIMESTAMP_FORMAT)


def _format_timestamp(value: datetime) -> str:
    return f"{value.year:04d}-{value.month:02d}-{value.day:02d} {value.hour}:{value.minute:02d}"


def _format_float(value: float) -> str:
    return format(value, ".15g")


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(str(path)))


def _load_rows(
    csv_path: Path,
    time_col: str,
) -> Tuple[List[str], List[str], List[Tuple[datetime, Dict[str, str], Dict[str, float]]]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if not fieldnames:
            raise ValueError(f"No header found in {csv_path}")
        if time_col not in fieldnames:
            raise ValueError(f"Missing timestamp column '{time_col}' in {csv_path}")

        numeric_cols = [col for col in fieldnames if col != time_col]
        rows: List[Tuple[datetime, Dict[str, str], Dict[str, float]]] = []

        for line_no, row in enumerate(reader, start=2):
            timestamp = _parse_timestamp(row[time_col])
            numeric_values: Dict[str, float] = {}
            for col in numeric_cols:
                raw_value = row[col]
                if raw_value == "":
                    raise ValueError(
                        f"Found an empty value in column '{col}' at line {line_no}; "
                        "this script expects missingness to come from absent timestamps."
                    )
                try:
                    numeric_values[col] = float(raw_value)
                except ValueError as exc:
                    raise ValueError(
                        f"Column '{col}' at line {line_no} is not numeric: {raw_value!r}"
                    ) from exc
            rows.append((timestamp, row, numeric_values))

    rows.sort(key=lambda item: item[0])
    for previous, current in zip(rows, rows[1:]):
        if previous[0] == current[0]:
            raise ValueError(f"Duplicate timestamp found: {_format_timestamp(current[0])}")

    return fieldnames, numeric_cols, rows


def _collect_gaps(timestamps: Sequence[datetime], interval_min: int) -> List[Gap]:
    gaps: List[Gap] = []
    for prev_timestamp, next_timestamp in zip(timestamps, timestamps[1:]):
        gap_minutes = int((next_timestamp - prev_timestamp).total_seconds() // 60)
        if gap_minutes <= 0:
            raise ValueError("Timestamps must be strictly increasing")
        if gap_minutes % interval_min != 0:
            raise ValueError(
                f"Timestamp gap {gap_minutes} is not divisible by the interval {interval_min}: "
                f"{_format_timestamp(prev_timestamp)} -> {_format_timestamp(next_timestamp)}"
            )
        missing_slots = gap_minutes // interval_min - 1
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


def _build_full_grid(start: datetime, end: datetime, interval_min: int) -> List[datetime]:
    if end < start:
        raise ValueError("End timestamp must be >= start timestamp")
    interval = timedelta(minutes=interval_min)
    total_minutes = int((end - start).total_seconds() // 60)
    if total_minutes % interval_min != 0:
        raise ValueError("The full timestamp range is not divisible by the target interval")
    total_steps = total_minutes // interval_min
    return [start + interval * step for step in range(total_steps + 1)]


def _interpolate_series(known_points: Dict[int, float], total_length: int) -> List[float]:
    if not known_points:
        raise ValueError("Cannot interpolate an empty series")

    known_positions = sorted(known_points)
    values = [0.0] * total_length

    first_pos = known_positions[0]
    last_pos = known_positions[-1]
    first_value = known_points[first_pos]
    last_value = known_points[last_pos]

    for pos in range(0, first_pos):
        values[pos] = first_value
    for pos in range(last_pos + 1, total_length):
        values[pos] = last_value

    for pos in known_positions:
        values[pos] = known_points[pos]

    for left_pos, right_pos in zip(known_positions, known_positions[1:]):
        left_value = known_points[left_pos]
        right_value = known_points[right_pos]
        distance = right_pos - left_pos
        if distance <= 1:
            continue
        step = (right_value - left_value) / distance
        for offset in range(1, distance):
            values[left_pos + offset] = left_value + step * offset

    return values


def _write_summary(
    summary_path: Path,
    input_path: Path,
    output_path: Path,
    time_col: str,
    interval_min: int,
    original_rows: int,
    regularized_rows: int,
    gaps: Sequence[Gap],
) -> None:
    largest_gaps = sorted(gaps, key=lambda gap: gap.missing_slots, reverse=True)[:10]
    max_gap = max((gap.gap_minutes for gap in gaps), default=interval_min)

    lines = [
        f"input_path={input_path.as_posix()}",
        f"output_path={output_path.as_posix()}",
        f"time_col={time_col}",
        f"interval_minutes={interval_min}",
        f"original_rows={original_rows}",
        f"regularized_rows={regularized_rows}",
        f"inserted_interpolated_rows={regularized_rows - original_rows}",
        f"non_{interval_min}min_intervals={len(gaps)}",
        f"max_gap_minutes={max_gap}",
        "",
        "largest_gaps:",
    ]
    for gap in largest_gaps:
        lines.append(
            "- "
            f"{_format_timestamp(gap.prev_timestamp)} -> {_format_timestamp(gap.next_timestamp)}: "
            f"gap_minutes={gap.gap_minutes}, missing_slots={gap.missing_slots}"
        )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_path = _absolute_path(args.input)
    output_path = _absolute_path(args.output)
    summary_path = (
        _absolute_path(args.summary)
        if args.summary is not None
        else output_path.with_suffix(output_path.suffix + ".summary.txt")
    )

    fieldnames, numeric_cols, rows = _load_rows(input_path, args.time_col)
    timestamps = [timestamp for timestamp, _, _ in rows]
    gaps = _collect_gaps(timestamps, args.interval_min)
    full_grid = _build_full_grid(timestamps[0], timestamps[-1], args.interval_min)
    full_grid_index = {timestamp: idx for idx, timestamp in enumerate(full_grid)}

    original_rows_by_timestamp = {
        timestamp: dict(row)
        for timestamp, row, _ in rows
    }

    interpolated_values: Dict[str, List[float]] = {}
    for col in numeric_cols:
        known_points = {
            full_grid_index[timestamp]: numeric_values[col]
            for timestamp, _, numeric_values in rows
        }
        interpolated_values[col] = _interpolate_series(known_points, len(full_grid))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, timestamp in enumerate(full_grid):
            if timestamp in original_rows_by_timestamp:
                out_row = original_rows_by_timestamp[timestamp]
            else:
                out_row = {args.time_col: _format_timestamp(timestamp)}
                for col in numeric_cols:
                    out_row[col] = _format_float(interpolated_values[col][idx])

            out_row[args.time_col] = _format_timestamp(timestamp)
            writer.writerow(out_row)

    _write_summary(
        summary_path=summary_path,
        input_path=input_path,
        output_path=output_path,
        time_col=args.time_col,
        interval_min=args.interval_min,
        original_rows=len(rows),
        regularized_rows=len(full_grid),
        gaps=gaps,
    )

    print(f"Wrote interpolated dataset to {output_path}")
    print(f"Wrote summary to {summary_path}")
    print(f"Original rows: {len(rows)}")
    print(f"Regularized rows: {len(full_grid)}")
    print(f"Inserted rows: {len(full_grid) - len(rows)}")


if __name__ == "__main__":
    main()
