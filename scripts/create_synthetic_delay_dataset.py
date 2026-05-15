#!/usr/bin/env python3

import argparse
import csv
import json
import math
import os
import random
from pathlib import Path


EDGE_PREFIXES = {
    "feed_to_stage1": ("feed_", "stage1_"),
    "stage1_to_stage2": ("stage1_", "stage2_"),
    "stage2_to_stage3": ("stage2_", "stage3_"),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inject known synthetic delay kernels between adjacent stages in LiquidSugar."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/LiquidSugar_linear_interpolated.csv"),
        help="Regularized source CSV used as the base dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write the delayed dataset and truth files.",
    )
    parser.add_argument(
        "--edge",
        choices=sorted(EDGE_PREFIXES),
        default="stage1_to_stage2",
        help="Adjacent stage pair where the synthetic delay is injected.",
    )
    parser.add_argument(
        "--mode",
        choices=["piecewise_constant", "linear", "sinusoidal", "bimodal"],
        default="piecewise_constant",
        help="Shape of the ground-truth delay process.",
    )
    parser.add_argument("--time-col", default="TimeStamp", help="Timestamp column name.")
    parser.add_argument("--target-col", default="yield_flow", help="Prediction target column to inject with the same delayed upstream signal.")
    parser.add_argument(
        "--forecast-horizon",
        type=int,
        default=4,
        help="Prediction horizon H. target-col at row t+H is injected from the delayed upstream signal at row t.",
    )
    parser.add_argument("--interval-min", type=int, default=15, help="Expected sampling interval in minutes.")
    parser.add_argument("--l-max", type=int, default=23, help="Largest discrete lag used in the truth kernel.")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.35,
        help="Strength of injected upstream influence. target_new=(1-alpha)*orig + alpha*delayed.",
    )
    parser.add_argument(
        "--noise-std-ratio",
        type=float,
        default=0.01,
        help="Gaussian noise scale as a fraction of each target column's std.",
    )
    parser.add_argument(
        "--target-alpha",
        type=float,
        default=0.8,
        help="Strength of delayed upstream influence injected into target-col. Set to 0 to keep the original target.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the additive noise.")

    parser.add_argument(
        "--piecewise-lags",
        default="2,6,10",
        help="Comma-separated lag values used by piecewise_constant mode.",
    )
    parser.add_argument(
        "--piecewise-fractions",
        default="0.25,0.5,0.25",
        help="Comma-separated segment fractions used by piecewise_constant mode.",
    )
    parser.add_argument("--linear-start-lag", type=float, default=2.0, help="Start lag for linear mode.")
    parser.add_argument("--linear-end-lag", type=float, default=10.0, help="End lag for linear mode.")
    parser.add_argument("--sin-base-lag", type=float, default=6.0, help="Center lag for sinusoidal mode.")
    parser.add_argument("--sin-amplitude", type=float, default=3.0, help="Lag amplitude for sinusoidal mode.")
    parser.add_argument(
        "--sin-period",
        type=float,
        default=96.0 * 7.0,
        help="Sinusoidal period in timesteps. Default is one week for 15-minute data.",
    )
    parser.add_argument(
        "--bimodal-lags",
        default="3,8",
        help="Comma-separated lag values used by bimodal mode.",
    )
    parser.add_argument(
        "--bimodal-weights",
        default="0.7,0.3",
        help="Comma-separated mixture weights used by bimodal mode.",
    )
    return parser.parse_args()


def _absolute_path(path):
    return Path(os.path.abspath(str(path)))


def _parse_number_list(raw_value, cast_type):
    parts = [part.strip() for part in raw_value.split(",") if part.strip()]
    if not parts:
        raise ValueError("Expected at least one comma-separated value")
    return [cast_type(part) for part in parts]


def _mean(values):
    return sum(values) / float(len(values))


def _std(values, mean_value):
    variance = sum((value - mean_value) ** 2 for value in values) / float(len(values))
    return math.sqrt(max(variance, 0.0))


def _format_float(value):
    return format(value, ".15g")


def _load_numeric_csv(csv_path, time_col):
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if not fieldnames:
            raise ValueError("CSV header is missing")
        if time_col not in fieldnames:
            raise ValueError("Missing timestamp column: %s" % time_col)

        numeric_cols = [col for col in fieldnames if col != time_col]
        rows = []
        timestamps = []
        values_by_col = {col: [] for col in numeric_cols}
        for line_no, row in enumerate(reader, start=2):
            timestamps.append(row[time_col])
            numeric_row = {}
            for col in numeric_cols:
                raw_value = row[col]
                try:
                    numeric_value = float(raw_value)
                except ValueError:
                    raise ValueError("Column %s at line %d is not numeric: %r" % (col, line_no, raw_value))
                numeric_row[col] = numeric_value
                values_by_col[col].append(numeric_value)
            rows.append({"raw": row, "numeric": numeric_row})
    return fieldnames, numeric_cols, rows, timestamps, values_by_col


def _verify_regular_timestamps(timestamps, interval_min):
    from datetime import datetime

    fmt = "%Y-%m-%d %H:%M"
    parsed = [datetime.strptime(value, fmt) for value in timestamps]
    expected_seconds = interval_min * 60
    for prev, cur in zip(parsed, parsed[1:]):
        if (cur - prev).total_seconds() != expected_seconds:
            raise ValueError(
                "Input dataset must already be regularized before synthetic delay injection: %s -> %s"
                % (prev.strftime(fmt), cur.strftime(fmt))
            )


def _select_stage_columns(fieldnames, time_col, edge):
    source_prefix, target_prefix = EDGE_PREFIXES[edge]
    source_cols = [col for col in fieldnames if col != time_col and col.startswith(source_prefix)]
    target_cols = [col for col in fieldnames if col != time_col and col.startswith(target_prefix)]
    if not source_cols or not target_cols:
        raise ValueError("Unable to find source/target columns for edge %s" % edge)
    return source_cols, target_cols


def _fractional_kernel(center_lag, l_max):
    center = min(max(float(center_lag), 0.0), float(l_max))
    left = int(math.floor(center))
    right = int(math.ceil(center))
    kernel = [0.0] * (l_max + 1)
    if left == right:
        kernel[left] = 1.0
        return kernel
    right_weight = center - left
    left_weight = 1.0 - right_weight
    kernel[left] += left_weight
    kernel[right] += right_weight
    return kernel


def _piecewise_kernel(t_idx, total_steps, l_max, piecewise_lags, piecewise_fractions):
    position = (t_idx + 0.5) / float(total_steps)
    cumulative = 0.0
    for lag, fraction in zip(piecewise_lags, piecewise_fractions):
        cumulative += fraction
        if position <= cumulative + 1e-12:
            return _fractional_kernel(lag, l_max)
    return _fractional_kernel(piecewise_lags[-1], l_max)


def _build_truth_kernels(total_steps, args):
    if args.mode == "piecewise_constant":
        piecewise_lags = _parse_number_list(args.piecewise_lags, float)
        piecewise_fractions = _parse_number_list(args.piecewise_fractions, float)
        if len(piecewise_lags) != len(piecewise_fractions):
            raise ValueError("piecewise lags and fractions must have the same length")
        total_fraction = sum(piecewise_fractions)
        if abs(total_fraction - 1.0) > 1e-6:
            raise ValueError("piecewise fractions must sum to 1.0")
        return [
            _piecewise_kernel(t_idx, total_steps, args.l_max, piecewise_lags, piecewise_fractions)
            for t_idx in range(total_steps)
        ]

    if args.mode == "linear":
        if total_steps <= 1:
            center_lags = [args.linear_start_lag]
        else:
            center_lags = [
                args.linear_start_lag
                + (args.linear_end_lag - args.linear_start_lag) * t_idx / float(total_steps - 1)
                for t_idx in range(total_steps)
            ]
        return [_fractional_kernel(center_lag, args.l_max) for center_lag in center_lags]

    if args.mode == "sinusoidal":
        if args.sin_period <= 0:
            raise ValueError("sin-period must be positive")
        return [
            _fractional_kernel(
                args.sin_base_lag + args.sin_amplitude * math.sin(2.0 * math.pi * t_idx / args.sin_period),
                args.l_max,
            )
            for t_idx in range(total_steps)
        ]

    if args.mode == "bimodal":
        bimodal_lags = _parse_number_list(args.bimodal_lags, int)
        bimodal_weights = _parse_number_list(args.bimodal_weights, float)
        if len(bimodal_lags) != len(bimodal_weights):
            raise ValueError("bimodal lags and weights must have the same length")
        if abs(sum(bimodal_weights) - 1.0) > 1e-6:
            raise ValueError("bimodal weights must sum to 1.0")
        kernel = [0.0] * (args.l_max + 1)
        for lag, weight in zip(bimodal_lags, bimodal_weights):
            if lag < 0 or lag > args.l_max:
                raise ValueError("bimodal lag %d is outside [0, %d]" % (lag, args.l_max))
            kernel[lag] += weight
        return [list(kernel) for _ in range(total_steps)]

    raise ValueError("Unknown mode: %s" % args.mode)


def _delayed_series_from_kernel(source_z, kernels):
    delayed = []
    total_steps = len(source_z)
    for t_idx in range(total_steps):
        acc = 0.0
        kernel = kernels[t_idx]
        for lag, weight in enumerate(kernel):
            src_idx = t_idx - lag
            if src_idx < 0:
                src_idx = 0
            acc += weight * source_z[src_idx]
        delayed.append(acc)
    return delayed


def _truth_frame_rows(timestamps, edge, kernels):
    rows = []
    for timestamp, kernel in zip(timestamps, kernels):
        expected_lag = sum(lag * weight for lag, weight in enumerate(kernel))
        dominant_lag = max(range(len(kernel)), key=lambda lag: kernel[lag])
        row = {
            "TimeStamp": timestamp,
            "%s_true_expected_lag" % edge: _format_float(expected_lag),
            "%s_true_argmax_lag" % edge: str(dominant_lag),
        }
        for lag, weight in enumerate(kernel):
            row["%s_true_pi_lag%d" % (edge, lag)] = _format_float(weight)
        rows.append(row)
    return rows


def _write_truth_csv(truth_path, timestamps, edge, kernels):
    rows = _truth_frame_rows(timestamps, edge, kernels)
    fieldnames = list(rows[0].keys())
    with truth_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_dataset_csv(dataset_path, fieldnames, rows, time_col):
    with dataset_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _dataset_truth_fieldnames(edge, l_max):
    names = [
        "lag_gt",
        "lag_expected_gt",
        "%s_lag_gt" % edge,
        "%s_lag_expected_gt" % edge,
        "%s_true_expected_lag" % edge,
        "%s_true_argmax_lag" % edge,
    ]
    names.extend("%s_true_pi_lag%d" % (edge, lag) for lag in range(l_max + 1))
    return names


def _add_truth_columns(raw_row, edge, kernel):
    expected_lag = sum(lag * weight for lag, weight in enumerate(kernel))
    dominant_lag = max(range(len(kernel)), key=lambda lag: kernel[lag])
    raw_row["lag_gt"] = str(dominant_lag)
    raw_row["lag_expected_gt"] = _format_float(expected_lag)
    raw_row["%s_lag_gt" % edge] = str(dominant_lag)
    raw_row["%s_lag_expected_gt" % edge] = _format_float(expected_lag)
    raw_row["%s_true_expected_lag" % edge] = _format_float(expected_lag)
    raw_row["%s_true_argmax_lag" % edge] = str(dominant_lag)
    for lag, weight in enumerate(kernel):
        raw_row["%s_true_pi_lag%d" % (edge, lag)] = _format_float(weight)


def main():
    args = parse_args()
    input_path = _absolute_path(args.input)
    if args.output_dir is None:
        output_dir = _absolute_path(Path("data/synthetic_delay") / ("%s_%s" % (args.edge, args.mode)))
    else:
        output_dir = _absolute_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fieldnames, numeric_cols, rows, timestamps, values_by_col = _load_numeric_csv(input_path, args.time_col)
    _verify_regular_timestamps(timestamps, args.interval_min)
    source_cols, target_cols = _select_stage_columns(fieldnames, args.time_col, args.edge)
    if args.target_alpha > 0.0 and args.target_col not in values_by_col:
        raise ValueError("target-col %s is not present in %s" % (args.target_col, input_path))

    kernels = _build_truth_kernels(len(rows), args)

    source_stats = {}
    for col in source_cols:
        mean_value = _mean(values_by_col[col])
        std_value = _std(values_by_col[col], mean_value)
        source_stats[col] = (mean_value, std_value if std_value > 1e-12 else 1.0)

    target_stats = {}
    for col in target_cols:
        mean_value = _mean(values_by_col[col])
        std_value = _std(values_by_col[col], mean_value)
        target_stats[col] = (mean_value, std_value if std_value > 1e-12 else 1.0)

    target_col_stats = None
    if args.target_alpha > 0.0:
        target_mean = _mean(values_by_col[args.target_col])
        target_std = _std(values_by_col[args.target_col], target_mean)
        target_col_stats = (target_mean, target_std if target_std > 1e-12 else 1.0)

    delayed_source_cache = {}
    for col in source_cols:
        mean_value, std_value = source_stats[col]
        source_z = [(value - mean_value) / std_value for value in values_by_col[col]]
        delayed_source_cache[col] = _delayed_series_from_kernel(source_z, kernels)

    rng = random.Random(args.seed)
    output_rows = []
    for row_idx, row_bundle in enumerate(rows):
        raw_row = dict(row_bundle["raw"])
        for target_idx, target_col in enumerate(target_cols):
            source_col = source_cols[target_idx % len(source_cols)]
            target_mean, target_std = target_stats[target_col]
            delayed_source_value = delayed_source_cache[source_col][row_idx]
            mapped_source_value = target_mean + delayed_source_value * target_std
            noise = rng.gauss(0.0, args.noise_std_ratio * target_std) if args.noise_std_ratio > 0 else 0.0
            original_value = row_bundle["numeric"][target_col]
            synthetic_value = (1.0 - args.alpha) * original_value + args.alpha * mapped_source_value + noise
            raw_row[target_col] = _format_float(synthetic_value)
        _add_truth_columns(raw_row, args.edge, kernels[row_idx])
        output_rows.append(raw_row)

    if args.target_alpha > 0.0:
        if args.forecast_horizon < 0:
            raise ValueError("forecast-horizon must be non-negative")
        target_mean, target_std = target_col_stats
        for sample_idx in range(0, len(output_rows) - args.forecast_horizon):
            target_idx = sample_idx + args.forecast_horizon
            delayed_source_value = sum(delayed_source_cache[col][sample_idx] for col in source_cols) / float(len(source_cols))
            mapped_source_value = target_mean + delayed_source_value * target_std
            noise = rng.gauss(0.0, args.noise_std_ratio * target_std) if args.noise_std_ratio > 0 else 0.0
            original_value = rows[target_idx]["numeric"][args.target_col]
            synthetic_value = (1.0 - args.target_alpha) * original_value + args.target_alpha * mapped_source_value + noise
            output_rows[target_idx][args.target_col] = _format_float(synthetic_value)

    dataset_path = output_dir / ("LiquidSugar_%s_%s.csv" % (args.edge, args.mode))
    truth_path = output_dir / "delay_truth.csv"
    manifest_path = output_dir / "manifest.json"

    dataset_fieldnames = list(fieldnames)
    for name in _dataset_truth_fieldnames(args.edge, args.l_max):
        if name not in dataset_fieldnames:
            dataset_fieldnames.append(name)

    _write_dataset_csv(dataset_path, dataset_fieldnames, output_rows, args.time_col)
    _write_truth_csv(truth_path, timestamps, args.edge, kernels)

    manifest = {
        "input_path": input_path.as_posix(),
        "dataset_path": dataset_path.as_posix(),
        "truth_path": truth_path.as_posix(),
        "edge": args.edge,
        "mode": args.mode,
        "source_cols": source_cols,
        "target_cols": target_cols,
        "alpha": args.alpha,
        "target_col": args.target_col,
        "target_alpha": args.target_alpha,
        "forecast_horizon": args.forecast_horizon,
        "noise_std_ratio": args.noise_std_ratio,
        "interval_min": args.interval_min,
        "l_max": args.l_max,
        "seed": args.seed,
        "n_rows": len(rows),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("Wrote synthetic dataset to %s" % dataset_path)
    print("Wrote delay truth to %s" % truth_path)
    print("Wrote manifest to %s" % manifest_path)
    print("Injected edge: %s" % args.edge)
    print("Mode: %s" % args.mode)


if __name__ == "__main__":
    main()
