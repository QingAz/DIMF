#!/usr/bin/env python3

import argparse
import csv
import json
import math
import os
from datetime import datetime, timedelta
from pathlib import Path

TIME_FORMAT = "%Y-%m-%d %H:%M"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare DIMF delay estimates against synthetic ground-truth delay kernels."
    )
    parser.add_argument("--estimates", type=Path, required=True, help="Path to test_delay_estimates.csv")
    parser.add_argument("--truth", type=Path, required=True, help="Path to delay_truth.csv")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for summary files. Defaults to the estimate file's directory.",
    )
    parser.add_argument(
        "--edge",
        default=None,
        help="Optional edge filter, e.g. stage1_to_stage2. If omitted, all common edges are evaluated.",
    )
    parser.add_argument(
        "--raw-reference",
        type=Path,
        default=None,
        help="Optional raw pre-interpolation CSV used to flag interpolated timestamps and long gap spans.",
    )
    parser.add_argument(
        "--interval-min",
        type=int,
        default=15,
        help="Sampling interval in minutes used when deriving gap spans from the raw reference CSV.",
    )
    parser.add_argument(
        "--long-gap-min-slots",
        type=int,
        default=8,
        help="Treat raw gaps with at least this many missing slots as long interpolation spans.",
    )
    return parser.parse_args()


def _absolute_path(path):
    return Path(os.path.abspath(str(path)))


def _read_csv(csv_path):
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    return fieldnames, rows


def _parse_timestamp(raw_value):
    return datetime.strptime(raw_value, TIME_FORMAT)


def _discover_edges(fieldnames, marker):
    suffix = "_%s_expected_lag" % marker
    return sorted([name[:-len(suffix)] for name in fieldnames if name.endswith(suffix)])


def _distribution_from_row(row, edge, marker):
    prefix = "%s_%s_pi_lag" % (edge, marker)
    cols = [key for key in row.keys() if key.startswith(prefix)]
    cols = sorted(cols, key=lambda name: int(name.split("lag")[-1]))
    return [float(row[col]) for col in cols]


def _kl_divergence(p, q):
    total = 0.0
    for p_i, q_i in zip(p, q):
        if p_i <= 0.0:
            continue
        total += p_i * math.log(p_i / max(q_i, 1e-12))
    return total


def _js_divergence(p, q):
    m = [(p_i + q_i) * 0.5 for p_i, q_i in zip(p, q)]
    return 0.5 * _kl_divergence(p, m) + 0.5 * _kl_divergence(q, m)


def _entropy(values):
    return -sum(value * math.log(max(value, 1e-12)) for value in values)


def _load_raw_gap_metadata(raw_reference_path, interval_min, long_gap_min_slots):
    if raw_reference_path is None:
        return None

    _, raw_rows = _read_csv(raw_reference_path)
    timestamps = [_parse_timestamp(row["TimeStamp"]) for row in raw_rows]
    raw_timestamp_set = set(timestamps)
    interval = timedelta(minutes=interval_min)
    long_gap_spans = []
    any_gap_spans = []

    for prev_ts, cur_ts in zip(timestamps, timestamps[1:]):
        delta_minutes = int((cur_ts - prev_ts).total_seconds() // 60)
        if delta_minutes <= interval_min:
            continue
        if delta_minutes % interval_min != 0:
            continue

        missing_slots = delta_minutes // interval_min - 1
        gap_start = prev_ts + interval
        gap_end = cur_ts - interval
        span = {
            "start": gap_start,
            "end": gap_end,
            "missing_slots": missing_slots,
            "gap_minutes": delta_minutes,
        }
        any_gap_spans.append(span)
        if missing_slots >= long_gap_min_slots:
            long_gap_spans.append(span)

    return {
        "raw_timestamp_set": raw_timestamp_set,
        "any_gap_spans": any_gap_spans,
        "long_gap_spans": long_gap_spans,
    }


def _in_gap_span(timestamp, spans):
    return any(span["start"] <= timestamp <= span["end"] for span in spans)


def _summarize_rows(rows):
    if not rows:
        return None

    n = float(len(rows))
    expected_mae = sum(row["expected_abs_error"] for row in rows) / n
    expected_rmse = math.sqrt(sum(row["expected_sq_error"] for row in rows) / n)
    argmax_mae = sum(row["argmax_abs_error"] for row in rows) / n
    argmax_acc = sum(row["argmax_hit"] for row in rows) / n
    mean_js = sum(row["js_divergence"] for row in rows) / n
    mean_entropy = sum(row["pred_entropy"] for row in rows) / n

    return {
        "n_matched": int(n),
        "expected_lag_mae": expected_mae,
        "expected_lag_rmse": expected_rmse,
        "argmax_lag_mae": argmax_mae,
        "argmax_lag_accuracy": argmax_acc,
        "mean_js_divergence": mean_js,
        "mean_pred_entropy": mean_entropy,
    }


def main():
    args = parse_args()
    estimates_path = _absolute_path(args.estimates)
    truth_path = _absolute_path(args.truth)
    raw_reference_path = _absolute_path(args.raw_reference) if args.raw_reference is not None else None
    output_dir = _absolute_path(args.output_dir) if args.output_dir is not None else estimates_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    estimate_fields, estimate_rows = _read_csv(estimates_path)
    truth_fields, truth_rows = _read_csv(truth_path)
    raw_gap_metadata = _load_raw_gap_metadata(raw_reference_path, args.interval_min, args.long_gap_min_slots)

    pred_edges = set(_discover_edges(estimate_fields, "pred"))
    true_edges = set(_discover_edges(truth_fields, "true"))
    common_edges = sorted(pred_edges & true_edges)
    if args.edge is not None:
        common_edges = [edge for edge in common_edges if edge == args.edge]
    if not common_edges:
        raise ValueError("No common edges found between estimate and truth files")

    truth_by_time = {row["TimeStamp"]: row for row in truth_rows}
    comparison_rows = []
    summary = {}

    for edge in common_edges:
        matched_rows = []
        for pred_row in estimate_rows:
            timestamp = pred_row["TimeStamp"]
            true_row = truth_by_time.get(timestamp)
            if true_row is None:
                continue
            timestamp_dt = _parse_timestamp(timestamp)

            pred_expected = float(pred_row["%s_pred_expected_lag" % edge])
            true_expected = float(true_row["%s_true_expected_lag" % edge])
            pred_argmax = int(float(pred_row["%s_pred_argmax_lag" % edge]))
            true_argmax = int(float(true_row["%s_true_argmax_lag" % edge]))
            pred_pi = _distribution_from_row(pred_row, edge, "pred")
            true_pi = _distribution_from_row(true_row, edge, "true")
            if len(pred_pi) != len(true_pi):
                raise ValueError("Lag support mismatch for edge %s at %s" % (edge, timestamp))

            js_div = _js_divergence(true_pi, pred_pi)
            pred_entropy = _entropy(pred_pi)
            is_raw_timestamp = None
            is_interpolated_timestamp = None
            in_any_gap_span = None
            in_long_gap_span = None
            if raw_gap_metadata is not None:
                is_raw_timestamp = int(timestamp_dt in raw_gap_metadata["raw_timestamp_set"])
                is_interpolated_timestamp = int(not is_raw_timestamp)
                in_any_gap_span = int(_in_gap_span(timestamp_dt, raw_gap_metadata["any_gap_spans"]))
                in_long_gap_span = int(_in_gap_span(timestamp_dt, raw_gap_metadata["long_gap_spans"]))

            matched_rows.append(
                {
                    "TimeStamp": timestamp,
                    "edge": edge,
                    "true_expected_lag": true_expected,
                    "pred_expected_lag": pred_expected,
                    "expected_abs_error": abs(pred_expected - true_expected),
                    "expected_sq_error": (pred_expected - true_expected) ** 2,
                    "true_argmax_lag": true_argmax,
                    "pred_argmax_lag": pred_argmax,
                    "argmax_hit": 1 if pred_argmax == true_argmax else 0,
                    "argmax_abs_error": abs(pred_argmax - true_argmax),
                    "js_divergence": js_div,
                    "pred_entropy": pred_entropy,
                    "is_raw_timestamp": is_raw_timestamp,
                    "is_interpolated_timestamp": is_interpolated_timestamp,
                    "in_any_gap_span": in_any_gap_span,
                    "in_long_gap_span": in_long_gap_span,
                }
            )

        if not matched_rows:
            continue

        overall_summary = _summarize_rows(matched_rows)
        summary[edge] = dict(overall_summary)
        if raw_gap_metadata is not None:
            outside_long_gap_rows = [row for row in matched_rows if row["in_long_gap_span"] == 0]
            inside_long_gap_rows = [row for row in matched_rows if row["in_long_gap_span"] == 1]
            outside_summary = _summarize_rows(outside_long_gap_rows)
            inside_summary = _summarize_rows(inside_long_gap_rows)
            summary[edge]["outside_long_gap_spans"] = outside_summary
            summary[edge]["inside_long_gap_spans"] = inside_summary
        comparison_rows.extend(matched_rows)

    if not summary:
        raise ValueError("No matched rows found after joining estimates and truth")

    comparison_csv_path = output_dir / "delay_recovery_comparison.csv"
    summary_json_path = output_dir / "delay_recovery_summary.json"

    comparison_fieldnames = [
        "TimeStamp",
        "edge",
        "true_expected_lag",
        "pred_expected_lag",
        "expected_abs_error",
        "expected_sq_error",
        "true_argmax_lag",
        "pred_argmax_lag",
        "argmax_hit",
        "argmax_abs_error",
        "js_divergence",
        "pred_entropy",
        "is_raw_timestamp",
        "is_interpolated_timestamp",
        "in_any_gap_span",
        "in_long_gap_span",
    ]
    with comparison_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=comparison_fieldnames)
        writer.writeheader()
        writer.writerows(comparison_rows)

    summary_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("Wrote comparison rows to %s" % comparison_csv_path)
    print("Wrote summary to %s" % summary_json_path)
    for edge in sorted(summary):
        metrics = summary[edge]
        print(
            "%s: n=%d expected_mae=%.6f expected_rmse=%.6f argmax_acc=%.6f mean_js=%.6f mean_entropy=%.6f"
            % (
                edge,
                metrics["n_matched"],
                metrics["expected_lag_mae"],
                metrics["expected_lag_rmse"],
                metrics["argmax_lag_accuracy"],
                metrics["mean_js_divergence"],
                metrics["mean_pred_entropy"],
            )
        )
        if metrics.get("outside_long_gap_spans") is not None:
            sub = metrics["outside_long_gap_spans"]
            print(
                "  outside_long_gap_spans: n=%d expected_mae=%.6f argmax_acc=%.6f mean_js=%.6f"
                % (
                    sub["n_matched"],
                    sub["expected_lag_mae"],
                    sub["argmax_lag_accuracy"],
                    sub["mean_js_divergence"],
                )
            )


if __name__ == "__main__":
    main()
