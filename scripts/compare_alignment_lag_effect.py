#!/usr/bin/env python3

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

TIME_FORMAT = "%Y-%m-%d %H:%M"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare DIMF runs with and without alignment on lag prediction and forecasting."
    )
    parser.add_argument("--aligned-estimates", type=Path, required=True, help="Aligned run test_delay_estimates.csv")
    parser.add_argument("--noalign-estimates", type=Path, required=True, help="No-alignment run test_delay_estimates.csv")
    parser.add_argument("--raw-dataset", type=Path, required=True, help="Raw-gap lagged dataset CSV")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for comparison outputs")
    parser.add_argument("--edge", default="stage1_to_stage2", help="Edge name to compare")
    parser.add_argument("--time-col", default="TimeStamp", help="Timestamp column name")
    parser.add_argument("--split-col", default="split", help="Split column name")
    parser.add_argument("--lag-col", default="lag_gt", help="Ground-truth lag column name")
    parser.add_argument(
        "--aligned-metrics",
        type=Path,
        default=None,
        help="Optional aligned run test_metrics.json",
    )
    parser.add_argument(
        "--noalign-metrics",
        type=Path,
        default=None,
        help="Optional no-alignment run test_metrics.json",
    )
    return parser.parse_args()


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(str(path)))


def _load_metrics(path: Path) -> Dict[str, float]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _pi_columns(frame: pd.DataFrame, edge: str) -> List[str]:
    prefix = "%s_pred_pi_lag" % edge
    cols = [col for col in frame.columns if col.startswith(prefix)]
    return sorted(cols, key=lambda name: int(name.split("lag")[-1]))


def _entropy(values: np.ndarray) -> np.ndarray:
    safe = np.clip(values, 1e-12, None)
    return -(safe * np.log(safe)).sum(axis=1)


def _summarize_subset(frame: pd.DataFrame) -> Dict[str, float]:
    if frame.empty:
        return {
            "n": 0,
            "expected_lag_mae": None,
            "expected_lag_rmse": None,
            "argmax_lag_mae": None,
            "argmax_lag_accuracy": None,
            "mean_pred_entropy": None,
            "mean_pred_expected_lag": None,
        }

    true_lag = frame["lag_gt"].to_numpy(dtype=np.float64)
    pred_expected = frame["pred_expected_lag"].to_numpy(dtype=np.float64)
    pred_argmax = frame["pred_argmax_lag"].to_numpy(dtype=np.float64)
    pred_entropy = frame["pred_entropy"].to_numpy(dtype=np.float64)

    expected_err = pred_expected - true_lag
    argmax_err = pred_argmax - true_lag

    return {
        "n": int(len(frame)),
        "expected_lag_mae": float(np.abs(expected_err).mean()),
        "expected_lag_rmse": float(np.sqrt(np.square(expected_err).mean())),
        "argmax_lag_mae": float(np.abs(argmax_err).mean()),
        "argmax_lag_accuracy": float((pred_argmax == true_lag).mean()),
        "mean_pred_entropy": float(pred_entropy.mean()),
        "mean_pred_expected_lag": float(pred_expected.mean()),
    }


def _prepare_joined_frame(
    estimates_path: Path,
    raw_test: pd.DataFrame,
    edge: str,
) -> pd.DataFrame:
    estimates = pd.read_csv(estimates_path)
    estimates["TimeStamp"] = pd.to_datetime(estimates["TimeStamp"]).dt.strftime(TIME_FORMAT)
    pi_cols = _pi_columns(estimates, edge)
    if not pi_cols:
        raise ValueError("No delay distribution columns found for edge %s in %s" % (edge, estimates_path))

    expected_col = "%s_pred_expected_lag" % edge
    argmax_col = "%s_pred_argmax_lag" % edge
    required = ["TimeStamp", expected_col, argmax_col] + pi_cols
    missing = [col for col in required if col not in estimates.columns]
    if missing:
        raise ValueError("Missing estimate columns in %s: %s" % (estimates_path, ", ".join(missing)))

    merged = raw_test[["TimeStamp", "lag_gt"]].merge(estimates[required], on="TimeStamp", how="inner")
    merged = merged.rename(
        columns={
            expected_col: "pred_expected_lag",
            argmax_col: "pred_argmax_lag",
        }
    )
    merged["pred_entropy"] = _entropy(merged[pi_cols].to_numpy(dtype=np.float64))
    return merged


def _confusion_table(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["lag_gt", "pred_argmax_lag", "count"])
    return (
        frame.groupby(["lag_gt", "pred_argmax_lag"], sort=True)
        .size()
        .reset_index(name="count")
        .sort_values(["lag_gt", "pred_argmax_lag"])
        .reset_index(drop=True)
    )


def _per_lag_rows(label: str, frame: pd.DataFrame) -> List[Dict[str, float]]:
    rows = []
    for lag_value in sorted(frame["lag_gt"].unique().tolist()):
        subset = frame.loc[frame["lag_gt"] == lag_value].copy()
        summary = _summarize_subset(subset)
        row = {"model": label, "lag_gt": int(lag_value)}
        row.update(summary)
        rows.append(row)
    return rows


def _table(headers: List[str], rows: List[List[str]]) -> str:
    sep = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return ("%." + str(digits) + "f") % float(value)


def main():
    args = parse_args()
    aligned_estimates = _absolute_path(args.aligned_estimates)
    noalign_estimates = _absolute_path(args.noalign_estimates)
    raw_dataset_path = _absolute_path(args.raw_dataset)
    output_dir = _absolute_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(raw_dataset_path)
    if args.split_col not in raw_df.columns or args.lag_col not in raw_df.columns:
        raise ValueError("Raw dataset must contain '%s' and '%s'" % (args.split_col, args.lag_col))
    raw_test = raw_df.loc[raw_df[args.split_col] == "test", [args.time_col, args.lag_col]].copy()
    raw_test = raw_test.rename(columns={args.time_col: "TimeStamp", args.lag_col: "lag_gt"})
    raw_test["TimeStamp"] = pd.to_datetime(raw_test["TimeStamp"]).dt.strftime(TIME_FORMAT)
    raw_test["lag_gt"] = raw_test["lag_gt"].astype(int)

    joined_aligned = _prepare_joined_frame(aligned_estimates, raw_test, args.edge)
    joined_noalign = _prepare_joined_frame(noalign_estimates, raw_test, args.edge)

    aligned_metrics = _load_metrics(_absolute_path(args.aligned_metrics) if args.aligned_metrics is not None else None)
    noalign_metrics = _load_metrics(_absolute_path(args.noalign_metrics) if args.noalign_metrics is not None else None)

    summaries = {}
    joined_frames = {
        "aligned": joined_aligned,
        "noalign": joined_noalign,
    }
    for label, frame in joined_frames.items():
        lagged = frame.loc[frame["lag_gt"] > 0].copy()
        nolag = frame.loc[frame["lag_gt"] == 0].copy()
        summaries[label] = {
            "overall": _summarize_subset(frame),
            "lagged_only": _summarize_subset(lagged),
            "no_lag_only": _summarize_subset(nolag),
            "per_lag": _per_lag_rows(label, frame),
        }

    comparison_rows = raw_test.merge(
        joined_aligned.rename(
            columns={
                "pred_expected_lag": "aligned_pred_expected_lag",
                "pred_argmax_lag": "aligned_pred_argmax_lag",
                "pred_entropy": "aligned_pred_entropy",
            }
        )[
            ["TimeStamp", "aligned_pred_expected_lag", "aligned_pred_argmax_lag", "aligned_pred_entropy"]
        ],
        on="TimeStamp",
        how="inner",
    ).merge(
        joined_noalign.rename(
            columns={
                "pred_expected_lag": "noalign_pred_expected_lag",
                "pred_argmax_lag": "noalign_pred_argmax_lag",
                "pred_entropy": "noalign_pred_entropy",
            }
        )[
            ["TimeStamp", "noalign_pred_expected_lag", "noalign_pred_argmax_lag", "noalign_pred_entropy"]
        ],
        on="TimeStamp",
        how="inner",
    )
    comparison_rows["aligned_expected_abs_error"] = (
        comparison_rows["aligned_pred_expected_lag"] - comparison_rows["lag_gt"]
    ).abs()
    comparison_rows["noalign_expected_abs_error"] = (
        comparison_rows["noalign_pred_expected_lag"] - comparison_rows["lag_gt"]
    ).abs()
    comparison_rows["aligned_argmax_hit"] = (
        comparison_rows["aligned_pred_argmax_lag"] == comparison_rows["lag_gt"]
    ).astype(int)
    comparison_rows["noalign_argmax_hit"] = (
        comparison_rows["noalign_pred_argmax_lag"] == comparison_rows["lag_gt"]
    ).astype(int)

    per_lag_records = []
    lag_values = sorted(comparison_rows["lag_gt"].unique().tolist())
    for lag_value in lag_values:
        aligned_row = next(row for row in summaries["aligned"]["per_lag"] if row["lag_gt"] == lag_value)
        noalign_row = next(row for row in summaries["noalign"]["per_lag"] if row["lag_gt"] == lag_value)
        per_lag_records.append(
            {
                "lag_gt": int(lag_value),
                "n": int(aligned_row["n"]),
                "aligned_expected_lag_mae": aligned_row["expected_lag_mae"],
                "noalign_expected_lag_mae": noalign_row["expected_lag_mae"],
                "aligned_argmax_lag_accuracy": aligned_row["argmax_lag_accuracy"],
                "noalign_argmax_lag_accuracy": noalign_row["argmax_lag_accuracy"],
                "aligned_mean_pred_expected_lag": aligned_row["mean_pred_expected_lag"],
                "noalign_mean_pred_expected_lag": noalign_row["mean_pred_expected_lag"],
            }
        )

    summary = {
        "raw_dataset": raw_dataset_path.as_posix(),
        "edge": args.edge,
        "n_test_rows_raw": int(len(raw_test)),
        "n_test_samples_compared": {
            "aligned": int(len(joined_aligned)),
            "noalign": int(len(joined_noalign)),
        },
        "forecast_metrics": {
            "aligned": {
                "MAE": aligned_metrics.get("MAE"),
                "RMSE": aligned_metrics.get("RMSE"),
                "R2": aligned_metrics.get("R2"),
            },
            "noalign": {
                "MAE": noalign_metrics.get("MAE"),
                "RMSE": noalign_metrics.get("RMSE"),
                "R2": noalign_metrics.get("R2"),
            },
        },
        "lag_recovery": summaries,
        "per_lag_comparison": per_lag_records,
        "confusion": {
            "aligned": _confusion_table(joined_aligned).to_dict(orient="records"),
            "noalign": _confusion_table(joined_noalign).to_dict(orient="records"),
        },
    }

    forecast_rows = [
        [
            "aligned",
            _fmt(summary["forecast_metrics"]["aligned"]["MAE"]),
            _fmt(summary["forecast_metrics"]["aligned"]["RMSE"]),
            _fmt(summary["forecast_metrics"]["aligned"]["R2"]),
        ],
        [
            "noalign",
            _fmt(summary["forecast_metrics"]["noalign"]["MAE"]),
            _fmt(summary["forecast_metrics"]["noalign"]["RMSE"]),
            _fmt(summary["forecast_metrics"]["noalign"]["R2"]),
        ],
    ]
    lag_rows = []
    for label in ["aligned", "noalign"]:
        for subset_name, pretty_name in [
            ("overall", "overall"),
            ("lagged_only", "lagged_only"),
            ("no_lag_only", "no_lag_only"),
        ]:
            metrics = summaries[label][subset_name]
            lag_rows.append(
                [
                    label,
                    pretty_name,
                    _fmt(metrics["n"], 0),
                    _fmt(metrics["expected_lag_mae"]),
                    _fmt(metrics["argmax_lag_accuracy"]),
                    _fmt(metrics["mean_pred_entropy"]),
                    _fmt(metrics["mean_pred_expected_lag"]),
                ]
            )

    per_lag_table_rows = [
        [
            _fmt(row["lag_gt"], 0),
            _fmt(row["n"], 0),
            _fmt(row["aligned_expected_lag_mae"]),
            _fmt(row["noalign_expected_lag_mae"]),
            _fmt(row["aligned_argmax_lag_accuracy"]),
            _fmt(row["noalign_argmax_lag_accuracy"]),
            _fmt(row["aligned_mean_pred_expected_lag"]),
            _fmt(row["noalign_mean_pred_expected_lag"]),
        ]
        for row in per_lag_records
    ]

    aligned_lagged = summaries["aligned"]["lagged_only"]
    noalign_lagged = summaries["noalign"]["lagged_only"]
    aligned_overall = summaries["aligned"]["overall"]
    noalign_overall = summaries["noalign"]["overall"]
    takeaways = [
        "Lagged samples only: aligned expected-lag MAE %s vs noalign %s."
        % (_fmt(aligned_lagged["expected_lag_mae"]), _fmt(noalign_lagged["expected_lag_mae"])),
        "Lagged samples only: aligned argmax accuracy %s vs noalign %s."
        % (_fmt(aligned_lagged["argmax_lag_accuracy"]), _fmt(noalign_lagged["argmax_lag_accuracy"])),
        "Forecasting MAE: aligned %s vs noalign %s."
        % (
            _fmt(summary["forecast_metrics"]["aligned"]["MAE"]),
            _fmt(summary["forecast_metrics"]["noalign"]["MAE"]),
        ),
        "Overall mean predicted lag: aligned %s vs noalign %s."
        % (_fmt(aligned_overall["mean_pred_expected_lag"]), _fmt(noalign_overall["mean_pred_expected_lag"])),
    ]

    report = "\n".join(
        [
            "# Alignment Comparison on Raw-Gap Lagged LiquidSugar",
            "",
            "Raw dataset: `%s`" % raw_dataset_path.as_posix(),
            "Compared edge: `%s`" % args.edge,
            "Matched test samples: aligned=%d, noalign=%d"
            % (len(joined_aligned), len(joined_noalign)),
            "",
            "## Forecast Metrics",
            "",
            _table(["model", "MAE", "RMSE", "R2"], forecast_rows),
            "",
            "## Lag Recovery",
            "",
            _table(
                ["model", "subset", "n", "expected_lag_mae", "argmax_acc", "mean_entropy", "mean_pred_expected"],
                lag_rows,
            ),
            "",
            "## Per True Lag",
            "",
            _table(
                [
                    "lag_gt",
                    "n",
                    "aligned_exp_mae",
                    "noalign_exp_mae",
                    "aligned_acc",
                    "noalign_acc",
                    "aligned_pred_mean",
                    "noalign_pred_mean",
                ],
                per_lag_table_rows,
            ),
            "",
            "## Takeaways",
            "",
            "\n".join(["- %s" % item for item in takeaways]),
            "",
        ]
    )

    joined_path = output_dir / "alignment_test_joined.csv"
    per_lag_path = output_dir / "alignment_per_lag_comparison.csv"
    summary_path = output_dir / "alignment_comparison_summary.json"
    report_path = output_dir / "alignment_comparison_report.md"

    comparison_rows.to_csv(joined_path, index=False)
    pd.DataFrame(per_lag_records).to_csv(per_lag_path, index=False)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(report + "\n", encoding="utf-8")

    print("Wrote joined comparison to %s" % joined_path)
    print("Wrote per-lag comparison to %s" % per_lag_path)
    print("Wrote summary to %s" % summary_path)
    print("Wrote report to %s" % report_path)
    print(
        "Lagged-only expected_lag_mae: aligned=%s noalign=%s"
        % (_fmt(aligned_lagged["expected_lag_mae"]), _fmt(noalign_lagged["expected_lag_mae"]))
    )
    print(
        "Lagged-only argmax_acc: aligned=%s noalign=%s"
        % (_fmt(aligned_lagged["argmax_lag_accuracy"]), _fmt(noalign_lagged["argmax_lag_accuracy"]))
    )


if __name__ == "__main__":
    main()
