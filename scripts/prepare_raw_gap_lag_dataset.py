#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Drop interpolated rows from a lagged LiquidSugar CSV and keep the original-gap timeline."
    )
    parser.add_argument("--input", type=Path, required=True, help="Full lagged CSV with is_interpolated markers.")
    parser.add_argument("--output", type=Path, required=True, help="Destination CSV path for the raw-gap copy.")
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help="Optional JSON summary path. Defaults to <output>.summary.json",
    )
    parser.add_argument("--time-col", default="TimeStamp", help="Timestamp column name.")
    parser.add_argument("--split-col", default="split", help="Split column name.")
    parser.add_argument("--lag-col", default="lag_gt", help="Ground-truth lag column name.")
    parser.add_argument(
        "--interpolated-col",
        default="is_interpolated",
        help="Column that flags rows inserted during regularization.",
    )
    return parser.parse_args()


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(str(path)))


def _to_bool_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        return numeric.fillna(0).astype(int).ne(0)

    lowered = series.fillna("").astype(str).str.strip().str.lower()
    return lowered.isin(["1", "true", "t", "yes", "y"])


def main():
    args = parse_args()
    input_path = _absolute_path(args.input)
    output_path = _absolute_path(args.output)
    summary_path = (
        _absolute_path(args.summary_path)
        if args.summary_path is not None
        else output_path.with_suffix(output_path.suffix + ".summary.json")
    )

    df = pd.read_csv(input_path)
    required_cols = [args.time_col, args.interpolated_col, args.split_col, args.lag_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError("Missing required columns: %s" % ", ".join(missing_cols))

    df[args.time_col] = pd.to_datetime(df[args.time_col])
    interp_mask = _to_bool_mask(df[args.interpolated_col])
    raw_df = df.loc[~interp_mask].copy()
    raw_df = raw_df.sort_values(args.time_col).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(output_path, index=False)

    split_counts = (
        raw_df[args.split_col]
        .value_counts(sort=False)
        .reindex(["train", "val", "test"])
        .fillna(0)
        .astype(int)
        .to_dict()
    )
    lag_counts = (
        raw_df[args.lag_col]
        .value_counts(sort=False)
        .sort_index()
        .astype(int)
        .to_dict()
    )
    test_lag_counts = (
        raw_df.loc[raw_df[args.split_col] == "test", args.lag_col]
        .value_counts(sort=False)
        .sort_index()
        .astype(int)
        .to_dict()
    )

    summary = {
        "input_path": input_path.as_posix(),
        "output_path": output_path.as_posix(),
        "rows_input": int(len(df)),
        "rows_output": int(len(raw_df)),
        "rows_dropped_as_interpolated": int(interp_mask.sum()),
        "time_min": raw_df[args.time_col].min().strftime("%Y-%m-%d %H:%M"),
        "time_max": raw_df[args.time_col].max().strftime("%Y-%m-%d %H:%M"),
        "split_row_count": split_counts,
        "lag_value_count_all_rows": {str(key): int(value) for key, value in lag_counts.items()},
        "lag_value_count_test_rows": {str(key): int(value) for key, value in test_lag_counts.items()},
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("Wrote raw-gap dataset to %s" % output_path)
    print("Wrote summary to %s" % summary_path)
    print("Rows kept: %d / %d" % (len(raw_df), len(df)))
    print("Split counts: %s" % split_counts)
    print("Test lag counts: %s" % {str(key): int(value) for key, value in test_lag_counts.items()})


if __name__ == "__main__":
    main()
