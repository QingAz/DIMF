#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

TIME_FORMAT = "%Y-%m-%d %H:%M"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export p/m/d_hat lag component diagnostics from test delay estimates."
    )
    parser.add_argument("--estimates", type=Path, required=True, help="Path to test_delay_estimates.csv")
    parser.add_argument("--raw-dataset", type=Path, required=True, help="Raw-gap lag dataset CSV")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path")
    parser.add_argument("--edge", default="stage1_to_stage2", help="Lag edge name")
    parser.add_argument("--time-col", default="TimeStamp", help="Raw timestamp column")
    parser.add_argument("--split-col", default="split", help="Raw split column")
    parser.add_argument("--lag-col", default="lag_gt", help="Raw lag target column")
    parser.add_argument("--test-split", default="test", help="Split value to export")
    return parser.parse_args()


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(str(path)))


def _pi_columns(frame: pd.DataFrame, edge: str) -> List[str]:
    prefix = f"{edge}_pred_pi_lag"
    cols = [col for col in frame.columns if col.startswith(prefix)]
    return sorted(cols, key=lambda name: int(name.split("lag")[-1]))


def _normalized_timestamp(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series).dt.strftime(TIME_FORMAT)


def export_components(args: argparse.Namespace) -> pd.DataFrame:
    estimates_path = _absolute_path(args.estimates)
    raw_path = _absolute_path(args.raw_dataset)
    output_path = _absolute_path(args.output)

    estimates = pd.read_csv(estimates_path)
    estimates["timestamp"] = _normalized_timestamp(estimates["TimeStamp"])
    pi_cols = _pi_columns(estimates, args.edge)
    if not pi_cols:
        raise ValueError(f"No {args.edge} lag probability columns found in {estimates_path}")

    pi = estimates[pi_cols].to_numpy(dtype=np.float64)
    lag_axis = np.arange(pi.shape[1], dtype=np.float64)
    p = 1.0 - pi[:, 0]
    d_hat = (pi * lag_axis[None, :]).sum(axis=1)
    positive_mass = np.clip(p, 1e-12, None)
    m = (pi[:, 1:] * lag_axis[None, 1:]).sum(axis=1) / positive_mass
    m = np.where(p > 1e-12, m, 0.0)

    estimate_components = pd.DataFrame(
        {
            "timestamp": estimates["timestamp"],
            "p": p,
            "m": m,
            "d_hat": d_hat,
        }
    )

    raw = pd.read_csv(raw_path)
    required = [args.time_col, args.split_col, args.lag_col]
    missing = [col for col in required if col not in raw.columns]
    if missing:
        raise ValueError(f"Raw dataset is missing columns: {', '.join(missing)}")

    raw_test = raw.loc[raw[args.split_col] == args.test_split].copy()
    raw_test["timestamp"] = _normalized_timestamp(raw_test[args.time_col])
    raw_keep = pd.DataFrame(
        {
            "timestamp": raw_test["timestamp"],
            "d_true": raw_test[args.lag_col].astype(int),
        }
    )

    if "inject_flag" in raw_test.columns:
        raw_keep["in_block"] = raw_test["inject_flag"].astype(int)
    else:
        raw_keep["in_block"] = raw_keep["d_true"].gt(0).astype(int)

    if "segment_id" in raw_test.columns:
        raw_keep["block_id"] = np.where(raw_keep["in_block"].to_numpy() > 0, raw_test["segment_id"].astype(int), -1)
    else:
        raw_keep["block_id"] = -1

    if "segment_dmax_gt" in raw_test.columns:
        raw_keep["dmax"] = np.where(raw_keep["in_block"].to_numpy() > 0, raw_test["segment_dmax_gt"].fillna(0).astype(int), 0)
    elif "bump_dmax_gt" in raw_test.columns:
        raw_keep["dmax"] = np.where(raw_keep["in_block"].to_numpy() > 0, raw_test["bump_dmax_gt"].fillna(0).astype(int), 0)
    else:
        raw_keep["dmax"] = 0

    joined = raw_keep.merge(estimate_components, on="timestamp", how="inner")
    joined["is_positive"] = joined["d_true"].gt(0).astype(int)
    joined = joined[
        [
            "timestamp",
            "block_id",
            "in_block",
            "dmax",
            "d_true",
            "is_positive",
            "p",
            "m",
            "d_hat",
        ]
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joined.to_csv(output_path, index=False)
    return joined


def main() -> None:
    args = parse_args()
    joined = export_components(args)
    print(f"Wrote {len(joined)} rows to {_absolute_path(args.output)}")


if __name__ == "__main__":
    main()
