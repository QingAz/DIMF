#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Combine raw/train.csv, raw/val.csv, raw/test.csv into one predefined-split CSV."
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing train.csv/val.csv/test.csv")
    parser.add_argument("--output", type=Path, required=True, help="Destination CSV path")
    parser.add_argument("--time-col", default="TimeStamp", help="Timestamp column name")
    parser.add_argument("--split-col", default="split", help="Split column name")
    return parser.parse_args()


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(str(path)))


def main():
    args = parse_args()
    input_dir = _absolute_path(args.input_dir)
    output_path = _absolute_path(args.output)

    frames = []
    for split_name in ["train", "val", "test"]:
        csv_path = input_dir / f"{split_name}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing split CSV: {csv_path}")
        frame = pd.read_csv(csv_path)
        if args.split_col not in frame.columns:
            frame[args.split_col] = split_name
        else:
            frame[args.split_col] = split_name
        frames.append(frame)

    merged = pd.concat(frames, axis=0, ignore_index=True)
    if args.time_col not in merged.columns:
        raise ValueError(f"Missing time column '{args.time_col}' in merged data")

    merged[args.time_col] = pd.to_datetime(merged[args.time_col])
    merged = merged.sort_values(args.time_col).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)

    print(f"Wrote merged predefined-split CSV to {output_path}")
    print(f"Rows: {len(merged)}")
    print(
        "Split counts: %s"
        % (
            merged[args.split_col]
            .value_counts(sort=False)
            .reindex(["train", "val", "test"])
            .fillna(0)
            .astype(int)
            .to_dict()
        )
    )


if __name__ == "__main__":
    main()
