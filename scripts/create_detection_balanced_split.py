#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a detection-balanced split by moving long block-out segments into validation."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input raw-gap CSV")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV")
    parser.add_argument("--summary", type=Path, default=None, help="Optional summary JSON")
    parser.add_argument("--time-col", default="TimeStamp")
    parser.add_argument("--split-col", default="split")
    parser.add_argument("--segment-col", default="segment_id")
    parser.add_argument("--inject-col", default="inject_flag")
    parser.add_argument("--lag-col", default="lag_gt")
    parser.add_argument("--val-negative-segment", type=int, default=28)
    return parser.parse_args()


def _sample_count(rows: int, history_steps: int = 96, horizon_steps: int = 4) -> int:
    return max(int(rows) - history_steps - horizon_steps + 1, 0)


def _split_summary(df: pd.DataFrame, split_col: str, segment_col: str, inject_col: str, lag_col: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for split_name, part in df.groupby(split_col, sort=True):
        block_in_rows = int((part[inject_col] > 0).sum())
        block_out_rows = int((part[inject_col] == 0).sum())
        lag_pos_rows = int((part[lag_col] > 0).sum())
        valid_block_in_samples = 0
        valid_block_out_samples = 0
        for _, seg in part.groupby(segment_col, sort=True):
            n_samples = _sample_count(len(seg))
            if int(seg[inject_col].max()) > 0:
                valid_block_in_samples += n_samples
            else:
                valid_block_out_samples += n_samples
        rows.append(
            {
                "split": split_name,
                "rows": int(len(part)),
                "block_in_rows": block_in_rows,
                "block_out_rows": block_out_rows,
                "lag_positive_rows": lag_pos_rows,
                "valid_block_in_samples_approx": int(valid_block_in_samples),
                "valid_block_out_samples_approx": int(valid_block_out_samples),
                "n_block_in_segments": int(part.loc[part[inject_col] > 0, segment_col].nunique()),
                "n_block_out_segments": int(part.loc[part[inject_col] == 0, segment_col].nunique()),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    input_path = Path(os.path.abspath(str(args.input)))
    output_path = Path(os.path.abspath(str(args.output)))
    summary_path = Path(os.path.abspath(str(args.summary))) if args.summary else output_path.with_suffix(output_path.suffix + ".summary.json")

    df = pd.read_csv(input_path)
    required = [args.time_col, args.split_col, args.segment_col, args.inject_col, args.lag_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Input is missing columns: {', '.join(missing)}")

    segment_id = int(args.val_negative_segment)
    seg_mask = df[args.segment_col].astype(int).eq(segment_id)
    if not seg_mask.any():
        raise ValueError(f"Segment {segment_id} not found")
    seg = df.loc[seg_mask]
    if int(seg[args.inject_col].max()) != 0:
        raise ValueError(f"Segment {segment_id} is not a block-out negative segment")
    if _sample_count(len(seg)) <= 0:
        raise ValueError(f"Segment {segment_id} is too short to produce validation samples")

    out = df.copy()
    out.loc[seg_mask, args.split_col] = "val"

    summary = {
        "input": input_path.as_posix(),
        "output": output_path.as_posix(),
        "moved_to_val_negative_segment": segment_id,
        "moved_rows": int(seg_mask.sum()),
        "before": _split_summary(df, args.split_col, args.segment_col, args.inject_col, args.lag_col),
        "after": _split_summary(out, args.split_col, args.segment_col, args.inject_col, args.lag_col),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
