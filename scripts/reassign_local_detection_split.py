#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reassign predefined splits by global segment-id ranges for local-detection experiments."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input raw-gap CSV")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV with reassigned split column")
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional summary JSON path; defaults next to output",
    )
    parser.add_argument("--segment-col", default="segment_id")
    parser.add_argument("--split-col", default="split")
    parser.add_argument("--inject-col", default="inject_flag")
    parser.add_argument("--dmax-col", default="segment_dmax_gt")
    parser.add_argument("--train-max-segment", type=int, required=True)
    parser.add_argument("--val-max-segment", type=int, required=True)
    return parser.parse_args()


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(str(path)))


def _assign_split(segment_id: int, train_max_segment: int, val_max_segment: int) -> str:
    if segment_id <= train_max_segment:
        return "train"
    if segment_id <= val_max_segment:
        return "val"
    return "test"


def main() -> None:
    args = parse_args()
    input_path = _absolute_path(args.input)
    output_path = _absolute_path(args.output)
    summary_json_path = (
        _absolute_path(args.summary_json)
        if args.summary_json is not None
        else output_path.with_suffix(output_path.suffix + ".summary.json")
    )

    df = pd.read_csv(input_path)
    df[args.segment_col] = df[args.segment_col].fillna(-1).astype(int)
    df[args.split_col] = df[args.segment_col].map(
        lambda seg: _assign_split(int(seg), int(args.train_max_segment), int(args.val_max_segment))
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    grouped = (
        df.groupby([args.split_col, args.segment_col], sort=True)
        .agg(
            n_rows=(args.segment_col, "size"),
            inject_max=(args.inject_col, "max"),
            dmax_mode=(args.dmax_col, lambda s: int(pd.Series(s).fillna(0).astype(int).mode().iloc[0])),
        )
        .reset_index()
    )
    grouped["inject_max"] = grouped["inject_max"].fillna(0).astype(int).clip(lower=0, upper=1)

    split_rows = []
    for split_name, part in grouped.groupby(args.split_col, sort=False):
        pos_part = part.loc[part["inject_max"] > 0]
        neg_part = part.loc[part["inject_max"] == 0]
        dmax_counts = {
            str(int(k)): int(v)
            for k, v in pos_part["dmax_mode"].value_counts().sort_index().items()
        }
        split_rows.append(
            {
                "split": split_name,
                "n_rows": int(df.loc[df[args.split_col] == split_name].shape[0]),
                "n_segments": int(part.shape[0]),
                "n_positive_segments": int(pos_part.shape[0]),
                "n_negative_segments": int(neg_part.shape[0]),
                "positive_dmax_counts": dmax_counts,
                "segment_ids": [int(x) for x in part[args.segment_col].tolist()],
            }
        )

    report = {
        "input": input_path.as_posix(),
        "output": output_path.as_posix(),
        "train_max_segment": int(args.train_max_segment),
        "val_max_segment": int(args.val_max_segment),
        "splits": split_rows,
    }
    summary_json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
