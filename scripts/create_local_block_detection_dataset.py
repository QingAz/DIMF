#!/usr/bin/env python3

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataprocess import (
    _regularize_split_with_gap_policy,
    _sample_indices_from_regularized_split,
    _split_predefined_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a local-block detection dataset with synthetic local windows and center masks."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input raw-gap CSV")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV with sample-center mask")
    parser.add_argument("--summary-json", type=Path, default=None, help="Optional summary JSON path")
    parser.add_argument("--summary-csv", type=Path, default=None, help="Optional summary CSV path")
    parser.add_argument("--time-col", default="TimeStamp")
    parser.add_argument("--split-col", default="split")
    parser.add_argument("--segment-col", default="segment_id")
    parser.add_argument("--inject-col", default="inject_flag")
    parser.add_argument("--lag-col", default="lag_gt")
    parser.add_argument("--sample-keep-col", default="sample_center_keep")
    parser.add_argument("--local-k", type=int, default=8, help="Number of rows kept on each side of a block")
    parser.add_argument(
        "--pos-neg-ratio",
        default="1:1",
        help="Positive-to-negative target ratio for kept centers, e.g. 1:1 or 1:2",
    )
    parser.add_argument("--history-steps", type=int, default=96)
    parser.add_argument("--horizon-steps", type=int, default=4)
    parser.add_argument("--collection-interval-min", type=int, default=15)
    parser.add_argument("--gap-break-min", type=int, default=120)
    parser.add_argument("--gap-fill-min", type=int, default=60)
    parser.add_argument("--fillna", default="ffill")
    parser.add_argument("--use-delta-t", action="store_true", default=True)
    parser.add_argument("--no-use-delta-t", dest="use_delta_t", action="store_false")
    return parser.parse_args()


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(str(path)))


def _parse_ratio(text: str) -> Tuple[int, int]:
    parts = str(text).split(":")
    if len(parts) != 2:
        raise ValueError(f"pos-neg ratio must look like '1:1' or '1:2', got {text!r}")
    left = int(parts[0])
    right = int(parts[1])
    if left <= 0 or right <= 0:
        raise ValueError("ratio values must be positive")
    return left, right


def _select_evenly(indices: List[int], keep_count: int) -> List[int]:
    if keep_count <= 0 or not indices:
        return []
    if keep_count >= len(indices):
        return list(indices)
    positions = np.linspace(0, len(indices) - 1, num=keep_count)
    chosen = np.unique(np.rint(positions).astype(np.int64))
    if len(chosen) < keep_count:
        extras = [idx for idx in range(len(indices)) if idx not in set(chosen)]
        need = keep_count - len(chosen)
        chosen = np.concatenate([chosen, np.asarray(extras[:need], dtype=np.int64)])
    return [indices[int(pos)] for pos in sorted(set(chosen.tolist()))[:keep_count]]


def _segment_purity_check(df: pd.DataFrame, split_col: str, segment_col: str, inject_col: str) -> None:
    purity = (
        df.groupby([split_col, segment_col])[inject_col]
        .nunique(dropna=False)
        .reset_index(name="n_unique")
    )
    bad = purity.loc[purity["n_unique"] > 1]
    if not bad.empty:
        raise ValueError(
            "Expected pure positive/negative segments before creating local detection mask; "
            f"found mixed labels in {len(bad)} segment(s)"
        )


def _build_local_windows(
    df: pd.DataFrame,
    time_col: str,
    split_col: str,
    segment_col: str,
    inject_col: str,
    lag_col: str,
    sample_keep_col: str,
    local_k: int,
    pos_neg_ratio: Tuple[int, int],
    history_steps: int,
    horizon_steps: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ratio_pos, ratio_neg = pos_neg_ratio
    output_parts: List[pd.DataFrame] = []
    summary_rows: List[Dict[str, object]] = []
    new_segment_id = 0

    for split_name in ["train", "val", "test"]:
        split_df = (
            df.loc[df[split_col] == split_name]
            .sort_values(time_col)
            .reset_index()
            .rename(columns={"index": "_global_row"})
        )
        pos_segments = []
        for segment_id, seg in split_df.groupby(segment_col, sort=True):
            segment_label = int(seg[inject_col].max())
            if segment_label > 0:
                pos_segments.append((int(segment_id), seg.copy()))

        for block_rank, (segment_id, seg) in enumerate(pos_segments, start=1):
            seg_local_positions = seg.index.to_numpy(dtype=np.int64)
            start_local = int(seg_local_positions.min())
            end_local = int(seg_local_positions.max())
            left = max(0, start_local - int(local_k) - max(int(history_steps) - 1, 0))
            right = min(len(split_df) - 1, end_local + int(local_k) + int(horizon_steps))
            window = split_df.iloc[left : right + 1].copy().reset_index(drop=True)

            candidate_left = max(0, start_local - int(local_k) - left)
            candidate_right = min(len(window) - 1, end_local + int(local_k) - left)
            window["source_segment_id"] = window[segment_col].astype(int)
            window["target_block_segment_id"] = int(segment_id)
            target_dmax = int(seg.get("segment_dmax_gt", pd.Series([0])).fillna(0).astype(int).mode().iloc[0])
            block_width = int(len(seg))

            original_positions = split_df.iloc[left : right + 1].index.to_numpy(dtype=np.int64)
            target_block_mask = (original_positions >= start_local) & (original_positions <= end_local)

            # 每个 synthetic local window 只保留一个目标 block 的标签语义。
            window[inject_col] = target_block_mask.astype(np.int64)
            if lag_col in window.columns:
                window[lag_col] = np.where(target_block_mask, window[lag_col].fillna(0).astype(int), 0)
            if "lag_binary_gt" in window.columns:
                window["lag_binary_gt"] = window[lag_col].gt(0).astype(int)
            if "segment_dmax_gt" in window.columns:
                window["segment_dmax_gt"] = target_dmax
            if "bump_dmax_gt" in window.columns:
                window["bump_dmax_gt"] = target_dmax
            window["target_block_width"] = block_width
            window["local_window_flag"] = 1
            window[sample_keep_col] = 0
            window[segment_col] = int(new_segment_id)

            candidate = window.iloc[candidate_left : candidate_right + 1].copy()
            pos_rows = candidate.index[candidate[inject_col] > 0].astype(int).tolist()
            neg_rows = candidate.index[candidate[inject_col] == 0].astype(int).tolist()
            target_pos = min(len(pos_rows), int(np.ceil(len(neg_rows) * float(ratio_pos) / float(ratio_neg)))) if neg_rows else 0
            selected_pos = _select_evenly(pos_rows, target_pos)
            selected_keep = sorted(set(selected_pos + neg_rows))
            if selected_keep:
                window.loc[selected_keep, sample_keep_col] = 1

            summary_rows.append(
                {
                    "split": split_name,
                    "segment_id": int(new_segment_id),
                    "source_block_segment_id": segment_id,
                    "block_rank_in_split": block_rank,
                    "block_start_time": pd.to_datetime(seg[time_col].iloc[0]).strftime("%Y-%m-%d %H:%M"),
                    "block_end_time": pd.to_datetime(seg[time_col].iloc[-1]).strftime("%Y-%m-%d %H:%M"),
                    "block_rows": int(len(seg)),
                    "window_rows": int(len(window)),
                    "history_context_rows": int(max(0, candidate_left)),
                    "future_context_rows": int(max(0, len(window) - 1 - candidate_right)),
                    "window_negative_rows": int(len(neg_rows)),
                    "window_positive_rows": int(len(pos_rows)),
                    "selected_negative_centers_raw": int(len(neg_rows)),
                    "selected_positive_centers_raw": int(len(selected_pos)),
                    "selected_total_centers_raw": int(len(selected_keep)),
                    "target_dmax": target_dmax,
                    "target_block_width": block_width,
                }
            )
            output_parts.append(window)
            new_segment_id += 1

    out = pd.concat(output_parts, ignore_index=True).sort_values([split_col, segment_col, time_col]).reset_index(drop=True)
    return out, pd.DataFrame(summary_rows)


def _exact_split_audit(
    df: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    raw_parts = _split_predefined_rows(df, time_col=args.time_col, split_col=args.split_col)
    rows: List[Dict[str, object]] = []
    for split_name, raw_part in zip(["train", "val", "test"], raw_parts):
        reg = _regularize_split_with_gap_policy(
            df_part=raw_part,
            time_col=args.time_col,
            collection_interval_min=int(args.collection_interval_min),
            gap_break_min=int(args.gap_break_min),
            gap_fill_min=int(args.gap_fill_min),
            fillna=str(args.fillna),
            use_delta_t=bool(args.use_delta_t),
            sample_keep_col=str(args.sample_keep_col),
            respect_existing_segment_id=True,
        )
        sample_indices = _sample_indices_from_regularized_split(
            reg,
            time_col=args.time_col,
            history_steps=int(args.history_steps),
            horizon_steps=int(args.horizon_steps),
            collection_interval_min=int(args.collection_interval_min),
            sample_keep_col=str(args.sample_keep_col),
        )
        centers = reg.iloc[sample_indices].copy()
        centers["label"] = centers[args.inject_col].fillna(0).astype(int)
        pos = int((centers["label"] > 0).sum())
        neg = int((centers["label"] == 0).sum())
        pos_segments = int(centers.loc[centers["label"] > 0, args.segment_col].nunique())
        neg_segments = int(centers.loc[centers["label"] == 0, args.segment_col].nunique())
        rows.append(
            {
                "split": split_name,
                "n_segments": int(centers[args.segment_col].nunique()),
                "n_windows": int(centers.loc[centers["label"] > 0, args.segment_col].nunique()),
                "n_positive": pos,
                "n_negative": neg,
                "pos_neg_ratio": float(pos / max(neg, 1)),
                "n_positive_segments": pos_segments,
                "n_negative_segments": neg_segments,
                "near_negative_ratio": 1.0 if neg > 0 else 0.0,
                "far_negative_ratio": 0.0,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    input_path = _absolute_path(args.input)
    output_path = _absolute_path(args.output)
    summary_json_path = (
        _absolute_path(args.summary_json)
        if args.summary_json is not None
        else output_path.with_suffix(output_path.suffix + ".summary.json")
    )
    summary_csv_path = (
        _absolute_path(args.summary_csv)
        if args.summary_csv is not None
        else output_path.with_suffix(output_path.suffix + ".summary.csv")
    )

    df = pd.read_csv(input_path)
    df[args.time_col] = pd.to_datetime(df[args.time_col])
    df = df.sort_values([args.split_col, args.time_col]).reset_index(drop=True)

    required = [args.time_col, args.split_col, args.segment_col, args.inject_col, args.lag_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {', '.join(missing)}")

    _segment_purity_check(df, split_col=args.split_col, segment_col=args.segment_col, inject_col=args.inject_col)

    out_df, block_summary = _build_local_windows(
        df=df,
        time_col=args.time_col,
        split_col=args.split_col,
        segment_col=args.segment_col,
        inject_col=args.inject_col,
        lag_col=args.lag_col,
        sample_keep_col=args.sample_keep_col,
        local_k=int(args.local_k),
        pos_neg_ratio=_parse_ratio(args.pos_neg_ratio),
        history_steps=int(args.history_steps),
        horizon_steps=int(args.horizon_steps),
    )

    split_summary = _exact_split_audit(out_df, args)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_json_path.parent.mkdir(parents=True, exist_ok=True)
    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)

    out_df.to_csv(output_path, index=False)
    block_summary.to_csv(summary_csv_path, index=False)
    split_summary.to_csv(summary_csv_path.with_name(summary_csv_path.stem + "_split_audit.csv"), index=False)

    report = {
        "input": input_path.as_posix(),
        "output": output_path.as_posix(),
        "local_k": int(args.local_k),
        "pos_neg_ratio": str(args.pos_neg_ratio),
        "sample_keep_col": str(args.sample_keep_col),
        "n_rows": int(len(out_df)),
        "n_kept_centers_raw_rows": int(out_df[args.sample_keep_col].sum()),
        "block_summary_rows": block_summary.to_dict(orient="records"),
        "split_summary_rows": split_summary.to_dict(orient="records"),
    }
    summary_json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(split_summary.to_csv(index=False))
    print(json.dumps({"output": output_path.as_posix(), "summary_json": summary_json_path.as_posix()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
