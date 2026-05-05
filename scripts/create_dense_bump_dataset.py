#!/usr/bin/env python3

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Increase the positive lag coverage of a local bump dataset to a target density."
    )
    parser.add_argument("--input-full", type=Path, required=True)
    parser.add_argument("--output-full", type=Path, required=True)
    parser.add_argument("--output-rawgap", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--time-col", default="TimeStamp")
    parser.add_argument("--split-col", default="split")
    parser.add_argument("--segment-col", default="segment_id")
    parser.add_argument("--lag-col", default="lag_gt")
    parser.add_argument("--inject-col", default="inject_flag")
    parser.add_argument("--interpolated-col", default="is_interpolated")
    parser.add_argument("--lag-binary-col", default="lag_binary_gt")
    parser.add_argument("--segment-dmax-col", default="segment_dmax_gt")
    parser.add_argument("--bump-dmax-col", default="bump_dmax_gt")
    parser.add_argument("--graph-col", default="g_stage1_to_stage2")
    parser.add_argument("--target-positive-fraction", type=float, default=0.80)
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


def _summarize(frame: pd.DataFrame, split_col: str, lag_col: str) -> dict:
    rows = {}
    for split_name, part in frame.groupby(split_col, sort=False):
        pos = int(part[lag_col].fillna(0).astype(int).gt(0).sum())
        total = int(len(part))
        rows[str(split_name)] = {
            "rows": total,
            "positive_rows": pos,
            "positive_fraction": float(pos / total) if total else 0.0,
        }
    total_pos = int(frame[lag_col].fillna(0).astype(int).gt(0).sum())
    total_rows = int(len(frame))
    rows["all"] = {
        "rows": total_rows,
        "positive_rows": total_pos,
        "positive_fraction": float(total_pos / total_rows) if total_rows else 0.0,
    }
    return rows


def _segment_dmax(seg: pd.DataFrame, lag_col: str, segment_dmax_col: str, bump_dmax_col: str) -> int:
    candidates = []
    for col in [segment_dmax_col, bump_dmax_col, lag_col]:
        if col in seg.columns:
            values = pd.to_numeric(seg[col], errors="coerce").fillna(0).astype(int)
            candidates.append(int(values.max()))
    return max(candidates) if candidates else 0


def _build_candidates(
    frame: pd.DataFrame,
    split_col: str,
    segment_col: str,
    lag_col: str,
    segment_dmax_col: str,
    bump_dmax_col: str,
) -> dict:
    out = {}
    for split_name, split_part in frame.groupby(split_col, sort=False):
        candidates = []
        for _, seg in split_part.groupby(segment_col, sort=False):
            seg_idx = seg.index.to_numpy(dtype=np.int64)
            lag_values = pd.to_numeric(seg[lag_col], errors="coerce").fillna(0).astype(int).to_numpy()
            pos_local = np.flatnonzero(lag_values > 0)
            if pos_local.size == 0:
                continue
            seg_dmax = _segment_dmax(seg, lag_col, segment_dmax_col, bump_dmax_col)
            zero_local = np.flatnonzero(lag_values <= 0)
            for local_idx in zero_local.tolist():
                nearest_pos = min(pos_local.tolist(), key=lambda pos_idx: abs(pos_idx - local_idx))
                assigned_lag = int(lag_values[nearest_pos])
                if assigned_lag <= 0:
                    assigned_lag = max(seg_dmax, 1)
                candidates.append(
                    {
                        "row_index": int(seg_idx[local_idx]),
                        "distance": int(abs(nearest_pos - local_idx)),
                        "assigned_lag": int(assigned_lag),
                        "assigned_dmax": int(max(seg_dmax, assigned_lag)),
                    }
                )
        candidates.sort(key=lambda row: (row["distance"], row["row_index"]))
        out[str(split_name)] = candidates
    return out


def _apply_density(
    frame: pd.DataFrame,
    split_col: str,
    lag_col: str,
    inject_col: str,
    lag_binary_col: str,
    segment_dmax_col: str,
    bump_dmax_col: str,
    graph_col: str,
    target_positive_fraction: float,
    candidates_by_split: dict,
) -> pd.DataFrame:
    out = frame.copy()
    for split_name, split_part in out.groupby(split_col, sort=False):
        split_mask = out[split_col].astype(str).eq(str(split_name))
        lag_values = pd.to_numeric(out.loc[split_mask, lag_col], errors="coerce").fillna(0).astype(int)
        current_pos = int(lag_values.gt(0).sum())
        total_rows = int(split_mask.sum())
        target_pos = int(min(total_rows, max(current_pos, int(math.ceil(total_rows * target_positive_fraction)))))
        need = max(target_pos - current_pos, 0)
        for row in candidates_by_split.get(str(split_name), [])[:need]:
            idx = int(row["row_index"])
            assigned_lag = int(row["assigned_lag"])
            assigned_dmax = int(row["assigned_dmax"])
            out.at[idx, lag_col] = assigned_lag
            if inject_col in out.columns:
                out.at[idx, inject_col] = 1
            if lag_binary_col in out.columns:
                out.at[idx, lag_binary_col] = 1
            if segment_dmax_col in out.columns:
                out.at[idx, segment_dmax_col] = assigned_dmax
            if bump_dmax_col in out.columns:
                out.at[idx, bump_dmax_col] = assigned_dmax
            if graph_col in out.columns:
                out.at[idx, graph_col] = 1
    return out


def main() -> None:
    args = parse_args()
    input_full = _absolute_path(args.input_full)
    output_full = _absolute_path(args.output_full)
    output_rawgap = _absolute_path(args.output_rawgap)
    summary_json = (
        _absolute_path(args.summary_json)
        if args.summary_json is not None
        else output_rawgap.with_suffix(output_rawgap.suffix + ".summary.json")
    )

    if not (0.0 < float(args.target_positive_fraction) <= 1.0):
        raise ValueError("--target-positive-fraction must be in (0, 1].")

    df = pd.read_csv(input_full)
    required = [args.time_col, args.split_col, args.segment_col, args.lag_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError("Missing required columns: %s" % ", ".join(missing))

    df[args.time_col] = pd.to_datetime(df[args.time_col])
    df = df.sort_values([args.split_col, args.segment_col, args.time_col]).reset_index(drop=True)

    before_full = _summarize(df, args.split_col, args.lag_col)
    candidates_by_split = _build_candidates(
        frame=df,
        split_col=args.split_col,
        segment_col=args.segment_col,
        lag_col=args.lag_col,
        segment_dmax_col=args.segment_dmax_col,
        bump_dmax_col=args.bump_dmax_col,
    )
    dense_df = _apply_density(
        frame=df,
        split_col=args.split_col,
        lag_col=args.lag_col,
        inject_col=args.inject_col,
        lag_binary_col=args.lag_binary_col,
        segment_dmax_col=args.segment_dmax_col,
        bump_dmax_col=args.bump_dmax_col,
        graph_col=args.graph_col,
        target_positive_fraction=float(args.target_positive_fraction),
        candidates_by_split=candidates_by_split,
    )
    after_full = _summarize(dense_df, args.split_col, args.lag_col)

    output_full.parent.mkdir(parents=True, exist_ok=True)
    dense_df.to_csv(output_full, index=False)

    if args.interpolated_col not in dense_df.columns:
        raise ValueError("Missing interpolated marker column: %s" % args.interpolated_col)
    interp_mask = _to_bool_mask(dense_df[args.interpolated_col])
    raw_df = dense_df.loc[~interp_mask].copy().sort_values(args.time_col).reset_index(drop=True)
    raw_df.to_csv(output_rawgap, index=False)
    after_raw = _summarize(raw_df, args.split_col, args.lag_col)

    summary = {
        "input_full": input_full.as_posix(),
        "output_full": output_full.as_posix(),
        "output_rawgap": output_rawgap.as_posix(),
        "target_positive_fraction": float(args.target_positive_fraction),
        "before_full": before_full,
        "after_full": after_full,
        "after_rawgap": after_raw,
        "rows_dropped_as_interpolated": int(interp_mask.sum()),
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
