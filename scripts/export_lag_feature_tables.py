#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


EDGE_PREFIX = "stage1_to_stage2"
CANDIDATE_A = 0.7549
CANDIDATE_B = 0.2160
LOCALIZATION_A = 0.0550
LOCALIZATION_B = 0.4635
TIME_FORMAT = "%Y-%m-%d %H:%M"


def _path(text: str | Path) -> Path:
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _raw_split_lookup(raw_csv: Path, split_name: str) -> pd.DataFrame:
    raw = pd.read_csv(raw_csv)
    raw["TimeStamp"] = pd.to_datetime(raw["TimeStamp"])
    part = raw.loc[raw["split"].astype(str) == split_name].sort_values("TimeStamp").reset_index(drop=True).copy()
    part = part.reset_index().rename(columns={"index": "raw_row_index"})
    lag_gt = part.get("lag_gt", pd.Series(np.zeros(len(part)), index=part.index)).fillna(0).astype(float)
    in_block = (
        part["inject_flag"].fillna(0).astype(int)
        if "inject_flag" in part.columns
        else lag_gt.gt(0).astype(int)
    )
    if "segment_dmax_gt" in part.columns:
        dmax = part["segment_dmax_gt"].fillna(0).astype(float)
    elif "bump_dmax_gt" in part.columns:
        dmax = part["bump_dmax_gt"].fillna(0).astype(float)
    else:
        dmax = lag_gt.clip(lower=0.0)
    out = pd.DataFrame(
        {
            "timestamp": part["TimeStamp"].dt.strftime(TIME_FORMAT),
            "raw_row_index": part["raw_row_index"].astype(int),
            "segment_id": part.get("segment_id", pd.Series(np.arange(len(part)), index=part.index)).fillna(-1).astype(int),
            "in_block": in_block.astype(int),
            "d_true": lag_gt,
            "dmax": dmax,
            "original_split": split_name,
        }
    )
    out["block_id"] = np.where(out["in_block"].to_numpy(dtype=int) > 0, out["segment_id"].to_numpy(dtype=int), -1)
    for optional_col in ["bump_dmax_gt", "segment_dmax_gt", "g_stage1_to_stage2", "bump_type", "bump_shape"]:
        if optional_col in part.columns:
            out[optional_col] = part[optional_col].to_numpy()
    return out


def _estimate_frame(eval_root: Path, split_name: str, edge: str) -> pd.DataFrame:
    path = eval_root / split_name / "test_delay_estimates.csv"
    frame = pd.read_csv(path)
    prefix = f"{edge}_pred_pi_lag"
    pi_cols = sorted(
        [col for col in frame.columns if col.startswith(prefix)],
        key=lambda name: int(name.split("lag")[-1]),
    )
    if not pi_cols:
        raise ValueError(f"No lag probability columns found for edge {edge!r} in {path}")
    expected_col = f"{edge}_pred_expected_lag"
    argmax_col = f"{edge}_pred_argmax_lag"
    out = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(frame["TimeStamp"]).dt.strftime(TIME_FORMAT),
            "expected_lag": pd.to_numeric(frame[expected_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64),
            "argmax_lag": pd.to_numeric(frame[argmax_col], errors="coerce").fillna(0).to_numpy(dtype=np.int64),
        }
    )
    pi = frame[pi_cols].to_numpy(dtype=np.float64)
    pi = np.where(np.isfinite(pi), pi, 0.0)
    out["nonzero_prob"] = np.clip(1.0 - pi[:, 0], 0.0, 1.0)
    out["max_prob"] = pi.max(axis=1)
    out["max_positive_prob"] = pi[:, 1:].max(axis=1) if pi.shape[1] > 1 else np.zeros(len(pi), dtype=np.float64)
    sorted_pi = np.sort(pi, axis=1)
    out["top1_top2_margin"] = sorted_pi[:, -1] - sorted_pi[:, -2] if pi.shape[1] >= 2 else sorted_pi[:, -1]
    entropy = -(pi * np.log(np.clip(pi, 1e-12, 1.0))).sum(axis=1)
    out["entropy"] = entropy
    return out


def _percentile_rank_by_segment(frame: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    out = np.zeros(len(frame), dtype=np.float64)
    for _, idx in frame.groupby("segment_id", sort=False).groups.items():
        idx_arr = frame.index.get_indexer(idx)
        out[idx_arr] = (
            pd.Series(values[idx_arr])
            .rank(method="average", pct=True)
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
        )
    return out


def _segment_zscore(frame: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    out = np.zeros(len(frame), dtype=np.float64)
    for _, idx in frame.groupby("segment_id", sort=False).groups.items():
        idx_arr = frame.index.get_indexer(idx)
        local = values[idx_arr]
        mean = float(np.nanmean(local)) if local.size else 0.0
        std = float(np.nanstd(local)) if local.size else 0.0
        if std <= 1e-12:
            out[idx_arr] = 0.0
        else:
            out[idx_arr] = (local - mean) / std
    return out


def _rolling_mean_by_segment(frame: pd.DataFrame, values: np.ndarray, window: int) -> np.ndarray:
    out = np.zeros(len(frame), dtype=np.float64)
    for _, idx in frame.groupby("segment_id", sort=False).groups.items():
        idx_arr = frame.index.get_indexer(idx)
        out[idx_arr] = (
            pd.Series(values[idx_arr])
            .rolling(window=int(window), center=True, min_periods=1)
            .mean()
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
        )
    return out


def _build_feature_table(raw_csv: Path, eval_root: Path, split_name: str, edge: str) -> pd.DataFrame:
    raw_lookup = _raw_split_lookup(raw_csv, split_name=split_name)
    estimates = _estimate_frame(eval_root, split_name=split_name, edge=edge)
    joined = raw_lookup.merge(estimates, on="timestamp", how="inner")
    joined = joined.sort_values(["segment_id", "timestamp", "raw_row_index"]).reset_index(drop=True)
    joined.insert(0, "split", split_name)
    joined.insert(1, "source_split", split_name)
    joined["t"] = joined.groupby("segment_id").cumcount().astype(int)
    joined["is_positive"] = (joined["d_true"].to_numpy(dtype=np.float64) > 0).astype(int)

    p = joined["nonzero_prob"].to_numpy(dtype=np.float64)
    expected = joined["expected_lag"].to_numpy(dtype=np.float64)
    raw_m = np.where(p > 1e-8, expected / np.maximum(p, 1e-8), 0.0)
    joined["p"] = p
    joined["m"] = raw_m
    joined["d_hat_raw"] = expected
    joined["pred_argmax_lag"] = joined["argmax_lag"].to_numpy(dtype=np.int64)
    joined["raw_m"] = raw_m
    joined["positive_margin"] = 2.0 * p - 1.0
    joined["argmax_is_positive"] = (joined["argmax_lag"].to_numpy(dtype=np.int64) > 0).astype(int)

    rank = _percentile_rank_by_segment(joined, raw_m)
    joined["candidate_score_model_rank"] = rank
    joined["candidate_score_heuristic"] = np.clip(0.5 + 0.5 * joined["positive_margin"].to_numpy(dtype=np.float64), 0.0, 1.0)
    joined["candidate_score_model"] = np.clip(p * 0.15, 0.0, 1.0)
    joined["candidate_score"] = np.clip(CANDIDATE_A * rank + CANDIDATE_B, 0.0, 1.0)
    joined["localization_score"] = np.clip(LOCALIZATION_A * rank + LOCALIZATION_B, 0.0, 1.0)
    joined["localization_mask"] = (joined["candidate_score"].to_numpy(dtype=np.float64) >= 0.25).astype(int)
    joined["segment_z"] = _segment_zscore(joined, raw_m)
    joined["residual_score"] = expected - _rolling_mean_by_segment(joined, expected, window=5)

    ordered_cols = [
        "split",
        "source_split",
        "timestamp",
        "raw_row_index",
        "segment_id",
        "t",
        "block_id",
        "dmax",
        "in_block",
        "d_true",
        "is_positive",
        "p",
        "m",
        "d_hat_raw",
        "pred_argmax_lag",
        "original_split",
        "raw_m",
        "expected_lag",
        "argmax_lag",
        "nonzero_prob",
        "max_prob",
        "max_positive_prob",
        "entropy",
        "top1_top2_margin",
        "positive_margin",
        "argmax_is_positive",
        "segment_z",
        "residual_score",
        "candidate_score_heuristic",
        "candidate_score_model",
        "candidate_score_model_rank",
        "candidate_score",
        "localization_score",
        "localization_mask",
    ]
    optional = [col for col in joined.columns if col not in ordered_cols]
    return joined[[col for col in ordered_cols if col in joined.columns] + optional].copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export q40/r46b-compatible lag feature tables from split delay estimates.")
    parser.add_argument("--raw-csv", required=True)
    parser.add_argument("--eval-root", required=True, help="Root produced by run_transfer_checkpoint_eval.py")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--edge", default=EDGE_PREFIX)
    parser.add_argument("--splits", default="train,val,test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_csv = _path(args.raw_csv)
    eval_root = _path(args.eval_root)
    out_dir = _path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, Any]] = []
    for split_name in [part.strip() for part in str(args.splits).split(",") if part.strip()]:
        table = _build_feature_table(raw_csv=raw_csv, eval_root=eval_root, split_name=split_name, edge=str(args.edge))
        table.to_csv(out_dir / f"{split_name}_feature_timeseries.csv", index=False)
        summary_rows.append(
            {
                "split": split_name,
                "n_rows": int(len(table)),
                "n_positive": int((table["d_true"].to_numpy(dtype=np.float64) > 0).sum()),
                "candidate_score_min": float(table["candidate_score"].min()) if len(table) else float("nan"),
                "candidate_score_max": float(table["candidate_score"].max()) if len(table) else float("nan"),
                "localization_score_min": float(table["localization_score"].min()) if len(table) else float("nan"),
                "localization_score_max": float(table["localization_score"].max()) if len(table) else float("nan"),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "feature_export_summary.csv", index=False)
    (out_dir / "feature_export_summary.json").write_text(
        json.dumps(
            {
                "raw_csv": raw_csv.as_posix(),
                "eval_root": eval_root.as_posix(),
                "edge": str(args.edge),
                "splits": summary_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(summary.to_csv(index=False, float_format="%.6f"))
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
