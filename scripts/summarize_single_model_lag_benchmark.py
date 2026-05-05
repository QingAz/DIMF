#!/usr/bin/env python3

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

TIME_FORMAT = "%Y-%m-%d %H:%M"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize lag benchmark metrics for a single DIMF run."
    )
    parser.add_argument("--estimates", type=Path, required=True, help="Path to test_delay_estimates.csv")
    parser.add_argument("--raw-dataset", type=Path, required=True, help="Raw-gap lagged dataset CSV")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for summary outputs")
    parser.add_argument("--metrics", type=Path, default=None, help="Optional test_metrics.json")
    parser.add_argument("--bump-plan", type=Path, default=None, help="Optional local_bump_plan.json")
    parser.add_argument("--edge", default="stage1_to_stage2", help="Edge name to summarize")
    parser.add_argument("--time-col", default="TimeStamp", help="Timestamp column name")
    parser.add_argument("--split-col", default="split", help="Split column name")
    parser.add_argument("--lag-col", default="lag_gt", help="Ground-truth lag column name")
    return parser.parse_args()


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(str(path)))


def _load_json(path: Path) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_bump_plan(path: Path) -> Dict[int, int]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        plan = json.load(f)
    mapping: Dict[int, int] = {}
    for item in plan:
        if "segment_id" in item and "dmax" in item:
            mapping[int(item["segment_id"])] = int(item["dmax"])
    return mapping


def _pi_columns(frame: pd.DataFrame, edge: str) -> List[str]:
    prefix = f"{edge}_pred_pi_lag"
    cols = [col for col in frame.columns if col.startswith(prefix)]
    return sorted(cols, key=lambda name: int(name.split("lag")[-1]))


def _entropy(values: np.ndarray) -> np.ndarray:
    safe = np.clip(values, 1e-12, None)
    return -(safe * np.log(safe)).sum(axis=1)


def _nonzero_score(frame: pd.DataFrame, pi_cols: List[str]) -> np.ndarray:
    lag0_col = None
    for col in pi_cols:
        if col.endswith("lag0"):
            lag0_col = col
            break
    if lag0_col is None:
        lag0_col = pi_cols[0]
    return 1.0 - frame[lag0_col].to_numpy(dtype=np.float64)


def _pr_curve(scores: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(-scores)
    scores_sorted = scores[order]
    labels_sorted = labels[order]
    tp = 0
    fp = 0
    fn = int(labels_sorted.sum())
    precision = []
    recall = []
    thresholds = []
    last_score = None
    for score, label in zip(scores_sorted, labels_sorted):
        if last_score is None or score != last_score:
            if last_score is not None:
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                precision.append(prec)
                recall.append(rec)
                thresholds.append(last_score)
            last_score = score
        if label:
            tp += 1
            fn -= 1
        else:
            fp += 1
    if last_score is not None:
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision.append(prec)
        recall.append(rec)
        thresholds.append(last_score)
    return (
        np.asarray(precision, dtype=np.float64),
        np.asarray(recall, dtype=np.float64),
        np.asarray(thresholds, dtype=np.float64),
    )


def _auprc(precision: np.ndarray, recall: np.ndarray) -> float:
    if precision.size == 0:
        return 0.0
    order = np.argsort(recall)
    return float(np.trapz(precision[order], recall[order]))


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(bool)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = pd.Series(scores).rank(method="average").to_numpy(dtype=np.float64)
    pos_rank_sum = float(ranks[labels].sum())
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)


def _best_f1(precision: np.ndarray, recall: np.ndarray, thresholds: np.ndarray) -> Dict[str, float]:
    if precision.size == 0:
        return {"threshold": None, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    f1 = 2.0 * precision * recall / (precision + recall + 1e-12)
    idx = int(np.argmax(f1))
    return {
        "threshold": float(thresholds[idx]),
        "precision": float(precision[idx]),
        "recall": float(recall[idx]),
        "f1": float(f1[idx]),
    }


def _false_alarm_rate(scores: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    pred_pos = scores >= threshold
    true_pos = labels.astype(bool)
    fp = int(np.logical_and(pred_pos, ~true_pos).sum())
    neg = int((~true_pos).sum())
    return float(fp) / float(neg) if neg > 0 else 0.0


def _binary_detection_summary(scores: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    labels = labels.astype(np.int64)
    prec, rec, thr = _pr_curve(scores, labels)
    best = _best_f1(prec, rec, thr)
    predicted_positive_ratio = (
        float((scores >= best["threshold"]).mean()) if best["threshold"] is not None else 0.0
    )
    true_pos = labels.astype(bool)
    pos_scores = scores[true_pos]
    neg_scores = scores[~true_pos]
    return {
        "n_positive": int(true_pos.sum()),
        "n_negative": int((~true_pos).sum()),
        "base_rate": float(true_pos.mean()) if true_pos.size else 0.0,
        "auprc": _auprc(prec, rec),
        "auroc": _auroc(scores, labels),
        "best_threshold": best["threshold"],
        "best_precision": best["precision"],
        "best_recall": best["recall"],
        "best_f1": best["f1"],
        "predicted_positive_ratio": predicted_positive_ratio,
        "mean_score_positive": float(pos_scores.mean()) if pos_scores.size else None,
        "mean_score_negative": float(neg_scores.mean()) if neg_scores.size else None,
        "score_margin": float(pos_scores.mean() - neg_scores.mean()) if pos_scores.size and neg_scores.size else None,
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
        raise ValueError(f"No delay distribution columns found for edge {edge} in {estimates_path}")

    expected_col = f"{edge}_pred_expected_lag"
    argmax_col = f"{edge}_pred_argmax_lag"
    required = ["TimeStamp", expected_col, argmax_col] + pi_cols
    missing = [col for col in required if col not in estimates.columns]
    if missing:
        raise ValueError(f"Missing estimate columns in {estimates_path}: {', '.join(missing)}")

    keep_cols = ["TimeStamp", "lag_gt"]
    for optional in ["segment_id", "inject_flag", "bump_dmax_gt", "segment_dmax_gt"]:
        if optional in raw_test.columns:
            keep_cols.append(optional)

    merged = raw_test[keep_cols].merge(estimates[required], on="TimeStamp", how="inner")
    merged = merged.rename(
        columns={
            expected_col: "pred_expected_lag",
            argmax_col: "pred_argmax_lag",
        }
    )
    merged["pred_entropy"] = _entropy(merged[pi_cols].to_numpy(dtype=np.float64))
    merged["pred_nonzero_prob"] = _nonzero_score(merged, pi_cols)
    return merged


def _benchmark_for_frame(frame: pd.DataFrame) -> Dict[str, Any]:
    if frame.empty:
        return {
            "n_rows": 0,
            "n_positive_rows": 0,
            "block_in_expected_lag_mae": None,
            "localization": {
                "auprc": 0.0,
                "best_threshold": None,
                "best_precision": 0.0,
                "best_recall": 0.0,
                "best_f1": 0.0,
            },
            "block_out_false_alarm_rate": 0.0,
            "mean_pred_expected_lag_when_true_zero": None,
        }

    labels = (frame["lag_gt"].to_numpy(dtype=np.int64) > 0).astype(np.int64)
    scores = frame["pred_nonzero_prob"].to_numpy(dtype=np.float64)
    lag_detection = _binary_detection_summary(scores, labels)
    best = {
        "threshold": lag_detection["best_threshold"],
        "precision": lag_detection["best_precision"],
        "recall": lag_detection["best_recall"],
        "f1": lag_detection["best_f1"],
    }
    far = _false_alarm_rate(scores, labels, best["threshold"]) if best["threshold"] is not None else 0.0
    predicted_nonzero_ratio = (
        float((scores >= best["threshold"]).mean()) if best["threshold"] is not None else 0.0
    )
    block_detection = None
    if "inject_flag" in frame.columns:
        block_labels = frame["inject_flag"].fillna(0).to_numpy(dtype=np.int64)
        block_detection = _binary_detection_summary(scores, block_labels)

    lagged = frame.loc[frame["lag_gt"] > 0]
    zero_lag = frame.loc[frame["lag_gt"] == 0]
    return {
        "n_rows": int(len(frame)),
        "n_positive_rows": int(len(lagged)),
        "block_in_expected_lag_mae": float(np.abs(lagged["pred_expected_lag"] - lagged["lag_gt"]).mean())
        if not lagged.empty else None,
        "localization": {
            "auprc": lag_detection["auprc"],
            "auroc": lag_detection["auroc"],
            "best_threshold": best["threshold"],
            "best_precision": best["precision"],
            "best_recall": best["recall"],
            "best_f1": best["f1"],
            "predicted_nonzero_ratio": predicted_nonzero_ratio,
        },
        "block_detection": block_detection,
        "block_out_false_alarm_rate": far,
        "mean_pred_expected_lag_when_true_zero": float(zero_lag["pred_expected_lag"].mean())
        if not zero_lag.empty else None,
    }


def _conditional_bias_table(joined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for true_lag in sorted(joined["lag_gt"].unique().tolist()):
        subset = joined.loc[joined["lag_gt"] == true_lag]
        rows.append(
            {
                "true_lag": int(true_lag),
                "n_rows": int(len(subset)),
                "mean_pred_expected_lag": float(subset["pred_expected_lag"].mean()),
                "mean_pred_argmax_lag": float(subset["pred_argmax_lag"].mean()),
                "mean_pred_nonzero_prob": float(subset["pred_nonzero_prob"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _positive_lag_blocks(joined: pd.DataFrame) -> List[pd.DataFrame]:
    working = joined.copy()
    if "TimeStamp" in working.columns:
        working["_sort_time"] = pd.to_datetime(working["TimeStamp"])
        working = working.sort_values("_sort_time").drop(columns=["_sort_time"]).reset_index(drop=True)
    lag_positive = working["lag_gt"].to_numpy(dtype=np.int64) > 0
    blocks: List[pd.DataFrame] = []
    start_idx = 0
    in_block = False
    for idx, is_positive in enumerate(lag_positive):
        if is_positive and not in_block:
            start_idx = idx
            in_block = True
        elif not is_positive and in_block:
            blocks.append(working.iloc[start_idx:idx].copy())
            in_block = False
    if in_block:
        blocks.append(working.iloc[start_idx:].copy())
    return blocks


def _peak_block_table(joined: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for block_id, block in enumerate(_positive_lag_blocks(joined), start=1):
        true_peak = float(block["lag_gt"].max())
        true_dmax = int(block["segment_dmax_gt"].max()) if "segment_dmax_gt" in block.columns else int(true_peak)
        pred_peak = float(block["pred_expected_lag"].max())
        rounded_peak = int(np.floor(pred_peak + 0.5))
        rows.append(
            {
                "block_id": block_id,
                "start_time": str(block["TimeStamp"].iloc[0]) if "TimeStamp" in block.columns else "",
                "end_time": str(block["TimeStamp"].iloc[-1]) if "TimeStamp" in block.columns else "",
                "n_samples": int(len(block)),
                "true_peak_lag": true_peak,
                "true_dmax": true_dmax,
                "pred_peak_expected_lag": pred_peak,
                "pred_peak_rounded_lag": rounded_peak,
                "peak_error": abs(pred_peak - true_peak),
                "peak_hit_at_0": int(rounded_peak == int(true_peak)),
                "peak_hit_at_pm1": int(abs(rounded_peak - int(true_peak)) <= 1),
            }
        )
    return pd.DataFrame(rows)


def _summarize_peak_blocks(blocks: pd.DataFrame) -> Dict[str, Any]:
    if blocks.empty:
        return {
            "n_blocks": 0,
            "peak_error": None,
            "peak_hit_at_0": None,
            "peak_hit_at_pm1": None,
        }
    return {
        "n_blocks": int(len(blocks)),
        "peak_error": float(blocks["peak_error"].mean()),
        "peak_hit_at_0": float(blocks["peak_hit_at_0"].mean()),
        "peak_hit_at_pm1": float(blocks["peak_hit_at_pm1"].mean()),
    }


def main() -> None:
    args = parse_args()
    estimates_path = _absolute_path(args.estimates)
    raw_dataset_path = _absolute_path(args.raw_dataset)
    output_dir = _absolute_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(raw_dataset_path)
    if args.split_col not in raw_df.columns or args.lag_col not in raw_df.columns:
        raise ValueError(f"Raw dataset must contain {args.split_col} and {args.lag_col}")

    base_cols = [args.time_col, args.lag_col]
    for optional in ["segment_id", "inject_flag", "bump_dmax_gt", "segment_dmax_gt"]:
        if optional in raw_df.columns:
            base_cols.append(optional)
    raw_test = raw_df.loc[raw_df[args.split_col] == "test", base_cols].copy()
    raw_test = raw_test.rename(columns={args.time_col: "TimeStamp", args.lag_col: "lag_gt"})
    raw_test["TimeStamp"] = pd.to_datetime(raw_test["TimeStamp"]).dt.strftime(TIME_FORMAT)
    raw_test["lag_gt"] = raw_test["lag_gt"].astype(int)
    if "segment_id" in raw_test.columns:
        raw_test["segment_id"] = raw_test["segment_id"].astype(int)

    joined = _prepare_joined_frame(estimates_path, raw_test, args.edge)

    if args.bump_plan is not None:
        bump_map = _load_bump_plan(_absolute_path(args.bump_plan))
        if bump_map and "segment_id" in joined.columns and "segment_dmax_gt" not in joined.columns:
            joined["segment_dmax_gt"] = joined["segment_id"].map(bump_map).fillna(0).astype(int)

    metrics = _load_json(_absolute_path(args.metrics) if args.metrics is not None else None)
    benchmark = _benchmark_for_frame(joined)
    conditional_bias = _conditional_bias_table(joined)
    peak_blocks = _peak_block_table(joined)
    peak_summary = _summarize_peak_blocks(peak_blocks)

    summary: Dict[str, Any] = {
        "raw_dataset": raw_dataset_path.as_posix(),
        "estimates": estimates_path.as_posix(),
        "edge": args.edge,
        "forecast_metrics": {
            "MAE": metrics.get("MAE"),
            "RMSE": metrics.get("RMSE"),
            "R2": metrics.get("R2"),
        },
        "benchmark": benchmark,
        "peak": peak_summary,
        "conditional_bias": conditional_bias.to_dict(orient="records"),
    }

    if "segment_dmax_gt" in joined.columns:
        by_dmax: Dict[str, Any] = {}
        for dmax_value in sorted(
            [int(value) for value in joined["segment_dmax_gt"].dropna().unique().tolist() if int(value) > 0]
        ):
            subset = joined.loc[joined["segment_dmax_gt"] == dmax_value].copy()
            metrics_by_dmax = _benchmark_for_frame(subset)
            metrics_by_dmax["n_segments"] = int(subset["segment_id"].nunique()) if "segment_id" in subset.columns else None
            dmax_blocks = peak_blocks.loc[peak_blocks["true_dmax"] == dmax_value] if not peak_blocks.empty else peak_blocks
            metrics_by_dmax["peak"] = _summarize_peak_blocks(dmax_blocks)
            by_dmax[str(dmax_value)] = metrics_by_dmax
        summary["benchmark_by_dmax"] = by_dmax

    joined.to_csv(output_dir / "test_joined_single_model.csv", index=False)
    conditional_bias.to_csv(output_dir / "conditional_bias_table.csv", index=False)
    peak_blocks.to_csv(output_dir / "benchmark_peak_summary.csv", index=False)
    (output_dir / "benchmark_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote joined test rows to {output_dir / 'test_joined_single_model.csv'}")
    print(f"Wrote conditional bias table to {output_dir / 'conditional_bias_table.csv'}")
    print(f"Wrote benchmark summary to {output_dir / 'benchmark_summary.json'}")


if __name__ == "__main__":
    main()
