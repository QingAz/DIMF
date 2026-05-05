#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit detection failure modes: near/far negatives, dmax breakdown, and shortcut probe."
    )
    parser.add_argument("--scores", type=Path, required=True, help="sample_detection_scores.csv from detection audit")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for outputs")
    parser.add_argument("--k-list", default="4,8", help="Comma-separated near/far thresholds in steps")
    return parser.parse_args()


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(str(path)))


def _pr_curve(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(-scores)
    scores_sorted = scores[order]
    labels_sorted = labels[order].astype(bool)
    tp = 0
    fp = 0
    fn = int(labels_sorted.sum())
    precision: List[float] = []
    recall: List[float] = []
    thresholds: List[float] = []
    last_score = None
    for score, label in zip(scores_sorted, labels_sorted):
        if last_score is None or score != last_score:
            if last_score is not None:
                precision.append(tp / (tp + fp) if tp + fp else 0.0)
                recall.append(tp / (tp + fn) if tp + fn else 0.0)
                thresholds.append(float(last_score))
            last_score = float(score)
        if label:
            tp += 1
            fn -= 1
        else:
            fp += 1
    if last_score is not None:
        precision.append(tp / (tp + fp) if tp + fp else 0.0)
        recall.append(tp / (tp + fn) if tp + fn else 0.0)
        thresholds.append(float(last_score))
    return (
        np.asarray(precision, dtype=np.float64),
        np.asarray(recall, dtype=np.float64),
        np.asarray(thresholds, dtype=np.float64),
    )


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels_bool = labels.astype(bool)
    n_pos = int(labels_bool.sum())
    n_neg = int((~labels_bool).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = pd.Series(scores).rank(method="average").to_numpy(dtype=np.float64)
    pos_rank_sum = float(ranks[labels_bool].sum())
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)


def _binary_metrics(scores: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    labels = labels.astype(np.int64)
    scores = scores.astype(np.float64)
    precision, recall, thresholds = _pr_curve(scores, labels)
    if precision.size:
        f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
        best_idx = int(np.argmax(f1))
        best_threshold = float(thresholds[best_idx])
        best_precision = float(precision[best_idx])
        best_recall = float(recall[best_idx])
        best_f1 = float(f1[best_idx])
        order = np.argsort(recall)
        auprc = float(np.trapz(precision[order], recall[order]))
    else:
        best_threshold = float("inf")
        best_precision = 0.0
        best_recall = 0.0
        best_f1 = 0.0
        auprc = 0.0
    pred_pos = scores >= best_threshold
    labels_bool = labels.astype(bool)
    neg_mask = ~labels_bool
    far = float(np.logical_and(pred_pos, neg_mask).sum() / max(int(neg_mask.sum()), 1))
    pos_scores = scores[labels_bool]
    neg_scores = scores[neg_mask]
    return {
        "auprc": auprc,
        "auroc": _auroc(scores, labels),
        "best_threshold": best_threshold,
        "best_precision": best_precision,
        "best_recall": best_recall,
        "best_f1": best_f1,
        "pred_positive_ratio": float(pred_pos.mean()) if pred_pos.size else 0.0,
        "far": far,
        "p_in_block_mean": float(pos_scores.mean()) if pos_scores.size else float("nan"),
        "p_out_block_mean": float(neg_scores.mean()) if neg_scores.size else float("nan"),
    }


def _sorted_scores(scores: pd.DataFrame) -> pd.DataFrame:
    out = scores.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    return out.sort_values(["split", "timestamp", "segment_id"]).reset_index(drop=True)


def _nearest_positive_distances(split_df: pd.DataFrame) -> np.ndarray:
    labels = split_df["in_block"].to_numpy(dtype=np.int64)
    n = len(labels)
    prev_pos = np.full(n, -10**9, dtype=np.int64)
    next_pos = np.full(n, 10**9, dtype=np.int64)

    last = -10**9
    for i in range(n):
        if labels[i] == 1:
            last = i
        prev_pos[i] = last

    last = 10**9
    for i in range(n - 1, -1, -1):
        if labels[i] == 1:
            last = i
        next_pos[i] = last

    prev_dist = np.where(prev_pos > -10**8, np.arange(n) - prev_pos, 10**9)
    next_dist = np.where(next_pos < 10**8, next_pos - np.arange(n), 10**9)
    dist = np.minimum(prev_dist, next_dist)
    dist[labels == 1] = 0
    return dist.astype(np.int64)


def _negative_composition(scores: pd.DataFrame, k_values: List[int]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for split_name, split_df in scores.groupby("split", sort=False):
        split_df = split_df.sort_values("timestamp").reset_index(drop=True)
        neg = split_df.loc[split_df["in_block"] == 0].copy()
        if neg.empty:
            for k in k_values:
                rows.append(
                    {
                        "split": split_name,
                        "k_steps": int(k),
                        "n_negative": 0,
                        "n_near_negative": 0,
                        "n_far_negative": 0,
                        "near_ratio": 0.0,
                        "far_ratio": 0.0,
                        "near_mean_p": float("nan"),
                        "far_mean_p": float("nan"),
                    }
                )
            continue
        distances = _nearest_positive_distances(split_df)
        neg = neg.assign(distance_to_block=distances[split_df["in_block"].to_numpy(dtype=np.int64) == 0])
        for k in k_values:
            near = neg.loc[neg["distance_to_block"] <= int(k)]
            far = neg.loc[neg["distance_to_block"] > int(k)]
            rows.append(
                {
                    "split": split_name,
                    "k_steps": int(k),
                    "n_negative": int(len(neg)),
                    "n_near_negative": int(len(near)),
                    "n_far_negative": int(len(far)),
                    "near_ratio": float(len(near) / max(len(neg), 1)),
                    "far_ratio": float(len(far) / max(len(neg), 1)),
                    "near_mean_p": float(near["p"].mean()) if not near.empty else float("nan"),
                    "far_mean_p": float(far["p"].mean()) if not far.empty else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def _dmax_breakdown(scores: pd.DataFrame) -> pd.DataFrame:
    test_df = scores.loc[scores["split"] == "test"].copy()
    neg = test_df.loc[test_df["in_block"] == 0].copy()
    rows: List[Dict[str, object]] = []
    for dmax in sorted(int(v) for v in test_df.loc[test_df["in_block"] == 1, "dmax"].dropna().unique() if int(v) > 0):
        pos = test_df.loc[(test_df["in_block"] == 1) & (test_df["dmax"].astype(int) == int(dmax))].copy()
        if pos.empty or neg.empty:
            continue
        labels = np.concatenate([np.ones(len(pos), dtype=np.int64), np.zeros(len(neg), dtype=np.int64)])
        score_values = np.concatenate([pos["p"].to_numpy(dtype=np.float64), neg["p"].to_numpy(dtype=np.float64)])
        metrics = _binary_metrics(score_values, labels)
        rows.append(
            {
                "split": "test",
                "dmax": int(dmax),
                "n_positive": int(len(pos)),
                "n_negative": int(len(neg)),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def _positive_segments(split_df: pd.DataFrame) -> pd.DataFrame:
    pos = (
        split_df.loc[split_df["in_block"] == 1]
        .groupby("segment_id", as_index=False)
        .agg(
            block_start=("row_index", "min"),
            block_end=("row_index", "max"),
            block_width=("row_index", "size"),
            block_dmax=("dmax", lambda s: int(pd.Series(s).mode().iloc[0])),
        )
        .sort_values("block_start")
        .reset_index(drop=True)
    )
    return pos


def _probe_features(scores: pd.DataFrame) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for split_name, split_df in scores.groupby("split", sort=False):
        split_df = split_df.sort_values("timestamp").reset_index(drop=True).copy()
        split_df["row_index"] = np.arange(len(split_df), dtype=np.int64)
        pos_segments = _positive_segments(split_df)

        prev_end = np.full(len(split_df), np.nan, dtype=np.float64)
        next_start = np.full(len(split_df), np.nan, dtype=np.float64)
        nearest_width = np.full(len(split_df), np.nan, dtype=np.float64)
        nearest_dmax = np.full(len(split_df), np.nan, dtype=np.float64)
        left_dist = np.full(len(split_df), np.nan, dtype=np.float64)
        right_dist = np.full(len(split_df), np.nan, dtype=np.float64)

        for i, row in split_df.iterrows():
            idx = int(row["row_index"])
            segment_id = int(row["segment_id"])
            if int(row["in_block"]) == 1:
                seg = pos_segments.loc[pos_segments["segment_id"] == segment_id].iloc[0]
                left_dist[i] = idx - int(seg["block_start"])
                right_dist[i] = int(seg["block_end"]) - idx
                nearest_width[i] = float(seg["block_width"])
                nearest_dmax[i] = float(seg["block_dmax"])
                prev_end[i] = 0.0
                next_start[i] = 0.0
                continue

            left_blocks = pos_segments.loc[pos_segments["block_end"] < idx]
            right_blocks = pos_segments.loc[pos_segments["block_start"] > idx]
            left_gap = float(idx - left_blocks["block_end"].max()) if not left_blocks.empty else np.nan
            right_gap = float(right_blocks["block_start"].min() - idx) if not right_blocks.empty else np.nan
            prev_end[i] = left_gap
            next_start[i] = right_gap
            candidates = []
            if not left_blocks.empty:
                left_seg = left_blocks.iloc[-1]
                candidates.append((left_gap, left_seg))
            if not right_blocks.empty:
                right_seg = right_blocks.iloc[0]
                candidates.append((right_gap, right_seg))
            if candidates:
                gap, nearest = sorted(candidates, key=lambda item: (np.inf if pd.isna(item[0]) else item[0]))[0]
                nearest_width[i] = float(nearest["block_width"])
                nearest_dmax[i] = float(nearest["block_dmax"])
            left_dist[i] = left_gap
            right_dist[i] = right_gap

        split_df["dist_to_left_block_edge"] = left_dist
        split_df["dist_to_right_block_edge"] = right_dist
        split_df["segment_length_float"] = split_df["segment_length"].astype(np.float64)
        split_df["segment_first_half"] = (split_df["segment_rel_pos"] <= 0.5).astype(np.int64)
        split_df["nearest_block_width"] = nearest_width
        split_df["nearest_block_dmax"] = nearest_dmax
        split_df["dist_to_prev_block_end"] = prev_end
        split_df["dist_to_next_block_start"] = next_start
        split_df["split"] = split_name
        rows.append(split_df)
    return pd.concat(rows, ignore_index=True)


def _shortcut_probe(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols = [
        "segment_rel_pos",
        "dist_to_left_block_edge",
        "dist_to_right_block_edge",
        "dist_to_prev_block_end",
        "dist_to_next_block_start",
        "segment_first_half",
        "segment_length_float",
        "nearest_block_width",
        "nearest_block_dmax",
    ]
    train = features.loc[features["split"] == "train"].copy()
    val = features.loc[features["split"] == "val"].copy()
    test = features.loc[features["split"] == "test"].copy()

    fill_values = train[feature_cols].median(numeric_only=True)
    x_train = train[feature_cols].fillna(fill_values)
    y_train = train["in_block"].astype(int)

    model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced")),
        ]
    )
    model.fit(x_train, y_train)

    metrics_rows: List[Dict[str, object]] = []
    score_rows: List[pd.DataFrame] = []
    for split_name, split_df in [("val", val), ("test", test)]:
        x = split_df[feature_cols].fillna(fill_values)
        score = model.predict_proba(x)[:, 1]
        metrics = _binary_metrics(score, split_df["in_block"].to_numpy(dtype=np.int64))
        metrics_rows.append({"split": split_name, **metrics})
        scored = split_df[["split", "timestamp", "segment_id", "in_block"]].copy()
        scored["probe_score"] = score
        score_rows.append(scored)

    coef = model.named_steps["clf"].coef_.reshape(-1)
    coef_df = pd.DataFrame({"feature": feature_cols, "coefficient": coef}).sort_values("coefficient", ascending=False)
    metrics_df = pd.DataFrame(metrics_rows)
    score_df = pd.concat(score_rows, ignore_index=True)
    return metrics_df, coef_df, score_df


def main() -> None:
    args = parse_args()
    scores_path = _absolute_path(args.scores)
    output_dir = _absolute_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scores = pd.read_csv(scores_path)
    scores = _sorted_scores(scores)
    k_values = [int(part.strip()) for part in str(args.k_list).split(",") if part.strip()]

    negative_df = _negative_composition(scores, k_values)
    negative_df.to_csv(output_dir / "negative_composition_audit.csv", index=False)

    dmax_df = _dmax_breakdown(scores)
    dmax_df.to_csv(output_dir / "test_dmax_detection_breakdown.csv", index=False)

    feature_df = _probe_features(scores)
    shortcut_metrics, shortcut_coef, shortcut_scores = _shortcut_probe(feature_df)
    shortcut_metrics.to_csv(output_dir / "shortcut_probe_metrics.csv", index=False)
    shortcut_coef.to_csv(output_dir / "shortcut_probe_coefficients.csv", index=False)
    shortcut_scores.to_csv(output_dir / "shortcut_probe_scores.csv", index=False)

    report = {
        "scores": scores_path.as_posix(),
        "output_dir": output_dir.as_posix(),
        "k_values": k_values,
        "negative_composition_rows": negative_df.to_dict(orient="records"),
        "dmax_breakdown_rows": dmax_df.to_dict(orient="records"),
        "shortcut_probe_rows": shortcut_metrics.to_dict(orient="records"),
    }
    (output_dir / "detection_generalization_diagnostics_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(negative_df.to_csv(index=False))
    print(dmax_df.to_csv(index=False))
    print(shortcut_metrics.to_csv(index=False))


if __name__ == "__main__":
    main()
