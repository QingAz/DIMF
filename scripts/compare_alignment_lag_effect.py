#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from alignment_peak_metrics import attach_peak_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare aligned vs no-align delay estimates on raw-gap test rows.")
    parser.add_argument("--aligned-estimates", type=Path, required=True)
    parser.add_argument("--noalign-estimates", type=Path, required=True)
    parser.add_argument("--aligned-metrics", type=Path, required=True)
    parser.add_argument("--noalign-metrics", type=Path, required=True)
    parser.add_argument("--raw-dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--edge", default="stage1_to_stage2")
    return parser.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _pi_columns(frame: pd.DataFrame, edge: str) -> List[str]:
    prefix = "%s_pred_pi_lag" % edge
    cols = [col for col in frame.columns if col.startswith(prefix)]
    return sorted(cols, key=lambda name: int(name.split("lag")[-1]))


def _entropy(pi: np.ndarray) -> np.ndarray:
    clipped = np.clip(pi.astype(float), 1e-12, 1.0)
    return -(clipped * np.log(clipped)).sum(axis=1)


def _estimate_features(path: Path, edge: str, model: str) -> pd.DataFrame:
    estimates = pd.read_csv(path)
    if "TimeStamp" not in estimates.columns:
        raise ValueError("Missing TimeStamp column in %s" % path)

    expected_col = "%s_pred_expected_lag" % edge
    argmax_col = "%s_pred_argmax_lag" % edge
    required = [expected_col, argmax_col]
    missing = [col for col in required if col not in estimates.columns]
    if missing:
        raise ValueError("Missing columns in %s: %s" % (path, ", ".join(missing)))

    pi_cols = _pi_columns(estimates, edge)
    if not pi_cols:
        raise ValueError("No probability columns for edge %s in %s" % (edge, path))

    pi = estimates[pi_cols].to_numpy(dtype=float)
    out = pd.DataFrame(
        {
            "TimeStamp": pd.to_datetime(estimates["TimeStamp"]),
            "%s_pred_expected_lag" % model: estimates[expected_col].astype(float),
            "%s_pred_argmax_lag" % model: estimates[argmax_col].astype(int),
            "%s_pred_entropy" % model: _entropy(pi),
            "%s_pred_nonzero_prob" % model: 1.0 - estimates[pi_cols[0]].astype(float),
        }
    )
    return out


def _joined_frame(raw_dataset: Path, aligned_path: Path, noalign_path: Path, edge: str) -> pd.DataFrame:
    raw = pd.read_csv(raw_dataset)
    raw["TimeStamp"] = pd.to_datetime(raw["TimeStamp"])
    raw_cols = [
        "TimeStamp",
        "lag_gt",
        "segment_id",
        "inject_flag",
        "bump_dmax_gt",
        "segment_dmax_gt",
    ]
    for col in raw_cols:
        if col not in raw.columns:
            raise ValueError("Missing raw dataset column: %s" % col)
    if "split" in raw.columns:
        raw = raw[raw["split"].astype(str) == "test"].copy()

    aligned = _estimate_features(aligned_path, edge, "aligned")
    noalign = _estimate_features(noalign_path, edge, "noalign")
    joined = raw[raw_cols].merge(aligned, on="TimeStamp", how="inner").merge(noalign, on="TimeStamp", how="inner")
    joined = joined.sort_values("TimeStamp").reset_index(drop=True)

    for model in ("aligned", "noalign"):
        joined["%s_expected_abs_error" % model] = (
            joined["%s_pred_expected_lag" % model].astype(float) - joined["lag_gt"].astype(float)
        ).abs()
        joined["%s_argmax_hit" % model] = (
            joined["%s_pred_argmax_lag" % model].astype(int) == joined["lag_gt"].astype(int)
        ).astype(int)

    return joined


def _metric_or_none(values: pd.Series) -> Optional[float]:
    if len(values) == 0:
        return None
    return float(values.mean())


def _lag_recovery_subset(frame: pd.DataFrame, model: str) -> Dict[str, Any]:
    if frame.empty:
        return {
            "n": 0,
            "expected_lag_mae": None,
            "expected_lag_rmse": None,
            "argmax_lag_mae": None,
            "argmax_lag_accuracy": None,
            "mean_pred_entropy": None,
            "mean_pred_expected_lag": None,
        }
    expected_error = frame["%s_pred_expected_lag" % model].astype(float) - frame["lag_gt"].astype(float)
    argmax_error = frame["%s_pred_argmax_lag" % model].astype(float) - frame["lag_gt"].astype(float)
    return {
        "n": int(len(frame)),
        "expected_lag_mae": float(expected_error.abs().mean()),
        "expected_lag_rmse": float(np.sqrt((expected_error ** 2).mean())),
        "argmax_lag_mae": float(argmax_error.abs().mean()),
        "argmax_lag_accuracy": float((argmax_error == 0).mean()),
        "mean_pred_entropy": _metric_or_none(frame["%s_pred_entropy" % model]),
        "mean_pred_expected_lag": _metric_or_none(frame["%s_pred_expected_lag" % model]),
    }


def _lag_recovery(joined: pd.DataFrame, model: str) -> Dict[str, Any]:
    lagged = joined[joined["lag_gt"].astype(float) > 0]
    no_lag = joined[joined["lag_gt"].astype(float) == 0]
    per_lag: List[Dict[str, Any]] = []
    for lag_value, group in joined.groupby("lag_gt"):
        row = _lag_recovery_subset(group, model)
        row["model"] = model
        row["lag_gt"] = int(lag_value)
        per_lag.append(row)
    return {
        "overall": _lag_recovery_subset(joined, model),
        "lagged_only": _lag_recovery_subset(lagged, model),
        "no_lag_only": _lag_recovery_subset(no_lag, model),
        "per_lag": sorted(per_lag, key=lambda row: row["lag_gt"]),
    }


def _average_precision(y_true: np.ndarray, score: np.ndarray) -> float:
    positives = int(y_true.sum())
    if positives == 0:
        return 0.0
    order = np.argsort(-score)
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted)
    precision = tp / (np.arange(len(y_sorted)) + 1.0)
    return float((precision * y_sorted).sum() / positives)


def _best_f1(y_true: np.ndarray, score: np.ndarray) -> Tuple[float, float, float, float]:
    best_threshold = 0.0
    best_precision = 0.0
    best_recall = 0.0
    best_f1 = -1.0
    for threshold in np.unique(score):
        pred = score >= threshold
        tp = int(np.logical_and(pred, y_true == 1).sum())
        fp = int(np.logical_and(pred, y_true == 0).sum())
        fn = int(np.logical_and(~pred, y_true == 1).sum())
        precision = float(tp / (tp + fp)) if tp + fp else 0.0
        recall = float(tp / (tp + fn)) if tp + fn else 0.0
        f1 = float(2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0
        if f1 > best_f1:
            best_threshold = float(threshold)
            best_precision = precision
            best_recall = recall
            best_f1 = f1
    return best_threshold, best_precision, best_recall, max(best_f1, 0.0)


def _localization(frame: pd.DataFrame, model: str) -> Dict[str, float]:
    if frame.empty:
        return {
            "auprc": 0.0,
            "best_threshold": 0.0,
            "best_precision": 0.0,
            "best_recall": 0.0,
            "best_f1": 0.0,
        }
    y_true = (frame["lag_gt"].astype(float).to_numpy() > 0).astype(int)
    score = frame["%s_pred_nonzero_prob" % model].astype(float).to_numpy()
    threshold, precision, recall, f1 = _best_f1(y_true, score)
    return {
        "auprc": _average_precision(y_true, score),
        "best_threshold": threshold,
        "best_precision": precision,
        "best_recall": recall,
        "best_f1": f1,
    }


def _false_alarm_rate(frame: pd.DataFrame, model: str, threshold: float) -> float:
    no_lag = frame[frame["lag_gt"].astype(float) == 0]
    if no_lag.empty:
        return 0.0
    return float((no_lag["%s_pred_nonzero_prob" % model].astype(float) >= threshold).mean())


def _per_lag_comparison(joined: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for lag_value, group in joined.groupby("lag_gt"):
        rows.append(
            {
                "lag_gt": int(lag_value),
                "n": int(len(group)),
                "aligned_expected_lag_mae": float(group["aligned_expected_abs_error"].mean()),
                "noalign_expected_lag_mae": float(group["noalign_expected_abs_error"].mean()),
                "aligned_argmax_lag_accuracy": float(group["aligned_argmax_hit"].mean()),
                "noalign_argmax_lag_accuracy": float(group["noalign_argmax_hit"].mean()),
                "aligned_mean_pred_expected_lag": float(group["aligned_pred_expected_lag"].mean()),
                "noalign_mean_pred_expected_lag": float(group["noalign_pred_expected_lag"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("lag_gt").reset_index(drop=True)


def _confusion(joined: pd.DataFrame, model: str) -> List[Dict[str, int]]:
    rows: List[Dict[str, int]] = []
    grouped = joined.groupby(["lag_gt", "%s_pred_argmax_lag" % model]).size().reset_index(name="count")
    for row in grouped.itertuples(index=False):
        rows.append(
            {
                "lag_gt": int(getattr(row, "lag_gt")),
                "pred_argmax_lag": int(getattr(row, "%s_pred_argmax_lag" % model)),
                "count": int(getattr(row, "count")),
            }
        )
    return rows


def _benchmark_by_dmax(joined: pd.DataFrame) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for dmax, group in joined[joined["segment_dmax_gt"].astype(float) > 0].groupby("segment_dmax_gt"):
        dmax_key = str(int(dmax))
        output[dmax_key] = {"n_samples": int(len(group)), "note": "Prediction MAE/RMSE not computed per-dmax (requires per-sample forecast errors)."}
        lagged = group[group["lag_gt"].astype(float) > 0]
        for model in ("aligned", "noalign"):
            loc = _localization(group, model)
            output[dmax_key][model] = {
                "block_in_expected_lag_mae": float(lagged["%s_expected_abs_error" % model].mean()) if not lagged.empty else None,
                "localization": loc,
                "block_out_false_alarm_rate": _false_alarm_rate(group, model, loc["best_threshold"]),
            }
    return output


def _write_report(summary: Dict[str, Any], output_path: Path) -> None:
    peak = summary.get("benchmark", {}).get("peak", {})
    lines = [
        "# Alignment Comparison",
        "",
        "## Alignment Metrics",
        "",
        "| metric | aligned | noalign |",
        "| --- | --- | --- |",
        "| Lagged-only expected lag MAE | %.6f | %.6f |"
        % (
            summary["lag_recovery"]["aligned"]["lagged_only"]["expected_lag_mae"],
            summary["lag_recovery"]["noalign"]["lagged_only"]["expected_lag_mae"],
        ),
        "| Localization AUPRC | %.6f | %.6f |"
        % (
            summary["benchmark"]["localization"]["aligned"]["auprc"],
            summary["benchmark"]["localization"]["noalign"]["auprc"],
        ),
        "| Block-out false alarm rate | %.6f | %.6f |"
        % (
            summary["benchmark"]["block_out_false_alarm_rate"]["aligned"],
            summary["benchmark"]["block_out_false_alarm_rate"]["noalign"],
        ),
    ]
    if peak:
        lines.extend(
            [
                "| Peak error | %.6f | %.6f |"
                % (peak["aligned"]["peak_error"], peak["noalign"]["peak_error"]),
                "| Peak hit@0 | %.6f | %.6f |"
                % (peak["aligned"]["peak_hit_at_0"], peak["noalign"]["peak_hit_at_0"]),
                "| Peak hit@+/-1 | %.6f | %.6f |"
                % (peak["aligned"]["peak_hit_at_pm1"], peak["noalign"]["peak_hit_at_pm1"]),
            ]
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    joined = _joined_frame(args.raw_dataset, args.aligned_estimates, args.noalign_estimates, args.edge)
    aligned_metrics = _load_json(args.aligned_metrics)
    noalign_metrics = _load_json(args.noalign_metrics)

    loc_aligned = _localization(joined, "aligned")
    loc_noalign = _localization(joined, "noalign")
    per_lag = _per_lag_comparison(joined)

    summary: Dict[str, Any] = {
        "raw_dataset": str(args.raw_dataset.resolve()),
        "edge": args.edge,
        "n_test_rows_raw": int((pd.read_csv(args.raw_dataset)["split"].astype(str) == "test").sum()),
        "n_test_samples_compared": {"aligned": int(len(joined)), "noalign": int(len(joined))},
        "forecast_metrics": {"aligned": aligned_metrics, "noalign": noalign_metrics},
        "lag_recovery": {"aligned": _lag_recovery(joined, "aligned"), "noalign": _lag_recovery(joined, "noalign")},
        "per_lag_comparison": per_lag.to_dict(orient="records"),
        "confusion": {"aligned": _confusion(joined, "aligned"), "noalign": _confusion(joined, "noalign")},
        "benchmark": {
            "forecast_improvement": {
                "MAE": float(noalign_metrics.get("MAE", np.nan) - aligned_metrics.get("MAE", np.nan)),
                "RMSE": float(noalign_metrics.get("RMSE", np.nan) - aligned_metrics.get("RMSE", np.nan)),
            },
            "block_in_expected_lag_mae": {
                "aligned": _lag_recovery(joined, "aligned")["lagged_only"]["expected_lag_mae"],
                "noalign": _lag_recovery(joined, "noalign")["lagged_only"]["expected_lag_mae"],
            },
            "localization": {"aligned": loc_aligned, "noalign": loc_noalign},
            "block_out_false_alarm_rate": {
                "aligned": _false_alarm_rate(joined, "aligned", loc_aligned["best_threshold"]),
                "noalign": _false_alarm_rate(joined, "noalign", loc_noalign["best_threshold"]),
            },
        },
        "benchmark_by_dmax": _benchmark_by_dmax(joined),
    }

    peak_blocks = attach_peak_metrics(summary, joined)

    joined_out = joined.copy()
    joined_out["TimeStamp"] = joined_out["TimeStamp"].dt.strftime("%Y-%m-%d %H:%M")
    joined_out.to_csv(args.output_dir / "alignment_test_joined.csv", index=False)
    per_lag.to_csv(args.output_dir / "alignment_per_lag_comparison.csv", index=False)
    peak_blocks.to_csv(args.output_dir / "alignment_peak_summary.csv", index=False)

    summary_path = args.output_dir / "alignment_comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_report(summary, args.output_dir / "alignment_comparison_report.md")

    print("Wrote: %s" % summary_path)


if __name__ == "__main__":
    main()
