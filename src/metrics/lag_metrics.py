from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


UNIMODAL_SHAPES = {"fixed", "random_discrete", "gaussian"}


def _safe_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))


def _normalize_dist(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    arr = np.clip(arr, 0.0, None)
    denom = np.clip(arr.sum(axis=-1, keepdims=True), 1e-12, None)
    return arr / denom


def _kl(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    q = _normalize_dist(q)
    p = _normalize_dist(p)
    terms = np.zeros_like(q, dtype=np.float64)
    mask = q > 0
    terms[mask] = q[mask] * np.log(q[mask] / np.clip(p[mask], 1e-12, None))
    return np.sum(terms, axis=-1)


def _js(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    q = _normalize_dist(q)
    p = _normalize_dist(p)
    m = 0.5 * (q + p)
    return 0.5 * _kl(q, m) + 0.5 * _kl(p, m)


def compute_lag_metrics(
    pred_pi: np.ndarray,
    gt_pi: np.ndarray,
    lag_flag: np.ndarray,
    lag_value: Optional[np.ndarray] = None,
    shape_type: Optional[np.ndarray] = None,
    occurrence_score: Optional[np.ndarray] = None,
    topk_radius: int = 1,
    false_alarm_threshold: float = 0.5,
) -> Dict[str, float]:
    pred_pi = _normalize_dist(pred_pi)
    gt_pi = _normalize_dist(gt_pi)
    if pred_pi.shape != gt_pi.shape:
        raise ValueError(f"pred_pi and gt_pi shape mismatch: {pred_pi.shape} vs {gt_pi.shape}")
    n, k = pred_pi.shape
    lag_axis = np.arange(k, dtype=np.float64)
    lag_flag = np.asarray(lag_flag).astype(bool)
    if lag_value is None:
        lag_value = gt_pi.argmax(axis=-1)
    lag_value = np.asarray(lag_value).astype(int)
    expected_pred = (pred_pi * lag_axis[None, :]).sum(axis=-1)
    expected_gt = (gt_pi * lag_axis[None, :]).sum(axis=-1)
    abs_expected = np.abs(expected_pred - expected_gt)

    pred_argmax = pred_pi.argmax(axis=-1)
    gt_argmax = gt_pi.argmax(axis=-1)
    if shape_type is None:
        unimodal = np.ones(n, dtype=bool)
    else:
        shape_arr = np.asarray(shape_type).astype(str)
        unimodal = np.isin(shape_arr, list(UNIMODAL_SHAPES))
    argmax_mask = lag_flag & unimodal

    hard_idx = np.clip(lag_value, 0, k - 1)
    top1_mass = pred_pi[np.arange(n), hard_idx]
    around_mass = []
    for idx, hard in enumerate(hard_idx):
        lo = max(0, hard - int(topk_radius))
        hi = min(k, hard + int(topk_radius) + 1)
        around_mass.append(float(pred_pi[idx, lo:hi].sum()))
    around_mass = np.asarray(around_mass, dtype=np.float64)

    if occurrence_score is None:
        occurrence_score = 1.0 - pred_pi[:, 0]
    occurrence_score = np.asarray(occurrence_score, dtype=np.float64)
    if len(np.unique(lag_flag.astype(int))) == 2:
        occurrence_auprc = float(average_precision_score(lag_flag.astype(int), occurrence_score))
    else:
        occurrence_auprc = float("nan")
    no_lag_mask = ~lag_flag
    pred_lag_positive = occurrence_score > float(false_alarm_threshold)

    return {
        "n_samples": int(n),
        "expected_lag_mae_all": _safe_mean(abs_expected),
        "expected_lag_mae_injected": _safe_mean(abs_expected[lag_flag]),
        "expected_lag_mae_no_lag": _safe_mean(abs_expected[no_lag_mask]),
        "argmax_lag_accuracy": _safe_mean((pred_argmax[argmax_mask] == gt_argmax[argmax_mask]).astype(float)),
        "soft_kl": _safe_mean(_kl(gt_pi, pred_pi)),
        "soft_js": _safe_mean(_js(gt_pi, pred_pi)),
        "top1_mass": _safe_mean(top1_mass[lag_flag]),
        "topk_mass_around_true": _safe_mean(around_mass[lag_flag]),
        "occurrence_auprc": occurrence_auprc,
        "no_lag_false_alarm_rate": _safe_mean(pred_lag_positive[no_lag_mask].astype(float)),
    }


def metrics_by_group(
    pred_pi: np.ndarray,
    gt_pi: np.ndarray,
    lag_flag: np.ndarray,
    group_values: Iterable[object],
    group_name: str,
    lag_value: Optional[np.ndarray] = None,
    shape_type: Optional[np.ndarray] = None,
    occurrence_score: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    group_values = np.asarray(list(group_values))
    rows = []
    for value in sorted(pd.unique(group_values)):
        mask = group_values == value
        if not np.any(mask):
            continue
        metrics = compute_lag_metrics(
            pred_pi=pred_pi[mask],
            gt_pi=gt_pi[mask],
            lag_flag=np.asarray(lag_flag)[mask],
            lag_value=None if lag_value is None else np.asarray(lag_value)[mask],
            shape_type=None if shape_type is None else np.asarray(shape_type)[mask],
            occurrence_score=None if occurrence_score is None else np.asarray(occurrence_score)[mask],
        )
        rows.append({group_name: value, **metrics})
    return pd.DataFrame(rows)


def prediction_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = y_pred - y_true
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "prediction_mae": float(np.mean(np.abs(err))),
        "prediction_mse": float(np.mean(err ** 2)),
        "prediction_rmse": float(np.sqrt(np.mean(err ** 2))),
        "prediction_r2": float(1.0 - ss_res / (ss_tot + 1e-12)),
    }


def save_lag_metric_tables(
    output_dir: str | Path,
    pred_pi: np.ndarray,
    gt_pi: np.ndarray,
    lag_flag: np.ndarray,
    lag_value: np.ndarray,
    shape_type: np.ndarray,
    occurrence_score: Optional[np.ndarray] = None,
    y_true: Optional[np.ndarray] = None,
    y_pred: Optional[np.ndarray] = None,
) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    overall = compute_lag_metrics(
        pred_pi=pred_pi,
        gt_pi=gt_pi,
        lag_flag=lag_flag,
        lag_value=lag_value,
        shape_type=shape_type,
        occurrence_score=occurrence_score,
    )
    by_shape = metrics_by_group(
        pred_pi,
        gt_pi,
        lag_flag,
        group_values=shape_type,
        group_name="shape_type",
        lag_value=lag_value,
        shape_type=shape_type,
        occurrence_score=occurrence_score,
    )
    by_lag = metrics_by_group(
        pred_pi,
        gt_pi,
        lag_flag,
        group_values=lag_value,
        group_name="lag_value",
        lag_value=lag_value,
        shape_type=shape_type,
        occurrence_score=occurrence_score,
    )
    paths = {
        "overall": output_dir / "lag_metrics_overall.csv",
        "by_shape": output_dir / "lag_metrics_by_shape.csv",
        "by_lag": output_dir / "lag_metrics_by_lag.csv",
        "prediction": output_dir / "prediction_metrics.csv",
    }
    pd.DataFrame([overall]).to_csv(paths["overall"], index=False)
    by_shape.to_csv(paths["by_shape"], index=False)
    by_lag.to_csv(paths["by_lag"], index=False)
    if y_true is not None and y_pred is not None:
        pd.DataFrame([prediction_metrics(y_true, y_pred)]).to_csv(paths["prediction"], index=False)
    else:
        pd.DataFrame([{}]).to_csv(paths["prediction"], index=False)
    return paths
