from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import pandas as pd


DEFAULT_Q40_EVIDENCE_FEATURE_COLUMNS = [
    "candidate_score",
    "d_raw",
    "expected_lag",
    "p_nonzero",
    "entropy",
    "peak_prob",
    "margin",
    "localization_score",
    "stability",
    "local_area",
    "local_std",
    "slope",
    "curvature",
]


@dataclass(frozen=True)
class DRawCalibration:
    source_col: str
    mode: str
    a: float = 1.0
    b: float = 0.0
    clip_to_dmax: bool = False
    dmax_col: str = "dmax"


@dataclass(frozen=True)
class FeatureNormalizer:
    feature_columns: List[str]
    median: List[float]
    mean: List[float]
    std: List[float]

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        missing = [col for col in self.feature_columns if col not in frame.columns]
        if missing:
            raise ValueError(f"Q40 feature input is missing columns: {', '.join(missing)}")
        values = frame[self.feature_columns].to_numpy(dtype=np.float64)
        median = np.asarray(self.median, dtype=np.float64)
        mean = np.asarray(self.mean, dtype=np.float64)
        std = np.asarray(self.std, dtype=np.float64)
        values = np.where(np.isfinite(values), values, median[None, :])
        return ((values - mean[None, :]) / std[None, :]).astype(np.float32)


def _as_float(frame: pd.DataFrame, col: str, default: float = 0.0) -> np.ndarray:
    if col in frame.columns:
        return frame[col].to_numpy(dtype=np.float64)
    return np.full(len(frame), float(default), dtype=np.float64)


def _first_existing(frame: pd.DataFrame, names: Sequence[str], default: float = 0.0) -> np.ndarray:
    out = np.full(len(frame), float(default), dtype=np.float64)
    filled = np.zeros(len(frame), dtype=bool)
    for name in names:
        if name not in frame.columns:
            continue
        values = frame[name].to_numpy(dtype=np.float64)
        valid = np.isfinite(values) & ~filled
        out[valid] = values[valid]
        filled[valid] = True
        if bool(filled.all()):
            break
    return out


def _sort_frame(frame: pd.DataFrame, group_col: str, time_col: str) -> pd.DataFrame:
    sort_cols = []
    if group_col in frame.columns:
        sort_cols.append(group_col)
    if time_col in frame.columns:
        sort_cols.append(time_col)
    elif "timestamp" in frame.columns:
        sort_cols.append("timestamp")
    elif "TimeStamp" in frame.columns:
        sort_cols.append("TimeStamp")
    if not sort_cols:
        return frame.reset_index(drop=True).copy()
    return frame.sort_values(sort_cols).reset_index(drop=True).copy()


def _rolling_by_group(
    frame: pd.DataFrame,
    values: np.ndarray,
    group_col: str,
    window: int,
    how: str,
) -> np.ndarray:
    window = max(int(window), 1)
    out = np.zeros(len(frame), dtype=np.float64)
    if group_col not in frame.columns:
        local = pd.Series(values)
        roller = local.rolling(window=window, center=True, min_periods=1)
        return getattr(roller, how)().fillna(0.0).to_numpy(dtype=np.float64)

    for _, idx in frame.groupby(group_col, sort=False).groups.items():
        idx_arr = frame.index.get_indexer(idx)
        local = pd.Series(values[idx_arr])
        roller = local.rolling(window=window, center=True, min_periods=1)
        out[idx_arr] = getattr(roller, how)().fillna(0.0).to_numpy(dtype=np.float64)
    return out


def _neighbor_by_group(frame: pd.DataFrame, values: np.ndarray, group_col: str, shift: int) -> np.ndarray:
    out = np.zeros(len(frame), dtype=np.float64)
    if group_col not in frame.columns:
        return pd.Series(values).shift(int(shift), fill_value=0.0).to_numpy(dtype=np.float64)

    for _, idx in frame.groupby(group_col, sort=False).groups.items():
        idx_arr = frame.index.get_indexer(idx)
        out[idx_arr] = (
            pd.Series(values[idx_arr])
            .shift(int(shift), fill_value=0.0)
            .to_numpy(dtype=np.float64)
        )
    return out


def _percentile_rank_by_group(frame: pd.DataFrame, values: np.ndarray, group_col: str) -> np.ndarray:
    out = np.zeros(len(frame), dtype=np.float64)
    if group_col not in frame.columns:
        return pd.Series(values).rank(method="average", pct=True).fillna(0.0).to_numpy(dtype=np.float64)

    for _, idx in frame.groupby(group_col, sort=False).groups.items():
        idx_arr = frame.index.get_indexer(idx)
        out[idx_arr] = (
            pd.Series(values[idx_arr])
            .rank(method="average", pct=True)
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
        )
    return out


def fit_d_raw_calibration(
    frame: pd.DataFrame,
    label_col: str = "d_true",
    source_col: str = "raw_m",
    mode: str = "affine",
    clip_to_dmax: bool = False,
    dmax_col: str = "dmax",
) -> DRawCalibration:
    if source_col not in frame.columns:
        raise ValueError(f"Cannot calibrate d_raw: missing source column {source_col!r}")
    mode = str(mode).lower()
    if mode not in {"none", "identity", "affine"}:
        raise ValueError(f"Unknown d_raw calibration mode: {mode}")
    if mode in {"none", "identity"}:
        return DRawCalibration(
            source_col=source_col,
            mode="identity",
            clip_to_dmax=bool(clip_to_dmax),
            dmax_col=dmax_col,
        )
    if label_col not in frame.columns:
        raise ValueError(f"Affine d_raw calibration requires label column {label_col!r}")

    labels = frame[label_col].to_numpy(dtype=np.float64)
    source = frame[source_col].to_numpy(dtype=np.float64)
    mask = (labels > 0) & np.isfinite(labels) & np.isfinite(source)
    if not np.any(mask):
        return DRawCalibration(
            source_col=source_col,
            mode="identity",
            clip_to_dmax=bool(clip_to_dmax),
            dmax_col=dmax_col,
        )
    a, b = np.linalg.lstsq(
        np.column_stack([source[mask], np.ones(int(mask.sum()), dtype=np.float64)]),
        labels[mask],
        rcond=None,
    )[0]
    return DRawCalibration(
        source_col=source_col,
        mode="affine",
        a=float(a),
        b=float(b),
        clip_to_dmax=bool(clip_to_dmax),
        dmax_col=dmax_col,
    )


def apply_d_raw_calibration(frame: pd.DataFrame, calibration: DRawCalibration) -> np.ndarray:
    source = _as_float(frame, calibration.source_col, default=0.0)
    if calibration.mode == "affine":
        out = float(calibration.a) * source + float(calibration.b)
    else:
        out = source.copy()
    out = np.where(np.isfinite(out), out, 0.0)
    out = np.clip(out, 0.0, None)
    if calibration.clip_to_dmax and calibration.dmax_col in frame.columns:
        dmax = np.maximum(_as_float(frame, calibration.dmax_col, default=np.inf), 0.0)
        out = np.minimum(out, dmax)
    return out


def add_q40_evidence_features(
    frame: pd.DataFrame,
    calibration: DRawCalibration,
    group_col: str = "segment_id",
    time_col: str = "t",
    window: int = 5,
    include_relative_features: bool = False,
    include_q40_prior_features: bool = False,
) -> pd.DataFrame:
    out = _sort_frame(frame, group_col=group_col, time_col=time_col)
    d_raw = apply_d_raw_calibration(out, calibration)
    out["d_raw"] = d_raw
    out["p_nonzero"] = np.clip(_first_existing(out, ["nonzero_prob", "p"], default=0.0), 0.0, 1.0)
    out["peak_prob"] = np.clip(
        _first_existing(out, ["max_positive_prob", "max_prob", "p_nonzero"], default=0.0),
        0.0,
        1.0,
    )
    out["margin"] = _first_existing(out, ["top1_top2_margin", "positive_margin"], default=0.0)
    out["candidate_score"] = np.clip(_as_float(out, "candidate_score", default=0.0), 0.0, 1.0)
    out["expected_lag"] = _first_existing(
        out,
        ["expected_lag", "d_hat_raw", "d_hat", "pred_expected_lag"],
        default=0.0,
    )
    out["entropy"] = _as_float(out, "entropy", default=0.0)
    out["localization_score"] = np.clip(
        _first_existing(out, ["localization_score", "candidate_score"], default=0.0),
        0.0,
        1.0,
    )

    out["local_area"] = _rolling_by_group(out, d_raw, group_col=group_col, window=window, how="sum")
    local_std = _rolling_by_group(out, d_raw, group_col=group_col, window=window, how="std")
    out["local_std"] = np.where(np.isfinite(local_std), local_std, 0.0)
    prev_raw = _neighbor_by_group(out, d_raw, group_col=group_col, shift=1)
    next_raw = _neighbor_by_group(out, d_raw, group_col=group_col, shift=-1)
    out["slope"] = d_raw - prev_raw
    out["curvature"] = next_raw - 2.0 * d_raw + prev_raw
    out["stability"] = 1.0 / (
        1.0
        + np.abs(out["slope"].to_numpy(dtype=np.float64))
        + out["local_std"].to_numpy(dtype=np.float64)
    )

    if bool(include_relative_features):
        for col in DEFAULT_Q40_EVIDENCE_FEATURE_COLUMNS:
            if col not in out.columns:
                continue
            values = out[col].to_numpy(dtype=np.float64)
            out[f"{col}_seq_rank"] = _percentile_rank_by_group(out, values, group_col=group_col)
            local_mean = _rolling_by_group(out, values, group_col=group_col, window=window, how="mean")
            out[f"{col}_local_contrast"] = values - np.where(np.isfinite(local_mean), local_mean, 0.0)

    if bool(include_q40_prior_features):
        out["q40_prior_selected"] = np.clip(_first_existing(out, ["q40_final_selected"], default=0.0), 0.0, 1.0)
        out["q40_prior_strong_selected"] = np.clip(
            _first_existing(out, ["q40_strong_selected"], default=0.0),
            0.0,
            1.0,
        )
        out["q40_prior_low_lag_selected"] = np.clip(
            _first_existing(out, ["low_lag_high_conf_selected"], default=0.0),
            0.0,
            1.0,
        )
        out["q40_prior_weak_plateau_selected"] = np.clip(
            _first_existing(out, ["weak_plateau_selected"], default=0.0),
            0.0,
            1.0,
        )
        out["q40_prior_p_pos"] = np.clip(_first_existing(out, ["q40_p_pos"], default=0.0), 0.0, 1.0)
        out["q40_prior_d_hat"] = np.maximum(_first_existing(out, ["q40_d_hat"], default=0.0), 0.0)
    return out


def fit_feature_normalizer(frame: pd.DataFrame, feature_columns: Sequence[str]) -> FeatureNormalizer:
    values = frame[list(feature_columns)].to_numpy(dtype=np.float64)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("Q40 verifier requires at least one feature column")
    finite = np.where(np.isfinite(values), values, np.nan)
    median = np.zeros(values.shape[1], dtype=np.float64)
    for col_idx in range(values.shape[1]):
        present = finite[:, col_idx][np.isfinite(finite[:, col_idx])]
        median[col_idx] = float(np.median(present)) if present.size else 0.0
    clean = np.where(np.isfinite(values), values, median[None, :])
    mean = clean.mean(axis=0)
    std = clean.std(axis=0)
    std = np.where(std > 1e-6, std, 1.0)
    return FeatureNormalizer(
        feature_columns=list(feature_columns),
        median=median.astype(float).tolist(),
        mean=mean.astype(float).tolist(),
        std=std.astype(float).tolist(),
    )


def split_by_group(
    frame: pd.DataFrame,
    group_col: str = "segment_id",
    val_fraction: float = 0.25,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if group_col not in frame.columns:
        rng = np.random.default_rng(int(seed))
        order = np.arange(len(frame))
        rng.shuffle(order)
        n_val = max(1, int(round(len(order) * float(val_fraction))))
        val_idx = order[:n_val]
        train_idx = order[n_val:]
        return frame.iloc[train_idx].reset_index(drop=True), frame.iloc[val_idx].reset_index(drop=True)

    groups = np.asarray(sorted(frame[group_col].dropna().unique().tolist()))
    if groups.size < 2:
        raise ValueError("Internal validation split requires at least two groups")
    rng = np.random.default_rng(int(seed))
    rng.shuffle(groups)
    n_val = max(1, int(round(groups.size * float(val_fraction))))
    n_val = min(n_val, groups.size - 1)
    val_groups = set(groups[:n_val].tolist())
    val_mask = frame[group_col].isin(val_groups)
    return (
        frame.loc[~val_mask].reset_index(drop=True),
        frame.loc[val_mask].reset_index(drop=True),
    )
