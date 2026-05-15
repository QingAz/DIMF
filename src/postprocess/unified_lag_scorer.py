from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Sampler, TensorDataset


DEFAULT_UNIFIED_FEATURE_COLUMNS = [
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

DEFAULT_RELATIVE_FEATURE_COLUMNS = [
    *[f"{col}_seq_rank" for col in DEFAULT_UNIFIED_FEATURE_COLUMNS],
    *[f"{col}_local_contrast" for col in DEFAULT_UNIFIED_FEATURE_COLUMNS],
]

DEFAULT_Q40_PRIOR_FEATURE_COLUMNS = [
    "q40_prior_selected",
    "q40_prior_strong_selected",
    "q40_prior_low_lag_selected",
    "q40_prior_weak_plateau_selected",
    "q40_prior_p_pos",
    "q40_prior_d_hat",
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
            raise ValueError(f"Unified scorer input is missing feature columns: {', '.join(missing)}")
        values = frame[self.feature_columns].to_numpy(dtype=np.float64)
        median = np.asarray(self.median, dtype=np.float64)
        mean = np.asarray(self.mean, dtype=np.float64)
        std = np.asarray(self.std, dtype=np.float64)
        values = np.where(np.isfinite(values), values, median[None, :])
        return ((values - mean[None, :]) / std[None, :]).astype(np.float32)


@dataclass(frozen=True)
class HardNegativeSamplingConfig:
    enabled: bool = False
    d_raw_threshold: float = 1.0
    expected_lag_threshold: float = 1.0
    p_nonzero_threshold: float = 0.3
    candidate_score_threshold: float = 0.25
    localization_score_quantile: float = 0.75
    hard_top_fraction: float = 0.30
    easy_bottom_fraction: float = 0.30
    positive_fraction: float = 0.35
    hard_negative_fraction: float = 0.45
    medium_negative_fraction: float = 0.0
    easy_negative_fraction: float = 0.20
    max_hard_per_positive: float = 20.0


class UnifiedLagScorer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, dropout: float = 0.10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


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


def add_unified_evidence_features(
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
    out["stability"] = 1.0 / (1.0 + np.abs(out["slope"].to_numpy(dtype=np.float64)) + out["local_std"].to_numpy(dtype=np.float64))

    if bool(include_relative_features):
        for col in DEFAULT_UNIFIED_FEATURE_COLUMNS:
            if col not in out.columns:
                continue
            values = out[col].to_numpy(dtype=np.float64)
            out[f"{col}_seq_rank"] = _percentile_rank_by_group(out, values, group_col=group_col)
            local_mean = _rolling_by_group(out, values, group_col=group_col, window=window, how="mean")
            out[f"{col}_local_contrast"] = values - np.where(np.isfinite(local_mean), local_mean, 0.0)

    if bool(include_q40_prior_features):
        out["q40_prior_selected"] = np.clip(_first_existing(out, ["q40_final_selected"], default=0.0), 0.0, 1.0)
        out["q40_prior_strong_selected"] = np.clip(_first_existing(out, ["q40_strong_selected"], default=0.0), 0.0, 1.0)
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


def available_feature_columns(
    frame: pd.DataFrame,
    requested: Iterable[str] | None = None,
) -> List[str]:
    columns = list(requested) if requested is not None else list(DEFAULT_UNIFIED_FEATURE_COLUMNS)
    return [col for col in columns if col in frame.columns]


def fit_feature_normalizer(frame: pd.DataFrame, feature_columns: Sequence[str]) -> FeatureNormalizer:
    values = frame[list(feature_columns)].to_numpy(dtype=np.float64)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("Unified scorer requires at least one feature column")
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


def _safe_quantile(values: np.ndarray, quantile: float, default: float) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float(default)
    q = min(max(float(quantile), 0.0), 1.0)
    return float(np.nanquantile(finite, q))


def hard_negative_sampling_groups(
    frame: pd.DataFrame,
    label_col: str = "d_true",
    config: HardNegativeSamplingConfig | None = None,
) -> tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    cfg = config if config is not None else HardNegativeSamplingConfig()
    if label_col not in frame.columns:
        raise ValueError(f"Hard-negative sampling requires label column {label_col!r}")

    labels = frame[label_col].to_numpy(dtype=np.float64)
    positive = labels > 0
    negative = ~positive

    d_raw = _as_float(frame, "d_raw", default=0.0)
    expected_lag = _as_float(frame, "expected_lag", default=0.0)
    p_nonzero = _as_float(frame, "p_nonzero", default=0.0)
    candidate_score = _as_float(frame, "candidate_score", default=0.0)
    localization_score = _as_float(frame, "localization_score", default=0.0)

    localization_threshold = _safe_quantile(
        localization_score[negative],
        quantile=float(cfg.localization_score_quantile),
        default=float("inf"),
    )
    hard_conditions = {
        "d_raw": d_raw > float(cfg.d_raw_threshold),
        "expected_lag": expected_lag > float(cfg.expected_lag_threshold),
        "p_nonzero": p_nonzero > float(cfg.p_nonzero_threshold),
        "candidate_score": candidate_score > float(cfg.candidate_score_threshold),
        "localization_score": localization_score >= float(localization_threshold),
    }
    hard_score = np.zeros(len(frame), dtype=np.int64)
    for mask in hard_conditions.values():
        hard_score += mask.astype(np.int64)

    neg_idx = np.flatnonzero(negative).astype(np.int64)
    hard_idx = np.empty(0, dtype=np.int64)
    easy_idx = np.empty(0, dtype=np.int64)
    medium_idx = np.empty(0, dtype=np.int64)
    if neg_idx.size > 0:
        neg_rank_frame = pd.DataFrame(
            {
                "row_index": neg_idx,
                "hard_score": hard_score[neg_idx],
                "d_raw": d_raw[neg_idx],
                "expected_lag": expected_lag[neg_idx],
                "p_nonzero": p_nonzero[neg_idx],
                "candidate_score": candidate_score[neg_idx],
                "localization_score": localization_score[neg_idx],
            }
        )
        n_neg = int(neg_idx.size)
        n_hard = min(
            int(math.ceil(float(cfg.hard_top_fraction) * n_neg)) if float(cfg.hard_top_fraction) > 0 else 0,
            n_neg,
        )
        n_easy = min(
            int(math.ceil(float(cfg.easy_bottom_fraction) * n_neg)) if float(cfg.easy_bottom_fraction) > 0 else 0,
            max(n_neg - n_hard, 0),
        )
        hard_rank = neg_rank_frame.sort_values(
            ["hard_score", "d_raw", "expected_lag", "p_nonzero", "candidate_score", "localization_score", "row_index"],
            ascending=[False, False, False, False, False, False, True],
            kind="mergesort",
        )
        hard_idx = hard_rank["row_index"].to_numpy(dtype=np.int64)[:n_hard]
        hard_set = set(hard_idx.tolist())
        easy_rank = neg_rank_frame.sort_values(
            ["hard_score", "d_raw", "expected_lag", "p_nonzero", "candidate_score", "localization_score", "row_index"],
            ascending=[True, True, True, True, True, True, True],
            kind="mergesort",
        )
        if hard_set:
            easy_rank = easy_rank.loc[~easy_rank["row_index"].isin(hard_set)]
        easy_idx = easy_rank["row_index"].to_numpy(dtype=np.int64)[:n_easy]
        easy_set = set(easy_idx.tolist())
        medium_idx = np.asarray(
            [idx for idx in neg_idx.tolist() if idx not in hard_set and idx not in easy_set],
            dtype=np.int64,
        )

    groups = {
        "positive": np.flatnonzero(positive).astype(np.int64),
        "hard_negative": hard_idx,
        "easy_negative": easy_idx,
        "medium_negative": medium_idx,
    }
    metadata = {
        "enabled": bool(cfg.enabled),
        "pool_counts": {name: int(idx.size) for name, idx in groups.items()},
        "empty_pools": [name for name, idx in groups.items() if int(idx.size) == 0],
        "pool_fractions_in_train": {
            name: float(idx.size / max(len(frame), 1))
            for name, idx in groups.items()
        },
        "thresholds": {
            "d_raw_threshold": float(cfg.d_raw_threshold),
            "expected_lag_threshold": float(cfg.expected_lag_threshold),
            "p_nonzero_threshold": float(cfg.p_nonzero_threshold),
            "candidate_score_threshold": float(cfg.candidate_score_threshold),
            "localization_score_quantile": float(cfg.localization_score_quantile),
            "localization_score_threshold": float(localization_threshold),
            "hard_top_fraction": float(cfg.hard_top_fraction),
            "easy_bottom_fraction": float(cfg.easy_bottom_fraction),
        },
        "trigger_counts": {
            name: int(np.logical_and(mask, negative).sum())
            for name, mask in hard_conditions.items()
        },
        "hard_score_distribution": {
            str(score): int(np.sum(hard_score[neg_idx] == score))
            for score in sorted(np.unique(hard_score[neg_idx]).tolist())
        }
        if neg_idx.size > 0
        else {},
    }
    return groups, metadata


GROUP_CODE_TO_NAME = {
    0: "positive",
    1: "hard_negative",
    2: "medium_negative",
    3: "easy_negative",
}

GROUP_NAME_TO_CODE = {name: code for code, name in GROUP_CODE_TO_NAME.items()}


def annotate_sampling_groups(
    frame: pd.DataFrame,
    label_col: str = "d_true",
    config: HardNegativeSamplingConfig | None = None,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    groups, metadata = hard_negative_sampling_groups(frame, label_col=label_col, config=config)
    out = frame.copy()
    group_code = np.full(len(out), -1, dtype=np.int64)
    for name, indices in groups.items():
        code = GROUP_NAME_TO_CODE[name]
        if indices.size > 0:
            group_code[indices] = int(code)
    if np.any(group_code < 0):
        raise ValueError("Sampling group annotation failed to assign every row")
    out["sampling_group_code"] = group_code
    out["sampling_group"] = pd.Series(group_code).map(GROUP_CODE_TO_NAME).astype(str)
    return out, metadata


def _allocate_batch_counts(
    batch_size: int,
    specs: Sequence[tuple[str, float, int]],
) -> Dict[str, int]:
    if int(batch_size) <= 0:
        raise ValueError("Batch size must be positive")
    active = [(name, float(weight), int(size)) for name, weight, size in specs if float(weight) > 0 and int(size) > 0]
    if not active:
        raise ValueError("Hard-negative sampler requires at least one non-empty sampling pool")
    if len(active) > int(batch_size):
        raise ValueError("Batch size is too small for the number of active hard-negative pools")

    total_weight = sum(weight for _, weight, _ in active)
    raw = {name: float(batch_size) * weight / max(total_weight, 1e-12) for name, weight, _ in active}
    counts = {name: max(1, int(math.floor(raw[name]))) for name, _, _ in active}

    while sum(counts.values()) < int(batch_size):
        best_name = max(
            active,
            key=lambda item: (raw[item[0]] - counts[item[0]], raw[item[0]], item[0]),
        )[0]
        counts[best_name] += 1

    while sum(counts.values()) > int(batch_size):
        reducible = [name for name, value in counts.items() if value > 1]
        if not reducible:
            break
        worst_name = min(
            reducible,
            key=lambda name: (raw[name] - counts[name], raw[name], name),
        )
        counts[worst_name] -= 1

    full = {name: 0 for name, _, _ in specs}
    full.update(counts)
    return full


class StratifiedHardNegativeBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        index_pools: Dict[str, np.ndarray],
        batch_size: int,
        steps_per_epoch: int,
        positive_fraction: float,
        hard_negative_fraction: float,
        medium_negative_fraction: float,
        easy_negative_fraction: float,
        max_hard_per_epoch: int,
        seed: int,
    ) -> None:
        self.index_pools = {
            name: np.asarray(indices, dtype=np.int64)
            for name, indices in index_pools.items()
        }
        self.batch_size = int(batch_size)
        self.steps_per_epoch = max(int(steps_per_epoch), 1)
        self.positive_fraction = float(positive_fraction)
        self.hard_negative_fraction = float(hard_negative_fraction)
        self.medium_negative_fraction = float(medium_negative_fraction)
        self.easy_negative_fraction = float(easy_negative_fraction)
        self.max_hard_per_epoch = max(int(max_hard_per_epoch), 0)
        self.seed = int(seed)
        self._epoch = 0

    def __len__(self) -> int:
        return self.steps_per_epoch

    def _make_epoch_pools(self, rng: np.random.Generator) -> Dict[str, np.ndarray]:
        positive_pool = self.index_pools.get("positive", np.empty(0, dtype=np.int64))
        hard_pool_full = self.index_pools.get("hard_negative", np.empty(0, dtype=np.int64))
        easy_pool = self.index_pools.get("easy_negative", np.empty(0, dtype=np.int64))
        medium_pool = self.index_pools.get("medium_negative", np.empty(0, dtype=np.int64))

        if self.max_hard_per_epoch > 0 and hard_pool_full.size > self.max_hard_per_epoch:
            hard_pool = rng.choice(hard_pool_full, size=self.max_hard_per_epoch, replace=False).astype(np.int64)
        else:
            hard_pool = hard_pool_full

        easy_available = easy_pool
        if easy_available.size == 0 and medium_pool.size > 0:
            easy_available = medium_pool
        else:
            preview_counts = _allocate_batch_counts(
                batch_size=self.batch_size,
                specs=[
                    ("positive", self.positive_fraction, int(positive_pool.size)),
                    ("hard_negative", self.hard_negative_fraction, int(hard_pool.size)),
                    ("easy_negative", self.easy_negative_fraction, int(easy_available.size + medium_pool.size)),
                ],
            )
            desired_easy_unique = int(preview_counts["easy_negative"]) * int(self.steps_per_epoch)
            if easy_available.size < desired_easy_unique and medium_pool.size > 0:
                medium_candidates = medium_pool
                if easy_available.size > 0:
                    easy_set = set(easy_available.tolist())
                    medium_candidates = np.asarray(
                        [idx for idx in medium_pool.tolist() if idx not in easy_set],
                        dtype=np.int64,
                    )
                need = min(max(desired_easy_unique - int(easy_available.size), 0), int(medium_candidates.size))
                if need > 0:
                    supplement = rng.choice(medium_candidates, size=need, replace=False).astype(np.int64)
                    easy_available = np.concatenate([easy_available, supplement], axis=0)

        return {
            "positive": positive_pool,
            "hard_negative": hard_pool,
            "medium_negative": medium_pool,
            "easy_negative": easy_available,
        }

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1
        epoch_pools = self._make_epoch_pools(rng)
        batch_counts = _allocate_batch_counts(
            batch_size=self.batch_size,
            specs=[
                ("positive", self.positive_fraction, int(epoch_pools["positive"].size)),
                ("hard_negative", self.hard_negative_fraction, int(epoch_pools["hard_negative"].size)),
                ("medium_negative", self.medium_negative_fraction, int(epoch_pools["medium_negative"].size)),
                ("easy_negative", self.easy_negative_fraction, int(epoch_pools["easy_negative"].size)),
            ],
        )
        pool_names = [name for name, count in batch_counts.items() if count > 0]
        for _ in range(self.steps_per_epoch):
            parts: List[np.ndarray] = []
            for name in pool_names:
                count = int(batch_counts[name])
                pool = epoch_pools[name]
                if count <= 0 or pool.size == 0:
                    continue
                parts.append(rng.choice(pool, size=count, replace=True).astype(np.int64))
            if not parts:
                raise RuntimeError("Hard-negative sampler produced an empty batch")
            batch = np.concatenate(parts, axis=0)
            rng.shuffle(batch)
            yield batch.tolist()


def _build_train_loader(
    frame: pd.DataFrame,
    normalizer: FeatureNormalizer,
    label_col: str,
    batch_size: int,
    seed: int,
    hard_negative_sampling: HardNegativeSamplingConfig | None,
) -> tuple[DataLoader, Dict[str, Any]]:
    dataset = _tensor_dataset(frame, normalizer, label_col=label_col)
    cfg = hard_negative_sampling if hard_negative_sampling is not None else HardNegativeSamplingConfig()
    if not bool(cfg.enabled):
        return DataLoader(dataset, batch_size=int(batch_size), shuffle=True), {
            "enabled": False,
            "mode": "shuffle",
            "batch_size": int(batch_size),
            "steps_per_epoch": int(max(math.ceil(len(frame) / max(int(batch_size), 1)), 1)),
        }

    groups, group_meta = hard_negative_sampling_groups(frame, label_col=label_col, config=cfg)
    positive_count = int(groups["positive"].size)
    max_hard_per_epoch = int(max(float(cfg.max_hard_per_positive) * max(positive_count, 0), 0.0))
    sampler = StratifiedHardNegativeBatchSampler(
        index_pools=groups,
        batch_size=int(batch_size),
        steps_per_epoch=max(int(math.ceil(len(frame) / max(int(batch_size), 1))), 1),
        positive_fraction=float(cfg.positive_fraction),
        hard_negative_fraction=float(cfg.hard_negative_fraction),
        medium_negative_fraction=float(cfg.medium_negative_fraction),
        easy_negative_fraction=float(cfg.easy_negative_fraction),
        max_hard_per_epoch=max_hard_per_epoch,
        seed=int(seed),
    )
    preview_rng = np.random.default_rng(int(seed))
    preview_pools = sampler._make_epoch_pools(preview_rng)
    preview_batch_counts = _allocate_batch_counts(
        batch_size=int(batch_size),
        specs=[
            ("positive", float(cfg.positive_fraction), int(preview_pools["positive"].size)),
            ("hard_negative", float(cfg.hard_negative_fraction), int(preview_pools["hard_negative"].size)),
            ("medium_negative", float(cfg.medium_negative_fraction), int(preview_pools["medium_negative"].size)),
            ("easy_negative", float(cfg.easy_negative_fraction), int(preview_pools["easy_negative"].size)),
        ],
    )
    sampling_info = {
        "enabled": True,
        "mode": "stratified_hard_negative",
        "batch_size": int(batch_size),
        "steps_per_epoch": int(len(sampler)),
        "batch_counts": {name: int(count) for name, count in preview_batch_counts.items()},
        "target_batch_fractions": {
            "positive": float(cfg.positive_fraction),
            "hard_negative": float(cfg.hard_negative_fraction),
            "medium_negative": float(cfg.medium_negative_fraction),
            "easy_negative": float(cfg.easy_negative_fraction),
        },
        "effective_batch_fractions": {
            name: float(count / max(int(batch_size), 1))
            for name, count in preview_batch_counts.items()
        },
        "active_pools": [name for name, count in preview_batch_counts.items() if int(count) > 0],
        "max_hard_per_epoch": int(max_hard_per_epoch),
        "preview_epoch_pool_counts": {
            name: int(pool.size)
            for name, pool in preview_pools.items()
        },
    }
    sampling_info.update(group_meta)
    return DataLoader(dataset, batch_sampler=sampler), sampling_info


def _tensor_dataset(
    frame: pd.DataFrame,
    normalizer: FeatureNormalizer,
    label_col: str,
) -> TensorDataset:
    if label_col not in frame.columns:
        raise ValueError(f"Unified scorer training requires label column {label_col!r}")
    x = torch.from_numpy(normalizer.transform(frame)).float()
    labels = (frame[label_col].to_numpy(dtype=np.float64) > 0).astype(np.float32)
    y = torch.from_numpy(labels).float()
    d_raw = torch.from_numpy(frame["d_raw"].to_numpy(dtype=np.float64).astype(np.float32)).float()
    target = torch.from_numpy(frame[label_col].to_numpy(dtype=np.float64).astype(np.float32)).float()
    if "sampling_group_code" in frame.columns:
        group_code = torch.from_numpy(frame["sampling_group_code"].to_numpy(dtype=np.int64)).long()
    else:
        default_code = np.where(labels > 0, GROUP_NAME_TO_CODE["positive"], GROUP_NAME_TO_CODE["easy_negative"])
        group_code = torch.from_numpy(default_code.astype(np.int64)).long()
    if "teacher_target" in frame.columns:
        teacher_target = torch.from_numpy(frame["teacher_target"].to_numpy(dtype=np.float64).astype(np.float32)).float()
    else:
        teacher_target = torch.full((len(frame),), -1.0, dtype=torch.float32)
    return TensorDataset(x, y, d_raw, target, group_code, teacher_target)


def _loss_on_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    pos_loss_weight: float,
    hard_neg_loss_weight: float,
    medium_neg_loss_weight: float,
    easy_neg_loss_weight: float,
    mag_loss_weight: float,
    zero_loss_weight: float,
    rate_loss_weight: float,
    rate_target_slack: float,
    rank_loss_weight: float,
    rank_margin: float,
    teacher_loss_weight: float,
) -> Dict[str, float]:
    total = 0.0
    total_pos = 0.0
    total_hard_neg = 0.0
    total_medium_neg = 0.0
    total_easy_neg = 0.0
    total_mag = 0.0
    total_zero = 0.0
    total_rate = 0.0
    total_rank = 0.0
    total_teacher = 0.0
    rows = 0
    model.eval()
    with torch.no_grad():
        for x, y, d_raw, target, group_code, teacher_target in loader:
            x = x.to(device)
            y = y.to(device)
            d_raw = d_raw.to(device)
            target = target.to(device)
            group_code = group_code.to(device)
            teacher_target = teacher_target.to(device)
            logits = model(x)
            c = torch.sigmoid(logits)
            positive = group_code == int(GROUP_NAME_TO_CODE["positive"])
            hard_neg = group_code == int(GROUP_NAME_TO_CODE["hard_negative"])
            medium_neg = group_code == int(GROUP_NAME_TO_CODE["medium_negative"])
            easy_neg = group_code == int(GROUP_NAME_TO_CODE["easy_negative"])
            if torch.any(positive):
                pos_loss = F.binary_cross_entropy_with_logits(logits[positive], torch.ones_like(logits[positive]))
            else:
                pos_loss = logits.new_tensor(0.0)
            if torch.any(hard_neg):
                hard_neg_loss = F.binary_cross_entropy_with_logits(logits[hard_neg], torch.zeros_like(logits[hard_neg]))
            else:
                hard_neg_loss = logits.new_tensor(0.0)
            if torch.any(medium_neg):
                medium_neg_loss = F.binary_cross_entropy_with_logits(logits[medium_neg], torch.zeros_like(logits[medium_neg]))
            else:
                medium_neg_loss = logits.new_tensor(0.0)
            if torch.any(easy_neg):
                easy_neg_loss = F.binary_cross_entropy_with_logits(logits[easy_neg], torch.zeros_like(logits[easy_neg]))
            else:
                easy_neg_loss = logits.new_tensor(0.0)
            if torch.any(positive):
                mag = F.smooth_l1_loss(c[positive] * d_raw[positive], target[positive])
            else:
                mag = logits.new_tensor(0.0)
            zero = y <= 0.5
            if torch.any(zero):
                zero_loss = torch.mean((c[zero] * d_raw[zero]) ** 2)
            else:
                zero_loss = logits.new_tensor(0.0)
            batch_positive_rate = float(positive.float().mean().item()) if int(positive.numel()) > 0 else 0.0
            target_rate = batch_positive_rate + float(rate_target_slack)
            rate_loss = F.relu(c.mean() - target_rate) ** 2
            rank_terms: List[torch.Tensor] = []
            if torch.any(positive) and torch.any(hard_neg):
                rank_terms.append(F.relu(float(rank_margin) - (c[positive].mean() - c[hard_neg].mean())))
            if torch.any(positive) and torch.any(medium_neg):
                rank_terms.append(0.5 * F.relu(float(rank_margin) - (c[positive].mean() - c[medium_neg].mean())))
            if torch.any(positive) and torch.any(easy_neg):
                rank_terms.append(0.5 * F.relu(float(rank_margin) - (c[positive].mean() - c[easy_neg].mean())))
            rank_loss = torch.stack(rank_terms).sum() if rank_terms else logits.new_tensor(0.0)
            teacher_mask = teacher_target >= 0.0
            if torch.any(teacher_mask):
                teacher_loss = F.binary_cross_entropy_with_logits(
                    logits[teacher_mask],
                    teacher_target[teacher_mask],
                )
            else:
                teacher_loss = logits.new_tensor(0.0)
            loss = (
                float(pos_loss_weight) * pos_loss
                + float(hard_neg_loss_weight) * hard_neg_loss
                + float(medium_neg_loss_weight) * medium_neg_loss
                + float(easy_neg_loss_weight) * easy_neg_loss
                + float(mag_loss_weight) * mag
                + float(zero_loss_weight) * zero_loss
                + float(rate_loss_weight) * rate_loss
                + float(rank_loss_weight) * rank_loss
                + float(teacher_loss_weight) * teacher_loss
            )
            total += float(loss.item()) * int(y.numel())
            total_pos += float(pos_loss.item()) * int(y.numel())
            total_hard_neg += float(hard_neg_loss.item()) * int(y.numel())
            total_medium_neg += float(medium_neg_loss.item()) * int(y.numel())
            total_easy_neg += float(easy_neg_loss.item()) * int(y.numel())
            total_mag += float(mag.item()) * int(y.numel())
            total_zero += float(zero_loss.item()) * int(y.numel())
            total_rate += float(rate_loss.item()) * int(y.numel())
            total_rank += float(rank_loss.item()) * int(y.numel())
            total_teacher += float(teacher_loss.item()) * int(y.numel())
            rows += int(y.numel())
    return {
        "loss": total / max(rows, 1),
        "pos": total_pos / max(rows, 1),
        "hard_neg": total_hard_neg / max(rows, 1),
        "medium_neg": total_medium_neg / max(rows, 1),
        "easy_neg": total_easy_neg / max(rows, 1),
        "mag": total_mag / max(rows, 1),
        "zero": total_zero / max(rows, 1),
        "rate": total_rate / max(rows, 1),
        "rank": total_rank / max(rows, 1),
        "teacher": total_teacher / max(rows, 1),
    }


def train_unified_scorer(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    label_col: str = "d_true",
    hidden_dim: int = 32,
    dropout: float = 0.10,
    epochs: int = 120,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    pos_loss_weight: float = 2.0,
    hard_neg_loss_weight: float = 1.0,
    medium_neg_loss_weight: float = 0.5,
    easy_neg_loss_weight: float = 0.5,
    mag_loss_weight: float = 1.0,
    zero_loss_weight: float = 0.0,
    rate_loss_weight: float = 0.0,
    rate_target_slack: float = 0.10,
    rank_loss_weight: float = 0.0,
    rank_margin: float = 0.10,
    teacher_loss_weight: float = 0.0,
    pos_weight_cap: float = 20.0,
    hard_negative_sampling: HardNegativeSamplingConfig | None = None,
    seed: int = 42,
    device: str | torch.device = "cpu",
) -> tuple[UnifiedLagScorer, FeatureNormalizer, pd.DataFrame, Dict[str, Any]]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    normalizer = fit_feature_normalizer(train_frame, feature_columns)
    train_loader, sampling_info = _build_train_loader(
        train_frame,
        normalizer=normalizer,
        label_col=label_col,
        batch_size=int(batch_size),
        seed=int(seed),
        hard_negative_sampling=hard_negative_sampling,
    )
    val_dataset = _tensor_dataset(val_frame, normalizer, label_col=label_col)
    val_loader = DataLoader(val_dataset, batch_size=int(batch_size), shuffle=False)

    torch_device = torch.device(device)
    model = UnifiedLagScorer(input_dim=len(feature_columns), hidden_dim=hidden_dim, dropout=dropout).to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    history: List[Dict[str, float]] = []
    best_state = None
    best_val = float("inf")
    for epoch in range(1, int(epochs) + 1):
        model.train()
        total = 0.0
        total_pos = 0.0
        total_hard_neg = 0.0
        total_medium_neg = 0.0
        total_easy_neg = 0.0
        total_mag = 0.0
        total_zero = 0.0
        total_rate = 0.0
        total_rank = 0.0
        total_teacher = 0.0
        rows = 0
        for x, y, d_raw, target, group_code, teacher_target in train_loader:
            x = x.to(torch_device)
            y = y.to(torch_device)
            d_raw = d_raw.to(torch_device)
            target = target.to(torch_device)
            group_code = group_code.to(torch_device)
            teacher_target = teacher_target.to(torch_device)
            logits = model(x)
            c = torch.sigmoid(logits)
            positive = group_code == int(GROUP_NAME_TO_CODE["positive"])
            hard_neg = group_code == int(GROUP_NAME_TO_CODE["hard_negative"])
            medium_neg = group_code == int(GROUP_NAME_TO_CODE["medium_negative"])
            easy_neg = group_code == int(GROUP_NAME_TO_CODE["easy_negative"])
            if torch.any(positive):
                pos_loss = F.binary_cross_entropy_with_logits(logits[positive], torch.ones_like(logits[positive]))
            else:
                pos_loss = logits.new_tensor(0.0)
            if torch.any(hard_neg):
                hard_neg_loss = F.binary_cross_entropy_with_logits(logits[hard_neg], torch.zeros_like(logits[hard_neg]))
            else:
                hard_neg_loss = logits.new_tensor(0.0)
            if torch.any(medium_neg):
                medium_neg_loss = F.binary_cross_entropy_with_logits(logits[medium_neg], torch.zeros_like(logits[medium_neg]))
            else:
                medium_neg_loss = logits.new_tensor(0.0)
            if torch.any(easy_neg):
                easy_neg_loss = F.binary_cross_entropy_with_logits(logits[easy_neg], torch.zeros_like(logits[easy_neg]))
            else:
                easy_neg_loss = logits.new_tensor(0.0)
            if torch.any(positive):
                mag = F.smooth_l1_loss(c[positive] * d_raw[positive], target[positive])
            else:
                mag = logits.new_tensor(0.0)
            zero = y <= 0.5
            if torch.any(zero):
                zero_loss = torch.mean((c[zero] * d_raw[zero]) ** 2)
            else:
                zero_loss = logits.new_tensor(0.0)
            batch_positive_rate = float(positive.float().mean().item()) if int(positive.numel()) > 0 else 0.0
            target_rate = batch_positive_rate + float(rate_target_slack)
            rate_loss = F.relu(c.mean() - target_rate) ** 2
            rank_terms: List[torch.Tensor] = []
            if torch.any(positive) and torch.any(hard_neg):
                rank_terms.append(F.relu(float(rank_margin) - (c[positive].mean() - c[hard_neg].mean())))
            if torch.any(positive) and torch.any(medium_neg):
                rank_terms.append(0.5 * F.relu(float(rank_margin) - (c[positive].mean() - c[medium_neg].mean())))
            if torch.any(positive) and torch.any(easy_neg):
                rank_terms.append(0.5 * F.relu(float(rank_margin) - (c[positive].mean() - c[easy_neg].mean())))
            rank_loss = torch.stack(rank_terms).sum() if rank_terms else logits.new_tensor(0.0)
            teacher_mask = teacher_target >= 0.0
            if torch.any(teacher_mask):
                teacher_loss = F.binary_cross_entropy_with_logits(
                    logits[teacher_mask],
                    teacher_target[teacher_mask],
                )
            else:
                teacher_loss = logits.new_tensor(0.0)
            loss = (
                float(pos_loss_weight) * pos_loss
                + float(hard_neg_loss_weight) * hard_neg_loss
                + float(medium_neg_loss_weight) * medium_neg_loss
                + float(easy_neg_loss_weight) * easy_neg_loss
                + float(mag_loss_weight) * mag
                + float(zero_loss_weight) * zero_loss
                + float(rate_loss_weight) * rate_loss
                + float(rank_loss_weight) * rank_loss
                + float(teacher_loss_weight) * teacher_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * int(y.numel())
            total_pos += float(pos_loss.item()) * int(y.numel())
            total_hard_neg += float(hard_neg_loss.item()) * int(y.numel())
            total_medium_neg += float(medium_neg_loss.item()) * int(y.numel())
            total_easy_neg += float(easy_neg_loss.item()) * int(y.numel())
            total_mag += float(mag.item()) * int(y.numel())
            total_zero += float(zero_loss.item()) * int(y.numel())
            total_rate += float(rate_loss.item()) * int(y.numel())
            total_rank += float(rank_loss.item()) * int(y.numel())
            total_teacher += float(teacher_loss.item()) * int(y.numel())
            rows += int(y.numel())

        train_stats = {
            "loss": total / max(rows, 1),
            "pos": total_pos / max(rows, 1),
            "hard_neg": total_hard_neg / max(rows, 1),
            "medium_neg": total_medium_neg / max(rows, 1),
            "easy_neg": total_easy_neg / max(rows, 1),
            "mag": total_mag / max(rows, 1),
            "zero": total_zero / max(rows, 1),
            "rate": total_rate / max(rows, 1),
            "rank": total_rank / max(rows, 1),
            "teacher": total_teacher / max(rows, 1),
        }
        val_stats = _loss_on_loader(
            model,
            val_loader,
            device=torch_device,
            pos_loss_weight=pos_loss_weight,
            hard_neg_loss_weight=hard_neg_loss_weight,
            medium_neg_loss_weight=medium_neg_loss_weight,
            easy_neg_loss_weight=easy_neg_loss_weight,
            mag_loss_weight=mag_loss_weight,
            zero_loss_weight=zero_loss_weight,
            rate_loss_weight=rate_loss_weight,
            rate_target_slack=rate_target_slack,
            rank_loss_weight=rank_loss_weight,
            rank_margin=rank_margin,
            teacher_loss_weight=teacher_loss_weight,
        )
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_stats["loss"]),
                "train_pos": float(train_stats["pos"]),
                "train_hard_neg": float(train_stats["hard_neg"]),
                "train_medium_neg": float(train_stats["medium_neg"]),
                "train_easy_neg": float(train_stats["easy_neg"]),
                "train_mag": float(train_stats["mag"]),
                "train_zero": float(train_stats["zero"]),
                "train_rate": float(train_stats["rate"]),
                "train_rank": float(train_stats["rank"]),
                "train_teacher": float(train_stats["teacher"]),
                "val_loss": float(val_stats["loss"]),
                "val_pos": float(val_stats["pos"]),
                "val_hard_neg": float(val_stats["hard_neg"]),
                "val_medium_neg": float(val_stats["medium_neg"]),
                "val_easy_neg": float(val_stats["easy_neg"]),
                "val_mag": float(val_stats["mag"]),
                "val_zero": float(val_stats["zero"]),
                "val_rate": float(val_stats["rate"]),
                "val_rank": float(val_stats["rank"]),
                "val_teacher": float(val_stats["teacher"]),
            }
        )
        if val_stats["loss"] < best_val:
            best_val = float(val_stats["loss"])
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, normalizer, pd.DataFrame(history), sampling_info


def diagnostic_score_table(
    frames_by_split: Dict[str, pd.DataFrame],
    group_col: str = "sampling_group",
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    target_groups = ["positive", "hard_negative", "easy_negative"]
    for split_name, frame in frames_by_split.items():
        if group_col not in frame.columns:
            continue
        for group_name in target_groups:
            local = frame.loc[frame[group_col].astype(str) == str(group_name)].copy()
            confidence = local["unified_confidence"].to_numpy(dtype=np.float64) if "unified_confidence" in local.columns else np.asarray([], dtype=np.float64)
            dsoft = local["unified_d_soft"].to_numpy(dtype=np.float64) if "unified_d_soft" in local.columns else np.asarray([], dtype=np.float64)
            rows.append(
                {
                    "split": str(split_name),
                    "group": str(group_name),
                    "count": int(len(local)),
                    "c_p10": float(np.nanquantile(confidence, 0.10)) if confidence.size else float("nan"),
                    "c_p50": float(np.nanquantile(confidence, 0.50)) if confidence.size else float("nan"),
                    "c_p90": float(np.nanquantile(confidence, 0.90)) if confidence.size else float("nan"),
                    "dsoft_p50": float(np.nanquantile(dsoft, 0.50)) if dsoft.size else float("nan"),
                    "dsoft_p90": float(np.nanquantile(dsoft, 0.90)) if dsoft.size else float("nan"),
                }
            )
    return pd.DataFrame(rows)


@torch.no_grad()
def predict_unified_scorer(
    model: nn.Module,
    frame: pd.DataFrame,
    normalizer: FeatureNormalizer,
    threshold: float = 0.5,
    rank_threshold: float = 0.0,
    batch_size: int = 4096,
    device: str | torch.device = "cpu",
) -> pd.DataFrame:
    torch_device = torch.device(device)
    model.to(torch_device)
    model.eval()
    x = torch.from_numpy(normalizer.transform(frame)).float()
    loader = DataLoader(TensorDataset(x), batch_size=int(batch_size), shuffle=False)
    scores: List[np.ndarray] = []
    for (batch_x,) in loader:
        logits = model(batch_x.to(torch_device))
        scores.append(torch.sigmoid(logits).detach().cpu().numpy())
    confidence = np.concatenate(scores, axis=0) if scores else np.zeros(len(frame), dtype=np.float64)

    out = frame.copy()
    d_raw = out["d_raw"].to_numpy(dtype=np.float64)
    d_soft = confidence * d_raw
    confidence_rank = _percentile_rank_by_group(
        out,
        confidence.astype(np.float64),
        group_col="segment_id" if "segment_id" in out.columns else "",
    )
    selected = (confidence >= float(threshold)) & (confidence_rank >= float(rank_threshold))
    out["unified_confidence"] = confidence
    out["unified_confidence_rank"] = confidence_rank
    out["unified_d_soft"] = d_soft
    out["unified_selected"] = selected.astype(int)
    out["unified_d_hat"] = np.where(selected, np.rint(d_soft), 0.0)
    out["unified_rank_threshold"] = float(rank_threshold)
    out["p_pos"] = out["unified_confidence"]
    out["d_hat"] = out["unified_d_hat"]
    out["peak_score"] = out["p_pos"] * out["d_hat"]
    return out


def _runs(mask: np.ndarray) -> List[tuple[int, int]]:
    runs: List[tuple[int, int]] = []
    start = None
    for i, value in enumerate(mask.astype(bool)):
        if value and start is None:
            start = i
        elif not value and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _positive_blocks(frame: pd.DataFrame, label_col: str, group_col: str) -> List[np.ndarray]:
    labels = frame[label_col].to_numpy(dtype=np.float64)
    blocks: List[np.ndarray] = []
    if group_col not in frame.columns:
        groups = [np.arange(len(frame), dtype=int)]
    else:
        groups = [frame.index.get_indexer(idx) for idx in frame.groupby(group_col, sort=False).groups.values()]
    for idx_arr in groups:
        local = labels[idx_arr] > 0
        for start, end in _runs(local):
            blocks.append(idx_arr[start : end + 1])
    return blocks


def selection_metrics(
    frame: pd.DataFrame,
    label_col: str = "d_true",
    group_col: str = "segment_id",
    selected_col: str = "unified_selected",
    d_hat_col: str = "unified_d_hat",
    score_col: str = "unified_confidence",
) -> Dict[str, Any]:
    for col in [label_col, selected_col, d_hat_col]:
        if col not in frame.columns:
            raise ValueError(f"Unified scorer metrics require column {col!r}")
    labels = frame[label_col].to_numpy(dtype=np.float64)
    true = labels > 0
    pred = frame[selected_col].to_numpy(dtype=np.float64) > 0
    d_hat = frame[d_hat_col].to_numpy(dtype=np.float64)
    tp = int(np.logical_and(pred, true).sum())
    fp = int(np.logical_and(pred, ~true).sum())
    fn = int(np.logical_and(~pred, true).sum())
    tn = int(np.logical_and(~pred, ~true).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    metrics: Dict[str, Any] = {
        "n_rows": int(len(frame)),
        "n_positive": int(true.sum()),
        "n_selected": int(pred.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "overall_recall": float(recall),
        "FAR": float(fp / max(fp + tn, 1)),
        "precision": float(precision),
        "F1": float(2.0 * precision * recall / max(precision + recall, 1e-12)),
        "pos_MAE": float(np.mean(np.abs(d_hat[true] - labels[true]))) if true.any() else float("nan"),
        "zero_E_d_hat": float(np.mean(d_hat[~true])) if (~true).any() else float("nan"),
        "selected_zero_E_d_hat": float(np.mean(d_hat[np.logical_and(pred, ~true)]))
        if np.any(np.logical_and(pred, ~true))
        else 0.0,
    }
    if score_col in frame.columns and true.any() and (~true).any():
        scores = frame[score_col].to_numpy(dtype=np.float64)
        metrics["AUPRC"] = float(average_precision_score(true.astype(int), scores))
        metrics["AUROC"] = float(roc_auc_score(true.astype(int), scores))
    else:
        metrics["AUPRC"] = float("nan")
        metrics["AUROC"] = float("nan")

    for value in sorted(float(v) for v in pd.Series(labels[true]).dropna().unique()):
        group = labels == value
        key = f"d{int(value)}"
        metrics[f"{key}_recall"] = float(np.logical_and(pred, group).sum() / max(int(group.sum()), 1))
        metrics[f"{key}_selected"] = int(np.logical_and(pred, group).sum())

    blocks = _positive_blocks(frame, label_col=label_col, group_col=group_col)
    if blocks:
        peak_errors = []
        peak_hits = []
        peak_false_alarm_lengths = []
        for block in blocks:
            true_peak = float(np.nanmax(labels[block]))
            pred_peak = float(np.nanmax(d_hat[block]))
            peak_errors.append(abs(pred_peak - true_peak))
            peak_hits.append(float(abs(int(np.floor(pred_peak + 0.5)) - int(true_peak)) <= 1))
        for start, end in _runs(np.logical_and(pred, ~true)):
            peak_false_alarm_lengths.append(end - start + 1)
        metrics.update(
            {
                "peak_error": float(np.mean(peak_errors)),
                "peak_hit_at_pm1": float(np.mean(peak_hits)),
                "n_positive_blocks": int(len(blocks)),
                "n_false_alarm_runs": int(len(peak_false_alarm_lengths)),
                "mean_false_alarm_run_length": float(np.mean(peak_false_alarm_lengths))
                if peak_false_alarm_lengths
                else 0.0,
            }
        )
    else:
        metrics.update(
            {
                "peak_error": float("nan"),
                "peak_hit_at_pm1": float("nan"),
                "n_positive_blocks": 0,
                "n_false_alarm_runs": 0,
                "mean_false_alarm_run_length": 0.0,
            }
        )
    metrics["selector_score"] = selector_score(metrics)
    return metrics


def selector_score(metrics: Dict[str, Any]) -> float:
    peak_hit = 0.0 if not np.isfinite(float(metrics.get("peak_hit_at_pm1", 0.0))) else float(metrics["peak_hit_at_pm1"])
    peak_error = 0.0 if not np.isfinite(float(metrics.get("peak_error", 0.0))) else float(metrics["peak_error"])
    return float(
        3.0 * float(metrics["overall_recall"])
        - float(metrics["FAR"])
        + 2.0 * peak_hit
        - 0.25 * peak_error
    )


def apply_threshold(
    frame: pd.DataFrame,
    threshold: float,
    d_floor: float = 0.0,
    rank_threshold: float = 0.0,
) -> pd.DataFrame:
    out = frame.copy()
    confidence = out["unified_confidence"].to_numpy(dtype=np.float64)
    if "unified_confidence_rank" in out.columns:
        confidence_rank = out["unified_confidence_rank"].to_numpy(dtype=np.float64)
    else:
        confidence_rank = _percentile_rank_by_group(
            out,
            confidence.astype(np.float64),
            group_col="segment_id" if "segment_id" in out.columns else "",
        )
        out["unified_confidence_rank"] = confidence_rank
    d_soft = out["unified_d_soft"].to_numpy(dtype=np.float64)
    selected = (
        (confidence >= float(threshold))
        & (confidence_rank >= float(rank_threshold))
        & (d_soft >= float(d_floor))
    )
    out["unified_selected"] = selected.astype(int)
    out["unified_d_hat"] = np.where(selected, np.rint(d_soft), 0.0)
    out["unified_threshold"] = float(threshold)
    out["unified_d_floor"] = float(d_floor)
    out["unified_rank_threshold"] = float(rank_threshold)
    out["p_pos"] = out["unified_confidence"]
    out["d_hat"] = out["unified_d_hat"]
    out["peak_score"] = out["p_pos"] * out["d_hat"]
    return out


def threshold_grid(
    scored_frame: pd.DataFrame,
    thresholds: Sequence[float],
    rank_thresholds: Sequence[float] | None = None,
    label_col: str = "d_true",
    group_col: str = "segment_id",
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    active_rank_thresholds = list(rank_thresholds) if rank_thresholds is not None else [0.0]
    for threshold in thresholds:
        for rank_threshold in active_rank_thresholds:
            frame = apply_threshold(
                scored_frame,
                threshold=float(threshold),
                d_floor=0.0,
                rank_threshold=float(rank_threshold),
            )
            rows.append(
                {
                    "threshold": float(threshold),
                    "d_floor": 0.0,
                    "rank_threshold": float(rank_threshold),
                    **selection_metrics(frame, label_col=label_col, group_col=group_col),
                }
            )
    return pd.DataFrame(rows)


def threshold_dfloor_grid(
    scored_frame: pd.DataFrame,
    thresholds: Sequence[float],
    d_floors: Sequence[float],
    rank_thresholds: Sequence[float] | None = None,
    label_col: str = "d_true",
    group_col: str = "segment_id",
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    active_rank_thresholds = list(rank_thresholds) if rank_thresholds is not None else [0.0]
    for threshold in thresholds:
        for d_floor in d_floors:
            for rank_threshold in active_rank_thresholds:
                frame = apply_threshold(
                    scored_frame,
                    threshold=float(threshold),
                    d_floor=float(d_floor),
                    rank_threshold=float(rank_threshold),
                )
                rows.append(
                    {
                        "threshold": float(threshold),
                        "d_floor": float(d_floor),
                        "rank_threshold": float(rank_threshold),
                        **selection_metrics(frame, label_col=label_col, group_col=group_col),
                    }
                )
    return pd.DataFrame(rows)


def select_threshold(grid: pd.DataFrame) -> float:
    ranked = grid.sort_values(
        [
            "selector_score",
            "overall_recall",
            "FAR",
            "AUPRC",
            "peak_hit_at_pm1",
            "peak_error",
            "threshold",
            "rank_threshold",
            "d_floor",
        ],
        ascending=[False, False, True, False, False, True, True, True, True],
    ).reset_index(drop=True)
    return float(ranked.iloc[0]["threshold"])


def select_threshold_with_constraints(
    grid: pd.DataFrame,
    recall_min: float = 0.5,
    far_max: float = 0.6,
    mae_weight: float = 0.3,
) -> Dict[str, Any]:
    required = {"threshold", "d_floor", "rank_threshold", "overall_recall", "FAR", "pos_MAE"}
    missing = [col for col in required if col not in grid.columns]
    if missing:
        raise ValueError(f"Threshold constraint search is missing columns: {', '.join(missing)}")
    working = grid.copy()
    working["constraint_valid"] = (
        (working["overall_recall"].to_numpy(dtype=np.float64) >= float(recall_min))
        & (working["FAR"].to_numpy(dtype=np.float64) <= float(far_max))
    ).astype(int)
    mae = working["pos_MAE"].to_numpy(dtype=np.float64)
    mae = np.where(np.isfinite(mae), mae, 999.0)
    working["constraint_cost"] = working["FAR"].to_numpy(dtype=np.float64) + float(mae_weight) * mae

    feasible = working.loc[working["constraint_valid"].to_numpy(dtype=np.int64) > 0].copy()
    if not feasible.empty:
        ranked = feasible.sort_values(
            ["constraint_cost", "FAR", "pos_MAE", "overall_recall", "threshold", "rank_threshold", "d_floor"],
            ascending=[True, True, True, False, True, True, True],
        ).reset_index(drop=True)
        selected = ranked.iloc[0]
        return {
            "selection_status": "valid",
            "threshold": float(selected["threshold"]),
            "d_floor": float(selected["d_floor"]),
            "rank_threshold": float(selected["rank_threshold"]),
            "constraint_cost": float(selected["constraint_cost"]),
            "metrics": selected.to_dict(),
        }

    ranked = working.sort_values(
        ["constraint_cost", "FAR", "pos_MAE", "overall_recall", "threshold", "rank_threshold", "d_floor"],
        ascending=[True, True, True, False, True, True, True],
    ).reset_index(drop=True)
    selected = ranked.iloc[0]
    return {
        "selection_status": "no_valid_threshold",
        "threshold": float(selected["threshold"]),
        "d_floor": float(selected["d_floor"]),
        "rank_threshold": float(selected["rank_threshold"]),
        "constraint_cost": float(selected["constraint_cost"]),
        "metrics": selected.to_dict(),
    }


def select_far_constrained_threshold(
    grid: pd.DataFrame,
    target_far: float,
    far_tolerance: float = 0.0,
) -> Dict[str, Any]:
    if "threshold" not in grid.columns or "d_floor" not in grid.columns:
        raise ValueError("FAR-constrained threshold search requires threshold and d_floor columns")
    working = grid.copy()
    working["target_far"] = float(target_far)
    working["far_gap"] = np.abs(working["FAR"].to_numpy(dtype=np.float64) - float(target_far))
    feasible = working.loc[working["FAR"].to_numpy(dtype=np.float64) <= float(target_far) + float(far_tolerance)].copy()
    if not feasible.empty:
        ranked = feasible.sort_values(
            [
                "overall_recall",
                "peak_hit_at_pm1",
                "pos_MAE",
                "far_gap",
                "threshold",
                "d_floor",
            ],
            ascending=[False, False, True, True, True, True],
        ).reset_index(drop=True)
        selected = ranked.iloc[0]
        return {
            "selection_status": "feasible_under_far_cap",
            "target_far": float(target_far),
            "threshold": float(selected["threshold"]),
            "d_floor": float(selected["d_floor"]),
            "far_gap": float(selected["far_gap"]),
            "metrics": selected.to_dict(),
        }

    ranked = working.sort_values(
        [
            "far_gap",
            "overall_recall",
            "peak_hit_at_pm1",
            "pos_MAE",
            "FAR",
            "threshold",
            "d_floor",
        ],
        ascending=[True, False, False, True, True, True, True],
    ).reset_index(drop=True)
    selected = ranked.iloc[0]
    return {
        "selection_status": "closest_far_match",
        "target_far": float(target_far),
        "threshold": float(selected["threshold"]),
        "d_floor": float(selected["d_floor"]),
        "far_gap": float(selected["far_gap"]),
        "metrics": selected.to_dict(),
    }


def compact_output(frame: pd.DataFrame, label_col: str = "d_true", group_col: str = "segment_id") -> pd.DataFrame:
    cols: List[str] = []
    for col in ["split", "source_split", "timestamp", "TimeStamp", "raw_row_index", group_col, "t", "block_id", label_col]:
        if col in frame.columns and col not in cols:
            cols.append(col)
    cols.extend(
        [
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
            "unified_confidence",
            "unified_confidence_rank",
            "unified_d_soft",
            "unified_selected",
            "unified_d_hat",
            "unified_threshold",
            "unified_d_floor",
            "unified_rank_threshold",
            "peak_score",
        ]
    )
    return frame[[col for col in cols if col in frame.columns]].copy()


def artifacts_to_dict(
    calibration: DRawCalibration,
    normalizer: FeatureNormalizer,
    feature_columns: Sequence[str],
    threshold: float,
    model_args: Dict[str, Any],
    d_floor: float = 0.0,
    rank_threshold: float = 0.0,
) -> Dict[str, Any]:
    return {
        "calibration": asdict(calibration),
        "normalizer": asdict(normalizer),
        "feature_columns": list(feature_columns),
        "threshold": float(threshold),
        "d_floor": float(d_floor),
        "rank_threshold": float(rank_threshold),
        "model_args": dict(model_args),
    }
