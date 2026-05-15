from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Q40FinalSelectorConfig:
    label_col: str = "d_true"
    group_col: str = "segment_id"
    time_col: str = "t"
    candidate_threshold: float = 0.25
    low_lag_loc_threshold: float = 0.490
    strong_raw_m_min: float = 3.0
    strong_loc_percentile_q: float = 40.0
    weak_candidate_min: float = 0.25
    weak_candidate_max: float = 0.30
    weak_raw_m_min: float = 1.5
    weak_raw_m_max: float = 2.4
    weak_expected_lag_min: float = 7.0
    weak_loc_min: float = 0.455
    weak_loc_max: float = 0.490
    weak_score_min: int = 3
    weak_segment_length_min: int = 8
    weak_segment_mean_raw_m_min: float = 1.6
    weak_segment_mean_raw_m_max: float = 2.6
    weak_segment_std_raw_m_max: float = 0.05
    weak_segment_mean_loc_min: float = 0.46
    weak_segment_max_loc_min: float = 0.464


def config_to_dict(cfg: Q40FinalSelectorConfig) -> Dict[str, Any]:
    return asdict(cfg)


def required_columns(cfg: Q40FinalSelectorConfig) -> List[str]:
    return [
        cfg.group_col,
        cfg.time_col,
        "dmax",
        "candidate_score",
        "localization_score",
        "raw_m",
        "calibrated_raw_m",
        "expected_lag",
    ]


def validate_frame(frame: pd.DataFrame, cfg: Q40FinalSelectorConfig, require_labels: bool = True) -> None:
    missing = [col for col in required_columns(cfg) if col not in frame.columns]
    if require_labels and cfg.label_col not in frame.columns:
        missing.append(cfg.label_col)
    if missing:
        raise ValueError(f"Q40 final selector input is missing columns: {', '.join(missing)}")


def _runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    runs: List[Tuple[int, int]] = []
    start = None
    for pos, value in enumerate(mask.astype(bool)):
        if value and start is None:
            start = pos
        elif not value and start is not None:
            runs.append((start, pos - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _source_names(mid_high: np.ndarray, low_lag: np.ndarray, plateau: np.ndarray) -> np.ndarray:
    sources = np.full(len(mid_high), "none", dtype=object)
    sources[mid_high] = "q40_strong"
    sources[low_lag] = "low_lag_high_conf"
    sources[plateau] = "weak_plateau"
    overlap = mid_high.astype(int) + low_lag.astype(int) + plateau.astype(int) > 1
    sources[overlap] = "multiple"
    return sources


def weak_plateau_mask(
    frame: pd.DataFrame,
    cfg: Q40FinalSelectorConfig,
    primary_selected: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame]:
    validate_frame(frame, cfg, require_labels=False)
    candidate = frame["candidate_score"].to_numpy(dtype=np.float64)
    cal = frame["calibrated_raw_m"].to_numpy(dtype=np.float64)
    expected = frame["expected_lag"].to_numpy(dtype=np.float64)
    loc = frame["localization_score"].to_numpy(dtype=np.float64)
    weak_score = (
        ((cal >= cfg.weak_raw_m_min) & (cal <= cfg.weak_raw_m_max)).astype(int)
        + (expected >= cfg.weak_expected_lag_min).astype(int)
        + ((loc >= cfg.weak_loc_min) & (loc < cfg.weak_loc_max)).astype(int)
    )
    weak_rows = (
        (~primary_selected.astype(bool))
        & (candidate >= cfg.weak_candidate_min)
        & (candidate <= cfg.weak_candidate_max)
        & (weak_score >= int(cfg.weak_score_min))
    )

    selected = np.zeros(len(frame), dtype=bool)
    plateaus: List[Dict[str, Any]] = []
    ordered = frame.sort_values([cfg.group_col, cfg.time_col])
    for group_value, idx in ordered.groupby(cfg.group_col, sort=False).groups.items():
        idx_list = list(idx)
        local = weak_rows[idx_list]
        for start, end in _runs(local):
            run_idx = np.asarray(idx_list[start : end + 1], dtype=int)
            if run_idx.size < int(cfg.weak_segment_length_min):
                continue
            run_cal = cal[run_idx]
            run_loc = loc[run_idx]
            mean_cal = float(np.nanmean(run_cal))
            std_cal = float(np.nanstd(run_cal))
            mean_loc = float(np.nanmean(run_loc))
            max_loc = float(np.nanmax(run_loc))
            if not (cfg.weak_segment_mean_raw_m_min <= mean_cal <= cfg.weak_segment_mean_raw_m_max):
                continue
            if std_cal > cfg.weak_segment_std_raw_m_max:
                continue
            if mean_loc < cfg.weak_segment_mean_loc_min or max_loc < cfg.weak_segment_max_loc_min:
                continue

            selected[run_idx] = True
            row: Dict[str, Any] = {
                cfg.group_col: group_value,
                "start_t": int(frame.loc[run_idx[0], cfg.time_col]),
                "end_t": int(frame.loc[run_idx[-1], cfg.time_col]),
                "length": int(run_idx.size),
                "mean_calibrated_raw_m": mean_cal,
                "std_calibrated_raw_m": std_cal,
                "mean_localization_score": mean_loc,
                "max_localization_score": max_loc,
            }
            if cfg.label_col in frame.columns:
                labels = frame.loc[run_idx, cfg.label_col].to_numpy(dtype=np.float64)
                row.update(
                    {
                        "n_d_true2": int((labels == 2.0).sum()),
                        "n_positive": int((labels > 0).sum()),
                        "n_zero": int((labels <= 0).sum()),
                    }
                )
            plateaus.append(row)

    return selected, pd.DataFrame(plateaus)


def apply_q40_final_selector(
    frame: pd.DataFrame,
    cfg: Q40FinalSelectorConfig | None = None,
) -> tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]:
    cfg = cfg or Q40FinalSelectorConfig()
    validate_frame(frame, cfg, require_labels=False)
    out = frame.copy()

    candidate = out["candidate_score"].to_numpy(dtype=np.float64) >= cfg.candidate_threshold
    loc = out["localization_score"].to_numpy(dtype=np.float64)
    cal = out["calibrated_raw_m"].to_numpy(dtype=np.float64)
    strong_candidate = candidate & (cal >= cfg.strong_raw_m_min)
    if not np.any(strong_candidate):
        raise ValueError("Q40 final selector found no strong_candidate rows")
    loc_threshold = float(np.nanpercentile(loc[strong_candidate], cfg.strong_loc_percentile_q))

    mid_high = strong_candidate & (loc >= loc_threshold)
    low_lag_high_conf = candidate & (cal < cfg.strong_raw_m_min) & (loc >= cfg.low_lag_loc_threshold)
    primary_selected = mid_high | low_lag_high_conf
    plateau, plateaus = weak_plateau_mask(out, cfg, primary_selected=primary_selected)
    selected = primary_selected | plateau
    d_hat = np.where(selected, cal, 0.0)
    p_pos = np.where(selected, loc, 0.0)

    out["q40_strong_candidate"] = strong_candidate.astype(int)
    out["q40_localization_threshold"] = loc_threshold
    out["q40_strong_selected"] = mid_high.astype(int)
    out["low_lag_high_conf_selected"] = low_lag_high_conf.astype(int)
    out["weak_plateau_selected"] = plateau.astype(int)
    out["q40_final_selected"] = selected.astype(int)
    out["q40_prediction_source"] = _source_names(mid_high, low_lag_high_conf, plateau)
    out["p_pos"] = p_pos
    out["d_hat"] = d_hat
    out["peak_score"] = p_pos * d_hat

    metadata = {
        "method": "q40_final_block_lag_selector",
        "strong_candidate": "candidate_score >= 0.25 and calibrated_raw_m >= 3.0",
        "strong_selector": "localization_score >= percentile(localization_score[strong_candidate], q=40)",
        "low_lag_high_conf": "candidate_score >= 0.25 and calibrated_raw_m < 3.0 and localization_score >= 0.490",
        "weak_plateau": "support-aware weak-lag plateau detector",
        "q40_localization_threshold": loc_threshold,
        "n_strong_candidate": int(strong_candidate.sum()),
        "n_q40_strong_selected": int(mid_high.sum()),
        "n_low_lag_high_conf_selected": int(low_lag_high_conf.sum()),
        "n_weak_plateau_selected": int(plateau.sum()),
        "n_final_selected": int(selected.sum()),
        "n_plateaus": int(len(plateaus)),
        "config": config_to_dict(cfg),
    }
    return out, metadata, plateaus


def positive_blocks(frame: pd.DataFrame, cfg: Q40FinalSelectorConfig) -> List[np.ndarray]:
    labels = frame[cfg.label_col].to_numpy(dtype=np.float64)
    blocks: List[np.ndarray] = []
    for idx in frame.groupby(cfg.group_col, sort=False).groups.values():
        idx_arr = np.asarray(list(idx), dtype=int)
        local = labels[idx_arr] > 0
        for start, end in _runs(local):
            blocks.append(idx_arr[start : end + 1])
    return blocks


def selection_metrics(frame: pd.DataFrame, cfg: Q40FinalSelectorConfig | None = None) -> Dict[str, Any]:
    cfg = cfg or Q40FinalSelectorConfig()
    validate_frame(frame, cfg, require_labels=True)
    labels = frame[cfg.label_col].to_numpy(dtype=np.float64)
    true = labels > 0
    pred = frame["q40_final_selected"].to_numpy(dtype=np.float64) > 0
    d_hat = frame["d_hat"].to_numpy(dtype=np.float64)
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
    }
    for value in [2.0, 4.0, 6.0]:
        group = labels == value
        metrics[f"d{int(value)}_recall"] = float(np.logical_and(pred, group).sum() / max(int(group.sum()), 1))
        metrics[f"d{int(value)}_selected"] = int(np.logical_and(pred, group).sum())

    blocks = positive_blocks(frame, cfg)
    if blocks:
        peak_errors = []
        peak_hits = []
        for block in blocks:
            true_peak = float(np.nanmax(labels[block]))
            pred_peak = float(np.nanmax(d_hat[block]))
            peak_errors.append(abs(pred_peak - true_peak))
            peak_hits.append(float(abs(int(np.floor(pred_peak + 0.5)) - int(true_peak)) <= 1))
        metrics.update(
            {
                "peak_error": float(np.mean(peak_errors)),
                "peak_hit_at_pm1": float(np.mean(peak_hits)),
                "n_positive_blocks": int(len(blocks)),
            }
        )
    else:
        metrics.update({"peak_error": float("nan"), "peak_hit_at_pm1": float("nan"), "n_positive_blocks": 0})
    return metrics
