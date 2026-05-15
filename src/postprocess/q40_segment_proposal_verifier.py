from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, TensorDataset

from src.postprocess.q40_final_block_lag_selector import (
    Q40FinalSelectorConfig,
    selection_metrics as q40_selection_metrics,
)
from src.postprocess.q40_common import (
    FeatureNormalizer,
    fit_feature_normalizer,
)


DEFAULT_SEGMENT_BASE_COLUMNS = [
    "d_raw",
    "expected_lag",
    "p_nonzero",
    "entropy",
    "peak_prob",
    "margin",
    "candidate_score",
    "localization_score",
    "q40_d_hat",
]


class Q40SegmentProposalVerifier(nn.Module):
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


def _with_point_ranks(frame: pd.DataFrame, group_col: str) -> pd.DataFrame:
    out = frame.copy()
    for src, dst in [
        ("d_raw", "rank_d_raw"),
        ("candidate_score", "rank_candidate_score"),
        ("localization_score", "rank_localization_score"),
    ]:
        values = out[src].to_numpy(dtype=np.float64) if src in out.columns else np.zeros(len(out), dtype=np.float64)
        out[dst] = _percentile_rank_by_group(out, values, group_col=group_col)
    return out


def _segment_rows(group: pd.DataFrame, time_col: str, merge_gap: int, min_len: int) -> List[pd.DataFrame]:
    selected = group["q40_selected"].to_numpy(dtype=np.float64) > 0
    if not selected.any():
        return []
    sel_pos = np.flatnonzero(selected)
    times = group[time_col].to_numpy(dtype=np.float64)
    rows: List[pd.DataFrame] = []
    start_pos = int(sel_pos[0])
    last_pos = int(sel_pos[0])
    last_time = float(times[last_pos])
    for pos in sel_pos[1:]:
        pos = int(pos)
        cur_time = float(times[pos])
        if cur_time - last_time <= float(merge_gap + 1):
            last_pos = pos
            last_time = cur_time
            continue
        segment = group.iloc[start_pos : last_pos + 1].copy()
        if len(segment) >= int(min_len):
            rows.append(segment)
        start_pos = pos
        last_pos = pos
        last_time = cur_time
    segment = group.iloc[start_pos : last_pos + 1].copy()
    if len(segment) >= int(min_len):
        rows.append(segment)
    return rows


def _aggregate_segment(
    segment: pd.DataFrame,
    group_value: Any,
    segment_index: int,
    group_col: str,
    time_col: str,
    include_q40_segment_features: bool = False,
) -> Dict[str, Any]:
    def _finite(values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float64)
        return arr[np.isfinite(arr)]

    row: Dict[str, Any] = {
        group_col: group_value,
        "q40_segment_index": int(segment_index),
        "segment_uid": f"{group_value}::{segment_index}",
        "start_t": float(segment[time_col].iloc[0]),
        "end_t": float(segment[time_col].iloc[-1]),
        "length": int(len(segment)),
        "segment_label": int((segment["d_true"].to_numpy(dtype=np.float64) > 0).any()) if "d_true" in segment.columns else 0,
        "q40_selected_count": int((segment["q40_selected"].to_numpy(dtype=np.float64) > 0).sum()),
    }
    d_raw = segment["d_raw"].to_numpy(dtype=np.float64) if "d_raw" in segment.columns else np.zeros(len(segment), dtype=np.float64)
    slope = np.diff(d_raw) if len(d_raw) > 1 else np.empty(0, dtype=np.float64)
    curvature = np.diff(d_raw, n=2) if len(d_raw) > 2 else np.empty(0, dtype=np.float64)
    row["area"] = float(np.nansum(d_raw))
    row["mean_abs_slope"] = float(np.nanmean(np.abs(slope))) if slope.size else 0.0
    row["max_slope"] = float(np.nanmax(np.abs(slope))) if slope.size else 0.0
    row["curvature_mean"] = float(np.nanmean(np.abs(curvature))) if curvature.size else 0.0
    row["plateau_ratio"] = float(np.mean(np.abs(slope) <= 0.1)) if slope.size else 1.0
    for col in DEFAULT_SEGMENT_BASE_COLUMNS:
        values = segment[col].to_numpy(dtype=np.float64) if col in segment.columns else np.zeros(len(segment), dtype=np.float64)
        finite = _finite(values)
        row[f"{col}_mean"] = float(finite.mean()) if finite.size else 0.0
        row[f"{col}_max"] = float(finite.max()) if finite.size else 0.0
        row[f"{col}_std"] = float(finite.std()) if finite.size else 0.0
    for col in ["rank_d_raw", "rank_candidate_score", "rank_localization_score"]:
        values = segment[col].to_numpy(dtype=np.float64) if col in segment.columns else np.zeros(len(segment), dtype=np.float64)
        finite = _finite(values)
        row[f"segment_mean_{col}"] = float(finite.mean()) if finite.size else 0.0
    if bool(include_q40_segment_features):
        proposal = segment.loc[segment["q40_selected"].to_numpy(dtype=np.float64) > 0].copy()
        q40_len = int(len(proposal))
        row["q40_segment_length"] = q40_len
        for src, dst, reducer in [
            ("q40_d_hat", "q40_mean_d_hat", "mean"),
            ("q40_d_hat", "q40_max_d_hat", "max"),
            ("q40_candidate_score", "q40_mean_candidate_score", "mean"),
            ("q40_localization_score", "q40_mean_localization_score", "mean"),
            ("q40_rank_score", "q40_mean_rank_score", "mean"),
        ]:
            values = proposal[src].to_numpy(dtype=np.float64) if src in proposal.columns else np.zeros(q40_len, dtype=np.float64)
            finite = _finite(values)
            if reducer == "max":
                row[dst] = float(finite.max()) if finite.size else 0.0
            else:
                row[dst] = float(finite.mean()) if finite.size else 0.0
        margin_values = proposal["q40_margin_to_threshold"].to_numpy(dtype=np.float64) if "q40_margin_to_threshold" in proposal.columns else np.zeros(q40_len, dtype=np.float64)
        finite_margin = _finite(margin_values)
        row["q40_min_margin_to_threshold"] = float(finite_margin.min()) if finite_margin.size else 0.0
        strong_mask = proposal["q40_strong_candidate"].to_numpy(dtype=np.float64) > 0 if "q40_strong_candidate" in proposal.columns else np.zeros(q40_len, dtype=bool)
        weak_mask = np.zeros(q40_len, dtype=bool)
        if "low_lag_high_conf_selected" in proposal.columns:
            weak_mask |= proposal["low_lag_high_conf_selected"].to_numpy(dtype=np.float64) > 0
        if "weak_plateau_selected" in proposal.columns:
            weak_mask |= proposal["weak_plateau_selected"].to_numpy(dtype=np.float64) > 0
        row["q40_has_strong_candidate"] = int(strong_mask.any()) if q40_len else 0
        row["q40_has_weak_candidate"] = int(weak_mask.any()) if q40_len else 0
        row["q40_strong_ratio"] = float(strong_mask.mean()) if q40_len else 0.0
        row["q40_weak_ratio"] = float(weak_mask.mean()) if q40_len else 0.0
    return row


def build_segment_dataset(
    frame: pd.DataFrame,
    group_col: str = "segment_id",
    time_col: str = "t",
    merge_gap: int = 1,
    min_len: int = 1,
    include_q40_segment_features: bool = False,
) -> pd.DataFrame:
    if "q40_selected" not in frame.columns:
        raise ValueError("Segment verifier requires q40_selected column")
    ordered = frame.sort_values([group_col, time_col]).reset_index(drop=True).copy()
    ordered = _with_point_ranks(ordered, group_col=group_col)
    rows: List[Dict[str, Any]] = []
    for group_value, group in ordered.groupby(group_col, sort=False):
        for seg_idx, segment in enumerate(_segment_rows(group, time_col=time_col, merge_gap=merge_gap, min_len=min_len), start=1):
            rows.append(
                _aggregate_segment(
                    segment,
                    group_value=group_value,
                    segment_index=seg_idx,
                    group_col=group_col,
                    time_col=time_col,
                    include_q40_segment_features=bool(include_q40_segment_features),
                )
            )
    return pd.DataFrame(rows)


def segment_feature_columns(frame: pd.DataFrame) -> List[str]:
    excluded = {
        "segment_uid",
        "segment_label",
        "q40_segment_index",
        "start_t",
        "end_t",
        "q40_selected_count",
        "segment_id",
    }
    return [col for col in frame.columns if col not in excluded and pd.api.types.is_numeric_dtype(frame[col])]


def _tensor_dataset(
    frame: pd.DataFrame,
    normalizer: FeatureNormalizer,
    feature_columns: Sequence[str],
) -> TensorDataset:
    features = normalizer.transform(frame[list(feature_columns)])
    labels = frame["segment_label"].to_numpy(dtype=np.float32)
    return TensorDataset(torch.from_numpy(features), torch.from_numpy(labels))


def train_q40_segment_verifier(
    fit_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    hidden_dim: int = 32,
    dropout: float = 0.10,
    epochs: int = 80,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    pos_weight: float = 1.0,
    seed: int = 42,
    device: str = "cpu",
) -> tuple[Q40SegmentProposalVerifier, FeatureNormalizer, pd.DataFrame]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    normalizer = fit_feature_normalizer(fit_frame, feature_columns=feature_columns)
    fit_ds = _tensor_dataset(fit_frame, normalizer=normalizer, feature_columns=feature_columns)
    val_ds = _tensor_dataset(val_frame, normalizer=normalizer, feature_columns=feature_columns)
    fit_loader = DataLoader(fit_ds, batch_size=int(batch_size), shuffle=True)
    fit_eval_loader = DataLoader(fit_ds, batch_size=int(batch_size), shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=int(batch_size), shuffle=False)
    model = Q40SegmentProposalVerifier(len(list(feature_columns)), hidden_dim=int(hidden_dim), dropout=float(dropout)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    pos_weight_tensor = torch.tensor([float(pos_weight)], dtype=torch.float32, device=device)
    rows: List[Dict[str, Any]] = []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        for xb, yb in fit_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb, pos_weight=pos_weight_tensor)
            loss.backward()
            optimizer.step()
        rows.append(
            {
                "epoch": int(epoch),
                "fit_loss": _mean_loss(model, fit_eval_loader, device=device, pos_weight=float(pos_weight)),
                "val_loss": _mean_loss(model, val_loader, device=device, pos_weight=float(pos_weight)),
            }
        )
    return model, normalizer, pd.DataFrame(rows)


def _mean_loss(model: Q40SegmentProposalVerifier, loader: DataLoader, device: str, pos_weight: float) -> float:
    model.eval()
    total_loss = 0.0
    total_rows = 0
    pos_weight_tensor = torch.tensor([float(pos_weight)], dtype=torch.float32, device=device)
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb, pos_weight=pos_weight_tensor)
            total_loss += float(loss.item()) * int(yb.numel())
            total_rows += int(yb.numel())
    return total_loss / max(total_rows, 1)


def predict_q40_segment_verifier(
    model: Q40SegmentProposalVerifier,
    frame: pd.DataFrame,
    normalizer: FeatureNormalizer,
    feature_columns: Sequence[str],
    threshold: float = 0.5,
    device: str = "cpu",
) -> pd.DataFrame:
    out = frame.copy()
    if out.empty:
        out["verifier_segment_confidence"] = np.empty(0, dtype=np.float64)
        out["verifier_segment_keep"] = np.empty(0, dtype=np.int64)
        out["verifier_segment_threshold"] = np.empty(0, dtype=np.float64)
        return out
    xb = torch.from_numpy(normalizer.transform(out[list(feature_columns)])).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(xb)
        probs = torch.sigmoid(logits).cpu().numpy().astype(np.float64)
    out["verifier_segment_confidence"] = probs
    out["verifier_segment_keep"] = (probs >= float(threshold)).astype(int)
    out["verifier_segment_threshold"] = float(threshold)
    return out


def apply_segment_decisions_to_timeseries(
    frame: pd.DataFrame,
    segment_frame: pd.DataFrame,
    group_col: str = "segment_id",
    time_col: str = "t",
) -> pd.DataFrame:
    out = frame.copy()
    out["verifier_segment_confidence"] = 0.0
    out["verifier_segment_keep"] = 0
    out["verifier_segment_index"] = -1
    q40_selected = out["q40_selected"].to_numpy(dtype=np.float64) > 0
    q40_d_hat = out["q40_d_hat"].to_numpy(dtype=np.float64) if "q40_d_hat" in out.columns else np.zeros(len(out), dtype=np.float64)
    keep_mask = np.zeros(len(out), dtype=bool)
    for row in segment_frame.itertuples(index=False):
        mask = (
            (out[group_col] == getattr(row, group_col))
            & (out[time_col] >= float(row.start_t))
            & (out[time_col] <= float(row.end_t))
        )
        out.loc[mask, "verifier_segment_confidence"] = float(row.verifier_segment_confidence)
        out.loc[mask, "verifier_segment_keep"] = int(row.verifier_segment_keep)
        out.loc[mask, "verifier_segment_index"] = int(row.q40_segment_index)
        if int(row.verifier_segment_keep) > 0:
            keep_mask |= mask.to_numpy(dtype=bool)
    selected_final = q40_selected & keep_mask
    out["verifier_selected_final"] = selected_final.astype(int)
    out["verifier_d_hat_final"] = np.where(selected_final, q40_d_hat, 0.0)
    return out


def segment_score_distribution_table(frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for split_name, frame in frames.items():
        if frame.empty or "verifier_segment_confidence" not in frame.columns:
            continue
        for group_name, mask in [
            ("positive_segment", frame["segment_label"].to_numpy(dtype=np.float64) > 0),
            ("false_positive_segment", frame["segment_label"].to_numpy(dtype=np.float64) <= 0),
        ]:
            values = frame.loc[mask, "verifier_segment_confidence"].to_numpy(dtype=np.float64)
            rows.append(
                {
                    "split": split_name,
                    "group": group_name,
                    "count": int(values.size),
                    "v_p10": float(np.nanpercentile(values, 10.0)) if values.size else float("nan"),
                    "v_p50": float(np.nanpercentile(values, 50.0)) if values.size else float("nan"),
                    "v_p90": float(np.nanpercentile(values, 90.0)) if values.size else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def verifier_end_to_end_metrics(
    frame: pd.DataFrame,
    label_col: str = "d_true",
    group_col: str = "segment_id",
    time_col: str = "t",
) -> Dict[str, Any]:
    proxy = frame.copy()
    proxy["q40_final_selected"] = proxy["verifier_selected_final"].to_numpy(dtype=np.float64)
    proxy["d_hat"] = proxy["verifier_d_hat_final"].to_numpy(dtype=np.float64)
    cfg = Q40FinalSelectorConfig(label_col=label_col, group_col=group_col, time_col=time_col)
    return q40_selection_metrics(proxy, cfg)


def segment_classification_metrics(frame: pd.DataFrame) -> Dict[str, Any]:
    if frame.empty:
        return {
            "n_segments": 0,
            "n_positive_segments": 0,
            "segment_precision": float("nan"),
            "segment_recall": float("nan"),
            "segment_F1": float("nan"),
            "segment_AUPRC": float("nan"),
        }
    label = frame["segment_label"].to_numpy(dtype=np.float64) > 0
    pred = frame["verifier_segment_keep"].to_numpy(dtype=np.float64) > 0
    score = frame["verifier_segment_confidence"].to_numpy(dtype=np.float64)
    tp = int(np.logical_and(pred, label).sum())
    fp = int(np.logical_and(pred, ~label).sum())
    fn = int(np.logical_and(~pred, label).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    auprc = float(average_precision_score(label.astype(int), score)) if np.unique(label.astype(int)).size > 1 else float("nan")
    return {
        "n_segments": int(len(frame)),
        "n_positive_segments": int(label.sum()),
        "segment_precision": float(precision),
        "segment_recall": float(recall),
        "segment_F1": float(2.0 * precision * recall / max(precision + recall, 1e-12)),
        "segment_AUPRC": auprc,
    }
