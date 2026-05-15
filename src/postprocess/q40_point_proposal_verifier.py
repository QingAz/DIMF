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


DEFAULT_Q40_POINT_VERIFIER_FEATURE_COLUMNS = [
    "d_raw",
    "expected_lag",
    "p_nonzero",
    "entropy",
    "peak_prob",
    "margin",
    "candidate_score",
    "localization_score",
    "q40_d_hat",
    "q40_candidate_score",
    "q40_localization_score",
    "q40_rank_score",
    "q40_margin_to_threshold",
]

DEFAULT_Q40_POINT_VERIFIER_MINIMAL_FEATURE_COLUMNS = [
    "d_raw",
    "expected_lag",
    "p_nonzero",
    "entropy",
    "candidate_score",
    "localization_score",
    "q40_d_hat",
]


class Q40PointProposalVerifier(nn.Module):
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


def prepare_q40_proposal_frame(
    frame: pd.DataFrame,
    group_col: str = "segment_id",
) -> pd.DataFrame:
    out = frame.copy()
    out["q40_selected"] = (
        out["q40_final_selected"].to_numpy(dtype=np.float64)
        if "q40_final_selected" in out.columns
        else np.zeros(len(out), dtype=np.float64)
    )
    if "q40_d_hat" in out.columns:
        out["q40_d_hat"] = np.maximum(out["q40_d_hat"].to_numpy(dtype=np.float64), 0.0)
    elif "d_hat" in out.columns:
        out["q40_d_hat"] = np.maximum(out["d_hat"].to_numpy(dtype=np.float64), 0.0)
    else:
        out["q40_d_hat"] = np.zeros(len(out), dtype=np.float64)

    out["q40_candidate_score"] = (
        out["candidate_score"].to_numpy(dtype=np.float64)
        if "candidate_score" in out.columns
        else np.zeros(len(out), dtype=np.float64)
    )
    out["q40_localization_score"] = (
        out["localization_score"].to_numpy(dtype=np.float64)
        if "localization_score" in out.columns
        else np.zeros(len(out), dtype=np.float64)
    )
    if "candidate_score_model_rank" in out.columns:
        out["q40_rank_score"] = out["candidate_score_model_rank"].to_numpy(dtype=np.float64)
    else:
        out["q40_rank_score"] = _percentile_rank_by_group(
            out,
            out["q40_localization_score"].to_numpy(dtype=np.float64),
            group_col=group_col,
        )
    if "q40_localization_threshold" in out.columns:
        threshold = out["q40_localization_threshold"].to_numpy(dtype=np.float64)
        out["q40_margin_to_threshold"] = out["q40_localization_score"].to_numpy(dtype=np.float64) - threshold
    else:
        out["q40_margin_to_threshold"] = np.zeros(len(out), dtype=np.float64)

    out["proposal_label"] = (
        out["d_true"].to_numpy(dtype=np.float64) > 0
    ).astype(np.float64) if "d_true" in out.columns else np.zeros(len(out), dtype=np.float64)
    return out


def available_feature_columns(
    frame: pd.DataFrame,
    requested: Iterable[str] | None = None,
) -> List[str]:
    columns = list(requested) if requested is not None else list(DEFAULT_Q40_POINT_VERIFIER_FEATURE_COLUMNS)
    return [col for col in columns if col in frame.columns]


def proposal_only_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if "q40_selected" not in frame.columns:
        raise ValueError("Proposal verifier requires q40_selected column")
    return frame.loc[frame["q40_selected"].to_numpy(dtype=np.float64) > 0].reset_index(drop=True).copy()


def proposal_inventory_metrics(frame: pd.DataFrame) -> Dict[str, Any]:
    if "q40_selected" not in frame.columns:
        raise ValueError("Proposal inventory requires q40_selected column")
    if "d_true" not in frame.columns:
        raise ValueError("Proposal inventory requires d_true column")
    true = frame["d_true"].to_numpy(dtype=np.float64) > 0
    pred = frame["q40_selected"].to_numpy(dtype=np.float64) > 0
    positive_proposal = int(np.logical_and(pred, true).sum())
    false_positive_proposal = int(np.logical_and(pred, ~true).sum())
    all_true_positive = int(true.sum())
    missed_positive = int(np.logical_and(~pred, true).sum())
    return {
        "total_time_points": int(len(frame)),
        "q40_selected_count": int(pred.sum()),
        "positive_proposal_count": positive_proposal,
        "false_positive_proposal_count": false_positive_proposal,
        "missed_positive_count": missed_positive,
        "all_true_positive_points": all_true_positive,
        "q40_proposal_recall": float(positive_proposal / max(all_true_positive, 1)),
    }


def capped_auto_pos_weight(
    proposal_frame: pd.DataFrame,
    cap: float = 3.0,
    default: float = 1.0,
) -> float:
    metrics = proposal_inventory_metrics(proposal_frame)
    positive = int(metrics["positive_proposal_count"])
    false_positive = int(metrics["false_positive_proposal_count"])
    if positive <= 0:
        return float(default)
    ratio = float(false_positive) / float(positive)
    return float(min(max(ratio, 1.0), float(cap)))


def compact_proposal_output(
    frame: pd.DataFrame,
    label_col: str = "d_true",
    group_col: str = "segment_id",
) -> pd.DataFrame:
    cols: List[str] = []
    for col in [
        "split",
        "source_split",
        "timestamp",
        "raw_row_index",
        group_col,
        "t",
        "block_id",
        label_col,
        "proposal_label",
        "q40_selected",
        "q40_d_hat",
        "candidate_score",
        "localization_score",
        "d_raw",
        "expected_lag",
        "p_nonzero",
        "entropy",
        "peak_prob",
        "margin",
        "q40_candidate_score",
        "q40_localization_score",
        "q40_rank_score",
        "q40_margin_to_threshold",
        "verifier_confidence",
        "verifier_keep",
        "verifier_selected_final",
        "verifier_d_hat_final",
        "verifier_threshold",
    ]:
        if col in frame.columns and col not in cols:
            cols.append(col)
    return frame[cols].copy()


def _tensor_dataset(
    frame: pd.DataFrame,
    normalizer: FeatureNormalizer,
    feature_columns: Sequence[str],
) -> TensorDataset:
    features = normalizer.transform(frame[list(feature_columns)])
    labels = frame["proposal_label"].to_numpy(dtype=np.float32)
    return TensorDataset(
        torch.from_numpy(features),
        torch.from_numpy(labels),
    )


def _loss_on_loader(
    model: Q40PointProposalVerifier,
    loader: DataLoader,
    device: str,
    pos_weight: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_rows = 0
    scores: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    pos_weight_tensor = torch.tensor([float(pos_weight)], dtype=torch.float32, device=device)
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb, pos_weight=pos_weight_tensor)
            total_loss += float(loss.item()) * int(yb.numel())
            total_rows += int(yb.numel())
            scores.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(yb.cpu().numpy())
    if scores:
        score_arr = np.concatenate(scores).astype(np.float64)
        label_arr = np.concatenate(labels).astype(np.float64)
    else:
        score_arr = np.empty(0, dtype=np.float64)
        label_arr = np.empty(0, dtype=np.float64)
    mean_loss = total_loss / max(total_rows, 1)
    return mean_loss, score_arr, label_arr


def train_q40_point_verifier(
    fit_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    hidden_dim: int = 32,
    dropout: float = 0.10,
    epochs: int = 80,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    pos_weight: float = 2.0,
    seed: int = 42,
    device: str = "cpu",
) -> tuple[Q40PointProposalVerifier, FeatureNormalizer, pd.DataFrame]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    normalizer = fit_feature_normalizer(fit_frame, feature_columns=feature_columns)
    fit_ds = _tensor_dataset(fit_frame, normalizer=normalizer, feature_columns=feature_columns)
    val_ds = _tensor_dataset(val_frame, normalizer=normalizer, feature_columns=feature_columns)
    fit_loader = DataLoader(fit_ds, batch_size=int(batch_size), shuffle=True)
    eval_fit_loader = DataLoader(fit_ds, batch_size=int(batch_size), shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=int(batch_size), shuffle=False)

    model = Q40PointProposalVerifier(
        input_dim=len(list(feature_columns)),
        hidden_dim=int(hidden_dim),
        dropout=float(dropout),
    ).to(device)
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

        fit_loss, fit_score, fit_label = _loss_on_loader(model, eval_fit_loader, device=device, pos_weight=float(pos_weight))
        val_loss, val_score, val_label = _loss_on_loader(model, val_loader, device=device, pos_weight=float(pos_weight))
        fit_auprc = (
            float(average_precision_score(fit_label, fit_score))
            if fit_label.size and np.unique(fit_label).size > 1
            else float("nan")
        )
        val_auprc = (
            float(average_precision_score(val_label, val_score))
            if val_label.size and np.unique(val_label).size > 1
            else float("nan")
        )
        rows.append(
            {
                "epoch": int(epoch),
                "fit_loss": fit_loss,
                "val_loss": val_loss,
                "fit_auprc": fit_auprc,
                "val_auprc": val_auprc,
                "fit_positive_rate": float(fit_label.mean()) if fit_label.size else float("nan"),
                "val_positive_rate": float(val_label.mean()) if val_label.size else float("nan"),
            }
        )
    return model, normalizer, pd.DataFrame(rows)


def predict_q40_point_verifier(
    model: Q40PointProposalVerifier,
    frame: pd.DataFrame,
    normalizer: FeatureNormalizer,
    feature_columns: Sequence[str],
    threshold: float = 0.5,
    device: str = "cpu",
) -> pd.DataFrame:
    out = frame.copy()
    proposal_mask = out["q40_selected"].to_numpy(dtype=np.float64) > 0
    confidence = np.zeros(len(out), dtype=np.float64)
    if proposal_mask.any():
        proposal_frame = out.loc[proposal_mask].reset_index(drop=True)
        features = normalizer.transform(proposal_frame[list(feature_columns)])
        xb = torch.from_numpy(features).to(device)
        model.eval()
        with torch.no_grad():
            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy().astype(np.float64)
        confidence[np.flatnonzero(proposal_mask)] = probs
    keep = proposal_mask & (confidence >= float(threshold))
    q40_d_hat = out["q40_d_hat"].to_numpy(dtype=np.float64) if "q40_d_hat" in out.columns else np.zeros(len(out), dtype=np.float64)
    out["verifier_confidence"] = confidence
    out["verifier_keep"] = keep.astype(int)
    out["verifier_selected_final"] = keep.astype(int)
    out["verifier_d_hat_final"] = np.where(keep, q40_d_hat, 0.0)
    out["verifier_threshold"] = float(threshold)
    return out


def proposal_classification_metrics(frame: pd.DataFrame) -> Dict[str, Any]:
    proposal = proposal_only_frame(frame)
    if proposal.empty:
        return {
            "n_proposals": 0,
            "n_positive_proposals": 0,
            "n_negative_proposals": 0,
            "proposal_precision": float("nan"),
            "proposal_recall": float("nan"),
            "proposal_F1": float("nan"),
            "proposal_FPR": float("nan"),
            "proposal_AUPRC": float("nan"),
        }
    label = proposal["proposal_label"].to_numpy(dtype=np.float64) > 0
    pred = proposal["verifier_keep"].to_numpy(dtype=np.float64) > 0
    score = proposal["verifier_confidence"].to_numpy(dtype=np.float64)
    tp = int(np.logical_and(pred, label).sum())
    fp = int(np.logical_and(pred, ~label).sum())
    fn = int(np.logical_and(~pred, label).sum())
    tn = int(np.logical_and(~pred, ~label).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    auprc = (
        float(average_precision_score(label.astype(np.int64), score))
        if np.unique(label.astype(np.int64)).size > 1
        else float("nan")
    )
    return {
        "n_proposals": int(len(proposal)),
        "n_positive_proposals": int(label.sum()),
        "n_negative_proposals": int((~label).sum()),
        "proposal_precision": float(precision),
        "proposal_recall": float(recall),
        "proposal_F1": float(f1),
        "proposal_FPR": float(fp / max(fp + tn, 1)),
        "proposal_AUPRC": auprc,
    }


def proposal_score_distribution_table(frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for split_name, frame in frames.items():
        proposal = proposal_only_frame(frame)
        if proposal.empty or "verifier_confidence" not in proposal.columns:
            continue
        score = proposal["verifier_confidence"].to_numpy(dtype=np.float64)
        is_positive = proposal["proposal_label"].to_numpy(dtype=np.float64) > 0
        groups = {
            "positive_proposal": is_positive,
            "false_positive_proposal": ~is_positive,
        }
        for group_name, mask in groups.items():
            group_scores = score[mask]
            if group_scores.size == 0:
                rows.append(
                    {
                        "split": split_name,
                        "group": group_name,
                        "count": 0,
                        "v_p10": float("nan"),
                        "v_p50": float("nan"),
                        "v_p90": float("nan"),
                    }
                )
                continue
            rows.append(
                {
                    "split": split_name,
                    "group": group_name,
                    "count": int(group_scores.size),
                    "v_p10": float(np.nanpercentile(group_scores, 10.0)),
                    "v_p50": float(np.nanpercentile(group_scores, 50.0)),
                    "v_p90": float(np.nanpercentile(group_scores, 90.0)),
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


def threshold_grid(
    frame: pd.DataFrame,
    model: Q40PointProposalVerifier,
    normalizer: FeatureNormalizer,
    feature_columns: Sequence[str],
    thresholds: Sequence[float],
    label_col: str = "d_true",
    group_col: str = "segment_id",
    time_col: str = "t",
    device: str = "cpu",
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for threshold in thresholds:
        scored = predict_q40_point_verifier(
            model,
            frame,
            normalizer=normalizer,
            feature_columns=feature_columns,
            threshold=float(threshold),
            device=device,
        )
        proposal_metrics = proposal_classification_metrics(scored)
        lag_metrics = verifier_end_to_end_metrics(
            scored,
            label_col=label_col,
            group_col=group_col,
            time_col=time_col,
        )
        rows.append(
            {
                "threshold": float(threshold),
                **proposal_metrics,
                **{f"lag_{key}": value for key, value in lag_metrics.items()},
            }
        )
    return pd.DataFrame(rows)
