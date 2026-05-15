from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LagLossWeights:
    lambda_soft_lag: float = 1.0
    lambda_expected_lag: float = 0.1
    lambda_occurrence: float = 0.5
    occurrence_pos_weight: Optional[float] = None
    lambda_entropy: float = 0.01
    lambda_smooth: float = 0.005
    lambda_positive_smooth: float = 0.0
    lambda_positive_ce: float = 0.0
    positive_ce_class_weights: Optional[list[float]] = None
    use_gaussian_lag_label: bool = True
    gaussian_lag_sigma: float = 0.7
    enable_segment_aware_temporal_loss: bool = False
    lambda_shape_curvature: float = 0.0
    shape_curvature_ids: Optional[list[int]] = None


class CandidateLagWindowEncoder(nn.Module):
    """
    Encodes a causal source patch for each candidate lag.

    For lag l at the current sample time t, the patch is
    [t-l-radius, ..., t-l]. Invalid windows are returned through the valid mask
    instead of being clamped into the softmax support.
    """

    def __init__(
        self,
        max_lag: int,
        radius: int = 2,
        hidden_dim: int = 64,
        mlp_hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        mode: str = "causal",
    ):
        super().__init__()
        if radius < 0:
            raise ValueError("lag window radius must be non-negative")
        if mode != "causal":
            raise ValueError("Only causal lag window encoding is supported")
        self.max_lag = int(max_lag)
        self.radius = int(radius)
        self.patch_len = self.radius + 1
        self.mode = str(mode)
        inner_dim = int(mlp_hidden_dim or hidden_dim)
        self.patch_encoder = nn.Sequential(
            nn.Linear(self.patch_len, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, hidden_dim),
        )

    def forward(self, source_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if source_seq.ndim != 3:
            raise ValueError("source_seq must be shaped [B, L, d_source]")
        bsz, length, d_source = source_seq.shape
        lag_idx = torch.arange(self.max_lag + 1, device=source_seq.device)
        end = (length - 1) - lag_idx
        start = end - self.radius
        valid = start >= 0
        if not torch.any(valid):
            raise ValueError(
                f"No valid lag windows for sequence length {length}, radius {self.radius}, max_lag {self.max_lag}"
            )

        offsets = torch.arange(self.radius, -1, -1, device=source_seq.device)
        positions = (end[:, None] - offsets[None, :]).clamp(min=0, max=max(length - 1, 0))
        flat = source_seq[:, positions.reshape(-1), :]
        patches = flat.reshape(bsz, self.max_lag + 1, self.patch_len, d_source).permute(0, 3, 1, 2)
        patch_repr = self.patch_encoder(patches)
        return patch_repr, valid


class STDALagIdentifier(nn.Module):
    """
    Lightweight STDA-style prior generator for DIMF delay alignment.

    The module predicts a lag distribution for every source feature. Inside the
    integrated DIMF path, those distributions are consumed as soft delay priors
    by DelayAlignment. The edge-level distribution used by lag diagnostics is the
    feature-importance weighted average of the feature-level distributions.
    """

    def __init__(
        self,
        d_source: int,
        d_target: int,
        max_lag: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        temperature: float = 0.7,
        use_temporal_decay: bool = True,
        use_feature_attention: bool = True,
        dropout: float = 0.0,
        use_sequence_smoother: bool = False,
        sequence_smoother_hidden_dim: Optional[int] = None,
        sequence_smoother_layers: int = 1,
        sequence_smoother_dropout: float = 0.0,
        sequence_smoother_residual_scale: float = 0.5,
        use_candidate_window_encoder: bool = False,
        lag_window_radius: int = 2,
        lag_window_hidden_dim: Optional[int] = None,
        lag_window_mode: str = "causal",
        keep_old_point_identifier: bool = True,
    ):
        super().__init__()
        if max_lag < 0:
            raise ValueError("max_lag must be non-negative")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.d_source = int(d_source)
        self.d_target = int(d_target)
        self.max_lag = int(max_lag)
        self.temperature = float(temperature)
        self.use_temporal_decay = bool(use_temporal_decay)
        self.use_feature_attention = bool(use_feature_attention)
        self.use_sequence_smoother = bool(use_sequence_smoother)
        self.sequence_smoother_residual_scale = float(sequence_smoother_residual_scale)
        self.use_candidate_window_encoder = bool(use_candidate_window_encoder)
        self.keep_old_point_identifier = bool(keep_old_point_identifier)

        self.source_value_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.source_context_encoder = nn.GRU(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=max(1, int(num_layers)),
            dropout=dropout if int(num_layers) > 1 else 0.0,
            batch_first=True,
        )
        self.target_encoder = nn.GRU(
            input_size=d_target,
            hidden_size=hidden_dim,
            num_layers=max(1, int(num_layers)),
            dropout=dropout if int(num_layers) > 1 else 0.0,
            batch_first=True,
        )
        self.target_repr_proj = nn.Linear(hidden_dim, hidden_dim)
        self.score_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.candidate_window_encoder = None
        if self.use_candidate_window_encoder:
            self.candidate_window_encoder = CandidateLagWindowEncoder(
                max_lag=self.max_lag,
                radius=int(lag_window_radius),
                hidden_dim=hidden_dim,
                mlp_hidden_dim=int(lag_window_hidden_dim or hidden_dim),
                dropout=dropout,
                mode=str(lag_window_mode),
            )
        self.feature_importance_head = nn.Linear(hidden_dim, 1)
        self.occurrence_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.lag_bias = nn.Parameter(torch.zeros(max_lag + 1))
        if self.use_temporal_decay:
            self.gamma_raw = nn.Parameter(torch.zeros(d_source))
        else:
            self.register_parameter("gamma_raw", None)

        smoother_in_dim = self.max_lag + 2
        smoother_hidden = int(sequence_smoother_hidden_dim or max(16, hidden_dim // 2))
        smoother_layers = max(1, int(sequence_smoother_layers))
        if self.use_sequence_smoother:
            self.sequence_smoother = nn.GRU(
                input_size=smoother_in_dim,
                hidden_size=smoother_hidden,
                num_layers=smoother_layers,
                dropout=float(sequence_smoother_dropout) if smoother_layers > 1 else 0.0,
                batch_first=True,
                bidirectional=True,
            )
            self.sequence_smoother_proj = nn.Linear(smoother_hidden * 2, smoother_in_dim)
        else:
            self.sequence_smoother = None
            self.sequence_smoother_proj = None

    def _source_feature_context(self, source_seq: torch.Tensor) -> torch.Tensor:
        bsz, length, d_source = source_seq.shape
        flat = source_seq.transpose(1, 2).reshape(bsz * d_source, length, 1)
        _, h = self.source_context_encoder(flat)
        ctx = h[-1].reshape(bsz, d_source, -1)
        return ctx

    def _lag_candidates(self, source_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, length, d_source = source_seq.shape
        lag_idx = torch.arange(self.max_lag + 1, device=source_seq.device)
        raw_pos = (length - 1) - lag_idx
        valid = raw_pos >= 0
        gather_pos = raw_pos.clamp(min=0)
        # source_seq[:, gather_pos, :] -> [B, K, d_source], then [B, d_source, K, 1]
        values = source_seq[:, gather_pos, :].permute(0, 2, 1).unsqueeze(-1)
        return values, valid

    def _apply_sequence_smoother(
        self,
        scores: torch.Tensor,
        occurrence_logit: torch.Tensor,
        valid: torch.Tensor,
        segment_id: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            not self.use_sequence_smoother
            or self.sequence_smoother is None
            or self.sequence_smoother_proj is None
            or scores.shape[0] <= 1
        ):
            return scores, occurrence_logit

        valid_mask = valid[None, None, :]
        safe_scores = scores.masked_fill(~valid_mask, -20.0)
        smoother_in = torch.cat([safe_scores, occurrence_logit.unsqueeze(-1)], dim=-1)

        def smooth_chunk(chunk: torch.Tensor) -> torch.Tensor:
            if chunk.shape[0] <= 1:
                return chunk
            seq = chunk.transpose(0, 1).contiguous()
            smoothed_seq, _ = self.sequence_smoother(seq)
            delta = self.sequence_smoother_proj(smoothed_seq)
            smoothed_seq = seq + self.sequence_smoother_residual_scale * delta
            return smoothed_seq.transpose(0, 1).contiguous()

        if segment_id is not None:
            segment_id = segment_id.to(device=scores.device).reshape(-1)
            if segment_id.shape[0] != scores.shape[0]:
                raise ValueError("segment_id length must match batch size")
            chunks = []
            start = 0
            for end in range(1, scores.shape[0] + 1):
                if end == scores.shape[0] or segment_id[end] != segment_id[start]:
                    chunks.append(smooth_chunk(smoother_in[start:end]))
                    start = end
            smoothed = torch.cat(chunks, dim=0)
        else:
            smoothed = smooth_chunk(smoother_in)

        score_delta = smoothed[..., : scores.shape[-1]]
        occurrence_delta = smoothed[..., scores.shape[-1]]
        score_delta = score_delta.masked_fill(~valid_mask, float("-inf"))
        return score_delta, occurrence_delta

    def forward(
        self,
        source_seq: torch.Tensor,
        target_seq: Optional[torch.Tensor] = None,
        target_repr: Optional[torch.Tensor] = None,
        source_mask: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
        pi_prior: Optional[torch.Tensor] = None,
        segment_id: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        del source_mask, target_mask
        if source_seq.ndim != 3:
            raise ValueError("source_seq must be shaped [B, L, d_source]")
        if source_seq.shape[-1] != self.d_source:
            raise ValueError(f"source_seq has d_source={source_seq.shape[-1]}, expected {self.d_source}")
        if target_repr is None:
            if target_seq is None:
                raise ValueError("Either target_seq or target_repr is required")
            if target_seq.ndim != 3:
                raise ValueError("target_seq must be shaped [B, L, d_target]")
            _, h = self.target_encoder(target_seq)
            target_repr = h[-1]
        target_context = self.target_repr_proj(target_repr)

        source_ctx = self._source_feature_context(source_seq)
        if self.use_candidate_window_encoder:
            if self.candidate_window_encoder is None:
                raise RuntimeError("candidate window encoder is not initialized")
            lag_repr, valid = self.candidate_window_encoder(source_seq)
        else:
            lag_values, valid = self._lag_candidates(source_seq)
            lag_repr = self.source_value_proj(lag_values)
        lag_repr = self.score_proj(lag_repr)

        scores = torch.einsum("bdkh,bh->bdk", lag_repr, target_context) / (lag_repr.shape[-1] ** 0.5)
        scores = scores + self.lag_bias[None, None, :]
        if self.use_temporal_decay and self.gamma_raw is not None:
            gamma = F.softplus(self.gamma_raw)
            lag_axis = torch.arange(self.max_lag + 1, device=source_seq.device, dtype=scores.dtype)
            scores = scores - gamma[None, :, None] * lag_axis[None, None, :]
        scores = scores.masked_fill(~valid[None, None, :], float("-inf"))

        if pi_prior is not None:
            prior = pi_prior.to(device=scores.device, dtype=scores.dtype)
            if prior.ndim == 2:
                prior = prior[:, None, :]
            if prior.ndim != 3:
                raise ValueError("pi_prior must be shaped [B, K] or [B, d_source, K]")
            if prior.shape[-1] != scores.shape[-1]:
                raise ValueError("pi_prior lag dimension does not match max_lag + 1")
            if prior.shape[1] == 1:
                prior = prior.expand(-1, scores.shape[1], -1)
            if prior.shape[1] != scores.shape[1]:
                raise ValueError("pi_prior feature dimension does not match d_source")
            scores = scores + torch.log(prior.clamp(min=1e-8))

        target_for_occ = target_context[:, None, :].expand(-1, self.d_source, -1)
        lag_occurrence_logit = self.occurrence_head(torch.cat([source_ctx, target_for_occ], dim=-1)).squeeze(-1)
        scores, lag_occurrence_logit = self._apply_sequence_smoother(
            scores,
            lag_occurrence_logit,
            valid,
            segment_id=segment_id,
        )
        occ_prob = torch.sigmoid(lag_occurrence_logit)

        raw_pi_lag = F.softmax(scores / self.temperature, dim=-1)
        if raw_pi_lag.shape[-1] > 1:
            if torch.any(valid[1:]):
                pos_pi = raw_pi_lag[..., 1:]
                pos_pi = pos_pi / pos_pi.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                pi_lag = torch.cat(
                    [
                        (1.0 - occ_prob).unsqueeze(-1),
                        occ_prob.unsqueeze(-1) * pos_pi,
                    ],
                    dim=-1,
                )
            else:
                pi_lag = torch.zeros_like(raw_pi_lag)
                pi_lag[..., 0] = 1.0
        else:
            pi_lag = raw_pi_lag

        lag_axis = torch.arange(self.max_lag + 1, device=source_seq.device, dtype=pi_lag.dtype)
        expected_lag = (pi_lag * lag_axis[None, None, :]).sum(dim=-1)
        argmax_lag = pi_lag.argmax(dim=-1)

        if self.use_feature_attention:
            feature_importance = F.softmax(self.feature_importance_head(source_ctx).squeeze(-1), dim=-1)
        else:
            feature_importance = torch.full(
                (source_seq.shape[0], self.d_source),
                1.0 / float(self.d_source),
                device=source_seq.device,
                dtype=source_seq.dtype,
            )
        pi_edge = torch.einsum("bd,bdk->bk", feature_importance, pi_lag)
        expected_edge = (pi_edge * lag_axis[None, :]).sum(dim=-1)
        occurrence_logit_edge = (feature_importance * lag_occurrence_logit).sum(dim=-1)

        return {
            "pi_lag": pi_lag,
            "pi_edge": pi_edge,
            "expected_lag": expected_lag,
            "expected_edge": expected_edge,
            "argmax_lag": argmax_lag,
            "argmax_edge": pi_edge.argmax(dim=-1),
            "lag_occurrence_logit": lag_occurrence_logit,
            "occurrence_logit_edge": occurrence_logit_edge,
            "feature_importance": feature_importance,
            "scores": scores,
            "raw_pi_lag": raw_pi_lag,
            "lag_candidate_valid": valid,
        }


def make_gaussian_lag_target(
    lag_gt: torch.Tensor,
    max_lag: int,
    sigma: float,
    positive_only: bool = True,
) -> torch.Tensor:
    if sigma <= 0:
        raise ValueError("gaussian_lag_sigma must be positive")
    if max_lag < 0:
        raise ValueError("max_lag must be non-negative")
    device = lag_gt.device
    dtype = torch.float32 if not torch.is_floating_point(lag_gt) else lag_gt.dtype
    start = 1 if positive_only else 0
    lag_values = torch.arange(start, max_lag + 1, device=device, dtype=dtype)
    center = lag_gt.to(device=device, dtype=dtype).unsqueeze(-1)
    target = torch.exp(-0.5 * ((lag_values[None, :] - center) / float(sigma)) ** 2)
    return target / target.sum(dim=-1, keepdim=True).clamp(min=1e-8)


def lag_identifier_loss(
    outputs: Dict[str, torch.Tensor],
    lag_soft_gt: torch.Tensor,
    lag_flag: torch.Tensor,
    shape_id: Optional[torch.Tensor] = None,
    weights: Optional[LagLossWeights] = None,
    segment_id: Optional[torch.Tensor] = None,
    lag_gt: Optional[torch.Tensor] = None,
    sample_index: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    weights = weights or LagLossWeights()
    pi = outputs["pi_edge"].clamp(min=1e-8, max=1.0)
    target = lag_soft_gt.to(device=pi.device, dtype=pi.dtype).clamp(min=0.0)
    target = target / target.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    lag_flag = lag_flag.to(device=pi.device, dtype=pi.dtype)
    lag_axis = torch.arange(pi.shape[-1], device=pi.device, dtype=pi.dtype)
    hard_gt = target.argmax(dim=-1).long() if lag_gt is None else lag_gt.to(device=pi.device).long()
    segment_id = None if segment_id is None else segment_id.to(device=pi.device)
    sample_index = None if sample_index is None else sample_index.to(device=pi.device)

    soft_lag_per_sample = -(target * pi.log()).sum(dim=-1)
    soft_lag = soft_lag_per_sample.mean()
    expected_pred = (pi * lag_axis[None, :]).sum(dim=-1)
    expected_gt = (target * lag_axis[None, :]).sum(dim=-1)
    expected = (expected_pred - expected_gt).abs().mean()
    occurrence_pos_weight = None
    if weights.occurrence_pos_weight is not None:
        occurrence_pos_weight = pi.new_tensor(float(weights.occurrence_pos_weight))
    occurrence_per_sample = F.binary_cross_entropy_with_logits(
        outputs["occurrence_logit_edge"],
        lag_flag,
        pos_weight=occurrence_pos_weight,
        reduction="none",
    )
    occurrence = occurrence_per_sample.mean()

    entropy = -(pi * pi.log()).sum(dim=-1)
    if shape_id is not None:
        shape_id = shape_id.to(device=pi.device)
        sharp_mask = (shape_id == 1) | (shape_id == 2)
        entropy_term = entropy[sharp_mask].mean() if torch.any(sharp_mask) else pi.new_tensor(0.0)
    else:
        positive_mask = lag_flag > 0.5
        entropy_term = entropy[positive_mask].mean() if torch.any(positive_mask) else pi.new_tensor(0.0)

    smooth = pi.new_tensor(0.0)
    positive_smooth = pi.new_tensor(0.0)
    temporal_enabled = bool(weights.enable_segment_aware_temporal_loss)
    if temporal_enabled and expected_pred.shape[0] > 1 and segment_id is not None and sample_index is not None:
        order = torch.argsort(sample_index)
        pred_ordered = expected_pred[order]
        gt_ordered = expected_gt[order]
        flag_ordered = lag_flag[order]
        seg_ordered = segment_id[order]
        idx_ordered = sample_index[order]
        adjacent = idx_ordered[1:] == (idx_ordered[:-1] + 1)
        same_segment = seg_ordered[1:] == seg_ordered[:-1]
        same_positive = (flag_ordered[1:] > 0.5) & (flag_ordered[:-1] > 0.5)
        pair_mask = adjacent & same_segment & same_positive
        if torch.any(pair_mask):
            pred_delta = pred_ordered[1:] - pred_ordered[:-1]
            gt_delta = gt_ordered[1:] - gt_ordered[:-1]
            smooth = pred_delta[pair_mask].abs().mean()
            positive_smooth = (pred_delta[pair_mask] - gt_delta[pair_mask]).abs().mean()

    positive_ce = pi.new_tensor(0.0)
    positive_kl = pi.new_tensor(0.0)
    if pi.shape[-1] > 1:
        positive_mask = (lag_flag > 0.5) & (hard_gt > 0) & (hard_gt < pi.shape[-1])
        if torch.any(positive_mask):
            pos_pi = pi[positive_mask, 1:]
            pos_pi = pos_pi / pos_pi.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            if weights.use_gaussian_lag_label:
                q = make_gaussian_lag_target(
                    hard_gt[positive_mask],
                    max_lag=pi.shape[-1] - 1,
                    sigma=float(weights.gaussian_lag_sigma),
                    positive_only=True,
                ).to(dtype=pi.dtype)
                positive_kl = F.kl_div(pos_pi.clamp(min=1e-8).log(), q, reduction="batchmean")
                positive_ce = positive_kl
            else:
                pos_target = hard_gt[positive_mask].long() - 1
                class_weight = None
                if weights.positive_ce_class_weights is not None:
                    class_weight = pi.new_tensor(weights.positive_ce_class_weights)
                    if class_weight.numel() != pi.shape[-1] - 1:
                        raise ValueError(
                            "positive_ce_class_weights length must equal the number of positive lag classes"
                        )
                positive_ce = F.nll_loss(pos_pi.clamp(min=1e-8).log(), pos_target, weight=class_weight)

    shape_curvature = pi.new_tensor(0.0)
    if (
        temporal_enabled
        and shape_id is not None
        and expected_pred.shape[0] > 2
        and segment_id is not None
        and sample_index is not None
    ):
        shape_id = shape_id.to(device=pi.device)
        order = torch.argsort(sample_index)
        pred_ordered = expected_pred[order]
        gt_ordered = expected_gt[order]
        flag_ordered = lag_flag[order]
        seg_ordered = segment_id[order]
        idx_ordered = sample_index[order]
        shape_ordered = shape_id[order]
        curvature_ids = weights.shape_curvature_ids if weights.shape_curvature_ids is not None else [5]
        curvature_id_tensor = torch.tensor(curvature_ids, device=pi.device, dtype=shape_ordered.dtype)
        shape_mask = torch.isin(shape_ordered, curvature_id_tensor)
        triplet_mask = (
            shape_mask[2:]
            & shape_mask[1:-1]
            & shape_mask[:-2]
            & (flag_ordered[2:] > 0.5)
            & (flag_ordered[1:-1] > 0.5)
            & (flag_ordered[:-2] > 0.5)
            & (seg_ordered[2:] == seg_ordered[1:-1])
            & (seg_ordered[1:-1] == seg_ordered[:-2])
            & (idx_ordered[1:-1] == (idx_ordered[:-2] + 1))
            & (idx_ordered[2:] == (idx_ordered[1:-1] + 1))
        )
        if torch.any(triplet_mask):
            pred_curvature = pred_ordered[2:] - 2.0 * pred_ordered[1:-1] + pred_ordered[:-2]
            gt_curvature = gt_ordered[2:] - 2.0 * gt_ordered[1:-1] + gt_ordered[:-2]
            shape_curvature = (pred_curvature[triplet_mask] - gt_curvature[triplet_mask]).abs().mean()

    total = (
        float(weights.lambda_soft_lag) * soft_lag
        + float(weights.lambda_expected_lag) * expected
        + float(weights.lambda_occurrence) * occurrence
        + float(weights.lambda_entropy) * entropy_term
        + float(weights.lambda_smooth) * smooth
        + float(weights.lambda_positive_smooth) * positive_smooth
        + float(weights.lambda_positive_ce) * positive_ce
        + float(weights.lambda_shape_curvature) * shape_curvature
    )
    return {
        "loss": total,
        "soft_lag": soft_lag,
        "expected": expected,
        "occurrence": occurrence,
        "entropy": entropy_term,
        "smooth": smooth,
        "positive_smooth": positive_smooth,
        "positive_ce": positive_ce,
        "positive_kl": positive_kl,
        "shape_curvature": shape_curvature,
    }
