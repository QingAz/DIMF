from __future__ import annotations

from typing import Dict, Optional

import torch


def _normalize(values: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    values = values.float()
    v_min = values.min()
    v_max = values.max()
    if float((v_max - v_min).abs().item()) < eps:
        return torch.zeros_like(values)
    return (values - v_min) / (v_max - v_min + eps)


@torch.no_grad()
def attention_mass_score(feature_importance_batches: list[torch.Tensor]) -> torch.Tensor:
    if not feature_importance_batches:
        raise ValueError("feature_importance_batches must not be empty")
    values = torch.cat([batch.detach().cpu() for batch in feature_importance_batches], dim=0)
    return values.mean(dim=0)


def gradient_energy_score(source_seq: torch.Tensor) -> torch.Tensor:
    if source_seq.grad is None:
        return torch.zeros(source_seq.shape[-1], device=source_seq.device)
    grad = source_seq.grad.detach()
    return grad.pow(2).mean(dim=(0, 1))


@torch.no_grad()
def entropy_penalty_score(pi_lag_batches: list[torch.Tensor]) -> torch.Tensor:
    if not pi_lag_batches:
        raise ValueError("pi_lag_batches must not be empty")
    pi = torch.cat([batch.detach().cpu() for batch in pi_lag_batches], dim=0).clamp(min=1e-8)
    entropy = -(pi * pi.log()).sum(dim=-1)
    return entropy.mean(dim=0)


def combine_feature_scores(
    attention_mass: torch.Tensor,
    gradient_energy: Optional[torch.Tensor] = None,
    ablation_importance: Optional[torch.Tensor] = None,
    entropy_penalty: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    score = _normalize(attention_mass.detach().cpu())
    if gradient_energy is not None:
        score = score + _normalize(gradient_energy.detach().cpu())
    if ablation_importance is not None:
        score = score + _normalize(ablation_importance.detach().cpu())
    if entropy_penalty is not None:
        score = score - _normalize(entropy_penalty.detach().cpu())
    return score


def select_feature_mask(
    feature_score: torch.Tensor,
    top_k: Optional[int] = None,
    top_ratio: Optional[float] = 0.3,
) -> torch.Tensor:
    score = feature_score.detach().cpu().float()
    n_features = int(score.numel())
    if n_features == 0:
        raise ValueError("feature_score must contain at least one feature")
    if top_k is None:
        ratio = 1.0 if top_ratio is None else float(top_ratio)
        top_k = max(1, int(round(n_features * ratio)))
    top_k = int(max(1, min(int(top_k), n_features)))
    keep_idx = torch.topk(score, k=top_k, largest=True).indices
    mask = torch.zeros(n_features, dtype=torch.bool)
    mask[keep_idx] = True
    return mask


def apply_feature_screening_to_prior(
    pi_prior: torch.Tensor,
    feature_mask: Optional[torch.Tensor],
    weak_prior_mix: float = 0.0,
) -> torch.Tensor:
    if feature_mask is None:
        return pi_prior
    if pi_prior.ndim != 3:
        raise ValueError("pi_prior must be shaped [B, d_source, K] for feature screening")
    mask = feature_mask.to(device=pi_prior.device, dtype=torch.bool)
    if mask.numel() != pi_prior.shape[1]:
        raise ValueError("feature_mask length must match pi_prior feature dimension")
    out = pi_prior.clone()
    uniform = torch.full_like(out[:, ~mask, :], 1.0 / float(pi_prior.shape[-1]))
    if weak_prior_mix <= 0.0:
        out[:, ~mask, :] = uniform
    else:
        out[:, ~mask, :] = (
            float(weak_prior_mix) * out[:, ~mask, :]
            + (1.0 - float(weak_prior_mix)) * uniform
        )
    return out


def screening_report(
    attention_mass: torch.Tensor,
    feature_score: torch.Tensor,
    feature_mask: torch.Tensor,
) -> Dict[str, object]:
    return {
        "n_features": int(feature_score.numel()),
        "n_selected": int(feature_mask.sum().item()),
        "selected_indices": [int(idx) for idx in torch.nonzero(feature_mask, as_tuple=False).view(-1).tolist()],
        "attention_mass": [float(v) for v in attention_mass.detach().cpu().tolist()],
        "feature_score": [float(v) for v in feature_score.detach().cpu().tolist()],
    }
