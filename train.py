from __future__ import annotations
import json
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
if sys.platform == "win32":
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        libbin = os.path.join(conda_prefix, "Library", "bin")
        os.environ["PATH"] = libbin + ";" + os.environ.get("PATH", "")
        try:
            os.add_dll_directory(libbin)
        except Exception:
            pass

import torch
import torch.nn.functional as F
import yaml
import joblib
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.seed import set_seed
from src.utils.logger import JsonlLogger
from src.utils.metrics import mae, mse, rmse, r2
from src.data.dataprocess import load_and_prepare
from src.data.dataset import MultistageWindowDataset, WindowSpec
from src.models.dimf import DIMF, alignment_consistency_loss, entropy_loss, tv_loss

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--H", type=int, default=None)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()

def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a YAML mapping")
    return data


def _deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str) -> Dict[str, Any]:
    config_path = Path(path).resolve()
    default_path = (Path(__file__).resolve().parent / "configs" / "default.yaml").resolve()
    default_cfg = _load_yaml(default_path)
    if config_path == default_path:
        return default_cfg
    override_cfg = _load_yaml(config_path)
    return _deep_merge_dict(default_cfg, override_cfg)


def _validate_strict_requirements(cfg: Dict[str, Any]) -> None:
    if not bool(cfg.get("strict_requirements", False)):
        return

    data_cfg = cfg.get("data", {})
    strict_checks = {
        "use_delta_t": True,
        "use_missing_mask": True,
        "include_target_history": False,
    }
    errors = []

    for key, expected in strict_checks.items():
        actual = data_cfg.get(key)
        if actual != expected:
            errors.append(f"data.{key} must be {expected!r} in strict mode, got {actual!r}")

    fillna = data_cfg.get("fillna")
    if fillna not in {"ffill", "bfill"}:
        errors.append(f"data.fillna must be 'ffill' or 'bfill' in strict mode, got {fillna!r}")

    if errors:
        raise ValueError("Strict requirement check failed:\n- " + "\n- ".join(errors))


def _checkpoint_metric_value(metric_name: str, val_loss: float, val_metrics: Dict[str, float]) -> float:
    if metric_name == "val_loss":
        return float(val_loss)
    if metric_name == "val_pred":
        return float(val_metrics["pred"])
    if metric_name.startswith("val_"):
        key = metric_name[4:]
        if key in val_metrics:
            return float(val_metrics[key])
    raise ValueError(f"Unknown checkpoint_metric: {metric_name}")


def _compute_lag_class_weights(
    lag_targets: np.ndarray,
    n_classes: int,
    weighting: str,
    positive_only: bool = False,
) -> np.ndarray | None:
    weighting = str(weighting or "none").lower()
    if positive_only:
        valid = lag_targets[(lag_targets > 0) & (lag_targets < n_classes)]
    else:
        valid = lag_targets[(lag_targets >= 0) & (lag_targets < n_classes)]
    if valid.size == 0 or weighting == "none":
        return None

    counts = np.bincount(valid.astype(np.int64), minlength=n_classes).astype(np.float32)
    present = counts > 0
    if not np.any(present):
        return None

    weights = np.zeros(n_classes, dtype=np.float32)
    if weighting == "inverse_frequency":
        weights[present] = float(valid.size) / counts[present]
    elif weighting == "sqrt_inverse_frequency":
        weights[present] = np.sqrt(float(valid.size) / counts[present])
    else:
        raise ValueError(f"Unknown lag supervision weighting: {weighting}")

    weights[present] *= float(np.sum(present)) / float(weights[present].sum())
    return weights


def _compute_occurrence_pos_weight(
    lag_targets: np.ndarray,
    weighting: str,
) -> float | None:
    weighting = str(weighting or "balanced").lower()
    valid = lag_targets[lag_targets >= 0]
    if valid.size == 0 or weighting == "none":
        return None
    if weighting != "balanced":
        raise ValueError(f"Unknown lag occurrence weighting: {weighting}")

    pos = int(np.sum(valid > 0))
    neg = int(np.sum(valid == 0))
    if pos == 0 or neg == 0:
        return None
    return float(neg) / float(pos)


def lag_supervision_terms(
    pi: Dict[str, torch.Tensor],
    X: Dict[str, torch.Tensor],
    edge: str,
    target_key: str,
    class_weights: torch.Tensor | None = None,
    occurrence_pos_weight: torch.Tensor | None = None,
    lambda_occurrence: float = 1.0,
    lambda_positive: float = 1.0,
) -> Dict[str, torch.Tensor]:
    if edge not in pi or target_key not in X:
        reference = next(iter(pi.values()))
        zero = reference.new_tensor(0.0)
        return {"lag_occ": zero, "lag_pos": zero, "lag_sup": zero}

    arr = pi[edge]
    arr_last = arr[:, -1, :] if arr.dim() == 3 else arr
    lag_target = X[target_key].long()
    valid = (lag_target >= 0) & (lag_target < arr_last.shape[-1])
    if not torch.any(valid):
        zero = arr_last.new_tensor(0.0)
        return {"lag_occ": zero, "lag_pos": zero, "lag_sup": zero}

    eps = 1e-6
    probs = arr_last[valid].clamp(min=eps, max=1.0 - eps)
    target = lag_target[valid]
    occ_target = target.gt(0).to(probs.dtype)
    occ_prob = (1.0 - probs[:, 0]).clamp(min=eps, max=1.0 - eps)

    lag_occ = lag_binary_aux_term_from_probs(
        occ_prob=occ_prob,
        occ_target=occ_target,
        occurrence_pos_weight=occurrence_pos_weight,
        focal_gamma=0.0,
    )

    lag_pos = probs.new_tensor(0.0)
    positive_mask = target > 0
    if torch.any(positive_mask):
        pos_probs = probs[positive_mask, 1:]
        pos_probs = pos_probs / pos_probs.sum(dim=-1, keepdim=True).clamp(min=eps)
        pos_target = target[positive_mask] - 1
        pos_class_weights = None
        if class_weights is not None:
            if class_weights.numel() == probs.shape[-1]:
                pos_class_weights = class_weights[1:]
            elif class_weights.numel() == probs.shape[-1] - 1:
                pos_class_weights = class_weights
            else:
                raise ValueError("Positive-lag class weights must have K or K-1 entries")
            pos_class_weights = pos_class_weights.to(device=pos_probs.device, dtype=pos_probs.dtype)
        lag_pos = F.nll_loss(
            pos_probs.clamp(min=eps).log(),
            pos_target,
            weight=pos_class_weights,
            reduction="mean",
        )

    lag_sup = float(lambda_occurrence) * lag_occ + float(lambda_positive) * lag_pos
    return {"lag_occ": lag_occ, "lag_pos": lag_pos, "lag_sup": lag_sup}


def lag_binary_aux_term_from_probs(
    occ_prob: torch.Tensor,
    occ_target: torch.Tensor,
    occurrence_pos_weight: torch.Tensor | None = None,
    focal_gamma: float = 0.0,
) -> torch.Tensor:
    eps = 1e-6
    occ_prob = occ_prob.clamp(min=eps, max=1.0 - eps)
    occ_target = occ_target.to(dtype=occ_prob.dtype)

    if occurrence_pos_weight is None:
        bce = F.binary_cross_entropy(occ_prob, occ_target, reduction="none")
    else:
        pos_weight = occurrence_pos_weight.to(device=occ_prob.device, dtype=occ_prob.dtype)
        bce = -(
            pos_weight * occ_target * occ_prob.log()
            + (1.0 - occ_target) * (1.0 - occ_prob).log()
        )

    if focal_gamma > 0.0:
        p_t = occ_target * occ_prob + (1.0 - occ_target) * (1.0 - occ_prob)
        bce = ((1.0 - p_t).clamp(min=0.0) ** float(focal_gamma)) * bce
    return bce.mean()


def lag_binary_aux_term(
    pi: Dict[str, torch.Tensor],
    X: Dict[str, torch.Tensor],
    edge: str,
    target_key: str,
    occurrence_pos_weight: torch.Tensor | None = None,
    focal_gamma: float = 0.0,
) -> torch.Tensor:
    if edge not in pi or target_key not in X:
        reference = next(iter(pi.values()))
        return reference.new_tensor(0.0)

    arr = pi[edge]
    arr_last = arr[:, -1, :] if arr.dim() == 3 else arr
    lag_target = X[target_key].long()
    valid = (lag_target >= 0) & (lag_target < arr_last.shape[-1])
    if not torch.any(valid):
        return arr_last.new_tensor(0.0)

    probs = arr_last[valid].clamp(min=1e-6, max=1.0 - 1e-6)
    occ_target = lag_target[valid].gt(0).to(probs.dtype)
    occ_prob = 1.0 - probs[:, 0]
    return lag_binary_aux_term_from_probs(
        occ_prob=occ_prob,
        occ_target=occ_target,
        occurrence_pos_weight=occurrence_pos_weight,
        focal_gamma=focal_gamma,
    )


def lag_positive_expected_aux_term(
    pi: Dict[str, torch.Tensor],
    X: Dict[str, torch.Tensor],
    edge: str,
    target_key: str,
) -> torch.Tensor:
    if edge not in pi or target_key not in X:
        reference = next(iter(pi.values()))
        return reference.new_tensor(0.0)

    arr = pi[edge]
    arr_last = arr[:, -1, :] if arr.dim() == 3 else arr
    lag_target = X[target_key].long()
    valid = (lag_target > 0) & (lag_target < arr_last.shape[-1])
    if not torch.any(valid):
        return arr_last.new_tensor(0.0)

    probs = arr_last[valid]
    lag_axis = torch.arange(probs.shape[-1], device=probs.device, dtype=probs.dtype)
    pred_expected = (probs * lag_axis[None, :]).sum(dim=-1)
    true_lag = lag_target[valid].to(dtype=probs.dtype)
    return (pred_expected - true_lag).abs().mean()


def lag_soft_target_term(
    pi: Dict[str, torch.Tensor],
    X: Dict[str, torch.Tensor],
    edge: str,
    target_key: str,
) -> torch.Tensor:
    if edge not in pi or target_key not in X:
        reference = next(iter(pi.values()))
        return reference.new_tensor(0.0)

    arr = pi[edge]
    arr_last = arr[:, -1, :] if arr.dim() == 3 else arr
    target = X[target_key].to(device=arr_last.device, dtype=arr_last.dtype)
    if target.ndim != 2:
        raise ValueError(f"Soft lag target '{target_key}' must be shaped [B, K]")
    if target.shape[-1] != arr_last.shape[-1]:
        raise ValueError(
            f"Soft lag target '{target_key}' has K={target.shape[-1]}, expected {arr_last.shape[-1]}"
        )

    valid = target.sum(dim=-1) > 0.0
    if not torch.any(valid):
        return arr_last.new_tensor(0.0)
    target = target[valid].clamp(min=0.0)
    target = target / target.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    probs = arr_last[valid].clamp(min=1e-6, max=1.0)
    return -(target * probs.log()).sum(dim=-1).mean()


def lag_zero_penalty_term(
    pi: Dict[str, torch.Tensor],
    X: Dict[str, torch.Tensor],
    edge: str,
    target_key: str,
    mode: str = "nll_zero",
) -> torch.Tensor:
    if edge not in pi or target_key not in X:
        reference = next(iter(pi.values()))
        return reference.new_tensor(0.0)

    arr = pi[edge]
    arr_last = arr[:, -1, :] if arr.dim() == 3 else arr
    lag_target = X[target_key].long()
    valid = lag_target.eq(0)
    if not torch.any(valid):
        return arr_last.new_tensor(0.0)

    probs = arr_last[valid].clamp(min=1e-6, max=1.0)
    mode = str(mode or "nll_zero").lower()
    if mode in {"nll_zero", "zero_nll", "logprob_zero"}:
        return -(probs[:, 0].clamp(min=1e-6).log()).mean()
    if mode in {"expected_lag", "mean_expected_lag"}:
        lag_axis = torch.arange(probs.shape[-1], device=probs.device, dtype=probs.dtype)
        pred_expected = (probs * lag_axis[None, :]).sum(dim=-1)
        return pred_expected.mean()
    raise ValueError(f"Unknown lag_zero_mode: {mode}")

def to_device(batch, device):
    X, y = batch
    X = {k: v.to(device) for k, v in X.items()}
    y = y.to(device)
    return X, y

@torch.no_grad()
def _update_delay_stats(acc: Dict[str, float], pi: Dict[str, torch.Tensor], batch_n: int) -> None:
    for edge, arr in pi.items():
        arr_last = arr[:, -1, :] if arr.dim() == 3 else arr
        lag_axis = torch.arange(arr_last.shape[-1], device=arr_last.device, dtype=arr_last.dtype)
        exp_lag = (arr_last * lag_axis[None, :]).sum(dim=-1)
        peak_prob = arr_last.max(dim=-1).values
        acc[f"{edge}_expected_lag_sum"] += float(exp_lag.sum().item())
        acc[f"{edge}_peak_prob_sum"] += float(peak_prob.sum().item())
        acc[f"{edge}_count"] += batch_n

def _finalize_delay_stats(acc: Dict[str, float]) -> Dict[str, float]:
    out = {}
    for key, value in acc.items():
        if not key.endswith("_count"):
            continue
        edge = key[:-6]
        count = max(int(value), 1)
        out[f"{edge}_expected_lag"] = acc[f"{edge}_expected_lag_sum"] / count
        out[f"{edge}_peak_prob"] = acc[f"{edge}_peak_prob_sum"] / count
    return out


def _scheduled_regularization_weight(
    target_value: float,
    epoch: int,
    pred_warmup_epochs: int,
    ramp_epochs: int,
) -> float:
    """
    第 9 点修改：两阶段 warm-up 调度
    1. 前 pred_warmup_epochs 个 epoch 只优化预测损失，正则权重固定为 0
    2. 随后在 ramp_epochs 个 epoch 内，将权重从 0 线性升到目标值
    """
    if pred_warmup_epochs < 0 or ramp_epochs < 0:
        raise ValueError("pred_warmup_epochs and ramp_epochs must be non-negative")
    if epoch <= pred_warmup_epochs:
        return 0.0
    if ramp_epochs == 0:
        return float(target_value)
    progress = min(epoch - pred_warmup_epochs, ramp_epochs) / float(ramp_epochs)
    return float(target_value) * progress

@torch.no_grad()
def eval_epoch_metrics(
    model,
    loader,
    device,
    lam_align,
    lam_ent,
    lam_tv,
    lam_lag_sup,
    lam_lag_binary_aux,
    lam_lag_pos_expected_aux,
    lam_lag_soft_target_aux,
    lam_lag_zero,
    align_loss_temp,
    lag_supervision_edge: str | None = None,
    lag_supervision_key: str | None = None,
    lag_class_weights: torch.Tensor | None = None,
    lag_occurrence_pos_weight: torch.Tensor | None = None,
    lag_lambda_occurrence: float = 1.0,
    lag_lambda_positive: float = 1.0,
    lag_binary_aux_gamma: float = 0.0,
    lag_soft_target_key: str | None = None,
    lag_zero_mode: str = "nll_zero",
):
    model.eval()
    tot_loss, tot_pred, tot_align, tot_ent, tot_tv, tot_lag_occ, tot_lag_pos, tot_lag_sup, tot_lag_bin, tot_lag_posexp, tot_lag_soft, tot_lag_zero, n = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0,
    )
    delay_stats = {}
    for batch in loader:
        X, y = to_device(batch, device)
        y_hat, pi = model(X)
        pred = (y_hat - y).abs().mean()
        # 第 8 点修改：基于窗口末端当前时刻 t 的表示，计算 batch 内对称 InfoNCE。
        align = alignment_consistency_loss(model.latest_alignment_cache, temperature=align_loss_temp)
        ent = sum(entropy_loss(v) for v in pi.values())
        tv  = sum(tv_loss(v) for v in pi.values())
        lag_occ = pred.new_tensor(0.0)
        lag_pos = pred.new_tensor(0.0)
        lag_sup = pred.new_tensor(0.0)
        lag_bin = pred.new_tensor(0.0)
        lag_posexp = pred.new_tensor(0.0)
        lag_soft = pred.new_tensor(0.0)
        lag_zero = pred.new_tensor(0.0)
        if lam_lag_sup > 0.0 and lag_supervision_edge is not None and lag_supervision_key is not None:
            lag_terms = lag_supervision_terms(
                pi,
                X,
                edge=lag_supervision_edge,
                target_key=lag_supervision_key,
                class_weights=lag_class_weights,
                occurrence_pos_weight=lag_occurrence_pos_weight,
                lambda_occurrence=lag_lambda_occurrence,
                lambda_positive=lag_lambda_positive,
            )
            lag_occ = lag_terms["lag_occ"]
            lag_pos = lag_terms["lag_pos"]
            lag_sup = lag_terms["lag_sup"]
        if lam_lag_binary_aux > 0.0 and lag_supervision_edge is not None and lag_supervision_key is not None:
            lag_bin = lag_binary_aux_term(
                pi,
                X,
                edge=lag_supervision_edge,
                target_key=lag_supervision_key,
                occurrence_pos_weight=lag_occurrence_pos_weight,
                focal_gamma=lag_binary_aux_gamma,
            )
        if lam_lag_pos_expected_aux > 0.0 and lag_supervision_edge is not None and lag_supervision_key is not None:
            lag_posexp = lag_positive_expected_aux_term(
                pi,
                X,
                edge=lag_supervision_edge,
                target_key=lag_supervision_key,
            )
        if lam_lag_soft_target_aux > 0.0 and lag_supervision_edge is not None and lag_soft_target_key is not None:
            lag_soft = lag_soft_target_term(
                pi,
                X,
                edge=lag_supervision_edge,
                target_key=lag_soft_target_key,
            )
        if lam_lag_zero > 0.0 and lag_supervision_edge is not None and lag_supervision_key is not None:
            lag_zero = lag_zero_penalty_term(
                pi,
                X,
                edge=lag_supervision_edge,
                target_key=lag_supervision_key,
                mode=lag_zero_mode,
            )
        loss = (
            pred
            + lam_align * align
            + lam_ent * ent
            + lam_tv * tv
            + lam_lag_sup * lag_sup
            + lam_lag_binary_aux * lag_bin
            + lam_lag_pos_expected_aux * lag_posexp
            + lam_lag_soft_target_aux * lag_soft
            + lam_lag_zero * lag_zero
        )
        batch_n = y.shape[0]
        tot_loss += float(loss.item()) * batch_n
        tot_pred += float(pred.item()) * batch_n
        tot_align += float(align.item()) * batch_n
        tot_ent += float(ent.item()) * batch_n
        tot_tv += float(tv.item()) * batch_n
        tot_lag_occ += float(lag_occ.item()) * batch_n
        tot_lag_pos += float(lag_pos.item()) * batch_n
        tot_lag_sup += float(lag_sup.item()) * batch_n
        tot_lag_bin += float(lag_bin.item()) * batch_n
        tot_lag_posexp += float(lag_posexp.item()) * batch_n
        tot_lag_soft += float(lag_soft.item()) * batch_n
        tot_lag_zero += float(lag_zero.item()) * batch_n
        n += batch_n
        if not delay_stats:
            for edge in pi:
                delay_stats[f"{edge}_expected_lag_sum"] = 0.0
                delay_stats[f"{edge}_peak_prob_sum"] = 0.0
                delay_stats[f"{edge}_count"] = 0.0
        _update_delay_stats(delay_stats, pi, batch_n)

    denom = max(n, 1)
    out = {
        "loss": tot_loss / denom,
        "pred": tot_pred / denom,
        "align": tot_align / denom,
        "ent": tot_ent / denom,
        "tv": tot_tv / denom,
        "lag_occ": tot_lag_occ / denom,
        "lag_pos": tot_lag_pos / denom,
        "lag_sup": tot_lag_sup / denom,
        "lag_bin": tot_lag_bin / denom,
        "lag_posexp": tot_lag_posexp / denom,
        "lag_soft": tot_lag_soft / denom,
        "lag_zero": tot_lag_zero / denom,
    }
    out.update(_finalize_delay_stats(delay_stats))
    return out

def _prediction_frame(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    input_timestamps: np.ndarray,
    target_timestamps: np.ndarray,
    target_cols: list[str] | None = None,
) -> pd.DataFrame:
    # 第 5 节修改：预测结果同时导出“输入时刻 t”和“目标时刻 t+H”，
    # 避免把当前窗口时刻和预测目标时刻混为同一个时间戳。
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim == 1:
        y_true = y_true[:, None]
    if y_pred.ndim == 1:
        y_pred = y_pred[:, None]
    if y_true.ndim != 2 or y_pred.ndim != 2 or y_true.shape != y_pred.shape:
        raise ValueError("Prediction export expects matching [N, D] y_true/y_pred arrays")
    if input_timestamps.shape[0] != y_true.shape[0] or target_timestamps.shape[0] != y_true.shape[0]:
        raise ValueError("Prediction timestamps must align with y_true/y_pred length")
    if target_cols is None:
        target_cols = ["y"] if y_true.shape[1] == 1 else [f"target_{idx}" for idx in range(y_true.shape[1])]
    if len(target_cols) != y_true.shape[1]:
        raise ValueError("target_cols length must match prediction dimension")
    cols = {
        "InputTimeStamp": input_timestamps,
        "TargetTimeStamp": target_timestamps,
    }
    for idx, name in enumerate(target_cols):
        cols[f"{name}_true"] = y_true[:, idx]
        cols[f"{name}_pred"] = y_pred[:, idx]
    return pd.DataFrame(cols)


def _delay_estimate_frame(sample_timestamps: np.ndarray, pis: Dict[str, list], collection_interval_min: int) -> pd.DataFrame:
    # 第 5 节修改：滞后估计文件严格对应当前时刻 t，并同时导出“步数”和“分钟”两种解释量。
    n_samples = len(sample_timestamps)
    cols = {"TimeStamp": sample_timestamps}
    for edge, arrs in pis.items():
        arr = np.concatenate(arrs, axis=0)
        arr_last = arr[:, -1, :] if arr.ndim == 3 else arr
        if arr_last.shape[0] != n_samples:
            raise ValueError(
                f"Delay estimate/sample timestamp mismatch for {edge}: "
                f"{arr_last.shape[0]} estimates vs {n_samples} timestamps"
            )
        lag_axis = np.arange(arr_last.shape[1], dtype=np.float32)
        cols[f"{edge}_pred_expected_lag"] = (arr_last * lag_axis[None, :]).sum(axis=1)
        cols[f"{edge}_pred_argmax_lag"] = arr_last.argmax(axis=1)
        cols[f"{edge}_pred_expected_lag_minutes"] = cols[f"{edge}_pred_expected_lag"] * float(collection_interval_min)
        cols[f"{edge}_pred_argmax_lag_minutes"] = cols[f"{edge}_pred_argmax_lag"] * float(collection_interval_min)
        for lag in range(arr_last.shape[1]):
            cols[f"{edge}_pred_pi_lag{lag}"] = arr_last[:, lag]
    return pd.DataFrame(cols)

@torch.no_grad()
def eval_test(
    model,
    loader,
    device,
    scaler_y,
    output_dir: str,
    input_timestamps: np.ndarray,
    target_timestamps: np.ndarray,
    collection_interval_min: int,
    target_cols: list[str] | None = None,
):
    model.eval()
    y_true, y_pred = [], []
    pis = {"feed_to_stage1": [], "stage1_to_stage2": [], "stage2_to_stage3": []}
    for X, y in loader:
        X = {k: v.to(device) for k, v in X.items()}
        y = y.to(device)
        y_hat, pi = model(X)
        y_true.append(y.cpu().numpy())
        y_pred.append(y_hat.cpu().numpy())
        for k in pis:
            pis[k].append(pi[k].cpu().numpy())

    # 单点任务下，batch 维拼接后应得到 [N]。
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    y_true_2d = y_true[:, None] if y_true.ndim == 1 else y_true
    y_pred_2d = y_pred[:, None] if y_pred.ndim == 1 else y_pred

    scaled_res = {
        "scaled_MSE": mse(y_true_2d, y_pred_2d),
        "scaled_MAE": mae(y_true_2d, y_pred_2d),
        "scaled_RMSE": rmse(y_true_2d, y_pred_2d),
        "scaled_R2": r2(y_true_2d.reshape(-1), y_pred_2d.reshape(-1)),
    }

    y_true_inv = scaler_y.inverse_transform(y_true_2d)
    y_pred_inv = scaler_y.inverse_transform(y_pred_2d)

    res = {
        "MSE": mse(y_true_inv, y_pred_inv),
        "MAE": mae(y_true_inv, y_pred_inv),
        "RMSE": rmse(y_true_inv, y_pred_inv),
        "R2": r2(y_true_inv.reshape(-1), y_pred_inv.reshape(-1)),
        "n_test": int(y_true_inv.shape[0]),
        "target_dim": int(y_true_inv.shape[1]),
    }
    res.update(scaled_res)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _prediction_frame(y_true_inv, y_pred_inv, input_timestamps, target_timestamps, target_cols).to_csv(out_dir / "test_pred_vs_true.csv", index=False)
    _prediction_frame(y_true_2d, y_pred_2d, input_timestamps, target_timestamps, target_cols).to_csv(out_dir / "test_pred_vs_true_scaled.csv", index=False)

    avg_pi = {}
    last_step_pi = {}
    for k, arrs in pis.items():
        arr = np.concatenate(arrs, axis=0)
        if arr.ndim == 3:                   # [N, L, K]
            avg_pi[k] = arr.mean(axis=(0, 1))
            last_step_pi[k] = arr[:, -1, :]
        else:                               # [N, K]
            avg_pi[k] = arr.mean(axis=0)
            last_step_pi[k] = arr
    np.save(out_dir / "test_delay_pi.npy", avg_pi, allow_pickle=True)
    np.save(out_dir / "test_delay_laststep_pi.npy", last_step_pi, allow_pickle=True)
    _delay_estimate_frame(input_timestamps, pis, collection_interval_min).to_csv(out_dir / "test_delay_estimates.csv", index=False)
    return res

def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.H is not None:
        cfg["data"]["H"] = int(args.H)
    _validate_strict_requirements(cfg)

    set_seed(int(cfg.get("seed", 42)))

    prepared, _ = load_and_prepare(
        csv_path=cfg["data"]["csv_path"],
        time_col=cfg["data"]["time_col"],
        target_col=cfg["data"]["target_col"],
        target_cols=cfg["data"].get("target_cols"),
        feed_prefix=cfg["data"]["feed_prefix"],
        stage1_prefix=cfg["data"]["stage1_prefix"],
        stage2_prefix=cfg["data"]["stage2_prefix"],
        stage3_prefix=cfg["data"]["stage3_prefix"],
        fillna=cfg["data"].get("fillna", "ffill"),
        use_delta_t=bool(cfg["data"].get("use_delta_t", True)),
        train_ratio=float(cfg["data"]["train_ratio"]),
        val_ratio=float(cfg["data"]["val_ratio"]),
        test_ratio=float(cfg["data"]["test_ratio"]),
        split_mode=str(cfg["data"].get("split_mode", "rows")),
        history_steps=int(cfg["data"]["L"]),
        horizon_steps=int(cfg["data"]["H"]),
        collection_interval_min=int(cfg["data"].get("collection_interval_min", 15)),
        gap_break_min=int(cfg["data"].get("gap_break_min", 120)),
        gap_fill_min=int(cfg["data"].get("gap_fill_min", 60)),
        use_missing_mask=bool(cfg["data"].get("use_missing_mask", True)),
        include_target_history=bool(cfg["data"].get("include_target_history", False)),
        split_col=str(cfg["data"].get("split_col", "split")),
    )

    # save scalers (fit on train only)
    Path(cfg["logging"]["scaler_path"]).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"scaler_x": prepared.scaler_x, "scaler_y": prepared.scaler_y}, cfg["logging"]["scaler_path"])

    spec = WindowSpec(L=int(cfg["data"]["L"]), H=int(cfg["data"]["H"]))
    ds_tr = MultistageWindowDataset(
        prepared.X_groups_train,
        prepared.y_train,
        spec,
        indices=prepared.sample_indices_train,
        extra_targets=prepared.extra_targets_train,
    )
    ds_va = MultistageWindowDataset(
        prepared.X_groups_val,
        prepared.y_val,
        spec,
        indices=prepared.sample_indices_val,
        extra_targets=prepared.extra_targets_val,
    )
    ds_te = MultistageWindowDataset(
        prepared.X_groups_test,
        prepared.y_test,
        spec,
        indices=prepared.sample_indices_test,
        extra_targets=prepared.extra_targets_test,
    )

    dl_tr = DataLoader(ds_tr, batch_size=int(cfg["train"]["batch_size"]), shuffle=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=int(cfg["train"]["batch_size"]), shuffle=False)
    dl_te = DataLoader(ds_te, batch_size=int(cfg["train"]["batch_size"]), shuffle=False)

    device = torch.device(args.device)
    lam_align_target = float(cfg["train"]["lambda_align"])
    lam_ent_target  = float(cfg["train"]["lambda_ent"])
    lam_tv_target   = float(cfg["train"]["lambda_tv"])
    lam_lag_sup_target = float(cfg["train"].get("lambda_lag_supervision", 0.0))
    lam_lag_binary_aux_target = float(cfg["train"].get("lambda_lag_binary_aux", 0.0))
    lam_lag_pos_expected_aux_target = float(cfg["train"].get("lambda_lag_pos_expected_aux", 0.0))
    lam_lag_soft_target_aux_target = float(cfg["train"].get("lambda_lag_soft_target_aux", 0.0))
    lam_lag_zero_target = float(cfg["train"].get("lambda_lag_zero", 0.0))
    align_loss_temp = float(cfg["train"]["align_loss_temp"])
    pred_warmup_epochs = int(cfg["train"]["pred_warmup_epochs"])
    ramp_epochs = int(cfg["train"]["ramp_epochs"])
    grad_clip = float(cfg["train"].get("grad_clip", 1.0))
    lag_supervision_edge = cfg["train"].get("lag_supervision_edge")
    lag_supervision_key = cfg["train"].get("lag_supervision_key")
    lag_soft_target_key = cfg["train"].get("lag_soft_target_target_key")
    if lag_soft_target_key is None and lag_supervision_edge is not None:
        lag_soft_target_key = f"{lag_supervision_edge}_lag_soft_gt"
    lag_supervision_weighting = str(cfg["train"].get("lag_supervision_weighting", "none"))
    lag_occurrence_weighting = str(cfg["train"].get("lag_occurrence_weighting", "balanced"))
    lag_lambda_occurrence = float(cfg["train"].get("lambda_lag_occurrence", 1.0))
    lag_lambda_positive = float(cfg["train"].get("lambda_lag_positive", 1.0))
    lag_binary_aux_gamma = float(cfg["train"].get("lag_binary_aux_gamma", 0.0))
    lag_binary_aux_pos_weight = cfg["train"].get("lag_binary_aux_pos_weight")
    lag_zero_mode = str(cfg["train"].get("lag_zero_mode", "nll_zero"))
    checkpoint_metric = str(cfg["train"].get("checkpoint_metric", "val_loss"))
    checkpoint_after_warmup_only = bool(cfg["train"].get("checkpoint_after_warmup_only", False))

    # 第 5 节修改：预测值对应目标时刻 t+H，而滞后分布对应当前时刻 t，二者时间戳分别导出。
    test_input_timestamps = prepared.timestamps_test[ds_te.indices]
    test_target_timestamps = prepared.timestamps_test[ds_te.indices + int(cfg["data"]["H"])]
    model = DIMF(
        group_dims=prepared.group_dims,
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        num_layers=int(cfg["model"]["num_layers"]),
        dropout=float(cfg["model"]["dropout"]),
        attn_dim=int(cfg["model"]["attn_dim"]),
        L_max=int(cfg["data"]["L_max"]),
        lead_steps=int(cfg["data"]["H"]),
        encoder_type=str(cfg["model"].get("encoder", "gru")),
        transformer_nhead=int(cfg["model"].get("transformer_nhead", 4)),
        transformer_ff_dim=cfg["model"].get("transformer_ff_dim", None),
        max_len=int(cfg["data"]["L"]),
        lag_emb=bool(cfg["model"].get("lag_emb", True)),
        use_alignment=bool(cfg["model"].get("use_alignment", True)),
        align_tau=float(cfg["model"].get("align_tau", 1.0)),
        align_dropout=float(cfg["model"].get("align_dropout", 0.0)),
        align_feed_to_stage1=cfg["model"].get("align_feed_to_stage1"),
        align_stage1_to_stage2=cfg["model"].get("align_stage1_to_stage2"),
        align_stage2_to_stage3=cfg["model"].get("align_stage2_to_stage3"),
        use_lag_bias=bool(cfg["model"].get("use_lag_bias", True)),
        head_stage=str(cfg["model"].get("head_stage", "stage3")),
        use_feed_context=bool(cfg["model"].get("use_feed_context", True)),
        lag_head_mode=str(cfg["model"].get("lag_head_mode", "softmax")),
        output_dim=len(prepared.target_cols or [cfg["data"]["target_col"]]),
    ).to(device)

    lag_class_weights = None
    lag_occurrence_pos_weight = None
    train_lag_targets = None
    if (
        lag_supervision_key is not None
        and prepared.extra_targets_train is not None
        and lag_supervision_key in prepared.extra_targets_train
    ):
        train_lag_targets = prepared.extra_targets_train[lag_supervision_key][ds_tr.indices]
    if (
        lam_lag_sup_target > 0.0
        and train_lag_targets is not None
    ):
        weight_values = _compute_lag_class_weights(
            lag_targets=train_lag_targets,
            n_classes=int(cfg["data"]["L_max"]) + 1,
            weighting=lag_supervision_weighting,
            positive_only=True,
        )
        if weight_values is not None:
            lag_class_weights = torch.tensor(weight_values, dtype=torch.float32, device=device)
    if (lam_lag_sup_target > 0.0 or lam_lag_binary_aux_target > 0.0) and train_lag_targets is not None:
        if lag_binary_aux_pos_weight is not None:
            lag_occurrence_pos_weight = torch.tensor(float(lag_binary_aux_pos_weight), dtype=torch.float32, device=device)
        else:
            occurrence_pos_weight = _compute_occurrence_pos_weight(
                lag_targets=train_lag_targets,
                weighting=lag_occurrence_weighting,
            )
            if occurrence_pos_weight is not None:
                lag_occurrence_pos_weight = torch.tensor(occurrence_pos_weight, dtype=torch.float32, device=device)

    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"].get("weight_decay", 0.0)),
    )

    logger = JsonlLogger(cfg["logging"]["log_path"])
    ckpt_path = cfg["logging"]["ckpt_path"]
    Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
    best, best_epoch = 1e18, 0
    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        model.train()
        # 第 9 点修改：每个 epoch 先根据 warm-up 规则计算当前生效的正则权重。
        current_lam_align = _scheduled_regularization_weight(
            lam_align_target,
            epoch,
            pred_warmup_epochs,
            ramp_epochs,
        )
        current_lam_ent = _scheduled_regularization_weight(
            lam_ent_target,
            epoch,
            pred_warmup_epochs,
            ramp_epochs,
        )
        current_lam_tv = _scheduled_regularization_weight(
            lam_tv_target,
            epoch,
            pred_warmup_epochs,
            ramp_epochs,
        )
        current_lam_lag_sup = _scheduled_regularization_weight(
            lam_lag_sup_target,
            epoch,
            pred_warmup_epochs,
            ramp_epochs,
        )
        current_lam_lag_binary_aux = _scheduled_regularization_weight(
            lam_lag_binary_aux_target,
            epoch,
            pred_warmup_epochs,
            ramp_epochs,
        )
        current_lam_lag_pos_expected_aux = _scheduled_regularization_weight(
            lam_lag_pos_expected_aux_target,
            epoch,
            pred_warmup_epochs,
            ramp_epochs,
        )
        current_lam_lag_soft_target_aux = _scheduled_regularization_weight(
            lam_lag_soft_target_aux_target,
            epoch,
            pred_warmup_epochs,
            ramp_epochs,
        )
        current_lam_lag_zero = _scheduled_regularization_weight(
            lam_lag_zero_target,
            epoch,
            pred_warmup_epochs,
            ramp_epochs,
        )
        tot_loss, tot_pred, tot_align, tot_ent, tot_tv, tot_lag_occ, tot_lag_pos, tot_lag_sup, tot_lag_bin, tot_lag_posexp, tot_lag_soft, tot_lag_zero, n = (
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0,
        )
        train_delay_stats = {}

        pbar = tqdm(dl_tr, desc=f"Epoch {epoch}")
        for batch in pbar:
            X, y = to_device(batch, device)
            y_hat, pi = model(X)

            pred = (y_hat - y).abs().mean()
            # 第 8 点修改：一致性损失直接读取 forward 后缓存的对齐结果，不再额外重复对齐计算。
            align = alignment_consistency_loss(model.latest_alignment_cache, temperature=align_loss_temp)
            ent  = sum(entropy_loss(v) for v in pi.values())
            tv   = sum(tv_loss(v) for v in pi.values())
            lag_occ = pred.new_tensor(0.0)
            lag_pos = pred.new_tensor(0.0)
            lag_sup = pred.new_tensor(0.0)
            lag_bin = pred.new_tensor(0.0)
            lag_posexp = pred.new_tensor(0.0)
            lag_soft = pred.new_tensor(0.0)
            lag_zero = pred.new_tensor(0.0)
            if current_lam_lag_sup > 0.0 and lag_supervision_edge is not None and lag_supervision_key is not None:
                lag_terms = lag_supervision_terms(
                    pi,
                    X,
                    edge=str(lag_supervision_edge),
                    target_key=str(lag_supervision_key),
                    class_weights=lag_class_weights,
                    occurrence_pos_weight=lag_occurrence_pos_weight,
                    lambda_occurrence=lag_lambda_occurrence,
                    lambda_positive=lag_lambda_positive,
                )
                lag_occ = lag_terms["lag_occ"]
                lag_pos = lag_terms["lag_pos"]
                lag_sup = lag_terms["lag_sup"]
            if current_lam_lag_binary_aux > 0.0 and lag_supervision_edge is not None and lag_supervision_key is not None:
                lag_bin = lag_binary_aux_term(
                    pi,
                    X,
                    edge=str(lag_supervision_edge),
                    target_key=str(lag_supervision_key),
                    occurrence_pos_weight=lag_occurrence_pos_weight,
                    focal_gamma=lag_binary_aux_gamma,
                )
            if current_lam_lag_pos_expected_aux > 0.0 and lag_supervision_edge is not None and lag_supervision_key is not None:
                lag_posexp = lag_positive_expected_aux_term(
                    pi,
                    X,
                    edge=str(lag_supervision_edge),
                    target_key=str(lag_supervision_key),
                )
            if current_lam_lag_soft_target_aux > 0.0 and lag_supervision_edge is not None and lag_soft_target_key is not None:
                lag_soft = lag_soft_target_term(
                    pi,
                    X,
                    edge=str(lag_supervision_edge),
                    target_key=str(lag_soft_target_key),
                )
            if current_lam_lag_zero > 0.0 and lag_supervision_edge is not None and lag_supervision_key is not None:
                lag_zero = lag_zero_penalty_term(
                    pi,
                    X,
                    edge=str(lag_supervision_edge),
                    target_key=str(lag_supervision_key),
                    mode=lag_zero_mode,
                )
            loss = (
                pred
                + current_lam_align * align
                + current_lam_ent * ent
                + current_lam_tv * tv
                + current_lam_lag_sup * lag_sup
                + current_lam_lag_binary_aux * lag_bin
                + current_lam_lag_pos_expected_aux * lag_posexp
                + current_lam_lag_soft_target_aux * lag_soft
                + current_lam_lag_zero * lag_zero
            )

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            batch_n = y.shape[0]
            tot_loss += float(loss.item()) * batch_n
            tot_pred += float(pred.item()) * batch_n
            tot_align += float(align.item()) * batch_n
            tot_ent += float(ent.item()) * batch_n
            tot_tv += float(tv.item()) * batch_n
            tot_lag_occ += float(lag_occ.item()) * batch_n
            tot_lag_pos += float(lag_pos.item()) * batch_n
            tot_lag_sup += float(lag_sup.item()) * batch_n
            tot_lag_bin += float(lag_bin.item()) * batch_n
            tot_lag_posexp += float(lag_posexp.item()) * batch_n
            tot_lag_soft += float(lag_soft.item()) * batch_n
            tot_lag_zero += float(lag_zero.item()) * batch_n
            n += batch_n
            if not train_delay_stats:
                for edge in pi:
                    train_delay_stats[f"{edge}_expected_lag_sum"] = 0.0
                    train_delay_stats[f"{edge}_peak_prob_sum"] = 0.0
                    train_delay_stats[f"{edge}_count"] = 0.0
            _update_delay_stats(train_delay_stats, pi, batch_n)
            pbar.set_postfix(
                loss=float(loss.item()),
                pred=float(pred.item()),
                align=float(align.item()),
                ent=float(ent.item()),
                tv=float(tv.item()),
                lag_occ=float(lag_occ.item()),
                lag_pos=float(lag_pos.item()),
                lag_sup=float(lag_sup.item()),
                lag_bin=float(lag_bin.item()),
                lag_posexp=float(lag_posexp.item()),
                lag_soft=float(lag_soft.item()),
                lag_zero=float(lag_zero.item()),
                lam_align=current_lam_align,
                lam_ent=current_lam_ent,
                lam_tv=current_lam_tv,
                lam_lag_sup=current_lam_lag_sup,
                lam_lag_bin=current_lam_lag_binary_aux,
                lam_lag_posexp=current_lam_lag_pos_expected_aux,
                lam_lag_soft=current_lam_lag_soft_target_aux,
                lam_lag_zero=current_lam_lag_zero,
            )

        denom = max(n, 1)
        train_metrics = {
            "loss": tot_loss / denom,
            "pred": tot_pred / denom,
            "align": tot_align / denom,
            "ent": tot_ent / denom,
            "tv": tot_tv / denom,
            "lag_occ": tot_lag_occ / denom,
            "lag_pos": tot_lag_pos / denom,
            "lag_sup": tot_lag_sup / denom,
            "lag_bin": tot_lag_bin / denom,
            "lag_posexp": tot_lag_posexp / denom,
            "lag_soft": tot_lag_soft / denom,
            "lag_zero": tot_lag_zero / denom,
        }
        train_metrics.update(_finalize_delay_stats(train_delay_stats))
        val_metrics = eval_epoch_metrics(
            model,
            dl_va,
            device,
            current_lam_align,
            current_lam_ent,
            current_lam_tv,
            current_lam_lag_sup,
            current_lam_lag_binary_aux,
            current_lam_lag_pos_expected_aux,
            current_lam_lag_soft_target_aux,
            current_lam_lag_zero,
            align_loss_temp,
            lag_supervision_edge=str(lag_supervision_edge) if lag_supervision_edge is not None else None,
            lag_supervision_key=str(lag_supervision_key) if lag_supervision_key is not None else None,
            lag_class_weights=lag_class_weights,
            lag_occurrence_pos_weight=lag_occurrence_pos_weight,
            lag_lambda_occurrence=lag_lambda_occurrence,
            lag_lambda_positive=lag_lambda_positive,
            lag_binary_aux_gamma=lag_binary_aux_gamma,
            lag_soft_target_key=str(lag_soft_target_key) if lag_soft_target_key is not None else None,
            lag_zero_mode=lag_zero_mode,
        )
        train_loss = train_metrics["loss"]
        val = val_metrics["loss"]
        print(
            "Epoch %d: train_loss=%.6f val_loss=%.6f train_pred=%.6f train_align=%.6f train_ent=%.6f train_tv=%.6f train_lag_occ=%.6f train_lag_pos=%.6f train_lag_sup=%.6f train_lag_bin=%.6f train_lag_posexp=%.6f train_lag_soft=%.6f train_lag_zero=%.6f w_align=%.4f w_ent=%.4f w_tv=%.4f w_lag_sup=%.4f w_lag_bin=%.4f w_lag_posexp=%.4f w_lag_soft=%.4f w_lag_zero=%.4f val_stage12_lag=%.3f val_stage12_peak=%.3f H=%d"
            % (
                epoch,
                train_loss,
                val,
                train_metrics["pred"],
                train_metrics["align"],
                train_metrics["ent"],
                train_metrics["tv"],
                train_metrics["lag_occ"],
                train_metrics["lag_pos"],
                train_metrics["lag_sup"],
                train_metrics["lag_bin"],
                train_metrics["lag_posexp"],
                train_metrics["lag_soft"],
                train_metrics["lag_zero"],
                current_lam_align,
                current_lam_ent,
                current_lam_tv,
                current_lam_lag_sup,
                current_lam_lag_binary_aux,
                current_lam_lag_pos_expected_aux,
                current_lam_lag_soft_target_aux,
                current_lam_lag_zero,
                val_metrics["stage1_to_stage2_expected_lag"],
                val_metrics["stage1_to_stage2_peak_prob"],
                int(cfg["data"]["H"]),
            )
        )
        logger.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val,
                "train_pred": train_metrics["pred"],
                "train_align": train_metrics["align"],
                "train_ent": train_metrics["ent"],
                "train_tv": train_metrics["tv"],
                "train_lag_occ": train_metrics["lag_occ"],
                "train_lag_pos": train_metrics["lag_pos"],
                "train_lag_sup": train_metrics["lag_sup"],
                "train_lag_bin": train_metrics["lag_bin"],
                "train_lag_posexp": train_metrics["lag_posexp"],
                "train_lag_soft": train_metrics["lag_soft"],
                "train_lag_zero": train_metrics["lag_zero"],
                "current_lambda_align": current_lam_align,
                "current_lambda_ent": current_lam_ent,
                "current_lambda_tv": current_lam_tv,
                "current_lambda_lag_supervision": current_lam_lag_sup,
                "current_lambda_lag_binary_aux": current_lam_lag_binary_aux,
                "current_lambda_lag_pos_expected_aux": current_lam_lag_pos_expected_aux,
                "current_lambda_lag_soft_target_aux": current_lam_lag_soft_target_aux,
                "current_lambda_lag_zero": current_lam_lag_zero,
                "lambda_lag_occurrence": lag_lambda_occurrence,
                "lambda_lag_positive": lag_lambda_positive,
                "lambda_lag_binary_aux": lam_lag_binary_aux_target,
                "lambda_lag_pos_expected_aux": lam_lag_pos_expected_aux_target,
                "lambda_lag_soft_target_aux": lam_lag_soft_target_aux_target,
                "lambda_lag_zero": lam_lag_zero_target,
                "lag_binary_aux_gamma": lag_binary_aux_gamma,
                "lag_binary_aux_pos_weight": None if lag_binary_aux_pos_weight is None else float(lag_binary_aux_pos_weight),
                "lag_soft_target_key": lag_soft_target_key,
                "lag_zero_mode": lag_zero_mode,
                "val_pred": val_metrics["pred"],
                "val_align": val_metrics["align"],
                "val_ent": val_metrics["ent"],
                "val_tv": val_metrics["tv"],
                "val_lag_occ": val_metrics["lag_occ"],
                "val_lag_pos": val_metrics["lag_pos"],
                "val_lag_sup": val_metrics["lag_sup"],
                "val_lag_bin": val_metrics["lag_bin"],
                "val_lag_posexp": val_metrics["lag_posexp"],
                "val_lag_soft": val_metrics["lag_soft"],
                "val_lag_zero": val_metrics["lag_zero"],
                "train_feed_to_stage1_expected_lag": train_metrics["feed_to_stage1_expected_lag"],
                "train_feed_to_stage1_peak_prob": train_metrics["feed_to_stage1_peak_prob"],
                "train_stage1_to_stage2_expected_lag": train_metrics["stage1_to_stage2_expected_lag"],
                "train_stage1_to_stage2_peak_prob": train_metrics["stage1_to_stage2_peak_prob"],
                "train_stage2_to_stage3_expected_lag": train_metrics["stage2_to_stage3_expected_lag"],
                "train_stage2_to_stage3_peak_prob": train_metrics["stage2_to_stage3_peak_prob"],
                "val_feed_to_stage1_expected_lag": val_metrics["feed_to_stage1_expected_lag"],
                "val_feed_to_stage1_peak_prob": val_metrics["feed_to_stage1_peak_prob"],
                "val_stage1_to_stage2_expected_lag": val_metrics["stage1_to_stage2_expected_lag"],
                "val_stage1_to_stage2_peak_prob": val_metrics["stage1_to_stage2_peak_prob"],
                "val_stage2_to_stage3_expected_lag": val_metrics["stage2_to_stage3_expected_lag"],
                "val_stage2_to_stage3_peak_prob": val_metrics["stage2_to_stage3_peak_prob"],
                "checkpoint_metric_name": checkpoint_metric,
                "checkpoint_metric_value": _checkpoint_metric_value(checkpoint_metric, val, val_metrics),
                "H": int(cfg["data"]["H"]),
            }
        )

        checkpoint_value = _checkpoint_metric_value(checkpoint_metric, val, val_metrics)
        checkpoint_eligible = (not checkpoint_after_warmup_only) or (epoch > pred_warmup_epochs)
        if checkpoint_eligible and checkpoint_value < best - 1e-6:
            best, best_epoch = checkpoint_value, epoch
            torch.save({"model": model.state_dict(), "cfg": cfg}, ckpt_path)

    print(f"[Done] best_{checkpoint_metric}={best:.6f} at epoch={best_epoch}")
    print(f"Completed epochs: {int(cfg['train']['epochs'])}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Scaler: {cfg['logging']['scaler_path']}")

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    output_dir = cfg["logging"].get("output_dir", "outputs")
    test_res = eval_test(
        model,
        dl_te,
        device,
        prepared.scaler_y,
        output_dir,
        test_input_timestamps,
        test_target_timestamps,
        int(cfg["data"]["collection_interval_min"]),
        target_cols=prepared.target_cols,
    )
    test_res["best_checkpoint_metric"] = checkpoint_metric
    test_res["best_checkpoint_value"] = float(best)
    test_res["best_epoch"] = int(best_epoch)
    test_res["epochs_ran"] = int(cfg["train"]["epochs"])
    test_res["seed"] = int(cfg.get("seed", 42))
    print("=== Test Metrics ===")
    for k, v in test_res.items():
        print(f"{k}: {v}")
    with open(Path(output_dir) / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_res, f, ensure_ascii=False, indent=2)
    print(f"Saved: {Path(output_dir) / 'test_pred_vs_true.csv'}")
    print(f"Saved: {Path(output_dir) / 'test_pred_vs_true_scaled.csv'}")
    print(f"Saved: {Path(output_dir) / 'test_delay_pi.npy'}")
    print(f"Saved: {Path(output_dir) / 'test_delay_laststep_pi.npy'}")
    print(f"Saved: {Path(output_dir) / 'test_delay_estimates.csv'}")
    print(f"Saved: {Path(output_dir) / 'test_metrics.json'}")

if __name__ == "__main__":
    main()
