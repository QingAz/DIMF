from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_lag_distribution_heatmap(
    output_dir: str | Path,
    gt_pi: np.ndarray,
    pred_pi: np.ndarray,
    max_rows: int = 200,
    filename: str = "lag_distribution_heatmap.png",
) -> Path:
    plt = _mpl()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n = min(int(max_rows), gt_pi.shape[0], pred_pi.shape[0])
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True, sharey=True)
    axes[0].imshow(gt_pi[:n].T, aspect="auto", origin="lower", interpolation="nearest")
    axes[0].set_title("Ground truth lag distribution")
    axes[0].set_ylabel("lag")
    im = axes[1].imshow(pred_pi[:n].T, aspect="auto", origin="lower", interpolation="nearest")
    axes[1].set_title("Predicted lag distribution")
    axes[1].set_xlabel("sample")
    axes[1].set_ylabel("lag")
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    path = output_dir / filename
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_expected_lag_curve(
    output_dir: str | Path,
    gt_pi: np.ndarray,
    pred_pi: np.ndarray,
    filename: str = "expected_lag_curve.png",
) -> Path:
    plt = _mpl()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lag_axis = np.arange(gt_pi.shape[-1], dtype=np.float64)
    gt = (gt_pi * lag_axis[None, :]).sum(axis=-1)
    pred = (pred_pi * lag_axis[None, :]).sum(axis=-1)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(gt, label="gt", linewidth=1.4)
    ax.plot(pred, label="pred", linewidth=1.2)
    ax.set_xlabel("sample")
    ax.set_ylabel("expected lag")
    ax.legend(frameon=False)
    path = output_dir / filename
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_viterbi_lag_curve(
    output_dir: str | Path,
    gt_pi: np.ndarray,
    pred_pi: np.ndarray,
    viterbi_path: np.ndarray,
    filename: str = "viterbi_lag_curve.png",
) -> Path:
    plt = _mpl()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lag_axis = np.arange(gt_pi.shape[-1], dtype=np.float64)
    gt = (gt_pi * lag_axis[None, :]).sum(axis=-1)
    pred = (pred_pi * lag_axis[None, :]).sum(axis=-1)
    viterbi_path = np.asarray(viterbi_path, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(gt, label="gt", linewidth=1.4)
    ax.plot(pred, label="pred_expected", linewidth=0.9, alpha=0.45)
    ax.plot(viterbi_path, label="viterbi", linewidth=1.2)
    ax.set_xlabel("sample")
    ax.set_ylabel("lag")
    ax.legend(frameon=False)
    path = output_dir / filename
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_by_shape_bar_chart(
    output_dir: str | Path,
    by_shape_metrics: pd.DataFrame,
    filename: str = "by_shape_lag_metrics.png",
) -> Optional[Path]:
    if by_shape_metrics.empty or "shape_type" not in by_shape_metrics:
        return None
    plt = _mpl()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = by_shape_metrics.copy()
    x = np.arange(len(frame))
    width = 0.4
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.bar(x - width / 2, frame["expected_lag_mae_all"], width=width, label="expected MAE")
    ax.bar(x + width / 2, frame["soft_js"], width=width, label="JS")
    ax.set_xticks(x)
    ax.set_xticklabels(frame["shape_type"].astype(str), rotation=30, ha="right")
    ax.legend(frameon=False)
    path = output_dir / filename
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_no_lag_false_alarm_plot(
    output_dir: str | Path,
    pred_pi: np.ndarray,
    lag_flag: np.ndarray,
    filename: str = "no_lag_false_alarm.png",
) -> Optional[Path]:
    no_lag = np.asarray(lag_flag).astype(int) == 0
    if not np.any(no_lag):
        return None
    plt = _mpl()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    p_pos = 1.0 - pred_pi[no_lag, 0]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(p_pos, bins=30, color="#4c78a8", alpha=0.85)
    ax.set_xlabel("P(lag > 0) on no-lag samples")
    ax.set_ylabel("count")
    path = output_dir / filename
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_selected_feature_lag_heatmap(
    output_dir: str | Path,
    pi_lag: np.ndarray,
    feature_indices: np.ndarray,
    sample_index: int = 0,
    filename: str = "selected_feature_lag_heatmap.png",
) -> Optional[Path]:
    if pi_lag.ndim != 3 or pi_lag.shape[0] == 0:
        return None
    plt = _mpl()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_indices = np.asarray(feature_indices, dtype=int)
    feature_indices = feature_indices[(feature_indices >= 0) & (feature_indices < pi_lag.shape[1])]
    if feature_indices.size == 0:
        return None
    arr = pi_lag[int(np.clip(sample_index, 0, pi_lag.shape[0] - 1)), feature_indices, :]
    fig, ax = plt.subplots(figsize=(8, max(2.5, 0.35 * len(feature_indices))))
    im = ax.imshow(arr, aspect="auto", origin="lower", interpolation="nearest")
    ax.set_xlabel("lag")
    ax.set_ylabel("source feature")
    ax.set_yticks(np.arange(len(feature_indices)))
    ax.set_yticklabels([str(int(i)) for i in feature_indices])
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    path = output_dir / filename
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
