#!/usr/bin/env python3

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataprocess import load_and_prepare
from src.data.dataset import MultistageWindowDataset, WindowSpec
from src.models.dimf import DIMF
from src.utils.seed import set_seed
from train import load_config

TIME_FORMAT = "%Y-%m-%d %H:%M"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a split-aware detection audit for a selected checkpoint and export segment diagnostics."
    )
    parser.add_argument("--config", type=Path, required=True, help="Training config used for the run")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for audit outputs")
    parser.add_argument("--edge", default="stage1_to_stage2", help="Lag edge name")
    parser.add_argument(
        "--target-key",
        default="stage1_to_stage2_in_block_gt",
        help="Detection target key used during training",
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional checkpoint override")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Torch device")
    parser.add_argument(
        "--context-segments",
        type=int,
        default=1,
        help="How many neighboring raw segments to include on each side in the worst-segment panels",
    )
    parser.add_argument("--low-auroc-k", type=int, default=3, help="Number of lowest-AUROC test segments to plot")
    parser.add_argument("--high-gap-k", type=int, default=2, help="Number of largest gap test segments to plot")
    return parser.parse_args()


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(str(path)))


def _timestamp(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series).dt.strftime(TIME_FORMAT)


def _build_model(cfg: Dict[str, Any], prepared, device: torch.device) -> DIMF:
    return DIMF(
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
        lag_head_mode=str(cfg["model"].get("lag_head_mode", "softmax")),
    ).to(device)


def _make_eval_loaders(cfg: Dict[str, Any], prepared) -> Dict[str, Dict[str, Any]]:
    spec = WindowSpec(L=int(cfg["data"]["L"]), H=int(cfg["data"]["H"]))
    batch_size = int(cfg["train"]["batch_size"])

    split_payloads = {
        "train": {
            "X_groups": prepared.X_groups_train,
            "y": prepared.y_train,
            "indices": prepared.sample_indices_train,
            "extra_targets": prepared.extra_targets_train,
            "timestamps": prepared.timestamps_train,
        },
        "val": {
            "X_groups": prepared.X_groups_val,
            "y": prepared.y_val,
            "indices": prepared.sample_indices_val,
            "extra_targets": prepared.extra_targets_val,
            "timestamps": prepared.timestamps_val,
        },
        "test": {
            "X_groups": prepared.X_groups_test,
            "y": prepared.y_test,
            "indices": prepared.sample_indices_test,
            "extra_targets": prepared.extra_targets_test,
            "timestamps": prepared.timestamps_test,
        },
    }

    out: Dict[str, Dict[str, Any]] = {}
    for split_name, payload in split_payloads.items():
        ds = MultistageWindowDataset(
            payload["X_groups"],
            payload["y"],
            spec,
            indices=payload["indices"],
            extra_targets=payload["extra_targets"],
        )
        out[split_name] = {
            "dataset": ds,
            "loader": DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False),
            "sample_timestamps": np.asarray(payload["timestamps"])[ds.indices],
        }
    return out


def _load_selected_checkpoint(args: argparse.Namespace, cfg: Dict[str, Any]) -> Path:
    if args.checkpoint is not None:
        return _absolute_path(args.checkpoint)

    output_dir = Path(cfg["logging"]["output_dir"])
    output_dir = output_dir if output_dir.is_absolute() else ROOT / output_dir

    test_metrics_path = output_dir / "test_metrics.json"
    if test_metrics_path.exists():
        metrics = json.loads(test_metrics_path.read_text(encoding="utf-8"))
        ckpt_path = metrics.get("eval_checkpoint_path")
        if ckpt_path:
            return _absolute_path(Path(ckpt_path) if Path(ckpt_path).is_absolute() else ROOT / ckpt_path)

    selection_path = output_dir / "checkpoint_alignment_selection.json"
    if selection_path.exists():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        ckpt_path = selection.get("selected", {}).get("checkpoint_path")
        if ckpt_path:
            return _absolute_path(Path(ckpt_path) if Path(ckpt_path).is_absolute() else ROOT / ckpt_path)

    fallback = Path(cfg["logging"]["ckpt_path"])
    if not fallback.is_absolute():
        fallback = ROOT / fallback
    return _absolute_path(fallback)


def _raw_split_frame(cfg: Dict[str, Any], split_name: str) -> pd.DataFrame:
    data_cfg = cfg["data"]
    raw_path = Path(data_cfg["csv_path"])
    if not raw_path.is_absolute():
        raw_path = ROOT / raw_path
    raw = pd.read_csv(raw_path)
    raw[data_cfg["time_col"]] = pd.to_datetime(raw[data_cfg["time_col"]])
    split_col = str(data_cfg.get("split_col", "split"))
    out = raw.loc[raw[split_col] == split_name].sort_values(data_cfg["time_col"]).reset_index(drop=True).copy()
    out["timestamp"] = _timestamp(out[data_cfg["time_col"]])
    if "inject_flag" in out.columns:
        out["in_block"] = out["inject_flag"].fillna(0).astype(int)
    else:
        out["in_block"] = out.get("lag_gt", 0)
        out["in_block"] = out["in_block"].fillna(0).astype(int).gt(0).astype(int)
    if "segment_id" not in out.columns:
        out["segment_id"] = np.arange(len(out), dtype=np.int64)
    out["segment_id"] = out["segment_id"].fillna(-1).astype(int)
    if "segment_dmax_gt" in out.columns:
        out["dmax"] = out["segment_dmax_gt"].fillna(0).astype(int)
    elif "bump_dmax_gt" in out.columns:
        out["dmax"] = out["bump_dmax_gt"].fillna(0).astype(int)
    else:
        out["dmax"] = 0
    out["lag_gt"] = out.get("lag_gt", 0).fillna(0).astype(int)
    return out[
        [
            "timestamp",
            "segment_id",
            "in_block",
            "dmax",
            "lag_gt",
        ]
    ].copy()


@torch.no_grad()
def _collect_split_scores(
    model: DIMF,
    loader: DataLoader,
    device: torch.device,
    sample_timestamps: np.ndarray,
    raw_lookup: pd.DataFrame,
    split_name: str,
    edge: str,
) -> pd.DataFrame:
    model.eval()
    probs_rows: List[np.ndarray] = []
    for X, _ in loader:
        X = {k: v.to(device) for k, v in X.items()}
        _, pi = model(X)
        if edge not in pi:
            raise ValueError(f"Missing edge {edge!r} in model outputs")
        arr = pi[edge]
        arr_last = arr[:, -1, :] if arr.dim() == 3 else arr
        probs_rows.append(arr_last.detach().cpu().numpy())

    if not probs_rows:
        raise ValueError(f"No batches found for split {split_name}")
    probs = np.concatenate(probs_rows, axis=0)
    if probs.shape[0] != len(sample_timestamps):
        raise ValueError(
            f"Prediction length mismatch for split {split_name}: "
            f"{probs.shape[0]} scores vs {len(sample_timestamps)} timestamps"
        )

    lag_axis = np.arange(probs.shape[1], dtype=np.float64)
    pred = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(sample_timestamps).strftime(TIME_FORMAT),
            "p": 1.0 - probs[:, 0],
            "pred_expected_lag": (probs * lag_axis[None, :]).sum(axis=1),
            "pred_argmax_lag": probs.argmax(axis=1).astype(int),
        }
    )
    joined = raw_lookup.merge(pred, on="timestamp", how="inner")
    joined.insert(0, "split", split_name)
    joined["is_positive"] = joined["lag_gt"].gt(0).astype(int)
    joined = joined.sort_values(["timestamp", "segment_id"]).reset_index(drop=True)
    joined["segment_length"] = joined.groupby("segment_id")["timestamp"].transform("size")
    joined["segment_index"] = joined.groupby("segment_id").cumcount()
    denom = np.maximum(joined["segment_length"].to_numpy(dtype=np.float64) - 1.0, 1.0)
    joined["segment_rel_pos"] = joined["segment_index"].to_numpy(dtype=np.float64) / denom
    return joined[
        [
            "split",
            "timestamp",
            "segment_id",
            "segment_length",
            "segment_index",
            "segment_rel_pos",
            "in_block",
            "dmax",
            "lag_gt",
            "is_positive",
            "p",
            "pred_expected_lag",
            "pred_argmax_lag",
        ]
    ].copy()


def _pr_curve(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(-scores)
    scores_sorted = scores[order]
    labels_sorted = labels[order].astype(bool)
    tp = 0
    fp = 0
    fn = int(labels_sorted.sum())
    precision: List[float] = []
    recall: List[float] = []
    thresholds: List[float] = []
    last_score = None
    for score, label in zip(scores_sorted, labels_sorted):
        if last_score is None or score != last_score:
            if last_score is not None:
                precision.append(tp / (tp + fp) if tp + fp else 0.0)
                recall.append(tp / (tp + fn) if tp + fn else 0.0)
                thresholds.append(float(last_score))
            last_score = float(score)
        if label:
            tp += 1
            fn -= 1
        else:
            fp += 1
    if last_score is not None:
        precision.append(tp / (tp + fp) if tp + fp else 0.0)
        recall.append(tp / (tp + fn) if tp + fn else 0.0)
        thresholds.append(float(last_score))
    return (
        np.asarray(precision, dtype=np.float64),
        np.asarray(recall, dtype=np.float64),
        np.asarray(thresholds, dtype=np.float64),
    )


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels_bool = labels.astype(bool)
    n_pos = int(labels_bool.sum())
    n_neg = int((~labels_bool).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = pd.Series(scores).rank(method="average").to_numpy(dtype=np.float64)
    pos_rank_sum = float(ranks[labels_bool].sum())
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)


def _binary_metrics(scores: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    labels = labels.astype(np.int64)
    scores = scores.astype(np.float64)
    precision, recall, thresholds = _pr_curve(scores, labels)
    if precision.size:
        f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
        best_idx = int(np.argmax(f1))
        best_threshold = float(thresholds[best_idx])
        best_precision = float(precision[best_idx])
        best_recall = float(recall[best_idx])
        best_f1 = float(f1[best_idx])
        order = np.argsort(recall)
        auprc = float(np.trapz(precision[order], recall[order]))
    else:
        best_threshold = float("inf")
        best_precision = 0.0
        best_recall = 0.0
        best_f1 = 0.0
        auprc = 0.0

    pred_pos = scores >= best_threshold
    labels_bool = labels.astype(bool)
    neg_mask = ~labels_bool
    far = float(np.logical_and(pred_pos, neg_mask).sum() / max(int(neg_mask.sum()), 1))
    pos_scores = scores[labels_bool]
    neg_scores = scores[neg_mask]
    return {
        "auprc": auprc,
        "auroc": _auroc(scores, labels),
        "best_threshold": best_threshold,
        "best_precision": best_precision,
        "best_recall": best_recall,
        "best_f1": best_f1,
        "pred_positive_ratio": float(pred_pos.mean()) if pred_pos.size else 0.0,
        "far": far,
        "mean_score_positive": float(pos_scores.mean()) if pos_scores.size else float("nan"),
        "mean_score_negative": float(neg_scores.mean()) if neg_scores.size else float("nan"),
        "score_margin": float(pos_scores.mean() - neg_scores.mean()) if pos_scores.size and neg_scores.size else float("nan"),
    }


def _segment_one_vs_opposite(split_frame: pd.DataFrame) -> pd.DataFrame:
    split_frame = split_frame.sort_values(["timestamp", "segment_id"]).reset_index(drop=True)
    pos_pool = split_frame.loc[split_frame["in_block"] == 1].copy()
    neg_pool = split_frame.loc[split_frame["in_block"] == 0].copy()
    rows: List[Dict[str, Any]] = []

    for segment_id, group in split_frame.groupby("segment_id", sort=False):
        segment = group.sort_values("timestamp").copy()
        segment_label = int(segment["in_block"].mode().iloc[0])
        if segment_label == 1:
            eval_pos = segment
            eval_neg = neg_pool
        else:
            eval_pos = pos_pool
            eval_neg = segment

        scores = np.concatenate(
            [
                eval_pos["p"].to_numpy(dtype=np.float64),
                eval_neg["p"].to_numpy(dtype=np.float64),
            ]
        )
        labels = np.concatenate(
            [
                np.ones(len(eval_pos), dtype=np.int64),
                np.zeros(len(eval_neg), dtype=np.int64),
            ]
        )
        metrics = _binary_metrics(scores, labels)
        rows.append(
            {
                "split": str(segment["split"].iloc[0]),
                "segment_id": int(segment_id),
                "segment_label": segment_label,
                "segment_start_time": str(segment["timestamp"].iloc[0]),
                "segment_end_time": str(segment["timestamp"].iloc[-1]),
                "segment_length": int(len(segment)),
                "n_in_block": int(len(eval_pos)),
                "n_out_block": int(len(eval_neg)),
                "segment_in_block_rows": int(segment["in_block"].sum()),
                "segment_out_block_rows": int((segment["in_block"] == 0).sum()),
                "n_lag_positive": int(segment["is_positive"].sum()),
                "dmax": int(segment["dmax"].mode().iloc[0]) if not segment["dmax"].empty else 0,
                "segment_p_mean": float(segment["p"].mean()),
                "segment_p_std": float(segment["p"].std(ddof=0)),
                "segment_p_min": float(segment["p"].min()),
                "segment_p_max": float(segment["p"].max()),
                "p_in_block_mean": float(eval_pos["p"].mean()) if not eval_pos.empty else float("nan"),
                "p_out_block_mean": float(eval_neg["p"].mean()) if not eval_neg.empty else float("nan"),
                **metrics,
            }
        )
    out = pd.DataFrame(rows)
    out["p_out_minus_p_in"] = out["p_out_block_mean"] - out["p_in_block_mean"]
    return out.sort_values(["split", "segment_start_time", "segment_id"]).reset_index(drop=True)


def _split_summaries(samples: pd.DataFrame, segment_audit: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for split_name, split_frame in samples.groupby("split", sort=False):
        sample_metrics = _binary_metrics(
            split_frame["p"].to_numpy(dtype=np.float64),
            split_frame["in_block"].to_numpy(dtype=np.int64),
        )
        seg_view = segment_audit.loc[segment_audit["split"] == split_name].copy()
        segment_metrics = _binary_metrics(
            seg_view["segment_p_mean"].to_numpy(dtype=np.float64),
            seg_view["segment_label"].to_numpy(dtype=np.int64),
        )
        rows.append(
            {
                "split": split_name,
                "n_samples": int(len(split_frame)),
                "n_segments": int(seg_view["segment_id"].nunique()),
                "n_positive_segments": int(seg_view["segment_label"].sum()),
                "n_negative_segments": int((seg_view["segment_label"] == 0).sum()),
                "row_block_auroc": sample_metrics["auroc"],
                "row_block_auprc": sample_metrics["auprc"],
                "row_best_f1": sample_metrics["best_f1"],
                "row_best_threshold": sample_metrics["best_threshold"],
                "row_pred_positive_ratio": sample_metrics["pred_positive_ratio"],
                "row_far": sample_metrics["far"],
                "row_p_in_block": sample_metrics["mean_score_positive"],
                "row_p_out_block": sample_metrics["mean_score_negative"],
                "row_score_margin": sample_metrics["score_margin"],
                "segment_block_auroc": segment_metrics["auroc"],
                "segment_block_auprc": segment_metrics["auprc"],
                "segment_best_f1": segment_metrics["best_f1"],
                "segment_best_threshold": segment_metrics["best_threshold"],
                "segment_pred_positive_ratio": segment_metrics["pred_positive_ratio"],
                "segment_far": segment_metrics["far"],
                "segment_p_in_block": segment_metrics["mean_score_positive"],
                "segment_p_out_block": segment_metrics["mean_score_negative"],
                "segment_score_margin": segment_metrics["score_margin"],
            }
        )
    return pd.DataFrame(rows)


def _select_worst_segments(segment_audit: pd.DataFrame, low_auroc_k: int, high_gap_k: int) -> pd.DataFrame:
    test_view = segment_audit.loc[segment_audit["split"] == "test"].copy()
    if test_view.empty:
        return pd.DataFrame(columns=segment_audit.columns)

    lowest = test_view.sort_values(["auroc", "p_out_minus_p_in"], ascending=[True, False]).head(max(low_auroc_k, 0))
    largest_gap = test_view.sort_values(["p_out_minus_p_in", "auroc"], ascending=[False, True]).head(max(high_gap_k, 0))
    combined = pd.concat([lowest, largest_gap], ignore_index=True)
    combined = combined.drop_duplicates(subset=["split", "segment_id"], keep="first").reset_index(drop=True)
    combined["selection_reason"] = ""
    low_keys = {(row.split, row.segment_id) for row in lowest.itertuples()}
    gap_keys = {(row.split, row.segment_id) for row in largest_gap.itertuples()}

    reasons: List[str] = []
    for row in combined.itertuples():
        tags: List[str] = []
        key = (row.split, row.segment_id)
        if key in low_keys:
            tags.append("low_auroc")
        if key in gap_keys:
            tags.append("high_gap")
        reasons.append("+".join(tags))
    combined["selection_reason"] = reasons
    return combined


def _tick_positions(n: int, max_ticks: int = 6) -> Iterable[int]:
    if n <= 1:
        return [0]
    if n <= max_ticks:
        return list(range(n))
    return np.linspace(0, n - 1, num=max_ticks, dtype=int).tolist()


def _render_worst_panels(
    samples: pd.DataFrame,
    worst_segments: pd.DataFrame,
    output_dir: Path,
    context_segments: int,
) -> None:
    panels_dir = output_dir / "worst_test_segment_panels"
    panels_dir.mkdir(parents=True, exist_ok=True)
    if worst_segments.empty:
        return

    test_samples = samples.loc[samples["split"] == "test"].copy()
    test_samples["timestamp_dt"] = pd.to_datetime(test_samples["timestamp"])
    segment_order = (
        test_samples.groupby("segment_id", as_index=False)["timestamp_dt"]
        .min()
        .sort_values("timestamp_dt")["segment_id"]
        .astype(int)
        .tolist()
    )
    segment_pos = {segment_id: idx for idx, segment_id in enumerate(segment_order)}

    n_panels = len(worst_segments)
    fig, axes = plt.subplots(n_panels, 1, figsize=(15, max(3.4 * n_panels, 4.0)), sharey=True)
    axes_arr = np.atleast_1d(axes)

    for ax, row in zip(axes_arr, worst_segments.itertuples()):
        center = segment_pos[int(row.segment_id)]
        left_idx = max(0, center - max(context_segments, 0))
        right_idx = min(len(segment_order) - 1, center + max(context_segments, 0))
        context_ids = segment_order[left_idx : right_idx + 1]
        view = test_samples.loc[test_samples["segment_id"].isin(context_ids)].copy()
        view = view.sort_values("timestamp_dt").reset_index(drop=True)
        x = np.arange(len(view), dtype=np.int64)

        current_mask = view["segment_id"].eq(int(row.segment_id)).to_numpy()
        block_mask = view["in_block"].eq(1).to_numpy()
        lag_mask = view["is_positive"].eq(1).to_numpy()

        ax.plot(x, view["p"], color="#2563eb", linewidth=1.8, label="p_t")
        ax.axhline(float(row.best_threshold), color="#dc2626", linewidth=1.2, linestyle="--", alpha=0.8, label="best-F1 thr")
        ax.fill_between(x, 0.0, 1.0, where=block_mask, color="#fde68a", alpha=0.18, step="pre", label="true in_block")
        highlight_color = "#f97316" if int(row.segment_label) == 1 else "#6b7280"
        ax.fill_between(x, 0.0, 1.0, where=current_mask, color=highlight_color, alpha=0.16, step="pre", label="selected segment")
        if lag_mask.any():
            ax.scatter(x[lag_mask], view.loc[lag_mask, "p"], s=20, color="#111827", zorder=4, label="lag>0 rows")

        tick_idx = list(dict.fromkeys(int(v) for v in _tick_positions(len(view))))
        tick_labels = [view["timestamp_dt"].iloc[i].strftime("%m-%d %H:%M") for i in tick_idx]
        ax.set_xticks(tick_idx)
        ax.set_xticklabels(tick_labels, rotation=0, fontsize=8)
        ax.set_ylim(-0.03, 1.03)
        ax.set_ylabel("p_t")
        ax.set_title(
            (
                f"test segment {int(row.segment_id)} | label={int(row.segment_label)} dmax={int(row.dmax)} "
                f"| AUROC={row.auroc:.3f} AUPRC={row.auprc:.3f} "
                f"| p_in={row.p_in_block_mean:.3f} p_out={row.p_out_block_mean:.3f} "
                f"| {row.selection_reason}"
            ),
            fontsize=10,
        )

    handles, labels = axes_arr[0].get_legend_handles_labels()
    unique = {}
    for handle, label in zip(handles, labels):
        if label not in unique:
            unique[label] = handle
    fig.legend(unique.values(), unique.keys(), loc="upper center", ncol=min(4, len(unique)), frameon=False, bbox_to_anchor=(0.5, 0.995))
    axes_arr[-1].set_xlabel("sample index within local context")
    fig.suptitle("G1' Worst Test Segments: Detection Score Panels", y=1.01, fontsize=14)
    fig.tight_layout()
    fig.savefig(panels_dir / "worst_test_segments.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    for row in worst_segments.itertuples():
        center = segment_pos[int(row.segment_id)]
        left_idx = max(0, center - max(context_segments, 0))
        right_idx = min(len(segment_order) - 1, center + max(context_segments, 0))
        context_ids = segment_order[left_idx : right_idx + 1]
        view = test_samples.loc[test_samples["segment_id"].isin(context_ids)].copy()
        view = view.sort_values("timestamp_dt").reset_index(drop=True)
        x = np.arange(len(view), dtype=np.int64)
        current_mask = view["segment_id"].eq(int(row.segment_id)).to_numpy()
        block_mask = view["in_block"].eq(1).to_numpy()
        lag_mask = view["is_positive"].eq(1).to_numpy()

        fig_single, ax_single = plt.subplots(figsize=(14, 3.8))
        ax_single.plot(x, view["p"], color="#2563eb", linewidth=1.9, label="p_t")
        ax_single.axhline(float(row.best_threshold), color="#dc2626", linewidth=1.2, linestyle="--", alpha=0.8, label="best-F1 thr")
        ax_single.fill_between(x, 0.0, 1.0, where=block_mask, color="#fde68a", alpha=0.18, step="pre", label="true in_block")
        highlight_color = "#f97316" if int(row.segment_label) == 1 else "#6b7280"
        ax_single.fill_between(x, 0.0, 1.0, where=current_mask, color=highlight_color, alpha=0.16, step="pre", label="selected segment")
        if lag_mask.any():
            ax_single.scatter(x[lag_mask], view.loc[lag_mask, "p"], s=20, color="#111827", zorder=4, label="lag>0 rows")
        tick_idx = list(dict.fromkeys(int(v) for v in _tick_positions(len(view))))
        tick_labels = [view["timestamp_dt"].iloc[i].strftime("%m-%d %H:%M") for i in tick_idx]
        ax_single.set_xticks(tick_idx)
        ax_single.set_xticklabels(tick_labels, fontsize=8)
        ax_single.set_ylim(-0.03, 1.03)
        ax_single.set_xlabel("sample index within local context")
        ax_single.set_ylabel("p_t")
        ax_single.set_title(
            (
                f"test segment {int(row.segment_id)} | label={int(row.segment_label)} dmax={int(row.dmax)} "
                f"| AUROC={row.auroc:.3f} | p_in={row.p_in_block_mean:.3f} p_out={row.p_out_block_mean:.3f}"
            ),
            fontsize=10,
        )
        handles, labels = ax_single.get_legend_handles_labels()
        unique = {}
        for handle, label in zip(handles, labels):
            if label not in unique:
                unique[label] = handle
        ax_single.legend(unique.values(), unique.keys(), loc="upper right", frameon=False)
        fig_single.tight_layout()
        fig_single.savefig(panels_dir / f"segment_{int(row.segment_id)}.png", dpi=180, bbox_inches="tight")
        plt.close(fig_single)


def main() -> None:
    args = parse_args()
    config_path = _absolute_path(args.config)
    output_dir = _absolute_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(str(config_path))
    set_seed(int(cfg.get("seed", 42)))
    device = torch.device(args.device)

    prepared, _ = load_and_prepare(
        csv_path=cfg["data"]["csv_path"],
        time_col=cfg["data"]["time_col"],
        target_col=cfg["data"]["target_col"],
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
        sample_keep_col=(
            str(cfg["data"]["sample_keep_col"])
            if cfg["data"].get("sample_keep_col") is not None
            else None
        ),
        respect_existing_segment_id=bool(cfg["data"].get("respect_existing_segment_id", False)),
    )
    loaders = _make_eval_loaders(cfg, prepared)
    model = _build_model(cfg, prepared, device)
    checkpoint_path = _load_selected_checkpoint(args, cfg)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])

    sample_frames: List[pd.DataFrame] = []
    for split_name in ["train", "val", "test"]:
        raw_lookup = _raw_split_frame(cfg, split_name)
        split_frame = _collect_split_scores(
            model=model,
            loader=loaders[split_name]["loader"],
            device=device,
            sample_timestamps=loaders[split_name]["sample_timestamps"],
            raw_lookup=raw_lookup,
            split_name=split_name,
            edge=args.edge,
        )
        sample_frames.append(split_frame)

    samples = pd.concat(sample_frames, ignore_index=True)
    samples.to_csv(output_dir / "sample_detection_scores.csv", index=False)

    segment_audit_parts = []
    for split_name, split_frame in samples.groupby("split", sort=False):
        segment_audit_parts.append(_segment_one_vs_opposite(split_frame))
    segment_audit = pd.concat(segment_audit_parts, ignore_index=True)
    segment_audit.to_csv(output_dir / "segment_detection_audit.csv", index=False)

    split_summary = _split_summaries(samples, segment_audit)
    split_summary.to_csv(output_dir / "split_detection_summary.csv", index=False)

    worst_segments = _select_worst_segments(
        segment_audit,
        low_auroc_k=int(args.low_auroc_k),
        high_gap_k=int(args.high_gap_k),
    )
    worst_segments.to_csv(output_dir / "worst_test_segments.csv", index=False)
    _render_worst_panels(
        samples=samples,
        worst_segments=worst_segments,
        output_dir=output_dir,
        context_segments=int(args.context_segments),
    )

    report = {
        "config": config_path.as_posix(),
        "checkpoint_path": checkpoint_path.as_posix(),
        "edge": args.edge,
        "target_key": args.target_key,
        "output_dir": output_dir.as_posix(),
        "n_samples": int(len(samples)),
        "n_segments": int(segment_audit["segment_id"].nunique()),
        "split_summary": split_summary.to_dict(orient="records"),
    }
    (output_dir / "segment_detection_audit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(split_summary.to_csv(index=False))
    if not worst_segments.empty:
        print(worst_segments[["split", "segment_id", "segment_label", "auroc", "p_in_block_mean", "p_out_block_mean", "selection_reason"]].to_csv(index=False))
    print(json.dumps({"checkpoint_path": checkpoint_path.as_posix()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
