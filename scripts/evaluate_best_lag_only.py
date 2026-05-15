#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_lag_grounded import (
    STDALagIdentifier,
    _edge_cfg,
    _history_steps,
    _load_existing_train_test_prepared,
    collect_lag_outputs,
    load_config,
    postprocess_lag_eval_for_eval,
)
import joblib
import matplotlib
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import MultistageWindowDataset, WindowSpec
from src.postprocess.viterbi_lag_decoder import viterbi_decode_lag


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the best lag-only decoder without y-target training.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


def _make_identifier(cfg: Dict[str, Any], prepared, edge_cfg: Dict[str, Any], device: torch.device) -> STDALagIdentifier:
    lag_cfg = cfg.get("lag_identifier", {})
    return STDALagIdentifier(
        d_source=prepared.group_dims[edge_cfg["source_stage"]],
        d_target=prepared.group_dims[edge_cfg["target_stage"]],
        max_lag=int(lag_cfg.get("max_lag", cfg.get("lag_injection", {}).get("max_lag", cfg["data"].get("L_max", 12)))),
        hidden_dim=int(lag_cfg.get("hidden_dim", 64)),
        num_layers=int(lag_cfg.get("num_layers", 2)),
        temperature=float(lag_cfg.get("lag_temperature", lag_cfg.get("temperature", 0.7))),
        use_temporal_decay=bool(lag_cfg.get("use_temporal_decay", True)),
        use_feature_attention=bool(lag_cfg.get("use_feature_attention", True)),
        dropout=float(lag_cfg.get("dropout", 0.0)),
        use_sequence_smoother=bool(lag_cfg.get("use_sequence_smoother", False)),
        sequence_smoother_hidden_dim=lag_cfg.get("sequence_smoother_hidden_dim"),
        sequence_smoother_layers=int(lag_cfg.get("sequence_smoother_layers", 1)),
        sequence_smoother_dropout=float(lag_cfg.get("sequence_smoother_dropout", 0.0)),
        sequence_smoother_residual_scale=float(lag_cfg.get("sequence_smoother_residual_scale", 0.5)),
        use_candidate_window_encoder=bool(lag_cfg.get("use_candidate_window_encoder", False)),
        lag_window_radius=int(lag_cfg.get("lag_window_radius", 2)),
        lag_window_hidden_dim=lag_cfg.get("lag_window_hidden_dim"),
        lag_window_mode=str(lag_cfg.get("lag_window_mode", "causal")),
        keep_old_point_identifier=bool(lag_cfg.get("keep_old_point_identifier", True)),
    ).to(device)


def _load_identifier(
    cfg: Dict[str, Any],
    prepared,
    edge_cfg: Dict[str, Any],
    checkpoint: Path,
    device: torch.device,
    use_sequence_smoother: bool | None = None,
) -> STDALagIdentifier:
    identifier = _make_identifier(cfg, prepared, edge_cfg, device)
    try:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location=device)
    identifier.load_state_dict(state)
    if use_sequence_smoother is not None:
        identifier.use_sequence_smoother = bool(use_sequence_smoother)
    identifier.eval()
    return identifier


def _mode_filter_by_segment(values: np.ndarray, segment_id: np.ndarray, width: int, n_states: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    out = values.copy()
    half = int(width) // 2
    for seg in np.unique(segment_id):
        idx = np.flatnonzero(segment_id == seg)
        vals = values[idx]
        filtered = vals.copy()
        for j in range(len(vals)):
            lo = max(0, j - half)
            hi = min(len(vals), j + half + 1)
            win = vals[lo:hi]
            counts = np.bincount(win, minlength=n_states)
            modes = np.flatnonzero(counts == counts.max())
            med = np.median(win)
            filtered[j] = int(modes[np.argmin(np.abs(modes - med))])
        out[idx] = filtered
    return out


def _metrics(pred: np.ndarray, gt: np.ndarray, sample_index: np.ndarray, segment_id: np.ndarray, prefix: str) -> Dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64)
    pred_int = np.rint(pred).astype(np.int64)
    gt = np.asarray(gt, dtype=np.int64)
    no_lag = gt == 0
    positive = gt > 0
    out = {
        f"{prefix}mae_all": float(np.abs(pred - gt).mean()),
        f"{prefix}mae_injected": float(np.abs(pred[positive] - gt[positive]).mean()),
        f"{prefix}mae_no_lag": float(np.abs(pred[no_lag] - gt[no_lag]).mean()),
        f"{prefix}accuracy_all": float((pred_int == gt).mean()),
        f"{prefix}accuracy_injected": float((pred_int[positive] == gt[positive]).mean()),
        f"{prefix}no_lag_false_alarm_rate": float((pred_int[no_lag] > 0).mean()),
    }
    ranges = [(0, 1200, "first_0_1200"), (3000, 4000, "second_3000_4000"), (5800, 6500, "last_5800_6500")]
    for lo, hi, name in ranges:
        mask = (sample_index >= lo) & (sample_index <= hi)
        out[f"{prefix}{name}_mae"] = float(np.abs(pred[mask] - gt[mask]).mean())
        out[f"{prefix}{name}_accuracy"] = float((pred_int[mask] == gt[mask]).mean())
    seg5 = segment_id == 5
    out[f"{prefix}segment5_mae"] = float(np.abs(pred[seg5] - gt[seg5]).mean())
    out[f"{prefix}segment5_accuracy"] = float((pred_int[seg5] == gt[seg5]).mean())
    return out


def _save_plot(path: Path, sample_index: np.ndarray, gt: np.ndarray, pred: np.ndarray, title: str, mask=None) -> None:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if mask is None:
        mask = np.ones_like(gt, dtype=bool)
    order = np.argsort(sample_index[mask], kind="stable")
    fig, ax = plt.subplots(figsize=(16, 3.6))
    ax.plot(sample_index[mask][order], gt[mask][order], label="gt", linewidth=1.8)
    ax.plot(sample_index[mask][order], pred[mask][order], label="pred", linewidth=1.3)
    ax.set_title(title)
    ax.set_xlabel("sample")
    ax.set_ylabel("lag")
    ax.set_ylim(-0.25, 5.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config.resolve())
    output_dir = (args.output_dir or Path(cfg["logging"]["output_dir"])).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    figs_dir = output_dir / "figs"
    figs_dir.mkdir(exist_ok=True)
    checkpoint = args.checkpoint.resolve()

    prepared = _load_existing_train_test_prepared(cfg)
    batch_size = int(args.batch_size or cfg.get("training", {}).get("batch_size", 256))
    spec = WindowSpec(L=_history_steps(cfg["data"]), H=int(cfg["data"]["H"]))
    ds_te = MultistageWindowDataset(
        prepared.X_groups_test,
        prepared.y_test,
        spec,
        prepared.sample_indices_test,
        prepared.extra_targets_test,
    )
    dl_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False, drop_last=False)

    device = torch.device(args.device)
    edge_cfg = _edge_cfg(cfg)
    edge_name = str(edge_cfg.get("name", "stage1_to_stage2"))
    lag_post = cfg.get("lag_eval_postprocess", {})
    adaptive_cfg = lag_post.get("adaptive_viterbi", {})
    strong = adaptive_cfg.get("strong", {})
    weak = adaptive_cfg.get("weak", {})
    trigger = adaptive_cfg.get("trigger", {})
    stabilizer = lag_post.get("stabilizer", {})

    smooth_identifier = _load_identifier(cfg, prepared, edge_cfg, checkpoint, device)
    smooth_eval = collect_lag_outputs(smooth_identifier, dl_te, device, edge_cfg, edge_name)
    smooth_eval, _ = postprocess_lag_eval_for_eval(smooth_eval, cfg)

    eval_identifier = _load_identifier(
        cfg,
        prepared,
        edge_cfg,
        checkpoint,
        device,
        use_sequence_smoother=bool(lag_post.get("eval_sequence_smoother", False)),
    )
    lag_eval = collect_lag_outputs(eval_identifier, dl_te, device, edge_cfg, edge_name)
    lag_eval, postprocess_info = postprocess_lag_eval_for_eval(lag_eval, cfg)

    pi = np.asarray(lag_eval["pred_pi"], dtype=np.float64)
    gt = np.asarray(lag_eval["lag_value"], dtype=np.int64)
    sample_index = np.asarray(lag_eval["sample_index"], dtype=np.int64)
    time_index = np.asarray(lag_eval.get("time_index", sample_index), dtype=np.int64)
    segment_id = np.asarray(lag_eval["segment_id"], dtype=np.int64)
    region_id = np.asarray(lag_eval.get("region_id", np.full_like(segment_id, -1)), dtype=np.int64)
    axis = np.arange(pi.shape[1], dtype=np.float64)
    raw_expected = pi @ axis
    smooth_expected = np.asarray(smooth_eval["pred_pi"], dtype=np.float64) @ axis
    smooth_occ = np.asarray(smooth_eval["occurrence_score"], dtype=np.float64)

    strong_path = viterbi_decode_lag(
        pi,
        segment_id=segment_id,
        smooth_lambda=float(strong.get("smooth_lambda", 1.5)),
        switch_penalty=float(strong.get("switch_penalty", 1.5)),
        pos_to_zero_penalty=float(strong.get("pos_to_zero_penalty", 3.0)),
    )
    weak_path = viterbi_decode_lag(
        pi,
        segment_id=segment_id,
        smooth_lambda=float(weak.get("smooth_lambda", 0.5)),
        switch_penalty=float(weak.get("switch_penalty", 2.0)),
        pos_to_zero_penalty=float(weak.get("pos_to_zero_penalty", 3.0)),
    )
    adaptive_path = strong_path.copy()
    chosen_segments = []
    for seg in sorted(np.unique(segment_id)):
        mask = segment_id == seg
        smooth_mean = float(smooth_expected[mask].mean())
        smooth_range = float(smooth_expected[mask].max() - smooth_expected[mask].min())
        occ_median = float(np.median(smooth_occ[mask]))
        use_weak = (
            occ_median > float(trigger.get("occurrence_median_gt", 0.5))
            and smooth_mean < float(trigger.get("smooth_expected_mean_lt", 1.2))
            and smooth_range > float(trigger.get("smooth_expected_range_gt", 3.0))
        )
        if use_weak:
            adaptive_path[mask] = weak_path[mask]
            chosen_segments.append(
                {
                    "segment_id": int(seg),
                    "smooth_pred_mean": smooth_mean,
                    "smooth_pred_range": smooth_range,
                    "smooth_occ_median": occ_median,
                }
            )

    stable_path = adaptive_path
    if str(stabilizer.get("type", "segment_mode_filter")) == "segment_mode_filter":
        stable_path = _mode_filter_by_segment(
            adaptive_path,
            segment_id,
            width=int(stabilizer.get("window", 61)),
            n_states=pi.shape[1],
        )

    metrics = {
        "checkpoint": str(checkpoint),
        "config": str(args.config.resolve()),
        "postprocess": "lag_only_nosmooth_adaptive_viterbi_stable",
        "lag_postprocess": json.dumps(postprocess_info),
        "strong_viterbi": json.dumps(strong),
        "weak_viterbi": json.dumps(weak),
        "adaptive_trigger": json.dumps(trigger),
        "adaptive_chosen_segments": json.dumps(chosen_segments),
        "stabilizer": json.dumps(stabilizer),
        **_metrics(raw_expected, gt, sample_index, segment_id, "raw_nosmooth_"),
        **_metrics(adaptive_path, gt, sample_index, segment_id, "adaptive_"),
        **_metrics(stable_path, gt, sample_index, segment_id, "stable_"),
    }
    pd.DataFrame([metrics]).to_csv(output_dir / "lag_eval_metrics.csv", index=False)

    frame = pd.DataFrame(
        {
            "sample": np.arange(len(gt)),
            "sample_index": sample_index,
            "time_index": time_index,
            "segment_id": segment_id,
            "region_id": region_id,
            "lag_gt": gt,
            "lag_flag": lag_eval["lag_flag"],
            "shape_type": lag_eval["shape_type"],
            "raw_expected_nosmooth": raw_expected,
            "argmax_nosmooth": pi.argmax(axis=1),
            "viterbi_strong": strong_path,
            "viterbi_weak": weak_path,
            "pred_lag_adaptive": adaptive_path,
            "pred_lag_stable": stable_path,
            "smooth_expected_for_trigger": smooth_expected,
            "smooth_occurrence_for_trigger": smooth_occ,
        }
    )
    for lag in range(pi.shape[1]):
        frame[f"pi_{lag}_nosmooth"] = pi[:, lag]
    frame.to_csv(output_dir / "lag_eval_predictions.csv", index=False)

    seg_rows = []
    for seg in sorted(np.unique(segment_id)):
        mask = segment_id == seg
        seg_rows.append(
            {
                "segment_id": int(seg),
                "n": int(mask.sum()),
                "sample_min": int(sample_index[mask].min()),
                "sample_max": int(sample_index[mask].max()),
                "gt_min": int(gt[mask].min()),
                "gt_max": int(gt[mask].max()),
                "pos_rate": float((gt[mask] > 0).mean()),
                "stable_mae": float(np.abs(stable_path[mask] - gt[mask]).mean()),
                "stable_accuracy": float((stable_path[mask] == gt[mask]).mean()),
                "adaptive_mae": float(np.abs(adaptive_path[mask] - gt[mask]).mean()),
            }
        )
    pd.DataFrame(seg_rows).to_csv(output_dir / "segment_metrics.csv", index=False)

    joblib.dump({"scaler_x": prepared.scaler_x, "scaler_y": prepared.scaler_y}, output_dir / "scaler.pkl")
    output_checkpoint = output_dir / "best_lag_identifier.pt"
    if checkpoint.resolve() != output_checkpoint.resolve():
        shutil.copy2(checkpoint, output_checkpoint)
    (output_dir / "eval_manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "config": str(args.config.resolve()),
                "checkpoint": str(output_checkpoint),
                "source_checkpoint": str(checkpoint),
                "postprocess": metrics["postprocess"],
                "strong_viterbi": strong,
                "weak_viterbi": weak,
                "adaptive_trigger": trigger,
                "adaptive_chosen_segments": chosen_segments,
                "stabilizer": stabilizer,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    _save_plot(figs_dir / "gt_pred_lag.png", sample_index, gt, stable_path, "GT vs Pred lag")

    print(json.dumps({"output_dir": str(output_dir), "stable_metrics": {k: v for k, v in metrics.items() if k.startswith("stable_")}}, indent=2))


if __name__ == "__main__":
    main()
