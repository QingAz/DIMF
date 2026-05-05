#!/usr/bin/env python3

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_detection_segment_audit import _build_model, _make_eval_loaders
from scripts.select_and_audit_detection_checkpoints import _load_prepared
from scripts.run_g2_magnitude_postprocess import (
    _absolute_path,
    _comparison_rows,
    _hysteresis,
    _merge_gaps,
    _metrics,
    _raw_lookup,
    _remove_short,
    _score_metrics,
)
from src.utils.seed import set_seed
from train import load_config

TIME_FORMAT = "%Y-%m-%d %H:%M"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a G2 magnitude postprocess sanity check with a balanced postprocess-val set and residual scores."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/r4_g2_mag_only_pos_seed42.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/r4_g2_mag_only_pos_seed42/best.ckpt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/r12_g2_magnitude_residual_sanity"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--edge", default="stage1_to_stage2")
    parser.add_argument("--far-max", type=float, default=0.60)
    parser.add_argument("--min-segment-samples", type=int, default=20)
    parser.add_argument("--min-positive-segments", type=int, default=6)
    parser.add_argument("--background-windows", default="25,49,97")
    return parser.parse_args()


def _parse_int_list(text: str) -> List[int]:
    return [int(part.strip()) for part in str(text).split(",") if part.strip()]


@torch.no_grad()
def _collect_magnitude_series(
    cfg: Dict[str, Any],
    checkpoint_path: Path,
    output_dir: Path,
    device: torch.device,
    edge: str,
    splits: List[str],
) -> Dict[str, pd.DataFrame]:
    set_seed(int(cfg.get("seed", 42)))
    prepared, _ = _load_prepared(cfg)
    loaders = _make_eval_loaders(cfg, prepared)
    model = _build_model(cfg, prepared, device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    frames: Dict[str, pd.DataFrame] = {}
    for split_name in splits:
        probs_rows: List[np.ndarray] = []
        for X, _ in loaders[split_name]["loader"]:
            X = {k: v.to(device) for k, v in X.items()}
            _, pi = model(X)
            if edge not in pi:
                raise ValueError(f"Missing edge {edge!r} in model outputs")
            arr = pi[edge]
            arr_last = arr[:, -1, :] if arr.dim() == 3 else arr
            probs_rows.append(arr_last.detach().cpu().numpy())
        if not probs_rows:
            raise ValueError(f"No batches for split={split_name}")
        probs = np.concatenate(probs_rows, axis=0)
        lag_axis = np.arange(probs.shape[1], dtype=np.float64)
        p = 1.0 - probs[:, 0]
        d_hat = (probs * lag_axis[None, :]).sum(axis=1)
        positive_mass = np.clip(p, 1e-12, None)
        m = (probs[:, 1:] * lag_axis[None, 1:]).sum(axis=1) / positive_mass
        m = np.where(p > 1e-12, m, 0.0)
        pred = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(loaders[split_name]["sample_timestamps"]).strftime(TIME_FORMAT),
                "p": p,
                "m": m,
                "d_hat_raw": d_hat,
                "pred_argmax_lag": probs.argmax(axis=1).astype(int),
            }
        )
        joined = _raw_lookup(cfg, split_name).merge(pred, on="timestamp", how="inner")
        joined.insert(0, "split", split_name)
        joined["source_split"] = split_name
        joined = joined.sort_values(["timestamp", "segment_id"]).reset_index(drop=True)
        joined["t"] = joined.groupby("segment_id").cumcount()
        joined["is_positive"] = joined["d_true"].gt(0).astype(int)
        joined = joined[
            [
                "split",
                "source_split",
                "timestamp",
                "raw_row_index",
                "segment_id",
                "t",
                "block_id",
                "dmax",
                "in_block",
                "d_true",
                "is_positive",
                "p",
                "m",
                "d_hat_raw",
                "pred_argmax_lag",
            ]
        ].copy()
        joined.to_csv(output_dir / f"g2_{split_name}_series.csv", index=False)
        frames[split_name] = joined

    pd.concat(frames.values(), ignore_index=True).to_csv(output_dir / "g2_train_val_test_series.csv", index=False)
    return frames


def _segment_audit(series: pd.DataFrame) -> pd.DataFrame:
    return (
        series.groupby(["source_split", "segment_id"], sort=False)
        .agg(
            n_rows=("timestamp", "size"),
            n_in_block=("in_block", "sum"),
            n_out_block=("in_block", lambda s: int((s.to_numpy(dtype=int) == 0).sum())),
            n_lag_positive=("d_true", lambda s: int((s.to_numpy(dtype=int) > 0).sum())),
            dmax=("dmax", "max"),
            m_mean=("m", "mean"),
            m_median=("m", "median"),
            m_p90=("m", lambda s: float(np.nanpercentile(s, 90))),
            start_time=("timestamp", "min"),
            end_time=("timestamp", "max"),
        )
        .reset_index()
    )


def _build_postproc_val(
    frames: Dict[str, pd.DataFrame],
    min_segment_samples: int,
    min_positive_segments: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_series = pd.concat(frames.values(), ignore_index=True)
    audit = _segment_audit(all_series)
    audit["segment_label"] = np.where(audit["n_in_block"].to_numpy(dtype=int) > 0, "in_block", "out_block")
    audit["selected_for_postproc_val"] = 0
    audit["selection_reason"] = ""

    selected_ids: List[int] = []

    old_val_pos = audit.loc[
        (audit["source_split"] == "val")
        & (audit["segment_label"] == "in_block")
        & (audit["n_rows"] >= int(min_segment_samples))
    ].sort_values("segment_id")
    for segment_id in old_val_pos["segment_id"].tolist():
        selected_ids.append(int(segment_id))
        audit.loc[audit["segment_id"] == int(segment_id), "selection_reason"] = "old_val_positive"

    train_pos = audit.loc[
        (audit["source_split"] == "train")
        & (audit["segment_label"] == "in_block")
        & (audit["n_rows"] >= int(min_segment_samples))
    ].copy()
    for dmax in [2, 4, 6]:
        if dmax in set(audit.loc[audit["segment_id"].isin(selected_ids), "dmax"].astype(int).tolist()):
            continue
        part = train_pos.loc[train_pos["dmax"].astype(int) == dmax].sort_values("segment_id", ascending=False)
        if not part.empty:
            segment_id = int(part.iloc[0]["segment_id"])
            if segment_id not in selected_ids:
                selected_ids.append(segment_id)
                audit.loc[audit["segment_id"] == segment_id, "selection_reason"] = f"train_positive_fill_dmax{dmax}"

    if len(selected_ids) < int(min_positive_segments):
        selected_set = set(selected_ids)
        fill = train_pos.loc[~train_pos["segment_id"].isin(selected_set)].sort_values("segment_id", ascending=False)
        for segment_id in fill["segment_id"].tolist():
            if len(selected_ids) >= int(min_positive_segments):
                break
            selected_ids.append(int(segment_id))
            audit.loc[audit["segment_id"] == int(segment_id), "selection_reason"] = "train_positive_fill_count"

    n_positive = len(selected_ids)
    train_neg = audit.loc[
        (audit["source_split"] == "train")
        & (audit["segment_label"] == "out_block")
        & (audit["n_rows"] >= int(min_segment_samples))
    ].sort_values("segment_id", ascending=False)
    for segment_id in train_neg.head(max(n_positive, 1))["segment_id"].tolist():
        selected_ids.append(int(segment_id))
        audit.loc[audit["segment_id"] == int(segment_id), "selection_reason"] = "train_negative_balance"

    selected_ids = sorted(set(selected_ids))
    audit["selected_for_postproc_val"] = audit["segment_id"].isin(selected_ids).astype(int)
    selected = all_series.loc[all_series["segment_id"].isin(selected_ids)].copy()
    selected["original_split"] = selected["split"]
    selected["split"] = "postproc_val"
    selected = selected.sort_values(["timestamp", "segment_id"]).reset_index(drop=True)

    if selected.empty:
        raise ValueError("No rows selected for postproc-val")
    if not selected["in_block"].gt(0).any() or not selected["in_block"].eq(0).any():
        raise ValueError("Postproc-val must include both in-block and out-block rows")
    return selected, audit.sort_values(["source_split", "segment_id"]).reset_index(drop=True)


def _rolling_by_segment(frame: pd.DataFrame, values: np.ndarray, window: int, how: str) -> np.ndarray:
    out = np.zeros(len(frame), dtype=np.float64)
    for _, idx in frame.groupby("segment_id", sort=False).groups.items():
        idx_arr = np.asarray(idx, dtype=int)
        local = pd.Series(values[idx_arr])
        roller = local.rolling(window=int(window), center=True, min_periods=1)
        if how == "mean":
            out[idx_arr] = roller.mean().to_numpy(dtype=np.float64)
        elif how == "median":
            out[idx_arr] = roller.median().to_numpy(dtype=np.float64)
        else:
            raise ValueError(f"Unknown rolling method: {how}")
    return out


def _score_values(frame: pd.DataFrame, score_type: str, background_window: int) -> np.ndarray:
    m = frame["m"].to_numpy(dtype=np.float64)
    if score_type == "raw_m":
        return m
    if score_type == "resid_ma":
        return m - _rolling_by_segment(frame, m, int(background_window), "mean")
    if score_type == "resid_median":
        return m - _rolling_by_segment(frame, m, int(background_window), "median")
    if score_type == "segment_z":
        out = np.zeros(len(frame), dtype=np.float64)
        for _, idx in frame.groupby("segment_id", sort=False).groups.items():
            idx_arr = np.asarray(idx, dtype=int)
            local = m[idx_arr]
            std = float(np.std(local))
            out[idx_arr] = (local - float(np.mean(local))) / max(std, 1e-6)
        return out
    raise ValueError(f"Unknown score_type: {score_type}")


def _smooth_values(frame: pd.DataFrame, values: np.ndarray, smoothing: str) -> np.ndarray:
    if smoothing == "none":
        return values.astype(np.float64)
    if not smoothing.startswith("ma"):
        raise ValueError(f"Unknown score smoothing: {smoothing}")
    return _rolling_by_segment(frame, values.astype(np.float64), int(smoothing[2:]), "mean")


def _apply_score_config(frame: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    out = frame.sort_values(["timestamp", "segment_id"]).reset_index(drop=True).copy()
    raw_score = _score_values(out, str(cfg["score_type"]), int(cfg.get("background_window") or 0))
    score = _smooth_values(out, raw_score, str(cfg.get("score_smoothing", "none")))
    segment_ids = out["segment_id"].to_numpy(dtype=np.int64)
    if cfg["threshold_type"] == "single":
        mask = score > float(cfg["tau_high"])
    elif cfg["threshold_type"] == "hysteresis":
        mask = _hysteresis(score, segment_ids, tau_high=float(cfg["tau_high"]), tau_low=float(cfg["tau_low"]))
    else:
        raise ValueError(f"Unknown threshold_type: {cfg['threshold_type']}")
    mask = _remove_short(mask, segment_ids, int(cfg["min_len"]))
    mask = _merge_gaps(mask, segment_ids, int(cfg["merge_gap"]))
    out["det_score_raw"] = raw_score
    out["det_score"] = score
    out["pred_score"] = score
    out["pred_mask"] = mask.astype(int)
    out["d_hat"] = np.where(mask, out["m"].to_numpy(dtype=np.float64), 0.0)
    return out


def _enriched_metrics(frame: pd.DataFrame) -> Dict[str, Any]:
    row = _metrics(frame)
    scores = frame["pred_score"].to_numpy(dtype=np.float64)
    in_block = frame["in_block"].to_numpy(dtype=np.int64) > 0
    mask = frame["pred_mask"].to_numpy(dtype=np.int64) > 0
    block_row = _score_metrics(scores, in_block.astype(np.int64))
    row["block_row_auprc"] = block_row["auprc"]
    row["block_row_best_f1"] = block_row["best_f1"]
    row["block_out_far"] = float(np.logical_and(mask, ~in_block).sum() / max(int((~in_block).sum()), 1))

    seg_rows: List[Dict[str, Any]] = []
    for segment_id, part in frame.groupby("segment_id", sort=False):
        seg_rows.append(
            {
                "segment_id": int(segment_id),
                "label": int(part["in_block"].gt(0).any()),
                "score_max": float(part["pred_score"].max()),
                "score_mean": float(part["pred_score"].mean()),
                "pred_positive": int(part["pred_mask"].gt(0).any()),
                "dmax": int(part["dmax"].max()),
            }
        )
    seg = pd.DataFrame(seg_rows)
    seg_scores = seg["score_max"].to_numpy(dtype=np.float64)
    seg_labels = seg["label"].to_numpy(dtype=np.int64)
    seg_metrics = _score_metrics(seg_scores, seg_labels)
    seg_pred = seg["pred_positive"].to_numpy(dtype=bool)
    seg_true = seg_labels.astype(bool)
    row["segment_auprc"] = seg_metrics["auprc"]
    row["segment_best_f1"] = seg_metrics["best_f1"]
    row["segment_far"] = float(np.logical_and(seg_pred, ~seg_true).sum() / max(int((~seg_true).sum()), 1))
    row["segment_pred_positive_ratio"] = float(seg_pred.mean()) if len(seg_pred) else 0.0
    row["n_segments"] = int(len(seg))
    row["n_positive_segments"] = int(seg_true.sum())
    row["n_negative_segments"] = int((~seg_true).sum())
    return row


def _threshold_candidates(values: np.ndarray, score_type: str) -> List[float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return [0.0]
    quantiles = [50, 60, 70, 75, 80, 85, 90, 95, 97, 99]
    vals = [float(np.nanpercentile(finite, q)) for q in quantiles]
    if score_type != "raw_m":
        vals.extend([0.0, 0.25, 0.5, 1.0])
    return sorted({round(v, 6) for v in vals if np.isfinite(v)})


def _grid_for_score(frame: pd.DataFrame, score_type: str, background_window: int) -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []
    for score_smoothing in ["none", "ma3", "ma5"]:
        base_score = _score_values(frame, score_type, int(background_window))
        score = _smooth_values(frame, base_score, score_smoothing)
        thresholds = _threshold_candidates(score, score_type)
        for tau in thresholds:
            for min_len in [1, 3, 5]:
                for merge_gap in [0, 1, 3]:
                    configs.append(
                        {
                            "score_type": score_type,
                            "background_window": int(background_window),
                            "score_smoothing": score_smoothing,
                            "threshold_type": "single",
                            "tau_high": float(tau),
                            "tau_low": np.nan,
                            "min_len": int(min_len),
                            "merge_gap": int(merge_gap),
                        }
                    )
        for i, tau_high in enumerate(thresholds):
            low_pool = thresholds[:i]
            for tau_low in low_pool[-5:]:
                if tau_low >= tau_high:
                    continue
                for min_len in [1, 3, 5]:
                    for merge_gap in [0, 1, 3]:
                        configs.append(
                            {
                                "score_type": score_type,
                                "background_window": int(background_window),
                                "score_smoothing": score_smoothing,
                                "threshold_type": "hysteresis",
                                "tau_high": float(tau_high),
                                "tau_low": float(tau_low),
                                "min_len": int(min_len),
                                "merge_gap": int(merge_gap),
                            }
                        )
    return configs


def _candidate_configs(frame: pd.DataFrame, background_windows: List[int]) -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []
    configs.extend(_grid_for_score(frame, "raw_m", 0))
    configs.extend(_grid_for_score(frame, "segment_z", 0))
    for window in background_windows:
        configs.extend(_grid_for_score(frame, "resid_ma", int(window)))
        configs.extend(_grid_for_score(frame, "resid_median", int(window)))
    seen = set()
    unique: List[Dict[str, Any]] = []
    for cfg in configs:
        key = (
            cfg["score_type"],
            cfg["background_window"],
            cfg["score_smoothing"],
            cfg["threshold_type"],
            cfg["tau_high"],
            cfg["tau_low"] if np.isfinite(cfg["tau_low"]) else None,
            cfg["min_len"],
            cfg["merge_gap"],
        )
        if key not in seen:
            seen.add(key)
            unique.append(cfg)
    return unique


def _evaluate_grid(series: pd.DataFrame, background_windows: List[int]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for cfg in _candidate_configs(series, background_windows):
        metrics = _enriched_metrics(_apply_score_config(series, cfg))
        metrics.update(cfg)
        block_mae = metrics.get("block_in_mae", np.nan)
        if not np.isfinite(block_mae):
            block_mae = 1e6
        metrics["selector_score"] = (
            float(metrics.get("segment_auprc", 0.0))
            + 0.5 * float(metrics.get("block_row_auprc", 0.0))
            - 0.5 * float(metrics.get("block_out_far", 0.0))
            - 0.2 * float(block_mae)
        )
        rows.append(metrics)
    return pd.DataFrame(rows).sort_values(
        ["segment_auprc", "block_row_auprc", "block_out_far", "block_in_mae"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)


def _select_best(grid: pd.DataFrame, threshold_type: str, score_family: str, far_max: float) -> pd.Series:
    part = grid.loc[grid["threshold_type"] == threshold_type].copy()
    if score_family == "raw":
        part = part.loc[part["score_type"] == "raw_m"].copy()
    elif score_family == "residual":
        part = part.loc[part["score_type"] != "raw_m"].copy()
    else:
        raise ValueError(f"Unknown score family: {score_family}")
    if part.empty:
        raise ValueError(f"No grid rows for threshold_type={threshold_type}, score_family={score_family}")
    eligible = part.loc[part["block_out_far"] <= float(far_max)].copy()
    selector_status = "far_constrained"
    if eligible.empty:
        eligible = part.copy()
        selector_status = "fallback_no_far_pass"
    ranked = eligible.sort_values(
        ["segment_auprc", "block_row_auprc", "block_out_far", "block_in_mae", "peak_hit_at_pm1"],
        ascending=[False, False, True, True, False],
    ).reset_index(drop=True)
    selected = ranked.iloc[0].copy()
    selected["selector_status"] = selector_status
    return selected


def _row_to_config(row: pd.Series) -> Dict[str, Any]:
    return {
        "score_type": str(row["score_type"]),
        "background_window": int(row["background_window"]),
        "score_smoothing": str(row["score_smoothing"]),
        "threshold_type": str(row["threshold_type"]),
        "tau_high": float(row["tau_high"]),
        "tau_low": float(row["tau_low"]) if pd.notna(row["tau_low"]) else np.nan,
        "min_len": int(row["min_len"]),
        "merge_gap": int(row["merge_gap"]),
    }


def _evaluate_selected(
    test_series: pd.DataFrame,
    selections: List[pd.Series],
    output_dir: Path,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for selected in selections:
        cfg = _row_to_config(selected)
        processed = _apply_score_config(test_series, cfg)
        metrics = _enriched_metrics(processed)
        metrics.update(cfg)
        metrics["score_family"] = "raw" if cfg["score_type"] == "raw_m" else "residual"
        metrics["selected_by"] = "postproc_val_segment_auprc_far_block_mae"
        metrics["selector_status"] = str(selected.get("selector_status", ""))
        rows.append(metrics)
        name = f"{metrics['score_family']}_{cfg['threshold_type']}_{cfg['score_type']}_test_series.csv"
        processed.to_csv(output_dir / name, index=False)
    return pd.DataFrame(rows)


def _audit_split_composition(frames: Dict[str, pd.DataFrame], postproc_val: pd.DataFrame) -> pd.DataFrame:
    parts = {**frames, "postproc_val": postproc_val}
    rows: List[Dict[str, Any]] = []
    for name, frame in parts.items():
        rows.append(
            {
                "split": name,
                "n_rows": int(len(frame)),
                "n_segments": int(frame["segment_id"].nunique()),
                "n_in_block_rows": int(frame["in_block"].sum()),
                "n_out_block_rows": int((frame["in_block"].to_numpy(dtype=int) == 0).sum()),
                "n_lag_positive_rows": int(frame["d_true"].gt(0).sum()),
                "m_mean": float(frame["m"].mean()),
                "m_in_block_mean": float(frame.loc[frame["in_block"] > 0, "m"].mean()) if frame["in_block"].gt(0).any() else np.nan,
                "m_out_block_mean": float(frame.loc[frame["in_block"] == 0, "m"].mean()) if frame["in_block"].eq(0).any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    config_path = _absolute_path(args.config)
    checkpoint_path = _absolute_path(args.checkpoint)
    output_dir = _absolute_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(str(config_path))
    background_windows = _parse_int_list(args.background_windows)
    frames = _collect_magnitude_series(
        cfg=cfg,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        device=torch.device(args.device),
        edge=str(args.edge),
        splits=["train", "val", "test"],
    )

    postproc_val, segment_audit = _build_postproc_val(
        frames,
        min_segment_samples=int(args.min_segment_samples),
        min_positive_segments=int(args.min_positive_segments),
    )
    postproc_val.to_csv(output_dir / "g2_postproc_val_balanced_series.csv", index=False)
    segment_audit.to_csv(output_dir / "g2_postproc_val_segment_audit.csv", index=False)
    split_audit = _audit_split_composition(frames, postproc_val)
    split_audit.to_csv(output_dir / "g2_postproc_split_composition_audit.csv", index=False)

    val_grid = _evaluate_grid(postproc_val, background_windows)
    val_grid.to_csv(output_dir / "g2_residual_postproc_val_grid.csv", index=False)

    selections = [
        _select_best(val_grid, "single", "raw", float(args.far_max)),
        _select_best(val_grid, "hysteresis", "raw", float(args.far_max)),
        _select_best(val_grid, "single", "residual", float(args.far_max)),
        _select_best(val_grid, "hysteresis", "residual", float(args.far_max)),
    ]
    selected_val = pd.DataFrame([row.to_dict() for row in selections])
    selected_val.to_csv(output_dir / "g2_residual_postproc_selected_val.csv", index=False)

    test_selected = _evaluate_selected(frames["test"], selections, output_dir)
    test_selected.to_csv(output_dir / "g2_residual_postproc_test_selected.csv", index=False)

    sanity_rows = test_selected.copy()
    sanity_rows["method"] = sanity_rows.apply(
        lambda row: f"{row['score_family']} {row['threshold_type']} ({row['score_type']})",
        axis=1,
    )
    sanity_cols = [
        "method",
        "score_type",
        "background_window",
        "score_smoothing",
        "threshold_type",
        "tau_high",
        "tau_low",
        "min_len",
        "merge_gap",
        "segment_auprc",
        "block_row_auprc",
        "block_out_far",
        "far",
        "auprc",
        "best_f1",
        "fixed_f1",
        "predicted_nonzero_ratio",
        "block_in_mae",
        "peak_error",
        "peak_hit_at_pm1",
        "selector_status",
    ]
    sanity_rows[sanity_cols].to_csv(output_dir / "g2_residual_sanity_test_table.csv", index=False)

    comparison_rows = []
    for _, row in test_selected.iterrows():
        comparison_rows.append(
            {
                "threshold_type": row["threshold_type"],
                "score_family": row["score_family"],
                "score_type": row["score_type"],
                "background_window": row["background_window"],
                "segment_auprc": row["segment_auprc"],
                "block_row_auprc": row["block_row_auprc"],
                "block_out_far": row["block_out_far"],
                "far": row["far"],
                "block_in_mae": row["block_in_mae"],
                "peak_error": row["peak_error"],
                "peak_hit_at_pm1": row["peak_hit_at_pm1"],
            }
        )
    pd.DataFrame(comparison_rows).to_csv(output_dir / "g2_residual_raw_vs_residual_summary.csv", index=False)

    final_compare = _comparison_rows(test_selected.rename(columns={"score_family": "method"}))
    final_compare.to_csv(output_dir / "g2_residual_final_comparison_legacy_metrics.csv", index=False)

    report = {
        "config": config_path.as_posix(),
        "checkpoint": checkpoint_path.as_posix(),
        "edge": str(args.edge),
        "far_max": float(args.far_max),
        "background_windows": background_windows,
        "postproc_val_note": (
            "Fixed G2 checkpoint. Postprocess-val uses original val positive segments plus train segments "
            "to add evaluable out-block negatives and dmax coverage; test remains original test."
        ),
        "outputs": {
            "postproc_val_series": (output_dir / "g2_postproc_val_balanced_series.csv").as_posix(),
            "segment_audit": (output_dir / "g2_postproc_val_segment_audit.csv").as_posix(),
            "split_audit": (output_dir / "g2_postproc_split_composition_audit.csv").as_posix(),
            "val_grid": (output_dir / "g2_residual_postproc_val_grid.csv").as_posix(),
            "selected_val": (output_dir / "g2_residual_postproc_selected_val.csv").as_posix(),
            "test_selected": (output_dir / "g2_residual_postproc_test_selected.csv").as_posix(),
            "sanity_table": (output_dir / "g2_residual_sanity_test_table.csv").as_posix(),
        },
        "split_audit": split_audit.to_dict(orient="records"),
        "selected_val": selected_val.to_dict(orient="records"),
        "test_selected": test_selected.to_dict(orient="records"),
    }
    (output_dir / "g2_residual_postproc_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Postprocess split audit:")
    print(split_audit.to_csv(index=False))
    print("Selected val configs:")
    print(selected_val.to_csv(index=False))
    print("Sanity test table:")
    print(sanity_rows[sanity_cols].to_csv(index=False))


if __name__ == "__main__":
    main()
