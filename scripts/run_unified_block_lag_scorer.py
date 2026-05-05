#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.postprocess.q40_final_block_lag_selector import (  # noqa: E402
    _source_names,
    Q40FinalSelectorConfig,
    apply_q40_final_selector,
    selection_metrics as q40_selection_metrics,
    weak_plateau_mask,
)
from src.postprocess.unified_lag_scorer import (  # noqa: E402
    DEFAULT_UNIFIED_FEATURE_COLUMNS,
    DEFAULT_Q40_PRIOR_FEATURE_COLUMNS,
    DEFAULT_RELATIVE_FEATURE_COLUMNS,
    HardNegativeSamplingConfig,
    add_unified_evidence_features,
    annotate_sampling_groups,
    apply_threshold,
    artifacts_to_dict,
    available_feature_columns,
    compact_output,
    diagnostic_score_table,
    fit_d_raw_calibration,
    predict_unified_scorer,
    select_threshold_with_constraints,
    selection_metrics,
    split_by_group,
    threshold_dfloor_grid,
    threshold_grid,
    train_unified_scorer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/evaluate the unified row-level lag scorer on liquidsugar block-lag outputs."
    )
    parser.add_argument("--old-train", default="outputs/r18_light_veto_filter_smoke2/light_veto_train_filtered.csv")
    parser.add_argument("--old-eval", default="outputs/r18_light_veto_filter_smoke2/light_veto_eval_filtered.csv")
    parser.add_argument("--seed134-train", default="outputs/r33_seed134_e2_light_veto_filter/light_veto_train_filtered.csv")
    parser.add_argument("--seed134-eval", default="outputs/r33_seed134_e2_light_veto_filter/light_veto_eval_filtered.csv")
    parser.add_argument("--runs", default="old,seed134_e2", help="Comma-separated run names: old, seed134_e2")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/r40_unified_block_lag_scorer"))
    parser.add_argument("--label-col", default="d_true")
    parser.add_argument("--group-col", default="segment_id")
    parser.add_argument("--time-col", default="t")
    parser.add_argument("--d-raw-source-col", default="raw_m")
    parser.add_argument("--d-raw-calibration", choices=["identity", "affine"], default="affine")
    parser.add_argument("--clip-d-raw-to-dmax", action="store_true")
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--relative-features", action="store_true")
    parser.add_argument("--q40-prior-features", action="store_true")
    parser.add_argument("--pos-loss-weight", type=float, default=2.0)
    parser.add_argument("--hard-neg-loss-weight", type=float, default=1.0)
    parser.add_argument("--medium-neg-loss-weight", type=float, default=0.5)
    parser.add_argument("--easy-neg-loss-weight", type=float, default=0.5)
    parser.add_argument("--mag-loss-weight", type=float, default=1.0)
    parser.add_argument("--zero-loss-weight", type=float, default=0.0)
    parser.add_argument("--rate-loss-weight", type=float, default=0.0)
    parser.add_argument("--rate-target-slack", type=float, default=0.10)
    parser.add_argument("--rank-loss-weight", type=float, default=0.0)
    parser.add_argument("--rank-margin", type=float, default=0.10)
    parser.add_argument("--teacher-loss-weight", type=float, default=0.0)
    parser.add_argument("--pos-weight-cap", type=float, default=20.0)
    parser.add_argument("--hard-negative-sampling", action="store_true")
    parser.add_argument("--hard-neg-d-raw-threshold", type=float, default=1.0)
    parser.add_argument("--hard-neg-expected-lag-threshold", type=float, default=1.0)
    parser.add_argument("--hard-neg-p-nonzero-threshold", type=float, default=0.3)
    parser.add_argument("--hard-neg-candidate-score-threshold", type=float, default=0.25)
    parser.add_argument("--hard-neg-localization-quantile", type=float, default=0.75)
    parser.add_argument("--hard-neg-top-fraction", type=float, default=0.30)
    parser.add_argument("--easy-neg-bottom-fraction", type=float, default=0.30)
    parser.add_argument("--hard-neg-positive-fraction", type=float, default=0.25)
    parser.add_argument("--hard-neg-fraction", type=float, default=0.50)
    parser.add_argument("--medium-neg-fraction", type=float, default=0.0)
    parser.add_argument("--easy-neg-fraction", type=float, default=0.25)
    parser.add_argument("--hard-neg-max-per-positive", type=float, default=20.0)
    parser.add_argument("--thresholds", default="0.05:0.95:0.05", help="start:end:step or comma-separated values")
    parser.add_argument("--d-floors", default="0.0,0.5,0.75,1.0,1.25")
    parser.add_argument("--rank-thresholds", default="0.0")
    parser.add_argument("--val-recall-min", type=float, default=0.5)
    parser.add_argument("--val-far-max", type=float, default=0.6)
    parser.add_argument("--val-mae-weight", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _path(text: str | Path) -> Path:
    path = Path(os.path.expandvars(str(text))).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _read_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if len(frame.columns) > 0:
        first = frame.columns[0]
        frame = frame.loc[frame[first].astype(str) != first].reset_index(drop=True)
    for col in frame.columns:
        if col not in {"split", "source_split", "timestamp", "TimeStamp", "original_split", "q40_prediction_source"}:
            try:
                frame[col] = pd.to_numeric(frame[col])
            except (TypeError, ValueError):
                pass
    return frame


def _parse_thresholds(text: str) -> List[float]:
    text = str(text).strip()
    if ":" in text:
        start_s, end_s, step_s = text.split(":")
        start = float(start_s)
        end = float(end_s)
        step = float(step_s)
        values = list(np.arange(start, end + 0.5 * step, step))
    else:
        values = [float(part.strip()) for part in text.split(",") if part.strip()]
    values.append(0.5)
    return sorted({round(float(value), 8) for value in values if 0.0 <= float(value) <= 1.0})


def _parse_floats(text: str) -> List[float]:
    return [float(part.strip()) for part in str(text).split(",") if part.strip()]


def _q40_apply_with_fit_threshold(
    fit_frame: pd.DataFrame,
    score_frame: pd.DataFrame,
    label_col: str,
    group_col: str,
    time_col: str,
) -> pd.DataFrame:
    cfg = Q40FinalSelectorConfig(label_col=label_col, group_col=group_col, time_col=time_col)
    fit = _fit_q40_affine_raw_m(fit_frame, label_col=label_col)
    enriched_fit = _q40_enrich(fit_frame, fit)
    enriched_score = _q40_enrich(score_frame, fit)

    candidate_fit = enriched_fit["candidate_score"].to_numpy(dtype=np.float64) >= cfg.candidate_threshold
    cal_fit = enriched_fit["calibrated_raw_m"].to_numpy(dtype=np.float64)
    strong_fit = candidate_fit & (cal_fit >= cfg.strong_raw_m_min)
    loc_fit = enriched_fit["localization_score"].to_numpy(dtype=np.float64)
    loc_threshold = (
        float(np.nanpercentile(loc_fit[strong_fit], cfg.strong_loc_percentile_q))
        if np.any(strong_fit)
        else float(cfg.low_lag_loc_threshold)
    )

    candidate = enriched_score["candidate_score"].to_numpy(dtype=np.float64) >= cfg.candidate_threshold
    loc = enriched_score["localization_score"].to_numpy(dtype=np.float64)
    cal = enriched_score["calibrated_raw_m"].to_numpy(dtype=np.float64)
    strong_candidate = candidate & (cal >= cfg.strong_raw_m_min)
    mid_high = strong_candidate & (loc >= loc_threshold)
    low_lag_high_conf = candidate & (cal < cfg.strong_raw_m_min) & (loc >= cfg.low_lag_loc_threshold)
    primary_selected = mid_high | low_lag_high_conf
    plateau, _ = weak_plateau_mask(enriched_score, cfg, primary_selected=primary_selected)
    selected_mask = primary_selected | plateau

    out = enriched_score.copy()
    out["q40_strong_candidate"] = strong_candidate.astype(int)
    out["q40_localization_threshold"] = float(loc_threshold)
    out["q40_strong_selected"] = mid_high.astype(int)
    out["low_lag_high_conf_selected"] = low_lag_high_conf.astype(int)
    out["weak_plateau_selected"] = plateau.astype(int)
    out["q40_final_selected"] = selected_mask.astype(int)
    out["q40_prediction_source"] = _source_names(mid_high, low_lag_high_conf, plateau)
    out["q40_p_pos"] = np.where(selected_mask, loc, 0.0)
    out["q40_d_hat"] = np.where(selected_mask, cal, 0.0)
    out["teacher_target"] = out["q40_final_selected"].to_numpy(dtype=np.float64)
    return out


def _json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_sanitize(item) for item in value]
    if isinstance(value, tuple):
        return [_json_sanitize(item) for item in value]
    if isinstance(value, np.generic):
        return _json_sanitize(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_sanitize(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _fit_q40_affine_raw_m(train: pd.DataFrame, label_col: str) -> Dict[str, float]:
    positive = train[label_col].to_numpy(dtype=np.float64) > 0
    x = train.loc[positive, "raw_m"].to_numpy(dtype=np.float64)
    y = train.loc[positive, label_col].to_numpy(dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size == 0:
        raise ValueError("Cannot fit q40 raw_m calibration without positive finite rows")
    a, b = np.linalg.lstsq(np.column_stack([x, np.ones_like(x)]), y, rcond=None)[0]
    pred = a * x + b
    return {
        "a": float(a),
        "b": float(b),
        "fit_rows": int(x.size),
        "fit_mae": float(np.mean(np.abs(pred - y))),
    }


def _q40_enrich(frame: pd.DataFrame, fit: Dict[str, float]) -> pd.DataFrame:
    out = frame.copy()
    calibrated = fit["a"] * out["raw_m"].to_numpy(dtype=np.float64) + fit["b"]
    if "dmax" in out.columns:
        dmax = np.maximum(out["dmax"].to_numpy(dtype=np.float64), 0.0)
        out["calibrated_raw_m"] = np.clip(calibrated, 0.0, dmax)
    else:
        out["calibrated_raw_m"] = np.clip(calibrated, 0.0, None)
    return out


def _q40_baseline(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    label_col: str,
    group_col: str,
    time_col: str,
    out_dir: Path,
) -> Dict[str, Any]:
    cfg = Q40FinalSelectorConfig(label_col=label_col, group_col=group_col, time_col=time_col)
    fit = _fit_q40_affine_raw_m(train, label_col=label_col)
    q40_eval = _q40_enrich(evaluation, fit)
    q40_frame, metadata, plateaus = apply_q40_final_selector(q40_eval, cfg)
    metrics = q40_selection_metrics(q40_frame, cfg)
    q40_frame.to_csv(out_dir / "q40_baseline_eval_timeseries.csv", index=False)
    plateaus.to_csv(out_dir / "q40_baseline_eval_weak_plateaus.csv", index=False)
    return {
        "fit": fit,
        "metadata": metadata,
        "metrics": metrics,
    }


def _run_one(
    name: str,
    train_path: Path,
    eval_path: Path,
    out_dir: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    run_dir = out_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    train_raw = _read_csv(train_path)
    eval_raw = _read_csv(eval_path)

    fit_raw, val_raw = split_by_group(
        train_raw,
        group_col=str(args.group_col),
        val_fraction=float(args.val_fraction),
        seed=int(args.seed),
    )
    calibration = fit_d_raw_calibration(
        fit_raw,
        label_col=str(args.label_col),
        source_col=str(args.d_raw_source_col),
        mode=str(args.d_raw_calibration),
        clip_to_dmax=bool(args.clip_d_raw_to_dmax),
    )
    need_q40_prior = bool(args.q40_prior_features) or float(args.teacher_loss_weight) > 0.0
    if need_q40_prior:
        fit_source = _q40_apply_with_fit_threshold(
            fit_frame=fit_raw,
            score_frame=fit_raw,
            label_col=str(args.label_col),
            group_col=str(args.group_col),
            time_col=str(args.time_col),
        )
        val_source = _q40_apply_with_fit_threshold(
            fit_frame=fit_raw,
            score_frame=val_raw,
            label_col=str(args.label_col),
            group_col=str(args.group_col),
            time_col=str(args.time_col),
        )
        eval_source = _q40_apply_with_fit_threshold(
            fit_frame=train_raw,
            score_frame=eval_raw,
            label_col=str(args.label_col),
            group_col=str(args.group_col),
            time_col=str(args.time_col),
        )
    else:
        fit_source = fit_raw
        val_source = val_raw
        eval_source = eval_raw
    fit_frame = add_unified_evidence_features(
        fit_source,
        calibration=calibration,
        group_col=str(args.group_col),
        time_col=str(args.time_col),
        window=int(args.window),
        include_relative_features=bool(args.relative_features),
        include_q40_prior_features=bool(args.q40_prior_features),
    )
    val_frame = add_unified_evidence_features(
        val_source,
        calibration=calibration,
        group_col=str(args.group_col),
        time_col=str(args.time_col),
        window=int(args.window),
        include_relative_features=bool(args.relative_features),
        include_q40_prior_features=bool(args.q40_prior_features),
    )
    eval_frame = add_unified_evidence_features(
        eval_source,
        calibration=calibration,
        group_col=str(args.group_col),
        time_col=str(args.time_col),
        window=int(args.window),
        include_relative_features=bool(args.relative_features),
        include_q40_prior_features=bool(args.q40_prior_features),
    )
    requested_features = list(DEFAULT_UNIFIED_FEATURE_COLUMNS)
    if bool(args.relative_features):
        requested_features.extend(DEFAULT_RELATIVE_FEATURE_COLUMNS)
    if bool(args.q40_prior_features):
        requested_features.extend(DEFAULT_Q40_PRIOR_FEATURE_COLUMNS)
    feature_columns = available_feature_columns(fit_frame, requested_features)
    missing = [col for col in requested_features if col not in feature_columns]
    if missing:
        raise ValueError(f"Unified scorer is missing expected evidence features: {', '.join(missing)}")

    hard_negative_sampling = HardNegativeSamplingConfig(
        enabled=bool(args.hard_negative_sampling),
        d_raw_threshold=float(args.hard_neg_d_raw_threshold),
        expected_lag_threshold=float(args.hard_neg_expected_lag_threshold),
        p_nonzero_threshold=float(args.hard_neg_p_nonzero_threshold),
        candidate_score_threshold=float(args.hard_neg_candidate_score_threshold),
        localization_score_quantile=float(args.hard_neg_localization_quantile),
        hard_top_fraction=float(args.hard_neg_top_fraction),
        easy_bottom_fraction=float(args.easy_neg_bottom_fraction),
        positive_fraction=float(args.hard_neg_positive_fraction),
        hard_negative_fraction=float(args.hard_neg_fraction),
        medium_negative_fraction=float(args.medium_neg_fraction),
        easy_negative_fraction=float(args.easy_neg_fraction),
        max_hard_per_positive=float(args.hard_neg_max_per_positive),
    )

    fit_frame, fit_group_meta = annotate_sampling_groups(
        fit_frame,
        label_col=str(args.label_col),
        config=hard_negative_sampling if bool(args.hard_negative_sampling) else None,
    )
    val_frame, val_group_meta = annotate_sampling_groups(
        val_frame,
        label_col=str(args.label_col),
        config=hard_negative_sampling if bool(args.hard_negative_sampling) else None,
    )
    eval_frame, eval_group_meta = annotate_sampling_groups(
        eval_frame,
        label_col=str(args.label_col),
        config=hard_negative_sampling if bool(args.hard_negative_sampling) else None,
    )

    model, normalizer, history, sampling_info = train_unified_scorer(
        fit_frame,
        val_frame,
        feature_columns=feature_columns,
        label_col=str(args.label_col),
        hidden_dim=int(args.hidden_dim),
        dropout=float(args.dropout),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        pos_loss_weight=float(args.pos_loss_weight),
        hard_neg_loss_weight=float(args.hard_neg_loss_weight),
        medium_neg_loss_weight=float(args.medium_neg_loss_weight),
        easy_neg_loss_weight=float(args.easy_neg_loss_weight),
        mag_loss_weight=float(args.mag_loss_weight),
        zero_loss_weight=float(args.zero_loss_weight),
        rate_loss_weight=float(args.rate_loss_weight),
        rate_target_slack=float(args.rate_target_slack),
        rank_loss_weight=float(args.rank_loss_weight),
        rank_margin=float(args.rank_margin),
        teacher_loss_weight=float(args.teacher_loss_weight),
        pos_weight_cap=float(args.pos_weight_cap),
        hard_negative_sampling=hard_negative_sampling,
        seed=int(args.seed),
        device=str(args.device),
    )
    history.to_csv(run_dir / "unified_train_history.csv", index=False)

    val_scored_soft = predict_unified_scorer(
        model,
        val_frame,
        normalizer=normalizer,
        threshold=0.5,
        rank_threshold=0.0,
        device=str(args.device),
    )
    grid = threshold_dfloor_grid(
        val_scored_soft,
        thresholds=_parse_thresholds(str(args.thresholds)),
        d_floors=_parse_floats(str(args.d_floors)),
        rank_thresholds=_parse_floats(str(args.rank_thresholds)),
        label_col=str(args.label_col),
        group_col=str(args.group_col),
    )
    grid.to_csv(run_dir / "unified_val_threshold_grid.csv", index=False)
    threshold_pick = select_threshold_with_constraints(
        grid,
        recall_min=float(args.val_recall_min),
        far_max=float(args.val_far_max),
        mae_weight=float(args.val_mae_weight),
    )
    threshold = float(threshold_pick["threshold"])
    d_floor = float(threshold_pick["d_floor"])
    rank_threshold = float(threshold_pick["rank_threshold"])

    fit_scored = predict_unified_scorer(
        model,
        fit_frame,
        normalizer=normalizer,
        threshold=0.5,
        rank_threshold=0.0,
        device=str(args.device),
    )
    fit_scored = apply_threshold(fit_scored, threshold=threshold, d_floor=d_floor, rank_threshold=rank_threshold)
    val_scored = apply_threshold(val_scored_soft, threshold=threshold, d_floor=d_floor, rank_threshold=rank_threshold)
    eval_scored = predict_unified_scorer(
        model,
        eval_frame,
        normalizer=normalizer,
        threshold=0.5,
        rank_threshold=0.0,
        device=str(args.device),
    )
    eval_scored = apply_threshold(eval_scored, threshold=threshold, d_floor=d_floor, rank_threshold=rank_threshold)

    fit_scored.to_csv(run_dir / "unified_fit_timeseries.csv", index=False)
    val_scored.to_csv(run_dir / "unified_val_timeseries.csv", index=False)
    eval_scored.to_csv(run_dir / "unified_eval_timeseries.csv", index=False)
    diagnostics = diagnostic_score_table({"train": fit_scored, "val": val_scored, "eval": eval_scored})
    diagnostics.to_csv(run_dir / "unified_score_diagnostics.csv", index=False)
    compact_output(eval_scored, label_col=str(args.label_col), group_col=str(args.group_col)).to_csv(
        run_dir / "unified_eval_outputs.csv",
        index=False,
    )

    metrics = {
        "fit": selection_metrics(fit_scored, label_col=str(args.label_col), group_col=str(args.group_col)),
        "val": selection_metrics(val_scored, label_col=str(args.label_col), group_col=str(args.group_col)),
        "eval": selection_metrics(eval_scored, label_col=str(args.label_col), group_col=str(args.group_col)),
    }
    q40 = _q40_baseline(
        train=train_raw,
        evaluation=eval_raw,
        label_col=str(args.label_col),
        group_col=str(args.group_col),
        time_col=str(args.time_col),
        out_dir=run_dir,
    )

    checkpoint = {
        "state_dict": model.state_dict(),
        "artifacts": artifacts_to_dict(
            calibration=calibration,
            normalizer=normalizer,
            feature_columns=feature_columns,
            threshold=threshold,
            rank_threshold=rank_threshold,
            d_floor=d_floor,
            model_args={
                "hidden_dim": int(args.hidden_dim),
                "dropout": float(args.dropout),
                "relative_features": bool(args.relative_features),
                "q40_prior_features": bool(args.q40_prior_features),
                "pos_loss_weight": float(args.pos_loss_weight),
                "hard_neg_loss_weight": float(args.hard_neg_loss_weight),
                "medium_neg_loss_weight": float(args.medium_neg_loss_weight),
                "easy_neg_loss_weight": float(args.easy_neg_loss_weight),
                "mag_loss_weight": float(args.mag_loss_weight),
                "zero_loss_weight": float(args.zero_loss_weight),
                "rate_loss_weight": float(args.rate_loss_weight),
                "rate_target_slack": float(args.rate_target_slack),
                "rank_loss_weight": float(args.rank_loss_weight),
                "rank_margin": float(args.rank_margin),
                "teacher_loss_weight": float(args.teacher_loss_weight),
                "pos_weight_cap": float(args.pos_weight_cap),
                "hard_negative_sampling": {
                    "enabled": bool(hard_negative_sampling.enabled),
                    "d_raw_threshold": float(hard_negative_sampling.d_raw_threshold),
                    "expected_lag_threshold": float(hard_negative_sampling.expected_lag_threshold),
                    "p_nonzero_threshold": float(hard_negative_sampling.p_nonzero_threshold),
                    "candidate_score_threshold": float(hard_negative_sampling.candidate_score_threshold),
                    "localization_score_quantile": float(hard_negative_sampling.localization_score_quantile),
                    "hard_top_fraction": float(hard_negative_sampling.hard_top_fraction),
                    "easy_bottom_fraction": float(hard_negative_sampling.easy_bottom_fraction),
                    "positive_fraction": float(hard_negative_sampling.positive_fraction),
                    "hard_negative_fraction": float(hard_negative_sampling.hard_negative_fraction),
                    "medium_negative_fraction": float(hard_negative_sampling.medium_negative_fraction),
                    "easy_negative_fraction": float(hard_negative_sampling.easy_negative_fraction),
                    "max_hard_per_positive": float(hard_negative_sampling.max_hard_per_positive),
                },
            },
        ),
    }
    torch.save(checkpoint, run_dir / "unified_lag_scorer.pt")

    comparison = {
        "run": name,
        "unified_eval_overall_recall": metrics["eval"]["overall_recall"],
        "unified_eval_FAR": metrics["eval"]["FAR"],
        "unified_eval_zero_E_d_hat": metrics["eval"]["zero_E_d_hat"],
        "unified_eval_AUPRC": metrics["eval"]["AUPRC"],
        "unified_eval_peak_hit_at_pm1": metrics["eval"]["peak_hit_at_pm1"],
        "unified_eval_pos_MAE": metrics["eval"]["pos_MAE"],
        "threshold_selection_status": str(threshold_pick["selection_status"]),
        "selected_threshold": float(threshold),
        "selected_d_floor": float(d_floor),
        "selected_rank_threshold": float(rank_threshold),
        "q40_eval_overall_recall": q40["metrics"]["overall_recall"],
        "q40_eval_FAR": q40["metrics"]["FAR"],
        "q40_eval_zero_E_d_hat": q40["metrics"]["zero_E_d_hat"],
        "q40_eval_peak_hit_at_pm1": q40["metrics"]["peak_hit_at_pm1"],
        "q40_eval_pos_MAE": q40["metrics"]["pos_MAE"],
    }
    pd.DataFrame([comparison]).to_csv(run_dir / "unified_vs_q40_eval_summary.csv", index=False)

    report = {
        "component": "unified_block_lag_scorer",
        "run": name,
        "train_path": train_path.as_posix(),
        "eval_path": eval_path.as_posix(),
        "threshold": float(threshold),
        "d_floor": float(d_floor),
        "rank_threshold": float(rank_threshold),
        "threshold_selection": threshold_pick,
        "feature_columns": feature_columns,
        "missing_feature_columns": missing,
        "calibration": {
            "source_col": calibration.source_col,
            "mode": calibration.mode,
            "a": calibration.a,
            "b": calibration.b,
            "clip_to_dmax": calibration.clip_to_dmax,
        },
        "loss_weights": {
            "relative_features": bool(args.relative_features),
            "q40_prior_features": bool(args.q40_prior_features),
            "pos_loss_weight": float(args.pos_loss_weight),
            "hard_neg_loss_weight": float(args.hard_neg_loss_weight),
            "medium_neg_loss_weight": float(args.medium_neg_loss_weight),
            "easy_neg_loss_weight": float(args.easy_neg_loss_weight),
            "mag_loss_weight": float(args.mag_loss_weight),
            "zero_loss_weight": float(args.zero_loss_weight),
            "rate_loss_weight": float(args.rate_loss_weight),
            "rate_target_slack": float(args.rate_target_slack),
            "rank_loss_weight": float(args.rank_loss_weight),
            "rank_margin": float(args.rank_margin),
            "teacher_loss_weight": float(args.teacher_loss_weight),
            "pos_weight_cap": float(args.pos_weight_cap),
        },
        "train_sampling": sampling_info,
        "split_group_metadata": {
            "train": fit_group_meta,
            "val": val_group_meta,
            "eval": eval_group_meta,
        },
        "metrics": metrics,
        "q40_baseline": q40,
        "comparison": comparison,
        "outputs": {
            "checkpoint": (run_dir / "unified_lag_scorer.pt").as_posix(),
            "history": (run_dir / "unified_train_history.csv").as_posix(),
            "threshold_grid": (run_dir / "unified_val_threshold_grid.csv").as_posix(),
            "diagnostics": (run_dir / "unified_score_diagnostics.csv").as_posix(),
            "eval_timeseries": (run_dir / "unified_eval_timeseries.csv").as_posix(),
            "eval_outputs": (run_dir / "unified_eval_outputs.csv").as_posix(),
            "q40_baseline_eval_timeseries": (run_dir / "q40_baseline_eval_timeseries.csv").as_posix(),
        },
    }
    _write_json(run_dir / "unified_lag_scorer_report.json", report)
    return comparison


def _run_specs(args: argparse.Namespace) -> List[Tuple[str, Path, Path]]:
    available = {
        "old": (_path(args.old_train), _path(args.old_eval)),
        "seed134_e2": (_path(args.seed134_train), _path(args.seed134_eval)),
    }
    requested = [part.strip() for part in str(args.runs).split(",") if part.strip()]
    specs: List[Tuple[str, Path, Path]] = []
    for name in requested:
        if name not in available:
            raise ValueError(f"Unknown run {name!r}; available: {', '.join(sorted(available))}")
        train_path, eval_path = available[name]
        specs.append((name, train_path, eval_path))
    return specs


def main() -> None:
    args = parse_args()
    out_dir = _path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, train_path, eval_path in _run_specs(args):
        rows.append(_run_one(name, train_path, eval_path, out_dir, args))
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "unified_vs_q40_eval_summary.csv", index=False)
    _write_json(
        out_dir / "unified_block_lag_scorer_report.json",
        {
            "component": "unified_block_lag_scorer",
            "runs": rows,
            "output_dir": out_dir.as_posix(),
        },
    )
    print(summary.to_csv(index=False))


if __name__ == "__main__":
    main()
