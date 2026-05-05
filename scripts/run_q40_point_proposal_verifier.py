#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.postprocess.q40_final_block_lag_selector import (  # noqa: E402
    Q40FinalSelectorConfig,
    _source_names,
    selection_metrics as q40_selection_metrics,
    weak_plateau_mask,
)
from src.postprocess.q40_point_proposal_verifier import (  # noqa: E402
    DEFAULT_Q40_POINT_VERIFIER_FEATURE_COLUMNS,
    DEFAULT_Q40_POINT_VERIFIER_MINIMAL_FEATURE_COLUMNS,
    available_feature_columns,
    capped_auto_pos_weight,
    compact_proposal_output,
    prepare_q40_proposal_frame,
    predict_q40_point_verifier,
    proposal_classification_metrics,
    proposal_inventory_metrics,
    proposal_only_frame,
    proposal_score_distribution_table,
    threshold_grid,
    train_q40_point_verifier,
    verifier_end_to_end_metrics,
)
from src.postprocess.q40_common import (  # noqa: E402
    add_q40_evidence_features,
    fit_d_raw_calibration,
    split_by_group,
)


def _path(text: str | Path) -> Path:
    path = Path(text).expanduser()
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
    raw = str(text).strip()
    if ":" in raw:
        start_s, end_s, step_s = raw.split(":")
        start = float(start_s)
        end = float(end_s)
        step = float(step_s)
        values = list(np.arange(start, end + 0.5 * step, step))
    else:
        values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    values.append(0.5)
    return sorted({round(float(value), 8) for value in values if 0.0 <= float(value) <= 1.0})


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


def _apply_fixed_q40(
    reference_frame: pd.DataFrame,
    score_frame: pd.DataFrame,
    label_col: str,
    group_col: str,
    time_col: str,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    cfg = Q40FinalSelectorConfig(label_col=label_col, group_col=group_col, time_col=time_col)
    fit = _fit_q40_affine_raw_m(reference_frame, label_col=label_col)
    ref = _q40_enrich(reference_frame, fit)
    score = _q40_enrich(score_frame, fit)

    candidate_ref = ref["candidate_score"].to_numpy(dtype=np.float64) >= cfg.candidate_threshold
    cal_ref = ref["calibrated_raw_m"].to_numpy(dtype=np.float64)
    strong_ref = candidate_ref & (cal_ref >= cfg.strong_raw_m_min)
    loc_ref = ref["localization_score"].to_numpy(dtype=np.float64)
    loc_threshold = (
        float(np.nanpercentile(loc_ref[strong_ref], cfg.strong_loc_percentile_q))
        if np.any(strong_ref)
        else float(cfg.low_lag_loc_threshold)
    )

    candidate = score["candidate_score"].to_numpy(dtype=np.float64) >= cfg.candidate_threshold
    cal = score["calibrated_raw_m"].to_numpy(dtype=np.float64)
    loc = score["localization_score"].to_numpy(dtype=np.float64)
    strong_candidate = candidate & (cal >= cfg.strong_raw_m_min)
    strong_selected = strong_candidate & (loc >= loc_threshold)
    low_lag_high_conf = candidate & (cal < cfg.strong_raw_m_min) & (loc >= cfg.low_lag_loc_threshold)
    primary_selected = strong_selected | low_lag_high_conf
    weak_plateau, plateaus = weak_plateau_mask(score, cfg, primary_selected=primary_selected)
    selected = primary_selected | weak_plateau

    out = score.copy()
    out["q40_strong_candidate"] = strong_candidate.astype(int)
    out["q40_localization_threshold"] = float(loc_threshold)
    out["q40_strong_selected"] = strong_selected.astype(int)
    out["low_lag_high_conf_selected"] = low_lag_high_conf.astype(int)
    out["weak_plateau_selected"] = weak_plateau.astype(int)
    out["q40_final_selected"] = selected.astype(int)
    out["q40_prediction_source"] = _source_names(strong_selected, low_lag_high_conf, weak_plateau)
    out["p_pos"] = np.where(selected, loc, 0.0)
    out["d_hat"] = np.where(selected, cal, 0.0)
    out["q40_d_hat"] = out["d_hat"].to_numpy(dtype=np.float64)
    metadata = {
        "affine_raw_m": fit,
        "q40_localization_threshold": float(loc_threshold),
        "n_strong_candidate": int(strong_candidate.sum()),
        "n_strong_selected": int(strong_selected.sum()),
        "n_low_lag_high_conf_selected": int(low_lag_high_conf.sum()),
        "n_weak_plateau_selected": int(weak_plateau.sum()),
        "n_final_selected": int(selected.sum()),
        "n_plateaus": int(len(plateaus)),
    }
    return out, metadata


def _export_q40_snapshot(
    frame: pd.DataFrame,
    out_prefix: Path,
    label_col: str,
    group_col: str,
) -> pd.DataFrame:
    proposal_frame = prepare_q40_proposal_frame(frame, group_col=group_col)
    proposal_frame.to_csv(out_prefix.with_name(out_prefix.name + "_timeseries.csv"), index=False)
    compact_proposal_output(proposal_frame, label_col=label_col, group_col=group_col).to_csv(
        out_prefix.with_name(out_prefix.name + "_outputs.csv"),
        index=False,
    )
    proposal_only_frame(proposal_frame).to_csv(
        out_prefix.with_name(out_prefix.name + "_proposals.csv"),
        index=False,
    )
    return proposal_frame


def _inventory_row(run: str, split: str, frame: pd.DataFrame) -> Dict[str, Any]:
    return {
        "run": run,
        "split": split,
        **proposal_inventory_metrics(frame),
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

    fit_q40_raw, fit_q40_meta = _apply_fixed_q40(
        reference_frame=fit_raw,
        score_frame=fit_raw,
        label_col=str(args.label_col),
        group_col=str(args.group_col),
        time_col=str(args.time_col),
    )
    val_q40_raw, val_q40_meta = _apply_fixed_q40(
        reference_frame=fit_raw,
        score_frame=val_raw,
        label_col=str(args.label_col),
        group_col=str(args.group_col),
        time_col=str(args.time_col),
    )
    eval_q40_raw, eval_q40_meta = _apply_fixed_q40(
        reference_frame=train_raw,
        score_frame=eval_raw,
        label_col=str(args.label_col),
        group_col=str(args.group_col),
        time_col=str(args.time_col),
    )

    calibration = fit_d_raw_calibration(
        fit_raw,
        label_col=str(args.label_col),
        source_col=str(args.d_raw_source_col),
        mode=str(args.d_raw_calibration),
        clip_to_dmax=bool(args.clip_d_raw_to_dmax),
    )
    fit_enriched = add_q40_evidence_features(
        fit_q40_raw,
        calibration=calibration,
        group_col=str(args.group_col),
        time_col=str(args.time_col),
        window=int(args.window),
    )
    val_enriched = add_q40_evidence_features(
        val_q40_raw,
        calibration=calibration,
        group_col=str(args.group_col),
        time_col=str(args.time_col),
        window=int(args.window),
    )
    eval_enriched = add_q40_evidence_features(
        eval_q40_raw,
        calibration=calibration,
        group_col=str(args.group_col),
        time_col=str(args.time_col),
        window=int(args.window),
    )

    fit_frame = _export_q40_snapshot(fit_enriched, run_dir / "q40_fixed_fit", label_col=str(args.label_col), group_col=str(args.group_col))
    val_frame = _export_q40_snapshot(val_enriched, run_dir / "q40_fixed_val", label_col=str(args.label_col), group_col=str(args.group_col))
    eval_frame = _export_q40_snapshot(eval_enriched, run_dir / "q40_fixed_eval", label_col=str(args.label_col), group_col=str(args.group_col))

    inventory_rows = [
        _inventory_row(name, "fit", fit_frame),
        _inventory_row(name, "val", val_frame),
        _inventory_row(name, "eval", eval_frame),
    ]
    pd.DataFrame(inventory_rows).to_csv(run_dir / "q40_proposal_inventory.csv", index=False)

    requested_features = list(DEFAULT_Q40_POINT_VERIFIER_FEATURE_COLUMNS)
    if bool(args.minimal_features):
        requested_features = list(DEFAULT_Q40_POINT_VERIFIER_MINIMAL_FEATURE_COLUMNS)
    feature_columns = available_feature_columns(fit_frame, requested=requested_features)
    missing = [col for col in requested_features if col not in feature_columns]
    if missing and bool(args.require_all_requested_features):
        raise ValueError(f"Verifier is missing requested feature columns: {', '.join(missing)}")

    fit_proposals = proposal_only_frame(fit_frame)
    val_proposals = proposal_only_frame(val_frame)
    if fit_proposals.empty or val_proposals.empty:
        raise ValueError("Point-level verifier requires non-empty q40 proposals on fit and val splits")

    if bool(args.auto_pos_weight):
        pos_weight = capped_auto_pos_weight(
            fit_frame,
            cap=float(args.pos_weight_cap),
            default=float(args.pos_weight_default),
        )
    else:
        pos_weight = float(args.pos_weight)

    model, normalizer, history = train_q40_point_verifier(
        fit_proposals,
        val_proposals,
        feature_columns=feature_columns,
        hidden_dim=int(args.hidden_dim),
        dropout=float(args.dropout),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        pos_weight=float(pos_weight),
        seed=int(args.seed),
        device=str(args.device),
    )
    history.to_csv(run_dir / "verifier_train_history.csv", index=False)

    fit_scored = predict_q40_point_verifier(
        model,
        fit_frame,
        normalizer=normalizer,
        feature_columns=feature_columns,
        threshold=float(args.decision_threshold),
        device=str(args.device),
    )
    val_scored = predict_q40_point_verifier(
        model,
        val_frame,
        normalizer=normalizer,
        feature_columns=feature_columns,
        threshold=float(args.decision_threshold),
        device=str(args.device),
    )
    eval_scored = predict_q40_point_verifier(
        model,
        eval_frame,
        normalizer=normalizer,
        feature_columns=feature_columns,
        threshold=float(args.decision_threshold),
        device=str(args.device),
    )

    fit_scored.to_csv(run_dir / "verifier_fit_timeseries.csv", index=False)
    val_scored.to_csv(run_dir / "verifier_val_timeseries.csv", index=False)
    eval_scored.to_csv(run_dir / "verifier_eval_timeseries.csv", index=False)
    score_distribution = proposal_score_distribution_table(
        {
            "fit": fit_scored,
            "val": val_scored,
            "eval": eval_scored,
        }
    )
    score_distribution.to_csv(run_dir / "verifier_score_distribution.csv", index=False)
    compact_proposal_output(eval_scored, label_col=str(args.label_col), group_col=str(args.group_col)).to_csv(
        run_dir / "verifier_eval_outputs.csv",
        index=False,
    )

    grid = threshold_grid(
        val_frame,
        model=model,
        normalizer=normalizer,
        feature_columns=feature_columns,
        thresholds=_parse_thresholds(str(args.thresholds)),
        label_col=str(args.label_col),
        group_col=str(args.group_col),
        time_col=str(args.time_col),
        device=str(args.device),
    )
    grid.to_csv(run_dir / "verifier_val_threshold_grid.csv", index=False)

    q40_cfg = Q40FinalSelectorConfig(
        label_col=str(args.label_col),
        group_col=str(args.group_col),
        time_col=str(args.time_col),
    )
    q40_eval_metrics = q40_selection_metrics(eval_q40_raw, q40_cfg)
    verifier_eval_metrics = verifier_end_to_end_metrics(
        eval_scored,
        label_col=str(args.label_col),
        group_col=str(args.group_col),
        time_col=str(args.time_col),
    )
    row = {
        "run": name,
        "verifier_eval_overall_recall": verifier_eval_metrics["overall_recall"],
        "verifier_eval_FAR": verifier_eval_metrics["FAR"],
        "verifier_eval_zero_E_d_hat": verifier_eval_metrics["zero_E_d_hat"],
        "verifier_eval_peak_hit_at_pm1": verifier_eval_metrics["peak_hit_at_pm1"],
        "verifier_eval_pos_MAE": verifier_eval_metrics["pos_MAE"],
        "verifier_eval_n_selected": verifier_eval_metrics["n_selected"],
        "q40_eval_overall_recall": q40_eval_metrics["overall_recall"],
        "q40_eval_FAR": q40_eval_metrics["FAR"],
        "q40_eval_zero_E_d_hat": q40_eval_metrics["zero_E_d_hat"],
        "q40_eval_peak_hit_at_pm1": q40_eval_metrics["peak_hit_at_pm1"],
        "q40_eval_pos_MAE": q40_eval_metrics["pos_MAE"],
        "q40_eval_n_selected": q40_eval_metrics["n_selected"],
        "decision_threshold": float(args.decision_threshold),
        "pos_weight": float(pos_weight),
        "n_fit_proposals": int(len(fit_proposals)),
        "n_val_proposals": int(len(val_proposals)),
        "n_eval_proposals": int(int(eval_frame["q40_selected"].sum())),
    }
    report = {
        "run": name,
        "feature_columns": feature_columns,
        "missing_requested_features": missing,
        "decision_threshold": float(args.decision_threshold),
        "auto_pos_weight": bool(args.auto_pos_weight),
        "pos_weight": float(pos_weight),
        "pos_weight_cap": float(args.pos_weight_cap),
        "pos_weight_default": float(args.pos_weight_default),
        "fit_q40_metadata": fit_q40_meta,
        "val_q40_metadata": val_q40_meta,
        "eval_q40_metadata": eval_q40_meta,
        "proposal_inventory": {
            "fit": inventory_rows[0],
            "val": inventory_rows[1],
            "eval": inventory_rows[2],
        },
        "proposal_metrics": {
            "fit": proposal_classification_metrics(fit_scored),
            "val": proposal_classification_metrics(val_scored),
            "eval": proposal_classification_metrics(eval_scored),
        },
        "lag_metrics": {
            "q40_eval": q40_eval_metrics,
            "verifier_fit": verifier_end_to_end_metrics(
                fit_scored,
                label_col=str(args.label_col),
                group_col=str(args.group_col),
                time_col=str(args.time_col),
            ),
            "verifier_val": verifier_end_to_end_metrics(
                val_scored,
                label_col=str(args.label_col),
                group_col=str(args.group_col),
                time_col=str(args.time_col),
            ),
            "verifier_eval": verifier_eval_metrics,
        },
        "outputs": {
            "q40_fixed_fit_timeseries": (run_dir / "q40_fixed_fit_timeseries.csv").as_posix(),
            "q40_fixed_val_timeseries": (run_dir / "q40_fixed_val_timeseries.csv").as_posix(),
            "q40_fixed_eval_timeseries": (run_dir / "q40_fixed_eval_timeseries.csv").as_posix(),
            "q40_proposal_inventory": (run_dir / "q40_proposal_inventory.csv").as_posix(),
            "verifier_eval_timeseries": (run_dir / "verifier_eval_timeseries.csv").as_posix(),
            "verifier_score_distribution": (run_dir / "verifier_score_distribution.csv").as_posix(),
            "verifier_val_threshold_grid": (run_dir / "verifier_val_threshold_grid.csv").as_posix(),
        },
    }
    _write_json(run_dir / "q40_point_verifier_report.json", report)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Step-0 q40 proposal export and r45a point-level proposal verifier.")
    parser.add_argument("--old-train", default="outputs/r18_light_veto_filter_smoke2/light_veto_train_filtered.csv")
    parser.add_argument("--old-eval", default="outputs/r18_light_veto_filter_smoke2/light_veto_eval_filtered.csv")
    parser.add_argument("--seed134-train", default="outputs/r33_seed134_e2_light_veto_filter/light_veto_train_filtered.csv")
    parser.add_argument("--seed134-eval", default="outputs/r33_seed134_e2_light_veto_filter/light_veto_eval_filtered.csv")
    parser.add_argument("--runs", default="old,seed134_e2")
    parser.add_argument("--output-dir", default="outputs/r45a_q40_point_proposal_verifier")
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
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pos-weight", type=float, default=2.0)
    parser.add_argument("--auto-pos-weight", action="store_true", default=True)
    parser.add_argument("--no-auto-pos-weight", dest="auto_pos_weight", action="store_false")
    parser.add_argument("--pos-weight-cap", type=float, default=3.0)
    parser.add_argument("--pos-weight-default", type=float, default=1.0)
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    parser.add_argument("--thresholds", default="0.05:0.95:0.05")
    parser.add_argument("--minimal-features", action="store_true")
    parser.add_argument("--require-all-requested-features", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = _path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_specs: Dict[str, Tuple[Path, Path]] = {
        "old": (_path(args.old_train), _path(args.old_eval)),
        "seed134_e2": (_path(args.seed134_train), _path(args.seed134_eval)),
    }
    names = [part.strip() for part in str(args.runs).split(",") if part.strip()]
    rows: List[Dict[str, Any]] = []
    for name in names:
        if name not in run_specs:
            raise ValueError(f"Unknown run name: {name}")
        train_path, eval_path = run_specs[name]
        rows.append(_run_one(name, train_path, eval_path, out_dir, args))

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "q40_point_verifier_eval_summary.csv", index=False)
    _write_json(
        out_dir / "q40_point_verifier_summary_report.json",
        {
            "component": "q40_point_proposal_verifier",
            "runs": rows,
            "output_dir": out_dir.as_posix(),
        },
    )
    print(summary.to_csv(index=False, float_format="%.6f"))
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
