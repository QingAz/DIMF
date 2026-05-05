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
    apply_threshold,
    compact_output,
    select_far_constrained_threshold,
    selection_metrics,
    split_by_group,
    threshold_dfloor_grid,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Version-1 threshold-only resweep for unified block lag scorer: FAR-constrained theta_c + d_floor."
    )
    parser.add_argument("--base-run-dir", default="outputs/r40_unified_block_lag_scorer")
    parser.add_argument("--old-train", default="outputs/r18_light_veto_filter_smoke2/light_veto_train_filtered.csv")
    parser.add_argument("--old-eval", default="outputs/r18_light_veto_filter_smoke2/light_veto_eval_filtered.csv")
    parser.add_argument("--seed134-train", default="outputs/r33_seed134_e2_light_veto_filter/light_veto_train_filtered.csv")
    parser.add_argument("--seed134-eval", default="outputs/r33_seed134_e2_light_veto_filter/light_veto_eval_filtered.csv")
    parser.add_argument("--runs", default="old,seed134_e2")
    parser.add_argument("--output-dir", default="outputs/r40_unified_block_lag_scorer_v1_threshold")
    parser.add_argument("--label-col", default="d_true")
    parser.add_argument("--group-col", default="segment_id")
    parser.add_argument("--time-col", default="t")
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--thresholds", default="0.05:0.99:0.01")
    parser.add_argument("--d-floors", default="0.5,0.75,1.0,1.25")
    parser.add_argument("--far-tolerance", type=float, default=0.0)
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
    return sorted({round(float(value), 8) for value in values if 0.0 <= float(value) <= 1.0})


def _parse_d_floors(text: str) -> List[float]:
    return sorted({round(float(part.strip()), 8) for part in str(text).split(",") if part.strip()})


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


def _apply_q40(
    fit_frame: pd.DataFrame,
    score_frame: pd.DataFrame,
    label_col: str,
    group_col: str,
    time_col: str,
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
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
    plateau, plateaus = weak_plateau_mask(enriched_score, cfg, primary_selected=primary_selected)
    selected_mask = primary_selected | plateau

    selected = enriched_score.copy()
    selected["q40_strong_candidate"] = strong_candidate.astype(int)
    selected["q40_localization_threshold"] = float(loc_threshold)
    selected["q40_strong_selected"] = mid_high.astype(int)
    selected["low_lag_high_conf_selected"] = low_lag_high_conf.astype(int)
    selected["weak_plateau_selected"] = plateau.astype(int)
    selected["q40_final_selected"] = selected_mask.astype(int)
    selected["q40_prediction_source"] = _source_names(mid_high, low_lag_high_conf, plateau)
    selected["p_pos"] = np.where(selected_mask, loc, 0.0)
    selected["d_hat"] = np.where(selected_mask, cal, 0.0)
    selected["peak_score"] = selected["p_pos"] * selected["d_hat"]

    metrics = q40_selection_metrics(selected, cfg)
    metadata = {
        "mode": "fit_threshold_then_apply",
        "q40_localization_threshold": float(loc_threshold),
        "n_fit_strong_candidate": int(strong_fit.sum()),
        "n_score_strong_candidate": int(strong_candidate.sum()),
        "n_plateaus": int(len(plateaus)),
    }
    return selected, metrics, {"fit": fit, "metadata": metadata, "n_plateaus": int(len(plateaus))}


def _load_base_scored_frames(base_run_dir: Path, run_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    run_dir = base_run_dir / run_name
    val_path = run_dir / "unified_val_timeseries.csv"
    eval_path = run_dir / "unified_eval_timeseries.csv"
    if not val_path.exists():
        raise FileNotFoundError(f"Missing base validation timeseries: {val_path}")
    if not eval_path.exists():
        raise FileNotFoundError(f"Missing base eval timeseries: {eval_path}")
    return _read_csv(val_path), _read_csv(eval_path)


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


def _run_one(
    name: str,
    train_path: Path,
    eval_path: Path,
    base_run_dir: Path,
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

    base_val, base_eval = _load_base_scored_frames(base_run_dir, name)
    base_val_metrics = selection_metrics(base_val, label_col=str(args.label_col), group_col=str(args.group_col))
    base_eval_metrics = selection_metrics(base_eval, label_col=str(args.label_col), group_col=str(args.group_col))

    q40_val_frame, q40_val_metrics, q40_val_meta = _apply_q40(
        fit_frame=fit_raw,
        score_frame=val_raw,
        label_col=str(args.label_col),
        group_col=str(args.group_col),
        time_col=str(args.time_col),
    )
    q40_eval_frame, q40_eval_metrics, q40_eval_meta = _apply_q40(
        fit_frame=train_raw,
        score_frame=eval_raw,
        label_col=str(args.label_col),
        group_col=str(args.group_col),
        time_col=str(args.time_col),
    )
    q40_val_frame.to_csv(run_dir / "q40_val_timeseries.csv", index=False)
    q40_eval_frame.to_csv(run_dir / "q40_eval_timeseries.csv", index=False)

    grid = threshold_dfloor_grid(
        base_val,
        thresholds=_parse_thresholds(str(args.thresholds)),
        d_floors=_parse_d_floors(str(args.d_floors)),
        label_col=str(args.label_col),
        group_col=str(args.group_col),
    )
    grid["target_far"] = float(q40_val_metrics["FAR"])
    grid["q40_val_far_gap"] = np.abs(grid["FAR"].to_numpy(dtype=np.float64) - float(q40_val_metrics["FAR"]))
    grid.to_csv(run_dir / "far_constrained_threshold_grid.csv", index=False)

    selected = select_far_constrained_threshold(
        grid,
        target_far=float(q40_val_metrics["FAR"]),
        far_tolerance=float(args.far_tolerance),
    )
    threshold = float(selected["threshold"])
    d_floor = float(selected["d_floor"])

    tuned_val = apply_threshold(base_val, threshold=threshold, d_floor=d_floor)
    tuned_eval = apply_threshold(base_eval, threshold=threshold, d_floor=d_floor)
    tuned_val.to_csv(run_dir / "unified_val_timeseries.csv", index=False)
    tuned_eval.to_csv(run_dir / "unified_eval_timeseries.csv", index=False)
    compact_output(tuned_eval, label_col=str(args.label_col), group_col=str(args.group_col)).to_csv(
        run_dir / "unified_eval_outputs.csv",
        index=False,
    )

    tuned_val_metrics = selection_metrics(tuned_val, label_col=str(args.label_col), group_col=str(args.group_col))
    tuned_eval_metrics = selection_metrics(tuned_eval, label_col=str(args.label_col), group_col=str(args.group_col))

    comparison = {
        "run": name,
        "selected_threshold": threshold,
        "selected_d_floor": d_floor,
        "selection_status": str(selected["selection_status"]),
        "q40_val_FAR": q40_val_metrics["FAR"],
        "base_unified_val_FAR": base_val_metrics["FAR"],
        "tuned_unified_val_FAR": tuned_val_metrics["FAR"],
        "q40_val_recall": q40_val_metrics["overall_recall"],
        "base_unified_val_recall": base_val_metrics["overall_recall"],
        "tuned_unified_val_recall": tuned_val_metrics["overall_recall"],
        "q40_eval_FAR": q40_eval_metrics["FAR"],
        "base_unified_eval_FAR": base_eval_metrics["FAR"],
        "tuned_unified_eval_FAR": tuned_eval_metrics["FAR"],
        "q40_eval_recall": q40_eval_metrics["overall_recall"],
        "base_unified_eval_recall": base_eval_metrics["overall_recall"],
        "tuned_unified_eval_recall": tuned_eval_metrics["overall_recall"],
        "base_unified_eval_pos_MAE": base_eval_metrics["pos_MAE"],
        "tuned_unified_eval_pos_MAE": tuned_eval_metrics["pos_MAE"],
        "q40_eval_pos_MAE": q40_eval_metrics["pos_MAE"],
        "base_unified_eval_peak_hit_at_pm1": base_eval_metrics["peak_hit_at_pm1"],
        "tuned_unified_eval_peak_hit_at_pm1": tuned_eval_metrics["peak_hit_at_pm1"],
        "q40_eval_peak_hit_at_pm1": q40_eval_metrics["peak_hit_at_pm1"],
    }
    pd.DataFrame([comparison]).to_csv(run_dir / "v1_threshold_comparison.csv", index=False)

    report = {
        "component": "unified_block_threshold_v1",
        "run": name,
        "base_run_dir": (base_run_dir / name).as_posix(),
        "train_path": train_path.as_posix(),
        "eval_path": eval_path.as_posix(),
        "selected": selected,
        "base_metrics": {"val": base_val_metrics, "eval": base_eval_metrics},
        "tuned_metrics": {"val": tuned_val_metrics, "eval": tuned_eval_metrics},
        "q40_metrics": {"val": q40_val_metrics, "eval": q40_eval_metrics},
        "q40_meta": {"val": q40_val_meta, "eval": q40_eval_meta},
        "comparison": comparison,
        "outputs": {
            "threshold_grid": (run_dir / "far_constrained_threshold_grid.csv").as_posix(),
            "unified_val_timeseries": (run_dir / "unified_val_timeseries.csv").as_posix(),
            "unified_eval_timeseries": (run_dir / "unified_eval_timeseries.csv").as_posix(),
            "unified_eval_outputs": (run_dir / "unified_eval_outputs.csv").as_posix(),
            "q40_val_timeseries": (run_dir / "q40_val_timeseries.csv").as_posix(),
            "q40_eval_timeseries": (run_dir / "q40_eval_timeseries.csv").as_posix(),
        },
    }
    _write_json(run_dir / "v1_threshold_report.json", report)
    return comparison


def main() -> None:
    args = parse_args()
    base_run_dir = _path(args.base_run_dir)
    out_dir = _path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, train_path, eval_path in _run_specs(args):
        rows.append(_run_one(name, train_path, eval_path, base_run_dir, out_dir, args))

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "v1_threshold_comparison.csv", index=False)
    _write_json(
        out_dir / "v1_threshold_report.json",
        {
            "component": "unified_block_threshold_v1",
            "base_run_dir": base_run_dir.as_posix(),
            "runs": rows,
            "output_dir": out_dir.as_posix(),
        },
    )
    print(summary.to_csv(index=False))


if __name__ == "__main__":
    main()
