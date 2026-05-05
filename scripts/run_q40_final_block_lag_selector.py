#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.postprocess.q40_final_block_lag_selector import (  # noqa: E402
    Q40FinalSelectorConfig,
    apply_q40_final_selector,
    config_to_dict,
    selection_metrics,
)


def _path(text: str | Path) -> Path:
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _read_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    first = frame.columns[0]
    frame = frame.loc[frame[first].astype(str) != first].reset_index(drop=True)
    for col in frame.columns:
        if col not in {"split", "source_split", "timestamp", "original_split"}:
            converted = pd.to_numeric(frame[col], errors="coerce")
            if converted.notna().any() or frame[col].isna().all():
                frame[col] = converted
    return frame


def _fit_affine_raw_m(train: pd.DataFrame, label_col: str) -> Dict[str, float]:
    positive = train[label_col].to_numpy(dtype=np.float64) > 0
    x = train.loc[positive, "raw_m"].to_numpy(dtype=np.float64)
    y = train.loc[positive, label_col].to_numpy(dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size == 0:
        raise ValueError("Cannot fit affine raw_m calibration without positive finite rows")
    a, b = np.linalg.lstsq(np.column_stack([x, np.ones_like(x)]), y, rcond=None)[0]
    pred = a * x + b
    return {
        "a": float(a),
        "b": float(b),
        "fit_rows": int(x.size),
        "fit_mae": float(np.mean(np.abs(pred - y))),
    }


def _enrich(frame: pd.DataFrame, fit: Dict[str, float]) -> pd.DataFrame:
    out = frame.copy()
    calibrated = fit["a"] * out["raw_m"].to_numpy(dtype=np.float64) + fit["b"]
    dmax = np.maximum(out["dmax"].to_numpy(dtype=np.float64), 0.0)
    out["calibrated_raw_m"] = np.clip(calibrated, 0.0, dmax)
    return out


def _compact_outputs(frame: pd.DataFrame, cfg: Q40FinalSelectorConfig) -> pd.DataFrame:
    cols = []
    for col in ["split", "source_split", "timestamp", "raw_row_index", cfg.group_col, cfg.time_col, "block_id", "dmax", cfg.label_col]:
        if col in frame.columns and col not in cols:
            cols.append(col)
    cols.extend(
        [
            "candidate_score",
            "localization_score",
            "raw_m",
            "calibrated_raw_m",
            "expected_lag",
            "q40_localization_threshold",
            "q40_strong_candidate",
            "q40_strong_selected",
            "low_lag_high_conf_selected",
            "weak_plateau_selected",
            "q40_final_selected",
            "q40_prediction_source",
            "p_pos",
            "d_hat",
            "peak_score",
        ]
    )
    return frame[[col for col in cols if col in frame.columns]].copy()


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


def _run_one(
    name: str,
    train_path: Path,
    eval_path: Path,
    out_dir: Path,
    cfg: Q40FinalSelectorConfig,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    train = _read_csv(train_path)
    evaluation = _read_csv(eval_path)
    fit = _fit_affine_raw_m(train, label_col=cfg.label_col)
    frame = _enrich(evaluation, fit)
    final_frame, metadata, plateaus = apply_q40_final_selector(frame, cfg)
    metrics = selection_metrics(final_frame, cfg)
    row = {
        "run": name,
        **metrics,
        "q40_localization_threshold": metadata["q40_localization_threshold"],
        "n_strong_candidate": metadata["n_strong_candidate"],
        "n_q40_strong_selected": metadata["n_q40_strong_selected"],
        "n_low_lag_high_conf_selected": metadata["n_low_lag_high_conf_selected"],
        "n_weak_plateau_selected": metadata["n_weak_plateau_selected"],
        "n_plateaus": metadata["n_plateaus"],
    }

    final_frame.to_csv(out_dir / f"q40_final_{name}_timeseries.csv", index=False)
    _compact_outputs(final_frame, cfg).to_csv(out_dir / f"q40_final_{name}_outputs.csv", index=False)
    plateaus.to_csv(out_dir / f"q40_final_{name}_weak_plateaus.csv", index=False)
    details = {
        "run": name,
        "train_path": train_path.as_posix(),
        "eval_path": eval_path.as_posix(),
        "affine_raw_m": fit,
        "selector_metadata": metadata,
        "metrics": metrics,
    }
    _write_json(out_dir / f"q40_final_{name}_report.json", details)
    return row, details


def _mean_summary(rows: pd.DataFrame) -> Dict[str, Any]:
    metrics = ["overall_recall", "FAR", "d2_recall", "d4_recall", "d6_recall", "pos_MAE", "peak_hit_at_pm1"]
    return {f"{col}_mean": float(rows[col].mean()) for col in metrics}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the fixed q=40 final block-lag selector.")
    parser.add_argument("--old-train", default="outputs/r18_light_veto_filter_smoke2/light_veto_train_filtered.csv")
    parser.add_argument("--old-eval", default="outputs/r18_light_veto_filter_smoke2/light_veto_eval_filtered.csv")
    parser.add_argument("--seed134-train", default="outputs/r33_seed134_e2_light_veto_filter/light_veto_train_filtered.csv")
    parser.add_argument("--seed134-eval", default="outputs/r33_seed134_e2_light_veto_filter/light_veto_eval_filtered.csv")
    parser.add_argument("--output-dir", default="outputs/r39_q40_final_block_lag_selector")
    parser.add_argument("--label-col", default="d_true")
    parser.add_argument("--group-col", default="segment_id")
    parser.add_argument("--time-col", default="t")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = _path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = Q40FinalSelectorConfig(
        label_col=str(args.label_col),
        group_col=str(args.group_col),
        time_col=str(args.time_col),
    )
    run_specs: List[Tuple[str, Path, Path]] = [
        ("old", _path(args.old_train), _path(args.old_eval)),
        ("seed134_e2", _path(args.seed134_train), _path(args.seed134_eval)),
    ]

    rows: List[Dict[str, Any]] = []
    details: Dict[str, Any] = {}
    for name, train_path, eval_path in run_specs:
        row, detail = _run_one(name, train_path, eval_path, out_dir, cfg)
        rows.append(row)
        details[name] = detail

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "q40_final_metrics_by_run.csv", index=False)
    summary = _mean_summary(metrics)
    pd.DataFrame([summary]).to_csv(out_dir / "q40_final_metrics_summary.csv", index=False)
    _write_json(
        out_dir / "q40_final_report.json",
        {
            "component": "q40_final_block_lag_selector",
            "status": "final_fixed_q40",
            "config": config_to_dict(cfg),
            "runs": details,
            "mean_summary": summary,
            "outputs": {
                "metrics_by_run": (out_dir / "q40_final_metrics_by_run.csv").as_posix(),
                "metrics_summary": (out_dir / "q40_final_metrics_summary.csv").as_posix(),
                "old_timeseries": (out_dir / "q40_final_old_timeseries.csv").as_posix(),
                "old_outputs": (out_dir / "q40_final_old_outputs.csv").as_posix(),
                "seed134_timeseries": (out_dir / "q40_final_seed134_e2_timeseries.csv").as_posix(),
                "seed134_outputs": (out_dir / "q40_final_seed134_e2_outputs.csv").as_posix(),
            },
        },
    )

    display_cols = [
        "run",
        "overall_recall",
        "FAR",
        "d2_recall",
        "d4_recall",
        "d6_recall",
        "pos_MAE",
        "peak_hit_at_pm1",
        "n_selected",
        "n_weak_plateau_selected",
    ]
    print(metrics[display_cols].to_csv(index=False, float_format="%.6f"))
    print("Mean summary:")
    print(pd.DataFrame([summary]).to_csv(index=False, float_format="%.6f"))
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
