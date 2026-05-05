#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.postprocess.q40_final_block_lag_selector import (  # noqa: E402
    Q40FinalSelectorConfig,
    _source_names,
    selection_metrics,
    weak_plateau_mask,
)


def _path(text: str | Path) -> Path:
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _read_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for col in frame.columns:
        if col not in {"split", "source_split", "timestamp", "original_split"}:
            frame[col] = pd.to_numeric(frame[col], errors="ignore")
    return frame


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


def _percentile_rank_by_group(frame: pd.DataFrame, values: np.ndarray, group_col: str) -> np.ndarray:
    out = np.zeros(len(frame), dtype=np.float64)
    if group_col not in frame.columns:
        return pd.Series(values).rank(method="average", pct=True).fillna(0.0).to_numpy(dtype=np.float64)
    for _, idx in frame.groupby(group_col, sort=False).groups.items():
        idx_arr = frame.index.get_indexer(idx)
        out[idx_arr] = (
            pd.Series(values[idx_arr])
            .rank(method="average", pct=True)
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
        )
    return out


def prepare_q40_proposal_frame(frame: pd.DataFrame, group_col: str = "segment_id") -> pd.DataFrame:
    out = frame.copy()
    out["q40_selected"] = (
        out["q40_final_selected"].to_numpy(dtype=np.float64)
        if "q40_final_selected" in out.columns
        else np.zeros(len(out), dtype=np.float64)
    )
    if "q40_d_hat" in out.columns:
        out["q40_d_hat"] = np.maximum(out["q40_d_hat"].to_numpy(dtype=np.float64), 0.0)
    elif "d_hat" in out.columns:
        out["q40_d_hat"] = np.maximum(out["d_hat"].to_numpy(dtype=np.float64), 0.0)
    else:
        out["q40_d_hat"] = np.zeros(len(out), dtype=np.float64)

    out["q40_candidate_score"] = (
        out["candidate_score"].to_numpy(dtype=np.float64)
        if "candidate_score" in out.columns
        else np.zeros(len(out), dtype=np.float64)
    )
    out["q40_localization_score"] = (
        out["localization_score"].to_numpy(dtype=np.float64)
        if "localization_score" in out.columns
        else np.zeros(len(out), dtype=np.float64)
    )
    if "candidate_score_model_rank" in out.columns:
        out["q40_rank_score"] = out["candidate_score_model_rank"].to_numpy(dtype=np.float64)
    else:
        out["q40_rank_score"] = _percentile_rank_by_group(
            out,
            out["q40_localization_score"].to_numpy(dtype=np.float64),
            group_col=group_col,
        )
    if "q40_localization_threshold" in out.columns:
        threshold = out["q40_localization_threshold"].to_numpy(dtype=np.float64)
        out["q40_margin_to_threshold"] = out["q40_localization_score"].to_numpy(dtype=np.float64) - threshold
    else:
        out["q40_margin_to_threshold"] = np.zeros(len(out), dtype=np.float64)

    out["proposal_label"] = (
        out["d_true"].to_numpy(dtype=np.float64) > 0
    ).astype(np.float64) if "d_true" in out.columns else np.zeros(len(out), dtype=np.float64)
    return out


def proposal_only_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if "q40_selected" not in frame.columns:
        raise ValueError("Proposal verifier requires q40_selected column")
    return frame.loc[frame["q40_selected"].to_numpy(dtype=np.float64) > 0].reset_index(drop=True).copy()


def compact_proposal_output(
    frame: pd.DataFrame,
    label_col: str = "d_true",
    group_col: str = "segment_id",
) -> pd.DataFrame:
    cols: list[str] = []
    for col in [
        "split",
        "source_split",
        "timestamp",
        "raw_row_index",
        group_col,
        "t",
        "block_id",
        label_col,
        "proposal_label",
        "q40_selected",
        "q40_d_hat",
        "candidate_score",
        "localization_score",
        "d_raw",
        "expected_lag",
        "p_nonzero",
        "entropy",
        "peak_prob",
        "margin",
        "q40_candidate_score",
        "q40_localization_score",
        "q40_rank_score",
        "q40_margin_to_threshold",
    ]:
        if col in frame.columns and col not in cols:
            cols.append(col)
    return frame[cols].copy()


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export fixed-q40 proposal snapshots from prepared feature tables.")
    parser.add_argument("--fit-series", required=True)
    parser.add_argument("--val-series", required=True)
    parser.add_argument("--eval-series", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", default="transfer")
    parser.add_argument("--label-col", default="d_true")
    parser.add_argument("--group-col", default="segment_id")
    parser.add_argument("--time-col", default="t")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fit_series = _read_csv(_path(args.fit_series))
    val_series = _read_csv(_path(args.val_series))
    eval_series = _read_csv(_path(args.eval_series))
    out_dir = _path(args.output_dir) / str(args.run_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    fit_q40, fit_meta = _apply_fixed_q40(
        reference_frame=fit_series,
        score_frame=fit_series,
        label_col=str(args.label_col),
        group_col=str(args.group_col),
        time_col=str(args.time_col),
    )
    val_q40, val_meta = _apply_fixed_q40(
        reference_frame=fit_series,
        score_frame=val_series,
        label_col=str(args.label_col),
        group_col=str(args.group_col),
        time_col=str(args.time_col),
    )
    eval_q40, eval_meta = _apply_fixed_q40(
        reference_frame=fit_series,
        score_frame=eval_series,
        label_col=str(args.label_col),
        group_col=str(args.group_col),
        time_col=str(args.time_col),
    )

    fit_prop = _export_q40_snapshot(fit_q40, out_dir / "q40_fixed_fit", label_col=str(args.label_col), group_col=str(args.group_col))
    val_prop = _export_q40_snapshot(val_q40, out_dir / "q40_fixed_val", label_col=str(args.label_col), group_col=str(args.group_col))
    eval_prop = _export_q40_snapshot(eval_q40, out_dir / "q40_fixed_eval", label_col=str(args.label_col), group_col=str(args.group_col))

    cfg = Q40FinalSelectorConfig(label_col=str(args.label_col), group_col=str(args.group_col), time_col=str(args.time_col))
    fit_metrics = selection_metrics(fit_q40, cfg)
    val_metrics = selection_metrics(val_q40, cfg)
    eval_metrics = selection_metrics(eval_q40, cfg)
    summary = pd.DataFrame(
        [
            {"split": "fit", **fit_metrics},
            {"split": "val", **val_metrics},
            {"split": "eval", **eval_metrics},
        ]
    )
    summary.to_csv(out_dir / "q40_fixed_snapshot_metrics.csv", index=False)
    (out_dir / "q40_fixed_snapshot_report.json").write_text(
        json.dumps(
            _json_sanitize(
                {
                    "run_name": str(args.run_name),
                    "fit_meta": fit_meta,
                    "val_meta": val_meta,
                    "eval_meta": eval_meta,
                    "fit_metrics": fit_metrics,
                    "val_metrics": val_metrics,
                    "eval_metrics": eval_metrics,
                    "outputs": {
                        "fit_timeseries": (out_dir / "q40_fixed_fit_timeseries.csv").as_posix(),
                        "val_timeseries": (out_dir / "q40_fixed_val_timeseries.csv").as_posix(),
                        "eval_timeseries": (out_dir / "q40_fixed_eval_timeseries.csv").as_posix(),
                    },
                    "rows": {
                        "fit": int(len(fit_prop)),
                        "val": int(len(val_prop)),
                        "eval": int(len(eval_prop)),
                    },
                }
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(summary.to_csv(index=False, float_format="%.6f"))
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
