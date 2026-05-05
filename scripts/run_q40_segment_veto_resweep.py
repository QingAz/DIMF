#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.postprocess.q40_final_block_lag_selector import (  # noqa: E402
    Q40FinalSelectorConfig,
    selection_metrics as q40_selection_metrics,
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
    return sorted({round(float(value), 8) for value in values if 0.0 <= float(value) <= 1.0})


def _apply_veto_only(frame: pd.DataFrame, theta_drop: float) -> pd.DataFrame:
    out = frame.copy()
    q40_selected = out["q40_selected"].to_numpy(dtype=np.float64) > 0
    q40_d_hat = out["q40_d_hat"].to_numpy(dtype=np.float64) if "q40_d_hat" in out.columns else np.zeros(len(out), dtype=np.float64)
    segment_conf = out["verifier_segment_confidence"].to_numpy(dtype=np.float64) if "verifier_segment_confidence" in out.columns else np.ones(len(out), dtype=np.float64)
    seg_idx = out["verifier_segment_index"].to_numpy(dtype=np.float64) if "verifier_segment_index" in out.columns else np.full(len(out), -1, dtype=np.float64)
    drop_segment = q40_selected & (seg_idx >= 0) & (segment_conf <= float(theta_drop))
    keep = q40_selected & (~drop_segment)
    out["veto_theta_drop"] = float(theta_drop)
    out["veto_drop_segment"] = drop_segment.astype(int)
    out["veto_selected_final"] = keep.astype(int)
    out["veto_d_hat_final"] = np.where(keep, q40_d_hat, 0.0)
    return out


def _lag_metrics(frame: pd.DataFrame, label_col: str, group_col: str, time_col: str) -> Dict[str, Any]:
    proxy = frame.copy()
    proxy["q40_final_selected"] = proxy["veto_selected_final"].to_numpy(dtype=np.float64)
    proxy["d_hat"] = proxy["veto_d_hat_final"].to_numpy(dtype=np.float64)
    cfg = Q40FinalSelectorConfig(label_col=label_col, group_col=group_col, time_col=time_col)
    return q40_selection_metrics(proxy, cfg)


def _q40_metrics(frame: pd.DataFrame, label_col: str, group_col: str, time_col: str) -> Dict[str, Any]:
    proxy = frame.copy()
    proxy["q40_final_selected"] = proxy["q40_selected"].to_numpy(dtype=np.float64)
    proxy["d_hat"] = proxy["q40_d_hat"].to_numpy(dtype=np.float64)
    cfg = Q40FinalSelectorConfig(label_col=label_col, group_col=group_col, time_col=time_col)
    return q40_selection_metrics(proxy, cfg)


def _threshold_grid(frame: pd.DataFrame, thresholds: List[float], label_col: str, group_col: str, time_col: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    q40_metrics = _q40_metrics(frame, label_col=label_col, group_col=group_col, time_col=time_col)
    q40_recall = float(q40_metrics["overall_recall"])
    for theta in thresholds:
        scored = _apply_veto_only(frame, theta_drop=float(theta))
        metrics = _lag_metrics(scored, label_col=label_col, group_col=group_col, time_col=time_col)
        rows.append(
            {
                "theta_drop": float(theta),
                "q40_val_recall": q40_recall,
                "veto_recall": float(metrics["overall_recall"]),
                "veto_FAR": float(metrics["FAR"]),
                "veto_zero_E_d_hat": float(metrics["zero_E_d_hat"]),
                "veto_pos_MAE": float(metrics["pos_MAE"]),
                "veto_peak_hit_at_pm1": float(metrics["peak_hit_at_pm1"]),
                "recall_delta_vs_q40": float(metrics["overall_recall"] - q40_recall),
                "valid_relax_005": bool(metrics["overall_recall"] >= q40_recall - 0.05),
                "valid_relax_010": bool(metrics["overall_recall"] >= q40_recall - 0.10),
            }
        )
    return pd.DataFrame(rows)


def _select_theta(grid: pd.DataFrame) -> Dict[str, Any]:
    stage = "relax_005"
    valid = grid.loc[grid["valid_relax_005"]].copy()
    if valid.empty:
        stage = "relax_010"
        valid = grid.loc[grid["valid_relax_010"]].copy()
    if valid.empty:
        best = grid.sort_values(["veto_FAR", "theta_drop"], ascending=[True, True]).iloc[0]
        return {
            "status": "no_valid_threshold",
            "selection_stage": stage,
            "theta_drop": float(best["theta_drop"]),
        }
    best = valid.sort_values(["veto_FAR", "theta_drop"], ascending=[True, True]).iloc[0]
    return {
        "status": "valid",
        "selection_stage": stage,
        "theta_drop": float(best["theta_drop"]),
    }


def _run_one(name: str, source_root: Path, out_dir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    run_in = source_root / name
    run_out = out_dir / name
    run_out.mkdir(parents=True, exist_ok=True)

    val_frame = _read_csv(run_in / "segment_verifier_val_timeseries.csv")
    eval_frame = _read_csv(run_in / "segment_verifier_eval_timeseries.csv")

    thresholds = _parse_thresholds(str(args.theta_drop_grid))
    val_grid = _threshold_grid(
        val_frame,
        thresholds=thresholds,
        label_col=str(args.label_col),
        group_col=str(args.group_col),
        time_col=str(args.time_col),
    )
    val_grid.to_csv(run_out / "veto_val_threshold_grid.csv", index=False)
    pick = _select_theta(val_grid)
    theta_drop = float(pick["theta_drop"])

    val_scored = _apply_veto_only(val_frame, theta_drop=theta_drop)
    eval_scored = _apply_veto_only(eval_frame, theta_drop=theta_drop)
    val_scored.to_csv(run_out / "veto_val_timeseries.csv", index=False)
    eval_scored.to_csv(run_out / "veto_eval_timeseries.csv", index=False)

    q40_val = _q40_metrics(val_frame, label_col=str(args.label_col), group_col=str(args.group_col), time_col=str(args.time_col))
    q40_eval = _q40_metrics(eval_frame, label_col=str(args.label_col), group_col=str(args.group_col), time_col=str(args.time_col))
    veto_val = _lag_metrics(val_scored, label_col=str(args.label_col), group_col=str(args.group_col), time_col=str(args.time_col))
    veto_eval = _lag_metrics(eval_scored, label_col=str(args.label_col), group_col=str(args.group_col), time_col=str(args.time_col))

    row = {
        "run": name,
        "selection_status": str(pick["status"]),
        "selection_stage": str(pick["selection_stage"]),
        "theta_drop": float(theta_drop),
        "q40_val_recall": q40_val["overall_recall"],
        "veto_val_recall": veto_val["overall_recall"],
        "q40_val_FAR": q40_val["FAR"],
        "veto_val_FAR": veto_val["FAR"],
        "q40_eval_recall": q40_eval["overall_recall"],
        "veto_eval_recall": veto_eval["overall_recall"],
        "q40_eval_FAR": q40_eval["FAR"],
        "veto_eval_FAR": veto_eval["FAR"],
        "q40_eval_zero_E_d_hat": q40_eval["zero_E_d_hat"],
        "veto_eval_zero_E_d_hat": veto_eval["zero_E_d_hat"],
        "q40_eval_pos_MAE": q40_eval["pos_MAE"],
        "veto_eval_pos_MAE": veto_eval["pos_MAE"],
    }
    _write_json(
        run_out / "veto_resweep_report.json",
        {
            "run": name,
            "selection": pick,
            "q40_val_metrics": q40_val,
            "veto_val_metrics": veto_val,
            "q40_eval_metrics": q40_eval,
            "veto_eval_metrics": veto_eval,
            "outputs": {
                "veto_val_threshold_grid": (run_out / "veto_val_threshold_grid.csv").as_posix(),
                "veto_val_timeseries": (run_out / "veto_val_timeseries.csv").as_posix(),
                "veto_eval_timeseries": (run_out / "veto_eval_timeseries.csv").as_posix(),
            },
        },
    )
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="r46a veto-only re-sweep for q40 segment verifier outputs.")
    parser.add_argument("--source-root", default="outputs/r45c_q40_segment_proposal_verifier_smoke")
    parser.add_argument("--runs", default="old,seed134_e2")
    parser.add_argument("--output-dir", default="outputs/r46a_q40_segment_veto_resweep")
    parser.add_argument("--label-col", default="d_true")
    parser.add_argument("--group-col", default="segment_id")
    parser.add_argument("--time-col", default="t")
    parser.add_argument("--theta-drop-grid", default="0.05:0.50:0.05")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = _path(args.source_root)
    out_dir = _path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for name in [part.strip() for part in str(args.runs).split(",") if part.strip()]:
        rows.append(_run_one(name, source_root, out_dir, args))
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "veto_resweep_summary.csv", index=False)
    print(summary.to_csv(index=False, float_format="%.6f"))
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
