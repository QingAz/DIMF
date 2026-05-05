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


def _q40_metrics(frame: pd.DataFrame, label_col: str, group_col: str, time_col: str) -> Dict[str, Any]:
    proxy = frame.copy()
    proxy["q40_final_selected"] = proxy["q40_selected"].to_numpy(dtype=np.float64)
    proxy["d_hat"] = proxy["q40_d_hat"].to_numpy(dtype=np.float64)
    cfg = Q40FinalSelectorConfig(label_col=label_col, group_col=group_col, time_col=time_col)
    return q40_selection_metrics(proxy, cfg)


def _lag_metrics(frame: pd.DataFrame, label_col: str, group_col: str, time_col: str) -> Dict[str, Any]:
    proxy = frame.copy()
    proxy["q40_final_selected"] = proxy["veto_selected_final"].to_numpy(dtype=np.float64)
    proxy["d_hat"] = proxy["veto_d_hat_final"].to_numpy(dtype=np.float64)
    cfg = Q40FinalSelectorConfig(label_col=label_col, group_col=group_col, time_col=time_col)
    return q40_selection_metrics(proxy, cfg)


def _strong_thresholds(fit_segments: pd.DataFrame) -> Dict[str, float]:
    loc_q75 = float(np.nanquantile(fit_segments["localization_score_mean"].to_numpy(dtype=np.float64), 0.75))
    cand_q75 = float(np.nanquantile(fit_segments["candidate_score_max"].to_numpy(dtype=np.float64), 0.75))
    return {
        "q40_max_d_hat_min": 4.0,
        "localization_score_mean_q75": loc_q75,
        "candidate_score_max_q75": cand_q75,
    }


def _annotate_strong_segments(segment_frame: pd.DataFrame, thresholds: Dict[str, float]) -> pd.DataFrame:
    out = segment_frame.copy()
    strong = (
        (out.get("q40_max_d_hat", out.get("q40_d_hat_max", 0.0)) >= float(thresholds["q40_max_d_hat_min"]))
        | (out["localization_score_mean"].to_numpy(dtype=np.float64) >= float(thresholds["localization_score_mean_q75"]))
        | (out["candidate_score_max"].to_numpy(dtype=np.float64) >= float(thresholds["candidate_score_max_q75"]))
        | (out.get("q40_has_strong_candidate", 0.0).to_numpy(dtype=np.float64) > 0)
    )
    out["segment_is_strong"] = strong.astype(int)
    out["segment_is_weak"] = (~strong).astype(int)
    return out


def _apply_strongkeep_veto(timeseries: pd.DataFrame, segments: pd.DataFrame, theta_drop: float, group_col: str) -> pd.DataFrame:
    key_cols = [group_col, "q40_segment_index"]
    seg_meta = segments[key_cols + ["segment_is_strong", "verifier_segment_confidence"]].copy()
    out = timeseries.merge(
        seg_meta,
        how="left",
        left_on=[group_col, "verifier_segment_index"],
        right_on=key_cols,
    )
    q40_selected = out["q40_selected"].to_numpy(dtype=np.float64) > 0
    q40_d_hat = out["q40_d_hat"].to_numpy(dtype=np.float64)
    is_strong = out["segment_is_strong"].fillna(0).to_numpy(dtype=np.float64) > 0
    segment_conf = out["verifier_segment_confidence_y"].fillna(0.0).to_numpy(dtype=np.float64) if "verifier_segment_confidence_y" in out.columns else out["verifier_segment_confidence_x"].fillna(0.0).to_numpy(dtype=np.float64)
    weak_drop = q40_selected & (~is_strong) & (segment_conf <= float(theta_drop))
    keep = q40_selected & (is_strong | (~weak_drop))
    out["theta_drop"] = float(theta_drop)
    out["segment_is_strong"] = is_strong.astype(int)
    out["weak_drop_segment"] = weak_drop.astype(int)
    out["veto_selected_final"] = keep.astype(int)
    out["veto_d_hat_final"] = np.where(keep, q40_d_hat, 0.0)
    return out


def _threshold_grid(
    val_ts: pd.DataFrame,
    val_segments: pd.DataFrame,
    thresholds: List[float],
    label_col: str,
    group_col: str,
    time_col: str,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    q40 = _q40_metrics(val_ts, label_col=label_col, group_col=group_col, time_col=time_col)
    q40_recall = float(q40["overall_recall"])
    for theta in thresholds:
        scored = _apply_strongkeep_veto(val_ts, val_segments, theta_drop=float(theta), group_col=group_col)
        metrics = _lag_metrics(scored, label_col=label_col, group_col=group_col, time_col=time_col)
        rows.append(
            {
                "theta_drop": float(theta),
                "q40_val_recall": q40_recall,
                "veto_recall": float(metrics["overall_recall"]),
                "veto_FAR": float(metrics["FAR"]),
                "veto_zero_E_d_hat": float(metrics["zero_E_d_hat"]),
                "veto_pos_MAE": float(metrics["pos_MAE"]),
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
        return {"status": "no_valid_threshold", "selection_stage": stage, "theta_drop": float(best["theta_drop"])}
    best = valid.sort_values(["veto_FAR", "theta_drop"], ascending=[True, True]).iloc[0]
    return {"status": "valid", "selection_stage": stage, "theta_drop": float(best["theta_drop"])}


def _drop_audit(segment_frame: pd.DataFrame, theta_drop: float) -> pd.DataFrame:
    out = segment_frame.copy()
    strong = out["segment_is_strong"].to_numpy(dtype=np.float64) > 0
    score = out["verifier_segment_confidence"].to_numpy(dtype=np.float64)
    label = out["segment_label"].to_numpy(dtype=np.float64) > 0
    dropped = (~strong) & (score <= float(theta_drop))
    rows: List[Dict[str, Any]] = []
    for name, mask in [
        ("dropped_positive_strong", dropped & label & strong),
        ("dropped_positive_weak", dropped & label & (~strong)),
        ("dropped_false_positive_strong", dropped & (~label) & strong),
        ("dropped_false_positive_weak", dropped & (~label) & (~strong)),
    ]:
        rows.append({"group": name, "count": int(mask.sum())})
    return pd.DataFrame(rows)


def _run_one(name: str, source_root: Path, out_dir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    run_in = source_root / name
    run_out = out_dir / name
    run_out.mkdir(parents=True, exist_ok=True)

    fit_segments = _read_csv(run_in / "segment_fit_scored.csv")
    val_segments = _read_csv(run_in / "segment_val_scored.csv")
    eval_segments = _read_csv(run_in / "segment_eval_scored.csv")
    val_ts = _read_csv(run_in / "segment_verifier_val_timeseries.csv")
    eval_ts = _read_csv(run_in / "segment_verifier_eval_timeseries.csv")

    strong_cfg = _strong_thresholds(fit_segments)
    fit_segments = _annotate_strong_segments(fit_segments, strong_cfg)
    val_segments = _annotate_strong_segments(val_segments, strong_cfg)
    eval_segments = _annotate_strong_segments(eval_segments, strong_cfg)
    fit_segments.to_csv(run_out / "segment_fit_scored_strong.csv", index=False)
    val_segments.to_csv(run_out / "segment_val_scored_strong.csv", index=False)
    eval_segments.to_csv(run_out / "segment_eval_scored_strong.csv", index=False)

    grid = _threshold_grid(
        val_ts,
        val_segments,
        thresholds=_parse_thresholds(str(args.theta_drop_grid)),
        label_col=str(args.label_col),
        group_col=str(args.group_col),
        time_col=str(args.time_col),
    )
    grid.to_csv(run_out / "strongkeep_veto_val_threshold_grid.csv", index=False)
    pick = _select_theta(grid)
    theta_drop = float(pick["theta_drop"])

    val_scored = _apply_strongkeep_veto(val_ts, val_segments, theta_drop=theta_drop, group_col=str(args.group_col))
    eval_scored = _apply_strongkeep_veto(eval_ts, eval_segments, theta_drop=theta_drop, group_col=str(args.group_col))
    val_scored.to_csv(run_out / "strongkeep_veto_val_timeseries.csv", index=False)
    eval_scored.to_csv(run_out / "strongkeep_veto_eval_timeseries.csv", index=False)

    q40_val = _q40_metrics(val_ts, label_col=str(args.label_col), group_col=str(args.group_col), time_col=str(args.time_col))
    veto_val = _lag_metrics(val_scored, label_col=str(args.label_col), group_col=str(args.group_col), time_col=str(args.time_col))
    q40_eval = _q40_metrics(eval_ts, label_col=str(args.label_col), group_col=str(args.group_col), time_col=str(args.time_col))
    veto_eval = _lag_metrics(eval_scored, label_col=str(args.label_col), group_col=str(args.group_col), time_col=str(args.time_col))

    audit = _drop_audit(eval_segments, theta_drop=theta_drop)
    audit.to_csv(run_out / "eval_drop_audit.csv", index=False)

    row = {
        "run": name,
        "selection_status": str(pick["status"]),
        "selection_stage": str(pick["selection_stage"]),
        "theta_drop": float(theta_drop),
        "q40_eval_recall": q40_eval["overall_recall"],
        "strongkeep_veto_eval_recall": veto_eval["overall_recall"],
        "q40_eval_FAR": q40_eval["FAR"],
        "strongkeep_veto_eval_FAR": veto_eval["FAR"],
        "q40_eval_zero_E_d_hat": q40_eval["zero_E_d_hat"],
        "strongkeep_veto_eval_zero_E_d_hat": veto_eval["zero_E_d_hat"],
        "q40_eval_pos_MAE": q40_eval["pos_MAE"],
        "strongkeep_veto_eval_pos_MAE": veto_eval["pos_MAE"],
        "n_eval_segments": int(len(eval_segments)),
        "n_eval_strong_segments": int((eval_segments["segment_is_strong"].to_numpy(dtype=np.float64) > 0).sum()),
    }
    _write_json(
        run_out / "strongkeep_veto_report.json",
        {
            "run": name,
            "strong_thresholds": strong_cfg,
            "selection": pick,
            "q40_val_metrics": q40_val,
            "strongkeep_veto_val_metrics": veto_val,
            "q40_eval_metrics": q40_eval,
            "strongkeep_veto_eval_metrics": veto_eval,
            "outputs": {
                "threshold_grid": (run_out / "strongkeep_veto_val_threshold_grid.csv").as_posix(),
                "val_timeseries": (run_out / "strongkeep_veto_val_timeseries.csv").as_posix(),
                "eval_timeseries": (run_out / "strongkeep_veto_eval_timeseries.csv").as_posix(),
                "eval_drop_audit": (run_out / "eval_drop_audit.csv").as_posix(),
            },
        },
    )
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="r46b strong-keep + weak-verifier veto re-sweep.")
    parser.add_argument("--source-root", default="outputs/r45c_q40_segment_proposal_verifier_smoke")
    parser.add_argument("--runs", default="old,seed134_e2")
    parser.add_argument("--output-dir", default="outputs/r46b_q40_segment_strongkeep_veto")
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
    summary.to_csv(out_dir / "strongkeep_veto_summary.csv", index=False)
    print(summary.to_csv(index=False, float_format="%.6f"))
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
