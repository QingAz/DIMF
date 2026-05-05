#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.postprocess.q40_final_block_lag_selector import (  # noqa: E402
    Q40FinalSelectorConfig,
    selection_metrics as q40_selection_metrics,
)
from src.postprocess.q40_segment_proposal_verifier import (  # noqa: E402
    apply_segment_decisions_to_timeseries,
    build_segment_dataset,
    predict_q40_segment_verifier,
    segment_classification_metrics,
    segment_feature_columns,
    segment_score_distribution_table,
    train_q40_segment_verifier,
    verifier_end_to_end_metrics,
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


def _run_one(name: str, proposal_root: Path, out_dir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    run_in = proposal_root / name
    run_out = out_dir / name
    run_out.mkdir(parents=True, exist_ok=True)

    fit_frame = _read_csv(run_in / "q40_fixed_fit_timeseries.csv")
    val_frame = _read_csv(run_in / "q40_fixed_val_timeseries.csv")
    eval_frame = _read_csv(run_in / "q40_fixed_eval_timeseries.csv")

    fit_segments = build_segment_dataset(
        fit_frame,
        group_col=str(args.group_col),
        time_col=str(args.time_col),
        merge_gap=int(args.merge_gap),
        min_len=int(args.min_len),
        include_q40_segment_features=bool(args.q40_segment_features),
    )
    val_segments = build_segment_dataset(
        val_frame,
        group_col=str(args.group_col),
        time_col=str(args.time_col),
        merge_gap=int(args.merge_gap),
        min_len=int(args.min_len),
        include_q40_segment_features=bool(args.q40_segment_features),
    )
    eval_segments = build_segment_dataset(
        eval_frame,
        group_col=str(args.group_col),
        time_col=str(args.time_col),
        merge_gap=int(args.merge_gap),
        min_len=int(args.min_len),
        include_q40_segment_features=bool(args.q40_segment_features),
    )
    fit_segments.to_csv(run_out / "segment_fit_candidates.csv", index=False)
    val_segments.to_csv(run_out / "segment_val_candidates.csv", index=False)
    eval_segments.to_csv(run_out / "segment_eval_candidates.csv", index=False)

    if fit_segments.empty or val_segments.empty:
        raise ValueError("Segment verifier requires non-empty fit and val segment tables")

    feature_columns = segment_feature_columns(fit_segments)
    model, normalizer, history = train_q40_segment_verifier(
        fit_segments,
        val_segments,
        feature_columns=feature_columns,
        hidden_dim=int(args.hidden_dim),
        dropout=float(args.dropout),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        pos_weight=float(args.pos_weight),
        seed=int(args.seed),
        device=str(args.device),
    )
    history.to_csv(run_out / "segment_verifier_train_history.csv", index=False)

    fit_seg_scored = predict_q40_segment_verifier(
        model,
        fit_segments,
        normalizer=normalizer,
        feature_columns=feature_columns,
        threshold=float(args.decision_threshold),
        device=str(args.device),
    )
    val_seg_scored = predict_q40_segment_verifier(
        model,
        val_segments,
        normalizer=normalizer,
        feature_columns=feature_columns,
        threshold=float(args.decision_threshold),
        device=str(args.device),
    )
    eval_seg_scored = predict_q40_segment_verifier(
        model,
        eval_segments,
        normalizer=normalizer,
        feature_columns=feature_columns,
        threshold=float(args.decision_threshold),
        device=str(args.device),
    )
    fit_seg_scored.to_csv(run_out / "segment_fit_scored.csv", index=False)
    val_seg_scored.to_csv(run_out / "segment_val_scored.csv", index=False)
    eval_seg_scored.to_csv(run_out / "segment_eval_scored.csv", index=False)

    score_dist = segment_score_distribution_table(
        {
            "fit": fit_seg_scored,
            "val": val_seg_scored,
            "eval": eval_seg_scored,
        }
    )
    score_dist.to_csv(run_out / "segment_score_distribution.csv", index=False)

    fit_ts = apply_segment_decisions_to_timeseries(
        fit_frame,
        fit_seg_scored,
        group_col=str(args.group_col),
        time_col=str(args.time_col),
    )
    val_ts = apply_segment_decisions_to_timeseries(
        val_frame,
        val_seg_scored,
        group_col=str(args.group_col),
        time_col=str(args.time_col),
    )
    eval_ts = apply_segment_decisions_to_timeseries(
        eval_frame,
        eval_seg_scored,
        group_col=str(args.group_col),
        time_col=str(args.time_col),
    )
    fit_ts.to_csv(run_out / "segment_verifier_fit_timeseries.csv", index=False)
    val_ts.to_csv(run_out / "segment_verifier_val_timeseries.csv", index=False)
    eval_ts.to_csv(run_out / "segment_verifier_eval_timeseries.csv", index=False)

    q40_cfg = Q40FinalSelectorConfig(
        label_col=str(args.label_col),
        group_col=str(args.group_col),
        time_col=str(args.time_col),
    )
    q40_eval = eval_frame.copy()
    q40_eval["q40_final_selected"] = q40_eval["q40_selected"].to_numpy(dtype=np.float64)
    q40_eval["d_hat"] = q40_eval["q40_d_hat"].to_numpy(dtype=np.float64)
    q40_eval_metrics = q40_selection_metrics(q40_eval, q40_cfg)
    verifier_eval_metrics = verifier_end_to_end_metrics(
        eval_ts,
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
        "q40_eval_overall_recall": q40_eval_metrics["overall_recall"],
        "q40_eval_FAR": q40_eval_metrics["FAR"],
        "q40_eval_zero_E_d_hat": q40_eval_metrics["zero_E_d_hat"],
        "q40_eval_peak_hit_at_pm1": q40_eval_metrics["peak_hit_at_pm1"],
        "q40_eval_pos_MAE": q40_eval_metrics["pos_MAE"],
        "decision_threshold": float(args.decision_threshold),
        "n_fit_segments": int(len(fit_segments)),
        "n_val_segments": int(len(val_segments)),
        "n_eval_segments": int(len(eval_segments)),
    }
    report = {
        "run": name,
        "feature_columns": feature_columns,
        "merge_gap": int(args.merge_gap),
        "min_len": int(args.min_len),
        "q40_segment_features": bool(args.q40_segment_features),
        "decision_threshold": float(args.decision_threshold),
        "segment_metrics": {
            "fit": segment_classification_metrics(fit_seg_scored),
            "val": segment_classification_metrics(val_seg_scored),
            "eval": segment_classification_metrics(eval_seg_scored),
        },
        "lag_metrics": {
            "q40_eval": q40_eval_metrics,
            "verifier_eval": verifier_eval_metrics,
        },
        "outputs": {
            "segment_fit_candidates": (run_out / "segment_fit_candidates.csv").as_posix(),
            "segment_val_candidates": (run_out / "segment_val_candidates.csv").as_posix(),
            "segment_eval_candidates": (run_out / "segment_eval_candidates.csv").as_posix(),
            "segment_score_distribution": (run_out / "segment_score_distribution.csv").as_posix(),
            "segment_verifier_eval_timeseries": (run_out / "segment_verifier_eval_timeseries.csv").as_posix(),
        },
    }
    _write_json(run_out / "q40_segment_verifier_report.json", report)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run r45b segment-level verifier on fixed q40 proposal tables.")
    parser.add_argument("--proposal-root", default="outputs/r45a_q40_point_proposal_verifier_smoke3")
    parser.add_argument("--runs", default="old,seed134_e2")
    parser.add_argument("--output-dir", default="outputs/r45b_q40_segment_proposal_verifier")
    parser.add_argument("--label-col", default="d_true")
    parser.add_argument("--group-col", default="segment_id")
    parser.add_argument("--time-col", default="t")
    parser.add_argument("--merge-gap", type=int, default=1)
    parser.add_argument("--min-len", type=int, default=1)
    parser.add_argument("--q40-segment-features", action="store_true")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pos-weight", type=float, default=1.0)
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    proposal_root = _path(args.proposal_root)
    out_dir = _path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for name in [part.strip() for part in str(args.runs).split(",") if part.strip()]:
        rows.append(_run_one(name, proposal_root, out_dir, args))
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "q40_segment_verifier_eval_summary.csv", index=False)
    print(summary.to_csv(index=False, float_format="%.6f"))
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
