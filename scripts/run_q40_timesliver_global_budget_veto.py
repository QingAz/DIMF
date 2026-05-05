#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _load_q40_selector_module() -> Any:
    module_path = ROOT / "src" / "postprocess" / "q40_final_block_lag_selector.py"
    module_name = "q40_final_block_lag_selector"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load q40 selector module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_Q40_SELECTOR = _load_q40_selector_module()
Q40FinalSelectorConfig = _Q40_SELECTOR.Q40FinalSelectorConfig
q40_selection_metrics = _Q40_SELECTOR.selection_metrics


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
        if col not in {"split", "source_split", "timestamp", "TimeStamp", "original_split", "q40_prediction_source", "segment_uid"}:
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


def _parse_float_list(text: str) -> List[float]:
    return [float(part.strip()) for part in str(text).split(",") if part.strip()]


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


def _resolve_attr_csv(name: str, old_root: Path, seed_root: Path) -> Path:
    if name == "old":
        return old_root / "old" / "timesliver_segment_attr_scores.csv"
    if name == "seed134_e2":
        return seed_root / "seed134_e2" / "timesliver_segment_attr_scores.csv"
    raise ValueError(f"Unknown run name: {name}")


def _normalized_sequence_rank(values: pd.Series) -> pd.Series:
    n = len(values)
    if n <= 1:
        return pd.Series(np.full(n, 0.5, dtype=np.float64), index=values.index)
    ranks = values.rank(method="average", ascending=True).to_numpy(dtype=np.float64)
    normalized = (ranks - 1.0) / float(n - 1)
    return pd.Series(normalized, index=values.index)


def _annotate_combined_scores(segment_frame: pd.DataFrame, verifier_weight: float, attr_weight: float, sequence_col: str) -> pd.DataFrame:
    out = segment_frame.copy()
    out["attr_score"] = out["attr_score"].fillna(0.0)
    out["rank_v"] = 1.0
    out["rank_attr"] = 1.0
    out["combined_score"] = 1.0
    weak_mask = out["segment_is_strong"].fillna(0).to_numpy(dtype=np.float64) <= 0
    weak = out.loc[weak_mask].copy()
    if weak.empty:
        return out
    weak["rank_v"] = weak.groupby(sequence_col)["verifier_segment_confidence"].transform(_normalized_sequence_rank)
    weak["rank_attr"] = weak.groupby(sequence_col)["attr_score"].transform(_normalized_sequence_rank)
    weak["combined_score"] = float(verifier_weight) * weak["rank_v"] + float(attr_weight) * weak["rank_attr"]
    out.loc[weak.index, "rank_v"] = weak["rank_v"].to_numpy(dtype=np.float64)
    out.loc[weak.index, "rank_attr"] = weak["rank_attr"].to_numpy(dtype=np.float64)
    out.loc[weak.index, "combined_score"] = weak["combined_score"].to_numpy(dtype=np.float64)
    return out


def _fit_protection_thresholds(fit_segments: pd.DataFrame) -> Dict[str, float]:
    loc_col = "q40_mean_localization_score" if "q40_mean_localization_score" in fit_segments.columns else "localization_score_mean"
    cand_col = "q40_mean_candidate_score" if "q40_mean_candidate_score" in fit_segments.columns else "candidate_score_mean"
    loc_q75 = float(np.nanquantile(fit_segments[loc_col].to_numpy(dtype=np.float64), 0.75))
    cand_q75 = float(np.nanquantile(fit_segments[cand_col].to_numpy(dtype=np.float64), 0.75))
    return {
        "max_d_hat_min": 4.0,
        "loc_mean_q75": loc_q75,
        "cand_mean_q75": cand_q75,
        "loc_col": loc_col,
        "cand_col": cand_col,
    }


def _annotate_protected(segment_frame: pd.DataFrame, thresholds: Dict[str, float]) -> pd.DataFrame:
    out = segment_frame.copy()
    max_dhat_col = "q40_max_d_hat" if "q40_max_d_hat" in out.columns else "q40_d_hat_max"
    loc_col = str(thresholds["loc_col"])
    cand_col = str(thresholds["cand_col"])
    protected = (
        (out["segment_is_strong"].fillna(0).to_numpy(dtype=np.float64) > 0)
        | (out[max_dhat_col].to_numpy(dtype=np.float64) >= float(thresholds["max_d_hat_min"]))
        | (out[loc_col].to_numpy(dtype=np.float64) >= float(thresholds["loc_mean_q75"]))
        | (out[cand_col].to_numpy(dtype=np.float64) >= float(thresholds["cand_mean_q75"]))
    )
    out["segment_is_protected"] = protected.astype(int)
    out["drop_candidate"] = ((out["segment_is_strong"].fillna(0).to_numpy(dtype=np.float64) <= 0) & (~protected)).astype(int)
    return out


def _apply_global_budget(
    segment_frame: pd.DataFrame,
    global_drop_budget: float,
    min_drop_k: int,
    max_drop_ratio: float,
) -> pd.DataFrame:
    out = segment_frame.copy()
    out["drop_weak_segment"] = 0
    out["global_drop_budget"] = float(global_drop_budget)
    out["drop_count"] = 0
    candidates = out.loc[out["drop_candidate"].to_numpy(dtype=np.float64) > 0].copy()
    n = len(candidates)
    if n <= 0:
        return out
    raw_count = int(math.floor(float(global_drop_budget) * float(n)))
    max_count = max(int(min_drop_k), int(math.floor(float(max_drop_ratio) * float(n))))
    drop_count = min(max(raw_count, int(min_drop_k)), max_count, n)
    chosen = candidates.sort_values(
        ["combined_score", "attr_score", "verifier_segment_confidence", "segment_id", "q40_segment_index"],
        ascending=[True, True, True, True, True],
    ).head(drop_count)
    if len(chosen) > 0:
        out.loc[chosen.index, "drop_weak_segment"] = 1
    out["drop_count"] = int(drop_count)
    return out


def _apply_timeseries_veto(timeseries: pd.DataFrame, segments: pd.DataFrame, group_col: str) -> pd.DataFrame:
    seg_meta = segments[
        [group_col, "q40_segment_index", "segment_is_strong", "segment_is_protected", "drop_candidate", "rank_v", "rank_attr", "combined_score", "drop_weak_segment"]
    ].copy()
    out = timeseries.merge(
        seg_meta,
        how="left",
        left_on=[group_col, "verifier_segment_index"],
        right_on=[group_col, "q40_segment_index"],
    )
    q40_selected = out["q40_selected"].to_numpy(dtype=np.float64) > 0
    q40_d_hat = out["q40_d_hat"].to_numpy(dtype=np.float64)
    drop_weak = out["drop_weak_segment"].fillna(0).to_numpy(dtype=np.float64) > 0
    keep = q40_selected & (~drop_weak)
    out["segment_is_strong"] = out["segment_is_strong"].fillna(0).to_numpy(dtype=np.float64).astype(int)
    out["segment_is_protected"] = out["segment_is_protected"].fillna(0).to_numpy(dtype=np.float64).astype(int)
    out["drop_candidate"] = out["drop_candidate"].fillna(0).to_numpy(dtype=np.float64).astype(int)
    out["rank_v"] = out["rank_v"].fillna(1.0)
    out["rank_attr"] = out["rank_attr"].fillna(1.0)
    out["combined_score"] = out["combined_score"].fillna(1.0)
    out["drop_weak_segment"] = drop_weak.astype(int)
    out["veto_selected_final"] = keep.astype(int)
    out["veto_d_hat_final"] = np.where(keep, q40_d_hat, 0.0)
    return out


def _drop_audit(segment_frame: pd.DataFrame) -> pd.DataFrame:
    strong = segment_frame["segment_is_strong"].to_numpy(dtype=np.float64) > 0
    label = segment_frame["segment_label"].to_numpy(dtype=np.float64) > 0
    dropped = segment_frame["drop_weak_segment"].to_numpy(dtype=np.float64) > 0
    rows: List[Dict[str, Any]] = []
    for name, mask in [
        ("dropped_positive_strong", dropped & label & strong),
        ("dropped_positive_weak", dropped & label & (~strong)),
        ("dropped_false_positive_strong", dropped & (~label) & strong),
        ("dropped_false_positive_weak", dropped & (~label) & (~strong)),
    ]:
        rows.append({"group": name, "count": int(mask.sum())})
    return pd.DataFrame(rows)


def _audit_counts(audit: pd.DataFrame) -> Dict[str, int]:
    return {str(row["group"]): int(row["count"]) for row in audit.to_dict(orient="records")}


def _grid_search(
    val_ts: pd.DataFrame,
    val_segments: pd.DataFrame,
    global_drop_budgets: Sequence[float],
    verifier_weight: float,
    attr_weight: float,
    min_drop_k: int,
    max_drop_ratio: float,
    label_col: str,
    group_col: str,
    time_col: str,
) -> pd.DataFrame:
    q40 = _q40_metrics(val_ts, label_col=label_col, group_col=group_col, time_col=time_col)
    q40_recall = float(q40["overall_recall"])
    ranked = _annotate_combined_scores(val_segments, verifier_weight=verifier_weight, attr_weight=attr_weight, sequence_col=group_col)
    rows: List[Dict[str, Any]] = []
    for budget in global_drop_budgets:
        dropped = _apply_global_budget(ranked, global_drop_budget=float(budget), min_drop_k=min_drop_k, max_drop_ratio=max_drop_ratio)
        audit = _audit_counts(_drop_audit(dropped))
        scored = _apply_timeseries_veto(val_ts, dropped, group_col=group_col)
        metrics = _lag_metrics(scored, label_col=label_col, group_col=group_col, time_col=time_col)
        candidate_count = int((dropped["drop_candidate"].to_numpy(dtype=np.float64) > 0).sum())
        rows.append(
            {
                "global_drop_budget": float(budget),
                "drop_candidate_count": candidate_count,
                "actual_drop_count": int(dropped["drop_count"].iloc[0]) if len(dropped) > 0 else 0,
                "q40_val_recall": q40_recall,
                "veto_recall": float(metrics["overall_recall"]),
                "veto_FAR": float(metrics["FAR"]),
                "veto_zero_E_d_hat": float(metrics["zero_E_d_hat"]),
                "veto_pos_MAE": float(metrics["pos_MAE"]),
                "recall_drop": float(q40_recall - float(metrics["overall_recall"])),
                "dropped_positive_total": int(audit.get("dropped_positive_strong", 0) + audit.get("dropped_positive_weak", 0)),
                "dropped_positive_strong": int(audit.get("dropped_positive_strong", 0)),
                "dropped_positive_weak": int(audit.get("dropped_positive_weak", 0)),
                "dropped_false_positive_strong": int(audit.get("dropped_false_positive_strong", 0)),
                "dropped_false_positive_weak": int(audit.get("dropped_false_positive_weak", 0)),
                "valid_relax_005": bool(float(q40_recall - metrics["overall_recall"]) <= 0.05),
                "valid_zero_drop": bool(int(audit.get("dropped_positive_strong", 0) + audit.get("dropped_positive_weak", 0)) == 0),
            }
        )
    return pd.DataFrame(rows)


def _select_budget(grid: pd.DataFrame) -> Dict[str, Any]:
    valid = grid.loc[grid["valid_relax_005"] & grid["valid_zero_drop"]].copy()
    stage = "zero_drop_relax_005"
    if valid.empty:
        valid = grid.loc[grid["valid_relax_005"]].copy()
        stage = "relax_005"
    if valid.empty:
        min_dropped_positive = int(grid["dropped_positive_total"].min())
        valid = grid.loc[grid["dropped_positive_total"] == min_dropped_positive].copy()
        stage = "min_dropped_positive"
    best = valid.sort_values(
        ["veto_FAR", "actual_drop_count", "global_drop_budget"],
        ascending=[True, False, True],
    ).iloc[0]
    return {
        "status": "valid" if stage != "min_dropped_positive" else "fallback",
        "selection_stage": stage,
        "global_drop_budget": float(best["global_drop_budget"]),
        "actual_drop_count": int(best["actual_drop_count"]),
    }


def _run_config(
    segments: pd.DataFrame,
    timeseries: pd.DataFrame,
    out_path: Path,
    verifier_weight: float,
    attr_weight: float,
    global_drop_budget: float,
    min_drop_k: int,
    max_drop_ratio: float,
    group_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ranked = _annotate_combined_scores(segments, verifier_weight=verifier_weight, attr_weight=attr_weight, sequence_col=group_col)
    dropped = _apply_global_budget(ranked, global_drop_budget=global_drop_budget, min_drop_k=min_drop_k, max_drop_ratio=max_drop_ratio)
    dropped.to_csv(out_path, index=False)
    scored = _apply_timeseries_veto(timeseries, dropped, group_col=group_col)
    return dropped, scored


def _run_one(
    name: str,
    timeseries_root: Path,
    strong_root: Path,
    old_attr_root: Path,
    seed_attr_root: Path,
    out_dir: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    run_out = out_dir / name
    run_out.mkdir(parents=True, exist_ok=True)

    attr_csv = _read_csv(_resolve_attr_csv(name, old_root=old_attr_root, seed_root=seed_attr_root))
    fit_segments = _read_csv(strong_root / name / "segment_fit_scored_strong.csv")
    val_segments = _read_csv(strong_root / name / "segment_val_scored_strong.csv")
    eval_segments = _read_csv(strong_root / name / "segment_eval_scored_strong.csv")
    val_ts = _read_csv(timeseries_root / name / "segment_verifier_val_timeseries.csv")
    eval_ts = _read_csv(timeseries_root / name / "segment_verifier_eval_timeseries.csv")

    thresholds = _fit_protection_thresholds(fit_segments)
    val_attr = attr_csv.loc[attr_csv["split"] == "val", ["segment_uid", "attr_score"]].copy()
    eval_attr = attr_csv.loc[attr_csv["split"] == "eval", ["segment_uid", "attr_score"]].copy()
    val_segments = _annotate_protected(val_segments.merge(val_attr, how="left", on="segment_uid"), thresholds=thresholds)
    eval_segments = _annotate_protected(eval_segments.merge(eval_attr, how="left", on="segment_uid"), thresholds=thresholds)
    val_segments["attr_score"] = val_segments["attr_score"].fillna(0.0)
    eval_segments["attr_score"] = eval_segments["attr_score"].fillna(0.0)

    verifier_weight = float(args.verifier_weight)
    attr_weight = float(args.attr_weight)
    min_drop_k = int(args.min_drop_k)
    max_drop_ratio = float(args.max_drop_ratio)

    grid = _grid_search(
        val_ts=val_ts,
        val_segments=val_segments,
        global_drop_budgets=_parse_float_list(str(args.global_drop_budgets)),
        verifier_weight=verifier_weight,
        attr_weight=attr_weight,
        min_drop_k=min_drop_k,
        max_drop_ratio=max_drop_ratio,
        label_col=str(args.label_col),
        group_col=str(args.group_col),
        time_col=str(args.time_col),
    )
    grid.to_csv(run_out / "global_budget_val_grid.csv", index=False)
    pick = _select_budget(grid)

    val_final_segments, val_scored = _run_config(
        segments=val_segments,
        timeseries=val_ts,
        out_path=run_out / "segment_val_globalbudget_scored.csv",
        verifier_weight=verifier_weight,
        attr_weight=attr_weight,
        global_drop_budget=float(pick["global_drop_budget"]),
        min_drop_k=min_drop_k,
        max_drop_ratio=max_drop_ratio,
        group_col=str(args.group_col),
    )
    eval_final_segments, eval_scored = _run_config(
        segments=eval_segments,
        timeseries=eval_ts,
        out_path=run_out / "segment_eval_globalbudget_scored.csv",
        verifier_weight=verifier_weight,
        attr_weight=attr_weight,
        global_drop_budget=float(pick["global_drop_budget"]),
        min_drop_k=min_drop_k,
        max_drop_ratio=max_drop_ratio,
        group_col=str(args.group_col),
    )
    val_scored.to_csv(run_out / "global_budget_veto_val_timeseries.csv", index=False)
    eval_scored.to_csv(run_out / "global_budget_veto_eval_timeseries.csv", index=False)

    q40_val = _q40_metrics(val_ts, label_col=str(args.label_col), group_col=str(args.group_col), time_col=str(args.time_col))
    veto_val = _lag_metrics(val_scored, label_col=str(args.label_col), group_col=str(args.group_col), time_col=str(args.time_col))
    q40_eval = _q40_metrics(eval_ts, label_col=str(args.label_col), group_col=str(args.group_col), time_col=str(args.time_col))
    veto_eval = _lag_metrics(eval_scored, label_col=str(args.label_col), group_col=str(args.group_col), time_col=str(args.time_col))

    eval_audit = _drop_audit(eval_final_segments)
    eval_audit.to_csv(run_out / "eval_drop_audit.csv", index=False)
    audit_counts = _audit_counts(eval_audit)

    row = {
        "run": name,
        "selection_status": str(pick["status"]),
        "selection_stage": str(pick["selection_stage"]),
        "verifier_weight": verifier_weight,
        "attr_weight": attr_weight,
        "global_drop_budget": float(pick["global_drop_budget"]),
        "actual_drop_count": int(pick["actual_drop_count"]),
        "q40_eval_recall": q40_eval["overall_recall"],
        "global_budget_eval_recall": veto_eval["overall_recall"],
        "q40_eval_FAR": q40_eval["FAR"],
        "global_budget_eval_FAR": veto_eval["FAR"],
        "q40_eval_zero_E_d_hat": q40_eval["zero_E_d_hat"],
        "global_budget_eval_zero_E_d_hat": veto_eval["zero_E_d_hat"],
        "q40_eval_pos_MAE": q40_eval["pos_MAE"],
        "global_budget_eval_pos_MAE": veto_eval["pos_MAE"],
        "dropped_positive_strong": int(audit_counts.get("dropped_positive_strong", 0)),
        "dropped_positive_weak": int(audit_counts.get("dropped_positive_weak", 0)),
        "dropped_false_positive_strong": int(audit_counts.get("dropped_false_positive_strong", 0)),
        "dropped_false_positive_weak": int(audit_counts.get("dropped_false_positive_weak", 0)),
        "n_eval_segments": int(len(eval_final_segments)),
        "n_eval_strong_segments": int((eval_final_segments["segment_is_strong"].to_numpy(dtype=np.float64) > 0).sum()),
        "n_eval_drop_candidates": int((eval_final_segments["drop_candidate"].to_numpy(dtype=np.float64) > 0).sum()),
    }
    _write_json(
        run_out / "global_budget_veto_report.json",
        {
            "run": name,
            "protection_thresholds": thresholds,
            "selection": pick,
            "q40_val_metrics": q40_val,
            "global_budget_val_metrics": veto_val,
            "q40_eval_metrics": q40_eval,
            "global_budget_eval_metrics": veto_eval,
            "outputs": {
                "grid": (run_out / "global_budget_val_grid.csv").as_posix(),
                "segment_val": (run_out / "segment_val_globalbudget_scored.csv").as_posix(),
                "segment_eval": (run_out / "segment_eval_globalbudget_scored.csv").as_posix(),
                "val_timeseries": (run_out / "global_budget_veto_val_timeseries.csv").as_posix(),
                "eval_timeseries": (run_out / "global_budget_veto_eval_timeseries.csv").as_posix(),
                "eval_drop_audit": (run_out / "eval_drop_audit.csv").as_posix(),
            },
        },
    )
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="r47d global weak-segment budget veto with protected segments.")
    parser.add_argument("--timeseries-root", default="outputs/r45c_q40_segment_proposal_verifier_smoke")
    parser.add_argument("--strong-root", default="outputs/r46b_q40_segment_strongkeep_veto_smoke")
    parser.add_argument("--old-attr-root", default="outputs/r47a_timesliver_attr_diagnostic_smoke_old")
    parser.add_argument("--seed-attr-root", default="outputs/r47a_timesliver_attr_diagnostic_smoke_seed")
    parser.add_argument("--runs", default="old,seed134_e2")
    parser.add_argument("--output-dir", default="outputs/r47d_timesliver_global_budget_veto")
    parser.add_argument("--label-col", default="d_true")
    parser.add_argument("--group-col", default="segment_id")
    parser.add_argument("--time-col", default="t")
    parser.add_argument("--verifier-weight", type=float, default=0.3)
    parser.add_argument("--attr-weight", type=float, default=0.7)
    parser.add_argument("--global-drop-budgets", default="0.05,0.10,0.20,0.30")
    parser.add_argument("--min-drop-k", type=int, default=1)
    parser.add_argument("--max-drop-ratio", type=float, default=0.30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timeseries_root = _path(args.timeseries_root)
    strong_root = _path(args.strong_root)
    old_attr_root = _path(args.old_attr_root)
    seed_attr_root = _path(args.seed_attr_root)
    out_dir = _path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for name in [part.strip() for part in str(args.runs).split(",") if part.strip()]:
        rows.append(_run_one(name, timeseries_root, strong_root, old_attr_root, seed_attr_root, out_dir, args))
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "global_budget_veto_summary.csv", index=False)
    print(summary.to_csv(index=False, float_format="%.6f"))
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
