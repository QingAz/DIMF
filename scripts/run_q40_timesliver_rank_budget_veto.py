#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

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


def _parse_weight_grid(text: str) -> List[Tuple[float, float]]:
    pairs: List[Tuple[float, float]] = []
    for part in str(text).split(","):
        raw = part.strip()
        if not raw:
            continue
        left, right = raw.split(":")
        weight_v = float(left)
        weight_attr = float(right)
        total = weight_v + weight_attr
        if total <= 0:
            continue
        pairs.append((weight_v / total, weight_attr / total))
    dedup: List[Tuple[float, float]] = []
    seen = set()
    for pair in pairs:
        key = (round(pair[0], 8), round(pair[1], 8))
        if key not in seen:
            seen.add(key)
            dedup.append(pair)
    return dedup


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


def _annotate_combined_scores(
    segment_frame: pd.DataFrame,
    verifier_weight: float,
    attr_weight: float,
    sequence_col: str,
) -> pd.DataFrame:
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


def _drop_weak_segments(segment_frame: pd.DataFrame, drop_budget: float, sequence_col: str) -> pd.DataFrame:
    out = segment_frame.copy()
    out["drop_weak_segment"] = 0
    weak_mask = out["segment_is_strong"].fillna(0).to_numpy(dtype=np.float64) <= 0
    weak = out.loc[weak_mask].copy()
    if weak.empty:
        return out
    drop_indices: List[int] = []
    for _, seq in weak.groupby(sequence_col, sort=False):
        n = len(seq)
        k = int(math.floor(float(drop_budget) * float(n) + 0.5))
        if k <= 0:
            continue
        chosen = seq.sort_values(["combined_score", "attr_score", "verifier_segment_confidence", "q40_segment_index"], ascending=[True, True, True, True]).head(k)
        drop_indices.extend(chosen.index.tolist())
    if drop_indices:
        out.loc[drop_indices, "drop_weak_segment"] = 1
    return out


def _apply_rank_budget_veto(timeseries: pd.DataFrame, segments: pd.DataFrame, group_col: str) -> pd.DataFrame:
    seg_meta = segments[
        [group_col, "q40_segment_index", "segment_is_strong", "rank_v", "rank_attr", "combined_score", "drop_weak_segment"]
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
    return {
        str(row["group"]): int(row["count"])
        for row in audit.to_dict(orient="records")
    }


def _grid_search(
    val_ts: pd.DataFrame,
    val_segments: pd.DataFrame,
    drop_budgets: Sequence[float],
    weights: Sequence[Tuple[float, float]],
    label_col: str,
    group_col: str,
    time_col: str,
) -> pd.DataFrame:
    q40 = _q40_metrics(val_ts, label_col=label_col, group_col=group_col, time_col=time_col)
    q40_recall = float(q40["overall_recall"])
    rows: List[Dict[str, Any]] = []
    for verifier_weight, attr_weight in weights:
        ranked = _annotate_combined_scores(
            val_segments,
            verifier_weight=verifier_weight,
            attr_weight=attr_weight,
            sequence_col=group_col,
        )
        for budget in drop_budgets:
            dropped = _drop_weak_segments(ranked, drop_budget=float(budget), sequence_col=group_col)
            audit = _audit_counts(_drop_audit(dropped))
            scored = _apply_rank_budget_veto(val_ts, dropped, group_col=group_col)
            metrics = _lag_metrics(scored, label_col=label_col, group_col=group_col, time_col=time_col)
            rows.append(
                {
                    "verifier_weight": float(verifier_weight),
                    "attr_weight": float(attr_weight),
                    "drop_budget": float(budget),
                    "q40_val_recall": q40_recall,
                    "veto_recall": float(metrics["overall_recall"]),
                    "veto_FAR": float(metrics["FAR"]),
                    "veto_zero_E_d_hat": float(metrics["zero_E_d_hat"]),
                    "veto_pos_MAE": float(metrics["pos_MAE"]),
                    "recall_delta_vs_q40": float(metrics["overall_recall"] - q40_recall),
                    "dropped_positive_total": int(audit.get("dropped_positive_strong", 0) + audit.get("dropped_positive_weak", 0)),
                    "dropped_positive_strong": int(audit.get("dropped_positive_strong", 0)),
                    "dropped_positive_weak": int(audit.get("dropped_positive_weak", 0)),
                    "dropped_false_positive_strong": int(audit.get("dropped_false_positive_strong", 0)),
                    "dropped_false_positive_weak": int(audit.get("dropped_false_positive_weak", 0)),
                    "valid_relax_005": bool(metrics["overall_recall"] >= q40_recall - 0.05),
                    "valid_relax_010": bool(metrics["overall_recall"] >= q40_recall - 0.10),
                }
            )
    return pd.DataFrame(rows)


def _select_config(grid: pd.DataFrame) -> Dict[str, Any]:
    stage = "relax_005"
    valid = grid.loc[grid["valid_relax_005"]].copy()
    if valid.empty:
        stage = "relax_010"
        valid = grid.loc[grid["valid_relax_010"]].copy()
    if valid.empty:
        stage = "fallback_all"
        valid = grid.copy()
    min_drop = int(valid["dropped_positive_total"].min())
    valid = valid.loc[valid["dropped_positive_total"] == min_drop].copy()
    best = valid.sort_values(
        ["veto_FAR", "dropped_positive_weak", "drop_budget", "attr_weight"],
        ascending=[True, True, True, False],
    ).iloc[0]
    status = "valid" if stage != "fallback_all" else "no_valid_threshold"
    return {
        "status": status,
        "selection_stage": stage,
        "verifier_weight": float(best["verifier_weight"]),
        "attr_weight": float(best["attr_weight"]),
        "drop_budget": float(best["drop_budget"]),
    }


def _run_config(
    name: str,
    split: str,
    segments: pd.DataFrame,
    timeseries: pd.DataFrame,
    out_path: Path,
    verifier_weight: float,
    attr_weight: float,
    drop_budget: float,
    group_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ranked = _annotate_combined_scores(segments, verifier_weight=verifier_weight, attr_weight=attr_weight, sequence_col=group_col)
    dropped = _drop_weak_segments(ranked, drop_budget=drop_budget, sequence_col=group_col)
    dropped["run"] = name
    dropped["split"] = split
    dropped["verifier_weight"] = float(verifier_weight)
    dropped["attr_weight"] = float(attr_weight)
    dropped["drop_budget"] = float(drop_budget)
    dropped.to_csv(out_path, index=False)
    scored = _apply_rank_budget_veto(timeseries, dropped, group_col=group_col)
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
    val_ts = _read_csv(timeseries_root / name / "segment_verifier_val_timeseries.csv")
    eval_ts = _read_csv(timeseries_root / name / "segment_verifier_eval_timeseries.csv")
    val_segments = _read_csv(strong_root / name / "segment_val_scored_strong.csv")
    eval_segments = _read_csv(strong_root / name / "segment_eval_scored_strong.csv")

    val_attr = attr_csv.loc[attr_csv["split"] == "val", ["segment_uid", "attr_score"]].copy()
    eval_attr = attr_csv.loc[attr_csv["split"] == "eval", ["segment_uid", "attr_score"]].copy()
    val_segments = val_segments.merge(val_attr, how="left", on="segment_uid")
    eval_segments = eval_segments.merge(eval_attr, how="left", on="segment_uid")
    val_segments["attr_score"] = val_segments["attr_score"].fillna(0.0)
    eval_segments["attr_score"] = eval_segments["attr_score"].fillna(0.0)

    grid = _grid_search(
        val_ts=val_ts,
        val_segments=val_segments,
        drop_budgets=_parse_float_list(str(args.drop_budgets)),
        weights=_parse_weight_grid(str(args.weight_grid)),
        label_col=str(args.label_col),
        group_col=str(args.group_col),
        time_col=str(args.time_col),
    )
    grid.to_csv(run_out / "rank_budget_val_grid.csv", index=False)
    pick = _select_config(grid)

    val_final_segments, val_scored = _run_config(
        name=name,
        split="val",
        segments=val_segments,
        timeseries=val_ts,
        out_path=run_out / "segment_val_rankbudget_scored.csv",
        verifier_weight=float(pick["verifier_weight"]),
        attr_weight=float(pick["attr_weight"]),
        drop_budget=float(pick["drop_budget"]),
        group_col=str(args.group_col),
    )
    eval_final_segments, eval_scored = _run_config(
        name=name,
        split="eval",
        segments=eval_segments,
        timeseries=eval_ts,
        out_path=run_out / "segment_eval_rankbudget_scored.csv",
        verifier_weight=float(pick["verifier_weight"]),
        attr_weight=float(pick["attr_weight"]),
        drop_budget=float(pick["drop_budget"]),
        group_col=str(args.group_col),
    )
    val_scored.to_csv(run_out / "rank_budget_veto_val_timeseries.csv", index=False)
    eval_scored.to_csv(run_out / "rank_budget_veto_eval_timeseries.csv", index=False)

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
        "verifier_weight": float(pick["verifier_weight"]),
        "attr_weight": float(pick["attr_weight"]),
        "drop_budget": float(pick["drop_budget"]),
        "q40_eval_recall": q40_eval["overall_recall"],
        "rank_budget_eval_recall": veto_eval["overall_recall"],
        "q40_eval_FAR": q40_eval["FAR"],
        "rank_budget_eval_FAR": veto_eval["FAR"],
        "q40_eval_zero_E_d_hat": q40_eval["zero_E_d_hat"],
        "rank_budget_eval_zero_E_d_hat": veto_eval["zero_E_d_hat"],
        "q40_eval_pos_MAE": q40_eval["pos_MAE"],
        "rank_budget_eval_pos_MAE": veto_eval["pos_MAE"],
        "dropped_positive_strong": int(audit_counts.get("dropped_positive_strong", 0)),
        "dropped_positive_weak": int(audit_counts.get("dropped_positive_weak", 0)),
        "dropped_false_positive_strong": int(audit_counts.get("dropped_false_positive_strong", 0)),
        "dropped_false_positive_weak": int(audit_counts.get("dropped_false_positive_weak", 0)),
        "n_eval_segments": int(len(eval_final_segments)),
        "n_eval_strong_segments": int((eval_final_segments["segment_is_strong"].to_numpy(dtype=np.float64) > 0).sum()),
    }
    _write_json(
        run_out / "rank_budget_veto_report.json",
        {
            "run": name,
            "selection": pick,
            "q40_val_metrics": q40_val,
            "rank_budget_val_metrics": veto_val,
            "q40_eval_metrics": q40_eval,
            "rank_budget_eval_metrics": veto_eval,
            "outputs": {
                "grid": (run_out / "rank_budget_val_grid.csv").as_posix(),
                "segment_val": (run_out / "segment_val_rankbudget_scored.csv").as_posix(),
                "segment_eval": (run_out / "segment_eval_rankbudget_scored.csv").as_posix(),
                "val_timeseries": (run_out / "rank_budget_veto_val_timeseries.csv").as_posix(),
                "eval_timeseries": (run_out / "rank_budget_veto_eval_timeseries.csv").as_posix(),
                "eval_drop_audit": (run_out / "eval_drop_audit.csv").as_posix(),
            },
        },
    )
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="r47c TimeSliver rank-budget veto on weak q40 segments.")
    parser.add_argument("--timeseries-root", default="outputs/r45c_q40_segment_proposal_verifier_smoke")
    parser.add_argument("--strong-root", default="outputs/r46b_q40_segment_strongkeep_veto_smoke")
    parser.add_argument("--old-attr-root", default="outputs/r47a_timesliver_attr_diagnostic_smoke_old")
    parser.add_argument("--seed-attr-root", default="outputs/r47a_timesliver_attr_diagnostic_smoke_seed")
    parser.add_argument("--runs", default="old,seed134_e2")
    parser.add_argument("--output-dir", default="outputs/r47c_timesliver_rank_budget_veto")
    parser.add_argument("--label-col", default="d_true")
    parser.add_argument("--group-col", default="segment_id")
    parser.add_argument("--time-col", default="t")
    parser.add_argument("--drop-budgets", default="0.05,0.10,0.15")
    parser.add_argument("--weight-grid", default="0.5:0.5,0.3:0.7,0.7:0.3")
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
    summary.to_csv(out_dir / "rank_budget_veto_summary.csv", index=False)
    print(summary.to_csv(index=False, float_format="%.6f"))
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
