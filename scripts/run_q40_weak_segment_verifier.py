#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.postprocess.q40_final_block_lag_selector import (  # noqa: E402
    Q40FinalSelectorConfig,
    selection_metrics as q40_selection_metrics,
)
from src.postprocess.q40_segment_proposal_verifier import (  # noqa: E402
    Q40SegmentProposalVerifier,
    build_segment_dataset,
    segment_feature_columns,
)
from src.postprocess.q40_common import (  # noqa: E402
    FeatureNormalizer,
    fit_feature_normalizer,
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


def _parse_threshold_grid(text: str) -> List[float]:
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


def _resolve_attr_csv(name: str, old_root: Path, seed_root: Path) -> Path:
    if name == "old":
        return old_root / "old" / "timesliver_segment_attr_scores.csv"
    if name == "seed134_e2":
        return seed_root / "seed134_e2" / "timesliver_segment_attr_scores.csv"
    raise ValueError(f"Unknown run name: {name}")


def _normalized_rank(values: pd.Series) -> pd.Series:
    n = len(values)
    if n <= 1:
        return pd.Series(np.full(n, 0.5, dtype=np.float64), index=values.index)
    ranks = values.rank(method="average", ascending=True).to_numpy(dtype=np.float64)
    normalized = (ranks - 1.0) / float(n - 1)
    return pd.Series(normalized, index=values.index)


def _annotate_overlap_ratio(segment_frame: pd.DataFrame, timeseries: pd.DataFrame, group_col: str, time_col: str) -> pd.DataFrame:
    out = segment_frame.copy()
    overlap_points: List[int] = []
    overlap_ratio: List[float] = []
    max_d_true: List[float] = []
    for row in out.itertuples(index=False):
        mask = (
            (timeseries[group_col] == getattr(row, group_col))
            & (timeseries[time_col] >= float(row.start_t))
            & (timeseries[time_col] <= float(row.end_t))
        )
        d_true = timeseries.loc[mask, "d_true"].to_numpy(dtype=np.float64)
        pos = d_true > 0
        overlap_points.append(int(pos.sum()))
        overlap_ratio.append(float(pos.mean()) if d_true.size else 0.0)
        max_d_true.append(float(np.nanmax(d_true)) if d_true.size else 0.0)
    out["overlap_points"] = overlap_points
    out["overlap_ratio"] = overlap_ratio
    out["max_d_true_inside"] = max_d_true
    return out


def _merge_attr_and_strong(
    split_name: str,
    segment_frame: pd.DataFrame,
    strong_frame: pd.DataFrame,
    attr_csv: pd.DataFrame,
    group_col: str,
) -> pd.DataFrame:
    out = segment_frame.copy()
    strong_cols = ["segment_uid", "segment_is_strong", "segment_is_weak", "verifier_segment_confidence"]
    keep_cols = [col for col in strong_cols if col in strong_frame.columns]
    if keep_cols:
        out = out.merge(strong_frame[keep_cols], how="left", on="segment_uid")
    split_attr = attr_csv.loc[attr_csv["split"] == split_name, ["segment_uid", "attr_score"]].copy()
    out = out.merge(split_attr, how="left", on="segment_uid")
    if "segment_is_strong" not in out.columns:
        out["segment_is_strong"] = 0
    else:
        out["segment_is_strong"] = out["segment_is_strong"].fillna(0).astype(int)
    if "segment_is_weak" not in out.columns:
        out["segment_is_weak"] = 1 - out["segment_is_strong"]
    else:
        out["segment_is_weak"] = out["segment_is_weak"].fillna(1 - out["segment_is_strong"]).astype(int)
    out["attr_score"] = out["attr_score"].fillna(0.0)
    if "verifier_segment_confidence" not in out.columns:
        out["verifier_segment_confidence"] = 0.5
    weak_mask = out["segment_is_weak"].to_numpy(dtype=np.float64) > 0
    out["rank_attr"] = 1.0
    out["rank_v_old"] = 1.0
    if weak_mask.any():
        weak = out.loc[weak_mask].copy()
        weak["rank_attr"] = weak.groupby(group_col)["attr_score"].transform(_normalized_rank)
        weak["rank_v_old"] = weak.groupby(group_col)["verifier_segment_confidence"].transform(_normalized_rank)
        out.loc[weak.index, "rank_attr"] = weak["rank_attr"].to_numpy(dtype=np.float64)
        out.loc[weak.index, "rank_v_old"] = weak["rank_v_old"].to_numpy(dtype=np.float64)
    out["combined_score"] = 0.3 * out["rank_v_old"].to_numpy(dtype=np.float64) + 0.7 * out["rank_attr"].to_numpy(dtype=np.float64)
    return out


def _feature_columns(frame: pd.DataFrame) -> List[str]:
    base = segment_feature_columns(frame)
    excluded = {
        "segment_is_strong",
        "segment_is_weak",
        "verifier_segment_keep",
        "verifier_segment_threshold",
        "overlap_points",
        "overlap_ratio",
        "max_d_true_inside",
        "soft_label",
        "train_weight",
    }
    cols = [col for col in base if col not in excluded]
    for extra in ["attr_score", "rank_attr", "combined_score"]:
        if extra in frame.columns and extra not in cols:
            cols.append(extra)
    return cols


def _weak_subset(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame["segment_is_weak"].to_numpy(dtype=np.float64) > 0].reset_index(drop=True).copy()


def _prepare_targets(frame: pd.DataFrame, label_mode: str, positive_weight: float) -> pd.DataFrame:
    out = frame.copy()
    if str(label_mode) == "soft":
        out["soft_label"] = out["overlap_ratio"].clip(lower=0.0, upper=1.0)
    else:
        out["soft_label"] = (out["segment_label"].to_numpy(dtype=np.float64) > 0).astype(np.float64)
    positive_mask = out["overlap_points"].to_numpy(dtype=np.float64) > 0
    out["train_weight"] = np.where(positive_mask, float(positive_weight), 1.0)
    return out


def _tensor_dataset(frame: pd.DataFrame, normalizer: FeatureNormalizer, feature_columns: Sequence[str]) -> TensorDataset:
    features = normalizer.transform(frame[list(feature_columns)]).astype(np.float32)
    labels = frame["soft_label"].to_numpy(dtype=np.float32)
    weights = frame["train_weight"].to_numpy(dtype=np.float32)
    return TensorDataset(torch.from_numpy(features), torch.from_numpy(labels), torch.from_numpy(weights))


def _mean_loss(model: Q40SegmentProposalVerifier, loader: DataLoader, device: str) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for xb, yb, wb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            wb = wb.to(device)
            logits = model(xb)
            loss_vec = F.binary_cross_entropy_with_logits(logits, yb, reduction="none")
            loss = (loss_vec * wb).sum() / torch.clamp(wb.sum(), min=1.0)
            total += float(loss.item()) * int(yb.numel())
            count += int(yb.numel())
    return total / max(count, 1)


def _train_model(
    fit_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    hidden_dim: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
    device: str,
) -> tuple[Q40SegmentProposalVerifier, FeatureNormalizer, pd.DataFrame]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    normalizer = fit_feature_normalizer(fit_frame, feature_columns=feature_columns)
    fit_ds = _tensor_dataset(fit_frame, normalizer=normalizer, feature_columns=feature_columns)
    val_ds = _tensor_dataset(val_frame, normalizer=normalizer, feature_columns=feature_columns)
    fit_loader = DataLoader(fit_ds, batch_size=int(batch_size), shuffle=True)
    fit_eval_loader = DataLoader(fit_ds, batch_size=int(batch_size), shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=int(batch_size), shuffle=False)
    model = Q40SegmentProposalVerifier(len(list(feature_columns)), hidden_dim=int(hidden_dim), dropout=float(dropout)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    rows: List[Dict[str, Any]] = []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        for xb, yb, wb in fit_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            wb = wb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss_vec = F.binary_cross_entropy_with_logits(logits, yb, reduction="none")
            loss = (loss_vec * wb).sum() / torch.clamp(wb.sum(), min=1.0)
            loss.backward()
            optimizer.step()
        rows.append(
            {
                "epoch": int(epoch),
                "fit_loss": _mean_loss(model, fit_eval_loader, device=device),
                "val_loss": _mean_loss(model, val_loader, device=device),
            }
        )
    return model, normalizer, pd.DataFrame(rows)


def _score_weak_segments(
    model: Q40SegmentProposalVerifier,
    frame: pd.DataFrame,
    normalizer: FeatureNormalizer,
    feature_columns: Sequence[str],
    device: str,
) -> pd.DataFrame:
    out = frame.copy()
    out["weak_verifier_confidence"] = 1.0
    weak = _weak_subset(out)
    if weak.empty:
        return out
    xb = torch.from_numpy(normalizer.transform(weak[list(feature_columns)]).astype(np.float32)).to(device)
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(xb)).cpu().numpy().astype(np.float64)
    out.loc[weak.index, "weak_verifier_confidence"] = probs
    return out


def _apply_threshold_to_segments(frame: pd.DataFrame, theta: float) -> pd.DataFrame:
    out = frame.copy()
    strong = out["segment_is_strong"].to_numpy(dtype=np.float64) > 0
    weak_keep = out["weak_verifier_confidence"].to_numpy(dtype=np.float64) >= float(theta)
    out["weak_verifier_threshold"] = float(theta)
    out["weak_verifier_keep"] = np.where(strong, 1, weak_keep.astype(int))
    return out


def _apply_segments_to_timeseries(frame: pd.DataFrame, segment_frame: pd.DataFrame, group_col: str, time_col: str) -> pd.DataFrame:
    out = frame.copy()
    out["weak_verifier_segment_index"] = -1
    out["weak_verifier_confidence"] = 1.0
    out["weak_verifier_keep"] = 0
    q40_selected = out["q40_selected"].to_numpy(dtype=np.float64) > 0
    q40_d_hat = out["q40_d_hat"].to_numpy(dtype=np.float64)
    keep_mask = np.zeros(len(out), dtype=bool)
    for row in segment_frame.itertuples(index=False):
        mask = (
            (out[group_col] == getattr(row, group_col))
            & (out[time_col] >= float(row.start_t))
            & (out[time_col] <= float(row.end_t))
        )
        out.loc[mask, "weak_verifier_segment_index"] = int(row.q40_segment_index)
        out.loc[mask, "weak_verifier_confidence"] = float(row.weak_verifier_confidence)
        out.loc[mask, "weak_verifier_keep"] = int(row.weak_verifier_keep)
        if int(row.weak_verifier_keep) > 0:
            keep_mask |= mask.to_numpy(dtype=bool)
    selected_final = q40_selected & keep_mask
    out["weak_verifier_selected_final"] = selected_final.astype(int)
    out["weak_verifier_d_hat_final"] = np.where(selected_final, q40_d_hat, 0.0)
    return out


def _end_to_end_metrics(frame: pd.DataFrame, label_col: str, group_col: str, time_col: str) -> Dict[str, Any]:
    proxy = frame.copy()
    proxy["q40_final_selected"] = proxy["weak_verifier_selected_final"].to_numpy(dtype=np.float64)
    proxy["d_hat"] = proxy["weak_verifier_d_hat_final"].to_numpy(dtype=np.float64)
    cfg = Q40FinalSelectorConfig(label_col=label_col, group_col=group_col, time_col=time_col)
    return q40_selection_metrics(proxy, cfg)


def _q40_metrics(frame: pd.DataFrame, label_col: str, group_col: str, time_col: str) -> Dict[str, Any]:
    proxy = frame.copy()
    proxy["q40_final_selected"] = proxy["q40_selected"].to_numpy(dtype=np.float64)
    proxy["d_hat"] = proxy["q40_d_hat"].to_numpy(dtype=np.float64)
    cfg = Q40FinalSelectorConfig(label_col=label_col, group_col=group_col, time_col=time_col)
    return q40_selection_metrics(proxy, cfg)


def _weak_distribution(frame: pd.DataFrame, split_name: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    weak = _weak_subset(frame)
    for group_name, mask in [
        ("positive_weak_segment", weak["overlap_points"].to_numpy(dtype=np.float64) > 0),
        ("false_positive_weak_segment", weak["overlap_points"].to_numpy(dtype=np.float64) <= 0),
    ]:
        values = weak.loc[mask, "weak_verifier_confidence"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "split": split_name,
                "group": group_name,
                "count": int(values.size),
                "v_p10": float(np.nanpercentile(values, 10.0)) if values.size else float("nan"),
                "v_p50": float(np.nanpercentile(values, 50.0)) if values.size else float("nan"),
                "v_p90": float(np.nanpercentile(values, 90.0)) if values.size else float("nan"),
            }
        )
    return rows


def _drop_audit(frame: pd.DataFrame) -> pd.DataFrame:
    strong = frame["segment_is_strong"].to_numpy(dtype=np.float64) > 0
    positive = frame["overlap_points"].to_numpy(dtype=np.float64) > 0
    keep = frame["weak_verifier_keep"].to_numpy(dtype=np.float64) > 0
    dropped = ~keep
    rows: List[Dict[str, Any]] = []
    for name, mask in [
        ("dropped_positive_strong", dropped & positive & strong),
        ("dropped_positive_weak", dropped & positive & (~strong)),
        ("dropped_false_positive_strong", dropped & (~positive) & strong),
        ("dropped_false_positive_weak", dropped & (~positive) & (~strong)),
    ]:
        rows.append({"group": name, "count": int(mask.sum())})
    return pd.DataFrame(rows)


def _audit_count(audit: pd.DataFrame, key: str) -> int:
    match = audit.loc[audit["group"] == key, "count"]
    return int(match.iloc[0]) if not match.empty else 0


def _pooled_group_frame(frames: Iterable[pd.DataFrame], run_names: Iterable[str], group_col: str) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for name, frame in zip(run_names, frames):
        if frame.empty:
            continue
        copy = frame.copy()
        copy[group_col] = copy[group_col].astype(str).map(lambda x: f"{name}::{x}")
        rows.append(copy)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _select_threshold(payloads: List[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    eligible = [
        item for item in payloads
        if int(item["val_weak_count"]) >= int(args.val_min_weak_candidates)
        and int(item["val_weak_positive_count"]) > 0
        and int(item["val_weak_negative_count"]) > 0
    ]
    thresholds = _parse_threshold_grid(str(args.threshold_grid))
    if not eligible:
        return {
            "status": "no_eligible_val",
            "selection_stage": "fallback_default",
            "theta": float(args.default_threshold),
            "eligible_runs": [],
        }
    pooled_q40 = _pooled_group_frame([item["val_timeseries"] for item in eligible], [item["run"] for item in eligible], group_col=str(args.group_col))
    q40_metrics = _q40_metrics(pooled_q40, label_col=str(args.label_col), group_col=str(args.group_col), time_col=str(args.time_col))
    q40_recall = float(q40_metrics["overall_recall"])
    rows: List[Dict[str, Any]] = []
    for theta in thresholds:
        ts_frames: List[pd.DataFrame] = []
        for item in eligible:
            seg = _apply_threshold_to_segments(item["val_scored_segments"], theta=float(theta))
            ts = _apply_segments_to_timeseries(item["val_timeseries"], seg, group_col=str(args.group_col), time_col=str(args.time_col))
            ts_frames.append(ts)
        pooled = _pooled_group_frame(ts_frames, [item["run"] for item in eligible], group_col=str(args.group_col))
        metrics = _end_to_end_metrics(pooled, label_col=str(args.label_col), group_col=str(args.group_col), time_col=str(args.time_col))
        rows.append(
            {
                "theta": float(theta),
                "q40_recall": q40_recall,
                "recall": float(metrics["overall_recall"]),
                "FAR": float(metrics["FAR"]),
                "recall_drop": float(q40_recall - float(metrics["overall_recall"])),
                "valid_relax_005": bool(float(q40_recall - metrics["overall_recall"]) <= 0.05),
            }
        )
    grid = pd.DataFrame(rows)
    valid = grid.loc[grid["valid_relax_005"]].copy()
    stage = "relax_005"
    if valid.empty:
        valid = grid.sort_values(["recall_drop", "FAR", "theta"], ascending=[True, True, True]).head(1).copy()
        stage = "min_recall_drop"
    best = valid.sort_values(["FAR", "theta"], ascending=[True, True]).iloc[0]
    return {
        "status": "valid" if stage == "relax_005" else "fallback",
        "selection_stage": stage,
        "theta": float(best["theta"]),
        "eligible_runs": [item["run"] for item in eligible],
        "grid": grid,
    }


def _run_one(name: str, proposal_root: Path, strong_root: Path, old_attr_root: Path, seed_attr_root: Path, out_dir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    run_out = out_dir / name
    run_out.mkdir(parents=True, exist_ok=True)

    run_in = proposal_root / name
    fit_ts = _read_csv(run_in / "q40_fixed_fit_timeseries.csv")
    val_ts = _read_csv(run_in / "q40_fixed_val_timeseries.csv")
    eval_ts = _read_csv(run_in / "q40_fixed_eval_timeseries.csv")
    strong_fit = _read_csv(strong_root / name / "segment_fit_scored_strong.csv")
    strong_val = _read_csv(strong_root / name / "segment_val_scored_strong.csv")
    strong_eval = _read_csv(strong_root / name / "segment_eval_scored_strong.csv")
    attr_csv = _read_csv(_resolve_attr_csv(name, old_root=old_attr_root, seed_root=seed_attr_root))

    fit_seg = build_segment_dataset(fit_ts, group_col=str(args.group_col), time_col=str(args.time_col), merge_gap=int(args.merge_gap), min_len=int(args.min_len), include_q40_segment_features=True)
    val_seg = build_segment_dataset(val_ts, group_col=str(args.group_col), time_col=str(args.time_col), merge_gap=int(args.merge_gap), min_len=int(args.min_len), include_q40_segment_features=True)
    eval_seg = build_segment_dataset(eval_ts, group_col=str(args.group_col), time_col=str(args.time_col), merge_gap=int(args.merge_gap), min_len=int(args.min_len), include_q40_segment_features=True)

    fit_seg = _annotate_overlap_ratio(_merge_attr_and_strong("fit", fit_seg, strong_fit, attr_csv, group_col=str(args.group_col)), fit_ts, group_col=str(args.group_col), time_col=str(args.time_col))
    val_seg = _annotate_overlap_ratio(_merge_attr_and_strong("val", val_seg, strong_val, attr_csv, group_col=str(args.group_col)), val_ts, group_col=str(args.group_col), time_col=str(args.time_col))
    eval_seg = _annotate_overlap_ratio(_merge_attr_and_strong("eval", eval_seg, strong_eval, attr_csv, group_col=str(args.group_col)), eval_ts, group_col=str(args.group_col), time_col=str(args.time_col))

    fit_seg = _prepare_targets(fit_seg, label_mode=str(args.label_mode), positive_weight=float(args.positive_weight))
    val_seg = _prepare_targets(val_seg, label_mode=str(args.label_mode), positive_weight=float(args.positive_weight))
    eval_seg = _prepare_targets(eval_seg, label_mode=str(args.label_mode), positive_weight=float(args.positive_weight))

    fit_seg.to_csv(run_out / "segment_fit_candidates.csv", index=False)
    val_seg.to_csv(run_out / "segment_val_candidates.csv", index=False)
    eval_seg.to_csv(run_out / "segment_eval_candidates.csv", index=False)

    fit_weak = _weak_subset(fit_seg)
    val_weak = _weak_subset(val_seg)
    eval_weak = _weak_subset(eval_seg)
    if fit_weak.empty:
        raise ValueError(f"{name}: weak segment verifier requires non-empty fit weak table")
    val_train = val_weak if not val_weak.empty else fit_weak.copy()

    features = _feature_columns(fit_seg)
    model, normalizer, history = _train_model(
        fit_weak,
        val_train,
        feature_columns=features,
        hidden_dim=int(args.hidden_dim),
        dropout=float(args.dropout),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        seed=int(args.seed),
        device=str(args.device),
    )
    history.to_csv(run_out / "weak_verifier_train_history.csv", index=False)

    fit_scored = _score_weak_segments(model, fit_seg, normalizer=normalizer, feature_columns=features, device=str(args.device))
    val_scored = _score_weak_segments(model, val_seg, normalizer=normalizer, feature_columns=features, device=str(args.device))
    eval_scored = _score_weak_segments(model, eval_seg, normalizer=normalizer, feature_columns=features, device=str(args.device))
    pd.DataFrame(_weak_distribution(fit_scored, "fit") + _weak_distribution(val_scored, "val") + _weak_distribution(eval_scored, "eval")).to_csv(run_out / "weak_score_distribution.csv", index=False)

    return {
        "run": name,
        "fit_scored_segments": fit_scored,
        "val_scored_segments": val_scored,
        "eval_scored_segments": eval_scored,
        "fit_timeseries": fit_ts,
        "val_timeseries": val_ts,
        "eval_timeseries": eval_ts,
        "feature_columns": features,
        "val_weak_count": int(len(val_weak)),
        "val_weak_positive_count": int((val_weak["overlap_points"].to_numpy(dtype=np.float64) > 0).sum()),
        "val_weak_negative_count": int((val_weak["overlap_points"].to_numpy(dtype=np.float64) <= 0).sum()),
        "history_path": (run_out / "weak_verifier_train_history.csv").as_posix(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="r48 weak-only segment verifier with attr features.")
    parser.add_argument("--proposal-root", default="outputs/r45a_q40_point_proposal_verifier_smoke3")
    parser.add_argument("--strong-root", default="outputs/r46b_q40_segment_strongkeep_veto_smoke")
    parser.add_argument("--old-attr-root", default="outputs/r47a_timesliver_attr_diagnostic_smoke_old")
    parser.add_argument("--seed-attr-root", default="outputs/r47a_timesliver_attr_diagnostic_smoke_seed")
    parser.add_argument("--runs", default="old,seed134_e2")
    parser.add_argument("--output-dir", default="outputs/r48_q40_weak_segment_verifier")
    parser.add_argument("--label-col", default="d_true")
    parser.add_argument("--group-col", default="segment_id")
    parser.add_argument("--time-col", default="t")
    parser.add_argument("--merge-gap", type=int, default=1)
    parser.add_argument("--min-len", type=int, default=1)
    parser.add_argument("--label-mode", choices=["hard", "soft"], default="hard")
    parser.add_argument("--positive-weight", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--threshold-grid", default="0.05:0.95:0.05")
    parser.add_argument("--default-threshold", type=float, default=0.5)
    parser.add_argument("--val-min-weak-candidates", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    proposal_root = _path(args.proposal_root)
    strong_root = _path(args.strong_root)
    old_attr_root = _path(args.old_attr_root)
    seed_attr_root = _path(args.seed_attr_root)
    out_dir = _path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payloads = []
    for name in [part.strip() for part in str(args.runs).split(",") if part.strip()]:
        payloads.append(_run_one(name, proposal_root, strong_root, old_attr_root, seed_attr_root, out_dir, args))

    threshold_pick = _select_threshold(payloads, args)
    if "grid" in threshold_pick:
        threshold_pick["grid"].to_csv(out_dir / "pooled_val_threshold_grid.csv", index=False)

    rows: List[Dict[str, Any]] = []
    for payload in payloads:
        run = payload["run"]
        run_out = out_dir / run
        theta = float(threshold_pick["theta"])
        fit_seg = _apply_threshold_to_segments(payload["fit_scored_segments"], theta=theta)
        val_seg = _apply_threshold_to_segments(payload["val_scored_segments"], theta=theta)
        eval_seg = _apply_threshold_to_segments(payload["eval_scored_segments"], theta=theta)
        fit_ts = _apply_segments_to_timeseries(payload["fit_timeseries"], fit_seg, group_col=str(args.group_col), time_col=str(args.time_col))
        val_ts = _apply_segments_to_timeseries(payload["val_timeseries"], val_seg, group_col=str(args.group_col), time_col=str(args.time_col))
        eval_ts = _apply_segments_to_timeseries(payload["eval_timeseries"], eval_seg, group_col=str(args.group_col), time_col=str(args.time_col))

        fit_seg.to_csv(run_out / "segment_fit_scored.csv", index=False)
        val_seg.to_csv(run_out / "segment_val_scored.csv", index=False)
        eval_seg.to_csv(run_out / "segment_eval_scored.csv", index=False)
        fit_ts.to_csv(run_out / "weak_verifier_fit_timeseries.csv", index=False)
        val_ts.to_csv(run_out / "weak_verifier_val_timeseries.csv", index=False)
        eval_ts.to_csv(run_out / "weak_verifier_eval_timeseries.csv", index=False)

        q40_eval = _q40_metrics(payload["eval_timeseries"], label_col=str(args.label_col), group_col=str(args.group_col), time_col=str(args.time_col))
        verifier_eval = _end_to_end_metrics(eval_ts, label_col=str(args.label_col), group_col=str(args.group_col), time_col=str(args.time_col))
        audit = _drop_audit(eval_seg)
        audit.to_csv(run_out / "eval_drop_audit.csv", index=False)

        report = {
            "run": run,
            "feature_columns": payload["feature_columns"],
            "label_mode": str(args.label_mode),
            "positive_weight": float(args.positive_weight),
            "threshold_pick": {k: v for k, v in threshold_pick.items() if k != "grid"},
            "val_weak_count": int(payload["val_weak_count"]),
            "val_weak_positive_count": int(payload["val_weak_positive_count"]),
            "val_weak_negative_count": int(payload["val_weak_negative_count"]),
            "lag_metrics": {
                "q40_eval": q40_eval,
                "weak_verifier_eval": verifier_eval,
            },
        }
        _write_json(run_out / "weak_segment_verifier_report.json", report)

        rows.append(
            {
                "run": run,
                "selection_status": str(threshold_pick["status"]),
                "selection_stage": str(threshold_pick["selection_stage"]),
                "theta": theta,
                "label_mode": str(args.label_mode),
                "positive_weight": float(args.positive_weight),
                "q40_eval_recall": q40_eval["overall_recall"],
                "weak_verifier_eval_recall": verifier_eval["overall_recall"],
                "q40_eval_FAR": q40_eval["FAR"],
                "weak_verifier_eval_FAR": verifier_eval["FAR"],
                "q40_eval_zero_E_d_hat": q40_eval["zero_E_d_hat"],
                "weak_verifier_eval_zero_E_d_hat": verifier_eval["zero_E_d_hat"],
                "q40_eval_pos_MAE": q40_eval["pos_MAE"],
                "weak_verifier_eval_pos_MAE": verifier_eval["pos_MAE"],
                "dropped_positive_weak": _audit_count(audit, "dropped_positive_weak"),
                "dropped_false_positive_weak": _audit_count(audit, "dropped_false_positive_weak"),
                "val_weak_count": int(payload["val_weak_count"]),
                "val_weak_positive_count": int(payload["val_weak_positive_count"]),
                "val_weak_negative_count": int(payload["val_weak_negative_count"]),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "weak_segment_verifier_eval_summary.csv", index=False)
    print(summary.to_csv(index=False, float_format="%.6f"))
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
