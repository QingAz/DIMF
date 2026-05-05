#!/usr/bin/env python3

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

_TRAPEZOID = getattr(np, "trapezoid", np.trapz)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hybrid G2 postprocessing: segment-z mask, raw/smoothed raw magnitude inside the mask."
    )
    parser.add_argument(
        "--series-dir",
        type=Path,
        default=Path("outputs/r12b_g2_magnitude_residual_sanity"),
        help="Directory containing g2_postproc_val_balanced_series.csv and g2_test_series.csv.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/r13_g2_hybrid_mask_amplitude"))
    parser.add_argument("--far-max-primary", type=float, default=0.05)
    parser.add_argument("--far-max-fallback", type=float, default=0.10)
    return parser.parse_args()


def _absolute_path(path: Path) -> Path:
    path = Path(os.path.expandvars(str(path))).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _runs(mask: np.ndarray) -> List[tuple[int, int]]:
    runs: List[tuple[int, int]] = []
    start = None
    for i, value in enumerate(mask.astype(bool)):
        if value and start is None:
            start = i
        elif not value and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _hysteresis(scores: np.ndarray, segment_ids: np.ndarray, tau_high: float, tau_low: float) -> np.ndarray:
    mask = np.zeros(len(scores), dtype=bool)
    for segment_id in pd.unique(segment_ids):
        idx = np.flatnonzero(segment_ids == segment_id)
        active = False
        for pos in idx:
            score = float(scores[pos])
            active = score > tau_low if active else score > tau_high
            mask[pos] = active
    return mask


def _remove_short(mask: np.ndarray, segment_ids: np.ndarray, min_len: int) -> np.ndarray:
    if int(min_len) <= 1:
        return mask.copy()
    out = mask.copy()
    for segment_id in pd.unique(segment_ids):
        idx = np.flatnonzero(segment_ids == segment_id)
        local = out[idx]
        for start, end in _runs(local):
            if end - start + 1 < int(min_len):
                local[start : end + 1] = False
        out[idx] = local
    return out


def _merge_gaps(mask: np.ndarray, segment_ids: np.ndarray, merge_gap: int) -> np.ndarray:
    if int(merge_gap) <= 0:
        return mask.copy()
    out = mask.copy()
    for segment_id in pd.unique(segment_ids):
        idx = np.flatnonzero(segment_ids == segment_id)
        local = out[idx]
        runs = _runs(local)
        if len(runs) <= 1:
            continue
        for (_, prev_end), (next_start, _) in zip(runs, runs[1:]):
            if next_start - prev_end - 1 <= int(merge_gap):
                local[prev_end + 1 : next_start] = True
        out[idx] = local
    return out


def _rolling_by_segment(frame: pd.DataFrame, values: np.ndarray, window: int, how: str) -> np.ndarray:
    out = np.zeros(len(frame), dtype=np.float64)
    for _, idx in frame.groupby("segment_id", sort=False).groups.items():
        idx_arr = np.asarray(idx, dtype=int)
        local = pd.Series(values[idx_arr])
        roller = local.rolling(window=int(window), center=True, min_periods=1)
        if how == "mean":
            out[idx_arr] = roller.mean().to_numpy(dtype=np.float64)
        elif how == "median":
            out[idx_arr] = roller.median().to_numpy(dtype=np.float64)
        else:
            raise ValueError(f"Unknown rolling method: {how}")
    return out


def _pr_curve(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = labels.astype(bool)
    order = np.argsort(-scores)
    scores_sorted = scores[order]
    labels_sorted = labels[order]
    tp = 0
    fp = 0
    fn = int(labels_sorted.sum())
    precision: List[float] = []
    recall: List[float] = []
    thresholds: List[float] = []
    last_score = None
    for score, label in zip(scores_sorted, labels_sorted):
        if last_score is None or score != last_score:
            if last_score is not None:
                precision.append(tp / (tp + fp) if tp + fp else 0.0)
                recall.append(tp / (tp + fn) if tp + fn else 0.0)
                thresholds.append(float(last_score))
            last_score = float(score)
        if label:
            tp += 1
            fn -= 1
        else:
            fp += 1
    if last_score is not None:
        precision.append(tp / (tp + fp) if tp + fp else 0.0)
        recall.append(tp / (tp + fn) if tp + fn else 0.0)
        thresholds.append(float(last_score))
    return (
        np.asarray(precision, dtype=np.float64),
        np.asarray(recall, dtype=np.float64),
        np.asarray(thresholds, dtype=np.float64),
    )


def _score_metrics(scores: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    precision, recall, thresholds = _pr_curve(scores.astype(np.float64), labels.astype(np.int64))
    if precision.size == 0:
        return {"auprc": 0.0, "best_threshold": np.inf, "best_precision": 0.0, "best_recall": 0.0, "best_f1": 0.0}
    order = np.argsort(recall)
    auprc = float(_TRAPEZOID(precision[order], recall[order]))
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
    best_idx = int(np.argmax(f1))
    return {
        "auprc": auprc,
        "best_threshold": float(thresholds[best_idx]),
        "best_precision": float(precision[best_idx]),
        "best_recall": float(recall[best_idx]),
        "best_f1": float(f1[best_idx]),
    }


def _fixed_f1(mask: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    pred = mask.astype(bool)
    true = labels.astype(bool)
    tp = int(np.logical_and(pred, true).sum())
    fp = int(np.logical_and(pred, ~true).sum())
    fn = int(np.logical_and(~pred, true).sum())
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall + 1e-12) if precision + recall > 0 else 0.0
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def _positive_lag_blocks(frame: pd.DataFrame) -> List[pd.DataFrame]:
    blocks: List[pd.DataFrame] = []
    for _, segment in frame.sort_values(["timestamp"]).groupby("segment_id", sort=False):
        labels = segment["d_true"].to_numpy(dtype=np.int64) > 0
        for start, end in _runs(labels):
            blocks.append(segment.iloc[start : end + 1].copy())
    return blocks


def _peak_summary(frame: pd.DataFrame) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for block_id, block in enumerate(_positive_lag_blocks(frame), start=1):
        true_peak = float(block["d_true"].max())
        pred_peak = float(block["d_hat"].max())
        rounded = int(np.floor(pred_peak + 0.5))
        rows.append(
            {
                "block_id": block_id,
                "segment_id": int(block["segment_id"].iloc[0]),
                "true_peak_lag": true_peak,
                "pred_peak_lag": pred_peak,
                "peak_error": abs(pred_peak - true_peak),
                "peak_hit_at_0": int(rounded == int(true_peak)),
                "peak_hit_at_pm1": int(abs(rounded - int(true_peak)) <= 1),
            }
        )
    if not rows:
        return {"n_blocks": 0, "peak_error": np.nan, "peak_hit_at_0": np.nan, "peak_hit_at_pm1": np.nan}
    peak = pd.DataFrame(rows)
    return {
        "n_blocks": int(len(peak)),
        "peak_error": float(peak["peak_error"].mean()),
        "peak_hit_at_0": float(peak["peak_hit_at_0"].mean()),
        "peak_hit_at_pm1": float(peak["peak_hit_at_pm1"].mean()),
    }


def _metrics(frame: pd.DataFrame) -> Dict[str, Any]:
    labels = frame["d_true"].to_numpy(dtype=np.int64) > 0
    mask = frame["pred_mask"].to_numpy(dtype=np.int64) > 0
    scores = frame["pred_score"].to_numpy(dtype=np.float64)
    fixed = _fixed_f1(mask, labels)
    scored = _score_metrics(scores, labels.astype(np.int64))
    zero = ~labels
    lagged = frame.loc[labels].copy()
    zero_frame = frame.loc[zero].copy()
    block_out = frame.loc[frame["in_block"].to_numpy(dtype=np.int64) == 0].copy()
    peak = _peak_summary(frame)
    row: Dict[str, Any] = {
        "n_rows": int(len(frame)),
        "n_positive": int(labels.sum()),
        "auprc": scored["auprc"],
        "best_threshold": scored["best_threshold"],
        "precision": scored["best_precision"],
        "recall": scored["best_recall"],
        "best_f1": scored["best_f1"],
        "fixed_precision": fixed["precision"],
        "fixed_recall": fixed["recall"],
        "fixed_f1": fixed["f1"],
        "far": float(np.logical_and(mask, zero).sum() / max(int(zero.sum()), 1)),
        "predicted_nonzero_ratio": float(mask.mean()) if mask.size else 0.0,
        "block_in_mae": float(np.abs(lagged["d_hat"] - lagged["d_true"]).mean()) if not lagged.empty else np.nan,
        "zero_mean": float(zero_frame["d_hat"].mean()) if not zero_frame.empty else np.nan,
        "block_out_mean": float(block_out["d_hat"].mean()) if not block_out.empty else np.nan,
        "peak_error": peak["peak_error"],
        "peak_hit_at_0": peak["peak_hit_at_0"],
        "peak_hit_at_pm1": peak["peak_hit_at_pm1"],
    }
    for dmax in [2, 4, 6]:
        part = lagged.loc[lagged["dmax"] == dmax]
        row[f"dmax{dmax}_mae"] = float(np.abs(part["d_hat"] - part["d_true"]).mean()) if not part.empty else np.nan
    return row


def _enriched_metrics(frame: pd.DataFrame) -> Dict[str, Any]:
    row = _metrics(frame)
    scores = frame["pred_score"].to_numpy(dtype=np.float64)
    in_block = frame["in_block"].to_numpy(dtype=np.int64) > 0
    mask = frame["pred_mask"].to_numpy(dtype=np.int64) > 0
    block_row = _score_metrics(scores, in_block.astype(np.int64))
    row["block_row_auprc"] = block_row["auprc"]
    row["block_row_best_f1"] = block_row["best_f1"]
    row["block_out_far"] = float(np.logical_and(mask, ~in_block).sum() / max(int((~in_block).sum()), 1))

    seg_rows: List[Dict[str, Any]] = []
    for segment_id, part in frame.groupby("segment_id", sort=False):
        seg_rows.append(
            {
                "segment_id": int(segment_id),
                "label": int(part["in_block"].gt(0).any()),
                "score_max": float(part["pred_score"].max()),
                "pred_positive": int(part["pred_mask"].gt(0).any()),
            }
        )
    seg = pd.DataFrame(seg_rows)
    seg_scores = seg["score_max"].to_numpy(dtype=np.float64)
    seg_labels = seg["label"].to_numpy(dtype=np.int64)
    seg_metrics = _score_metrics(seg_scores, seg_labels)
    seg_pred = seg["pred_positive"].to_numpy(dtype=bool)
    seg_true = seg_labels.astype(bool)
    row["segment_auprc"] = seg_metrics["auprc"]
    row["segment_best_f1"] = seg_metrics["best_f1"]
    row["segment_far"] = float(np.logical_and(seg_pred, ~seg_true).sum() / max(int((~seg_true).sum()), 1))
    row["segment_pred_positive_ratio"] = float(seg_pred.mean()) if len(seg_pred) else 0.0
    row["n_segments"] = int(len(seg))
    row["n_positive_segments"] = int(seg_true.sum())
    row["n_negative_segments"] = int((~seg_true).sum())
    return row


def _flatten_baseline_summary(method: str, path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    bench = data.get("benchmark", {})
    loc = bench.get("localization", {})
    peak = data.get("peak", {})
    row = {
        "method": method,
        "segment_auprc": np.nan,
        "block_row_auprc": np.nan,
        "block_out_far": np.nan,
        "far": bench.get("block_out_false_alarm_rate"),
        "auprc": loc.get("auprc"),
        "best_f1": loc.get("best_f1"),
        "predicted_nonzero_ratio": loc.get("predicted_nonzero_ratio"),
        "block_in_mae": bench.get("block_in_expected_lag_mae"),
        "peak_error": peak.get("peak_error"),
        "peak_hit_at_pm1": peak.get("peak_hit_at_pm1"),
    }
    by_dmax = data.get("benchmark_by_dmax", {})
    for dmax in [2, 4, 6]:
        row[f"dmax{dmax}_mae"] = by_dmax.get(str(dmax), {}).get("block_in_expected_lag_mae")
    return row


def _segment_z(frame: pd.DataFrame) -> np.ndarray:
    m = frame["m"].to_numpy(dtype=np.float64)
    out = np.zeros(len(frame), dtype=np.float64)
    for _, idx in frame.groupby("segment_id", sort=False).groups.items():
        idx_arr = np.asarray(idx, dtype=int)
        local = m[idx_arr]
        out[idx_arr] = (local - float(np.mean(local))) / max(float(np.std(local)), 1e-6)
    return out


def _smooth_raw_m(frame: pd.DataFrame, smoothing: str) -> np.ndarray:
    values = frame["m"].to_numpy(dtype=np.float64)
    if smoothing == "none":
        return values
    if not smoothing.startswith("ma"):
        raise ValueError(f"Unknown amplitude smoothing: {smoothing}")
    return _rolling_by_segment(frame, values, int(smoothing[2:]), "mean")


def _mask_from_segment_z(frame: pd.DataFrame, cfg: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    score = _segment_z(frame)
    segment_ids = frame["segment_id"].to_numpy(dtype=np.int64)
    if cfg["mask_type"] == "single":
        mask = score > float(cfg["tau_high"])
    elif cfg["mask_type"] == "hysteresis":
        mask = _hysteresis(score, segment_ids, tau_high=float(cfg["tau_high"]), tau_low=float(cfg["tau_low"]))
    else:
        raise ValueError(f"Unknown mask type: {cfg['mask_type']}")
    mask = _remove_short(mask, segment_ids, int(cfg["min_len"]))
    mask = _merge_gaps(mask, segment_ids, int(cfg["merge_gap"]))
    return score, mask


def _apply_hybrid(frame: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    out = frame.sort_values(["timestamp", "segment_id"]).reset_index(drop=True).copy()
    z, mask = _mask_from_segment_z(out, cfg)
    amp = _smooth_raw_m(out, str(cfg["amplitude_smoothing"]))
    out["segment_z"] = z
    out["pred_score"] = z
    out["pred_mask"] = mask.astype(int)
    out["amplitude"] = amp
    out["d_hat"] = np.where(mask, amp, 0.0)
    return out


def _base_configs() -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []
    for tau in [0.5, 1.0, 1.5, 2.0]:
        for min_len in [1, 2, 3, 4]:
            for merge_gap in [0, 1, 2]:
                for amp_smooth in ["none", "ma3", "ma5"]:
                    configs.append(
                        {
                            "method_id": "H1" if amp_smooth == "none" else "H3",
                            "mask_type": "single",
                            "tau_high": float(tau),
                            "tau_low": np.nan,
                            "min_len": int(min_len),
                            "merge_gap": int(merge_gap),
                            "amplitude_smoothing": amp_smooth,
                        }
                    )
    for tau_high in [1.0, 1.5, 2.0]:
        for tau_low in [0.25, 0.5, 1.0]:
            if tau_low >= tau_high:
                continue
            for min_len in [1, 2, 3, 4]:
                for merge_gap in [0, 1, 2]:
                    for amp_smooth in ["none", "ma3", "ma5"]:
                        configs.append(
                            {
                                "method_id": "H2" if amp_smooth == "none" else "H4",
                                "mask_type": "hysteresis",
                                "tau_high": float(tau_high),
                                "tau_low": float(tau_low),
                                "min_len": int(min_len),
                                "merge_gap": int(merge_gap),
                                "amplitude_smoothing": amp_smooth,
                            }
                        )
    return configs


def _method_label(row: pd.Series | Dict[str, Any]) -> str:
    method_id = str(row["method_id"])
    amp = str(row["amplitude_smoothing"])
    mask = str(row["mask_type"])
    if method_id == "H1":
        return "H1 hybrid: z single mask + raw_m"
    if method_id == "H2":
        return "H2 hybrid: z hysteresis mask + raw_m"
    if method_id == "H3":
        return f"H3 hybrid: z single mask + {amp} raw_m"
    if method_id == "H4":
        return f"H4 hybrid: z hysteresis mask + {amp} raw_m"
    return f"{method_id}: {mask} + {amp}"


def _evaluate_grid(val_series: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for cfg in _base_configs():
        processed = _apply_hybrid(val_series, cfg)
        metrics = _enriched_metrics(processed)
        row = {**cfg, **metrics}
        row["method"] = _method_label(row)
        rows.append(row)
    return pd.DataFrame(rows)


def _select_by_rule(grid: pd.DataFrame, method_id: str, far_primary: float, far_fallback: float) -> pd.Series:
    part = grid.loc[grid["method_id"] == method_id].copy()
    if part.empty:
        raise ValueError(f"No grid rows for method_id={method_id}")

    selector_status = f"block_out_far_le_{far_primary:g}"
    eligible = part.loc[part["block_out_far"] <= float(far_primary)].copy()
    if eligible.empty:
        selector_status = f"fallback_block_out_far_le_{far_fallback:g}"
        eligible = part.loc[part["block_out_far"] <= float(far_fallback)].copy()
    if eligible.empty:
        selector_status = "fallback_no_far_pass"
        eligible = part.copy()

    ranked = eligible.sort_values(
        ["segment_auprc", "block_in_mae", "peak_hit_at_pm1", "block_row_auprc", "block_out_far"],
        ascending=[False, True, False, False, True],
    ).reset_index(drop=True)
    selected = ranked.iloc[0].copy()
    selected["selector_status"] = selector_status
    return selected


def _evaluate_selected(test_series: pd.DataFrame, selections: List[pd.Series], output_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for selected in selections:
        cfg = {
            "method_id": str(selected["method_id"]),
            "mask_type": str(selected["mask_type"]),
            "tau_high": float(selected["tau_high"]),
            "tau_low": float(selected["tau_low"]) if pd.notna(selected["tau_low"]) else np.nan,
            "min_len": int(selected["min_len"]),
            "merge_gap": int(selected["merge_gap"]),
            "amplitude_smoothing": str(selected["amplitude_smoothing"]),
        }
        processed = _apply_hybrid(test_series, cfg)
        metrics = _enriched_metrics(processed)
        row = {**cfg, **metrics}
        row["method"] = _method_label(row)
        row["selector_status"] = str(selected.get("selector_status", ""))
        rows.append(row)
        safe_name = row["method_id"].lower() + "_" + row["mask_type"] + "_" + row["amplitude_smoothing"] + "_test_series.csv"
        processed.to_csv(output_dir / safe_name, index=False)
    return pd.DataFrame(rows)


def _metric_subset(row: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "method",
        "segment_auprc",
        "block_row_auprc",
        "block_out_far",
        "far",
        "auprc",
        "best_f1",
        "fixed_f1",
        "predicted_nonzero_ratio",
        "block_in_mae",
        "peak_error",
        "peak_hit_at_pm1",
        "dmax2_mae",
        "dmax4_mae",
        "dmax6_mae",
    ]
    return {key: row.get(key) for key in keys}


def _baseline_rows(series_dir: Path, test_series: pd.DataFrame) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for method, path in {
        "R2-4": ROOT / "outputs/r2_c3_late_decay_exp_seed42/benchmark_single_model/benchmark_summary.json",
        "R2-5": ROOT / "outputs/r2_c3_d1_lite_seed42/benchmark_single_model/benchmark_summary.json",
        "G2 raw": ROOT / "outputs/r4_g2_mag_only_pos_seed42/benchmark_single_model/benchmark_summary.json",
    }.items():
        base = _flatten_baseline_summary(method, path)
        if base is not None:
            rows.append(base)

    residual_path = series_dir / "g2_residual_sanity_test_table.csv"
    if residual_path.exists():
        residual = pd.read_csv(residual_path)
        for _, row in residual.loc[residual["method"].astype(str).str.startswith("residual")].iterrows():
            rows.append(
                {
                    "method": "residual-only " + str(row["threshold_type"]),
                    "segment_auprc": row.get("segment_auprc"),
                    "block_row_auprc": row.get("block_row_auprc"),
                    "block_out_far": row.get("block_out_far"),
                    "far": row.get("far"),
                    "auprc": row.get("auprc"),
                    "best_f1": row.get("best_f1"),
                    "fixed_f1": row.get("fixed_f1"),
                    "predicted_nonzero_ratio": row.get("predicted_nonzero_ratio"),
                    "block_in_mae": row.get("block_in_mae"),
                    "peak_error": row.get("peak_error"),
                    "peak_hit_at_pm1": row.get("peak_hit_at_pm1"),
                }
            )
    return rows


def _render_segment_panels(test_series: pd.DataFrame, selected_rows: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panel_dir = output_dir / "figures"
    panel_dir.mkdir(parents=True, exist_ok=True)

    representative: List[int] = []
    pos = (
        test_series.loc[test_series["d_true"] > 0]
        .groupby("segment_id")
        .agg(dmax=("dmax", "max"), n_pos=("d_true", "size"))
        .reset_index()
    )
    for dmax in [2, 4, 6]:
        part = pos.loc[pos["dmax"] == dmax].sort_values("n_pos", ascending=False)
        if not part.empty:
            representative.append(int(part.iloc[0]["segment_id"]))
    neg = (
        test_series.loc[test_series["in_block"] == 0]
        .groupby("segment_id")
        .size()
        .sort_values(ascending=False)
    )
    if not neg.empty:
        representative.append(int(neg.index[0]))
    representative = list(dict.fromkeys(representative))[:6]

    if selected_rows.empty or not representative:
        return
    best = selected_rows.sort_values(
        ["block_out_far", "block_in_mae", "peak_error"],
        ascending=[True, True, True],
    ).iloc[0]
    cfg = {
        "method_id": str(best["method_id"]),
        "mask_type": str(best["mask_type"]),
        "tau_high": float(best["tau_high"]),
        "tau_low": float(best["tau_low"]) if pd.notna(best["tau_low"]) else np.nan,
        "min_len": int(best["min_len"]),
        "merge_gap": int(best["merge_gap"]),
        "amplitude_smoothing": str(best["amplitude_smoothing"]),
    }
    processed = _apply_hybrid(test_series, cfg)

    fig, axes = plt.subplots(len(representative), 1, figsize=(14, 3.0 * len(representative)), constrained_layout=True)
    if len(representative) == 1:
        axes = [axes]
    for ax, segment_id in zip(axes, representative):
        part = processed.loc[processed["segment_id"] == segment_id].sort_values("t")
        x = part["t"].to_numpy()
        ymax = max(
            8.0,
            float(np.nanmax(part[["m", "segment_z", "d_hat", "d_true"]].to_numpy(dtype=np.float64))),
        )
        in_block = part["in_block"].to_numpy(dtype=int) > 0
        ax.fill_between(x, 0, ymax, where=in_block, alpha=0.15, color="#f58518", step="mid", label="true block")
        ax.plot(x, part["d_true"], color="#111111", linewidth=2.0, label="true lag")
        ax.plot(x, part["m"], color="#4c78a8", linewidth=1.2, label="raw_m")
        ax.plot(x, part["segment_z"], color="#54a24b", linewidth=1.0, label="segment_z")
        ax.plot(x, part["d_hat"], color="#e45756", linewidth=1.5, label="hybrid d_hat")
        ax.fill_between(x, 0, part["pred_mask"] * ymax, color="#e45756", alpha=0.10, step="mid", label="hybrid mask")
        ax.set_title(f"segment {segment_id}, dmax={int(part['dmax'].max())}, n={len(part)}")
        ax.set_ylabel("lag / score")
        ax.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"R13 Hybrid Sanity Panels: {_method_label(best)}", fontsize=14, fontweight="bold")
    fig.savefig(panel_dir / "r13_hybrid_sanity_panels.png", bbox_inches="tight", dpi=140)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    series_dir = _absolute_path(args.series_dir)
    output_dir = _absolute_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    val_path = series_dir / "g2_postproc_val_balanced_series.csv"
    test_path = series_dir / "g2_test_series.csv"
    if not val_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing required R12b series files under {series_dir}")
    val_series = pd.read_csv(val_path)
    test_series = pd.read_csv(test_path)

    val_grid = _evaluate_grid(val_series)
    val_grid.to_csv(output_dir / "g2_hybrid_val_grid.csv", index=False)

    selections = [
        _select_by_rule(val_grid, method_id, float(args.far_max_primary), float(args.far_max_fallback))
        for method_id in ["H1", "H2", "H3", "H4"]
    ]
    selected_val = pd.DataFrame([row.to_dict() for row in selections])
    selected_val.to_csv(output_dir / "g2_hybrid_selected_val.csv", index=False)

    test_selected = _evaluate_selected(test_series, selections, output_dir)
    test_selected.to_csv(output_dir / "g2_hybrid_test_selected.csv", index=False)

    comparison_rows: List[Dict[str, Any]] = []
    comparison_rows.extend(_baseline_rows(series_dir, test_series))
    for _, row in test_selected.iterrows():
        comparison_rows.append(_metric_subset(row.to_dict()))
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(output_dir / "g2_hybrid_final_comparison.csv", index=False)

    summary_cols = [
        "method",
        "method_id",
        "mask_type",
        "amplitude_smoothing",
        "tau_high",
        "tau_low",
        "min_len",
        "merge_gap",
        "segment_auprc",
        "block_row_auprc",
        "block_out_far",
        "far",
        "block_in_mae",
        "peak_error",
        "peak_hit_at_pm1",
        "selector_status",
    ]
    test_selected[summary_cols].to_csv(output_dir / "g2_hybrid_sanity_test_table.csv", index=False)
    _render_segment_panels(test_series, test_selected, output_dir)

    report = {
        "series_dir": series_dir.as_posix(),
        "far_max_primary": float(args.far_max_primary),
        "far_max_fallback": float(args.far_max_fallback),
        "outputs": {
            "val_grid": (output_dir / "g2_hybrid_val_grid.csv").as_posix(),
            "selected_val": (output_dir / "g2_hybrid_selected_val.csv").as_posix(),
            "test_selected": (output_dir / "g2_hybrid_test_selected.csv").as_posix(),
            "comparison": (output_dir / "g2_hybrid_final_comparison.csv").as_posix(),
            "sanity_table": (output_dir / "g2_hybrid_sanity_test_table.csv").as_posix(),
        },
        "selected_val": selected_val.to_dict(orient="records"),
        "test_selected": test_selected.to_dict(orient="records"),
    }
    (output_dir / "g2_hybrid_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Selected val configs:")
    print(selected_val.to_csv(index=False))
    print("Hybrid test table:")
    print(test_selected[summary_cols].to_csv(index=False))
    print("Final comparison:")
    print(comparison.to_csv(index=False))


if __name__ == "__main__":
    main()
