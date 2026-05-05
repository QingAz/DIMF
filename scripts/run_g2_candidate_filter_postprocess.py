#!/usr/bin/env python3

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_g2_hybrid_mask_amplitude import (  # noqa: E402
    _absolute_path,
    _baseline_rows,
    _enriched_metrics,
    _merge_gaps,
    _metric_subset,
    _remove_short,
    _rolling_by_segment,
    _runs,
    _score_metrics,
    _segment_z,
    _smooth_raw_m,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="R14 diagnostic postprocessing: mask overlap audit plus raw-m candidate/segment-z filter variants."
    )
    parser.add_argument("--series-dir", type=Path, default=Path("outputs/r12b_g2_magnitude_residual_sanity"))
    parser.add_argument("--hybrid-dir", type=Path, default=Path("outputs/r13_g2_hybrid_mask_amplitude_local"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/r14_g2_candidate_filter_local"))
    return parser.parse_args()


def _sort_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(["segment_id", "t"]).reset_index(drop=True).copy()


def _metric_value(num: int | float, den: int | float) -> float:
    return float(num / den) if den else np.nan


def _lag_positive_blocks(frame: pd.DataFrame) -> List[pd.DataFrame]:
    blocks: List[pd.DataFrame] = []
    for _, segment in frame.groupby("segment_id", sort=False):
        labels = segment["d_true"].to_numpy(dtype=np.float64) > 0
        for start, end in _runs(labels):
            blocks.append(segment.iloc[start : end + 1])
    return blocks


def _injected_blocks(frame: pd.DataFrame) -> List[pd.DataFrame]:
    blocks: List[pd.DataFrame] = []
    in_block = frame.loc[frame["in_block"].to_numpy(dtype=np.int64) > 0]
    for _, block in in_block.groupby(["segment_id", "block_id"], sort=False):
        blocks.append(block)
    return blocks


def _run_lengths(mask: np.ndarray, segment_ids: np.ndarray) -> List[int]:
    lengths: List[int] = []
    for segment_id in pd.unique(segment_ids):
        idx = np.flatnonzero(segment_ids == segment_id)
        for start, end in _runs(mask[idx]):
            lengths.append(int(end - start + 1))
    return lengths


def _audit_mask(name: str, frame: pd.DataFrame) -> Dict[str, Any]:
    frame = _sort_frame(frame)
    mask = frame["pred_mask"].to_numpy(dtype=np.int64) > 0
    lagpos = frame["d_true"].to_numpy(dtype=np.float64) > 0
    in_block = frame["in_block"].to_numpy(dtype=np.int64) > 0
    segment_ids = frame["segment_id"].to_numpy(dtype=np.int64)
    n_mask = int(mask.sum())

    lag_blocks = _lag_positive_blocks(frame)
    inj_blocks = _injected_blocks(frame)
    lag_block_any = [bool(block["pred_mask"].to_numpy(dtype=np.int64).sum() > 0) for block in lag_blocks]
    lag_block_overlap = [
        float(block["pred_mask"].to_numpy(dtype=np.int64).mean()) if len(block) else np.nan for block in lag_blocks
    ]
    inj_block_any = [bool(block["pred_mask"].to_numpy(dtype=np.int64).sum() > 0) for block in inj_blocks]
    inj_block_overlap = [
        float(block["pred_mask"].to_numpy(dtype=np.int64).mean()) if len(block) else np.nan for block in inj_blocks
    ]
    seg_mask_rows = (
        frame.assign(mask=mask.astype(int)).groupby("segment_id", sort=False)["mask"].sum().to_numpy(dtype=np.float64)
    )
    run_lengths = _run_lengths(mask, segment_ids)

    row: Dict[str, Any] = {
        "method": name,
        "n_rows": int(len(frame)),
        "n_mask_rows": n_mask,
        "mask_ratio": float(mask.mean()) if len(mask) else np.nan,
        "recall_d_true_gt0": _metric_value(int(np.logical_and(mask, lagpos).sum()), int(lagpos.sum())),
        "recall_in_block": _metric_value(int(np.logical_and(mask, in_block).sum()), int(in_block.sum())),
        "precision_d_true_gt0": _metric_value(int(np.logical_and(mask, lagpos).sum()), n_mask),
        "precision_in_block": _metric_value(int(np.logical_and(mask, in_block).sum()), n_mask),
        "far_vs_d_true0": _metric_value(int(np.logical_and(mask, ~lagpos).sum()), int((~lagpos).sum())),
        "block_out_far": _metric_value(int(np.logical_and(mask, ~in_block).sum()), int((~in_block).sum())),
        "lagpos_block_any_recall": float(np.mean(lag_block_any)) if lag_block_any else np.nan,
        "lagpos_block_overlap_mean": float(np.nanmean(lag_block_overlap)) if lag_block_overlap else np.nan,
        "injected_block_any_recall": float(np.mean(inj_block_any)) if inj_block_any else np.nan,
        "injected_block_overlap_mean": float(np.nanmean(inj_block_overlap)) if inj_block_overlap else np.nan,
        "masked_segment_ratio": float((seg_mask_rows > 0).mean()) if len(seg_mask_rows) else np.nan,
        "mask_rows_per_segment_mean": float(np.mean(seg_mask_rows)) if len(seg_mask_rows) else np.nan,
        "mask_rows_per_masked_segment_mean": float(np.mean(seg_mask_rows[seg_mask_rows > 0]))
        if np.any(seg_mask_rows > 0)
        else 0.0,
        "mask_runs_per_segment_mean": float(len(run_lengths) / max(len(seg_mask_rows), 1)),
        "mask_run_len_mean": float(np.mean(run_lengths)) if run_lengths else 0.0,
        "mask_run_len_median": float(np.median(run_lengths)) if run_lengths else 0.0,
        "mask_run_len_max": int(max(run_lengths)) if run_lengths else 0,
    }
    return row


def _audit_by_dmax(name: str, frame: pd.DataFrame) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    mask = frame["pred_mask"].to_numpy(dtype=np.int64) > 0
    for dmax in [2, 4, 6]:
        part = frame.loc[frame["dmax"] == dmax].copy()
        if part.empty:
            continue
        local_mask = mask[part.index.to_numpy(dtype=int)]
        lagpos = part["d_true"].to_numpy(dtype=np.float64) > 0
        in_block = part["in_block"].to_numpy(dtype=np.int64) > 0
        n_mask = int(local_mask.sum())
        rows.append(
            {
                "method": name,
                "dmax": dmax,
                "n_rows": int(len(part)),
                "n_mask_rows": n_mask,
                "mask_ratio": float(local_mask.mean()) if len(local_mask) else np.nan,
                "recall_d_true_gt0": _metric_value(int(np.logical_and(local_mask, lagpos).sum()), int(lagpos.sum())),
                "recall_in_block": _metric_value(int(np.logical_and(local_mask, in_block).sum()), int(in_block.sum())),
                "precision_d_true_gt0": _metric_value(int(np.logical_and(local_mask, lagpos).sum()), n_mask),
                "precision_in_block": _metric_value(int(np.logical_and(local_mask, in_block).sum()), n_mask),
                "block_out_far": _metric_value(int(np.logical_and(local_mask, ~in_block).sum()), int((~in_block).sum())),
            }
        )
    return rows


def _load_existing_masks(series_dir: Path, hybrid_dir: Path) -> Dict[str, pd.DataFrame]:
    files = {
        "residual single": series_dir / "residual_single_segment_z_test_series.csv",
        "residual hysteresis": series_dir / "residual_hysteresis_segment_z_test_series.csv",
        "H1 hybrid": hybrid_dir / "h1_single_none_test_series.csv",
        "H2 hybrid": hybrid_dir / "h2_hysteresis_none_test_series.csv",
        "H3 hybrid": hybrid_dir / "h3_single_ma3_test_series.csv",
        "H4 hybrid": hybrid_dir / "h4_hysteresis_ma3_test_series.csv",
    }
    loaded: Dict[str, pd.DataFrame] = {}
    for name, path in files.items():
        if path.exists():
            loaded[name] = _sort_frame(pd.read_csv(path))
    return loaded


def _local_quantile_threshold(frame: pd.DataFrame, scores: np.ndarray, quantile: float) -> np.ndarray:
    threshold = np.zeros(len(frame), dtype=np.float64)
    for _, idx in frame.groupby("segment_id", sort=False).groups.items():
        idx_arr = np.asarray(idx, dtype=int)
        threshold[idx_arr] = float(np.quantile(scores[idx_arr], quantile))
    return threshold


def _single_candidate_mask(frame: pd.DataFrame, scores: np.ndarray, quantile: float) -> np.ndarray:
    return scores >= _local_quantile_threshold(frame, scores, quantile)


def _peak_candidate_mask(frame: pd.DataFrame, scores: np.ndarray, quantile: float, grow_window: int) -> np.ndarray:
    mask = np.zeros(len(frame), dtype=bool)
    for _, idx in frame.groupby("segment_id", sort=False).groups.items():
        idx_arr = np.asarray(idx, dtype=int)
        local = scores[idx_arr]
        threshold = float(np.quantile(local, quantile))
        n = len(local)
        for i, value in enumerate(local):
            left = local[i - 1] if i > 0 else -np.inf
            right = local[i + 1] if i < n - 1 else -np.inf
            is_peak = value >= threshold and value >= left and value >= right and (value > left or value > right)
            if not is_peak:
                continue
            start = max(0, i - int(grow_window))
            end = min(n - 1, i + int(grow_window))
            mask[idx_arr[start : end + 1]] = True
    return mask


def _candidate_configs() -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []
    smoothings = ["none", "ma3", "ma5"]
    raw_quantiles = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90]
    peak_quantiles = [0.80, 0.85, 0.90]
    z_thresholds = [-1.5, -1.0, -0.5, 0.0, 0.5]
    min_lens = [1, 2, 3]
    merge_gaps = [0, 1, 2]

    for smoothing in smoothings:
        for quantile in raw_quantiles:
            for min_len in min_lens:
                for merge_gap in merge_gaps:
                    configs.append(
                        {
                            "method_id": "K1",
                            "candidate_type": "raw_m_single_threshold",
                            "filter_type": "none",
                            "raw_smoothing": smoothing,
                            "raw_quantile": quantile,
                            "peak_quantile": np.nan,
                            "grow_window": np.nan,
                            "z_threshold": np.nan,
                            "min_len": min_len,
                            "merge_gap": merge_gap,
                        }
                    )
                    for z_threshold in z_thresholds:
                        configs.append(
                            {
                                "method_id": "K2",
                                "candidate_type": "raw_m_single_threshold",
                                "filter_type": "segment_z_filter",
                                "raw_smoothing": smoothing,
                                "raw_quantile": quantile,
                                "peak_quantile": np.nan,
                                "grow_window": np.nan,
                                "z_threshold": z_threshold,
                                "min_len": min_len,
                                "merge_gap": merge_gap,
                            }
                        )

    for smoothing in smoothings:
        for quantile in peak_quantiles:
            for grow_window in [2, 4, 6]:
                for min_len in min_lens:
                    for merge_gap in merge_gaps:
                        configs.append(
                            {
                                "method_id": "K3",
                                "candidate_type": "raw_m_peak_trigger",
                                "filter_type": "none",
                                "raw_smoothing": smoothing,
                                "raw_quantile": np.nan,
                                "peak_quantile": quantile,
                                "grow_window": grow_window,
                                "z_threshold": np.nan,
                                "min_len": min_len,
                                "merge_gap": merge_gap,
                            }
                        )
                        for z_threshold in z_thresholds:
                            configs.append(
                                {
                                    "method_id": "K4",
                                    "candidate_type": "raw_m_peak_trigger",
                                    "filter_type": "segment_z_veto",
                                    "raw_smoothing": smoothing,
                                    "raw_quantile": np.nan,
                                    "peak_quantile": quantile,
                                    "grow_window": grow_window,
                                    "z_threshold": z_threshold,
                                    "min_len": min_len,
                                    "merge_gap": merge_gap,
                                }
                            )
    return configs


def _method_label(row: Dict[str, Any] | pd.Series) -> str:
    method_id = str(row["method_id"])
    smoothing = str(row["raw_smoothing"])
    if method_id == "K1":
        return f"K1 raw single ({smoothing})"
    if method_id == "K2":
        return f"K2 raw single + z filter ({smoothing})"
    if method_id == "K3":
        return f"K3 raw peak grow ({smoothing})"
    if method_id == "K4":
        return f"K4 raw peak grow + z veto ({smoothing})"
    return method_id


def _apply_candidate_filter(frame: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    out = _sort_frame(frame)
    segment_ids = out["segment_id"].to_numpy(dtype=np.int64)
    z = _segment_z(out)
    raw_score = _smooth_raw_m(out, str(cfg["raw_smoothing"]))

    if cfg["candidate_type"] == "raw_m_single_threshold":
        candidate = _single_candidate_mask(out, raw_score, float(cfg["raw_quantile"]))
    elif cfg["candidate_type"] == "raw_m_peak_trigger":
        candidate = _peak_candidate_mask(out, raw_score, float(cfg["peak_quantile"]), int(cfg["grow_window"]))
    else:
        raise ValueError(f"Unknown candidate_type: {cfg['candidate_type']}")

    candidate = _remove_short(candidate, segment_ids, int(cfg["min_len"]))
    candidate = _merge_gaps(candidate, segment_ids, int(cfg["merge_gap"]))

    if cfg["filter_type"] in {"segment_z_filter", "segment_z_veto"}:
        mask = np.logical_and(candidate, z > float(cfg["z_threshold"]))
        mask = _remove_short(mask, segment_ids, int(cfg["min_len"]))
        mask = _merge_gaps(mask, segment_ids, int(cfg["merge_gap"]))
    elif cfg["filter_type"] == "none":
        mask = candidate
    else:
        raise ValueError(f"Unknown filter_type: {cfg['filter_type']}")

    suppressed = float(np.nanmin(raw_score) - 1.0) if len(raw_score) else -1.0
    out["segment_z"] = z
    out["raw_score"] = raw_score
    out["candidate_mask"] = candidate.astype(int)
    out["pred_mask"] = mask.astype(int)
    out["pred_score"] = np.where(mask, raw_score, suppressed)
    out["amplitude"] = raw_score
    out["d_hat"] = np.where(mask, raw_score, 0.0)
    return out


def _evaluate_grid(val_series: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for cfg in _candidate_configs():
        processed = _apply_candidate_filter(val_series, cfg)
        metrics = _enriched_metrics(processed)
        row = {**cfg, **metrics}
        row["method"] = _method_label(row)
        rows.append(row)
    return pd.DataFrame(rows)


def _select_configs(grid: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    sort_cols = [
        "block_in_mae",
        "peak_error",
        "peak_hit_at_pm1",
        "block_out_far",
        "far",
        "recall",
        "segment_auprc",
    ]
    ascending = [True, True, False, True, True, False, False]
    for method_id in ["K1", "K2", "K3", "K4"]:
        part = grid.loc[grid["method_id"] == method_id].copy()
        selected = part.sort_values(sort_cols, ascending=ascending).iloc[0].copy()
        selected["selector"] = "mae_first"
        rows.append(selected.to_dict())
    return pd.DataFrame(rows)


def _evaluate_selected(test_series: pd.DataFrame, selected: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, selected_row in selected.iterrows():
        cfg = {
            "method_id": str(selected_row["method_id"]),
            "candidate_type": str(selected_row["candidate_type"]),
            "filter_type": str(selected_row["filter_type"]),
            "raw_smoothing": str(selected_row["raw_smoothing"]),
            "raw_quantile": float(selected_row["raw_quantile"]) if pd.notna(selected_row["raw_quantile"]) else np.nan,
            "peak_quantile": float(selected_row["peak_quantile"]) if pd.notna(selected_row["peak_quantile"]) else np.nan,
            "grow_window": int(selected_row["grow_window"]) if pd.notna(selected_row["grow_window"]) else np.nan,
            "z_threshold": float(selected_row["z_threshold"]) if pd.notna(selected_row["z_threshold"]) else np.nan,
            "min_len": int(selected_row["min_len"]),
            "merge_gap": int(selected_row["merge_gap"]),
        }
        processed = _apply_candidate_filter(test_series, cfg)
        metrics = _enriched_metrics(processed)
        row = {**cfg, **metrics}
        row["method"] = _method_label(row)
        row["selector"] = str(selected_row.get("selector", ""))
        rows.append(row)
        safe_name = str(row["method_id"]).lower() + "_" + str(row["raw_smoothing"]) + "_test_series.csv"
        processed.to_csv(output_dir / safe_name, index=False)
    return pd.DataFrame(rows)


def _select_far_cap_configs(grid: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    sort_cols = ["block_in_mae", "peak_error", "peak_hit_at_pm1", "block_out_far", "far"]
    ascending = [True, True, False, True, True]
    for cap in [0.10, 0.20, 0.30, 0.40, 0.50]:
        part = grid.loc[grid["block_out_far"] <= cap].copy()
        if part.empty:
            continue
        selected = part.sort_values(sort_cols, ascending=ascending).iloc[0].copy()
        selected["selector"] = f"block_out_far_cap_{cap:g}"
        selected["val_block_out_far_cap"] = cap
        selected["val_block_in_mae"] = float(selected["block_in_mae"])
        selected["val_block_out_far"] = float(selected["block_out_far"])
        rows.append(selected.to_dict())
    return pd.DataFrame(rows)


def _evaluate_far_cap_selected(test_series: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, selected_row in selected.iterrows():
        cfg = {
            "method_id": str(selected_row["method_id"]),
            "candidate_type": str(selected_row["candidate_type"]),
            "filter_type": str(selected_row["filter_type"]),
            "raw_smoothing": str(selected_row["raw_smoothing"]),
            "raw_quantile": float(selected_row["raw_quantile"]) if pd.notna(selected_row["raw_quantile"]) else np.nan,
            "peak_quantile": float(selected_row["peak_quantile"]) if pd.notna(selected_row["peak_quantile"]) else np.nan,
            "grow_window": int(selected_row["grow_window"]) if pd.notna(selected_row["grow_window"]) else np.nan,
            "z_threshold": float(selected_row["z_threshold"]) if pd.notna(selected_row["z_threshold"]) else np.nan,
            "min_len": int(selected_row["min_len"]),
            "merge_gap": int(selected_row["merge_gap"]),
        }
        processed = _apply_candidate_filter(test_series, cfg)
        metrics = _enriched_metrics(processed)
        row = {**cfg, **metrics}
        row["method"] = _method_label(row)
        row["selector"] = str(selected_row.get("selector", ""))
        row["val_block_out_far_cap"] = float(selected_row["val_block_out_far_cap"])
        row["val_block_in_mae"] = float(selected_row["val_block_in_mae"])
        row["val_block_out_far"] = float(selected_row["val_block_out_far"])
        rows.append(row)
    return pd.DataFrame(rows)


def _render_panels(test_series: pd.DataFrame, selected: pd.DataFrame, output_dir: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    pos = (
        test_series.loc[test_series["d_true"] > 0]
        .groupby("segment_id")
        .agg(dmax=("dmax", "max"), n_pos=("d_true", "size"))
        .reset_index()
    )
    representative: List[int] = []
    for dmax in [2, 4, 6]:
        part = pos.loc[pos["dmax"] == dmax].sort_values("n_pos", ascending=False)
        if not part.empty:
            representative.append(int(part.iloc[0]["segment_id"]))
    representative = list(dict.fromkeys(representative))[:4]
    if not representative or selected.empty:
        return

    best = selected.sort_values(["block_in_mae", "peak_error", "block_out_far"], ascending=[True, True, True]).iloc[0]
    cfg = {
        "method_id": str(best["method_id"]),
        "candidate_type": str(best["candidate_type"]),
        "filter_type": str(best["filter_type"]),
        "raw_smoothing": str(best["raw_smoothing"]),
        "raw_quantile": float(best["raw_quantile"]) if pd.notna(best["raw_quantile"]) else np.nan,
        "peak_quantile": float(best["peak_quantile"]) if pd.notna(best["peak_quantile"]) else np.nan,
        "grow_window": int(best["grow_window"]) if pd.notna(best["grow_window"]) else np.nan,
        "z_threshold": float(best["z_threshold"]) if pd.notna(best["z_threshold"]) else np.nan,
        "min_len": int(best["min_len"]),
        "merge_gap": int(best["merge_gap"]),
    }
    processed = _apply_candidate_filter(test_series, cfg)

    fig, axes = plt.subplots(len(representative), 1, figsize=(14, 3.0 * len(representative)), constrained_layout=True)
    if len(representative) == 1:
        axes = [axes]
    for ax, segment_id in zip(axes, representative):
        part = processed.loc[processed["segment_id"] == segment_id].sort_values("t")
        x = part["t"].to_numpy(dtype=np.float64)
        ymax = max(8.0, float(np.nanmax(part[["raw_score", "segment_z", "d_hat", "d_true"]].to_numpy(dtype=np.float64))))
        ax.fill_between(
            x,
            0,
            ymax,
            where=part["in_block"].to_numpy(dtype=np.int64) > 0,
            color="#f58518",
            alpha=0.15,
            step="mid",
            label="true block",
        )
        ax.plot(x, part["d_true"], color="#111111", linewidth=2.0, label="true lag")
        ax.plot(x, part["raw_score"], color="#4c78a8", linewidth=1.2, label="raw_m score")
        ax.plot(x, part["segment_z"], color="#54a24b", linewidth=1.0, label="segment_z")
        ax.fill_between(x, 0, part["candidate_mask"] * ymax, color="#72b7b2", alpha=0.12, step="mid", label="candidate")
        ax.fill_between(x, 0, part["pred_mask"] * ymax, color="#e45756", alpha=0.14, step="mid", label="final mask")
        ax.plot(x, part["d_hat"], color="#e45756", linewidth=1.5, label="d_hat")
        ax.set_title(f"segment {segment_id}, dmax={int(part['dmax'].max())}")
        ax.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"R14 Candidate/Filter Panels: {_method_label(best)}", fontsize=14, fontweight="bold")
    fig.savefig(fig_dir / "r14_candidate_filter_sanity_panels.png", bbox_inches="tight", dpi=140)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    series_dir = _absolute_path(args.series_dir)
    hybrid_dir = _absolute_path(args.hybrid_dir)
    output_dir = _absolute_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    val_series = _sort_frame(pd.read_csv(series_dir / "g2_postproc_val_balanced_series.csv"))
    test_series = _sort_frame(pd.read_csv(series_dir / "g2_test_series.csv"))

    existing_masks = _load_existing_masks(series_dir, hybrid_dir)
    audit_rows = [_audit_mask(name, frame) for name, frame in existing_masks.items()]
    audit_by_dmax_rows: List[Dict[str, Any]] = []
    for name, frame in existing_masks.items():
        audit_by_dmax_rows.extend(_audit_by_dmax(name, frame.reset_index(drop=True)))
    mask_audit = pd.DataFrame(audit_rows)
    mask_audit_by_dmax = pd.DataFrame(audit_by_dmax_rows)
    mask_audit.to_csv(output_dir / "mask_overlap_audit.csv", index=False)
    mask_audit_by_dmax.to_csv(output_dir / "mask_overlap_by_dmax.csv", index=False)

    val_grid = _evaluate_grid(val_series)
    val_grid.to_csv(output_dir / "candidate_filter_val_grid.csv", index=False)

    selected_val = _select_configs(val_grid)
    selected_val.to_csv(output_dir / "candidate_filter_selected_val.csv", index=False)

    test_selected = _evaluate_selected(test_series, selected_val, output_dir)
    test_selected.to_csv(output_dir / "candidate_filter_test_selected.csv", index=False)

    far_cap_selected_val = _select_far_cap_configs(val_grid)
    far_cap_selected_val.to_csv(output_dir / "candidate_filter_far_cap_selected_val.csv", index=False)
    far_cap_test = _evaluate_far_cap_selected(test_series, far_cap_selected_val)
    far_cap_test.to_csv(output_dir / "candidate_filter_far_cap_test.csv", index=False)

    selected_mask_audit = [_audit_mask(str(row["method"]), pd.read_csv(output_dir / (str(row["method_id"]).lower() + "_" + str(row["raw_smoothing"]) + "_test_series.csv"))) for _, row in test_selected.iterrows()]
    pd.DataFrame(selected_mask_audit).to_csv(output_dir / "candidate_filter_selected_mask_audit.csv", index=False)

    comparison_rows: List[Dict[str, Any]] = []
    comparison_rows.extend(_baseline_rows(series_dir, test_series))
    r13_path = hybrid_dir / "g2_hybrid_final_comparison.csv"
    if r13_path.exists():
        r13 = pd.read_csv(r13_path)
        for _, row in r13.loc[r13["method"].astype(str).str.startswith("H")].iterrows():
            comparison_rows.append(row.to_dict())
    for _, row in test_selected.iterrows():
        comparison_rows.append(_metric_subset(row.to_dict()))
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(output_dir / "candidate_filter_final_comparison.csv", index=False)

    _render_panels(test_series, test_selected, output_dir)

    report = {
        "series_dir": series_dir.as_posix(),
        "hybrid_dir": hybrid_dir.as_posix(),
        "outputs": {
            "mask_audit": (output_dir / "mask_overlap_audit.csv").as_posix(),
            "mask_audit_by_dmax": (output_dir / "mask_overlap_by_dmax.csv").as_posix(),
            "val_grid": (output_dir / "candidate_filter_val_grid.csv").as_posix(),
            "selected_val": (output_dir / "candidate_filter_selected_val.csv").as_posix(),
            "test_selected": (output_dir / "candidate_filter_test_selected.csv").as_posix(),
            "far_cap_selected_val": (output_dir / "candidate_filter_far_cap_selected_val.csv").as_posix(),
            "far_cap_test": (output_dir / "candidate_filter_far_cap_test.csv").as_posix(),
            "comparison": (output_dir / "candidate_filter_final_comparison.csv").as_posix(),
        },
        "selected_val": selected_val.to_dict(orient="records"),
        "test_selected": test_selected.to_dict(orient="records"),
        "far_cap_test": far_cap_test.to_dict(orient="records"),
    }
    (output_dir / "candidate_filter_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("Mask overlap audit:")
    keep = [
        "method",
        "recall_d_true_gt0",
        "recall_in_block",
        "precision_d_true_gt0",
        "precision_in_block",
        "block_out_far",
        "lagpos_block_any_recall",
        "mask_rows_per_segment_mean",
    ]
    print(mask_audit[keep].to_csv(index=False))
    print("Selected val configs:")
    print(
        selected_val[
            [
                "method",
                "raw_quantile",
                "peak_quantile",
                "grow_window",
                "z_threshold",
                "min_len",
                "merge_gap",
                "block_in_mae",
                "peak_error",
                "block_out_far",
                "far",
            ]
        ].to_csv(index=False)
    )
    print("Selected test configs:")
    print(
        test_selected[
            [
                "method",
                "block_in_mae",
                "peak_error",
                "peak_hit_at_pm1",
                "block_out_far",
                "far",
                "segment_auprc",
                "block_row_auprc",
            ]
        ].to_csv(index=False)
    )
    if not far_cap_test.empty:
        print("FAR-cap selected test configs:")
        print(
            far_cap_test[
                [
                    "val_block_out_far_cap",
                    "method",
                    "block_in_mae",
                    "peak_error",
                    "peak_hit_at_pm1",
                    "block_out_far",
                    "far",
                    "fixed_recall",
                    "fixed_precision",
                ]
            ].to_csv(index=False)
        )


if __name__ == "__main__":
    main()
