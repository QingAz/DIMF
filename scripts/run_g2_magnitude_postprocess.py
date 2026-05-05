#!/usr/bin/env python3

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_detection_segment_audit import _build_model, _make_eval_loaders
from scripts.select_and_audit_detection_checkpoints import _load_prepared
from train import load_config
from src.utils.seed import set_seed

TIME_FORMAT = "%Y-%m-%d %H:%M"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export G2 magnitude series and run simple post-processing grid search."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/r4_g2_mag_only_pos_seed42.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/r4_g2_mag_only_pos_seed42/best.ckpt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/r11_g2_magnitude_postprocess"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--edge", default="stage1_to_stage2")
    parser.add_argument("--time-col", default="TimeStamp")
    parser.add_argument("--split-col", default="split")
    parser.add_argument("--lag-col", default="lag_gt")
    return parser.parse_args()


def _absolute_path(path: Path) -> Path:
    path = Path(os.path.expandvars(str(path))).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _timestamp(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series).dt.strftime(TIME_FORMAT)


def _raw_lookup(cfg: Dict[str, Any], split_name: str) -> pd.DataFrame:
    data_cfg = cfg["data"]
    raw_path = _absolute_path(Path(data_cfg["csv_path"]))
    raw = pd.read_csv(raw_path)
    raw[data_cfg["time_col"]] = pd.to_datetime(raw[data_cfg["time_col"]])
    split_col = str(data_cfg.get("split_col", "split"))
    part = raw.loc[raw[split_col] == split_name].sort_values(data_cfg["time_col"]).copy()
    part = part.reset_index().rename(columns={"index": "raw_row_index"})
    d_true = part.get("lag_gt", pd.Series(np.zeros(len(part)), index=part.index)).fillna(0).astype(int)
    if "inject_flag" in part.columns:
        in_block = part["inject_flag"].fillna(0).astype(int)
    else:
        in_block = d_true.gt(0).astype(int)
    out = pd.DataFrame(
        {
            "timestamp": _timestamp(part[data_cfg["time_col"]]),
            "raw_row_index": part["raw_row_index"].astype(int),
            "segment_id": part.get("segment_id", pd.Series(np.arange(len(part)), index=part.index)).fillna(-1).astype(int),
            "in_block": in_block,
            "d_true": d_true,
        }
    )
    if "segment_dmax_gt" in part.columns:
        out["dmax"] = part["segment_dmax_gt"].fillna(0).astype(int)
    elif "bump_dmax_gt" in part.columns:
        out["dmax"] = part["bump_dmax_gt"].fillna(0).astype(int)
    else:
        out["dmax"] = out["d_true"].clip(lower=0).astype(int)
    out["block_id"] = np.where(out["in_block"].to_numpy(dtype=int) > 0, out["segment_id"], -1)
    return out


@torch.no_grad()
def _collect_magnitude_series(
    cfg: Dict[str, Any],
    checkpoint_path: Path,
    output_dir: Path,
    device: torch.device,
    edge: str,
) -> Dict[str, pd.DataFrame]:
    set_seed(int(cfg.get("seed", 42)))
    prepared, _ = _load_prepared(cfg)
    loaders = _make_eval_loaders(cfg, prepared)
    model = _build_model(cfg, prepared, device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    frames: Dict[str, pd.DataFrame] = {}
    for split_name in ["val", "test"]:
        probs_rows: List[np.ndarray] = []
        for X, _ in loaders[split_name]["loader"]:
            X = {k: v.to(device) for k, v in X.items()}
            _, pi = model(X)
            if edge not in pi:
                raise ValueError(f"Missing edge {edge!r} in model outputs")
            arr = pi[edge]
            arr_last = arr[:, -1, :] if arr.dim() == 3 else arr
            probs_rows.append(arr_last.detach().cpu().numpy())
        probs = np.concatenate(probs_rows, axis=0)
        lag_axis = np.arange(probs.shape[1], dtype=np.float64)
        p = 1.0 - probs[:, 0]
        d_hat = (probs * lag_axis[None, :]).sum(axis=1)
        positive_mass = np.clip(p, 1e-12, None)
        m = (probs[:, 1:] * lag_axis[None, 1:]).sum(axis=1) / positive_mass
        m = np.where(p > 1e-12, m, 0.0)
        pred = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(loaders[split_name]["sample_timestamps"]).strftime(TIME_FORMAT),
                "p": p,
                "m": m,
                "d_hat_raw": d_hat,
                "pred_argmax_lag": probs.argmax(axis=1).astype(int),
            }
        )
        joined = _raw_lookup(cfg, split_name).merge(pred, on="timestamp", how="inner")
        joined.insert(0, "split", split_name)
        joined = joined.sort_values(["timestamp", "segment_id"]).reset_index(drop=True)
        joined["t"] = joined.groupby("segment_id").cumcount()
        joined["is_positive"] = joined["d_true"].gt(0).astype(int)
        joined = joined[
            [
                "split",
                "timestamp",
                "raw_row_index",
                "segment_id",
                "t",
                "block_id",
                "dmax",
                "in_block",
                "d_true",
                "is_positive",
                "p",
                "m",
                "d_hat_raw",
                "pred_argmax_lag",
            ]
        ].copy()
        joined.to_csv(output_dir / f"g2_{split_name}_series.csv", index=False)
        frames[split_name] = joined
    pd.concat(frames.values(), ignore_index=True).to_csv(output_dir / "g2_val_test_series.csv", index=False)
    return frames


def _smooth(frame: pd.DataFrame, smoothing: str) -> np.ndarray:
    if smoothing == "none":
        return frame["m"].to_numpy(dtype=np.float64)
    if not smoothing.startswith("ma"):
        raise ValueError(f"Unknown smoothing: {smoothing}")
    window = int(smoothing[2:])
    out = np.zeros(len(frame), dtype=np.float64)
    for _, idx in frame.groupby("segment_id", sort=False).groups.items():
        values = frame.loc[idx, "m"].to_numpy(dtype=np.float64)
        out[np.asarray(idx, dtype=int)] = (
            pd.Series(values).rolling(window=window, center=True, min_periods=1).mean().to_numpy(dtype=np.float64)
        )
    return out


def _runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    runs: List[Tuple[int, int]] = []
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
            if active:
                active = score > tau_low
            else:
                active = score > tau_high
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
            gap = next_start - prev_end - 1
            if gap <= int(merge_gap):
                local[prev_end + 1 : next_start] = True
        out[idx] = local
    return out


def _apply_config(frame: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    out = frame.sort_values(["timestamp", "segment_id"]).reset_index(drop=True).copy()
    out["m_smooth"] = _smooth(out, str(cfg["smoothing"]))
    segment_ids = out["segment_id"].to_numpy(dtype=np.int64)
    if cfg["threshold_type"] == "single":
        mask = out["m_smooth"].to_numpy(dtype=np.float64) > float(cfg["tau_high"])
    elif cfg["threshold_type"] == "hysteresis":
        mask = _hysteresis(
            out["m_smooth"].to_numpy(dtype=np.float64),
            segment_ids,
            tau_high=float(cfg["tau_high"]),
            tau_low=float(cfg["tau_low"]),
        )
    else:
        raise ValueError(f"Unknown threshold_type: {cfg['threshold_type']}")
    mask = _remove_short(mask, segment_ids, int(cfg["min_len"]))
    mask = _merge_gaps(mask, segment_ids, int(cfg["merge_gap"]))
    out["pred_mask"] = mask.astype(int)
    out["d_hat"] = np.where(mask, out["m_smooth"].to_numpy(dtype=np.float64), 0.0)
    out["pred_score"] = out["d_hat"].astype(float)
    return out


def _pr_curve(scores: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    auprc = float(np.trapz(precision[order], recall[order]))
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
                "start_time": str(block["timestamp"].iloc[0]),
                "end_time": str(block["timestamp"].iloc[-1]),
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


def _score_row(row: Dict[str, Any]) -> float:
    block_mae = row.get("block_in_mae", np.nan)
    if not np.isfinite(block_mae):
        block_mae = 1e6
    return float(row.get("auprc", 0.0) - 0.5 * row.get("far", 0.0) - 0.2 * block_mae)


def _coarse_grid() -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []
    smoothings = ["none", "ma3", "ma5"]
    single_thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    tau_highs = [1.0, 1.5, 2.0, 2.5, 3.0]
    tau_lows = [0.5, 1.0, 1.5, 2.0]
    for smoothing in smoothings:
        for tau in single_thresholds:
            for min_len in [1, 3, 5]:
                for merge_gap in [0, 1, 3]:
                    configs.append(
                        {
                            "stage": "coarse",
                            "smoothing": smoothing,
                            "threshold_type": "single",
                            "tau_high": float(tau),
                            "tau_low": np.nan,
                            "min_len": int(min_len),
                            "merge_gap": int(merge_gap),
                        }
                    )
        for tau_high in tau_highs:
            for tau_low in tau_lows:
                if tau_low >= tau_high:
                    continue
                for min_len in [1, 3, 5]:
                    for merge_gap in [0, 1, 3]:
                        configs.append(
                            {
                                "stage": "coarse",
                                "smoothing": smoothing,
                                "threshold_type": "hysteresis",
                                "tau_high": float(tau_high),
                                "tau_low": float(tau_low),
                                "min_len": int(min_len),
                                "merge_gap": int(merge_gap),
                            }
                        )
    return configs


def _fine_grid(top_rows: pd.DataFrame) -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []
    for _, row in top_rows.iterrows():
        smoothings = sorted(set([str(row["smoothing"]), "none", "ma3", "ma5", "ma7"]))
        min_lens = sorted({max(1, min(5, int(row["min_len"]) + delta)) for delta in [-1, 0, 1]})
        merge_gaps = sorted({max(0, min(3, int(row["merge_gap"]) + delta)) for delta in [-1, 0, 1]})
        if row["threshold_type"] == "single":
            taus = sorted({round(max(0.1, float(row["tau_high"]) + delta), 3) for delta in [-0.5, -0.25, 0.0, 0.25, 0.5]})
            for smoothing in smoothings:
                for tau in taus:
                    for min_len in min_lens:
                        for merge_gap in merge_gaps:
                            configs.append(
                                {
                                    "stage": "fine",
                                    "smoothing": smoothing,
                                    "threshold_type": "single",
                                    "tau_high": float(tau),
                                    "tau_low": np.nan,
                                    "min_len": int(min_len),
                                    "merge_gap": int(merge_gap),
                                }
                            )
        else:
            highs = sorted({round(max(0.1, float(row["tau_high"]) + delta), 3) for delta in [-0.5, -0.25, 0.0, 0.25, 0.5]})
            lows = sorted({round(max(0.05, float(row["tau_low"]) + delta), 3) for delta in [-0.5, -0.25, 0.0, 0.25, 0.5]})
            for smoothing in smoothings:
                for tau_high in highs:
                    for tau_low in lows:
                        if tau_low >= tau_high:
                            continue
                        for min_len in min_lens:
                            for merge_gap in merge_gaps:
                                configs.append(
                                    {
                                        "stage": "fine",
                                        "smoothing": smoothing,
                                        "threshold_type": "hysteresis",
                                        "tau_high": float(tau_high),
                                        "tau_low": float(tau_low),
                                        "min_len": int(min_len),
                                        "merge_gap": int(merge_gap),
                                    }
                                )
    seen = set()
    unique: List[Dict[str, Any]] = []
    for cfg in configs:
        key = (
            cfg["smoothing"],
            cfg["threshold_type"],
            cfg["tau_high"],
            cfg["tau_low"] if np.isfinite(cfg["tau_low"]) else None,
            cfg["min_len"],
            cfg["merge_gap"],
        )
        if key not in seen:
            seen.add(key)
            unique.append(cfg)
    return unique


def _evaluate_grid(series: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    coarse = _coarse_grid()
    for cfg in coarse:
        metrics = _metrics(_apply_config(series, cfg))
        metrics.update(cfg)
        metrics["selector_score"] = _score_row(metrics)
        rows.append(metrics)
    coarse_df = pd.DataFrame(rows)
    top_for_fine = (
        coarse_df.sort_values(["auprc", "far", "block_in_mae", "peak_hit_at_pm1"], ascending=[False, True, True, False])
        .groupby("threshold_type", group_keys=False)
        .head(4)
    )
    for cfg in _fine_grid(top_for_fine):
        metrics = _metrics(_apply_config(series, cfg))
        metrics.update(cfg)
        metrics["selector_score"] = _score_row(metrics)
        rows.append(metrics)
    out = pd.DataFrame(rows).drop_duplicates(
        subset=["smoothing", "threshold_type", "tau_high", "tau_low", "min_len", "merge_gap"],
        keep="last",
    )
    return out.sort_values(["auprc", "far", "block_in_mae", "peak_hit_at_pm1"], ascending=[False, True, True, False]).reset_index(drop=True)


def _select_best(grid: pd.DataFrame, threshold_type: str) -> pd.Series:
    part = grid.loc[grid["threshold_type"] == threshold_type].copy()
    if part.empty:
        raise ValueError(f"No grid rows for threshold_type={threshold_type}")
    return (
        part.sort_values(["auprc", "far", "block_in_mae", "peak_hit_at_pm1"], ascending=[False, True, True, False])
        .reset_index(drop=True)
        .iloc[0]
    )


def _row_to_config(row: pd.Series) -> Dict[str, Any]:
    return {
        "stage": str(row["stage"]),
        "smoothing": str(row["smoothing"]),
        "threshold_type": str(row["threshold_type"]),
        "tau_high": float(row["tau_high"]),
        "tau_low": float(row["tau_low"]) if pd.notna(row["tau_low"]) else None,
        "min_len": int(row["min_len"]),
        "merge_gap": int(row["merge_gap"]),
    }


def _flatten_baseline_summary(method: str, path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    bench = data.get("benchmark", {})
    loc = bench.get("localization", {})
    peak = data.get("peak", {})
    row = {
        "method": method,
        "auprc": loc.get("auprc"),
        "best_f1": loc.get("best_f1"),
        "far": bench.get("block_out_false_alarm_rate"),
        "predicted_nonzero_ratio": loc.get("predicted_nonzero_ratio"),
        "zero_mean": bench.get("mean_pred_expected_lag_when_true_zero"),
        "block_out_mean": None,
        "block_in_mae": bench.get("block_in_expected_lag_mae"),
        "peak_error": peak.get("peak_error"),
        "peak_hit_at_0": peak.get("peak_hit_at_0"),
        "peak_hit_at_pm1": peak.get("peak_hit_at_pm1"),
    }
    by_dmax = data.get("benchmark_by_dmax", {})
    for dmax in [2, 4, 6]:
        item = by_dmax.get(str(dmax), {})
        row[f"dmax{dmax}_mae"] = item.get("block_in_expected_lag_mae")
    return row


def _pi_columns(frame: pd.DataFrame, edge: str) -> List[str]:
    prefix = f"{edge}_pred_pi_lag"
    cols = [col for col in frame.columns if col.startswith(prefix)]
    return sorted(cols, key=lambda name: int(name.split("lag")[-1]))


def _estimate_baseline_row(method: str, estimates_path: Path, raw_path: Path, edge: str) -> Dict[str, Any] | None:
    if not estimates_path.exists() or not raw_path.exists():
        return None

    estimates = pd.read_csv(estimates_path)
    raw = pd.read_csv(raw_path)
    pi_cols = _pi_columns(estimates, edge)
    expected_col = f"{edge}_pred_expected_lag"
    if not pi_cols or expected_col not in estimates.columns:
        return None

    estimates["timestamp"] = _timestamp(estimates["TimeStamp"])
    raw_test = raw.loc[raw["split"] == "test"].copy()
    raw_test["timestamp"] = _timestamp(raw_test["TimeStamp"])
    keep = pd.DataFrame(
        {
            "timestamp": raw_test["timestamp"],
            "segment_id": raw_test.get("segment_id", pd.Series(np.arange(len(raw_test)), index=raw_test.index)).fillna(-1).astype(int),
            "d_true": raw_test.get("lag_gt", pd.Series(np.zeros(len(raw_test)), index=raw_test.index)).fillna(0).astype(int),
        }
    )
    keep["in_block"] = (
        raw_test["inject_flag"].fillna(0).astype(int)
        if "inject_flag" in raw_test.columns
        else keep["d_true"].gt(0).astype(int)
    )
    if "segment_dmax_gt" in raw_test.columns:
        keep["dmax"] = raw_test["segment_dmax_gt"].fillna(0).astype(int)
    elif "bump_dmax_gt" in raw_test.columns:
        keep["dmax"] = raw_test["bump_dmax_gt"].fillna(0).astype(int)
    else:
        keep["dmax"] = keep["d_true"].clip(lower=0).astype(int)

    pi = estimates[pi_cols].to_numpy(dtype=np.float64)
    pred = pd.DataFrame(
        {
            "timestamp": estimates["timestamp"],
            "d_hat": estimates[expected_col].to_numpy(dtype=np.float64),
            "pred_score": 1.0 - pi[:, 0],
        }
    )
    joined = keep.merge(pred, on="timestamp", how="inner").sort_values(["timestamp", "segment_id"]).reset_index(drop=True)
    scored = _score_metrics(
        joined["pred_score"].to_numpy(dtype=np.float64),
        joined["d_true"].gt(0).to_numpy(dtype=np.int64),
    )
    joined["pred_mask"] = (joined["pred_score"].to_numpy(dtype=np.float64) >= float(scored["best_threshold"])).astype(int)
    metrics = _metrics(joined)
    metrics["method"] = method
    return {key: metrics.get(key) for key in ["method", "auprc", "best_f1", "far", "predicted_nonzero_ratio", "zero_mean", "block_out_mean", "block_in_mae", "peak_error", "peak_hit_at_0", "peak_hit_at_pm1", "dmax2_mae", "dmax4_mae", "dmax6_mae"]}


def _comparison_rows(test_rows: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    baseline_paths = {
        "noalign": ROOT / "outputs/localblock_mixed_evalsafe_10_9_10_noalign_seed42/benchmark_single_model/benchmark_summary.json",
        "R2-4": ROOT / "outputs/r2_c3_late_decay_exp_seed42/benchmark_single_model/benchmark_summary.json",
        "R2-5": ROOT / "outputs/r2_c3_d1_lite_seed42/benchmark_single_model/benchmark_summary.json",
        "G2 raw": ROOT / "outputs/r4_g2_mag_only_pos_seed42/benchmark_single_model/benchmark_summary.json",
    }
    for method, path in baseline_paths.items():
        row = _flatten_baseline_summary(method, path)
        if row is None and method == "noalign":
            row = _estimate_baseline_row(
                method,
                ROOT / "outputs/localblock_mixed_evalsafe_10_9_10_noalign_seed42/test_delay_estimates.csv",
                ROOT / "data/processed/LiquidSugar_local_block_mixed_evalsafe_10_9_10_rawgap.csv",
                "stage1_to_stage2",
            )
        if row is not None:
            rows.append(row)
        else:
            rows.append({"method": method})
    for _, row in test_rows.iterrows():
        rows.append(
            {
                "method": f"G2 + {row['threshold_type']}",
                "auprc": row["auprc"],
                "best_f1": row["best_f1"],
                "far": row["far"],
                "predicted_nonzero_ratio": row["predicted_nonzero_ratio"],
                "zero_mean": row["zero_mean"],
                "block_out_mean": row["block_out_mean"],
                "block_in_mae": row["block_in_mae"],
                "peak_error": row["peak_error"],
                "peak_hit_at_0": row["peak_hit_at_0"],
                "peak_hit_at_pm1": row["peak_hit_at_pm1"],
                "dmax2_mae": row["dmax2_mae"],
                "dmax4_mae": row["dmax4_mae"],
                "dmax6_mae": row["dmax6_mae"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    config_path = _absolute_path(args.config)
    checkpoint_path = _absolute_path(args.checkpoint)
    output_dir = _absolute_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(str(config_path))
    frames = _collect_magnitude_series(
        cfg=cfg,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        device=torch.device(args.device),
        edge=str(args.edge),
    )

    val_grid = _evaluate_grid(frames["val"])
    val_grid.to_csv(output_dir / "g2_postproc_val_grid.csv", index=False)

    selected_rows = []
    selected_series = {}
    selected_configs: Dict[str, Dict[str, Any]] = {}
    for threshold_type in ["single", "hysteresis"]:
        selected = _select_best(val_grid, threshold_type)
        cfg_selected = _row_to_config(selected)
        selected_configs[threshold_type] = cfg_selected
        test_processed = _apply_config(frames["test"], cfg_selected)
        test_metrics = _metrics(test_processed)
        test_metrics.update(cfg_selected)
        test_metrics["selected_by"] = "val_auprc_far_mae_peak"
        selected_rows.append(test_metrics)
        selected_series[threshold_type] = test_processed
        test_processed.to_csv(output_dir / f"g2_test_{threshold_type}_postproc_series.csv", index=False)

    test_selected = pd.DataFrame(selected_rows)
    test_selected.to_csv(output_dir / "g2_postproc_test_selected.csv", index=False)

    comparison = _comparison_rows(test_selected)
    comparison.to_csv(output_dir / "g2_postproc_final_comparison.csv", index=False)

    report = {
        "config": config_path.as_posix(),
        "checkpoint": checkpoint_path.as_posix(),
        "edge": str(args.edge),
        "outputs": {
            "val_series": (output_dir / "g2_val_series.csv").as_posix(),
            "test_series": (output_dir / "g2_test_series.csv").as_posix(),
            "val_grid": (output_dir / "g2_postproc_val_grid.csv").as_posix(),
            "test_selected": (output_dir / "g2_postproc_test_selected.csv").as_posix(),
            "comparison": (output_dir / "g2_postproc_final_comparison.csv").as_posix(),
        },
        "selected_configs": selected_configs,
        "test_selected": test_selected.to_dict(orient="records"),
    }
    (output_dir / "g2_postproc_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(val_grid.head(20).to_csv(index=False))
    print(test_selected.to_csv(index=False))
    print(comparison.to_csv(index=False))


if __name__ == "__main__":
    main()
