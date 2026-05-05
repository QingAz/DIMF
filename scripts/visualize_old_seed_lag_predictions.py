#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-codex"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = {
    "mid_high_z": "#1b9e77",
    "low_lag_high_conf": "#377eb8",
    "weak_plateau": "#e7298a",
    "selected_overlap": "#984ea3",
}


def _path(text: str | Path) -> Path:
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _read_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    first = frame.columns[0]
    frame = frame.loc[frame[first].astype(str) != first].reset_index(drop=True)
    for col in frame.columns:
        if col not in {"split", "source_split", "timestamp", "original_split"}:
            converted = pd.to_numeric(frame[col], errors="coerce")
            if converted.notna().any() or frame[col].isna().all():
                frame[col] = converted
    return frame


def _fit_affine_raw_m(train: pd.DataFrame, label_col: str) -> Dict[str, float]:
    positive = train[label_col].to_numpy(dtype=np.float64) > 0
    x = train.loc[positive, "raw_m"].to_numpy(dtype=np.float64)
    y = train.loc[positive, label_col].to_numpy(dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    a, b = np.linalg.lstsq(np.column_stack([x, np.ones_like(x)]), y, rcond=None)[0]
    pred = a * x + b
    return {"a": float(a), "b": float(b), "fit_rows": int(x.size), "fit_mae": float(np.mean(np.abs(pred - y)))}


def _enrich(frame: pd.DataFrame, fit: Dict[str, float]) -> pd.DataFrame:
    out = frame.copy()
    calibrated = fit["a"] * out["raw_m"].to_numpy(dtype=np.float64) + fit["b"]
    out["calibrated_raw_m"] = np.clip(calibrated, 0.0, np.maximum(out["dmax"].to_numpy(dtype=np.float64), 0.0))
    return out


def _runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    runs: List[Tuple[int, int]] = []
    start = None
    for idx, value in enumerate(mask.astype(bool)):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _positive_blocks(frame: pd.DataFrame, label_col: str, group_col: str) -> List[np.ndarray]:
    blocks: List[np.ndarray] = []
    for idx in frame.groupby(group_col, sort=False).groups.values():
        idx_arr = frame.index.get_indexer(idx)
        labels = frame.iloc[idx_arr][label_col].to_numpy(dtype=np.float64) > 0
        start = None
        for pos, value in enumerate(labels):
            if value and start is None:
                start = pos
            elif not value and start is not None:
                blocks.append(idx_arr[start:pos])
                start = None
        if start is not None:
            blocks.append(idx_arr[start:])
    return blocks


def _weak_plateau_mask(frame: pd.DataFrame, strong_mask: np.ndarray) -> tuple[np.ndarray, List[Dict[str, Any]]]:
    candidate = frame["candidate_score"].to_numpy(dtype=np.float64)
    cal = frame["calibrated_raw_m"].to_numpy(dtype=np.float64)
    expected = frame["expected_lag"].to_numpy(dtype=np.float64)
    loc = frame["localization_score"].to_numpy(dtype=np.float64)
    weak_score = (
        ((cal >= 1.5) & (cal <= 2.4)).astype(int)
        + (expected >= 7.0).astype(int)
        + ((loc >= 0.455) & (loc < 0.490)).astype(int)
    )
    weak_rows = (~strong_mask.astype(bool)) & (candidate >= 0.25) & (candidate <= 0.30) & (weak_score >= 3)
    selected = np.zeros(len(frame), dtype=bool)
    plateaus: List[Dict[str, Any]] = []
    ordered = frame.sort_values(["segment_id", "t"])
    for segment_id, idx in ordered.groupby("segment_id", sort=False).groups.items():
        idx_list = list(idx)
        local = weak_rows[idx_list]
        for start, end in _runs(local):
            run_idx = np.asarray(idx_list[start : end + 1], dtype=int)
            if run_idx.size < 8:
                continue
            run_cal = cal[run_idx]
            run_loc = loc[run_idx]
            mean_cal = float(np.nanmean(run_cal))
            if not (1.6 <= mean_cal <= 2.6):
                continue
            if float(np.nanstd(run_cal)) > 0.05:
                continue
            if float(np.nanmean(run_loc)) < 0.46 or float(np.nanmax(run_loc)) < 0.464:
                continue
            selected[run_idx] = True
            labels = frame.loc[run_idx, "d_true"].to_numpy(dtype=np.float64)
            plateaus.append(
                {
                    "segment_id": segment_id,
                    "start_t": int(frame.loc[run_idx[0], "t"]),
                    "end_t": int(frame.loc[run_idx[-1], "t"]),
                    "length": int(run_idx.size),
                    "n_d_true2": int((labels == 2.0).sum()),
                    "n_positive": int((labels > 0).sum()),
                    "n_zero": int((labels <= 0).sum()),
                    "mean_localization_score": float(np.nanmean(run_loc)),
                    "max_localization_score": float(np.nanmax(run_loc)),
                }
            )
    return selected, plateaus


def _apply_old_best(frame: pd.DataFrame, z_threshold: float) -> tuple[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]:
    out = frame.copy()
    candidate = out["candidate_score"].to_numpy(dtype=np.float64) >= 0.25
    loc = out["localization_score"].to_numpy(dtype=np.float64)
    cal = out["calibrated_raw_m"].to_numpy(dtype=np.float64)
    strong_candidate = candidate & (cal >= 3.0)
    strong_loc = loc[strong_candidate]
    loc_mean = float(np.nanmean(strong_loc))
    loc_std = float(np.nanstd(strong_loc))
    loc_threshold = loc_mean + float(z_threshold) * max(loc_std, 1e-12)

    mid_high_z = strong_candidate & (loc >= loc_threshold)
    low_lag_high_conf = candidate & (cal < 3.0) & (loc >= 0.490)
    weak_plateau, plateaus = _weak_plateau_mask(out, mid_high_z)
    selected = mid_high_z | low_lag_high_conf | weak_plateau

    out["mid_high_z_selected"] = mid_high_z.astype(int)
    out["low_lag_high_conf_selected"] = low_lag_high_conf.astype(int)
    out["weak_plateau_selected"] = weak_plateau.astype(int)
    out["final_selected"] = selected.astype(int)
    out["d_hat_final"] = np.where(selected, cal, 0.0)
    out["abs_error_positive"] = np.where(out["d_true"].to_numpy(dtype=np.float64) > 0, np.abs(out["d_hat_final"] - out["d_true"]), np.nan)

    source = np.full(len(out), "not_selected", dtype=object)
    source[mid_high_z] = "mid_high_z"
    source[low_lag_high_conf] = "low_lag_high_conf"
    source[weak_plateau] = "weak_plateau"
    overlap = (mid_high_z.astype(int) + low_lag_high_conf.astype(int) + weak_plateau.astype(int)) > 1
    source[overlap] = "selected_overlap"
    out["prediction_source"] = source

    info = {
        "z_threshold": float(z_threshold),
        "strong_candidate_rows": int(strong_candidate.sum()),
        "strong_candidate_loc_mean": loc_mean,
        "strong_candidate_loc_std": loc_std,
        "strong_candidate_loc_threshold": loc_threshold,
        "mid_high_z_rows": int(mid_high_z.sum()),
        "low_lag_high_conf_rows": int(low_lag_high_conf.sum()),
        "weak_plateau_rows": int(weak_plateau.sum()),
        "final_selected_rows": int(selected.sum()),
    }
    return out, info, plateaus


def _metrics(frame: pd.DataFrame, label_col: str, group_col: str) -> Dict[str, Any]:
    labels = frame[label_col].to_numpy(dtype=np.float64)
    true = labels > 0
    pred = frame["final_selected"].to_numpy(dtype=bool)
    d_hat = frame["d_hat_final"].to_numpy(dtype=np.float64)
    tp = int(np.logical_and(pred, true).sum())
    fp = int(np.logical_and(pred, ~true).sum())
    fn = int(np.logical_and(~pred, true).sum())
    tn = int(np.logical_and(~pred, ~true).sum())
    row: Dict[str, Any] = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "overall_recall": float(tp / max(tp + fn, 1)),
        "FAR": float(fp / max(fp + tn, 1)),
        "precision": float(tp / max(tp + fp, 1)),
        "pos_MAE": float(np.mean(np.abs(d_hat[true] - labels[true]))) if true.any() else float("nan"),
        "zero_E_d_hat": float(np.mean(d_hat[~true])) if (~true).any() else float("nan"),
    }
    for value in [2.0, 4.0, 6.0]:
        group = labels == value
        selected = pred & group
        row[f"d{int(value)}_recall"] = float(selected.sum() / max(int(group.sum()), 1))
        row[f"d{int(value)}_selected"] = int(selected.sum())
        row[f"d{int(value)}_MAE"] = float(np.mean(np.abs(d_hat[group] - labels[group]))) if group.any() else float("nan")

    errors = []
    hits = []
    for block in _positive_blocks(frame, label_col=label_col, group_col=group_col):
        true_peak = float(np.nanmax(labels[block]))
        pred_peak = float(np.nanmax(d_hat[block]))
        errors.append(abs(pred_peak - true_peak))
        hits.append(float(abs(int(np.floor(pred_peak + 0.5)) - int(true_peak)) <= 1))
    row["peak_error"] = float(np.mean(errors)) if errors else float("nan")
    row["peak_hit_at_pm1"] = float(np.mean(hits)) if hits else float("nan")
    row["n_positive_blocks"] = int(len(errors))
    return row


def _segment_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for segment_id, part in frame.sort_values(["segment_id", "t"]).groupby("segment_id", sort=False):
        labels = part["d_true"].to_numpy(dtype=np.float64)
        pred = part["final_selected"].to_numpy(dtype=bool)
        d_hat = part["d_hat_final"].to_numpy(dtype=np.float64)
        positive = labels > 0
        rows.append(
            {
                "segment_id": int(segment_id),
                "rows": int(len(part)),
                "positive_rows": int(positive.sum()),
                "true_peak": float(np.nanmax(labels)),
                "pred_peak": float(np.nanmax(d_hat)),
                "selected_rows": int(pred.sum()),
                "tp": int(np.logical_and(pred, positive).sum()),
                "fp": int(np.logical_and(pred, ~positive).sum()),
                "fn": int(np.logical_and(~pred, positive).sum()),
                "recall": float(np.logical_and(pred, positive).sum() / max(int(positive.sum()), 1)),
                "pos_MAE": float(np.mean(np.abs(d_hat[positive] - labels[positive]))) if positive.any() else float("nan"),
                "mid_high_z_rows": int(part["mid_high_z_selected"].sum()),
                "low_lag_high_conf_rows": int(part["low_lag_high_conf_selected"].sum()),
                "weak_plateau_rows": int(part["weak_plateau_selected"].sum()),
            }
        )
    return pd.DataFrame(rows)


def _false_positive_runs(frame: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    ordered = frame.sort_values(["segment_id", "t"])
    for segment_id, part in ordered.groupby("segment_id", sort=False):
        idx = part.index.to_numpy()
        local = (part["final_selected"].to_numpy(dtype=bool)) & (part["d_true"].to_numpy(dtype=np.float64) <= 0)
        for start, end in _runs(local):
            run_idx = idx[start : end + 1]
            rows.append(
                {
                    "segment_id": int(segment_id),
                    "start_t": int(frame.loc[run_idx[0], "t"]),
                    "end_t": int(frame.loc[run_idx[-1], "t"]),
                    "length": int(len(run_idx)),
                    "mean_d_hat": float(frame.loc[run_idx, "d_hat_final"].mean()),
                    "max_d_hat": float(frame.loc[run_idx, "d_hat_final"].max()),
                    "source_counts": ",".join(
                        f"{key}:{value}" for key, value in frame.loc[run_idx, "prediction_source"].value_counts().to_dict().items()
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(["length", "max_d_hat"], ascending=[False, False]).reset_index(drop=True)


def _shade_positive(ax: plt.Axes, part: pd.DataFrame) -> None:
    t = part["t"].to_numpy(dtype=np.float64)
    positive = part["d_true"].to_numpy(dtype=np.float64) > 0
    if t.size == 0:
        return
    for start, end in _runs(positive):
        lo = float(t[start])
        hi = float(t[end])
        ax.axvspan(lo - 0.5, hi + 0.5, color="#4daf4a", alpha=0.12, linewidth=0)


def _plot_positive_windows(frame: pd.DataFrame, out_dir: Path, info: Dict[str, Any]) -> None:
    positive_segments = (
        frame.loc[frame["d_true"].to_numpy(dtype=np.float64) > 0, "segment_id"]
        .drop_duplicates()
        .astype(int)
        .tolist()
    )
    fig, axes = plt.subplots(len(positive_segments), 2, figsize=(18, 3.1 * len(positive_segments)), squeeze=False)
    fig.suptitle("Old seed best full component: local lag prediction windows", fontsize=15)
    for row, segment_id in enumerate(positive_segments):
        part_all = frame.loc[frame["segment_id"].astype(int).eq(segment_id)].sort_values("t")
        pos_t = part_all.loc[part_all["d_true"].to_numpy(dtype=np.float64) > 0, "t"]
        lo = max(int(pos_t.min()) - 40, int(part_all["t"].min()))
        hi = min(int(pos_t.max()) + 40, int(part_all["t"].max()))
        part = part_all.loc[part_all["t"].between(lo, hi)].copy()
        t = part["t"].to_numpy(dtype=np.float64)

        ax = axes[row, 0]
        _shade_positive(ax, part)
        ax.step(t, part["d_true"], where="mid", color="black", linewidth=2.0, label="d_true")
        ax.plot(t, part["calibrated_raw_m"], color="#999999", linewidth=1.0, alpha=0.7, label="calibrated_raw_m")
        ax.step(t, part["d_hat_final"], where="mid", color="#e41a1c", linewidth=1.6, label="d_hat")
        for source, color in COLORS.items():
            source_part = part.loc[part["prediction_source"].eq(source)]
            if not source_part.empty:
                ax.scatter(
                    source_part["t"],
                    source_part["d_hat_final"],
                    s=20,
                    color=color,
                    label=source,
                    zorder=4,
                )
        ax.set_title(f"segment {segment_id}: lag prediction")
        ax.set_ylabel("lag")
        ax.set_ylim(-0.25, max(7.0, float(part[["d_true", "d_hat_final", "calibrated_raw_m"]].max().max()) + 0.7))
        ax.grid(alpha=0.25)
        if row == 0:
            ax.legend(loc="upper left", ncol=3, fontsize=8)

        ax2 = axes[row, 1]
        _shade_positive(ax2, part)
        ax2.plot(t, part["localization_score"], color="#e7298a", linewidth=1.4, label="localization_score")
        ax2.axhline(float(info["strong_candidate_loc_threshold"]), color="#e7298a", linestyle="--", linewidth=1.0, label="z=-0.5 threshold")
        ax2.axhline(0.490, color="#e7298a", linestyle=":", linewidth=1.0, label="low-lag 0.490")
        ax2.plot(t, part["candidate_score"], color="#1f77b4", linewidth=1.1, label="candidate_score")
        ax2.axhline(0.25, color="#1f77b4", linestyle="--", linewidth=1.0)
        ax2.set_title(f"segment {segment_id}: selector scores")
        ax2.set_ylabel("score")
        ax2.set_ylim(0.22, 1.02)
        ax2.grid(alpha=0.25)
        if row == 0:
            ax2.legend(loc="upper left", ncol=2, fontsize=8)

    for ax in axes[-1, :]:
        ax.set_xlabel("t")
    fig.tight_layout()
    fig.savefig(out_dir / "old_seed_best_positive_segment_windows.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_overview(frame: pd.DataFrame, out_dir: Path) -> None:
    segments = frame["segment_id"].drop_duplicates().astype(int).tolist()
    fig, axes = plt.subplots(len(segments), 1, figsize=(18, 2.1 * len(segments)), sharex=False)
    fig.suptitle("Old seed best full component: all test segment lag overview", fontsize=15)
    for ax, segment_id in zip(np.ravel(axes), segments):
        part = frame.loc[frame["segment_id"].astype(int).eq(segment_id)].sort_values("t")
        t = part["t"].to_numpy(dtype=np.float64)
        _shade_positive(ax, part)
        ax.step(t, part["d_true"], where="mid", color="black", linewidth=1.6, label="d_true")
        ax.step(t, part["d_hat_final"], where="mid", color="#e41a1c", linewidth=1.0, label="d_hat")
        selected = part.loc[part["final_selected"].astype(bool)]
        ax.scatter(selected["t"], selected["d_hat_final"], s=8, color="#e41a1c", alpha=0.55)
        ax.set_ylabel(f"seg {segment_id}")
        ax.set_ylim(-0.2, 6.8)
        ax.grid(alpha=0.18)
    axes[0].legend(loc="upper left", ncol=2, fontsize=8)
    axes[-1].set_xlabel("t")
    fig.tight_layout()
    fig.savefig(out_dir / "old_seed_best_all_segment_overview.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_scatter(frame: pd.DataFrame, segment_summary: pd.DataFrame, out_dir: Path) -> None:
    positive = frame.loc[frame["d_true"].to_numpy(dtype=np.float64) > 0].copy()
    rng = np.random.default_rng(7)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax = axes[0]
    jitter = rng.normal(0, 0.035, size=len(positive))
    selected = positive["final_selected"].to_numpy(dtype=bool)
    ax.scatter(positive.loc[~selected, "d_true"] + jitter[~selected], positive.loc[~selected, "d_hat_final"], color="#bbbbbb", s=32, label="missed")
    ax.scatter(positive.loc[selected, "d_true"] + jitter[selected], positive.loc[selected, "d_hat_final"], color="#e41a1c", s=32, label="selected")
    ax.plot([0, 6.5], [0, 6.5], color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("d_true")
    ax.set_ylabel("d_hat")
    ax.set_title("positive rows: d_hat vs d_true")
    ax.set_xlim(1.2, 6.8)
    ax.set_ylim(-0.2, 6.8)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")

    by_d = positive.groupby("d_true").agg(
        recall=("final_selected", "mean"),
        mae=("abs_error_positive", "mean"),
        rows=("d_true", "size"),
    ).reset_index()
    ax = axes[1]
    x = np.arange(len(by_d))
    ax.bar(x - 0.18, by_d["recall"], width=0.36, color="#4daf4a", label="recall")
    ax.bar(x + 0.18, by_d["mae"] / max(float(by_d["mae"].max()), 1e-9), width=0.36, color="#984ea3", label="MAE / max")
    ax.set_xticks(x)
    ax.set_xticklabels([f"d={int(v)}" for v in by_d["d_true"]])
    ax.set_ylim(0, 1.05)
    ax.set_title("recall and scaled MAE by lag")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")

    ax = axes[2]
    source_counts = frame.loc[frame["final_selected"].astype(bool), "prediction_source"].value_counts()
    ax.bar(source_counts.index.astype(str), source_counts.values, color=[COLORS.get(key, "#777777") for key in source_counts.index])
    ax.set_title("selected rows by source")
    ax.set_ylabel("#rows")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "old_seed_best_prediction_scatter_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    positive_segments = segment_summary.loc[segment_summary["positive_rows"] > 0].copy()
    x = np.arange(len(positive_segments))
    ax.bar(x, positive_segments["recall"], color="#4daf4a")
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in positive_segments["segment_id"]])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("segment_id")
    ax.set_ylabel("row recall")
    ax.set_title("positive segment row recall")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "old_seed_best_positive_segment_recall.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize old-seed lag predictions for the best full component probe.")
    parser.add_argument("--train-series", default="outputs/r18_light_veto_filter_smoke2/light_veto_train_filtered.csv")
    parser.add_argument("--eval-series", default="outputs/r18_light_veto_filter_smoke2/light_veto_eval_filtered.csv")
    parser.add_argument("--output-dir", default="outputs/r36_old_seed_best_lag_prediction_viz")
    parser.add_argument("--label-col", default="d_true")
    parser.add_argument("--group-col", default="segment_id")
    parser.add_argument("--z-threshold", type=float, default=-0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = _path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train = _read_csv(_path(args.train_series))
    evaluation = _read_csv(_path(args.eval_series))
    fit = _fit_affine_raw_m(train, label_col=str(args.label_col))
    frame = _enrich(evaluation, fit)
    frame, selector_info, plateaus = _apply_old_best(frame, float(args.z_threshold))

    metrics = _metrics(frame, label_col=str(args.label_col), group_col=str(args.group_col))
    segment_summary = _segment_summary(frame)
    false_positive_runs = _false_positive_runs(frame)

    keep_cols = [
        "segment_id",
        "t",
        "d_true",
        "candidate_score",
        "localization_score",
        "raw_m",
        "calibrated_raw_m",
        "d_hat_final",
        "final_selected",
        "prediction_source",
        "mid_high_z_selected",
        "low_lag_high_conf_selected",
        "weak_plateau_selected",
        "expected_lag",
        "positive_margin",
        "abs_error_positive",
    ]
    frame[keep_cols].to_csv(out_dir / "old_seed_best_predictions.csv", index=False)
    pd.DataFrame([{**selector_info, **metrics}]).to_csv(out_dir / "old_seed_best_metrics.csv", index=False)
    segment_summary.to_csv(out_dir / "old_seed_best_by_segment.csv", index=False)
    false_positive_runs.to_csv(out_dir / "old_seed_best_false_positive_runs.csv", index=False)
    pd.DataFrame(plateaus).to_csv(out_dir / "old_seed_best_weak_plateau_segments.csv", index=False)

    _plot_positive_windows(frame, out_dir, selector_info)
    _plot_overview(frame, out_dir)
    _plot_scatter(frame, segment_summary, out_dir)

    report = {
        "affine_raw_m": fit,
        "selector": selector_info,
        "metrics": metrics,
        "outputs": {
            "predictions": (out_dir / "old_seed_best_predictions.csv").as_posix(),
            "metrics": (out_dir / "old_seed_best_metrics.csv").as_posix(),
            "by_segment": (out_dir / "old_seed_best_by_segment.csv").as_posix(),
            "false_positive_runs": (out_dir / "old_seed_best_false_positive_runs.csv").as_posix(),
            "positive_windows": (out_dir / "old_seed_best_positive_segment_windows.png").as_posix(),
            "overview": (out_dir / "old_seed_best_all_segment_overview.png").as_posix(),
            "scatter_summary": (out_dir / "old_seed_best_prediction_scatter_summary.png").as_posix(),
            "segment_recall": (out_dir / "old_seed_best_positive_segment_recall.png").as_posix(),
        },
    }
    (out_dir / "old_seed_best_visualization_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    display = {
        "overall_recall": metrics["overall_recall"],
        "FAR": metrics["FAR"],
        "d2_recall": metrics["d2_recall"],
        "d4_recall": metrics["d4_recall"],
        "d6_recall": metrics["d6_recall"],
        "pos_MAE": metrics["pos_MAE"],
        "peak_hit_at_pm1": metrics["peak_hit_at_pm1"],
        "selected_rows": selector_info["final_selected_rows"],
    }
    print(json.dumps(display, indent=2))
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
