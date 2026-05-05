#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-codex"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.visualize_old_seed_lag_predictions import (  # noqa: E402
    COLORS,
    _apply_old_best,
    _enrich,
    _false_positive_runs,
    _fit_affine_raw_m,
    _metrics,
    _path,
    _read_csv,
    _runs,
    _shade_positive,
)


SPLIT_ORDER = ["train", "val", "test"]


def _split_rank(value: str) -> int:
    return SPLIT_ORDER.index(value) if value in SPLIT_ORDER else len(SPLIT_ORDER)


def _apply_per_split(frame: pd.DataFrame, z_threshold: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    parts: List[pd.DataFrame] = []
    selector_infos: List[Dict[str, Any]] = []
    plateau_rows: List[Dict[str, Any]] = []
    for split_name, part in frame.groupby("source_split", sort=False):
        predicted, info, plateaus = _apply_old_best(part.reset_index(drop=True), z_threshold)
        predicted["source_split"] = split_name
        predicted["full_group_id"] = predicted["source_split"].astype(str) + ":" + predicted["segment_id"].astype(str)
        parts.append(predicted)
        selector_infos.append({"source_split": split_name, **info})
        for plateau in plateaus:
            plateau_rows.append({"source_split": split_name, **plateau})
    out = pd.concat(parts, ignore_index=True)
    out["_split_rank"] = out["source_split"].map(_split_rank)
    out = out.sort_values(["_split_rank", "segment_id", "t"]).drop(columns=["_split_rank"]).reset_index(drop=True)
    return out, pd.DataFrame(selector_infos), pd.DataFrame(plateau_rows)


def _metrics_by_split(frame: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    rows.append({"source_split": "all", **_metrics(frame, label_col="d_true", group_col="full_group_id")})
    for split_name, part in frame.groupby("source_split", sort=False):
        rows.append({"source_split": split_name, **_metrics(part, label_col="d_true", group_col="full_group_id")})
    return pd.DataFrame(rows)


def _segment_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    ordered = frame.sort_values(["source_split", "segment_id", "t"])
    for (split_name, segment_id), part in ordered.groupby(["source_split", "segment_id"], sort=False):
        labels = part["d_true"].to_numpy(dtype=np.float64)
        pred = part["final_selected"].to_numpy(dtype=bool)
        d_hat = part["d_hat_final"].to_numpy(dtype=np.float64)
        positive = labels > 0
        rows.append(
            {
                "source_split": split_name,
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


def _plot_full_overview(frame: pd.DataFrame, out_dir: Path) -> None:
    segments = (
        frame[["source_split", "segment_id"]]
        .drop_duplicates()
        .assign(_rank=lambda x: x["source_split"].map(_split_rank))
        .sort_values(["_rank", "segment_id"])
        .drop(columns=["_rank"])
        .to_dict(orient="records")
    )
    fig, axes = plt.subplots(len(segments), 1, figsize=(20, max(2.0 * len(segments), 8)), sharex=False)
    fig.suptitle("Old seed best full component: train/val/test lag overview", fontsize=16)
    axes_arr = np.ravel(axes)
    for ax, item in zip(axes_arr, segments):
        split_name = item["source_split"]
        segment_id = int(item["segment_id"])
        part = frame.loc[frame["source_split"].eq(split_name) & frame["segment_id"].astype(int).eq(segment_id)].sort_values("t")
        t = part["t"].to_numpy(dtype=np.float64)
        _shade_positive(ax, part)
        ax.step(t, part["d_true"], where="mid", color="black", linewidth=1.5, label="d_true")
        ax.step(t, part["d_hat_final"], where="mid", color="#e41a1c", linewidth=0.95, label="d_hat")
        selected = part.loc[part["final_selected"].astype(bool)]
        ax.scatter(selected["t"], selected["d_hat_final"], s=7, color="#e41a1c", alpha=0.45)
        ax.set_ylabel(f"{split_name}\\nseg {segment_id}", rotation=0, ha="right", va="center")
        ax.set_ylim(-0.2, 6.8)
        ax.grid(alpha=0.18)
    axes_arr[0].legend(loc="upper left", ncol=2, fontsize=8)
    axes_arr[-1].set_xlabel("t")
    fig.tight_layout()
    fig.savefig(out_dir / "old_seed_best_full_dataset_overview.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_positive_windows(frame: pd.DataFrame, selector_info: pd.DataFrame, out_dir: Path) -> None:
    positives = (
        frame.loc[frame["d_true"].to_numpy(dtype=np.float64) > 0, ["source_split", "segment_id"]]
        .drop_duplicates()
        .assign(_rank=lambda x: x["source_split"].map(_split_rank))
        .sort_values(["_rank", "segment_id"])
        .drop(columns=["_rank"])
        .to_dict(orient="records")
    )
    fig, axes = plt.subplots(len(positives), 2, figsize=(20, max(3.0 * len(positives), 9)), squeeze=False)
    fig.suptitle("Old seed best full component: positive windows across train/val/test", fontsize=16)
    thresholds = selector_info.set_index("source_split")["strong_candidate_loc_threshold"].to_dict()
    for row, item in enumerate(positives):
        split_name = item["source_split"]
        segment_id = int(item["segment_id"])
        part_all = frame.loc[frame["source_split"].eq(split_name) & frame["segment_id"].astype(int).eq(segment_id)].sort_values("t")
        pos_t = part_all.loc[part_all["d_true"].to_numpy(dtype=np.float64) > 0, "t"]
        lo = max(int(pos_t.min()) - 35, int(part_all["t"].min()))
        hi = min(int(pos_t.max()) + 35, int(part_all["t"].max()))
        part = part_all.loc[part_all["t"].between(lo, hi)].copy()
        t = part["t"].to_numpy(dtype=np.float64)

        ax = axes[row, 0]
        _shade_positive(ax, part)
        ax.step(t, part["d_true"], where="mid", color="black", linewidth=1.8, label="d_true")
        ax.plot(t, part["calibrated_raw_m"], color="#999999", linewidth=1.0, alpha=0.65, label="calibrated_raw_m")
        ax.step(t, part["d_hat_final"], where="mid", color="#e41a1c", linewidth=1.4, label="d_hat")
        for source, color in COLORS.items():
            source_part = part.loc[part["prediction_source"].eq(source)]
            if not source_part.empty:
                ax.scatter(source_part["t"], source_part["d_hat_final"], s=16, color=color, label=source, zorder=4)
        ax.set_title(f"{split_name} segment {segment_id}: lag prediction")
        ax.set_ylabel("lag")
        ax.set_ylim(-0.25, 7.0)
        ax.grid(alpha=0.24)
        if row == 0:
            ax.legend(loc="upper left", ncol=3, fontsize=8)

        ax2 = axes[row, 1]
        _shade_positive(ax2, part)
        ax2.plot(t, part["localization_score"], color="#e7298a", linewidth=1.2, label="localization_score")
        ax2.axhline(float(thresholds[split_name]), color="#e7298a", linestyle="--", linewidth=1.0, label="split z=-0.5 threshold")
        ax2.axhline(0.490, color="#e7298a", linestyle=":", linewidth=1.0, label="low-lag 0.490")
        ax2.plot(t, part["candidate_score"], color="#1f77b4", linewidth=1.0, label="candidate_score")
        ax2.axhline(0.25, color="#1f77b4", linestyle="--", linewidth=1.0)
        ax2.set_title(f"{split_name} segment {segment_id}: selector scores")
        ax2.set_ylabel("score")
        ax2.set_ylim(0.22, 1.02)
        ax2.grid(alpha=0.24)
        if row == 0:
            ax2.legend(loc="upper left", ncol=2, fontsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel("t")
    fig.tight_layout()
    fig.savefig(out_dir / "old_seed_best_full_dataset_positive_windows.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_split_summary(frame: pd.DataFrame, metrics: pd.DataFrame, segment_summary: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    split_metrics = metrics.loc[metrics["source_split"].isin(SPLIT_ORDER)].copy()
    split_metrics["_rank"] = split_metrics["source_split"].map(_split_rank)
    split_metrics = split_metrics.sort_values("_rank")
    x = np.arange(len(split_metrics))

    ax = axes[0]
    width = 0.22
    ax.bar(x - width, split_metrics["overall_recall"], width=width, color="#4daf4a", label="overall recall")
    ax.bar(x, split_metrics["d2_recall"], width=width, color="#377eb8", label="d=2 recall")
    ax.bar(x + width, split_metrics["d4_recall"], width=width, color="#984ea3", label="d=4 recall")
    ax.scatter(x + width * 2.1, split_metrics["d6_recall"], color="#ff7f00", label="d=6 recall", zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(split_metrics["source_split"])
    ax.set_ylim(0, 1.05)
    ax.set_title("recall by split")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower left", fontsize=8)

    ax = axes[1]
    ax.bar(x - 0.18, split_metrics["FAR"], width=0.36, color="#e41a1c", label="FAR")
    ax.bar(x + 0.18, split_metrics["pos_MAE"], width=0.36, color="#984ea3", label="pos-MAE")
    ax.set_xticks(x)
    ax.set_xticklabels(split_metrics["source_split"])
    ax.set_title("FAR and positive MAE")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left")

    ax = axes[2]
    selected = frame.loc[frame["final_selected"].astype(bool)]
    counts = selected.groupby(["source_split", "prediction_source"]).size().unstack(fill_value=0)
    counts = counts.reindex(SPLIT_ORDER).fillna(0)
    bottom = np.zeros(len(counts))
    for source, color in COLORS.items():
        values = counts[source].to_numpy(dtype=float) if source in counts.columns else np.zeros(len(counts))
        ax.bar(np.arange(len(counts)), values, bottom=bottom, color=color, label=source)
        bottom += values
    ax.set_xticks(np.arange(len(counts)))
    ax.set_xticklabels(counts.index)
    ax.set_title("selected rows by source")
    ax.set_ylabel("#rows")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "old_seed_best_full_dataset_split_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    positive_segments = segment_summary.loc[segment_summary["positive_rows"] > 0].copy()
    positive_segments["_rank"] = positive_segments["source_split"].map(_split_rank)
    positive_segments = positive_segments.sort_values(["_rank", "segment_id"])
    labels = [f"{row.source_split}\\n{int(row.segment_id)}" for row in positive_segments.itertuples()]
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.bar(np.arange(len(positive_segments)), positive_segments["recall"], color="#4daf4a")
    ax.set_xticks(np.arange(len(positive_segments)))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("row recall")
    ax.set_title("positive segment recall across full dataset")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "old_seed_best_full_dataset_positive_segment_recall.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize old-seed best lag predictions across train/val/test.")
    parser.add_argument("--train-series", default="outputs/r18_light_veto_filter_smoke2/light_veto_train_filtered.csv")
    parser.add_argument("--eval-series", default="outputs/r18_light_veto_filter_smoke2/light_veto_eval_filtered.csv")
    parser.add_argument("--output-dir", default="outputs/r37_old_seed_best_full_dataset_viz")
    parser.add_argument("--z-threshold", type=float, default=-0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = _path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = _read_csv(_path(args.train_series))
    evaluation = _read_csv(_path(args.eval_series))
    if "source_split" not in train.columns:
        train["source_split"] = "train"
    evaluation["source_split"] = "test"
    fit = _fit_affine_raw_m(train, label_col="d_true")
    full = pd.concat([train, evaluation], ignore_index=True)
    full = _enrich(full, fit)
    predicted, selector_info, plateaus = _apply_per_split(full, float(args.z_threshold))

    metrics = _metrics_by_split(predicted)
    segment_summary = _segment_summary(predicted)
    false_positive_runs = _false_positive_runs(predicted)

    keep_cols = [
        "source_split",
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
    predicted[keep_cols].to_csv(out_dir / "old_seed_best_full_dataset_predictions.csv", index=False)
    selector_info.to_csv(out_dir / "old_seed_best_full_dataset_selector_thresholds.csv", index=False)
    metrics.to_csv(out_dir / "old_seed_best_full_dataset_metrics_by_split.csv", index=False)
    segment_summary.to_csv(out_dir / "old_seed_best_full_dataset_by_segment.csv", index=False)
    false_positive_runs.to_csv(out_dir / "old_seed_best_full_dataset_false_positive_runs.csv", index=False)
    plateaus.to_csv(out_dir / "old_seed_best_full_dataset_weak_plateau_segments.csv", index=False)

    _plot_full_overview(predicted, out_dir)
    _plot_positive_windows(predicted, selector_info, out_dir)
    _plot_split_summary(predicted, metrics, segment_summary, out_dir)

    report = {
        "affine_raw_m": fit,
        "selector_thresholds_by_split": selector_info.to_dict(orient="records"),
        "metrics_by_split": metrics.to_dict(orient="records"),
        "outputs": {
            "predictions": (out_dir / "old_seed_best_full_dataset_predictions.csv").as_posix(),
            "metrics_by_split": (out_dir / "old_seed_best_full_dataset_metrics_by_split.csv").as_posix(),
            "by_segment": (out_dir / "old_seed_best_full_dataset_by_segment.csv").as_posix(),
            "overview": (out_dir / "old_seed_best_full_dataset_overview.png").as_posix(),
            "positive_windows": (out_dir / "old_seed_best_full_dataset_positive_windows.png").as_posix(),
            "split_summary": (out_dir / "old_seed_best_full_dataset_split_summary.png").as_posix(),
            "positive_segment_recall": (out_dir / "old_seed_best_full_dataset_positive_segment_recall.png").as_posix(),
        },
    }
    (out_dir / "old_seed_best_full_dataset_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(metrics.to_csv(index=False, float_format="%.6f"))
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
