#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


GROUP_SPECS = [
    ("positive", "positive: d_true > 0", "#1f77b4"),
    ("easy_negative", "easy negative", "#2ca02c"),
    ("hard_negative", "hard negative", "#d62728"),
    ("q40_false_positive", "q40 false positive", "#9467bd"),
]

METRIC_SPECS = [
    ("unified_confidence", "c_t"),
    ("unified_d_soft", "d_soft"),
    ("d_raw", "d_raw"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dump score distributions for current r40 unified lag scorer outputs."
    )
    parser.add_argument("--base-run-dir", default="outputs/r40_unified_block_lag_scorer")
    parser.add_argument("--runs", default="old,seed134_e2")
    parser.add_argument("--output-dir", default="outputs/r40_score_distribution_dump")
    parser.add_argument("--easy-d-raw-quantile", type=float, default=0.25)
    parser.add_argument("--hard-feature-quantile", type=float, default=0.75)
    parser.add_argument("--hard-min-high-features", type=int, default=2)
    return parser.parse_args()


def _path(text: str | Path) -> Path:
    path = Path(os.path.expandvars(str(text))).expanduser()
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


def _load_run_frame(base_run_dir: Path, run_name: str) -> pd.DataFrame:
    unified = _read_csv(base_run_dir / run_name / "unified_eval_timeseries.csv")
    q40 = _read_csv(base_run_dir / run_name / "q40_baseline_eval_timeseries.csv")
    key_cols = [col for col in ["split", "source_split", "timestamp", "raw_row_index", "segment_id", "t"] if col in unified.columns and col in q40.columns]
    q40_keep = q40[key_cols + [col for col in ["q40_final_selected", "q40_prediction_source"] if col in q40.columns]].copy()
    merged = unified.merge(q40_keep, on=key_cols, how="left")
    merged["q40_final_selected"] = merged["q40_final_selected"].fillna(0).astype(int)
    merged["q40_prediction_source"] = merged["q40_prediction_source"].fillna("none")
    return merged


def _annotate_groups(
    frame: pd.DataFrame,
    easy_d_raw_quantile: float,
    hard_feature_quantile: float,
    hard_min_high_features: int,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    out = frame.copy()
    neg = out["d_true"].to_numpy(dtype=np.float64) == 0
    pos = out["d_true"].to_numpy(dtype=np.float64) > 0
    q40_fp = neg & (out["q40_final_selected"].to_numpy(dtype=np.float64) > 0)

    neg_frame = out.loc[neg].copy()
    easy_threshold = float(np.nanquantile(neg_frame["d_raw"].to_numpy(dtype=np.float64), easy_d_raw_quantile))
    hard_features = ["d_raw", "expected_lag", "p_nonzero", "candidate_score"]
    high_thresholds = {
        col: float(np.nanquantile(neg_frame[col].to_numpy(dtype=np.float64), hard_feature_quantile))
        for col in hard_features
    }

    high_count = np.zeros(len(out), dtype=int)
    for col, threshold in high_thresholds.items():
        high_count += (out[col].to_numpy(dtype=np.float64) >= threshold).astype(int)

    hard = neg & ~q40_fp & (high_count >= int(hard_min_high_features))
    easy = neg & ~q40_fp & ~hard & (out["d_raw"].to_numpy(dtype=np.float64) <= easy_threshold)

    out["positive"] = pos.astype(int)
    out["easy_negative"] = easy.astype(int)
    out["hard_negative"] = hard.astype(int)
    out["q40_false_positive"] = q40_fp.astype(int)
    out["diagnostic_group"] = np.where(
        pos,
        "positive",
        np.where(
            easy,
            "easy_negative",
            np.where(
                hard,
                "hard_negative",
                np.where(q40_fp, "q40_false_positive", "other_negative"),
            ),
        ),
    )
    out["hard_high_feature_count"] = high_count

    metadata = {
        "easy_d_raw_quantile": float(easy_d_raw_quantile),
        "easy_d_raw_threshold": easy_threshold,
        "hard_feature_quantile": float(hard_feature_quantile),
        "hard_feature_thresholds": high_thresholds,
        "hard_min_high_features": int(hard_min_high_features),
        "group_counts": {name: int(out[name].sum()) for name, _, _ in GROUP_SPECS},
        "other_negative_count": int((out["diagnostic_group"] == "other_negative").sum()),
    }
    return out, metadata


def _bandwidth(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size <= 1:
        return 0.1
    std = float(np.nanstd(finite))
    if not np.isfinite(std) or std <= 1e-8:
        return 0.1
    return max(0.15 * std, 1e-3)


def _plot_metric_hist(ax: plt.Axes, frame: pd.DataFrame, metric: str, title: str) -> None:
    finite = frame[metric].to_numpy(dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        ax.set_title(f"{title} hist")
        ax.text(0.5, 0.5, "no finite values", ha="center", va="center", transform=ax.transAxes)
        return
    vmin = float(np.nanmin(finite))
    vmax = float(np.nanmax(finite))
    if metric == "unified_confidence":
        vmin, vmax = 0.0, 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-3

    bins = np.linspace(vmin, vmax, 40)
    for group_name, label, color in GROUP_SPECS:
        mask = frame[group_name].to_numpy(dtype=np.float64) > 0
        values = frame.loc[mask, metric].to_numpy(dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        ax.hist(values, bins=bins, density=True, alpha=0.28, color=color, label=f"{label} (n={values.size})")
    ax.set_title(f"{title} hist")
    ax.set_xlabel(title)
    ax.set_ylabel("density")
    ax.grid(alpha=0.2)


def _plot_metric_kde(ax: plt.Axes, frame: pd.DataFrame, metric: str, title: str) -> None:
    finite = frame[metric].to_numpy(dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        ax.set_title(f"{title} KDE")
        ax.text(0.5, 0.5, "no finite values", ha="center", va="center", transform=ax.transAxes)
        return
    vmin = float(np.nanmin(finite))
    vmax = float(np.nanmax(finite))
    if metric == "unified_confidence":
        vmin, vmax = 0.0, 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-3
    grid = np.linspace(vmin, vmax, 400)[:, None]

    for group_name, label, color in GROUP_SPECS:
        mask = frame[group_name].to_numpy(dtype=np.float64) > 0
        values = frame.loc[mask, metric].to_numpy(dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size < 2:
            continue
        kde = KernelDensity(kernel="gaussian", bandwidth=_bandwidth(values))
        kde.fit(values[:, None])
        density = np.exp(kde.score_samples(grid))
        ax.plot(grid[:, 0], density, color=color, linewidth=2.0, label=f"{label} (n={values.size})")
    ax.set_title(f"{title} KDE")
    ax.set_xlabel(title)
    ax.set_ylabel("density")
    ax.grid(alpha=0.2)


def _distribution_stats(frame: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for metric, _ in METRIC_SPECS:
        for group_name, label, _ in GROUP_SPECS:
            mask = frame[group_name].to_numpy(dtype=np.float64) > 0
            values = frame.loc[mask, metric].to_numpy(dtype=np.float64)
            values = values[np.isfinite(values)]
            rows.append(
                {
                    "metric": metric,
                    "group": group_name,
                    "group_label": label,
                    "n": int(values.size),
                    "mean": float(np.nanmean(values)) if values.size else np.nan,
                    "std": float(np.nanstd(values)) if values.size else np.nan,
                    "p10": float(np.nanpercentile(values, 10)) if values.size else np.nan,
                    "p25": float(np.nanpercentile(values, 25)) if values.size else np.nan,
                    "p50": float(np.nanpercentile(values, 50)) if values.size else np.nan,
                    "p75": float(np.nanpercentile(values, 75)) if values.size else np.nan,
                    "p90": float(np.nanpercentile(values, 90)) if values.size else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _plot_run(frame: pd.DataFrame, run_name: str, output_path: Path) -> None:
    fig, axes = plt.subplots(nrows=2, ncols=len(METRIC_SPECS), figsize=(5.6 * len(METRIC_SPECS), 8.4))
    if len(METRIC_SPECS) == 1:
        axes = np.asarray(axes).reshape(2, 1)
    for col_idx, (metric, title) in enumerate(METRIC_SPECS):
        _plot_metric_hist(axes[0, col_idx], frame, metric=metric, title=title)
        _plot_metric_kde(axes[1, col_idx], frame, metric=metric, title=title)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(f"r40 unified score distributions: {run_name}", fontsize=16, y=0.98)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _run_specs(args: argparse.Namespace) -> List[str]:
    return [part.strip() for part in str(args.runs).split(",") if part.strip()]


def main() -> None:
    args = parse_args()
    base_run_dir = _path(args.base_run_dir)
    out_dir = _path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, Any]] = []
    report: Dict[str, Any] = {
        "component": "dump_unified_score_distributions",
        "base_run_dir": base_run_dir.as_posix(),
        "runs": {},
    }

    for run_name in _run_specs(args):
        run_dir = out_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        frame = _load_run_frame(base_run_dir, run_name)
        annotated, metadata = _annotate_groups(
            frame,
            easy_d_raw_quantile=float(args.easy_d_raw_quantile),
            hard_feature_quantile=float(args.hard_feature_quantile),
            hard_min_high_features=int(args.hard_min_high_features),
        )
        annotated.to_csv(run_dir / "score_distribution_groups.csv", index=False)
        stats = _distribution_stats(annotated)
        stats.to_csv(run_dir / "score_distribution_stats.csv", index=False)
        _plot_run(annotated, run_name=run_name, output_path=run_dir / "score_distribution_panels.png")

        summary_rows.append(
            {
                "run": run_name,
                **metadata["group_counts"],
                "other_negative_count": metadata["other_negative_count"],
                "easy_d_raw_threshold": metadata["easy_d_raw_threshold"],
                **{f"hard_{key}_threshold": value for key, value in metadata["hard_feature_thresholds"].items()},
            }
        )
        report["runs"][run_name] = {
            "metadata": metadata,
            "outputs": {
                "groups_csv": (run_dir / "score_distribution_groups.csv").as_posix(),
                "stats_csv": (run_dir / "score_distribution_stats.csv").as_posix(),
                "panels_png": (run_dir / "score_distribution_panels.png").as_posix(),
            },
        }

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "score_distribution_summary.csv", index=False)
    _write_json(out_dir / "score_distribution_report.json", report)
    print(summary.to_csv(index=False))


if __name__ == "__main__":
    main()
