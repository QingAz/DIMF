#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

if not os.environ.get("MPLCONFIGDIR"):
    _mpl_dir = Path.cwd() / ".matplotlib-codex"
    _mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(_mpl_dir)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize predicted lag behavior on a selected low-lag cohort across one or more runs."
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run spec formatted as name=/abs/or/relative/path/to/test_joined_single_model.csv",
    )
    parser.add_argument(
        "--mask",
        choices=["lag_gt_zero", "inject_flag_zero"],
        default="lag_gt_zero",
        help="Which cohort to visualize.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for plots and summary CSV")
    parser.add_argument(
        "--title",
        default="",
        help="Figure title",
    )
    return parser.parse_args()


def _parse_run_spec(spec: str) -> Tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Run spec must be formatted as name=path, got: {spec}")
    name, path_str = spec.split("=", 1)
    name = name.strip()
    path = Path(path_str.strip())
    if not name:
        raise ValueError(f"Run name is empty in spec: {spec}")
    return name, path


def _require_columns(frame: pd.DataFrame, path: Path) -> None:
    required = {"lag_gt", "pred_expected_lag", "pred_argmax_lag", "pred_nonzero_prob"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")


def _bucketize_argmax(values: pd.Series) -> Dict[str, float]:
    arr = values.to_numpy(dtype=np.int64)
    n = max(len(arr), 1)
    return {
        "lag0": float(np.sum(arr == 0)) / n,
        "lag1": float(np.sum(arr == 1)) / n,
        "lag2_3": float(np.sum((arr >= 2) & (arr <= 3))) / n,
        "lag4p": float(np.sum(arr >= 4)) / n,
    }


def _select_mask_frame(frame: pd.DataFrame, mask_name: str, path: Path) -> pd.DataFrame:
    if mask_name == "lag_gt_zero":
        return frame.loc[frame["lag_gt"].to_numpy(dtype=np.int64) == 0].copy()
    if mask_name == "inject_flag_zero":
        if "inject_flag" not in frame.columns:
            raise ValueError(f"{path} is missing required column for inject_flag_zero mask: inject_flag")
        return frame.loc[frame["inject_flag"].to_numpy(dtype=np.int64) == 0].copy()
    raise ValueError(f"Unsupported mask: {mask_name}")


def _cohort_summary(name: str, mask_name: str, frame: pd.DataFrame) -> Dict[str, float | str]:
    if frame.empty:
        return {
            "run": name,
            "mask": mask_name,
            "n_rows": 0,
            "mean_pred_expected_lag": np.nan,
            "median_pred_expected_lag": np.nan,
            "p90_pred_expected_lag": np.nan,
            "mean_pred_nonzero_prob": np.nan,
            "median_pred_nonzero_prob": np.nan,
            "share_argmax_lag0": np.nan,
            "share_argmax_gt0": np.nan,
            "share_pred_expected_lag_gt0p5": np.nan,
            "share_pred_nonzero_prob_ge0p5": np.nan,
        }

    argmax = frame["pred_argmax_lag"].to_numpy(dtype=np.int64)
    pred_expected = frame["pred_expected_lag"].to_numpy(dtype=np.float64)
    pred_nonzero = frame["pred_nonzero_prob"].to_numpy(dtype=np.float64)
    return {
        "run": name,
        "mask": mask_name,
        "n_rows": int(len(frame)),
        "mean_pred_expected_lag": float(np.mean(pred_expected)),
        "median_pred_expected_lag": float(np.median(pred_expected)),
        "p90_pred_expected_lag": float(np.quantile(pred_expected, 0.9)),
        "mean_pred_nonzero_prob": float(np.mean(pred_nonzero)),
        "median_pred_nonzero_prob": float(np.median(pred_nonzero)),
        "share_argmax_lag0": float(np.mean(argmax == 0)),
        "share_argmax_gt0": float(np.mean(argmax > 0)),
        "share_pred_expected_lag_gt0p5": float(np.mean(pred_expected > 0.5)),
        "share_pred_nonzero_prob_ge0p5": float(np.mean(pred_nonzero >= 0.5)),
    }


def _plot_expected_hist(ax: plt.Axes, run_frames: Dict[str, pd.DataFrame], colors: Dict[str, str]) -> None:
    bins = np.linspace(0.0, 6.0, 25)
    for name, frame in run_frames.items():
        values = frame["pred_expected_lag"].to_numpy(dtype=np.float64)
        ax.hist(
            np.clip(values, bins[0], bins[-1]),
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2.0,
            label=name,
            color=colors[name],
        )
    ax.set_title("Predicted Expected Lag")
    ax.set_xlabel("pred_expected_lag")
    ax.set_ylabel("density")
    ax.grid(alpha=0.2)


def _plot_expected_cdf(ax: plt.Axes, run_frames: Dict[str, pd.DataFrame], colors: Dict[str, str]) -> None:
    for name, frame in run_frames.items():
        values = np.sort(frame["pred_expected_lag"].to_numpy(dtype=np.float64))
        y = np.linspace(0.0, 1.0, len(values), endpoint=False) if len(values) else np.array([])
        ax.plot(values, y, linewidth=2.0, label=name, color=colors[name])
    ax.set_title("CDF of Expected Lag")
    ax.set_xlabel("pred_expected_lag")
    ax.set_ylabel("CDF")
    ax.grid(alpha=0.2)


def _plot_nonzero_hist(ax: plt.Axes, run_frames: Dict[str, pd.DataFrame], colors: Dict[str, str]) -> None:
    bins = np.linspace(0.0, 1.0, 25)
    for name, frame in run_frames.items():
        values = frame["pred_nonzero_prob"].to_numpy(dtype=np.float64)
        ax.hist(
            np.clip(values, bins[0], bins[-1]),
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2.0,
            label=name,
            color=colors[name],
        )
    ax.set_title("Predicted Nonzero Probability")
    ax.set_xlabel("pred_nonzero_prob = 1 - pi(0)")
    ax.set_ylabel("density")
    ax.grid(alpha=0.2)


def _plot_argmax_bars(ax: plt.Axes, run_frames: Dict[str, pd.DataFrame], colors: Dict[str, str]) -> None:
    names = list(run_frames.keys())
    x = np.arange(len(names), dtype=np.float64)
    width = 0.72
    bottoms = np.zeros(len(names), dtype=np.float64)
    bucket_order = ["lag0", "lag1", "lag2_3", "lag4p"]
    bucket_labels = {
        "lag0": "argmax = 0",
        "lag1": "argmax = 1",
        "lag2_3": "argmax = 2-3",
        "lag4p": "argmax >= 4",
    }
    bucket_colors = {
        "lag0": "#1d4ed8",
        "lag1": "#60a5fa",
        "lag2_3": "#f59e0b",
        "lag4p": "#dc2626",
    }

    for bucket in bucket_order:
        values = []
        for name in names:
            values.append(_bucketize_argmax(run_frames[name]["pred_argmax_lag"])[bucket])
        values_arr = np.asarray(values, dtype=np.float64)
        ax.bar(x, values_arr, width=width, bottom=bottoms, color=bucket_colors[bucket], label=bucket_labels[bucket])
        bottoms += values_arr

    ax.set_title("Argmax Lag Bucket Share")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("share of rows")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.2)


def _default_title(mask_name: str) -> str:
    if mask_name == "inject_flag_zero":
        return "Uninjected-Region Lag Prediction Behavior"
    return "Zero-Region Lag Prediction Behavior"


def _output_stem(mask_name: str) -> str:
    if mask_name == "inject_flag_zero":
        return "inject_flag_zero_behavior"
    return "zero_region_behavior"


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    run_frames: Dict[str, pd.DataFrame] = {}
    summary_rows: List[Dict[str, float | str]] = []

    for spec in args.run:
        name, path = _parse_run_spec(spec)
        frame = pd.read_csv(path)
        _require_columns(frame, path)
        run_frames[name] = frame

        cohort = _select_mask_frame(frame, args.mask, path)
        summary_rows.append(_cohort_summary(name, args.mask, cohort))

    cohort_run_frames = {
        name: _select_mask_frame(frame, args.mask, Path(name))
        for name, frame in run_frames.items()
    }

    color_cycle = plt.get_cmap("tab10").colors
    colors = {name: color_cycle[idx % len(color_cycle)] for idx, name in enumerate(cohort_run_frames.keys())}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.reshape(2, 2)
    _plot_expected_hist(axes[0, 0], cohort_run_frames, colors)
    _plot_expected_cdf(axes[0, 1], cohort_run_frames, colors)
    _plot_nonzero_hist(axes[1, 0], cohort_run_frames, colors)
    _plot_argmax_bars(axes[1, 1], cohort_run_frames, colors)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=max(2, len(handles)), frameon=False, bbox_to_anchor=(0.5, 0.99))
    fig.suptitle(args.title or _default_title(args.mask), fontsize=15, y=1.02)
    fig.tight_layout()
    stem = _output_stem(args.mask)
    fig.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / f"{stem}_summary.csv", index=False)

    print(f"Wrote figure to {output_dir / f'{stem}.png'}")
    print(f"Wrote summary to {output_dir / f'{stem}_summary.csv'}")


if __name__ == "__main__":
    main()
