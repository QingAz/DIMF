#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot lag predictions by local segment blocks.")
    parser.add_argument("--root", type=Path, required=True, help="Root containing seed_* result directories.")
    parser.add_argument("--delete-full", action="store_true", help="Delete the concatenated gt_pred_lag.png.")
    return parser.parse_args()


def _segment_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby("segment_id", sort=True)
        .agg(
            sample_min=("sample_index", "min"),
            sample_max=("sample_index", "max"),
            region_id=("region_id", "first"),
            gt_min=("lag_gt", "min"),
            gt_max=("lag_gt", "max"),
            n=("lag_gt", "size"),
        )
        .reset_index()
    )


def _block_label(frame: pd.DataFrame, segment_id: int) -> str:
    local = frame[frame["segment_id"] == segment_id]
    shapes = sorted(str(item) for item in local["shape_type"].dropna().unique())
    shapes = [item for item in shapes if item != "none"] or ["none"]
    region = int(local["region_id"].iloc[0])
    return f"seg{segment_id:02d}_region{region}_{'_'.join(shapes)}"


def _save_block_plot(path: Path, frame: pd.DataFrame, block_segments: list[int], lag_segment: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    local = frame[frame["segment_id"].isin(block_segments)].copy()
    local = local.sort_values("sample_index", kind="stable")
    fig, ax = plt.subplots(figsize=(12, 3.8))

    ax.plot(local["sample_index"], local["lag_gt"], label="gt", linewidth=1.9)
    ax.plot(local["sample_index"], local["pred_lag_stable"], label="pred", linewidth=1.4)

    for seg in block_segments:
        seg_df = local[local["segment_id"] == seg]
        if seg_df.empty:
            continue
        xmin = float(seg_df["sample_index"].min())
        xmax = float(seg_df["sample_index"].max())
        if seg == lag_segment:
            ax.axvspan(xmin, xmax, color="#ff7f0e", alpha=0.08)
        ax.axvline(xmin, color="0.75", linewidth=0.8, linestyle="--")
        ax.text(
            (xmin + xmax) / 2.0,
            5.12,
            f"seg {seg}",
            ha="center",
            va="top",
            fontsize=8,
            color="0.35",
        )
    xmax = float(local["sample_index"].max())
    ax.axvline(xmax, color="0.75", linewidth=0.8, linestyle="--")

    shapes = sorted(str(item) for item in frame.loc[frame["segment_id"] == lag_segment, "shape_type"].unique())
    shapes = [item for item in shapes if item != "none"] or ["none"]
    ax.set_title(f"GT vs Pred lag - segment {lag_segment} ({', '.join(shapes)}) with no-lag context")
    ax.set_xlabel("sample")
    ax.set_ylabel("lag")
    ax.set_ylim(-0.25, 5.35)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_seed(seed_dir: Path, delete_full: bool) -> list[Path]:
    pred_path = seed_dir / "lag_eval_predictions.csv"
    if not pred_path.exists():
        return []
    frame = pd.read_csv(pred_path)
    summary = _segment_summary(frame)
    segments = summary["segment_id"].astype(int).tolist()
    positive_segments = summary.loc[summary["gt_max"] > 0, "segment_id"].astype(int).tolist()

    out_dir = seed_dir / "figs" / "segment_blocks"
    written = []
    for seg in positive_segments:
        idx = segments.index(seg)
        block = []
        if idx > 0:
            block.append(segments[idx - 1])
        block.append(seg)
        if idx + 1 < len(segments):
            block.append(segments[idx + 1])
        label = _block_label(frame, seg)
        out_path = out_dir / f"{label}.png"
        _save_block_plot(out_path, frame, block, seg)
        written.append(out_path)

    if delete_full:
        full_plot = seed_dir / "figs" / "gt_pred_lag.png"
        if full_plot.exists():
            full_plot.unlink()
    return written


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    all_written = []
    for seed_dir in sorted(root.glob("seed_*")):
        if seed_dir.is_dir():
            all_written.extend(plot_seed(seed_dir, delete_full=args.delete_full))
    for path in all_written:
        print(path)
    print(f"Wrote {len(all_written)} segment block plots.")


if __name__ == "__main__":
    main()
