#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
from typing import List, Tuple

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
        description="Plot local windows around contiguous inject_flag=0 regions in a benchmark joined table."
    )
    parser.add_argument("--joined", type=Path, required=True, help="Path to test_joined_single_model.csv")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for plots and summary files")
    parser.add_argument("--context", type=int, default=24, help="Context rows to show before and after each block")
    parser.add_argument("--title-prefix", default="Baseline", help="Label used in figure titles")
    return parser.parse_args()


def _find_uninjected_blocks(frame: pd.DataFrame) -> List[Tuple[int, int]]:
    flags = frame["inject_flag"].to_numpy(dtype=np.int64)
    blocks: List[Tuple[int, int]] = []
    start = None
    for idx, flag in enumerate(flags):
        if flag == 0 and start is None:
            start = idx
        if flag != 0 and start is not None:
            blocks.append((start, idx - 1))
            start = None
    if start is not None:
        blocks.append((start, len(flags) - 1))
    return blocks


def _summary_rows(frame: pd.DataFrame, blocks: List[Tuple[int, int]]) -> pd.DataFrame:
    rows = []
    for block_id, (start, end) in enumerate(blocks, start=1):
        view = frame.iloc[start : end + 1].copy()
        rows.append(
            {
                "block_id": block_id,
                "start_idx": int(start),
                "end_idx": int(end),
                "start_time": view["TimeStamp"].iloc[0].strftime("%Y-%m-%d %H:%M"),
                "end_time": view["TimeStamp"].iloc[-1].strftime("%Y-%m-%d %H:%M"),
                "n_rows": int(len(view)),
                "segment_id": int(view["segment_id"].mode().iloc[0]) if "segment_id" in view.columns else -1,
                "mean_pred_expected_lag": float(view["pred_expected_lag"].mean()),
                "mean_pred_nonzero_prob": float(view["pred_nonzero_prob"].mean()),
                "share_argmax_lag0": float((view["pred_argmax_lag"] == 0).mean()),
                "p90_pred_expected_lag": float(view["pred_expected_lag"].quantile(0.9)),
            }
        )
    return pd.DataFrame(rows)


def _plot_block(
    frame: pd.DataFrame,
    block: Tuple[int, int],
    output_path: Path,
    context: int,
    title_prefix: str,
) -> None:
    start, end = block
    left = max(0, start - context)
    right = min(len(frame) - 1, end + context)
    view = frame.iloc[left : right + 1].copy().reset_index(drop=True)
    x = np.arange(len(view), dtype=np.int64)
    highlight_start = start - left
    highlight_end = end - left

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    axes[0].plot(x, view["lag_gt"], color="#111827", linewidth=2.2, drawstyle="steps-post", label="true lag")
    axes[0].plot(x, view["pred_expected_lag"], color="#2563eb", linewidth=2.0, label="pred expected lag")
    axes[0].axvspan(highlight_start, highlight_end, color="#fde68a", alpha=0.35, label="inject_flag = 0")
    axes[0].set_ylabel("lag")
    axes[0].set_title(f"{title_prefix}: uninjected-region expected lag")
    axes[0].grid(alpha=0.2)
    axes[0].legend(loc="upper right", frameon=False)

    axes[1].plot(x, view["pred_nonzero_prob"], color="#dc2626", linewidth=2.0, label="pred nonzero prob")
    axes[1].axhline(0.5, color="#9ca3af", linewidth=1.2, linestyle="--", label="0.5 threshold")
    axes[1].axvspan(highlight_start, highlight_end, color="#fde68a", alpha=0.35)
    axes[1].set_ylabel("probability")
    axes[1].set_xlabel("sample index in local window")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].grid(alpha=0.2)
    axes[1].legend(loc="upper right", frameon=False)

    tick_positions = np.linspace(0, len(view) - 1, num=min(8, len(view)), dtype=int)
    tick_labels = [pd.to_datetime(view["TimeStamp"].iloc[pos]).strftime("%m-%d %H:%M") for pos in tick_positions]
    axes[1].set_xticks(tick_positions)
    axes[1].set_xticklabels(tick_labels, rotation=25, ha="right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.joined)
    required = {"TimeStamp", "inject_flag", "lag_gt", "pred_expected_lag", "pred_nonzero_prob", "pred_argmax_lag"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns in {args.joined}: {', '.join(missing)}")

    frame["TimeStamp"] = pd.to_datetime(frame["TimeStamp"])
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    blocks = _find_uninjected_blocks(frame)
    if not blocks:
        raise ValueError("No inject_flag=0 blocks found.")

    summary = _summary_rows(frame, blocks)
    summary.to_csv(output_dir / "uninjected_block_summary.csv", index=False)

    for block_id, block in enumerate(blocks, start=1):
        _plot_block(
            frame=frame,
            block=block,
            output_path=output_dir / f"uninjected_block_{block_id:03d}.png",
            context=args.context,
            title_prefix=args.title_prefix,
        )

    print(f"Wrote summary to {output_dir / 'uninjected_block_summary.csv'}")
    print(f"Wrote {len(blocks)} block plot(s) to {output_dir}")


if __name__ == "__main__":
    main()
