#!/usr/bin/env python3

import argparse
import math
import os
from pathlib import Path
from typing import Dict, List, Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot lag block panels for a single model: true lag vs predicted expected lag."
    )
    parser.add_argument("--joined", type=Path, required=True, help="Path to test_joined_single_model.csv")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for plots and summaries")
    parser.add_argument("--context", type=int, default=16, help="Context samples shown on each side of a lag block")
    parser.add_argument("--title-prefix", default="V5", help="Short label shown in titles and legends")
    return parser.parse_args()


def _find_lag_blocks(joined: pd.DataFrame) -> List[Dict[str, int]]:
    lag_mask = joined["lag_gt"].to_numpy(dtype=np.int64) > 0
    blocks: List[Dict[str, int]] = []
    start = None
    for idx, is_lagged in enumerate(lag_mask):
        if is_lagged and start is None:
            start = idx
        if not is_lagged and start is not None:
            blocks.append({"start": int(start), "end": int(idx - 1)})
            start = None
    if start is not None:
        blocks.append({"start": int(start), "end": int(len(joined) - 1)})
    return blocks


def _block_dmax(joined: pd.DataFrame, block: Dict[str, int]) -> int:
    view = joined.iloc[block["start"] : block["end"] + 1]
    if "segment_dmax_gt" not in view.columns:
        return 0
    return int(view["segment_dmax_gt"].mode().iloc[0])


def _block_summary_rows(joined: pd.DataFrame, blocks: List[Dict[str, int]]) -> pd.DataFrame:
    rows = []
    for idx, block in enumerate(blocks, start=1):
        block_view = joined.iloc[block["start"] : block["end"] + 1].copy()
        rows.append(
            {
                "block_id": idx,
                "start_time": block_view["TimeStamp"].iloc[0].strftime("%Y-%m-%d %H:%M"),
                "end_time": block_view["TimeStamp"].iloc[-1].strftime("%Y-%m-%d %H:%M"),
                "n_samples": int(len(block_view)),
                "true_dmax": _block_dmax(joined, block),
                "true_lag_values": ",".join(str(int(v)) for v in sorted(block_view["lag_gt"].unique().tolist())),
                "true_lag_mean": float(block_view["lag_gt"].mean()),
                "pred_expected_lag_mean": float(block_view["pred_expected_lag"].mean()),
                "block_expected_lag_mae": float((block_view["pred_expected_lag"] - block_view["lag_gt"]).abs().mean()),
            }
        )
    return pd.DataFrame(rows)


def _render_blocks(
    joined: pd.DataFrame,
    blocks: List[Dict[str, int]],
    output_path: Path,
    title: str,
    legend_label: str,
    context: int,
) -> None:
    if not blocks:
        raise ValueError("No lag blocks to render.")

    n_blocks = len(blocks)
    ncols = 2 if n_blocks > 1 else 1
    nrows = int(math.ceil(n_blocks / float(ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.8 * nrows), sharey=True)
    axes_arr = np.atleast_1d(axes).reshape(nrows, ncols)

    y_max = max(7.2, float(joined["lag_gt"].max()) + 0.8, float(joined["pred_expected_lag"].max()) + 0.8)

    for block_idx, block in enumerate(blocks):
        row_idx = block_idx // ncols
        col_idx = block_idx % ncols
        ax = axes_arr[row_idx, col_idx]

        left = max(0, block["start"] - context)
        right = min(len(joined) - 1, block["end"] + context)
        view = joined.iloc[left : right + 1].copy()
        x = np.arange(len(view), dtype=np.int64)

        ax.plot(x, view["lag_gt"], label="true lag", color="#111827", linewidth=2.2, drawstyle="steps-post")
        ax.plot(
            x,
            view["pred_expected_lag"],
            label=legend_label,
            color="#2563eb",
            linewidth=1.9,
        )

        lag_start = block["start"] - left
        lag_end = block["end"] - left
        ax.axvspan(lag_start, lag_end, color="#fde68a", alpha=0.35)
        ax.set_ylim(-0.3, y_max)
        ax.set_xlabel("sample index in local window")
        ax.set_ylabel("lag")

        dmax = _block_dmax(joined, block)
        ax.set_title(
            "Block %d (dmax=%d): %s to %s"
            % (
                block_idx + 1,
                dmax,
                joined.iloc[block["start"]]["TimeStamp"].strftime("%Y-%m-%d %H:%M"),
                joined.iloc[block["end"]]["TimeStamp"].strftime("%Y-%m-%d %H:%M"),
            ),
            fontsize=10,
        )

    for idx in range(n_blocks, nrows * ncols):
        axes_arr[idx // ncols, idx % ncols].axis("off")

    handles, labels = axes_arr[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.99))
    fig.suptitle(title, y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    joined = pd.read_csv(args.joined)
    joined["TimeStamp"] = pd.to_datetime(joined["TimeStamp"])
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    blocks = _find_lag_blocks(joined)
    if not blocks:
        raise ValueError("No lag blocks found in joined file.")

    summary_df = _block_summary_rows(joined, blocks)
    summary_df.to_csv(output_dir / "lag_block_summary_v5.csv", index=False)

    _render_blocks(
        joined=joined,
        blocks=blocks,
        output_path=output_dir / "lag_block_panels_v5.png",
        title="Lag Block Panels: True vs %s Predicted Expected Lag" % args.title_prefix,
        legend_label="%s expected lag" % args.title_prefix.lower(),
        context=args.context,
    )

    if "segment_dmax_gt" in joined.columns:
        for dmax_value in sorted(
            [int(value) for value in joined["segment_dmax_gt"].dropna().unique().tolist() if int(value) > 0]
        ):
            selected = [block for block in blocks if _block_dmax(joined, block) == dmax_value]
            if not selected:
                continue
            _render_blocks(
                joined=joined,
                blocks=selected,
                output_path=output_dir / ("lag_block_panels_v5_dmax%d.png" % dmax_value),
                title="Lag Block Panels (dmax=%d): True vs %s Expected Lag" % (dmax_value, args.title_prefix),
                legend_label="%s expected lag" % args.title_prefix.lower(),
                context=args.context,
            )


if __name__ == "__main__":
    main()
