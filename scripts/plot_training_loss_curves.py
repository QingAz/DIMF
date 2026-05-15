#!/usr/bin/env python3

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LINE_RE = re.compile(
    r"Epoch\s+(?P<epoch>\d+):.*?ent=(?P<ent>[-+0-9.eE]+),\s*loss=(?P<loss>[-+0-9.eE]+),\s*pred=(?P<pred>[-+0-9.eE]+),\s*tv=(?P<tv>[-+0-9.eE]+)"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot training loss curves and reconstruct per-batch component trends from SLURM stderr."
    )
    parser.add_argument(
        "--aligned-log",
        type=Path,
        default=Path("outputs/rawgap_stage2lag_aligned/train_log.jsonl"),
        help="Aligned run train_log.jsonl",
    )
    parser.add_argument(
        "--noalign-log",
        type=Path,
        default=Path("outputs/rawgap_stage2lag_noalign/train_log.jsonl"),
        help="No-alignment run train_log.jsonl",
    )
    parser.add_argument(
        "--stderr-log",
        type=Path,
        default=Path("logs/slurm-dimf_rawgap_align-4726859.err"),
        help="Combined SLURM stderr that includes tqdm batch postfix lines",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/rawgap_stage2lag_alignment_compare/visuals"),
        help="Directory for training plots",
    )
    return parser.parse_args()


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(str(path)))


def _load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _plot_total_losses(aligned_rows: List[Dict], noalign_rows: List[Dict], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.0))

    for rows, label, color, linestyle, key in [
        (aligned_rows, "aligned train", "#2563eb", "-", "train_loss"),
        (aligned_rows, "aligned val", "#2563eb", "--", "val_loss"),
        (noalign_rows, "noalign train", "#ef4444", "-", "train_loss"),
        (noalign_rows, "noalign val", "#ef4444", "--", "val_loss"),
    ]:
        epochs = [row["epoch"] for row in rows]
        values = [row[key] for row in rows]
        ax.plot(epochs, values, label=label, color=color, linestyle=linestyle, linewidth=2.0)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Total Loss Curves")
    ax.legend(frameon=False, ncol=2)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _segment_runs(stderr_path: Path) -> List[List[Dict[str, float]]]:
    matches = []
    with stderr_path.open("r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.replace("\r", "\n")
            for piece in line.split("\n"):
                match = LINE_RE.search(piece)
                if not match:
                    continue
                matches.append(
                    {
                        "epoch": int(match.group("epoch")),
                        "loss": float(match.group("loss")),
                        "pred": float(match.group("pred")),
                        "ent": float(match.group("ent")),
                        "tv": float(match.group("tv")),
                    }
                )

    if not matches:
        return []

    runs = [[]]
    prev_epoch = None
    for item in matches:
        epoch = item["epoch"]
        if prev_epoch is not None and epoch < prev_epoch:
            runs.append([])
        runs[-1].append(item)
        prev_epoch = epoch
    return runs


def _aggregate_run(items: List[Dict[str, float]]) -> List[Dict[str, float]]:
    by_epoch: Dict[int, Dict[str, List[float]]] = {}
    for item in items:
        slot = by_epoch.setdefault(
            item["epoch"],
            {"loss": [], "pred": [], "ent": [], "tv": []},
        )
        for key in ["loss", "pred", "ent", "tv"]:
            slot[key].append(item[key])

    rows = []
    for epoch in sorted(by_epoch):
        slot = by_epoch[epoch]
        rows.append(
            {
                "epoch": epoch,
                "loss": float(np.mean(slot["loss"])),
                "pred": float(np.mean(slot["pred"])),
                "ent": float(np.mean(slot["ent"])),
                "tv": float(np.mean(slot["tv"])),
                "n_batches_seen": len(slot["loss"]),
            }
        )
    return rows


def _plot_components(run_rows: Dict[str, List[Dict[str, float]]], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6), sharex=False)
    specs = [
        ("pred", "Batch Mean Prediction Loss"),
        ("ent", "Batch Mean Entropy Term"),
        ("tv", "Batch Mean TV Term"),
    ]
    colors = {"aligned": "#2563eb", "noalign": "#ef4444"}

    for ax, (key, title) in zip(axes, specs):
        for label in ["aligned", "noalign"]:
            rows = run_rows.get(label, [])
            if not rows:
                continue
            epochs = [row["epoch"] for row in rows]
            values = [row[key] for row in rows]
            ax.plot(epochs, values, label=label, color=colors[label], linewidth=2.0)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Approximate batch mean")
    axes[-1].legend(frameon=False)
    fig.suptitle("Training Component Curves Reconstructed from tqdm Batch Logs", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_component_csv(run_rows: Dict[str, List[Dict[str, float]]], output_path: Path) -> None:
    lines = ["model,epoch,loss,pred,ent,tv,n_batches_seen"]
    for label in ["aligned", "noalign"]:
        for row in run_rows.get(label, []):
            lines.append(
                "%s,%d,%.10f,%.10f,%.10f,%.10f,%d"
                % (
                    label,
                    int(row["epoch"]),
                    float(row["loss"]),
                    float(row["pred"]),
                    float(row["ent"]),
                    float(row["tv"]),
                    int(row["n_batches_seen"]),
                )
            )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    aligned_log = _absolute_path(args.aligned_log)
    noalign_log = _absolute_path(args.noalign_log)
    stderr_log = _absolute_path(args.stderr_log)
    output_dir = _absolute_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    aligned_rows = _load_jsonl(aligned_log)
    noalign_rows = _load_jsonl(noalign_log)
    total_plot = output_dir / "training_total_loss_curves.png"
    _plot_total_losses(aligned_rows, noalign_rows, total_plot)

    run_segments = _segment_runs(stderr_log)
    run_rows = {}
    if len(run_segments) >= 1:
        run_rows["aligned"] = _aggregate_run(run_segments[0])
    if len(run_segments) >= 2:
        run_rows["noalign"] = _aggregate_run(run_segments[1])

    component_plot = output_dir / "training_component_curves.png"
    component_csv = output_dir / "training_component_curves.csv"
    if run_rows:
        _plot_components(run_rows, component_plot)
        _write_component_csv(run_rows, component_csv)
        print("Wrote: %s" % component_plot)
        print("Wrote: %s" % component_csv)

    print("Wrote: %s" % total_plot)


if __name__ == "__main__":
    main()
