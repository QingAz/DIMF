#!/usr/bin/env python3

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_detection_segment_audit import (
    _build_model,
    _collect_split_scores,
    _make_eval_loaders,
    _raw_split_frame,
    _segment_one_vs_opposite,
    _split_summaries,
)
from scripts.select_and_audit_detection_checkpoints import _load_prepared
from src.utils.seed import set_seed
from train import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run D1-D4 follow-up diagnostics for local-detection multiseed experiments."
    )
    parser.add_argument("--run-dirs", nargs="+", type=Path, required=True, help="Seed output directories")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for diagnostics")
    parser.add_argument("--device", default="cpu", help="Torch device for checkpoint trajectory evaluation")
    parser.add_argument("--success-panel-seeds", type=int, default=2, help="How many success seeds to visualize")
    parser.add_argument("--failure-panel-seeds", type=int, default=2, help="How many failure seeds to visualize")
    parser.add_argument("--panels-per-seed", type=int, default=3, help="How many segments to plot per chosen seed")
    parser.add_argument(
        "--max-checkpoints-per-seed",
        type=int,
        default=0,
        help="Optional cap for smoke tests; 0 means evaluate every checkpoint",
    )
    return parser.parse_args()


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(str(path)))


def _seed_from_config(config_path: Path) -> int:
    for line in config_path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("seed:"):
            return int(line.split(":", 1)[1].strip())
    raise ValueError(f"Could not parse seed from {config_path}")


def _load_run_summary(run_dir: Path) -> Dict[str, Any]:
    selection_json = run_dir / "detection_selected_audit" / "checkpoint_detection_selection.json"
    split_summary_csv = run_dir / "detection_selected_audit" / "split_detection_summary.csv"
    selection = json.loads(selection_json.read_text(encoding="utf-8"))
    split_summary = pd.read_csv(split_summary_csv)
    test_row = split_summary.loc[split_summary["split"] == "test"].iloc[0]
    config_path = _absolute_path(Path(selection["config"]))
    seed = _seed_from_config(config_path)
    diff = float(test_row["row_p_in_block"]) - float(test_row["row_p_out_block"])
    segment_auroc = float(test_row["segment_block_auroc"])
    success = bool(segment_auroc > 0.5 and diff > 0.0)
    return {
        "seed": seed,
        "run_dir": run_dir.as_posix(),
        "config_path": config_path.as_posix(),
        "selected_epoch": int(selection["selected"]["epoch"]),
        "row_auroc": float(test_row["row_block_auroc"]),
        "segment_auroc": segment_auroc,
        "block_auprc": float(test_row["row_block_auprc"]),
        "p_in_block": float(test_row["row_p_in_block"]),
        "p_out_block": float(test_row["row_p_out_block"]),
        "diff": diff,
        "success": success,
    }


def _load_test_segment_rows(run_dir: Path, seed_summary: Dict[str, Any]) -> pd.DataFrame:
    path = run_dir / "detection_selected_audit" / "segment_detection_audit.csv"
    df = pd.read_csv(path)
    df = df.loc[df["split"] == "test"].copy()
    df.insert(0, "seed", int(seed_summary["seed"]))
    df.insert(1, "success_seed", bool(seed_summary["success"]))
    df["diff"] = df["p_in_block_mean"].astype(float) - df["p_out_block_mean"].astype(float)
    df["segment_success"] = (df["diff"] > 0.0) & (df["auroc"].astype(float) > 0.5)
    return df


def _seed_segment_summary(segment_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for seed, part in segment_df.groupby("seed", sort=True):
        rows.append(
            {
                "seed": int(seed),
                "success_seed": bool(part["success_seed"].iloc[0]),
                "n_test_segments": int(len(part)),
                "n_segment_diff_positive": int((part["diff"] > 0.0).sum()),
                "n_segment_auroc_gt_0_5": int((part["auroc"] > 0.5).sum()),
                "n_segment_success": int(part["segment_success"].sum()),
                "positive_diff_ratio": float((part["diff"] > 0.0).mean()),
                "segment_success_ratio": float(part["segment_success"].mean()),
                "mean_segment_auroc": float(part["auroc"].mean()),
                "mean_diff": float(part["diff"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)


def _dmax_breakdowns(segment_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    grouped = (
        segment_df.groupby(["success_seed", "dmax"], sort=True)
        .agg(
            n_segments=("segment_id", "size"),
            mean_segment_auroc=("auroc", "mean"),
            mean_diff=("diff", "mean"),
            success_segment_ratio=("segment_success", "mean"),
        )
        .reset_index()
    )
    per_seed = (
        segment_df.groupby(["seed", "success_seed", "dmax"], sort=True)
        .agg(
            n_segments=("segment_id", "size"),
            mean_segment_auroc=("auroc", "mean"),
            mean_diff=("diff", "mean"),
            success_segment_ratio=("segment_success", "mean"),
        )
        .reset_index()
    )
    return grouped, per_seed


def _selected_panel_seeds(seed_summary_df: pd.DataFrame, n_success: int, n_failure: int) -> Tuple[List[int], List[int]]:
    success_view = seed_summary_df.loc[seed_summary_df["success"]].sort_values(
        ["diff", "segment_auroc", "row_auroc"],
        ascending=[False, False, False],
    )
    failure_view = seed_summary_df.loc[~seed_summary_df["success"]].sort_values(
        ["diff", "segment_auroc", "row_auroc"],
        ascending=[True, True, True],
    )
    return (
        success_view["seed"].astype(int).head(max(n_success, 0)).tolist(),
        failure_view["seed"].astype(int).head(max(n_failure, 0)).tolist(),
    )


def _selected_segment_ids(part: pd.DataFrame, is_success: bool, k: int) -> List[int]:
    if is_success:
        ranked = part.sort_values(["diff", "auroc", "best_f1"], ascending=[False, False, False])
    else:
        ranked = part.sort_values(["diff", "auroc", "best_f1"], ascending=[True, True, True])
    return ranked["segment_id"].astype(int).head(max(k, 0)).tolist()


def _contiguous_true_spans(mask: np.ndarray) -> Iterable[Tuple[int, int]]:
    start = None
    for idx, flag in enumerate(mask.tolist()):
        if flag and start is None:
            start = idx
        if not flag and start is not None:
            yield (start, idx - 1)
            start = None
    if start is not None:
        yield (start, len(mask) - 1)


def _render_seed_panels(
    seed: int,
    is_success: bool,
    run_dir: Path,
    segment_df: pd.DataFrame,
    panels_per_seed: int,
    output_dir: Path,
) -> pd.DataFrame:
    samples_path = run_dir / "detection_selected_audit" / "sample_detection_scores.csv"
    samples = pd.read_csv(samples_path)
    samples = samples.loc[samples["split"] == "test"].copy()

    seed_segments = segment_df.loc[segment_df["seed"] == seed].copy()
    chosen_ids = _selected_segment_ids(seed_segments, is_success=is_success, k=panels_per_seed)
    if not chosen_ids:
        return pd.DataFrame(columns=["seed", "segment_id", "panel_rank", "panel_group"])

    fig, axes = plt.subplots(len(chosen_ids), 1, figsize=(12, max(3.0 * len(chosen_ids), 4.0)), sharex=False)
    axes_arr = np.atleast_1d(axes)
    manifest_rows: List[Dict[str, Any]] = []
    panel_group = "success" if is_success else "failure"

    for panel_rank, (ax, segment_id) in enumerate(zip(axes_arr, chosen_ids), start=1):
        seg = samples.loc[samples["segment_id"] == int(segment_id)].copy()
        seg = seg.sort_values("segment_index").reset_index(drop=True)
        x = seg["segment_index"].to_numpy(dtype=np.int64)
        p = seg["p"].to_numpy(dtype=np.float64)
        in_block = seg["in_block"].to_numpy(dtype=np.int64) > 0

        for left, right in _contiguous_true_spans(in_block):
            ax.axvspan(left - 0.5, right + 0.5, color="tab:orange", alpha=0.18, lw=0)
        ax.plot(x, p, color="tab:blue", linewidth=2.0)
        ax.scatter(x[~in_block], p[~in_block], color="tab:blue", s=22, alpha=0.8)
        if in_block.any():
            ax.scatter(x[in_block], p[in_block], color="tab:orange", s=28)

        meta = seed_segments.loc[seed_segments["segment_id"] == int(segment_id)].iloc[0]
        title = (
            f"seed={seed} seg={int(segment_id)} dmax={int(meta['dmax'])} "
            f"diff={float(meta['diff']):+.4f} auroc={float(meta['auroc']):.3f}"
        )
        ax.set_title(title)
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel("p_t")
        ax.set_xlabel("segment index")
        ax.grid(alpha=0.25, linewidth=0.5)

        manifest_rows.append(
            {
                "seed": int(seed),
                "segment_id": int(segment_id),
                "panel_rank": int(panel_rank),
                "panel_group": panel_group,
                "dmax": int(meta["dmax"]),
                "diff": float(meta["diff"]),
                "segment_auroc": float(meta["auroc"]),
                "segment_auprc": float(meta["auprc"]),
                "best_f1": float(meta["best_f1"]),
            }
        )

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"seed_{seed}_{panel_group}_segments.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(manifest_rows)


def _build_eval_state(run_dir: Path, device: torch.device):
    selection_path = run_dir / "detection_selected_audit" / "checkpoint_detection_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    config_path = _absolute_path(Path(selection["config"]))
    cfg = load_config(str(config_path))
    set_seed(int(cfg.get("seed", 42)))
    prepared, _ = _load_prepared(cfg)
    loaders = _make_eval_loaders(cfg, prepared)
    model = _build_model(cfg, prepared, device)
    raw_lookup_test = _raw_split_frame(cfg, "test")
    return cfg, model, loaders, raw_lookup_test, int(selection["selected"]["epoch"])


@torch.no_grad()
def _evaluate_test_checkpoint(
    model,
    loader,
    sample_timestamps: np.ndarray,
    raw_lookup: pd.DataFrame,
    device: torch.device,
    checkpoint_path: Path,
) -> Dict[str, float]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    test_scores = _collect_split_scores(
        model=model,
        loader=loader,
        device=device,
        sample_timestamps=sample_timestamps,
        raw_lookup=raw_lookup,
        split_name="test",
        edge="stage1_to_stage2",
    )
    test_segment = _segment_one_vs_opposite(test_scores)
    test_split = _split_summaries(test_scores, test_segment).iloc[0]
    return {
        "test_row_auroc": float(test_split["row_block_auroc"]),
        "test_segment_auroc": float(test_split["segment_block_auroc"]),
        "test_diff": float(test_split["row_p_in_block"]) - float(test_split["row_p_out_block"]),
    }


def _checkpoint_trajectory_for_run(
    run_dir: Path,
    device: torch.device,
    max_checkpoints_per_seed: int,
) -> pd.DataFrame:
    seed_summary = _load_run_summary(run_dir)
    selection_csv = run_dir / "detection_selected_audit" / "checkpoint_detection_selection.csv"
    selection_df = pd.read_csv(selection_csv).sort_values("epoch").reset_index(drop=True)
    if max_checkpoints_per_seed > 0:
        selection_df = selection_df.head(int(max_checkpoints_per_seed)).reset_index(drop=True)

    cfg, model, loaders, raw_lookup_test, selected_epoch = _build_eval_state(run_dir, device)
    test_loader = loaders["test"]["loader"]
    sample_timestamps = loaders["test"]["sample_timestamps"]

    rows: List[Dict[str, Any]] = []
    for row in selection_df.itertuples():
        metrics = _evaluate_test_checkpoint(
            model=model,
            loader=test_loader,
            sample_timestamps=sample_timestamps,
            raw_lookup=raw_lookup_test,
            device=device,
            checkpoint_path=_absolute_path(Path(str(row.checkpoint_path))),
        )
        rows.append(
            {
                "seed": int(seed_summary["seed"]),
                "epoch": int(row.epoch),
                "checkpoint_path": str(row.checkpoint_path),
                "selected_epoch": int(selected_epoch),
                "is_selected": bool(int(row.epoch) == int(selected_epoch)),
                "val_mean_segment_auroc": float(row.mean_segment_auroc),
                "val_mean_segment_auprc": float(row.mean_segment_auprc),
                "val_positive_segment_ratio": float(row.positive_segment_ratio),
                "val_selector_score": float(row.selector_score),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def _selection_track_summary(trajectory_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for seed, part in trajectory_df.groupby("seed", sort=True):
        selected_view = part.loc[part["is_selected"]]
        if selected_view.empty:
            selected = part.sort_values(["val_selector_score", "epoch"], ascending=[False, False]).iloc[0]
            selected_available = False
        else:
            selected = selected_view.iloc[0]
            selected_available = True
        best_test_seg = part.sort_values(["test_segment_auroc", "test_diff", "test_row_auroc"], ascending=[False, False, False]).iloc[0]
        best_test_diff = part.sort_values(["test_diff", "test_segment_auroc"], ascending=[False, False]).iloc[0]
        rows.append(
            {
                "seed": int(seed),
                "selected_epoch_available": bool(selected_available),
                "selected_epoch": int(selected["epoch"]),
                "selected_val_mean_segment_auroc": float(selected["val_mean_segment_auroc"]),
                "selected_val_positive_segment_ratio": float(selected["val_positive_segment_ratio"]),
                "selected_test_row_auroc": float(selected["test_row_auroc"]),
                "selected_test_segment_auroc": float(selected["test_segment_auroc"]),
                "selected_test_diff": float(selected["test_diff"]),
                "best_test_segment_epoch": int(best_test_seg["epoch"]),
                "best_test_segment_auroc": float(best_test_seg["test_segment_auroc"]),
                "best_test_diff_epoch": int(best_test_diff["epoch"]),
                "best_test_diff": float(best_test_diff["test_diff"]),
            }
        )
    return pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)


def _plot_checkpoint_trajectory(part: pd.DataFrame, output_path: Path) -> None:
    part = part.sort_values("epoch").reset_index(drop=True)
    epochs = part["epoch"].to_numpy(dtype=np.int64)
    selected_view = part.loc[part["is_selected"], "epoch"]
    selected_epoch = int(selected_view.iloc[0]) if not selected_view.empty else None

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(epochs, part["val_mean_segment_auroc"], label="val mean segment AUROC", linewidth=2.0)
    axes[0].plot(epochs, part["val_positive_segment_ratio"], label="val positive-segment ratio", linewidth=2.0)
    axes[0].set_ylabel("val")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.25)

    axes[1].plot(epochs, part["test_row_auroc"], label="test row AUROC", linewidth=2.0)
    axes[1].plot(epochs, part["test_segment_auroc"], label="test segment AUROC", linewidth=2.0)
    axes[1].set_ylabel("test AUROC")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].legend(loc="best")
    axes[1].grid(alpha=0.25)

    axes[2].plot(epochs, part["test_diff"], label="test p_in - p_out", color="tab:green", linewidth=2.0)
    axes[2].axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.5)
    axes[2].set_ylabel("test diff")
    axes[2].set_xlabel("epoch")
    axes[2].legend(loc="best")
    axes[2].grid(alpha=0.25)

    if selected_epoch is not None:
        for ax in axes:
            ax.axvline(selected_epoch, color="tab:red", linewidth=1.3, linestyle="--", alpha=0.8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = _absolute_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = [_absolute_path(run_dir) for run_dir in args.run_dirs]
    seed_summary_df = pd.DataFrame([_load_run_summary(run_dir) for run_dir in run_dirs]).sort_values("seed").reset_index(drop=True)
    seed_summary_df.to_csv(output_dir / "seed_outcome_table.csv", index=False)

    d1_parts = [_load_test_segment_rows(run_dir, summary) for run_dir, summary in zip(run_dirs, seed_summary_df.to_dict(orient="records"))]
    d1_df = pd.concat(d1_parts, ignore_index=True).sort_values(["seed", "segment_id"]).reset_index(drop=True)
    d1_df.to_csv(output_dir / "test_segment_audit_5seed.csv", index=False)

    seed_segment_df = _seed_segment_summary(d1_df)
    seed_segment_df.to_csv(output_dir / "seed_test_segment_summary.csv", index=False)

    d2_grouped, d2_per_seed = _dmax_breakdowns(d1_df)
    d2_grouped.to_csv(output_dir / "dmax_success_vs_failure_summary.csv", index=False)
    d2_per_seed.to_csv(output_dir / "dmax_per_seed_summary.csv", index=False)

    success_panel_seeds, failure_panel_seeds = _selected_panel_seeds(
        seed_summary_df=seed_summary_df,
        n_success=int(args.success_panel_seeds),
        n_failure=int(args.failure_panel_seeds),
    )
    panel_manifest_parts = []
    for seed in success_panel_seeds:
        run_dir = next(run_dir for run_dir in run_dirs if _load_run_summary(run_dir)["seed"] == seed)
        panel_manifest_parts.append(
            _render_seed_panels(
                seed=seed,
                is_success=True,
                run_dir=run_dir,
                segment_df=d1_df,
                panels_per_seed=int(args.panels_per_seed),
                output_dir=output_dir / "seed_segment_panels",
            )
        )
    for seed in failure_panel_seeds:
        run_dir = next(run_dir for run_dir in run_dirs if _load_run_summary(run_dir)["seed"] == seed)
        panel_manifest_parts.append(
            _render_seed_panels(
                seed=seed,
                is_success=False,
                run_dir=run_dir,
                segment_df=d1_df,
                panels_per_seed=int(args.panels_per_seed),
                output_dir=output_dir / "seed_segment_panels",
            )
        )
    panel_manifest = (
        pd.concat(panel_manifest_parts, ignore_index=True)
        if panel_manifest_parts
        else pd.DataFrame(columns=["seed", "segment_id", "panel_rank", "panel_group"])
    )
    panel_manifest.to_csv(output_dir / "panel_manifest.csv", index=False)

    device = torch.device(args.device)
    trajectory_parts = []
    for run_dir in run_dirs:
        trajectory_parts.append(
            _checkpoint_trajectory_for_run(
                run_dir=run_dir,
                device=device,
                max_checkpoints_per_seed=int(args.max_checkpoints_per_seed),
            )
        )
    trajectory_df = pd.concat(trajectory_parts, ignore_index=True).sort_values(["seed", "epoch"]).reset_index(drop=True)
    trajectory_df.to_csv(output_dir / "all_checkpoint_trajectories.csv", index=False)

    selection_track_df = _selection_track_summary(trajectory_df)
    selection_track_df.to_csv(output_dir / "checkpoint_selection_track_summary.csv", index=False)

    for seed, part in trajectory_df.groupby("seed", sort=True):
        _plot_checkpoint_trajectory(part, output_dir / "checkpoint_trajectory_plots" / f"seed_{int(seed)}_trajectory.png")

    report = {
        "run_dirs": [run_dir.as_posix() for run_dir in run_dirs],
        "success_panel_seeds": success_panel_seeds,
        "failure_panel_seeds": failure_panel_seeds,
        "n_total_test_segments": int(d1_df.shape[0]),
        "n_success_seeds": int(seed_summary_df["success"].sum()),
    }
    (output_dir / "followup_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(seed_summary_df.to_csv(index=False))
    print(seed_segment_df.to_csv(index=False))
    print(selection_track_df.to_csv(index=False))


if __name__ == "__main__":
    main()
