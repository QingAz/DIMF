#!/usr/bin/env python3

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataprocess import load_and_prepare
from src.utils.seed import set_seed
from train import load_config
from scripts.run_detection_segment_audit import (
    _build_model,
    _collect_split_scores,
    _make_eval_loaders,
    _raw_split_frame,
    _render_worst_panels,
    _segment_one_vs_opposite,
    _select_worst_segments,
    _split_summaries,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select the best detection checkpoint by validation segment metrics and run a full audit."
    )
    parser.add_argument("--config", type=Path, required=True, help="Training config")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for selection outputs")
    parser.add_argument("--checkpoint-dir", type=Path, default=None, help="Directory containing periodic checkpoints")
    parser.add_argument("--edge", default="stage1_to_stage2", help="Lag edge name")
    parser.add_argument("--target-key", default="stage1_to_stage2_in_block_gt", help="Detection target key")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Torch device")
    parser.add_argument("--context-segments", type=int, default=1)
    parser.add_argument("--low-auroc-k", type=int, default=3)
    parser.add_argument("--high-gap-k", type=int, default=2)
    parser.add_argument(
        "--selector-rule",
        choices=["current", "auroc_then_ratio", "late_current"],
        default="current",
        help="Checkpoint selector rule applied to validation detection metrics.",
    )
    parser.add_argument(
        "--late-min-epoch",
        type=int,
        default=30,
        help="Minimum epoch used by the late_current selector.",
    )
    return parser.parse_args()


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(str(path)))


def _checkpoint_dir(cfg: Dict[str, Any], args: argparse.Namespace) -> Path:
    if args.checkpoint_dir is not None:
        return _absolute_path(args.checkpoint_dir)
    ckpt_dir = Path(cfg["logging"].get("checkpoint_dir", ""))
    if not ckpt_dir:
        ckpt_dir = Path(cfg["logging"]["ckpt_path"]).parent / "checkpoints"
    if not ckpt_dir.is_absolute():
        ckpt_dir = ROOT / ckpt_dir
    return _absolute_path(ckpt_dir)


def _load_prepared(cfg: Dict[str, Any]):
    return load_and_prepare(
        csv_path=cfg["data"]["csv_path"],
        time_col=cfg["data"]["time_col"],
        target_col=cfg["data"]["target_col"],
        feed_prefix=cfg["data"]["feed_prefix"],
        stage1_prefix=cfg["data"]["stage1_prefix"],
        stage2_prefix=cfg["data"]["stage2_prefix"],
        stage3_prefix=cfg["data"]["stage3_prefix"],
        fillna=cfg["data"].get("fillna", "ffill"),
        use_delta_t=bool(cfg["data"].get("use_delta_t", True)),
        train_ratio=float(cfg["data"]["train_ratio"]),
        val_ratio=float(cfg["data"]["val_ratio"]),
        test_ratio=float(cfg["data"]["test_ratio"]),
        split_mode=str(cfg["data"].get("split_mode", "rows")),
        history_steps=int(cfg["data"]["L"]),
        horizon_steps=int(cfg["data"]["H"]),
        collection_interval_min=int(cfg["data"].get("collection_interval_min", 15)),
        gap_break_min=int(cfg["data"].get("gap_break_min", 120)),
        gap_fill_min=int(cfg["data"].get("gap_fill_min", 60)),
        use_missing_mask=bool(cfg["data"].get("use_missing_mask", True)),
        include_target_history=bool(cfg["data"].get("include_target_history", False)),
        split_col=str(cfg["data"].get("split_col", "split")),
        sample_keep_col=(
            str(cfg["data"]["sample_keep_col"])
            if cfg["data"].get("sample_keep_col") is not None
            else None
        ),
        respect_existing_segment_id=bool(cfg["data"].get("respect_existing_segment_id", False)),
    )


def _evaluate_one_checkpoint(
    ckpt_path: Path,
    model,
    loaders,
    device: torch.device,
    cfg: Dict[str, Any],
    edge: str,
) -> Dict[str, Any]:
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model"])

    val_lookup = _raw_split_frame(cfg, "val")
    val_scores = _collect_split_scores(
        model=model,
        loader=loaders["val"]["loader"],
        device=device,
        sample_timestamps=loaders["val"]["sample_timestamps"],
        raw_lookup=val_lookup,
        split_name="val",
        edge=edge,
    )
    val_segment = _segment_one_vs_opposite(val_scores)
    val_split = _split_summaries(val_scores, val_segment).iloc[0].to_dict()
    segment_direction = (
        val_segment["p_in_block_mean"].to_numpy(dtype=float)
        > val_segment["p_out_block_mean"].to_numpy(dtype=float)
    )
    positive_view = val_segment.loc[val_segment["segment_label"] == 1].copy()
    negative_view = val_segment.loc[val_segment["segment_label"] == 0].copy()
    mean_segment_auroc = float(val_segment["auroc"].mean()) if not val_segment.empty else 0.5
    mean_segment_auprc = float(val_segment["auprc"].mean()) if not val_segment.empty else 0.0
    positive_segment_ratio = float(segment_direction.mean()) if len(segment_direction) else 0.0
    positive_only_ratio = (
        float((positive_view["p_in_block_mean"] > positive_view["p_out_block_mean"]).mean())
        if not positive_view.empty
        else 0.0
    )
    selector_score = mean_segment_auroc + 0.5 * positive_segment_ratio + 0.2 * mean_segment_auprc
    return {
        "epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_path": ckpt_path.as_posix(),
        "mean_segment_auroc": mean_segment_auroc,
        "mean_segment_auprc": mean_segment_auprc,
        "positive_segment_ratio": positive_segment_ratio,
        "positive_only_segment_ratio": positive_only_ratio,
        "mean_positive_segment_auroc": float(positive_view["auroc"].mean()) if not positive_view.empty else 0.5,
        "mean_negative_segment_auroc": float(negative_view["auroc"].mean()) if not negative_view.empty else 0.5,
        "selector_score": selector_score,
        **val_split,
    }


def _select_current(selection: pd.DataFrame) -> pd.Series:
    ranked = selection.sort_values(
        [
            "selector_score",
            "mean_segment_auroc",
            "positive_segment_ratio",
            "mean_segment_auprc",
            "segment_block_auroc",
            "row_block_auroc",
            "epoch",
        ],
        ascending=[False, False, False, False, False, False, False],
    ).reset_index(drop=True)
    return ranked.iloc[0]


def _select_auroc_then_ratio(selection: pd.DataFrame) -> pd.Series:
    ranked = selection.sort_values(
        [
            "mean_segment_auroc",
            "positive_segment_ratio",
            "mean_segment_auprc",
            "selector_score",
            "segment_block_auroc",
            "row_block_auroc",
            "epoch",
        ],
        ascending=[False, False, False, False, False, False, False],
    ).reset_index(drop=True)
    return ranked.iloc[0]


def _select_checkpoint(selection: pd.DataFrame, selector_rule: str, late_min_epoch: int) -> pd.Series:
    if selector_rule == "auroc_then_ratio":
        return _select_auroc_then_ratio(selection)
    if selector_rule == "late_current":
        late = selection.loc[selection["epoch"].astype(int) >= int(late_min_epoch)].copy()
        if late.empty:
            late = selection.copy()
        return _select_current(late)
    return _select_current(selection)


def _full_audit(
    model,
    checkpoint_path: Path,
    loaders,
    device: torch.device,
    cfg: Dict[str, Any],
    edge: str,
    output_dir: Path,
    context_segments: int,
    low_auroc_k: int,
    high_gap_k: int,
) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])

    sample_frames: List[pd.DataFrame] = []
    for split_name in ["train", "val", "test"]:
        raw_lookup = _raw_split_frame(cfg, split_name)
        split_frame = _collect_split_scores(
            model=model,
            loader=loaders[split_name]["loader"],
            device=device,
            sample_timestamps=loaders[split_name]["sample_timestamps"],
            raw_lookup=raw_lookup,
            split_name=split_name,
            edge=edge,
        )
        sample_frames.append(split_frame)
    samples = pd.concat(sample_frames, ignore_index=True)
    samples.to_csv(output_dir / "sample_detection_scores.csv", index=False)

    segment_parts = []
    for _, split_frame in samples.groupby("split", sort=False):
        segment_parts.append(_segment_one_vs_opposite(split_frame))
    segment_audit = pd.concat(segment_parts, ignore_index=True)
    segment_audit.to_csv(output_dir / "segment_detection_audit.csv", index=False)

    split_summary = _split_summaries(samples, segment_audit)
    split_summary.to_csv(output_dir / "split_detection_summary.csv", index=False)

    worst_segments = _select_worst_segments(segment_audit, low_auroc_k=low_auroc_k, high_gap_k=high_gap_k)
    worst_segments.to_csv(output_dir / "worst_test_segments.csv", index=False)
    _render_worst_panels(samples, worst_segments, output_dir, context_segments=context_segments)

    return {
        "split_summary": split_summary.to_dict(orient="records"),
        "worst_segments": worst_segments.to_dict(orient="records"),
    }


def main() -> None:
    args = parse_args()
    config_path = _absolute_path(args.config)
    output_dir = _absolute_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(str(config_path))
    set_seed(int(cfg.get("seed", 42)))
    device = torch.device(args.device)

    prepared, _ = _load_prepared(cfg)
    loaders = _make_eval_loaders(cfg, prepared)
    model = _build_model(cfg, prepared, device)

    ckpt_dir = _checkpoint_dir(cfg, args)
    checkpoints = sorted(ckpt_dir.glob("epoch_*.ckpt"))
    if not checkpoints:
        raise ValueError(f"No periodic checkpoints found in {ckpt_dir}")

    selection_rows = []
    for ckpt_path in checkpoints:
        selection_rows.append(
            _evaluate_one_checkpoint(
                ckpt_path=ckpt_path,
                model=model,
                loaders=loaders,
                device=device,
                cfg=cfg,
                edge=args.edge,
            )
        )
    selection = pd.DataFrame(selection_rows).sort_values("epoch").reset_index(drop=True)
    selection.to_csv(output_dir / "checkpoint_detection_selection.csv", index=False)

    selected = _select_checkpoint(
        selection,
        selector_rule=str(args.selector_rule),
        late_min_epoch=int(args.late_min_epoch),
    )
    selected_path = _absolute_path(Path(str(selected["checkpoint_path"])))
    audit = _full_audit(
        model=model,
        checkpoint_path=selected_path,
        loaders=loaders,
        device=device,
        cfg=cfg,
        edge=args.edge,
        output_dir=output_dir,
        context_segments=int(args.context_segments),
        low_auroc_k=int(args.low_auroc_k),
        high_gap_k=int(args.high_gap_k),
    )

    report = {
        "config": config_path.as_posix(),
        "checkpoint_dir": ckpt_dir.as_posix(),
        "selection_metric_primary": "selector_score",
        "selection_metric_secondary": "mean_segment_auroc",
        "selector_formula": "mean_segment_auroc + 0.5 * positive_segment_ratio + 0.2 * mean_segment_auprc",
        "selector_rule": str(args.selector_rule),
        "late_min_epoch": int(args.late_min_epoch),
        "selected": selected.to_dict(),
        "audit": audit,
    }
    (output_dir / "checkpoint_detection_selection.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(selection.to_csv(index=False))
    print(json.dumps({"selected_checkpoint_path": selected_path.as_posix()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
