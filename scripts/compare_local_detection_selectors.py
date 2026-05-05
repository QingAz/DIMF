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

from scripts.select_and_audit_detection_checkpoints import (
    _absolute_path,
    _checkpoint_dir,
    _full_audit,
    _load_prepared,
)
from scripts.run_detection_segment_audit import _build_model, _make_eval_loaders
from src.utils.seed import set_seed
from train import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare multiple selector rules on a completed local-detection training run."
    )
    parser.add_argument("--config", type=Path, required=True, help="Training config for the run")
    parser.add_argument("--run-dir", type=Path, required=True, help="Completed run output directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for selector-ablation outputs")
    parser.add_argument("--checkpoint-dir", type=Path, default=None, help="Optional checkpoint dir override")
    parser.add_argument("--device", default="cpu", help="Torch device")
    parser.add_argument("--late-min-epoch", type=int, default=30, help="Min epoch for the late selector")
    return parser.parse_args()


def _load_selection_table(run_dir: Path) -> pd.DataFrame:
    selection_csv = run_dir / "detection_selected_audit" / "checkpoint_detection_selection.csv"
    if not selection_csv.exists():
        raise FileNotFoundError(f"Missing selection CSV: {selection_csv}")
    return pd.read_csv(selection_csv).sort_values("epoch").reset_index(drop=True)


def _pick_current(selection: pd.DataFrame) -> pd.Series:
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


def _pick_auroc_ratio(selection: pd.DataFrame) -> pd.Series:
    ranked = selection.sort_values(
        [
            "mean_segment_auroc",
            "positive_segment_ratio",
            "mean_segment_auprc",
            "selector_score",
            "row_block_auroc",
            "epoch",
        ],
        ascending=[False, False, False, False, False, False],
    ).reset_index(drop=True)
    return ranked.iloc[0]


def _pick_late_current(selection: pd.DataFrame, min_epoch: int) -> pd.Series:
    late = selection.loc[selection["epoch"].astype(int) >= int(min_epoch)].copy()
    if late.empty:
        late = selection.copy()
    return _pick_current(late)


def _rule_catalog(selection: pd.DataFrame, min_epoch: int) -> List[Dict[str, Any]]:
    return [
        {
            "selector_name": "current_selector",
            "description": "current score = mean_segment_auroc + 0.5*positive_segment_ratio + 0.2*mean_segment_auprc",
            "selected": _pick_current(selection),
        },
        {
            "selector_name": "auroc_then_ratio",
            "description": "mean_segment_auroc first, positive_segment_ratio tie-break, then mean_segment_auprc",
            "selected": _pick_auroc_ratio(selection),
        },
        {
            "selector_name": f"late{int(min_epoch)}_current",
            "description": f"current selector, but only among checkpoints with epoch >= {int(min_epoch)}",
            "selected": _pick_late_current(selection, min_epoch=int(min_epoch)),
        },
    ]


def _seed_from_config(config_path: Path) -> int:
    for line in config_path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("seed:"):
            return int(line.split(":", 1)[1].strip())
    raise ValueError(f"Could not parse seed from {config_path}")


def main() -> None:
    args = parse_args()
    config_path = _absolute_path(args.config)
    run_dir = _absolute_path(args.run_dir)
    output_dir = _absolute_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(str(config_path))
    set_seed(int(cfg.get("seed", 42)))
    device = torch.device(args.device)

    prepared, _ = _load_prepared(cfg)
    loaders = _make_eval_loaders(cfg, prepared)
    model = _build_model(cfg, prepared, device)

    ckpt_dir = _checkpoint_dir(cfg, args)
    selection = _load_selection_table(run_dir)

    rules = _rule_catalog(selection, min_epoch=int(args.late_min_epoch))
    rows: List[Dict[str, Any]] = []
    seed = _seed_from_config(config_path)

    for rule in rules:
        selected = rule["selected"]
        selector_name = str(rule["selector_name"])
        rule_dir = output_dir / selector_name
        rule_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = _absolute_path(Path(str(selected["checkpoint_path"])))
        audit = _full_audit(
            model=model,
            checkpoint_path=checkpoint_path,
            loaders=loaders,
            device=device,
            cfg=cfg,
            edge="stage1_to_stage2",
            output_dir=rule_dir,
            context_segments=1,
            low_auroc_k=3,
            high_gap_k=2,
        )
        split_summary = pd.DataFrame(audit["split_summary"])
        test_row = split_summary.loc[split_summary["split"] == "test"].iloc[0]
        diff = float(test_row["row_p_in_block"]) - float(test_row["row_p_out_block"])
        success = bool(float(test_row["segment_block_auroc"]) > 0.5 and diff > 0.0)

        rows.append(
            {
                "seed": int(seed),
                "selector_name": selector_name,
                "description": str(rule["description"]),
                "selected_epoch": int(selected["epoch"]),
                "selected_checkpoint_path": checkpoint_path.as_posix(),
                "val_selector_score": float(selected["selector_score"]),
                "val_mean_segment_auroc": float(selected["mean_segment_auroc"]),
                "val_positive_segment_ratio": float(selected["positive_segment_ratio"]),
                "val_mean_segment_auprc": float(selected["mean_segment_auprc"]),
                "test_row_auroc": float(test_row["row_block_auroc"]),
                "test_segment_auroc": float(test_row["segment_block_auroc"]),
                "test_block_auprc": float(test_row["row_block_auprc"]),
                "test_p_in_block": float(test_row["row_p_in_block"]),
                "test_p_out_block": float(test_row["row_p_out_block"]),
                "test_diff": diff,
                "success": success,
            }
        )

    summary_df = pd.DataFrame(rows).sort_values(["seed", "selector_name"]).reset_index(drop=True)
    summary_df.to_csv(output_dir / "selector_ablation_summary.csv", index=False)

    report = {
        "seed": int(seed),
        "config": config_path.as_posix(),
        "run_dir": run_dir.as_posix(),
        "checkpoint_dir": ckpt_dir.as_posix(),
        "late_min_epoch": int(args.late_min_epoch),
        "selectors": summary_df.to_dict(orient="records"),
    }
    (output_dir / "selector_ablation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(summary_df.to_csv(index=False))


if __name__ == "__main__":
    main()
