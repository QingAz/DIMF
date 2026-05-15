#!/usr/bin/env python3

import argparse
import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml

import sys

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataprocess import load_and_prepare

TIME_FORMAT = "%Y-%m-%d %H:%M"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit lag labels, detection labels, and positive-only magnitude masks."
    )
    parser.add_argument("--config", type=Path, required=True, help="Training config to audit")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for audit outputs")
    parser.add_argument("--edge-key", default="stage1_to_stage2_lag_gt", help="Extra target key")
    return parser.parse_args()


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(str(path)))


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return data


def _timestamp(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series).dt.strftime(TIME_FORMAT)


def _raw_split_frame(cfg: Dict[str, Any], split_name: str) -> pd.DataFrame:
    data_cfg = cfg["data"]
    raw = pd.read_csv(data_cfg["csv_path"])
    raw[data_cfg["time_col"]] = pd.to_datetime(raw[data_cfg["time_col"]])
    if data_cfg.get("split_col", "split") not in raw.columns:
        raise ValueError("Raw dataset must contain predefined split column for this audit")
    split_col = data_cfg.get("split_col", "split")
    out = raw.loc[raw[split_col] == split_name].sort_values(data_cfg["time_col"]).reset_index(drop=True).copy()
    out["timestamp"] = _timestamp(out[data_cfg["time_col"]])
    out["_raw_row_in_split"] = np.arange(len(out), dtype=np.int64)
    return out


def _prepared_data(cfg: Dict[str, Any]):
    data_cfg = cfg["data"]
    return load_and_prepare(
        csv_path=data_cfg["csv_path"],
        time_col=data_cfg["time_col"],
        target_col=data_cfg["target_col"],
        feed_prefix=data_cfg["feed_prefix"],
        stage1_prefix=data_cfg["stage1_prefix"],
        stage2_prefix=data_cfg["stage2_prefix"],
        stage3_prefix=data_cfg["stage3_prefix"],
        fillna=data_cfg.get("fillna", "ffill"),
        use_delta_t=bool(data_cfg.get("use_delta_t", True)),
        train_ratio=float(data_cfg["train_ratio"]),
        val_ratio=float(data_cfg["val_ratio"]),
        test_ratio=float(data_cfg["test_ratio"]),
        split_mode=str(data_cfg.get("split_mode", "rows")),
        history_steps=int(data_cfg["L"]),
        horizon_steps=int(data_cfg["H"]),
        collection_interval_min=int(data_cfg.get("collection_interval_min", 15)),
        gap_break_min=int(data_cfg.get("gap_break_min", 120)),
        gap_fill_min=int(data_cfg.get("gap_fill_min", 60)),
        use_missing_mask=bool(data_cfg.get("use_missing_mask", True)),
        include_target_history=bool(data_cfg.get("include_target_history", False)),
        split_col=str(data_cfg.get("split_col", "split")),
        sample_keep_col=(
            str(data_cfg["sample_keep_col"])
            if data_cfg.get("sample_keep_col") is not None
            else None
        ),
        respect_existing_segment_id=bool(data_cfg.get("respect_existing_segment_id", False)),
    )[0]


def _split_payload(prepared, split_name: str, edge_key: str) -> Dict[str, Any]:
    if split_name == "train":
        return {
            "indices": prepared.sample_indices_train,
            "timestamps": prepared.timestamps_train,
            "extra_targets": prepared.extra_targets_train,
        }
    if split_name == "val":
        return {
            "indices": prepared.sample_indices_val,
            "timestamps": prepared.timestamps_val,
            "extra_targets": prepared.extra_targets_val,
        }
    if split_name == "test":
        return {
            "indices": prepared.sample_indices_test,
            "timestamps": prepared.timestamps_test,
            "extra_targets": prepared.extra_targets_test,
        }
    raise ValueError(f"Unknown split: {split_name}")


def _sample_frame(
    cfg: Dict[str, Any],
    prepared,
    split_name: str,
    edge_key: str,
) -> pd.DataFrame:
    payload = _split_payload(prepared, split_name, edge_key)
    indices = np.asarray(payload["indices"], dtype=np.int64)
    timestamps = np.asarray(payload["timestamps"])
    extra_targets = payload["extra_targets"] or {}
    if edge_key not in extra_targets:
        raise ValueError(f"Missing extra target {edge_key!r} in split {split_name}")

    raw = _raw_split_frame(cfg, split_name)
    raw_lookup_cols = ["timestamp", "lag_gt", "_raw_row_in_split"]
    for optional in ["inject_flag", "segment_id", "segment_dmax_gt", "bump_dmax_gt"]:
        if optional in raw.columns:
            raw_lookup_cols.append(optional)
    raw_lookup = raw[raw_lookup_cols].copy()

    label = extra_targets[edge_key][indices].astype(np.int64)
    out = pd.DataFrame(
        {
            "split": split_name,
            "sample_index": indices,
            "timestamp": timestamps[indices],
            "train_lag_label": label,
            "train_det_label": (label > 0).astype(np.int64),
        }
    )
    out["timestamp"] = pd.to_datetime(out["timestamp"]).dt.strftime(TIME_FORMAT)
    out = out.merge(raw_lookup, on="timestamp", how="left")
    out = out.rename(columns={"lag_gt": "raw_lag_gt"})
    out["raw_det_label"] = out["raw_lag_gt"].fillna(-1).astype(int).gt(0).astype(np.int64)
    out["label_matches_raw"] = out["train_lag_label"].eq(out["raw_lag_gt"])
    out["det_matches_raw"] = out["train_det_label"].eq(out["raw_det_label"])

    raw_lag_by_row = raw["lag_gt"].astype(int).to_numpy()
    prev_values: List[float] = []
    next_values: List[float] = []
    for row_idx in out["_raw_row_in_split"].fillna(-1).astype(int).to_numpy():
        prev_values.append(raw_lag_by_row[row_idx - 1] if row_idx > 0 else np.nan)
        next_values.append(raw_lag_by_row[row_idx + 1] if 0 <= row_idx < len(raw_lag_by_row) - 1 else np.nan)
    out["raw_lag_prev"] = prev_values
    out["raw_lag_next"] = next_values
    out["matches_prev_lag"] = out["train_lag_label"].eq(out["raw_lag_prev"])
    out["matches_next_lag"] = out["train_lag_label"].eq(out["raw_lag_next"])

    if "inject_flag" in out.columns:
        out["in_block"] = out["inject_flag"].fillna(0).astype(int)
    else:
        out["in_block"] = out["raw_lag_gt"].fillna(0).astype(int).gt(0).astype(int)
    return out


def _summarize_split(frame: pd.DataFrame, split_name: str, n_lag_classes: int) -> Dict[str, Any]:
    label = frame["train_lag_label"].astype(int)
    det = frame["train_det_label"].astype(int)
    mag_mask = label.gt(0) & label.lt(n_lag_classes)
    in_block = frame["in_block"].astype(int).gt(0)
    block_out = ~in_block
    return {
        "split": split_name,
        "n_samples": int(len(frame)),
        "n_positive": int(det.sum()),
        "n_zero": int((label == 0).sum()),
        "n_invalid_negative": int((label < 0).sum()),
        "n_invalid_ge_k": int((label >= n_lag_classes).sum()),
        "label_mismatch_raw": int((~frame["label_matches_raw"]).sum()),
        "det_mismatch_raw": int((~frame["det_matches_raw"]).sum()),
        "match_current_rate": float(frame["label_matches_raw"].mean()),
        "match_prev_rate": float(frame["matches_prev_lag"].mean()),
        "match_next_rate": float(frame["matches_next_lag"].mean()),
        "block_in_samples": int(in_block.sum()),
        "block_in_det0": int((in_block & det.eq(0)).sum()),
        "block_out_samples": int(block_out.sum()),
        "block_out_det1": int((block_out & det.eq(1)).sum()),
        "mag_mask_samples": int(mag_mask.sum()),
        "mag_mask_negative_samples": int((mag_mask & label.le(0)).sum()),
        "mag_mask_invalid_samples": int((mag_mask & ((label < 0) | (label >= n_lag_classes))).sum()),
    }


def _source_semantics() -> Dict[str, Any]:
    model_source = Path("src/models/delay_alignment.py").read_text(encoding="utf-8")
    train_source = Path("train.py").read_text(encoding="utf-8")
    return {
        "model_uses_nonzero_sigmoid": "nonzero_prob = torch.sigmoid(occ_logit)" in model_source,
        "model_sets_pi0_to_one_minus_nonzero": "pi0 = (1.0 - nonzero_prob).unsqueeze(-1)" in model_source,
        "model_sets_positive_mass_to_nonzero": "pi_pos = nonzero_prob.unsqueeze(-1) * pos_pi" in model_source,
        "bce_uses_one_minus_pi0": "occ_prob = 1.0 - probs[:, 0]" in train_source,
        "bce_target_is_lag_gt_positive": "occ_target = lag_target[valid].gt(0)" in train_source,
        "magnitude_mask_is_positive_lag": "valid = (lag_target > 0) & (lag_target < arr_last.shape[-1])" in train_source,
    }


def main() -> None:
    args = parse_args()
    config_path = _absolute_path(args.config)
    output_dir = _absolute_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_yaml(config_path)
    prepared = _prepared_data(cfg)
    n_lag_classes = int(cfg["data"]["L_max"]) + 1

    rows = []
    for split_name in ["train", "val", "test"]:
        frame = _sample_frame(cfg, prepared, split_name, args.edge_key)
        frame.to_csv(output_dir / f"label_mask_audit_samples_{split_name}.csv", index=False)
        rows.append(_summarize_split(frame, split_name, n_lag_classes))

    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "label_mask_audit_summary.csv", index=False)

    semantics = _source_semantics()
    report = {
        "config": config_path.as_posix(),
        "edge_key": args.edge_key,
        "n_lag_classes": n_lag_classes,
        "summary": rows,
        "source_semantics": semantics,
    }
    (output_dir / "label_mask_audit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(summary.to_csv(index=False))
    print(json.dumps({"source_semantics": semantics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
