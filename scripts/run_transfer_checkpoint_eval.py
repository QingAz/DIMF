#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import re

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train import eval_test, load_config  # noqa: E402
from src.data.dataset import MultistageWindowDataset, WindowSpec  # noqa: E402
from src.data.dataprocess import (  # noqa: E402
    _infer_groups,
    _mask_col_name,
    _regularize_split_with_gap_policy,
    _sample_indices_from_regularized_split,
    _split_predefined_rows,
    _split_rows,
)
from src.models.dimf import DIMF  # noqa: E402


def _path(text: str | Path) -> Path:
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _extract_extra_targets(df_part: pd.DataFrame) -> Dict[str, np.ndarray] | None:
    extra_targets: Dict[str, np.ndarray] = {}
    if "lag_gt" in df_part.columns:
        extra_targets["stage1_to_stage2_lag_gt"] = df_part["lag_gt"].fillna(-1).astype(np.int64).to_numpy()
    if "inject_flag" in df_part.columns:
        extra_targets["stage1_to_stage2_in_block_gt"] = df_part["inject_flag"].fillna(0).astype(np.int64).to_numpy()
    if "segment_dmax_gt" in df_part.columns:
        extra_targets["stage1_to_stage2_dmax_gt"] = df_part["segment_dmax_gt"].fillna(0).astype(np.int64).to_numpy()
    return extra_targets or None


def _build_model(cfg: Dict[str, Any], group_dims: Dict[str, int], device: torch.device) -> DIMF:
    return DIMF(
        group_dims=group_dims,
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        num_layers=int(cfg["model"]["num_layers"]),
        dropout=float(cfg["model"]["dropout"]),
        attn_dim=int(cfg["model"]["attn_dim"]),
        L_max=int(cfg["data"]["L_max"]),
        lead_steps=int(cfg["data"]["H"]),
        encoder_type=str(cfg["model"].get("encoder", "gru")),
        transformer_nhead=int(cfg["model"].get("transformer_nhead", 4)),
        transformer_ff_dim=cfg["model"].get("transformer_ff_dim", None),
        max_len=int(cfg["data"]["L"]),
        lag_emb=bool(cfg["model"].get("lag_emb", True)),
        use_alignment=bool(cfg["model"].get("use_alignment", True)),
        align_tau=float(cfg["model"].get("align_tau", 1.0)),
        align_dropout=float(cfg["model"].get("align_dropout", 0.0)),
        align_feed_to_stage1=cfg["model"].get("align_feed_to_stage1"),
        align_stage1_to_stage2=cfg["model"].get("align_stage1_to_stage2"),
        align_stage2_to_stage3=cfg["model"].get("align_stage2_to_stage3"),
        use_lag_bias=bool(cfg["model"].get("use_lag_bias", True)),
        lag_head_mode=str(cfg["model"].get("lag_head_mode", "softmax")),
    ).to(device)


def _prepare_splits_with_external_scalers(
    cfg: Dict[str, Any],
    csv_path: Path,
    scaler_x: Any,
    scaler_y: Any,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[str]]]:
    data_cfg = cfg["data"]
    df = pd.read_csv(csv_path)
    df[data_cfg["time_col"]] = pd.to_datetime(df[data_cfg["time_col"]])
    df = df.sort_values(data_cfg["time_col"]).reset_index(drop=True)

    groups = _infer_groups(
        df,
        time_col=data_cfg["time_col"],
        target_col=data_cfg["target_col"],
        feed_prefix=data_cfg["feed_prefix"],
        s1_prefix=data_cfg["stage1_prefix"],
        s2_prefix=data_cfg["stage2_prefix"],
        s3_prefix=data_cfg["stage3_prefix"],
    )
    if bool(data_cfg.get("include_target_history", False)) and data_cfg["target_col"] not in groups["stage3"]:
        groups["stage3"] = groups["stage3"] + [data_cfg["target_col"]]

    split_mode = str(data_cfg.get("split_mode", "rows"))
    if split_mode in {"rows", "valid_segments"}:
        raw_parts = _split_rows(
            df,
            train_ratio=float(data_cfg["train_ratio"]),
            val_ratio=float(data_cfg["val_ratio"]),
            test_ratio=float(data_cfg["test_ratio"]),
        )
    elif split_mode == "predefined_valid_segments":
        raw_parts = _split_predefined_rows(
            df,
            time_col=data_cfg["time_col"],
            split_col=str(data_cfg.get("split_col", "split")),
        )
    else:
        raise ValueError(f"Unsupported split_mode for transfer eval: {split_mode}")

    sample_keep_col = (
        str(data_cfg["sample_keep_col"])
        if data_cfg.get("sample_keep_col") is not None
        else None
    )
    if sample_keep_col is not None and sample_keep_col not in df.columns:
        sample_keep_col = None

    regularized_parts = []
    for part in raw_parts:
        regularized_parts.append(
            _regularize_split_with_gap_policy(
                df_part=part,
                time_col=data_cfg["time_col"],
                collection_interval_min=int(data_cfg.get("collection_interval_min", 15)),
                gap_break_min=int(data_cfg.get("gap_break_min", 120)),
                gap_fill_min=int(data_cfg.get("gap_fill_min", 60)),
                fillna=data_cfg.get("fillna", "ffill"),
                use_delta_t=bool(data_cfg.get("use_delta_t", True)),
                sample_keep_col=sample_keep_col,
                respect_existing_segment_id=bool(data_cfg.get("respect_existing_segment_id", False)),
            )
        )
    df_train, df_val, df_test = regularized_parts

    groups_with_aux: Dict[str, List[str]] = {}
    for group_name, base_cols in groups.items():
        cols = list(base_cols)
        if bool(data_cfg.get("use_delta_t", True)):
            cols.append("delta_t_min")
        if bool(data_cfg.get("use_missing_mask", True)):
            cols.extend([_mask_col_name(col) for col in base_cols])
        groups_with_aux[group_name] = cols

    split_frames = {"train": df_train, "val": df_val, "test": df_test}
    for df_part in split_frames.values():
        df_part.attrs["source_csv"] = csv_path.as_posix()
    sample_indices = {
        split_name: _sample_indices_from_regularized_split(
            df_part,
            time_col=data_cfg["time_col"],
            history_steps=int(data_cfg["L"]),
            horizon_steps=int(data_cfg["H"]),
            collection_interval_min=int(data_cfg.get("collection_interval_min", 15)),
            sample_keep_col=sample_keep_col,
        )
        for split_name, df_part in split_frames.items()
    }

    x_cols_all = sorted(set(sum(groups_with_aux.values(), [])))
    mask_cols_all = sorted([col for col in x_cols_all if col.startswith("mask_")])
    scaled_x_cols = [col for col in x_cols_all if col not in mask_cols_all]
    if int(getattr(scaler_x, "n_features_in_", len(scaled_x_cols))) != len(scaled_x_cols):
        raise ValueError(
            "Loaded scaler_x feature dimension does not match transfer dataset feature layout: "
            f"{getattr(scaler_x, 'n_features_in_', 'unknown')} vs {len(scaled_x_cols)}"
        )
    scaled_col_to_idx = {col: idx for idx, col in enumerate(scaled_x_cols)}

    prepared: Dict[str, Dict[str, Any]] = {}
    for split_name, df_part in split_frames.items():
        scaled_x_all = scaler_x.transform(df_part[scaled_x_cols].values).astype(np.float32)
        x_groups: Dict[str, np.ndarray] = {}
        for group_name, cols in groups_with_aux.items():
            arrays = []
            for col in cols:
                if col in mask_cols_all:
                    arrays.append(df_part[[col]].values.astype(np.float32))
                else:
                    arrays.append(scaled_x_all[:, [scaled_col_to_idx[col]]])
            x_groups[group_name] = np.concatenate(arrays, axis=1).astype(np.float32)
        y = scaler_y.transform(df_part[[data_cfg["target_col"]]].values).astype(np.float32).reshape(-1)
        prepared[split_name] = {
            "frame": df_part,
            "X_groups": x_groups,
            "y": y,
            "indices": sample_indices[split_name],
            "timestamps": df_part[data_cfg["time_col"]].dt.strftime("%Y-%m-%d %H:%M").to_numpy(),
            "extra_targets": _extract_extra_targets(df_part),
        }
    return prepared, groups_with_aux


def _run_one_split(
    split_name: str,
    model: Any,
    scaler_y: Any,
    prepared_split: Dict[str, Any],
    cfg: Dict[str, Any],
    output_dir: Path,
    device: torch.device,
) -> Dict[str, Any]:
    spec = WindowSpec(L=int(cfg["data"]["L"]), H=int(cfg["data"]["H"]))
    ds = MultistageWindowDataset(
        prepared_split["X_groups"],
        prepared_split["y"],
        spec,
        indices=prepared_split["indices"],
        extra_targets=prepared_split["extra_targets"],
    )
    loader = DataLoader(ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=False, drop_last=False)
    timestamps = np.asarray(prepared_split["timestamps"])
    input_timestamps = timestamps[ds.indices]
    target_timestamps = timestamps[ds.indices + int(cfg["data"]["H"])]
    split_out = output_dir / split_name
    metrics = eval_test(
        model=model,
        loader=loader,
        device=device,
        scaler_y=scaler_y,
        output_dir=str(split_out),
        input_timestamps=input_timestamps,
        target_timestamps=target_timestamps,
        collection_interval_min=int(cfg["data"].get("collection_interval_min", 15)),
    )
    metrics["split"] = split_name
    metrics["n_samples"] = int(len(ds))
    metrics["source_csv"] = prepared_split["frame"].attrs.get("source_csv", "")
    (split_out / f"{split_name}_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a trained DIMF checkpoint on a different raw-gap dataset without retraining.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--csv-path", required=True, help="Transfer raw-gap CSV to evaluate on.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path. Defaults to config logging.ckpt_path.")
    parser.add_argument("--scaler-path", default=None, help="Scaler bundle path. Defaults to config logging.scaler_path.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--splits", default="train,val,test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(str(_path(args.config)))
    csv_path = _path(args.csv_path)
    output_dir = _path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = _path(args.checkpoint) if args.checkpoint else _path(cfg["logging"]["ckpt_path"])
    scaler_path = _path(args.scaler_path) if args.scaler_path else _path(cfg["logging"]["scaler_path"])

    scaler_bundle = joblib.load(scaler_path)
    scaler_x = scaler_bundle["scaler_x"]
    scaler_y = scaler_bundle["scaler_y"]
    prepared, _ = _prepare_splits_with_external_scalers(cfg, csv_path=csv_path, scaler_x=scaler_x, scaler_y=scaler_y)

    device = torch.device(args.device)
    group_dims = {
        group_name: array.shape[1]
        for group_name, array in prepared["train"]["X_groups"].items()
    }
    model = _build_model(cfg, group_dims, device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])

    metrics_rows: List[Dict[str, Any]] = []
    requested_splits = [part.strip() for part in re.split(r"[,+]", str(args.splits)) if part.strip()]
    for split_name in requested_splits:
        if split_name not in prepared:
            raise ValueError(f"Unknown split {split_name!r}; expected one of train,val,test")
        metrics_rows.append(
            _run_one_split(
                split_name=split_name,
                model=model,
                scaler_y=scaler_y,
                prepared_split=prepared[split_name],
                cfg=cfg,
                output_dir=output_dir,
                device=device,
            )
        )

    summary = {
        "config": str(_path(args.config)),
        "csv_path": csv_path.as_posix(),
        "checkpoint": checkpoint_path.as_posix(),
        "scaler_path": scaler_path.as_posix(),
        "device": str(device),
        "splits": requested_splits,
        "metrics": metrics_rows,
    }
    (output_dir / "transfer_eval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(metrics_rows).to_csv(output_dir / "transfer_eval_metrics_by_split.csv", index=False)
    print(pd.DataFrame(metrics_rows).to_csv(index=False, float_format="%.6f"))
    print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main()
