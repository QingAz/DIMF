#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import sysconfig
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
if sys.platform == "win32":
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        exe_path = Path(sys.executable).resolve()
        conda_prefix = str(exe_path.parent)
        if exe_path.parent.name.lower() in {"scripts", "bin"}:
            conda_prefix = str(exe_path.parent.parent)
    libbin = os.path.join(conda_prefix, "Library", "bin")
    site_packages = sysconfig.get_paths().get("purelib")
    torch_lib = os.path.join(site_packages, "torch", "lib") if site_packages else None
    if torch_lib and os.path.isdir(torch_lib):
        os.environ["PATH"] = torch_lib + ";" + os.environ.get("PATH", "")
        try:
            os.add_dll_directory(torch_lib)
        except Exception:
            pass
    if os.path.isdir(libbin):
        os.environ["PATH"] = libbin + ";" + os.environ.get("PATH", "")
        try:
            os.add_dll_directory(libbin)
        except Exception:
            pass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataprocess import load_and_prepare
from src.data.dataset import MultistageWindowDataset, WindowSpec
from src.data.lag_injection import ID_TO_SHAPE, SHAPE_TO_ID, inject_lag_csv
from src.metrics.lag_metrics import compute_lag_metrics, prediction_metrics, save_lag_metric_tables
from src.metrics.lag_visualization import (
    save_by_shape_bar_chart,
    save_expected_lag_curve,
    save_lag_distribution_heatmap,
    save_no_lag_false_alarm_plot,
    save_selected_feature_lag_heatmap,
    save_viterbi_lag_curve,
)
from src.models.dimf import DIMF
from src.models.lag_feature_screening import (
    apply_feature_screening_to_prior,
    attention_mass_score,
    combine_feature_scores,
    entropy_penalty_score,
    screening_report,
    select_feature_mask,
)
from src.models.stda_lag_identifier import LagLossWeights, STDALagIdentifier, lag_identifier_loss
from src.postprocess.viterbi_lag_decoder import path_to_onehot, viterbi_decode_lag
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lag-grounded DIMF training and evaluation.")
    parser.add_argument("--config", type=Path, default=Path("configs/lag_grounded_dimf.yaml"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--summary-only", action="store_true", help="Only run lag injection and summary export.")
    parser.add_argument(
        "--lag-only",
        action="store_true",
        help="Auxiliary lag-diagnostic mode for the DIMF lag-guided prior generator; it does not run y prediction.",
    )
    parser.add_argument(
        "--raw-adapt",
        action="store_true",
        help="Use a pretrained DIMF lag-guided prior generator and train DIMF on raw y targets without synthetic lag labels.",
    )
    parser.add_argument(
        "--eval-stage2-no-joint",
        action="store_true",
        help="Evaluate existing best_lag_identifier.pt and best_val_pred_dimf.pt without Stage3 joint fine-tuning.",
    )
    parser.add_argument("--checkpoint-dir", type=Path, help="Checkpoint directory used by --eval-stage2-no-joint.")
    parser.add_argument("--eval-output-dir", type=Path, help="Output directory used by --eval-stage2-no-joint.")
    return parser.parse_args()


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if key == "shapes":
            out[key] = value
        elif isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: Path) -> Dict[str, Any]:
    cfg = _load_yaml(path)
    base_path = cfg.pop("base_config", None)
    if base_path:
        base_file = (path.parent / base_path).resolve()
        return _deep_merge(load_config(base_file), cfg)
    return cfg


def _target_cols(data_cfg: Dict[str, Any]) -> list[str]:
    raw = data_cfg.get("target_cols")
    if raw is None:
        return [data_cfg["target_col"]]
    if isinstance(raw, str):
        return [part.strip() for part in raw.split(",") if part.strip()]
    return list(raw)


def _edge_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    edges = cfg.get("lag_edges") or cfg.get("lag_injection", {}).get("lag_edges")
    if edges:
        return dict(edges[0])
    return {
        "name": "stage1_to_stage2",
        "source_stage": "stage1",
        "target_stage": "stage2",
        "source_features": "all",
        "target_features": "all",
        "inject_strength": 0.5,
    }


def _history_steps(data_cfg: Dict[str, Any]) -> int:
    if "seq_len" in data_cfg:
        return int(data_cfg["seq_len"])
    if "window_size" in data_cfg:
        return int(data_cfg["window_size"])
    return int(data_cfg.get("L", 12))


def _prepare_injected_dataset(cfg: Dict[str, Any], results_dir: Path) -> Path:
    data_cfg = cfg["data"]
    lag_cfg = dict(cfg.get("lag_injection", {}))
    lag_cfg.setdefault("summary_dir", str(results_dir))
    injected_dir = results_dir / "dataset"
    injected_csv = injected_dir / "lag_injected.csv"
    metadata_csv = injected_dir / "lag_metadata.csv"
    prefix_by_stage = {
        "feed": data_cfg.get("feed_prefix", "feed_"),
        "stage1": data_cfg.get("stage1_prefix", "stage1_"),
        "stage2": data_cfg.get("stage2_prefix", "stage2_"),
        "stage3": data_cfg.get("stage3_prefix", "stage3_"),
    }
    inject_lag_csv(
        input_csv=data_cfg["csv_path"],
        output_csv=injected_csv,
        metadata_csv=metadata_csv,
        time_col=data_cfg["time_col"],
        target_cols=_target_cols(data_cfg),
        lag_injection_cfg=lag_cfg,
        lag_edges=cfg.get("lag_edges") or lag_cfg.get("lag_edges"),
        prefix_by_stage=prefix_by_stage,
    )
    return injected_csv


def _load_prepared(cfg: Dict[str, Any], injected_csv: Path):
    data_cfg = dict(cfg["data"])
    data_cfg["csv_path"] = str(injected_csv)
    data_cfg["split_mode"] = "predefined_valid_segments"
    data_cfg["split_col"] = "model_split"
    return load_and_prepare(
        csv_path=data_cfg["csv_path"],
        time_col=data_cfg["time_col"],
        target_col=data_cfg["target_col"],
        target_cols=data_cfg.get("target_cols"),
        feed_prefix=data_cfg["feed_prefix"],
        stage1_prefix=data_cfg["stage1_prefix"],
        stage2_prefix=data_cfg["stage2_prefix"],
        stage3_prefix=data_cfg["stage3_prefix"],
        fillna=data_cfg.get("fillna", "ffill"),
        use_delta_t=bool(data_cfg.get("use_delta_t", True)),
        train_ratio=float(data_cfg.get("train_ratio", 0.7)),
        val_ratio=float(data_cfg.get("val_ratio", 0.1)),
        test_ratio=float(data_cfg.get("test_ratio", 0.2)),
        split_mode=data_cfg["split_mode"],
        history_steps=_history_steps(data_cfg),
        horizon_steps=int(data_cfg["H"]),
        collection_interval_min=int(data_cfg.get("collection_interval_min", 15)),
        gap_break_min=int(data_cfg.get("gap_break_min", 120)),
        gap_fill_min=int(data_cfg.get("gap_fill_min", 60)),
        use_missing_mask=bool(data_cfg.get("use_missing_mask", True)),
        include_target_history=bool(data_cfg.get("include_target_history", False)),
        split_col=data_cfg["split_col"],
        sample_keep_col=data_cfg.get("sample_keep_col"),
        respect_existing_segment_id=bool(data_cfg.get("respect_existing_segment_id", False)),
    )[0]


def _load_raw_prepared(cfg: Dict[str, Any]):
    data_cfg = cfg["data"]
    return load_and_prepare(
        csv_path=data_cfg["csv_path"],
        time_col=data_cfg["time_col"],
        target_col=data_cfg["target_col"],
        target_cols=data_cfg.get("target_cols"),
        feed_prefix=data_cfg["feed_prefix"],
        stage1_prefix=data_cfg["stage1_prefix"],
        stage2_prefix=data_cfg["stage2_prefix"],
        stage3_prefix=data_cfg["stage3_prefix"],
        fillna=data_cfg.get("fillna", "ffill"),
        use_delta_t=bool(data_cfg.get("use_delta_t", False)),
        train_ratio=float(data_cfg.get("train_ratio", 0.7)),
        val_ratio=float(data_cfg.get("val_ratio", 0.1)),
        test_ratio=float(data_cfg.get("test_ratio", 0.2)),
        split_mode=str(data_cfg.get("split_mode", "rows")),
        history_steps=_history_steps(data_cfg),
        horizon_steps=int(data_cfg["H"]),
        collection_interval_min=int(data_cfg.get("collection_interval_min", 15)),
        gap_break_min=int(data_cfg.get("gap_break_min", data_cfg.get("collection_interval_min", 15))),
        gap_fill_min=int(data_cfg.get("gap_fill_min", 0)),
        use_missing_mask=bool(data_cfg.get("use_missing_mask", False)),
        include_target_history=bool(data_cfg.get("include_target_history", False)),
        split_col=str(data_cfg.get("split_col", "split")),
        sample_keep_col=data_cfg.get("sample_keep_col"),
        respect_existing_segment_id=bool(data_cfg.get("respect_existing_segment_id", False)),
    )[0]


def _raw_group_columns(
    df: pd.DataFrame,
    data_cfg: Dict[str, Any],
    target_cols: list[str],
) -> Dict[str, list[str]]:
    excluded = {
        data_cfg["time_col"],
        *target_cols,
        "split",
        "model_split",
        "is_interpolated",
        "segment_id",
        "region_id",
        "tile_id",
        "lag_gt",
        "lag_binary_gt",
        "inject_flag",
        "bump_dmax_gt",
        "segment_dmax_gt",
        "g_stage1_to_stage2",
        "lag_shape_gt",
        "lag_pattern_gt",
    }
    prefixes = {
        "feed": data_cfg.get("feed_prefix", "feed_"),
        "stage1": data_cfg.get("stage1_prefix", "stage1_"),
        "stage2": data_cfg.get("stage2_prefix", "stage2_"),
        "stage3": data_cfg.get("stage3_prefix", "stage3_"),
    }
    groups = {}
    for group_name, prefix in prefixes.items():
        cols = [col for col in df.columns if col not in excluded and col.startswith(prefix)]
        if not cols:
            raise ValueError(f"No columns found for group {group_name!r} with prefix {prefix!r}")
        groups[group_name] = sorted(cols)
    return groups


def _segment_window_indices(
    df: pd.DataFrame,
    data_cfg: Dict[str, Any],
    L: int,
    H: int,
) -> np.ndarray:
    if "segment_id" not in df.columns:
        return np.arange(L - 1, len(df) - H, dtype=np.int64)
    time_col = data_cfg["time_col"]
    interval_min = int(data_cfg.get("collection_interval_min", 15))
    expected_span = pd.Timedelta(minutes=interval_min * (L + H - 1))
    indices = []
    for _, seg in df.groupby("segment_id", sort=False):
        seg = seg.reset_index().rename(columns={"index": "_global_index"})
        if len(seg) < L + H:
            continue
        for local_t in range(L - 1, len(seg) - H):
            start = local_t - L + 1
            end = local_t + H
            if pd.api.types.is_datetime64_any_dtype(seg[time_col]):
                if seg[time_col].iloc[end] - seg[time_col].iloc[start] != expected_span:
                    continue
            indices.append(int(seg["_global_index"].iloc[local_t]))
    if not indices:
        raise ValueError("No valid train/test windows found in existing split dataset")
    return np.asarray(indices, dtype=np.int64)


def _ensure_segment_metadata(df: pd.DataFrame, data_cfg: Dict[str, Any]) -> pd.DataFrame:
    df = df.copy()
    time_col = data_cfg["time_col"]
    df = df.sort_values(time_col).reset_index(drop=True)
    if "time_index" not in df.columns:
        df["time_index"] = np.arange(len(df), dtype=np.int64)
    if "region_id" not in df.columns and "tile_id" in df.columns:
        df["region_id"] = df["tile_id"]
    if "segment_id" in df.columns and bool(data_cfg.get("respect_existing_segment_id", False)):
        return df

    interval = pd.Timedelta(minutes=int(data_cfg.get("collection_interval_min", 15)))
    boundary = df[time_col].diff().ne(interval).fillna(True)
    if "region_id" in df.columns:
        region = df["region_id"].fillna(-1)
        boundary = boundary | region.ne(region.shift()).fillna(True)
    else:
        flag_col = None
        for candidate in ("lag_flag", "lag_binary_gt", "inject_flag"):
            if candidate in df.columns:
                flag_col = candidate
                break
        if flag_col is not None:
            flag = df[flag_col].fillna(0).astype(int)
            boundary = boundary | flag.ne(flag.shift()).fillna(True)
        if "lag_shape_gt" in df.columns:
            shape = df["lag_shape_gt"].fillna("none").astype(str)
            boundary = boundary | shape.ne(shape.shift()).fillna(True)
        if "lag_pattern_gt" in df.columns:
            pattern = df["lag_pattern_gt"].fillna("none").astype(str)
            boundary = boundary | pattern.ne(pattern.shift()).fillna(True)

    df["segment_id"] = boundary.astype(int).cumsum().astype(np.int64) - 1
    return df


def _shape_ids_from_existing(df: pd.DataFrame) -> np.ndarray:
    if "lag_shape_gt" not in df.columns:
        return np.zeros(len(df), dtype=np.int64)
    mapping = {
        "none": SHAPE_TO_ID["none"],
        "block": SHAPE_TO_ID["fixed"],
        "bump": SHAPE_TO_ID["local_bump"],
        "constant": SHAPE_TO_ID["fixed"],
        "fixed": SHAPE_TO_ID["fixed"],
        "random_discrete": SHAPE_TO_ID["random_discrete"],
        "gaussian": SHAPE_TO_ID["gaussian"],
        "ramp": SHAPE_TO_ID["ramp"],
        "sinusoidal": SHAPE_TO_ID["sinusoidal"],
        "bimodal": SHAPE_TO_ID["bimodal"],
        "local_bump": SHAPE_TO_ID["local_bump"],
    }
    return df["lag_shape_gt"].fillna("none").astype(str).map(mapping).fillna(0).astype(np.int64).to_numpy()


def _extra_targets_from_existing(df: pd.DataFrame, max_lag: int, edge_name: str = "stage1_to_stage2") -> Dict[str, np.ndarray]:
    lag = df["lag_gt"].fillna(0).astype(np.int64).clip(lower=0, upper=max_lag).to_numpy()
    if "lag_binary_gt" in df.columns:
        flag = df["lag_binary_gt"].fillna(0).astype(np.int64).to_numpy()
    elif "inject_flag" in df.columns:
        flag = df["inject_flag"].fillna(0).astype(np.int64).to_numpy()
    else:
        flag = (lag > 0).astype(np.int64)
    soft = np.zeros((len(df), max_lag + 1), dtype=np.float32)
    soft[np.arange(len(df)), lag] = 1.0
    sample_index = (
        df["sample_index"].fillna(-1).astype(np.int64).to_numpy()
        if "sample_index" in df.columns
        else np.arange(len(df), dtype=np.int64)
    )
    time_index = (
        df["time_index"].fillna(-1).astype(np.int64).to_numpy()
        if "time_index" in df.columns
        else sample_index
    )
    out = {
        f"{edge_name}_lag_gt": lag.astype(np.int64),
        f"{edge_name}_lag_expected_gt": lag.astype(np.float32),
        f"{edge_name}_lag_flag": flag.astype(np.int64),
        f"{edge_name}_shape_id": _shape_ids_from_existing(df),
        f"{edge_name}_lag_soft_gt": soft,
        "sample_index": sample_index,
        "time_index": time_index,
        f"{edge_name}_sample_index": sample_index,
        f"{edge_name}_time_index": time_index,
    }
    if "segment_id" in df.columns:
        segment_id = df["segment_id"].fillna(-1).astype(np.int64).to_numpy()
        out["segment_id"] = segment_id
        out[f"{edge_name}_segment_id"] = segment_id
    if "region_id" in df.columns:
        region_id = df["region_id"].fillna(-1).astype(np.int64).to_numpy()
        out["region_id"] = region_id
        out[f"{edge_name}_region_id"] = region_id
    return out


def _load_existing_train_test_prepared(cfg: Dict[str, Any]):
    data_cfg = cfg["data"]
    train_csv = Path(data_cfg["train_csv"])
    test_csv = Path(data_cfg["test_csv"])
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    time_col = data_cfg["time_col"]
    train_df[time_col] = pd.to_datetime(train_df[time_col])
    test_df[time_col] = pd.to_datetime(test_df[time_col])
    train_df = _ensure_segment_metadata(train_df, data_cfg)
    test_df = _ensure_segment_metadata(test_df, data_cfg)
    full_train_df_for_scaler = train_df.copy()
    val_df = None
    val_segment_ids = data_cfg.get("val_segment_ids")
    val_region_ids = data_cfg.get("val_region_ids")
    if val_segment_ids is not None:
        if "segment_id" not in train_df.columns:
            raise ValueError("data.val_segment_ids requires segment_id metadata")
        val_ids = {int(item) for item in val_segment_ids}
        val_mask = train_df["segment_id"].astype(int).isin(val_ids)
        if not bool(val_mask.any()):
            raise ValueError(f"data.val_segment_ids did not match any train rows: {sorted(val_ids)}")
        val_df = train_df.loc[val_mask].copy()
        train_df = train_df.loc[~val_mask].copy()
    elif val_region_ids is not None:
        if "region_id" not in train_df.columns:
            raise ValueError("data.val_region_ids requires region_id metadata")
        val_ids = {int(item) for item in val_region_ids}
        val_mask = train_df["region_id"].astype(int).isin(val_ids)
        if not bool(val_mask.any()):
            raise ValueError(f"data.val_region_ids did not match any train rows: {sorted(val_ids)}")
        val_df = train_df.loc[val_mask].copy()
        train_df = train_df.loc[~val_mask].copy()
    elif data_cfg.get("val_fraction_per_segment") is not None:
        if "segment_id" not in train_df.columns:
            raise ValueError("data.val_fraction_per_segment requires segment_id metadata")
        val_fraction = float(data_cfg.get("val_fraction_per_segment", 0.0))
        if not 0.0 < val_fraction < 1.0:
            raise ValueError("data.val_fraction_per_segment must be between 0 and 1")
        min_val_samples = int(data_cfg.get("val_min_samples_per_segment", _history_steps(data_cfg) + int(data_cfg["H"])))
        min_train_samples = int(data_cfg.get("val_min_train_samples_per_segment", _history_steps(data_cfg) + int(data_cfg["H"])))
        split_mode = str(data_cfg.get("val_split_mode", "tail"))
        block_size = int(data_cfg.get("val_block_size", max(min_val_samples, 64)))
        val_indices = []
        for _, seg in train_df.groupby("segment_id", sort=False):
            if len(seg) < min_val_samples + min_train_samples:
                continue
            n_val = max(min_val_samples, int(round(len(seg) * val_fraction)))
            n_val = min(n_val, len(seg) - min_train_samples)
            if n_val <= 0:
                continue
            if split_mode == "uniform_blocks":
                local_idx = seg.index.to_numpy()
                block = max(min_val_samples, min(block_size, n_val))
                n_blocks = max(1, int(round(n_val / block)))
                n_blocks = min(n_blocks, max(1, (len(seg) - min_train_samples) // block))
                max_start = max(0, len(seg) - block)
                if n_blocks == 1:
                    starts = [max_start // 2]
                else:
                    starts = np.linspace(0, max_start, num=n_blocks).round().astype(int).tolist()
                chosen = set()
                for start in starts:
                    end = min(len(seg), int(start) + block)
                    chosen.update(local_idx[int(start):end].tolist())
                val_indices.extend(sorted(chosen))
            elif split_mode == "tail":
                val_indices.extend(seg.index[-n_val:].tolist())
            else:
                raise ValueError(f"Unsupported data.val_split_mode: {split_mode}")
        if not val_indices:
            raise ValueError("data.val_fraction_per_segment did not produce any validation rows")
        val_mask = train_df.index.isin(val_indices)
        val_df = train_df.loc[val_mask].copy()
        train_df = train_df.loc[~val_mask].copy()
    if train_df.empty:
        raise ValueError("Validation split consumed all train rows")

    train_df = train_df.sort_values([c for c in ["segment_id", time_col] if c in train_df.columns]).reset_index(drop=True)
    if val_df is not None:
        val_df = val_df.sort_values([c for c in ["segment_id", time_col] if c in val_df.columns]).reset_index(drop=True)
    test_df = test_df.sort_values([c for c in ["segment_id", time_col] if c in test_df.columns]).reset_index(drop=True)

    target_cols = _target_cols(data_cfg)
    groups = _raw_group_columns(train_df, data_cfg, target_cols)
    x_cols_all = sorted(set(sum(groups.values(), [])))
    scaler_df = full_train_df_for_scaler if bool(data_cfg.get("fit_scaler_on_full_train_before_val", False)) else train_df
    scaler_x = StandardScaler().fit(scaler_df[x_cols_all].astype(float).to_numpy())
    scaler_y = StandardScaler().fit(scaler_df[target_cols].astype(float).to_numpy())
    col_to_idx = {col: idx for idx, col in enumerate(x_cols_all)}

    def transform(df: pd.DataFrame):
        scaled = scaler_x.transform(df[x_cols_all].astype(float).to_numpy()).astype(np.float32)
        x_groups = {}
        for group_name, cols in groups.items():
            x_groups[group_name] = scaled[:, [col_to_idx[col] for col in cols]].astype(np.float32)
        y = scaler_y.transform(df[target_cols].astype(float).to_numpy()).astype(np.float32)
        if len(target_cols) == 1:
            y = y.reshape(-1)
        return x_groups, y

    L = _history_steps(data_cfg)
    H = int(data_cfg["H"])
    max_lag = int(cfg.get("lag_identifier", {}).get("max_lag", data_cfg.get("L_max", 12)))
    edge_name = str(_edge_cfg(cfg).get("name", "stage1_to_stage2"))
    x_train, y_train = transform(train_df)
    x_val, y_val = (None, None) if val_df is None else transform(val_df)
    x_test, y_test = transform(test_df)
    segment_ids_train = train_df["segment_id"].to_numpy() if "segment_id" in train_df.columns else None
    segment_ids_val = None if val_df is None or "segment_id" not in val_df.columns else val_df["segment_id"].to_numpy()
    segment_ids_test = test_df["segment_id"].to_numpy() if "segment_id" in test_df.columns else None
    return SimpleNamespace(
        X_groups_train=x_train,
        y_train=y_train,
        X_groups_val=x_val,
        y_val=y_val,
        X_groups_test=x_test,
        y_test=y_test,
        group_dims={key: value.shape[1] for key, value in x_train.items()},
        scaler_x=scaler_x,
        scaler_y=scaler_y,
        sample_indices_train=_segment_window_indices(train_df, data_cfg, L, H),
        sample_indices_val=None if val_df is None else _segment_window_indices(val_df, data_cfg, L, H),
        sample_indices_test=_segment_window_indices(test_df, data_cfg, L, H),
        segment_ids_train=segment_ids_train,
        segment_ids_val=segment_ids_val,
        segment_ids_test=segment_ids_test,
        timestamps_train=train_df[time_col].dt.strftime("%Y-%m-%d %H:%M").to_numpy(),
        timestamps_val=None if val_df is None else val_df[time_col].dt.strftime("%Y-%m-%d %H:%M").to_numpy(),
        timestamps_test=test_df[time_col].dt.strftime("%Y-%m-%d %H:%M").to_numpy(),
        extra_targets_train=_extra_targets_from_existing(train_df, max_lag=max_lag, edge_name=edge_name),
        extra_targets_val=None if val_df is None else _extra_targets_from_existing(val_df, max_lag=max_lag, edge_name=edge_name),
        extra_targets_test=_extra_targets_from_existing(test_df, max_lag=max_lag, edge_name=edge_name),
        target_cols=target_cols,
    )


def _datasets_and_loaders(cfg: Dict[str, Any], prepared):
    spec = WindowSpec(L=_history_steps(cfg["data"]), H=int(cfg["data"]["H"]))
    ds_tr = MultistageWindowDataset(prepared.X_groups_train, prepared.y_train, spec, prepared.sample_indices_train, prepared.extra_targets_train)
    ds_va = None
    if getattr(prepared, "X_groups_val", None) is not None:
        ds_va = MultistageWindowDataset(prepared.X_groups_val, prepared.y_val, spec, prepared.sample_indices_val, prepared.extra_targets_val)
    ds_te = MultistageWindowDataset(prepared.X_groups_test, prepared.y_test, spec, prepared.sample_indices_test, prepared.extra_targets_test)
    batch_size = int(cfg.get("training", {}).get("batch_size", 64))
    return (
        ds_tr,
        ds_va,
        ds_te,
        DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=False),
        None if ds_va is None else DataLoader(ds_va, batch_size=batch_size, shuffle=False),
        DataLoader(ds_te, batch_size=batch_size, shuffle=False),
    )


def _dataset_extra(ds: MultistageWindowDataset, key: str) -> Optional[np.ndarray]:
    values = ds.extra_targets.get(key)
    if values is None:
        return None
    return np.asarray(values)[ds.indices]


def _make_lag_balanced_sampler(ds: MultistageWindowDataset, edge_name: str, sampler_cfg: Dict[str, Any]):
    if not bool(sampler_cfg.get("enabled", False)):
        return None, {"enabled": False}

    lag = _dataset_extra(ds, f"{edge_name}_lag_gt")
    if lag is None:
        return None, {"enabled": False, "reason": f"missing {edge_name}_lag_gt"}
    lag = lag.astype(np.int64)
    flag = _dataset_extra(ds, f"{edge_name}_lag_flag")
    flag = (lag > 0).astype(np.int64) if flag is None else flag.astype(np.int64)
    shape = _dataset_extra(ds, f"{edge_name}_shape_id")
    shape = np.zeros_like(lag, dtype=np.int64) if shape is None else shape.astype(np.int64)

    n = int(len(lag))
    if n == 0:
        return None, {"enabled": False, "reason": "empty dataset"}

    positive = (flag > 0) & (lag > 0)
    no_lag = ~positive
    weights = np.zeros(n, dtype=np.float64)
    positive_fraction = float(sampler_cfg.get("positive_fraction", 0.55))
    positive_fraction = min(max(positive_fraction, 0.0), 1.0)

    if positive.any():
        if bool(sampler_cfg.get("balance_lag_classes", True)):
            positive_lags = sorted(int(item) for item in np.unique(lag[positive]))
            per_class_mass = positive_fraction / max(len(positive_lags), 1)
            for lag_value in positive_lags:
                mask = positive & (lag == lag_value)
                weights[mask] = per_class_mass / max(int(mask.sum()), 1)
        else:
            weights[positive] = positive_fraction / max(int(positive.sum()), 1)

        if bool(sampler_cfg.get("balance_shapes", True)):
            positive_shapes = sorted(int(item) for item in np.unique(shape[positive]) if int(item) != 0)
            for shape_id in positive_shapes:
                mask = positive & (shape == shape_id)
                if not mask.any():
                    continue
                shape_scale = float(sampler_cfg.get("shape_weights", {}).get(ID_TO_SHAPE.get(shape_id, str(shape_id)), 1.0))
                weights[mask] *= shape_scale

    if no_lag.any():
        weights[no_lag] = (1.0 - positive_fraction) / max(int(no_lag.sum()), 1)

    if not np.isfinite(weights).all() or float(weights.sum()) <= 0.0:
        return None, {"enabled": False, "reason": "invalid sampler weights"}

    weights = weights / weights.mean()
    num_samples = int(sampler_cfg.get("num_samples") or n)
    generator = torch.Generator()
    generator.manual_seed(int(sampler_cfg.get("seed", 0)))
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=num_samples,
        replacement=bool(sampler_cfg.get("replacement", True)),
        generator=generator,
    )
    info = {
        "enabled": True,
        "strategy": str(sampler_cfg.get("strategy", "shape_lag_balanced")),
        "num_samples": int(num_samples),
        "positive_fraction_target": float(positive_fraction),
        "positive_samples": int(positive.sum()),
        "no_lag_samples": int(no_lag.sum()),
        "lag_counts": {str(int(k)): int(v) for k, v in zip(*np.unique(lag, return_counts=True))},
        "shape_counts": {ID_TO_SHAPE.get(int(k), str(int(k))): int(v) for k, v in zip(*np.unique(shape, return_counts=True))},
    }
    return sampler, info


def _make_lag_train_loader(
    cfg: Dict[str, Any],
    ds_tr: MultistageWindowDataset,
    batch_size: int,
    edge_name: str,
) -> tuple[DataLoader, Dict[str, Any]]:
    training_cfg = cfg.get("training", {})
    sampler_cfg = dict(training_cfg.get("lag_identifier_sampler") or {})
    sampler_cfg.setdefault("seed", int(cfg.get("seed", 0)))
    sampler, sampler_info = _make_lag_balanced_sampler(ds_tr, edge_name, sampler_cfg)
    if sampler is not None:
        return DataLoader(ds_tr, batch_size=batch_size, sampler=sampler, drop_last=False), sampler_info
    shuffle = bool(training_cfg.get("lag_identifier_shuffle", False))
    return DataLoader(ds_tr, batch_size=batch_size, shuffle=shuffle, drop_last=False), sampler_info


def _apply_lag_identifier_finetune_mode(identifier: STDALagIdentifier, cfg: Dict[str, Any]) -> Dict[str, Any]:
    lag_cfg = cfg.get("lag_identifier", {})
    training_cfg = cfg.get("training", {})
    mode = str(lag_cfg.get("fine_tune_mode", training_cfg.get("lag_identifier_fine_tune_mode", "all")))
    mode_patterns = {
        "all": None,
        "bias_only": ["lag_bias"],
        "bias_occurrence": ["lag_bias", "occurrence_head"],
        "heads_only": ["lag_bias", "score_proj", "feature_importance_head", "occurrence_head"],
        "smoother_heads": [
            "lag_bias",
            "score_proj",
            "feature_importance_head",
            "occurrence_head",
            "sequence_smoother",
            "sequence_smoother_proj",
        ],
    }
    patterns = lag_cfg.get("trainable_patterns", training_cfg.get("lag_identifier_trainable_patterns"))
    if patterns is None:
        patterns = mode_patterns.get(mode)
    if patterns is None:
        total = sum(param.numel() for param in identifier.parameters())
        return {"mode": mode, "trainable_parameters": int(total), "total_parameters": int(total)}
    patterns = [str(item) for item in patterns]
    trainable = 0
    total = 0
    trainable_names = []
    for name, param in identifier.named_parameters():
        total += int(param.numel())
        keep = any(pattern in name for pattern in patterns)
        param.requires_grad = bool(keep)
        if keep:
            trainable += int(param.numel())
            trainable_names.append(name)
    return {
        "mode": mode,
        "patterns": patterns,
        "trainable_parameters": int(trainable),
        "total_parameters": int(total),
        "trainable_names": trainable_names,
    }


def _to_device(batch, device):
    x, y = batch
    return {k: v.to(device) for k, v in x.items()}, y.to(device)


def _lag_targets(x: Dict[str, torch.Tensor], edge: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    soft_key = f"{edge}_lag_soft_gt"
    flag_key = f"{edge}_lag_flag"
    hard_key = f"{edge}_lag_gt"
    shape_key = f"{edge}_shape_id"
    if soft_key not in x:
        raise KeyError(f"Missing lag soft target {soft_key!r}; run lag injection first")
    soft = x[soft_key].float()
    flag = x[flag_key].float() if flag_key in x else x[hard_key].gt(0).float()
    hard = x[hard_key].long() if hard_key in x else soft.argmax(dim=-1).long()
    shape = x[shape_key].long() if shape_key in x else torch.zeros_like(hard)
    return soft, flag, hard, shape


def _lag_metadata(
    x: Dict[str, torch.Tensor],
    edge: str,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    segment = x.get(f"{edge}_segment_id", x.get("segment_id"))
    sample = x.get(f"{edge}_sample_index", x.get("sample_index", x.get("time_index")))
    time_index = x.get(f"{edge}_time_index", x.get("time_index", sample))
    return segment, sample, time_index


def _loss_weights(cfg: Dict[str, Any], stage: str) -> LagLossWeights:
    loss_cfg = cfg.get("loss", {})
    if stage == "joint":
        return LagLossWeights(
            lambda_soft_lag=float(loss_cfg.get("joint_lambda_soft_lag", 0.2)),
            lambda_expected_lag=float(loss_cfg.get("joint_lambda_expected_lag", 0.05)),
            lambda_occurrence=float(loss_cfg.get("joint_lambda_occurrence", 0.1)),
            occurrence_pos_weight=loss_cfg.get("joint_occurrence_pos_weight", loss_cfg.get("occurrence_pos_weight")),
            lambda_entropy=float(loss_cfg.get("lambda_entropy", 0.01)),
            lambda_smooth=float(loss_cfg.get("lambda_smooth", 0.005)),
            lambda_positive_smooth=float(loss_cfg.get("joint_lambda_positive_smooth", loss_cfg.get("lambda_positive_smooth", 0.0))),
            lambda_positive_ce=float(loss_cfg.get("joint_lambda_positive_ce", loss_cfg.get("lambda_positive_ce", 0.0))),
            positive_ce_class_weights=loss_cfg.get("positive_ce_class_weights"),
            use_gaussian_lag_label=bool(loss_cfg.get("use_gaussian_lag_label", True)),
            gaussian_lag_sigma=float(loss_cfg.get("gaussian_lag_sigma", 0.7)),
            enable_segment_aware_temporal_loss=bool(loss_cfg.get("enable_segment_aware_temporal_loss", False)),
            lambda_shape_curvature=float(loss_cfg.get("joint_lambda_shape_curvature", loss_cfg.get("lambda_shape_curvature", 0.0))),
            shape_curvature_ids=loss_cfg.get("shape_curvature_ids"),
        )
    return LagLossWeights(
        lambda_soft_lag=float(loss_cfg.get("lambda_soft_lag", 1.0)),
        lambda_expected_lag=float(loss_cfg.get("lambda_expected_lag", 0.1)),
        lambda_occurrence=float(loss_cfg.get("lambda_occurrence", 0.5)),
        occurrence_pos_weight=loss_cfg.get("occurrence_pos_weight"),
        lambda_entropy=float(loss_cfg.get("lambda_entropy", 0.01)),
        lambda_smooth=float(loss_cfg.get("lambda_smooth", 0.005)),
        lambda_positive_smooth=float(loss_cfg.get("lambda_positive_smooth", 0.0)),
        lambda_positive_ce=float(loss_cfg.get("lambda_positive_ce", 0.0)),
        positive_ce_class_weights=loss_cfg.get("positive_ce_class_weights"),
        use_gaussian_lag_label=bool(loss_cfg.get("use_gaussian_lag_label", True)),
        gaussian_lag_sigma=float(loss_cfg.get("gaussian_lag_sigma", 0.7)),
        enable_segment_aware_temporal_loss=bool(loss_cfg.get("enable_segment_aware_temporal_loss", False)),
        lambda_shape_curvature=float(loss_cfg.get("lambda_shape_curvature", 0.0)),
        shape_curvature_ids=loss_cfg.get("shape_curvature_ids"),
    )


def _lag_forward(identifier, x, edge_cfg, edge_name: Optional[str] = None, segment_id=None):
    if hasattr(identifier, "has_lag_guided_alignment") and identifier.has_lag_guided_alignment():
        if segment_id is None and edge_name is not None:
            segment_id, _, _ = _lag_metadata(x, edge_name)
        return identifier.predict_lag_prior(x, segment_id=segment_id)
    if segment_id is None and edge_name is not None:
        segment_id, _, _ = _lag_metadata(x, edge_name)
    return identifier(
        x[edge_cfg["source_stage"]],
        target_seq=x[edge_cfg["target_stage"]],
        segment_id=segment_id,
    )


def train_lag_identifier_epoch(
    identifier,
    loader,
    optimizer,
    device,
    edge_cfg,
    edge_name,
    weights,
    teacher=None,
    distill_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    identifier.train()
    if teacher is not None:
        teacher.eval()
    distill_cfg = dict(distill_cfg or {})
    totals: Dict[str, float] = {}
    n = 0
    for batch in loader:
        x, _ = _to_device(batch, device)
        soft, flag, hard, shape = _lag_targets(x, edge_name)
        segment_id, _, time_index = _lag_metadata(x, edge_name)
        out = _lag_forward(identifier, x, edge_cfg, edge_name=edge_name, segment_id=segment_id)
        losses = lag_identifier_loss(
            out,
            soft,
            flag,
            shape,
            weights,
            segment_id=segment_id,
            lag_gt=hard,
            sample_index=time_index,
        )
        if teacher is not None and bool(distill_cfg.get("enabled", False)):
            with torch.no_grad():
                teacher_out = _lag_forward(teacher, x, edge_cfg, edge_name=edge_name, segment_id=segment_id)
            student_pi = out["pi_edge"].clamp(min=1e-8, max=1.0)
            teacher_pi = teacher_out["pi_edge"].detach().clamp(min=1e-8, max=1.0)
            kl_per_sample = (teacher_pi * (teacher_pi.log() - student_pi.log())).sum(dim=-1)
            scope = str(distill_cfg.get("scope", "all"))
            if scope == "positive":
                mask = flag > 0.5
            elif scope == "no_lag":
                mask = flag <= 0.5
            else:
                mask = torch.ones_like(flag, dtype=torch.bool)
            if bool(mask.any()):
                teacher_kl = kl_per_sample[mask].mean()
            else:
                teacher_kl = kl_per_sample.mean()
            teacher_occ = torch.sigmoid(teacher_out["occurrence_logit_edge"].detach())
            student_occ = torch.sigmoid(out["occurrence_logit_edge"])
            teacher_occ_mse = F.mse_loss(student_occ, teacher_occ)
            losses["teacher_kl"] = teacher_kl
            losses["teacher_occ_mse"] = teacher_occ_mse
            losses["loss"] = (
                losses["loss"]
                + float(distill_cfg.get("kl_weight", 0.0)) * teacher_kl
                + float(distill_cfg.get("occurrence_weight", 0.0)) * teacher_occ_mse
            )
        optimizer.zero_grad()
        losses["loss"].backward()
        optimizer.step()
        batch_n = int(soft.shape[0])
        n += batch_n
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().item()) * batch_n
    return {key: value / max(n, 1) for key, value in totals.items()}


@torch.no_grad()
def collect_lag_outputs(identifier, loader, device, edge_cfg, edge_name):
    identifier.eval()
    pred_pi, pred_feature_pi, expected_edge, argmax_edge = [], [], [], []
    gt_pi, flags, hard, shape, occ, feat_imp = [], [], [], [], [], []
    segment_ids, region_ids, sample_indices, time_indices = [], [], [], []
    for batch in loader:
        x, _ = _to_device(batch, device)
        soft, flag, hard_lag, shape_id = _lag_targets(x, edge_name)
        segment_id, sample_index, time_index = _lag_metadata(x, edge_name)
        out = _lag_forward(identifier, x, edge_cfg, edge_name=edge_name, segment_id=segment_id)
        pred_pi.append(out["pi_edge"].cpu().numpy())
        pred_feature_pi.append(out["pi_lag"].cpu().numpy())
        expected_edge.append(out["expected_edge"].cpu().numpy())
        argmax_edge.append(out["argmax_edge"].cpu().numpy())
        gt_pi.append(soft.cpu().numpy())
        flags.append(flag.cpu().numpy())
        hard.append(hard_lag.cpu().numpy())
        shape.append(shape_id.cpu().numpy())
        occ.append(torch.sigmoid(out["occurrence_logit_edge"]).cpu().numpy())
        feat_imp.append(out["feature_importance"].cpu())
        if segment_id is not None:
            segment_ids.append(segment_id.detach().cpu().numpy())
        region_id = x.get(f"{edge_name}_region_id", x.get("region_id"))
        if region_id is not None:
            region_ids.append(region_id.detach().cpu().numpy())
        if sample_index is not None:
            sample_indices.append(sample_index.detach().cpu().numpy())
        if time_index is not None:
            time_indices.append(time_index.detach().cpu().numpy())
    shape_id = np.concatenate(shape, axis=0).astype(int)
    shape_type = np.asarray([ID_TO_SHAPE.get(int(idx), "none") for idx in shape_id])
    out = {
        "pred_pi": np.concatenate(pred_pi, axis=0),
        "pred_feature_pi": np.concatenate(pred_feature_pi, axis=0),
        "expected_edge": np.concatenate(expected_edge, axis=0),
        "argmax_lag": np.concatenate(argmax_edge, axis=0).astype(int),
        "gt_pi": np.concatenate(gt_pi, axis=0),
        "lag_flag": np.concatenate(flags, axis=0).astype(int),
        "lag_value": np.concatenate(hard, axis=0).astype(int),
        "shape_id": shape_id,
        "shape_type": shape_type,
        "occurrence_score": np.concatenate(occ, axis=0),
        "feature_importance_batches": feat_imp,
    }
    if segment_ids:
        out["segment_id"] = np.concatenate(segment_ids, axis=0).astype(int)
    if region_ids:
        out["region_id"] = np.concatenate(region_ids, axis=0).astype(int)
    if sample_indices:
        out["sample_index"] = np.concatenate(sample_indices, axis=0).astype(int)
    if time_indices:
        out["time_index"] = np.concatenate(time_indices, axis=0).astype(int)
    return out


@torch.no_grad()
def collect_lag_prior_outputs(identifier, loader, device, edge_cfg, edge_name):
    identifier.eval()
    pred_pi, pred_feature_pi, expected_edge, argmax_edge, occ = [], [], [], [], []
    segment_ids, sample_indices, time_indices = [], [], []
    for batch in loader:
        x, _ = _to_device(batch, device)
        segment_id, sample_index, time_index = _lag_metadata(x, edge_name)
        out = _lag_forward(identifier, x, edge_cfg, edge_name=edge_name, segment_id=segment_id)
        pred_pi.append(out["pi_edge"].cpu().numpy())
        pred_feature_pi.append(out["pi_lag"].cpu().numpy())
        expected_edge.append(out["expected_edge"].cpu().numpy())
        argmax_edge.append(out["argmax_edge"].cpu().numpy())
        occ.append(torch.sigmoid(out["occurrence_logit_edge"]).cpu().numpy())
        if segment_id is not None:
            segment_ids.append(segment_id.detach().cpu().numpy())
        if sample_index is not None:
            sample_indices.append(sample_index.detach().cpu().numpy())
        if time_index is not None:
            time_indices.append(time_index.detach().cpu().numpy())
    out = {
        "pred_pi": np.concatenate(pred_pi, axis=0),
        "pred_feature_pi": np.concatenate(pred_feature_pi, axis=0),
        "expected_edge": np.concatenate(expected_edge, axis=0),
        "argmax_lag": np.concatenate(argmax_edge, axis=0).astype(int),
        "occurrence_score": np.concatenate(occ, axis=0),
    }
    if segment_ids:
        out["segment_id"] = np.concatenate(segment_ids, axis=0).astype(int)
    if sample_indices:
        out["sample_index"] = np.concatenate(sample_indices, axis=0).astype(int)
    if time_indices:
        out["time_index"] = np.concatenate(time_indices, axis=0).astype(int)
    return out


def _normalize_np_dist(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    arr = np.clip(arr, 0.0, None)
    denom = np.clip(arr.sum(axis=-1, keepdims=True), 1e-12, None)
    return arr / denom


def _lag_abs_expected_error(lag_eval: Dict[str, Any]) -> np.ndarray:
    pred_pi = _normalize_np_dist(lag_eval["pred_pi"])
    gt_pi = _normalize_np_dist(lag_eval["gt_pi"])
    axis = np.arange(pred_pi.shape[-1], dtype=np.float64)
    return np.abs((pred_pi * axis[None, :]).sum(axis=-1) - (gt_pi * axis[None, :]).sum(axis=-1))


def _worst_group_mae(
    lag_eval: Dict[str, Any],
    abs_error: np.ndarray,
    group_key: str,
    mask: np.ndarray,
    min_samples: int,
) -> float:
    if group_key not in lag_eval:
        return float("nan")
    groups = np.asarray(lag_eval[group_key])
    mask = np.asarray(mask, dtype=bool) & np.isfinite(abs_error)
    values = []
    for value in pd.unique(groups[mask]):
        group_mask = mask & (groups == value)
        if int(group_mask.sum()) >= int(min_samples):
            values.append(float(np.mean(abs_error[group_mask])))
    return float(max(values)) if values else float("nan")


def _lag_checkpoint_score(
    lag_eval: Dict[str, Any],
    val_metrics: Dict[str, float],
    cfg: Dict[str, Any],
) -> tuple[float, Dict[str, float]]:
    selection_cfg = dict(cfg.get("checkpoint_selection") or {})
    mode = str(selection_cfg.get("mode", "expected_lag_mae_injected"))
    if mode != "lag_balanced_composite":
        metric = float(val_metrics.get("expected_lag_mae_injected", np.inf))
        return metric, {"score": metric, "expected_lag_mae_injected": metric}

    abs_error = _lag_abs_expected_error(lag_eval)
    lag_flag = np.asarray(lag_eval["lag_flag"]).astype(bool)
    lag_value = np.asarray(lag_eval["lag_value"]).astype(int)
    positive = lag_flag & (lag_value > 0)
    min_samples = int(selection_cfg.get("min_group_samples", 8))
    components = {
        "expected_lag_mae_injected": float(val_metrics.get("expected_lag_mae_injected", np.nan)),
        "expected_lag_mae_no_lag": float(val_metrics.get("expected_lag_mae_no_lag", np.nan)),
        "no_lag_false_alarm_rate": float(val_metrics.get("no_lag_false_alarm_rate", np.nan)),
        "worst_shape_expected_lag_mae": _worst_group_mae(lag_eval, abs_error, "shape_type", positive, min_samples),
        "worst_lag_class_mae": _worst_group_mae(lag_eval, abs_error, "lag_value", positive, min_samples),
        "worst_segment_mae": _worst_group_mae(lag_eval, abs_error, "segment_id", positive, min_samples),
        "worst_region_mae": _worst_group_mae(lag_eval, abs_error, "region_id", positive, min_samples),
    }
    weights = {
        "expected_lag_mae_injected": 1.0,
        "expected_lag_mae_no_lag": 0.5,
        "no_lag_false_alarm_rate": 0.5,
        "worst_shape_expected_lag_mae": 0.8,
        "worst_lag_class_mae": 0.5,
        "worst_segment_mae": 0.8,
        "worst_region_mae": 0.0,
    }
    weights.update({str(k): float(v) for k, v in dict(selection_cfg.get("weights") or {}).items()})
    score = 0.0
    used_weight = 0.0
    for key, weight in weights.items():
        value = float(components.get(key, np.nan))
        if weight == 0.0 or not np.isfinite(value):
            continue
        score += float(weight) * value
        used_weight += abs(float(weight))
    if used_weight <= 0.0:
        score = float(val_metrics.get("expected_lag_mae_injected", np.inf))
    components["score"] = float(score)
    return float(score), components


def make_dimf(cfg: Dict[str, Any], prepared, device):
    model_cfg = cfg.get("model", {})
    delay_cfg = cfg.get("delay_prior", {})
    return DIMF(
        group_dims=prepared.group_dims,
        hidden_dim=int(model_cfg.get("hidden_dim", 64)),
        num_layers=int(model_cfg.get("num_layers", 2)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        attn_dim=int(model_cfg.get("attn_dim", model_cfg.get("hidden_dim", 64))),
        L_max=int(cfg["data"].get("L_max", cfg.get("lag_injection", {}).get("max_lag", 12))),
        lead_steps=int(cfg["data"]["H"]),
        encoder_type=str(model_cfg.get("encoder", "gru")),
        transformer_nhead=int(model_cfg.get("transformer_nhead", 4)),
        transformer_ff_dim=model_cfg.get("transformer_ff_dim"),
        max_len=_history_steps(cfg["data"]),
        lag_emb=bool(model_cfg.get("lag_emb", True)),
        use_alignment=bool(model_cfg.get("use_alignment", True)),
        align_tau=float(model_cfg.get("align_tau", 1.0)),
        align_dropout=float(model_cfg.get("align_dropout", 0.0)),
        align_feed_to_stage1=model_cfg.get("align_feed_to_stage1"),
        align_stage1_to_stage2=model_cfg.get("align_stage1_to_stage2"),
        align_stage2_to_stage3=model_cfg.get("align_stage2_to_stage3"),
        use_lag_bias=bool(model_cfg.get("use_lag_bias", True)),
        head_stage=str(model_cfg.get("head_stage", "stage3")),
        use_feed_context=bool(model_cfg.get("use_feed_context", True)),
        lag_head_mode=str(model_cfg.get("lag_head_mode", "softmax")),
        output_dim=len(prepared.target_cols or [cfg["data"]["target_col"]]),
        delay_prior_lambda=float(delay_cfg.get("lambda_prior", 1.0)),
        delay_prior_mode=str(delay_cfg.get("prior_mode", "soft_distribution")),
        delay_prior_sigma=float(delay_cfg.get("sigma_prior", 1.5)),
    ).to(device)


def _attach_lag_guided_alignment_to_dimf(dimf, identifier, edge_cfg, edge_name, cfg, feature_mask=None):
    delay_cfg = cfg.get("delay_prior", {})
    dimf.attach_lag_guided_prior_generator(
        identifier,
        edge_name=edge_name,
        source_stage=edge_cfg["source_stage"],
        target_stage=edge_cfg["target_stage"],
        feature_mask=feature_mask,
        lambda_prior=float(delay_cfg.get("lambda_prior", 1.0)),
        prior_mode=str(delay_cfg.get("prior_mode", "soft_distribution")),
        sigma_prior=float(delay_cfg.get("sigma_prior", 1.5)),
        weak_prior_mix=float(delay_cfg.get("weak_prior_mix", 0.0)),
    )
    return dimf


def _load_dimf_checkpoint_with_lag_head(dimf, identifier, checkpoint_path: Path, edge_cfg, edge_name, cfg, feature_mask=None):
    state = torch.load(checkpoint_path, map_location=next(dimf.parameters()).device)
    has_lag_guided_prior_state = any(str(key).startswith("lag_identifier.") for key in state.keys())
    if has_lag_guided_prior_state:
        _attach_lag_guided_alignment_to_dimf(dimf, identifier, edge_cfg, edge_name, cfg, feature_mask=feature_mask)
        dimf.load_state_dict(state)
    else:
        dimf.load_state_dict(state)
        _attach_lag_guided_alignment_to_dimf(dimf, identifier, edge_cfg, edge_name, cfg, feature_mask=feature_mask)
    return dimf


def _dimf_prior(model_or_identifier, x, edge_cfg, edge_name, delay_cfg, feature_mask, segment_id=None):
    if hasattr(model_or_identifier, "has_lag_guided_alignment") and model_or_identifier.has_lag_guided_alignment():
        return model_or_identifier.infer_lag_guided_delay_priors(x, segment_id=segment_id)
    out = _lag_forward(model_or_identifier, x, edge_cfg, edge_name=edge_name, segment_id=segment_id)
    pi_prior = out["pi_lag"]
    pi_prior = apply_feature_screening_to_prior(
        pi_prior,
        feature_mask.to(pi_prior.device) if feature_mask is not None else None,
        weak_prior_mix=float(delay_cfg.get("weak_prior_mix", 0.0)),
    )
    return out, {
        edge_name: {
            "pi_prior": pi_prior,
            "lambda_prior": float(delay_cfg.get("lambda_prior", 1.0)),
            "prior_mode": str(delay_cfg.get("prior_mode", "soft_distribution")),
            "sigma_prior": float(delay_cfg.get("sigma_prior", 1.5)),
        }
    }


def train_dimf_epoch(dimf, identifier, loader, optimizer, device, edge_cfg, edge_name, cfg, feature_mask, joint: bool) -> Dict[str, float]:
    dimf.train()
    identifier.train(mode=joint)
    delay_cfg = cfg.get("delay_prior", {})
    weights = _loss_weights(cfg, "joint" if joint else "stage2")
    small_lambda_lag = float(cfg.get("loss", {}).get("small_lambda_lag", 0.1))
    total_pred = total_lag = total = 0.0
    n = 0
    for batch in loader:
        x, y = _to_device(batch, device)
        soft, flag, hard, shape = _lag_targets(x, edge_name)
        segment_id, _, time_index = _lag_metadata(x, edge_name)
        if joint:
            lag_out, priors = _dimf_prior(dimf, x, edge_cfg, edge_name, delay_cfg, feature_mask, segment_id=segment_id)
            lag_loss = lag_identifier_loss(
                lag_out,
                soft,
                flag,
                shape,
                weights,
                segment_id=segment_id,
                lag_gt=hard,
                sample_index=time_index,
            )["loss"]
        else:
            with torch.no_grad():
                lag_out, priors = _dimf_prior(dimf, x, edge_cfg, edge_name, delay_cfg, feature_mask, segment_id=segment_id)
                lag_loss = lag_identifier_loss(
                    lag_out,
                    soft,
                    flag,
                    shape,
                    weights,
                    segment_id=segment_id,
                    lag_gt=hard,
                    sample_index=time_index,
                )["loss"]
        y_hat, _ = dimf(x, delay_priors=priors)
        pred_loss = (y_hat - y).abs().mean()
        loss = pred_loss + (small_lambda_lag if not joint else 1.0) * lag_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_n = int(y.shape[0])
        n += batch_n
        total += float(loss.detach().item()) * batch_n
        total_pred += float(pred_loss.detach().item()) * batch_n
        total_lag += float(lag_loss.detach().item()) * batch_n
    return {"loss": total / max(n, 1), "pred": total_pred / max(n, 1), "lag": total_lag / max(n, 1)}


def train_dimf_y_only_epoch(
    dimf,
    loader,
    optimizer,
    device,
    edge_cfg,
    edge_name,
    cfg,
    feature_mask,
    use_lag_prior: bool = True,
) -> Dict[str, float]:
    dimf.train()
    if use_lag_prior and getattr(dimf, "lag_identifier", None) is not None:
        dimf.lag_identifier.eval()
    delay_cfg = cfg.get("delay_prior", {})
    total_pred = 0.0
    n = 0
    for batch in loader:
        x, y = _to_device(batch, device)
        priors = None
        if use_lag_prior:
            segment_id, _, _ = _lag_metadata(x, edge_name)
            with torch.no_grad():
                _, priors = _dimf_prior(dimf, x, edge_cfg, edge_name, delay_cfg, feature_mask, segment_id=segment_id)
        y_hat, _ = dimf(x, delay_priors=priors)
        pred_loss = (y_hat - y).abs().mean()
        optimizer.zero_grad()
        pred_loss.backward()
        optimizer.step()
        batch_n = int(y.shape[0])
        n += batch_n
        total_pred += float(pred_loss.detach().item()) * batch_n
    pred = total_pred / max(n, 1)
    return {"loss": pred, "pred": pred}


@torch.no_grad()
def evaluate_prediction(
    dimf,
    identifier,
    loader,
    device,
    edge_cfg,
    edge_name,
    cfg,
    feature_mask,
    scaler_y,
    use_lag_prior: bool = True,
):
    dimf.eval()
    if identifier is not None:
        identifier.eval()
    y_true, y_pred = [], []
    delay_cfg = cfg.get("delay_prior", {})
    for batch in loader:
        x, y = _to_device(batch, device)
        priors = None
        if use_lag_prior:
            segment_id, _, _ = _lag_metadata(x, edge_name)
            _, priors = _dimf_prior(dimf, x, edge_cfg, edge_name, delay_cfg, feature_mask, segment_id=segment_id)
        y_hat, _ = dimf(x, delay_priors=priors)
        y_true.append(y.cpu().numpy())
        y_pred.append(y_hat.cpu().numpy())
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    y_true_2d = y_true[:, None] if y_true.ndim == 1 else y_true
    y_pred_2d = y_pred[:, None] if y_pred.ndim == 1 else y_pred
    y_true_inv = scaler_y.inverse_transform(y_true_2d)
    y_pred_inv = scaler_y.inverse_transform(y_pred_2d)
    return y_true_inv, y_pred_inv, prediction_metrics(y_true_inv, y_pred_inv)


def _safe_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.mean(values)) if values.size else float("nan")


def _viterbi_cfg(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cfg = cfg or {}
    viterbi_cfg = dict(cfg.get("viterbi", {}))
    if "use_viterbi_decode" in cfg and "use_viterbi_decode" not in viterbi_cfg:
        viterbi_cfg["use_viterbi_decode"] = cfg["use_viterbi_decode"]
    viterbi_cfg.setdefault("use_viterbi_decode", viterbi_cfg.get("enabled", False))
    viterbi_cfg.setdefault("viterbi_smooth_lambda", viterbi_cfg.get("smooth_lambda", viterbi_cfg.get("transition_penalty", 0.8)))
    viterbi_cfg.setdefault("viterbi_switch_penalty", viterbi_cfg.get("switch_penalty", 1.5))
    viterbi_cfg.setdefault("viterbi_pos_to_zero_penalty", viterbi_cfg.get("pos_to_zero_penalty", 2.0))
    return viterbi_cfg


def _temporal_order(lag_eval: Dict[str, Any], segment_ids: Optional[np.ndarray]) -> np.ndarray:
    n = int(np.asarray(lag_eval["pred_pi"]).shape[0])
    if "time_index" in lag_eval:
        time_index = np.asarray(lag_eval["time_index"])
    elif "sample_index" in lag_eval:
        time_index = np.asarray(lag_eval["sample_index"])
    else:
        time_index = np.arange(n)
    if segment_ids is None and "segment_id" in lag_eval:
        segment_ids = np.asarray(lag_eval["segment_id"])
    if segment_ids is None:
        return np.argsort(time_index, kind="stable")
    return np.lexsort((time_index, np.asarray(segment_ids)))


def _decode_viterbi_for_eval(
    lag_eval: Dict[str, Any],
    cfg: Optional[Dict[str, Any]],
    segment_ids: Optional[np.ndarray] = None,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    viterbi_cfg = _viterbi_cfg(cfg)
    if not bool(viterbi_cfg.get("use_viterbi_decode", False)):
        return None, None, viterbi_cfg
    if segment_ids is None and "segment_id" in lag_eval:
        segment_ids = np.asarray(lag_eval["segment_id"])
    order = _temporal_order(lag_eval, segment_ids)
    sorted_segments = None if segment_ids is None else np.asarray(segment_ids)[order]
    sorted_path = viterbi_decode_lag(
        np.asarray(lag_eval["pred_pi"])[order],
        segment_id=sorted_segments,
        smooth_lambda=float(viterbi_cfg.get("viterbi_smooth_lambda", 0.8)),
        switch_penalty=float(viterbi_cfg.get("viterbi_switch_penalty", 1.5)),
        pos_to_zero_penalty=float(viterbi_cfg.get("viterbi_pos_to_zero_penalty", 2.0)),
    )
    path = np.empty_like(sorted_path)
    path[order] = sorted_path
    return path, order, viterbi_cfg


def _lag_postprocess_cfg(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cfg = cfg or {}
    if "lag_postprocess" in cfg:
        return dict(cfg.get("lag_postprocess") or {})
    post = cfg.get("postprocess", {})
    if isinstance(post, dict):
        return dict(post.get("lag") or {})
    return {}


def postprocess_lag_eval_for_eval(
    lag_eval: Dict[str, Any],
    cfg: Optional[Dict[str, Any]],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    pp_cfg = _lag_postprocess_cfg(cfg)
    gate_cfg = dict(pp_cfg.get("segment_zero_gate") or {})
    if not bool(gate_cfg.get("enabled", False)):
        return lag_eval, {"segment_zero_gate_enabled": False}
    if "segment_id" not in lag_eval:
        return lag_eval, {"segment_zero_gate_enabled": False, "segment_zero_gate_reason": "missing_segment_id"}

    pi = np.asarray(lag_eval["pred_pi"], dtype=np.float64).copy()
    occurrence = np.asarray(lag_eval.get("occurrence_score", 1.0 - pi[:, 0]), dtype=np.float64).copy()
    segment_ids = np.asarray(lag_eval["segment_id"])
    threshold = float(gate_cfg.get("occurrence_median_threshold", 0.1))
    min_samples = int(gate_cfg.get("min_segment_samples", 1))

    zeroed_segments = []
    for segment in np.unique(segment_ids):
        mask = segment_ids == segment
        if int(mask.sum()) < min_samples:
            continue
        median_occ = float(np.median(occurrence[mask]))
        if median_occ <= threshold:
            pi[mask, :] = 0.0
            pi[mask, 0] = 1.0
            occurrence[mask] = 0.0
            zeroed_segments.append(
                {
                    "segment_id": int(segment),
                    "n_samples": int(mask.sum()),
                    "median_occurrence": median_occ,
                }
            )

    info = {
        "segment_zero_gate_enabled": True,
        "segment_zero_gate_threshold": threshold,
        "segment_zero_gate_min_samples": min_samples,
        "segment_zero_gate_zeroed_segments": zeroed_segments,
    }
    if not zeroed_segments:
        return lag_eval, info

    out = dict(lag_eval)
    out["pred_pi"] = pi.astype(np.float32)
    out["occurrence_score"] = occurrence.astype(np.float32)
    lag_axis = np.arange(pi.shape[1], dtype=np.float64)
    out["expected_edge"] = (pi * lag_axis[None, :]).sum(axis=1).astype(np.float32)
    out["argmax_lag"] = pi.argmax(axis=1).astype(int)
    if "pred_feature_pi" in out:
        feature_pi = np.asarray(out["pred_feature_pi"]).copy()
        for item in zeroed_segments:
            mask = segment_ids == int(item["segment_id"])
            feature_pi[mask, :, :] = 0.0
            feature_pi[mask, :, 0] = 1.0
        out["pred_feature_pi"] = feature_pi
    return out, info


def _lag_eval_metrics(
    lag_eval: Dict[str, Any],
    viterbi_path: Optional[np.ndarray] = None,
    false_alarm_threshold: float = 0.5,
) -> Dict[str, float]:
    pi = np.asarray(lag_eval["pred_pi"], dtype=np.float64)
    lag_axis = np.arange(pi.shape[1], dtype=np.float64)
    raw_expected = (pi * lag_axis[None, :]).sum(axis=1)
    raw_argmax = pi.argmax(axis=1)
    lag_gt = np.asarray(lag_eval["lag_value"], dtype=np.int64)
    lag_flag = np.asarray(lag_eval["lag_flag"]).astype(bool)
    no_lag = ~lag_flag
    occurrence_score = np.asarray(lag_eval.get("occurrence_score", 1.0 - pi[:, 0]), dtype=np.float64)
    metrics = {
        "raw_argmax_lag_accuracy": _safe_mean((raw_argmax == lag_gt).astype(float)),
        "raw_expected_lag_mae_all": _safe_mean(np.abs(raw_expected - lag_gt)),
        "raw_expected_lag_mae_injected": _safe_mean(np.abs(raw_expected[lag_flag] - lag_gt[lag_flag])),
        "raw_expected_lag_mae_no_lag": _safe_mean(np.abs(raw_expected[no_lag] - lag_gt[no_lag])),
        "raw_no_lag_false_alarm_rate": _safe_mean((occurrence_score[no_lag] > float(false_alarm_threshold)).astype(float)),
    }
    if viterbi_path is not None:
        path = np.asarray(viterbi_path, dtype=np.int64)
        metrics.update(
            {
                "viterbi_lag_accuracy_all": _safe_mean((path == lag_gt).astype(float)),
                "viterbi_lag_accuracy_injected": _safe_mean((path[lag_flag] == lag_gt[lag_flag]).astype(float)),
                "viterbi_lag_mae_all": _safe_mean(np.abs(path - lag_gt)),
                "viterbi_lag_mae_injected": _safe_mean(np.abs(path[lag_flag] - lag_gt[lag_flag])),
                "viterbi_lag_mae_no_lag": _safe_mean(np.abs(path[no_lag] - lag_gt[no_lag])),
                "viterbi_no_lag_false_alarm_rate": _safe_mean((path[no_lag] > 0).astype(float)),
            }
        )
    return metrics


def _settings_suffix(cfg: Optional[Dict[str, Any]], viterbi_enabled: bool) -> str:
    cfg = cfg or {}
    lag_cfg = cfg.get("lag_identifier", {})
    return (
        f"seq{_history_steps(cfg.get('data', {}))}"
        f"_cw{int(bool(lag_cfg.get('use_candidate_window_encoder', False)))}"
        f"_r{int(lag_cfg.get('lag_window_radius', 2))}"
        f"_gauss{int(bool(cfg.get('loss', {}).get('use_gaussian_lag_label', True)))}"
        f"_vit{int(bool(viterbi_enabled))}"
    )


def _lag_prediction_frame(
    lag_eval: Dict[str, Any],
    segment_ids: Optional[np.ndarray] = None,
    viterbi_path: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    pi = np.asarray(lag_eval["pred_pi"])
    lag_axis = np.arange(pi.shape[1], dtype=np.float64)
    expected_edge = lag_eval.get("expected_edge", (pi * lag_axis[None, :]).sum(axis=1))
    raw_argmax = lag_eval.get("argmax_lag", pi.argmax(axis=1))
    true_expected = (np.asarray(lag_eval["gt_pi"]) * lag_axis[None, :]).sum(axis=1)
    frame = pd.DataFrame(
        {
            "sample": np.arange(pi.shape[0]),
            "true_expected_lag": true_expected,
            "raw_expected": expected_edge,
            "pred_expected_lag": expected_edge,
            "expected_edge": expected_edge,
            "raw_argmax": raw_argmax,
            "argmax_lag": raw_argmax,
            "lag_gt": lag_eval["lag_value"],
            "lag_value": lag_eval["lag_value"],
            "lag_flag": lag_eval["lag_flag"],
            "shape_type": lag_eval["shape_type"],
            "occurrence_score": lag_eval["occurrence_score"],
        }
    )
    if "sample_index" in lag_eval:
        frame["sample_index"] = np.asarray(lag_eval["sample_index"])
    if "time_index" in lag_eval:
        frame["time_index"] = np.asarray(lag_eval["time_index"])
    if "segment_id" in lag_eval:
        frame["segment_id"] = np.asarray(lag_eval["segment_id"])
    elif segment_ids is not None:
        frame["segment_id"] = np.asarray(segment_ids)
    if "region_id" in lag_eval:
        frame["region_id"] = np.asarray(lag_eval["region_id"])
    if viterbi_path is not None:
        frame["viterbi_path"] = np.asarray(viterbi_path)
    for lag in range(pi.shape[1]):
        frame[f"pi_{lag}"] = pi[:, lag]
        frame[f"pi_edge_lag{lag}"] = pi[:, lag]
    return frame


def save_outputs(
    results_dir: Path,
    lag_eval: Dict[str, Any],
    y_true,
    y_pred,
    feature_mask,
    cfg: Optional[Dict[str, Any]] = None,
    segment_ids: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    if "segment_id" in lag_eval:
        segment_ids = np.asarray(lag_eval["segment_id"])
    lag_eval, postprocess_info = postprocess_lag_eval_for_eval(lag_eval, cfg)
    if "segment_id" in lag_eval:
        segment_ids = np.asarray(lag_eval["segment_id"])
    paths = save_lag_metric_tables(
        results_dir,
        lag_eval["pred_pi"],
        lag_eval["gt_pi"],
        lag_eval["lag_flag"],
        lag_eval["lag_value"],
        lag_eval["shape_type"],
        occurrence_score=lag_eval["occurrence_score"],
        y_true=y_true,
        y_pred=y_pred,
    )
    figs_dir = results_dir / "figs"
    save_lag_distribution_heatmap(figs_dir, lag_eval["gt_pi"], lag_eval["pred_pi"])
    save_expected_lag_curve(figs_dir, lag_eval["gt_pi"], lag_eval["pred_pi"])
    by_shape = pd.read_csv(paths["by_shape"])
    save_by_shape_bar_chart(figs_dir, by_shape)
    save_no_lag_false_alarm_plot(figs_dir, lag_eval["pred_pi"], lag_eval["lag_flag"])
    selected = torch.nonzero(feature_mask, as_tuple=False).view(-1).cpu().numpy() if feature_mask is not None else np.arange(min(8, lag_eval["pred_feature_pi"].shape[1]))
    save_selected_feature_lag_heatmap(figs_dir, lag_eval["pred_feature_pi"], selected)
    lag_metrics = pd.read_csv(paths["overall"]).iloc[0].to_dict()
    pred_metrics = pd.read_csv(paths["prediction"]).iloc[0].to_dict()
    full_metrics = {**pred_metrics, **lag_metrics}

    viterbi_path, viterbi_order, viterbi_cfg = _decode_viterbi_for_eval(lag_eval, cfg, segment_ids=segment_ids)
    full_metrics.update(_lag_eval_metrics(lag_eval, viterbi_path=viterbi_path))

    prediction_frame = _lag_prediction_frame(lag_eval, segment_ids=segment_ids, viterbi_path=viterbi_path)
    prediction_frame.to_csv(results_dir / "test_lag_predictions.csv", index=False)
    prediction_frame.to_csv(results_dir / "lag_eval_predictions.csv", index=False)

    if viterbi_path is not None:
        viterbi_pi = path_to_onehot(viterbi_path, lag_eval["pred_pi"].shape[1])
        viterbi_tmp_dir = results_dir / "_viterbi_tmp"
        viterbi_paths = save_lag_metric_tables(
            viterbi_tmp_dir,
            viterbi_pi,
            lag_eval["gt_pi"],
            lag_eval["lag_flag"],
            lag_eval["lag_value"],
            lag_eval["shape_type"],
            occurrence_score=(viterbi_path > 0).astype(float),
            y_true=None,
            y_pred=None,
        )
        rename_map = {
            "overall": "viterbi_lag_metrics_overall.csv",
            "by_shape": "viterbi_lag_metrics_by_shape.csv",
            "by_lag": "viterbi_lag_metrics_by_lag.csv",
        }
        for key, filename in rename_map.items():
            src = viterbi_paths[key]
            dst = results_dir / filename
            src.replace(dst)
            viterbi_paths[key] = dst
        viterbi_paths["prediction"].unlink(missing_ok=True)
        try:
            viterbi_tmp_dir.rmdir()
        except OSError:
            pass

        lag_axis = np.arange(lag_eval["pred_pi"].shape[1], dtype=np.float64)
        viterbi_curve = pd.DataFrame(
            {
                "sample": np.arange(viterbi_path.shape[0]),
                "true_expected_lag": (lag_eval["gt_pi"] * lag_axis[None, :]).sum(axis=1),
                "pred_expected_lag": (lag_eval["pred_pi"] * lag_axis[None, :]).sum(axis=1),
                "viterbi_lag": viterbi_path,
                "lag_flag": lag_eval["lag_flag"],
                "lag_value": lag_eval["lag_value"],
                "shape_type": lag_eval["shape_type"],
            }
        )
        if segment_ids is not None:
            viterbi_curve["segment_id"] = np.asarray(segment_ids)
        viterbi_curve.to_csv(results_dir / "viterbi_lag_curve.csv", index=False)
        suffix = _settings_suffix(cfg, viterbi_enabled=True)
        if viterbi_order is None:
            viterbi_order = np.arange(viterbi_path.shape[0])
        save_viterbi_lag_curve(
            figs_dir,
            lag_eval["gt_pi"][viterbi_order],
            lag_eval["pred_pi"][viterbi_order],
            viterbi_path[viterbi_order],
            filename=f"viterbi_lag_curve_{suffix}.png",
        )
        viterbi_metrics = pd.read_csv(viterbi_paths["overall"]).iloc[0].to_dict()
        for key, value in viterbi_metrics.items():
            full_metrics[f"viterbi_distribution_{key}"] = value
        full_metrics["use_viterbi_decode"] = True
        full_metrics["viterbi_smooth_lambda"] = float(viterbi_cfg.get("viterbi_smooth_lambda", 0.8))
        full_metrics["viterbi_switch_penalty"] = float(viterbi_cfg.get("viterbi_switch_penalty", 1.5))
        full_metrics["viterbi_pos_to_zero_penalty"] = float(viterbi_cfg.get("viterbi_pos_to_zero_penalty", 2.0))
    else:
        full_metrics["use_viterbi_decode"] = False

    full_metrics["lag_postprocess"] = json.dumps(postprocess_info)
    pd.DataFrame([full_metrics]).to_csv(results_dir / "lag_eval_metrics.csv", index=False)

    return full_metrics


def save_prediction_only_outputs(
    results_dir: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_indices: Optional[np.ndarray],
    pred_metrics: Dict[str, float],
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([pred_metrics]).to_csv(results_dir / "prediction_metrics.csv", index=False)

    y_true_2d = np.asarray(y_true)
    y_pred_2d = np.asarray(y_pred)
    if y_true_2d.ndim == 1:
        y_true_2d = y_true_2d[:, None]
    if y_pred_2d.ndim == 1:
        y_pred_2d = y_pred_2d[:, None]
    n = int(y_true_2d.shape[0])
    if sample_indices is None:
        sample_indices = np.arange(n, dtype=np.int64)
    sample_indices = np.asarray(sample_indices, dtype=np.int64)

    pd.DataFrame(
        {
            "sample": np.arange(n, dtype=np.int64),
            "sample_index": sample_indices,
            "y_true": y_true_2d[:, 0],
            "y_pred": y_pred_2d[:, 0],
            "error": y_pred_2d[:, 0] - y_true_2d[:, 0],
            "abs_error": np.abs(y_pred_2d[:, 0] - y_true_2d[:, 0]),
        }
    ).to_csv(results_dir / "y_target_predictions.csv", index=False)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs_dir = results_dir / "figs"
    figs_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 3.8))
    ax.plot(sample_indices, y_true_2d[:, 0], label="gt", linewidth=1.4)
    ax.plot(sample_indices, y_pred_2d[:, 0], label="pred", linewidth=1.2)
    ax.set_title("Y target GT vs Pred")
    ax.set_xlabel("sample")
    ax.set_ylabel("yield_flow")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figs_dir / "y_target_gt_pred.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true_2d[:, 0], y_pred_2d[:, 0], s=5, alpha=0.45)
    lo = float(min(y_true_2d[:, 0].min(), y_pred_2d[:, 0].min()))
    hi = float(max(y_true_2d[:, 0].max(), y_pred_2d[:, 0].max()))
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.0)
    ax.set_title("Y target scatter")
    ax.set_xlabel("gt")
    ax.set_ylabel("pred")
    fig.tight_layout()
    fig.savefig(figs_dir / "y_target_scatter.png", dpi=160)
    plt.close(fig)


def save_lag_prior_outputs(results_dir: Path, lag_eval: Dict[str, Any]) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    pi = np.asarray(lag_eval["pred_pi"])
    frame = pd.DataFrame(
        {
            "sample": np.arange(pi.shape[0], dtype=np.int64),
            "pred_expected_lag": np.asarray(lag_eval["expected_edge"]),
            "pred_argmax_lag": np.asarray(lag_eval["argmax_lag"]),
            "occurrence_score": np.asarray(lag_eval["occurrence_score"]),
        }
    )
    if "sample_index" in lag_eval:
        frame["sample_index"] = np.asarray(lag_eval["sample_index"])
    if "time_index" in lag_eval:
        frame["time_index"] = np.asarray(lag_eval["time_index"])
    if "segment_id" in lag_eval:
        frame["segment_id"] = np.asarray(lag_eval["segment_id"])
    for lag in range(pi.shape[1]):
        frame[f"pi_edge_lag{lag}"] = pi[:, lag]
    frame.to_csv(results_dir / "raw_lag_prior_predictions.csv", index=False)


def write_ablation_table(results_dir: Path, full_metrics: Dict[str, float], current_variant: str) -> None:
    variants = [
        "A_original_dimf",
        "B_dimf_hard_lag_supervision",
        "C_dimf_soft_lag_supervision",
        "D_dimf_lag_guided_prior_generator",
        "E_dimf_lag_guided_delay_alignment",
        "F_dimf_lag_guided_alignment_feature_screening",
    ]
    metric_names = [
        "prediction_mae",
        "prediction_rmse",
        "expected_lag_mae_injected",
        "soft_js",
        "occurrence_auprc",
        "no_lag_false_alarm_rate",
    ]
    table_path = results_dir / "ablation_lag_grounded_dimf.csv"
    if table_path.exists():
        frame = pd.read_csv(table_path)
        rows = frame.to_dict("records")
    else:
        rows = []
        for variant in variants:
            row = {"variant": variant}
            for metric in metric_names:
                row[metric] = np.nan
            row["status"] = "not_run_in_this_invocation"
            rows.append(row)
    row_by_variant = {row["variant"]: row for row in rows}
    if current_variant not in row_by_variant:
        row_by_variant[current_variant] = {"variant": current_variant}
    for metric in metric_names:
        row_by_variant[current_variant][metric] = full_metrics.get(metric, np.nan)
    row_by_variant[current_variant]["status"] = "current_run"
    ordered = [row_by_variant[variant] for variant in variants if variant in row_by_variant]
    ordered.extend(row for variant, row in row_by_variant.items() if variant not in variants)
    pd.DataFrame(ordered).to_csv(table_path, index=False)


def _feature_mask_from_report(checkpoint_dir: Path, n_features: int) -> Optional[torch.Tensor]:
    report_path = checkpoint_dir / "feature_screening_report.json"
    if not report_path.exists():
        return None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    selected = report.get("selected_indices")
    if selected is None:
        return None
    mask = torch.zeros(n_features, dtype=torch.bool)
    for idx in selected:
        idx = int(idx)
        if 0 <= idx < n_features:
            mask[idx] = True
    return mask


def run_raw_y_only_adaptation(
    cfg: Dict[str, Any],
    prepared,
    dl_tr,
    dl_va,
    dl_te,
    dimf,
    identifier,
    device,
    edge_cfg,
    edge_name,
    results_dir: Path,
    feature_mask: Optional[torch.Tensor],
    checkpoint_summary: Dict[str, Dict[str, Any]],
) -> None:
    raw_cfg = dict(cfg.get("raw_adaptation") or {})
    prior_source = str(raw_cfg.get("prior_source", "pretrained")).lower()
    if prior_source not in {"none", "pretrained", "random"}:
        raise ValueError("raw_adaptation.prior_source must be one of: none, pretrained, random")
    use_lag_prior = prior_source != "none" and bool(cfg.get("delay_prior", {}).get("enabled", True))
    if use_lag_prior:
        for param in identifier.parameters():
            param.requires_grad = False
        _attach_lag_guided_alignment_to_dimf(dimf, identifier, edge_cfg, edge_name, cfg, feature_mask=feature_mask)

    training_cfg = cfg.get("training", {})
    trainable_params = [param for param in dimf.parameters() if param.requires_grad]
    if not trainable_params:
        raise ValueError("Raw adaptation has no trainable DIMF parameters")
    opt_dimf = torch.optim.Adam(trainable_params, lr=float(training_cfg.get("lr_dimf", 1e-3)))

    best_pred = float("inf")
    best_pred_epoch = 0
    epochs = int(training_cfg.get("epochs_dimf", 100))
    for epoch in range(1, epochs + 1):
        train_log = train_dimf_y_only_epoch(
            dimf,
            dl_tr,
            opt_dimf,
            device,
            edge_cfg,
            edge_name,
            cfg,
            feature_mask,
            use_lag_prior=use_lag_prior,
        )
        if dl_va is not None:
            _, _, pred_m = evaluate_prediction(
                dimf,
                identifier,
                dl_va,
                device,
                edge_cfg,
                edge_name,
                cfg,
                feature_mask,
                prepared.scaler_y,
                use_lag_prior=use_lag_prior,
            )
            metric = float(pred_m["prediction_rmse"])
            metric_name = "val_prediction_rmse"
            print(f"[raw adapt epoch {epoch}] loss={train_log['loss']:.5f} val_rmse={metric:.5f}")
        else:
            metric = float(train_log["pred"])
            metric_name = "train_prediction_mae"
            print(f"[raw adapt epoch {epoch}] loss={train_log['loss']:.5f}")
        if metric < best_pred:
            best_pred = metric
            best_pred_epoch = epoch
            torch.save(dimf.state_dict(), results_dir / "best_raw_adapt_dimf.pt")

    checkpoint_summary["raw_adaptation"] = {
        "selection_metric": metric_name,
        "best_epoch": int(best_pred_epoch),
        "best_metric": float(best_pred),
    }
    dimf.load_state_dict(torch.load(results_dir / "best_raw_adapt_dimf.pt", map_location=device))
    torch.save({"dimf": dimf.state_dict(), "lag_identifier": identifier.state_dict()}, results_dir / "best_raw_adapt_stage2_only.pt")

    y_true, y_pred, pred_m = evaluate_prediction(
        dimf,
        identifier,
        dl_te,
        device,
        edge_cfg,
        edge_name,
        cfg,
        feature_mask,
        prepared.scaler_y,
        use_lag_prior=use_lag_prior,
    )
    save_prediction_only_outputs(
        results_dir,
        y_true,
        y_pred,
        getattr(prepared, "sample_indices_test", None),
        pred_m,
    )
    lag_prior_output = None
    if use_lag_prior:
        lag_prior_eval = collect_lag_prior_outputs(dimf, dl_te, device, edge_cfg, edge_name)
        save_lag_prior_outputs(results_dir, lag_prior_eval)
        lag_prior_output = "raw_lag_prior_predictions.csv"
    (results_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "stage": "raw_y_only_adaptation",
                "prior_source": prior_source,
                "prediction": pred_m,
                "checkpoint_selection": checkpoint_summary,
                "lag_prior": {
                    "enabled": bool(use_lag_prior),
                    "has_ground_truth": False,
                    "output": lag_prior_output,
                },
                "gap_policy": {
                    "gap_break_min": int(cfg["data"].get("gap_break_min", cfg["data"].get("collection_interval_min", 15))),
                    "gap_fill_min": int(cfg["data"].get("gap_fill_min", 0)),
                    "use_delta_t": bool(cfg["data"].get("use_delta_t", False)),
                    "use_missing_mask": bool(cfg["data"].get("use_missing_mask", False)),
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved raw adaptation outputs under: {results_dir}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config.resolve())
    raw_adaptation = bool(
        args.raw_adapt
        or cfg.get("raw_adaptation", {}).get("enabled", False)
        or cfg.get("training", {}).get("raw_adaptation", False)
    )
    set_seed(int(cfg.get("seed", cfg.get("lag_injection", {}).get("random_seed", 42))))
    results_dir = Path(cfg.get("logging", {}).get("output_dir", "results/lag_grounded")).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    if raw_adaptation:
        prepared = _load_raw_prepared(cfg)
        if args.summary_only:
            print("Using raw CSV with gap-aware contiguous-window sampling; no lag injection was run.")
            print(f"train samples: {len(prepared.sample_indices_train)}")
            print(f"val samples: {0 if prepared.sample_indices_val is None else len(prepared.sample_indices_val)}")
            print(f"test samples: {len(prepared.sample_indices_test)}")
            return
    elif "train_csv" in cfg["data"] and "test_csv" in cfg["data"]:
        prepared = _load_existing_train_test_prepared(cfg)
        if args.summary_only:
            print("Using existing train/test CSV files; no new lag injection was run.")
            print(f"train samples: {len(prepared.sample_indices_train)}")
            print(f"test samples: {len(prepared.sample_indices_test)}")
            return
    else:
        injected_csv = _prepare_injected_dataset(cfg, results_dir)
        if args.summary_only:
            print(f"Saved lag-injected CSV: {injected_csv}")
            print(f"Saved summary: {results_dir / 'lag_injection_summary.json'}")
            return
        prepared = _load_prepared(cfg, injected_csv)
    joblib.dump({"scaler_x": prepared.scaler_x, "scaler_y": prepared.scaler_y}, results_dir / "scaler.pkl")
    ds_tr, ds_va, ds_te, dl_tr, dl_va, dl_te = _datasets_and_loaders(cfg, prepared)
    batch_size = int(cfg.get("training", {}).get("batch_size", 64))

    device = torch.device(args.device)
    edge_cfg = _edge_cfg(cfg)
    edge_name = str(edge_cfg.get("name", "stage1_to_stage2"))
    lag_cfg = cfg.get("lag_identifier", {})
    training_cfg = cfg.get("training", {})
    lag_identifier_shuffle = bool(training_cfg.get("lag_identifier_shuffle", False))
    dl_tr_temporal, lag_sampler_info = _make_lag_train_loader(cfg, ds_tr, batch_size, edge_name)
    del ds_tr, ds_va, ds_te
    loss_weights_stage1 = _loss_weights(cfg, "stage1")
    seq_len = _history_steps(cfg["data"])
    viterbi_cfg = _viterbi_cfg(cfg)
    print(
        "Lag identifier settings: "
        + json.dumps(
            {
                "use_candidate_window_encoder": bool(lag_cfg.get("use_candidate_window_encoder", False)),
                "lag_window_radius": int(lag_cfg.get("lag_window_radius", 2)),
                "seq_len": int(seq_len),
                "use_gaussian_lag_label": bool(loss_weights_stage1.use_gaussian_lag_label),
                "gaussian_lag_sigma": float(loss_weights_stage1.gaussian_lag_sigma),
                "enable_segment_aware_temporal_loss": bool(loss_weights_stage1.enable_segment_aware_temporal_loss),
                "use_viterbi_decode": bool(viterbi_cfg.get("use_viterbi_decode", False)),
                "viterbi_smooth_lambda": float(viterbi_cfg.get("viterbi_smooth_lambda", 0.8)),
                "viterbi_switch_penalty": float(viterbi_cfg.get("viterbi_switch_penalty", 1.5)),
                "viterbi_pos_to_zero_penalty": float(viterbi_cfg.get("viterbi_pos_to_zero_penalty", 2.0)),
                "lag_identifier_sampler": lag_sampler_info,
            },
            sort_keys=True,
        )
    )
    if not bool(loss_weights_stage1.enable_segment_aware_temporal_loss):
        print("Segment-aware temporal smooth/slope losses are disabled during training; Viterbi is eval-only.")
    if lag_identifier_shuffle:
        print("Lag identifier training DataLoader shuffle is enabled.")
    if lag_sampler_info.get("enabled"):
        print("Lag identifier balanced sampler is enabled: " + json.dumps(lag_sampler_info, sort_keys=True))
    if bool(viterbi_cfg.get("use_viterbi_decode", False)) and dl_va is None:
        print("No val split is used; Viterbi parameters are fixed config defaults and are not tuned on test.")
    identifier = STDALagIdentifier(
        d_source=prepared.group_dims[edge_cfg["source_stage"]],
        d_target=prepared.group_dims[edge_cfg["target_stage"]],
        max_lag=int(lag_cfg.get("max_lag", cfg.get("lag_injection", {}).get("max_lag", cfg["data"].get("L_max", 12)))),
        hidden_dim=int(lag_cfg.get("hidden_dim", 64)),
        num_layers=int(lag_cfg.get("num_layers", 2)),
        temperature=float(lag_cfg.get("lag_temperature", lag_cfg.get("temperature", 0.7))),
        use_temporal_decay=bool(lag_cfg.get("use_temporal_decay", True)),
        use_feature_attention=bool(lag_cfg.get("use_feature_attention", True)),
        dropout=float(lag_cfg.get("dropout", 0.0)),
        use_sequence_smoother=bool(lag_cfg.get("use_sequence_smoother", False)),
        sequence_smoother_hidden_dim=lag_cfg.get("sequence_smoother_hidden_dim"),
        sequence_smoother_layers=int(lag_cfg.get("sequence_smoother_layers", 1)),
        sequence_smoother_dropout=float(lag_cfg.get("sequence_smoother_dropout", 0.0)),
        sequence_smoother_residual_scale=float(lag_cfg.get("sequence_smoother_residual_scale", 0.5)),
        use_candidate_window_encoder=bool(lag_cfg.get("use_candidate_window_encoder", False)),
        lag_window_radius=int(lag_cfg.get("lag_window_radius", 2)),
        lag_window_hidden_dim=lag_cfg.get("lag_window_hidden_dim"),
        lag_window_mode=str(lag_cfg.get("lag_window_mode", "causal")),
        keep_old_point_identifier=bool(lag_cfg.get("keep_old_point_identifier", True)),
    ).to(device)
    init_checkpoint = lag_cfg.get("init_checkpoint", training_cfg.get("lag_identifier_init_checkpoint"))
    if init_checkpoint:
        init_path = Path(str(init_checkpoint))
        if not init_path.is_absolute():
            init_path = (ROOT / init_path).resolve()
        try:
            init_state = torch.load(init_path, map_location=device, weights_only=True)
        except TypeError:
            init_state = torch.load(init_path, map_location=device)
        identifier.load_state_dict(init_state)
        print(f"[stage1] initialized lag identifier from: {init_path}")

    distill_cfg = dict(training_cfg.get("lag_identifier_teacher_distill") or {})
    teacher_identifier = None
    if bool(distill_cfg.get("enabled", False)):
        teacher_identifier = copy.deepcopy(identifier).to(device)
        teacher_checkpoint = distill_cfg.get("checkpoint")
        if teacher_checkpoint:
            teacher_path = Path(str(teacher_checkpoint))
            if not teacher_path.is_absolute():
                teacher_path = (ROOT / teacher_path).resolve()
            try:
                teacher_state = torch.load(teacher_path, map_location=device, weights_only=True)
            except TypeError:
                teacher_state = torch.load(teacher_path, map_location=device)
            teacher_identifier.load_state_dict(teacher_state)
        teacher_identifier.eval()
        for param in teacher_identifier.parameters():
            param.requires_grad = False
        print(
            "[stage1] teacher distillation enabled: "
            + json.dumps(
                {
                    "checkpoint": str(distill_cfg.get("checkpoint", "init_checkpoint_state")),
                    "kl_weight": float(distill_cfg.get("kl_weight", 0.0)),
                    "occurrence_weight": float(distill_cfg.get("occurrence_weight", 0.0)),
                    "scope": str(distill_cfg.get("scope", "all")),
                },
                sort_keys=True,
            )
        )

    checkpoint_summary: Dict[str, Dict[str, Any]] = {}
    if args.eval_stage2_no_joint:
        checkpoint_dir = (args.checkpoint_dir or results_dir).resolve()
        eval_dir = (args.eval_output_dir or (results_dir / "stage2_no_joint_eval")).resolve()
        eval_dir.mkdir(parents=True, exist_ok=True)
        identifier.load_state_dict(torch.load(checkpoint_dir / "best_lag_identifier.pt", map_location=device))
        dimf = make_dimf(cfg, prepared, device)
        n_source = int(prepared.group_dims[edge_cfg["source_stage"]])
        feature_mask = _feature_mask_from_report(checkpoint_dir, n_source)
        _load_dimf_checkpoint_with_lag_head(
            dimf,
            identifier,
            checkpoint_dir / "best_val_pred_dimf.pt",
            edge_cfg,
            edge_name,
            cfg,
            feature_mask=feature_mask,
        )
        y_true, y_pred, pred_m = evaluate_prediction(
            dimf,
            identifier,
            dl_te,
            device,
            edge_cfg,
            edge_name,
            cfg,
            feature_mask,
            prepared.scaler_y,
        )
        lag_eval = collect_lag_outputs(dimf, dl_te, device, edge_cfg, edge_name)
        segment_ids = None
        if getattr(prepared, "segment_ids_test", None) is not None:
            segment_ids = prepared.segment_ids_test[prepared.sample_indices_test]
        full_metrics = save_outputs(eval_dir, lag_eval, y_true, y_pred, feature_mask, cfg=cfg, segment_ids=segment_ids)
        (eval_dir / "run_summary.json").write_text(
            json.dumps(
                {
                    "checkpoint_dir": str(checkpoint_dir),
                    "stage": "stage2_no_joint",
                    "prediction": pred_m,
                    "lag": {k: full_metrics.get(k) for k in full_metrics},
                    "feature_mask": None
                    if feature_mask is None
                    else np.flatnonzero(feature_mask.cpu().numpy()).astype(int).tolist(),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Saved Stage2 no-joint test outputs under: {eval_dir}")
        return

    if raw_adaptation:
        raw_prior_source = str(cfg.get("raw_adaptation", {}).get("prior_source", "pretrained")).lower()
        if raw_prior_source == "pretrained" and not init_checkpoint:
            raise ValueError("Raw adaptation with prior_source=pretrained requires lag_identifier.init_checkpoint or training.lag_identifier_init_checkpoint")
        if raw_prior_source not in {"none", "pretrained", "random"}:
            raise ValueError("raw_adaptation.prior_source must be one of: none, pretrained, random")
        checkpoint_summary["stage0_lag_pretrain"] = {
            "selection_metric": f"{raw_prior_source}_lag_prior",
            "best_epoch": None,
            "best_metric": None,
            "checkpoint": str(init_path) if init_checkpoint else None,
        }
        feature_mask = None
        feature_report = cfg.get("raw_adaptation", {}).get("feature_screening_report")
        if feature_report:
            report_dir = Path(str(feature_report))
            if report_dir.is_file():
                report_dir = report_dir.parent
            if not report_dir.is_absolute():
                report_dir = (ROOT / report_dir).resolve()
            feature_mask = _feature_mask_from_report(report_dir, prepared.group_dims[edge_cfg["source_stage"]])
        dimf = make_dimf(cfg, prepared, device)
        run_raw_y_only_adaptation(
            cfg,
            prepared,
            dl_tr,
            dl_va,
            dl_te,
            dimf,
            identifier,
            device,
            edge_cfg,
            edge_name,
            results_dir,
            feature_mask,
            checkpoint_summary,
        )
        return

    finetune_info = _apply_lag_identifier_finetune_mode(identifier, cfg)
    trainable_params = [param for param in identifier.parameters() if param.requires_grad]
    if not trainable_params:
        raise ValueError("Lag identifier fine-tune mode left no trainable parameters")
    print("[stage1] fine-tune mode: " + json.dumps(finetune_info, sort_keys=True))
    opt_lag = torch.optim.Adam(trainable_params, lr=float(training_cfg.get("lr_lag_identifier", 1e-3)))
    best_lag = float("inf")
    best_lag_epoch = 0
    lag_epochs = int(training_cfg.get("epochs_lag_identifier", 50))
    checkpoint_selection_cfg = dict(cfg.get("checkpoint_selection") or {})
    checkpoint_selection_mode = str(checkpoint_selection_cfg.get("mode", "expected_lag_mae_injected"))
    include_epoch0 = bool(checkpoint_selection_cfg.get("include_epoch0", True))
    checkpoint_selection_rows = []
    if init_checkpoint:
        output_checkpoint = results_dir / "best_lag_identifier.pt"
        if dl_va is not None and include_epoch0:
            init_eval = collect_lag_outputs(identifier, dl_va, device, edge_cfg, edge_name)
            init_metrics = compute_lag_metrics(
                init_eval["pred_pi"],
                init_eval["gt_pi"],
                init_eval["lag_flag"],
                init_eval["lag_value"],
                init_eval["shape_type"],
                init_eval["occurrence_score"],
            )
            best_lag, init_selection = _lag_checkpoint_score(init_eval, init_metrics, cfg)
            checkpoint_selection_rows.append(
                {
                    "epoch": 0,
                    "train_loss": float("nan"),
                    **{f"val_{key}": float(value) for key, value in init_metrics.items() if np.isscalar(value)},
                    **{f"selection_{key}": float(value) for key, value in init_selection.items()},
                }
            )
            print(
                f"[stage1 epoch 0] selection_score={best_lag:.5f} "
                f"val_expected_lag_mae_injected={float(init_metrics.get('expected_lag_mae_injected', np.nan)):.5f}"
            )
        elif dl_va is not None:
            print("[stage1 epoch 0] excluded from checkpoint selection by checkpoint_selection.include_epoch0=false")
        torch.save(identifier.state_dict(), output_checkpoint)
    if lag_epochs <= 0:
        checkpoint_path = results_dir / "best_lag_identifier.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"epochs_lag_identifier <= 0 requires an existing checkpoint: {checkpoint_path}")
        if checkpoint_selection_rows:
            pd.DataFrame(checkpoint_selection_rows).to_csv(results_dir / "checkpoint_selection_curve.csv", index=False)
        checkpoint_summary["stage1"] = {
            "selection_metric": "pretrained_checkpoint",
            "best_epoch": 0,
            "best_metric": None,
            "checkpoint": str(checkpoint_path),
        }
        print(f"[stage1] skipped; using existing checkpoint: {checkpoint_path}")
    else:
        for epoch in range(1, lag_epochs + 1):
            train_log = train_lag_identifier_epoch(
                identifier,
                dl_tr_temporal,
                opt_lag,
                device,
                edge_cfg,
                edge_name,
                loss_weights_stage1,
                teacher=teacher_identifier,
                distill_cfg=distill_cfg,
            )
            if dl_va is not None:
                val_eval = collect_lag_outputs(identifier, dl_va, device, edge_cfg, edge_name)
                val_metrics = compute_lag_metrics(
                    val_eval["pred_pi"],
                    val_eval["gt_pi"],
                    val_eval["lag_flag"],
                    val_eval["lag_value"],
                    val_eval["shape_type"],
                    val_eval["occurrence_score"],
                )
                metric, selection_components = _lag_checkpoint_score(val_eval, val_metrics, cfg)
                checkpoint_selection_rows.append(
                    {
                        "epoch": int(epoch),
                        "train_loss": float(train_log["loss"]),
                        **{f"val_{key}": float(value) for key, value in val_metrics.items() if np.isscalar(value)},
                        **{f"selection_{key}": float(value) for key, value in selection_components.items()},
                    }
                )
                if metric < best_lag:
                    best_lag = metric
                    best_lag_epoch = epoch
                    torch.save(identifier.state_dict(), results_dir / "best_lag_identifier.pt")
                print(
                    f"[stage1 epoch {epoch}] loss={train_log['loss']:.5f} "
                    f"selection_score={metric:.5f} "
                    f"val_expected_lag_mae_injected={float(val_metrics.get('expected_lag_mae_injected', np.nan)):.5f}"
                )
            else:
                metric = float(train_log["loss"])
                if metric < best_lag:
                    best_lag = metric
                    best_lag_epoch = epoch
                    torch.save(identifier.state_dict(), results_dir / "best_lag_identifier.pt")
                print(f"[stage1 epoch {epoch}] loss={train_log['loss']:.5f} best_train_loss={best_lag:.5f}@{best_lag_epoch}")
        if checkpoint_selection_rows:
            pd.DataFrame(checkpoint_selection_rows).to_csv(results_dir / "checkpoint_selection_curve.csv", index=False)
        checkpoint_summary["stage1"] = {
            "selection_metric": checkpoint_selection_mode if dl_va is not None else "train_loss",
            "best_epoch": int(best_lag_epoch),
            "best_metric": float(best_lag),
        }
    identifier.load_state_dict(torch.load(results_dir / "best_lag_identifier.pt", map_location=device))

    if args.lag_only:
        lag_eval = collect_lag_outputs(identifier, dl_te, device, edge_cfg, edge_name)
        lag_eval, postprocess_info = postprocess_lag_eval_for_eval(lag_eval, cfg)
        lag_metrics = compute_lag_metrics(
            lag_eval["pred_pi"],
            lag_eval["gt_pi"],
            lag_eval["lag_flag"],
            lag_eval["lag_value"],
            lag_eval["shape_type"],
            lag_eval["occurrence_score"],
        )
        lag_metrics["lag_postprocess"] = json.dumps(postprocess_info)
        pd.DataFrame([lag_metrics]).to_csv(results_dir / "lag_identifier_test_metrics.csv", index=False)
        pd.DataFrame(
            _lag_prediction_frame(lag_eval)
        ).to_csv(results_dir / "lag_identifier_test_curve.csv", index=False)
        print(f"Saved test lag curve data: {results_dir / 'lag_identifier_test_curve.csv'}")
        print(f"Saved lag identifier metrics: {results_dir / 'lag_identifier_test_metrics.csv'}")
        return

    feature_mask = None
    if bool(cfg.get("feature_screening", {}).get("enabled", True)):
        screening_loader = dl_va if dl_va is not None else dl_tr_temporal
        train_eval = collect_lag_outputs(identifier, screening_loader, device, edge_cfg, edge_name)
        attention_mass = attention_mass_score(train_eval["feature_importance_batches"])
        entropy_penalty = entropy_penalty_score([torch.tensor(train_eval["pred_feature_pi"])])
        feature_score = combine_feature_scores(attention_mass, entropy_penalty=entropy_penalty)
        feature_mask = select_feature_mask(
            feature_score,
            top_k=cfg.get("feature_screening", {}).get("top_k"),
            top_ratio=cfg.get("feature_screening", {}).get("top_ratio", 0.3),
        )
        (results_dir / "feature_screening_report.json").write_text(
            json.dumps(screening_report(attention_mass, feature_score, feature_mask), indent=2) + "\n",
            encoding="utf-8",
        )

    dimf = make_dimf(cfg, prepared, device)
    _attach_lag_guided_alignment_to_dimf(dimf, identifier, edge_cfg, edge_name, cfg, feature_mask=feature_mask)
    for param in identifier.parameters():
        param.requires_grad = False
    opt_dimf = torch.optim.Adam(dimf.parameters(), lr=float(training_cfg.get("lr_dimf", 1e-3)))
    best_pred = float("inf")
    best_pred_epoch = 0
    for epoch in range(1, int(training_cfg.get("epochs_dimf", 100)) + 1):
        train_log = train_dimf_epoch(dimf, identifier, dl_tr_temporal, opt_dimf, device, edge_cfg, edge_name, cfg, feature_mask, joint=False)
        if dl_va is not None:
            yv, pv, pred_m = evaluate_prediction(dimf, identifier, dl_va, device, edge_cfg, edge_name, cfg, feature_mask, prepared.scaler_y)
            if pred_m["prediction_rmse"] < best_pred:
                best_pred = pred_m["prediction_rmse"]
                best_pred_epoch = epoch
                torch.save(dimf.state_dict(), results_dir / "best_val_pred_dimf.pt")
            print(f"[stage2 epoch {epoch}] loss={train_log['loss']:.5f} val_rmse={pred_m['prediction_rmse']:.5f}")
            del yv, pv
        else:
            metric = float(train_log["pred"])
            if metric < best_pred:
                best_pred = metric
                best_pred_epoch = epoch
                torch.save(dimf.state_dict(), results_dir / "best_val_pred_dimf.pt")
            print(f"[stage2 epoch {epoch}] loss={train_log['loss']:.5f} pred={train_log['pred']:.5f} lag={train_log['lag']:.5f} best_train_pred={best_pred:.5f}@{best_pred_epoch}")
    checkpoint_summary["stage2"] = {
        "selection_metric": "val_prediction_rmse" if dl_va is not None else "train_pred",
        "best_epoch": int(best_pred_epoch),
        "best_metric": float(best_pred),
    }

    dimf.load_state_dict(torch.load(results_dir / "best_val_pred_dimf.pt", map_location=device))
    torch.save({"dimf": dimf.state_dict(), "lag_identifier": identifier.state_dict()}, results_dir / "best_stage2_only.pt")
    y_true, y_pred, pred_m = evaluate_prediction(dimf, identifier, dl_te, device, edge_cfg, edge_name, cfg, feature_mask, prepared.scaler_y)

    if bool(cfg.get("logging", {}).get("ytarget_only_output", False)):
        save_prediction_only_outputs(
            results_dir,
            y_true,
            y_pred,
            getattr(prepared, "sample_indices_test", None),
            pred_m,
        )
        (results_dir / "run_summary.json").write_text(
            json.dumps(
                {
                    "prediction": pred_m,
                    "checkpoint_selection": checkpoint_summary,
                    "stage": "stage2_only_y_target",
                    "lag_checkpoint": str(results_dir / "best_lag_identifier.pt"),
                    "dimf_checkpoint": str(results_dir / "best_val_pred_dimf.pt"),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Saved y-target metrics and figures under: {results_dir}")
        return

    lag_eval = collect_lag_outputs(dimf, dl_te, device, edge_cfg, edge_name)
    segment_ids = None
    if getattr(prepared, "segment_ids_test", None) is not None:
        segment_ids = prepared.segment_ids_test[prepared.sample_indices_test]
    full_metrics = save_outputs(results_dir, lag_eval, y_true, y_pred, feature_mask, cfg=cfg, segment_ids=segment_ids)
    current_variant = str(
        cfg.get("ablation", {}).get(
            "variant",
            "F_dimf_lag_guided_alignment_feature_screening",
        )
    )
    write_ablation_table(results_dir, full_metrics, current_variant=current_variant)
    (results_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "prediction": pred_m,
                "lag": {k: full_metrics.get(k) for k in full_metrics},
                "checkpoint_selection": checkpoint_summary,
                "lag_identifier_settings": {
                    "use_candidate_window_encoder": bool(lag_cfg.get("use_candidate_window_encoder", False)),
                    "lag_window_radius": int(lag_cfg.get("lag_window_radius", 2)),
                    "seq_len": int(seq_len),
                    "use_gaussian_lag_label": bool(loss_weights_stage1.use_gaussian_lag_label),
                    "gaussian_lag_sigma": float(loss_weights_stage1.gaussian_lag_sigma),
                    "enable_segment_aware_temporal_loss": bool(loss_weights_stage1.enable_segment_aware_temporal_loss),
                    "use_viterbi_decode": bool(viterbi_cfg.get("use_viterbi_decode", False)),
                    "viterbi_smooth_lambda": float(viterbi_cfg.get("viterbi_smooth_lambda", 0.8)),
                    "viterbi_switch_penalty": float(viterbi_cfg.get("viterbi_switch_penalty", 1.5)),
                    "viterbi_pos_to_zero_penalty": float(viterbi_cfg.get("viterbi_pos_to_zero_penalty", 2.0)),
                },
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"Saved lag metrics and figures under: {results_dir}")


if __name__ == "__main__":
    main()
