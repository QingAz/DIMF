from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

@dataclass
class PreparedData:
    X_groups_train: Dict[str, np.ndarray]
    y_train: np.ndarray
    X_groups_val: Dict[str, np.ndarray]
    y_val: np.ndarray
    X_groups_test: Dict[str, np.ndarray]
    y_test: np.ndarray
    group_dims: Dict[str, int]
    scaler_x: StandardScaler
    scaler_y: StandardScaler
    sample_indices_train: Optional[np.ndarray] = None
    sample_indices_val: Optional[np.ndarray] = None
    sample_indices_test: Optional[np.ndarray] = None

def _infer_groups(df: pd.DataFrame, time_col: str, target_col: str,
                  feed_prefix: str, s1_prefix: str, s2_prefix: str, s3_prefix: str):
    cols = [c for c in df.columns if c not in [time_col, target_col]]
    groups = {"feed": [], "stage1": [], "stage2": [], "stage3": []}
    for c in cols:
        if c.startswith(feed_prefix):
            groups["feed"].append(c)
        elif c.startswith(s1_prefix):
            groups["stage1"].append(c)
        elif c.startswith(s2_prefix):
            groups["stage2"].append(c)
        elif c.startswith(s3_prefix):
            groups["stage3"].append(c)
    for k in groups:
        groups[k] = sorted(groups[k])
    return groups

def _split_rows(df: pd.DataFrame, train_ratio: float, val_ratio: float, test_ratio: float):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return df.iloc[:n_train].copy(), df.iloc[n_train:n_train+n_val].copy(), df.iloc[n_train+n_val:].copy()

def _valid_segment_end_indices(
    df: pd.DataFrame,
    time_col: str,
    total_steps: int,
    collection_interval_min: int,
) -> np.ndarray:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    target_delta = pd.Timedelta(minutes=collection_interval_min * (total_steps - 1))
    ends = []
    # Match MultistageNet's segment enumeration exactly:
    # it scans end indices in range(seq_len + pred_len, len(data)),
    # so the final possible endpoint is intentionally excluded.
    for end in range(total_steps, len(df)):
        window = df.iloc[end - total_steps:end]
        if window[time_col].iloc[-1] - window[time_col].iloc[0] == target_delta:
            ends.append(end)
    return np.asarray(ends, dtype=np.int64)

def _split_valid_segments(
    df: pd.DataFrame,
    time_col: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    history_steps: int,
    horizon_steps: int,
    collection_interval_min: int,
):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    total_steps = history_steps + horizon_steps
    valid_end = _valid_segment_end_indices(df, time_col, total_steps, collection_interval_min)
    if len(valid_end) == 0:
        raise ValueError("No valid contiguous windows found for the requested history/horizon")

    n_train = int(len(valid_end) * train_ratio)
    n_test = int(len(valid_end) * test_ratio)
    n_val = len(valid_end) - n_train - n_test
    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError("Split ratios produce an empty train/val/test partition")

    train_end = valid_end[:n_train]
    val_end = valid_end[n_train:n_train + n_val]
    test_end = valid_end[n_train + n_val:]

    train_border1, train_border2 = 0, int(val_end.min())
    val_border1, val_border2 = int(val_end.min() - total_steps), int(test_end.min())
    test_border1, test_border2 = int(test_end.min() - total_steps), len(df)

    parts = (
        df.iloc[train_border1:train_border2].copy(),
        df.iloc[val_border1:val_border2].copy(),
        df.iloc[test_border1:test_border2].copy(),
    )

    def to_local_t(sample_end: np.ndarray, border1: int) -> np.ndarray:
        return sample_end - border1 - horizon_steps - 1

    sample_indices = (
        to_local_t(train_end, train_border1),
        to_local_t(val_end, val_border1),
        to_local_t(test_end, test_border1),
    )
    return parts, sample_indices

def load_and_prepare(
    csv_path: str,
    time_col: str,
    target_col: str,
    feed_prefix: str,
    stage1_prefix: str,
    stage2_prefix: str,
    stage3_prefix: str,
    fillna: str = "ffill",
    use_delta_t: bool = True,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    split_mode: str = "rows",
    history_steps: Optional[int] = None,
    horizon_steps: Optional[int] = None,
    collection_interval_min: int = 15,
    include_target_history: bool = False,
) -> Tuple[PreparedData, Dict[str, List[str]]]:
    df = pd.read_csv(csv_path)
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).reset_index(drop=True)

    # delta_t feature (minutes)
    if use_delta_t:
        df["delta_t_min"] = (df[time_col].diff().dt.total_seconds().fillna(0.0) / 60.0).astype(np.float32)

    # fill missing values (only features/target, not timestamp)
    feature_cols = [c for c in df.columns if c != time_col]
    if fillna == "ffill":
        df[feature_cols] = df[feature_cols].ffill().bfill()
    elif fillna == "bfill":
        df[feature_cols] = df[feature_cols].bfill().ffill()
    elif fillna == "zero":
        df[feature_cols] = df[feature_cols].fillna(0.0)
    else:
        raise ValueError(f"Unknown fillna: {fillna}")

    groups = _infer_groups(df, time_col, target_col, feed_prefix, stage1_prefix, stage2_prefix, stage3_prefix)
    if include_target_history and target_col not in groups["stage3"]:
        groups["stage3"] = groups["stage3"] + [target_col]
    if use_delta_t:
        for k in groups:
            groups[k] = groups[k] + ["delta_t_min"]

    if split_mode == "rows":
        (df_train, df_val, df_test) = _split_rows(df, train_ratio, val_ratio, test_ratio)
        sample_indices_train = sample_indices_val = sample_indices_test = None
    elif split_mode == "valid_segments":
        if history_steps is None or horizon_steps is None:
            raise ValueError("history_steps and horizon_steps are required when split_mode='valid_segments'")
        (df_train, df_val, df_test), sample_indices = _split_valid_segments(
            df=df,
            time_col=time_col,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            history_steps=history_steps,
            horizon_steps=horizon_steps,
            collection_interval_min=collection_interval_min,
        )
        sample_indices_train, sample_indices_val, sample_indices_test = sample_indices
    else:
        raise ValueError(f"Unknown split_mode: {split_mode}")

    # fit scalers on TRAIN only (avoid leakage)
    x_cols_all = sorted(set(sum(groups.values(), [])))
    scaler_x = StandardScaler().fit(df_train[x_cols_all].values)
    scaler_y = StandardScaler().fit(df_train[[target_col]].values)

    col_to_idx = {c: i for i, c in enumerate(x_cols_all)}

    def transform(df_part):
        X_all = scaler_x.transform(df_part[x_cols_all].values).astype(np.float32)
        X_groups = {}
        for g, cols in groups.items():
            idxs = [col_to_idx[c] for c in cols]
            X_groups[g] = X_all[:, idxs]
        y = scaler_y.transform(df_part[[target_col]].values).astype(np.float32).reshape(-1)
        return X_groups, y

    Xtr, ytr = transform(df_train)
    Xva, yva = transform(df_val)
    Xte, yte = transform(df_test)

    group_dims = {k: v.shape[1] for k, v in Xtr.items()}

    prepared = PreparedData(
        X_groups_train=Xtr, y_train=ytr,
        X_groups_val=Xva, y_val=yva,
        X_groups_test=Xte, y_test=yte,
        group_dims=group_dims,
        scaler_x=scaler_x, scaler_y=scaler_y,
        sample_indices_train=sample_indices_train,
        sample_indices_val=sample_indices_val,
        sample_indices_test=sample_indices_test,
    )
    return prepared, groups
