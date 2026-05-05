from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

@dataclass
class PreparedData:
    """
    训练入口真正消费的数据载体。

    这里同时保存：
    1. train/val/test 三个切分下、按工段拆分好的特征矩阵；
    2. 目标值与标准化器；
    3. 合法样本中心索引、时间戳以及辅助监督目标。
    """
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
    timestamps_train: Optional[np.ndarray] = None
    timestamps_val: Optional[np.ndarray] = None
    timestamps_test: Optional[np.ndarray] = None
    extra_targets_train: Optional[Dict[str, np.ndarray]] = None
    extra_targets_val: Optional[Dict[str, np.ndarray]] = None
    extra_targets_test: Optional[Dict[str, np.ndarray]] = None

def _infer_groups(df: pd.DataFrame, time_col: str, target_col: str,
                  feed_prefix: str, s1_prefix: str, s2_prefix: str, s3_prefix: str):
    # 通过列名前缀把原始变量归入四个工段，time/target 列不参与分组。
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


def _mask_col_name(col: str) -> str:
    # mask 列命名统一收口在这里，避免不同模块手写字符串时不一致。
    return f"mask_{col}"

def _split_rows(df: pd.DataFrame, train_ratio: float, val_ratio: float, test_ratio: float):
    # 最朴素的切分方式：按时间排序后的行号直接切 train/val/test。
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return df.iloc[:n_train].copy(), df.iloc[n_train:n_train+n_val].copy(), df.iloc[n_train+n_val:].copy()


def _split_predefined_rows(
    df: pd.DataFrame,
    time_col: str,
    split_col: str,
):
    # 使用数据表里预先给好的 split 标签切分，并在每个切分内重新按时间排序。
    parts = []
    for split_name in ["train", "val", "test"]:
        df_part = (
            df.loc[df[split_col] == split_name]
            .sort_values(time_col)
            .reset_index(drop=True)
            .copy()
        )
        if df_part.empty:
            raise ValueError(f"No rows found for split='{split_name}' in column '{split_col}'")
        # 进入各自切分后，split 列不再参与后续正则化和特征构造。
        if split_col in df_part.columns:
            df_part = df_part.drop(columns=[split_col])
        parts.append(df_part)
    return tuple(parts)

def _valid_segment_end_indices(
    df: pd.DataFrame,
    time_col: str,
    total_steps: int,
    collection_interval_min: int,
) -> np.ndarray:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    # 一个合法样本需要覆盖 history+future 共 total_steps 个规则时间点。
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
    # 先枚举整个原始序列中所有“能形成完整连续窗口”的终点位置。
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

    # 为了让每个切分仍保留形成这些样本所需的历史上下文，
    # val/test 的起点会适当前移 total_steps。
    train_border1, train_border2 = 0, int(val_end.min())
    val_border1, val_border2 = int(val_end.min() - total_steps), int(test_end.min())
    test_border1, test_border2 = int(test_end.min() - total_steps), len(df)

    parts = (
        df.iloc[train_border1:train_border2].copy(),
        df.iloc[val_border1:val_border2].copy(),
        df.iloc[test_border1:test_border2].copy(),
    )

    def to_local_t(sample_end: np.ndarray, border1: int) -> np.ndarray:
        # sample_end 指向窗口右边界后一位，因此要减去 horizon 和 1 才能回到样本中心时刻 t。
        return sample_end - border1 - horizon_steps - 1

    sample_indices = (
        to_local_t(train_end, train_border1),
        to_local_t(val_end, val_border1),
        to_local_t(test_end, test_border1),
    )
    return parts, sample_indices


def _split_predefined_valid_segments(
    df: pd.DataFrame,
    time_col: str,
    split_col: str,
    history_steps: int,
    horizon_steps: int,
    collection_interval_min: int,
):
    total_steps = history_steps + horizon_steps
    parts = []
    sample_indices = []

    for split_name in ["train", "val", "test"]:
        # 预定义切分模式下，每个 split 内部仍然只保留可形成完整连续窗口的样本。
        df_part = (
            df.loc[df[split_col] == split_name]
            .sort_values(time_col)
            .reset_index(drop=True)
            .copy()
        )
        if df_part.empty:
            raise ValueError(f"No rows found for split='{split_name}' in column '{split_col}'")

        valid_end = _valid_segment_end_indices(
            df_part,
            time_col=time_col,
            total_steps=total_steps,
            collection_interval_min=collection_interval_min,
        )
        if len(valid_end) == 0:
            raise ValueError(
                f"No valid contiguous windows found for split='{split_name}' under predefined split mode"
            )

        parts.append(df_part)
        sample_indices.append(valid_end - horizon_steps - 1)

    return tuple(parts), tuple(sample_indices)


def _regularize_split_with_gap_policy(
    df_part: pd.DataFrame,
    time_col: str,
    collection_interval_min: int,
    gap_break_min: int,
    gap_fill_min: int,
    fillna: str,
    use_delta_t: bool,
    sample_keep_col: Optional[str] = None,
    respect_existing_segment_id: bool = False,
    ) -> pd.DataFrame:
    """
    按照“两阈值策略”处理单个时间切分：
    1. gap > G_break 时切成新的 segment；
    2. 在每个 segment 内仅对 gap <= G_fill 的小缺口补齐到 15 min 网格；
    3. gap 介于 (G_fill, G_break] 时保留原始断点，不做强行补齐。
    """
    if df_part.empty:
        return df_part.copy()
    if gap_fill_min > gap_break_min:
        raise ValueError("gap_fill_min must be <= gap_break_min")

    # 先按时间排序；如果显式要求保留已有 segment_id，则在 segment 内部再按时间排序。
    if respect_existing_segment_id and "segment_id" in df_part.columns:
        df_part = df_part.sort_values(["segment_id", time_col]).reset_index(drop=True).copy()
    else:
        df_part = df_part.sort_values(time_col).reset_index(drop=True).copy()
    interval = pd.Timedelta(minutes=collection_interval_min)
    meta_cols = {"segment_id", "is_real_observation", "is_small_gap_fill", "delta_t_min"}
    if sample_keep_col:
        meta_cols.add(sample_keep_col)
    value_cols = [c for c in df_part.columns if c != time_col and c not in meta_cols]
    mask_cols = [_mask_col_name(c) for c in value_cols]

    if respect_existing_segment_id and "segment_id" in df_part.columns:
        raw_segment_id = df_part["segment_id"].fillna(-1).astype(np.int64)
    else:
        delta_minutes = df_part[time_col].diff().dt.total_seconds().div(60.0)
        raw_segment_id = delta_minutes.gt(float(gap_break_min)).fillna(False).cumsum().astype(np.int64)
    df_part["_raw_segment_id"] = raw_segment_id

    regularized_parts = []
    for segment_id, segment_df in df_part.groupby("_raw_segment_id", sort=True):
        # 每个 segment 独立处理，避免跨大缺口做插补或传播统计量。
        segment_df = segment_df.drop(columns="_raw_segment_id").sort_values(time_col).reset_index(drop=True).copy()
        segment_df["is_real_observation"] = 1
        segment_df["is_small_gap_fill"] = 0
        segment_df["segment_id"] = int(segment_id)
        if use_delta_t:
            # 使用原始（未规则化）真实观测之间的时间差。
            # 对插补出来的规则网格位置，后面会继承“下一条真实观测”的原始 gap 大小，
            # 从而保留小缺口发生过的证据，而不是在规则化后全部退化成 15 min。
            segment_df["delta_t_min"] = (
                segment_df[time_col].diff().dt.total_seconds().div(60.0).fillna(0.0).astype(np.float32)
            )
        for col in value_cols:
            # 第 3 点修改：mask=1 表示该特征在原始数据中真实存在，mask=0 表示后续需要依赖重采样/插补。
            segment_df[_mask_col_name(col)] = segment_df[col].notna().astype(np.int8)

        # 先在 segment 内建立完整的规则时间网格，再只保留“真实点 + 可补齐的小缺口”。
        full_grid = pd.date_range(
            start=segment_df[time_col].iloc[0],
            end=segment_df[time_col].iloc[-1],
            freq=interval,
        )
        fillable_times = []
        for prev_ts, cur_ts in zip(segment_df[time_col], segment_df[time_col].iloc[1:]):
            gap = cur_ts - prev_ts
            gap_minutes = int(gap.total_seconds() // 60)
            if gap <= interval:
                continue
            # 只有 gap 足够小、且正好是采样间隔的整数倍时，才允许补齐中间点。
            if gap_minutes <= gap_fill_min and gap_minutes % collection_interval_min == 0:
                n_missing = gap_minutes // collection_interval_min - 1
                for step in range(1, n_missing + 1):
                    fillable_times.append(prev_ts + step * interval)

        # reindex 之后，原本缺失的规则时间点会显式出现在表里，便于后续筛选“可补的小缺口”。
        segment_grid = segment_df.set_index(time_col).reindex(full_grid)
        segment_grid.index.name = time_col
        segment_grid["segment_id"] = int(segment_id)
        segment_grid["is_real_observation"] = segment_grid["is_real_observation"].fillna(0).astype(np.int8)
        segment_grid["is_small_gap_fill"] = pd.Index(segment_grid.index).isin(fillable_times).astype(np.int8)
        if use_delta_t:
            # 补齐位置没有原始真实观测；这里让它继承“下一条真实观测”的原始 gap 大小，
            # 让模型能在整个小缺口区域感知到不规则采样信号。
            segment_grid["delta_t_min"] = (
                segment_grid["delta_t_min"].bfill().fillna(0.0).astype(np.float32)
            )
        if sample_keep_col:
            segment_grid[sample_keep_col] = segment_grid[sample_keep_col].fillna(0).astype(np.int8)
        for col in value_cols:
            segment_grid[_mask_col_name(col)] = segment_grid[_mask_col_name(col)].fillna(0).astype(np.int8)

        keep_mask = segment_grid["is_real_observation"].eq(1) | segment_grid["is_small_gap_fill"].eq(1)
        segment_grid = segment_grid.loc[keep_mask].copy()

        # 只对保留下来的小缺口做补值，避免跨中/大缺口传播信息。
        fill_cols = value_cols
        if fillna == "ffill":
            segment_grid[fill_cols] = segment_grid[fill_cols].ffill().bfill()
        elif fillna == "bfill":
            segment_grid[fill_cols] = segment_grid[fill_cols].bfill().ffill()
        elif fillna == "zero":
            segment_grid[fill_cols] = segment_grid[fill_cols].fillna(0.0)
        elif fillna in {"none", None}:
            pass
        else:
            raise ValueError(f"Unknown fillna: {fillna}")

        regularized_parts.append(segment_grid.reset_index())

    out = pd.concat(regularized_parts, ignore_index=True)
    aux_cols = ["delta_t_min"] if use_delta_t else []
    ordered_cols = [time_col] + value_cols + aux_cols + mask_cols
    if sample_keep_col:
        ordered_cols.append(sample_keep_col)
    ordered_cols += ["segment_id", "is_real_observation", "is_small_gap_fill"]
    out = out[ordered_cols].sort_values(time_col).reset_index(drop=True)
    return out


def _sample_indices_from_regularized_split(
    df_part: pd.DataFrame,
    time_col: str,
    history_steps: Optional[int],
    horizon_steps: Optional[int],
    collection_interval_min: int,
    sample_keep_col: Optional[str] = None,
) -> Optional[np.ndarray]:
    if history_steps is None or horizon_steps is None:
        return None
    required_cols = {time_col, "segment_id", "is_real_observation"}
    missing_cols = required_cols.difference(df_part.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns for sample construction: {sorted(missing_cols)}")

    # 第 4 点修改：严格按 segment 内滑动窗口构造样本。
    # 一个候选样本只有在下面四个条件都满足时才保留：
    # 1. 窗口完全处于同一个 segment；
    # 2. 当前时刻 t 为真实观测点；
    # 3. 标签时刻 t+H 为真实观测点；
    # 4. 该 segment 长度至少满足 L+H。
    total_steps = history_steps + horizon_steps
    expected_span = pd.Timedelta(minutes=collection_interval_min * (total_steps - 1))
    sample_indices = []

    for _, segment_df in df_part.groupby("segment_id", sort=True):
        segment_df = segment_df.reset_index().rename(columns={"index": "_global_index"})
        seg_len = len(segment_df)
        if seg_len < total_steps:
            continue

        for local_t in range(history_steps - 1, seg_len - horizon_steps):
            start_local = local_t - history_steps + 1
            label_local = local_t + horizon_steps

            # 保证整个 [t-L+1, ..., t, ..., t+H] 处于 15 min 规则连续网格上。
            actual_span = segment_df[time_col].iloc[label_local] - segment_df[time_col].iloc[start_local]
            if actual_span != expected_span:
                continue

            # 当前时刻和标签时刻都必须对应原始真实观测点，而不是补齐点。
            if int(segment_df["is_real_observation"].iloc[local_t]) != 1:
                continue
            if int(segment_df["is_real_observation"].iloc[label_local]) != 1:
                continue
            if sample_keep_col and sample_keep_col in segment_df.columns:
                if int(segment_df[sample_keep_col].iloc[local_t]) != 1:
                    continue

            sample_indices.append(int(segment_df["_global_index"].iloc[local_t]))

    if len(sample_indices) == 0:
        raise ValueError("No valid in-segment samples found under the requested L/H and observation constraints")
    return np.asarray(sample_indices, dtype=np.int64)

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
    gap_break_min: int = 120,
    gap_fill_min: int = 60,
    use_missing_mask: bool = True,
    include_target_history: bool = False,
    split_col: str = "split",
    sample_keep_col: Optional[str] = None,
    respect_existing_segment_id: bool = False,
) -> Tuple[PreparedData, Dict[str, List[str]]]:
    """
    从原始 CSV 走完整个 DIMF 输入准备流程：
    1. 读表并按列前缀识别工段；
    2. 先切分，再在各切分内做时间规则化与缺口处理；
    3. 组装“原始特征 + delta_t + mask”；
    4. 仅用训练集拟合标准化器；
    5. 返回模型训练所需的全部张量化前数据。
    """
    df = pd.read_csv(csv_path)
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).reset_index(drop=True)

    groups = _infer_groups(df, time_col, target_col, feed_prefix, stage1_prefix, stage2_prefix, stage3_prefix)
    if include_target_history and target_col not in groups["stage3"]:
        groups["stage3"] = groups["stage3"] + [target_col]

    # 第二点修改：先按时间顺序切分原始数据，再在各自切分内独立做“两阈值策略”，避免跨切分信息泄露。
    if split_mode in {"rows", "valid_segments"}:
        raw_parts = _split_rows(df, train_ratio, val_ratio, test_ratio)
    elif split_mode == "predefined_valid_segments":
        if split_col not in df.columns:
            raise ValueError(f"Missing split column '{split_col}' for predefined split mode")
        raw_parts = _split_predefined_rows(df, time_col=time_col, split_col=split_col)
    else:
        raise ValueError(f"Unknown split_mode: {split_mode}")

    df_train, df_val, df_test = [
        _regularize_split_with_gap_policy(
            df_part=part,
            time_col=time_col,
            collection_interval_min=collection_interval_min,
            gap_break_min=gap_break_min,
            gap_fill_min=gap_fill_min,
            fillna=fillna,
            use_delta_t=use_delta_t,
            sample_keep_col=sample_keep_col,
            respect_existing_segment_id=respect_existing_segment_id,
        )
        for part in raw_parts
    ]

    # 第 3 点修改：模型输入顺序固定为“工段原始特征 + 共享 delta_t + 当前工段 mask”。
    groups_with_aux = {}
    for group_name, base_cols in groups.items():
        # 这里的列顺序就是后面张量最后一维的顺序，必须保持稳定。
        group_cols = list(base_cols)
        if use_delta_t:
            group_cols.append("delta_t_min")
        if use_missing_mask:
            group_cols.extend([_mask_col_name(col) for col in base_cols])
        groups_with_aux[group_name] = group_cols
    groups = groups_with_aux

    sample_indices_train = _sample_indices_from_regularized_split(
        df_train,
        time_col=time_col,
        history_steps=history_steps,
        horizon_steps=horizon_steps,
        collection_interval_min=collection_interval_min,
        sample_keep_col=sample_keep_col,
    )
    sample_indices_val = _sample_indices_from_regularized_split(
        df_val,
        time_col=time_col,
        history_steps=history_steps,
        horizon_steps=horizon_steps,
        collection_interval_min=collection_interval_min,
        sample_keep_col=sample_keep_col,
    )
    sample_indices_test = _sample_indices_from_regularized_split(
        df_test,
        time_col=time_col,
        history_steps=history_steps,
        horizon_steps=horizon_steps,
        collection_interval_min=collection_interval_min,
        sample_keep_col=sample_keep_col,
    )

    # fit scalers on TRAIN only (avoid leakage)
    # mask 列保留 0/1 含义，不参与标准化；其余数值特征统一按训练集统计量缩放。
    x_cols_all = sorted(set(sum(groups.values(), [])))
    mask_cols_all = sorted([c for c in x_cols_all if c.startswith("mask_")])
    scaled_x_cols = [c for c in x_cols_all if c not in mask_cols_all]
    scaler_x = StandardScaler().fit(df_train[scaled_x_cols].values)
    scaler_y = StandardScaler().fit(df_train[[target_col]].values)
    scaled_col_to_idx = {c: i for i, c in enumerate(scaled_x_cols)}

    def transform(df_part):
        # 先一次性标准化全部可缩放列，再按工段拆回字典，减少重复计算。
        scaled_x_all = scaler_x.transform(df_part[scaled_x_cols].values).astype(np.float32)
        X_groups = {}
        for g, cols in groups.items():
            group_arrays = []
            for col in cols:
                if col in mask_cols_all:
                    # mask 保持 0/1 语义，不做标准化。
                    group_arrays.append(df_part[[col]].values.astype(np.float32))
                else:
                    group_arrays.append(scaled_x_all[:, [scaled_col_to_idx[col]]])
            X_groups[g] = np.concatenate(group_arrays, axis=1).astype(np.float32)
        y = scaler_y.transform(df_part[[target_col]].values).astype(np.float32).reshape(-1)
        return X_groups, y

    def extract_extra_targets(df_part: pd.DataFrame) -> Optional[Dict[str, np.ndarray]]:
        # 这里统一收口所有“不是主回归目标、但训练时可能会用到”的辅助监督标签。
        extra_targets = {}
        if "lag_gt" in df_part.columns:
            extra_targets["stage1_to_stage2_lag_gt"] = (
                df_part["lag_gt"].fillna(-1).astype(np.int64).to_numpy()
            )
        if "inject_flag" in df_part.columns:
            extra_targets["stage1_to_stage2_in_block_gt"] = (
                df_part["inject_flag"].fillna(0).astype(np.int64).to_numpy()
            )
        if "segment_dmax_gt" in df_part.columns:
            extra_targets["stage1_to_stage2_dmax_gt"] = (
                df_part["segment_dmax_gt"].fillna(0).astype(np.int64).to_numpy()
            )
        return extra_targets or None

    Xtr, ytr = transform(df_train)
    Xva, yva = transform(df_val)
    Xte, yte = transform(df_test)
    extra_tr = extract_extra_targets(df_train)
    extra_va = extract_extra_targets(df_val)
    extra_te = extract_extra_targets(df_test)

    group_dims = {k: v.shape[1] for k, v in Xtr.items()}

    # 时间戳保留成字符串，便于直接导出到 csv/日志里做可视化和误差对齐分析。
    prepared = PreparedData(
        X_groups_train=Xtr, y_train=ytr,
        X_groups_val=Xva, y_val=yva,
        X_groups_test=Xte, y_test=yte,
        group_dims=group_dims,
        scaler_x=scaler_x, scaler_y=scaler_y,
        sample_indices_train=sample_indices_train,
        sample_indices_val=sample_indices_val,
        sample_indices_test=sample_indices_test,
        timestamps_train=df_train[time_col].dt.strftime("%Y-%m-%d %H:%M").to_numpy(),
        timestamps_val=df_val[time_col].dt.strftime("%Y-%m-%d %H:%M").to_numpy(),
        timestamps_test=df_test[time_col].dt.strftime("%Y-%m-%d %H:%M").to_numpy(),
        extra_targets_train=extra_tr,
        extra_targets_val=extra_va,
        extra_targets_test=extra_te,
    )
    return prepared, groups
