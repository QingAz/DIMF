from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import json
import math

import numpy as np
import pandas as pd


EDGE_STAGE_PREFIX = {
    "feed": "feed_",
    "stage1": "stage1_",
    "stage2": "stage2_",
    "stage3": "stage3_",
}

SHAPE_TYPES = (
    "fixed",
    "random_discrete",
    "gaussian",
    "ramp",
    "sinusoidal",
    "bimodal",
    "local_bump",
)

SHAPE_TO_ID = {"none": 0, **{name: idx + 1 for idx, name in enumerate(SHAPE_TYPES)}}
ID_TO_SHAPE = {idx: name for name, idx in SHAPE_TO_ID.items()}


@dataclass
class LagInjectionResult:
    dataframe: pd.DataFrame
    metadata: pd.DataFrame
    summary: Dict[str, Any]


def chronological_train_test_split(n_rows: int, split_ratio: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
    if not 0.0 < split_ratio < 1.0:
        raise ValueError("split_ratio must be in (0, 1)")
    n_train = int(n_rows * float(split_ratio))
    if n_train <= 0 or n_train >= n_rows:
        raise ValueError("split_ratio creates an empty train or test split")
    return np.arange(n_train, dtype=np.int64), np.arange(n_train, n_rows, dtype=np.int64)


def gaussian_lag_distribution(center: float, max_lag: int, sigma: float) -> np.ndarray:
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    axis = np.arange(max_lag + 1, dtype=np.float64)
    q = np.exp(-0.5 * ((axis - float(center)) / float(sigma)) ** 2)
    total = q.sum()
    if total <= 0:
        q[0] = 1.0
        return q.astype(np.float32)
    return (q / total).astype(np.float32)


def one_hot_lag_distribution(lag: int, max_lag: int, smoothing: float = 0.0, sigma: float = 1.0) -> np.ndarray:
    lag = int(np.clip(lag, 0, max_lag))
    if smoothing > 0.0:
        return gaussian_lag_distribution(float(lag), max_lag=max_lag, sigma=max(float(sigma), 1e-6))
    q = np.zeros(max_lag + 1, dtype=np.float32)
    q[lag] = 1.0
    return q


def no_lag_distribution(max_lag: int) -> np.ndarray:
    q = np.zeros(max_lag + 1, dtype=np.float32)
    q[0] = 1.0
    return q


def _normalize_weights(weights: Mapping[str, float], allowed: Iterable[str]) -> List[Tuple[str, float]]:
    allowed_set = set(allowed)
    items = [(str(k), float(v)) for k, v in weights.items() if str(k) in allowed_set and float(v) > 0.0]
    if not items:
        items = [(name, 1.0) for name in allowed]
    total = sum(v for _, v in items)
    return [(k, v / total) for k, v in items]


def _choose_shape(rng: np.random.Generator, weights: Mapping[str, float], allowed_shapes: Optional[Sequence[str]]) -> str:
    allowed = [name for name in (allowed_shapes or SHAPE_TYPES) if name in SHAPE_TYPES]
    weighted = _normalize_weights(weights, allowed)
    names = [name for name, _ in weighted]
    probs = [prob for _, prob in weighted]
    return str(rng.choice(names, p=probs))


def _resolve_columns(
    df: pd.DataFrame,
    stage: str,
    feature_spec: Any,
    prefix_by_stage: Mapping[str, str],
    time_col: str,
    target_cols: Sequence[str],
) -> List[str]:
    if stage not in prefix_by_stage:
        raise ValueError(f"Unknown stage: {stage}")
    prefix = prefix_by_stage[stage]
    candidates = [
        col
        for col in df.columns
        if col != time_col and col not in set(target_cols) and col.startswith(prefix)
    ]
    candidates = sorted(candidates)
    if feature_spec in (None, "all"):
        if not candidates:
            raise ValueError(f"No columns found for stage {stage!r} with prefix {prefix!r}")
        return candidates
    if isinstance(feature_spec, str):
        names = [part.strip() for part in feature_spec.split(",") if part.strip()]
        if names and all(name in df.columns for name in names):
            return names
        indices = [int(part) for part in names]
    else:
        indices = [int(idx) for idx in feature_spec]
    out = []
    for idx in indices:
        if idx < 0 or idx >= len(candidates):
            raise IndexError(f"Feature index {idx} is out of range for stage {stage!r}")
        out.append(candidates[idx])
    return out


def _select_injection_positions(
    split_positions: np.ndarray,
    ratio: float,
    max_lag: int,
    granularity: str,
    rng: np.random.Generator,
    block_min_len: int,
    block_max_len: int,
) -> List[np.ndarray]:
    valid = split_positions[split_positions - split_positions[0] >= max_lag]
    if len(valid) == 0 or ratio <= 0.0:
        return []
    target_count = int(round(len(valid) * float(np.clip(ratio, 0.0, 1.0))))
    if target_count <= 0:
        return []

    granularity = str(granularity or "block").lower()
    if granularity == "window":
        chosen = np.sort(rng.choice(valid, size=min(target_count, len(valid)), replace=False))
        return [np.asarray([idx], dtype=np.int64) for idx in chosen]
    if granularity != "block":
        raise ValueError("injection_granularity must be 'block' or 'window'")

    valid_set = set(int(v) for v in valid)
    selected: set[int] = set()
    blocks: List[np.ndarray] = []
    attempts = 0
    while len(selected) < target_count and attempts < max(1000, target_count * 20):
        attempts += 1
        remaining = np.asarray([idx for idx in valid if int(idx) not in selected], dtype=np.int64)
        if len(remaining) == 0:
            break
        start = int(rng.choice(remaining))
        length = int(rng.integers(max(1, block_min_len), max(block_min_len, block_max_len) + 1))
        block = []
        for idx in range(start, min(start + length, int(split_positions[-1]) + 1)):
            if idx in valid_set and idx not in selected:
                block.append(idx)
            if len(selected) + len(block) >= target_count:
                break
        if block:
            selected.update(block)
            blocks.append(np.asarray(block, dtype=np.int64))
    return blocks


def _shape_distributions(
    shape_type: str,
    block_positions: np.ndarray,
    max_lag: int,
    min_lag: int,
    cfg: Mapping[str, Any],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(block_positions)
    if n == 0:
        return np.zeros((0, max_lag + 1), dtype=np.float32), np.zeros(0, dtype=np.int64)

    fixed_lags = [int(v) for v in cfg.get("fixed_lags", [2, 4, 6, 8, 10])]
    fixed_lags = [int(np.clip(v, min_lag, max_lag)) for v in fixed_lags if min_lag <= int(v) <= max_lag]
    if not fixed_lags:
        fixed_lags = [int(np.clip(min_lag, 1, max_lag))]
    sigma_range = cfg.get("gaussian_sigma_range", [0.8, 2.0])
    sigma_low, sigma_high = float(sigma_range[0]), float(sigma_range[1])
    smoothing = float(cfg.get("hard_label_smoothing", 0.0))
    hard_sigma = float(cfg.get("hard_label_smoothing_sigma", 1.0))

    def random_lag() -> int:
        return int(rng.integers(min_lag, max_lag + 1))

    q_rows: List[np.ndarray] = []
    if shape_type == "fixed":
        lag = int(rng.choice(fixed_lags))
        q = one_hot_lag_distribution(lag, max_lag=max_lag, smoothing=smoothing, sigma=hard_sigma)
        q_rows = [q.copy() for _ in range(n)]
    elif shape_type == "random_discrete":
        lag = int(rng.choice(fixed_lags))
        q = one_hot_lag_distribution(lag, max_lag=max_lag, smoothing=smoothing, sigma=hard_sigma)
        q_rows = [q.copy() for _ in range(n)]
    elif shape_type == "gaussian":
        center = float(random_lag())
        sigma = float(rng.uniform(sigma_low, sigma_high))
        q = gaussian_lag_distribution(center, max_lag=max_lag, sigma=sigma)
        q_rows = [q.copy() for _ in range(n)]
    elif shape_type == "ramp":
        start, end = float(random_lag()), float(random_lag())
        if n == 1:
            centers = [start]
        else:
            centers = np.linspace(start, end, n)
        sigma = float(rng.uniform(sigma_low, sigma_high))
        q_rows = [gaussian_lag_distribution(center, max_lag=max_lag, sigma=sigma) for center in centers]
    elif shape_type == "sinusoidal":
        center = float(rng.uniform(min_lag, max_lag))
        amplitude = float(rng.uniform(1.0, max(1.0, (max_lag - min_lag) / 2.0)))
        period = float(rng.uniform(max(4.0, n / 2.0), max(5.0, n * 2.0)))
        sigma = float(rng.uniform(sigma_low, sigma_high))
        for pos in range(n):
            lag = np.clip(center + amplitude * math.sin(2.0 * math.pi * pos / period), min_lag, max_lag)
            q_rows.append(gaussian_lag_distribution(lag, max_lag=max_lag, sigma=sigma))
    elif shape_type == "bimodal":
        lag1 = random_lag()
        lag2 = random_lag()
        if lag1 == lag2:
            lag2 = int(np.clip(lag1 + max(1, (max_lag - min_lag) // 2), min_lag, max_lag))
        sigma1 = float(rng.uniform(sigma_low, sigma_high))
        sigma2 = float(rng.uniform(sigma_low, sigma_high))
        weight = float(rng.uniform(0.35, 0.65))
        q = weight * gaussian_lag_distribution(lag1, max_lag=max_lag, sigma=sigma1)
        q += (1.0 - weight) * gaussian_lag_distribution(lag2, max_lag=max_lag, sigma=sigma2)
        q = (q / q.sum()).astype(np.float32)
        q_rows = [q.copy() for _ in range(n)]
    elif shape_type == "local_bump":
        strong_lag = int(rng.choice(fixed_lags))
        weak_lag = int(np.clip(cfg.get("local_bump_weak_lag", min_lag), min_lag, max_lag))
        sigma = float(rng.uniform(sigma_low, sigma_high))
        q_weak = gaussian_lag_distribution(weak_lag, max_lag=max_lag, sigma=max(sigma, 1.0))
        q_bump = gaussian_lag_distribution(strong_lag, max_lag=max_lag, sigma=sigma)
        lo = int(round(n * 0.35))
        hi = max(lo + 1, int(round(n * 0.65)))
        for pos in range(n):
            q_rows.append(q_bump.copy() if lo <= pos < hi else q_weak.copy())
    else:
        raise ValueError(f"Unknown shape_type: {shape_type}")

    q_arr = np.stack(q_rows).astype(np.float32)
    hard = q_arr.argmax(axis=1).astype(np.int64)
    return q_arr, hard


def _delayed_mixture(source: np.ndarray, t: int, q: np.ndarray) -> float:
    acc = 0.0
    for lag, weight in enumerate(q):
        if weight <= 0.0:
            continue
        src_idx = t - lag
        if src_idx < 0:
            continue
        acc += float(weight) * float(source[src_idx])
    return acc


def _inject_columns(
    df: pd.DataFrame,
    positions: np.ndarray,
    q_rows: np.ndarray,
    source_cols: Sequence[str],
    target_cols: Sequence[str],
    inject_strength: float,
) -> None:
    if len(positions) == 0:
        return
    rho = float(np.clip(inject_strength, 0.0, 1.0))
    if rho <= 0.0:
        return

    source_stats = {}
    for col in source_cols:
        values = df[col].astype(float).to_numpy()
        std = float(np.nanstd(values))
        source_stats[col] = (float(np.nanmean(values)), std if std > 1e-12 else 1.0, values)
    target_stats = {}
    for col in target_cols:
        values = df[col].astype(float).to_numpy()
        std = float(np.nanstd(values))
        target_stats[col] = (float(np.nanmean(values)), std if std > 1e-12 else 1.0)

    for target_idx, target_col in enumerate(target_cols):
        source_col = source_cols[target_idx % len(source_cols)]
        src_mean, src_std, src_values = source_stats[source_col]
        tgt_mean, tgt_std = target_stats[target_col]
        target_values = df[target_col].astype(float).to_numpy(copy=True)
        source_z = (src_values - src_mean) / src_std
        for row_pos, q in zip(positions, q_rows):
            t = int(row_pos)
            lagged_z = _delayed_mixture(source_z, t, q)
            lagged_in_target_scale = tgt_mean + tgt_std * lagged_z
            old = float(target_values[t])
            target_values[t] = (1.0 - rho) * old + rho * lagged_in_target_scale
        df[target_col] = target_values


def inject_lag_into_dataframe(
    df: pd.DataFrame,
    time_col: str,
    target_cols: Sequence[str],
    lag_injection_cfg: Mapping[str, Any],
    lag_edges: Optional[Sequence[Mapping[str, Any]]] = None,
    prefix_by_stage: Optional[Mapping[str, str]] = None,
) -> LagInjectionResult:
    cfg = dict(lag_injection_cfg or {})
    prefix_by_stage = dict(prefix_by_stage or EDGE_STAGE_PREFIX)
    if not bool(cfg.get("enabled", True)):
        metadata = pd.DataFrame()
        return LagInjectionResult(dataframe=df.copy(), metadata=metadata, summary={})

    out = df.sort_values(time_col).reset_index(drop=True).copy()
    n_rows = len(out)
    max_lag = int(cfg.get("max_lag", 12))
    min_lag = int(cfg.get("min_lag", 1))
    if min_lag < 1 or max_lag < min_lag:
        raise ValueError("Expected 1 <= min_lag <= max_lag")

    rng = np.random.default_rng(int(cfg.get("random_seed", 42)))
    train_pos, test_pos = chronological_train_test_split(n_rows, float(cfg.get("split_ratio", 0.8)))
    train_val_ratio = float(cfg.get("train_val_ratio", 0.875))
    n_model_train = max(1, int(len(train_pos) * train_val_ratio))
    model_split = np.asarray(["test"] * n_rows, dtype=object)
    model_split[train_pos[:n_model_train]] = "train"
    model_split[train_pos[n_model_train:]] = "val"
    out["model_split"] = model_split

    lag_split = np.asarray(["test"] * n_rows, dtype=object)
    lag_split[train_pos] = "train"
    out["split"] = lag_split

    edges = list(lag_edges or cfg.get("lag_edges") or [])
    if not edges:
        edges = [
            {
                "name": "stage1_to_stage2",
                "source_stage": "stage1",
                "target_stage": "stage2",
                "source_features": "all",
                "target_features": "all",
            }
        ]

    metadata_rows: List[Dict[str, Any]] = []
    summary_frames = []
    for edge_cfg in edges:
        edge_name = str(edge_cfg.get("name") or f"{edge_cfg['source_stage']}_to_{edge_cfg['target_stage']}")
        source_stage = str(edge_cfg.get("source_stage", "stage1"))
        target_stage = str(edge_cfg.get("target_stage", "stage2"))
        source_cols = _resolve_columns(
            out,
            source_stage,
            edge_cfg.get("source_features", "all"),
            prefix_by_stage,
            time_col,
            target_cols,
        )
        target_stage_cols = _resolve_columns(
            out,
            target_stage,
            edge_cfg.get("target_features", "all"),
            prefix_by_stage,
            time_col,
            target_cols,
        )
        strength_range = edge_cfg.get("inject_strength_range", cfg.get("inject_strength_range", None))
        if strength_range is not None:
            inject_strength = float(rng.uniform(float(strength_range[0]), float(strength_range[1])))
        else:
            inject_strength = float(edge_cfg.get("inject_strength", cfg.get("inject_strength", 0.5)))

        q_all = np.zeros((n_rows, max_lag + 1), dtype=np.float32)
        q_all[:, 0] = 1.0
        lag_flag = np.zeros(n_rows, dtype=np.int64)
        lag_value = np.zeros(n_rows, dtype=np.int64)
        shape_type = np.asarray(["none"] * n_rows, dtype=object)
        shape_id = np.zeros(n_rows, dtype=np.int64)
        injected_rows: Dict[int, Tuple[np.ndarray, int, str]] = {}

        for split_name, split_positions, ratio_key in [
            ("train", train_pos, "injection_ratio_train"),
            ("test", test_pos, "injection_ratio_test"),
        ]:
            ratio = float(cfg.get(ratio_key, cfg.get("injection_ratio", 0.6)))
            blocks = _select_injection_positions(
                split_positions=split_positions,
                ratio=ratio,
                max_lag=max_lag,
                granularity=str(cfg.get("injection_granularity", "block")),
                rng=rng,
                block_min_len=int(cfg.get("block_min_len", max(8, max_lag + 1))),
                block_max_len=int(cfg.get("block_max_len", max(24, 2 * max_lag + 1))),
            )
            allowed_shapes = cfg.get(f"{split_name}_shapes")
            for block in blocks:
                shape = _choose_shape(rng, cfg.get("shapes", {}), allowed_shapes)
                q_rows, hard_rows = _shape_distributions(shape, block, max_lag, min_lag, cfg, rng)
                _inject_columns(out, block, q_rows, source_cols, target_stage_cols, inject_strength)
                for row_pos, q, hard_lag in zip(block, q_rows, hard_rows):
                    q_all[int(row_pos)] = q
                    lag_flag[int(row_pos)] = 1 if int(hard_lag) > 0 or q[1:].sum() > 1e-6 else 0
                    lag_value[int(row_pos)] = int(hard_lag)
                    shape_type[int(row_pos)] = shape if lag_flag[int(row_pos)] else "none"
                    shape_id[int(row_pos)] = SHAPE_TO_ID[str(shape_type[int(row_pos)])]
                    injected_rows[int(row_pos)] = (q, int(hard_lag), str(shape_type[int(row_pos)]))

        out[f"{edge_name}_lag_flag"] = lag_flag
        out[f"{edge_name}_lag_gt"] = lag_value
        out[f"{edge_name}_lag_expected_gt"] = (q_all * np.arange(max_lag + 1, dtype=np.float32)[None, :]).sum(axis=1)
        out[f"{edge_name}_true_expected_lag"] = out[f"{edge_name}_lag_expected_gt"]
        out[f"{edge_name}_true_argmax_lag"] = lag_value
        out[f"{edge_name}_shape_id"] = shape_id
        out[f"{edge_name}_shape_type"] = shape_type
        out["lag_gt"] = lag_value
        out["lag_expected_gt"] = out[f"{edge_name}_lag_expected_gt"]
        out["lag_flag"] = lag_flag
        out["shape_id"] = shape_id
        out["shape_type"] = shape_type
        for lag in range(max_lag + 1):
            out[f"{edge_name}_true_pi_lag{lag}"] = q_all[:, lag]

        for row_idx in range(n_rows):
            row = {
                "row_index": row_idx,
                "TimeStamp": out[time_col].iloc[row_idx],
                "split": lag_split[row_idx],
                "model_split": model_split[row_idx],
                "valid_for_injection": int(
                    (row_idx - int(train_pos[0]) >= max_lag)
                    if lag_split[row_idx] == "train"
                    else (row_idx - int(test_pos[0]) >= max_lag)
                ),
                "lag_flag": int(lag_flag[row_idx]),
                "lag_value": int(lag_value[row_idx]),
                "lag_expected": float(out[f"{edge_name}_lag_expected_gt"].iloc[row_idx]),
                "shape_type": str(shape_type[row_idx]),
                "shape_id": int(shape_id[row_idx]),
                "source_stage": source_stage,
                "target_stage": target_stage,
                "source_feature_id": ",".join(str(source_cols.index(col)) for col in source_cols),
                "target_feature_id": ",".join(str(target_stage_cols.index(col)) for col in target_stage_cols),
                "edge_name": edge_name,
                "inject_strength": float(inject_strength),
            }
            for lag in range(max_lag + 1):
                row[f"lag_soft_{lag}"] = float(q_all[row_idx, lag])
            metadata_rows.append(row)
        summary_frames.append(pd.DataFrame({"split": lag_split, "lag_flag": lag_flag, "lag_value": lag_value, "shape_type": shape_type}))

    metadata = pd.DataFrame(metadata_rows)
    summary = summarize_lag_injection(metadata, max_lag=max_lag, save_dir=cfg.get("summary_dir"))
    return LagInjectionResult(dataframe=out, metadata=metadata, summary=summary)


def inject_lag_csv(
    input_csv: str | Path,
    output_csv: str | Path,
    metadata_csv: str | Path,
    time_col: str,
    target_cols: Sequence[str],
    lag_injection_cfg: Mapping[str, Any],
    lag_edges: Optional[Sequence[Mapping[str, Any]]] = None,
    prefix_by_stage: Optional[Mapping[str, str]] = None,
) -> LagInjectionResult:
    df = pd.read_csv(input_csv)
    result = inject_lag_into_dataframe(
        df,
        time_col=time_col,
        target_cols=target_cols,
        lag_injection_cfg=lag_injection_cfg,
        lag_edges=lag_edges,
        prefix_by_stage=prefix_by_stage,
    )
    output_csv = Path(output_csv)
    metadata_csv = Path(metadata_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    metadata_csv.parent.mkdir(parents=True, exist_ok=True)
    result.dataframe.to_csv(output_csv, index=False)
    result.metadata.to_csv(metadata_csv, index=False)
    return result


def summarize_lag_injection(
    metadata: pd.DataFrame,
    max_lag: Optional[int] = None,
    save_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    if metadata.empty:
        summary = {"empty": True}
    else:
        soft_cols = sorted(
            [col for col in metadata.columns if col.startswith("lag_soft_")],
            key=lambda name: int(name.rsplit("_", 1)[-1]),
        )
        soft = metadata[soft_cols].to_numpy(dtype=np.float64) if soft_cols else np.zeros((len(metadata), 0))
        sums = soft.sum(axis=1) if soft.size else np.array([])
        no_lag_rows = metadata["lag_flag"].astype(int).eq(0).to_numpy()
        lag_values = metadata["lag_value"].astype(int).to_numpy()
        split_summary = {}
        for split_name, split_df in metadata.groupby("split"):
            if "valid_for_injection" in split_df.columns:
                denom_df = split_df.loc[split_df["valid_for_injection"].astype(int).eq(1)]
                if denom_df.empty:
                    denom_df = split_df
            else:
                denom_df = split_df
            split_summary[str(split_name)] = {
                "n_rows": int(len(split_df)),
                "n_valid_for_injection": int(len(denom_df)),
                "lag_flag_ratio": float(denom_df["lag_flag"].astype(float).mean()),
            }
        boundary_leakage = False
        if {"split", "row_index"}.issubset(metadata.columns):
            for _, row in metadata.loc[metadata["lag_flag"].astype(int).eq(1)].iterrows():
                lag = int(row["lag_value"])
                if lag <= 0:
                    continue
                src_idx = int(row["row_index"]) - lag
                if src_idx < 0:
                    boundary_leakage = True
                    break
                src_split = metadata.loc[metadata["row_index"].eq(src_idx), "split"]
                if src_split.empty or str(src_split.iloc[0]) != str(row["split"]):
                    boundary_leakage = True
                    break
        summary = {
            "split_summary": split_summary,
            "lag_value_counts": {str(k): int(v) for k, v in metadata["lag_value"].value_counts().sort_index().items()},
            "shape_type_counts": {str(k): int(v) for k, v in metadata["shape_type"].value_counts().sort_index().items()},
            "lag_soft_shape": list(soft.shape),
            "lag_soft_rows_sum_to_1": bool(np.allclose(sums, 1.0, atol=1e-5)) if soft.size else False,
            "no_lag_q0_is_1": bool(np.allclose(soft[no_lag_rows, 0], 1.0, atol=1e-5)) if soft.size and np.any(no_lag_rows) else True,
            "has_out_of_range_lag": bool(np.any(lag_values < 0) or (max_lag is not None and np.any(lag_values > int(max_lag)))),
            "has_train_test_boundary_leakage": bool(boundary_leakage),
        }

    printable = {
        "train_lag_flag_ratio": summary.get("split_summary", {}).get("train", {}).get("lag_flag_ratio"),
        "test_lag_flag_ratio": summary.get("split_summary", {}).get("test", {}).get("lag_flag_ratio"),
        "lag_value_counts": summary.get("lag_value_counts"),
        "shape_type_counts": summary.get("shape_type_counts"),
        "lag_soft_shape": summary.get("lag_soft_shape"),
        "lag_soft_rows_sum_to_1": summary.get("lag_soft_rows_sum_to_1"),
        "no_lag_q0_is_1": summary.get("no_lag_q0_is_1"),
        "has_out_of_range_lag": summary.get("has_out_of_range_lag"),
        "has_train_test_boundary_leakage": summary.get("has_train_test_boundary_leakage"),
    }
    print(json.dumps(printable, ensure_ascii=False, indent=2))

    if save_dir is not None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        (save_path / "lag_injection_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        rows = []
        for split_name, values in summary.get("split_summary", {}).items():
            rows.append({"split": split_name, **values})
        if rows:
            pd.DataFrame(rows).to_csv(save_path / "lag_injection_summary.csv", index=False)
        else:
            pd.DataFrame([summary]).to_csv(save_path / "lag_injection_summary.csv", index=False)
    return summary
