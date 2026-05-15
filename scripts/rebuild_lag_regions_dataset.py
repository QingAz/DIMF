#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild train/test lag regimes as distributed sinusoidal/constant regions."
    )
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--test", type=Path, default=Path("data/test.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/lag_regions"))
    parser.add_argument("--time-col", default="TimeStamp")
    parser.add_argument("--target-col", default="yield_flow")
    parser.add_argument("--source-prefix", default="stage1_")
    parser.add_argument("--target-prefix", default="stage2_")
    parser.add_argument("--train-regions", type=int, default=10)
    parser.add_argument("--test-regions", type=int, default=5)
    parser.add_argument(
        "--active-ratio",
        type=float,
        default=0.6,
        help="Fraction of each coarse region used as the lag-active block.",
    )
    parser.add_argument("--sin-max-lag", type=int, default=5)
    parser.add_argument("--const-max-lag", type=int, default=7)
    parser.add_argument("--rho", type=float, default=0.55)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _stage_cols(df: pd.DataFrame, prefix: str, excluded: Iterable[str]) -> List[str]:
    excluded = set(excluded)
    cols = [col for col in df.columns if col.startswith(prefix) and col not in excluded]
    if not cols:
        raise ValueError(f"No columns found with prefix {prefix!r}")
    return sorted(cols)


def _build_region_plan(n_rows: int, n_regions: int, active_ratio: float) -> List[Tuple[int, int, int]]:
    if n_regions <= 0:
        raise ValueError("n_regions must be positive")
    if not 0.0 < active_ratio <= 1.0:
        raise ValueError("active_ratio must be in (0, 1]")
    edges = np.linspace(0, n_rows, n_regions + 1, dtype=int)
    plan = []
    for region_idx in range(n_regions):
        lo, hi = int(edges[region_idx]), int(edges[region_idx + 1])
        width = max(hi - lo, 1)
        active_width = max(1, int(round(width * active_ratio)))
        start = lo + max(0, (width - active_width) // 2)
        end = min(start + active_width, hi)
        plan.append((region_idx, start, end))
    return plan


def _segment_ids_from_region(region_id: np.ndarray) -> np.ndarray:
    """Build contiguous segment IDs from generated lag-region membership."""
    region = np.asarray(region_id, dtype=np.int64)
    if region.size == 0:
        return region
    boundary = np.ones(region.shape[0], dtype=bool)
    boundary[1:] = region[1:] != region[:-1]
    return np.cumsum(boundary, dtype=np.int64) - 1


def _sinusoidal_lag(length: int, max_lag: int, phase: float = 0.0) -> np.ndarray:
    if length <= 0:
        return np.zeros(0, dtype=np.int64)
    x = np.linspace(0.0, 2.0 * np.pi, length, endpoint=False) + float(phase)
    values = 0.5 * (np.sin(x) + 1.0) * float(max_lag)
    return np.rint(values).astype(np.int64).clip(0, max_lag)


def _constant_lag(region_idx: int, max_lag: int) -> int:
    return int(region_idx % (max_lag + 1))


def _lagged_value(source: np.ndarray, row_idx: int, lag: int) -> float:
    src_idx = max(0, int(row_idx) - int(lag))
    return float(source[src_idx])


def _apply_lag_mixing(
    df: pd.DataFrame,
    lag: np.ndarray,
    source_cols: List[str],
    target_cols: List[str],
    rho: float,
) -> pd.DataFrame:
    out = df.copy()
    rho = float(np.clip(rho, 0.0, 1.0))
    source_stats: Dict[str, Tuple[float, float, np.ndarray]] = {}
    for col in source_cols:
        values = out[col].astype(float).to_numpy()
        std = float(np.nanstd(values))
        source_stats[col] = (float(np.nanmean(values)), std if std > 1e-12 else 1.0, values)
    target_stats: Dict[str, Tuple[float, float]] = {}
    for col in target_cols:
        values = out[col].astype(float).to_numpy()
        std = float(np.nanstd(values))
        target_stats[col] = (float(np.nanmean(values)), std if std > 1e-12 else 1.0)

    active_rows = np.flatnonzero(lag > 0)
    for target_idx, target_col in enumerate(target_cols):
        source_col = source_cols[target_idx % len(source_cols)]
        src_mean, src_std, src_values = source_stats[source_col]
        tgt_mean, tgt_std = target_stats[target_col]
        source_z = (src_values - src_mean) / src_std
        target_values = out[target_col].astype(float).to_numpy(copy=True)
        for row_idx in active_rows:
            lagged_z = _lagged_value(source_z, int(row_idx), int(lag[row_idx]))
            lagged_scaled = tgt_mean + tgt_std * lagged_z
            target_values[row_idx] = (1.0 - rho) * target_values[row_idx] + rho * lagged_scaled
        out[target_col] = target_values
    return out


def rebuild_split(
    df: pd.DataFrame,
    split_name: str,
    n_regions: int,
    active_ratio: float,
    sin_max_lag: int,
    const_max_lag: int,
    source_cols: List[str],
    target_cols: List[str],
    rho: float,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy().reset_index(drop=True)
    n_rows = len(out)
    lag = np.zeros(n_rows, dtype=np.int64)
    inject_flag = np.zeros(n_rows, dtype=np.int64)
    shape = np.asarray(["none"] * n_rows, dtype=object)
    pattern = np.asarray(["none"] * n_rows, dtype=object)
    region_id = np.full(n_rows, -1, dtype=np.int64)

    rows = []
    for region_idx, start, end in _build_region_plan(n_rows, n_regions, active_ratio):
        length = max(end - start, 0)
        if length <= 0:
            continue
        if region_idx % 2 == 0:
            phase = float(rng.uniform(0.0, 2.0 * np.pi))
            local_lag = _sinusoidal_lag(length, max_lag=sin_max_lag, phase=phase)
            local_shape = "sinusoidal"
            local_pattern = f"sin_0_{sin_max_lag}"
        else:
            const_lag = _constant_lag(region_idx, const_max_lag)
            local_lag = np.full(length, const_lag, dtype=np.int64)
            local_shape = "fixed"
            local_pattern = f"const_{const_lag}"
        lag[start:end] = local_lag
        inject_flag[start:end] = (local_lag > 0).astype(np.int64)
        shape[start:end] = np.where(local_lag > 0, local_shape, "none")
        pattern[start:end] = np.where(local_lag > 0, local_pattern, "none")
        region_id[start:end] = region_idx
        rows.append(
            {
                "split": split_name,
                "region_id": region_idx,
                "start_row": int(start),
                "end_row_exclusive": int(end),
                "n_rows": int(length),
                "shape": local_shape,
                "pattern": local_pattern,
                "lag_min": int(local_lag.min()),
                "lag_max": int(local_lag.max()),
                "positive_rows": int((local_lag > 0).sum()),
            }
        )

    out = _apply_lag_mixing(out, lag, source_cols, target_cols, rho)
    out["lag_gt"] = lag
    out["lag_binary_gt"] = (lag > 0).astype(np.int64)
    out["inject_flag"] = inject_flag
    out["bump_dmax_gt"] = 0
    out["segment_dmax_gt"] = lag
    out["g_stage1_to_stage2"] = lag
    out["lag_shape_gt"] = shape
    out["lag_pattern_gt"] = pattern
    out["region_id"] = region_id
    out["tile_id"] = region_id
    out["segment_id"] = _segment_ids_from_region(region_id)
    out["split"] = split_name
    return out, pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(int(args.seed))
    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)
    excluded = {
        args.time_col,
        args.target_col,
        "split",
        "model_split",
        "is_interpolated",
        "segment_id",
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
    source_cols = _stage_cols(train_df, args.source_prefix, excluded)
    target_cols = _stage_cols(train_df, args.target_prefix, excluded)

    train_out, train_regions = rebuild_split(
        train_df,
        split_name="train",
        n_regions=args.train_regions,
        active_ratio=args.active_ratio,
        sin_max_lag=args.sin_max_lag,
        const_max_lag=args.const_max_lag,
        source_cols=source_cols,
        target_cols=target_cols,
        rho=args.rho,
        rng=rng,
    )
    test_out, test_regions = rebuild_split(
        test_df,
        split_name="test",
        n_regions=args.test_regions,
        active_ratio=args.active_ratio,
        sin_max_lag=args.sin_max_lag,
        const_max_lag=args.const_max_lag,
        source_cols=source_cols,
        target_cols=target_cols,
        rho=args.rho,
        rng=rng,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.csv"
    test_path = args.output_dir / "test.csv"
    train_out.to_csv(train_path, index=False)
    test_out.to_csv(test_path, index=False)
    regions = pd.concat([train_regions, test_regions], ignore_index=True)
    regions.to_csv(args.output_dir / "lag_regions.csv", index=False)

    summary = {
        "train_path": str(train_path),
        "test_path": str(test_path),
        "source_cols": source_cols,
        "target_cols": target_cols,
        "train_rows": int(len(train_out)),
        "test_rows": int(len(test_out)),
        "train_positive_ratio": float((train_out["lag_gt"].astype(int) > 0).mean()),
        "test_positive_ratio": float((test_out["lag_gt"].astype(int) > 0).mean()),
        "active_ratio": float(args.active_ratio),
        "sin_max_lag": int(args.sin_max_lag),
        "const_max_lag": int(args.const_max_lag),
        "rho": float(args.rho),
        "seed": int(args.seed),
        "train_lag_counts": {str(k): int(v) for k, v in train_out["lag_gt"].value_counts().sort_index().items()},
        "test_lag_counts": {str(k): int(v) for k, v in test_out["lag_gt"].value_counts().sort_index().items()},
        "train_segment_count": int(train_out["segment_id"].nunique()),
        "test_segment_count": int(test_out["segment_id"].nunique()),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
