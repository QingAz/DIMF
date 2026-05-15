#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a lag-injected dataset where y has a one-to-one lag percentage response."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/lag_regions_legacy622_lag5"),
        help="Directory containing train.csv and test.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/lag_regions_legacy622_lag5_y_lagmap_pct"),
        help="Directory to write the mapped dataset.",
    )
    parser.add_argument("--target-col", default="yield_flow")
    parser.add_argument("--lag-col", default="lag_gt")
    parser.add_argument(
        "--pct-per-lag",
        type=float,
        default=0.01,
        help="Per-lag y multiplier step. Default maps lag 3 to +3%% and lag 5 to +5%%.",
    )
    parser.add_argument(
        "--max-lag",
        type=int,
        default=5,
        help="Maximum lag to include in the mapping.",
    )
    return parser.parse_args()


def build_mapping(max_lag: int, pct_per_lag: float) -> Dict[int, float]:
    if max_lag < 0:
        raise ValueError("max_lag must be non-negative")
    return {lag: float(lag) * float(pct_per_lag) for lag in range(max_lag + 1)}


def transform_split(
    input_csv: Path,
    output_csv: Path,
    target_col: str,
    lag_col: str,
    mapping: Dict[int, float],
) -> Dict[str, object]:
    df = pd.read_csv(input_csv)
    missing = [col for col in (target_col, lag_col) if col not in df.columns]
    if missing:
        raise ValueError(f"{input_csv} is missing required columns: {missing}")

    original_y = df[target_col].astype(float)
    lag_values = df[lag_col].fillna(0).astype(int)
    pct = lag_values.map(mapping).fillna(0.0).astype(float)
    multiplier = 1.0 + pct
    mapped_y = original_y * multiplier

    df[f"{target_col}_original"] = original_y
    df["y_lag_response_pct"] = pct
    df["y_lag_response_percent"] = pct * 100.0
    df["y_lag_response_multiplier"] = multiplier
    df["y_lag_response_delta"] = mapped_y - original_y
    df[target_col] = mapped_y

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    by_lag = {}
    for lag, group in df.groupby(lag_col):
        lag_int = int(lag)
        by_lag[str(lag_int)] = {
            "n_rows": int(len(group)),
            "response_percent": float(mapping.get(lag_int, 0.0) * 100.0),
            "mean_y_original": float(group[f"{target_col}_original"].mean()),
            "mean_y_mapped": float(group[target_col].mean()),
            "mean_delta": float(group["y_lag_response_delta"].mean()),
        }

    return {
        "rows": int(len(df)),
        "target_col": target_col,
        "lag_col": lag_col,
        "mean_y_original": float(original_y.mean()),
        "mean_y_mapped": float(mapped_y.mean()),
        "mean_delta": float((mapped_y - original_y).mean()),
        "by_lag": by_lag,
    }


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    mapping = build_mapping(max_lag=int(args.max_lag), pct_per_lag=float(args.pct_per_lag))

    if not input_dir.exists():
        raise FileNotFoundError(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_summary = {}
    for split in ("train", "test"):
        split_summary[split] = transform_split(
            input_csv=input_dir / f"{split}.csv",
            output_csv=output_dir / f"{split}.csv",
            target_col=str(args.target_col),
            lag_col=str(args.lag_col),
            mapping=mapping,
        )

    for name in ("lag_regions.csv",):
        src = input_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)

    summary = {
        "source_dataset": str(input_dir),
        "output_dataset": str(output_dir),
        "rule": "yield_flow := yield_flow_original * (1 + lag_gt * pct_per_lag)",
        "pct_per_lag": float(args.pct_per_lag),
        "lag_to_y_response_percent": {str(k): float(v * 100.0) for k, v in mapping.items()},
        "splits": split_summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output_dir / "README_y_lagmap.md").write_text(
        "# Lag-Mapped Y Target Dataset\n\n"
        "This dataset keeps the existing stage1-to-stage2 lag injection and adds a one-to-one y response.\n\n"
        "Rule:\n\n"
        "```text\n"
        f"{args.target_col} = {args.target_col}_original * (1 + {args.lag_col} * {float(args.pct_per_lag)})\n"
        "```\n\n"
        "With the default mapping, lag 0 is unchanged, lag 3 maps to +3%, and lag 5 maps to +5%.\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
