#!/usr/bin/env python3

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


PEAK_COLUMNS = [
    "block_id",
    "start_time",
    "end_time",
    "n_samples",
    "true_peak_lag",
    "true_dmax",
    "aligned_peak_expected_lag",
    "aligned_peak_rounded_lag",
    "aligned_peak_error",
    "aligned_peak_hit_at_0",
    "aligned_peak_hit_at_pm1",
    "noalign_peak_expected_lag",
    "noalign_peak_rounded_lag",
    "noalign_peak_error",
    "noalign_peak_hit_at_0",
    "noalign_peak_hit_at_pm1",
]


def _as_optional_float(value) -> Optional[float]:
    if value is None:
        return None
    if pd.isna(value):
        return None
    return float(value)


def _positive_lag_blocks(joined: pd.DataFrame) -> List[pd.DataFrame]:
    blocks: List[pd.DataFrame] = []
    in_block = False
    start_idx = 0
    lag_positive = joined["lag_gt"].to_numpy(dtype=float) > 0

    for idx, is_positive in enumerate(lag_positive):
        if is_positive and not in_block:
            start_idx = idx
            in_block = True
        elif not is_positive and in_block:
            blocks.append(joined.iloc[start_idx:idx].copy())
            in_block = False
    if in_block:
        blocks.append(joined.iloc[start_idx:].copy())
    return blocks


def _rounded_lag(value: float) -> int:
    return int(np.floor(float(value) + 0.5))


def build_peak_block_table(joined: pd.DataFrame) -> pd.DataFrame:
    if joined.empty:
        return pd.DataFrame(columns=PEAK_COLUMNS)

    working = joined.copy()
    if "TimeStamp" in working.columns:
        working["_sort_time"] = pd.to_datetime(working["TimeStamp"])
        working = working.sort_values("_sort_time").drop(columns=["_sort_time"]).reset_index(drop=True)
    else:
        working = working.reset_index(drop=True)

    rows: List[Dict[str, object]] = []
    for block_id, block in enumerate(_positive_lag_blocks(working), start=1):
        true_peak = float(block["lag_gt"].max())
        if "segment_dmax_gt" in block.columns and block["segment_dmax_gt"].notna().any():
            true_dmax = int(block["segment_dmax_gt"].max())
        else:
            true_dmax = int(true_peak)

        row: Dict[str, object] = {
            "block_id": block_id,
            "start_time": str(block["TimeStamp"].iloc[0]) if "TimeStamp" in block.columns else "",
            "end_time": str(block["TimeStamp"].iloc[-1]) if "TimeStamp" in block.columns else "",
            "n_samples": int(len(block)),
            "true_peak_lag": true_peak,
            "true_dmax": true_dmax,
        }

        for model in ("aligned", "noalign"):
            peak_value = float(block["%s_pred_expected_lag" % model].max())
            rounded = _rounded_lag(peak_value)
            error = abs(peak_value - true_peak)
            row["%s_peak_expected_lag" % model] = peak_value
            row["%s_peak_rounded_lag" % model] = rounded
            row["%s_peak_error" % model] = error
            row["%s_peak_hit_at_0" % model] = int(rounded == int(true_peak))
            row["%s_peak_hit_at_pm1" % model] = int(abs(rounded - int(true_peak)) <= 1)

        rows.append(row)

    return pd.DataFrame(rows, columns=PEAK_COLUMNS)


def summarize_peak_blocks(blocks: pd.DataFrame, model: str) -> Dict[str, Optional[float]]:
    if blocks.empty:
        return {
            "n_blocks": 0,
            "peak_error": None,
            "peak_hit_at_0": None,
            "peak_hit_at_pm1": None,
            "mean_true_peak_lag": None,
            "mean_pred_peak_expected_lag": None,
            "mean_pred_peak_rounded_lag": None,
        }

    return {
        "n_blocks": int(len(blocks)),
        "peak_error": _as_optional_float(blocks["%s_peak_error" % model].mean()),
        "peak_hit_at_0": _as_optional_float(blocks["%s_peak_hit_at_0" % model].mean()),
        "peak_hit_at_pm1": _as_optional_float(blocks["%s_peak_hit_at_pm1" % model].mean()),
        "mean_true_peak_lag": _as_optional_float(blocks["true_peak_lag"].mean()),
        "mean_pred_peak_expected_lag": _as_optional_float(blocks["%s_peak_expected_lag" % model].mean()),
        "mean_pred_peak_rounded_lag": _as_optional_float(blocks["%s_peak_rounded_lag" % model].mean()),
    }


def attach_peak_metrics(summary: Dict, joined: pd.DataFrame) -> pd.DataFrame:
    peak_blocks = build_peak_block_table(joined)
    benchmark = summary.setdefault("benchmark", {})
    benchmark["peak"] = {
        "aligned": summarize_peak_blocks(peak_blocks, "aligned"),
        "noalign": summarize_peak_blocks(peak_blocks, "noalign"),
    }

    by_dmax = summary.setdefault("benchmark_by_dmax", {})
    for dmax_key, item in by_dmax.items():
        try:
            dmax_value = int(dmax_key)
        except ValueError:
            continue
        dmax_blocks = peak_blocks[peak_blocks["true_dmax"] == dmax_value] if not peak_blocks.empty else peak_blocks
        item.setdefault("aligned", {})["peak"] = summarize_peak_blocks(dmax_blocks, "aligned")
        item.setdefault("noalign", {})["peak"] = summarize_peak_blocks(dmax_blocks, "noalign")

    return peak_blocks
