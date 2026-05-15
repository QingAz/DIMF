#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.postprocess.viterbi_lag_decoder import viterbi_decode_lag


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep Viterbi/mode-filter lag postprocess settings on saved predictions.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def pi_from_frame(frame: pd.DataFrame) -> np.ndarray:
    pi_cols = [col for col in frame.columns if col.startswith("pi_") and col.endswith("_nosmooth")]
    pi_cols = sorted(pi_cols, key=lambda item: int(item.split("_")[1]))
    if not pi_cols:
        raise ValueError("No pi_*_nosmooth columns found")
    pi = frame[pi_cols].astype(float).to_numpy()
    pi = np.clip(pi, 0.0, None)
    return pi / np.clip(pi.sum(axis=1, keepdims=True), 1e-12, None)


def _mode_filter_by_segment(values: np.ndarray, segment_id: np.ndarray, width: int, n_states: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    out = values.copy()
    half = int(width) // 2
    for seg in np.unique(segment_id):
        idx = np.flatnonzero(segment_id == seg)
        vals = values[idx]
        filtered = vals.copy()
        for j in range(len(vals)):
            lo = max(0, j - half)
            hi = min(len(vals), j + half + 1)
            win = vals[lo:hi]
            counts = np.bincount(win, minlength=n_states)
            modes = np.flatnonzero(counts == counts.max())
            med = np.median(win)
            filtered[j] = int(modes[np.argmin(np.abs(modes - med))])
        out[idx] = filtered
    return out


def _metrics(pred: np.ndarray, gt: np.ndarray, sample_index: np.ndarray, segment_id: np.ndarray, prefix: str) -> dict:
    pred = np.asarray(pred, dtype=np.float64)
    pred_int = np.rint(pred).astype(np.int64)
    gt = np.asarray(gt, dtype=np.int64)
    no_lag = gt == 0
    positive = gt > 0
    out = {
        f"{prefix}mae_all": float(np.abs(pred - gt).mean()),
        f"{prefix}mae_injected": float(np.abs(pred[positive] - gt[positive]).mean()),
        f"{prefix}mae_no_lag": float(np.abs(pred[no_lag] - gt[no_lag]).mean()),
        f"{prefix}accuracy_all": float((pred_int == gt).mean()),
        f"{prefix}accuracy_injected": float((pred_int[positive] == gt[positive]).mean()),
        f"{prefix}no_lag_false_alarm_rate": float((pred_int[no_lag] > 0).mean()),
    }
    ranges = [(0, 1200, "first_0_1200"), (3000, 4000, "second_3000_4000"), (5800, 6500, "last_5800_6500")]
    for lo, hi, name in ranges:
        mask = (sample_index >= lo) & (sample_index <= hi)
        out[f"{prefix}{name}_mae"] = float(np.abs(pred[mask] - gt[mask]).mean())
        out[f"{prefix}{name}_accuracy"] = float((pred_int[mask] == gt[mask]).mean())
    seg5 = segment_id == 5
    out[f"{prefix}segment5_mae"] = float(np.abs(pred[seg5] - gt[seg5]).mean())
    out[f"{prefix}segment5_accuracy"] = float((pred_int[seg5] == gt[seg5]).mean())
    return out


def summarize(name: str, pred: np.ndarray, frame: pd.DataFrame, params: dict) -> dict:
    gt = frame["lag_gt"].astype(int).to_numpy()
    sample_index = frame["sample_index"].astype(int).to_numpy()
    segment_id = frame["segment_id"].astype(int).to_numpy()
    metrics = _metrics(pred, gt, sample_index, segment_id, "")
    return {
        "name": name,
        **params,
        **metrics,
    }


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.predictions)
    pi = pi_from_frame(frame)
    gt = frame["lag_gt"].astype(int).to_numpy()
    segment_id = frame["segment_id"].astype(int).to_numpy()
    sample_index = frame["sample_index"].astype(int).to_numpy()
    axis = np.arange(pi.shape[1], dtype=np.float64)
    raw_expected = pi @ axis

    rows = [summarize("raw_expected", raw_expected, frame, {"smooth": np.nan, "switch": np.nan, "poszero": np.nan, "window": 0})]
    if "pred_lag_stable" in frame:
        rows.append(
            summarize(
                "current_stable",
                frame["pred_lag_stable"].astype(float).to_numpy(),
                frame,
                {"smooth": np.nan, "switch": np.nan, "poszero": np.nan, "window": np.nan},
            )
        )

    smooth_values = [0.1, 0.3, 0.5, 0.8, 1.5]
    switch_values = [0.5, 1.0, 1.5, 2.0]
    poszero_values = [1.0, 2.0, 3.0]
    windows = [0, 41, 61, 81]

    for smooth, switch, poszero in itertools.product(smooth_values, switch_values, poszero_values):
        path = viterbi_decode_lag(
            pi,
            segment_id=segment_id,
            smooth_lambda=float(smooth),
            switch_penalty=float(switch),
            pos_to_zero_penalty=float(poszero),
        )
        rows.append(
            summarize(
                "viterbi",
                path,
                frame,
                {"smooth": smooth, "switch": switch, "poszero": poszero, "window": 0},
            )
        )
        for window in windows:
            if window <= 1:
                continue
            stable = _mode_filter_by_segment(path, segment_id, width=int(window), n_states=pi.shape[1])
            rows.append(
                summarize(
                    "viterbi_mode",
                    stable,
                    frame,
                    {"smooth": smooth, "switch": switch, "poszero": poszero, "window": window},
                )
            )

    out = pd.DataFrame(rows)
    out = out.sort_values(["mae_all", "mae_injected", "segment5_mae"], kind="stable")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")
    print(out.head(int(args.top_k)).to_string(index=False))


if __name__ == "__main__":
    main()
