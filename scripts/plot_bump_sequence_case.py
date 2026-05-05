#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-codex"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _path(text: str | Path) -> Path:
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _read_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for col in frame.columns:
        if col not in {"split", "source_split", "timestamp", "original_split", "q40_prediction_source"}:
            try:
                frame[col] = pd.to_numeric(frame[col])
            except (TypeError, ValueError):
                pass
    return frame


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    start = None
    for idx, value in enumerate(mask.astype(bool)):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            out.append((start, idx - 1))
            start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out


def _merge_case(feature_path: Path, q40_path: Path, r46b_path: Path) -> pd.DataFrame:
    feature = _read_csv(feature_path)
    q40 = _read_csv(q40_path)
    r46b = _read_csv(r46b_path)
    keys = [col for col in ["timestamp", "raw_row_index", "segment_id", "t"] if col in feature.columns]
    q40_cols = keys + [col for col in ["q40_d_hat", "q40_selected", "q40_final_selected"] if col in q40.columns]
    r46b_cols = keys + [col for col in ["veto_d_hat_final", "veto_selected_final", "weak_drop_segment", "segment_is_strong"] if col in r46b.columns]
    merged = feature.merge(q40[q40_cols], on=keys, how="left")
    merged = merged.merge(r46b[r46b_cols], on=keys, how="left")
    if "q40_selected" not in merged.columns and "q40_final_selected" in merged.columns:
        merged["q40_selected"] = merged["q40_final_selected"]
    merged["q40_d_hat"] = pd.to_numeric(merged.get("q40_d_hat", 0.0), errors="coerce").fillna(0.0)
    merged["veto_d_hat_final"] = pd.to_numeric(merged.get("veto_d_hat_final", 0.0), errors="coerce").fillna(0.0)
    merged["q40_selected"] = pd.to_numeric(merged.get("q40_selected", 0.0), errors="coerce").fillna(0.0)
    merged["veto_selected_final"] = pd.to_numeric(merged.get("veto_selected_final", 0.0), errors="coerce").fillna(0.0)
    merged["weak_drop_segment"] = pd.to_numeric(merged.get("weak_drop_segment", 0.0), errors="coerce").fillna(0.0)
    merged["segment_is_strong"] = pd.to_numeric(merged.get("segment_is_strong", 0.0), errors="coerce").fillna(0.0)
    return merged


def plot_segment(
    frame: pd.DataFrame,
    segment_id: int,
    output_path: Path,
    title: str,
    minimal_r46b: bool = False,
    t_min: int | None = None,
    t_max: int | None = None,
) -> None:
    part = frame.loc[frame["segment_id"].astype(int) == int(segment_id)].sort_values("t").copy()
    if part.empty:
        raise ValueError(f"segment_id={segment_id} not found")
    if t_min is not None:
        part = part.loc[part["t"].to_numpy(dtype=np.float64) >= float(t_min)].copy()
    if t_max is not None:
        part = part.loc[part["t"].to_numpy(dtype=np.float64) <= float(t_max)].copy()
    if part.empty:
        raise ValueError(f"segment_id={segment_id} has no rows in requested t-window")

    x = part["t"].to_numpy(dtype=np.float64)
    d_true = part["d_true"].to_numpy(dtype=np.float64)
    raw = part["expected_lag"].to_numpy(dtype=np.float64)
    q40 = part["q40_d_hat"].to_numpy(dtype=np.float64)
    r46b = part["veto_d_hat_final"].to_numpy(dtype=np.float64)
    q40_sel = part["q40_selected"].to_numpy(dtype=np.float64) > 0
    r46b_sel = part["veto_selected_final"].to_numpy(dtype=np.float64) > 0
    weak_drop = part["weak_drop_segment"].to_numpy(dtype=np.float64) > 0

    fig, (ax, ax2) = plt.subplots(
        2,
        1,
        figsize=(16, 7),
        gridspec_kw={"height_ratios": [4.5, 1.2]},
        constrained_layout=True,
    )

    ymax = max(1.0, float(np.nanmax(np.concatenate([d_true, raw, q40, r46b])) + 0.8))
    for start, end in _runs(d_true > 0):
        ax.axvspan(float(x[start]) - 0.5, float(x[end]) + 0.5, color="#444444", alpha=0.10, linewidth=0)
    if not minimal_r46b:
        for start, end in _runs(q40_sel):
            ax.axvspan(float(x[start]) - 0.5, float(x[end]) + 0.5, color="#e41a1c", alpha=0.05, linewidth=0)
    for start, end in _runs(weak_drop):
        ax.axvspan(float(x[start]) - 0.5, float(x[end]) + 0.5, color="#377eb8", alpha=0.08, linewidth=0)

    ax.step(x, d_true, where="mid", color="black", linewidth=2.0, label="d_true")
    if not minimal_r46b:
        ax.plot(x, raw, color="#ff7f00", linewidth=1.3, label="raw expected_lag")
        ax.step(x, q40, where="mid", color="#e41a1c", linewidth=1.4, label="q40")
    ax.step(x, r46b, where="mid", color="#377eb8", linewidth=1.6, label="r46b")
    ax.set_ylim(-0.1, ymax)
    ax.set_ylabel("lag")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", ncol=2 if minimal_r46b else 4, fontsize=9)

    if not minimal_r46b:
        ax2.step(x, q40_sel.astype(float), where="mid", color="#e41a1c", linewidth=1.2, label="q40 selected")
    ax2.step(x, r46b_sel.astype(float) + 0.05, where="mid", color="#377eb8", linewidth=1.2, label="r46b kept")
    ax2.step(x, weak_drop.astype(float) + 0.10, where="mid", color="#4daf4a", linewidth=1.0, label="r46b drop")
    ax2.set_ylim(-0.1, 1.3)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["off", "on"])
    ax2.set_xlabel("t")
    ax2.grid(alpha=0.20)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a single bump val/test segment with raw, q40, and r46b.")
    parser.add_argument("--feature", required=True)
    parser.add_argument("--q40", required=True)
    parser.add_argument("--r46b", required=True)
    parser.add_argument("--segment-id", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--minimal-r46b", action="store_true")
    parser.add_argument("--t-min", type=int, default=None)
    parser.add_argument("--t-max", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = _merge_case(
        feature_path=_path(args.feature),
        q40_path=_path(args.q40),
        r46b_path=_path(args.r46b),
    )
    plot_segment(
        frame=frame,
        segment_id=int(args.segment_id),
        output_path=_path(args.output),
        title=str(args.title),
        minimal_r46b=bool(args.minimal_r46b),
        t_min=args.t_min,
        t_max=args.t_max,
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
