#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

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
        if col not in {"split", "source_split", "timestamp", "original_split", "bump_type", "bump_shape"}:
            frame[col] = pd.to_numeric(frame[col], errors="ignore")
    return frame


def _merge_series(feature_eval: pd.DataFrame, q40_eval: pd.DataFrame, r46b_eval: pd.DataFrame) -> pd.DataFrame:
    keys = [col for col in ["timestamp", "raw_row_index", "segment_id", "t"] if col in feature_eval.columns]
    if not keys:
        raise ValueError("Cannot merge bump evaluation tables: no shared key columns")
    q40_cols = keys + [col for col in ["q40_selected", "q40_d_hat", "q40_final_selected", "d_hat"] if col in q40_eval.columns]
    r46b_cols = keys + [col for col in ["veto_selected_final", "veto_d_hat_final", "weak_drop_segment", "segment_is_strong"] if col in r46b_eval.columns]
    merged = feature_eval.merge(q40_eval[q40_cols], on=keys, how="left", suffixes=("", "_q40dup"))
    merged = merged.merge(r46b_eval[r46b_cols], on=keys, how="left", suffixes=("", "_r46dup"))
    if "q40_d_hat" not in merged.columns:
        if "d_hat" in q40_eval.columns:
            merged["q40_d_hat"] = pd.to_numeric(merged["d_hat"], errors="coerce").fillna(0.0)
        else:
            merged["q40_d_hat"] = 0.0
    if "q40_selected" not in merged.columns:
        if "q40_final_selected" in merged.columns:
            merged["q40_selected"] = pd.to_numeric(merged["q40_final_selected"], errors="coerce").fillna(0.0)
        else:
            merged["q40_selected"] = (pd.to_numeric(merged["q40_d_hat"], errors="coerce").fillna(0.0) > 0).astype(int)
    if "veto_d_hat_final" not in merged.columns:
        merged["veto_d_hat_final"] = 0.0
    if "veto_selected_final" not in merged.columns:
        merged["veto_selected_final"] = (pd.to_numeric(merged["veto_d_hat_final"], errors="coerce").fillna(0.0) > 0).astype(int)
    return merged.sort_values(keys).reset_index(drop=True)


def _runs(mask: np.ndarray) -> List[tuple[int, int]]:
    runs: List[tuple[int, int]] = []
    start = None
    for idx, value in enumerate(mask.astype(bool)):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _positive_blocks(frame: pd.DataFrame, group_col: str, label_col: str) -> List[np.ndarray]:
    blocks: List[np.ndarray] = []
    for _, idx in frame.groupby(group_col, sort=False).groups.items():
        idx_arr = frame.index.get_indexer(idx)
        labels = frame.iloc[idx_arr][label_col].to_numpy(dtype=np.float64) > 0
        for start, end in _runs(labels):
            blocks.append(idx_arr[start : end + 1])
    return blocks


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan")
    x = x[mask]
    y = y[mask]
    if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _method_summary(frame: pd.DataFrame, d_hat_col: str, label_col: str, group_col: str, time_col: str) -> Dict[str, Any]:
    d_true = frame[label_col].to_numpy(dtype=np.float64)
    d_hat = np.maximum(pd.to_numeric(frame[d_hat_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64), 0.0)
    bump_mask = d_true > 0
    outside_mask = ~bump_mask
    blocks = _positive_blocks(frame, group_col=group_col, label_col=label_col)

    peak_time_errors: List[float] = []
    peak_value_errors: List[float] = []
    shape_corrs: List[float] = []
    block_rows: List[Dict[str, Any]] = []
    time_values = frame[time_col].to_numpy(dtype=np.float64)
    for block_id, idx in enumerate(blocks, start=1):
        block_true = d_true[idx]
        block_hat = d_hat[idx]
        block_time = time_values[idx]
        true_peak_pos = int(np.nanargmax(block_true))
        pred_peak_pos = int(np.nanargmax(block_hat))
        true_peak_time = float(block_time[true_peak_pos])
        pred_peak_time = float(block_time[pred_peak_pos])
        true_peak_value = float(np.nanmax(block_true))
        pred_peak_value = float(np.nanmax(block_hat))
        peak_time_error = abs(pred_peak_time - true_peak_time)
        peak_value_error = abs(pred_peak_value - true_peak_value)
        corr = _safe_corr(block_hat, block_true)
        peak_time_errors.append(peak_time_error)
        peak_value_errors.append(peak_value_error)
        shape_corrs.append(corr)
        block_rows.append(
            {
                "block_id": block_id,
                "segment_id": int(frame.iloc[idx[0]][group_col]),
                "start_t": float(block_time[0]),
                "end_t": float(block_time[-1]),
                "true_peak_time": true_peak_time,
                "pred_peak_time": pred_peak_time,
                "peak_time_error": peak_time_error,
                "true_peak_value": true_peak_value,
                "pred_peak_value": pred_peak_value,
                "peak_value_error": peak_value_error,
                "shape_corr": corr,
            }
        )

    return {
        "n_rows": int(len(frame)),
        "n_bump_rows": int(bump_mask.sum()),
        "n_outside_rows": int(outside_mask.sum()),
        "bump_in_mae": float(np.mean(np.abs(d_hat[bump_mask] - d_true[bump_mask]))) if bump_mask.any() else float("nan"),
        "outside_far": float(np.mean(d_hat[outside_mask] > 0)) if outside_mask.any() else float("nan"),
        "outside_mean_d_hat": float(np.mean(d_hat[outside_mask])) if outside_mask.any() else float("nan"),
        "peak_time_error": float(np.nanmean(peak_time_errors)) if peak_time_errors else float("nan"),
        "peak_value_error": float(np.nanmean(peak_value_errors)) if peak_value_errors else float("nan"),
        "shape_corr": float(np.nanmean(shape_corrs)) if shape_corrs else float("nan"),
        "n_blocks": int(len(blocks)),
        "block_rows": block_rows,
    }


def _representative_segments(frame: pd.DataFrame, type_col: str, label_col: str, max_per_type: int) -> Dict[str, List[int]]:
    positives = frame.loc[frame[label_col].to_numpy(dtype=np.float64) > 0].copy()
    if positives.empty:
        return {}
    out: Dict[str, List[int]] = {}
    for type_value, part in positives.groupby(type_col, sort=False):
        view = (
            part.groupby("segment_id", sort=False)
            .agg(n_positive=(label_col, lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0.0) > 0).sum())))
            .reset_index()
            .sort_values(["n_positive", "segment_id"], ascending=[False, True])
        )
        out[str(type_value)] = view["segment_id"].astype(int).head(int(max_per_type)).tolist()
    return out


def _plot_panels(frame: pd.DataFrame, out_dir: Path, type_col: str, label_col: str, max_per_type: int) -> None:
    picks = _representative_segments(frame, type_col=type_col, label_col=label_col, max_per_type=max_per_type)
    for type_value, segments in picks.items():
        if not segments:
            continue
        fig, axes = plt.subplots(len(segments), 1, figsize=(14, 3.2 * len(segments)), constrained_layout=True)
        if len(segments) == 1:
            axes = [axes]
        for ax, segment_id in zip(axes, segments):
            part = frame.loc[frame["segment_id"].astype(int) == int(segment_id)].sort_values("t").copy()
            x = part["t"].to_numpy(dtype=np.float64)
            d_true = part[label_col].to_numpy(dtype=np.float64)
            q40_hat = part["q40_d_hat"].to_numpy(dtype=np.float64)
            r46b_hat = part["veto_d_hat_final"].to_numpy(dtype=np.float64)
            expected = part["expected_lag"].to_numpy(dtype=np.float64) if "expected_lag" in part.columns else np.zeros(len(part), dtype=np.float64)
            q40_sel = part["q40_selected"].to_numpy(dtype=np.float64) > 0
            ymax = max(1.0, float(np.nanmax(np.concatenate([d_true, q40_hat, r46b_hat, expected])) + 0.8))
            for start, end in _runs(d_true > 0):
                ax.axvspan(float(x[start]) - 0.5, float(x[end]) + 0.5, color="#444444", alpha=0.10, linewidth=0)
            if q40_sel.any():
                for start, end in _runs(q40_sel):
                    ax.axvspan(float(x[start]) - 0.5, float(x[end]) + 0.5, color="#e41a1c", alpha=0.06, linewidth=0)
            ax.step(x, d_true, where="mid", color="black", linewidth=2.0, label="d_true bump")
            ax.step(x, q40_hat, where="mid", color="#e41a1c", linewidth=1.4, label="q40_d_hat")
            ax.step(x, r46b_hat, where="mid", color="#377eb8", linewidth=1.4, label="r46b_d_hat")
            ax.plot(x, expected, color="#ff7f00", linewidth=1.2, label="expected_lag")
            ax.set_ylim(-0.1, ymax)
            ax.set_title(f"{type_col}={type_value}, segment={segment_id}")
            ax.grid(alpha=0.25)
        axes[0].legend(loc="upper left", ncol=4, fontsize=8)
        axes[-1].set_xlabel("t")
        fig.savefig(out_dir / f"bump_panel_{type_col}_{type_value}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate q40 and r46b on bump-test outputs.")
    parser.add_argument("--feature-eval", required=True)
    parser.add_argument("--q40-eval", required=True)
    parser.add_argument("--r46b-eval", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label-col", default="d_true")
    parser.add_argument("--group-col", default="segment_id")
    parser.add_argument("--time-col", default="t")
    parser.add_argument("--max-panels-per-type", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = _path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_eval = _read_csv(_path(args.feature_eval))
    q40_eval = _read_csv(_path(args.q40_eval))
    r46b_eval = _read_csv(_path(args.r46b_eval))
    merged = _merge_series(feature_eval=feature_eval, q40_eval=q40_eval, r46b_eval=r46b_eval)
    merged.to_csv(out_dir / "bump_eval_joined.csv", index=False)

    type_col = "bump_type" if "bump_type" in merged.columns else ("bump_shape" if "bump_shape" in merged.columns else "dmax")

    rows: List[Dict[str, Any]] = []
    block_rows: List[Dict[str, Any]] = []
    for method, col in [
        ("q40", "q40_d_hat"),
        ("r46b", "veto_d_hat_final"),
        ("expected_lag", "expected_lag"),
    ]:
        metrics = _method_summary(
            merged,
            d_hat_col=col,
            label_col=str(args.label_col),
            group_col=str(args.group_col),
            time_col=str(args.time_col),
        )
        rows.append({k: v for k, v in metrics.items() if k != "block_rows"} | {"method": method})
        for item in metrics["block_rows"]:
            block_rows.append({"method": method, **item})

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "bump_method_summary.csv", index=False)
    pd.DataFrame(block_rows).to_csv(out_dir / "bump_block_metrics.csv", index=False)
    _plot_panels(
        merged,
        out_dir=out_dir,
        type_col=type_col,
        label_col=str(args.label_col),
        max_per_type=int(args.max_panels_per_type),
    )

    report = {
        "type_col": type_col,
        "feature_eval": str(_path(args.feature_eval)),
        "q40_eval": str(_path(args.q40_eval)),
        "r46b_eval": str(_path(args.r46b_eval)),
        "summary": rows,
        "outputs": {
            "joined": (out_dir / "bump_eval_joined.csv").as_posix(),
            "method_summary": (out_dir / "bump_method_summary.csv").as_posix(),
            "block_metrics": (out_dir / "bump_block_metrics.csv").as_posix(),
        },
    }
    (out_dir / "bump_eval_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(summary.to_csv(index=False, float_format="%.6f"))
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
