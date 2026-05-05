#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


DEFAULT_R10_CONFIGS = [
    "configs/r10_detlocal_v3_seed42.yaml",
    "configs/r10_detlocal_v3_seed134.yaml",
    "configs/r10_detlocal_v3_seed321.yaml",
    "configs/r10_detlocal_v3_seed456.yaml",
    "configs/r10_detlocal_v3_seed712.yaml",
]


def _path(text: str | Path) -> Path:
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _run(cmd: List[str], env: Dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def _json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_sanitize(v) for v in value]
    if isinstance(value, tuple):
        return [_json_sanitize(v) for v in value]
    if isinstance(value, np.generic):
        return _json_sanitize(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _method_metric(frame: pd.DataFrame, method: str, metric: str) -> float:
    row = frame.loc[frame["method"].astype(str) == str(method)]
    if row.empty:
        return float("nan")
    return float(row.iloc[0][metric])


def _drop_count(audit: pd.DataFrame, group: str) -> int:
    row = audit.loc[audit["group"].astype(str) == str(group)]
    if row.empty:
        return 0
    return int(row.iloc[0]["count"])


def _summarize_seed(seed_name: str, seed_root: Path) -> Dict[str, Any]:
    bump_summary = pd.read_csv(seed_root / "bump_eval" / "bump_method_summary.csv")
    veto_summary = pd.read_csv(seed_root / "r46b" / "strongkeep_veto_summary.csv")
    audit = pd.read_csv(seed_root / "r46b" / "bump_transfer" / "eval_drop_audit.csv")

    row = {
        "seed_run": seed_name,
        "raw_bump_in_mae": _method_metric(bump_summary, "expected_lag", "bump_in_mae"),
        "raw_outside_far": _method_metric(bump_summary, "expected_lag", "outside_far"),
        "raw_outside_mean_d_hat": _method_metric(bump_summary, "expected_lag", "outside_mean_d_hat"),
        "raw_peak_time_error": _method_metric(bump_summary, "expected_lag", "peak_time_error"),
        "raw_peak_value_error": _method_metric(bump_summary, "expected_lag", "peak_value_error"),
        "raw_shape_corr": _method_metric(bump_summary, "expected_lag", "shape_corr"),
        "q40_bump_in_mae": _method_metric(bump_summary, "q40", "bump_in_mae"),
        "q40_outside_far": _method_metric(bump_summary, "q40", "outside_far"),
        "q40_outside_mean_d_hat": _method_metric(bump_summary, "q40", "outside_mean_d_hat"),
        "q40_peak_time_error": _method_metric(bump_summary, "q40", "peak_time_error"),
        "q40_peak_value_error": _method_metric(bump_summary, "q40", "peak_value_error"),
        "q40_shape_corr": _method_metric(bump_summary, "q40", "shape_corr"),
        "r46b_bump_in_mae": _method_metric(bump_summary, "r46b", "bump_in_mae"),
        "r46b_outside_far": _method_metric(bump_summary, "r46b", "outside_far"),
        "r46b_outside_mean_d_hat": _method_metric(bump_summary, "r46b", "outside_mean_d_hat"),
        "r46b_peak_time_error": _method_metric(bump_summary, "r46b", "peak_time_error"),
        "r46b_peak_value_error": _method_metric(bump_summary, "r46b", "peak_value_error"),
        "r46b_shape_corr": _method_metric(bump_summary, "r46b", "shape_corr"),
        "r46b_selection_status": str(veto_summary.iloc[0]["selection_status"]),
        "r46b_selection_stage": str(veto_summary.iloc[0]["selection_stage"]),
        "r46b_theta_drop": float(veto_summary.iloc[0]["theta_drop"]),
        "q40_eval_recall": float(veto_summary.iloc[0]["q40_eval_recall"]),
        "r46b_eval_recall": float(veto_summary.iloc[0]["strongkeep_veto_eval_recall"]),
        "q40_eval_far": float(veto_summary.iloc[0]["q40_eval_FAR"]),
        "r46b_eval_far": float(veto_summary.iloc[0]["strongkeep_veto_eval_FAR"]),
        "dropped_positive_strong": _drop_count(audit, "dropped_positive_strong"),
        "dropped_positive_weak": _drop_count(audit, "dropped_positive_weak"),
        "dropped_false_positive_strong": _drop_count(audit, "dropped_false_positive_strong"),
        "dropped_false_positive_weak": _drop_count(audit, "dropped_false_positive_weak"),
    }
    row["r46b_minus_q40_outside_far"] = row["r46b_outside_far"] - row["q40_outside_far"]
    row["r46b_minus_q40_bump_in_mae"] = row["r46b_bump_in_mae"] - row["q40_bump_in_mae"]
    row["r46b_minus_q40_shape_corr"] = row["r46b_shape_corr"] - row["q40_shape_corr"]
    return row


def _aggregate_seed_rows(seed_rows: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [col for col in seed_rows.columns if col != "seed_run" and pd.api.types.is_numeric_dtype(seed_rows[col])]
    rows: List[Dict[str, Any]] = []
    for col in numeric_cols:
        values = pd.to_numeric(seed_rows[col], errors="coerce").to_numpy(dtype=np.float64)
        finite = values[np.isfinite(values)]
        rows.append(
            {
                "metric": col,
                "mean": float(np.mean(finite)) if finite.size else float("nan"),
                "std": float(np.std(finite, ddof=0)) if finite.size else float("nan"),
                "median": float(np.median(finite)) if finite.size else float("nan"),
                "min": float(np.min(finite)) if finite.size else float("nan"),
                "max": float(np.max(finite)) if finite.size else float("nan"),
                "n": int(finite.size),
            }
        )
    return pd.DataFrame(rows)


def _prune_seed_root_to_summaries(seed_root: Path) -> None:
    keep_files = {
        seed_root / "bump_eval" / "bump_method_summary.csv",
        seed_root / "r46b" / "strongkeep_veto_summary.csv",
        seed_root / "r46b" / "bump_transfer" / "eval_drop_audit.csv",
    }

    for subdir in ["transfer_block", "transfer_bump", "features_block", "features_bump", "q40_snapshots", "r45c"]:
        path = seed_root / subdir
        if path.exists():
            shutil.rmtree(path)

    for root in [seed_root / "bump_eval", seed_root / "r46b"]:
        if not root.exists():
            continue
        for file_path in root.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path in keep_files:
                continue
            file_path.unlink()

        for dir_path in sorted([p for p in root.rglob("*") if p.is_dir()], key=lambda p: len(p.parts), reverse=True):
            try:
                next(dir_path.iterdir())
            except StopIteration:
                dir_path.rmdir()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run raw/q40/r46b bump-transfer comparison across multiple seeds.")
    parser.add_argument("--configs", default=",".join(DEFAULT_R10_CONFIGS))
    parser.add_argument(
        "--bump-csv",
        default="data/processed/LiquidSugar_local_bump_mixed_balanced_evalsafe_segmentsplit_v3_rawgap.csv",
    )
    parser.add_argument("--output-root", default="outputs/r50_multiseed_bump_compare")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--artifact-mode",
        choices=["full", "summary"],
        default="full",
        help="Keep all compare artifacts or prune to compact summaries after each seed finishes.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    python_bin = str(_path(args.python_bin)) if not Path(str(args.python_bin)).is_absolute() else str(Path(str(args.python_bin)))
    bump_csv = _path(args.bump_csv)
    output_root = _path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/mpl")

    config_paths = [_path(part.strip()) for part in str(args.configs).split(",") if part.strip()]
    seed_rows: List[Dict[str, Any]] = []

    for config_path in config_paths:
        cfg = _load_config(config_path)
        seed_name = config_path.stem
        train_block_csv = _path(cfg["data"]["csv_path"])
        seed_root = output_root / seed_name
        block_eval_root = seed_root / "transfer_block"
        bump_eval_root = seed_root / "transfer_bump"
        block_feature_root = seed_root / "features_block"
        bump_feature_root = seed_root / "features_bump"
        q40_root = seed_root / "q40_snapshots"
        r45c_root = seed_root / "r45c"
        r46b_root = seed_root / "r46b"
        bump_out_root = seed_root / "bump_eval"

        final_summary_path = bump_out_root / "bump_method_summary.csv"
        if bool(args.skip_existing) and final_summary_path.exists():
            print(f"[skip-existing] {seed_name}", flush=True)
            seed_rows.append(_summarize_seed(seed_name, seed_root))
            continue

        _run(
            [
                python_bin,
                str(ROOT / "scripts/run_transfer_checkpoint_eval.py"),
                "--config",
                str(config_path),
                "--csv-path",
                str(train_block_csv),
                "--output-dir",
                str(block_eval_root),
                "--splits",
                "train,val",
                "--device",
                str(args.device),
            ],
            env=env,
        )
        _run(
            [
                python_bin,
                str(ROOT / "scripts/run_transfer_checkpoint_eval.py"),
                "--config",
                str(config_path),
                "--csv-path",
                str(bump_csv),
                "--output-dir",
                str(bump_eval_root),
                "--splits",
                "test",
                "--device",
                str(args.device),
            ],
            env=env,
        )
        _run(
            [
                python_bin,
                str(ROOT / "scripts/export_lag_feature_tables.py"),
                "--raw-csv",
                str(train_block_csv),
                "--eval-root",
                str(block_eval_root),
                "--output-dir",
                str(block_feature_root),
                "--splits",
                "train,val",
            ],
            env=env,
        )
        _run(
            [
                python_bin,
                str(ROOT / "scripts/export_lag_feature_tables.py"),
                "--raw-csv",
                str(bump_csv),
                "--eval-root",
                str(bump_eval_root),
                "--output-dir",
                str(bump_feature_root),
                "--splits",
                "test",
            ],
            env=env,
        )
        _run(
            [
                python_bin,
                str(ROOT / "scripts/export_fixed_q40_snapshots.py"),
                "--fit-series",
                str(block_feature_root / "train_feature_timeseries.csv"),
                "--val-series",
                str(block_feature_root / "val_feature_timeseries.csv"),
                "--eval-series",
                str(bump_feature_root / "test_feature_timeseries.csv"),
                "--output-dir",
                str(q40_root),
                "--run-name",
                "bump_transfer",
            ],
            env=env,
        )
        _run(
            [
                python_bin,
                str(ROOT / "scripts/run_q40_segment_proposal_verifier.py"),
                "--proposal-root",
                str(q40_root),
                "--runs",
                "bump_transfer",
                "--output-dir",
                str(r45c_root),
                "--q40-segment-features",
            ],
            env=env,
        )
        _run(
            [
                python_bin,
                str(ROOT / "scripts/run_q40_segment_strongkeep_veto.py"),
                "--source-root",
                str(r45c_root),
                "--runs",
                "bump_transfer",
                "--output-dir",
                str(r46b_root),
            ],
            env=env,
        )
        _run(
            [
                python_bin,
                str(ROOT / "scripts/evaluate_bump_q40_r46b.py"),
                "--feature-eval",
                str(bump_feature_root / "test_feature_timeseries.csv"),
                "--q40-eval",
                str(q40_root / "bump_transfer" / "q40_fixed_eval_timeseries.csv"),
                "--r46b-eval",
                str(r46b_root / "bump_transfer" / "strongkeep_veto_eval_timeseries.csv"),
                "--output-dir",
                str(bump_out_root),
            ],
            env=env,
        )

        seed_rows.append(_summarize_seed(seed_name, seed_root))
        if str(args.artifact_mode) == "summary":
            _prune_seed_root_to_summaries(seed_root)

    seed_summary = pd.DataFrame(seed_rows)
    seed_summary.to_csv(output_root / "multiseed_seed_summary.csv", index=False)
    metric_summary = _aggregate_seed_rows(seed_summary)
    metric_summary.to_csv(output_root / "multiseed_metric_summary.csv", index=False)
    report = {
        "configs": [str(path) for path in config_paths],
        "bump_csv": str(bump_csv),
        "output_root": str(output_root),
        "n_seeds": int(len(seed_summary)),
        "seeds": seed_summary["seed_run"].astype(str).tolist() if not seed_summary.empty else [],
    }
    (output_root / "multiseed_report.json").write_text(
        json.dumps(_json_sanitize(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(seed_summary.to_csv(index=False, float_format="%.6f"))
    print(metric_summary.to_csv(index=False, float_format="%.6f"))
    print(f"Wrote {output_root}")


if __name__ == "__main__":
    main()
