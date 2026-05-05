#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep lambda_zero for unified block lag scorer and collect comparison tables."
    )
    parser.add_argument("--lambdas", default="1.0,2.0,5.0")
    parser.add_argument("--runs", default="old,seed134_e2")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-root", default="outputs/r41_unified_zero_loss_sweep")
    parser.add_argument("--base-summary", default="outputs/r40_unified_block_lag_scorer/unified_vs_q40_eval_summary.csv")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _path(text: str | Path) -> Path:
    path = Path(os.path.expandvars(str(text))).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _parse_lambdas(text: str) -> List[float]:
    return [float(part.strip()) for part in str(text).split(",") if part.strip()]


def _format_lambda(value: float) -> str:
    return str(value).replace(".", "p")


def _json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_sanitize(item) for item in value]
    if isinstance(value, tuple):
        return [_json_sanitize(item) for item in value]
    if isinstance(value, np.generic):
        return _json_sanitize(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_sanitize(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _run_one(lambda_zero: float, runs: str, epochs: int, device: str, output_root: Path, seed: int) -> Path:
    run_dir = output_root / f"lambda_zero_{_format_lambda(lambda_zero)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_unified_block_lag_scorer.py"),
        "--runs",
        str(runs),
        "--epochs",
        str(int(epochs)),
        "--device",
        str(device),
        "--seed",
        str(int(seed)),
        "--zero-loss-weight",
        str(float(lambda_zero)),
        "--output-dir",
        run_dir.as_posix(),
    ]
    subprocess.run(cmd, check=True, cwd=ROOT)
    return run_dir


def main() -> None:
    args = parse_args()
    output_root = _path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    base_summary = pd.read_csv(_path(args.base_summary))

    rows: List[Dict[str, Any]] = []
    run_records: List[Dict[str, Any]] = []
    for lambda_zero in _parse_lambdas(args.lambdas):
        run_dir = _run_one(
            lambda_zero=float(lambda_zero),
            runs=str(args.runs),
            epochs=int(args.epochs),
            device=str(args.device),
            output_root=output_root,
            seed=int(args.seed),
        )
        summary = pd.read_csv(run_dir / "unified_vs_q40_eval_summary.csv")
        summary["lambda_zero"] = float(lambda_zero)
        rows.append(summary)
        run_records.append(
            {
                "lambda_zero": float(lambda_zero),
                "run_dir": run_dir.as_posix(),
            }
        )

    combined = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    merged = combined.merge(
        base_summary.rename(
            columns={
                "unified_eval_overall_recall": "base_unified_eval_overall_recall",
                "unified_eval_FAR": "base_unified_eval_FAR",
                "unified_eval_zero_E_d_hat": "base_unified_eval_zero_E_d_hat",
                "unified_eval_AUPRC": "base_unified_eval_AUPRC",
                "unified_eval_peak_hit_at_pm1": "base_unified_eval_peak_hit_at_pm1",
                "unified_eval_pos_MAE": "base_unified_eval_pos_MAE",
            }
        ),
        on="run",
        how="left",
    )
    merged.to_csv(output_root / "lambda_zero_sweep_summary.csv", index=False)
    _write_json(
        output_root / "lambda_zero_sweep_report.json",
        {
            "component": "unified_zero_loss_sweep",
            "lambdas": _parse_lambdas(args.lambdas),
            "runs": str(args.runs),
            "epochs": int(args.epochs),
            "device": str(args.device),
            "base_summary": _path(args.base_summary).as_posix(),
            "outputs": {
                "summary_csv": (output_root / "lambda_zero_sweep_summary.csv").as_posix(),
            },
            "run_records": run_records,
        },
    )
    print(merged.to_csv(index=False))


if __name__ == "__main__":
    main()
