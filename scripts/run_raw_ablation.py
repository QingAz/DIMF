#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict

import pandas as pd
import yaml


DEFAULT_VARIANTS = ["no_prior", "pretrained_0p1", "pretrained_0p3", "pretrained_0p5", "random_0p3"]
METRIC_KEYS = ["prediction_mae", "prediction_mse", "prediction_rmse", "prediction_r2"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run raw LiquidSugar y-only ablations for pretrained lag priors.")
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/raw_liquidsugar_y_only_adapt.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("results/raw_ablation"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=None, help="Override training.epochs_dimf for quick probes.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def log(message: str) -> None:
    print(message, flush=True)


def dump_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def variant_override(variant: str) -> Dict[str, Any]:
    if variant == "no_prior":
        return {
            "raw_adaptation": {"enabled": True, "prior_source": "none"},
            "delay_prior": {"enabled": False, "lambda_prior": 0.0},
            "lag_identifier": {"init_checkpoint": None},
        }
    if variant.startswith("pretrained_"):
        lambda_prior = _lambda_from_variant(variant, prefix="pretrained_")
        return {
            "raw_adaptation": {"enabled": True, "prior_source": "pretrained"},
            "delay_prior": {"enabled": True, "lambda_prior": lambda_prior},
        }
    if variant.startswith("random_"):
        lambda_prior = _lambda_from_variant(variant, prefix="random_")
        return {
            "raw_adaptation": {"enabled": True, "prior_source": "random"},
            "delay_prior": {"enabled": True, "lambda_prior": lambda_prior},
            "lag_identifier": {"init_checkpoint": None},
        }
    raise ValueError(f"Unknown variant: {variant}")


def _lambda_from_variant(variant: str, prefix: str) -> float:
    raw = variant.removeprefix(prefix).replace("p", ".")
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Could not parse lambda from variant {variant!r}") from exc


def build_run_config(base_config: Path, output_dir: Path, seed: int, variant: str, epochs: int | None) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "base_config": str(base_config),
        "seed": int(seed),
        "logging": {"output_dir": str(output_dir)},
        "ablation": {"variant": variant},
    }
    cfg.update(variant_override(variant))
    if epochs is not None:
        cfg.setdefault("training", {})["epochs_dimf"] = int(epochs)
    return cfg


def run_step(cmd: list[str], cwd: Path, label: str) -> None:
    log(f"[{label}] START")
    log(f"[{label}] CMD: {' '.join(cmd)}")
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    with subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    ) as proc:
        assert proc.stdout is not None
        for line in proc.stdout:
            print(f"[{label}] {line}", end="", flush=True)
        code = proc.wait()
    if code != 0:
        raise subprocess.CalledProcessError(code, cmd)
    log(f"[{label}] DONE")


def format_mean_std(values: list[float]) -> str:
    if len(values) == 1:
        return f"{values[0]:.6f}"
    return f"{mean(values):.6f} +/- {pstdev(values):.6f}"


def write_summary(output_root: Path, rows: list[dict]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    rows_path = output_root / "raw_ablation_metrics.csv"
    fieldnames = ["variant", "seed", "output_dir"] + METRIC_KEYS
    with rows_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    summary_rows = []
    for variant in sorted({str(row["variant"]) for row in rows}):
        subset = [row for row in rows if row["variant"] == variant]
        summary = {"variant": variant, "n_runs": len(subset)}
        for key in METRIC_KEYS:
            values = [float(row[key]) for row in subset]
            summary[f"{key}_mean_std"] = format_mean_std(values)
            summary[f"{key}_mean"] = mean(values)
            summary[f"{key}_std"] = 0.0 if len(values) == 1 else pstdev(values)
        summary_rows.append(summary)

    summary_path = output_root / "raw_ablation_summary.csv"
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    log(f"Saved: {rows_path}")
    log(f"Saved: {summary_path}")


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    base_config = args.config if args.config.is_absolute() else project_root / args.config
    output_root = args.output_root if args.output_root.is_absolute() else project_root / args.output_root
    generated_root = output_root / "_generated_configs"
    output_root.mkdir(parents=True, exist_ok=True)

    log(f"[setup] Project root: {project_root}")
    log(f"[setup] Base config: {base_config}")
    log(f"[setup] Output root: {output_root}")
    log(f"[setup] Seeds: {args.seeds}")
    log(f"[setup] Variants: {args.variants}")
    log(f"[setup] Device: {args.device}")

    rows: list[dict] = []
    total = len(args.variants) * len(args.seeds)
    run_idx = 0
    for variant in args.variants:
        variant_override(variant)  # validate early
        for seed in args.seeds:
            run_idx += 1
            label = f"{variant}/seed_{seed}"
            log(f"[{label}] ===== RUN {run_idx}/{total} =====")
            run_dir = output_root / variant / f"seed_{seed}"
            cfg_path = generated_root / variant / f"seed_{seed}.yaml"
            cfg = build_run_config(base_config, run_dir, int(seed), variant, args.epochs)
            dump_yaml(cfg_path, cfg)
            log(f"[{label}] generated config: {cfg_path}")

            metrics_path = run_dir / "prediction_metrics.csv"
            if args.skip_existing and metrics_path.exists():
                log(f"[{label}] skip existing metrics: {metrics_path}")
            elif args.dry_run:
                log(f"[{label}] dry-run skip training")
                continue
            else:
                cmd = [
                    sys.executable,
                    "-u",
                    str(project_root / "scripts" / "train_lag_grounded.py"),
                    "--config",
                    str(cfg_path),
                    "--raw-adapt",
                    "--device",
                    args.device,
                ]
                run_step(cmd, project_root, label)

            if not metrics_path.exists():
                raise FileNotFoundError(f"Missing prediction metrics: {metrics_path}")
            metrics = pd.read_csv(metrics_path).iloc[0].to_dict()
            row = {"variant": variant, "seed": int(seed), "output_dir": str(run_dir)}
            for key in METRIC_KEYS:
                row[key] = float(metrics[key])
            rows.append(row)
            log(
                f"[{label}] rmse={row['prediction_rmse']:.6f} "
                f"mae={row['prediction_mae']:.6f} r2={row['prediction_r2']:.6f}"
            )

    if args.dry_run:
        log("Dry run complete. No training was launched.")
        return
    write_summary(output_root, rows)


if __name__ == "__main__":
    main()
