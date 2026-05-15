#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path
from statistics import mean, pstdev

import pandas as pd
import yaml


SUMMARY_KEYS = [
    "stable_mae_all",
    "stable_mae_injected",
    "stable_mae_no_lag",
    "stable_accuracy_all",
    "stable_accuracy_injected",
    "stable_no_lag_false_alarm_rate",
    "stable_second_3000_4000_mae",
    "stable_segment5_mae",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate lag-only runs over multiple seeds.")
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/lag_regions_lag_only_best.yaml"))
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("results/lag_only_multiseed"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def dump_yaml(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def format_mean_std(values: list[float]) -> str:
    if len(values) == 1:
        return f"{values[0]:.6f}"
    return f"{mean(values):.6f} +/- {pstdev(values):.6f}"


def log(message: str) -> None:
    print(message, flush=True)


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


def write_summary(output_root: Path, rows: list[dict]) -> Path:
    summary_path = output_root / "multiseed_lag_metrics.csv"
    fieldnames = ["seed", "output_dir"] + SUMMARY_KEYS
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        summary = {"seed": "mean+/-std", "output_dir": ""}
        for key in SUMMARY_KEYS:
            values = [float(row[key]) for row in rows]
            summary[key] = format_mean_std(values)
        writer.writerow(summary)
    return summary_path


def main() -> None:
    args = parse_args()
    log("[setup] Parse arguments")
    project_root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    output_root = args.output_root if args.output_root.is_absolute() else project_root / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    generated_root = output_root / "_generated_configs"
    log(f"[setup] Project root: {project_root}")
    log(f"[setup] Base config: {config_path}")
    log(f"[setup] Output root: {output_root}")
    log(f"[setup] Seeds: {args.seeds}")
    log(f"[setup] Device: {args.device}")

    rows = []
    for idx, seed in enumerate(args.seeds, start=1):
        seed_label = f"seed_{seed}"
        log(f"[{seed_label}] ===== RUN {idx}/{len(args.seeds)} =====")
        seed_dir = output_root / f"seed_{seed}"
        seed_cfg_path = generated_root / f"seed_{seed}.yaml"
        seed_cfg = {
            "base_config": str(config_path),
            "seed": int(seed),
            "logging": {
                "output_dir": str(seed_dir),
            },
        }
        dump_yaml(seed_cfg_path, seed_cfg)

        log(f"[{seed_label}] Step 1/5 generated config: {seed_cfg_path}")
        log(f"[{seed_label}] Step 2/5 output dir: {seed_dir}")

        train_cmd = [
            sys.executable,
            "-u",
            str(project_root / "scripts" / "train_lag_grounded.py"),
            "--config",
            str(seed_cfg_path),
            "--lag-only",
            "--device",
            args.device,
        ]
        eval_cmd = [
            sys.executable,
            "-u",
            str(project_root / "scripts" / "evaluate_best_lag_only.py"),
            "--config",
            str(seed_cfg_path),
            "--checkpoint",
            str(seed_dir / "best_lag_identifier.pt"),
            "--output-dir",
            str(seed_dir),
            "--device",
            args.device,
        ]

        log(f"[{seed_label}] Step 3/5 train command ready")
        log(f"[{seed_label}] Step 4/5 eval command ready")
        if args.dry_run:
            log(f"[{seed_label}] Dry-run skip train/eval")
            continue

        run_step(train_cmd, project_root, f"{seed_label}/train")
        run_step(eval_cmd, project_root, f"{seed_label}/eval")

        log(f"[{seed_label}] Step 5/5 read lag metrics")
        metrics_path = seed_dir / "lag_eval_metrics.csv"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing lag metrics: {metrics_path}")
        metrics = pd.read_csv(metrics_path).iloc[0].to_dict()
        row = {"seed": int(seed), "output_dir": str(seed_dir)}
        for key in SUMMARY_KEYS:
            row[key] = metrics[key]
        rows.append(row)
        log(
            f"[{seed_label}] Metrics: "
            f"stable_mae_all={float(row['stable_mae_all']):.6f}, "
            f"stable_second_3000_4000_mae={float(row['stable_second_3000_4000_mae']):.6f}, "
            f"stable_segment5_mae={float(row['stable_segment5_mae']):.6f}"
        )
        log(f"[{seed_label}] ===== DONE =====")

    if args.dry_run:
        log("Dry run complete. No training was launched.")
        return

    log("[summary] Writing multiseed summary")
    summary_path = write_summary(output_root, rows)
    log("=== Lag multi-seed summary ===")
    for key in SUMMARY_KEYS:
        values = [float(row[key]) for row in rows]
        log(f"{key}: {format_mean_std(values)}")
    log(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
