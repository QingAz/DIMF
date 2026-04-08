#!/usr/bin/env python3

import argparse
import copy
import csv
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean, pstdev

import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Run DIMF sequentially over multiple seeds and summarize metrics.")
    parser.add_argument("--config", type=str, default="configs/multistage_aligned.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--run-suffix", type=str, default="5seed")
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_config(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        try:
            yaml.safe_dump(payload, f, sort_keys=False)
        except TypeError:
            yaml.safe_dump(payload, f)


def resolve_project_path(project_root: Path, path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return project_root / path


def with_suffix_name(path: Path, suffix: str) -> Path:
    return path.with_name(f"{path.name}_{suffix}")


def seed_config(base_cfg, seed: int, run_suffix: str):
    cfg = copy.deepcopy(base_cfg)
    cfg["seed"] = int(seed)

    output_root = with_suffix_name(Path(cfg["logging"]["output_dir"]), run_suffix)
    ckpt_root = with_suffix_name(Path(cfg["logging"]["ckpt_path"]).parent, run_suffix)
    scaler_root = with_suffix_name(Path(cfg["logging"]["scaler_path"]).parent, run_suffix)

    seed_name = f"seed_{seed}"
    cfg["logging"]["output_dir"] = str(output_root / seed_name)
    cfg["logging"]["log_path"] = str(output_root / seed_name / "train_log.jsonl")
    cfg["logging"]["ckpt_path"] = str(ckpt_root / seed_name / Path(cfg["logging"]["ckpt_path"]).name)
    cfg["logging"]["scaler_path"] = str(scaler_root / seed_name / Path(cfg["logging"]["scaler_path"]).name)
    return cfg, output_root


def write_seed_config(project_root: Path, cfg, output_root: Path, seed: int) -> Path:
    gen_dir = resolve_project_path(project_root, str(output_root / "_generated_configs"))
    gen_dir.mkdir(parents=True, exist_ok=True)
    config_path = gen_dir / f"seed_{seed}.yaml"
    dump_yaml(config_path, cfg)
    return config_path


def load_metrics(metrics_path: Path):
    with metrics_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def format_mean_std(values):
    if len(values) == 1:
        return f"{values[0]:.6f}"
    return f"{mean(values):.6f} ± {pstdev(values):.6f}"


def write_summary(output_root: Path, rows):
    output_root.mkdir(parents=True, exist_ok=True)
    metrics_order = ["MSE", "MAE", "RMSE", "R2", "scaled_MSE", "scaled_MAE", "scaled_RMSE", "scaled_R2", "n_test"]

    csv_path = output_root / "multiseed_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seed"] + metrics_order)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

        summary_row = {"seed": "mean±std"}
        for key in metrics_order:
            values = [float(r[key]) for r in rows]
            if key == "n_test":
                summary_row[key] = str(int(values[0])) if len(set(values)) == 1 else format_mean_std(values)
            else:
                summary_row[key] = format_mean_std(values)
        writer.writerow(summary_row)

    summary = {
        "n_runs": len(rows),
        "seeds": [int(r["seed"]) for r in rows],
        "summary": {},
    }
    for key in metrics_order:
        values = [float(r[key]) for r in rows]
        if key == "n_test":
            summary["summary"][key] = {
                "value": int(values[0]) if len(set(values)) == 1 else values,
            }
        else:
            summary["summary"][key] = {
                "mean": mean(values),
                "std": pstdev(values),
                "formatted": format_mean_std(values),
            }

    with (output_root / "multiseed_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return csv_path, output_root / "multiseed_summary.json"


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config_path = resolve_project_path(project_root, args.config)
    base_cfg = load_config(config_path)

    seeds = args.seeds if args.seeds else list(base_cfg.get("seeds", [base_cfg.get("seed", 42)]))
    if not seeds:
        raise ValueError("No seeds were provided and config does not contain a non-empty 'seeds' list.")

    rows = []
    last_output_root = None

    for idx, seed in enumerate(seeds, start=1):
        seed_cfg, output_root = seed_config(base_cfg, int(seed), args.run_suffix)
        last_output_root = output_root
        seed_output_dir = resolve_project_path(project_root, seed_cfg["logging"]["output_dir"])
        metrics_path = seed_output_dir / "test_metrics.json"
        generated_cfg = write_seed_config(project_root, seed_cfg, output_root, int(seed))

        print(f"[{idx}/{len(seeds)}] seed={seed}")
        print(f"  config: {generated_cfg}")
        print(f"  output: {seed_output_dir}")

        if not args.dry_run:
            subprocess.run(
                [sys.executable, str(project_root / "train.py"), "--config", str(generated_cfg), "--device", args.device],
                cwd=str(project_root),
                check=True,
            )

            if not metrics_path.exists():
                raise FileNotFoundError(f"Missing metrics file after seed {seed}: {metrics_path}")

            metrics = load_metrics(metrics_path)
            row = {"seed": int(seed)}
            for key in ["MSE", "MAE", "RMSE", "R2", "scaled_MSE", "scaled_MAE", "scaled_RMSE", "scaled_R2", "n_test"]:
                row[key] = metrics[key]
            rows.append(row)

    if args.dry_run:
        print("Dry run complete. No training was launched.")
        return

    csv_path, json_path = write_summary(resolve_project_path(project_root, str(last_output_root)), rows)
    print("=== Multi-seed Summary ===")
    for key in ["MSE", "MAE", "RMSE", "R2", "scaled_MSE", "scaled_MAE", "scaled_RMSE", "scaled_R2"]:
        values = [float(r[key]) for r in rows]
        print(f"{key}: {format_mean_std(values)}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
