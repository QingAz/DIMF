#!/usr/bin/env python3

import argparse
import copy
import csv
import itertools
import json
import random
import subprocess
import sys
from pathlib import Path
from statistics import mean, pstdev

import yaml


METRIC_KEYS = [
    "best_val",
    "MSE",
    "MAE",
    "RMSE",
    "R2",
    "scaled_MSE",
    "scaled_MAE",
    "scaled_RMSE",
    "scaled_R2",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run a hyperparameter sweep for DIMF.")
    parser.add_argument("--sweep-config", type=str, default="configs/sweep_multistage_aligned.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path):
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


def set_nested_value(cfg, dotted_key: str, value):
    parts = dotted_key.split(".")
    cur = cfg
    for key in parts[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[parts[-1]] = value


def generate_trials(search_cfg):
    search = search_cfg["search"]
    param_space = search["parameters"]
    keys = list(param_space.keys())
    value_lists = [list(param_space[key]) for key in keys]

    total = 1
    for values in value_lists:
        total *= len(values)

    if total > 50000:
        raise ValueError(f"Search space too large ({total} combinations). Shrink it before launching.")

    all_combos = [dict(zip(keys, combo)) for combo in itertools.product(*value_lists)]
    mode = str(search.get("mode", "random")).lower()
    n_trials = int(search.get("n_trials", len(all_combos)))

    if mode == "grid":
        return all_combos[:n_trials]

    if mode != "random":
        raise ValueError(f"Unsupported search mode: {mode}")

    rng = random.Random(int(search.get("random_seed", 42)))
    rng.shuffle(all_combos)
    return all_combos[:n_trials]


def format_mean_std(values):
    if len(values) == 1:
        return f"{values[0]:.6f}"
    return f"{mean(values):.6f} ± {pstdev(values):.6f}"


def summarize_metric(values):
    return {
        "mean": mean(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "formatted": format_mean_std(values),
    }


def load_best_val(log_path: Path):
    best = None
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            val = float(record["val_loss"])
            best = val if best is None else min(best, val)
    return best


def build_trial_config(base_cfg, params):
    cfg = copy.deepcopy(base_cfg)
    for dotted_key, value in params.items():
        set_nested_value(cfg, dotted_key, value)
    return cfg


def build_trial_paths(run_name: str, trial_name: str):
    output_root = Path("outputs") / run_name / trial_name
    artifact_root = Path("artifacts") / run_name / trial_name
    return output_root, artifact_root


def build_seed_config(trial_cfg, output_root: Path, artifact_root: Path, seed: int):
    cfg = copy.deepcopy(trial_cfg)
    cfg["seed"] = int(seed)
    seed_name = f"seed_{seed}"
    cfg["logging"]["output_dir"] = str(output_root / seed_name)
    cfg["logging"]["log_path"] = str(output_root / seed_name / "train_log.jsonl")
    cfg["logging"]["ckpt_path"] = str(artifact_root / seed_name / "best.ckpt")
    cfg["logging"]["scaler_path"] = str(artifact_root / seed_name / "scaler.pkl")
    return cfg


def write_seed_config(generated_root: Path, trial_name: str, seed: int, cfg) -> Path:
    config_path = generated_root / trial_name / f"seed_{seed}.yaml"
    dump_yaml(config_path, cfg)
    return config_path


def summarize_seed_rows(rows):
    summary = {}
    for key in METRIC_KEYS:
        values = [float(r[key]) for r in rows]
        summary[key] = summarize_metric(values)

    n_test_values = [int(r["n_test"]) for r in rows]
    summary["n_test"] = {
        "value": n_test_values[0] if len(set(n_test_values)) == 1 else n_test_values,
    }
    return summary


def write_trial_outputs(output_root: Path, params, rows, summary):
    output_root.mkdir(parents=True, exist_ok=True)
    metrics_order = METRIC_KEYS + ["n_test"]

    csv_path = output_root / "multiseed_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seed"] + metrics_order)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

        summary_row = {"seed": "mean±std"}
        for key in METRIC_KEYS:
            summary_row[key] = summary[key]["formatted"]
        n_test_value = summary["n_test"]["value"]
        summary_row["n_test"] = str(n_test_value) if isinstance(n_test_value, int) else json.dumps(n_test_value)
        writer.writerow(summary_row)

    summary_json = output_root / "multiseed_summary.json"
    payload = {
        "params": params,
        "n_runs": len(rows),
        "seeds": [int(r["seed"]) for r in rows],
        "summary": summary,
    }
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return csv_path, summary_json


def resolve_selection_metric(row, metric: str):
    if metric in row:
        return metric

    mean_metric = f"{metric}_mean"
    if mean_metric in row:
        return mean_metric

    raise KeyError(f"Selection metric '{metric}' is not available.")


def sort_trials(rows, metric: str, mode: str):
    metric_field = resolve_selection_metric(rows[0], metric)
    reverse = mode.lower() == "max"
    ranked = sorted(rows, key=lambda row: float(row[metric_field]), reverse=reverse)
    return ranked, metric_field


def write_outputs(output_root: Path, rows, best_trial, selection_metric: str, selection_metric_field: str, selection_mode: str, best_cfg):
    output_root.mkdir(parents=True, exist_ok=True)

    metric_fields = []
    for key in METRIC_KEYS:
        metric_fields.extend([f"{key}_mean", f"{key}_std"])

    reserved_fields = {"trial", "n_runs", "n_test"} | set(metric_fields)
    param_fields = sorted([k for k in rows[0].keys() if k not in reserved_fields])
    fieldnames = ["trial"] + param_fields + ["n_runs", "n_test"] + metric_fields

    results_csv = output_root / "sweep_results.csv"
    with results_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

        summary_row = {"trial": "mean±std across trials", "n_runs": rows[0]["n_runs"]}
        for key in param_fields:
            summary_row[key] = ""
        n_test_values = [float(r["n_test"]) for r in rows]
        summary_row["n_test"] = str(int(n_test_values[0])) if len(set(n_test_values)) == 1 else format_mean_std(n_test_values)
        for key in metric_fields:
            values = [float(r[key]) for r in rows]
            summary_row[key] = format_mean_std(values)
        writer.writerow(summary_row)

    summary = {
        "selection_metric": selection_metric,
        "selection_metric_field": selection_metric_field,
        "selection_mode": selection_mode,
        "n_trials": len(rows),
        "best_trial": best_trial,
        "mean_std": {},
    }
    for key in metric_fields:
        values = [float(r[key]) for r in rows]
        summary["mean_std"][key] = summarize_metric(values)
    n_test_values = [int(r["n_test"]) for r in rows]
    summary["mean_std"]["n_test"] = {
        "value": n_test_values[0] if len(set(n_test_values)) == 1 else n_test_values,
    }

    summary_json = output_root / "sweep_summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    best_cfg_path = output_root / "best_config.yaml"
    dump_yaml(best_cfg_path, best_cfg)
    return results_csv, summary_json, best_cfg_path


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]

    sweep_cfg_path = resolve_project_path(project_root, args.sweep_config)
    sweep_cfg = load_yaml(sweep_cfg_path)
    base_cfg_path = resolve_project_path(project_root, sweep_cfg["base_config"])
    base_cfg = load_yaml(base_cfg_path)

    run_name = str(sweep_cfg.get("run_name", "dimf_sweep"))
    selection_metric = str(sweep_cfg.get("selection_metric", "best_val"))
    selection_mode = str(sweep_cfg.get("selection_mode", "min"))
    trials = generate_trials(sweep_cfg)
    seeds = args.seeds if args.seeds else list(base_cfg.get("seeds", [base_cfg.get("seed", 42)]))
    if not seeds:
        raise ValueError("No seeds were provided and config does not contain a non-empty 'seeds' list.")

    output_root = project_root / "outputs" / run_name
    generated_root = output_root / "_generated_configs"
    rows = []

    for trial_index, params in enumerate(trials, start=1):
        trial_name = f"trial_{trial_index:03d}"
        trial_cfg = build_trial_config(base_cfg, params)
        trial_output_rel, trial_artifact_rel = build_trial_paths(run_name, trial_name)
        trial_output_dir = resolve_project_path(project_root, str(trial_output_rel))

        print(f"[{trial_index}/{len(trials)}] {trial_name}")
        for key, value in params.items():
            print(f"  {key} = {value}")
        print(f"  seeds = {seeds}")
        print(f"  output: {trial_output_dir}")

        seed_rows = []
        for seed in seeds:
            seed_cfg = build_seed_config(trial_cfg, trial_output_rel, trial_artifact_rel, int(seed))
            config_path = write_seed_config(generated_root, trial_name, int(seed), seed_cfg)
            seed_output_dir = resolve_project_path(project_root, seed_cfg["logging"]["output_dir"])
            metrics_path = seed_output_dir / "test_metrics.json"
            train_log_path = seed_output_dir / "train_log.jsonl"

            print(f"    seed={seed}")
            print(f"      config: {config_path}")
            print(f"      output: {seed_output_dir}")

            if args.dry_run:
                continue

            subprocess.run(
                [sys.executable, str(project_root / "train.py"), "--config", str(config_path), "--device", args.device],
                cwd=str(project_root),
                check=True,
            )

            if not metrics_path.exists():
                raise FileNotFoundError(f"Missing metrics file for {trial_name} seed {seed}: {metrics_path}")

            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            best_val = float(metrics.get("best_val", load_best_val(train_log_path)))
            seed_rows.append(
                {
                    "seed": int(seed),
                    "best_val": best_val,
                    "MSE": float(metrics["MSE"]),
                    "MAE": float(metrics["MAE"]),
                    "RMSE": float(metrics["RMSE"]),
                    "R2": float(metrics["R2"]),
                    "scaled_MSE": float(metrics["scaled_MSE"]),
                    "scaled_MAE": float(metrics["scaled_MAE"]),
                    "scaled_RMSE": float(metrics["scaled_RMSE"]),
                    "scaled_R2": float(metrics["scaled_R2"]),
                    "n_test": int(metrics["n_test"]),
                }
            )

        if args.dry_run:
            continue

        trial_summary = summarize_seed_rows(seed_rows)
        write_trial_outputs(trial_output_dir, params, seed_rows, trial_summary)

        n_test_value = trial_summary["n_test"]["value"]
        if not isinstance(n_test_value, int):
            raise ValueError(f"Expected a shared n_test per trial, got: {n_test_value}")

        row = {
            "trial": trial_name,
            "n_runs": len(seed_rows),
            "n_test": n_test_value,
        }
        row.update(params)
        for key in METRIC_KEYS:
            row[f"{key}_mean"] = float(trial_summary[key]["mean"])
            row[f"{key}_std"] = float(trial_summary[key]["std"])
        rows.append(row)

    if args.dry_run:
        print("Dry run complete. No training was launched.")
        return

    ranked, selection_metric_field = sort_trials(rows, selection_metric, selection_mode)
    best_trial = copy.deepcopy(ranked[0])
    best_params = {key: best_trial[key] for key in trials[0].keys()}
    best_cfg = build_trial_config(base_cfg, best_params)

    results_csv, summary_json, best_cfg_path = write_outputs(
        output_root,
        rows,
        best_trial,
        selection_metric,
        selection_metric_field,
        selection_mode,
        best_cfg,
    )

    print("=== Sweep Complete ===")
    print(f"Best trial: {best_trial['trial']}")
    print(f"Selection metric ({selection_mode}): {selection_metric_field} = {best_trial[selection_metric_field]:.6f}")
    print(f"MSE: {best_trial['MSE_mean']:.6f} ± {best_trial['MSE_std']:.6f}")
    print(f"RMSE: {best_trial['RMSE_mean']:.6f} ± {best_trial['RMSE_std']:.6f}")
    print(f"scaled_MSE: {best_trial['scaled_MSE_mean']:.6f} ± {best_trial['scaled_MSE_std']:.6f}")
    print(f"scaled_MAE: {best_trial['scaled_MAE_mean']:.6f} ± {best_trial['scaled_MAE_std']:.6f}")
    print(f"Saved: {results_csv}")
    print(f"Saved: {summary_json}")
    print(f"Saved: {best_cfg_path}")


if __name__ == "__main__":
    main()
