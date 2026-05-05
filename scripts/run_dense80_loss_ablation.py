#!/usr/bin/env python3

import argparse
import copy
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Union

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_VARIANTS: Dict[str, Dict[str, Any]] = {
    "baseline_final": {},
    "no_ent": {"train.lambda_ent": 0.0},
    "no_tv": {"train.lambda_tv": 0.0},
    "no_ent_tv": {"train.lambda_ent": 0.0, "train.lambda_tv": 0.0},
    "no_align": {"train.lambda_align": 0.0},
    "no_lag_sup": {
        "train.lambda_lag_supervision": 0.0,
        "train.lambda_lag_occurrence": 0.0,
        "train.lambda_lag_positive": 0.0,
    },
    "no_lag_posexp": {"train.lambda_lag_pos_expected_aux": 0.0},
}


def _path(text: Union[str, Path]) -> Path:
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _command_path(text: Union[str, Path]) -> str:
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    # Keep symlinks intact so virtualenv interpreters do not collapse back to
    # the underlying system Python.
    return os.fspath(path)


def _run(cmd: List[str], env: Dict[str, str] = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env, cwd=str(ROOT))


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _dump_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _set_nested(cfg: Dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cur = cfg
    for key in parts[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[parts[-1]] = value


def _train_log_summary(path: Path) -> Dict[str, Any]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    best_idx = min(range(len(rows)), key=lambda idx: float(rows[idx]["val_pred"]))
    best = rows[best_idx]
    final = rows[-1]
    out = {
        "best_epoch_by_val_pred": int(best["epoch"]),
        "best_val_pred": float(best["val_pred"]),
        "final_epoch": int(final["epoch"]),
        "final_val_pred": float(final["val_pred"]),
        "final_val_align": float(final["val_align"]),
        "final_val_ent": float(final["val_ent"]),
        "final_val_tv": float(final["val_tv"]),
        "final_val_lag_sup": float(final["val_lag_sup"]),
        "final_val_lag_posexp": float(final["val_lag_posexp"]),
        "final_val_stage12_lag": float(final["val_stage1_to_stage2_expected_lag"]),
        "final_val_stage12_peak": float(final["val_stage1_to_stage2_peak_prob"]),
        "weighted_final_val_align": float(final["val_align"]) * float(final["current_lambda_align"]),
        "weighted_final_val_ent": float(final["val_ent"]) * float(final["current_lambda_ent"]),
        "weighted_final_val_tv": float(final["val_tv"]) * float(final["current_lambda_tv"]),
        "weighted_final_val_lag_sup": float(final["val_lag_sup"]) * float(final["current_lambda_lag_supervision"]),
        "weighted_final_val_lag_posexp": float(final["val_lag_posexp"]) * float(final["current_lambda_lag_pos_expected_aux"]),
    }
    return out


def _resolve_compare_result_root(compare_root: Path, variant: str, seed: int) -> Path:
    candidates = [
        compare_root / f"seed_{seed}",
        compare_root / variant,
        compare_root,
    ]
    for candidate in candidates:
        if (candidate / "bump_eval" / "bump_method_summary.csv").exists() and (
            candidate / "r46b" / "strongkeep_veto_summary.csv"
        ).exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find compare outputs for variant={variant!r}, seed={seed} under {compare_root}"
    )


def _q40_summary(compare_root: Path, variant: str, seed: int) -> Dict[str, Any]:
    result_root = _resolve_compare_result_root(compare_root, variant=variant, seed=seed)
    bump_summary = pd.read_csv(result_root / "bump_eval" / "bump_method_summary.csv")
    veto_summary = pd.read_csv(result_root / "r46b" / "strongkeep_veto_summary.csv")
    q40 = bump_summary.loc[bump_summary["method"].astype(str) == "q40"].iloc[0]
    return {
        "q40_bump_in_mae": float(q40["bump_in_mae"]),
        "q40_outside_far": float(q40["outside_far"]),
        "q40_peak_time_error": float(q40["peak_time_error"]),
        "q40_peak_value_error": float(q40["peak_value_error"]),
        "q40_shape_corr": float(q40["shape_corr"]),
        "q40_eval_recall": float(veto_summary.iloc[0]["q40_eval_recall"]),
        "q40_eval_far": float(veto_summary.iloc[0]["q40_eval_FAR"]),
        "r46b_eval_recall": float(veto_summary.iloc[0]["strongkeep_veto_eval_recall"]),
        "r46b_eval_far": float(veto_summary.iloc[0]["strongkeep_veto_eval_FAR"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dense80 loss ablation on a single seed and evaluate q40 lag quality.")
    parser.add_argument(
        "--base-config",
        default="configs/multistage_localbump_mixed_balanced_evalsafe_aligned_segmentsplit_v3_nobias_tau1p2_posexp0p1_dense80.yaml",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument(
        "--variants",
        default=",".join(DEFAULT_VARIANTS.keys()),
        help="Comma-separated variant names to run.",
    )
    parser.add_argument(
        "--bump-csv",
        default="data/processed/LiquidSugar_local_bump_mixed_balanced_evalsafe_segmentsplit_v3_dense80_rawgap.csv",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/dense80_loss_ablation_seed42",
    )
    parser.add_argument(
        "--compare-root",
        default="outputs/dense80_loss_ablation_compare_seed42",
    )
    parser.add_argument(
        "--compare-artifact-mode",
        choices=["full", "summary"],
        default="summary",
        help="Whether compare outputs should keep all intermediate artifacts or be pruned to compact summaries.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    python_bin = _command_path(args.python_bin)
    base_cfg = _load_yaml(_path(args.base_config))
    output_root = _path(args.output_root)
    compare_root = _path(args.compare_root)
    bump_csv = str(_path(args.bump_csv))
    variants = [part.strip() for part in str(args.variants).split(",") if part.strip()]

    env = dict(**os.environ)
    env.setdefault("MPLCONFIGDIR", "/tmp/mpl")

    summary_rows: List[Dict[str, Any]] = []
    for variant in variants:
        if variant not in DEFAULT_VARIANTS:
            raise ValueError(f"Unknown variant: {variant}")

        cfg = copy.deepcopy(base_cfg)
        cfg["seed"] = int(args.seed)
        cfg["seeds"] = [int(args.seed)]
        cfg.setdefault("train", {})
        cfg["train"]["eval_checkpoint"] = "final"

        variant_output = output_root / variant
        variant_artifact = ROOT / "artifacts" / output_root.name / variant
        cfg["logging"]["output_dir"] = str(variant_output)
        cfg["logging"]["log_path"] = str(variant_output / "train_log.jsonl")
        cfg["logging"]["ckpt_path"] = str(variant_artifact / "best.ckpt")
        cfg["logging"]["scaler_path"] = str(variant_artifact / "scaler.pkl")
        cfg["logging"]["final_ckpt_path"] = str(variant_artifact / "best.ckpt")

        for dotted_key, value in DEFAULT_VARIANTS[variant].items():
            _set_nested(cfg, dotted_key, value)

        cfg_path = output_root / "_generated_configs" / f"{variant}.yaml"
        _dump_yaml(cfg_path, cfg)

        compare_variant_root = compare_root / variant
        try:
            final_summary_path = _resolve_compare_result_root(compare_variant_root, variant=variant, seed=int(args.seed)) / "bump_eval" / "bump_method_summary.csv"
        except FileNotFoundError:
            final_summary_path = compare_variant_root / f"seed_{args.seed}" / "bump_eval" / "bump_method_summary.csv"
        if not (bool(args.skip_existing) and final_summary_path.exists()):
            _run([python_bin, str(ROOT / "train.py"), "--config", str(cfg_path), "--device", str(args.device)], env=env)
            _run(
                [
                    python_bin,
                    str(ROOT / "scripts" / "run_multiseed_bump_component_compare.py"),
                    "--configs",
                    str(cfg_path),
                    "--bump-csv",
                    bump_csv,
                    "--output-root",
                    str(compare_variant_root),
                    "--python-bin",
                    python_bin,
                    "--device",
                    str(args.device),
                    "--artifact-mode",
                    str(args.compare_artifact_mode),
                ],
                env=env,
            )

        row = {
            "variant": variant,
            "seed": int(args.seed),
            "base_config": str(_path(args.base_config).relative_to(ROOT)),
            "train_lr": float(cfg["train"]["lr"]),
            **_train_log_summary(_path(cfg["logging"]["log_path"])),
            **_q40_summary(compare_variant_root, variant=variant, seed=int(args.seed)),
        }
        row["overrides"] = json.dumps(DEFAULT_VARIANTS[variant], ensure_ascii=False, sort_keys=True)
        summary_rows.append(row)

    summary_path = compare_root / "dense80_loss_ablation_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(summary_rows[0].keys())
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(pd.DataFrame(summary_rows).to_csv(index=False))
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
