#!/usr/bin/env python3

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import yaml


DEFAULT_CONFIGS = [
    "configs/multistage_aligned_stage12_piecewise_delay.yaml",
    "configs/multistage_aligned_stage12_linear_delay.yaml",
    "configs/multistage_aligned_stage12_sinusoidal_delay.yaml",
    "configs/multistage_aligned_stage12_bimodal_delay.yaml",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run DIMF synthetic-delay recovery experiments and compare estimated delays with ground truth."
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=DEFAULT_CONFIGS,
        help="One or more config files. Defaults to the stage1->stage2 piecewise/linear/bimodal suite.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device passed through to train.py")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only, without launching them")
    parser.add_argument("--skip-train", action="store_true", help="Skip train.py and only run delay evaluation")
    parser.add_argument("--skip-eval", action="store_true", help="Skip delay evaluation and only run training")
    parser.add_argument("--skip-plots", action="store_true", help="Skip delay recovery plotting")
    parser.add_argument(
        "--summary-dir",
        type=str,
        default="outputs/delay_recovery_suite",
        help="Directory used for the suite-level summary files.",
    )
    return parser.parse_args()


def resolve_project_path(project_root, path_value):
    path = Path(path_value)
    if path.is_absolute():
        return path
    return project_root / path


def load_config(path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def format_command(cmd):
    return " ".join(str(part) for part in cmd)


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _append_gap_analysis_args(cmd, project_root, cfg):
    analysis_cfg = cfg.get("analysis", {})
    raw_reference_value = analysis_cfg.get("raw_reference_path")
    if not raw_reference_value:
        return

    interval_min = int(cfg.get("data", {}).get("collection_interval_min", 15))
    long_gap_min_slots = int(analysis_cfg.get("long_gap_min_slots", 8))
    raw_reference_path = resolve_project_path(project_root, raw_reference_value)
    cmd.extend(
        [
            "--raw-reference",
            str(raw_reference_path),
            "--interval-min",
            str(interval_min),
            "--long-gap-min-slots",
            str(long_gap_min_slots),
        ]
    )


def write_suite_summary(summary_dir, rows):
    summary_dir.mkdir(parents=True, exist_ok=True)
    csv_path = summary_dir / "delay_recovery_suite_summary.csv"
    json_path = summary_dir / "delay_recovery_suite_summary.json"

    fieldnames = [
        "config",
        "output_dir",
        "edge",
        "n_matched",
        "expected_lag_mae",
        "expected_lag_rmse",
        "argmax_lag_mae",
        "argmax_lag_accuracy",
        "mean_js_divergence",
        "mean_pred_entropy",
        "outside_long_gap_n_matched",
        "outside_long_gap_expected_lag_mae",
        "outside_long_gap_expected_lag_rmse",
        "outside_long_gap_argmax_lag_mae",
        "outside_long_gap_argmax_lag_accuracy",
        "outside_long_gap_mean_js_divergence",
        "outside_long_gap_mean_pred_entropy",
        "inside_long_gap_n_matched",
        "inside_long_gap_expected_lag_mae",
        "inside_long_gap_expected_lag_rmse",
        "inside_long_gap_argmax_lag_mae",
        "inside_long_gap_argmax_lag_accuracy",
        "inside_long_gap_mean_js_divergence",
        "inside_long_gap_mean_pred_entropy",
        "MSE",
        "MAE",
        "RMSE",
        "R2",
        "scaled_MSE",
        "scaled_MAE",
        "scaled_RMSE",
        "scaled_R2",
        "best_val",
        "n_test",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    payload = {"runs": rows}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return csv_path, json_path


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    summary_rows = []

    for idx, config_value in enumerate(args.configs, start=1):
        config_path = resolve_project_path(project_root, config_value)
        cfg = load_config(config_path)
        analysis_cfg = cfg.get("analysis", {})
        truth_value = analysis_cfg.get("delay_truth_path")
        target_edge = analysis_cfg.get("target_edge")
        if not truth_value or not target_edge:
            raise ValueError(
                "Config %s must define analysis.delay_truth_path and analysis.target_edge" % config_path
            )

        output_dir = resolve_project_path(project_root, cfg["logging"]["output_dir"])
        truth_path = resolve_project_path(project_root, truth_value)
        estimates_path = output_dir / "test_delay_estimates.csv"
        prediction_csv_path = output_dir / "test_pred_vs_true.csv"
        recovery_summary_path = output_dir / "delay_recovery_summary.json"
        test_metrics_path = output_dir / "test_metrics.json"
        plot_dir = output_dir / "plots"

        train_cmd = [sys.executable, str(project_root / "train.py"), "--config", str(config_path), "--device", args.device]
        eval_cmd = [
            sys.executable,
            str(project_root / "scripts" / "evaluate_delay_estimates.py"),
            "--estimates",
            str(estimates_path),
            "--truth",
            str(truth_path),
            "--edge",
            str(target_edge),
            "--output-dir",
            str(output_dir),
        ]
        _append_gap_analysis_args(eval_cmd, project_root, cfg)
        plot_cmd = [
            sys.executable,
            str(project_root / "scripts" / "plot_delay_recovery.py"),
            "--estimates",
            str(estimates_path),
            "--truth",
            str(truth_path),
            "--edge",
            str(target_edge),
            "--output-dir",
            str(plot_dir),
            "--title-prefix",
            str(Path(config_path).stem),
        ]
        _append_gap_analysis_args(plot_cmd, project_root, cfg)
        pred_plot_cmd = [
            sys.executable,
            str(project_root / "scripts" / "plot_test_predictions.py"),
            "--predictions",
            str(prediction_csv_path),
            "--output-dir",
            str(plot_dir),
            "--title-prefix",
            str(Path(config_path).stem),
        ]

        print("[%d/%d] %s" % (idx, len(args.configs), config_path))
        print("  output_dir: %s" % output_dir)
        print("  train_cmd: %s" % format_command(train_cmd))
        print("  eval_cmd:  %s" % format_command(eval_cmd))
        print("  plot_cmd:  %s" % format_command(plot_cmd))
        print("  pred_plot: %s" % format_command(pred_plot_cmd))

        if args.dry_run:
            continue

        if not args.skip_train:
            subprocess.run(train_cmd, cwd=str(project_root), check=True)

        if not args.skip_eval:
            subprocess.run(eval_cmd, cwd=str(project_root), check=True)

        if not args.skip_plots:
            subprocess.run(plot_cmd, cwd=str(project_root), check=True)
            subprocess.run(pred_plot_cmd, cwd=str(project_root), check=True)

        if recovery_summary_path.exists() and test_metrics_path.exists():
            summary_payload = load_json(recovery_summary_path)
            test_metrics = load_json(test_metrics_path)
            metrics = summary_payload.get(target_edge)
            if metrics:
                outside_metrics = metrics.get("outside_long_gap_spans") or {}
                inside_metrics = metrics.get("inside_long_gap_spans") or {}
                summary_rows.append(
                    {
                        "config": str(config_path),
                        "output_dir": str(output_dir),
                        "edge": str(target_edge),
                        "n_matched": int(metrics["n_matched"]),
                        "expected_lag_mae": float(metrics["expected_lag_mae"]),
                        "expected_lag_rmse": float(metrics["expected_lag_rmse"]),
                        "argmax_lag_mae": float(metrics["argmax_lag_mae"]),
                        "argmax_lag_accuracy": float(metrics["argmax_lag_accuracy"]),
                        "mean_js_divergence": float(metrics["mean_js_divergence"]),
                        "mean_pred_entropy": float(metrics["mean_pred_entropy"]),
                        "outside_long_gap_n_matched": outside_metrics.get("n_matched"),
                        "outside_long_gap_expected_lag_mae": outside_metrics.get("expected_lag_mae"),
                        "outside_long_gap_expected_lag_rmse": outside_metrics.get("expected_lag_rmse"),
                        "outside_long_gap_argmax_lag_mae": outside_metrics.get("argmax_lag_mae"),
                        "outside_long_gap_argmax_lag_accuracy": outside_metrics.get("argmax_lag_accuracy"),
                        "outside_long_gap_mean_js_divergence": outside_metrics.get("mean_js_divergence"),
                        "outside_long_gap_mean_pred_entropy": outside_metrics.get("mean_pred_entropy"),
                        "inside_long_gap_n_matched": inside_metrics.get("n_matched"),
                        "inside_long_gap_expected_lag_mae": inside_metrics.get("expected_lag_mae"),
                        "inside_long_gap_expected_lag_rmse": inside_metrics.get("expected_lag_rmse"),
                        "inside_long_gap_argmax_lag_mae": inside_metrics.get("argmax_lag_mae"),
                        "inside_long_gap_argmax_lag_accuracy": inside_metrics.get("argmax_lag_accuracy"),
                        "inside_long_gap_mean_js_divergence": inside_metrics.get("mean_js_divergence"),
                        "inside_long_gap_mean_pred_entropy": inside_metrics.get("mean_pred_entropy"),
                        "MSE": float(test_metrics["MSE"]),
                        "MAE": float(test_metrics["MAE"]),
                        "RMSE": float(test_metrics["RMSE"]),
                        "R2": float(test_metrics["R2"]),
                        "scaled_MSE": float(test_metrics["scaled_MSE"]),
                        "scaled_MAE": float(test_metrics["scaled_MAE"]),
                        "scaled_RMSE": float(test_metrics["scaled_RMSE"]),
                        "scaled_R2": float(test_metrics["scaled_R2"]),
                        "best_val": float(test_metrics["best_val"]),
                        "n_test": int(test_metrics["n_test"]),
                    }
                )

    if args.dry_run:
        print("Dry run complete. No commands were launched.")
        return

    if not summary_rows:
        print("No suite summary was written because no delay_recovery_summary.json files were produced.")
        return

    suite_summary_dir = resolve_project_path(project_root, args.summary_dir)
    csv_path, json_path = write_suite_summary(suite_summary_dir, summary_rows)
    print("Saved: %s" % csv_path)
    print("Saved: %s" % json_path)


if __name__ == "__main__":
    main()
