#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render summary tables for single-seed baseline and 5-seed v5 benchmark."
    )
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--baseline-conditional", type=Path, required=True)
    parser.add_argument("--multiseed-benchmark-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_conditional_row(path: Path, true_lag: int):
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(float(row["true_lag"])) == int(true_lag):
                return row
    return None


def _fmt_scalar(value: float) -> str:
    return f"{float(value):.3f}"


def _fmt_mean_std(values) -> str:
    if not values:
        return "N/A"
    if len(values) == 1:
        return _fmt_scalar(values[0])
    return f"{mean(values):.3f} +/- {pstdev(values):.3f}"


def _render_table(df: pd.DataFrame, title: str, subtitle: str, output_path: Path):
    n_rows, n_cols = df.shape
    fig_w = max(8.0, 2.2 * n_cols)
    fig_h = max(2.8, 1.1 + 0.65 * n_rows)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.5)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#d9e8fb")
            cell.set_text_props(weight="bold")
        elif row % 2 == 1:
            cell.set_facecolor("#f7f9fc")
        else:
            cell.set_facecolor("#ffffff")
        cell.set_edgecolor("#8da0b6")

    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.97)
    ax.set_title(subtitle, fontsize=10, pad=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_summary = _load_json(args.baseline_summary)
    baseline_cond = _load_conditional_row(args.baseline_conditional, true_lag=0)
    if baseline_cond is None:
        raise ValueError("Could not find true_lag=0 row in baseline conditional table.")

    noalign_row = {
        "Model": "noalign",
        "Block-in MAE": _fmt_scalar(baseline_summary["benchmark"]["block_in_expected_lag_mae"]["noalign"]),
        "AUPRC": _fmt_scalar(baseline_summary["benchmark"]["localization"]["noalign"]["auprc"]),
        "FAR": _fmt_scalar(baseline_summary["benchmark"]["block_out_false_alarm_rate"]["noalign"]),
        "E[d_hat|d=0]": _fmt_scalar(float(baseline_cond["noalign_mean_pred_expected_lag"])),
    }
    baseline_row = {
        "Model": "no-bias + tau=1.2",
        "Block-in MAE": _fmt_scalar(baseline_summary["benchmark"]["block_in_expected_lag_mae"]["aligned"]),
        "AUPRC": _fmt_scalar(baseline_summary["benchmark"]["localization"]["aligned"]["auprc"]),
        "FAR": _fmt_scalar(baseline_summary["benchmark"]["block_out_false_alarm_rate"]["aligned"]),
        "E[d_hat|d=0]": _fmt_scalar(float(baseline_cond["aligned_mean_pred_expected_lag"])),
    }

    per_seed = []
    for seed_dir in sorted(path for path in args.multiseed_benchmark_root.iterdir() if path.is_dir() and path.name.startswith("seed_")):
        summary_path = seed_dir / "alignment_comparison_summary.json"
        joined_path = seed_dir / "alignment_test_joined.csv"
        if not summary_path.exists() or not joined_path.exists():
            continue
        summary = _load_json(summary_path)

        d0_values = []
        with joined_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if int(float(row["lag_gt"])) == 0:
                    d0_values.append(float(row["aligned_pred_expected_lag"]))

        per_seed.append(
            {
                "summary": summary,
                "d0": mean(d0_values) if d0_values else None,
            }
        )

    if not per_seed:
        raise ValueError("No seed benchmark summaries found under multiseed root.")

    v5_row = {
        "Model": "v5 (+lambda_pos=0.1)",
        "Block-in MAE": _fmt_mean_std([item["summary"]["benchmark"]["block_in_expected_lag_mae"]["aligned"] for item in per_seed]),
        "AUPRC": _fmt_mean_std([item["summary"]["benchmark"]["localization"]["aligned"]["auprc"] for item in per_seed]),
        "FAR": _fmt_mean_std([item["summary"]["benchmark"]["block_out_false_alarm_rate"]["aligned"] for item in per_seed]),
        "E[d_hat|d=0]": _fmt_mean_std([item["d0"] for item in per_seed if item["d0"] is not None]),
    }

    comparison_df = pd.DataFrame([noalign_row, baseline_row, v5_row])
    comparison_csv = output_dir / "final_comparison_table.csv"
    comparison_png = output_dir / "final_comparison_table.png"
    comparison_df.to_csv(comparison_csv, index=False)
    _render_table(
        comparison_df,
        title="Current Final Comparison",
        subtitle="v5 uses 5-seed mean+/-std; noalign and no-bias baseline are current single-seed references.",
        output_path=comparison_png,
    )

    dmax_rows = []
    for dmax in [2, 4, 6]:
        items = []
        sample_counts = []
        for item in per_seed:
            by_dmax = item["summary"].get("benchmark_by_dmax", {})
            if str(dmax) not in by_dmax:
                continue
            payload = by_dmax[str(dmax)]
            items.append(payload)
            sample_counts.append(int(payload.get("n_samples", 0)))
        if not items:
            dmax_rows.append(
                {
                    "true dmax": f"d{dmax}",
                    "n_samples": "0",
                    "Block-in MAE": "N/A",
                    "AUPRC": "N/A",
                    "FAR": "N/A",
                }
            )
            continue
        dmax_rows.append(
            {
                "true dmax": f"d{dmax}",
                "n_samples": str(int(mean(sample_counts))),
                "Block-in MAE": _fmt_mean_std([float(x["aligned"]["block_in_expected_lag_mae"]) for x in items]),
                "AUPRC": _fmt_mean_std([float(x["aligned"]["localization"]["auprc"]) for x in items]),
                "FAR": _fmt_mean_std([float(x["aligned"]["block_out_false_alarm_rate"]) for x in items]),
            }
        )

    dmax_df = pd.DataFrame(dmax_rows)
    dmax_csv = output_dir / "conditioned_by_true_dmax_5seed.csv"
    dmax_png = output_dir / "conditioned_by_true_dmax_5seed.png"
    dmax_df.to_csv(dmax_csv, index=False)
    _render_table(
        dmax_df,
        title="5-Seed Conditioned by True dmax",
        subtitle="Rows summarize v5 benchmark metrics aggregated across seeds.",
        output_path=dmax_png,
    )

    print(f"Wrote: {comparison_csv}")
    print(f"Wrote: {comparison_png}")
    print(f"Wrote: {dmax_csv}")
    print(f"Wrote: {dmax_png}")


if __name__ == "__main__":
    main()
