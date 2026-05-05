#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

if not os.environ.get("MPLCONFIGDIR"):
    _mpl_dir = Path.cwd() / ".matplotlib-codex"
    _mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(_mpl_dir)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from alignment_peak_metrics import attach_peak_metrics, build_peak_block_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create alignment-only summary tables and visualizations for a raw-gap comparison."
    )
    parser.add_argument("--summary", type=Path, required=True, help="Path to alignment_comparison_summary.json")
    parser.add_argument("--joined", type=Path, required=True, help="Path to alignment_test_joined.csv")
    parser.add_argument("--per-lag", type=Path, required=True, help="Path to alignment_per_lag_comparison.csv")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for figures and markdown summary")
    parser.add_argument("--aligned-estimates", type=Path, default=None)
    parser.add_argument("--noalign-estimates", type=Path, default=None)
    parser.add_argument("--edge", default="stage1_to_stage2")
    parser.add_argument("--thresholds", default="0.3,0.5")
    return parser.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _parse_thresholds(raw: str) -> List[float]:
    thresholds: List[float] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        value = float(chunk)
        if value <= 0.0 or value >= 1.0:
            raise ValueError("Each threshold must be in (0, 1), got %s" % chunk)
        thresholds.append(value)
    if not thresholds:
        raise ValueError("At least one threshold must be provided.")
    return sorted(set(thresholds))


def _threshold_tag(tau: float) -> str:
    return ("%.2f" % tau).replace(".", "p")


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#d1d5db", alpha=0.55)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_grouped_bar(
    ax: plt.Axes,
    labels: Sequence[str],
    aligned: Sequence[float],
    noalign: Sequence[float],
    title: str,
    ylabel: str,
) -> None:
    x = np.arange(len(labels), dtype=np.float64)
    width = 0.36
    ax.bar(x - width / 2, aligned, width=width, color="#2563eb", label="aligned")
    ax.bar(x + width / 2, noalign, width=width, color="#ef4444", label="noalign")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(frameon=False)
    _style_axis(ax)


def _write_subset_plot(summary: Dict[str, Any], output_path: Path) -> None:
    lag_recovery = summary["lag_recovery"]
    benchmark = summary["benchmark"]
    peak = benchmark.get("peak", {})

    labels = ["Lagged MAE", "Argmax acc", "AUPRC", "best-F1", "FAR"]
    aligned = [
        lag_recovery["aligned"]["lagged_only"]["expected_lag_mae"],
        lag_recovery["aligned"]["lagged_only"]["argmax_lag_accuracy"],
        benchmark["localization"]["aligned"]["auprc"],
        benchmark["localization"]["aligned"]["best_f1"],
        benchmark["block_out_false_alarm_rate"]["aligned"],
    ]
    noalign = [
        lag_recovery["noalign"]["lagged_only"]["expected_lag_mae"],
        lag_recovery["noalign"]["lagged_only"]["argmax_lag_accuracy"],
        benchmark["localization"]["noalign"]["auprc"],
        benchmark["localization"]["noalign"]["best_f1"],
        benchmark["block_out_false_alarm_rate"]["noalign"],
    ]

    fig, axes = plt.subplots(1, 2 if peak else 1, figsize=(13.5 if peak else 8.5, 4.6))
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])
    _plot_grouped_bar(axes[0], labels, aligned, noalign, "Alignment Detection And Lag Recovery", "value")

    if peak:
        peak_labels = ["Peak Error", "Hit@0", "Hit@+/-1"]
        peak_aligned = [
            peak["aligned"]["peak_error"],
            peak["aligned"]["peak_hit_at_0"],
            peak["aligned"]["peak_hit_at_pm1"],
        ]
        peak_noalign = [
            peak["noalign"]["peak_error"],
            peak["noalign"]["peak_hit_at_0"],
            peak["noalign"]["peak_hit_at_pm1"],
        ]
        _plot_grouped_bar(axes[1], peak_labels, peak_aligned, peak_noalign, "Peak Accuracy", "value")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_per_lag_plot(per_lag: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(per_lag["lag_gt"], per_lag["aligned_expected_lag_mae"], marker="o", color="#2563eb", label="aligned")
    ax.plot(per_lag["lag_gt"], per_lag["noalign_expected_lag_mae"], marker="o", color="#ef4444", label="noalign")
    ax.set_xlabel("true lag")
    ax.set_ylabel("expected-lag MAE")
    ax.set_title("Expected-Lag Error By True Lag", fontsize=12, fontweight="bold")
    ax.legend(frameon=False)
    _style_axis(ax)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _find_lag_blocks(joined: pd.DataFrame) -> List[Tuple[int, int]]:
    lag_mask = joined["lag_gt"].astype(float).to_numpy() > 0
    blocks: List[Tuple[int, int]] = []
    start = None
    for idx, is_lagged in enumerate(lag_mask):
        if is_lagged and start is None:
            start = idx
        elif not is_lagged and start is not None:
            blocks.append((start, idx - 1))
            start = None
    if start is not None:
        blocks.append((start, len(lag_mask) - 1))
    return blocks


def _write_block_plot(joined: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    working = joined.copy()
    working["TimeStamp"] = pd.to_datetime(working["TimeStamp"])
    working = working.sort_values("TimeStamp").reset_index(drop=True)
    blocks = _find_lag_blocks(working)
    if not blocks:
        raise ValueError("No lag blocks were found in alignment_test_joined.csv.")

    n_blocks = len(blocks)
    ncols = 2
    nrows = int(np.ceil(n_blocks / float(ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.0, max(3.2, 2.8 * nrows)), sharey=True)
    axes_arr = np.atleast_1d(axes).reshape(-1)

    summary_rows: List[Dict[str, Any]] = []
    for block_id, (start, end) in enumerate(blocks, start=1):
        ax = axes_arr[block_id - 1]
        left = max(0, start - 8)
        right = min(len(working), end + 9)
        view = working.iloc[left:right].copy()
        local_x = np.arange(len(view), dtype=np.float64)
        lag_start = start - left
        lag_end = end - left

        ax.axvspan(lag_start, lag_end, color="#fde68a", alpha=0.45)
        ax.plot(local_x, view["lag_gt"], color="#111827", linewidth=2.0, drawstyle="steps-post", label="true lag")
        ax.plot(local_x, view["aligned_pred_expected_lag"], color="#2563eb", linewidth=1.8, label="aligned")
        ax.plot(local_x, view["noalign_pred_expected_lag"], color="#ef4444", linewidth=1.8, label="noalign")
        ax.set_title("Block %d: %s to %s" % (block_id, str(working["TimeStamp"].iloc[start]), str(working["TimeStamp"].iloc[end])), fontsize=9)
        ax.set_xlabel("sample index in local window")
        ax.set_ylabel("lag")
        _style_axis(ax)

        block = working.iloc[start : end + 1]
        summary_rows.append(
            {
                "block_id": block_id,
                "start_time": str(block["TimeStamp"].iloc[0]),
                "end_time": str(block["TimeStamp"].iloc[-1]),
                "n_samples": int(len(block)),
                "true_lag_mode": int(block["lag_gt"].mode().iloc[0]),
                "true_peak_lag": float(block["lag_gt"].max()),
                "aligned_mean_pred_expected_lag": float(block["aligned_pred_expected_lag"].mean()),
                "noalign_mean_pred_expected_lag": float(block["noalign_pred_expected_lag"].mean()),
                "aligned_expected_lag_mae": float(block["aligned_expected_abs_error"].mean()),
                "noalign_expected_lag_mae": float(block["noalign_expected_abs_error"].mean()),
            }
        )

    for ax in axes_arr[n_blocks:]:
        ax.axis("off")

    handles, labels = axes_arr[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Lag Block Panels: True vs Predicted Expected Lag", fontsize=14, fontweight="bold", y=0.995)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(summary_rows)


def _pi_columns(frame: pd.DataFrame, edge: str) -> List[str]:
    prefix = "%s_pred_pi_lag" % edge
    cols = [col for col in frame.columns if col.startswith(prefix)]
    return sorted(cols, key=lambda name: int(name.split("lag")[-1]))


def _load_threshold_features(estimates_path: Path, edge: str, model: str) -> pd.DataFrame:
    estimates = pd.read_csv(estimates_path)
    pi_cols = _pi_columns(estimates, edge)
    if not pi_cols:
        raise ValueError("No probability columns for edge %s in %s" % (edge, estimates_path))
    nonzero_cols = pi_cols[1:]
    nonzero_prob = 1.0 - estimates[pi_cols[0]].astype(float)
    nonzero_pi = estimates[nonzero_cols].to_numpy(dtype=float)
    nonzero_lag_values = np.asarray([int(col.split("lag")[-1]) for col in nonzero_cols], dtype=int)
    argmax_nonzero = np.zeros(len(estimates), dtype=int)
    if len(nonzero_cols):
        argmax_nonzero = nonzero_lag_values[np.argmax(nonzero_pi, axis=1)]
    return pd.DataFrame(
        {
            "TimeStamp": pd.to_datetime(estimates["TimeStamp"]),
            "%s_pred_nonzero_prob" % model: nonzero_prob,
            "%s_pred_argmax_nonzero_lag" % model: argmax_nonzero,
        }
    )


def _threshold_metrics(true_lag: pd.Series, pred_lag: pd.Series) -> Dict[str, float]:
    true = true_lag.astype(int).to_numpy()
    pred = pred_lag.astype(int).to_numpy()
    true_nonzero = true > 0
    pred_nonzero = pred > 0
    tp = int(np.logical_and(true_nonzero, pred_nonzero).sum())
    fp = int(np.logical_and(~true_nonzero, pred_nonzero).sum())
    fn = int(np.logical_and(true_nonzero, ~pred_nonzero).sum())
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    recall = float(tp / (tp + fn)) if tp + fn else 0.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0
    lagged = true_nonzero
    return {
        "pred_nonzero_rate": float(pred_nonzero.mean()),
        "detect_precision": precision,
        "detect_recall": recall,
        "detect_f1": f1,
        "overall_exact_match": float((pred == true).mean()),
        "lagged_only_exact_match": float((pred[lagged] == true[lagged]).mean()) if lagged.any() else 0.0,
        "overall_mae": float(np.abs(pred - true).mean()),
        "lagged_only_mae": float(np.abs(pred[lagged] - true[lagged]).mean()) if lagged.any() else 0.0,
    }


def _attach_threshold_predictions(joined: pd.DataFrame, args: argparse.Namespace, thresholds: Sequence[float]) -> pd.DataFrame:
    if args.aligned_estimates is None or args.noalign_estimates is None:
        return joined.copy()

    enriched = joined.copy()
    enriched["TimeStamp"] = pd.to_datetime(enriched["TimeStamp"])
    for model, estimates_path in (("aligned", args.aligned_estimates), ("noalign", args.noalign_estimates)):
        required = ["%s_pred_nonzero_prob" % model, "%s_pred_argmax_nonzero_lag" % model]
        missing = [col for col in required if col not in enriched.columns]
        if missing:
            features = _load_threshold_features(estimates_path, args.edge, model)
            enriched = enriched.merge(features[["TimeStamp"] + missing], on="TimeStamp", how="left")

    for tau in thresholds:
        tag = _threshold_tag(tau)
        for model in ("aligned", "noalign"):
            pred_col = "%s_pred_thresholded_lag_tau%s" % (model, tag)
            hit_col = "%s_pred_threshold_hit_tau%s" % (model, tag)
            err_col = "%s_pred_threshold_abs_error_tau%s" % (model, tag)
            enriched[pred_col] = np.where(
                enriched["%s_pred_nonzero_prob" % model].astype(float) >= tau,
                enriched["%s_pred_argmax_nonzero_lag" % model].astype(int),
                0,
            )
            enriched[hit_col] = (enriched[pred_col].astype(int) == enriched["lag_gt"].astype(int)).astype(int)
            enriched[err_col] = (enriched[pred_col].astype(int) - enriched["lag_gt"].astype(int)).abs()
    return enriched


def _write_threshold_summary(enriched: pd.DataFrame, thresholds: Sequence[float], output_dir: Path) -> None:
    lines = [
        "# Thresholded Lag Summary",
        "",
        "```text",
        "if P(lag>0) < tau: predicted lag = 0",
        "else: predicted lag = argmax over non-zero lag bins",
        "```",
        "",
        "## Metrics",
        "",
        "| tau | model | pred_nonzero_rate | precision | recall | F1 | overall exact | lagged-only exact | overall MAE | lagged-only MAE |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for tau in thresholds:
        tag = _threshold_tag(tau)
        rows: List[Dict[str, Any]] = []
        for model in ("aligned", "noalign"):
            pred_col = "%s_pred_thresholded_lag_tau%s" % (model, tag)
            metrics = _threshold_metrics(enriched["lag_gt"], enriched[pred_col])
            row = {"tau": tau, "model": model}
            row.update(metrics)
            rows.append(row)
            lines.append(
                "| %.2f | %s | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |"
                % (
                    tau,
                    model,
                    metrics["pred_nonzero_rate"],
                    metrics["detect_precision"],
                    metrics["detect_recall"],
                    metrics["detect_f1"],
                    metrics["overall_exact_match"],
                    metrics["lagged_only_exact_match"],
                    metrics["overall_mae"],
                    metrics["lagged_only_mae"],
                )
            )
        pd.DataFrame(rows).to_csv(output_dir / ("lag_block_summary_threshold_pgt0_%s.csv" % tag), index=False)
    (output_dir / "thresholded_lag_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_summary_markdown(summary: Dict[str, Any], block_summary: pd.DataFrame, peak_summary: pd.DataFrame, output_path: Path) -> None:
    benchmark = summary["benchmark"]
    peak = benchmark.get("peak", {})
    lines = [
        "# Raw-Gap Alignment Summary",
        "",
        "## Core Alignment Metrics",
        "",
        "| metric | aligned | noalign |",
        "| --- | --- | --- |",
        "| Lagged-only expected lag MAE | %.3f | %.3f |"
        % (
            summary["lag_recovery"]["aligned"]["lagged_only"]["expected_lag_mae"],
            summary["lag_recovery"]["noalign"]["lagged_only"]["expected_lag_mae"],
        ),
        "| Lagged-only argmax acc | %.3f | %.3f |"
        % (
            summary["lag_recovery"]["aligned"]["lagged_only"]["argmax_lag_accuracy"],
            summary["lag_recovery"]["noalign"]["lagged_only"]["argmax_lag_accuracy"],
        ),
        "| Localization AUPRC | %.3f | %.3f |"
        % (benchmark["localization"]["aligned"]["auprc"], benchmark["localization"]["noalign"]["auprc"]),
        "| Localization best-F1 | %.3f | %.3f |"
        % (benchmark["localization"]["aligned"]["best_f1"], benchmark["localization"]["noalign"]["best_f1"]),
        "| Block-out false alarm rate | %.3f | %.3f |"
        % (benchmark["block_out_false_alarm_rate"]["aligned"], benchmark["block_out_false_alarm_rate"]["noalign"]),
    ]
    if peak:
        lines.extend(
            [
                "| Peak error | %.3f | %.3f |" % (peak["aligned"]["peak_error"], peak["noalign"]["peak_error"]),
                "| Peak hit@0 | %.3f | %.3f |" % (peak["aligned"]["peak_hit_at_0"], peak["noalign"]["peak_hit_at_0"]),
                "| Peak hit@+/-1 | %.3f | %.3f |" % (peak["aligned"]["peak_hit_at_pm1"], peak["noalign"]["peak_hit_at_pm1"]),
            ]
        )

    lines.extend(
        [
            "",
            "## Figure Files",
            "",
            "- [Lag subset metrics](./lag_subset_metrics.png)",
            "- [Lag MAE by true lag](./lag_mae_by_true_lag.png)",
            "- [Lag block panels](./lag_block_panels.png)",
            "",
            "## Lag Blocks",
            "",
            "| block | start | end | true peak | aligned peak | aligned peak error | hit@+/-1 |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in peak_summary.itertuples(index=False):
        lines.append(
            "| %d | %s | %s | %.3f | %.3f | %.3f | %d |"
            % (
                int(row.block_id),
                str(row.start_time),
                str(row.end_time),
                float(row.true_peak_lag),
                float(row.aligned_peak_expected_lag),
                float(row.aligned_peak_error),
                int(row.aligned_peak_hit_at_pm1),
            )
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = _load_json(args.summary)
    joined = pd.read_csv(args.joined)
    per_lag = pd.read_csv(args.per_lag)

    peak_summary = attach_peak_metrics(summary, joined)
    if peak_summary.empty:
        peak_summary = build_peak_block_table(joined)
    _save_json(args.summary, summary)

    peak_summary.to_csv(output_dir / "alignment_peak_summary.csv", index=False)
    _write_subset_plot(summary, output_dir / "lag_subset_metrics.png")
    _write_per_lag_plot(per_lag, output_dir / "lag_mae_by_true_lag.png")
    block_summary = _write_block_plot(joined, output_dir / "lag_block_panels.png")
    block_summary.to_csv(output_dir / "lag_block_summary.csv", index=False)

    thresholds = _parse_thresholds(args.thresholds)
    enriched = _attach_threshold_predictions(joined, args, thresholds)
    enriched.to_csv(output_dir / "alignment_test_joined_thresholded.csv", index=False)
    if args.aligned_estimates is not None and args.noalign_estimates is not None:
        _write_threshold_summary(enriched, thresholds, output_dir)

    _write_summary_markdown(summary, block_summary, peak_summary, output_dir / "alignment_summary_zh.md")

    print("Wrote: %s" % (output_dir / "lag_subset_metrics.png"))
    print("Wrote: %s" % (output_dir / "lag_mae_by_true_lag.png"))
    print("Wrote: %s" % (output_dir / "lag_block_panels.png"))
    print("Wrote: %s" % (output_dir / "alignment_summary_zh.md"))


if __name__ == "__main__":
    main()
