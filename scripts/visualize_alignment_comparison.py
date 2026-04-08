#!/usr/bin/env python3

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create summary tables and visualizations for the raw-gap alignment comparison."
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("outputs/rawgap_stage2lag_alignment_compare/alignment_comparison_summary.json"),
        help="Path to alignment_comparison_summary.json",
    )
    parser.add_argument(
        "--joined",
        type=Path,
        default=Path("outputs/rawgap_stage2lag_alignment_compare/alignment_test_joined.csv"),
        help="Path to alignment_test_joined.csv",
    )
    parser.add_argument(
        "--per-lag",
        type=Path,
        default=Path("outputs/rawgap_stage2lag_alignment_compare/alignment_per_lag_comparison.csv"),
        help="Path to alignment_per_lag_comparison.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/rawgap_stage2lag_alignment_compare/visuals"),
        help="Directory for figures and markdown summary",
    )
    parser.add_argument(
        "--aligned-estimates",
        type=Path,
        default=None,
        help="Optional aligned run test_delay_estimates.csv for thresholded lag visualizations",
    )
    parser.add_argument(
        "--noalign-estimates",
        type=Path,
        default=None,
        help="Optional no-alignment run test_delay_estimates.csv for thresholded lag visualizations",
    )
    parser.add_argument(
        "--edge",
        default="stage1_to_stage2",
        help="Edge name used to read lag probabilities from test_delay_estimates.csv",
    )
    parser.add_argument(
        "--thresholds",
        default="0.3,0.5",
        help="Comma-separated thresholds for the display rule P(lag>0) >= tau",
    )
    return parser.parse_args()


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(str(path)))


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _pi_columns(frame: pd.DataFrame, edge: str) -> List[str]:
    prefix = "%s_pred_pi_lag" % edge
    cols = [col for col in frame.columns if col.startswith(prefix)]
    return sorted(cols, key=lambda name: int(name.split("lag")[-1]))


def _threshold_tag(tau: float) -> str:
    return str(tau).replace(".", "p")


def _parse_thresholds(raw: str) -> List[float]:
    values = []
    for chunk in str(raw).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        tau = float(chunk)
        if not (0.0 < tau < 1.0):
            raise ValueError("Each threshold must be in (0, 1), got %s" % chunk)
        values.append(tau)
    if not values:
        raise ValueError("At least one threshold must be provided")
    return sorted(set(values))


def _find_lag_blocks(joined: pd.DataFrame) -> List[Dict[str, int]]:
    lag_mask = joined["lag_gt"].to_numpy(dtype=np.int64) > 0
    blocks = []
    start = None
    for idx, is_lagged in enumerate(lag_mask):
        if is_lagged and start is None:
            start = idx
        if not is_lagged and start is not None:
            blocks.append({"start": start, "end": idx - 1})
            start = None
    if start is not None:
        blocks.append({"start": start, "end": len(joined) - 1})
    return blocks


def _write_forecast_plot(summary: Dict, output_path: Path) -> None:
    metrics = summary["forecast_metrics"]
    labels = ["aligned", "noalign"]
    mae = [metrics[label]["MAE"] for label in labels]
    rmse = [metrics[label]["RMSE"] for label in labels]
    r2 = [metrics[label]["R2"] for label in labels]

    x = np.arange(len(labels), dtype=np.float64)
    width = 0.34

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.bar(x - width / 2.0, mae, width=width, label="MAE", color="#2563eb")
    ax.bar(x + width / 2.0, rmse, width=width, label="RMSE", color="#f59e0b")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Forecast error")
    ax.set_title("Forecast Metrics on Raw-Gap Test Set")
    ax.legend(frameon=False)

    text = "\n".join(["%s R2 = %.3f" % (label, value) for label, value in zip(labels, r2)])
    ax.text(
        0.98,
        0.98,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#d1d5db"},
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_subset_plot(summary: Dict, output_path: Path) -> None:
    subsets = ["overall", "lagged_only", "no_lag_only"]
    labels = ["overall", "lagged only", "no-lag only"]
    models = ["aligned", "noalign"]
    colors = {"aligned": "#2563eb", "noalign": "#ef4444"}
    x = np.arange(len(subsets), dtype=np.float64)
    width = 0.34

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    for idx, model in enumerate(models):
        offset = (-0.5 + idx) * width
        mae_values = [summary["lag_recovery"][model][subset]["expected_lag_mae"] for subset in subsets]
        acc_values = [summary["lag_recovery"][model][subset]["argmax_lag_accuracy"] for subset in subsets]
        axes[0].bar(x + offset, mae_values, width=width, label=model, color=colors[model])
        axes[1].bar(x + offset, acc_values, width=width, label=model, color=colors[model])

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Expected lag MAE")
    axes[0].set_title("Lag Error by Subset")

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Argmax lag accuracy")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_title("Argmax Accuracy by Subset")
    axes[1].legend(frameon=False, loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_per_lag_plot(per_lag: pd.DataFrame, output_path: Path) -> None:
    lag_values = per_lag["lag_gt"].astype(int).tolist()
    x = np.arange(len(lag_values), dtype=np.float64)
    width = 0.34

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.bar(
        x - width / 2.0,
        per_lag["aligned_expected_lag_mae"].to_numpy(dtype=np.float64),
        width=width,
        label="aligned",
        color="#2563eb",
    )
    ax.bar(
        x + width / 2.0,
        per_lag["noalign_expected_lag_mae"].to_numpy(dtype=np.float64),
        width=width,
        label="noalign",
        color="#ef4444",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(value) for value in lag_values])
    ax.set_xlabel("True lag")
    ax.set_ylabel("Expected lag MAE")
    ax.set_title("Expected-Lag Error by True Lag")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _load_threshold_features(estimates_path: Path, edge: str, label: str) -> pd.DataFrame:
    estimates = pd.read_csv(estimates_path)
    estimates["TimeStamp"] = pd.to_datetime(estimates["TimeStamp"])
    pi_cols = _pi_columns(estimates, edge)
    if not pi_cols:
        raise ValueError("No probability columns for edge %s in %s" % (edge, estimates_path))

    pi = estimates[pi_cols].to_numpy(dtype=np.float64)
    lag_values = np.asarray([int(col.split("lag")[-1]) for col in pi_cols], dtype=np.int64)
    nonzero_prob = 1.0 - pi[:, 0]
    peak_prob = pi.max(axis=1)
    if pi.shape[1] > 1:
        nonzero_pi = pi[:, 1:]
        nonzero_lag_values = lag_values[1:]
        argmax_nonzero = nonzero_lag_values[nonzero_pi.argmax(axis=1)]
    else:
        argmax_nonzero = np.zeros(pi.shape[0], dtype=np.int64)

    return pd.DataFrame(
        {
            "TimeStamp": estimates["TimeStamp"],
            "%s_pred_nonzero_prob" % label: nonzero_prob,
            "%s_pred_peak_prob" % label: peak_prob,
            "%s_pred_argmax_nonzero_lag" % label: argmax_nonzero,
        }
    )


def _attach_threshold_predictions(
    joined: pd.DataFrame,
    aligned_estimates: Path,
    noalign_estimates: Path,
    edge: str,
    thresholds: List[float],
) -> pd.DataFrame:
    enriched = joined.copy()
    enriched["TimeStamp"] = pd.to_datetime(enriched["TimeStamp"])
    aligned_features = _load_threshold_features(aligned_estimates, edge=edge, label="aligned")
    noalign_features = _load_threshold_features(noalign_estimates, edge=edge, label="noalign")
    enriched = enriched.merge(aligned_features, on="TimeStamp", how="left")
    enriched = enriched.merge(noalign_features, on="TimeStamp", how="left")

    if enriched[["aligned_pred_nonzero_prob", "noalign_pred_nonzero_prob"]].isna().any().any():
        raise ValueError("Threshold features could not be aligned to joined comparison rows")

    lag_gt = enriched["lag_gt"].to_numpy(dtype=np.int64)
    for tau in thresholds:
        tau_tag = _threshold_tag(tau)
        for label in ("aligned", "noalign"):
            nonzero_prob = enriched["%s_pred_nonzero_prob" % label].to_numpy(dtype=np.float64)
            argmax_nonzero = enriched["%s_pred_argmax_nonzero_lag" % label].to_numpy(dtype=np.int64)
            pred = np.where(nonzero_prob >= tau, argmax_nonzero, 0).astype(np.int64)
            col = "%s_pred_thresholded_lag_tau%s" % (label, tau_tag)
            enriched[col] = pred
            enriched["%s_pred_threshold_hit_tau%s" % (label, tau_tag)] = (pred == lag_gt).astype(np.int64)
            enriched["%s_pred_threshold_abs_error_tau%s" % (label, tau_tag)] = np.abs(pred - lag_gt)

    return enriched


def _write_block_plot(joined: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    joined = joined.copy()
    joined["TimeStamp"] = pd.to_datetime(joined["TimeStamp"])
    blocks = _find_lag_blocks(joined)
    if not blocks:
        raise ValueError("No lag blocks were found in alignment_test_joined.csv")

    context = 16
    n_blocks = len(blocks)
    ncols = 2
    nrows = int(math.ceil(n_blocks / float(ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.8 * nrows), sharey=True)
    axes_arr = np.atleast_1d(axes).reshape(nrows, ncols)

    block_rows = []
    for block_idx, block in enumerate(blocks):
        row_idx = block_idx // ncols
        col_idx = block_idx % ncols
        ax = axes_arr[row_idx, col_idx]

        left = max(0, block["start"] - context)
        right = min(len(joined) - 1, block["end"] + context)
        view = joined.iloc[left : right + 1].copy()
        x = np.arange(len(view), dtype=np.int64)

        ax.plot(x, view["lag_gt"], label="true lag", color="#111827", linewidth=2.2, drawstyle="steps-post")
        ax.plot(x, view["aligned_pred_expected_lag"], label="aligned expected lag", color="#2563eb", linewidth=1.8)
        ax.plot(x, view["noalign_pred_expected_lag"], label="noalign expected lag", color="#ef4444", linewidth=1.8)

        lag_start = block["start"] - left
        lag_end = block["end"] - left
        ax.axvspan(lag_start, lag_end, color="#fde68a", alpha=0.35)
        ax.set_ylim(-0.3, 7.2)
        ax.set_title(
            "Block %d: %s to %s"
            % (
                block_idx + 1,
                joined.iloc[block["start"]]["TimeStamp"].strftime("%Y-%m-%d %H:%M"),
                joined.iloc[block["end"]]["TimeStamp"].strftime("%Y-%m-%d %H:%M"),
            ),
            fontsize=10,
        )
        ax.set_xlabel("sample index in local window")
        ax.set_ylabel("lag")

        block_view = joined.iloc[block["start"] : block["end"] + 1]
        block_rows.append(
            {
                "block_id": block_idx + 1,
                "start_time": block_view["TimeStamp"].iloc[0].strftime("%Y-%m-%d %H:%M"),
                "end_time": block_view["TimeStamp"].iloc[-1].strftime("%Y-%m-%d %H:%M"),
                "n_samples": int(len(block_view)),
                "true_lag_mode": int(block_view["lag_gt"].mode().iloc[0]),
                "aligned_mean_pred_expected_lag": float(block_view["aligned_pred_expected_lag"].mean()),
                "noalign_mean_pred_expected_lag": float(block_view["noalign_pred_expected_lag"].mean()),
                "aligned_expected_lag_mae": float(block_view["aligned_expected_abs_error"].mean()),
                "noalign_expected_lag_mae": float(block_view["noalign_expected_abs_error"].mean()),
            }
        )

    for idx in range(n_blocks, nrows * ncols):
        axes_arr[idx // ncols, idx % ncols].axis("off")

    handles, labels = axes_arr[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.99))
    fig.suptitle("Lag Block Panels: True vs Predicted Expected Lag", y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return pd.DataFrame(block_rows)


def _write_threshold_block_plot(joined: pd.DataFrame, tau: float, output_path: Path) -> pd.DataFrame:
    joined = joined.copy()
    joined["TimeStamp"] = pd.to_datetime(joined["TimeStamp"])
    blocks = _find_lag_blocks(joined)
    if not blocks:
        raise ValueError("No lag blocks were found in alignment_test_joined.csv")

    tau_tag = _threshold_tag(tau)
    aligned_col = "aligned_pred_thresholded_lag_tau%s" % tau_tag
    noalign_col = "noalign_pred_thresholded_lag_tau%s" % tau_tag
    context = 16
    n_blocks = len(blocks)
    ncols = 2
    nrows = int(math.ceil(n_blocks / float(ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.8 * nrows), sharey=True)
    axes_arr = np.atleast_1d(axes).reshape(nrows, ncols)

    block_rows = []
    for block_idx, block in enumerate(blocks):
        row_idx = block_idx // ncols
        col_idx = block_idx % ncols
        ax = axes_arr[row_idx, col_idx]

        left = max(0, block["start"] - context)
        right = min(len(joined) - 1, block["end"] + context)
        view = joined.iloc[left : right + 1].copy()
        x = np.arange(len(view), dtype=np.int64)

        ax.plot(x, view["lag_gt"], label="true lag", color="#111827", linewidth=2.2, drawstyle="steps-post")
        ax.plot(x, view[aligned_col], label="aligned thresholded lag", color="#2563eb", linewidth=1.8, drawstyle="steps-post")
        ax.plot(x, view[noalign_col], label="noalign thresholded lag", color="#ef4444", linewidth=1.8, drawstyle="steps-post")

        lag_start = block["start"] - left
        lag_end = block["end"] - left
        ax.axvspan(lag_start, lag_end, color="#fde68a", alpha=0.35)
        ax.set_ylim(-0.3, 7.2)
        ax.set_title(
            "Block %d: %s to %s"
            % (
                block_idx + 1,
                joined.iloc[block["start"]]["TimeStamp"].strftime("%Y-%m-%d %H:%M"),
                joined.iloc[block["end"]]["TimeStamp"].strftime("%Y-%m-%d %H:%M"),
            ),
            fontsize=10,
        )
        ax.set_xlabel("sample index in local window")
        ax.set_ylabel("lag")

        block_view = joined.iloc[block["start"] : block["end"] + 1]
        block_rows.append(
            {
                "block_id": block_idx + 1,
                "start_time": block_view["TimeStamp"].iloc[0].strftime("%Y-%m-%d %H:%M"),
                "end_time": block_view["TimeStamp"].iloc[-1].strftime("%Y-%m-%d %H:%M"),
                "n_samples": int(len(block_view)),
                "true_lag_mode": int(block_view["lag_gt"].mode().iloc[0]),
                "aligned_mean_thresholded_lag": float(block_view[aligned_col].mean()),
                "noalign_mean_thresholded_lag": float(block_view[noalign_col].mean()),
                "aligned_thresholded_lag_mae": float((block_view[aligned_col] - block_view["lag_gt"]).abs().mean()),
                "noalign_thresholded_lag_mae": float((block_view[noalign_col] - block_view["lag_gt"]).abs().mean()),
            }
        )

    for idx in range(n_blocks, nrows * ncols):
        axes_arr[idx // ncols, idx % ncols].axis("off")

    handles, labels = axes_arr[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.99))
    fig.suptitle(
        "Lag Block Panels: Thresholded Lag, P(lag>0) >= %.2f" % tau,
        y=1.02,
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return pd.DataFrame(block_rows)


def _threshold_metrics(frame: pd.DataFrame, pred_col: str) -> Dict[str, float]:
    true_lag = frame["lag_gt"].to_numpy(dtype=np.int64)
    pred_lag = frame[pred_col].to_numpy(dtype=np.int64)
    true_nonzero = true_lag > 0
    pred_nonzero = pred_lag > 0
    tp = int(np.logical_and(true_nonzero, pred_nonzero).sum())
    fp = int(np.logical_and(~true_nonzero, pred_nonzero).sum())
    fn = int(np.logical_and(true_nonzero, ~pred_nonzero).sum())
    precision = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "pred_nonzero_rate": float(pred_nonzero.mean()),
        "detect_precision": float(precision),
        "detect_recall": float(recall),
        "detect_f1": float(f1),
        "overall_exact_match": float((pred_lag == true_lag).mean()),
        "lagged_only_exact_match": float((pred_lag[true_nonzero] == true_lag[true_nonzero]).mean()) if true_nonzero.any() else 0.0,
        "overall_mae": float(np.abs(pred_lag - true_lag).mean()),
        "lagged_only_mae": float(np.abs(pred_lag[true_nonzero] - true_lag[true_nonzero]).mean()) if true_nonzero.any() else 0.0,
    }


def _write_threshold_summary_markdown(
    joined: pd.DataFrame,
    thresholds: List[float],
    output_path: Path,
) -> None:
    metric_rows = []
    for tau in thresholds:
        tau_tag = _threshold_tag(tau)
        for label in ("aligned", "noalign"):
            metrics = _threshold_metrics(joined, "%s_pred_thresholded_lag_tau%s" % (label, tau_tag))
            metric_rows.append(
                "| %.2f | %s | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |"
                % (
                    tau,
                    label,
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

    lines = [
        "# Thresholded Lag Summary",
        "",
        "展示规则统一定义为：",
        "",
        "```text",
        "if P(lag>0) < tau: predicted lag = 0",
        "else: predicted lag = argmax over non-zero lag bins",
        "```",
        "",
        "其中 `P(lag>0) = 1 - pi_lag0`。",
        "",
        "## Metrics",
        "",
        "| tau | model | pred_nonzero_rate | precision | recall | F1 | overall exact | lagged-only exact | overall MAE | lagged-only MAE |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    lines.extend(metric_rows)
    lines.extend(["", "## Figure Files", ""])
    for tau in thresholds:
        tau_tag = _threshold_tag(tau)
        lines.extend(
            [
                "### tau = %.2f" % tau,
                "",
                "- Thresholded block plot: `lag_block_panels_threshold_pgt0_%s.png`" % tau_tag,
                "- Thresholded block summary: `lag_block_summary_threshold_pgt0_%s.csv`" % tau_tag,
                "",
            ]
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_summary_markdown(
    summary: Dict,
    per_lag: pd.DataFrame,
    block_summary: pd.DataFrame,
    output_path: Path,
    visuals_dir: Path,
) -> None:
    aligned = summary["lag_recovery"]["aligned"]
    noalign = summary["lag_recovery"]["noalign"]

    lines = [
        "# Raw-Gap Alignment 结果汇总",
        "",
        "## 结论",
        "",
        "1. 预测 `yield_flow` 时，alignment 打开后 `MAE` 从 `%.3f` 降到 `%.3f`，有小幅收益。"
        % (summary["forecast_metrics"]["noalign"]["MAE"], summary["forecast_metrics"]["aligned"]["MAE"]),
        "2. 只看真正加了 lag 的样本，alignment 的 `expected lag MAE` 从 `%.3f` 降到 `%.3f`，说明它确实在恢复 lag 大小。"
        % (noalign["lagged_only"]["expected_lag_mae"], aligned["lagged_only"]["expected_lag_mae"]),
        "3. 但 alignment 当前几乎把 `stage1_to_stage2` 的预测 lag 压在 `5 step` 左右，所以 `argmax lag accuracy` 仍然是 `0`。"
        " 这让它在 `lag=4/6` 上明显占优，但在 `lag=0/2` 上会系统性偏大。",
        "",
        "## 图表",
        "",
        "- Forecast 指标：[forecast_metrics.png](./forecast_metrics.png)",
        "- Lag 子集对比：[lag_subset_metrics.png](./lag_subset_metrics.png)",
        "- 各真实 lag 的误差：[lag_mae_by_true_lag.png](./lag_mae_by_true_lag.png)",
        "- 各 lag block 时间轴：[lag_block_panels.png](./lag_block_panels.png)",
        "",
        "## 关键数字",
        "",
        "| 指标 | aligned | noalign |",
        "| --- | --- | --- |",
        "| Forecast MAE | %.3f | %.3f |"
        % (summary["forecast_metrics"]["aligned"]["MAE"], summary["forecast_metrics"]["noalign"]["MAE"]),
        "| Forecast RMSE | %.3f | %.3f |"
        % (summary["forecast_metrics"]["aligned"]["RMSE"], summary["forecast_metrics"]["noalign"]["RMSE"]),
        "| Forecast R2 | %.3f | %.3f |"
        % (summary["forecast_metrics"]["aligned"]["R2"], summary["forecast_metrics"]["noalign"]["R2"]),
        "| Lagged-only expected lag MAE | %.3f | %.3f |"
        % (aligned["lagged_only"]["expected_lag_mae"], noalign["lagged_only"]["expected_lag_mae"]),
        "| Lagged-only argmax acc | %.3f | %.3f |"
        % (aligned["lagged_only"]["argmax_lag_accuracy"], noalign["lagged_only"]["argmax_lag_accuracy"]),
        "| Mean predicted lag on lagged samples | %.3f | %.3f |"
        % (aligned["lagged_only"]["mean_pred_expected_lag"], noalign["lagged_only"]["mean_pred_expected_lag"]),
        "",
        "## 分真实 lag",
        "",
        "| true lag | n | aligned MAE | noalign MAE | aligned pred mean | noalign pred mean |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    for row in per_lag.itertuples(index=False):
        lines.append(
            "| %d | %d | %.3f | %.3f | %.3f | %.3f |"
            % (
                int(row.lag_gt),
                int(row.n),
                float(row.aligned_expected_lag_mae),
                float(row.noalign_expected_lag_mae),
                float(row.aligned_mean_pred_expected_lag),
                float(row.noalign_mean_pred_expected_lag),
            )
        )

    lines.extend(
        [
            "",
            "## Lag Block 摘要",
            "",
            "| block | start | end | true lag | aligned pred mean | noalign pred mean | aligned MAE | noalign MAE |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )

    for row in block_summary.itertuples(index=False):
        lines.append(
            "| %d | %s | %s | %d | %.3f | %.3f | %.3f | %.3f |"
            % (
                int(row.block_id),
                row.start_time,
                row.end_time,
                int(row.true_lag_mode),
                float(row.aligned_mean_pred_expected_lag),
                float(row.noalign_mean_pred_expected_lag),
                float(row.aligned_expected_lag_mae),
                float(row.noalign_expected_lag_mae),
            )
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    summary_path = _absolute_path(args.summary)
    joined_path = _absolute_path(args.joined)
    per_lag_path = _absolute_path(args.per_lag)
    output_dir = _absolute_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = _load_json(summary_path)
    joined = pd.read_csv(joined_path)
    per_lag = pd.read_csv(per_lag_path)

    forecast_plot = output_dir / "forecast_metrics.png"
    subset_plot = output_dir / "lag_subset_metrics.png"
    per_lag_plot = output_dir / "lag_mae_by_true_lag.png"
    block_plot = output_dir / "lag_block_panels.png"
    block_summary_path = output_dir / "lag_block_summary.csv"
    summary_md_path = output_dir / "alignment_summary_zh.md"
    threshold_summary_md_path = output_dir / "thresholded_lag_summary.md"

    _write_forecast_plot(summary, forecast_plot)
    _write_subset_plot(summary, subset_plot)
    _write_per_lag_plot(per_lag, per_lag_plot)
    block_summary = _write_block_plot(joined, block_plot)
    block_summary.to_csv(block_summary_path, index=False)
    _write_summary_markdown(summary, per_lag, block_summary, summary_md_path, output_dir)

    if args.aligned_estimates is not None and args.noalign_estimates is not None:
        thresholds = _parse_thresholds(args.thresholds)
        thresholded_joined = _attach_threshold_predictions(
            joined=joined,
            aligned_estimates=_absolute_path(args.aligned_estimates),
            noalign_estimates=_absolute_path(args.noalign_estimates),
            edge=args.edge,
            thresholds=thresholds,
        )
        thresholded_joined_path = output_dir / "alignment_test_joined_thresholded.csv"
        thresholded_joined.to_csv(thresholded_joined_path, index=False)
        print("Wrote: %s" % thresholded_joined_path)

        for tau in thresholds:
            tau_tag = _threshold_tag(tau)
            threshold_block_plot = output_dir / ("lag_block_panels_threshold_pgt0_%s.png" % tau_tag)
            threshold_block_summary = output_dir / ("lag_block_summary_threshold_pgt0_%s.csv" % tau_tag)
            block_threshold_summary = _write_threshold_block_plot(thresholded_joined, tau=tau, output_path=threshold_block_plot)
            block_threshold_summary.to_csv(threshold_block_summary, index=False)
            print("Wrote: %s" % threshold_block_plot)
            print("Wrote: %s" % threshold_block_summary)

        _write_threshold_summary_markdown(thresholded_joined, thresholds, threshold_summary_md_path)
        print("Wrote: %s" % threshold_summary_md_path)

    print("Wrote: %s" % forecast_plot)
    print("Wrote: %s" % subset_plot)
    print("Wrote: %s" % per_lag_plot)
    print("Wrote: %s" % block_plot)
    print("Wrote: %s" % block_summary_path)
    print("Wrote: %s" % summary_md_path)


if __name__ == "__main__":
    main()
