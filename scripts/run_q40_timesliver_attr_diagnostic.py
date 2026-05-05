#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


FEATURE_COLUMNS = [
    "d_raw",
    "expected_lag",
    "p_nonzero",
    "entropy",
    "candidate_score",
    "localization_score",
    "q40_d_hat",
    "q40_selected",
]


@dataclass(frozen=True)
class SampleMeta:
    split: str
    segment_uid: str
    segment_id: int
    q40_segment_index: int
    segment_label: int
    segment_is_weak: int
    start_t: float
    end_t: float
    window_start_t: float
    window_end_t: float


class TimeSliverLite(nn.Module):
    def __init__(self, input_dim: int, n_bins: int = 4, hidden_dim: int = 16, kernel_size: int = 3) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.n_bins = int(n_bins)
        self.hidden_dim = int(hidden_dim)
        self.conv = nn.Conv1d(self.input_dim, self.hidden_dim, kernel_size=int(kernel_size), padding=int(kernel_size) // 2)
        self.linear = nn.Linear(self.input_dim * self.n_bins * self.hidden_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        valid_mask: torch.Tensor,
        segment_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.conv(x.transpose(1, 2)).transpose(1, 2)
        q = q * valid_mask.unsqueeze(-1)
        p = torch.einsum("bts,bth->bsh", z, q)
        logits = self.linear(p.reshape(p.shape[0], -1)).squeeze(-1)
        weight = self.linear.weight.reshape(self.input_dim * self.n_bins, self.hidden_dim)
        contrib = torch.einsum("bts,bth,sh->bt", z, q, weight)
        contrib = contrib * valid_mask
        abs_contrib = contrib.abs()
        attr = (abs_contrib * segment_mask).sum(dim=1) / (abs_contrib.sum(dim=1) + 1e-6)
        return logits, attr


def _path(text: str | Path) -> Path:
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _read_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if len(frame.columns) > 0:
        first = frame.columns[0]
        frame = frame.loc[frame[first].astype(str) != first].reset_index(drop=True)
    for col in frame.columns:
        if col not in {"split", "source_split", "timestamp", "TimeStamp", "original_split", "q40_prediction_source", "segment_uid"}:
            try:
                frame[col] = pd.to_numeric(frame[col])
            except (TypeError, ValueError):
                pass
    return frame


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


def _extract_window_samples(
    timeseries: pd.DataFrame,
    segment_frame: pd.DataFrame,
    split: str,
    group_col: str,
    time_col: str,
    radius: int,
) -> tuple[List[np.ndarray], List[np.ndarray], List[SampleMeta]]:
    rows_x: List[np.ndarray] = []
    rows_seg_mask: List[np.ndarray] = []
    metas: List[SampleMeta] = []
    ordered = timeseries.sort_values([group_col, time_col]).reset_index(drop=True)
    for seg in segment_frame.itertuples(index=False):
        group_value = getattr(seg, group_col)
        start_t = float(seg.start_t)
        end_t = float(seg.end_t)
        local = ordered.loc[ordered[group_col] == group_value].copy()
        local = local.sort_values(time_col).reset_index(drop=True)
        window = local.loc[(local[time_col] >= start_t - float(radius)) & (local[time_col] <= end_t + float(radius))].copy()
        if window.empty:
            continue
        x = window[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
        seg_mask = ((window[time_col].to_numpy(dtype=np.float64) >= start_t) & (window[time_col].to_numpy(dtype=np.float64) <= end_t)).astype(np.float32)
        rows_x.append(x)
        rows_seg_mask.append(seg_mask)
        metas.append(
            SampleMeta(
                split=split,
                segment_uid=str(seg.segment_uid),
                segment_id=int(getattr(seg, group_col)),
                q40_segment_index=int(seg.q40_segment_index),
                segment_label=int(seg.segment_label),
                segment_is_weak=int(seg.segment_is_weak),
                start_t=start_t,
                end_t=end_t,
                window_start_t=float(window[time_col].iloc[0]),
                window_end_t=float(window[time_col].iloc[-1]),
            )
        )
    return rows_x, rows_seg_mask, metas


def _fit_normalizer_and_bins(samples: Sequence[np.ndarray], n_bins: int) -> tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    all_values = np.concatenate(samples, axis=0)
    mean = np.zeros(all_values.shape[1], dtype=np.float64)
    std = np.ones(all_values.shape[1], dtype=np.float64)
    for col in range(all_values.shape[1]):
        values = all_values[:, col]
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            mean[col] = 0.0
            std[col] = 1.0
            continue
        mean[col] = float(finite.mean())
        std[col] = float(finite.std()) if float(finite.std()) > 1e-6 else 1.0
    std = np.where(std > 1e-6, std, 1.0)
    normalized = (all_values - mean[None, :]) / std[None, :]
    edges: List[np.ndarray] = []
    for col in range(normalized.shape[1]):
        values = normalized[:, col]
        values = values[np.isfinite(values)]
        if values.size == 0:
            edges.append(np.asarray([], dtype=np.float64))
            continue
        qs = np.linspace(0.0, 1.0, int(n_bins) + 1)[1:-1]
        edge = np.unique(np.quantile(values, qs))
        edges.append(np.asarray(edge, dtype=np.float64))
    return mean.astype(np.float64), std.astype(np.float64), edges


def _encode_samples(
    samples: Sequence[np.ndarray],
    segment_masks: Sequence[np.ndarray],
    labels: Sequence[int],
    mean: np.ndarray,
    std: np.ndarray,
    edges: Sequence[np.ndarray],
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    max_len = max(len(sample) for sample in samples)
    n_feat = len(FEATURE_COLUMNS)
    x_pad = np.zeros((len(samples), max_len, n_feat), dtype=np.float32)
    z_pad = np.zeros((len(samples), max_len, n_feat * int(n_bins)), dtype=np.float32)
    valid_mask = np.zeros((len(samples), max_len), dtype=np.float32)
    seg_mask = np.zeros((len(samples), max_len), dtype=np.float32)
    y = np.asarray(labels, dtype=np.float32)
    for i, sample in enumerate(samples):
        norm = (sample - mean[None, :]) / std[None, :]
        t = norm.shape[0]
        x_pad[i, :t, :] = np.where(np.isfinite(norm), norm, 0.0).astype(np.float32)
        valid_mask[i, :t] = 1.0
        seg_mask[i, :t] = segment_masks[i].astype(np.float32)
        for feat_idx in range(n_feat):
            vals = norm[:, feat_idx]
            bin_idx = np.digitize(vals, edges[feat_idx], right=False)
            for pos, b in enumerate(bin_idx.tolist()):
                z_pad[i, pos, feat_idx * int(n_bins) + int(b)] = 1.0
    return x_pad, z_pad, valid_mask, seg_mask, y


def _make_loader(
    x: np.ndarray,
    z: np.ndarray,
    valid_mask: np.ndarray,
    seg_mask: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(x),
        torch.from_numpy(z),
        torch.from_numpy(valid_mask),
        torch.from_numpy(seg_mask),
        torch.from_numpy(y),
    )
    return DataLoader(ds, batch_size=int(batch_size), shuffle=bool(shuffle))


def _epoch_eval(model: TimeSliverLite, loader: DataLoader, device: str) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_rows = 0
    logits_list: List[np.ndarray] = []
    prob_list: List[np.ndarray] = []
    attr_list: List[np.ndarray] = []
    with torch.no_grad():
        for xb, zb, vb, sb, yb in loader:
            xb = xb.to(device)
            zb = zb.to(device)
            vb = vb.to(device)
            sb = sb.to(device)
            yb = yb.to(device)
            logits, attr = model(xb, zb, vb, sb)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            total_loss += float(loss.item()) * int(yb.numel())
            total_rows += int(yb.numel())
            logits_list.append(logits.cpu().numpy())
            prob_list.append(torch.sigmoid(logits).cpu().numpy())
            attr_list.append(attr.cpu().numpy())
    mean_loss = total_loss / max(total_rows, 1)
    probs = np.concatenate(prob_list).astype(np.float64) if prob_list else np.empty(0, dtype=np.float64)
    attrs = np.concatenate(attr_list).astype(np.float64) if attr_list else np.empty(0, dtype=np.float64)
    logits = np.concatenate(logits_list).astype(np.float64) if logits_list else np.empty(0, dtype=np.float64)
    return mean_loss, probs, attrs, logits


def _attr_distribution_table(samples_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for split_name, split_df in samples_df.groupby("split", sort=False):
        weak_df = split_df.loc[split_df["segment_is_weak"] == 1].copy()
        for group_name, mask in [
            ("positive_weak_segment", weak_df["segment_label"] == 1),
            ("false_positive_weak_segment", weak_df["segment_label"] == 0),
        ]:
            values = weak_df.loc[mask, "attr_score"].to_numpy(dtype=np.float64)
            rows.append(
                {
                    "split": split_name,
                    "group": group_name,
                    "count": int(values.size),
                    "attr_p10": float(np.nanpercentile(values, 10.0)) if values.size else float("nan"),
                    "attr_p50": float(np.nanpercentile(values, 50.0)) if values.size else float("nan"),
                    "attr_p90": float(np.nanpercentile(values, 90.0)) if values.size else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def _run_one(name: str, timeseries_root: Path, strong_root: Path, out_dir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    run_out = out_dir / name
    run_out.mkdir(parents=True, exist_ok=True)
    split_frames = {}
    strong_frames = {}
    for split in ["fit", "val", "eval"]:
        split_frames[split] = _read_csv(timeseries_root / name / f"segment_verifier_{split}_timeseries.csv")
        strong_frames[split] = _read_csv(strong_root / name / f"segment_{split}_scored_strong.csv")

    sample_x: Dict[str, List[np.ndarray]] = {}
    sample_segmask: Dict[str, List[np.ndarray]] = {}
    sample_meta: Dict[str, List[SampleMeta]] = {}
    for split in ["fit", "val", "eval"]:
        x, segm, meta = _extract_window_samples(
            split_frames[split],
            strong_frames[split],
            split=split,
            group_col=str(args.group_col),
            time_col=str(args.time_col),
            radius=int(args.radius),
        )
        sample_x[split] = x
        sample_segmask[split] = segm
        sample_meta[split] = meta
    if not sample_x["fit"] or not sample_x["val"] or not sample_x["eval"]:
        raise ValueError("r47a requires non-empty fit/val/eval segment windows")

    fit_mean, fit_std, bin_edges = _fit_normalizer_and_bins(sample_x["fit"], n_bins=int(args.n_bins))
    encoded = {}
    for split in ["fit", "val", "eval"]:
        labels = [meta.segment_label for meta in sample_meta[split]]
        encoded[split] = _encode_samples(
            sample_x[split],
            sample_segmask[split],
            labels=labels,
            mean=fit_mean,
            std=fit_std,
            edges=bin_edges,
            n_bins=int(args.n_bins),
        )

    loaders = {
        "fit_train": _make_loader(*encoded["fit"], batch_size=int(args.batch_size), shuffle=True),
        "fit_eval": _make_loader(*encoded["fit"], batch_size=int(args.batch_size), shuffle=False),
        "val": _make_loader(*encoded["val"], batch_size=int(args.batch_size), shuffle=False),
        "eval": _make_loader(*encoded["eval"], batch_size=int(args.batch_size), shuffle=False),
    }

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    model = TimeSliverLite(input_dim=len(FEATURE_COLUMNS), n_bins=int(args.n_bins), hidden_dim=int(args.hidden_dim)).to(str(args.device))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    history_rows: List[Dict[str, Any]] = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        for xb, zb, vb, sb, yb in loaders["fit_train"]:
            xb = xb.to(str(args.device))
            zb = zb.to(str(args.device))
            vb = vb.to(str(args.device))
            sb = sb.to(str(args.device))
            yb = yb.to(str(args.device))
            optimizer.zero_grad()
            logits, _ = model(xb, zb, vb, sb)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            loss.backward()
            optimizer.step()
        fit_loss, _, _, _ = _epoch_eval(model, loaders["fit_eval"], device=str(args.device))
        val_loss, _, _, _ = _epoch_eval(model, loaders["val"], device=str(args.device))
        history_rows.append({"epoch": int(epoch), "fit_loss": fit_loss, "val_loss": val_loss})
    history = pd.DataFrame(history_rows)
    history.to_csv(run_out / "timesliver_lite_history.csv", index=False)

    sample_rows: List[Dict[str, Any]] = []
    split_metrics: Dict[str, Dict[str, Any]] = {}
    for split in ["fit", "val", "eval"]:
        loss, probs, attrs, logits = _epoch_eval(model, loaders[split if split != "fit" else "fit_eval"], device=str(args.device))
        labels = np.asarray([meta.segment_label for meta in sample_meta[split]], dtype=np.float64)
        split_metrics[split] = {
            "loss": float(loss),
            "n_segments": int(len(labels)),
            "n_positive": int(labels.sum()),
            "n_weak": int(sum(meta.segment_is_weak for meta in sample_meta[split])),
        }
        for meta, prob, attr, logit in zip(sample_meta[split], probs.tolist(), attrs.tolist(), logits.tolist()):
            sample_rows.append(
                {
                    "split": meta.split,
                    "segment_uid": meta.segment_uid,
                    "segment_id": meta.segment_id,
                    "q40_segment_index": meta.q40_segment_index,
                    "segment_label": meta.segment_label,
                    "segment_is_weak": meta.segment_is_weak,
                    "start_t": meta.start_t,
                    "end_t": meta.end_t,
                    "window_start_t": meta.window_start_t,
                    "window_end_t": meta.window_end_t,
                    "aux_logit": float(logit),
                    "aux_prob": float(prob),
                    "attr_score": float(attr),
                }
            )
    sample_df = pd.DataFrame(sample_rows)
    sample_df.to_csv(run_out / "timesliver_segment_attr_scores.csv", index=False)
    diagnostic = _attr_distribution_table(sample_df)
    diagnostic.to_csv(run_out / "timesliver_attr_diagnostic.csv", index=False)

    report = {
        "run": name,
        "feature_columns": FEATURE_COLUMNS,
        "radius": int(args.radius),
        "n_bins": int(args.n_bins),
        "hidden_dim": int(args.hidden_dim),
        "metrics": split_metrics,
        "outputs": {
            "history": (run_out / "timesliver_lite_history.csv").as_posix(),
            "segment_attr_scores": (run_out / "timesliver_segment_attr_scores.csv").as_posix(),
            "attr_diagnostic": (run_out / "timesliver_attr_diagnostic.csv").as_posix(),
        },
    }
    _write_json(run_out / "timesliver_attr_report.json", report)
    eval_diag = diagnostic.loc[diagnostic["split"] == "eval"].copy()
    pos_eval = eval_diag.loc[eval_diag["group"] == "positive_weak_segment"]
    fp_eval = eval_diag.loc[eval_diag["group"] == "false_positive_weak_segment"]
    return {
        "run": name,
        "fit_n_segments": split_metrics["fit"]["n_segments"],
        "fit_n_weak": split_metrics["fit"]["n_weak"],
        "eval_n_weak": split_metrics["eval"]["n_weak"],
        "eval_positive_weak_attr_p50": float(pos_eval["attr_p50"].iloc[0]) if not pos_eval.empty else float("nan"),
        "eval_false_positive_weak_attr_p50": float(fp_eval["attr_p50"].iloc[0]) if not fp_eval.empty else float("nan"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="r47a TimeSliver-lite attribution diagnostic on weak q40 segments.")
    parser.add_argument("--timeseries-root", default="outputs/r45c_q40_segment_proposal_verifier_smoke")
    parser.add_argument("--strong-root", default="outputs/r46b_q40_segment_strongkeep_veto_smoke")
    parser.add_argument("--runs", default="old,seed134_e2")
    parser.add_argument("--output-dir", default="outputs/r47a_timesliver_attr_diagnostic")
    parser.add_argument("--group-col", default="segment_id")
    parser.add_argument("--time-col", default="t")
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--n-bins", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timeseries_root = _path(args.timeseries_root)
    strong_root = _path(args.strong_root)
    out_dir = _path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for name in [part.strip() for part in str(args.runs).split(",") if part.strip()]:
        rows.append(_run_one(name, timeseries_root, strong_root, out_dir, args))
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "timesliver_attr_summary.csv", index=False)
    print(summary.to_csv(index=False, float_format="%.6f"))
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
