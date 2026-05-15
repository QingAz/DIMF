#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import sysconfig
from pathlib import Path
from typing import Any, Dict, Optional

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
_DLL_DIR_HANDLES = []
if sys.platform == "win32":
    exe_path = Path(sys.executable).resolve()
    prefixes = []
    for raw in (os.environ.get("CONDA_PREFIX"), sys.prefix, str(exe_path.parent)):
        if raw:
            path = Path(raw).resolve()
            if path.name.lower() in {"scripts", "bin"}:
                path = path.parent
            if path not in prefixes:
                prefixes.append(path)
    site_packages = sysconfig.get_paths().get("purelib")
    if site_packages:
        torch_lib = Path(site_packages) / "torch" / "lib"
    else:
        torch_lib = None
    primary_dirs = []
    path_dirs = []
    if torch_lib is not None:
        primary_dirs.append(torch_lib)
        path_dirs.append(torch_lib)
    for prefix in prefixes:
        libbin = prefix / "Library" / "bin"
        primary_dirs.append(libbin)
        path_dirs.append(libbin)
    seen_dirs = set()
    resolved_dirs = []
    for path in primary_dirs:
        path = Path(path)
        key = str(path).lower()
        if key in seen_dirs or not path.is_dir():
            continue
        seen_dirs.add(key)
        try:
            _DLL_DIR_HANDLES.append(os.add_dll_directory(str(path)))
        except Exception:
            pass
    seen_path_dirs = set()
    for path in path_dirs:
        path = Path(path)
        key = str(path).lower()
        if key in seen_path_dirs or not path.is_dir():
            continue
        seen_path_dirs.add(key)
        resolved_dirs.append(str(path))
    if resolved_dirs:
        os.environ["PATH"] = ";".join(resolved_dirs) + ";" + os.environ.get("PATH", "")

import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_lag_grounded import (
    STDALagIdentifier,
    _decode_viterbi_for_eval,
    _edge_cfg,
    _history_steps,
    _lag_eval_metrics,
    _lag_prediction_frame,
    _load_existing_train_test_prepared,
    _settings_suffix,
    collect_lag_outputs,
    load_config,
    postprocess_lag_eval_for_eval,
)
from src.postprocess.viterbi_lag_decoder import viterbi_decode_lag


VITERBI_VARIANTS = {
    "A_off": None,
    "B_default": {
        "viterbi_smooth_lambda": 0.8,
        "viterbi_switch_penalty": 1.5,
        "viterbi_pos_to_zero_penalty": 2.0,
    },
    "C_weak": {
        "viterbi_smooth_lambda": 0.3,
        "viterbi_switch_penalty": 0.5,
        "viterbi_pos_to_zero_penalty": 0.8,
    },
    "D_weaker": {
        "viterbi_smooth_lambda": 0.2,
        "viterbi_switch_penalty": 0.3,
        "viterbi_pos_to_zero_penalty": 0.5,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug S1-S2 lag recovery raw/Viterbi behavior.")
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/lag_regions_train_test.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


def _make_identifier(cfg: Dict[str, Any], prepared, edge_cfg: Dict[str, Any], device: torch.device) -> STDALagIdentifier:
    lag_cfg = cfg.get("lag_identifier", {})
    return STDALagIdentifier(
        d_source=prepared.group_dims[edge_cfg["source_stage"]],
        d_target=prepared.group_dims[edge_cfg["target_stage"]],
        max_lag=int(lag_cfg.get("max_lag", cfg.get("lag_injection", {}).get("max_lag", cfg["data"].get("L_max", 12)))),
        hidden_dim=int(lag_cfg.get("hidden_dim", 64)),
        num_layers=int(lag_cfg.get("num_layers", 2)),
        temperature=float(lag_cfg.get("lag_temperature", lag_cfg.get("temperature", 0.7))),
        use_temporal_decay=bool(lag_cfg.get("use_temporal_decay", True)),
        use_feature_attention=bool(lag_cfg.get("use_feature_attention", True)),
        dropout=float(lag_cfg.get("dropout", 0.0)),
        use_sequence_smoother=bool(lag_cfg.get("use_sequence_smoother", False)),
        sequence_smoother_hidden_dim=lag_cfg.get("sequence_smoother_hidden_dim"),
        sequence_smoother_layers=int(lag_cfg.get("sequence_smoother_layers", 1)),
        sequence_smoother_dropout=float(lag_cfg.get("sequence_smoother_dropout", 0.0)),
        sequence_smoother_residual_scale=float(lag_cfg.get("sequence_smoother_residual_scale", 0.5)),
        use_candidate_window_encoder=bool(lag_cfg.get("use_candidate_window_encoder", False)),
        lag_window_radius=int(lag_cfg.get("lag_window_radius", 2)),
        lag_window_hidden_dim=lag_cfg.get("lag_window_hidden_dim"),
        lag_window_mode=str(lag_cfg.get("lag_window_mode", "causal")),
        keep_old_point_identifier=bool(lag_cfg.get("keep_old_point_identifier", True)),
    ).to(device)


def _decode_variant(lag_eval: Dict[str, Any], params: Optional[Dict[str, float]]) -> Optional[np.ndarray]:
    if params is None:
        return None
    order = _temporal_order(lag_eval)
    segment = np.asarray(lag_eval["segment_id"])[order] if "segment_id" in lag_eval else None
    sorted_path = viterbi_decode_lag(
        np.asarray(lag_eval["pred_pi"])[order],
        segment_id=segment,
        smooth_lambda=float(params["viterbi_smooth_lambda"]),
        switch_penalty=float(params["viterbi_switch_penalty"]),
        pos_to_zero_penalty=float(params["viterbi_pos_to_zero_penalty"]),
    )
    path = np.empty_like(sorted_path)
    path[order] = sorted_path
    return path


def _temporal_order(lag_eval: Dict[str, Any]) -> np.ndarray:
    n = int(np.asarray(lag_eval["pred_pi"]).shape[0])
    time_index = np.asarray(lag_eval.get("time_index", lag_eval.get("sample_index", np.arange(n))))
    if "segment_id" in lag_eval:
        return np.lexsort((time_index, np.asarray(lag_eval["segment_id"])))
    return np.argsort(time_index, kind="stable")


def _pi_debug_metrics(lag_eval: Dict[str, Any], identifier: STDALagIdentifier) -> Dict[str, Any]:
    pi = np.asarray(lag_eval["pred_pi"], dtype=np.float64)
    lag_flag = np.asarray(lag_eval["lag_flag"]).astype(bool)
    no_lag = ~lag_flag
    occ = np.asarray(lag_eval.get("occurrence_score", 1.0 - pi[:, 0]), dtype=np.float64)
    out: Dict[str, Any] = {
        "mean_pi0_on_no_lag": _safe_mean(pi[no_lag, 0]),
        "mean_pi0_on_positive": _safe_mean(pi[lag_flag, 0]),
        "median_pi0_on_no_lag": _safe_median(pi[no_lag, 0]),
        "median_pi0_on_positive": _safe_median(pi[lag_flag, 0]),
        "mean_occ_prob_on_no_lag": _safe_mean(occ[no_lag]),
        "mean_occ_prob_on_positive": _safe_mean(occ[lag_flag]),
        "lag_bias": json.dumps([float(x) for x in identifier.lag_bias.detach().cpu().numpy()]),
    }
    for lag in range(pi.shape[1]):
        out[f"mean_pi{lag}_on_all"] = _safe_mean(pi[:, lag])
        out[f"mean_pi{lag}_on_no_lag"] = _safe_mean(pi[no_lag, lag])
        out[f"mean_pi{lag}_on_positive"] = _safe_mean(pi[lag_flag, lag])
    if lag_eval.get("feature_importance_batches"):
        imp = torch.cat(lag_eval["feature_importance_batches"], dim=0).numpy()
        mean_imp = imp.mean(axis=0)
        top = np.argsort(-mean_imp)[: min(5, mean_imp.shape[0])]
        out["feature_importance_top5"] = json.dumps(
            [{"feature": int(idx), "mean_importance": float(mean_imp[idx])} for idx in top]
        )
        out["feature_importance_top1_mass"] = float(mean_imp[top[0]]) if top.size else float("nan")
    return out


def _safe_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.mean(values)) if values.size else float("nan")


def _safe_median(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.median(values)) if values.size else float("nan")


def _save_debug_plot(frame: pd.DataFrame, output_path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sort_cols = [col for col in ["segment_id", "time_index", "sample_index"] if col in frame.columns]
    plot_df = frame.sort_values(sort_cols).reset_index(drop=True) if sort_cols else frame.reset_index(drop=True)
    x = np.arange(len(plot_df))

    fig, axes = plt.subplots(2, 1, figsize=(16, 6.5), sharex=True, height_ratios=[3, 1])
    axes[0].plot(x, plot_df["lag_gt"], label="gt", linewidth=1.8)
    axes[0].plot(x, plot_df["raw_expected"], label="raw_expected", linewidth=1.2)
    axes[0].plot(x, plot_df["raw_argmax"], label="raw_argmax", linewidth=0.8, alpha=0.55)
    if "viterbi_path_default" in plot_df:
        axes[0].plot(x, plot_df["viterbi_path_default"], label="viterbi_default", linewidth=1.0, alpha=0.8)
    if "viterbi_path_weak" in plot_df:
        axes[0].plot(x, plot_df["viterbi_path_weak"], label="viterbi_weak", linewidth=1.0, alpha=0.8)
    axes[0].set_ylabel("lag")
    axes[0].set_title(title)
    axes[0].legend(frameon=False, ncol=3)

    if "pi_0" in plot_df:
        axes[1].plot(x, plot_df["pi_0"], label="pi_0", linewidth=1.0)
    if "occurrence_score" in plot_df:
        axes[1].plot(x, plot_df["occurrence_score"], label="occ_prob", linewidth=0.9, alpha=0.8)
    axes[1].set_xlabel("sorted sample")
    axes[1].set_ylabel("prob")
    axes[1].legend(frameon=False, ncol=2)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config.resolve())
    results_dir = Path(cfg.get("logging", {}).get("output_dir", "results/lag_grounded")).resolve()
    output_dir = (args.output_dir or results_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = (args.checkpoint or (results_dir / "best_lag_identifier.pt")).resolve()

    if "train_csv" not in cfg["data"] or "test_csv" not in cfg["data"]:
        raise ValueError("debug_lag_recovery.py currently expects existing train_csv/test_csv config")
    prepared = _load_existing_train_test_prepared(cfg)
    spec_batch = int(args.batch_size or cfg.get("training", {}).get("batch_size", 256))
    from src.data.dataset import MultistageWindowDataset, WindowSpec

    spec = WindowSpec(L=_history_steps(cfg["data"]), H=int(cfg["data"]["H"]))
    ds_te = MultistageWindowDataset(
        prepared.X_groups_test,
        prepared.y_test,
        spec,
        prepared.sample_indices_test,
        prepared.extra_targets_test,
    )
    dl_te = DataLoader(ds_te, batch_size=spec_batch, shuffle=False, drop_last=False)

    device = torch.device(args.device)
    edge_cfg = _edge_cfg(cfg)
    edge_name = str(edge_cfg.get("name", "stage1_to_stage2"))
    identifier = _make_identifier(cfg, prepared, edge_cfg, device)
    try:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location=device)
    identifier.load_state_dict(state)
    identifier.eval()

    lag_eval = collect_lag_outputs(identifier, dl_te, device, edge_cfg, edge_name)
    lag_eval, postprocess_info = postprocess_lag_eval_for_eval(lag_eval, cfg)
    with torch.no_grad():
        first_x, _ = next(iter(dl_te))
        first_x = {key: value.to(device) for key, value in first_x.items()}
        valid = identifier(first_x[edge_cfg["source_stage"]], target_seq=first_x[edge_cfg["target_stage"]])[
            "lag_candidate_valid"
        ].detach().cpu().numpy().astype(bool)

    paths: Dict[str, Optional[np.ndarray]] = {}
    rows = []
    for name, params in VITERBI_VARIANTS.items():
        path = _decode_variant(lag_eval, params)
        paths[name] = path
        row = {
            "variant": name,
            "use_viterbi_decode": params is not None,
            **_lag_eval_metrics(lag_eval, viterbi_path=path),
        }
        if params:
            row.update(params)
        rows.append(row)

    ablation = pd.DataFrame(rows)
    ablation.to_csv(output_dir / "viterbi_ablation_metrics.csv", index=False)

    default_path = paths["B_default"]
    weak_path = paths["C_weak"]
    base_frame = _lag_prediction_frame(lag_eval, viterbi_path=default_path)
    if default_path is not None:
        base_frame["viterbi_path_default"] = default_path
    if weak_path is not None:
        base_frame["viterbi_path_weak"] = weak_path
    if paths["D_weaker"] is not None:
        base_frame["viterbi_path_weaker"] = paths["D_weaker"]
    sort_cols = [col for col in ["segment_id", "time_index", "sample_index"] if col in base_frame.columns]
    base_frame = base_frame.sort_values(sort_cols).reset_index(drop=True) if sort_cols else base_frame
    base_frame.to_csv(output_dir / "lag_eval_predictions_debug.csv", index=False)

    debug_metrics = {
        **_pi_debug_metrics(lag_eval, identifier),
        **_lag_eval_metrics(lag_eval, viterbi_path=default_path),
        "checkpoint": str(checkpoint),
        "seq_len": int(_history_steps(cfg["data"])),
        "use_candidate_window_encoder": bool(cfg.get("lag_identifier", {}).get("use_candidate_window_encoder", False)),
        "lag_window_radius": int(cfg.get("lag_identifier", {}).get("lag_window_radius", 2)),
        "use_gaussian_lag_label": bool(cfg.get("loss", {}).get("use_gaussian_lag_label", True)),
        "valid_lag_mask": json.dumps([bool(x) for x in valid.tolist()]),
        "lag5_valid": bool(valid[5]) if valid.shape[0] > 5 else False,
        "lag_postprocess": json.dumps(postprocess_info),
    }
    pd.DataFrame([debug_metrics]).to_csv(output_dir / "lag_eval_metrics_debug.csv", index=False)

    suffix = _settings_suffix(cfg, viterbi_enabled=True)
    _save_debug_plot(
        base_frame,
        output_dir / "debug_lag_raw_vs_viterbi.png",
        title=f"Raw vs Viterbi lag debug ({suffix})",
    )

    print(json.dumps({"output_dir": str(output_dir), "checkpoint": str(checkpoint)}, indent=2))
    print(ablation.to_string(index=False))
    print(pd.DataFrame([debug_metrics]).to_string(index=False))


if __name__ == "__main__":
    main()
