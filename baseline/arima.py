from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataprocess import load_and_prepare
from src.utils.metrics import mae, rmse, r2

try:
    from statsmodels.tsa.arima.model import ARIMA
except ImportError as e:
    raise ImportError("statsmodels is not installed. Please run: python -m pip install statsmodels") from e


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--H", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def inverse_scale_2d(arr: np.ndarray, scaler_y) -> np.ndarray:
    return scaler_y.inverse_transform(arr.reshape(-1, 1)).reshape(arr.shape)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "R2": r2(y_true, y_pred),
        "n_samples": int(y_true.shape[0]),
        "horizon": int(y_true.shape[1]),
    }


def summarize_metric(values: list[float]) -> Dict[str, float]:
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
    }


def rolling_arima_forecast(
    history_prefix: np.ndarray,
    series: np.ndarray,
    L: int,
    H: int,
    order: Tuple[int, int, int],
    trend: str,
    progress_prefix: str = "",
    progress_every: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    t_min = L - 1
    t_max = len(series) - H - 1
    if t_max < t_min:
        raise ValueError("Not enough samples for given L and H.")

    y_true_rows = []
    y_pred_rows = []

    total_steps = t_max - t_min + 1
    for step_idx, t in enumerate(range(t_min, t_max + 1), start=1):
        y_true_rows.append(series[t + 1 : t + H + 1])

        history = np.concatenate([history_prefix, series[: t + 1]], axis=0).astype(np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ARIMA(history, order=order, trend=trend)
            fit = model.fit()
            forecast = fit.forecast(steps=H)

        y_pred_rows.append(np.asarray(forecast, dtype=np.float32))
        if progress_every > 0 and (step_idx % progress_every == 0 or step_idx == total_steps):
            pct = 100.0 * step_idx / total_steps
            print(f"{progress_prefix} step {step_idx}/{total_steps} ({pct:.1f}%)")

    y_true = np.stack(y_true_rows).astype(np.float32)
    y_pred = np.stack(y_pred_rows).astype(np.float32)
    return y_true, y_pred


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.H is not None:
        cfg["data"]["H"] = int(args.H)

    seeds = (
        [int(s) for s in args.seeds]
        if args.seeds is not None and len(args.seeds) > 0
        else [int(s) for s in cfg.get("seeds", [int(cfg.get("seed", 42))])]
    )

    prepared, _ = load_and_prepare(
        csv_path=cfg["data"]["csv_path"],
        time_col=cfg["data"]["time_col"],
        target_col=cfg["data"]["target_col"],
        feed_prefix=cfg["data"]["feed_prefix"],
        stage1_prefix=cfg["data"]["stage1_prefix"],
        stage2_prefix=cfg["data"]["stage2_prefix"],
        stage3_prefix=cfg["data"]["stage3_prefix"],
        fillna=cfg["data"].get("fillna", "ffill"),
        use_delta_t=bool(cfg["data"].get("use_delta_t", True)),
        train_ratio=float(cfg["data"]["train_ratio"]),
        val_ratio=float(cfg["data"]["val_ratio"]),
        test_ratio=float(cfg["data"]["test_ratio"]),
    )

    L = int(cfg["data"]["L"])
    H = int(cfg["data"]["H"])

    arima_cfg = cfg.get("arima", {})
    order = tuple(arima_cfg.get("order", [3, 1, 0]))
    if len(order) != 3:
        raise ValueError("arima.order must be a 3-element list/tuple, e.g. [3,1,0].")
    order = (int(order[0]), int(order[1]), int(order[2]))
    trend = str(arima_cfg.get("trend", "n"))

    y_train = prepared.y_train
    y_val = prepared.y_val
    y_test = prepared.y_test

    y_test_true_scaled, _ = rolling_arima_forecast(
        history_prefix=np.concatenate([y_train, y_val], axis=0),
        series=y_test,
        L=L,
        H=H,
        order=order,
        trend=trend,
        progress_prefix="[Init-Test]",
        progress_every=int(arima_cfg.get("progress_every", 10)),
    )
    y_test_true = inverse_scale_2d(y_test_true_scaled, prepared.scaler_y)

    per_seed = []
    mae_vals, rmse_vals, r2_vals = [], [], []
    y_test_pred_last = None

    for seed in seeds:
        np.random.seed(int(seed))

        progress_every = int(arima_cfg.get("progress_every", 10))
        y_val_true_scaled, y_val_pred_scaled = rolling_arima_forecast(
            history_prefix=y_train,
            series=y_val,
            L=L,
            H=H,
            order=order,
            trend=trend,
            progress_prefix=f"[Seed {seed}][Val]",
            progress_every=progress_every,
        )
        y_test_true_scaled_seed, y_test_pred_scaled = rolling_arima_forecast(
            history_prefix=np.concatenate([y_train, y_val], axis=0),
            series=y_test,
            L=L,
            H=H,
            order=order,
            trend=trend,
            progress_prefix=f"[Seed {seed}][Test]",
            progress_every=progress_every,
        )

        y_val_true = inverse_scale_2d(y_val_true_scaled, prepared.scaler_y)
        y_val_pred = inverse_scale_2d(y_val_pred_scaled, prepared.scaler_y)
        y_test_true_seed = inverse_scale_2d(y_test_true_scaled_seed, prepared.scaler_y)
        y_test_pred = inverse_scale_2d(y_test_pred_scaled, prepared.scaler_y)
        y_test_pred_last = y_test_pred

        val_metrics = evaluate(y_val_true, y_val_pred)
        test_metrics = evaluate(y_test_true_seed, y_test_pred)

        per_seed.append(
            {
                "seed": int(seed),
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
            }
        )

        mae_vals.append(test_metrics["MAE"])
        rmse_vals.append(test_metrics["RMSE"])
        r2_vals.append(test_metrics["R2"])

        print(
            f"Seed {seed} Test: "
            f"MAE={test_metrics['MAE']:.4f}, RMSE={test_metrics['RMSE']:.4f}, R2={test_metrics['R2']:.4f}"
        )

    summary = {
        "MAE": summarize_metric(mae_vals),
        "RMSE": summarize_metric(rmse_vals),
        "R2": summarize_metric(r2_vals),
    }

    out_dir = PROJECT_ROOT / "outputs" / "baseline_arima"
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "L": L,
        "H": H,
        "seeds": seeds,
        "arima": {"order": list(order), "trend": trend},
        "per_seed": per_seed,
        "summary_test": summary,
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    cols = {}
    for i in range(H):
        cols[f"y_true_t{i + 1}"] = y_test_true[:, i]
    for i in range(H):
        cols[f"y_pred_t{i + 1}"] = y_test_pred_last[:, i]
    pd.DataFrame(cols).to_csv(out_dir / "test_pred_vs_true.csv", index=False)

    print("=== ARIMA Baseline Done ===")
    print(f"L={L}, H={H}, order={order}, trend={trend}")
    print("Seeds:", seeds)
    print(
        f"Test MAE: {summary['MAE']['mean']:.4f} +- {summary['MAE']['std']:.4f} | "
        f"RMSE: {summary['RMSE']['mean']:.4f} +- {summary['RMSE']['std']:.4f} | "
        f"R2: {summary['R2']['mean']:.4f} +- {summary['R2']['std']:.4f}"
    )
    print(f"Saved metrics: {out_dir / 'metrics.json'}")
    print(f"Saved preds:   {out_dir / 'test_pred_vs_true.csv'}")


if __name__ == "__main__":
    main()
