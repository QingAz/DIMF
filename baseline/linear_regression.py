from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataprocess import load_and_prepare
from src.utils.metrics import mae, rmse, r2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--H", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_windows(
    X_groups: Dict[str, np.ndarray], y: np.ndarray, L: int, H: int
) -> Tuple[np.ndarray, np.ndarray]:
    group_order = sorted(X_groups.keys())
    T = next(iter(X_groups.values())).shape[0]

    t_min = L - 1
    t_max = T - H - 1

    x_rows = []
    y_rows = []
    for t in range(t_min, t_max + 1):
        feature = np.concatenate(
            [X_groups[g][t - L + 1 : t + 1].reshape(-1) for g in group_order], axis=0
        )
        target = y[t + 1 : t + H + 1]
        x_rows.append(feature)
        y_rows.append(target)

    X = np.stack(x_rows).astype(np.float32)
    Y = np.stack(y_rows).astype(np.float32)
    return X, Y


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

    Xtr, ytr = build_windows(prepared.X_groups_train, prepared.y_train, L=L, H=H)
    Xva, yva = build_windows(prepared.X_groups_val, prepared.y_val, L=L, H=H)
    Xte, yte = build_windows(prepared.X_groups_test, prepared.y_test, L=L, H=H)

    yte_inv = inverse_scale_2d(yte, prepared.scaler_y)
    per_seed = []
    mae_vals, rmse_vals, r2_vals = [], [], []
    yte_hat_last = None

    for seed in seeds:
        # LinearRegression is deterministic for fixed data; repeated seeds keep the same evaluation protocol.
        print(f"\n########## Seed {seed} ##########")

        model = MultiOutputRegressor(LinearRegression())
        model.fit(Xtr, ytr)

        yva_hat = model.predict(Xva).astype(np.float32)
        yte_hat = model.predict(Xte).astype(np.float32)

        yva_inv = inverse_scale_2d(yva, prepared.scaler_y)
        yva_hat_inv = inverse_scale_2d(yva_hat, prepared.scaler_y)
        yte_hat_inv = inverse_scale_2d(yte_hat, prepared.scaler_y)
        yte_hat_last = yte_hat_inv

        val_metrics = evaluate(yva_inv, yva_hat_inv)
        test_metrics = evaluate(yte_inv, yte_hat_inv)
        per_seed.append({"seed": int(seed), "val_metrics": val_metrics, "test_metrics": test_metrics})

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

    out_dir = PROJECT_ROOT / "outputs" / "baseline_linear_regression"
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "L": L,
        "H": H,
        "seeds": seeds,
        "model": "LinearRegression",
        "per_seed": per_seed,
        "summary_test": summary,
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    cols = {}
    for i in range(H):
        cols[f"y_true_t{i + 1}"] = yte_inv[:, i]
    for i in range(H):
        cols[f"y_pred_t{i + 1}"] = yte_hat_last[:, i]
    pd.DataFrame(cols).to_csv(out_dir / "test_pred_vs_true.csv", index=False)

    print("=== Linear Regression Baseline Done ===")
    print(f"L={L}, H={H}")
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
