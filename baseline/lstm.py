from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataprocess import load_and_prepare
from src.utils.metrics import mae, rmse, r2
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--H", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
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
        x_seq = np.concatenate([X_groups[g][t - L + 1 : t + 1] for g in group_order], axis=1)
        y_target = y[t + 1 : t + H + 1]
        x_rows.append(x_seq)
        y_rows.append(y_target)

    X = np.stack(x_rows).astype(np.float32)  # [N, L, D]
    Y = np.stack(y_rows).astype(np.float32)  # [N, H]
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


class LSTMBaseline(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float, horizon: int):
        super().__init__()
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=effective_dropout,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_n = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        pred = model(xb)
        loss = criterion(pred, yb)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        bs = xb.size(0)
        total_loss += float(loss.item()) * bs
        total_n += bs

    return total_loss / max(total_n, 1)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    for xb, _ in loader:
        xb = xb.to(device)
        preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds, axis=0)


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

    train_cfg = cfg.get("train", {})
    model_cfg = cfg.get("model", {})
    hidden_dim = int(model_cfg.get("hidden_dim", 64))
    num_layers = int(model_cfg.get("num_layers", 2))
    dropout = float(model_cfg.get("dropout", 0.0))
    epochs = int(train_cfg.get("epochs", 200))
    batch_size = int(train_cfg.get("batch_size", 256))
    lr = float(train_cfg.get("lr", 5e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    patience = int(train_cfg.get("early_stop_patience", 50))

    device = torch.device(args.device)

    tr_loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    va_loader = DataLoader(
        TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva)),
        batch_size=batch_size,
        shuffle=False,
    )
    te_loader = DataLoader(
        TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte)),
        batch_size=batch_size,
        shuffle=False,
    )

    yte_inv = inverse_scale_2d(yte, prepared.scaler_y)
    per_seed = []
    mae_vals, rmse_vals, r2_vals = [], [], []
    yte_hat_last = None

    for seed in seeds:
        print(f"\n########## Seed {seed} ##########")
        set_seed(int(seed))

        model = LSTMBaseline(
            input_dim=Xtr.shape[2],
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            horizon=H,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()

        best_state = None
        best_val = float("inf")
        bad_epochs = 0

        for epoch in range(1, epochs + 1):
            train_loss = run_epoch(model, tr_loader, criterion, device, optimizer=optimizer)
            val_loss = run_epoch(model, va_loader, criterion, device, optimizer=None)
            print(f"Epoch {epoch:03d}/{epochs} | train_mse={train_loss:.6f} | val_mse={val_loss:.6f}")

            if val_loss < best_val - 1e-8:
                best_val = val_loss
                bad_epochs = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print(f"Early stop at epoch {epoch} (patience={patience})")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        yva_hat = predict(model, va_loader, device)
        yte_hat = predict(model, te_loader, device)

        yva_inv = inverse_scale_2d(yva, prepared.scaler_y)
        yva_hat_inv = inverse_scale_2d(yva_hat, prepared.scaler_y)
        yte_hat_inv = inverse_scale_2d(yte_hat, prepared.scaler_y)
        yte_hat_last = yte_hat_inv

        val_metrics = evaluate(yva_inv, yva_hat_inv)
        test_metrics = evaluate(yte_inv, yte_hat_inv)
        per_seed.append({"seed": int(seed), "best_val_mse": float(best_val), "val_metrics": val_metrics, "test_metrics": test_metrics})

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

    out_dir = PROJECT_ROOT / "outputs" / "baseline_lstm"
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "L": L,
        "H": H,
        "seeds": seeds,
        "model": {
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
        },
        "train": {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "early_stop_patience": patience,
        },
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

    print("=== LSTM Baseline Done ===")
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
