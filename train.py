from __future__ import annotations
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
if sys.platform == "win32":
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        libbin = os.path.join(conda_prefix, "Library", "bin")
        os.environ["PATH"] = libbin + ";" + os.environ.get("PATH", "")
        try:
            os.add_dll_directory(libbin)
        except Exception:
            pass

import torch
import yaml
import joblib
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.seed import set_seed
from src.utils.logger import JsonlLogger
from src.utils.metrics import mae, mse, rmse, r2
from src.data.dataprocess import load_and_prepare
from src.data.dataset import MultistageWindowDataset, WindowSpec
from src.models.dimf import DIMF, entropy_loss, tv_loss

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--H", type=int, default=None)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def to_device(batch, device):
    X, y = batch
    X = {k: v.to(device) for k, v in X.items()}
    y = y.to(device)
    return X, y

@torch.no_grad()
def eval_loss(model, loader, device, lam_ent, lam_tv):
    model.eval()
    tot, n = 0.0, 0
    for batch in loader:
        X, y = to_device(batch, device)
        y_hat, pi = model(X)
        pred = (y_hat - y).abs().mean()
        ent = sum(entropy_loss(v) for v in pi.values())
        tv  = sum(tv_loss(v) for v in pi.values())
        loss = pred + lam_ent * ent + lam_tv * tv
        tot += float(loss.item()) * y.shape[0]
        n += y.shape[0]
    return tot / max(n, 1)

def _prediction_frame(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    H = y_true.shape[1]
    cols = {}
    for i in range(H):
        cols[f"y_true_t{i+1}"] = y_true[:, i]
    for i in range(H):
        cols[f"y_pred_t{i+1}"] = y_pred[:, i]
    return pd.DataFrame(cols)

@torch.no_grad()
def eval_test(model, loader, device, scaler_y, output_dir: str):
    model.eval()
    y_true, y_pred = [], []
    pis = {"feed_to_stage1": [], "stage1_to_stage2": [], "stage2_to_stage3": []}
    for X, y in loader:
        X = {k: v.to(device) for k, v in X.items()}
        y = y.to(device)
        y_hat, pi = model(X)
        y_true.append(y.cpu().numpy())
        y_pred.append(y_hat.cpu().numpy())
        for k in pis:
            pis[k].append(pi[k].cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    scaled_res = {
        "scaled_MSE": mse(y_true, y_pred),
        "scaled_MAE": mae(y_true, y_pred),
        "scaled_RMSE": rmse(y_true, y_pred),
        "scaled_R2": r2(y_true, y_pred),
    }

    y_true_inv = scaler_y.inverse_transform(y_true.reshape(-1, 1)).reshape(y_true.shape)
    y_pred_inv = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).reshape(y_pred.shape)

    res = {
        "MSE": mse(y_true_inv, y_pred_inv),
        "MAE": mae(y_true_inv, y_pred_inv),
        "RMSE": rmse(y_true_inv, y_pred_inv),
        "R2": r2(y_true_inv, y_pred_inv),
        "n_test": int(y_true_inv.shape[0]),
    }
    res.update(scaled_res)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _prediction_frame(y_true_inv, y_pred_inv).to_csv(out_dir / "test_pred_vs_true.csv", index=False)
    _prediction_frame(y_true, y_pred).to_csv(out_dir / "test_pred_vs_true_scaled.csv", index=False)

    avg_pi = {}
    for k, arrs in pis.items():
        arr = np.concatenate(arrs, axis=0)
        if arr.ndim == 3:                   # [N, L, K]
            avg_pi[k] = arr.mean(axis=(0, 1))
        else:                               # [N, K]
            avg_pi[k] = arr.mean(axis=0)
    np.save(out_dir / "test_delay_pi.npy", avg_pi, allow_pickle=True)
    return res

def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.H is not None:
        cfg["data"]["H"] = int(args.H)

    set_seed(int(cfg.get("seed", 42)))

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
        split_mode=str(cfg["data"].get("split_mode", "rows")),
        history_steps=int(cfg["data"]["L"]),
        horizon_steps=int(cfg["data"]["H"]),
        collection_interval_min=int(cfg["data"].get("collection_interval_min", 15)),
        include_target_history=bool(cfg["data"].get("include_target_history", False)),
    )

    # save scalers (fit on train only)
    Path(cfg["logging"]["scaler_path"]).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"scaler_x": prepared.scaler_x, "scaler_y": prepared.scaler_y}, cfg["logging"]["scaler_path"])

    spec = WindowSpec(L=int(cfg["data"]["L"]), H=int(cfg["data"]["H"]))
    ds_tr = MultistageWindowDataset(
        prepared.X_groups_train,
        prepared.y_train,
        spec,
        indices=prepared.sample_indices_train,
    )
    ds_va = MultistageWindowDataset(
        prepared.X_groups_val,
        prepared.y_val,
        spec,
        indices=prepared.sample_indices_val,
    )
    ds_te = MultistageWindowDataset(
        prepared.X_groups_test,
        prepared.y_test,
        spec,
        indices=prepared.sample_indices_test,
    )

    dl_tr = DataLoader(ds_tr, batch_size=int(cfg["train"]["batch_size"]), shuffle=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=int(cfg["train"]["batch_size"]), shuffle=False)
    dl_te = DataLoader(ds_te, batch_size=int(cfg["train"]["batch_size"]), shuffle=False)

    device = torch.device(args.device)
    model = DIMF(
        group_dims=prepared.group_dims,
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        num_layers=int(cfg["model"]["num_layers"]),
        dropout=float(cfg["model"]["dropout"]),
        attn_dim=int(cfg["model"]["attn_dim"]),
        L_max=int(cfg["data"]["L_max"]),
        horizon=int(cfg["data"]["H"]),
        encoder_type=str(cfg["model"].get("encoder", "gru")),
        transformer_nhead=int(cfg["model"].get("transformer_nhead", 4)),
        transformer_ff_dim=cfg["model"].get("transformer_ff_dim", None),
        max_len=int(cfg["data"]["L"]),
        lag_emb=bool(cfg["model"].get("lag_emb", True)),
    ).to(device)

    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"].get("weight_decay", 0.0)),
    )

    logger = JsonlLogger(cfg["logging"]["log_path"])
    ckpt_path = cfg["logging"]["ckpt_path"]
    Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)

    lam_ent  = float(cfg["train"]["lambda_ent"])
    lam_tv   = float(cfg["train"]["lambda_tv"])
    grad_clip = float(cfg["train"].get("grad_clip", 1.0))
    patience  = int(cfg["train"]["early_stop_patience"])

    best, bad = 1e18, 0
    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        model.train()
        tot, n = 0.0, 0

        pbar = tqdm(dl_tr, desc=f"Epoch {epoch}")
        for batch in pbar:
            X, y = to_device(batch, device)
            y_hat, pi = model(X)

            pred = (y_hat - y).abs().mean()
            ent  = sum(entropy_loss(v) for v in pi.values())
            tv   = sum(tv_loss(v) for v in pi.values())
            loss = pred + lam_ent * ent + lam_tv * tv

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            tot += float(loss.item()) * y.shape[0]
            n += y.shape[0]
            pbar.set_postfix(loss=float(loss.item()), pred=float(pred.item()), ent=float(ent.item()), tv=float(tv.item()))

        val = eval_loss(model, dl_va, device, lam_ent, lam_tv)
        train_loss = tot / max(n, 1)
        print(f"Epoch {epoch}: train_loss={train_loss:.6f} val_loss={val:.6f} H={int(cfg['data']['H'])}")
        logger.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val, "H": int(cfg["data"]["H"])})

        if val < best - 1e-6:
            best, bad = val, 0
            torch.save({"model": model.state_dict(), "cfg": cfg}, ckpt_path)
        else:
            bad += 1
            if bad >= patience:
                break

    print(f"[Done] best_val={best:.6f}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Scaler: {cfg['logging']['scaler_path']}")

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    output_dir = cfg["logging"].get("output_dir", "outputs")
    test_res = eval_test(model, dl_te, device, prepared.scaler_y, output_dir)
    print("=== Test Metrics ===")
    for k, v in test_res.items():
        print(f"{k}: {v}")
    print(f"Saved: {Path(output_dir) / 'test_pred_vs_true.csv'}")
    print(f"Saved: {Path(output_dir) / 'test_pred_vs_true_scaled.csv'}")
    print(f"Saved: {Path(output_dir) / 'test_delay_pi.npy'}")

if __name__ == "__main__":
    main()
