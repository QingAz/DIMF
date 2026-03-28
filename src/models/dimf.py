from __future__ import annotations
from typing import Dict, Tuple
import torch
import torch.nn as nn

from .encoders import GRUEncoder, LSTMEncoder, TransformerEncoder
from .delay_alignment import DelayAlignment

class DIMF(nn.Module):
    """
    DIMF: 分段编码 + 相邻滞后对齐 + 最终预测（用 stage3）
    """
    def __init__(self, group_dims: Dict[str, int], hidden_dim: int, num_layers: int, dropout: float,
                 attn_dim: int, L_max: int, horizon: int, encoder_type: str = "gru",
                 transformer_nhead: int = 4, transformer_ff_dim: int = None, max_len: int = 512,
                 lag_emb: bool = True):
        super().__init__()
        enc_type = encoder_type.lower()
        def make_encoder(d_in: int):
            if enc_type == "gru":
                return GRUEncoder(d_in, hidden_dim, num_layers, dropout)
            if enc_type == "lstm":
                return LSTMEncoder(d_in, hidden_dim, num_layers, dropout)
            if enc_type == "transformer":
                return TransformerEncoder(
                    d_in,
                    hidden_dim,
                    num_layers=num_layers,
                    dropout=dropout,
                    nhead=transformer_nhead,
                    ff_dim=transformer_ff_dim,
                    max_len=max_len,
                )
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        self.enc_feed = make_encoder(group_dims["feed"])
        self.enc_s1   = make_encoder(group_dims["stage1"])
        self.enc_s2   = make_encoder(group_dims["stage2"])
        self.enc_s3   = make_encoder(group_dims["stage3"])

        self.align_0_1 = DelayAlignment(hidden_dim, attn_dim, L_max, lag_emb)
        self.align_1_2 = DelayAlignment(hidden_dim, attn_dim, L_max, lag_emb)
        self.align_2_3 = DelayAlignment(hidden_dim, attn_dim, L_max, lag_emb)

        def fuse():
            return nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )

        self.fuse_s1 = fuse()
        self.fuse_s2 = fuse()
        self.fuse_s3 = fuse()
        self.head = nn.Linear(hidden_dim, horizon)

    def forward(self, X_seq: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        E0_seq, E0_last = self.enc_feed(X_seq["feed"])
        E1_seq, E1_last = self.enc_s1(X_seq["stage1"])
        E2_seq, E2_last = self.enc_s2(X_seq["stage2"])
        E3_seq, E3_last = self.enc_s3(X_seq["stage3"])

        m1_seq, pi01 = self.align_0_1.forward_seq(E1_seq, E0_seq)   # feed -> stage1
        u1_seq = self.fuse_s1(torch.cat([E1_seq, m1_seq], dim=-1))

        m2_seq, pi12 = self.align_1_2.forward_seq(E2_seq, u1_seq)   # stage1 -> stage2
        u2_seq = self.fuse_s2(torch.cat([E2_seq, m2_seq], dim=-1))

        m3_seq, pi23 = self.align_2_3.forward_seq(E3_seq, u2_seq)   # stage2 -> stage3
        u3_seq = self.fuse_s3(torch.cat([E3_seq, m3_seq], dim=-1))
        u3 = u3_seq[:, -1, :]

        y_hat = self.head(u3)
        return y_hat, {"feed_to_stage1": pi01, "stage1_to_stage2": pi12, "stage2_to_stage3": pi23}

def entropy_loss(pi: torch.Tensor) -> torch.Tensor:
    # encourage peaked distribution (prefer a clearer delay)
    return -(pi * (pi.clamp(min=1e-12).log())).sum(dim=-1).mean()

def tv_loss(pi: torch.Tensor) -> torch.Tensor:
    # smoothness over time steps (L dimension)
    if pi.dim() < 3 or pi.shape[1] < 2:
        return pi.new_tensor(0.0)
    return (pi[:, 1:, :] - pi[:, :-1, :]).abs().mean()
