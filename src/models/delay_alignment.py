from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class DelayAlignment(nn.Module):
    """
    可微滞后对齐（相邻工段）：
      down_last: [B, D]        下游当前表示 e_t^(s)
      up_seq:    [B, L, D]     上游窗口内逐步表示 E^(s-1)

    备选滞后: ℓ=0..L_max，对应上游位置 idx=(L-1-ℓ)
    输出:
      msg: [B, D]             上游对齐消息 m_t
      pi:  [B, K]             π(ℓ|t)
    """
    def __init__(self, dim: int, attn_dim: int, L_max: int, lag_emb: bool = True):
        super().__init__()
        self.L_max = L_max
        self.Wq = nn.Linear(dim, attn_dim, bias=False)
        self.Wk = nn.Linear(dim, attn_dim, bias=False)
        self.Wv = nn.Linear(dim, dim, bias=False)
        self.scale = attn_dim ** 0.5

        self.lag_emb = lag_emb
        if lag_emb:
            self.emb = nn.Embedding(L_max + 1, dim)

    def forward(self, down_last: torch.Tensor, up_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D = up_seq.shape
        L_max = min(self.L_max, L - 1)
        idx = torch.arange(L_max + 1, device=up_seq.device)          # [K]
        pos = (L - 1 - idx).clamp(min=0)
        up_cand = up_seq[:, pos, :]                                   # [B, K, D]

        if self.lag_emb:
            up_cand = up_cand + self.emb(idx)[None, :, :]

        q = self.Wq(down_last)                                        # [B, A]
        k = self.Wk(up_cand)                                          # [B, K, A]
        alpha = torch.einsum("ba,bka->bk", q, k) / self.scale          # [B, K]
        pi = F.softmax(alpha, dim=-1)                                 # [B, K]

        v = self.Wv(up_cand)                                          # [B, K, D]
        msg = torch.einsum("bk,bkd->bd", pi, v)                        # [B, D]
        return msg, pi

    def forward_seq(self, down_seq: torch.Tensor, up_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D = up_seq.shape
        L_max = min(self.L_max, L - 1)
        idx = torch.arange(L_max + 1, device=up_seq.device)          # [K]
        t_idx = torch.arange(L, device=up_seq.device)                 # [L]
        pos = (t_idx[:, None] - idx[None, :]).clamp(min=0)             # [L, K]
        up_cand = up_seq[:, pos, :]                                   # [B, L, K, D]

        if self.lag_emb:
            up_cand = up_cand + self.emb(idx)[None, None, :, :]

        q = self.Wq(down_seq)                                         # [B, L, A]
        k = self.Wk(up_cand)                                          # [B, L, K, A]
        alpha = torch.einsum("bla,blka->blk", q, k) / self.scale       # [B, L, K]
        pi = F.softmax(alpha, dim=-1)                                 # [B, L, K]

        v = self.Wv(up_cand)                                          # [B, L, K, D]
        msg = torch.einsum("blk,blkd->bld", pi, v)                     # [B, L, D]
        return msg, pi
