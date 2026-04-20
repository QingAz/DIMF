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

    备选滞后: ℓ=0..L_max，对应上游位置 idx=(t-ℓ)
    输出:
      msg: [B, D]             上游对齐消息 m_t
      pi:  [B, K]             π(ℓ|t)
      raw:[B, D]              未经过 W_v 的对齐上游表征，用于后续一致性约束
    """
    def __init__(
        self,
        dim: int,
        attn_dim: int,
        L_max: int,
        lag_emb: bool = True,
        tau: float = 1.0,
        use_lag_bias: bool = True,
    ):
        super().__init__()
        if tau <= 0:
            raise ValueError("tau must be positive")
        self.L_max = L_max
        self.tau = float(tau)
        self.Wq = nn.Linear(dim, attn_dim, bias=False)
        self.Wk = nn.Linear(dim, attn_dim, bias=False)
        self.Wv = nn.Linear(dim, dim, bias=False)
        # 第 6 点修改：为每个候选 lag 学习独立的延迟先验偏置 b_l。
        self.lag_bias = nn.Parameter(torch.zeros(L_max + 1)) if use_lag_bias else None
        self.scale = attn_dim ** 0.5

        self.lag_emb = lag_emb
        if lag_emb:
            # 为每个候选 lag 提供一个可学习的“延迟身份向量”。
            self.emb = nn.Embedding(L_max + 1, dim)

    def _build_candidates_last(self, up_seq: torch.Tensor):
        B, L, D = up_seq.shape
        # lag_idx = [0, 1, ..., L_max]，表示“当前时刻往回看多少步”。
        lag_idx = torch.arange(self.L_max + 1, device=up_seq.device)            # [K]
        # 当前预测时刻默认是窗口末端 L-1，所以候选上游位置是 (L-1)-lag。
        raw_pos = (L - 1) - lag_idx                                              # [K]
        valid_mask = raw_pos >= 0
        # 对无效 lag 先临时夹到 0，后续再用 valid_mask 把得分强制设成 -inf。
        gather_pos = raw_pos.clamp(min=0)
        up_raw = up_seq[:, gather_pos, :]                                         # [B, K, D]
        return lag_idx, valid_mask, up_raw

    def _build_candidates_seq(self, up_seq: torch.Tensor):
        B, L, D = up_seq.shape
        lag_idx = torch.arange(self.L_max + 1, device=up_seq.device)             # [K]
        time_idx = torch.arange(L, device=up_seq.device)                          # [L]
        # 对每个时间步 t，都构造一组位置 (t-lag) 的候选上游表征。
        raw_pos = time_idx[:, None] - lag_idx[None, :]                            # [L, K]
        valid_mask = raw_pos >= 0
        gather_pos = raw_pos.clamp(min=0)
        up_raw = up_seq[:, gather_pos, :]                                          # [B, L, K, D]
        return lag_idx, valid_mask, up_raw

    def forward(self, down_last: torch.Tensor, up_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 单时刻版本：只为窗口末端时刻 t 计算一组 lag 分布。
        lag_idx, valid_mask, up_raw = self._build_candidates_last(up_seq)
        key_input = up_raw
        if self.lag_emb:
            # lag embedding 仅参与匹配打分，不改变真实上游值的聚合语义。
            key_input = key_input + self.emb(lag_idx)[None, :, :]

        q = self.Wq(down_last)                                        # [B, A]
        k = self.Wk(key_input)                                        # [B, K, A]
        # alpha 是“下游当前状态”与“每个候选 lag 上游状态”的兼容性分数。
        alpha = torch.einsum("ba,bka->bk", q, k) / self.scale          # [B, K]
        if self.lag_bias is not None:
            # 学习到的 lag 先验会整体偏向某些常见延迟。
            alpha = alpha + self.lag_bias[None, :]
        # 第 6 点修改：显式屏蔽超出历史窗口的无效 lag，而不是简单截断。
        alpha = alpha.masked_fill(~valid_mask[None, :], float("-inf"))
        pi = F.softmax(alpha / self.tau, dim=-1)                      # [B, K]

        v = self.Wv(up_raw)                                           # [B, K, D]
        # msg 是送给下游融合模块的投影后消息；
        # raw_msg 则保留原始语义空间，专门给一致性损失使用。
        msg = torch.einsum("bk,bkd->bd", pi, v)                        # [B, D]
        raw_msg = torch.einsum("bk,bkd->bd", pi, up_raw)               # [B, D]
        return msg, pi, raw_msg

    def forward_seq(self, down_seq: torch.Tensor, up_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 序列版本：为窗口里的每个时间步都生成一份 lag 分布。
        lag_idx, valid_mask, up_raw = self._build_candidates_seq(up_seq)
        key_input = up_raw
        if self.lag_emb:
            key_input = key_input + self.emb(lag_idx)[None, None, :, :]

        q = self.Wq(down_seq)                                         # [B, L, A]
        k = self.Wk(key_input)                                        # [B, L, K, A]
        alpha = torch.einsum("bla,blka->blk", q, k) / self.scale       # [B, L, K]
        if self.lag_bias is not None:
            alpha = alpha + self.lag_bias[None, None, :]
        alpha = alpha.masked_fill(~valid_mask[None, :, :], float("-inf"))
        pi = F.softmax(alpha / self.tau, dim=-1)                      # [B, L, K]

        v = self.Wv(up_raw)                                           # [B, L, K, D]
        # 对每个时间步，按 pi 在候选 lag 维度上做加权和。
        msg = torch.einsum("blk,blkd->bld", pi, v)                     # [B, L, D]
        raw_msg = torch.einsum("blk,blkd->bld", pi, up_raw)            # [B, L, D]
        return msg, pi, raw_msg


class NoDelayAlignment(nn.Module):
    """
    不做显式 delay search 的对照版本：
    - 消息直接使用同一时刻的上游表示
    - delay 分布退化为 lag=0 的确定分布
    """

    def __init__(self, L_max: int):
        super().__init__()
        self.L_max = L_max

    def forward(self, down_last: torch.Tensor, up_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 对照组不搜索 lag，直接把当前时刻上游状态视作“已对齐”结果。
        del down_last
        B, L, D = up_seq.shape
        K = self.L_max + 1
        pi = up_seq.new_zeros((B, K))
        pi[:, 0] = 1.0
        msg = up_seq[:, -1, :].reshape(B, D)
        raw_msg = up_seq[:, -1, :].reshape(B, D)
        return msg, pi, raw_msg

    def forward_seq(self, down_seq: torch.Tensor, up_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 序列版本同理：每个时间步的分布都退化成 lag=0。
        del down_seq
        B, L, D = up_seq.shape
        K = self.L_max + 1
        pi = up_seq.new_zeros((B, L, K))
        pi[:, :, 0] = 1.0
        msg = up_seq.reshape(B, L, D)
        raw_msg = up_seq.reshape(B, L, D)
        return msg, pi, raw_msg
