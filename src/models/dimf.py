from __future__ import annotations
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import GRUEncoder, LSTMEncoder, TransformerEncoder
from .delay_alignment import DelayAlignment, NoDelayAlignment

class GatedResidualFusion(nn.Module):
    """
    第 7 点修改：使用门控残差融合
        g_t = sigmoid(W_g [e_t ; m_t] + b_g)
        u_t = e_t + g_t ⊙ psi(m_t)
    """

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.msg_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, base_seq: torch.Tensor, msg_seq: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate(torch.cat([base_seq, msg_seq], dim=-1)))
        msg_update = self.msg_proj(msg_seq)
        return base_seq + gate * msg_update

class DIMF(nn.Module):
    """
    DIMF: 分段编码 + 相邻滞后对齐 + 最终单点预测（用 stage3）
    其中每个工段编码器接收的输入已在数据侧拼接为：[x^(s), delta_t, mask^(s)]。
    第 6 点修改后，对齐模块还会保留未经过 W_v 的对齐上游表征，供后续一致性约束使用。
    """
    def __init__(self, group_dims: Dict[str, int], hidden_dim: int, num_layers: int, dropout: float,
                 attn_dim: int, L_max: int, lead_steps: int, encoder_type: str = "gru",
                 transformer_nhead: int = 4, transformer_ff_dim: int = None, max_len: int = 512,
                 lag_emb: bool = True, use_alignment: bool = True, align_tau: float = 1.0,
                 align_dropout: float = 0.0, align_feed_to_stage1: bool | None = None,
                 align_stage1_to_stage2: bool | None = None, align_stage2_to_stage3: bool | None = None,
                 stage1_to_stage2_confidence_mode: str | None = None,
                 stage1_to_stage2_confidence_require_nonzero_argmax: bool = False,
                 stage1_to_stage2_confidence_peak_threshold: float | None = None,
                 stage1_to_stage2_confidence_nonzero_threshold: float | None = None,
                 stage1_to_stage2_confidence_sharpness: float = 20.0):
        super().__init__()
        # 第 1 点修改：lead_steps 仅表示固定提前量 H，预测头始终输出单个 y_{t+H}。
        self.lead_steps = lead_steps
        self.align_dropout = float(align_dropout)
        self.latest_alignment_cache = {}
        self.align_feed_to_stage1 = use_alignment if align_feed_to_stage1 is None else bool(align_feed_to_stage1)
        self.align_stage1_to_stage2 = use_alignment if align_stage1_to_stage2 is None else bool(align_stage1_to_stage2)
        self.align_stage2_to_stage3 = use_alignment if align_stage2_to_stage3 is None else bool(align_stage2_to_stage3)
        confidence_mode = None if stage1_to_stage2_confidence_mode is None else str(stage1_to_stage2_confidence_mode).lower()
        if confidence_mode in {"", "none"}:
            confidence_mode = None
        if confidence_mode not in {None, "hard", "soft"}:
            raise ValueError(f"Unknown stage1_to_stage2_confidence_mode: {stage1_to_stage2_confidence_mode}")
        self.stage1_to_stage2_confidence_mode = confidence_mode
        self.stage1_to_stage2_confidence_require_nonzero_argmax = bool(stage1_to_stage2_confidence_require_nonzero_argmax)
        self.stage1_to_stage2_confidence_peak_threshold = (
            None if stage1_to_stage2_confidence_peak_threshold is None else float(stage1_to_stage2_confidence_peak_threshold)
        )
        self.stage1_to_stage2_confidence_nonzero_threshold = (
            None if stage1_to_stage2_confidence_nonzero_threshold is None else float(stage1_to_stage2_confidence_nonzero_threshold)
        )
        self.stage1_to_stage2_confidence_sharpness = float(stage1_to_stage2_confidence_sharpness)
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

        def make_alignment(enabled: bool):
            if enabled:
                return DelayAlignment(hidden_dim, attn_dim, L_max, lag_emb, tau=align_tau)
            return NoDelayAlignment(L_max)

        self.align_0_1 = make_alignment(self.align_feed_to_stage1)
        self.align_1_2 = make_alignment(self.align_stage1_to_stage2)
        self.align_2_3 = make_alignment(self.align_stage2_to_stage3)

        self.fuse_s1 = GatedResidualFusion(hidden_dim, dropout)
        self.fuse_s2 = GatedResidualFusion(hidden_dim, dropout)
        self.fuse_s3 = GatedResidualFusion(hidden_dim, dropout)
        # 预测目标改为单点值，而不是长度为 H 的向量。
        self.head = nn.Linear(hidden_dim, 1)

    def _apply_alignment_dropout(self, msg_seq: torch.Tensor) -> torch.Tensor:
        # 第 7 点修改：训练期以概率 p_align 将整条对齐消息置零，模拟“对齐消息不稳定或缺失”。
        if (not self.training) or self.align_dropout <= 0.0:
            return msg_seq
        keep_mask = (
            torch.rand(*msg_seq.shape[:-1], 1, device=msg_seq.device) >= self.align_dropout
        ).to(msg_seq.dtype)
        return msg_seq * keep_mask

    def _apply_stage1_to_stage2_confidence_gate(
        self,
        msg_seq: torch.Tensor,
        pi_seq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.stage1_to_stage2_confidence_mode is None:
            conf = torch.ones_like(msg_seq[..., :1])
            return msg_seq, conf

        conf = torch.ones_like(msg_seq[..., :1])
        if self.stage1_to_stage2_confidence_require_nonzero_argmax:
            argmax_nonzero = pi_seq.argmax(dim=-1, keepdim=True).ne(0).to(msg_seq.dtype)
            conf = conf * argmax_nonzero

        peak_prob = pi_seq.max(dim=-1, keepdim=True).values
        nonzero_mass = 1.0 - pi_seq[..., :1]

        if self.stage1_to_stage2_confidence_peak_threshold is not None:
            threshold = self.stage1_to_stage2_confidence_peak_threshold
            if self.stage1_to_stage2_confidence_mode == "soft":
                conf = conf * torch.sigmoid((peak_prob - threshold) * self.stage1_to_stage2_confidence_sharpness)
            else:
                conf = conf * (peak_prob >= threshold).to(msg_seq.dtype)

        if self.stage1_to_stage2_confidence_nonzero_threshold is not None:
            threshold = self.stage1_to_stage2_confidence_nonzero_threshold
            if self.stage1_to_stage2_confidence_mode == "soft":
                conf = conf * torch.sigmoid((nonzero_mass - threshold) * self.stage1_to_stage2_confidence_sharpness)
            else:
                conf = conf * (nonzero_mass >= threshold).to(msg_seq.dtype)

        return msg_seq * conf, conf

    def forward(self, X_seq: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        E0_seq, E0_last = self.enc_feed(X_seq["feed"])
        E1_seq, E1_last = self.enc_s1(X_seq["stage1"])
        E2_seq, E2_last = self.enc_s2(X_seq["stage2"])
        E3_seq, E3_last = self.enc_s3(X_seq["stage3"])

        m1_seq, pi01, raw1_seq = self.align_0_1.forward_seq(E1_seq, E0_seq)   # feed -> stage1
        m1_seq = self._apply_alignment_dropout(m1_seq)
        u1_seq = self.fuse_s1(E1_seq, m1_seq)

        m2_seq, pi12, raw2_seq = self.align_1_2.forward_seq(E2_seq, u1_seq)   # stage1 -> stage2
        m2_seq = self._apply_alignment_dropout(m2_seq)
        m2_seq, conf12_seq = self._apply_stage1_to_stage2_confidence_gate(m2_seq, pi12)
        u2_seq = self.fuse_s2(E2_seq, m2_seq)

        m3_seq, pi23, raw3_seq = self.align_2_3.forward_seq(E3_seq, u2_seq)   # stage2 -> stage3
        m3_seq = self._apply_alignment_dropout(m3_seq)
        u3_seq = self.fuse_s3(E3_seq, m3_seq)
        u3 = u3_seq[:, -1, :]

        # 第 6/8 点修改：缓存每条边的对齐分布、未投影上游对齐表示、下游编码表示，
        # 供后续一致性损失直接读取。
        self.latest_alignment_cache = {
            "feed_to_stage1": {
                "active": self.align_feed_to_stage1,
                "pi": pi01,
                "aligned_upstream_raw": raw1_seq,
                "downstream_seq": E1_seq,
            },
            "stage1_to_stage2": {
                "active": self.align_stage1_to_stage2,
                "pi": pi12,
                "fusion_confidence": conf12_seq,
                "aligned_upstream_raw": raw2_seq,
                "downstream_seq": E2_seq,
            },
            "stage2_to_stage3": {
                "active": self.align_stage2_to_stage3,
                "pi": pi23,
                "aligned_upstream_raw": raw3_seq,
                "downstream_seq": E3_seq,
            },
        }

        # 输出形状整理为 [B]，与单点标签 y[t+H] 对齐。
        y_hat = self.head(u3).squeeze(-1)
        return y_hat, {"feed_to_stage1": pi01, "stage1_to_stage2": pi12, "stage2_to_stage3": pi23}

def entropy_loss(pi: torch.Tensor) -> torch.Tensor:
    # encourage peaked distribution (prefer a clearer delay)
    return -(pi * (pi.clamp(min=1e-12).log())).sum(dim=-1).mean()

def tv_loss(pi: torch.Tensor) -> torch.Tensor:
    # smoothness over time steps (L dimension)
    if pi.dim() < 3 or pi.shape[1] < 2:
        return pi.new_tensor(0.0)
    return (pi[:, 1:, :] - pi[:, :-1, :]).abs().mean()


def symmetric_info_nce_loss(
    downstream_repr: torch.Tensor,
    upstream_repr: torch.Tensor,
    temperature: float = 0.2,
) -> torch.Tensor:
    """
    第 8 点修改：对同一 batch 内的正样本配对做对称 InfoNCE。
    这里使用窗口末端当前时刻 t 的表示：
      - downstream_repr: e_t^(s)
      - upstream_repr:   \tilde e_t^(s-1)
    """
    if temperature <= 0:
        raise ValueError("InfoNCE temperature must be positive")
    if downstream_repr.ndim != 2 or upstream_repr.ndim != 2:
        raise ValueError("InfoNCE inputs must be rank-2 tensors shaped as [B, D]")
    if downstream_repr.shape != upstream_repr.shape:
        raise ValueError("Downstream/upstream representations for InfoNCE must share the same shape")

    z_down = F.normalize(downstream_repr, dim=-1)
    z_up = F.normalize(upstream_repr, dim=-1)
    logits = torch.matmul(z_down, z_up.transpose(0, 1)) / temperature
    labels = torch.arange(z_down.shape[0], device=z_down.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.transpose(0, 1), labels))


def alignment_consistency_loss(
    alignment_cache: Dict[str, Dict[str, torch.Tensor]],
    temperature: float = 0.2,
) -> torch.Tensor:
    """
    第 8 点修改：逐条工段边计算对齐一致性损失，并求和：
      L_align = sum_s InfoNCE(e_t^(s), \tilde e_t^(s-1))
    当前训练样本定义在窗口末端时刻 t，因此这里显式使用每个样本最后一个时间步。
    """
    if not alignment_cache:
        raise ValueError("alignment_cache is empty; run model forward before computing L_align")

    total = None
    for edge_name, edge_cache in alignment_cache.items():
        if not bool(edge_cache.get("active", True)):
            continue
        if "downstream_seq" not in edge_cache or "aligned_upstream_raw" not in edge_cache:
            raise ValueError(f"Alignment cache for {edge_name} is missing required tensors")
        downstream_last = edge_cache["downstream_seq"][:, -1, :]
        upstream_last = edge_cache["aligned_upstream_raw"][:, -1, :]
        edge_loss = symmetric_info_nce_loss(downstream_last, upstream_last, temperature=temperature)
        total = edge_loss if total is None else total + edge_loss
    if total is None:
        first_edge = next(iter(alignment_cache.values()))
        reference = first_edge["downstream_seq"]
        return reference.new_tensor(0.0)
    return total
