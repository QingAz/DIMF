from __future__ import annotations
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import GRUEncoder, LSTMEncoder, TransformerEncoder
from .delay_alignment import DelayAlignment, NoDelayAlignment

# 这个文件实现 DIMF（Delay-Integrated Multi-stage Forecasting）的主干逻辑。
# 整体思路可以概括成四步：
# 1. 先分别编码 feed / stage1 / stage2 / stage3 四个工段的窗口序列；
# 2. 再在相邻工段之间做可微滞后对齐，寻找“当前下游状态最可能对应的上游历史时刻”；
# 3. 将得到的上游对齐消息逐级注入下游工段表示，形成带跨工段信息的融合表示；
# 4. 最后只使用 stage3 窗口末端的融合表示，预测固定提前量 H 对应的单个目标值。
#
# 下面注释里会频繁使用这些记号：
# - B: batch size
# - L: 窗口长度（一个样本里包含多少个时间步）
# - D: 隐藏维度 hidden_dim
# - K: 候选 lag 数量，等于 L_max + 1

class GatedResidualFusion(nn.Module):
    """
    门控残差融合模块，用于把“本工段自身编码”与“来自上游的对齐消息”合成到一起。

    这里采用残差式而不是直接拼接后再过大 MLP，原因是：
    1. 保留一条稳定的恒等路径，避免上游消息质量不高时破坏本地表示；
    2. 让模型能按隐藏维度细粒度控制“注入多少上游信息”；
    3. 在做消融或训练早期时，门控学到接近 0 也不会让表示崩掉。

    计算形式为：
        g_t = sigmoid(W_g [e_t ; m_t] + b_g)
        u_t = e_t + g_t ⊙ psi(m_t)

    其中：
    - e_t: 当前工段在时刻 t 的编码表示
    - m_t: 从上游工段对齐得到的消息
    - psi: 对上游消息做非线性投影，提升可融合性
    - g_t: 每个隐藏维度独立学习的注入系数
    """

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        # gate 接收“本工段编码 + 上游对齐消息”的拼接向量，
        # 为每个隐藏维度学习一个 0~1 的残差注入系数。
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        # msg_proj 先把上游消息映射到更适合融合的空间，
        # 再由门控决定注入多少信息。
        self.msg_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, base_seq: torch.Tensor, msg_seq: torch.Tensor) -> torch.Tensor:
        # base_seq/msg_seq: [B, L, D]
        # 两个输入在时间维和隐藏维上必须对齐：
        # - base_seq 表示“这一工段自己在每个时刻学到了什么”
        # - msg_seq  表示“上游工段对齐后，在每个时刻传来了什么”
        #
        # 逐时间步地根据当前工段状态和对齐消息计算门控。
        gate = torch.sigmoid(self.gate(torch.cat([base_seq, msg_seq], dim=-1)))
        msg_update = self.msg_proj(msg_seq)
        # 残差形式可以保证：即便门控学到接近 0，模型也至少保留原始工段编码。
        return base_seq + gate * msg_update

class DIMF(nn.Module):
    """
    DIMF 主模型：分段编码 + 相邻滞后对齐 + 逐级融合 + 最终单点预测。

    数据流可以理解为：
        feed --对齐--> stage1 --对齐--> stage2 --对齐--> stage3 --预测--> y[t+H]

    关键设计点：
    1. 四个工段各自先做时序编码，抽取本工段内部动态；
    2. 相邻工段之间不直接“同时间步硬对齐”，而是搜索一个可学习的 lag 分布；
    3. 对齐后的上游信息按层级向后传递，使 stage3 能同时感受到多级上游影响；
    4. 预测头固定只输出一个标量，对应监督标签 y[t+H]。

    约定：
    - 每个工段编码器的输入都已经在数据侧拼接成 [x^(s), delta_t, mask^(s)]；
    - forward 返回预测值和各条边的对齐分布；
    - 同时会把更细的对齐中间量缓存在 latest_alignment_cache 中，
      供训练时额外的一致性损失直接复用。
    """
    def __init__(
        self,
        group_dims: Dict[str, int],
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        attn_dim: int,
        L_max: int,
        lead_steps: int,
        encoder_type: str = "gru",
        transformer_nhead: int = 4,
        transformer_ff_dim: int = None,
        max_len: int = 512,
        lag_emb: bool = True,
        use_alignment: bool = True,
        align_tau: float = 1.0,
        align_dropout: float = 0.0,
        align_feed_to_stage1: bool | None = None,
        align_stage1_to_stage2: bool | None = None,
        align_stage2_to_stage3: bool | None = None,
        use_lag_bias: bool = True,
        lag_head_mode: str = "softmax",
    ):
        super().__init__()
        # 第 1 点修改：lead_steps 仅表示固定提前量 H，预测头始终输出单个 y_{t+H}。
        self.lead_steps = lead_steps
        self.align_dropout = float(align_dropout)
        # forward 后会把每条工段边的对齐中间量缓存在这里，
        # 训练循环可以直接读取，避免重复前向。
        self.latest_alignment_cache = {}
        # 如果某条边的开关未单独设置，就默认沿用全局 use_alignment。
        self.align_feed_to_stage1 = use_alignment if align_feed_to_stage1 is None else bool(align_feed_to_stage1)
        self.align_stage1_to_stage2 = use_alignment if align_stage1_to_stage2 is None else bool(align_stage1_to_stage2)
        self.align_stage2_to_stage3 = use_alignment if align_stage2_to_stage3 is None else bool(align_stage2_to_stage3)
        enc_type = encoder_type.lower()

        def make_encoder(d_in: int):
            # 四个工段共享同一“编码范式”，但各自输入维度 d_in 可以不同。
            # 这样既能保持模型结构对称，也能适配不同工段拥有不同数量的原始特征。
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

        # 四个工段独立编码。命名里的 0/1/2/3 对应：
        # 0 -> feed, 1 -> stage1, 2 -> stage2, 3 -> stage3
        self.enc_feed = make_encoder(group_dims["feed"])
        self.enc_s1   = make_encoder(group_dims["stage1"])
        self.enc_s2   = make_encoder(group_dims["stage2"])
        self.enc_s3   = make_encoder(group_dims["stage3"])

        def make_alignment(enabled: bool):
            # 对齐模块按“边”独立创建：
            # - feed -> stage1
            # - stage1 -> stage2
            # - stage2 -> stage3
            #
            # 这样做的好处是每条边可以单独关闭，便于做局部消融实验，
            # 例如只观察 stage2 -> stage3 的滞后建模到底有没有帮助。
            if enabled:
                return DelayAlignment(
                    hidden_dim,
                    attn_dim,
                    L_max,
                    lag_emb,
                    tau=align_tau,
                    use_lag_bias=use_lag_bias,
                    lag_head_mode=lag_head_mode,
                )
            return NoDelayAlignment(L_max)

        self.align_0_1 = make_alignment(self.align_feed_to_stage1)
        self.align_1_2 = make_alignment(self.align_stage1_to_stage2)
        self.align_2_3 = make_alignment(self.align_stage2_to_stage3)

        # 每条边都先做“对齐消息构造”，再做“门控残差融合”。
        self.fuse_s1 = GatedResidualFusion(hidden_dim, dropout)
        self.fuse_s2 = GatedResidualFusion(hidden_dim, dropout)
        self.fuse_s3 = GatedResidualFusion(hidden_dim, dropout)
        # 预测目标改为单点值，而不是长度为 H 的向量。
        self.head = nn.Linear(hidden_dim, 1)

    def _apply_alignment_dropout(self, msg_seq: torch.Tensor) -> torch.Tensor:
        """
        对齐消息 dropout。

        与常规逐元素 dropout 不同，这里是以“样本-时间步”为单位整体丢弃一条消息，
        即某个样本在某个时刻的整条上游消息向量要么全部保留，要么全部置零。

        这么做的直觉是：
        - 更贴近“这一时刻上游对齐信息不可靠/缺失”的场景；
        - 避免只随机打掉部分维度，导致消息语义被碎片化污染；
        - 迫使模型在必要时回退到本工段自身编码，不要过度依赖上游消息。
        """
        if (not self.training) or self.align_dropout <= 0.0:
            return msg_seq
        # keep_mask 的最后一维为 1，意味着同一样本、同一时刻的全部隐藏维同步保留/丢弃。
        keep_mask = (
            torch.rand(*msg_seq.shape[:-1], 1, device=msg_seq.device) >= self.align_dropout
        ).to(msg_seq.dtype)
        return msg_seq * keep_mask

    def forward(self, X_seq: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        DIMF 的主前向过程。

        参数:
            X_seq:
                一个字典，至少包含以下键：
                - "feed"
                - "stage1"
                - "stage2"
                - "stage3"
                每个值都是形状 [B, L, D_in^(s)] 的张量。

        返回:
            y_hat:
                [B]，每个样本对应一个对未来固定提前量 H 的单点预测。
            alignment_pi:
                一个字典，保存三条工段边的 lag 概率分布：
                - feed_to_stage1: [B, L, K]
                - stage1_to_stage2: [B, L, K]
                - stage2_to_stage3: [B, L, K]
        """
        # 1. 各工段先独立编码，得到整段序列表征 E*_seq 和末端表征 E*_last。
        #    这里保留 *_last 是为了维持统一接口；当前预测路径真正用到的是序列表征，
        #    因为对齐和融合都发生在整个窗口的逐时间步层面。
        E0_seq, E0_last = self.enc_feed(X_seq["feed"])
        E1_seq, E1_last = self.enc_s1(X_seq["stage1"])
        E2_seq, E2_last = self.enc_s2(X_seq["stage2"])
        E3_seq, E3_last = self.enc_s3(X_seq["stage3"])
        del E0_last, E1_last, E2_last, E3_last

        # 2. feed -> stage1：
        #    根据 stage1 每个时刻的表示，去上游 feed 历史里搜索最可能的滞后位置，
        #    得到一条“对齐消息序列” m1_seq。随后将其与 stage1 自身编码 E1_seq 融合，
        #    形成带上游上下文的 stage1 新表示 u1_seq。
        m1_seq, pi01, raw1_seq = self.align_0_1.forward_seq(E1_seq, E0_seq)   # feed -> stage1
        m1_seq = self._apply_alignment_dropout(m1_seq)
        u1_seq = self.fuse_s1(E1_seq, m1_seq)

        # 3. stage1 -> stage2：
        #    这里上游输入不是原始 E1_seq，而是已经融合后的 u1_seq。
        #    这意味着“feed 对 stage1 的影响”也能继续通过 stage1 传递给 stage2，
        #    从而形成真正的逐级传播，而不是每一层都只看本层原始编码。
        m2_seq, pi12, raw2_seq = self.align_1_2.forward_seq(E2_seq, u1_seq)   # stage1 -> stage2
        m2_seq = self._apply_alignment_dropout(m2_seq)
        u2_seq = self.fuse_s2(E2_seq, m2_seq)

        # 4. stage2 -> stage3：
        #    同理，stage3 接收到的是“已经吸收了更上游信息”的 u2_seq。
        #    最终预测只读取窗口末端时刻的融合表示 u3，表示“在当前观测窗口结束时，
        #    stage3 对未来 H 步目标值的摘要认识”。
        m3_seq, pi23, raw3_seq = self.align_2_3.forward_seq(E3_seq, u2_seq)   # stage2 -> stage3
        m3_seq = self._apply_alignment_dropout(m3_seq)
        u3_seq = self.fuse_s3(E3_seq, m3_seq)
        u3 = u3_seq[:, -1, :]

        # 第 6/8 点修改：缓存每条边的对齐分布、未投影上游对齐表示、下游编码表示，
        # 供后续一致性损失直接读取。
        #
        # 注意这里缓存的是“未经过 W_v 的原始对齐上游表征” raw*_seq，
        # 因为一致性损失想比较的是语义空间中的表示是否对齐，
        # 而不是比较已经为消息传递目的变换过的投影向量。
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

        # 5. 单点预测头只读取最后一个时间步，对应监督标签 y[t+H]。
        #    head 先输出 [B, 1]，再 squeeze 成 [B]，这样更方便和一维标签直接算损失。
        y_hat = self.head(u3).squeeze(-1)
        return y_hat, {"feed_to_stage1": pi01, "stage1_to_stage2": pi12, "stage2_to_stage3": pi23}

def entropy_loss(pi: torch.Tensor) -> torch.Tensor:
    """
    对齐分布的负熵正则。

    输入:
        pi: [B, K] 或 [B, L, K]，最后一维是 lag 概率分布。

    作用:
    - 当 pi 很平时，说明模型觉得多个 lag 都“差不多像”，对齐不确定；
    - 最小化这个负熵项，会鼓励分布更尖锐、更确定；
    - 这通常有助于学出更清晰的时延结构。
    """
    return -(pi * (pi.clamp(min=1e-12).log())).sum(dim=-1).mean()

def tv_loss(pi: torch.Tensor) -> torch.Tensor:
    """
    时间总变分（temporal variation）正则。

    对序列版对齐分布 pi[:, t, :] 来说，这个损失惩罚相邻时间步分布的剧烈变化。
    直觉上，如果真实工艺延迟不会在相邻采样点之间瞬间大幅跳动，
    那么这个约束可以让 lag 轨迹更加平滑稳定。
    """
    if pi.dim() < 3 or pi.shape[1] < 2:
        return pi.new_tensor(0.0)
    return (pi[:, 1:, :] - pi[:, :-1, :]).abs().mean()


def symmetric_info_nce_loss(
    downstream_repr: torch.Tensor,
    upstream_repr: torch.Tensor,
    temperature: float = 0.2,
) -> torch.Tensor:
    """
    对称 InfoNCE 一致性损失。

    这里把“同一个样本在当前边上的下游表示”和“该样本对齐得到的上游表示”
    视作正样本对；同一 batch 中其他样本则充当负样本。

    这里使用窗口末端当前时刻 t 的表示：
      - downstream_repr: e_t^(s)，形状 [B, D]
      - upstream_repr:   \\tilde e_t^(s-1)，形状 [B, D]

    为什么做成对称形式：
    - 只做 downstream -> upstream 时，优化可能偏向单侧表示空间；
    - 加上 upstream -> downstream 后，两个方向都要把正确配对排到最高，
      一致性约束更均衡。
    """
    if temperature <= 0:
        raise ValueError("InfoNCE temperature must be positive")
    if downstream_repr.ndim != 2 or upstream_repr.ndim != 2:
        raise ValueError("InfoNCE inputs must be rank-2 tensors shaped as [B, D]")
    if downstream_repr.shape != upstream_repr.shape:
        raise ValueError("Downstream/upstream representations for InfoNCE must share the same shape")

    # 先做 L2 归一化，让相似度比较聚焦于“方向”而不是“向量模长”。
    z_down = F.normalize(downstream_repr, dim=-1)
    z_up = F.normalize(upstream_repr, dim=-1)
    # 每个样本与 batch 内所有样本做相似度比较，对角线位置是正样本。
    logits = torch.matmul(z_down, z_up.transpose(0, 1)) / temperature
    labels = torch.arange(z_down.shape[0], device=z_down.device)
    # 对称写法同时优化 down->up 和 up->down 两个方向，避免一侧主导。
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.transpose(0, 1), labels))


def alignment_consistency_loss(
    alignment_cache: Dict[str, Dict[str, torch.Tensor]],
    temperature: float = 0.2,
) -> torch.Tensor:
    """
    基于 latest_alignment_cache 计算跨工段对齐一致性损失。

    逐条工段边计算并求和：
      L_align = sum_s InfoNCE(e_t^(s), \\tilde e_t^(s-1))

    其中：
    - e_t^(s) 表示下游工段在监督时刻 t 的表示；
    - \\tilde e_t^(s-1) 表示按 lag 分布聚合得到的上游对齐表示。

    当前训练样本定义在窗口末端时刻 t，因此这里统一取每个样本最后一个时间步。
    如果某条边没有启用对齐，则直接跳过，不参与这项损失。
    """
    if not alignment_cache:
        raise ValueError("alignment_cache is empty; run model forward before computing L_align")

    total = None
    for edge_name, edge_cache in alignment_cache.items():
        # 某条边被显式关闭时，不把它纳入一致性损失。
        if not bool(edge_cache.get("active", True)):
            continue
        if "downstream_seq" not in edge_cache or "aligned_upstream_raw" not in edge_cache:
            raise ValueError(f"Alignment cache for {edge_name} is missing required tensors")
        # 这里约定监督时刻就是窗口末端，因此统一取最后一个时间步。
        downstream_last = edge_cache["downstream_seq"][:, -1, :]
        upstream_last = edge_cache["aligned_upstream_raw"][:, -1, :]
        edge_loss = symmetric_info_nce_loss(downstream_last, upstream_last, temperature=temperature)
        total = edge_loss if total is None else total + edge_loss
    if total is None:
        # 如果所有边都没启用，返回与设备/类型一致的零张量，方便训练脚本里直接相加，
        # 不需要专门写 if 分支判断 “当前是否存在可用的一致性损失”。
        first_edge = next(iter(alignment_cache.values()))
        reference = first_edge["downstream_seq"]
        return reference.new_tensor(0.0)
    return total
