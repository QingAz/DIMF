import torch
import torch.nn as nn

class GRUEncoder(nn.Module):
    """
    输入:  [B, L, D_in]
    输出:
      seq_out: [B, L, D_h]
      last:    [B, D_h]  (窗口末端表示)
    """
    def __init__(self, d_in: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        # batch_first=True 后，输入输出都统一使用 [B, L, D] 排布。
        self.gru = nn.GRU(
            input_size=d_in,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor):
        # seq_out 保留整段历史，last 只取末端，方便预测头直接读取。
        seq_out, _ = self.gru(x)
        last = seq_out[:, -1, :]
        return seq_out, last

class LSTMEncoder(nn.Module):
    """
    输入:  [B, L, D_in]
    输出:
      seq_out: [B, L, D_h]
      last:    [B, D_h]  (窗口末端表示)
    """
    def __init__(self, d_in: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        # LSTM 版本接口与 GRU 保持一致，便于在配置里直接切换编码器。
        self.lstm = nn.LSTM(
            input_size=d_in,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor):
        # 这里只返回输出序列，不额外暴露 h_n/c_n，因为 DIMF 只需要时序表征。
        seq_out, _ = self.lstm(x)
        last = seq_out[:, -1, :]
        return seq_out, last

class TransformerEncoder(nn.Module):
    """
    输入:  [B, L, D_in]
    输出:
      seq_out: [B, L, D_h]
      last:    [B, D_h]  (窗口末端表示)

    第 5 点修改：
    这里使用因果 Transformer，位置 t 只能访问 [0, ..., t] 的历史信息，
    不能看到窗口内更靠后的未来步。
    """
    def __init__(self, d_in: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.1,
                 nhead: int = 4, ff_dim: int = None, max_len: int = 512):
        super().__init__()
        # 先把原始特征投到统一隐藏维，再叠加位置嵌入。
        self.in_proj = nn.Linear(d_in, hidden_dim)
        self.pos_emb = nn.Embedding(max_len, hidden_dim)
        if ff_dim is None:
            ff_dim = hidden_dim * 4
        # 这里用标准的 TransformerEncoderLayer，因果约束由 forward 里的 mask 提供。
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.max_len = max_len

    def _build_causal_mask(self, length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # 上三角（不含主对角线）位置置为 -inf，禁止当前位置访问未来 token。
        future_mask = torch.triu(
            torch.ones(length, length, device=device, dtype=torch.bool),
            diagonal=1,
        )
        causal_mask = torch.zeros(length, length, device=device, dtype=dtype)
        causal_mask = causal_mask.masked_fill(future_mask, float("-inf"))
        return causal_mask

    def forward(self, x: torch.Tensor):
        B, L, _ = x.shape
        if L > self.max_len:
            raise ValueError(f"Sequence length {L} exceeds max_len {self.max_len}")
        pos = torch.arange(L, device=x.device)
        # 同一 batch 内所有样本共享 [0..L-1] 的相对位置编号。
        h = self.in_proj(x) + self.pos_emb(pos)[None, :, :]
        # 第 5 点修改：显式加入因果掩码，确保 t 时刻表示不访问未来信息。
        causal_mask = self._build_causal_mask(L, x.device, h.dtype)
        seq_out = self.encoder(h, mask=causal_mask)
        seq_out = self.norm(seq_out)
        last = seq_out[:, -1, :]
        return seq_out, last
