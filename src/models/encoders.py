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
        self.gru = nn.GRU(
            input_size=d_in,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor):
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
        self.lstm = nn.LSTM(
            input_size=d_in,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor):
        seq_out, _ = self.lstm(x)
        last = seq_out[:, -1, :]
        return seq_out, last

class TransformerEncoder(nn.Module):
    """
    输入:  [B, L, D_in]
    输出:
      seq_out: [B, L, D_h]
      last:    [B, D_h]  (窗口末端表示)
    """
    def __init__(self, d_in: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.1,
                 nhead: int = 4, ff_dim: int = None, max_len: int = 512):
        super().__init__()
        self.in_proj = nn.Linear(d_in, hidden_dim)
        self.pos_emb = nn.Embedding(max_len, hidden_dim)
        if ff_dim is None:
            ff_dim = hidden_dim * 4
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

    def forward(self, x: torch.Tensor):
        B, L, _ = x.shape
        if L > self.max_len:
            raise ValueError(f"Sequence length {L} exceeds max_len {self.max_len}")
        pos = torch.arange(L, device=x.device)
        h = self.in_proj(x) + self.pos_emb(pos)[None, :, :]
        seq_out = self.encoder(h)
        seq_out = self.norm(seq_out)
        last = seq_out[:, -1, :]
        return seq_out, last
