from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

@dataclass
class WindowSpec:
    # L: 历史窗口长度；H: 预测提前量。
    L: int
    H: int

class MultistageWindowDataset(Dataset):
    """
    对每个时刻 t 构造样本：
      输入：各组 X[t-L+1 : t+1]  -> [L, d_g]
      标签：固定提前量的单点目标 y[t+H]
    """
    def __init__(
        self,
        X_groups: Dict[str, np.ndarray],
        y: np.ndarray,
        spec: WindowSpec,
        indices: Optional[np.ndarray] = None,
        extra_targets: Optional[Dict[str, np.ndarray]] = None,
    ):
        # X_groups 中每个数组都是 [T, d_group]，且不同工段共享同一时间轴长度 T。
        self.X_groups = X_groups
        self.y = y
        self.L = spec.L
        self.H = spec.H
        self.T = next(iter(X_groups.values())).shape[0]
        # extra_targets 用来携带与主回归目标并行的辅助监督，例如 lag_gt。
        self.extra_targets = extra_targets or {}

        # 样本中心时刻 t 的合法范围：
        # - 需要至少有 L 个历史点，所以 t >= L-1
        # - 需要还能看到 y[t+H]，所以 t <= T-H-1
        self.t_min = self.L - 1
        self.t_max = self.T - self.H - 1

        for g, X in X_groups.items():
            assert X.shape[0] == self.T
        for name, values in self.extra_targets.items():
            if values.shape[0] != self.T:
                raise ValueError(f"extra target '{name}' has length {values.shape[0]}, expected {self.T}")

        if indices is None:
            self.indices = np.arange(self.t_min, self.t_max + 1)
        else:
            # 如果上游已经算好了合法样本位置，就直接沿用，避免这里重复做时间连续性判断。
            idx = np.asarray(indices, dtype=np.int64)
            if idx.ndim != 1:
                raise ValueError("indices must be a 1D array")
            if len(idx) == 0:
                raise ValueError("indices must not be empty")
            if np.any(idx < self.t_min) or np.any(idx > self.t_max):
                raise ValueError("indices contain out-of-range sample positions")
            self.indices = idx

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        t = int(self.indices[idx])
        # 对每个工段都截取 [t-L+1, ..., t] 这段历史窗口，形状统一成 [L, d_g]。
        X_seq = {g: torch.from_numpy(X[t-self.L+1:t+1]).float() for g, X in self.X_groups.items()}
        for name, values in self.extra_targets.items():
            # 辅助监督目标也按“样本中心时刻 t”对齐取值。
            value = values[t]
            if np.issubdtype(values.dtype, np.integer):
                X_seq[name] = torch.tensor(int(value), dtype=torch.long)
            else:
                X_seq[name] = torch.tensor(float(value), dtype=torch.float32)
        # 第 1 点修改：H 表示“提前量步数”，这里只取单个监督点 y_{t+H}。
        y_target = torch.tensor(self.y[t + self.H], dtype=torch.float32)
        return X_seq, y_target
