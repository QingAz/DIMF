from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

@dataclass
class WindowSpec:
    L: int
    H: int

class MultistageWindowDataset(Dataset):
    """
    对每个时刻 t 构造样本：
      输入：各组 X[t-L+1 : t+1]  -> [L, d_g]
      标签：y[t+H]
    """
    def __init__(
        self,
        X_groups: Dict[str, np.ndarray],
        y: np.ndarray,
        spec: WindowSpec,
        indices: Optional[np.ndarray] = None,
    ):
        self.X_groups = X_groups
        self.y = y
        self.L = spec.L
        self.H = spec.H
        self.T = next(iter(X_groups.values())).shape[0]

        self.t_min = self.L - 1
        self.t_max = self.T - self.H - 1

        for g, X in X_groups.items():
            assert X.shape[0] == self.T

        if indices is None:
            self.indices = np.arange(self.t_min, self.t_max + 1)
        else:
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
        X_seq = {g: torch.from_numpy(X[t-self.L+1:t+1]).float() for g, X in self.X_groups.items()}
        y_target = torch.from_numpy(self.y[t + 1:t + self.H + 1]).float()
        return X_seq, y_target
