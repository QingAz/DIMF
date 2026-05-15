from __future__ import annotations

from typing import Optional

import numpy as np


def _as_numpy(values) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values)


def _normalize(pi: np.ndarray) -> np.ndarray:
    arr = _as_numpy(pi).astype(np.float64, copy=False)
    arr = np.clip(arr, 0.0, None)
    denom = np.clip(arr.sum(axis=-1, keepdims=True), 1e-12, None)
    return arr / denom


def _transition_scores(
    n_states: int,
    smooth_lambda: float,
    switch_penalty: float,
    pos_to_zero_penalty: float,
) -> np.ndarray:
    lag_axis = np.arange(n_states, dtype=np.float64)
    trans = -float(smooth_lambda) * np.abs(lag_axis[:, None] - lag_axis[None, :])
    trans[(lag_axis[:, None] == 0) & (lag_axis[None, :] > 0)] -= float(switch_penalty)
    trans[(lag_axis[:, None] > 0) & (lag_axis[None, :] == 0)] -= float(pos_to_zero_penalty)
    return trans


def _decode_one_segment(
    log_emit: np.ndarray,
    smooth_lambda: float,
    switch_penalty: float,
    pos_to_zero_penalty: float,
) -> np.ndarray:
    n_steps, n_states = log_emit.shape
    if n_steps == 0:
        return np.zeros(0, dtype=np.int64)

    transition = _transition_scores(
        n_states,
        smooth_lambda=smooth_lambda,
        switch_penalty=switch_penalty,
        pos_to_zero_penalty=pos_to_zero_penalty,
    )

    dp = np.empty((n_steps, n_states), dtype=np.float64)
    back = np.zeros((n_steps, n_states), dtype=np.int64)
    dp[0] = log_emit[0]

    for t in range(1, n_steps):
        scores = dp[t - 1][:, None] + transition
        back[t] = np.argmax(scores, axis=0)
        dp[t] = log_emit[t] + scores[back[t], np.arange(n_states)]

    path = np.empty(n_steps, dtype=np.int64)
    path[-1] = int(np.argmax(dp[-1]))
    for t in range(n_steps - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def viterbi_decode_lag(
    pi,
    segment_id: Optional[np.ndarray] = None,
    smooth_lambda: float = 0.8,
    switch_penalty: float = 1.5,
    pos_to_zero_penalty: float = 2.0,
) -> np.ndarray:
    """Decode lag states with segment-bounded Viterbi dynamic programming."""
    probs = _normalize(pi)
    if probs.ndim != 2:
        raise ValueError("pi must be shaped [T, K]")
    log_emit = np.log(np.clip(probs, 1e-12, None))
    n_samples = log_emit.shape[0]
    if segment_id is None:
        return _decode_one_segment(
            log_emit,
            smooth_lambda=smooth_lambda,
            switch_penalty=switch_penalty,
            pos_to_zero_penalty=pos_to_zero_penalty,
        )

    segment_id = _as_numpy(segment_id)
    if segment_id.shape[0] != n_samples:
        raise ValueError("segment_id length must match pi samples")

    path = np.empty(n_samples, dtype=np.int64)
    start = 0
    for end in range(1, n_samples + 1):
        if end == n_samples or segment_id[end] != segment_id[start]:
            path[start:end] = _decode_one_segment(
                log_emit[start:end],
                smooth_lambda=smooth_lambda,
                switch_penalty=switch_penalty,
                pos_to_zero_penalty=pos_to_zero_penalty,
            )
            start = end
    return path


def viterbi_decode_lag_path(
    pred_pi: np.ndarray,
    transition_penalty: float = 0.8,
    segment_ids: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Backward-compatible wrapper for the old single-penalty decoder."""
    return viterbi_decode_lag(
        pred_pi,
        segment_id=segment_ids,
        smooth_lambda=float(transition_penalty),
        switch_penalty=0.0,
        pos_to_zero_penalty=0.0,
    )


def path_to_onehot(path: np.ndarray, n_lags: int) -> np.ndarray:
    path = np.asarray(path, dtype=np.int64)
    onehot = np.zeros((path.shape[0], int(n_lags)), dtype=np.float32)
    valid = (path >= 0) & (path < int(n_lags))
    onehot[np.arange(path.shape[0])[valid], path[valid]] = 1.0
    return onehot
