"""Embedding pooling and normalization."""

from __future__ import annotations

import numpy as np


def l2_normalize(values: np.ndarray, axis: int = -1, epsilon: float = 1e-12) -> np.ndarray:
    """L2-normalize without producing NaNs for zero vectors."""

    array = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(array, axis=axis, keepdims=True)
    return array / np.maximum(norms, epsilon)


def mean_pool_tokens(local_tokens: np.ndarray) -> np.ndarray:
    """Mean-pool ``[B,N,D]`` local tokens into ``[B,D]`` vectors."""

    tokens = np.asarray(local_tokens)
    if tokens.ndim != 3:
        raise ValueError(f"expected [B,N,D] tokens, got shape {tokens.shape}")
    return tokens.mean(axis=1, dtype=np.float32)
