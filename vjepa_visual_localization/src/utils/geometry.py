"""Pose geometry functions shared by localization and evaluation."""

from __future__ import annotations

import numpy as np


def wrap_angles(values: np.ndarray | float) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return (array + np.pi) % (2.0 * np.pi) - np.pi


def position_distances(reference: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Return planar Euclidean distances from one pose to many poses."""

    reference = np.asarray(reference, dtype=np.float64)
    candidates = np.asarray(candidates, dtype=np.float64)
    return np.linalg.norm(candidates[:, :2] - reference[:2], axis=1)
