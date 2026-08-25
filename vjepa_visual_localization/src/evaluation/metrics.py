"""Position, yaw and retrieval metrics required by the specification."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from src.utils.geometry import wrap_angles


POSITION_RADII_M = (0.5, 1.0, 2.0, 5.0)
RETRIEVAL_K = (1, 5, 20)


def position_error(ground_truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(ground_truth)[:2] - np.asarray(prediction)[:2]))


def yaw_error(ground_truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(abs(wrap_angles(float(ground_truth[3]) - float(prediction[3]))))


def summarize_localization(
    ground_truth: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float | int]:
    """Compute required position and yaw aggregate metrics."""

    ground_truth = np.asarray(ground_truth, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    if ground_truth.shape != predictions.shape or ground_truth.ndim != 2 or ground_truth.shape[1] != 4:
        raise ValueError("ground_truth and predictions must both have shape [N,4]")
    errors = np.linalg.norm(ground_truth[:, :2] - predictions[:, :2], axis=1)
    yaw_errors = np.abs(wrap_angles(ground_truth[:, 3] - predictions[:, 3]))
    report: dict[str, float | int] = {
        "count": int(len(errors)),
        "mean_position_error_m": float(np.mean(errors)),
        "median_position_error_m": float(np.median(errors)),
        "p95_position_error_m": float(np.percentile(errors, 95)),
        "mean_yaw_error_rad": float(np.mean(yaw_errors)),
        "median_yaw_error_rad": float(np.median(yaw_errors)),
    }
    for radius in POSITION_RADII_M:
        report[f"recall_at_{radius:g}m"] = float(np.mean(errors <= radius))
    return report


def retrieval_recall(
    ground_truth_poses: Iterable[np.ndarray],
    candidate_poses: Iterable[np.ndarray],
    *,
    radius_m: float = 1.0,
) -> dict[str, float]:
    """Report whether top-1/5/20 contains a pose within ``radius_m``."""

    successes = {k: [] for k in RETRIEVAL_K}
    for ground_truth, candidates in zip(ground_truth_poses, candidate_poses, strict=True):
        candidates = np.asarray(candidates, dtype=np.float64)
        distances = np.linalg.norm(candidates[:, :2] - np.asarray(ground_truth)[:2], axis=1)
        for k in RETRIEVAL_K:
            successes[k].append(bool(np.any(distances[: min(k, len(distances))] <= radius_m)))
    return {
        f"retrieval_recall_at_{k}_within_{radius_m:g}m": float(np.mean(values))
        for k, values in successes.items()
    }
