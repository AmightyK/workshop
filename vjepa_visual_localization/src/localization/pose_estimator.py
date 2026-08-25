"""Top-1 and spatially gated weighted pose estimation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.retrieval.global_retriever import RetrievalResult
from src.utils.geometry import position_distances


@dataclass(frozen=True)
class PosePrediction:
    pose: np.ndarray
    source_id: str
    score: float
    method: str


class PoseEstimator:
    """Convert retrieval candidates into one world-frame pose."""

    def predict_top1(self, result: RetrievalResult) -> PosePrediction:
        if len(result.indices) == 0:
            raise ValueError("retrieval result is empty")
        return PosePrediction(
            pose=result.poses[0].copy(),
            source_id=str(result.ids[0]),
            score=float(result.scores[0]),
            method="top1",
        )

    def predict_weighted(
        self,
        result: RetrievalResult,
        *,
        alpha: float = 10.0,
        spatial_radius_m: float = 2.0,
    ) -> PosePrediction:
        """Average only candidates spatially connected to the top-1 match."""

        if len(result.indices) == 0:
            raise ValueError("retrieval result is empty")
        mask = position_distances(result.poses[0], result.poses) <= spatial_radius_m
        poses = result.poses[mask]
        scores = result.scores[mask]
        logits = alpha * (scores - scores.max())
        weights = np.exp(logits)
        weights /= weights.sum()
        xyz = np.sum(weights[:, None] * poses[:, :3], axis=0)
        yaw = np.arctan2(
            np.sum(weights * np.sin(poses[:, 3])),
            np.sum(weights * np.cos(poses[:, 3])),
        )
        return PosePrediction(
            pose=np.asarray([*xyz, yaw], dtype=np.float64),
            source_id=str(result.ids[0]),
            score=float(result.scores[0]),
            method="weighted_spatial_gate",
        )
