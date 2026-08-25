"""Ground-truth-free global visual localization inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from src.localization.pose_estimator import PoseEstimator, PosePrediction
from src.models.vjepa_encoder import EncoderOutput
from src.retrieval.global_retriever import GlobalRetriever, RetrievalResult


class Encoder(Protocol):
    def encode_video(self, video: np.ndarray) -> EncoderOutput: ...


@dataclass(frozen=True)
class LocalizationResult:
    prediction: PosePrediction
    retrieval: RetrievalResult
    query_embedding: np.ndarray | None = None

    @property
    def confidence_margin(self) -> float:
        if len(self.retrieval.scores) < 2:
            return float(self.retrieval.scores[0])
        return float(self.retrieval.scores[0] - self.retrieval.scores[1])


class GlobalVisualLocalizer:
    """Estimate pose using only query pixels and a prebuilt visual map."""

    def __init__(
        self,
        encoder: Encoder,
        retriever: GlobalRetriever,
        *,
        top_k: int = 20,
        pose_method: str = "top1",
        weighted_alpha: float = 10.0,
        weighted_radius_m: float = 2.0,
    ) -> None:
        self.encoder = encoder
        self.retriever = retriever
        self.top_k = top_k
        self.pose_method = pose_method
        self.weighted_alpha = weighted_alpha
        self.weighted_radius_m = weighted_radius_m
        self.estimator = PoseEstimator()

    def localize(self, frames: np.ndarray) -> LocalizationResult:
        encoded = self.encoder.encode_video(frames)
        retrieval = self.retriever.query(encoded.global_embedding[0], top_k=self.top_k)
        if self.pose_method == "top1":
            prediction = self.estimator.predict_top1(retrieval)
        elif self.pose_method == "weighted":
            prediction = self.estimator.predict_weighted(
                retrieval,
                alpha=self.weighted_alpha,
                spatial_radius_m=self.weighted_radius_m,
            )
        else:
            raise ValueError(f"unsupported pose method: {self.pose_method}")
        return LocalizationResult(
            prediction,
            retrieval,
            np.asarray(encoded.global_embedding[0], dtype=np.float32).copy(),
        )
