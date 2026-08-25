"""Brute-force cosine retrieval baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.mapping.map_database import VisualMap
from src.models.pooling import l2_normalize


@dataclass(frozen=True)
class RetrievalResult:
    indices: np.ndarray
    ids: np.ndarray
    scores: np.ndarray
    poses: np.ndarray
    timestamps: np.ndarray


class GlobalRetriever:
    """Retrieve map clips using normalized dot-product cosine similarity."""

    def __init__(self, visual_map: VisualMap) -> None:
        self.visual_map = visual_map
        self.embeddings = l2_normalize(visual_map.global_embeddings)

    def query(self, embedding: np.ndarray, *, top_k: int = 20) -> RetrievalResult:
        query = np.asarray(embedding, dtype=np.float32)
        if query.ndim == 2 and query.shape[0] == 1:
            query = query[0]
        if query.ndim != 1 or query.shape[0] != self.embeddings.shape[1]:
            raise ValueError(
                f"query embedding must have shape [{self.embeddings.shape[1]}], got {query.shape}"
            )
        count = min(max(int(top_k), 1), len(self.embeddings))
        scores = self.embeddings @ l2_normalize(query)
        # Stable ordering makes ties deterministic and testable.
        indices = np.argsort(-scores, kind="stable")[:count]
        return RetrievalResult(
            indices=indices,
            ids=self.visual_map.ids[indices],
            scores=scores[indices],
            poses=self.visual_map.poses[indices],
            timestamps=self.visual_map.timestamps[indices],
        )
