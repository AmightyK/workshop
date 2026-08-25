"""Build a global V-JEPA visual map from a synchronized mapping run."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np

from src.data.dataset import VideoPoseDataset
from src.mapping.map_database import VisualMap
from src.models.vjepa_encoder import EncoderOutput


class Encoder(Protocol):
    def encode_video(self, video: np.ndarray) -> EncoderOutput: ...


class GlobalMapBuilder:
    """Extract normalized global embeddings without introducing FAISS."""

    def __init__(self, encoder: Encoder, *, batch_size: int = 1) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.encoder = encoder
        self.batch_size = batch_size

    def build(
        self,
        dataset: VideoPoseDataset,
        *,
        output_dir: str | Path | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> VisualMap:
        embeddings: list[np.ndarray] = []
        poses: list[np.ndarray] = []
        timestamps: list[float] = []
        ids: list[str] = []
        for start in range(0, len(dataset), self.batch_size):
            items = [dataset[index] for index in range(start, min(start + self.batch_size, len(dataset)))]
            videos = np.stack([item.frames for item in items])
            output = self.encoder.encode_video(videos)
            if output.global_embedding.shape[0] != len(items):
                raise RuntimeError("encoder batch size does not match input batch")
            embeddings.extend(output.global_embedding)
            poses.extend(item.pose.as_array() for item in items)
            timestamps.extend(item.timestamp for item in items)
            ids.extend(item.id for item in items)

        map_metadata = dict(metadata or {})
        map_metadata.setdefault("mapping_run", str(dataset.run_dir.resolve()))
        if hasattr(self.encoder, "describe"):
            map_metadata["encoder"] = self.encoder.describe()  # type: ignore[attr-defined]
        visual_map = VisualMap(
            global_embeddings=np.asarray(embeddings, dtype=np.float32),
            poses=np.asarray(poses, dtype=np.float64),
            timestamps=np.asarray(timestamps, dtype=np.float64),
            ids=np.asarray(ids, dtype=str),
            metadata=map_metadata,
        )
        if output_dir is not None:
            visual_map.save(output_dir)
        return visual_map
