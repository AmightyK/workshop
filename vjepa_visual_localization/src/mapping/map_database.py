"""Numpy-backed global visual map database."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class VisualMap:
    """Memory-resident global embeddings with synchronized poses."""

    global_embeddings: np.ndarray
    poses: np.ndarray
    timestamps: np.ndarray
    ids: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.global_embeddings = np.asarray(self.global_embeddings)
        self.poses = np.asarray(self.poses)
        self.timestamps = np.asarray(self.timestamps)
        self.ids = np.asarray(self.ids)
        count = len(self.global_embeddings)
        if self.global_embeddings.ndim != 2:
            raise ValueError("global_embeddings must have shape [num_entries,D]")
        if self.poses.shape != (count, 4):
            raise ValueError(f"poses must have shape [{count},4]")
        if self.timestamps.shape != (count,) or self.ids.shape != (count,):
            raise ValueError("timestamps and ids must have one value per map entry")
        if count == 0:
            raise ValueError("visual map is empty")
        if not np.isfinite(self.global_embeddings).all() or not np.isfinite(self.poses).all():
            raise ValueError("visual map contains NaN or Inf")

    def save(self, directory: str | Path) -> None:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        np.save(target / "global_embeddings.npy", self.global_embeddings.astype(np.float16))
        np.save(target / "poses.npy", self.poses.astype(np.float64))
        np.save(target / "timestamps.npy", self.timestamps.astype(np.float64))
        np.save(target / "ids.npy", self.ids)
        metadata = dict(self.metadata)
        metadata.update(
            {
                "num_entries": len(self.global_embeddings),
                "embedding_dimension": self.global_embeddings.shape[1],
                "embedding_dtype": "float16",
                "pose_columns": ["x", "y", "z", "yaw"],
            }
        )
        (target / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: str | Path) -> "VisualMap":
        source = Path(directory)
        metadata_path = source / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return cls(
            global_embeddings=np.load(source / "global_embeddings.npy").astype(np.float32),
            poses=np.load(source / "poses.npy"),
            timestamps=np.load(source / "timestamps.npy"),
            ids=np.load(source / "ids.npy", allow_pickle=False),
            metadata=metadata,
        )
