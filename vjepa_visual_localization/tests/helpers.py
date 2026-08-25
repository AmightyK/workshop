from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from src.models.pooling import l2_normalize
from src.models.vjepa_encoder import EncoderOutput


class MeanColorEncoder:
    """Fast deterministic encoder used only to exercise pipeline mechanics."""

    def encode_video(self, video: np.ndarray) -> EncoderOutput:
        array = np.asarray(video)
        if array.ndim == 4:
            array = array[None]
        embedding = array.astype(np.float32).mean(axis=(1, 2, 3))
        embedding = l2_normalize(embedding)
        local = np.repeat(embedding[:, None, :], 4, axis=1)
        return EncoderOutput(embedding, local, (4,))

    def describe(self) -> dict[str, str]:
        return {"backend": "test_mean_color"}


def make_run(path: Path, colors: list[tuple[int, int, int]], fps: float = 4.0) -> Path:
    path.mkdir(parents=True)
    height, width = 48, 64
    writer = cv2.VideoWriter(
        str(path / "video.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV video writer is unavailable")
    for color in colors:
        # Input colors are RGB; OpenCV writes BGR.
        frame = np.full((height, width, 3), color[::-1], dtype=np.uint8)
        writer.write(frame)
    writer.release()
    with (path / "poses.csv").open("w", newline="", encoding="utf-8") as stream:
        writer_csv = csv.writer(stream)
        writer_csv.writerow(["timestamp", "x", "y", "z", "yaw"])
        duration = len(colors) / fps
        timestamp = 0.0
        while timestamp <= duration + 1e-9:
            writer_csv.writerow([timestamp, timestamp, 2.0 * timestamp, 0.0, 0.1 * timestamp])
            timestamp += 0.25
    return path
