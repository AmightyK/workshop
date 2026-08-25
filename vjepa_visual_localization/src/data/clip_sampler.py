"""Deterministic uniform clip sampling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    frame_count: int

    @property
    def duration_sec(self) -> float:
        return self.frame_count / self.fps


@dataclass(frozen=True)
class ClipSpec:
    """Frame indices and timestamps for one sampled video clip."""

    id: str
    center_timestamp: float
    frame_indices: tuple[int, ...]
    frame_timestamps: tuple[float, ...]


class UniformClipSampler:
    """Generate fixed-duration clips at a fixed center-time stride."""

    def __init__(
        self,
        *,
        clip_duration_sec: float,
        num_sampled_frames: int,
        stride_sec: float,
    ) -> None:
        if clip_duration_sec <= 0.0 or stride_sec <= 0.0:
            raise ValueError("clip duration and stride must be positive")
        if num_sampled_frames <= 0:
            raise ValueError("num_sampled_frames must be positive")
        self.clip_duration_sec = float(clip_duration_sec)
        self.num_sampled_frames = int(num_sampled_frames)
        self.stride_sec = float(stride_sec)

    def build(
        self,
        video: VideoInfo,
        *,
        valid_start_time: float | None = None,
        valid_end_time: float | None = None,
        time_offset_sec: float = 0.0,
    ) -> list[ClipSpec]:
        """Build clip descriptors constrained by video and pose coverage."""

        if video.fps <= 0.0 or video.frame_count <= 0:
            raise ValueError("video metadata is invalid")
        half = self.clip_duration_sec / 2.0
        first = time_offset_sec + half
        last = time_offset_sec + video.duration_sec - half
        if valid_start_time is not None:
            first = max(first, valid_start_time)
        if valid_end_time is not None:
            last = min(last, valid_end_time)
        if last + 1e-9 < first:
            return []

        count = int(np.floor((last - first) / self.stride_sec + 1e-9)) + 1
        clips: list[ClipSpec] = []
        for clip_index in range(count):
            center = first + clip_index * self.stride_sec
            sample_times = np.linspace(
                center - half,
                center + half,
                self.num_sampled_frames,
                endpoint=False,
                dtype=np.float64,
            )
            video_times = sample_times - time_offset_sec
            frame_indices = np.rint(video_times * video.fps).astype(np.int64)
            frame_indices = np.clip(frame_indices, 0, video.frame_count - 1)
            actual_times = frame_indices.astype(np.float64) / video.fps + time_offset_sec
            clips.append(
                ClipSpec(
                    id=f"clip_{clip_index:06d}",
                    center_timestamp=float(center),
                    frame_indices=tuple(int(value) for value in frame_indices),
                    frame_timestamps=tuple(float(value) for value in actual_times),
                )
            )
        return clips
