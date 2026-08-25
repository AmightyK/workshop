"""Synchronized video-clip and ground-truth-pose dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .clip_sampler import ClipSpec, UniformClipSampler, VideoInfo
from .pose_io import Pose, PoseSeries


@dataclass(frozen=True)
class ClipItem:
    """One deterministic localization sample."""

    id: str
    timestamp: float
    frames: np.ndarray
    pose: Pose
    source_pose_time_error: float
    frame_timestamps: tuple[float, ...]
    ground_truth_translation_m: float


@dataclass(frozen=True)
class VideoClipItem:
    """One video-only query clip with no ground-truth dependency."""

    id: str
    timestamp: float
    frames: np.ndarray
    frame_timestamps: tuple[float, ...]


def _read_video_info(path: Path) -> VideoInfo:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {path}")
    info = VideoInfo(
        fps=float(capture.get(cv2.CAP_PROP_FPS)),
        frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    capture.release()
    return info


def _read_clip_frames(path: Path, clip: ClipSpec) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {path}")
    frames: list[np.ndarray] = []
    try:
        for frame_index in clip.frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"failed to decode frame {frame_index} from {path}")
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    return np.stack(frames, axis=0)


class VideoClipDataset:
    """Video-only query path used when online ground truth is unavailable."""

    def __init__(
        self,
        run_dir: str | Path,
        *,
        clip_duration_sec: float = 2.0,
        num_sampled_frames: int = 16,
        stride_sec: float = 0.5,
        video_time_offset_sec: float = 0.0,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.video_path = self.run_dir / "video.mp4"
        if not self.video_path.is_file():
            raise FileNotFoundError(self.video_path)
        self.video_info = _read_video_info(self.video_path)
        self.sampler = UniformClipSampler(
            clip_duration_sec=clip_duration_sec,
            num_sampled_frames=num_sampled_frames,
            stride_sec=stride_sec,
        )
        self.clips = self.sampler.build(
            self.video_info,
            time_offset_sec=video_time_offset_sec,
        )
        if not self.clips:
            raise ValueError("video does not contain a complete clip")

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, index: int) -> VideoClipItem:
        clip = self.clips[index]
        return VideoClipItem(
            id=clip.id,
            timestamp=clip.center_timestamp,
            frames=_read_clip_frames(self.video_path, clip),
            frame_timestamps=clip.frame_timestamps,
        )


class VideoPoseDataset:
    """Load ``video.mp4`` and ``poses.csv`` from one mapping/query run."""

    def __init__(
        self,
        run_dir: str | Path,
        *,
        clip_duration_sec: float = 2.0,
        num_sampled_frames: int = 16,
        stride_sec: float = 0.5,
        pose_method: str = "interpolate",
        pose_tolerance_sec: float = 0.2,
        video_time_offset_sec: float = 0.0,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.video_path = self.run_dir / "video.mp4"
        self.pose_path = self.run_dir / "poses.csv"
        if not self.video_path.is_file():
            raise FileNotFoundError(self.video_path)
        if not self.pose_path.is_file():
            raise FileNotFoundError(self.pose_path)
        self.poses = PoseSeries.from_csv(self.pose_path)
        self.pose_method = pose_method
        self.pose_tolerance_sec = float(pose_tolerance_sec)

        self.video_info = _read_video_info(self.video_path)
        self.sampler = UniformClipSampler(
            clip_duration_sec=clip_duration_sec,
            num_sampled_frames=num_sampled_frames,
            stride_sec=stride_sec,
        )
        self.clips = self.sampler.build(
            self.video_info,
            valid_start_time=self.poses.start_time,
            valid_end_time=self.poses.end_time,
            time_offset_sec=video_time_offset_sec,
        )
        if not self.clips:
            raise ValueError("video and pose ranges do not contain a complete clip")

        # Validate synchronization eagerly instead of failing halfway through
        # an expensive embedding extraction job.
        for clip in self.clips:
            self.poses.sample(
                clip.center_timestamp,
                method=self.pose_method,
                tolerance_sec=self.pose_tolerance_sec,
            )

    def __len__(self) -> int:
        return len(self.clips)

    def _read_frames(self, clip: ClipSpec) -> np.ndarray:
        return _read_clip_frames(self.video_path, clip)

    def __getitem__(self, index: int) -> ClipItem:
        clip = self.clips[index]
        pose_sample = self.poses.sample(
            clip.center_timestamp,
            method=self.pose_method,
            tolerance_sec=self.pose_tolerance_sec,
        )
        first_pose = self.poses.sample(
            clip.frame_timestamps[0],
            method=self.pose_method,
            tolerance_sec=self.pose_tolerance_sec,
        ).pose
        last_pose = self.poses.sample(
            clip.frame_timestamps[-1],
            method=self.pose_method,
            tolerance_sec=self.pose_tolerance_sec,
        ).pose
        translation = float(
            np.hypot(last_pose.x - first_pose.x, last_pose.y - first_pose.y)
        )
        return ClipItem(
            id=clip.id,
            timestamp=clip.center_timestamp,
            frames=self._read_frames(clip),
            pose=pose_sample.pose,
            source_pose_time_error=pose_sample.source_time_error,
            frame_timestamps=clip.frame_timestamps,
            ground_truth_translation_m=translation,
        )
