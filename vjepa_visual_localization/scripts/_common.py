"""Shared CLI construction helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import VideoClipDataset, VideoPoseDataset
from src.models.vjepa_encoder import VJEPAEncoder
from src.utils.config import section


def dataset_from_config(run_dir: str | Path, config: dict[str, Any]) -> VideoPoseDataset:
    video = section(config, "video")
    synchronization = config.get("synchronization", {})
    return VideoPoseDataset(
        run_dir,
        clip_duration_sec=float(video.get("clip_duration_sec", 2.0)),
        num_sampled_frames=int(video.get("num_sampled_frames", 16)),
        stride_sec=float(video.get("stride_sec", 0.5)),
        pose_method=str(synchronization.get("pose_method", "interpolate")),
        pose_tolerance_sec=float(synchronization.get("pose_tolerance_sec", 0.2)),
        video_time_offset_sec=float(synchronization.get("video_time_offset_sec", 0.0)),
    )


def encoder_from_config(config: dict[str, Any]) -> VJEPAEncoder:
    model = section(config, "model")
    return VJEPAEncoder(
        checkpoint=str(model.get("checkpoint", "facebook/vjepa2-vitl-fpc64-256")),
        device=str(model.get("device", "cuda")),
        dtype=str(model.get("dtype", "float16")),
        return_local_tokens=bool(model.get("return_local_tokens", True)),
        normalize_embeddings=bool(model.get("normalize_embeddings", True)),
    )


def video_dataset_from_config(run_dir: str | Path, config: dict[str, Any]) -> VideoClipDataset:
    video = section(config, "video")
    synchronization = config.get("synchronization", {})
    return VideoClipDataset(
        run_dir,
        clip_duration_sec=float(video.get("clip_duration_sec", 2.0)),
        num_sampled_frames=int(video.get("num_sampled_frames", 16)),
        stride_sec=float(video.get("stride_sec", 0.5)),
        video_time_offset_sec=float(synchronization.get("video_time_offset_sec", 0.0)),
    )
