from __future__ import annotations

import numpy as np
import pytest

from scripts.run_live_localization import validate_map_video_profile
from src.mapping.map_database import VisualMap


def make_map(video: dict) -> VisualMap:
    return VisualMap(
        global_embeddings=np.ones((1, 4), dtype=np.float32),
        poses=np.zeros((1, 4), dtype=np.float64),
        timestamps=np.zeros(1, dtype=np.float64),
        ids=np.asarray(["clip_000000"]),
        metadata={"config": {"video": video}},
    )


def test_live_profile_accepts_matching_map_clip() -> None:
    video = {
        "clip_duration_sec": 1.0,
        "num_sampled_frames": 4,
        "sampling": "uniform",
        "stride_sec": 0.5,
    }

    validate_map_video_profile(make_map(video), dict(video))


def test_live_profile_rejects_old_map_clip() -> None:
    old_video = {
        "clip_duration_sec": 2.0,
        "num_sampled_frames": 16,
        "sampling": "uniform",
    }
    live_video = {
        "clip_duration_sec": 1.0,
        "num_sampled_frames": 4,
        "sampling": "uniform",
    }

    with pytest.raises(RuntimeError, match="map/query video profiles differ"):
        validate_map_video_profile(make_map(old_video), live_video)
