from __future__ import annotations

import numpy as np

from src.data.dataset import VideoClipDataset, VideoPoseDataset
from tests.helpers import make_run


def test_dataset_returns_deterministic_synchronized_clips(tmp_path) -> None:
    colors = [(index * 12, 20, 200 - index * 8) for index in range(16)]
    run = make_run(tmp_path / "run", colors)
    dataset = VideoPoseDataset(
        run,
        clip_duration_sec=2.0,
        num_sampled_frames=4,
        stride_sec=0.5,
        pose_tolerance_sec=0.2,
    )
    first = dataset[0]
    repeated = dataset[0]
    assert first.id == "clip_000000"
    assert first.frames.shape == (4, 48, 64, 3)
    assert first.pose.timestamp == first.timestamp
    assert first.source_pose_time_error <= 0.2
    assert first.ground_truth_translation_m > 0.0
    np.testing.assert_array_equal(first.frames, repeated.frames)
    np.testing.assert_allclose(first.pose.as_array(), repeated.pose.as_array())


def test_video_only_query_does_not_require_pose_file(tmp_path) -> None:
    run = make_run(tmp_path / "query", [(20, 40, 80)] * 12)
    (run / "poses.csv").unlink()
    dataset = VideoClipDataset(
        run,
        clip_duration_sec=1.0,
        num_sampled_frames=4,
        stride_sec=0.5,
    )
    item = dataset[0]
    assert item.frames.shape == (4, 48, 64, 3)
    assert not hasattr(item, "pose")
