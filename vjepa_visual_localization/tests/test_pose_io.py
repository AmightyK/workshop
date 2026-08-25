from __future__ import annotations

import math

import numpy as np
import pytest

from src.data.pose_io import Pose, PoseSeries


def test_interpolates_xyz_and_wraps_yaw_across_pi() -> None:
    poses = PoseSeries(
        [
            Pose(0.0, 0.0, 0.0, 0.0, math.radians(179.0)),
            Pose(1.0, 2.0, 4.0, 0.0, math.radians(-179.0)),
        ]
    )
    sample = poses.sample(0.5, tolerance_sec=0.5)
    np.testing.assert_allclose(sample.pose.as_array()[:3], [1.0, 2.0, 0.0])
    assert abs(abs(sample.pose.yaw) - math.pi) < math.radians(0.1)
    assert sample.source_time_error == pytest.approx(0.5)


def test_rejects_unsynchronized_timestamp() -> None:
    poses = PoseSeries([Pose(0.0, 0, 0, 0, 0), Pose(1.0, 1, 0, 0, 0)])
    with pytest.raises(ValueError, match="no pose within"):
        poses.sample(0.5, tolerance_sec=0.1)
