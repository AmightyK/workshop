from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from nav_pose_math import Pose2D, blend_pose, compose, map_to_odom, wrap_angle


class NavPoseMathTests(unittest.TestCase):
    def test_map_to_odom_reconstructs_visual_map_pose(self) -> None:
        map_base = Pose2D(10.0, 5.0, math.pi / 2.0)
        odom_base = Pose2D(2.0, 0.0, 0.0)
        transform = map_to_odom(map_base, odom_base)
        reconstructed = compose(transform, odom_base)
        self.assertAlmostEqual(reconstructed.x, map_base.x)
        self.assertAlmostEqual(reconstructed.y, map_base.y)
        self.assertAlmostEqual(reconstructed.yaw, map_base.yaw)

    def test_blend_uses_shortest_wrapped_yaw(self) -> None:
        current = Pose2D(0.0, 0.0, math.radians(179.0))
        target = Pose2D(2.0, 4.0, math.radians(-179.0))
        blended = blend_pose(current, target, 0.5)
        self.assertAlmostEqual(blended.x, 1.0)
        self.assertAlmostEqual(blended.y, 2.0)
        self.assertAlmostEqual(abs(wrap_angle(blended.yaw)), math.pi)


if __name__ == "__main__":
    unittest.main()
