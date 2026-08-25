from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from latent_route import (
    LatentRoutePlanner,
    orient_route_tangents,
    prune_hard_corner_checkpoints,
)


class LatentRoutePlannerTests(unittest.TestCase):
    def make_planner(self, poses: list[list[float]]) -> LatentRoutePlanner:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        np.save(root / "poses.npy", np.asarray(poses, dtype=np.float64))
        return LatentRoutePlanner(root, max_target_error_m=0.4)

    def test_segment_never_uses_a_later_return_loop_for_first_target(self) -> None:
        planner = self.make_planner(
            [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0], [1, 0, 0, 3.14]]
        )
        segment = planner.segment_to([1, 0, 0, 0], spacing_m=0.1)
        self.assertEqual(segment.end_index, 1)

    def test_second_segment_only_moves_forward_from_previous_index(self) -> None:
        planner = self.make_planner(
            [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0], [1, 0, 0, 3.14]]
        )
        segment = planner.segment_to([1, 0, 0, 3.14], start_index=2)
        self.assertEqual(segment.end_index, 3)

    def test_unrecorded_target_is_rejected(self) -> None:
        planner = self.make_planner([[0, 0, 0, 0], [1, 0, 0, 0]])
        with self.assertRaisesRegex(RuntimeError, "outside the recorded latent corridor"):
            planner.segment_to([4, 0, 0, 0])

    def test_target_heading_must_have_been_recorded(self) -> None:
        planner = self.make_planner([[0, 0, 0, 0], [1, 0, 0, 0]])
        with self.assertRaisesRegex(RuntimeError, "yaw error"):
            planner.segment_to([1, 0, 0, 1.8])

    def test_semantic_three_value_pose_uses_third_value_as_yaw(self) -> None:
        planner = self.make_planner(
            [[0, 0, 0, 0], [1, 0, 0, 1.2], [1, 0, 0, 0]]
        )
        planner.max_target_yaw_error_rad = 0.2
        index, error = planner.nearest_forward_index([1, 0, 1.2])
        self.assertEqual(index, 1)
        self.assertAlmostEqual(error, 0.0)

    def test_first_visit_can_last_more_than_eight_clips(self) -> None:
        # The robot enters the target radius, pauses/rotates for many clips,
        # then reaches the exact pose before leaving. A much later return loop
        # must not replace that exact pose.
        poses = [[0, 0, 0, 0]]
        poses.extend([[0.7, 0, 0, 0]] * 12)
        poses.append([1.0, 0, 0, 0])
        poses.extend([[2.0, 0, 0, 0]] * 3)
        poses.append([1.0, 0, 0, 0])
        planner = self.make_planner(poses)

        index, error = planner.nearest_forward_index([1, 0, 0])

        self.assertEqual(index, 13)
        self.assertAlmostEqual(error, 0.0)

    def test_downsample_ignores_in_place_camera_rotation(self) -> None:
        planner = self.make_planner(
            [[0, 0, 0, 0], [0.1, 0, 0, 0], [0.2, 0, 0, 1.0], [0.3, 0, 0, 1.0]]
        )
        segment = planner.segment_to(
            [0.3, 0, 0, 1.0], spacing_m=1.0, yaw_step_rad=0.5
        )
        self.assertEqual(len(segment.poses), 2)
        np.testing.assert_allclose(segment.poses[-1], [0.3, 0, 0, 1.0])

    def test_downsample_keeps_geometric_corner(self) -> None:
        planner = self.make_planner(
            [
                [0, 0, 0, 0],
                [0.4, 0, 0, 0],
                [0.8, 0, 0, 0],
                [0.8, 0.4, 0, 1.57],
                [0.8, 0.8, 0, 1.57],
            ]
        )
        segment = planner.segment_to(
            [0.8, 0.8, 1.57], spacing_m=2.0, yaw_step_rad=0.5
        )
        np.testing.assert_allclose(segment.poses[1, :2], [0.8, 0.0])
        np.testing.assert_allclose(segment.poses[-1, :2], [0.8, 0.8])

    def test_navigation_prunes_hard_corner_via_point_before_smoothing(self) -> None:
        poses = np.asarray(
            [
                [0.0, 0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0, 0.0],
                [2.0, 2.0, 0.0, 1.57],
                [2.0, 4.0, 0.0, 1.57],
            ]
        )

        pruned = prune_hard_corner_checkpoints(poses)

        np.testing.assert_allclose(pruned[:, :2], [[0, 0], [2, 2], [2, 4]])

    def test_intermediate_yaw_follows_path_tangent(self) -> None:
        poses = np.asarray(
            [[0.0, 0.0, 0.0, 1.4], [1.0, 0.0, 0.0, -1.2], [1.0, 1.0, 0.0, 2.2]]
        )

        oriented = orient_route_tangents(poses)

        self.assertAlmostEqual(oriented[0, 3], 0.0)
        self.assertAlmostEqual(oriented[1, 3], np.pi / 2.0)
        self.assertAlmostEqual(oriented[2, 3], 2.2)

    def test_short_final_leg_does_not_restore_recorded_camera_yaw(self) -> None:
        poses = np.asarray(
            [[0.0, 0.0, 0.0, 1.4], [1.0, 0.0, 0.0, -1.2], [1.2, 0.0, 0.0, 2.2]]
        )

        oriented = orient_route_tangents(poses)

        self.assertAlmostEqual(oriented[1, 3], 0.0)
        self.assertAlmostEqual(oriented[2, 3], 2.2)


if __name__ == "__main__":
    unittest.main()
