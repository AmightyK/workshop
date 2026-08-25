from __future__ import annotations

import numpy as np

from scripts.trajectory_evaluation import (
    corner_overshoot,
    compare_tracking,
    prefix_through_corner,
    summarize_tracking,
)


ROUTE = np.asarray([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]])


def test_tracking_comparison_accepts_stable_path_and_reduced_corner_overshoot() -> None:
    baseline_points = np.asarray(
        [[0, 0], [1, 0.1], [2.35, 0.0], [2.30, 0.5], [2.1, 1.5], [2, 2]]
    )
    candidate_points = np.asarray(
        [[0, 0], [1, 0.05], [2.08, 0.05], [2.07, 0.5], [2.04, 1.5], [2, 2]]
    )
    baseline = summarize_tracking(
        baseline_points, ROUTE, apex_xy=(2.0, 0.0), outgoing_yaw_rad=np.pi / 2
    )
    candidate = summarize_tracking(
        candidate_points, ROUTE, apex_xy=(2.0, 0.0), outgoing_yaw_rad=np.pi / 2
    )

    comparison = compare_tracking(baseline, candidate)

    assert comparison["tracking_not_worse"]
    assert comparison["corner_significantly_reduced"]
    assert comparison["corner_reduction_fraction"] > 0.25


def test_tracking_comparison_rejects_broad_regression() -> None:
    baseline_points = np.asarray([[0, 0], [1, 0.05], [2.05, 0.2], [2, 2]])
    poor_points = np.asarray([[0, 0], [1, 0.5], [2.5, 0.5], [2.5, 2]])
    baseline = summarize_tracking(
        baseline_points, ROUTE, apex_xy=(2, 0), outgoing_yaw_rad=np.pi / 2
    )
    poor = summarize_tracking(
        poor_points, ROUTE, apex_xy=(2, 0), outgoing_yaw_rad=np.pi / 2
    )

    comparison = compare_tracking(baseline, poor)

    assert not comparison["tracking_not_worse"]


def test_corner_overshoot_ignores_a_later_return_leg() -> None:
    points = np.asarray(
        [
            [0.0, -1.0], [0.0, 0.0], [0.05, 0.5], [0.02, 1.0],
            [0.0, 2.1], [3.0, 5.0], [0.0, 1.0],
        ]
    )

    overshoot = corner_overshoot(
        points, apex_xy=(0.0, 0.0), outgoing_yaw_rad=np.pi / 2
    )

    assert abs(overshoot - 0.05) < 1.0e-12


def test_prefix_through_corner_excludes_noncomparable_delivery_return() -> None:
    points = np.asarray(
        [
            [-2.0, 0.0], [-1.0, 0.0], [0.0, 0.0], [0.0, 1.0],
            [0.0, 2.1], [4.0, 4.0],
        ]
    )

    prefix = prefix_through_corner(
        points, apex_xy=(0.0, 0.0), outgoing_yaw_rad=np.pi / 2
    )

    assert prefix.tolist() == points[:5].tolist()
