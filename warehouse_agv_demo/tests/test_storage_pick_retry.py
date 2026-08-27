import numpy as np

from geometry_msgs.msg import PoseStamped

from latent_route import LatentRoutePlanner
from storage_pick_mission import prune_passed_checkpoints, staging_route_segment
from grasp_retry import grasp_attempts
from mission_lifecycle import (
    contains_ordered_sequence,
    planar_pose_error,
    require_home_pose,
)


def _pose(x: float, y: float) -> PoseStamped:
    pose = PoseStamped()
    pose.pose.position.x = x
    pose.pose.position.y = y
    return pose


def test_retry_drops_checkpoint_already_passed_along_route() -> None:
    pending = [_pose(0.0, 0.0), _pose(2.0, 0.0), _pose(4.0, 0.0)]

    pruned = prune_passed_checkpoints(pending, (2.4, 0.1))

    assert [(pose.pose.position.x, pose.pose.position.y) for pose in pruned] == [
        (4.0, 0.0)
    ]


def test_retry_keeps_checkpoint_that_is_still_ahead() -> None:
    pending = [_pose(2.0, 0.0), _pose(4.0, 0.0)]

    pruned = prune_passed_checkpoints(pending, (0.5, 0.1))

    assert pruned == pending


def test_redirect_from_c_to_b_skips_dock_prefix() -> None:
    pending = [
        _pose(10.8, -10.0),
        _pose(5.5, -10.0),
        _pose(-1.45, -10.0),
        _pose(-1.90, 0.8),
    ]

    pruned = prune_passed_checkpoints(pending, (-2.50, 3.8))

    assert [(pose.pose.position.x, pose.pose.position.y) for pose in pruned] == [
        (-1.90, 0.8)
    ]


def test_redirect_from_c_to_b_keeps_b_shelf_approach_after_staging() -> None:
    pending = [
        _pose(10.8, -10.0),
        _pose(5.5, -10.0),
        _pose(-1.45, -10.0),
        _pose(-1.90, 0.8),
        _pose(-5.04, 1.06),
    ]

    pruned = prune_passed_checkpoints(pending, (-2.50, 3.8))

    assert [(pose.pose.position.x, pose.pose.position.y) for pose in pruned] == [
        (-1.90, 0.8),
        (-5.04, 1.06),
    ]


def test_redirect_from_c_to_a_skips_dock_prefix() -> None:
    pending = [
        _pose(10.8, -10.0),
        _pose(5.5, -10.0),
        _pose(-1.45, -10.0),
        _pose(-1.90, -2.3),
    ]

    pruned = prune_passed_checkpoints(pending, (-2.50, 3.8))

    assert [(pose.pose.position.x, pose.pose.position.y) for pose in pruned] == [
        (-1.90, -2.3)
    ]


def test_redirect_from_a_to_c_keeps_only_forward_aisle_goal() -> None:
    pending = [
        _pose(10.8, -10.0),
        _pose(5.5, -10.0),
        _pose(-1.45, -10.0),
        _pose(-2.50, 3.8),
    ]

    pruned = prune_passed_checkpoints(pending, (-1.90, -2.3))

    assert [(pose.pose.position.x, pose.pose.position.y) for pose in pruned] == [
        (-2.50, 3.8)
    ]


def test_retry_does_not_cut_route_when_robot_is_far_off_corridor() -> None:
    pending = [_pose(0.0, 0.0), _pose(2.0, 0.0), _pose(4.0, 0.0)]

    pruned = prune_passed_checkpoints(pending, (1.0, 2.0))

    assert pruned == pending


def test_retry_selects_nearest_later_segment_without_returning_to_prefix() -> None:
    pending = [
        _pose(10.0, -10.0),
        _pose(0.0, -10.0),
        _pose(0.0, 0.0),
        _pose(-6.0, 0.0),
    ]

    pruned = prune_passed_checkpoints(pending, (0.1, -3.0))

    assert [(pose.pose.position.x, pose.pose.position.y) for pose in pruned] == [
        (0.0, 0.0),
        (-6.0, 0.0),
    ]


def test_storage_b_direct_staging_does_not_replay_storage_a_slots(tmp_path) -> None:
    # The recorded map reaches B only after visiting A01. A B mission must
    # retain B's latent index for its shelf approach without sending A01 as an
    # outbound checkpoint.
    poses = np.asarray(
        [
            [10.0, -10.0, 0.0, 3.14],
            [-2.0, -2.3, 0.0, 1.57],
            [-6.2, -1.94, 0.0, 1.57],  # A01 inspection detour
            [-2.0, 0.8, 0.0, 1.57],
            [-5.0, 1.06, 0.0, 1.57],  # B02 shelf approach
        ],
        dtype=np.float64,
    )
    np.save(tmp_path / "poses.npy", poses)
    planner = LatentRoutePlanner(tmp_path, max_target_error_m=0.4)
    route = {
        "direct_staging": True,
        "waypoints": [
            {"name": "dock_exit", "pose": [8.0, -10.0, 3.14]},
            {"name": "central_south", "pose": [-2.0, -10.0, 1.57]},
            {"name": "cabinet_A_pass", "pose": [-2.0, -2.3, 1.57]},
            {"name": "cabinet_B_staging", "pose": [-2.0, 0.8, 1.57]},
        ],
    }

    segment = staging_route_segment(route, planner)

    assert segment.end_index == 3
    assert not np.any(
        np.all(np.isclose(segment.poses[:, :2], [-6.2, -1.94]), axis=1)
    )
    np.testing.assert_allclose(segment.poses[-1, :2], [-2.0, 0.8])


def test_storage_c_direct_route_has_no_a_or_b_bypass_waypoint() -> None:
    import yaml
    from storage_pick_mission import ROUTE_CONFIG

    route = yaml.safe_load(ROUTE_CONFIG.read_text(encoding="utf-8"))["routes"][
        "storage_C"
    ]
    names = [waypoint["name"] for waypoint in route["waypoints"]]

    assert names == [
        "dock_exit",
        "east_cross_aisle",
        "central_south",
        "cabinet_C_staging",
    ]


def test_storage_b_direct_route_has_no_storage_a_goal() -> None:
    import yaml
    from storage_pick_mission import ROUTE_CONFIG

    route = yaml.safe_load(ROUTE_CONFIG.read_text(encoding="utf-8"))["routes"][
        "storage_B"
    ]

    assert [waypoint["name"] for waypoint in route["waypoints"]] == [
        "dock_exit",
        "east_cross_aisle",
        "central_south",
        "cabinet_B_staging",
    ]


def test_all_pick_cabinets_use_explicit_direct_staging_routes() -> None:
    import yaml
    from storage_pick_mission import ROUTE_CONFIG

    routes = yaml.safe_load(ROUTE_CONFIG.read_text(encoding="utf-8"))["routes"]

    assert all(routes[name]["direct_staging"] for name in (
        "storage_A", "storage_B", "storage_C"
    ))


def test_grasp_retry_budget_is_configurable_and_bounded() -> None:
    assert list(grasp_attempts(0)) == [1]
    assert list(grasp_attempts(2)) == [1, 2, 3]


def test_recycle_sequence_requires_drop_park_charge_then_ready() -> None:
    assert contains_ordered_sequence(
        ["PLACE_PACKAGE", "DROP", "PARK", "CHARGING_HOME", "READY_FOR_NEXT_TASK"]
    )
    assert not contains_ordered_sequence(
        ["DROP", "CHARGING_HOME", "PARK", "READY_FOR_NEXT_TASK"]
    )


def test_home_gate_uses_observed_pose_not_elapsed_time() -> None:
    assert planar_pose_error((13.92, -10.61), (13.90, -10.60, 3.14)) < 0.03
    assert require_home_pose((13.92, -10.61), (13.90, -10.60, 3.14), 0.38) < 0.03

    try:
        require_home_pose((13.0, -10.6), (13.90, -10.60, 3.14), 0.38)
    except RuntimeError as error:
        assert "Charging/home gate failed" in str(error)
    else:
        raise AssertionError("home gate accepted an out-of-tolerance pose")
