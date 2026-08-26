from geometry_msgs.msg import PoseStamped

from storage_pick_mission import prune_passed_checkpoints
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


def test_retry_does_not_cut_route_when_robot_is_far_off_corridor() -> None:
    pending = [_pose(0.0, 0.0), _pose(2.0, 0.0), _pose(4.0, 0.0)]

    pruned = prune_passed_checkpoints(pending, (1.0, 2.0))

    assert pruned == pending


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
