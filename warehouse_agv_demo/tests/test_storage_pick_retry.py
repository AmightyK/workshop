from geometry_msgs.msg import PoseStamped

from storage_pick_mission import prune_passed_checkpoints


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
