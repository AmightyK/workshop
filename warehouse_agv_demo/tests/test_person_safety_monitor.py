from __future__ import annotations

import math
import random
from types import SimpleNamespace

import pytest

from scripts.person_safety_monitor import (
    CAMERA_VISIBLE_SCRIPTED_WORKERS,
    Pose2D,
    crossing_worker_requires_stop,
    known_crossing_requires_stop,
    stale_worker_names,
    tracked_worker_requires_stop,
    worker_relative_to_agv,
    worker_requires_stop,
)
from scripts.random_people import (
    HUMAN_1_ACTIVATION_DISTANCE_M,
    HUMAN_1_CONTINUOUS,
    HUMAN_1_INITIAL_XY,
    HUMAN_1_REARM_EACH_MISSION,
    HUMAN_1_WAYPOINTS,
    HUMAN_2_WAYPOINTS,
    HUMAN_2_CONTINUOUS,
    RandomPeopleController,
    Walker,
    acquire_controller_lock,
    enforce_continuous_patrol_bounds,
    hold_return_until_agv_clears,
    worker_in_forward_camera_view,
    worker_yields_to_agv_at_crossing,
)


def test_worker_coordinates_are_transformed_into_agv_frame() -> None:
    agv = Pose2D(2.0, 3.0, math.pi / 2.0)

    forward, lateral = worker_relative_to_agv(agv, Pose2D(2.0, 4.0))

    assert abs(forward - 1.0) < 1e-12
    assert abs(lateral) < 1e-12


def test_only_worker_in_forward_crossing_corridor_requests_stop() -> None:
    agv = Pose2D(0.0, 0.0, 0.0)

    assert worker_requires_stop(agv, Pose2D(1.2, 0.4))
    assert not worker_requires_stop(agv, Pose2D(1.2, 0.8))
    assert not worker_requires_stop(agv, Pose2D(-1.0, 0.0))


def test_close_person_stops_in_any_direction_but_reverse_remains_normally_free() -> None:
    agv = Pose2D(0.0, 0.0, 0.0)

    assert worker_requires_stop(agv, Pose2D(-0.60, 0.0))
    assert not worker_requires_stop(agv, Pose2D(-1.0, 0.0))


def test_release_hysteresis_prevents_stop_go_chatter() -> None:
    agv = Pose2D(0.0, 0.0, 0.0)
    worker = Pose2D(1.7, 0.60)

    assert not worker_requires_stop(agv, worker, stopping=False)
    assert worker_requires_stop(agv, worker, stopping=True)


def test_known_worker_path_stops_only_near_shared_crossing() -> None:
    crossing = Pose2D(7.0, -10.0)
    agv = Pose2D(5.2, -10.0, 0.0)

    assert known_crossing_requires_stop(
        agv, Pose2D(7.0, -10.8), crossing
    )
    assert not known_crossing_requires_stop(
        agv, Pose2D(7.0, -12.0), crossing
    )
    assert not known_crossing_requires_stop(
        Pose2D(7.5, -10.0, 0.0), Pose2D(7.0, -10.2), crossing
    )


def test_worker_2_crossing_guard_protects_loaded_return_direction() -> None:
    worker = Pose2D(-1.5, -4.2)

    assert crossing_worker_requires_stop(
        "random_worker_5",
        Pose2D(-1.5, -2.2, -math.pi / 2.0),
        worker,
    )
    assert crossing_worker_requires_stop(
        "random_worker_5",
        Pose2D(-1.5, -6.2, math.pi / 2.0),
        worker,
    )
    assert not crossing_worker_requires_stop(
        "random_worker_5",
        Pose2D(-1.5, -7.0, -math.pi / 2.0),
        worker,
    )


def test_worker_1_crossing_guard_protects_return_from_c() -> None:
    worker = Pose2D(7.0, -9.8)

    assert crossing_worker_requires_stop(
        "random_worker_4",
        Pose2D(7.0, -7.9, -math.pi / 2.0),
        worker,
    )


def test_known_crossing_fallback_releases_after_track_is_usable() -> None:
    # Regression: this pose is inside the wide crossing fallback but outside
    # the immediate physical envelope. With a healthy track the predictive
    # planner must be allowed to accumulate its 1 s clearance confirmation.
    agv = Pose2D(5.95, -10.834, 0.0)
    worker = Pose2D(7.0, -10.0)

    assert tracked_worker_requires_stop(
        "random_worker_4",
        agv,
        worker,
        stopping=True,
        track_sample_count=1,
    )
    assert not tracked_worker_requires_stop(
        "random_worker_4",
        agv,
        worker,
        stopping=True,
        track_sample_count=2,
    )


def test_stale_worker_pose_forces_fail_safe_contract() -> None:
    timestamps = {"random_worker_1": 9.8, "random_worker_2": 8.0}

    assert stale_worker_names(timestamps, now=10.0, timeout_s=0.5) == [
        "random_worker_2"
    ]


def test_scripted_worker_contracts_are_state_driven_and_continuous() -> None:
    assert HUMAN_1_ACTIVATION_DISTANCE_M == 5.0
    assert HUMAN_1_INITIAL_XY == (7.0, -10.0)
    assert HUMAN_1_WAYPOINTS == ((7.0, -10.8), (7.0, -8.0))
    assert HUMAN_1_REARM_EACH_MISSION is True
    assert HUMAN_1_CONTINUOUS is True
    assert HUMAN_2_WAYPOINTS == ((-4.65, -4.2), (5.2, -4.2))
    assert HUMAN_2_CONTINUOUS is True


def test_only_one_worker_controller_can_own_a_partition(tmp_path) -> None:
    lock_path = tmp_path / "random_people.lock"
    first = acquire_controller_lock(lock_path)
    try:
        with pytest.raises(RuntimeError, match="already owns"):
            acquire_controller_lock(lock_path)
    finally:
        first.close()


def test_duplicate_mission_trigger_burst_arms_exactly_one_generation(
) -> None:
    controller = object.__new__(RandomPeopleController)
    controller.mission_generation = 0
    controller.last_mission_id = None

    for _ in range(5):
        controller.on_new_mission(SimpleNamespace(data=1787779000123456789))

    assert controller.mission_generation == 1

    controller.on_new_mission(SimpleNamespace(data=1787779000123456790))
    assert controller.mission_generation == 2


def test_initial_random_worker_target_is_in_front_of_spawn_heading() -> None:
    walker = Walker(
        "worker",
        ((-2.0, 0.0), (2.0, 0.0)),
        speed=0.5,
        x=0.0,
        y=0.0,
        yaw=0.0,
    )

    walker.choose_target(random.Random(7), prefer_forward=True)

    assert walker.target == (2.0, 0.0)


def test_worker_accepts_endpoint_crossed_between_pose_callbacks() -> None:
    walker = Walker(
        "worker",
        ((0.0, 0.0), (1.0, 0.0)),
        speed=0.5,
        x=0.0,
        y=0.0,
        target=(1.0, 0.0),
        target_origin=(0.0, 0.0),
    )

    walker.x = 1.12

    assert walker.reached_target()


def test_worker_2_hard_bound_reverses_even_after_large_endpoint_overshoot() -> None:
    walker = Walker(
        "random_worker_5",
        HUMAN_2_WAYPOINTS,
        speed=0.5,
        x=14.73,
        y=-4.2,
        target=HUMAN_2_WAYPOINTS[1],
        target_origin=HUMAN_2_WAYPOINTS[0],
        continuous=True,
    )

    assert enforce_continuous_patrol_bounds(walker)
    assert walker.target == HUMAN_2_WAYPOINTS[0]
    assert walker.target_origin == (14.73, -4.2)


def test_worker_does_not_accept_endpoint_before_segment_is_complete() -> None:
    walker = Walker(
        "worker",
        ((0.0, 0.0), (1.0, 0.0)),
        speed=0.5,
        x=0.65,
        y=0.0,
        target=(1.0, 0.0),
        target_origin=(0.0, 0.0),
    )

    assert not walker.reached_target()


def test_crossing_worker_waits_at_endpoint_until_agv_clears_intersection() -> None:
    walker = Walker(
        "worker",
        ((-4.8, -4.2), (3.0, -4.2)),
        speed=0.5,
        x=1.0,
        y=-4.2,
        crossing_xy=(-1.5, -4.2),
        agv_clearance_before_return_m=4.0,
    )

    assert hold_return_until_agv_clears(walker, (-1.5, -2.0))
    assert not hold_return_until_agv_clears(walker, (-1.5, 1.0))


def test_worker_2_east_endpoint_clears_the_loaded_return_ray() -> None:
    # Regression from the B return: the former endpoint (3.0, -4.2) was only
    # 5 cm lateral to the AGV heading and caused a permanent mutual WAIT.
    agv = Pose2D(1.14, -2.97, -0.597)
    endpoint = Pose2D(*HUMAN_2_WAYPOINTS[1])

    forward, lateral = worker_relative_to_agv(agv, endpoint)

    assert forward > 0.0
    assert abs(lateral) > 1.20
    assert not worker_requires_stop(agv, endpoint, stopping=True)


def test_worker_2_yields_when_agv_already_owns_the_crossing() -> None:
    walker = Walker(
        "random_worker_5",
        HUMAN_2_WAYPOINTS,
        speed=0.5,
        x=-3.2,
        y=-4.2,
        target=HUMAN_2_WAYPOINTS[1],
        crossing_xy=(-1.5, -4.2),
        yield_to_agv_at_crossing=True,
    )

    assert worker_yields_to_agv_at_crossing(walker, (-1.5, -2.2))


def test_committed_worker_2_keeps_crossing_so_agv_can_wait() -> None:
    walker = Walker(
        "random_worker_5",
        HUMAN_2_WAYPOINTS,
        speed=0.5,
        x=-2.2,
        y=-4.2,
        target=HUMAN_2_WAYPOINTS[1],
        crossing_xy=(-1.5, -4.2),
        yield_to_agv_at_crossing=True,
    )

    assert not worker_yields_to_agv_at_crossing(walker, (-1.5, -2.2))


def test_worker_2_does_not_yield_after_it_has_cleared_the_crossing() -> None:
    walker = Walker(
        "random_worker_5",
        HUMAN_2_WAYPOINTS,
        speed=0.5,
        x=0.5,
        y=-4.2,
        target=HUMAN_2_WAYPOINTS[1],
        crossing_xy=(-1.5, -4.2),
        yield_to_agv_at_crossing=True,
    )

    assert not worker_yields_to_agv_at_crossing(walker, (-1.5, -2.2))


def test_both_scripted_agv_stops_require_camera_visibility() -> None:
    assert CAMERA_VISIBLE_SCRIPTED_WORKERS == {
        "random_worker_4",
        "random_worker_5",
    }


def test_worker_1_reuses_the_same_segment_for_the_safe_return_trip() -> None:
    walker = Walker(
        "random_worker_4",
        HUMAN_1_WAYPOINTS,
        speed=0.5,
        x=7.0,
        y=-8.0,
        target=HUMAN_1_WAYPOINTS[1],
        target_origin=HUMAN_1_WAYPOINTS[0],
    )

    walker.choose_target(random.Random(1))

    assert walker.target == HUMAN_1_WAYPOINTS[0]
    assert walker.target_origin == (7.0, -8.0)


def test_worker_1_crossing_is_scripted_once_in_each_mission() -> None:
    walker = Walker(
        "random_worker_4",
        HUMAN_1_WAYPOINTS,
        speed=0.5,
        x=7.0,
        y=-10.0,
        yaw=math.pi / 2.0,
        target=HUMAN_1_WAYPOINTS[1],
        target_origin=HUMAN_1_INITIAL_XY,
        activation_distance_m=HUMAN_1_ACTIVATION_DISTANCE_M,
        activated=False,
        rearm_each_mission=True,
        last_pose_update=1.0,
    )
    controller = object.__new__(RandomPeopleController)
    controller.walkers = [walker]
    # 4.8 m validates the camera-gated 5.0 m early trigger. The former 3.2 m
    # trigger would leave this worker parked until the AGV was already close.
    controller.agv_xy = (11.8, -10.0)
    controller.agv_yaw = 0.0
    controller.feedback_bootstrap_deadline = 0.0
    controller.mission_generation = 0
    controller.generator = random.Random(1)
    commands = []
    controller.publish_velocity = (
        lambda current, linear, angular: commands.append((linear, angular))
    )

    controller.update(now=1.1)
    assert not walker.activated
    assert commands[-1] == (0.0, 0.0)

    controller.mission_generation = 1
    controller.update(now=1.2)
    assert not walker.activated
    assert commands[-1] == (0.0, 0.0)

    controller.agv_yaw = math.pi
    controller.update(now=1.25)
    assert walker.activated
    assert commands[-1][0] > 0.0

    walker.x, walker.y = HUMAN_1_WAYPOINTS[1]
    controller.update(now=1.3)
    assert walker.completed_mission_generation == 1
    assert not walker.activated
    assert walker.target == HUMAN_1_WAYPOINTS[0]
    assert commands[-1] == (0.0, 0.0)

    controller.mission_generation = 2
    controller.agv_xy = (9.8, -7.0)
    controller.agv_yaw = math.pi
    controller.update(now=1.4)
    assert walker.activated
    assert commands[-1][1] != 0.0


def test_worker_1_trigger_requires_real_forward_camera_visibility() -> None:
    worker = HUMAN_1_INITIAL_XY

    assert worker_in_forward_camera_view((10.0, -10.0), math.pi, worker)
    assert not worker_in_forward_camera_view((10.0, -10.0), 0.0, worker)


def test_world_reset_to_spawn_disarms_an_active_scripted_crossing() -> None:
    controller = RandomPeopleController(
        seed=1,
        speed_scale=1.0,
        worker4_mode="proximity",
        reset_on_start=False,
    )
    worker = controller.walkers_by_name["random_worker_4"]
    worker.x, worker.y = HUMAN_1_WAYPOINTS[-1]
    worker.activated = True
    controller.mission_generation = 3

    pose = SimpleNamespace(
        name="random_worker_4",
        position=SimpleNamespace(x=HUMAN_1_INITIAL_XY[0], y=HUMAN_1_INITIAL_XY[1]),
        orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
    )
    controller.on_worker_pose("random_worker_4", pose)

    assert not worker.activated
    assert worker.completed_mission_generation == 3
    assert worker.target == HUMAN_1_WAYPOINTS[-1]


def test_worker_1_starts_at_the_latest_github_position() -> None:
    assert HUMAN_1_INITIAL_XY == (7.0, -10.0)
