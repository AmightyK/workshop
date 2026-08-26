from __future__ import annotations

import math

from scripts.person_safety_monitor import (
    Pose2D,
    crossing_worker_requires_stop,
    known_crossing_requires_stop,
    worker_relative_to_agv,
    worker_requires_stop,
)
from scripts.random_people import (
    HUMAN_1_ACTIVATION_DISTANCE_M,
    HUMAN_2_CONTINUOUS,
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


def test_scripted_worker_contracts_are_state_driven_and_continuous() -> None:
    assert HUMAN_1_ACTIVATION_DISTANCE_M == 3.2
    assert HUMAN_2_CONTINUOUS is True
