from __future__ import annotations

import math

from scripts.person_safety_monitor import (
    Pose2D,
    known_crossing_requires_stop,
    worker_relative_to_agv,
    worker_requires_stop,
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
