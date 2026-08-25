from __future__ import annotations

from scripts.behavior_planner import (
    Decision,
    PlannerConfig,
    Pose2D,
    PredictiveBehaviorPlanner,
    TrajectoryTrack,
)


def track(points: list[tuple[float, float, float]]) -> TrajectoryTrack:
    result = TrajectoryTrack()
    for timestamp, x, y in points:
        result.update(Pose2D(x, y), timestamp)
    return result


def test_trajectory_history_estimates_crossing_velocity() -> None:
    person = track([(0.0, 1.0, -1.0), (1.0, 1.0, -0.5), (2.0, 1.0, 0.0)])

    vx, vy = person.velocity()

    assert abs(vx) < 1.0e-12
    assert abs(vy - 0.5) < 1.0e-12


def test_static_blocker_waits_then_resumes_when_path_is_predicted_clear() -> None:
    planner = PredictiveBehaviorPlanner()
    blocker = track([(0.0, 1.0, 0.0), (1.0, 1.0, 0.0)])

    waiting = planner.evaluate(
        person_id="human_1",
        scenario="human_1_static_until_close",
        ego=Pose2D(0.0, 0.0),
        ego_speed_mps=0.4,
        track=blocker,
        timestamp=1.0,
    )
    blocker.update(Pose2D(1.0, 1.8), 3.0)
    cleared = planner.evaluate(
        person_id="human_1",
        scenario="human_1_static_until_close",
        ego=Pose2D(0.0, 0.0),
        ego_speed_mps=0.4,
        track=blocker,
        timestamp=3.0,
    )

    assert waiting.decision is Decision.WAIT
    assert cleared.decision is Decision.PASS
    assert "resume" in cleared.reason


def test_crossing_human_waits_when_ttc_is_inside_pass_window() -> None:
    planner = PredictiveBehaviorPlanner()
    crossing = track([(0.0, 1.2, -0.8), (1.0, 1.2, -0.4)])

    report = planner.evaluate(
        person_id="human_2",
        scenario="human_2_continuous_crossing",
        ego=Pose2D(0.0, 0.0),
        ego_speed_mps=0.5,
        track=crossing,
        timestamp=1.0,
    )

    assert report.decision is Decision.WAIT
    assert report.time_to_collision_s is not None
    assert report.collision_probability >= 0.55


def test_crossing_human_can_be_passed_when_future_window_is_clear() -> None:
    planner = PredictiveBehaviorPlanner()
    crossing_away = track([(0.0, 1.5, 0.9), (1.0, 1.5, 1.5)])

    report = planner.evaluate(
        person_id="human_2",
        scenario="human_2_continuous_crossing",
        ego=Pose2D(0.0, 0.0),
        ego_speed_mps=0.4,
        track=crossing_away,
        timestamp=1.0,
    )

    assert report.decision is Decision.PASS
    assert report.predicted_free_space_window_s >= 2.0


def test_persistent_blockage_requests_only_a_bounded_replan() -> None:
    config = PlannerConfig(replan_after_s=2.0, replan_cooldown_s=10.0)
    planner = PredictiveBehaviorPlanner(config)
    blocker = track([(0.0, 1.0, 0.0), (1.0, 1.0, 0.0)])

    planner.evaluate(
        person_id="human_1", scenario="static", ego=Pose2D(0.0, 0.0),
        ego_speed_mps=0.3, track=blocker, timestamp=1.0,
    )
    replan = planner.evaluate(
        person_id="human_1", scenario="static", ego=Pose2D(0.0, 0.0),
        ego_speed_mps=0.3, track=blocker, timestamp=3.1,
    )
    second = planner.evaluate(
        person_id="human_1", scenario="static", ego=Pose2D(0.0, 0.0),
        ego_speed_mps=0.3, track=blocker, timestamp=3.2,
    )

    assert replan.decision is Decision.REPLAN
    assert second.decision is Decision.WAIT


def test_stalled_crossing_human_at_corridor_edge_authorizes_safe_overtake() -> None:
    config = PlannerConfig(overtake_after_s=2.0)
    planner = PredictiveBehaviorPlanner(config)
    edge = track([(0.0, 1.5, 0.60), (1.0, 1.5, 0.60)])
    planner.evaluate(
        person_id="human_2", scenario="human_2_continuous_crossing",
        ego=Pose2D(0.0, 0.0), ego_speed_mps=0.3, track=edge, timestamp=1.0,
    )

    report = planner.evaluate(
        person_id="human_2", scenario="human_2_continuous_crossing",
        ego=Pose2D(0.0, 0.0), ego_speed_mps=0.3, track=edge, timestamp=3.1,
    )

    assert report.decision is Decision.PASS
    assert "lateral overtake" in report.reason


def test_decision_json_contains_required_reason_and_metrics() -> None:
    planner = PredictiveBehaviorPlanner()
    person = track([(0.0, 3.0, 2.0), (1.0, 3.0, 2.0)])

    report = planner.evaluate(
        person_id="human_2", scenario="crossing", ego=Pose2D(0.0, 0.0),
        ego_speed_mps=0.5, track=person, timestamp=1.0,
    )
    payload = report.as_dict()

    assert payload["decision"] in {"WAIT", "PASS", "REPLAN"}
    assert payload["reason"]
    assert "collision_probability" in payload
    assert "time_to_collision_s" in payload
    assert "predicted_free_space_window_s" in payload
