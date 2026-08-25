from __future__ import annotations

import numpy as np

from src.prediction.latent_predictor import (
    LatentRolloutPredictor,
    behavior_case,
    summarize_evaluations,
)


def test_rollout_produces_all_three_required_horizons() -> None:
    predictor = LatentRolloutPredictor(normalize_rollouts=False)

    predictions, evaluations = predictor.observe(
        np.array([1.0, 0.0]), timestamp=1.0, scene="normal_driving"
    )

    assert [item.horizon for item in predictions] == [1, 2, 3]
    assert evaluations == []


def test_linear_latent_dynamics_have_zero_one_step_error() -> None:
    predictor = LatentRolloutPredictor(
        velocity_alpha=1.0, normalize_rollouts=False
    )
    predictor.observe(np.array([0.0, 1.0]), timestamp=0.0, scene="human_2")
    predictor.observe(np.array([1.0, 1.0]), timestamp=1.0, scene="human_2")
    _, evaluations = predictor.observe(
        np.array([2.0, 1.0]), timestamp=2.0, scene="human_2"
    )

    horizon_one = [row for row in evaluations if row.horizon == 1]
    # Two origins mature at index 2; the newest linear prediction is exact.
    assert horizon_one[-1].l1_latent_error == 0.0
    assert horizon_one[-1].cosine_similarity == 1.0
    assert horizon_one[-1].prediction_drift_error == 0.0


def test_metrics_are_aggregated_by_horizon_and_scene() -> None:
    predictor = LatentRolloutPredictor(normalize_rollouts=False)
    rows = []
    for index in range(5):
        _, matured = predictor.observe(
            np.array([float(index), 1.0]),
            timestamp=float(index),
            scene="shelf_approach",
        )
        rows.extend(matured)

    summary = summarize_evaluations(rows)

    assert summary["count"] == len(rows)
    assert set(summary["by_horizon"]) == {"1", "2", "3"}
    assert summary["by_scene"]["shelf_approach"]["count"] == len(rows)


def test_behavior_demo_cases_match_required_scenarios() -> None:
    assert behavior_case(
        decision="PASS",
        scenario="human_1_static_until_close",
        previous_decision="WAIT",
    ) == "human_leaves_path"
    assert behavior_case(
        decision="WAIT",
        scenario="human_2_continuous_crossing",
        previous_decision="PASS",
    ) == "human_continues_crossing"
    assert behavior_case(
        decision="PASS",
        scenario="human_2_continuous_crossing",
        previous_decision="WAIT",
    ) == "vehicle_can_safely_pass"
    assert behavior_case(
        decision="PASS",
        scenario="human_2_continuous_crossing",
        previous_decision="PASS",
        path_occupied=True,
    ) == "human_continues_crossing"
