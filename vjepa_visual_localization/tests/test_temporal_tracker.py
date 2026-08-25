from __future__ import annotations

import numpy as np

from src.localization.temporal_tracker import TemporalPoseTracker
from src.retrieval.global_retriever import RetrievalResult


def retrieval(poses, scores=None) -> RetrievalResult:
    poses = np.asarray(poses, dtype=float)
    count = len(poses)
    if scores is None:
        scores = np.linspace(0.98, 0.80, count)
    return RetrievalResult(
        indices=np.arange(count),
        ids=np.asarray([f"clip_{index}" for index in range(count)]),
        scores=np.asarray(scores, dtype=float),
        poses=poses,
        timestamps=np.arange(count, dtype=float),
    )


def test_temporal_prior_rejects_far_visual_top1() -> None:
    tracker = TemporalPoseTracker(smoothing_alpha=1.0)
    tracker.update(retrieval([[0, 0, 0, 0], [9, 0, 0, 0]]), timestamp=1.0)
    result = tracker.update(
        retrieval([[10, 0, 0, 0], [0.7, 0, 0, 0]], [0.99, 0.90]),
        timestamp=2.0,
        camera_moving=True,
    )
    assert result.state == "TRACKING"
    assert result.selected_rank == 1
    np.testing.assert_allclose(result.prediction.pose[:2], [0.7, 0.0])
    assert result.raw_jump_m == 10.0


def test_route_start_prior_prevents_ambiguous_global_initialization() -> None:
    tracker = TemporalPoseTracker(
        smoothing_alpha=1.0,
        initial_route_index=0,
        initial_route_window=3,
    )
    candidates = RetrievalResult(
        indices=np.asarray([60, 3, 0, 12]),
        ids=np.asarray(["clip_60", "clip_3", "clip_0", "clip_12"]),
        scores=np.asarray([0.96, 0.93, 0.91, 0.90]),
        poses=np.asarray(
            [
                [1.27, -9.96, 0, -3.13],
                [13.2, -10.6, 0, 3.10],
                [13.8, -10.6, 0, 3.14],
                [11.2, -10.2, 0, 2.90],
            ],
            dtype=float,
        ),
        timestamps=np.asarray([31.0, 2.5, 1.0, 7.0]),
    )

    result = tracker.update(candidates, timestamp=10.0)

    assert result.state == "INITIALIZED"
    assert result.raw_prediction.source_id == "clip_60"
    assert result.prediction.source_id == "clip_3"
    assert result.prediction.method == "temporal_initialize_route_prior"
    assert result.selected_rank == 1
    np.testing.assert_allclose(result.prediction.pose[:2], [13.2, -10.6])


def test_tracker_holds_instead_of_teleporting() -> None:
    tracker = TemporalPoseTracker(smoothing_alpha=1.0)
    tracker.update(retrieval([[0, 0, 0, 0]]), timestamp=1.0)
    result = tracker.update(retrieval([[12, 2, 0, 0]]), timestamp=2.0)
    assert result.state == "HOLDING"
    assert result.selected_rank is None
    np.testing.assert_allclose(result.prediction.pose[:2], [0.0, 0.0])
    assert result.rejected_streak == 1


def test_consistent_nearby_candidate_can_relocalize() -> None:
    tracker = TemporalPoseTracker(
        smoothing_alpha=1.0,
        max_translation_gate_m=0.8,
        relocalization_frames=3,
        relocalization_max_jump_m=4.0,
    )
    tracker.update(retrieval([[0, 0, 0, 0]]), timestamp=1.0)
    first = tracker.update(
        retrieval([[2.5, 0, 0, 0]], [0.95]),
        timestamp=2.0,
        camera_moving=True,
    )
    second = tracker.update(
        retrieval([[2.6, 0, 0, 0]], [0.96]),
        timestamp=3.0,
        camera_moving=True,
    )
    third = tracker.update(
        retrieval([[2.55, 0, 0, 0]], [0.97]),
        timestamp=4.0,
        camera_moving=True,
    )
    assert first.state == second.state == "HOLDING"
    assert third.state == "RELOCALIZED"
    np.testing.assert_allclose(third.prediction.pose[:2], [2.55, 0.0])


def test_large_repeated_alias_never_forces_relocalization() -> None:
    tracker = TemporalPoseTracker(relocalization_frames=2, relocalization_max_jump_m=4.0)
    tracker.update(retrieval([[0, 0, 0, 0]]), timestamp=1.0)
    for timestamp in (2.0, 3.0, 4.0):
        result = tracker.update(retrieval([[10, 0, 0, 0]], [0.99]), timestamp=timestamp)
        assert result.state == "HOLDING"
        np.testing.assert_allclose(result.prediction.pose[:2], [0.0, 0.0])


def test_tracker_does_not_move_backwards_in_latent_sequence() -> None:
    tracker = TemporalPoseTracker(smoothing_alpha=1.0)
    tracker.update(retrieval([[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0]]), timestamp=1.0)
    tracker.update(
        RetrievalResult(
            indices=np.asarray([1, 0, 2]),
            ids=np.asarray(["clip_1", "clip_0", "clip_2"]),
            scores=np.asarray([0.99, 0.90, 0.80]),
            poses=np.asarray([[1, 0, 0, 0], [0, 0, 0, 0], [2, 0, 0, 0]], dtype=float),
            timestamps=np.asarray([1.0, 0.0, 2.0]),
        ),
        timestamp=2.0,
        camera_moving=True,
    )
    result = tracker.update(
        RetrievalResult(
            indices=np.asarray([0, 2, 1]),
            ids=np.asarray(["clip_0", "clip_2", "clip_1"]),
            scores=np.asarray([0.995, 0.94, 0.93]),
            poses=np.asarray([[0, 0, 0, 0], [2, 0, 0, 0], [1, 0, 0, 0]], dtype=float),
            timestamps=np.asarray([0.0, 2.0, 1.0]),
        ),
        timestamp=3.0,
        camera_moving=True,
    )
    assert result.prediction.source_id != "clip_0"
    assert result.prediction.pose[0] >= 1.0


def test_camera_motion_advances_to_next_latent_instead_of_sticking() -> None:
    tracker = TemporalPoseTracker(
        smoothing_alpha=1.0,
        distance_penalty=0.0,
        yaw_penalty=0.0,
        forward_progress_bonus=0.055,
    )
    candidates = retrieval(
        [[0.0, 0, 0, 0], [0.4, 0, 0, 0], [0.8, 0, 0, 0]],
        [0.98, 0.95, 0.90],
    )
    tracker.update(candidates, timestamp=1.0)

    still = tracker.update(candidates, timestamp=2.0, camera_moving=False)
    assert still.prediction.source_id == "clip_0"

    moving = tracker.update(candidates, timestamp=3.0, camera_moving=True)
    assert moving.prediction.source_id == "clip_1"
    np.testing.assert_allclose(moving.prediction.pose[:2], [0.4, 0.0])


def test_sustained_camera_motion_cannot_stay_pinned_to_same_latent() -> None:
    tracker = TemporalPoseTracker(
        smoothing_alpha=1.0,
        distance_penalty=0.0,
        yaw_penalty=0.0,
        forward_progress_bonus=0.0,
        max_index_advance_per_sec=2.0,
    )
    candidates = retrieval(
        [[0.0, 0, 0, 0], [0.25, 0, 0, 0], [0.50, 0, 0, 0]],
        [0.99, 0.80, 0.70],
    )
    tracker.update(candidates, timestamp=1.0)

    first = tracker.update(
        candidates,
        timestamp=1.5,
        camera_moving=True,
        camera_progress_scale=1.0,
    )
    second = tracker.update(
        candidates,
        timestamp=2.0,
        camera_moving=True,
        camera_progress_scale=1.0,
    )

    assert first.prediction.source_id == "clip_1"
    assert second.prediction.source_id == "clip_2"


def test_stationary_frame_cannot_spend_saved_motion_credit() -> None:
    tracker = TemporalPoseTracker(
        smoothing_alpha=1.0,
        distance_penalty=0.0,
        yaw_penalty=0.0,
        forward_progress_bonus=0.0,
        max_index_advance_per_sec=3.0,
        stationary_translation_gate_m=0.18,
    )
    candidates = retrieval(
        [[0.0, 0, 0, 0], [0.10, 0, 0, 0], [0.20, 0, 0, 0]],
        [0.99, 0.90, 0.80],
    )
    tracker.update(candidates, timestamp=1.0)
    moving = tracker.update(
        candidates,
        timestamp=1.5,
        camera_moving=True,
        camera_progress_scale=1.0,
    )
    held = tracker.update(candidates, timestamp=2.0, camera_moving=False)

    assert moving.prediction.source_id == "clip_1"
    assert held.prediction.source_id == "clip_1"
    assert tracker.motion_credit < 1.0


def test_fractional_credit_cannot_select_unearned_second_step() -> None:
    tracker = TemporalPoseTracker(
        smoothing_alpha=1.0,
        distance_penalty=0.0,
        yaw_penalty=0.0,
        forward_progress_bonus=0.0,
        max_index_advance_per_sec=2.2,
    )
    candidates = retrieval(
        [[0.0, 0, 0, 0], [0.25, 0, 0, 0], [0.50, 0, 0, 0]],
        [0.50, 0.60, 0.99],
    )
    tracker.update(candidates, timestamp=1.0)
    result = tracker.update(
        candidates,
        timestamp=1.5,
        camera_moving=True,
        camera_progress_scale=1.0,
    )

    assert result.prediction.source_id == "clip_1"


def test_stationary_camera_cannot_drift_to_neighboring_shelf() -> None:
    tracker = TemporalPoseTracker(
        smoothing_alpha=1.0,
        stationary_translation_gate_m=0.18,
        relocalization_frames=2,
    )
    tracker.update(
        retrieval([[0.0, 0, 0, 0], [0.8, 0, 0, 0]]), timestamp=1.0
    )
    for timestamp in (2.0, 3.0, 4.0):
        result = tracker.update(
            retrieval([[0.8, 0, 0, 0], [0.05, 0, 0, 0]], [0.99, 0.90]),
            timestamp=timestamp,
            camera_moving=False,
        )
        assert result.state == "TRACKING"
        np.testing.assert_allclose(result.prediction.pose[:2], [0.05, 0.0])
