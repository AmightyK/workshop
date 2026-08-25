from __future__ import annotations

import json

import numpy as np

from scripts.latent_prediction_monitor import (
    LatentLogWriter,
    LogSample,
    classify_scene,
)


def sample(index: int, scene: str = "human_2_encounter") -> LogSample:
    return LogSample(
        timestamp=float(index),
        frame_rgb=np.full((48, 64, 3), 30 + index, dtype=np.uint8),
        latent=np.asarray([1.0, 0.1 * index, 0.2, 0.3], dtype=np.float32),
        pose=(float(index), 0.0, 0.0, 0.0),
        scene=scene,
        mission_state="NAVIGATE_TO_SHELF",
        behavior={
            "decision": "WAIT",
            "reason": "predicted crossing",
            "scenario": "human_2_continuous_crossing",
            "person_id": "random_worker_5",
            "behavior_case": "human_continues_crossing",
        },
    )


def test_writer_persists_frames_latents_pose_rollouts_metrics_and_plots(tmp_path) -> None:
    writer = LatentLogWriter(tmp_path, horizons=(1, 2, 3), velocity_alpha=0.65)
    for index in range(5):
        writer.write(sample(index))
    writer.close(dropped_samples=0)

    records = [
        json.loads(line)
        for line in (tmp_path / "samples.jsonl").read_text().splitlines()
    ]
    summary = json.loads((tmp_path / "summary.json").read_text())

    assert len(records) == 5
    assert records[0]["vehicle_pose"] == [0.0, 0.0, 0.0, 0.0]
    assert len(records[0]["rollouts"]) == 3
    assert (tmp_path / records[0]["raw_frame"]).is_file()
    assert (tmp_path / records[0]["latent_vector"]).is_file()
    assert (tmp_path / "latent_metrics.jsonl").is_file()
    assert (tmp_path / "latent_prediction_metrics.png").is_file()
    assert (
        tmp_path / "behavior_visualizations" / "human_continues_crossing.png"
    ).is_file()
    assert summary["count"] > 0


def test_scene_classification_covers_all_mission_categories() -> None:
    assert classify_scene("SHELF_APPROACH", {}) == "shelf_approach"
    assert classify_scene("GRASP_PACKAGE", {}) == "pick_up_operation"
    assert classify_scene("RETURN_TO_DROPOFF", {}) == "return_path"
    assert classify_scene(
        "NAVIGATE_TO_SHELF",
        {"scenario": "human_1_static_until_close", "person_id": "random_worker_4"},
    ) == "human_1_encounter"
