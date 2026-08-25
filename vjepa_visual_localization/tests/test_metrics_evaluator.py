from __future__ import annotations

import json

import numpy as np

from src.data.dataset import VideoPoseDataset
from src.evaluation.evaluator import GlobalBaselineEvaluator
from src.evaluation.metrics import summarize_localization
from src.mapping.map_builder import GlobalMapBuilder
from src.retrieval.global_retriever import GlobalRetriever
from tests.helpers import MeanColorEncoder, make_run


def test_required_metric_values() -> None:
    ground_truth = np.asarray([[0, 0, 0, 0], [0, 0, 0, 0]], dtype=float)
    predictions = np.asarray([[0.3, 0, 0, 0.1], [3.0, 0, 0, -0.1]], dtype=float)
    report = summarize_localization(ground_truth, predictions)
    assert report["count"] == 2
    assert report["recall_at_0.5m"] == 0.5
    assert report["recall_at_5m"] == 1.0


def test_end_to_end_map_retrieval_and_debug_logs(tmp_path) -> None:
    colors = [(20 + i * 10, 220 - i * 8, 30 + i * 5) for i in range(16)]
    mapping_run = make_run(tmp_path / "mapping", colors)
    query_run = make_run(tmp_path / "query", colors)
    dataset_args = dict(
        clip_duration_sec=1.0,
        num_sampled_frames=4,
        stride_sec=1.0,
        pose_tolerance_sec=0.2,
    )
    mapping = VideoPoseDataset(mapping_run, **dataset_args)
    query = VideoPoseDataset(query_run, **dataset_args)
    encoder = MeanColorEncoder()
    visual_map = GlobalMapBuilder(encoder, batch_size=2).build(mapping)
    predictions_path = tmp_path / "predictions.jsonl"
    metrics_path = tmp_path / "metrics.json"
    records, metrics = GlobalBaselineEvaluator(
        encoder, GlobalRetriever(visual_map), top_k=3
    ).run(query, predictions_path=predictions_path, metrics_path=metrics_path)
    assert len(records) == len(query)
    assert metrics["mean_position_error_m"] == 0.0
    first = json.loads(predictions_path.read_text().splitlines()[0])
    assert len(first["top_k_candidate_ids"]) == 3
    assert len(first["top_k_similarities"]) == 3
    assert "candidate_poses" in first and "position_error_m" in first
    assert first["ground_truth_translation_m"] > 0.0
    assert metrics["stationary_clips_skipped"] == 0
    assert json.loads(metrics_path.read_text())["count"] == len(query)
