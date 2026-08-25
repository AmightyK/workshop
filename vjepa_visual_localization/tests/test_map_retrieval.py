from __future__ import annotations

import numpy as np

from src.localization.global_localizer import GlobalVisualLocalizer
from src.localization.pose_estimator import PoseEstimator
from src.mapping.map_database import VisualMap
from src.retrieval.global_retriever import GlobalRetriever
from tests.helpers import MeanColorEncoder


def make_map() -> VisualMap:
    return VisualMap(
        global_embeddings=np.eye(3, dtype=np.float32),
        poses=np.asarray([[0, 0, 0, 3.13], [1, 0, 0, -3.13], [10, 0, 0, 0]], dtype=float),
        timestamps=np.asarray([0.0, 1.0, 2.0]),
        ids=np.asarray(["a", "b", "c"]),
    )


def test_map_round_trip_and_cosine_top1(tmp_path) -> None:
    visual_map = make_map()
    visual_map.save(tmp_path / "map")
    loaded = VisualMap.load(tmp_path / "map")
    result = GlobalRetriever(loaded).query(np.asarray([0.0, 1.0, 0.0]), top_k=2)
    assert result.ids.tolist() == ["b", "a"]
    prediction = PoseEstimator().predict_top1(result)
    np.testing.assert_allclose(prediction.pose, loaded.poses[1])


def test_weighted_pose_ignores_disconnected_candidate() -> None:
    result = GlobalRetriever(make_map()).query(np.asarray([0.7, 0.7, 0.0]), top_k=3)
    prediction = PoseEstimator().predict_weighted(result, spatial_radius_m=2.0)
    assert 0.0 <= prediction.pose[0] <= 1.0
    assert abs(abs(prediction.pose[3]) - np.pi) < 0.05


def test_global_localizer_uses_pixels_without_ground_truth() -> None:
    visual_map = VisualMap(
        global_embeddings=np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
        poses=np.asarray([[2, 3, 0, 0.2], [8, 9, 0, 1.0]], dtype=float),
        timestamps=np.asarray([0.0, 1.0]),
        ids=np.asarray(["red_place", "green_place"]),
    )
    red_clip = np.full((4, 16, 16, 3), [200, 0, 0], dtype=np.uint8)
    result = GlobalVisualLocalizer(
        MeanColorEncoder(), GlobalRetriever(visual_map), top_k=2
    ).localize(red_clip)
    assert result.prediction.source_id == "red_place"
    np.testing.assert_allclose(result.prediction.pose, [2, 3, 0, 0.2])
    assert result.query_embedding.shape == (3,)
    np.testing.assert_allclose(result.query_embedding, [1.0, 0.0, 0.0])
