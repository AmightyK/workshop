"""End-to-end evaluator for the global V-JEPA retrieval baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from src.data.dataset import VideoPoseDataset
from src.evaluation.metrics import position_error, retrieval_recall, summarize_localization, yaw_error
from src.localization.pose_estimator import PoseEstimator
from src.models.vjepa_encoder import EncoderOutput
from src.retrieval.global_retriever import GlobalRetriever
from src.utils.logging import write_json, write_jsonl


class Encoder(Protocol):
    def encode_video(self, video: np.ndarray) -> EncoderOutput: ...


class GlobalBaselineEvaluator:
    """Localize every query clip and preserve complete retrieval diagnostics."""

    def __init__(
        self,
        encoder: Encoder,
        retriever: GlobalRetriever,
        *,
        top_k: int = 20,
        pose_method: str = "top1",
        weighted_alpha: float = 10.0,
        weighted_radius_m: float = 2.0,
        min_translation_m: float = 0.0,
    ) -> None:
        self.encoder = encoder
        self.retriever = retriever
        self.top_k = top_k
        self.pose_method = pose_method
        self.weighted_alpha = weighted_alpha
        self.weighted_radius_m = weighted_radius_m
        if min_translation_m < 0.0:
            raise ValueError("min_translation_m must be non-negative")
        self.min_translation_m = float(min_translation_m)
        self.pose_estimator = PoseEstimator()

    def run(
        self,
        dataset: VideoPoseDataset,
        *,
        predictions_path: str | Path | None = None,
        metrics_path: str | Path | None = None,
    ) -> tuple[list[dict[str, object]], dict[str, float | int]]:
        records: list[dict[str, object]] = []
        ground_truth_poses: list[np.ndarray] = []
        predicted_poses: list[np.ndarray] = []
        candidate_pose_sets: list[np.ndarray] = []
        stationary_clips_skipped = 0
        for item in (dataset[index] for index in range(len(dataset))):
            if item.ground_truth_translation_m < self.min_translation_m:
                stationary_clips_skipped += 1
                continue
            encoded = self.encoder.encode_video(item.frames)
            retrieval = self.retriever.query(encoded.global_embedding[0], top_k=self.top_k)
            if self.pose_method == "top1":
                prediction = self.pose_estimator.predict_top1(retrieval)
            elif self.pose_method == "weighted":
                prediction = self.pose_estimator.predict_weighted(
                    retrieval,
                    alpha=self.weighted_alpha,
                    spatial_radius_m=self.weighted_radius_m,
                )
            else:
                raise ValueError(f"unsupported pose method: {self.pose_method}")
            ground_truth = item.pose.as_array()
            ground_truth_poses.append(ground_truth)
            predicted_poses.append(prediction.pose)
            candidate_pose_sets.append(retrieval.poses)
            records.append(
                {
                    "query_id": item.id,
                    "timestamp": item.timestamp,
                    "ground_truth_pose": ground_truth.tolist(),
                    "predicted_pose": prediction.pose.tolist(),
                    "pose_method": prediction.method,
                    "top_k_candidate_ids": retrieval.ids.tolist(),
                    "top_k_similarities": retrieval.scores.astype(float).tolist(),
                    "candidate_poses": retrieval.poses.tolist(),
                    "position_error_m": position_error(ground_truth, prediction.pose),
                    "yaw_error_rad": yaw_error(ground_truth, prediction.pose),
                    "source_pose_time_error_sec": item.source_pose_time_error,
                    "ground_truth_translation_m": item.ground_truth_translation_m,
                }
            )

        if not records:
            raise ValueError(
                "no moving query clips remain after applying min_translation_m="
                f"{self.min_translation_m:.3f}"
            )
        metrics = summarize_localization(
            np.asarray(ground_truth_poses), np.asarray(predicted_poses)
        )
        metrics.update(retrieval_recall(ground_truth_poses, candidate_pose_sets))
        metrics["stationary_clips_skipped"] = stationary_clips_skipped
        metrics["min_translation_m"] = self.min_translation_m
        if predictions_path is not None:
            write_jsonl(predictions_path, records)
        if metrics_path is not None:
            write_json(metrics_path, metrics)
        return records, metrics
