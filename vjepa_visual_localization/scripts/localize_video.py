#!/usr/bin/env python3
"""Localize a video-only query run without loading ground-truth poses."""

from __future__ import annotations

import argparse
import json

from _common import encoder_from_config, video_dataset_from_config
from src.localization.global_localizer import GlobalVisualLocalizer
from src.mapping.map_database import VisualMap
from src.retrieval.global_retriever import GlobalRetriever
from src.utils.config import load_config, section
from src.utils.logging import write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--query-run", required=True, help="directory containing video.mp4; poses.csv is not read")
    parser.add_argument("--map", default="outputs/map")
    parser.add_argument("--output", default="outputs/localization/predictions.jsonl")
    args = parser.parse_args()
    config = load_config(args.config)
    retrieval_config = section(config, "retrieval")
    estimator_config = config.get("pose_estimator", {})
    localizer = GlobalVisualLocalizer(
        encoder_from_config(config),
        GlobalRetriever(VisualMap.load(args.map)),
        top_k=int(retrieval_config.get("global_top_k", 20)),
        pose_method=str(estimator_config.get("method", "top1")),
        weighted_alpha=float(estimator_config.get("weighted_alpha", 10.0)),
        weighted_radius_m=float(estimator_config.get("weighted_radius_m", 2.0)),
    )
    dataset = video_dataset_from_config(args.query_run, config)
    rows = []
    for index in range(len(dataset)):
        clip = dataset[index]
        result = localizer.localize(clip.frames)
        row = {
            "query_id": clip.id,
            "timestamp": clip.timestamp,
            "predicted_pose": result.prediction.pose.tolist(),
            "pose_method": result.prediction.method,
            "confidence_margin": result.confidence_margin,
            "top_k_candidate_ids": result.retrieval.ids.tolist(),
            "top_k_similarities": result.retrieval.scores.astype(float).tolist(),
            "candidate_poses": result.retrieval.poses.tolist(),
        }
        rows.append(row)
        print(json.dumps(row))
    write_jsonl(args.output, rows)
    latest = rows[-1]
    print(f"Current visual pose at t={latest['timestamp']:.3f}: {latest['predicted_pose']}")


if __name__ == "__main__":
    main()
