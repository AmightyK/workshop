#!/usr/bin/env python3
"""Run global cosine retrieval and top-1 pose localization."""

from __future__ import annotations

import argparse
import json

from _common import dataset_from_config, encoder_from_config
from src.evaluation.evaluator import GlobalBaselineEvaluator
from src.mapping.map_database import VisualMap
from src.retrieval.global_retriever import GlobalRetriever
from src.utils.config import load_config, section


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--query-run", required=True)
    parser.add_argument("--map", default="outputs/map")
    parser.add_argument("--predictions", default="outputs/baseline/predictions.jsonl")
    parser.add_argument("--metrics", default="outputs/baseline/metrics.json")
    args = parser.parse_args()
    config = load_config(args.config)
    retrieval = section(config, "retrieval")
    estimator = config.get("pose_estimator", {})
    evaluation = config.get("evaluation", {})
    evaluator = GlobalBaselineEvaluator(
        encoder_from_config(config),
        GlobalRetriever(VisualMap.load(args.map)),
        top_k=int(retrieval.get("global_top_k", 20)),
        pose_method=str(estimator.get("method", "top1")),
        weighted_alpha=float(estimator.get("weighted_alpha", 10.0)),
        weighted_radius_m=float(estimator.get("weighted_radius_m", 2.0)),
        min_translation_m=float(evaluation.get("min_translation_m", 0.0)),
    )
    _, metrics = evaluator.run(
        dataset_from_config(args.query_run, config),
        predictions_path=args.predictions,
        metrics_path=args.metrics,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
