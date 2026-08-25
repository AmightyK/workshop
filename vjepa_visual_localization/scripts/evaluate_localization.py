#!/usr/bin/env python3
"""Recompute aggregate metrics from saved retrieval debug predictions."""

from __future__ import annotations

import argparse
import json

import numpy as np

from _common import PROJECT_ROOT  # noqa: F401 - establishes src import path
from src.evaluation.metrics import retrieval_recall, summarize_localization
from src.utils.logging import read_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="outputs/baseline/predictions.jsonl")
    parser.add_argument("--output", default="outputs/baseline/metrics.json")
    args = parser.parse_args()
    rows = read_jsonl(args.predictions)
    if not rows:
        raise ValueError("prediction log is empty")
    ground_truth = np.asarray([row["ground_truth_pose"] for row in rows], dtype=np.float64)
    predictions = np.asarray([row["predicted_pose"] for row in rows], dtype=np.float64)
    candidates = [np.asarray(row["candidate_poses"], dtype=np.float64) for row in rows]
    metrics = summarize_localization(ground_truth, predictions)
    metrics.update(retrieval_recall(ground_truth, candidates))
    write_json(args.output, metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
