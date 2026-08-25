#!/usr/bin/env python3
"""Save a sampled clip contact sheet with synchronized pose metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from _common import dataset_from_config
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--run", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output", default="outputs/dataset_sample.jpg")
    args = parser.parse_args()
    item = dataset_from_config(args.run, load_config(args.config))[args.index]
    selected = np.linspace(0, len(item.frames) - 1, min(8, len(item.frames)), dtype=int)
    thumbnails = [cv2.resize(cv2.cvtColor(item.frames[i], cv2.COLOR_RGB2BGR), (256, 144)) for i in selected]
    sheet = np.concatenate(thumbnails, axis=1)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(target), sheet)
    target.with_suffix(".json").write_text(
        json.dumps(
            {
                "id": item.id,
                "timestamp": item.timestamp,
                "pose": item.pose.as_array().tolist(),
                "source_pose_time_error_sec": item.source_pose_time_error,
                "frame_timestamps": item.frame_timestamps,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved {target} and {target.with_suffix('.json')}")


if __name__ == "__main__":
    main()
