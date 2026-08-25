#!/usr/bin/env python3
"""Build a normalized global V-JEPA visual map from a mapping run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import dataset_from_config, encoder_from_config
from src.mapping.map_builder import GlobalMapBuilder
from src.utils.config import load_config, section


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--run", required=True, help="directory containing video.mp4 and poses.csv")
    parser.add_argument("--output", default="outputs/map")
    args = parser.parse_args()
    config = load_config(args.config)
    dataset = dataset_from_config(args.run, config)
    encoder = encoder_from_config(config)
    batch_size = int(section(config, "mapping").get("batch_size", 1))
    metadata = {"config": config}
    camera_pipeline_path = Path(args.run) / "camera_pipeline.json"
    if camera_pipeline_path.exists():
        metadata["camera_pipeline"] = json.loads(
            camera_pipeline_path.read_text(encoding="utf-8")
        )
    visual_map = GlobalMapBuilder(encoder, batch_size=batch_size).build(
        dataset,
        output_dir=args.output,
        metadata=metadata,
    )
    print(
        f"Saved {len(visual_map.global_embeddings)} entries with dimension "
        f"{visual_map.global_embeddings.shape[1]} to {args.output}"
    )


if __name__ == "__main__":
    main()
