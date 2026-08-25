#!/usr/bin/env python3
"""Inspect raw video and V-JEPA preprocessing dimensions without inference."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import torch
from transformers import AutoVideoProcessor


def aspect_label(width: int, height: int) -> str:
    divisor = math.gcd(width, height)
    return f"{width // divisor}:{height // divisor}"


def parse_aspect(value: str) -> float:
    try:
        width, height = (float(part) for part in value.split(":", maxsplit=1))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("aspect must use WIDTH:HEIGHT, e.g. 16:9") from error
    if width <= 0.0 or height <= 0.0:
        raise argparse.ArgumentTypeError("aspect dimensions must be positive")
    return width / height


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="directory containing video.mp4")
    parser.add_argument("--checkpoint", default="facebook/vjepa2-vitl-fpc64-256")
    parser.add_argument("--output", default=None)
    parser.add_argument("--expected-aspect", type=parse_aspect, default=None)
    args = parser.parse_args()

    run_dir = Path(args.run)
    video_path = run_dir / "video.mp4"
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if width <= 0 or height <= 0:
        raise RuntimeError(f"invalid video dimensions: {width}x{height}")
    raw_aspect = width / height
    if args.expected_aspect is not None and not math.isclose(
        raw_aspect, args.expected_aspect, rel_tol=0.0, abs_tol=0.01
    ):
        raise RuntimeError(
            f"camera aspect is {aspect_label(width, height)} ({raw_aspect:.4f}), "
            f"expected {args.expected_aspect:.4f}"
        )

    processor = AutoVideoProcessor.from_pretrained(args.checkpoint)
    sample = torch.zeros((1, 3, height, width), dtype=torch.uint8)
    processed = processor(sample, return_tensors="pt")["pixel_values_videos"]
    model_height = int(processed.shape[-2])
    model_width = int(processed.shape[-1])
    report = {
        "checkpoint": args.checkpoint,
        "raw_video": {
            "path": str(video_path.resolve()),
            "width": width,
            "height": height,
            "aspect_ratio": raw_aspect,
            "aspect_label": aspect_label(width, height),
            "is_square": width == height,
            "fps": fps,
            "frame_count": frame_count,
            "matches_expected_aspect": args.expected_aspect is None or math.isclose(
                raw_aspect, args.expected_aspect, rel_tol=0.0, abs_tol=0.01
            ),
        },
        "vjepa_input": {
            "width": model_width,
            "height": model_height,
            "aspect_ratio": model_width / model_height,
            "aspect_label": aspect_label(model_width, model_height),
            "is_square": model_width == model_height,
            "resize_shortest_edge": processor.size.get("shortest_edge"),
            "center_crop": bool(processor.do_center_crop),
            "crop_size": dict(processor.crop_size),
            "note": "resize giữ tỉ lệ rồi center-crop; không kéo méo ảnh gốc thành vuông",
        },
    }
    output = Path(args.output) if args.output else run_dir / "camera_pipeline.json"
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
