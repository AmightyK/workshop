"""ROS Image conversion without a cv_bridge dependency."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def image_message_to_rgb(message: Any) -> np.ndarray:
    """Convert common 8-bit ROS image encodings to contiguous RGB."""

    encodings = {
        "rgb8": (3, None),
        "bgr8": (3, cv2.COLOR_BGR2RGB),
        "rgba8": (4, cv2.COLOR_RGBA2RGB),
        "bgra8": (4, cv2.COLOR_BGRA2RGB),
        "mono8": (1, cv2.COLOR_GRAY2RGB),
    }
    encoding = str(message.encoding).lower()
    if encoding not in encodings:
        raise ValueError(f"unsupported ROS image encoding: {message.encoding}")
    channels, conversion = encodings[encoding]
    width, height, step = int(message.width), int(message.height), int(message.step)
    expected_row_bytes = width * channels
    if step < expected_row_bytes:
        raise ValueError("ROS image step is smaller than encoded row width")
    raw = np.frombuffer(message.data, dtype=np.uint8)
    if raw.size < height * step:
        raise ValueError("ROS image data is truncated")
    rows = raw[: height * step].reshape(height, step)[:, :expected_row_bytes]
    image = rows.reshape(height, width, channels) if channels > 1 else rows.reshape(height, width)
    if conversion is not None:
        image = cv2.cvtColor(image, conversion)
    return np.ascontiguousarray(image)


def message_timestamp_sec(message: Any) -> float:
    """Read a ROS header timestamp as floating-point seconds."""

    return float(message.header.stamp.sec) + 1e-9 * float(message.header.stamp.nanosec)
