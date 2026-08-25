from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data.ros_image import image_message_to_rgb, message_timestamp_sec


@dataclass
class Stamp:
    sec: int = 2
    nanosec: int = 500_000_000


@dataclass
class Header:
    stamp: Stamp


@dataclass
class ImageMessage:
    width: int
    height: int
    step: int
    encoding: str
    data: bytes
    header: Header


def test_bgr_image_with_padded_rows_converts_to_rgb() -> None:
    # Two BGR pixels plus two bytes of transport padding.
    row = bytes([30, 20, 10, 60, 50, 40, 0, 0])
    message = ImageMessage(2, 1, 8, "bgr8", row, Header(Stamp()))
    rgb = image_message_to_rgb(message)
    np.testing.assert_array_equal(rgb, [[[10, 20, 30], [40, 50, 60]]])
    assert message_timestamp_sec(message) == 2.5
