from __future__ import annotations

import cv2
import numpy as np

from src.localization.image_motion import CameraMotionEstimator


def textured_frame() -> np.ndarray:
    rng = np.random.default_rng(7)
    gray = rng.integers(0, 256, (180, 320), dtype=np.uint8)
    return np.repeat(gray[..., None], 3, axis=2)


def test_forward_expansion_is_translation_cue() -> None:
    first = textured_frame()
    transform = cv2.getRotationMatrix2D((159.5, 89.5), 0.0, 1.06)
    second = cv2.warpAffine(first, transform, (320, 180))
    estimator = CameraMotionEstimator(forward_expansion_threshold=0.04)
    estimator.measure(first)
    result = estimator.measure(second)
    assert result.translating
    assert result.forward_expansion > 0.04


def test_horizontal_camera_yaw_does_not_look_like_forward_travel() -> None:
    first = textured_frame()
    transform = np.float32([[1, 0, 18], [0, 1, 0]])
    second = cv2.warpAffine(first, transform, (320, 180), borderMode=cv2.BORDER_REFLECT)
    estimator = CameraMotionEstimator(forward_expansion_threshold=0.04)
    estimator.measure(first)
    result = estimator.measure(second)
    assert result.pixel_change > 0.02
    assert not result.translating


def test_local_foreground_motion_does_not_advance_camera() -> None:
    first = textured_frame()
    second = first.copy()
    # Simulate a large textured worker crossing only the center of a static
    # warehouse view. Global background scale remains one.
    patch = first[45:145, 110:180].copy()
    second[45:145, 145:215] = patch
    estimator = CameraMotionEstimator(forward_expansion_threshold=0.04)
    estimator.measure(first)
    result = estimator.measure(second)

    assert result.feature_count > 100
    assert not result.translating
