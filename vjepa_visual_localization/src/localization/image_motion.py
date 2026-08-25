"""Image-only camera translation cue for temporal visual localization."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraMotionResult:
    pixel_change: float
    forward_expansion: float
    fit_inlier_ratio: float
    essential_inlier_ratio: float
    translation_z: float
    feature_count: int
    translating: bool


class CameraMotionEstimator:
    """Separate forward camera translation from local motion and in-place yaw.

    Forward travel creates an expanding optical-flow field. A person walking
    through a small part of the image is removed by the robust affine fit, and
    an in-place yaw is mostly a common horizontal shift with little expansion.
    This estimator consumes pixels only; no velocity, odometry or truth enters
    the V-JEPA localization path.
    """

    def __init__(
        self,
        *,
        forward_expansion_threshold: float = 0.035,
        min_fit_inlier_ratio: float = 0.14,
        min_essential_inlier_ratio: float = 0.35,
        max_forward_translation_z: float = -0.75,
        horizontal_fov_rad: float = 1.22,
        width: int = 320,
        height: int = 180,
    ) -> None:
        self.forward_expansion_threshold = float(forward_expansion_threshold)
        self.min_fit_inlier_ratio = float(min_fit_inlier_ratio)
        self.min_essential_inlier_ratio = float(min_essential_inlier_ratio)
        self.max_forward_translation_z = float(max_forward_translation_z)
        self.width = int(width)
        self.height = int(height)
        focal = self.width / (2.0 * math.tan(float(horizontal_fov_rad) / 2.0))
        self.camera_matrix = np.asarray(
            [
                [focal, 0.0, (self.width - 1) / 2.0],
                [0.0, focal, (self.height - 1) / 2.0],
                [0.0, 0.0, 1.0],
            ]
        )
        self.previous: np.ndarray | None = None

    def reset(self) -> None:
        self.previous = None

    def measure(self, rgb: np.ndarray) -> CameraMotionResult:
        gray = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (self.width, self.height), interpolation=cv2.INTER_AREA)
        if self.previous is None:
            self.previous = gray
            return CameraMotionResult(0.0, 0.0, 0.0, 0.0, 0.0, 0, False)

        pixel_change = float(
            np.quantile(
                np.abs(gray.astype(np.float32) - self.previous.astype(np.float32))
                / 255.0,
                0.70,
            )
        )
        previous_points = cv2.goodFeaturesToTrack(
            self.previous,
            maxCorners=600,
            qualityLevel=0.01,
            minDistance=6,
            blockSize=7,
        )
        forward_expansion = 0.0
        inlier_ratio = 0.0
        essential_inlier_ratio = 0.0
        translation_z = 0.0
        feature_count = 0
        if previous_points is not None:
            current_points, status, _ = cv2.calcOpticalFlowPyrLK(
                self.previous,
                gray,
                previous_points,
                None,
                winSize=(21, 21),
                maxLevel=3,
                criteria=(
                    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    30,
                    0.01,
                ),
            )
            matched = status[:, 0] == 1
            old = previous_points[matched].reshape(-1, 2)
            new = current_points[matched].reshape(-1, 2)
            feature_count = len(old)
            if feature_count >= 12:
                transform, inliers = cv2.estimateAffinePartial2D(
                    old,
                    new,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=2.0,
                    maxIters=2000,
                    confidence=0.99,
                )
                if transform is not None and inliers is not None:
                    image_scale = float(math.hypot(transform[0, 0], transform[0, 1]))
                    forward_expansion = image_scale - 1.0
                    inlier_ratio = float(np.mean(inliers))
                essential, essential_mask = cv2.findEssentialMat(
                    old,
                    new,
                    self.camera_matrix,
                    method=cv2.RANSAC,
                    prob=0.999,
                    threshold=1.0,
                )
                if essential is not None and essential_mask is not None:
                    try:
                        inlier_count, _, translation, _ = cv2.recoverPose(
                            essential,
                            old,
                            new,
                            self.camera_matrix,
                            mask=essential_mask,
                        )
                        essential_inlier_ratio = float(inlier_count / feature_count)
                        translation_z = float(translation[2, 0])
                    except cv2.error:
                        pass
        self.previous = gray
        affine_forward = (
            forward_expansion >= self.forward_expansion_threshold
            and inlier_ratio >= self.min_fit_inlier_ratio
        )
        essential_forward = (
            essential_inlier_ratio >= self.min_essential_inlier_ratio
            and translation_z <= self.max_forward_translation_z
        )
        return CameraMotionResult(
            pixel_change=pixel_change,
            forward_expansion=forward_expansion,
            fit_inlier_ratio=inlier_ratio,
            essential_inlier_ratio=essential_inlier_ratio,
            translation_z=translation_z,
            feature_count=feature_count,
            # Essential-matrix translation is scale-free and became unstable
            # around a moving worker / in-place rack turn. It may corroborate
            # the debug output, but route progress requires the robust global
            # expansion field. This keeps local foreground motion from moving
            # the camera through the latent map while the AGV is stopped.
            translating=affine_forward,
        )
