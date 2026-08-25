"""Trajectory and corner metrics used by the warehouse acceptance audit."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrackingMetrics:
    count: int
    rmse_m: float
    mean_error_m: float
    p95_error_m: float
    max_error_m: float
    corner_overshoot_m: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "rmse_m": self.rmse_m,
            "mean_error_m": self.mean_error_m,
            "p95_error_m": self.p95_error_m,
            "max_error_m": self.max_error_m,
            "corner_overshoot_m": self.corner_overshoot_m,
        }


def polyline_distances(points: np.ndarray, route: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    route = np.asarray(route, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape [N,2]")
    if route.ndim != 2 or route.shape[1] != 2 or len(route) < 2:
        raise ValueError("route must have shape [M,2] with M>=2")
    starts = route[:-1]
    vectors = route[1:] - starts
    lengths_sq = np.sum(vectors * vectors, axis=1)
    valid = lengths_sq > 1.0e-12
    if not np.any(valid):
        raise ValueError("route has no non-zero segment")
    starts, vectors, lengths_sq = starts[valid], vectors[valid], lengths_sq[valid]
    offsets = points[:, None, :] - starts[None, :, :]
    projection = np.sum(offsets * vectors[None, :, :], axis=2) / lengths_sq[None, :]
    projection = np.clip(projection, 0.0, 1.0)
    closest = starts[None, :, :] + projection[:, :, None] * vectors[None, :, :]
    return np.min(np.linalg.norm(points[:, None, :] - closest, axis=2), axis=1)


def corner_overshoot(
    points: np.ndarray,
    *,
    apex_xy: tuple[float, float],
    outgoing_yaw_rad: float,
    validation_distance_m: float = 2.0,
) -> float:
    """Maximum outgoing-segment lateral error after the closest apex sample."""
    points = np.asarray(points, dtype=np.float64)
    apex = np.asarray(apex_xy, dtype=np.float64)
    if len(points) == 0:
        return math.inf
    apex_index = int(np.argmin(np.linalg.norm(points - apex, axis=1)))
    offsets = points[apex_index:] - apex
    tangent = np.asarray(
        [math.cos(outgoing_yaw_rad), math.sin(outgoing_yaw_rad)], dtype=np.float64
    )
    normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float64)
    progress = offsets @ tangent
    lateral = np.abs(offsets @ normal)
    # Score only the first contiguous outgoing pass. A delivery can later
    # revisit the same projection on its return; that is not corner overshoot.
    covered: list[float] = []
    entered = False
    for progress_m, lateral_m in zip(progress, lateral):
        if 0.0 <= progress_m <= float(validation_distance_m):
            entered = True
            covered.append(float(lateral_m))
        elif entered:
            break
    return max(covered) if covered else math.inf


def prefix_through_corner(
    points: np.ndarray,
    *,
    apex_xy: tuple[float, float],
    outgoing_yaw_rad: float,
    validation_distance_m: float = 2.0,
) -> np.ndarray:
    """Return the shared route prefix through the first outgoing corner pass.

    Baseline patrol recordings stop outbound while a delivery mission later
    takes a direct A* return. Their prefix through the Shelf A corner is the
    common, controller-owned tracking interval.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
        raise ValueError("points must have non-empty shape [N,2]")
    apex = np.asarray(apex_xy, dtype=np.float64)
    apex_index = int(np.argmin(np.linalg.norm(points - apex, axis=1)))
    tangent = np.asarray(
        [math.cos(outgoing_yaw_rad), math.sin(outgoing_yaw_rad)], dtype=np.float64
    )
    progress = (points[apex_index:] - apex) @ tangent
    completed = np.flatnonzero(progress >= float(validation_distance_m))
    if len(completed) == 0:
        raise ValueError(
            "trajectory does not cover the requested outgoing corner distance"
        )
    end = apex_index + int(completed[0]) + 1
    return points[:end].copy()


def summarize_tracking(
    points: np.ndarray,
    route: np.ndarray,
    *,
    apex_xy: tuple[float, float],
    outgoing_yaw_rad: float,
) -> TrackingMetrics:
    errors = polyline_distances(points, route)
    return TrackingMetrics(
        count=len(errors),
        rmse_m=float(np.sqrt(np.mean(errors**2))),
        mean_error_m=float(np.mean(errors)),
        p95_error_m=float(np.percentile(errors, 95)),
        max_error_m=float(np.max(errors)),
        corner_overshoot_m=corner_overshoot(
            points, apex_xy=apex_xy, outgoing_yaw_rad=outgoing_yaw_rad
        ),
    )


def compare_tracking(
    baseline: TrackingMetrics,
    candidate: TrackingMetrics,
    *,
    max_tracking_degradation: float = 0.05,
    minimum_overshoot_reduction: float = 0.20,
) -> dict[str, float | bool]:
    rmse_limit = baseline.rmse_m * (1.0 + max_tracking_degradation)
    p95_limit = baseline.p95_error_m * (1.0 + max_tracking_degradation)
    overshoot_limit = baseline.corner_overshoot_m * (
        1.0 - minimum_overshoot_reduction
    )
    return {
        "tracking_not_worse": bool(
            candidate.rmse_m <= rmse_limit and candidate.p95_error_m <= p95_limit
        ),
        "corner_significantly_reduced": bool(
            candidate.corner_overshoot_m <= overshoot_limit
        ),
        "rmse_limit_m": rmse_limit,
        "p95_limit_m": p95_limit,
        "corner_overshoot_limit_m": overshoot_limit,
        "corner_reduction_fraction": (
            1.0 - candidate.corner_overshoot_m / baseline.corner_overshoot_m
            if baseline.corner_overshoot_m > 1.0e-9
            else 0.0
        ),
    }
