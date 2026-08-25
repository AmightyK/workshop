"""Small dependency-free 2-D pose helpers for Nav2 localization."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float


def wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def planar_distance(first: Pose2D, second: Pose2D) -> float:
    return math.hypot(second.x - first.x, second.y - first.y)


def yaw_distance(first: Pose2D, second: Pose2D) -> float:
    return abs(wrap_angle(second.yaw - first.yaw))


def map_to_odom(map_base: Pose2D, odom_base: Pose2D) -> Pose2D:
    """Solve T_map_odom = T_map_base * inverse(T_odom_base)."""
    yaw = wrap_angle(map_base.yaw - odom_base.yaw)
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    return Pose2D(
        map_base.x - (cosine * odom_base.x - sine * odom_base.y),
        map_base.y - (sine * odom_base.x + cosine * odom_base.y),
        yaw,
    )


def compose(first: Pose2D, second: Pose2D) -> Pose2D:
    cosine = math.cos(first.yaw)
    sine = math.sin(first.yaw)
    return Pose2D(
        first.x + cosine * second.x - sine * second.y,
        first.y + sine * second.x + cosine * second.y,
        wrap_angle(first.yaw + second.yaw),
    )


def blend_pose(current: Pose2D, target: Pose2D, alpha: float) -> Pose2D:
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    return Pose2D(
        current.x + alpha * (target.x - current.x),
        current.y + alpha * (target.y - current.y),
        wrap_angle(current.yaw + alpha * wrap_angle(target.yaw - current.yaw)),
    )
