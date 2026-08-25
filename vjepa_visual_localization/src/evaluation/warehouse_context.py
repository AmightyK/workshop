"""Pure helpers for warehouse-region and obstacle commentary."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import yaml


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class RegionCenter:
    name: str
    x: float
    y: float


class WarehouseRegionIndex:
    """Describe a world coordinate using rack zones from inventory YAML."""

    def __init__(self, centers: Sequence[RegionCenter], near_radius_m: float = 4.5) -> None:
        if not centers:
            raise ValueError("at least one warehouse region is required")
        self.centers = tuple(centers)
        self.near_radius_m = float(near_radius_m)

    @classmethod
    def from_inventory(
        cls, path: str | Path, *, near_radius_m: float = 4.5
    ) -> "WarehouseRegionIndex":
        with Path(path).open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        grouped: dict[str, list[tuple[float, float]]] = {}
        for location in config["locations"].values():
            grouped.setdefault(str(location["zone"]), []).append(
                (float(location["x"]), float(location["y"]))
            )
        centers = [
            RegionCenter(
                zone,
                float(np.mean([point[0] for point in points])),
                float(np.mean([point[1] for point in points])),
            )
            for zone, points in grouped.items()
        ]
        return cls(centers, near_radius_m=near_radius_m)

    def describe(self, x: float, y: float) -> str:
        """Return a concise Vietnamese area label for a map coordinate."""
        if math.hypot(x - 14.3, y + 10.6) <= 2.2:
            return "khu sạc AGV"
        if math.hypot(x - 10.8, y + 9.0) <= 2.0:
            return "khu đóng gói"

        ranked = sorted(
            (
                (math.hypot(x - center.x, y - center.y), center.name)
                for center in self.centers
            ),
            key=lambda item: (item[0], item[1]),
        )
        nearest_distance, nearest_name = ranked[0]
        if nearest_distance <= self.near_radius_m:
            second_distance, second_name = ranked[1]
            if (
                second_distance <= self.near_radius_m
                and second_distance - nearest_distance <= 0.65
            ):
                return f"lối đi giữa khu kệ {nearest_name} và {second_name}"
            return f"khu kệ {nearest_name}"
        if y <= -21.0:
            return "hành lang phía Nam"
        if y >= 20.0:
            return "hành lang phía Bắc"
        if x <= -10.0:
            return "hành lang phía Tây"
        if x >= 10.0:
            return "hành lang phía Đông"
        return "lối đi trung tâm"


@dataclass(frozen=True)
class EntityPose:
    name: str
    x: float
    y: float


@dataclass(frozen=True)
class ObstacleReport:
    name: str
    label: str
    clearance_m: float
    center_distance_m: float | None
    bearing_rad: float | None


def scan_sector_min(
    ranges: Iterable[float],
    *,
    angle_min: float,
    angle_increment: float,
    range_min: float,
    range_max: float,
    center_rad: float = 0.0,
    half_width_rad: float = 0.5,
) -> float:
    """Return the nearest valid LiDAR range in an angular sector."""
    nearest = math.inf
    for index, value in enumerate(ranges):
        distance = float(value)
        if not math.isfinite(distance) or not range_min <= distance <= range_max:
            continue
        angle = angle_min + index * angle_increment
        if abs(wrap_angle(angle - center_rad)) <= half_width_rad:
            nearest = min(nearest, distance)
    return nearest


def obstacle_label(name: str) -> str:
    if name.startswith("random_worker_"):
        return f"người đi bộ {name.removeprefix('random_worker_')}"
    if name.startswith("road_box_static_"):
        return f"thùng tĩnh {name.removeprefix('road_box_static_')}"
    return name


def identify_forward_obstacle(
    *,
    agv_x: float,
    agv_y: float,
    agv_yaw: float,
    entities: Iterable[EntityPose],
    lidar_clearance_m: float,
    detection_distance_m: float = 2.2,
    front_half_angle_rad: float = 0.65,
) -> ObstacleReport | None:
    """Fuse a LiDAR trigger with Gazebo labels used only for commentary."""
    if not math.isfinite(lidar_clearance_m) or lidar_clearance_m > detection_distance_m:
        return None
    candidates: list[tuple[float, float, EntityPose]] = []
    for entity in entities:
        dx, dy = entity.x - agv_x, entity.y - agv_y
        distance = math.hypot(dx, dy)
        bearing = wrap_angle(math.atan2(dy, dx) - agv_yaw)
        if distance <= detection_distance_m + 1.2 and abs(bearing) <= front_half_angle_rad:
            candidates.append((distance, bearing, entity))
    if candidates:
        distance, bearing, entity = min(candidates, key=lambda item: item[0])
        return ObstacleReport(
            entity.name,
            obstacle_label(entity.name),
            float(lidar_clearance_m),
            distance,
            bearing,
        )
    return ObstacleReport(
        "unclassified_lidar_obstacle",
        "vật cản LiDAR chưa định danh (kệ hoặc tường)",
        float(lidar_clearance_m),
        None,
        None,
    )
