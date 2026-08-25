from __future__ import annotations

import math

from src.evaluation.warehouse_context import (
    EntityPose,
    RegionCenter,
    WarehouseRegionIndex,
    identify_forward_obstacle,
    scan_sector_min,
)


def test_region_description_handles_named_and_general_areas() -> None:
    index = WarehouseRegionIndex(
        [RegionCenter("Q", 6.8, -13.0), RegionCenter("T", 1.2, -13.0)]
    )
    assert index.describe(14.0, -10.6) == "khu sạc AGV"
    assert index.describe(6.7, -13.1) == "khu kệ Q"
    assert index.describe(-12.0, 0.0) == "hành lang phía Tây"


def test_scan_sector_ignores_invalid_and_side_ranges() -> None:
    ranges = [0.4, math.inf, 1.2, 0.7, 0.3]
    nearest = scan_sector_min(
        ranges,
        angle_min=-1.0,
        angle_increment=0.5,
        range_min=0.1,
        range_max=10.0,
        half_width_rad=0.55,
    )
    assert nearest == 0.7


def test_obstacle_identification_uses_lidar_gate_and_forward_entity() -> None:
    entities = [
        EntityPose("road_box_static_2", 1.5, 0.1),
        EntityPose("random_worker_1", -0.5, 0.0),
    ]
    report = identify_forward_obstacle(
        agv_x=0.0,
        agv_y=0.0,
        agv_yaw=0.0,
        entities=entities,
        lidar_clearance_m=1.1,
    )
    assert report is not None
    assert report.name == "road_box_static_2"
    assert report.label == "thùng tĩnh 2"
    assert identify_forward_obstacle(
        agv_x=0.0,
        agv_y=0.0,
        agv_yaw=0.0,
        entities=entities,
        lidar_clearance_m=3.0,
    ) is None
