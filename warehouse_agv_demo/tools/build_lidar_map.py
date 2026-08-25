#!/usr/bin/env python3
"""Build a world-aligned occupancy map from live Gazebo LiDAR scans."""

from __future__ import annotations

import math
import os
import subprocess
import time
from pathlib import Path

import rclpy
from geometry_msgs.msg import Twist
from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.pose_pb2 import Pose
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.transport13 import Node as GazeboNode
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


ROOT = Path(__file__).resolve().parents[1]
MAP_DIR = ROOT / "maps"
MAP_STEM = MAP_DIR / "warehouse_lidar"

RESOLUTION = 0.05
MIN_X, MAX_X = -15.0, 15.0
MIN_Y, MAX_Y = -25.0, 25.0
WIDTH = math.ceil((MAX_X - MIN_X) / RESOLUTION)
HEIGHT = math.ceil((MAX_Y - MIN_Y) / RESOLUTION)

# Viewpoints avoid all fixed collisions. The scanner range is 12 m, so this
# grid overlaps enough to cover the walls, compact racks and connecting aisles.
VIEWPOINTS = (
    (-13.0, -22.0), (-7.8, -22.0), (-2.8, -22.0), (3.0, -22.0), (9.0, -22.0),
    (-13.0, -14.0), (-7.8, -14.0), (-2.8, -14.0), (3.0, -14.0), (9.0, -14.0),
    (-13.0, -3.5), (-7.8, -3.5), (-2.8, -3.5), (3.0, -3.5), (9.0, -3.5),
    (-11.5, 1.0), (0.0, 1.0), (9.0, 1.0),
    (-11.5, 7.0), (0.0, 7.0), (9.0, 7.0),
    (-12.0, 14.0), (0.0, 14.0), (10.0, 14.0),
    (-12.0, 22.0), (0.0, 22.0), (10.0, 22.0),
)


def yaw_from_pose(pose: Pose) -> float:
    q = pose.orientation
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


def cell_from_world(x: float, y: float) -> tuple[int, int] | None:
    column = int((x - MIN_X) / RESOLUTION)
    row_from_bottom = int((y - MIN_Y) / RESOLUTION)
    if 0 <= column < WIDTH and 0 <= row_from_bottom < HEIGHT:
        return column, row_from_bottom
    return None


def bresenham(
    start: tuple[int, int], end: tuple[int, int]
) -> list[tuple[int, int]]:
    x0, y0 = start
    x1, y1 = end
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    error = dx + dy
    cells = []
    while True:
        cells.append((x0, y0))
        if x0 == x1 and y0 == y1:
            return cells
        twice_error = 2 * error
        if twice_error >= dy:
            error += dy
            x0 += sx
        if twice_error <= dx:
            error += dx
            y0 += sy


class LidarMapBuilder(Node):
    def __init__(self) -> None:
        super().__init__("warehouse_lidar_map_builder")
        self.create_subscription(LaserScan, "/scan", self.on_scan, 10)
        self.velocity = self.create_publisher(Twist, "/cmd_vel", 10)
        self.gz_node = GazeboNode()
        self.people_enabled = self.gz_node.advertise(
            "/warehouse/random_people/enabled", Boolean
        )
        self.gz_node.subscribe(
            Pose_V, "/world/world_demo/pose/info", self.on_world_poses
        )
        self.latest_scan: LaserScan | None = None
        self.latest_pose: Pose | None = None
        self.scan_sequence = 0
        self.free = bytearray(WIDTH * HEIGHT)
        self.occupied = bytearray(WIDTH * HEIGHT)

    def set_people_enabled(self, enabled: bool) -> None:
        message = Boolean()
        message.data = enabled
        for _ in range(4):
            self.people_enabled.publish(message)
            time.sleep(0.05)

    def on_scan(self, message: LaserScan) -> None:
        self.latest_scan = message
        self.scan_sequence += 1

    def on_world_poses(self, message: Pose_V) -> None:
        for pose in message.pose:
            if pose.name == "warehouse_agv":
                stored = Pose()
                stored.CopyFrom(pose)
                self.latest_pose = stored
                return

    def wait_for_fresh_scan(self, scans_to_skip: int = 6) -> tuple[LaserScan, Pose]:
        target = self.scan_sequence + scans_to_skip
        deadline = time.monotonic() + 5.0
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if (
                self.scan_sequence >= target
                and self.latest_scan is not None
                and self.latest_pose is not None
            ):
                return self.latest_scan, self.latest_pose
        raise RuntimeError("Timed out waiting for a fresh /scan and world pose")

    def move_scanner(self, x: float, y: float) -> None:
        # Repeated zero commands settle the wheel controllers before the pose
        # update; the actual pose paired with each scan is still read back.
        zero = Twist()
        for _ in range(5):
            self.velocity.publish(zero)
            rclpy.spin_once(self, timeout_sec=0.03)
        request = (
            f'name: "warehouse_agv", position: {{x: {x}, y: {y}, z: 0.02}}, '
            "orientation: {w: 1}"
        )
        result = subprocess.run(
            [
                "gz", "service", "-s", "/world/world_demo/set_pose",
                "--reqtype", "gz.msgs.Pose", "--reptype", "gz.msgs.Boolean",
                "--timeout", "3000", "--req", request,
            ],
            check=True,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        if "true" not in result.stdout:
            raise RuntimeError(f"Gazebo rejected pose {x, y}: {result.stdout}")
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            pose = self.latest_pose
            if (
                pose is not None
                and math.hypot(pose.position.x - x, pose.position.y - y) < 0.20
            ):
                return
            time.sleep(0.02)
        raise RuntimeError(f"Gazebo pose feedback did not reach survey point {x, y}")

    def integrate(self, scan: LaserScan, pose: Pose) -> None:
        yaw = yaw_from_pose(pose)
        # Fixed scanner origin relative to base_link at the retracted lift pose.
        sensor_x = pose.position.x + math.cos(yaw) * -0.2063125
        sensor_y = pose.position.y + math.sin(yaw) * -0.2063125
        origin = cell_from_world(sensor_x, sensor_y)
        if origin is None:
            return

        angle = scan.angle_min
        for measured in scan.ranges:
            hit = math.isfinite(measured) and measured < scan.range_max - 0.05
            distance = measured if math.isfinite(measured) else scan.range_max
            distance = min(max(distance, scan.range_min), scan.range_max)
            ray_angle = yaw + angle
            end_x = sensor_x + math.cos(ray_angle) * distance
            end_y = sensor_y + math.sin(ray_angle) * distance
            endpoint = cell_from_world(end_x, end_y)
            angle += scan.angle_increment
            if endpoint is None:
                continue
            cells = bresenham(origin, endpoint)
            free_cells = cells[:-2] if hit and len(cells) > 2 else cells
            for column, row in free_cells:
                self.free[row * WIDTH + column] = 1
            if hit:
                column, row = cells[-1]
                self.occupied[row * WIDTH + column] = 1

    def save(self) -> None:
        MAP_DIR.mkdir(parents=True, exist_ok=True)
        image = bytearray([205]) * (WIDTH * HEIGHT)
        for row_from_bottom in range(HEIGHT):
            image_row = HEIGHT - 1 - row_from_bottom
            for column in range(WIDTH):
                source = row_from_bottom * WIDTH + column
                target = image_row * WIDTH + column
                if self.free[source]:
                    image[target] = 254
                if self.occupied[source]:
                    image[target] = 0

        pgm_path = MAP_STEM.with_suffix(".pgm")
        with pgm_path.open("wb") as stream:
            stream.write(f"P5\n{WIDTH} {HEIGHT}\n255\n".encode("ascii"))
            stream.write(image)
        MAP_STEM.with_suffix(".yaml").write_text(
            "image: warehouse_lidar.pgm\n"
            "mode: trinary\n"
            f"resolution: {RESOLUTION}\n"
            f"origin: [{MIN_X}, {MIN_Y}, 0.0]\n"
            "negate: 0\n"
            "occupied_thresh: 0.65\n"
            "free_thresh: 0.25\n",
            encoding="utf-8",
        )
        known = sum(1 for value in image if value != 205)
        occupied = sum(1 for value in image if value == 0)
        print(
            f"Saved {pgm_path}: {known / len(image):.1%} observed, "
            f"{occupied} occupied cells"
        )


def main() -> None:
    rclpy.init()
    builder = LidarMapBuilder()
    try:
        builder.wait_for_fresh_scan(1)
        # Dynamic people belong in live costmaps, never in the static map.
        builder.set_people_enabled(False)
        time.sleep(0.5)
        for index, (x, y) in enumerate(VIEWPOINTS, 1):
            builder.move_scanner(x, y)
            scan, pose = builder.wait_for_fresh_scan()
            builder.integrate(scan, pose)
            print(
                f"[{index:02d}/{len(VIEWPOINTS)}] scan at "
                f"({pose.position.x:.2f}, {pose.position.y:.2f})"
            )
        builder.save()
    finally:
        builder.set_people_enabled(True)
        if rclpy.ok():
            builder.velocity.publish(Twist())
        builder.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
