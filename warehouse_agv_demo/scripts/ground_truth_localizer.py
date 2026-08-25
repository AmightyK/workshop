#!/usr/bin/env python3
"""Publish map->odom from Gazebo ground truth for repeatable Nav2 tests."""

from __future__ import annotations

import json
import math
import threading
from copy import deepcopy

import rclpy
from geometry_msgs.msg import TransformStamped
from gz.msgs10.pose_pb2 import Pose
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.transport13 import Node as GazeboNode
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster


def yaw_from_pose(pose: Pose) -> float:
    q = pose.orientation
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


class GroundTruthLocalizer(Node):
    """Keep the Nav2 map frame aligned with the Gazebo world frame.

    Odometry and control still come from the warehouse AGV. Only localization is
    made deterministic so LiDAR costmaps can be evaluated without AMCL drift.
    """

    def __init__(self) -> None:
        super().__init__("gazebo_ground_truth_localizer")
        self.broadcaster = TransformBroadcaster(self)
        self.status_publisher = self.create_publisher(
            String, "/nav/localization_status", 10
        )
        self.last_status_ns = 0
        self.gz_node = GazeboNode()
        self.lock = threading.Lock()
        self.world_pose: Pose | None = None
        self.gz_node.subscribe(
            Pose_V, "/world/world_demo/pose/info", self._on_world_poses
        )
        self.create_subscription(Odometry, "/odom", self._on_odom, 20)
        self.get_logger().info(
            "NAV CONTROL=GAZEBO TRUTH REFERENCE; V-JEPA remains a shadow evaluator"
        )

    def _on_world_poses(self, message: Pose_V) -> None:
        for pose in message.pose:
            if pose.name == "warehouse_agv":
                with self.lock:
                    self.world_pose = pose
                return

    def _on_odom(self, odom: Odometry) -> None:
        with self.lock:
            world = self.world_pose
        if world is None:
            return

        q = odom.pose.pose.orientation
        odom_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        map_yaw = yaw_from_pose(world)
        map_to_odom_yaw = map_yaw - odom_yaw
        c, s = math.cos(map_to_odom_yaw), math.sin(map_to_odom_yaw)
        ox = odom.pose.pose.position.x
        oy = odom.pose.pose.position.y

        transform = TransformStamped()
        # Publish both the measured time and a short future allowance. Using
        # only a future-dated transform left no TF sample for the first sensor
        # frames, while using only the measured time occasionally left Nav2's
        # controller callback 10 ms ahead under GUI/GPU load.
        now = self.get_clock().now()
        transform.header.stamp = now.to_msg()
        transform.header.frame_id = "map"
        transform.child_frame_id = "odom"
        transform.transform.translation.x = world.position.x - (c * ox - s * oy)
        transform.transform.translation.y = world.position.y - (s * ox + c * oy)
        transform.transform.translation.z = 0.0
        transform.transform.rotation.z = math.sin(map_to_odom_yaw / 2.0)
        transform.transform.rotation.w = math.cos(map_to_odom_yaw / 2.0)
        future_transform = deepcopy(transform)
        future_transform.header.stamp = (
            now + Duration(seconds=0.25)
        ).to_msg()
        self.broadcaster.sendTransform([transform, future_transform])

        now_ns = self.get_clock().now().nanoseconds
        if now_ns - self.last_status_ns >= 500_000_000:
            status = String()
            status.data = json.dumps(
                {
                    "state": "ACTIVE",
                    "source": "GAZEBO_TRUTH_REFERENCE",
                    "uses_gazebo_truth": True,
                    "vjepa_shadow_mode": True,
                    "planner": "NAVFN_ASTAR",
                },
                sort_keys=True,
            )
            self.status_publisher.publish(status)
            self.last_status_ns = now_ns


def main() -> None:
    rclpy.init()
    node = GroundTruthLocalizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
