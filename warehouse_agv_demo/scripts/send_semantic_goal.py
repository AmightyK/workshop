#!/usr/bin/env python3
"""Resolve a warehouse object into a semantic anchor and send it to Nav2."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import rclpy
import yaml
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node


CONFIG = Path(__file__).resolve().parents[1] / "config" / "semantic_tasks.yaml"


def load_config() -> dict:
    with CONFIG.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def resolve_anchor(config: dict, object_name: str, destination: str) -> tuple[str, list[float]]:
    if object_name not in config["objects"]:
        choices = ", ".join(sorted(config["objects"]))
        raise ValueError(f"Unknown object '{object_name}'. Available: {choices}")

    if destination == "pickup":
        item = config["objects"][object_name]
        station_name, slot_name = item["location"], item["slot"]
    else:
        station_name, slot_name = "packing_station", "PACK01"

    slot = config["stations"][station_name]["slots"][slot_name]
    return slot["anchor"], slot["approach"]


def make_pose(frame_id: str, xyz_yaw: list[float], stamp) -> PoseStamped:
    x, y, yaw = map(float, xyz_yaw)
    pose = PoseStamped()
    pose.header.frame_id = frame_id
    pose.header.stamp = stamp
    pose.pose.position.x = x
    pose.pose.position.y = y
    pose.pose.orientation.z = math.sin(yaw / 2.0)
    pose.pose.orientation.w = math.cos(yaw / 2.0)
    return pose


class SemanticGoalSender(Node):
    def __init__(self) -> None:
        super().__init__("semantic_task_planner")
        self.client = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self.goal_preview = self.create_publisher(PoseStamped, "semantic_goal", 1)

    def send(self, anchor_name: str, coordinates: list[float], wait_seconds: float) -> bool:
        pose = make_pose("map", coordinates, self.get_clock().now().to_msg())
        discovery_deadline = time.monotonic() + 0.5
        while self.goal_preview.get_subscription_count() == 0 and time.monotonic() < discovery_deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
        self.goal_preview.publish(pose)
        self.get_logger().info(
            f"Resolved anchor {anchor_name}: x={pose.pose.position.x:.2f}, "
            f"y={pose.pose.position.y:.2f}"
        )
        if not self.client.wait_for_server(timeout_sec=wait_seconds):
            self.get_logger().error(
                "Nav2 action /navigate_to_pose is not available. The resolved "
                "pose was still published on /semantic_goal."
            )
            return False

        goal = NavigateToPose.Goal()
        goal.pose = pose
        future = self.client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        handle = future.result()
        if handle is None or not handle.accepted:
            self.get_logger().error(f"Nav2 rejected anchor {anchor_name}")
            return False

        self.get_logger().info(f"Nav2 accepted anchor {anchor_name}")
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        status = result_future.result().status
        self.get_logger().info(f"Nav2 finished {anchor_name} with status {status}")
        return status == 4  # action_msgs/GoalStatus.STATUS_SUCCEEDED


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--object", default="blue_box")
    parser.add_argument("--destination", choices=("pickup", "packing"), default="pickup")
    parser.add_argument("--dry-run", action="store_true", help="Resolve and print without ROS")
    parser.add_argument("--wait", type=float, default=5.0, help="Seconds to wait for Nav2")
    args = parser.parse_args()

    config = load_config()
    anchor, coordinates = resolve_anchor(config, args.object, args.destination)
    if args.dry_run:
        x, y, yaw = coordinates
        print(f"{args.object} -> {anchor}: x={x:.2f}, y={y:.2f}, yaw={yaw:.4f}")
        return

    rclpy.init()
    node = SemanticGoalSender()
    try:
        succeeded = node.send(anchor, coordinates, args.wait)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    if not succeeded:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
