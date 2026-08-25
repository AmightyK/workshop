#!/usr/bin/env python3
"""Move the five-stage lift while preserving exact scissor geometry."""

from __future__ import annotations

import argparse
import math
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64


STAGE_COUNT = 5
BAR_LENGTH = 0.28875
THETA_FOLDED = 0.180797
THETA_RAISED = 1.10
HALF_BAR = BAR_LENGTH / 2.0


class ScissorLiftController(Node):
    def __init__(self) -> None:
        super().__init__("scissor_lift_controller")
        self.platform = self.create_publisher(
            Float64, "/gripper/lift_position", 10
        )
        self.angle = self.create_publisher(
            Float64, "/gripper/scissor_angle_position", 10
        )
        self.slider = self.create_publisher(
            Float64, "/gripper/scissor_slider_position", 10
        )
        self.stages = [
            self.create_publisher(
                Float64, f"/gripper/scissor_stage_{index}_position", 10
            )
            for index in range(1, STAGE_COUNT + 1)
        ]

    def wait_for_bridge(self, timeout: float = 4.0) -> None:
        publishers = [self.platform, self.angle, self.slider, *self.stages]
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if all(pub.get_subscription_count() > 0 for pub in publishers):
                return
            rclpy.spin_once(self, timeout_sec=0.05)
        raise RuntimeError("ROS-Gazebo lift bridge is unavailable; run ./run_bridge.sh")

    def publish_geometry(self, theta: float) -> None:
        # Five equal X stages: total height is N * L * sin(theta).
        lift = STAGE_COUNT * BAR_LENGTH * (
            math.sin(theta) - math.sin(THETA_FOLDED)
        )
        angle = theta - THETA_FOLDED
        slider = HALF_BAR * (
            math.cos(THETA_FOLDED) - math.cos(theta)
        )

        self.platform.publish(Float64(data=lift))
        self.angle.publish(Float64(data=angle))
        self.slider.publish(Float64(data=slider))
        # Each visual stage follows the centre of its own X cell.
        stage_positions = tuple(
            lift * (2 * index - 1) / (2 * STAGE_COUNT)
            for index in range(1, STAGE_COUNT + 1)
        )
        for publisher, position in zip(self.stages, stage_positions):
            publisher.publish(Float64(data=position))

    def move(self, direction: str, duration: float, rate: float) -> None:
        start = THETA_FOLDED if direction == "up" else THETA_RAISED
        finish = THETA_RAISED if direction == "up" else THETA_FOLDED
        started = time.monotonic()
        period = 1.0 / rate
        while True:
            elapsed = time.monotonic() - started
            ratio = min(1.0, elapsed / duration)
            # Smooth acceleration while retaining the exact linkage equations.
            smooth = ratio * ratio * (3.0 - 2.0 * ratio)
            theta = start + (finish - start) * smooth
            self.publish_geometry(theta)
            rclpy.spin_once(self, timeout_sec=0.0)
            if ratio >= 1.0:
                break
            time.sleep(period)

        # Repeat the final command so every Gazebo controller receives it.
        for _ in range(10):
            self.publish_geometry(finish)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(period)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("direction", choices=("up", "down"), nargs="?", default="up")
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--rate", type=float, default=30.0)
    args = parser.parse_args()
    if args.duration <= 0.0 or args.rate <= 0.0:
        parser.error("--duration and --rate must be positive")

    rclpy.init()
    controller = ScissorLiftController()
    try:
        controller.wait_for_bridge()
        controller.move(args.direction, args.duration, args.rate)
        print(f"Scissor lift moved {args.direction} with connected-link kinematics.")
    finally:
        controller.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
