#!/usr/bin/env python3
"""Publish a stop gate only for the moving Gazebo worker models."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass

import rclpy
from gz.msgs10.pose_pb2 import Pose
from gz.transport13 import Node as GazeboNode
from rclpy.node import Node
from std_msgs.msg import Bool


WORKER_NAMES = tuple(f"random_worker_{index}" for index in range(1, 6))
POSE_TIMEOUT_S = 0.50


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float = 0.0


KNOWN_WORKER_CROSSINGS = {
    "random_worker_4": Pose2D(7.0, -10.0),
    "random_worker_5": Pose2D(-1.5, -5.0),
}


def yaw_from_pose(message: Pose) -> float:
    orientation = message.orientation
    return math.atan2(
        2.0 * (
            orientation.w * orientation.z
            + orientation.x * orientation.y
        ),
        1.0 - 2.0 * (orientation.y**2 + orientation.z**2),
    )


def worker_relative_to_agv(
    agv: Pose2D, worker: Pose2D
) -> tuple[float, float]:
    """Return forward and left offsets in the AGV body frame."""
    dx = worker.x - agv.x
    dy = worker.y - agv.y
    cosine = math.cos(agv.yaw)
    sine = math.sin(agv.yaw)
    return cosine * dx + sine * dy, -sine * dx + cosine * dy


def worker_requires_stop(
    agv: Pose2D, worker: Pose2D, *, stopping: bool = False
) -> bool:
    """Apply a forward crossing corridor plus an all-around close envelope."""
    forward, lateral = worker_relative_to_agv(agv, worker)
    distance = math.hypot(forward, lateral)
    # Release thresholds are deliberately wider than entry thresholds so a
    # person on the boundary cannot chatter the base between stop and go.
    if stopping:
        return distance <= 0.90 or (
            -0.10 <= forward <= 1.85 and abs(lateral) <= 0.70
        )
    return distance <= 0.75 or (
        0.0 <= forward <= 1.60 and abs(lateral) <= 0.55
    )


def known_crossing_requires_stop(
    agv: Pose2D,
    worker: Pose2D,
    crossing: Pose2D,
    *,
    stopping: bool = False,
) -> bool:
    """Use the scripted worker route to stop before a shared intersection."""
    crossing_forward, crossing_lateral = worker_relative_to_agv(agv, crossing)
    worker_distance = math.hypot(
        worker.x - crossing.x, worker.y - crossing.y
    )
    if stopping:
        return (
            -0.35 <= crossing_forward <= 2.40
            and abs(crossing_lateral) <= 1.05
            and worker_distance <= 1.55
        )
    return (
        0.0 <= crossing_forward <= 2.10
        and abs(crossing_lateral) <= 0.90
        and worker_distance <= 1.20
    )


class PersonSafetyMonitor(Node):
    def __init__(self) -> None:
        super().__init__("person_safety_monitor")
        self.lock = threading.Lock()
        self.agv: tuple[Pose2D, float] | None = None
        self.workers: dict[str, tuple[Pose2D, float]] = {}
        self.stopping = False
        self.publisher = self.create_publisher(Bool, "/warehouse/person_stop", 10)
        self.gz_node = GazeboNode()
        if not self.gz_node.subscribe(
            Pose, "/model/warehouse_agv/pose", self._on_agv
        ):
            raise RuntimeError("could not subscribe warehouse_agv Gazebo pose")
        for name in WORKER_NAMES:
            if not self.gz_node.subscribe(
                Pose,
                f"/model/{name}/pose",
                lambda message, worker_name=name: self._on_worker(
                    worker_name, message
                ),
            ):
                raise RuntimeError(f"could not subscribe {name} Gazebo pose")
        self.create_timer(0.05, self._publish_gate)
        self.get_logger().info(
            "Worker-only stop/resume gate active on /warehouse/person_stop"
        )

    @staticmethod
    def _pose2d(message: Pose) -> Pose2D:
        return Pose2D(
            float(message.position.x),
            float(message.position.y),
            yaw_from_pose(message),
        )

    def _on_agv(self, message: Pose) -> None:
        if message.name != "warehouse_agv":
            return
        with self.lock:
            self.agv = self._pose2d(message), time.monotonic()

    def _on_worker(self, name: str, message: Pose) -> None:
        if message.name != name:
            return
        with self.lock:
            self.workers[name] = self._pose2d(message), time.monotonic()

    def _publish_gate(self) -> None:
        now = time.monotonic()
        with self.lock:
            agv_sample = self.agv
            worker_samples = tuple(self.workers.items())
            previous = self.stopping
            should_stop = False
            blocking_name = ""
            if agv_sample is not None and now - agv_sample[1] <= POSE_TIMEOUT_S:
                agv = agv_sample[0]
                for name, (worker, timestamp) in worker_samples:
                    if now - timestamp > POSE_TIMEOUT_S:
                        continue
                    crossing = KNOWN_WORKER_CROSSINGS.get(name)
                    crossing_blocked = (
                        crossing is not None
                        and known_crossing_requires_stop(
                            agv, worker, crossing, stopping=previous
                        )
                    )
                    if (
                        crossing_blocked
                        or worker_requires_stop(agv, worker, stopping=previous)
                    ):
                        should_stop = True
                        blocking_name = name
                        break
            self.stopping = should_stop

        self.publisher.publish(Bool(data=should_stop))
        if should_stop != previous:
            if should_stop:
                self.get_logger().warn(
                    f"Stopping for {blocking_name}; retaining current Nav2 path"
                )
            else:
                self.get_logger().info(
                    "Worker cleared; resuming retained velocity/path"
                )


def main() -> None:
    rclpy.init()
    node = PersonSafetyMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.publisher.publish(Bool(data=False))
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
