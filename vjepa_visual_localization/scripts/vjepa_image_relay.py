#!/usr/bin/env python3
"""Threaded latest-frame ROS 2 DDS relay for a remote V-JEPA computer."""

from __future__ import annotations

import argparse
import threading
import time
from typing import Any

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import Image


DDS_IMAGE_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)


class LatestFrameSlot:
    """Thread-safe one-frame mailbox; superseded camera frames are dropped."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sequence = 0
        self._message: Any | None = None

    def put(self, message: Any) -> int:
        with self._lock:
            self._sequence += 1
            self._message = message
            return self._sequence

    def latest_after(self, sequence: int) -> tuple[int, Any] | None:
        with self._lock:
            if self._message is None or self._sequence <= sequence:
                return None
            return self._sequence, self._message


class VjepaImageRelay(Node):
    """Receive Gazebo camera callbacks and publish the latest frame via DDS."""

    def __init__(self, *, input_topic: str, output_topic: str, fps: float) -> None:
        super().__init__("vjepa_image_dds_relay")
        if fps <= 0.0:
            raise ValueError("fps must be positive")
        if input_topic == output_topic:
            raise ValueError("input and output image topics must differ")

        self.input_topic = input_topic
        self.output_topic = output_topic
        self.fps = float(fps)
        self.slot = LatestFrameSlot()
        self.received_frames = 0
        self.published_frames = 0
        self._stop = threading.Event()
        self.publisher = self.create_publisher(Image, output_topic, DDS_IMAGE_QOS)
        self.subscription = self.create_subscription(
            Image, input_topic, self._on_image, DDS_IMAGE_QOS
        )
        self.worker = threading.Thread(
            target=self._publish_loop,
            name="vjepa-dds-image-publisher",
            daemon=True,
        )
        self.worker.start()
        self.get_logger().info(
            f"ROS 2 DDS image relay: {input_topic} -> {output_topic} at {fps:g} FPS | "
            "QoS=BEST_EFFORT/VOLATILE/KEEP_LAST(1)"
        )

    def _on_image(self, message: Image) -> None:
        # The DDS callback performs no image conversion or GPU work. Replacing
        # this reference is intentionally O(1); the worker publishes only the
        # newest complete ROS Image and preserves its Gazebo header timestamp.
        self.received_frames += 1
        self.slot.put(message)

    def _publish_loop(self) -> None:
        period = 1.0 / self.fps
        next_deadline = time.monotonic() + period
        last_sequence = 0
        last_report = time.monotonic()
        last_received_report = 0
        last_published_report = 0
        while not self._stop.is_set():
            wait_sec = max(0.0, next_deadline - time.monotonic())
            if self._stop.wait(wait_sec):
                break
            now = time.monotonic()
            latest = self.slot.latest_after(last_sequence)
            if latest is not None:
                last_sequence, message = latest
                self.publisher.publish(message)
                self.published_frames += 1

            if now - last_report >= 5.0:
                received_delta = self.received_frames - last_received_report
                published_delta = self.published_frames - last_published_report
                dropped_delta = max(0, received_delta - published_delta)
                self.get_logger().info(
                    f"DDS image flow: received={received_delta}, "
                    f"published={published_delta}, superseded={dropped_delta} / 5s"
                )
                last_received_report = self.received_frames
                last_published_report = self.published_frames
                last_report = now

            next_deadline += period
            if next_deadline < now:
                next_deadline = now + period

    def close(self) -> None:
        self._stop.set()
        if self.worker.is_alive():
            self.worker.join(timeout=2.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-topic", default="/camera")
    parser.add_argument("--output-topic", default="/vjepa/camera/image_raw")
    parser.add_argument("--fps", type=float, default=32.0)
    args, ros_args = parser.parse_known_args()
    if args.fps <= 0.0:
        parser.error("--fps must be positive")
    if args.input_topic == args.output_topic:
        parser.error("--input-topic and --output-topic must differ")

    rclpy.init(args=ros_args)
    node = VjepaImageRelay(
        input_topic=args.input_topic,
        output_topic=args.output_topic,
        fps=args.fps,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
