#!/usr/bin/env python3
"""Record ROS camera video with Gazebo ground truth for visual-map building."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import threading
import time
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import rclpy
from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.pose_pb2 import Pose
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.transport13 import Node as GazeboNode
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from src.data.ros_image import image_message_to_rgb, message_timestamp_sec


def yaw_from_pose(pose: Pose) -> float:
    q = pose.orientation
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


class MappingRunRecorder(Node):
    """Ground truth is deliberately confined to this mapping-only process."""

    def __init__(
        self,
        *,
        run_dir: Path,
        camera_topic: str,
        model_name: str,
        fps: float,
        duration_sec: float | None,
        overwrite: bool,
        park_people: bool,
    ) -> None:
        super().__init__("vjepa_mapping_run_recorder")
        run_dir.mkdir(parents=True, exist_ok=True)
        self.video_path = run_dir / "video.mp4"
        self.pose_path = run_dir / "poses.csv"
        self.camera_metadata_path = run_dir / "camera_metadata.json"
        if not overwrite and (self.video_path.exists() or self.pose_path.exists()):
            raise FileExistsError(f"mapping run already exists: {run_dir}; use --overwrite")
        self.model_name = model_name
        self.fps = fps
        self.duration_sec = duration_sec
        self.frame_index = 0
        self.first_camera_timestamp: float | None = None
        self.writer: cv2.VideoWriter | None = None
        self.pose_stream = self.pose_path.open("w", newline="", encoding="utf-8")
        self.pose_writer = csv.writer(self.pose_stream)
        self.pose_writer.writerow(["timestamp", "x", "y", "z", "yaw"])
        self.latest_pose: Pose | None = None
        self.pose_lock = threading.Lock()
        self.gz_node = GazeboNode()
        self.gz_node.subscribe(Pose_V, "/world/world_demo/pose/info", self.on_poses)
        self.people_control = self.gz_node.advertise(
            "/warehouse/random_people/enabled", Boolean
        )
        self.park_people = park_people
        if park_people:
            self._publish_people_enabled(False, wait_for_discovery=True)
        self.create_subscription(Image, camera_topic, self.on_image, qos_profile_sensor_data)
        self.get_logger().info(f"Recording {camera_topic} and mapping-only Gazebo pose to {run_dir}")

    def _publish_people_enabled(
        self, enabled: bool, *, wait_for_discovery: bool = False
    ) -> None:
        if wait_for_discovery:
            deadline = time.monotonic() + 2.0
            while not self.people_control.has_connections() and time.monotonic() < deadline:
                time.sleep(0.05)
        message = Boolean()
        message.data = enabled
        for _ in range(4):
            self.people_control.publish(message)
            time.sleep(0.03)

    def on_poses(self, message: Pose_V) -> None:
        for pose in message.pose:
            if pose.name == self.model_name:
                stored = Pose()
                stored.CopyFrom(pose)
                with self.pose_lock:
                    self.latest_pose = stored
                return

    def on_image(self, message: Image) -> None:
        with self.pose_lock:
            pose = self.latest_pose
        if pose is None:
            return
        source_timestamp = message_timestamp_sec(message)
        if source_timestamp <= 0.0:
            source_timestamp = self.get_clock().now().nanoseconds * 1e-9
        if self.first_camera_timestamp is None:
            self.first_camera_timestamp = source_timestamp
        elapsed = source_timestamp - self.first_camera_timestamp
        scheduled_timestamp = self.frame_index / self.fps
        if elapsed + 0.5 / self.fps < scheduled_timestamp:
            return
        rgb = image_message_to_rgb(message)
        if self.writer is None:
            height, width = rgb.shape[:2]
            self.writer = cv2.VideoWriter(
                str(self.video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.fps,
                (width, height),
            )
            if not self.writer.isOpened():
                raise RuntimeError(f"could not open video writer: {self.video_path}")
            divisor = math.gcd(width, height)
            camera_metadata = {
                "ros_image": {
                    "width": int(message.width),
                    "height": int(message.height),
                    "encoding": str(message.encoding),
                    "step_bytes": int(message.step),
                    "aspect_ratio": width / height,
                    "aspect_label": f"{width // divisor}:{height // divisor}",
                    "is_square": width == height,
                },
                "recorded_video": {
                    "width": width,
                    "height": height,
                    "fps": self.fps,
                    "aspect_ratio": width / height,
                },
            }
            self.camera_metadata_path.write_text(
                json.dumps(camera_metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            self.get_logger().info(
                f"[CAMERA] raw={width}x{height}, "
                f"aspect={width // divisor}:{height // divisor}, square={width == height}"
            )

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        # MP4 has a fixed frame rate. Fill each elapsed output slot so frame
        # index, pose CSV timestamp and visual content always share one timebase,
        # even if ROS drops an occasional camera callback under GPU load.
        while scheduled_timestamp <= elapsed + 0.5 / self.fps:
            self.writer.write(bgr)
            self.pose_writer.writerow(
                [
                    f"{scheduled_timestamp:.6f}",
                    f"{pose.position.x:.9f}",
                    f"{pose.position.y:.9f}",
                    f"{pose.position.z:.9f}",
                    f"{yaw_from_pose(pose):.9f}",
                ]
            )
            self.frame_index += 1
            if self.frame_index % max(1, round(self.fps)) == 0:
                self.pose_stream.flush()
                self.get_logger().info(f"recorded {scheduled_timestamp:.1f}s")
            if self.duration_sec is not None and scheduled_timestamp >= self.duration_sec:
                rclpy.shutdown()
                return
            scheduled_timestamp = self.frame_index / self.fps

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
        self.pose_stream.close()
        if self.park_people:
            self._publish_people_enabled(True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--camera-topic", default="/camera")
    parser.add_argument("--model-name", default="warehouse_agv")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-people", action="store_true")
    args, ros_args = parser.parse_known_args()
    if args.fps <= 0.0 or (args.duration is not None and args.duration <= 0.0):
        parser.error("fps and duration must be positive")
    rclpy.init(args=ros_args)
    node = MappingRunRecorder(
        run_dir=Path(args.output),
        camera_topic=args.camera_topic,
        model_name=args.model_name,
        fps=args.fps,
        duration_sec=args.duration,
        overwrite=args.overwrite,
        park_people=not args.keep_people,
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
