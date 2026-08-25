#!/usr/bin/env python3
"""Use camera-only V-JEPA poses as the global localization source for Nav2.

Gazebo world pose is intentionally absent. Wheel odometry supplies the smooth
short-term odom->base_link motion required by the controller; timestamp-aligned
V-JEPA observations correct the global map->odom transform.
"""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster

from nav_pose_math import (
    Pose2D,
    blend_pose,
    map_to_odom,
    planar_distance,
    wrap_angle,
    yaw_distance,
)


@dataclass(frozen=True)
class TimedPose:
    timestamp: float
    pose: Pose2D


def stamp_seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def yaw_from_quaternion(quaternion) -> float:
    return math.atan2(
        2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
        1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z),
    )


class VjepaNavLocalizer(Node):
    def __init__(self) -> None:
        super().__init__("vjepa_nav_localizer")
        # V-JEPA provides global, lower-rate corrections while wheel odometry
        # remains the smooth short-term motion source.  Keep corrections
        # conservative: an image retrieval mismatch must never drag Nav2's
        # entire map frame to another aisle.
        self.declare_parameter("correction_alpha", 0.45)
        self.declare_parameter("max_sync_delta_sec", 0.35)
        self.declare_parameter("max_translation_correction_m", 0.90)
        self.declare_parameter("max_yaw_correction_rad", 0.55)
        self.declare_parameter("stale_odom_translation_m", 0.16)
        self.declare_parameter("stale_odom_yaw_rad", 0.16)
        self.declare_parameter("stale_visual_translation_m", 0.025)
        self.declare_parameter("stale_visual_yaw_rad", 0.025)
        self.correction_alpha = float(self.get_parameter("correction_alpha").value)
        self.max_sync_delta = float(self.get_parameter("max_sync_delta_sec").value)
        self.max_translation_correction = float(
            self.get_parameter("max_translation_correction_m").value
        )
        self.max_yaw_correction = float(
            self.get_parameter("max_yaw_correction_rad").value
        )
        self.stale_odom_translation = float(
            self.get_parameter("stale_odom_translation_m").value
        )
        self.stale_odom_yaw = float(self.get_parameter("stale_odom_yaw_rad").value)
        self.stale_visual_translation = float(
            self.get_parameter("stale_visual_translation_m").value
        )
        self.stale_visual_yaw = float(
            self.get_parameter("stale_visual_yaw_rad").value
        )

        self.odom_history: deque[TimedPose] = deque(maxlen=1200)
        self.transform: Pose2D | None = None
        self.anchor_transform: Pose2D | None = None
        self.last_seen_visual: TimedPose | None = None
        self.last_seen_visual_odom: TimedPose | None = None
        self.accepted_measurements = 0
        self.rejected_measurements = 0
        self.broadcaster = TransformBroadcaster(self)
        self.status_publisher = self.create_publisher(
            String, "/nav/localization_status", 10
        )
        # Keep the old diagnostic topic for scripts from the earlier prototype.
        self.legacy_status_publisher = self.create_publisher(
            String, "/vjepa_nav/status", 10
        )
        self.create_subscription(Odometry, "/odom", self.on_odom, 50)
        self.create_subscription(
            PoseWithCovarianceStamped, "/vjepa_pose", self.on_vjepa, 20
        )
        self.get_logger().info(
            "NAV LOCALIZATION=V-JEPA: /vjepa_pose + short-term /odom -> map->odom; "
            "Gazebo truth is not subscribed"
        )

    @staticmethod
    def odom_pose(message: Odometry) -> Pose2D:
        pose = message.pose.pose
        return Pose2D(
            float(pose.position.x),
            float(pose.position.y),
            yaw_from_quaternion(pose.orientation),
        )

    @staticmethod
    def visual_pose(message: PoseWithCovarianceStamped) -> Pose2D:
        pose = message.pose.pose
        return Pose2D(
            float(pose.position.x),
            float(pose.position.y),
            yaw_from_quaternion(pose.orientation),
        )

    def nearest_odom(self, timestamp: float) -> TimedPose | None:
        if not self.odom_history:
            return None
        candidate = min(
            self.odom_history, key=lambda sample: abs(sample.timestamp - timestamp)
        )
        if abs(candidate.timestamp - timestamp) > self.max_sync_delta:
            return None
        return candidate

    def publish_status(self, state: str, **values) -> None:
        message = String()
        message.data = json.dumps(
            {
                "state": state,
                "source": "V-JEPA_CAMERA",
                "uses_gazebo_truth": False,
                "planner": "NAVFN_ASTAR",
                "accepted_measurements": self.accepted_measurements,
                "rejected_measurements": self.rejected_measurements,
                **values,
            },
            sort_keys=True,
        )
        self.status_publisher.publish(message)
        self.legacy_status_publisher.publish(message)

    def on_odom(self, message: Odometry) -> None:
        timestamp = stamp_seconds(message.header.stamp)
        if timestamp <= 0.0:
            timestamp = self.get_clock().now().nanoseconds * 1e-9
        self.odom_history.append(TimedPose(timestamp, self.odom_pose(message)))
        if self.transform is not None:
            self.broadcast(message.header.stamp)

    def on_vjepa(self, message: PoseWithCovarianceStamped) -> None:
        timestamp = stamp_seconds(message.header.stamp)
        odom = self.nearest_odom(timestamp)
        visual = TimedPose(timestamp, self.visual_pose(message))
        if odom is None:
            self.rejected_measurements += 1
            self.publish_status("WAITING_SYNC", visual_timestamp=timestamp)
            return

        if (
            self.last_seen_visual is not None
            and timestamp <= self.last_seen_visual.timestamp
        ):
            return
        previous_visual = self.last_seen_visual
        previous_odom = self.last_seen_visual_odom
        # Track consecutive camera estimates even when one is rejected.  This
        # makes the stale-frame gate describe what the camera just did instead
        # of comparing with an arbitrarily old accepted observation.
        self.last_seen_visual = visual
        self.last_seen_visual_odom = odom
        if previous_visual is not None and previous_odom is not None:
            odom_translation = planar_distance(previous_odom.pose, odom.pose)
            odom_yaw = yaw_distance(previous_odom.pose, odom.pose)
            visual_translation = planar_distance(previous_visual.pose, visual.pose)
            visual_yaw = yaw_distance(previous_visual.pose, visual.pose)
            odom_moved = (
                odom_translation >= self.stale_odom_translation
                or odom_yaw >= self.stale_odom_yaw
            )
            visual_held = (
                visual_translation <= self.stale_visual_translation
                and visual_yaw <= self.stale_visual_yaw
            )
            if odom_moved and visual_held:
                self.rejected_measurements += 1
                self.publish_status(
                    "STALE_VISUAL_HOLD",
                    visual_timestamp=timestamp,
                    sync_delta_sec=abs(odom.timestamp - timestamp),
                    odom_translation_m=odom_translation,
                    visual_translation_m=visual_translation,
                )
                return

        target = map_to_odom(visual.pose, odom.pose)
        correction_m = 0.0
        correction_yaw = 0.0
        anchor_drift_m = 0.0
        anchor_drift_yaw = 0.0
        if self.transform is None:
            self.transform = target
            self.anchor_transform = target
            state = "INITIALIZED"
        else:
            if self.anchor_transform is None:  # defensive; initialized together
                self.anchor_transform = self.transform
            anchor_drift_m = planar_distance(self.anchor_transform, target)
            anchor_drift_yaw = yaw_distance(self.anchor_transform, target)
            # Wheel odometry is deliberately only the high-rate *relative*
            # motion source. Its skid-steer error accumulates over a long
            # warehouse route, so the valid map->odom correction must be
            # allowed to move away from the startup anchor.  The former fixed
            # 0.55 m anchor gate rejected every useful V-JEPA correction once
            # odometry drift crossed that total distance and eventually sent
            # Nav2 down the wrong aisle.  Outliers are instead bounded against
            # the previous accepted transform below; the temporal retriever
            # has already constrained candidates against the previous image.
            correction_m = planar_distance(self.transform, target)
            correction_yaw = yaw_distance(self.transform, target)
            if (
                correction_m > self.max_translation_correction
                or correction_yaw > self.max_yaw_correction
            ):
                self.rejected_measurements += 1
                self.publish_status(
                    "REJECTED_OUTLIER",
                    visual_timestamp=timestamp,
                    correction_m=correction_m,
                    correction_yaw_rad=correction_yaw,
                    anchor_drift_m=anchor_drift_m,
                    anchor_drift_yaw_rad=anchor_drift_yaw,
                )
                if self.rejected_measurements % 10 == 1:
                    self.get_logger().warning(
                        "Rejected V-JEPA correction outlier: "
                        f"correction={correction_m:.2f}m/"
                        f"{correction_yaw:.2f}rad"
                    )
                return
            self.transform = blend_pose(
                self.transform, target, self.correction_alpha
            )
            state = "ACTIVE"

        self.accepted_measurements += 1
        self.publish_status(
            state,
            visual_timestamp=timestamp,
            sync_delta_sec=abs(odom.timestamp - timestamp),
            correction_m=correction_m,
            correction_yaw_rad=correction_yaw,
            anchor_drift_m=anchor_drift_m,
            anchor_drift_yaw_rad=anchor_drift_yaw,
            vjepa_x=visual.pose.x,
            vjepa_y=visual.pose.y,
            map_to_odom_x=self.transform.x,
            map_to_odom_y=self.transform.y,
        )
        self.get_logger().info(
            f"[{state}] V-JEPA=({visual.pose.x:.2f},{visual.pose.y:.2f}) "
            f"map->odom=({self.transform.x:.2f},{self.transform.y:.2f},"
            f"{self.transform.yaw:.2f}) correction={correction_m:.2f}m"
        )

    def broadcast(self, stamp) -> None:
        if self.transform is None:
            return
        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = "map"
        transform.child_frame_id = "odom"
        transform.transform.translation.x = self.transform.x
        transform.transform.translation.y = self.transform.y
        transform.transform.rotation.z = math.sin(self.transform.yaw / 2.0)
        transform.transform.rotation.w = math.cos(self.transform.yaw / 2.0)
        self.broadcaster.sendTransform(transform)


def main() -> None:
    rclpy.init()
    node = VjepaNavLocalizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
