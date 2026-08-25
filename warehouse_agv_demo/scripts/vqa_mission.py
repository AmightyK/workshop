#!/usr/bin/env python3
"""Camera-gated Nav2 pickup and physical Gazebo delivery mission.

Nav2 owns coarse aisle motion. Once it reaches a semantic shelf anchor, an
image-based controller owns the low-speed final approach. A Gazebo
DetachableJoint stands in for the pressure/contact-confirmed vacuum interface
that would be used on the real robot; this file never teleports a payload.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
import yaml
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, Twist
from gz.msgs10.double_pb2 import Double
from gz.msgs10.empty_pb2 import Empty
from gz.msgs10.pose_pb2 import Pose
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.msgs10.stringmsg_pb2 import StringMsg
from gz.msgs10.twist_pb2 import Twist as GazeboTwist
from gz.transport13 import Node as GazeboNode
from nav_msgs.msg import Odometry
from nav2_msgs.action import NavigateToPose
from nav2_msgs.srv import Toggle
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import Float64, String

from vqa_oracle import answer as answer_vqa
from grasp_retry import grasp_attempts


CONFIG = Path(__file__).resolve().parents[1] / "config" / "semantic_tasks.yaml"
PIPELINE_CONFIG = Path(__file__).resolve().parents[1] / "config" / "pipeline.yaml"
NAV_TO_POSE_BT = (
    Path(__file__).resolve().parents[1] / "config" / "navigate_to_pose_no_spin.xml"
)
DEFAULT_CAMERA_EVIDENCE = (
    Path(__file__).resolve().parents[1]
    / "screenshots"
    / "vqa_storage_A_blue_detection.png"
)
SCISSOR_STAGE_COUNT = 5
SCISSOR_BAR_LENGTH = 0.28875
SCISSOR_THETA_FOLDED = 0.180797
SCISSOR_THETA_RAISED = 1.10
SCISSOR_LIFT_MAX = SCISSOR_STAGE_COUNT * SCISSOR_BAR_LENGTH * (
    math.sin(SCISSOR_THETA_RAISED) - math.sin(SCISSOR_THETA_FOLDED)
)

# The payload is intentionally smaller than the flat fork carriage.  Keep the
# dimensions here as a hard physical safety contract: a successful
# DetachableJoint state alone does not prove that the carton is actually
# sitting on the tray (it can still be attached at a bad lateral offset).
TRAY_SIZE_X_M = 0.394625
TRAY_SIZE_Y_M = 0.30625
TRAY_PLATFORM_THICKNESS_M = 0.025
PAYLOAD_SIZE_X_M = 0.16
PAYLOAD_SIZE_Y_M = 0.19
PAYLOAD_SIZE_Z_M = 0.18
TRAY_FIT_MARGIN_M = 0.005
PACKING_SURFACE_Z_M = 0.055


def release_approach_pose(
    drop_center: list[float],
    drop_yaw: float,
    payload_forward_m: float,
    payload_lateral_m: float,
    slide_extension_m: float,
) -> list[float]:
    """Return the AGV pose that puts the carton centre on ``drop_center``.

    The fork moves forward while the carton is still attached.  Navigating the
    AGV to the painted P centre and then extending the fork therefore releases
    the carton one fork-length away from the centre.  This pose compensates for
    that extension and for the measured residual offset of the carton on the
    tray; it keeps the correction fully in the normal Nav2 motion pipeline.
    """
    release_forward = float(payload_forward_m) + float(slide_extension_m)
    release_lateral = float(payload_lateral_m)
    cos_yaw = math.cos(float(drop_yaw))
    sin_yaw = math.sin(float(drop_yaw))
    return [
        float(drop_center[0]) - cos_yaw * release_forward + sin_yaw * release_lateral,
        float(drop_center[1]) - sin_yaw * release_forward - cos_yaw * release_lateral,
        float(drop_yaw),
    ]


def print_target_card(answer: dict) -> None:
    tty = sys.stdout.isatty()
    cyan = "\033[36m" if tty else ""
    green = "\033[32m" if tty else ""
    bold = "\033[1m" if tty else ""
    reset = "\033[0m" if tty else ""
    print(f"{cyan}{bold}╭─ CAMERA / VQA PICK TARGET ──────────────────────────────╮{reset}")
    print(f"{cyan}│{reset} Storage : {bold}{answer['storage']}{reset}")
    print(f"{cyan}│{reset} Slot    : {bold}{answer['slot']}{reset}")
    print(f"{cyan}│{reset} Model   : {answer['model']}")
    print(f"{cyan}│{reset} Anchor  : {answer['pickup_anchor']} "
          f"({answer['pickup_pose'][0]:.2f}, {answer['pickup_pose'][1]:.2f})")
    print(f"{cyan}│{reset} VQA     : {green}resolved · confidence 1.00{reset}")
    print(f"{cyan}{bold}╰─────────────────────────────────────────────────────────╯{reset}")


class GazeboPayload:
    """Physical Gazebo suction adapter backed by DetachableJoint."""

    def __init__(self) -> None:
        self.node = GazeboNode()
        self.observed: dict[str, Pose] = {}
        self.lock = threading.Lock()
        self.attachment_state: dict[str, bool] = {}
        self.attach_publishers = {}
        self.detach_publishers = {}
        self.descent_publishers = {}
        self.node.subscribe(Pose_V, "/world/world_demo/pose/info", self._on_poses)

    def _on_poses(self, message: Pose_V) -> None:
        with self.lock:
            for pose in message.pose:
                self.observed[pose.name] = pose

    def wait_for(self, *models: str, timeout: float = 8.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.lock:
                if all(model in self.observed for model in models):
                    return
            time.sleep(0.05)
        raise RuntimeError(f"Gazebo pose feedback missing: {models}")

    def verify_slot(
        self,
        model: str,
        expected: list[float],
        planar_tolerance: float = 0.08,
        height_tolerance: float = 0.05,
    ) -> float:
        with self.lock:
            pose = self.observed[model]
        expected_x, expected_y, expected_z = map(float, expected)
        planar_error = math.hypot(
            pose.position.x - expected_x,
            pose.position.y - expected_y,
        )
        height_error = abs(pose.position.z - expected_z)
        error = math.dist(
            (pose.position.x, pose.position.y, pose.position.z),
            (expected_x, expected_y, expected_z),
        )
        # A carton may settle a few centimetres forward on the shelf. Keep a
        # separate tight height gate so this never accepts a carton that fell
        # to the floor, while camera servo still owns the live XY alignment.
        if planar_error > planar_tolerance or height_error > height_tolerance:
            raise RuntimeError(
                f"VQA verification failed: {model} shelf error "
                f"xy={planar_error:.2f} m, z={height_error:.2f} m"
            )
        return error

    def verify_grasp_pose(
        self,
        model: str,
        expected_forward: float,
        forward_tolerance: float,
        lateral_tolerance: float,
        height_tolerance: float,
        expected_yaw: float,
        yaw_tolerance: float,
    ) -> dict:
        """Gate attachment using measured Gazebo poses, not a semantic lookup."""
        with self.lock:
            robot = self.observed["warehouse_agv"]
            box = self.observed[model]
        q = robot.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny, cosy)
        dx = box.position.x - robot.position.x
        dy = box.position.y - robot.position.y
        forward = math.cos(yaw) * dx + math.sin(yaw) * dy
        lateral = -math.sin(yaw) * dx + math.cos(yaw) * dy
        yaw_error = math.atan2(
            math.sin(expected_yaw - yaw), math.cos(expected_yaw - yaw)
        )
        height_error = abs(box.position.z - 0.83)
        forward_error = abs(forward - expected_forward)
        if (
            forward_error > forward_tolerance
            or abs(lateral) > lateral_tolerance
            or height_error > height_tolerance
            or abs(yaw_error) > yaw_tolerance
        ):
            raise RuntimeError(
                "Physical grasp gate failed: "
                f"forward={forward:.3f} m, lateral={lateral:+.3f} m, "
                f"yaw_error={math.degrees(yaw_error):+.2f} deg, "
                f"height_error={height_error:.3f} m"
            )
        return {
            "forward_m": forward,
            "lateral_m": lateral,
            "yaw_error_rad": yaw_error,
            "height_error_m": height_error,
        }

    def robot_yaw(self) -> float:
        with self.lock:
            pose = self.observed["warehouse_agv"]
        q = pose.orientation
        return math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )

    def reach_extension(self, retracted_x: float) -> float:
        """Read the real prismatic extension from Gazebo link feedback."""
        with self.lock:
            pose = self.observed["fork_reach"]
        return max(0.0, pose.position.x - retracted_x)

    def verify_payload_on_agv(self, model: str, max_distance_m: float = 0.85) -> float:
        """Reject resume/release when the selected carton is not on this AGV.

        This is deliberately a planar check.  The carton is carried above the
        AGV, so including the vertical tray height made a correctly carried
        box look ``0.83 m`` away from a robot whose model origin is on the
        floor.  The footprint gate below performs the separate height check.
        """
        with self.lock:
            robot = self.observed["warehouse_agv"]
            box = self.observed[model]
        distance = math.hypot(
            box.position.x - robot.position.x,
            box.position.y - robot.position.y,
        )
        if distance > max_distance_m:
            raise RuntimeError(
                f"Cannot resume delivery: {model} is {distance:.2f} m from the AGV; "
                "select the storage/color of the payload actually attached"
            )
        return distance

    @staticmethod
    def _yaw_from_pose(pose: Pose) -> float:
        q = pose.orientation
        return math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )

    def verify_payload_on_tray(
        self,
        model: str,
        tray_size_x: float = TRAY_SIZE_X_M,
        tray_size_y: float = TRAY_SIZE_Y_M,
        payload_size_x: float = PAYLOAD_SIZE_X_M,
        payload_size_y: float = PAYLOAD_SIZE_Y_M,
        payload_size_z: float = PAYLOAD_SIZE_Z_M,
        margin: float = TRAY_FIT_MARGIN_M,
        height_tolerance: float = 0.035,
    ) -> dict[str, float]:
        """Require the whole carton footprint to be inside the raised tray.

        Gazebo's detachable joint reports ``attached`` as soon as a fixed
        joint exists.  That signal does not constrain the carton to the tray
        outline, and was the reason a half-overhanging box could be accepted.
        We transform all four payload corners into the AGV/tray frame and
        reject the mission before it drives away if any corner is outside.
        """
        with self.lock:
            robot = self.observed["warehouse_agv"]
            box = self.observed[model]
            fork = self.observed.get("fork_carriage")

        robot_yaw = self._yaw_from_pose(robot)
        cos_robot = math.cos(robot_yaw)
        sin_robot = math.sin(robot_yaw)

        # PosePublisher exposes link poses relative to the AGV model.  The
        # fork carriage is centred today, but using its measured pose keeps
        # this gate correct if the tray is moved later.
        fork_x = float(fork.position.x) if fork is not None else 0.0
        fork_y = float(fork.position.y) if fork is not None else 0.0
        tray_center_x = robot.position.x + cos_robot * fork_x - sin_robot * fork_y
        tray_center_y = robot.position.y + sin_robot * fork_x + cos_robot * fork_y

        dx_world = box.position.x - tray_center_x
        dy_world = box.position.y - tray_center_y
        center_x = cos_robot * dx_world + sin_robot * dy_world
        center_y = -sin_robot * dx_world + cos_robot * dy_world
        box_yaw = self._yaw_from_pose(box)
        relative_yaw = box_yaw - robot_yaw
        cos_box = math.cos(relative_yaw)
        sin_box = math.sin(relative_yaw)

        corners = []
        for sx in (-0.5 * payload_size_x, 0.5 * payload_size_x):
            for sy in (-0.5 * payload_size_y, 0.5 * payload_size_y):
                corners.append(
                    (
                        center_x + cos_box * sx - sin_box * sy,
                        center_y + sin_box * sx + cos_box * sy,
                    )
                )
        min_x = min(corner[0] for corner in corners)
        max_x = max(corner[0] for corner in corners)
        min_y = min(corner[1] for corner in corners)
        max_y = max(corner[1] for corner in corners)
        tray_half_x = 0.5 * tray_size_x
        tray_half_y = 0.5 * tray_size_y
        fit = (
            min_x >= -tray_half_x + margin
            and max_x <= tray_half_x - margin
            and min_y >= -tray_half_y + margin
            and max_y <= tray_half_y - margin
        )

        # The tray link pose is relative to the AGV model.  The platform top
        # plus half the carton height is the expected centre after LOWER.
        tray_z = float(fork.position.z) if fork is not None else 0.363575
        expected_z = (
            float(robot.position.z)
            + tray_z
            + 0.5 * TRAY_PLATFORM_THICKNESS_M
            + 0.5 * payload_size_z
        )
        height_error = abs(float(box.position.z) - expected_z)
        if not fit or height_error > height_tolerance:
            raise RuntimeError(
                f"Payload tray gate failed: {model} footprint "
                f"x=[{min_x:+.3f},{max_x:+.3f}] y=[{min_y:+.3f},{max_y:+.3f}] "
                f"tray=±({tray_half_x - margin:.3f},{tray_half_y - margin:.3f}), "
                f"height_error={height_error:.3f} m"
            )
        return {
            "center_forward_m": center_x,
            "center_lateral_m": center_y,
            "min_x_m": min_x,
            "max_x_m": max_x,
            "min_y_m": min_y,
            "max_y_m": max_y,
            "height_error_m": height_error,
        }

    def robot_xy_distance_to(self, pose: list[float]) -> float:
        with self.lock:
            robot = self.observed["warehouse_agv"]
        return math.hypot(
            robot.position.x - float(pose[0]),
            robot.position.y - float(pose[1]),
        )

    def configure_grasp(self, model: str, timeout: float = 5.0) -> None:
        namespace = f"/warehouse_agv/gripper/{model}"
        self.attach_publishers[model] = self.node.advertise(
            f"{namespace}/attach", Empty
        )
        self.detach_publishers[model] = self.node.advertise(
            f"{namespace}/detach", Empty
        )
        self.descent_publishers[model] = self.node.advertise(
            f"/model/{model}/cmd_vel", GazeboTwist
        )

        def on_state(message: StringMsg) -> None:
            self.attachment_state[model] = message.data == "attached"

        self.node.subscribe(StringMsg, f"{namespace}/attached", on_state)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if (
                self.attach_publishers[model].has_connections()
                and self.descent_publishers[model].has_connections()
            ):
                self.stop_payload_motion(model)
                return
            time.sleep(0.05)
        raise RuntimeError(f"DetachableJoint attach topic unavailable for {model}")

    def stop_payload_motion(self, model: str) -> None:
        """Cancel residual shelf drift without changing the carton pose."""
        stop = GazeboTwist()
        for _ in range(5):
            self.descent_publishers[model].publish(stop)
            time.sleep(0.01)

    def attach(self, model: str, timeout: float = 3.0) -> None:
        if model not in self.attach_publishers:
            self.configure_grasp(model)
        self.attach_publishers[model].publish(Empty())
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.attachment_state.get(model, False):
                return
            time.sleep(0.05)
        raise RuntimeError(f"Physical suction joint did not attach {model}")

    def detach(self, model: str, timeout: float = 3.0) -> None:
        if model not in self.detach_publishers:
            self.configure_grasp(model)
        self.detach_publishers[model].publish(Empty())
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self.attachment_state.get(model, True):
                return
            time.sleep(0.05)
        raise RuntimeError(f"Physical suction joint did not detach {model}")

    def wait_for_drop(
        self,
        model: str,
        center: list[float],
        half_extents: list[float],
        target_center_z: float = 0.145,
        height_tolerance: float = 0.018,
        timeout: float = 4.0,
    ) -> tuple[float, float, float]:
        """Require the released carton to settle on the P platform."""
        center_x, center_y = map(float, center)
        half_x, half_y = map(float, half_extents)
        deadline = time.monotonic() + timeout
        stable_samples = 0
        previous: tuple[float, float, float] | None = None
        latest = (math.nan, math.nan, math.nan)
        while time.monotonic() < deadline:
            with self.lock:
                pose = self.observed[model]
                latest = (
                    float(pose.position.x),
                    float(pose.position.y),
                    float(pose.position.z),
                )
            x, y, z = latest
            inside = (
                abs(x - center_x) <= half_x
                and abs(y - center_y) <= half_y
            )
            # Packing P has a 55 mm top surface.  The 180 mm carton therefore
            # rests with its centre at z=0.145 m, rather than at floor height
            # z=0.09 m (which made it intersect the platform and look as if it
            # were floating or sinking).
            grounded = abs(z - float(target_center_z)) <= height_tolerance
            stable = previous is not None and math.dist(latest, previous) <= 0.004
            stable_samples = stable_samples + 1 if inside and grounded and stable else 0
            if stable_samples >= 5:
                self.stop_payload_motion(model)
                return latest
            previous = latest
            time.sleep(0.05)
        raise RuntimeError(
            f"Drop verification failed: {model} did not settle inside Packing P; "
            f"last pose=({latest[0]:.2f}, {latest[1]:.2f}, {latest[2]:.2f})"
        )

    def lower_released_payload(
        self,
        model: str,
        target_center_z: float = 0.145,
        speed_mps: float = 0.20,
        timeout: float = 4.0,
    ) -> None:
        """Move a detached, zero-gravity carton onto the P platform."""
        publisher = self.descent_publishers[model]
        deadline = time.monotonic() + timeout
        command = GazeboTwist()
        command.linear.z = -abs(speed_mps)
        try:
            while time.monotonic() < deadline:
                with self.lock:
                    z = float(self.observed[model].position.z)
                if z <= target_center_z + 0.005:
                    return
                publisher.publish(command)
                time.sleep(0.02)
            raise RuntimeError(
                f"Controlled drop timed out: {model} remained above floor"
            )
        finally:
            stop = GazeboTwist()
            for _ in range(5):
                publisher.publish(stop)
                time.sleep(0.01)


class VqaNav2Mission(Node):
    def __init__(self, pipeline: dict) -> None:
        super().__init__("vqa_nav2_task_planner")
        self.pipeline = pipeline
        self.nav_client = ActionClient(self, NavigateToPose, "/navigate_to_pose")
        self.collision_toggle = self.create_client(Toggle, "/collision_monitor/toggle")
        self.goal_preview = self.create_publisher(PoseStamped, "/semantic_goal", 1)
        self.mission_state = self.create_publisher(
            String, "/warehouse/mission_state", 10
        )
        self.current_state_message: String | None = None
        # Short manipulation phases can begin before DDS discovery finishes
        # and end between V-JEPA inference samples. Republish the active phase
        # while manipulation callbacks are being spun so evidence is not lost.
        self.create_timer(0.5, self._republish_state)
        self.base_velocity = self.create_publisher(Twist, "/cmd_vel", 10)
        self.scissor_angle = self.create_publisher(
            Float64, "/gripper/scissor_angle_position", 10
        )
        self.scissor_slider = self.create_publisher(
            Float64, "/gripper/scissor_slider_position", 10
        )
        self.gripper_lift = self.create_publisher(
            Float64, "/gripper/lift_position", 10
        )
        self.scissor_stages = [
            self.create_publisher(
                Float64, f"/gripper/scissor_stage_{index}_position", 10
            )
            for index in range(1, SCISSOR_STAGE_COUNT + 1)
        ]
        self.gripper_slide = self.create_publisher(
            Float64, "/gripper/hand_position", 10
        )
        # Send the same physical joint target over Gazebo Transport as a
        # deterministic fallback. DDS discovery once accepted the ROS target
        # while the bridge's first command still arrived too late, leaving the
        # slide at zero for an entire pick attempt.
        self.gz_command_node = GazeboNode()
        self.gz_gripper_slide = self.gz_command_node.advertise(
            "/model/warehouse_agv/joint/fork_reach_joint/cmd_pos", Double
        )
        self.camera_frame_rgb: np.ndarray | None = None
        self.camera_sequence = 0
        self.create_subscription(Image, "/camera", self._camera_callback, 10)
        self.odom_xy: tuple[float, float] | None = None
        self.create_subscription(Odometry, "/odom", self._odom_callback, 20)
        self.front_clearance = math.inf
        self.create_subscription(LaserScan, "/scan", self._scan_callback, 20)
        self.last_feedback = 0.0
        self.lift_position = 0.0

    def publish_state(self, state: str, **details: object) -> None:
        payload = {"state": state, "timestamp": time.time(), **details}
        self.current_state_message = String(
            data=json.dumps(payload, sort_keys=True)
        )
        self.mission_state.publish(self.current_state_message)
        print(f"[MISSION_STATE] {state}")

    def _republish_state(self) -> None:
        if self.current_state_message is not None:
            self.mission_state.publish(self.current_state_message)

    def _odom_callback(self, message: Odometry) -> None:
        position = message.pose.pose.position
        self.odom_xy = (position.x, position.y)

    def _scan_callback(self, message: LaserScan) -> None:
        """Track the closest valid return in a narrow forward safety cone."""
        values = []
        angle = message.angle_min
        for distance in message.ranges:
            wrapped = math.atan2(math.sin(angle), math.cos(angle))
            if (
                abs(wrapped) <= 0.14
                and math.isfinite(distance)
                and message.range_min <= distance <= message.range_max
            ):
                values.append(float(distance))
            angle += message.angle_increment
        self.front_clearance = min(values, default=math.inf)

    def _camera_callback(self, message: Image) -> None:
        """Keep the newest Gazebo RGB frame without depending on cv_bridge."""
        encoding = message.encoding.lower()
        channels = 4 if encoding in ("rgba8", "bgra8") else 3
        expected = message.width * channels
        raw = np.frombuffer(message.data, dtype=np.uint8)
        if message.height <= 0 or message.step < expected:
            return
        rows = raw.reshape(message.height, message.step)[:, :expected]
        frame = rows.reshape(message.height, message.width, channels)
        if encoding == "bgr8":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif encoding == "rgba8":
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        elif encoding == "bgra8":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        elif encoding != "rgb8":
            self.get_logger().warning(f"Unsupported camera encoding: {message.encoding}")
            return
        self.camera_frame_rgb = frame.copy()
        self.camera_sequence += 1

    def _fresh_camera_frame(self, timeout: float = 3.0) -> np.ndarray:
        sequence = self.camera_sequence
        deadline = time.monotonic() + timeout
        while self.camera_sequence <= sequence and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.10)
        if self.camera_frame_rgb is None or self.camera_sequence <= sequence:
            raise RuntimeError("VQA did not receive a fresh frame from /camera")
        return self.camera_frame_rgb.copy()

    def _analyze_color(self, rgb: np.ndarray, color: str) -> dict:
        """Return the best solid-color carton candidate in one RGB frame."""
        ranges = {
            "blue": ((95, 125, 45), (135, 255, 255)),
            "green": ((40, 90, 40), (88, 255, 255)),
            "red": ((0, 120, 55), (10, 255, 255)),
        }
        if color not in ranges:
            raise RuntimeError(f"No camera detector configured for color '{color}'")
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        lower, upper = ranges[color]
        mask = cv2.inRange(
            hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8)
        )
        kernel = np.ones((5, 5), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        count, _, stats, centers = cv2.connectedComponentsWithStats(mask)
        height, width = mask.shape
        candidates = []
        for index in range(1, count):
            x, y, box_width, box_height, area = map(int, stats[index])
            if area < 40 or area > int(width * height * 0.45):
                continue
            if box_width > width * 0.60 or box_height > height * 0.72:
                continue
            center_x, center_y = centers[index]
            aspect = box_width / max(1.0, float(box_height))
            fill_ratio = area / max(1.0, float(box_width * box_height))
            # Task cartons are compact, solid rectangles in the lower half of
            # the level camera image. Long orange rack beams can enter the red
            # HSV range; their sparse connected component previously produced
            # a huge false box spanning the image ceiling.
            if center_y < height * 0.44:
                continue
            if not 0.50 <= aspect <= 2.20 or fill_ratio < 0.55:
                continue
            center_distance = math.hypot(
                (center_x - width / 2.0) / width,
                (center_y - height / 2.0) / height,
            )
            # The requested slot is centered by its Nav2 approach pose.
            score = area / (1.0 + 4.0 * center_distance)
            candidates.append((score, area, x, y, box_width, box_height))
        if not candidates:
            raise RuntimeError(
                f"VQA camera frame contains no usable {color} box candidate"
            )

        _, area, x, y, box_width, box_height = max(candidates)
        confidence = min(0.99, 0.70 + area / float(width * height) * 8.0)
        return {
            "color": color,
            "bbox": [x, y, box_width, box_height],
            "center": [x + box_width / 2.0, y + box_height / 2.0],
            "frame_size": [width, height],
            "pixels": area,
            "confidence": confidence,
            "rgb": rgb,
        }

    def _save_detection(self, detection: dict, evidence_path: Path) -> None:
        x, y, box_width, box_height = detection["bbox"]
        rgb = detection["rgb"]
        annotated = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        frame_width, frame_height = detection["frame_size"]
        cv2.drawMarker(
            annotated,
            (frame_width // 2, frame_height // 2),
            (255, 255, 255),
            cv2.MARKER_CROSS,
            28,
            2,
        )
        cv2.rectangle(
            annotated, (x, y), (x + box_width, y + box_height), (0, 255, 255), 3
        )
        label = f"VQA: {detection['color'].upper()}"
        if "range_m" in detection:
            label += f"  range={detection['range_m']:.2f}m"
        cv2.putText(
            annotated, label, (max(8, x), max(25, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
        )
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(evidence_path), annotated):
            raise RuntimeError(f"Could not save VQA evidence to {evidence_path}")
        detection["image"] = str(evidence_path)
        detection.pop("rgb", None)

    def detect_camera_color(
        self, color: str, evidence_path: Path, timeout: float = 6.0
    ) -> dict:
        detection = self._analyze_color(self._fresh_camera_frame(timeout), color)
        self._save_detection(detection, evidence_path)
        return detection

    def _make_pose(self, xyz_yaw: list[float]) -> PoseStamped:
        x, y, yaw = map(float, xyz_yaw)
        goal = PoseStamped()
        goal.header.frame_id = "map"
        # Use the latest transform. A system-time stamp can be far ahead of
        # Gazebo simulation time and make an otherwise valid Nav2 goal abort.
        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.orientation.w = math.cos(yaw / 2.0)
        return goal

    def _feedback(self, message) -> None:
        now = time.monotonic()
        if now - self.last_feedback >= 2.0:
            remaining = message.feedback.distance_remaining
            print(f"[NAV2] distance_remaining={remaining:.2f} m")
            self.last_feedback = now

    def set_collision_monitor(self, enabled: bool) -> None:
        """Toggle the emergency velocity layer without disabling LiDAR costmaps."""
        if not self.collision_toggle.wait_for_service(timeout_sec=2.0):
            print("[SAFETY] collision monitor service unavailable; keeping current state")
            return
        request = Toggle.Request()
        request.enable = enabled
        future = self.collision_toggle.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)
        response = future.result()
        if response is None or not response.success:
            raise RuntimeError("Could not change collision monitor state")
        state = "enabled" if enabled else "disabled"
        nav_mode = self.pipeline["coarse_navigation"].get("mode", "dynamic")
        print(f"[SAFETY] collision monitor {state}; navigation mode={nav_mode}")

    def navigate(self, label: str, xyz_yaw: list[float], wait: float) -> None:
        if not self.nav_client.wait_for_server(timeout_sec=wait):
            raise RuntimeError("Nav2 /navigate_to_pose action server is unavailable")
        pose = self._make_pose(xyz_yaw)
        self.goal_preview.publish(pose)
        print(
            f"[NAV2] goal {label}: x={pose.pose.position.x:.2f}, "
            f"y={pose.pose.position.y:.2f}"
        )
        goal = NavigateToPose.Goal()
        goal.pose = pose
        goal.behavior_tree = str(NAV_TO_POSE_BT)
        send_future = self.nav_client.send_goal_async(
            goal, feedback_callback=self._feedback
        )
        rclpy.spin_until_future_complete(self, send_future)
        handle = send_future.result()
        if handle is None or not handle.accepted:
            raise RuntimeError(f"Nav2 rejected {label}")
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        status = result_future.result().status
        if status != GoalStatus.STATUS_SUCCEEDED:
            raise RuntimeError(f"Nav2 failed {label} with action status {status}")
        print(f"[NAV2] reached {label}")

    def _stop_base(self) -> None:
        stop = Twist()
        for _ in range(5):
            self.base_velocity.publish(stop)
            rclpy.spin_once(self, timeout_sec=0.03)

    def drive_distance(
        self,
        distance: float,
        speed: float = 0.05,
        hard_stop: float | None = None,
    ) -> None:
        """Closed-loop low-speed final creep using odometry distance."""
        deadline = time.monotonic() + 3.0
        while self.odom_xy is None and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.10)
        if self.odom_xy is None:
            raise RuntimeError("No odometry available for final visual-servo creep")
        start = self.odom_xy
        command = Twist()
        command.linear.x = math.copysign(abs(speed), distance)
        target = abs(distance)
        deadline = time.monotonic() + max(5.0, target / abs(speed) * 2.5)
        while time.monotonic() < deadline:
            current = self.odom_xy
            if current is not None and math.hypot(
                current[0] - start[0], current[1] - start[1]
            ) >= target:
                self._stop_base()
                return
            if distance > 0.0 and hard_stop is not None and self.front_clearance <= hard_stop:
                self._stop_base()
                raise RuntimeError(
                    f"LiDAR hard stop at {self.front_clearance:.3f} m during final creep"
                )
            self.base_velocity.publish(command)
            rclpy.spin_once(self, timeout_sec=0.05)
        self._stop_base()
        raise RuntimeError("Final visual-servo creep timed out")

    def visual_servo_to_target(
        self, color: str, evidence_path: Path
    ) -> dict:
        """Use camera pixels to center the carton and close most of the gap."""
        perception = self.pipeline["perception"]
        control = self.pipeline["fine_approach"]
        fov = float(perception["horizontal_fov_rad"])
        object_width = float(perception["target_width_m"])
        range_scale = float(perception["monocular_range_scale"])
        target_range = float(control["target_range_m"])
        hard_stop = float(control["lidar_hard_stop_m"])
        tolerance = float(control["horizontal_tolerance"])
        deadline = time.monotonic() + float(control["timeout_s"])
        last_report = 0.0
        last_search_report = 0.0
        search_speed = float(control.get("search_angular_speed", 0.45))
        acquisition_max_range = float(
            control.get("acquisition_max_range_m", 3.2)
        )
        target_acquired = False
        final_detection = None

        def search_for_target(reason: str) -> None:
            nonlocal last_search_report
            command = Twist()
            command.angular.z = search_speed
            self.base_velocity.publish(command)
            if time.monotonic() - last_search_report >= 1.0:
                print(
                    f"[VJEPA_RECOVERY] {reason}; "
                    f"camera đang quét {search_speed:.2f} rad/s"
                )
                last_search_report = time.monotonic()

        while time.monotonic() < deadline:
            try:
                detection = self._analyze_color(
                    self._fresh_camera_frame(1.5), color
                )
            except RuntimeError:
                # V-JEPA is the coarse global localizer. If its residual puts
                # the requested carton outside the initial camera FOV, scan in
                # place and let image evidence reacquire the target. Translation
                # remains disabled until a valid color detection is present.
                search_for_target(f"chưa thấy box {color}")
                rclpy.spin_once(self, timeout_sec=0.05)
                continue
            width, _ = detection["frame_size"]
            center_x, _ = detection["center"]
            box_width = detection["bbox"][2]
            focal_px = width / (2.0 * math.tan(fov / 2.0))
            range_m = range_scale * focal_px * object_width / box_width
            horizontal_error = (center_x - width / 2.0) / (width / 2.0)
            detection["range_m"] = range_m
            detection["horizontal_error"] = horizontal_error
            if not target_acquired and range_m > acquisition_max_range:
                search_for_target(
                    f"bỏ qua vật {color} xa {range_m:.1f} m"
                )
                rclpy.spin_once(self, timeout_sec=0.05)
                continue
            if not target_acquired:
                target_acquired = True
                self._stop_base()
                print(
                    f"[VJEPA_RECOVERY] đã bắt lại box {color} ở "
                    f"{range_m:.2f} m; chuyển sang visual servo"
                )
            final_detection = detection

            if time.monotonic() - last_report >= 1.0:
                print(
                    f"[VISUAL_SERVO] camera_range={range_m:.2f} m "
                    f"horizontal_error={horizontal_error:+.3f} "
                    f"front_lidar={self.front_clearance:.2f} m"
                )
                last_report = time.monotonic()

            if range_m <= target_range and abs(horizontal_error) <= tolerance:
                self._stop_base()
                self._save_detection(detection, evidence_path)
                creep = float(control["final_creep_m"])
                print(
                    f"[VISUAL_SERVO] image gate passed; odometry creep={creep:.2f} m"
                )
                self.drive_distance(creep, hard_stop=hard_stop)
                return detection

            command = Twist()
            # Correct yaw before translating. A simultaneous turn and drive
            # made this short-wheelbase platform approach the rack diagonally.
            if abs(horizontal_error) > tolerance:
                raw_yaw = -float(control["angular_gain"]) * horizontal_error
                yaw_speed = min(
                    float(control["max_angular_speed"]),
                    max(float(control["min_angular_speed"]), abs(raw_yaw)),
                )
                command.angular.z = math.copysign(yaw_speed, raw_yaw)
            elif range_m > target_range:
                if self.front_clearance <= hard_stop:
                    self._stop_base()
                    raise RuntimeError(
                        f"LiDAR hard stop at {self.front_clearance:.3f} m; "
                        "refusing to drive into the rack"
                    )
                command.linear.x = min(
                    float(control["max_linear_speed"]),
                    float(control["linear_gain"]) * (range_m - target_range),
                )
            self.base_velocity.publish(command)
            rclpy.spin_once(self, timeout_sec=0.05)

        self._stop_base()
        if final_detection is not None:
            self._save_detection(final_detection, evidence_path)
        raise RuntimeError("Camera visual servo could not reach the grasp gate")

    def target_geometry(self, detection: dict) -> tuple[float, float]:
        """Return monocular range and normalized horizontal image error."""
        perception = self.pipeline["perception"]
        width, _ = detection["frame_size"]
        center_x, _ = detection["center"]
        box_width = detection["bbox"][2]
        focal_px = width / (
            2.0 * math.tan(float(perception["horizontal_fov_rad"]) / 2.0)
        )
        range_m = (
            float(perception["monocular_range_scale"])
            * focal_px
            * float(perception["target_width_m"])
            / box_width
        )
        horizontal_error = (center_x - width / 2.0) / (width / 2.0)
        detection["range_m"] = range_m
        detection["horizontal_error"] = horizontal_error
        return range_m, horizontal_error

    def dock_square_and_center(
        self,
        color: str,
        evidence_path: Path,
        rack_yaw: float,
        yaw_feedback,
        yaw_tolerance: float,
    ) -> dict:
        """Face the rack first, then approach and validate the final centerline."""
        control = self.pipeline["fine_approach"]
        attempts = max(1, int(control.get("final_square_attempts", 3)))
        image_tolerance = float(
            control.get(
                "post_square_horizontal_tolerance",
                control["horizontal_tolerance"],
            )
        )
        target_range = float(control["target_range_m"])
        range_tolerance = float(
            control.get("post_square_range_tolerance_m", 0.10)
        )
        retreat = float(control.get("correction_retreat_m", 0.25))

        initial_error = math.atan2(
            math.sin(rack_yaw - yaw_feedback()),
            math.cos(rack_yaw - yaw_feedback()),
        )
        if abs(initial_error) <= yaw_tolerance:
            side = "FRONT"
        else:
            side = "LEFT" if initial_error > 0.0 else "RIGHT"
        print(
            f"[RACK_SIDE] rack is {side}; rotate {math.degrees(initial_error):+.1f} deg "
            "before translation"
        )
        self.square_to_rack(rack_yaw, yaw_feedback, yaw_tolerance, timeout=8.0)

        for attempt in range(1, attempts + 1):
            print(f"[DOCK] center approach pass {attempt}/{attempts}")
            self.visual_servo_to_target(color, evidence_path)
            self.square_to_rack(rack_yaw, yaw_feedback, yaw_tolerance, timeout=8.0)
            detection = self._analyze_color(self._fresh_camera_frame(2.0), color)
            range_m, horizontal_error = self.target_geometry(detection)
            centered = abs(horizontal_error) <= image_tolerance
            in_range = range_m <= target_range + range_tolerance
            print(
                f"[CENTER_GATE] range={range_m:.3f} m "
                f"horizontal_error={horizontal_error:+.3f} "
                f"square_yaw=PASS centered={'PASS' if centered else 'RETRY'}"
            )
            if centered and in_range:
                self._save_detection(detection, evidence_path)
                return detection
            if attempt < attempts:
                # A differential-drive base cannot strafe. Backing away gives
                # it room to turn toward the box, approach its centerline, and
                # square itself to the rack again.
                print(
                    f"[CENTER_RETRY] backing {retreat:.2f} m to correct lateral offset"
                )
                self.drive_distance(-retreat, speed=0.08)

        raise RuntimeError(
            "Final rack center gate failed after camera approach and square-up"
        )

    def square_to_rack(
        self,
        target_yaw: float,
        yaw_feedback,
        tolerance: float,
        timeout: float = 4.0,
    ) -> float:
        """Rotate the fixed camera, chassis and suction cups square to rack."""
        control = self.pipeline["fine_approach"]
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            error = math.atan2(
                math.sin(target_yaw - yaw_feedback()),
                math.cos(target_yaw - yaw_feedback()),
            )
            if abs(error) <= tolerance:
                self._stop_base()
                return error
            speed = min(
                float(control["max_angular_speed"]),
                max(
                    float(control["min_angular_speed"]),
                    float(control["angular_gain"]) * abs(error),
                ),
            )
            command = Twist()
            command.angular.z = math.copysign(speed, error)
            self.base_velocity.publish(command)
            rclpy.spin_once(self, timeout_sec=0.04)
        self._stop_base()
        raise RuntimeError("Robot could not square both suction cups to rack")

    def publish_lift_geometry(self, lift: float) -> None:
        sin_theta = math.sin(SCISSOR_THETA_FOLDED) + lift / (
            SCISSOR_STAGE_COUNT * SCISSOR_BAR_LENGTH
        )
        theta = math.asin(min(1.0, max(-1.0, sin_theta)))
        angle = theta - SCISSOR_THETA_FOLDED
        slider = (SCISSOR_BAR_LENGTH / 2.0) * (
            math.cos(SCISSOR_THETA_FOLDED) - math.cos(theta)
        )
        self.gripper_lift.publish(Float64(data=lift))
        self.scissor_angle.publish(Float64(data=angle))
        self.scissor_slider.publish(Float64(data=slider))
        stage_msgs = [
            Float64(
                data=lift * (2 * index - 1) / (2 * SCISSOR_STAGE_COUNT)
            )
            for index in range(1, SCISSOR_STAGE_COUNT + 1)
        ]
        for publisher, message in zip(self.scissor_stages, stage_msgs):
            publisher.publish(message)

    def move_lift_to(self, target: float, duration: float) -> None:
        """Move the lift to an absolute calibrated height with smooth motion."""
        target = min(SCISSOR_LIFT_MAX, max(0.0, float(target)))
        start = self.lift_position
        started = time.monotonic()
        deadline = started + max(0.05, duration)
        while time.monotonic() < deadline:
            ratio = min(1.0, (time.monotonic() - started) / max(0.05, duration))
            smooth = ratio * ratio * (3.0 - 2.0 * ratio)
            self.publish_lift_geometry(start + (target - start) * smooth)
            rclpy.spin_once(self, timeout_sec=0.04)
        self.lift_position = target
        self.publish_lift_geometry(target)

    def move_suction_slide_to(
        self,
        velocity: float,
        target_extension: float,
        extension_feedback,
        tolerance: float,
        timeout: float,
    ) -> float:
        """Command an absolute slide position and verify measured link motion.

        The previous velocity controller could lose its short command burst
        when Gazebo's real-time factor dropped during V-JEPA inference.  The
        position target persists in Gazebo and therefore remains deterministic
        while feedback is still the authority for every state transition.
        """
        del velocity  # kept in the call contract for configuration compatibility
        target_extension = max(0.0, float(target_extension))
        command = Float64(data=target_extension)
        gazebo_command = Double(data=target_extension)

        def publish_target() -> None:
            self.gripper_slide.publish(command)
            self.gz_gripper_slide.publish(gazebo_command)

        discovery_deadline = time.monotonic() + 5.0
        while (
            self.gripper_slide.get_subscription_count() == 0
            and time.monotonic() < discovery_deadline
        ):
            rclpy.spin_once(self, timeout_sec=0.05)
        if self.gripper_slide.get_subscription_count() == 0:
            raise RuntimeError(
                "Suction position controller is disconnected: "
                "/gripper/hand_position has no bridge subscriber"
            )
        extending = target_extension > extension_feedback()
        deadline = time.monotonic() + timeout
        last_report = 0.0
        while time.monotonic() < deadline:
            extension = extension_feedback()
            reached = (
                extension >= target_extension - tolerance
                if extending
                else extension <= target_extension + tolerance
            )
            if reached:
                publish_target()
                return extension
            publish_target()
            if time.monotonic() - last_report >= 0.5:
                print(
                    f"[SLIDE] target={target_extension:.3f} m "
                    f"measured={extension:.3f} m"
                )
                last_report = time.monotonic()
            rclpy.spin_once(self, timeout_sec=0.04)
        publish_target()
        extension = extension_feedback()
        raise RuntimeError(
            "Suction slide failed to reach feedback gate: "
            f"target={target_extension:.3f} m, measured={extension:.3f} m"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--command",
        default="Bring the blue box from Storage A to Packing Station",
    )
    parser.add_argument("--wait", type=float, default=10.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--pick-only", action="store_true",
        help="Stop after pulling the selected box onto the raised platform",
    )
    parser.add_argument(
        "--camera-evidence", type=Path, default=DEFAULT_CAMERA_EVIDENCE,
        help="Path for the annotated camera frame used by VQA",
    )
    parser.add_argument(
        "--skip-navigation",
        action="store_true",
        help="the caller already followed the recorded latent corridor to the pickup pose",
    )
    parser.add_argument(
        "--release-only",
        action="store_true",
        help="release an already attached payload after the caller reached Packing Station",
    )
    parser.add_argument(
        "--prepare-return-only",
        action="store_true",
        help="verify an attached payload and retreat from the rack before return Nav2",
    )
    args = parser.parse_args()
    if sum((args.pick_only, args.release_only, args.prepare_return_only)) > 1:
        parser.error(
            "--pick-only, --release-only and --prepare-return-only are mutually exclusive"
        )

    with CONFIG.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    with PIPELINE_CONFIG.open(encoding="utf-8") as stream:
        pipeline = yaml.safe_load(stream)["pipeline"]
    vqa = answer_vqa(args.command, config)
    answer = vqa["answer"]
    nav_source = os.environ.get(
        "WAREHOUSE_NAV_LOCALIZATION_SOURCE", "vjepa"
    ).lower()
    slot_config = config["stations"][answer["storage"]]["slots"][answer["slot"]]
    if nav_source == "vjepa" and "vjepa_approach" in slot_config:
        answer["pickup_pose"] = slot_config["vjepa_approach"]
        answer["pickup_anchor"] = f"{answer['pickup_anchor']}_vjepa"
    if args.dry_run:
        print(json.dumps(vqa, ensure_ascii=False, indent=2))
        return

    print_target_card(answer)
    item = config["objects"][answer["object"]]
    packing = config["stations"]["packing_station"]["slots"]["PACK01"]
    drop_center = list(packing["drop_center"])
    drop_half_extents = list(
        packing.get("drop_center_tolerance", packing["drop_half_extents"])
    )
    drop_surface_z = float(packing.get("drop_surface_z", PACKING_SURFACE_Z_M))
    payload = GazeboPayload()
    payload.wait_for(
        "warehouse_agv", answer["model"], "fork_reach", "fork_carriage"
    )
    payload.configure_grasp(answer["model"])

    rclpy.init()
    mission = VqaNav2Mission(pipeline)
    try:
        print("[STATE] PARSE_TASK -> RESOLVE_SEMANTIC_TARGET")
        mission.publish_state(
            "PARSE_TASK", storage=answer["storage"], slot=answer["slot"]
        )
        print("[VQA] target resolved; Nav2 owns coarse aisle navigation")
        mission.set_collision_monitor(
            bool(pipeline["coarse_navigation"]["use_collision_monitor"])
        )
        # A previous aborted run may have left the lift or inverted-L carriage
        # extended. Return both real joints to their safe navigation limits;
        # this publishes controller commands and never resets a model pose.
        grasp = pipeline["grasp"]
        lift = pipeline["lift_alignment"]
        slide_speed = float(grasp["extend_speed_mps"])
        slide_max = float(grasp["max_extension_m"])
        slide_tolerance = float(grasp["position_tolerance_m"])
        slide_timeout = float(grasp["motion_timeout_s"])
        retracted_x = float(grasp["retracted_link_x_m"])
        extension_feedback = lambda: payload.reach_extension(retracted_x)
        drop_center_z = drop_surface_z + 0.5 * float(
            grasp.get("payload_size_z_m", PAYLOAD_SIZE_Z_M)
        )

        if args.prepare_return_only:
            mission.publish_state("RETURN_TO_DROPOFF", payload=answer["model"])
            payload_distance = payload.verify_payload_on_agv(answer["model"])
            mission.move_suction_slide_to(
                -slide_speed, 0.0, extension_feedback,
                slide_tolerance, slide_timeout,
            )
            mission.move_lift_to(0.0, float(lift["duration_s"]))
            aisle_error = payload.robot_xy_distance_to(answer["pickup_pose"])
            print(
                f"[PAYLOAD_GATE] {answer['model']} is on the AGV "
                f"(relative distance={payload_distance:.2f} m)"
            )
            if 0.30 < aisle_error <= 1.20:
                retreat_distance = float(grasp.get("post_pick_retreat_m", 0.72))
                retreat_speed = float(
                    grasp.get("post_pick_retreat_speed_mps", 0.18)
                )
                print(
                    f"[RESUME_CLEARANCE] base is {aisle_error:.2f} m from "
                    f"the registered aisle pose; reversing {retreat_distance:.2f} m"
                )
                mission.drive_distance(-retreat_distance, speed=retreat_speed)
            else:
                print(
                    "[RESUME_CLEARANCE] base is already in/away from the clear "
                    "aisle; no extra retreat"
                )
            print("[MISSION] PAYLOAD READY FOR RETURN CORRIDOR")
            return

        if args.release_only:
            mission.publish_state("PLACE_PACKAGE", payload=answer["model"])
            payload_distance = payload.verify_payload_on_agv(answer["model"])
            print(
                "[LATENT_ROUTE] loaded AGV reached Packing Station through "
                "the recorded return corridor"
            )
            print(
                f"[PAYLOAD_GATE] {answer['model']} is on the AGV "
                f"(relative distance={payload_distance:.2f} m)"
            )
            print("[STATE] RETURN_CORRIDOR -> RELEASE_PAYLOAD")
            drop_tray_fit = payload.verify_payload_on_tray(
                answer["model"],
                tray_size_x=float(grasp.get("tray_size_x_m", TRAY_SIZE_X_M)),
                tray_size_y=float(grasp.get("tray_size_y_m", TRAY_SIZE_Y_M)),
                payload_size_x=float(
                    grasp.get("payload_size_x_m", PAYLOAD_SIZE_X_M)
                ),
                payload_size_y=float(
                    grasp.get("payload_size_y_m", PAYLOAD_SIZE_Y_M)
                ),
                payload_size_z=float(
                    grasp.get("payload_size_z_m", PAYLOAD_SIZE_Z_M)
                ),
                margin=float(grasp.get("tray_fit_margin_m", TRAY_FIT_MARGIN_M)),
                height_tolerance=float(
                    grasp.get("tray_height_tolerance_m", 0.035)
                ),
            )
            drop_pose = release_approach_pose(
                drop_center,
                float(packing["approach"][2]),
                drop_tray_fit["center_forward_m"],
                drop_tray_fit["center_lateral_m"],
                slide_max,
            )
            print(
                "[DROP_ALIGN] release-only Nav2 correction "
                f"goal=({drop_pose[0]:.2f}, {drop_pose[1]:.2f})"
            )
            mission.navigate(
                answer["destination_anchor"], drop_pose, args.wait
            )
            print("[DROP_ALIGN] AGV is inside P; extending fork for release")
            mission.move_suction_slide_to(
                -slide_speed, 0.0, extension_feedback,
                slide_tolerance, slide_timeout,
            )
            mission.move_lift_to(0.0, float(lift["duration_s"]))
            mission.move_suction_slide_to(
                slide_speed, slide_max, extension_feedback,
                slide_tolerance, slide_timeout,
            )
            payload.detach(answer["model"], float(grasp["attach_timeout_s"]))
            # The fork can leave a tiny inherited velocity at the instant the
            # fixed joint opens.  Clear all six velocity components before the
            # controlled vertical placement so the carton cannot drift away
            # from the computed P centre.
            payload.stop_payload_motion(answer["model"])
            mission.move_suction_slide_to(
                -slide_speed, 0.0, extension_feedback,
                slide_tolerance, slide_timeout,
            )
            payload.lower_released_payload(
                answer["model"], target_center_z=drop_center_z
            )
            landed = payload.wait_for_drop(
                answer["model"],
                drop_center,
                drop_half_extents,
                target_center_z=drop_center_z,
            )
            print(
                f"[DROP_GATE] carton settled inside P at "
                f"({landed[0]:.2f}, {landed[1]:.2f}, z={landed[2]:.2f})"
            )
            print(f"[DROP] {answer['model']} delivered to Packing Station")
            print("[MISSION] COMPLETE on recorded outbound + return latent route")
            mission.publish_state("MISSION_COMPLETE", payload=answer["model"])
            return

        print("[STATE] RESOLVE_SEMANTIC_TARGET -> PREPARE_HARDWARE")
        mission.move_suction_slide_to(
            -slide_speed, 0.0, extension_feedback,
            slide_tolerance, slide_timeout,
        )
        mission.move_lift_to(0.0, float(lift["duration_s"]))
        print("[STATE] RESOLVE_SEMANTIC_TARGET -> NAVIGATE_TO_STORAGE")
        mission.publish_state(
            "NAVIGATE_TO_SHELF", storage=answer["storage"], slot=answer["slot"]
        )
        if args.skip_navigation:
            print(
                "[LATENT_ROUTE] pickup pose reached by recorded mapping corridor; "
                "skipping direct A* shortcut"
            )
        else:
            mission.navigate(answer["pickup_anchor"], answer["pickup_pose"], args.wait)
        shelf_sequence_started = time.monotonic()

        rack_yaw = float(answer["pickup_pose"][2])
        yaw_tolerance = float(grasp["rack_yaw_tolerance_rad"])

        # Raise the recessed platform to the calibrated shelf centerline, then
        # hand velocity ownership from Nav2 to the camera controller.
        print("[STATE] NAVIGATE_TO_STORAGE -> ALIGN_LIFT")
        mission.publish_state("RAISE_LIFT", slot=answer["slot"])
        lift_target = float(lift["nominal_position_m"])
        lift_duration = float(lift["duration_s"])
        print(f"[LIFT] raising to {lift_target:.3f} m in {lift_duration:.2f} s")
        mission.move_lift_to(lift_target, lift_duration)
        payload.stop_payload_motion(answer["model"])
        slot_error = payload.verify_slot(answer["model"], item["shelf_pose"])
        max_grasp_retries = int(grasp.get("max_retries", 2))
        minimum_confidence = float(
            grasp.get("minimum_detection_confidence", 0.72)
        )
        retry_retreat = float(grasp.get("retry_retreat_m", 0.25))
        for grasp_attempt in grasp_attempts(max_grasp_retries):
            try:
                print(
                    "[STATE] ALIGN_LIFT -> VISUAL_SERVO "
                    f"attempt {grasp_attempt}/{max_grasp_retries + 1}"
                )
                mission.publish_state(
                    "ALIGN_PACKAGE",
                    color=item["color"],
                    slot=answer["slot"],
                    attempt=grasp_attempt,
                )
                detection = mission.dock_square_and_center(
                    item["color"],
                    args.camera_evidence,
                    rack_yaw,
                    payload.robot_yaw,
                    yaw_tolerance,
                )
                if float(detection["confidence"]) < minimum_confidence:
                    raise RuntimeError(
                        "Detection confidence gate failed: "
                        f"{detection['confidence']:.2f} < {minimum_confidence:.2f}"
                    )
                print(
                    f"[VQA] /camera detected {detection['color']} in {answer['slot']} "
                    f"bbox={detection['bbox']} confidence={detection['confidence']:.2f}; "
                    f"registered pose error={slot_error:.3f} m"
                )
                print(f"[VQA] annotated input saved to {detection['image']}")
                print(
                    "[ALIGN] detection confidence and position consistency "
                    "passed"
                )

                mission.publish_state(
                    "GRASP_PACKAGE",
                    payload=answer["model"],
                    attempt=grasp_attempt,
                )
                extended = mission.move_suction_slide_to(
                    slide_speed, slide_max, extension_feedback,
                    slide_tolerance, slide_timeout,
                )
                print(f"[SLIDE_GATE] extended={extended:.3f} m")
                contact = payload.verify_grasp_pose(
                    answer["model"],
                    float(grasp["expected_box_forward_m"]),
                    float(grasp["forward_tolerance_m"]),
                    float(grasp["lateral_tolerance_m"]),
                    float(grasp["height_tolerance_m"]),
                    rack_yaw,
                    yaw_tolerance,
                )
                print(
                    "[CONTACT_GATE] "
                    f"forward={contact['forward_m']:.3f} m "
                    f"lateral={contact['lateral_m']:+.3f} m "
                    f"yaw_error={math.degrees(contact['yaw_error_rad']):+.2f} deg"
                )
                payload.attach(answer["model"], float(grasp["attach_timeout_s"]))
                mission.publish_state(
                    "VERIFY_GRASP",
                    payload=answer["model"],
                    attempt=grasp_attempt,
                )
                if not payload.attachment_state.get(answer["model"], False):
                    raise RuntimeError("gripper state did not confirm attachment")
                print(
                    "[GRASP_VERIFICATION] confidence=PASS "
                    "position=PASS gripper_state=ATTACHED"
                )
                print(f"[PICK] physical suction joint attached only {answer['model']}")
                break
            except RuntimeError as grasp_error:
                print(
                    f"[GRASP_RETRY] attempt {grasp_attempt} failed: {grasp_error}"
                )
                if payload.attachment_state.get(answer["model"], False):
                    payload.detach(
                        answer["model"], float(grasp["attach_timeout_s"])
                    )
                # Retraction is a mandatory safety gate. If it fails, do not
                # drive or begin another alignment attempt.
                mission.move_suction_slide_to(
                    -slide_speed, 0.0, extension_feedback,
                    slide_tolerance, slide_timeout,
                )
                if grasp_attempt > max_grasp_retries:
                    raise RuntimeError(
                        "grasp failed after configured "
                        f"{max_grasp_retries + 1} attempts"
                    ) from grasp_error
                if retry_retreat > 0.0:
                    mission.drive_distance(-retry_retreat, speed=0.08)
                print("[GRASP_RETRY] realigning camera and retrying grasp")

        print("[STATE] ATTACH_PAYLOAD -> RETRACT_PAYLOAD")
        retracted = mission.move_suction_slide_to(
            -slide_speed, 0.0, extension_feedback,
            slide_tolerance, slide_timeout,
        )
        print(
            "[RETRACT_GATE] suction slide and box are on the platform; "
            f"extension={retracted:.3f} m"
        )
        payload_distance = payload.verify_payload_on_agv(answer["model"])
        print(
            "[GRASP_VERIFICATION] retracted_payload_position=PASS "
            f"payload_planar_distance={payload_distance:.3f} m"
        )
        tray_fit = payload.verify_payload_on_tray(
            answer["model"],
            tray_size_x=float(grasp.get("tray_size_x_m", TRAY_SIZE_X_M)),
            tray_size_y=float(grasp.get("tray_size_y_m", TRAY_SIZE_Y_M)),
            payload_size_x=float(
                grasp.get("payload_size_x_m", PAYLOAD_SIZE_X_M)
            ),
            payload_size_y=float(
                grasp.get("payload_size_y_m", PAYLOAD_SIZE_Y_M)
            ),
            payload_size_z=float(
                grasp.get("payload_size_z_m", PAYLOAD_SIZE_Z_M)
            ),
            margin=float(grasp.get("tray_fit_margin_m", TRAY_FIT_MARGIN_M)),
            height_tolerance=float(
                grasp.get("tray_height_tolerance_m", 0.035)
            ),
        )
        print(
            "[TRAY_GATE] payload footprint fully inside tray "
            f"center_forward={tray_fit['center_forward_m']:+.3f} m "
            f"center_lateral={tray_fit['center_lateral_m']:+.3f} m "
            f"height_error={tray_fit['height_error_m']:.3f} m"
        )

        # Never lower while the measured slide is still extended.
        if extension_feedback() > slide_tolerance:
            raise RuntimeError(
                "Refusing to lower: suction slide is not fully retracted"
            )
        print("[STATE] RETRACT_PAYLOAD -> LOWER_PAYLOAD")
        mission.move_lift_to(0.0, lift_duration)
        print(f"[LIFT] payload lowered in {lift_duration:.2f} s")

        retreat_distance = float(grasp.get("post_pick_retreat_m", 0.0))
        if retreat_distance > 0.0:
            retreat_speed = float(
                grasp.get("post_pick_retreat_speed_mps", 0.18)
            )
            print(
                "[STATE] LOWER_PAYLOAD -> CLEAR_RACK_INFLATION "
                f"│ reverse={retreat_distance:.2f} m at {retreat_speed:.2f} m/s"
            )
            mission.drive_distance(-retreat_distance, speed=retreat_speed)
            print("[CLEARANCE_GATE] AGV base is back in the navigable aisle")

        if args.pick_only:
            print(
                f"[MISSION] PICK COMPLETE: {answer['model']} remains attached "
                "on the lowered tray"
            )
            print(
                "[TIMING] shelf arrival -> lowered payload: "
                f"{time.monotonic() - shelf_sequence_started:.2f} s"
            )
            return

        print("[STATE] RETRACT_PAYLOAD -> NAVIGATE_TO_PACKING")
        mission.publish_state(
            "RETURN_TO_DROPOFF", destination="packing_station"
        )
        drop_tray_fit = payload.verify_payload_on_tray(
            answer["model"],
            tray_size_x=float(grasp.get("tray_size_x_m", TRAY_SIZE_X_M)),
            tray_size_y=float(grasp.get("tray_size_y_m", TRAY_SIZE_Y_M)),
            payload_size_x=float(
                grasp.get("payload_size_x_m", PAYLOAD_SIZE_X_M)
            ),
            payload_size_y=float(
                grasp.get("payload_size_y_m", PAYLOAD_SIZE_Y_M)
            ),
            payload_size_z=float(
                grasp.get("payload_size_z_m", PAYLOAD_SIZE_Z_M)
            ),
            margin=float(grasp.get("tray_fit_margin_m", TRAY_FIT_MARGIN_M)),
            height_tolerance=float(
                grasp.get("tray_height_tolerance_m", 0.035)
            ),
        )
        drop_pose = release_approach_pose(
            drop_center,
            float(packing["approach"][2]),
            drop_tray_fit["center_forward_m"],
            drop_tray_fit["center_lateral_m"],
            slide_max,
        )
        print(
            "[DROP_ALIGN] measured tray offset "
            f"({drop_tray_fit['center_forward_m']:+.3f},"
            f" {drop_tray_fit['center_lateral_m']:+.3f}) m; "
            f"Nav2 release approach=({drop_pose[0]:.2f}, {drop_pose[1]:.2f})"
        )
        mission.navigate(
            answer["destination_anchor"], drop_pose, args.wait
        )
        print("[DROP_ALIGN] AGV is inside P; extending fork for release")
        print("[STATE] NAVIGATE_TO_PACKING -> RELEASE_PAYLOAD")
        mission.publish_state("PLACE_PACKAGE", payload=answer["model"])
        mission.move_suction_slide_to(
            slide_speed, slide_max, extension_feedback,
            slide_tolerance, slide_timeout,
        )
        payload.detach(answer["model"], float(grasp["attach_timeout_s"]))
        payload.stop_payload_motion(answer["model"])
        mission.move_suction_slide_to(
            -slide_speed, 0.0, extension_feedback,
            slide_tolerance, slide_timeout,
        )
        mission.move_lift_to(0.0, float(lift["duration_s"]))
        payload.lower_released_payload(
            answer["model"], target_center_z=drop_center_z
        )
        landed = payload.wait_for_drop(
            answer["model"],
            drop_center,
            drop_half_extents,
            target_center_z=drop_center_z,
        )
        print(
            f"[DROP_GATE] carton settled inside P at "
            f"({landed[0]:.2f}, {landed[1]:.2f}, z={landed[2]:.2f})"
        )
        print(f"[DROP] {answer['model']} delivered to Packing Station")
        print("[MISSION] COMPLETE")
        mission.publish_state("MISSION_COMPLETE", payload=answer["model"])
    finally:
        mission.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
