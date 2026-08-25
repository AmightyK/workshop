#!/usr/bin/env python3
"""Live V-JEPA streaming-QA dashboard plus warehouse navigation map.

The visible estimate is the raw camera-only pose at the center of its rolling
clip. Gazebo truth is timestamp-aligned only for evaluation. Optional odometry
projection is disabled by default and, when explicitly enabled, remains
telemetry-only rather than being drawn as pure V-JEPA.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import math
import os
import sys
import threading
import time
import unicodedata
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
WAREHOUSE_ROOT = WORKSPACE_ROOT / "warehouse_agv_demo"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import rclpy
from action_msgs.msg import GoalStatus, GoalStatusArray
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.transport13 import Node as GazeboNode
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from nav_msgs.msg import Odometry, Path as NavPath
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import Float32MultiArray, String

from src.data.ros_image import image_message_to_rgb, message_timestamp_sec

from src.evaluation.warehouse_context import (
    EntityPose,
    WarehouseRegionIndex,
    identify_forward_obstacle,
    scan_sector_min,
    wrap_angle,
)


TELEMETRY_WINDOW = "VL-JEPA Warehouse Streaming QA"
MAP_WINDOW = "Warehouse Map - Truth GPS & Planning"
STREAMING_WIDTH = 1440
STREAMING_HEIGHT = 568
LOW_LATENCY_IMAGE_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)
ACTIVE_NAV2_GOAL_STATES = frozenset({
    GoalStatus.STATUS_ACCEPTED,
    GoalStatus.STATUS_EXECUTING,
    GoalStatus.STATUS_CANCELING,
})


@dataclass(frozen=True)
class PoseSample:
    timestamp: float
    x: float
    y: float
    z: float
    yaw: float


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def gazebo_timestamp(message: Any) -> float:
    try:
        return float(message.header.stamp.sec) + float(message.header.stamp.nsec) * 1e-9
    except (AttributeError, TypeError):
        return 0.0


def ros_timestamp(message: PoseWithCovarianceStamped) -> float:
    return float(message.header.stamp.sec) + float(message.header.stamp.nanosec) * 1e-9


def ascii_text(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value).replace("đ", "d").replace("Đ", "D")
    return "".join(character for character in normalized if unicodedata.category(character) != "Mn")


def differential_keyboard_command(
    *, forward: bool, backward: bool, left: bool, right: bool
) -> tuple[float, float]:
    """Return simultaneous WASD linear/angular commands for a diff-drive base."""
    linear = 0.0
    angular = 0.0
    if forward != backward:
        linear = 1.0 if forward else -0.5
    if left != right:
        angular = 0.75 if left else -0.75
    return linear, angular


def goal_status_array_has_active_goal(message: GoalStatusArray) -> bool:
    """Return whether an action status array contains a live Nav2 goal."""
    return any(
        item.status in ACTIVE_NAV2_GOAL_STATES for item in message.status_list
    )


def plan_for_display(
    plan: tuple[tuple[float, float], ...],
    *,
    received_at: float,
    nav_goal_active: bool,
    now: float,
    startup_grace: float = 3.0,
) -> tuple[tuple[float, float], ...]:
    """Keep the once-computed A* path visible for the complete Nav2 goal."""
    if nav_goal_active or now - received_at <= startup_grace:
        return plan
    return ()


class X11KeyboardState:
    """Poll held keys, including simultaneous W+D, through XWayland/X11."""

    KEY_NAMES = ("w", "s", "a", "d", "space", "Escape")

    def __init__(self) -> None:
        self.library = None
        self.display = None
        self.keycodes: dict[str, int] = {}
        library_name = ctypes.util.find_library("X11")
        if not library_name or not os.environ.get("DISPLAY"):
            return
        library = ctypes.CDLL(library_name)
        library.XOpenDisplay.argtypes = [ctypes.c_char_p]
        library.XOpenDisplay.restype = ctypes.c_void_p
        library.XStringToKeysym.argtypes = [ctypes.c_char_p]
        library.XStringToKeysym.restype = ctypes.c_ulong
        library.XKeysymToKeycode.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        library.XKeysymToKeycode.restype = ctypes.c_ubyte
        library.XQueryKeymap.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        library.XQueryKeymap.restype = ctypes.c_int
        library.XCloseDisplay.argtypes = [ctypes.c_void_p]
        display = library.XOpenDisplay(None)
        if not display:
            return
        self.library = library
        self.display = display
        self.keycodes = {
            name: int(
                library.XKeysymToKeycode(
                    display, library.XStringToKeysym(name.encode("ascii"))
                )
            )
            for name in self.KEY_NAMES
        }

    @property
    def available(self) -> bool:
        return self.library is not None and self.display is not None

    def pressed(self) -> set[str]:
        if not self.available:
            return set()
        keymap = ctypes.create_string_buffer(32)
        self.library.XQueryKeymap(self.display, keymap)
        raw = keymap.raw
        return {
            name
            for name, code in self.keycodes.items()
            if code > 0 and raw[code >> 3] & (1 << (code & 7))
        }

    def close(self) -> None:
        if self.available:
            self.library.XCloseDisplay(self.display)
        self.library = None
        self.display = None


def project_pose_with_odometry(
    visual_pose: PoseSample,
    odom_at_visual_time: PoseSample,
    current_odom: PoseSample,
) -> PoseSample:
    """Bring a delayed visual pose to the latest odom time without truth data."""
    dx = current_odom.x - odom_at_visual_time.x
    dy = current_odom.y - odom_at_visual_time.y
    cos_origin = math.cos(odom_at_visual_time.yaw)
    sin_origin = math.sin(odom_at_visual_time.yaw)
    relative_x = cos_origin * dx + sin_origin * dy
    relative_y = -sin_origin * dx + cos_origin * dy
    cos_visual = math.cos(visual_pose.yaw)
    sin_visual = math.sin(visual_pose.yaw)
    return PoseSample(
        current_odom.timestamp,
        visual_pose.x + cos_visual * relative_x - sin_visual * relative_y,
        visual_pose.y + sin_visual * relative_x + cos_visual * relative_y,
        visual_pose.z,
        wrap_angle(
            visual_pose.yaw
            + wrap_angle(current_odom.yaw - odom_at_visual_time.yaw)
        ),
    )


class LocalizationDashboardNode(Node):
    """Join V-JEPA output and truth only in an external evaluation process."""

    def __init__(
        self,
        inventory: Path,
        *,
        camera_topic: str = "/vjepa/camera/image_raw",
        odom_projection_enabled: bool = False,
    ) -> None:
        super().__init__("vjepa_localization_dashboard")
        self.regions = WarehouseRegionIndex.from_inventory(inventory)
        self.odom_projection_enabled = odom_projection_enabled
        self.lock = threading.Lock()
        self.truth_history: deque[PoseSample] = deque(maxlen=12000)
        self.odom_history: deque[PoseSample] = deque(maxlen=12000)
        self.truth_trail: deque[tuple[float, float]] = deque(maxlen=2400)
        self.vjepa_trail: deque[tuple[float, float]] = deque(maxlen=1200)
        self.entities: tuple[EntityPose, ...] = ()
        self.current_truth: PoseSample | None = None
        self.aligned_truth: PoseSample | None = None
        self.raw_vjepa_pose: PoseSample | None = None
        self.vjepa_pose: PoseSample | None = None
        self.projected_vjepa_pose: PoseSample | None = None
        self.relative_dx: float | None = None
        self.relative_dy: float | None = None
        self.position_error: float | None = None
        self.yaw_error: float | None = None
        self.raw_position_error: float | None = None
        self.raw_yaw_error: float | None = None
        self.time_delta: float | None = None
        self.front_clearance = math.inf
        self.command_linear = 0.0
        self.command_angular = 0.0
        self.keyboard_active = False
        self.keyboard_linear = 0.0
        self.keyboard_angular = 0.0
        self.debug: dict[str, Any] = {}
        self.camera_bgr: np.ndarray | None = None
        self.camera_timestamp = 0.0
        self.query_latent: np.ndarray | None = None
        self.nav_status: dict[str, Any] = {}
        self.astar_plan: tuple[tuple[float, float], ...] = ()
        self.astar_plan_received = -math.inf
        self.nav_goal_active = {
            "navigate_to_pose": False,
            "navigate_through_poses": False,
        }
        self.sequence = 0

        self.create_subscription(
            PoseWithCovarianceStamped, "/vjepa_pose", self._on_vjepa, 20
        )
        self.create_subscription(String, "/vjepa_localization/debug", self._on_debug, 20)
        self.create_subscription(
            Float32MultiArray, "/vjepa_latent", self._on_latent, 20
        )
        self.create_subscription(
            Image, camera_topic, self._on_camera, LOW_LATENCY_IMAGE_QOS
        )
        self.create_subscription(
            String, "/nav/localization_status", self._on_nav_status, 20
        )
        self.create_subscription(NavPath, "/plan", self._on_plan, 10)
        # The controller follows this rounded A* path. It arrives just after
        # the raw planner output and therefore becomes the blue planning line
        # shown on the map for the rest of the active action.
        self.create_subscription(NavPath, "/plan_smoothed", self._on_plan, 10)
        self.create_subscription(
            GoalStatusArray,
            "/navigate_to_pose/_action/status",
            lambda message: self._on_action_status("navigate_to_pose", message),
            10,
        )
        self.create_subscription(
            GoalStatusArray,
            "/navigate_through_poses/_action/status",
            lambda message: self._on_action_status(
                "navigate_through_poses", message
            ),
            10,
        )
        if self.odom_projection_enabled:
            self.create_subscription(
                Odometry, "/odom", self._on_odom, qos_profile_sensor_data
            )
        self.create_subscription(LaserScan, "/scan", self._on_scan, qos_profile_sensor_data)
        self.create_subscription(Twist, "/cmd_vel", self._on_cmd_vel, 20)
        # A priority mux sends manual commands around the Nav2 smoother, then
        # through the collision monitor. A direct publisher is a fallback when
        # the dashboard is used without Nav2.
        self.keyboard_mux_velocity = self.create_publisher(
            Twist, "/cmd_vel_keyboard", 10
        )
        self.keyboard_direct_velocity = self.create_publisher(Twist, "/cmd_vel", 10)
        self.gz_node = GazeboNode()
        if not self.gz_node.subscribe(
            Pose_V, "/world/world_demo/pose/info", self._on_world
        ):
            raise RuntimeError("không thể subscribe Gazebo pose/info")
        projection_state = "enabled (telemetry only)" if odom_projection_enabled else "disabled"
        self.get_logger().info(
            f"Dashboard camera={camera_topic}; raw camera-only V-JEPA; "
            f"odom projection is {projection_state}"
        )

    def _refresh_live_vjepa_locked(self) -> None:
        raw = self.raw_vjepa_pose
        if raw is None:
            return
        if not self.odom_projection_enabled:
            self.projected_vjepa_pose = None
            return
        live = raw
        if self.odom_history:
            odom_at_visual_time = min(
                self.odom_history,
                key=lambda item: abs(item.timestamp - raw.timestamp),
            )
            current_odom = self.odom_history[-1]
            if abs(odom_at_visual_time.timestamp - raw.timestamp) <= 0.35:
                live = project_pose_with_odometry(
                    raw, odom_at_visual_time, current_odom
                )
        self.projected_vjepa_pose = live

    def _on_world(self, message: Pose_V) -> None:
        timestamp = gazebo_timestamp(message)
        if timestamp <= 0.0:
            timestamp = self.get_clock().now().nanoseconds * 1e-9
        truth = None
        entities: list[EntityPose] = []
        for pose in message.pose:
            if pose.name == "warehouse_agv":
                q = pose.orientation
                truth = PoseSample(
                    timestamp,
                    float(pose.position.x),
                    float(pose.position.y),
                    float(pose.position.z),
                    yaw_from_quaternion(q.x, q.y, q.z, q.w),
                )
            elif pose.name.startswith(("road_box_static_", "random_worker_")):
                entities.append(
                    EntityPose(pose.name, float(pose.position.x), float(pose.position.y))
                )
        if truth is None:
            return
        with self.lock:
            self.truth_history.append(truth)
            self.current_truth = truth
            if (
                not self.truth_trail
                or math.hypot(
                    truth.x - self.truth_trail[-1][0],
                    truth.y - self.truth_trail[-1][1],
                )
                >= 0.05
            ):
                self.truth_trail.append((truth.x, truth.y))
            self.entities = tuple(entities)
            self._refresh_live_vjepa_locked()

    def _on_odom(self, message: Odometry) -> None:
        timestamp = (
            float(message.header.stamp.sec)
            + float(message.header.stamp.nanosec) * 1e-9
        )
        q = message.pose.pose.orientation
        odom = PoseSample(
            timestamp,
            float(message.pose.pose.position.x),
            float(message.pose.pose.position.y),
            float(message.pose.pose.position.z),
            yaw_from_quaternion(q.x, q.y, q.z, q.w),
        )
        with self.lock:
            self.odom_history.append(odom)
            self._refresh_live_vjepa_locked()

    def _on_scan(self, message: LaserScan) -> None:
        clearance = scan_sector_min(
            message.ranges,
            angle_min=float(message.angle_min),
            angle_increment=float(message.angle_increment),
            range_min=float(message.range_min),
            range_max=float(message.range_max),
            half_width_rad=0.65,
        )
        with self.lock:
            self.front_clearance = clearance

    def _on_cmd_vel(self, message: Twist) -> None:
        with self.lock:
            self.command_linear = float(message.linear.x)
            self.command_angular = float(message.angular.z)

    def _publish_keyboard_velocity(self, linear: float, angular: float) -> None:
        command = Twist()
        command.linear.x = float(linear)
        command.angular.z = float(angular)
        if self.keyboard_mux_velocity.get_subscription_count() > 0:
            self.keyboard_mux_velocity.publish(command)
        else:
            self.keyboard_direct_velocity.publish(command)

    def handle_keyboard_key(self, key: int) -> bool:
        """Translate an OpenCV window key into safe differential-drive teleop."""
        if key == 27:
            self.stop_keyboard()
            return False
        commands = {
            ord("w"): (1.00, 0.0), ord("W"): (1.00, 0.0),
            ord("s"): (-0.50, 0.0), ord("S"): (-0.50, 0.0),
            ord("a"): (0.0, 0.75), ord("A"): (0.0, 0.75),
            ord("d"): (0.0, -0.75), ord("D"): (0.0, -0.75),
            ord("q"): (0.65, 0.60), ord("Q"): (0.65, 0.60),
            ord("e"): (0.65, -0.60), ord("E"): (0.65, -0.60),
            ord("z"): (-0.40, -0.55), ord("Z"): (-0.40, -0.55),
            ord("c"): (-0.40, 0.55), ord("C"): (-0.40, 0.55),
            32: (0.0, 0.0),
            65362: (1.00, 0.0), 2490368: (1.00, 0.0),
            65364: (-0.50, 0.0), 2621440: (-0.50, 0.0),
            65361: (0.0, 0.75), 2424832: (0.0, 0.75),
            65363: (0.0, -0.75), 2555904: (0.0, -0.75),
        }
        command = commands.get(key)
        if command is None:
            return True
        self.keyboard_linear, self.keyboard_angular = command
        self.keyboard_active = command != (0.0, 0.0)
        self._publish_keyboard_velocity(*command)
        return True

    def refresh_keyboard_command(self) -> None:
        """Keep a selected manual command alive until Space or Esc is pressed."""
        if self.keyboard_active:
            self._publish_keyboard_velocity(
                self.keyboard_linear, self.keyboard_angular
            )

    def update_held_keyboard(self, pressed: set[str]) -> bool:
        """Publish a vehicle-style command from the currently held keys."""
        if "Escape" in pressed:
            self.stop_keyboard()
            return False
        linear, angular = differential_keyboard_command(
            forward="w" in pressed,
            backward="s" in pressed,
            left="a" in pressed,
            right="d" in pressed,
        )
        if "space" in pressed:
            linear, angular = 0.0, 0.0
        if linear == 0.0 and angular == 0.0:
            self.stop_keyboard()
            return True
        self.keyboard_linear = linear
        self.keyboard_angular = angular
        self.keyboard_active = True
        self._publish_keyboard_velocity(linear, angular)
        return True

    def stop_keyboard(self) -> None:
        if self.keyboard_active:
            self._publish_keyboard_velocity(0.0, 0.0)
        self.keyboard_active = False
        self.keyboard_linear = 0.0
        self.keyboard_angular = 0.0

    def _on_debug(self, message: String) -> None:
        try:
            value = json.loads(message.data)
        except json.JSONDecodeError:
            return
        with self.lock:
            self.debug = value

    def _on_camera(self, message: Image) -> None:
        try:
            rgb = image_message_to_rgb(message)
        except ValueError:
            return
        with self.lock:
            self.camera_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            self.camera_timestamp = message_timestamp_sec(message)

    def _on_latent(self, message: Float32MultiArray) -> None:
        latent = np.asarray(message.data, dtype=np.float32)
        if latent.ndim != 1 or latent.size < 2 or not np.isfinite(latent).all():
            return
        with self.lock:
            self.query_latent = latent

    def _on_nav_status(self, message: String) -> None:
        try:
            value = json.loads(message.data)
        except json.JSONDecodeError:
            return
        with self.lock:
            self.nav_status = value

    def _on_plan(self, message: NavPath) -> None:
        points = tuple(
            (float(item.pose.position.x), float(item.pose.position.y))
            for item in message.poses
        )
        with self.lock:
            self.astar_plan = points
            self.astar_plan_received = time.monotonic()

    def _on_action_status(
        self, action_name: str, message: GoalStatusArray
    ) -> None:
        with self.lock:
            self.nav_goal_active[action_name] = (
                goal_status_array_has_active_goal(message)
            )

    def _on_vjepa(self, message: PoseWithCovarianceStamped) -> None:
        timestamp = ros_timestamp(message)
        q = message.pose.pose.orientation
        prediction = PoseSample(
            timestamp,
            float(message.pose.pose.position.x),
            float(message.pose.pose.position.y),
            float(message.pose.pose.position.z),
            yaw_from_quaternion(q.x, q.y, q.z, q.w),
        )
        with self.lock:
            if not self.truth_history:
                return
            truth = min(
                self.truth_history,
                key=lambda item: abs(item.timestamp - timestamp),
            )
            raw_dx = prediction.x - truth.x
            raw_dy = prediction.y - truth.y
            self.raw_vjepa_pose = prediction
            self.vjepa_pose = prediction
            self.aligned_truth = truth
            self.raw_position_error = math.hypot(raw_dx, raw_dy)
            self.raw_yaw_error = abs(wrap_angle(prediction.yaw - truth.yaw))
            self.relative_dx = raw_dx
            self.relative_dy = raw_dy
            self.position_error = self.raw_position_error
            self.yaw_error = self.raw_yaw_error
            self.time_delta = abs(prediction.timestamp - truth.timestamp)
            if (
                not self.vjepa_trail
                or math.hypot(
                    prediction.x - self.vjepa_trail[-1][0],
                    prediction.y - self.vjepa_trail[-1][1],
                )
                >= 0.05
            ):
                self.vjepa_trail.append((prediction.x, prediction.y))
            self._refresh_live_vjepa_locked()
            self.sequence += 1

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            current = self.current_truth
            prediction = self.vjepa_pose
            raw_prediction = self.raw_vjepa_pose
            projected_prediction = self.projected_vjepa_pose
            aligned = self.aligned_truth
            entities = self.entities
            clearance = self.front_clearance
            angular = self.command_angular
            linear = self.command_linear
            debug = dict(self.debug)
            nav_status = dict(self.nav_status)
            camera_bgr = None if self.camera_bgr is None else self.camera_bgr.copy()
            camera_timestamp = self.camera_timestamp
            query_latent = (
                None if self.query_latent is None else self.query_latent.copy()
            )
            # This demo computes A* once per goal, so /plan is intentionally
            # not republished. Keep that path for the action's whole lifetime;
            # the short grace covers the plan/status callback startup race.
            astar_plan = plan_for_display(
                self.astar_plan,
                received_at=self.astar_plan_received,
                nav_goal_active=any(self.nav_goal_active.values()),
                now=time.monotonic(),
            )
            snapshot = {
                "sequence": self.sequence,
                "current_truth": current,
                "aligned_truth": aligned,
                "vjepa": prediction,
                "raw_vjepa": raw_prediction,
                "projected_vjepa": projected_prediction,
                "relative_dx": self.relative_dx,
                "relative_dy": self.relative_dy,
                "position_error": self.position_error,
                "yaw_error": self.yaw_error,
                "raw_position_error": self.raw_position_error,
                "raw_yaw_error": self.raw_yaw_error,
                "time_delta": self.time_delta,
                "truth_trail": tuple(self.truth_trail),
                "vjepa_trail": tuple(self.vjepa_trail),
                "astar_plan": astar_plan,
                "top1_similarity": debug.get("top1_similarity"),
                "confidence_margin": debug.get("confidence_margin"),
                "tracking_state": debug.get("tracking_state", "WAITING"),
                "camera_moving": debug.get("camera_moving", False),
                "camera_motion_score": debug.get("camera_motion_score"),
                "camera_pixel_change": debug.get("camera_pixel_change"),
                "camera_motion_inlier_ratio": debug.get("camera_motion_inlier_ratio"),
                "camera_progress_scale": debug.get("camera_progress_scale"),
                "motion_credit": debug.get("motion_credit"),
                "raw_jump_m": debug.get("raw_jump_m"),
                "accepted_step_m": debug.get("accepted_step_m"),
                "translation_gate_m": debug.get("translation_gate_m"),
                "selected_rank": debug.get("selected_rank"),
                "rejected_streak": debug.get("rejected_streak", 0),
                "nav_localization_state": nav_status.get("state", "WAITING"),
                "nav_localization_source": nav_status.get("source", "WAITING"),
                "nav_uses_gazebo_truth": nav_status.get("uses_gazebo_truth"),
                "nav_planner": nav_status.get("planner", "NAVFN_ASTAR"),
                "nav_correction_m": nav_status.get("correction_m"),
                "camera_bgr": camera_bgr,
                "camera_timestamp": camera_timestamp,
                "query_latent": query_latent,
                "source_id": debug.get("source_id"),
                "latent_dimension": debug.get("latent_dimension"),
                "compute_host": debug.get("compute_host", "waiting"),
                "inference_ms": debug.get("inference_ms"),
                "result_pose_topic": debug.get("pose_topic", "/vjepa_pose"),
                "result_latent_topic": debug.get("latent_topic", "/vjepa_latent"),
                "odom_projection_enabled": self.odom_projection_enabled,
            }
        obstacle = None
        if current is not None:
            obstacle = identify_forward_obstacle(
                agv_x=current.x,
                agv_y=current.y,
                agv_yaw=current.yaw,
                entities=entities,
                lidar_clearance_m=clearance,
                detection_distance_m=2.2,
                front_half_angle_rad=0.65,
            )
            snapshot["area"] = self.regions.describe(current.x, current.y)
        else:
            snapshot["area"] = "đang chờ Gazebo truth"
        snapshot["obstacle"] = obstacle
        snapshot["current_display_gap"] = (
            math.hypot(prediction.x - current.x, prediction.y - current.y)
            if prediction is not None and current is not None
            else None
        )
        snapshot["vjepa_age_ms"] = (
            max(0.0, current.timestamp - prediction.timestamp) * 1000.0
            if prediction is not None and current is not None
            else None
        )
        snapshot["linear_x"] = linear
        snapshot["angular_z"] = angular
        # Gazebo pose and camera stamps share simulation time even when this
        # standalone evaluator was started without use_sim_time. Prefer that
        # common clock so the on-screen latency never mixes wall and sim time.
        now = (
            current.timestamp
            if current is not None and current.timestamp > 0.0
            else 0.0
        )
        snapshot["camera_age_ms"] = (
            max(0.0, now - camera_timestamp) * 1000.0
            if camera_timestamp > 0.0 and now > 0.0
            else None
        )
        nearby_people: list[tuple[float, str]] = []
        people_ahead: list[tuple[float, str]] = []
        if current is not None:
            for entity in entities:
                if not entity.name.startswith("random_worker_"):
                    continue
                dx, dy = entity.x - current.x, entity.y - current.y
                distance = math.hypot(dx, dy)
                if distance <= 4.0:
                    nearby_people.append((distance, entity.name))
                forward = math.cos(current.yaw) * dx + math.sin(current.yaw) * dy
                lateral = abs(-math.sin(current.yaw) * dx + math.cos(current.yaw) * dy)
                if 0.0 < forward <= 3.5 and lateral <= 1.5:
                    people_ahead.append((distance, entity.name))
        snapshot["nearby_people"] = tuple(sorted(nearby_people))
        snapshot["people_ahead"] = tuple(sorted(people_ahead))
        return snapshot


class LatentProjector:
    """Project the saved map and live 1024-D V-JEPA vector into one PCA plane."""

    def __init__(self, map_directory: Path) -> None:
        embeddings = np.load(map_directory / "global_embeddings.npy").astype(np.float32)
        ids = np.load(map_directory / "ids.npy", allow_pickle=False)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-8)
        self.mean = embeddings.mean(axis=0)
        centered = embeddings - self.mean
        _, _, basis = np.linalg.svd(centered, full_matrices=False)
        self.basis = basis[:2]
        self.map_points = centered @ self.basis.T
        self.id_to_index = {str(value): index for index, value in enumerate(ids)}
        lower = np.quantile(self.map_points, 0.01, axis=0)
        upper = np.quantile(self.map_points, 0.99, axis=0)
        padding = np.maximum((upper - lower) * 0.12, 1e-3)
        self.lower = lower - padding
        self.upper = upper + padding
        self.dimension = int(embeddings.shape[1])

    def project(self, latent: np.ndarray | None) -> np.ndarray | None:
        if latent is None or latent.shape != (self.dimension,):
            return None
        vector = latent.astype(np.float32)
        vector /= max(float(np.linalg.norm(vector)), 1e-8)
        return (vector - self.mean) @ self.basis.T

    def selected(self, source_id: Any) -> np.ndarray | None:
        index = self.id_to_index.get(str(source_id))
        return None if index is None else self.map_points[index]

    def to_pixel(
        self, point: np.ndarray, x: int, y: int, width: int, height: int
    ) -> tuple[int, int]:
        normalized = (point - self.lower) / np.maximum(self.upper - self.lower, 1e-8)
        px = x + int(np.clip(normalized[0], 0.0, 1.0) * width)
        py = y + height - int(np.clip(normalized[1], 0.0, 1.0) * height)
        return px, py

    def draw(
        self,
        canvas: np.ndarray,
        bounds: tuple[int, int, int, int],
        query_latent: np.ndarray | None,
        source_id: Any,
    ) -> None:
        x, y, width, height = bounds
        cv2.rectangle(canvas, (x, y), (x + width, y + height), (244, 244, 240), -1)
        cv2.rectangle(canvas, (x, y), (x + width, y + height), (92, 98, 108), 1)
        for point in self.map_points:
            cv2.circle(
                canvas,
                self.to_pixel(point, x + 18, y + 40, width - 36, height - 62),
                1,
                (145, 150, 155),
                -1,
                cv2.LINE_AA,
            )
        cv2.putText(
            canvas,
            "V-JEPA LATENT SPACE (PCA)",
            (x + 16, y + 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.53,
            (45, 48, 54),
            1,
            cv2.LINE_AA,
        )
        plot = (x + 18, y + 40, width - 36, height - 62)
        instant = self.project(query_latent)
        stable = self.selected(source_id)
        if instant is not None:
            cv2.circle(canvas, self.to_pixel(instant, *plot), 7, (45, 45, 235), -1, cv2.LINE_AA)
        if stable is not None:
            cv2.circle(canvas, self.to_pixel(stable, *plot), 7, (235, 90, 35), -1, cv2.LINE_AA)
        if instant is not None and stable is not None:
            cv2.line(
                canvas,
                self.to_pixel(instant, *plot),
                self.to_pixel(stable, *plot),
                (170, 135, 115),
                1,
                cv2.LINE_AA,
            )


class PreparedQA:
    """Rotate prepared warehouse questions and temporally stabilize their answers."""

    def __init__(self, config_path: Path) -> None:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.questions = tuple(config["questions"])
        if not self.questions:
            raise ValueError("live question list is empty")
        self.interval_sec = float(config.get("interval_sec", 3.5))
        self.stabilize_frames = max(1, int(config.get("stabilize_frames", 2)))
        self.started = time.monotonic()
        self.last_sequence = -1
        self.instant: dict[str, str] = {}
        self.stable: dict[str, str] = {}
        self.candidate_key: dict[str, str] = {}
        self.candidate_count: dict[str, int] = {}

    @staticmethod
    def _answer(question_id: str, snapshot: dict[str, Any]) -> tuple[str, str]:
        if question_id == "area":
            area = ascii_text(str(snapshot["area"]))
            return area, f"Robot is in {area}."
        if question_id == "obstacle":
            obstacle = snapshot["obstacle"]
            if obstacle is None:
                return "clear", "No obstacle is detected in front."
            label = ascii_text(obstacle.label)
            return obstacle.name, f"Yes. {label} is {obstacle.clearance_m:.1f} m ahead."
        if question_id == "person":
            people = snapshot["people_ahead"] or snapshot["nearby_people"]
            if not people:
                return "no_person", "No person is close to the current route."
            distance, name = people[0]
            number = str(name).removeprefix("random_worker_")
            relation = "ahead" if snapshot["people_ahead"] else "nearby"
            return str(name), f"Worker {number} is {relation}, about {distance:.1f} m away."
        if question_id == "avoidance":
            obstacle = snapshot["obstacle"]
            angular = float(snapshot["angular_z"])
            if obstacle is None:
                return "none", "Nothing now; the LiDAR corridor is clear."
            label = ascii_text(obstacle.label)
            if abs(angular) < 0.12:
                return f"wait:{obstacle.name}", f"Waiting for {label} to clear."
            direction = "left" if angular > 0.0 else "right"
            return f"{obstacle.name}:{direction}", f"Turning {direction} to avoid {label}."
        if question_id == "motion":
            linear = float(snapshot["linear_x"])
            angular = float(snapshot["angular_z"])
            if abs(linear) < 0.03 and abs(angular) < 0.05:
                return "stopped", "The robot is stopped."
            if abs(angular) >= 0.12:
                direction = "left" if angular > 0.0 else "right"
                return f"turn:{direction}", f"The robot is moving and turning {direction}."
            return "forward", "The robot is moving forward on the aisle."
        if question_id == "route":
            count = len(snapshot["astar_plan"])
            if count < 2:
                return "no_plan", "A* is waiting for the next route goal."
            return "plan_active", f"Yes. The active A* path has {count} poses."
        if question_id == "tracking":
            state = str(snapshot["tracking_state"])
            similarity = snapshot["top1_similarity"]
            if isinstance(similarity, (int, float)):
                return state, f"Temporal tracking is {state}; similarity is {similarity:.3f}."
            return state, f"Temporal tracking is {state}; waiting for a latent match."
        if question_id == "comparison":
            error = snapshot["position_error"]
            if not isinstance(error, (int, float)):
                return "waiting", "Waiting for timestamp-aligned Gazebo truth."
            band = "low" if error < 0.5 else "medium" if error < 1.5 else "high"
            return band, f"V-JEPA differs from Gazebo truth by {error:.2f} m."
        return "unsupported", "No prepared answer for this question."

    def update(self, snapshot: dict[str, Any]) -> None:
        sequence = int(snapshot["sequence"])
        if sequence == self.last_sequence:
            return
        self.last_sequence = sequence
        for spec in self.questions:
            question_id = str(spec["id"])
            key, answer = self._answer(question_id, snapshot)
            self.instant[question_id] = answer
            if self.candidate_key.get(question_id) == key:
                self.candidate_count[question_id] = self.candidate_count.get(question_id, 0) + 1
            else:
                self.candidate_key[question_id] = key
                self.candidate_count[question_id] = 1
            if question_id not in self.stable or self.candidate_count[question_id] >= self.stabilize_frames:
                self.stable[question_id] = answer

    def active(self, now: float | None = None) -> dict[str, Any]:
        elapsed = (time.monotonic() if now is None else now) - self.started
        index = int(max(0.0, elapsed) / self.interval_sec) % len(self.questions)
        spec = self.questions[index]
        question_id = str(spec["id"])
        return {
            "index": index,
            "count": len(self.questions),
            "id": question_id,
            "question": str(spec["text"]),
            "instant": self.instant.get(question_id, "Collecting live context..."),
            "stable": self.stable.get(question_id, "Collecting live context..."),
        }


class DashboardRenderer:
    def __init__(
        self,
        map_yaml: Path,
        regions: WarehouseRegionIndex,
        latent_map: Path,
    ) -> None:
        with map_yaml.open(encoding="utf-8") as stream:
            metadata = yaml.safe_load(stream)
        image_path = Path(str(metadata["image"]))
        if not image_path.is_absolute():
            image_path = map_yaml.parent / image_path
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"không đọc được map image: {image_path}")
        self.map_gray = image
        self.resolution = float(metadata["resolution"])
        self.origin_x = float(metadata["origin"][0])
        self.origin_y = float(metadata["origin"][1])
        self.scale = min(0.82, 820.0 / image.shape[0])
        self.map_width = int(round(image.shape[1] * self.scale))
        self.map_height = int(round(image.shape[0] * self.scale))
        self.regions = regions
        self.latent = LatentProjector(latent_map)

    def world_to_pixel(self, x: float, y: float) -> tuple[int, int]:
        column = (x - self.origin_x) / self.resolution
        row = self.map_gray.shape[0] - 1.0 - (y - self.origin_y) / self.resolution
        return int(round(column * self.scale)), int(round(row * self.scale))

    @staticmethod
    def _polyline(
        image: np.ndarray,
        points: tuple[tuple[float, float], ...],
        convert,
        color: tuple[int, int, int],
        thickness: int,
    ) -> None:
        if len(points) < 2:
            return
        pixels = np.asarray([convert(x, y) for x, y in points], dtype=np.int32)
        cv2.polylines(image, [pixels], False, color, thickness, cv2.LINE_AA)

    def _arrow(
        self,
        image: np.ndarray,
        pose: PoseSample,
        color: tuple[int, int, int],
        radius: int = 9,
    ) -> None:
        start = self.world_to_pixel(pose.x, pose.y)
        end = self.world_to_pixel(
            pose.x + 0.8 * math.cos(pose.yaw),
            pose.y + 0.8 * math.sin(pose.yaw),
        )
        cv2.circle(image, start, radius, color, -1, cv2.LINE_AA)
        cv2.arrowedLine(image, start, end, color, 3, cv2.LINE_AA, tipLength=0.35)

    @staticmethod
    def _line(
        image: np.ndarray,
        text: str,
        x: int,
        y: int,
        *,
        color: tuple[int, int, int] = (225, 225, 225),
        scale: float = 0.52,
        thickness: int = 1,
    ) -> None:
        cv2.putText(
            image,
            ascii_text(text),
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    @staticmethod
    def _pose_text(pose: PoseSample | None) -> str:
        if pose is None:
            return "waiting..."
        return f"x={pose.x:+.2f}  y={pose.y:+.2f}  yaw={math.degrees(pose.yaw):+.1f} deg"

    @staticmethod
    def avoidance_comment(snapshot: dict[str, Any]) -> str:
        obstacle = snapshot["obstacle"]
        angular = float(snapshot["angular_z"])
        if obstacle is None:
            return "CLEAR: no obstacle in front"
        label = ascii_text(obstacle.label)
        if abs(angular) >= 0.12:
            direction = "LEFT" if angular > 0.0 else "RIGHT"
            return f"AVOID: turning {direction} around {label}"
        return f"OBSTACLE: {label} at {obstacle.clearance_m:.2f} m"

    def render_streaming(
        self, snapshot: dict[str, Any], qa: dict[str, Any]
    ) -> np.ndarray:
        """Render the reference-video layout: camera left, latent plane right."""
        width, height = STREAMING_WIDTH, STREAMING_HEIGHT
        canvas = np.full((height, width, 3), (18, 22, 28), dtype=np.uint8)
        cv2.rectangle(canvas, (0, 0), (width, 82), (43, 37, 32), -1)
        self._line(
            canvas,
            f"Query [{qa['index'] + 1}/{qa['count']}]: {qa['question']}",
            24,
            32,
            color=(245, 245, 245),
            scale=0.62,
            thickness=2,
        )
        self._line(
            canvas,
            f"Model: {qa['stable']}",
            24,
            64,
            color=(120, 235, 255),
            scale=0.58,
            thickness=2,
        )
        self._line(
            canvas,
            (
                f"DDS {snapshot['compute_host']} | "
                + (
                    f"infer {snapshot['inference_ms']:.0f} ms"
                    if isinstance(snapshot["inference_ms"], (int, float))
                    else "waiting for Orin"
                )
            ),
            1080,
            48,
            color=(100, 255, 180),
            scale=0.46,
            thickness=2,
        )

        camera_x, camera_y, camera_w, camera_h = 20, 98, 800, 450
        camera = snapshot["camera_bgr"]
        if camera is None:
            camera_view = np.full((camera_h, camera_w, 3), (30, 34, 40), dtype=np.uint8)
            self._line(
                camera_view,
                "Waiting for DDS /vjepa/camera/image_raw (640x360, 16:9)...",
                100,
                230,
                scale=0.56,
            )
        else:
            camera_view = cv2.resize(camera, (camera_w, camera_h), interpolation=cv2.INTER_AREA)
        canvas[camera_y : camera_y + camera_h, camera_x : camera_x + camera_w] = camera_view
        cv2.rectangle(
            canvas,
            (camera_x, camera_y),
            (camera_x + camera_w, camera_y + camera_h),
            (105, 112, 122),
            1,
        )
        cv2.rectangle(canvas, (camera_x, camera_y), (camera_x + 470, camera_y + 34), (18, 22, 28), -1)
        camera_age = snapshot["camera_age_ms"]
        age_text = (
            f"{camera_age:.0f} ms"
            if isinstance(camera_age, (int, float))
            else "waiting"
        )
        self._line(
            canvas,
            f"AGV CAMERA  16:9  |  latest frame: {age_text}  |  {ascii_text(str(snapshot['area']))}",
            camera_x + 10,
            camera_y + 23,
            color=(235, 235, 235),
            scale=0.43,
            thickness=1,
        )

        latent_bounds = (840, 98, 580, 450)
        self.latent.draw(
            canvas,
            latent_bounds,
            snapshot["query_latent"],
            snapshot["source_id"],
        )
        cv2.circle(canvas, (864, 522), 6, (45, 45, 235), -1)
        self._line(canvas, "Instant query embedding", 878, 527, color=(45, 45, 80), scale=0.42)
        cv2.circle(canvas, (1058, 522), 6, (235, 90, 35), -1)
        self._line(canvas, "Temporal match", 1072, 527, color=(45, 45, 80), scale=0.42)

        return canvas

    def render_map(self, snapshot: dict[str, Any]) -> np.ndarray:
        """Render the bare map with Truth GPS and planning overlaid on it."""
        map_image = cv2.resize(
            self.map_gray,
            (self.map_width, self.map_height),
            interpolation=cv2.INTER_NEAREST,
        )
        canvas = cv2.cvtColor(map_image, cv2.COLOR_GRAY2BGR)
        self._polyline(canvas, snapshot["astar_plan"], self.world_to_pixel, (255, 170, 0), 3)
        self._polyline(canvas, snapshot["truth_trail"], self.world_to_pixel, (30, 30, 245), 2)

        current = snapshot["current_truth"]
        if current is not None:
            self._arrow(canvas, current, (30, 30, 245))
        return canvas


def serializable_snapshot(
    snapshot: dict[str, Any], qa: dict[str, Any] | None = None
) -> dict[str, Any]:
    def pose_value(pose: PoseSample | None) -> list[float] | None:
        if pose is None:
            return None
        return [pose.x, pose.y, pose.z, pose.yaw]

    obstacle = snapshot["obstacle"]
    value = {
        "area": snapshot["area"],
        "gazebo_truth": pose_value(snapshot["current_truth"]),
        "vjepa_pose": pose_value(snapshot["vjepa"]),
        "vjepa_raw_pose": pose_value(snapshot["raw_vjepa"]),
        "vjepa_odom_projected_pose": pose_value(snapshot["projected_vjepa"]),
        "odom_projection_enabled": snapshot["odom_projection_enabled"],
        "relative_dx": snapshot["relative_dx"],
        "relative_dy": snapshot["relative_dy"],
        "position_error_m": snapshot["position_error"],
        "yaw_error_rad": snapshot["yaw_error"],
        "raw_timestamp_aligned_position_error_m": snapshot["raw_position_error"],
        "raw_timestamp_aligned_yaw_error_rad": snapshot["raw_yaw_error"],
        "current_display_gap_m": snapshot["current_display_gap"],
        "tracking_state": snapshot["tracking_state"],
        "camera_moving": snapshot["camera_moving"],
        "camera_motion_score": snapshot["camera_motion_score"],
        "camera_pixel_change": snapshot["camera_pixel_change"],
        "camera_motion_inlier_ratio": snapshot["camera_motion_inlier_ratio"],
        "camera_progress_scale": snapshot["camera_progress_scale"],
        "motion_credit": snapshot["motion_credit"],
        "raw_jump_m": snapshot["raw_jump_m"],
        "accepted_step_m": snapshot["accepted_step_m"],
        "translation_gate_m": snapshot["translation_gate_m"],
        "selected_rank": snapshot["selected_rank"],
        "rejected_streak": snapshot["rejected_streak"],
        "nav_localization_state": snapshot["nav_localization_state"],
        "nav_localization_source": snapshot["nav_localization_source"],
        "nav_uses_gazebo_truth": snapshot["nav_uses_gazebo_truth"],
        "nav_planner": snapshot["nav_planner"],
        "nav_correction_m": snapshot["nav_correction_m"],
        "astar_plan_points": len(snapshot["astar_plan"]),
        "obstacle": obstacle.label if obstacle is not None else None,
        "comment": DashboardRenderer.avoidance_comment(snapshot),
        "latent_dimension": snapshot["latent_dimension"],
        "compute_host": snapshot["compute_host"],
        "inference_ms": snapshot["inference_ms"],
        "result_pose_topic": snapshot["result_pose_topic"],
        "result_latent_topic": snapshot["result_latent_topic"],
        "camera_age_ms": snapshot["camera_age_ms"],
        "vjepa_age_ms": snapshot["vjepa_age_ms"],
    }
    if qa is not None:
        value["prepared_qa"] = {
            "index": qa["index"],
            "question": qa["question"],
            "instant": qa["instant"],
            "stabilized": qa["stable"],
        }
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inventory",
        type=Path,
        default=WAREHOUSE_ROOT / "config" / "inventory_locations.yaml",
    )
    parser.add_argument(
        "--map-yaml",
        type=Path,
        default=WAREHOUSE_ROOT / "maps" / "warehouse_lidar.yaml",
    )
    parser.add_argument(
        "--latent-map",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "autonomous_map_dense",
    )
    parser.add_argument(
        "--camera-topic",
        default="/vjepa/camera/image_raw",
        help="ROS 2 DDS image topic received by V-JEPA",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=PROJECT_ROOT / "configs" / "warehouse_live_questions.yaml",
    )
    parser.add_argument("--headless", action="store_true", help="print snapshots without GUI")
    parser.add_argument("--duration", type=float, default=0.0, help="optional run time for smoke tests")
    parser.add_argument("--refresh-hz", type=float, default=32.0)
    parser.add_argument(
        "--map-refresh-hz",
        type=float,
        default=5.0,
        help="redraw the heavier occupancy-map window independently of camera FPS",
    )
    parser.add_argument(
        "--odom-projection",
        action="store_true",
        help="add short-term wheel-odom projection to JSON telemetry only",
    )
    args, ros_args = parser.parse_known_args()
    if args.duration < 0.0 or args.refresh_hz <= 0.0 or args.map_refresh_hz <= 0.0:
        parser.error("duration must be non-negative and refresh rates must be positive")
    if not args.headless and not os.environ.get("DISPLAY"):
        parser.error("DISPLAY is not set; use --headless or run inside the desktop session")

    rclpy.init(args=ros_args)
    node = LocalizationDashboardNode(
        args.inventory.resolve(),
        camera_topic=args.camera_topic,
        odom_projection_enabled=args.odom_projection,
    )
    renderer = DashboardRenderer(
        args.map_yaml.resolve(), node.regions, args.latent_map.resolve()
    )
    qa_engine = PreparedQA(args.questions.resolve())
    started = time.monotonic()
    last_sequence = -1
    last_question_index = -1
    last_map_render = -math.inf
    cached_map: np.ndarray | None = None
    held_keyboard = X11KeyboardState() if not args.headless else None
    delay_ms = max(1, int(round(1000.0 / args.refresh_hz)))
    if not args.headless:
        cv2.namedWindow(TELEMETRY_WINDOW, cv2.WINDOW_NORMAL)
        cv2.namedWindow(MAP_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(TELEMETRY_WINDOW, STREAMING_WIDTH, STREAMING_HEIGHT)
        cv2.resizeWindow(MAP_WINDOW, renderer.map_width, renderer.map_height)
        print(
            "[KEYBOARD] Hold W/S to move and A/D simultaneously to steer; "
            "A or D alone rotates in place. SPACE stops; ESC closes. "
            f"held-key polling={'X11' if held_keyboard and held_keyboard.available else 'fallback'}.",
            flush=True,
        )
    try:
        while rclpy.ok():
            # Process one blocking callback and then drain the ready queue.
            # Camera QoS depth=1 means this always converges to the newest
            # frame instead of replaying an accumulated video backlog.
            rclpy.spin_once(node, timeout_sec=min(0.01, 1.0 / args.refresh_hz))
            for _ in range(31):
                rclpy.spin_once(node, timeout_sec=0.0)
            snapshot = node.snapshot()
            qa_engine.update(snapshot)
            qa = qa_engine.active()
            if int(qa["index"]) != last_question_index:
                print(
                    f"\n[VL-JEPA QUERY {qa['index'] + 1}/{qa['count']}] {qa['question']}\n"
                    f"[STABILIZED ANSWER] {qa['stable']}",
                    flush=True,
                )
                last_question_index = int(qa["index"])
            if args.headless:
                if snapshot["sequence"] != last_sequence:
                    print(
                        "[DASHBOARD] "
                        + json.dumps(serializable_snapshot(snapshot, qa), ensure_ascii=False),
                        flush=True,
                    )
                    last_sequence = int(snapshot["sequence"])
                time.sleep(1.0 / args.refresh_hz)
            else:
                cv2.imshow(TELEMETRY_WINDOW, renderer.render_streaming(snapshot, qa))
                now = time.monotonic()
                if (
                    cached_map is None
                    or now - last_map_render >= 1.0 / args.map_refresh_hz
                ):
                    cached_map = renderer.render_map(snapshot)
                    last_map_render = now
                cv2.imshow(MAP_WINDOW, cached_map)
                key = cv2.waitKeyEx(delay_ms)
                if held_keyboard and held_keyboard.available:
                    if not node.update_held_keyboard(held_keyboard.pressed()):
                        break
                else:
                    if key >= 0 and not node.handle_keyboard_key(key):
                        break
                    node.refresh_keyboard_command()
                if (
                    cv2.getWindowProperty(TELEMETRY_WINDOW, cv2.WND_PROP_VISIBLE) < 1
                    or cv2.getWindowProperty(MAP_WINDOW, cv2.WND_PROP_VISIBLE) < 1
                ):
                    break
            if args.duration and time.monotonic() - started >= args.duration:
                break
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_keyboard()
        if held_keyboard is not None:
            held_keyboard.close()
        if not args.headless:
            cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
