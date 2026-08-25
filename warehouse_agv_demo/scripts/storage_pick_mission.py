#!/usr/bin/env python3
"""Navigate a fixed cabinet route, then choose a color and run physical pick."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import rclpy
import yaml
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from lifecycle_msgs.srv import GetState
from nav2_msgs.action import NavigateThroughPoses, NavigateToPose
from nav2_msgs.msg import SpeedLimit
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import String

from latent_route import (
    LatentRoutePlanner,
    LatentRouteSegment,
    combine_route_segments,
    hard_corner_points,
    orient_route_tangents,
    prune_hard_corner_checkpoints,
    round_hard_corners,
)


ROOT = Path(__file__).resolve().parents[1]
TASK_CONFIG = ROOT / "config" / "semantic_tasks.yaml"
ROUTE_CONFIG = ROOT / "config" / "storage_routes.yaml"
VQA_MISSION = ROOT / "scripts" / "vqa_mission.py"
NAV_TO_POSE_BT = ROOT / "config" / "navigate_to_pose_no_spin.xml"
NAV_THROUGH_POSES_BT = ROOT / "config" / "navigate_through_poses_no_spin.xml"

COLOR_ALIASES = {
    "blue": {"blue", "xanh", "xanh duong"},
    "red": {"red", "do"},
    "green": {"green", "xanh la"},
}

TTY = sys.stdout.isatty()
CYAN = "\033[36m" if TTY else ""
GREEN = "\033[32m" if TTY else ""
YELLOW = "\033[33m" if TTY else ""
BOLD = "\033[1m" if TTY else ""
RESET = "\033[0m" if TTY else ""


def print_route_card(
    storage: str,
    route: dict,
    available: dict,
    latent_segment: LatentRouteSegment,
) -> None:
    nav_source = os.environ.get(
        "WAREHOUSE_NAV_LOCALIZATION_SOURCE", "ground_truth"
    ).lower()
    localization = (
        "V-JEPA camera + short-term odom (experimental)"
        if nav_source == "vjepa"
        else "truth-reference control; V-JEPA shadow comparison"
    )
    print(f"{CYAN}{BOLD}╭─ NAV2 SEMANTIC ROUTE ───────────────────────────────────╮{RESET}")
    print(f"{CYAN}│{RESET} Destination : {BOLD}{route['label']} ({storage}){RESET}")
    print(f"{CYAN}│{RESET} RGB boxes    : {', '.join(available)}")
    print(
        f"{CYAN}│{RESET} Route        : latent clips "
        f"{latent_segment.start_index}→{latent_segment.end_index} "
        f"({len(prune_hard_corner_checkpoints(latent_segment.poses[1:]))} checkpoints)"
    )
    print(f"{CYAN}│{RESET} Latent map   : {latent_segment.map_dir}")
    print(f"{CYAN}│{RESET} Planner      : NavFn A* + rounded-corner MPPI")
    print(f"{CYAN}│{RESET} Localization : {localization}")
    print(f"{CYAN}│{RESET} Latent use   : recorded outbound corridor + V-JEPA dashboard")
    print(
        f"{CYAN}│{RESET} Safety       : predicted WAIT/PASS/REPLAN + local LiDAR"
    )
    print(f"{CYAN}{BOLD}╰─────────────────────────────────────────────────────────╯{RESET}")


def normalize(value: str) -> str:
    text = unicodedata.normalize("NFD", value.strip().lower()).replace("đ", "d")
    return " ".join(
        "".join(char for char in text if unicodedata.category(char) != "Mn").split()
    )


def normalize_storage(value: str) -> str:
    cleaned = normalize(value).removeprefix("storage ").removeprefix("tu ")
    storage = f"storage_{cleaned.upper()}"
    if storage not in {"storage_A", "storage_B", "storage_C"}:
        raise ValueError("chỉ hỗ trợ tủ A, B hoặc C")
    return storage


def normalize_color(value: str) -> str:
    cleaned = normalize(value)
    for color, aliases in COLOR_ALIASES.items():
        if cleaned in aliases:
            return color
    raise ValueError("màu hợp lệ: blue/xanh dương, red/đỏ, green/xanh lá")


def prune_passed_checkpoints(
    poses: list[PoseStamped], current_xy: tuple[float, float] | None
) -> list[PoseStamped]:
    """Never send a checkpoint behind the robot again after an action retry."""
    pending = list(poses)
    if current_xy is None:
        return pending
    current_x, current_y = current_xy
    while len(pending) > 1:
        first = pending[0].pose.position
        following = pending[1].pose.position
        dx = float(following.x - first.x)
        dy = float(following.y - first.y)
        length_sq = dx * dx + dy * dy
        if length_sq <= 1e-6:
            pending.pop(0)
            continue
        offset_x = current_x - float(first.x)
        offset_y = current_y - float(first.y)
        projection = (offset_x * dx + offset_y * dy) / length_sq
        closest_projection = min(1.0, max(0.0, projection))
        closest_x = float(first.x) + closest_projection * dx
        closest_y = float(first.y) + closest_projection * dy
        cross_track = math.hypot(current_x - closest_x, current_y - closest_y)
        distance_to_first = math.hypot(offset_x, offset_y)
        reached = distance_to_first <= 0.45
        passed = projection >= 0.15 and cross_track <= 1.0
        if not (reached or passed):
            break
        pending.pop(0)
    return pending


def available_objects(tasks: dict, storage: str) -> dict[str, tuple[str, dict]]:
    return {
        str(item["color"]): (name, item)
        for name, item in tasks["objects"].items()
        if item["location"] == storage
    }


class CabinetRouteNavigator(Node):
    def __init__(self) -> None:
        super().__init__("cabinet_route_navigator")
        self.client = ActionClient(self, NavigateToPose, "/navigate_to_pose")
        self.corridor_client = ActionClient(
            self, NavigateThroughPoses, "/navigate_through_poses"
        )
        self.lifecycle = self.create_client(GetState, "/bt_navigator/get_state")
        self.preview = self.create_publisher(PoseStamped, "/semantic_goal", 1)
        self.mission_state = self.create_publisher(
            String, "/warehouse/mission_state", 10
        )
        self.speed_limit = self.create_publisher(SpeedLimit, "/speed_limit", 10)
        self.create_subscription(
            String,
            "/warehouse/behavior_decision",
            self._on_behavior_decision,
            10,
        )
        self.last_feedback = 0.0
        self.goal_initial_distance = 0.0
        self.last_distance_remaining = math.inf
        self.poses_remaining: int | None = None
        self.last_current_xy: tuple[float, float] | None = None
        self.progress_visible = False
        self.active_goal = None
        self.replan_requested = False
        route_config = yaml.safe_load(ROUTE_CONFIG.read_text(encoding="utf-8"))
        self.tracking = route_config.get("trajectory_tracking", {})
        self.active_corner_points = np.empty((0, 2), dtype=np.float64)
        self.speed_limited = False
        self.shelf_transition_xy: tuple[float, float] | None = None
        self.shelf_state_details: dict[str, object] = {}
        self.shelf_state_published = False
        self.current_state_message: String | None = None
        self.last_state_publish = -math.inf

    def publish_state(self, state: str, **details: object) -> None:
        payload = {
            "state": state,
            "timestamp": time.time(),
            **details,
        }
        self.current_state_message = String(
            data=json.dumps(payload, sort_keys=True)
        )
        self.mission_state.publish(self.current_state_message)
        self.last_state_publish = time.monotonic()
        self.get_logger().info(f"MISSION_STATE {state}")

    def _on_behavior_decision(self, message: String) -> None:
        """Turn a bounded REPLAN decision into one fresh Nav2 action goal."""
        try:
            report = json.loads(message.data)
        except json.JSONDecodeError:
            return
        if (
            report.get("decision") != "REPLAN"
            or self.active_goal is None
            or self.replan_requested
        ):
            return
        self.replan_requested = True
        self.get_logger().warning(
            "Predictive planner requested REPLAN: " + str(report.get("reason", ""))
        )
        # Cancellation is asynchronous and safe from the executor callback.
        # The normal retry path then resends only the remaining corridor,
        # forcing NavFn to compute a fresh path once.
        self.active_goal.cancel_goal_async()

    def _publish_speed_limit(self, limited: bool) -> None:
        if limited == self.speed_limited:
            return
        message = SpeedLimit()
        message.percentage = False
        message.speed_limit = (
            float(self.tracking.get("corner_speed_limit_mps", 0.40))
            if limited
            else 0.0
        )
        self.speed_limit.publish(message)
        self.speed_limited = limited

    def _update_route_phase(self, current_xy: tuple[float, float]) -> None:
        if len(self.active_corner_points):
            distance = float(
                np.min(
                    np.linalg.norm(
                        self.active_corner_points - np.asarray(current_xy), axis=1
                    )
                )
            )
            self._publish_speed_limit(
                distance <= float(
                    self.tracking.get("corner_slowdown_radius_m", 2.50)
                )
            )
        if self.shelf_transition_xy is not None and not self.shelf_state_published:
            distance = math.hypot(
                current_xy[0] - self.shelf_transition_xy[0],
                current_xy[1] - self.shelf_transition_xy[1],
            )
            if distance <= 1.0:
                self.publish_state("SHELF_APPROACH", **self.shelf_state_details)
                self.shelf_state_published = True

    def wait_until_active(self, timeout: float) -> None:
        """Wait for Nav2 lifecycle activation, not just action discovery."""
        deadline = time.monotonic() + timeout
        last_report = 0.0
        if not self.lifecycle.wait_for_service(timeout_sec=timeout):
            raise RuntimeError("không tìm thấy lifecycle service của bt_navigator")
        while time.monotonic() < deadline:
            future = self.lifecycle.call_async(GetState.Request())
            rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)
            response = future.result() if future.done() else None
            if response is not None and response.current_state.label == "active":
                return
            now = time.monotonic()
            if now - last_report >= 1.0:
                print("[ROUTE_WAIT] đang chờ Nav2 active và TF odom sẵn sàng")
                last_report = now
            time.sleep(0.25)
        raise RuntimeError("Nav2 chưa active sau thời gian chờ")

    @staticmethod
    def make_pose(values: list[float]) -> PoseStamped:
        x, y, yaw = map(float, values)
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation.z = math.sin(yaw / 2.0)
        pose.pose.orientation.w = math.cos(yaw / 2.0)
        return pose

    def feedback(self, message) -> None:
        now = time.monotonic()
        if (
            self.current_state_message is not None
            and now - self.last_state_publish >= 1.0
        ):
            self.mission_state.publish(self.current_state_message)
            self.last_state_publish = now
        remaining = max(0.0, float(message.feedback.distance_remaining))
        current = message.feedback.current_pose.pose.position
        self.last_current_xy = (float(current.x), float(current.y))
        self._update_route_phase(self.last_current_xy)
        if remaining > 0.05 or self.goal_initial_distance > 0.05:
            self.last_distance_remaining = remaining
        if hasattr(message.feedback, "number_of_poses_remaining"):
            count = int(message.feedback.number_of_poses_remaining)
            if count > 0:
                self.poses_remaining = count
        interval = 0.35 if TTY else 2.0
        if now - self.last_feedback >= interval:
            if self.goal_initial_distance <= 0.05 and remaining <= 0.05:
                # Nav2 may emit one zero-valued feedback before its first path.
                return
            self.goal_initial_distance = max(self.goal_initial_distance, remaining)
            progress = (
                min(1.0, 1.0 - remaining / self.goal_initial_distance)
                if self.goal_initial_distance > 0.05
                else 1.0
            )
            filled = int(round(progress * 22))
            bar = "█" * filled + "·" * (22 - filled)
            line = (
                f"  {GREEN}NAV2{RESET} [{bar}] {progress * 100:5.1f}% "
                f"│ còn {remaining:5.2f} m"
            )
            if TTY:
                print(f"\r{line:<92}", end="", flush=True)
                self.progress_visible = True
            else:
                print(line)
            self.last_feedback = now

    def finish_progress(self) -> None:
        if self.progress_visible:
            print()
            self.progress_visible = False

    def navigate(self, name: str, values: list[float], wait: float, retries: int) -> None:
        self.wait_until_active(wait)
        if not self.client.wait_for_server(timeout_sec=wait):
            raise RuntimeError("không tìm thấy Nav2 /navigate_to_pose")
        for attempt in range(1, retries + 2):
            self.replan_requested = False
            pose = self.make_pose(values)
            self.preview.publish(pose)
            self.goal_initial_distance = 0.0
            self.last_distance_remaining = math.inf
            self.last_feedback = 0.0
            print(
                f"\n{CYAN}▶ WAYPOINT{RESET} {BOLD}{name}{RESET} "
                f"({pose.pose.position.x:.2f}, {pose.pose.position.y:.2f}) "
                f"│ attempt {attempt}/{retries + 1}"
            )
            goal = NavigateToPose.Goal()
            goal.pose = pose
            goal.behavior_tree = str(NAV_TO_POSE_BT)
            sent = self.client.send_goal_async(goal, feedback_callback=self.feedback)
            rclpy.spin_until_future_complete(self, sent, timeout_sec=10.0)
            handle = sent.result() if sent.done() else None
            if handle is not None and handle.accepted:
                self.active_goal = handle
                result = handle.get_result_async()
                rclpy.spin_until_future_complete(self, result)
                if (
                    result.result() is not None
                    and result.result().status == GoalStatus.STATUS_SUCCEEDED
                ):
                    self.finish_progress()
                    self.active_goal = None
                    print(f"  {GREEN}✓ ARRIVED{RESET} {name}")
                    return
                self.active_goal = None
            self.finish_progress()
            if attempt <= retries:
                print(f"  {YELLOW}↻ RETRY{RESET} chờ TF ổn định rồi thử lại {name}")
                deadline = time.monotonic() + 2.0
                while time.monotonic() < deadline:
                    rclpy.spin_once(self, timeout_sec=0.1)
        raise RuntimeError(f"Nav2 không thể tới waypoint {name}")

    def navigate_latent_corridor(
        self,
        name: str,
        segment: LatentRouteSegment,
        wait: float,
        retries: int,
        *,
        shelf_transition_xy: tuple[float, float] | None = None,
        shelf_state_details: dict[str, object] | None = None,
        final_yaw_from_path: bool = False,
    ) -> None:
        """Follow stable spatial checkpoints sampled from the mapping traversal."""
        self.wait_until_active(wait)
        if not self.corridor_client.wait_for_server(timeout_sec=wait):
            raise RuntimeError("không tìm thấy Nav2 /navigate_through_poses")
        # The first saved pose is the current segment anchor. Avoid asking
        # Nav2 to stop immediately at the pose where the robot already sits.
        values = segment.poses[1:] if len(segment.poses) > 1 else segment.poses
        threshold = float(self.tracking.get("hard_turn_threshold_rad", 0.75))
        self.active_corner_points = hard_corner_points(
            values, angle_threshold_rad=threshold
        )
        values = round_hard_corners(
            values,
            angle_threshold_rad=threshold,
            radius_m=float(self.tracking.get("corner_rounding_radius_m", 0.90)),
            curve_samples=int(self.tracking.get("corner_curve_samples", 3)),
        )
        values = orient_route_tangents(values)
        if final_yaw_from_path and len(values) >= 2:
            final_delta = values[-1, :2] - values[-2, :2]
            if float(np.linalg.norm(final_delta)) > 0.05:
                values[-1, 3] = math.atan2(
                    float(final_delta[1]), float(final_delta[0])
                )
        self.shelf_transition_xy = shelf_transition_xy
        self.shelf_state_details = shelf_state_details or {}
        self.shelf_state_published = False
        poses = [self.make_pose(list(pose[[0, 1, 3]])) for pose in values]
        pending_poses = poses
        for attempt in range(1, retries + 2):
            self.replan_requested = False
            self.goal_initial_distance = 0.0
            self.last_distance_remaining = math.inf
            self.poses_remaining = len(pending_poses)
            self.last_current_xy = None
            self.last_feedback = 0.0
            print(
                f"\n{CYAN}▶ LATENT CORRIDOR{RESET} {BOLD}{name}{RESET} "
                f"│ clips {segment.start_index}→{segment.end_index} "
                f"│ {len(pending_poses)} checkpoints │ attempt {attempt}/{retries + 1}"
            )
            for pose in pending_poses:
                self.preview.publish(pose)
            goal = NavigateThroughPoses.Goal()
            goal.poses = pending_poses
            goal.behavior_tree = str(NAV_THROUGH_POSES_BT)
            sent = self.corridor_client.send_goal_async(
                goal, feedback_callback=self.feedback
            )
            rclpy.spin_until_future_complete(self, sent, timeout_sec=10.0)
            handle = sent.result() if sent.done() else None
            if handle is not None and handle.accepted:
                self.active_goal = handle
                result = handle.get_result_async()
                rclpy.spin_until_future_complete(self, result)
                outcome = result.result() if result.done() else None
                if outcome is not None and outcome.status == GoalStatus.STATUS_SUCCEEDED:
                    self.finish_progress()
                    self.active_goal = None
                    self._publish_speed_limit(False)
                    print(
                        f"  {GREEN}✓ LATENT CORRIDOR COMPLETE{RESET} {name} "
                        f"(target error {segment.target_error_m:.2f} m)"
                    )
                    return
                self.active_goal = None
                # MPPI may settle a few centimetres outside Nav2's strict goal
                # checker while its command is below the velocity deadband.
                # The visual servo owns final shelf accuracy, so never resend
                # the entire already-traversed corridor from this condition.
                if (
                    self.goal_initial_distance > 0.50
                    and self.last_distance_remaining <= 0.40
                ):
                    self.finish_progress()
                    self._publish_speed_limit(False)
                    print(
                        f"  {GREEN}✓ LATENT CORRIDOR COMPLETE{RESET} {name} "
                        f"(controller remainder {self.last_distance_remaining:.2f} m)"
                    )
                    return
            self.finish_progress()
            if attempt <= retries:
                remaining_count = max(
                    1,
                    min(
                        len(pending_poses),
                        self.poses_remaining or len(pending_poses),
                    ),
                )
                pending_poses = pending_poses[-remaining_count:]
                pending_poses = prune_passed_checkpoints(
                    pending_poses, self.last_current_xy
                )
                print(f"  {YELLOW}↻ RETRY{RESET} quay lại corridor latent {name}")
                deadline = time.monotonic() + 2.0
                while time.monotonic() < deadline:
                    rclpy.spin_once(self, timeout_sec=0.1)
        self._publish_speed_limit(False)
        raise RuntimeError(f"Nav2 không thể đi hết corridor latent {name}")

    def cancel_active_goal(self) -> None:
        self._publish_speed_limit(False)
        if self.active_goal is None or not rclpy.ok():
            return
        future = self.active_goal.cancel_goal_async()
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        self.active_goal = None


def select_color(
    tasks: dict, storage: str, requested: str | None, *, interactive: bool
) -> tuple[str, str, dict]:
    choices = available_objects(tasks, storage)
    print(f"{CYAN}◆ SELECT{RESET} {storage} có các màu: {BOLD}{', '.join(choices)}{RESET}")
    while True:
        if requested is None:
            if not interactive:
                raise RuntimeError("cần truyền --color khi không chạy terminal tương tác")
            try:
                requested = input("Chọn màu box cần pick: ")
            except EOFError as error:
                raise RuntimeError("terminal đã đóng trước khi chọn màu") from error
        try:
            color = normalize_color(requested)
            if color not in choices:
                raise ValueError(
                    f"tủ {storage[-1]} không có màu {color}; có: {', '.join(choices)}"
                )
            break
        except ValueError as error:
            if not interactive:
                raise
            print(f"{YELLOW}↻ SELECT RETRY{RESET} {error}")
            requested = None
    object_name, item = choices[color]
    print(
        f"{GREEN}✓ TARGET{RESET} {BOLD}{color.upper()}{RESET} → "
        f"slot {item['slot']} → model {item['model']}"
    )
    return color, object_name, item


def payload_route_segments(
    tasks: dict,
    storage: str,
    item: dict,
    latent_planner: LatentRoutePlanner,
    staging_segment: LatentRouteSegment,
) -> tuple[str, LatentRouteSegment, list[float]]:
    """Resolve the saved pickup leg and direct A* delivery goal."""
    slot = str(item["slot"])
    pickup_pose = tasks["stations"][storage]["slots"][slot].get("vjepa_approach")
    if pickup_pose is None:
        raise ValueError(f"slot {slot} chưa có vjepa_approach trong semantic_tasks.yaml")
    pickup_segment = latent_planner.segment_to(
        list(pickup_pose), start_index=staging_segment.end_index
    )
    packing_pose = list(
        tasks["stations"]["packing_station"]["slots"]["PACK01"]["approach"]
    )
    return slot, pickup_segment, packing_pose


def run_loaded_return(
    navigator: CabinetRouteNavigator,
    *,
    slot: str,
    packing_pose: list[float],
    wait: float,
    retries: int,
) -> None:
    print(
        f"\n{CYAN}{BOLD}↩ DIRECT A* DELIVERY{RESET} "
        f"slot {slot} → Packing Station"
    )
    navigator.navigate(
        f"{slot} → Packing Station", packing_pose, wait, retries
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage", required=True, help="A, B hoặc C")
    parser.add_argument("--color", default=None, help="để trống để chọn sau khi tới tủ")
    delivery = parser.add_mutually_exclusive_group()
    delivery.add_argument(
        "--deliver",
        dest="deliver",
        action="store_true",
        help="giao tới Packing Station (mặc định)",
    )
    delivery.add_argument(
        "--pick-only",
        dest="deliver",
        action="store_false",
        help="dừng tại kệ với box còn gắn trên khay",
    )
    parser.set_defaults(deliver=True)
    parser.add_argument(
        "--resume-delivery",
        action="store_true",
        help="box đã gắn trên khay: bỏ qua outbound/pick và chạy nhánh quay về",
    )
    parser.add_argument("--route-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--wait", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()
    if args.wait <= 0.0 or args.retries < 0:
        parser.error("wait phải dương và retries không được âm")
    if args.resume_delivery and args.route_only:
        parser.error("--resume-delivery không dùng cùng --route-only")
    if args.resume_delivery and args.color is None:
        parser.error("--resume-delivery cần --color để xác định payload đang gắn")

    tasks = yaml.safe_load(TASK_CONFIG.read_text(encoding="utf-8"))
    routes = yaml.safe_load(ROUTE_CONFIG.read_text(encoding="utf-8"))["routes"]
    try:
        storage = normalize_storage(args.storage)
    except ValueError as error:
        parser.error(str(error))
    route = routes[storage]
    available = available_objects(tasks, storage)
    try:
        latent_planner = LatentRoutePlanner()
        staging_pose = list(route["waypoints"][-1]["pose"])
        staging_segment = latent_planner.segment_to(staging_pose)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    outbound_segment = staging_segment
    selected_target: tuple[
        str, dict, str, LatentRouteSegment, list[float]
    ] | None = None
    # The one-command mission already knows its requested color. Resolve it
    # before motion and keep the staging + Shelf A approach in one action so
    # MPPI sees (and rounds) the shared second 90-degree turn. Interactive
    # missions retain the original stop-and-select behavior.
    if args.color is not None and not args.route_only and not args.resume_delivery:
        try:
            color, _, item = select_color(
                tasks, storage, args.color, interactive=False
            )
            slot, pickup_segment, packing_pose = payload_route_segments(
                tasks, storage, item, latent_planner, staging_segment
            )
            outbound_segment = combine_route_segments(
                staging_segment, pickup_segment
            )
            selected_target = (
                color, item, slot, pickup_segment, packing_pose
            )
        except (RuntimeError, ValueError) as error:
            parser.error(str(error))
    print_route_card(storage, route, available, outbound_segment)
    if args.resume_delivery:
        mode = "RESUME LOADED RETURN + DROP AT PACKING STATION"
    elif args.deliver:
        mode = "PICK + RETURN + DROP AT PACKING STATION"
    else:
        mode = "PICK ONLY (payload stays attached)"
    print(f"{GREEN}◆ MISSION{RESET} {mode}")
    if args.dry_run and not args.resume_delivery:
        if args.color is not None and selected_target is None:
            try:
                select_color(tasks, storage, args.color, interactive=False)
            except (RuntimeError, ValueError) as error:
                parser.error(str(error))
        return

    if args.resume_delivery:
        try:
            color, _, item = select_color(
                tasks, storage, args.color, interactive=False
            )
            slot, _, packing_pose = payload_route_segments(
                tasks, storage, item, latent_planner, staging_segment
            )
        except (RuntimeError, ValueError) as error:
            parser.error(str(error))
        print(
            f"{GREEN}✓ RESUME ROUTE{RESET} payload {color.upper()} "
            "trên khay → direct NavFn A* → Packing Station"
        )
        if args.dry_run:
            return
        storage_label = storage[-1]
        command = (
            f"Bring the {color} box from Storage {storage_label} "
            "to Packing Station"
        )
        subprocess.run(
            [
                sys.executable,
                "-u",
                str(VQA_MISSION),
                "--command",
                command,
                "--wait",
                str(args.wait),
                "--prepare-return-only",
            ],
            cwd=ROOT,
            check=True,
        )
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        navigator = CabinetRouteNavigator()
        try:
            navigator.publish_state(
                "RETURN_TO_DROPOFF", slot=slot, destination="packing_station"
            )
            run_loaded_return(
                navigator,
                slot=slot,
                packing_pose=packing_pose,
                wait=args.wait,
                retries=args.retries,
            )
        except KeyboardInterrupt:
            navigator.finish_progress()
            navigator.cancel_active_goal()
            raise SystemExit(130)
        finally:
            navigator.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        subprocess.run(
            [
                sys.executable,
                "-u",
                str(VQA_MISSION),
                "--command",
                command,
                "--wait",
                str(args.wait),
                "--release-only",
            ],
            cwd=ROOT,
            check=True,
        )
        print(
            f"\n{GREEN}{BOLD}✓ DELIVERY COMPLETE{RESET} "
            f"{color.upper()} box: Storage {storage_label} → Packing Station"
        )
        return

    # Keep Python's KeyboardInterrupt handler so Ctrl+C can cancel the active
    # Nav2 goal before shutting the ROS context down.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    navigator = CabinetRouteNavigator()
    interrupted = False
    try:
        navigator.publish_state(
            "NAVIGATE_TO_SHELF", storage=storage, destination=route["label"]
        )
        navigator.navigate_latent_corridor(
            (
                f"dock → {selected_target[2]}"
                if selected_target is not None
                else f"dock → {route['label']}"
            ),
            outbound_segment,
            args.wait,
            args.retries,
            shelf_transition_xy=(float(staging_pose[0]), float(staging_pose[1]))
            if selected_target is not None
            else None,
            shelf_state_details={
                "storage": storage,
                "slot": selected_target[2],
                "color": selected_target[0],
            }
            if selected_target is not None
            else None,
            final_yaw_from_path=selected_target is not None,
        )
    except KeyboardInterrupt:
        interrupted = True
        navigator.finish_progress()
        print(f"\n{YELLOW}■ CANCELLED{RESET} đã hủy goal Nav2 hiện tại")
        navigator.cancel_active_goal()
    finally:
        navigator.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    if interrupted:
        raise SystemExit(130)

    if selected_target is not None:
        print(
            f"\n{GREEN}{BOLD}✓ AGV đã tới {selected_target[2]} "
            f"qua một corridor liên tục{RESET}"
        )
    else:
        print(f"\n{GREEN}{BOLD}✓ AGV đã ở staging của {route['label']}{RESET}")
    if args.route_only:
        print(f"[SELECT] màu có sẵn: {', '.join(available)}")
        return
    if selected_target is not None:
        color, item, slot, pickup_segment, packing_pose = selected_target
    else:
        try:
            color, _, item = select_color(
                tasks, storage, args.color, interactive=sys.stdin.isatty()
            )
        except (RuntimeError, ValueError) as error:
            parser.error(str(error))
        try:
            slot, pickup_segment, packing_pose = payload_route_segments(
                tasks, storage, item, latent_planner, staging_segment
            )
        except (ValueError, RuntimeError) as error:
            parser.error(str(error))

    # Color is selected only after staging, then the final approach continues
    # along the exact same forward latent sequence used during mapping.
    if selected_target is None:
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        navigator = CabinetRouteNavigator()
        try:
            navigator.publish_state(
                "SHELF_APPROACH", storage=storage, slot=slot, color=color
            )
            navigator.navigate_latent_corridor(
                f"{route['label']} → {slot}",
                pickup_segment,
                args.wait,
                args.retries,
                final_yaw_from_path=True,
            )
        except KeyboardInterrupt:
            navigator.finish_progress()
            navigator.cancel_active_goal()
            raise SystemExit(130)
        finally:
            navigator.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()

    storage_label = storage[-1]
    command = f"Bring the {color} box from Storage {storage_label} to Packing Station"
    evidence = ROOT / "screenshots" / f"pick_{storage_label}_{color}.png"
    mission_command = [
        sys.executable,
        "-u",
        str(VQA_MISSION),
        "--command",
        command,
        "--wait",
        str(args.wait),
        "--camera-evidence",
        str(evidence),
        "--skip-navigation",
        # Route ownership returns to this process after the box is secured so
        # the loaded AGV can take a direct A* delivery goal.
        "--pick-only",
    ]
    subprocess.run(mission_command, cwd=ROOT, check=True)

    if not args.deliver:
        return

    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    navigator = CabinetRouteNavigator()
    try:
        navigator.publish_state(
            "RETURN_TO_DROPOFF", slot=slot, destination="packing_station"
        )
        run_loaded_return(
            navigator,
            slot=slot,
            packing_pose=packing_pose,
            wait=args.wait,
            retries=args.retries,
        )
    except KeyboardInterrupt:
        navigator.finish_progress()
        navigator.cancel_active_goal()
        raise SystemExit(130)
    finally:
        navigator.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    release_command = [
        sys.executable,
        "-u",
        str(VQA_MISSION),
        "--command",
        command,
        "--wait",
        str(args.wait),
        "--release-only",
    ]
    subprocess.run(release_command, cwd=ROOT, check=True)
    print(
        f"\n{GREEN}{BOLD}✓ DELIVERY COMPLETE{RESET} "
        f"{color.upper()} box: Storage {storage_label} → Packing Station"
    )


if __name__ == "__main__":
    main()
