#!/usr/bin/env python3
"""Drive a Nav2 warehouse patrol and narrate localization/avoidance events."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.transport13 import Node as GazeboNode
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

from src.evaluation.metrics import summarize_localization
from src.evaluation.warehouse_context import (
    EntityPose,
    ObstacleReport,
    WarehouseRegionIndex,
    identify_forward_obstacle,
    scan_sector_min,
    wrap_angle,
)
from src.utils.config import load_config


@dataclass(frozen=True)
class GroundTruthSample:
    timestamp: float
    x: float
    y: float
    z: float
    yaw: float

    def pose_array(self) -> np.ndarray:
        return np.asarray([self.x, self.y, self.z, self.yaw], dtype=np.float64)


def _yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _message_timestamp(message: Any) -> float:
    try:
        return float(message.header.stamp.sec) + float(message.header.stamp.nsec) * 1e-9
    except (AttributeError, TypeError):
        return 0.0


def _ros_timestamp(message: PoseWithCovarianceStamped) -> float:
    return float(message.header.stamp.sec) + float(message.header.stamp.nanosec) * 1e-9


class AutonomousWarehousePatrol(Node):
    """Nav2 waypoint driver with independent Gazebo/V-JEPA evaluation."""

    def __init__(
        self,
        *,
        config: dict[str, Any],
        phase: str,
        output_dir: Path,
    ) -> None:
        super().__init__(f"vjepa_{phase}_patrol")
        experiment = config["warehouse_experiment"]
        evaluation = config.get("evaluation", {})
        inventory_path = Path(str(experiment["inventory"]))
        if not inventory_path.is_absolute():
            inventory_path = (PROJECT_ROOT / inventory_path).resolve()
        self.regions = WarehouseRegionIndex.from_inventory(inventory_path)
        self.phase = phase
        self.experiment = experiment
        self.comparison_min_translation_m = float(
            evaluation.get("min_translation_m", 0.0)
        )
        self.comparison_motion_window_sec = float(
            evaluation.get("live_motion_window_sec", 1.0)
        )
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.event_stream = (self.output_dir / f"{phase}_events.jsonl").open(
            "w", encoding="utf-8"
        )
        self.comparison_stream = (self.output_dir / f"{phase}_comparison.csv").open(
            "w", newline="", encoding="utf-8"
        )
        self.comparison_writer = csv.DictWriter(
            self.comparison_stream,
            fieldnames=[
                "timestamp",
                "gazebo_x",
                "gazebo_y",
                "gazebo_z",
                "gazebo_yaw",
                "vjepa_x",
                "vjepa_y",
                "vjepa_z",
                "vjepa_yaw",
                "relative_dx",
                "relative_dy",
                "position_error_m",
                "yaw_error_rad",
                "ground_truth_time_delta_sec",
                "ground_truth_translation_m",
                "top1_similarity",
                "confidence_margin",
                "area",
                "obstacle",
            ],
        )
        self.comparison_writer.writeheader()
        self.comparisons: list[dict[str, Any]] = []
        self.stationary_predictions_skipped = 0
        self.comparison_motion_active: bool | None = None
        self.goal_results: list[dict[str, Any]] = []
        self.lock = threading.Lock()
        self.pose_history: deque[GroundTruthSample] = deque(maxlen=5000)
        self.entities: tuple[EntityPose, ...] = ()
        self.latest_front_clearance = math.inf
        self.latest_command = Twist()
        self.latest_debug: dict[str, Any] = {}
        self.current_obstacle: ObstacleReport | None = None
        self.previous_obstacle_name: str | None = None
        self.previous_avoidance_key: tuple[str, str] | None = None
        self.last_avoidance_report_time = -math.inf
        self.active_waypoint = "chưa có"
        self.distance_remaining = math.nan
        self.closed = False

        self.nav_client = ActionClient(self, NavigateToPose, "/navigate_to_pose")
        self.create_subscription(LaserScan, "/scan", self._on_scan, qos_profile_sensor_data)
        self.create_subscription(Twist, "/cmd_vel", self._on_cmd_vel, 20)
        if phase == "query":
            self.create_subscription(
                PoseWithCovarianceStamped, "/vjepa_pose", self._on_vjepa_pose, 20
            )
            self.create_subscription(String, "/vjepa_localization/debug", self._on_debug, 20)
        self.gz_node = GazeboNode()
        if not self.gz_node.subscribe(Pose_V, "/world/world_demo/pose/info", self._on_world):
            raise RuntimeError("could not subscribe to Gazebo world poses")
        interval = float(experiment.get("status_interval_sec", 1.0))
        self.create_timer(interval, self._report_status)
        self._event(
            "START",
            f"bắt đầu lượt {phase}; Nav2 tự lái, LiDAR giám sát vật cản",
        )

    def _event(self, kind: str, message: str, **details: Any) -> None:
        record = {
            "wall_time": time.time(),
            "phase": self.phase,
            "event": kind,
            "message": message,
            **details,
        }
        self.event_stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        self.event_stream.flush()
        self.get_logger().info(f"[{kind}] {message}")

    def _on_world(self, message: Pose_V) -> None:
        timestamp = _message_timestamp(message)
        if timestamp <= 0.0:
            timestamp = self.get_clock().now().nanoseconds * 1e-9
        gt: GroundTruthSample | None = None
        entities: list[EntityPose] = []
        for pose in message.pose:
            if pose.name == "warehouse_agv":
                q = pose.orientation
                gt = GroundTruthSample(
                    timestamp,
                    float(pose.position.x),
                    float(pose.position.y),
                    float(pose.position.z),
                    _yaw_from_quaternion(q.x, q.y, q.z, q.w),
                )
            elif pose.name.startswith(("road_box_static_", "random_worker_")):
                entities.append(
                    EntityPose(pose.name, float(pose.position.x), float(pose.position.y))
                )
        with self.lock:
            if gt is not None:
                self.pose_history.append(gt)
            self.entities = tuple(entities)

    def _on_scan(self, message: LaserScan) -> None:
        clearance = scan_sector_min(
            message.ranges,
            angle_min=float(message.angle_min),
            angle_increment=float(message.angle_increment),
            range_min=float(message.range_min),
            range_max=float(message.range_max),
            half_width_rad=float(
                self.experiment.get("obstacle_front_half_angle_rad", 0.65)
            ),
        )
        with self.lock:
            self.latest_front_clearance = clearance

    def _on_cmd_vel(self, message: Twist) -> None:
        with self.lock:
            self.latest_command = message

    def _on_debug(self, message: String) -> None:
        try:
            parsed = json.loads(message.data)
        except json.JSONDecodeError:
            return
        with self.lock:
            self.latest_debug = parsed

    def _nearest_ground_truth(self, timestamp: float) -> GroundTruthSample | None:
        with self.lock:
            history = tuple(self.pose_history)
        if not history:
            return None
        return min(history, key=lambda item: abs(item.timestamp - timestamp))

    def _ground_truth_translation(self, timestamp: float) -> float | None:
        """Measure actual Gazebo translation around one V-JEPA clip center."""
        with self.lock:
            history = tuple(self.pose_history)
        if not history:
            return None
        half_window = self.comparison_motion_window_sec / 2.0
        start_time = timestamp - half_window
        end_time = timestamp + half_window
        if history[0].timestamp > start_time or history[-1].timestamp < end_time:
            return None
        start = min(history, key=lambda item: abs(item.timestamp - start_time))
        end = min(history, key=lambda item: abs(item.timestamp - end_time))
        return math.hypot(end.x - start.x, end.y - start.y)

    def _on_vjepa_pose(self, message: PoseWithCovarianceStamped) -> None:
        timestamp = _ros_timestamp(message)
        ground_truth = self._nearest_ground_truth(timestamp)
        if ground_truth is None:
            self._event("VJEPA_WAIT", "đã có pose V-JEPA nhưng chưa có ground-truth Gazebo")
            return
        translation = self._ground_truth_translation(timestamp)
        if translation is None:
            return
        if translation < self.comparison_min_translation_m:
            self.stationary_predictions_skipped += 1
            if self.comparison_motion_active is not False:
                self._event(
                    "LOCALIZE_SKIP",
                    "AGV đang đứng yên; không đưa mẫu này vào so sánh V-JEPA/Gazebo",
                    ground_truth_translation_m=translation,
                )
            self.comparison_motion_active = False
            return
        if self.comparison_motion_active is False:
            self._event(
                "LOCALIZE_RESUME",
                "AGV đã di chuyển; tiếp tục so sánh V-JEPA với truth Gazebo",
                ground_truth_translation_m=translation,
            )
        self.comparison_motion_active = True
        q = message.pose.pose.orientation
        predicted = np.asarray(
            [
                message.pose.pose.position.x,
                message.pose.pose.position.y,
                message.pose.pose.position.z,
                _yaw_from_quaternion(q.x, q.y, q.z, q.w),
            ],
            dtype=np.float64,
        )
        truth = ground_truth.pose_array()
        dx, dy = float(predicted[0] - truth[0]), float(predicted[1] - truth[1])
        position_error = math.hypot(dx, dy)
        yaw_error = abs(wrap_angle(float(predicted[3] - truth[3])))
        area = self.regions.describe(ground_truth.x, ground_truth.y)
        with self.lock:
            debug = dict(self.latest_debug)
            obstacle = self.current_obstacle
        row = {
            "timestamp": timestamp,
            "gazebo_x": ground_truth.x,
            "gazebo_y": ground_truth.y,
            "gazebo_z": ground_truth.z,
            "gazebo_yaw": ground_truth.yaw,
            "vjepa_x": float(predicted[0]),
            "vjepa_y": float(predicted[1]),
            "vjepa_z": float(predicted[2]),
            "vjepa_yaw": float(predicted[3]),
            "relative_dx": dx,
            "relative_dy": dy,
            "position_error_m": position_error,
            "yaw_error_rad": yaw_error,
            "ground_truth_time_delta_sec": abs(ground_truth.timestamp - timestamp),
            "ground_truth_translation_m": translation,
            "top1_similarity": debug.get("top1_similarity"),
            "confidence_margin": debug.get("confidence_margin"),
            "area": area,
            "obstacle": obstacle.label if obstacle else "không",
        }
        self.comparison_writer.writerow(row)
        self.comparison_stream.flush()
        self.comparisons.append(row)
        self._event(
            "LOCALIZE",
            f"{area} | Gazebo=({truth[0]:.2f},{truth[1]:.2f}) "
            f"V-JEPA=({predicted[0]:.2f},{predicted[1]:.2f}) "
            f"lệch tương đối=({dx:+.2f},{dy:+.2f}) m, lỗi={position_error:.2f} m",
            **row,
        )

    def _current_context(self) -> tuple[GroundTruthSample | None, ObstacleReport | None, Twist]:
        with self.lock:
            gt = self.pose_history[-1] if self.pose_history else None
            entities = self.entities
            clearance = self.latest_front_clearance
            command = self.latest_command
        obstacle = None
        if gt is not None:
            obstacle = identify_forward_obstacle(
                agv_x=gt.x,
                agv_y=gt.y,
                agv_yaw=gt.yaw,
                entities=entities,
                lidar_clearance_m=clearance,
                detection_distance_m=float(
                    self.experiment.get("obstacle_detection_distance_m", 2.2)
                ),
                front_half_angle_rad=float(
                    self.experiment.get("obstacle_front_half_angle_rad", 0.65)
                ),
            )
        return gt, obstacle, command

    def _report_status(self) -> None:
        gt, obstacle, command = self._current_context()
        with self.lock:
            self.current_obstacle = obstacle
        if gt is None:
            self._event("STATUS", "đang chờ pose Gazebo")
            return
        area = self.regions.describe(gt.x, gt.y)
        obstacle_name = obstacle.name if obstacle else None
        if obstacle_name != self.previous_obstacle_name:
            if obstacle is None:
                self._event("CLEAR", f"{area}: phía trước không còn vật cản")
            else:
                self._event(
                    "OBSTACLE",
                    f"{area}: phát hiện {obstacle.label}, khoảng hở LiDAR "
                    f"{obstacle.clearance_m:.2f} m",
                    obstacle=obstacle.label,
                    clearance_m=obstacle.clearance_m,
                )
            self.previous_obstacle_name = obstacle_name

        threshold = float(self.experiment.get("avoidance_angular_speed_radps", 0.12))
        now = time.monotonic()
        if obstacle is not None and abs(command.angular.z) >= threshold:
            direction = "trái" if command.angular.z > 0.0 else "phải"
            key = (obstacle.name, direction)
            if key != self.previous_avoidance_key or now - self.last_avoidance_report_time >= 2.0:
                self._event(
                    "AVOID",
                    f"đang rẽ {direction} để né {obstacle.label} "
                    f"(LiDAR {obstacle.clearance_m:.2f} m)",
                    obstacle=obstacle.label,
                    direction=direction,
                    angular_z=float(command.angular.z),
                )
                self.previous_avoidance_key = key
                self.last_avoidance_report_time = now
        else:
            self.previous_avoidance_key = None

        obstacle_text = (
            f"có: {obstacle.label} ({obstacle.clearance_m:.2f} m)"
            if obstacle
            else "không"
        )
        remaining = (
            f"{self.distance_remaining:.1f} m"
            if math.isfinite(self.distance_remaining)
            else "chưa rõ"
        )
        self._event(
            "STATUS",
            f"{area} | pose Gazebo=({gt.x:.2f},{gt.y:.2f}) | "
            f"đích={self.active_waypoint}, còn={remaining} | vật cản={obstacle_text}",
        )

    def _feedback(self, message: Any) -> None:
        self.distance_remaining = float(message.feedback.distance_remaining)

    def run_route(self) -> bool:
        timeout = float(self.experiment.get("action_server_timeout_sec", 45.0))
        if not self.nav_client.wait_for_server(timeout_sec=timeout):
            self._event("ERROR", "không tìm thấy Nav2 action /navigate_to_pose")
            return False

        route_key = f"{self.phase}_waypoints"
        waypoints = list(
            self.experiment.get(route_key, self.experiment.get("waypoints", ()))
        )
        if not waypoints:
            raise RuntimeError(
                f"warehouse_experiment has no {route_key} or fallback waypoints"
            )
        self._event(
            "ROUTE",
            f"dùng {route_key}: {len(waypoints)} điểm theo corridor latent",
            route_key=route_key,
            waypoint_count=len(waypoints),
        )
        goal_retries = max(0, int(self.experiment.get("goal_retries", 2)))
        retry_delay = max(
            0.0, float(self.experiment.get("goal_retry_delay_sec", 2.0))
        )
        all_succeeded = True
        for index, waypoint in enumerate(waypoints, start=1):
            self.active_waypoint = str(waypoint["name"])
            self.distance_remaining = math.nan
            final_status = "rejected"
            succeeded = False
            attempts_used = 0
            for attempt in range(1, goal_retries + 2):
                attempts_used = attempt
                goal = NavigateToPose.Goal()
                goal.pose.header.frame_id = "map"
                # A zero stamp asks TF for the latest available transform. Gazebo's
                # world pose and /clock can otherwise lead TF by one sensor tick.
                goal.pose.pose.position.x = float(waypoint["x"])
                goal.pose.pose.position.y = float(waypoint["y"])
                yaw = float(waypoint["yaw"])
                goal.pose.pose.orientation.z = math.sin(yaw / 2.0)
                goal.pose.pose.orientation.w = math.cos(yaw / 2.0)
                self._event(
                    "GOAL",
                    f"gửi waypoint {index}/{len(waypoints)} {self.active_waypoint} "
                    f"=({waypoint['x']:.2f},{waypoint['y']:.2f}), "
                    f"lần {attempt}/{goal_retries + 1}",
                )
                send_future = self.nav_client.send_goal_async(
                    goal, feedback_callback=self._feedback
                )
                rclpy.spin_until_future_complete(self, send_future, timeout_sec=10.0)
                handle = send_future.result() if send_future.done() else None
                if handle is None or not handle.accepted:
                    final_status = "rejected"
                    self._event("GOAL_FAILED", f"Nav2 từ chối {self.active_waypoint}")
                else:
                    result_future = handle.get_result_async()
                    deadline = time.monotonic() + float(
                        self.experiment.get("goal_timeout_sec", 100.0)
                    )
                    while (
                        rclpy.ok()
                        and not result_future.done()
                        and time.monotonic() < deadline
                    ):
                        rclpy.spin_once(self, timeout_sec=0.10)
                    if not result_future.done():
                        cancel_future = handle.cancel_goal_async()
                        rclpy.spin_until_future_complete(
                            self, cancel_future, timeout_sec=3.0
                        )
                        final_status = "timeout"
                        self._event(
                            "GOAL_TIMEOUT",
                            f"hủy {self.active_waypoint} vì quá thời gian",
                        )
                    else:
                        status = int(result_future.result().status)
                        succeeded = status == GoalStatus.STATUS_SUCCEEDED
                        final_status = "succeeded" if succeeded else str(status)
                        self._event(
                            "GOAL_DONE",
                            f"{self.active_waypoint}: "
                            f"{'đã tới' if succeeded else f'thất bại status={status}'}",
                        )
                if succeeded or attempt > goal_retries:
                    break
                self._event(
                    "GOAL_RETRY",
                    f"đợi {retry_delay:.1f}s để TF ổn định rồi thử lại "
                    f"{self.active_waypoint}",
                )
                retry_deadline = time.monotonic() + retry_delay
                while rclpy.ok() and time.monotonic() < retry_deadline:
                    rclpy.spin_once(self, timeout_sec=0.10)

            self.goal_results.append(
                {
                    "name": self.active_waypoint,
                    "status": final_status,
                    "attempts": attempts_used,
                }
            )
            all_succeeded &= succeeded
        self.active_waypoint = "hoàn tất"
        return all_succeeded

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        metrics: dict[str, Any]
        if self.comparisons:
            truth = np.asarray(
                [
                    [row["gazebo_x"], row["gazebo_y"], row["gazebo_z"], row["gazebo_yaw"]]
                    for row in self.comparisons
                ],
                dtype=np.float64,
            )
            prediction = np.asarray(
                [
                    [row["vjepa_x"], row["vjepa_y"], row["vjepa_z"], row["vjepa_yaw"]]
                    for row in self.comparisons
                ],
                dtype=np.float64,
            )
            metrics = summarize_localization(truth, prediction)
        else:
            metrics = {"count": 0}
        summary = {
            "phase": self.phase,
            "goals": self.goal_results,
            "all_goals_succeeded": bool(
                self.goal_results
                and all(item["status"] == "succeeded" for item in self.goal_results)
            ),
            "live_localization_metrics": metrics,
            "stationary_predictions_skipped": self.stationary_predictions_skipped,
            "comparison_min_translation_m": self.comparison_min_translation_m,
        }
        (self.output_dir / f"{self.phase}_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self._event(
            "SUMMARY",
            f"lượt {self.phase} hoàn tất: {len(self.goal_results)} waypoint, "
            f"{len(self.comparisons)} phép so sánh V-JEPA/Gazebo",
            summary=summary,
        )
        self.comparison_stream.close()
        self.event_stream.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/warehouse_experiment.yaml")
    parser.add_argument("--phase", choices=("mapping", "query"), required=True)
    parser.add_argument("--output", default="outputs/autonomous_experiment")
    args, ros_args = parser.parse_known_args()
    config = load_config(args.config)
    rclpy.init(args=ros_args)
    node = AutonomousWarehousePatrol(
        config=config,
        phase=args.phase,
        output_dir=Path(args.output),
    )
    exit_code = 0
    try:
        if not node.run_route():
            exit_code = 1
    except KeyboardInterrupt:
        exit_code = 130
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
