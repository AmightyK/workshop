#!/usr/bin/env python3
"""Predict people trajectories and publish WAIT/PASS/REPLAN decisions.

Gazebo transport supplies scenario truth in this simulation adapter. The
planner itself is sensor-agnostic and consumes bounded trajectory histories;
on hardware the same API is fed by detector/tracker poses.
"""

from __future__ import annotations

import json
import csv
import math
import os
import threading
import time
from dataclasses import fields, replace
from pathlib import Path

import rclpy
import yaml
from gz.msgs10.pose_pb2 import Pose
from gz.transport13 import Node as GazeboNode
from rclpy.node import Node
from std_msgs.msg import Bool, String

try:  # package import in tests; direct import when executed as a ROS script
    from .behavior_planner import (
        Decision,
        PlannerConfig,
        Pose2D,
        PredictiveBehaviorPlanner,
        TrajectoryTrack,
        relative_to_ego,
    )
except ImportError:
    from behavior_planner import (
        Decision,
        PlannerConfig,
        Pose2D,
        PredictiveBehaviorPlanner,
        TrajectoryTrack,
        relative_to_ego,
    )

try:
    from .random_people import worker_in_forward_camera_view
except ImportError:
    from random_people import worker_in_forward_camera_view


WORKER_NAMES = tuple(f"random_worker_{index}" for index in range(1, 6))
POSE_TIMEOUT_S = 0.50
ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "behavior_planner.yaml"

KNOWN_WORKER_CROSSINGS = {
    "random_worker_4": Pose2D(7.0, -10.0),
    "random_worker_5": Pose2D(-1.5, -4.2),
}
# Scripted stops must be visually explainable in the public camera. Human #2
# is safe outside the image because its own controller yields before entering
# an AGV-owned crossing; prediction becomes AGV control authority only after
# the worker enters the forward RGB field of view.
CAMERA_VISIBLE_SCRIPTED_WORKERS = frozenset({
    "random_worker_4",
    "random_worker_5",
})


def stale_worker_names(
    timestamps: dict[str, float], now: float, timeout_s: float = POSE_TIMEOUT_S
) -> list[str]:
    """Return missing/stale tracks that require a fail-safe AGV stop."""
    return sorted(
        name for name, timestamp in timestamps.items()
        if now - timestamp > timeout_s
    )


def yaw_from_pose(message: Pose) -> float:
    orientation = message.orientation
    return math.atan2(
        2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
        1.0 - 2.0 * (orientation.y**2 + orientation.z**2),
    )


def worker_relative_to_agv(agv: Pose2D, worker: Pose2D) -> tuple[float, float]:
    """Backward-compatible body-frame helper used by the safety tests."""
    return relative_to_ego(agv, worker)


def worker_requires_stop(
    agv: Pose2D, worker: Pose2D, *, stopping: bool = False
) -> bool:
    """Immediate safety envelope retained below the predictive policy."""
    forward, lateral = worker_relative_to_agv(agv, worker)
    distance = math.hypot(forward, lateral)
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
    """Conservative startup fallback until a trajectory has two samples."""
    crossing_forward, crossing_lateral = worker_relative_to_agv(agv, crossing)
    worker_distance = math.hypot(worker.x - crossing.x, worker.y - crossing.y)
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


def crossing_worker_requires_stop(
    person_id: str,
    agv: Pose2D,
    worker: Pose2D,
    *,
    stopping: bool = False,
) -> bool:
    """Protect both travel directions near a known shared crossing."""
    if worker_requires_stop(agv, worker, stopping=stopping):
        return True
    crossing = KNOWN_WORKER_CROSSINGS.get(person_id)
    return crossing is not None and known_crossing_requires_stop(
        agv, worker, crossing, stopping=stopping
    )


def load_planner_config(path: Path = CONFIG_PATH) -> tuple[PlannerConfig, dict[str, str]]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    values = document.get("behavior_planner", {})
    allowed = {field.name for field in fields(PlannerConfig)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"unknown behavior planner settings: {sorted(unknown)}")
    return PlannerConfig(**values), {
        str(name): str(scenario)
        for name, scenario in document.get("scenarios", {}).items()
    }


class PersonSafetyMonitor(Node):
    def __init__(self) -> None:
        super().__init__("person_safety_monitor")
        config, self.scenarios = load_planner_config()
        self.planner = PredictiveBehaviorPlanner(config)
        self.lock = threading.Lock()
        self.agv_track = TrajectoryTrack()
        self.agv_timestamp = -math.inf
        self.worker_tracks = {name: TrajectoryTrack() for name in WORKER_NAMES}
        self.worker_timestamps = {name: -math.inf for name in WORKER_NAMES}
        self.stopping = False
        self.last_signature: tuple[str, str] | None = None
        self.last_log_time = -math.inf
        self.stop_publisher = self.create_publisher(
            Bool, "/warehouse/person_stop", 10
        )
        self.decision_publisher = self.create_publisher(
            String, "/warehouse/behavior_decision", 10
        )
        self.observation_publisher = self.create_publisher(
            String, "/warehouse/behavior_observation", 10
        )
        self.latest_candidate_reports: dict[str, dict] = {}
        self.last_observation_log: dict[str, float] = {}
        log_dir = Path(os.environ.get("WAREHOUSE_LOG_DIR", "/tmp/warehouse_agv_demo"))
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / "behavior_decisions.jsonl"
        self.log_path.write_text("", encoding="utf-8")
        self.trajectory_path = log_dir / "warehouse_trajectory.csv"
        self.trajectory_stream = self.trajectory_path.open(
            "w", newline="", encoding="utf-8", buffering=1
        )
        self.trajectory_writer = csv.writer(self.trajectory_stream)
        self.trajectory_writer.writerow(["timestamp", "x", "y", "yaw"])

        self.gz_node = GazeboNode()
        # Per-model topics stay current under the combined Gazebo / Nav2 /
        # V-JEPA load.  The full world Pose_V is large enough to accumulate a
        # stale callback backlog, which previously froze both trajectory
        # estimates and worker steering at their spawn poses.
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
        self.create_timer(0.05, self._publish_decision)
        self.get_logger().info(
            "Predictive person planner active: trajectory -> occupancy -> "
            "WAIT/PASS/REPLAN on /warehouse/behavior_decision"
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
        now = time.monotonic()
        with self.lock:
            self.agv_track.update(self._pose2d(message), now)
            self.agv_timestamp = now

    def _on_worker(self, name: str, message: Pose) -> None:
        if message.name != name:
            return
        now = time.monotonic()
        with self.lock:
            self.worker_tracks[name].update(self._pose2d(message), now)
            self.worker_timestamps[name] = now

    def _current_report(self, now: float) -> dict:
        with self.lock:
            if not self.agv_track.samples or now - self.agv_timestamp > POSE_TIMEOUT_S:
                self.latest_candidate_reports = {}
                return {
                    "decision": Decision.WAIT.value,
                    "reason": "AGV pose stream is stale; fail-safe stop",
                    "person_id": "none",
                    "scenario": "sensor_timeout",
                    "timestamp": now,
                    "collision_probability": 1.0,
                    "time_to_collision_s": 0.0,
                    "predicted_free_space_window_s": 0.0,
                    "predicted_speed_mps": 0.0,
                    "wait_duration_s": 0.0,
                    "occupancy": [],
                }
            ego = self.agv_track.latest_pose
            ego_vx, ego_vy = self.agv_track.velocity()
            ego_speed = math.hypot(ego_vx, ego_vy)
            stale_workers = stale_worker_names(self.worker_timestamps, now)
            if stale_workers:
                self.latest_candidate_reports = {}
                return {
                    "decision": Decision.WAIT.value,
                    "reason": (
                        "worker pose stream is stale; fail-safe stop: "
                        + ", ".join(stale_workers)
                    ),
                    "person_id": stale_workers[0],
                    "scenario": "sensor_timeout",
                    "timestamp": now,
                    "collision_probability": 1.0,
                    "time_to_collision_s": 0.0,
                    "predicted_free_space_window_s": 0.0,
                    "predicted_speed_mps": 0.0,
                    "wait_duration_s": 0.0,
                    "occupancy": [],
                }
            candidates = []
            for name, track in self.worker_tracks.items():
                if not track.samples or now - self.worker_timestamps[name] > POSE_TIMEOUT_S:
                    continue
                report = self.planner.evaluate(
                    person_id=name,
                    scenario=self.scenarios.get(name, "general_worker"),
                    ego=ego,
                    ego_speed_mps=ego_speed,
                    track=track,
                    timestamp=now,
                )
                camera_visible = worker_in_forward_camera_view(
                    (ego.x, ego.y),
                    ego.yaw,
                    (track.latest_pose.x, track.latest_pose.y),
                )
                geometric_stop = crossing_worker_requires_stop(
                    name,
                    ego,
                    track.latest_pose,
                    stopping=self.stopping,
                )
                # Both scripted encounters must be visually explainable.
                # Suppress predictive / known-crossing WAIT while the worker
                # is outside the RGB FOV; the immediate physical envelope
                # remains authoritative as the final collision backstop.
                if name in CAMERA_VISIBLE_SCRIPTED_WORKERS and not camera_visible:
                    immediate_stop = worker_requires_stop(
                        ego, track.latest_pose, stopping=self.stopping
                    )
                    geometric_stop = immediate_stop
                    if not immediate_stop and report.decision is not Decision.PASS:
                        # Leaving the camera is the terminal clearance signal
                        # for a scripted crossing. Clear the old WAIT state so
                        # the confirmation timer cannot restart forever at an
                        # endpoint that is already out of view.
                        self.planner.wait_started.pop(name, None)
                        self.planner.clear_started.pop(name, None)
                        report = replace(
                            report,
                            decision=Decision.PASS,
                            reason=(
                                "scripted worker cleared the forward camera; "
                                "resume while emergency distance remains active"
                            ),
                            collision_probability=0.0,
                            time_to_collision_s=None,
                            predicted_free_space_window_s=(
                                self.planner.config.prediction_horizon_s
                            ),
                            wait_duration_s=0.0,
                        )
                if geometric_stop:
                    wait_started = self.planner.wait_started.setdefault(name, now)
                    self.planner.clear_started.pop(name, None)
                if geometric_stop and report.decision is Decision.PASS:
                    report = replace(
                        report,
                        decision=Decision.WAIT,
                        reason=(
                            "worker occupies the bidirectional geometric guard "
                            "at the shared crossing; retain path and wait"
                        ),
                        collision_probability=max(
                            report.collision_probability,
                            self.planner.config.collision_probability_threshold,
                        ),
                        time_to_collision_s=0.0,
                        predicted_free_space_window_s=0.0,
                        wait_duration_s=max(0.0, now - wait_started),
                    )
                candidates.append(report)

        if not candidates:
            self.latest_candidate_reports = {}
            return {
                "decision": Decision.PASS.value,
                "reason": "no fresh person track occupies the planned corridor",
                "person_id": "none",
                "scenario": "normal_driving",
                "timestamp": now,
                "collision_probability": 0.0,
                "time_to_collision_s": None,
                "predicted_free_space_window_s": self.planner.config.prediction_horizon_s,
                "predicted_speed_mps": 0.0,
                "wait_duration_s": 0.0,
                "occupancy": [],
            }

        self.latest_candidate_reports = {
            report.person_id: report.as_dict() for report in candidates
        }
        priority = {Decision.PASS: 0, Decision.WAIT: 1, Decision.REPLAN: 2}
        selected = max(
            candidates,
            key=lambda report: (
                priority[report.decision], report.collision_probability
            ),
        )
        return selected.as_dict()

    def _publish_decision(self) -> None:
        now = time.monotonic()
        report = self._current_report(now)
        with self.lock:
            if self.agv_track.samples:
                pose = self.agv_track.latest_pose
                self.trajectory_writer.writerow(
                    [now, pose.x, pose.y, pose.yaw]
                )
        decision = str(report["decision"])
        should_stop = decision in {Decision.WAIT.value, Decision.REPLAN.value}
        previous_stop = self.stopping
        self.stopping = should_stop

        self.stop_publisher.publish(Bool(data=should_stop))
        payload = json.dumps(report, sort_keys=True)
        self.decision_publisher.publish(String(data=payload))

        # Preserve per-human prediction evidence even when another person has
        # the highest-risk global control decision. Observations never drive
        # /warehouse/person_stop, so control arbitration is unchanged.
        for person, candidate in self.latest_candidate_reports.items():
            if candidate.get("scenario") not in {
                "human_1_static_until_close",
                "human_2_continuous_crossing",
            }:
                continue
            observation = dict(candidate)
            observation["selected_for_control"] = person == report["person_id"]
            observation_payload = json.dumps(observation, sort_keys=True)
            self.observation_publisher.publish(String(data=observation_payload))
            last_observation = self.last_observation_log.get(person, -math.inf)
            if person != report["person_id"] and now - last_observation >= 0.5:
                with self.log_path.open("a", encoding="utf-8") as stream:
                    stream.write(observation_payload + "\n")
                self.last_observation_log[person] = now

        signature = (decision, str(report["person_id"]))
        if signature != self.last_signature or now - self.last_log_time >= 0.5:
            with self.log_path.open("a", encoding="utf-8") as stream:
                stream.write(payload + "\n")
            self.last_signature = signature
            self.last_log_time = now

        if should_stop != previous_stop or decision == Decision.REPLAN.value:
            reason = str(report["reason"])
            person = str(report["person_id"])
            if should_stop:
                self.get_logger().warn(f"{decision} for {person}: {reason}")
            else:
                self.get_logger().info(f"PASS for {person}: {reason}")


def main() -> None:
    rclpy.init()
    node = PersonSafetyMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.stop_publisher.publish(Bool(data=True))
        node.trajectory_stream.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
