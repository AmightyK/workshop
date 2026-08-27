#!/usr/bin/env python3
"""Pure trajectory prediction and behavior decisions for warehouse people.

The ROS/Gazebo adapter lives in :mod:`person_safety_monitor`.  Keeping the
world-model math here makes the safety policy deterministic and unit-testable
without a running simulator.
"""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import asdict, dataclass
from enum import Enum


class Decision(str, Enum):
    WAIT = "WAIT"
    PASS = "PASS"
    REPLAN = "REPLAN"


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float = 0.0


@dataclass(frozen=True)
class TrackSample:
    timestamp: float
    pose: Pose2D


@dataclass(frozen=True)
class PredictedOccupancy:
    horizon_s: float
    x: float
    y: float
    forward_m: float
    lateral_m: float
    separation_m: float
    collision_probability: float
    path_occupied: bool


@dataclass(frozen=True)
class PlannerConfig:
    prediction_horizon_s: float = 4.0
    prediction_step_s: float = 0.25
    nominal_approach_speed_mps: float = 0.55
    corridor_half_width_m: float = 0.62
    combined_safety_radius_m: float = 0.62
    emergency_radius_m: float = 0.72
    probability_sigma_m: float = 0.34
    collision_probability_threshold: float = 0.55
    required_pass_window_s: float = 2.0
    replan_after_s: float = 10.0
    replan_cooldown_s: float = 15.0
    stationary_speed_mps: float = 0.08
    overtake_after_s: float = 4.0
    overtake_min_lateral_m: float = 0.55
    overtake_min_separation_m: float = 1.20
    overtake_authorization_s: float = 5.0
    allow_stationary_overtake: bool = False
    clearance_confirmation_s: float = 1.0


@dataclass(frozen=True)
class DecisionReport:
    decision: Decision
    reason: str
    person_id: str
    scenario: str
    timestamp: float
    collision_probability: float
    time_to_collision_s: float | None
    predicted_free_space_window_s: float
    predicted_speed_mps: float
    wait_duration_s: float
    occupancy: tuple[PredictedOccupancy, ...]

    def as_dict(self) -> dict:
        result = asdict(self)
        result["decision"] = self.decision.value
        return result

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True)


class TrajectoryTrack:
    """Bounded person history with a least-squares constant-velocity model."""

    def __init__(self, history_size: int = 12) -> None:
        if history_size < 2:
            raise ValueError("history_size must be at least two")
        self.samples: deque[TrackSample] = deque(maxlen=int(history_size))

    def update(self, pose: Pose2D, timestamp: float) -> None:
        timestamp = float(timestamp)
        if self.samples and timestamp <= self.samples[-1].timestamp:
            if timestamp == self.samples[-1].timestamp:
                self.samples[-1] = TrackSample(timestamp, pose)
                return
            self.samples.clear()
        self.samples.append(TrackSample(timestamp, pose))

    @property
    def latest_pose(self) -> Pose2D:
        if not self.samples:
            raise RuntimeError("trajectory has no samples")
        return self.samples[-1].pose

    def velocity(self) -> tuple[float, float]:
        if len(self.samples) < 2:
            return 0.0, 0.0
        # Regression is less sensitive to one Gazebo pose jitter than a
        # two-frame finite difference. Rebase time to retain numeric accuracy.
        t0 = self.samples[0].timestamp
        times = [sample.timestamp - t0 for sample in self.samples]
        mean_t = sum(times) / len(times)
        denominator = sum((value - mean_t) ** 2 for value in times)
        if denominator <= 1.0e-9:
            return 0.0, 0.0
        mean_x = sum(sample.pose.x for sample in self.samples) / len(self.samples)
        mean_y = sum(sample.pose.y for sample in self.samples) / len(self.samples)
        vx = sum(
            (time_value - mean_t) * (sample.pose.x - mean_x)
            for time_value, sample in zip(times, self.samples)
        ) / denominator
        vy = sum(
            (time_value - mean_t) * (sample.pose.y - mean_y)
            for time_value, sample in zip(times, self.samples)
        ) / denominator
        return float(vx), float(vy)

    def predict(self, horizon_s: float) -> Pose2D:
        pose = self.latest_pose
        vx, vy = self.velocity()
        return Pose2D(
            pose.x + vx * float(horizon_s),
            pose.y + vy * float(horizon_s),
            pose.yaw,
        )


def relative_to_ego(ego: Pose2D, point: Pose2D) -> tuple[float, float]:
    dx = point.x - ego.x
    dy = point.y - ego.y
    cosine = math.cos(ego.yaw)
    sine = math.sin(ego.yaw)
    return cosine * dx + sine * dy, -sine * dx + cosine * dy


class PredictiveBehaviorPlanner:
    """Choose WAIT, PASS, or REPLAN from predicted future occupancy."""

    def __init__(self, config: PlannerConfig | None = None) -> None:
        self.config = config or PlannerConfig()
        self.wait_started: dict[str, float] = {}
        self.clear_started: dict[str, float] = {}
        self.last_replan: dict[str, float] = {}
        self.pass_authorized_until: dict[str, float] = {}

    def _rollout(
        self,
        ego: Pose2D,
        ego_speed_mps: float,
        track: TrajectoryTrack,
    ) -> tuple[PredictedOccupancy, ...]:
        config = self.config
        approach_speed = max(
            0.0,
            min(
                config.nominal_approach_speed_mps,
                max(float(ego_speed_mps), config.nominal_approach_speed_mps * 0.65),
            ),
        )
        count = int(round(config.prediction_horizon_s / config.prediction_step_s))
        rollout = []
        for index in range(count + 1):
            horizon = index * config.prediction_step_s
            person = track.predict(horizon)
            ego_future = Pose2D(
                ego.x + math.cos(ego.yaw) * approach_speed * horizon,
                ego.y + math.sin(ego.yaw) * approach_speed * horizon,
                ego.yaw,
            )
            forward, lateral = relative_to_ego(ego_future, person)
            separation = math.hypot(forward, lateral)
            path_occupied = (
                -config.combined_safety_radius_m <= forward
                <= config.nominal_approach_speed_mps * config.prediction_horizon_s
                + config.combined_safety_radius_m
                and abs(lateral) <= config.corridor_half_width_m
            )
            clearance = max(0.0, separation - config.combined_safety_radius_m)
            probability = (
                math.exp(
                    -0.5 * (clearance / max(config.probability_sigma_m, 1.0e-6)) ** 2
                )
                if path_occupied
                else 0.0
            )
            rollout.append(
                PredictedOccupancy(
                    horizon_s=horizon,
                    x=person.x,
                    y=person.y,
                    forward_m=forward,
                    lateral_m=lateral,
                    separation_m=separation,
                    collision_probability=probability,
                    path_occupied=path_occupied,
                )
            )
        return tuple(rollout)

    def evaluate(
        self,
        *,
        person_id: str,
        scenario: str,
        ego: Pose2D,
        ego_speed_mps: float,
        track: TrajectoryTrack,
        timestamp: float,
    ) -> DecisionReport:
        config = self.config
        timestamp = float(timestamp)
        occupancy = self._rollout(ego, ego_speed_mps, track)
        probability = max(item.collision_probability for item in occupancy)
        collision_times = [
            item.horizon_s
            for item in occupancy
            if item.collision_probability >= config.collision_probability_threshold
        ]
        time_to_collision = min(collision_times) if collision_times else None

        safe_run = 0.0
        best_safe_run = 0.0
        for item in occupancy:
            if item.collision_probability < config.collision_probability_threshold:
                safe_run += config.prediction_step_s
                best_safe_run = max(best_safe_run, safe_run)
            else:
                safe_run = 0.0
        free_window = min(config.prediction_horizon_s, best_safe_run)

        vx, vy = track.velocity()
        speed = math.hypot(vx, vy)
        current_separation = occupancy[0].separation_m
        immediate_emergency = current_separation <= config.emergency_radius_m
        unsafe_window = (
            time_to_collision is not None
            and time_to_collision <= config.required_pass_window_s
        )
        blocked = immediate_emergency or unsafe_window

        if blocked:
            self.clear_started.pop(person_id, None)
            wait_started = self.wait_started.setdefault(person_id, timestamp)
            wait_duration = max(0.0, timestamp - wait_started)
            last_replan = self.last_replan.get(person_id, -math.inf)
            pass_active = timestamp < self.pass_authorized_until.get(
                person_id, -math.inf
            )
            can_start_overtake = (
                config.allow_stationary_overtake
                and
                scenario == "human_2_continuous_crossing"
                and wait_duration >= config.overtake_after_s
                and speed <= config.stationary_speed_mps
                and abs(occupancy[0].lateral_m)
                >= config.overtake_min_lateral_m
                and current_separation >= config.overtake_min_separation_m
                and not immediate_emergency
            )
            if pass_active or can_start_overtake:
                decision = Decision.PASS
                if can_start_overtake:
                    self.pass_authorized_until[person_id] = (
                        timestamp + config.overtake_authorization_s
                    )
                reason = (
                    "safe lateral overtake authorized: crossing human is "
                    f"stationary at corridor edge {abs(occupancy[0].lateral_m):.2f}m, "
                    f"separation {current_separation:.2f}m; local MPPI retains "
                    "collision authority"
                )
            elif (
                scenario not in {
                    "human_1_static_until_close",
                    "human_1_continuous_crossing",
                    "human_2_continuous_crossing",
                }
                and
                wait_duration >= config.replan_after_s
                and timestamp - last_replan >= config.replan_cooldown_s
            ):
                decision = Decision.REPLAN
                self.last_replan[person_id] = timestamp
                reason = (
                    f"predicted corridor blockage persisted {wait_duration:.1f}s; "
                    "requesting one bounded Nav2 replan"
                )
            else:
                decision = Decision.WAIT
                if immediate_emergency:
                    reason = (
                        f"person inside {config.emergency_radius_m:.2f}m emergency "
                        "envelope; retain path and wait"
                    )
                else:
                    reason = (
                        f"collision probability {probability:.2f} with TTC "
                        f"{time_to_collision:.2f}s; free window "
                        f"{free_window:.2f}s is too short"
                    )
        else:
            self.pass_authorized_until.pop(person_id, None)
            previous_wait = self.wait_started.get(person_id)
            wait_duration = (
                max(0.0, timestamp - previous_wait)
                if previous_wait is not None
                else 0.0
            )
            if previous_wait is not None:
                clear_started = self.clear_started.setdefault(person_id, timestamp)
                clear_duration = max(0.0, timestamp - clear_started)
                if clear_duration < config.clearance_confirmation_s:
                    decision = Decision.WAIT
                    reason = (
                        f"predicted path has been clear for {clear_duration:.2f}s; "
                        f"confirming {config.clearance_confirmation_s:.2f}s continuous "
                        "clearance before resume"
                    )
                else:
                    self.wait_started.pop(person_id, None)
                    self.clear_started.pop(person_id, None)
                    decision = Decision.PASS
                    reason = (
                        "path cleared continuously; resume: predicted collision "
                        f"probability {probability:.2f}, free-space window "
                        f"{free_window:.2f}s"
                    )
            else:
                self.clear_started.pop(person_id, None)
                decision = Decision.PASS
                reason = (
                    f"safe pass: predicted collision probability {probability:.2f}, "
                    f"free-space window {free_window:.2f}s"
                )

        return DecisionReport(
            decision=decision,
            reason=reason,
            person_id=person_id,
            scenario=scenario,
            timestamp=timestamp,
            collision_probability=probability,
            time_to_collision_s=time_to_collision,
            predicted_free_space_window_s=free_window,
            predicted_speed_mps=speed,
            wait_duration_s=wait_duration,
            occupancy=occupancy,
        )
