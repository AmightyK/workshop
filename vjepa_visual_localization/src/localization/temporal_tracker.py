"""Temporal pose tracking for visually ambiguous warehouse aisles."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.localization.pose_estimator import PosePrediction
from src.retrieval.global_retriever import RetrievalResult


def wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class TemporalTrackingResult:
    prediction: PosePrediction
    state: str
    raw_prediction: PosePrediction
    selected_rank: int | None
    raw_jump_m: float
    accepted_step_m: float
    translation_gate_m: float
    yaw_gate_rad: float
    rejected_streak: int


class TemporalPoseTracker:
    """Track visual pose using only current retrieval and prior visual poses.

    The tracker never reads odometry or Gazebo truth. A candidate must be
    reachable from the previous accepted V-JEPA pose within a time-dependent
    translation/yaw gate. If no candidate passes, the previous pose is held so
    a visually repeated shelf cannot teleport the estimate across the map.
    """

    def __init__(
        self,
        *,
        max_linear_speed_mps: float = 1.25,
        max_angular_speed_radps: float = 2.4,
        base_translation_gate_m: float = 0.65,
        base_yaw_gate_rad: float = 0.55,
        max_translation_gate_m: float = 2.2,
        max_yaw_gate_rad: float = 2.8,
        distance_penalty: float = 0.04,
        yaw_penalty: float = 0.015,
        smoothing_alpha: float = 0.82,
        max_index_advance_per_sec: float = 4.0,
        forward_progress_bonus: float = 0.055,
        stationary_translation_gate_m: float = 0.18,
        initial_route_index: int | None = None,
        initial_route_window: int = 0,
        relocalization_frames: int = 4,
        relocalization_cluster_radius_m: float = 1.2,
        relocalization_min_similarity: float = 0.90,
        relocalization_max_jump_m: float = 4.0,
    ) -> None:
        if max_linear_speed_mps <= 0.0 or max_angular_speed_radps <= 0.0:
            raise ValueError("temporal speed limits must be positive")
        if not 0.0 < smoothing_alpha <= 1.0:
            raise ValueError("smoothing_alpha must be in (0, 1]")
        if relocalization_frames < 2:
            raise ValueError("relocalization_frames must be at least 2")
        if max_index_advance_per_sec <= 0.0:
            raise ValueError("max_index_advance_per_sec must be positive")
        self.max_linear_speed_mps = float(max_linear_speed_mps)
        self.max_angular_speed_radps = float(max_angular_speed_radps)
        self.base_translation_gate_m = float(base_translation_gate_m)
        self.base_yaw_gate_rad = float(base_yaw_gate_rad)
        self.max_translation_gate_m = float(max_translation_gate_m)
        self.max_yaw_gate_rad = float(max_yaw_gate_rad)
        self.distance_penalty = float(distance_penalty)
        self.yaw_penalty = float(yaw_penalty)
        self.smoothing_alpha = float(smoothing_alpha)
        self.max_index_advance_per_sec = float(max_index_advance_per_sec)
        self.forward_progress_bonus = float(forward_progress_bonus)
        self.stationary_translation_gate_m = float(stationary_translation_gate_m)
        if self.stationary_translation_gate_m < 0.0:
            raise ValueError("stationary_translation_gate_m must be non-negative")
        if initial_route_index is not None and initial_route_index < 0:
            raise ValueError("initial_route_index must be non-negative")
        if initial_route_window < 0:
            raise ValueError("initial_route_window must be non-negative")
        self.initial_route_index = (
            None if initial_route_index is None else int(initial_route_index)
        )
        self.initial_route_window = int(initial_route_window)
        self.relocalization_frames = int(relocalization_frames)
        self.relocalization_cluster_radius_m = float(relocalization_cluster_radius_m)
        self.relocalization_min_similarity = float(relocalization_min_similarity)
        self.relocalization_max_jump_m = float(relocalization_max_jump_m)
        self.reset()

    def reset(self) -> None:
        self.pose: np.ndarray | None = None
        self.source_id: str | None = None
        self.source_index: int | None = None
        self.timestamp: float | None = None
        self.rejected_streak = 0
        self.relocalization_pose: np.ndarray | None = None
        self.relocalization_count = 0
        # Fractional progress through the recorded latent sequence, inferred
        # only from image motion.  Unlike a fixed score bonus, this cannot
        # leave the tracker pinned indefinitely to one visually repetitive
        # shelf while the camera keeps moving forward.
        self.motion_credit = 0.0

    @staticmethod
    def _raw_prediction(retrieval: RetrievalResult) -> PosePrediction:
        if len(retrieval.indices) == 0:
            raise ValueError("retrieval result is empty")
        return PosePrediction(
            pose=retrieval.poses[0].copy(),
            source_id=str(retrieval.ids[0]),
            score=float(retrieval.scores[0]),
            method="raw_global_top1",
        )

    def _initialize(
        self, retrieval: RetrievalResult, timestamp: float
    ) -> TemporalTrackingResult:
        raw = self._raw_prediction(retrieval)
        rank = 0
        if self.initial_route_index is not None:
            # The warehouse demo always starts at the beginning of the saved
            # closed route.  Repeated rack/corridor views make a global top-1
            # unsafe before a previous visual pose exists, so use the best
            # visual match inside the configured start window as that first
            # prior.  Retrieval remains camera-only; no truth/odometry enters
            # this choice.
            lower = self.initial_route_index
            upper = lower + self.initial_route_window
            initial_ranks = np.flatnonzero(
                (retrieval.indices >= lower) & (retrieval.indices <= upper)
            )
            if len(initial_ranks) > 0:
                # Retrieval results are already ordered by similarity.
                rank = int(initial_ranks[0])
        self.pose = retrieval.poses[rank].copy()
        self.source_id = str(retrieval.ids[rank])
        self.source_index = int(retrieval.indices[rank])
        self.timestamp = float(timestamp)
        prediction = PosePrediction(
            self.pose.copy(),
            self.source_id,
            float(retrieval.scores[rank]),
            (
                "temporal_initialize_route_prior"
                if self.initial_route_index is not None
                else "temporal_initialize"
            ),
        )
        return TemporalTrackingResult(
            prediction, "INITIALIZED", raw, rank, 0.0, 0.0,
            self.base_translation_gate_m, self.base_yaw_gate_rad, 0,
        )

    def _smooth(self, target: np.ndarray) -> np.ndarray:
        assert self.pose is not None
        alpha = self.smoothing_alpha
        output = self.pose.copy()
        output[:3] += alpha * (target[:3] - output[:3])
        output[3] = wrap_angle(output[3] + alpha * wrap_angle(target[3] - output[3]))
        return output

    def update(
        self,
        retrieval: RetrievalResult,
        *,
        timestamp: float,
        camera_moving: bool = False,
        camera_progress_scale: float = 1.0,
    ) -> TemporalTrackingResult:
        """Choose and smooth a reachable candidate, or hold the prior pose."""
        if self.pose is None or self.timestamp is None:
            return self._initialize(retrieval, timestamp)
        if timestamp <= self.timestamp:
            self.reset()
            return self._initialize(retrieval, timestamp)

        raw = self._raw_prediction(retrieval)
        previous = self.pose.copy()
        dt = min(max(float(timestamp) - self.timestamp, 0.0), 2.5)
        if camera_moving:
            progress_scale = min(max(float(camera_progress_scale), 0.0), 1.5)
            self.motion_credit = min(
                6.0,
                self.motion_credit
                + self.max_index_advance_per_sec * dt * progress_scale,
            )
        else:
            # Keep at most a fractional remainder. A stopped/yawing camera
            # must never spend old credit to translate to a neighboring rack.
            self.motion_credit = min(self.motion_credit, 0.99)
        translation_gate = min(
            self.max_translation_gate_m,
            self.base_translation_gate_m + self.max_linear_speed_mps * dt,
        )
        yaw_gate = min(
            self.max_yaw_gate_rad,
            self.base_yaw_gate_rad + self.max_angular_speed_radps * dt,
        )
        distances = np.linalg.norm(retrieval.poses[:, :2] - previous[:2], axis=1)
        yaw_distances = np.abs(
            np.asarray([wrap_angle(value - previous[3]) for value in retrieval.poses[:, 3]])
        )
        valid = (distances <= translation_gate) & (yaw_distances <= yaw_gate)
        if not camera_moving:
            # A yawing or stationary camera may select a different view at the
            # same physical place, but it cannot translate the robot to a
            # neighboring repeated shelf. Wheel odometry remains responsible
            # for short-term motion while this visual position is held.
            valid &= distances <= self.stationary_translation_gate_m
        # Map clips are recorded in route order. Enforce forward continuity in
        # that latent sequence, including wrap-around for a closed loop. This
        # prevents repeated shelves from pulling the tracker back to an older
        # clip even when the old visual score is marginally higher.
        forward_steps: np.ndarray | None = None
        if self.source_index is not None:
            map_count = int(np.max(retrieval.indices)) + 1
            max_forward_steps = max(
                2, int(math.ceil(self.max_index_advance_per_sec * max(dt, 0.5)))
            )
            if (
                map_count > 2 * max_forward_steps
                and self.source_index >= map_count - max_forward_steps
            ):
                forward_steps = (retrieval.indices - self.source_index) % map_count
                sequence_valid = forward_steps <= max_forward_steps
            else:
                forward_steps = retrieval.indices - self.source_index
                sequence_valid = (forward_steps >= 0) & (
                    forward_steps <= max_forward_steps
                )
            valid &= sequence_valid
        raw_jump = float(distances[0])

        if np.any(valid):
            ranks = np.flatnonzero(valid)
            if (
                camera_moving
                and forward_steps is not None
                and self.motion_credit >= 1.0
            ):
                # Sustained forward optical flow earns latent-sequence steps.
                # Once at least one step is earned, do not let a marginally
                # higher repeated-shelf score select the current clip again.
                # Never select a future anchor before its full image-motion
                # credit has accumulated. ceil() allowed 1.01 credits to jump
                # two clips and made the estimate run ahead during avoidance.
                affordable_steps = max(1, int(math.floor(self.motion_credit)))
                advancing = (
                    valid
                    & (forward_steps > 0)
                    & (forward_steps <= affordable_steps)
                )
                if np.any(advancing):
                    ranks = np.flatnonzero(advancing)
            adjusted = (
                retrieval.scores[ranks]
                - self.distance_penalty * np.square(distances[ranks] / translation_gate)
                - self.yaw_penalty * np.square(yaw_distances[ranks] / yaw_gate)
            )
            if camera_moving and forward_steps is not None:
                # A changing camera view is an image-only motion cue. Favor the
                # immediately following recorded clips so a repeated rack does
                # not keep the tracker pinned to its old latent. The bonus
                # decays quickly and never bypasses the spatial/yaw hard gates.
                progress = forward_steps[ranks]
                positive = progress > 0
                progress_bonus = np.zeros_like(adjusted)
                progress_bonus[positive] = self.forward_progress_bonus * np.exp(
                    -0.55 * (progress[positive] - 1)
                )
                adjusted += progress_bonus
            rank = int(ranks[int(np.argmax(adjusted))])
            target = retrieval.poses[rank]
            self.pose = self._smooth(target)
            self.source_id = str(retrieval.ids[rank])
            self.source_index = int(retrieval.indices[rank])
            if forward_steps is not None:
                consumed_steps = max(0, int(forward_steps[rank]))
                self.motion_credit = max(0.0, self.motion_credit - consumed_steps)
            self.timestamp = float(timestamp)
            self.rejected_streak = 0
            self.relocalization_pose = None
            self.relocalization_count = 0
            step = float(np.linalg.norm(self.pose[:2] - previous[:2]))
            prediction = PosePrediction(
                self.pose.copy(),
                self.source_id,
                float(retrieval.scores[rank]),
                "temporal_prior",
            )
            return TemporalTrackingResult(
                prediction, "TRACKING", raw, rank, raw_jump, step,
                translation_gate, yaw_gate, 0,
            )

        self.rejected_streak += 1
        if (
            self.relocalization_pose is not None
            and np.linalg.norm(raw.pose[:2] - self.relocalization_pose[:2])
            <= self.relocalization_cluster_radius_m
        ):
            self.relocalization_count += 1
        else:
            self.relocalization_pose = raw.pose.copy()
            self.relocalization_count = 1

        can_relocalize = (
            camera_moving
            and self.relocalization_count >= self.relocalization_frames
            and raw.score >= self.relocalization_min_similarity
            and raw_jump <= self.relocalization_max_jump_m
        )
        if can_relocalize:
            self.pose = raw.pose.copy()
            self.source_id = raw.source_id
            self.source_index = int(retrieval.indices[0])
            self.timestamp = float(timestamp)
            self.rejected_streak = 0
            self.relocalization_pose = None
            self.relocalization_count = 0
            self.motion_credit = 0.0
            prediction = PosePrediction(
                self.pose.copy(), self.source_id, raw.score, "temporal_relocalized"
            )
            return TemporalTrackingResult(
                prediction, "RELOCALIZED", raw, 0, raw_jump, raw_jump,
                translation_gate, yaw_gate, 0,
            )

        # Advance the timestamp while holding. This prevents a long pause from
        # silently expanding the next gate enough to accept a cross-map alias.
        self.timestamp = float(timestamp)
        prediction = PosePrediction(
            previous,
            self.source_id or raw.source_id,
            raw.score,
            "temporal_hold",
        )
        return TemporalTrackingResult(
            prediction, "HOLDING", raw, None, raw_jump, 0.0,
            translation_gate, yaw_gate, self.rejected_streak,
        )
