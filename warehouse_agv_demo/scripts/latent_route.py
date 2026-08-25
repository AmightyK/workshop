"""Build Nav2 checkpoint corridors from the exact recorded V-JEPA route."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LATENT_MAP = (
    WORKSPACE_ROOT
    / "vjepa_visual_localization"
    / "outputs"
    / "autonomous_map_dense"
)


def wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def orient_route_tangents(
    poses: np.ndarray, *, lookahead_m: float = 0.35
) -> np.ndarray:
    """Use stable path tangents and preserve the final goal yaw.

    Saved mapping poses contain several in-place rotations.  Using the first
    5 cm displacement after each pose made those camera-yaw samples become
    Nav2 heading goals, so the short AGV visibly wagged left and right.  A
    longer spatial lookahead follows the aisle direction instead.
    """
    route = np.asarray(poses, dtype=np.float64).copy()
    if len(route) <= 1:
        return route
    for index in range(len(route) - 1):
        fallback_delta = None
        for lookahead in range(index + 1, len(route)):
            delta = route[lookahead, :2] - route[index, :2]
            if float(np.linalg.norm(delta)) >= 0.05:
                fallback_delta = delta
            if float(np.linalg.norm(delta)) >= float(lookahead_m):
                route[index, 3] = math.atan2(float(delta[1]), float(delta[0]))
                break
        else:
            # The final short leg can be below the normal lookahead.  It is
            # still a translation goal, so face it instead of restoring a
            # recorded in-place camera yaw at the penultimate checkpoint.
            if fallback_delta is not None:
                route[index, 3] = math.atan2(
                    float(fallback_delta[1]), float(fallback_delta[0])
                )
    return route


def prune_hard_corner_checkpoints(
    poses: np.ndarray, *, angle_threshold_rad: float = 1.05
) -> np.ndarray:
    """Remove via points that force a differential base to stop at a corner.

    NavFn still plans between the retained points and therefore remains
    responsible for obstacle avoidance. Removing only the geometric corner
    lets the global smoother create an approach arc before the intersection.
    """
    route = np.asarray(poses, dtype=np.float64)
    if len(route) <= 2:
        return route.copy()
    kept = [0]
    for index in range(1, len(route) - 1):
        incoming = route[index, :2] - route[kept[-1], :2]
        outgoing = route[index + 1, :2] - route[index, :2]
        if float(np.linalg.norm(incoming)) <= 0.05:
            continue
        if float(np.linalg.norm(outgoing)) <= 0.05:
            continue
        incoming_yaw = math.atan2(float(incoming[1]), float(incoming[0]))
        outgoing_yaw = math.atan2(float(outgoing[1]), float(outgoing[0]))
        turn = abs(wrap_angle(outgoing_yaw - incoming_yaw))
        if turn >= float(angle_threshold_rad):
            continue
        kept.append(index)
    kept.append(len(route) - 1)
    return route[np.asarray(kept, dtype=np.int64)].copy()


@dataclass(frozen=True)
class LatentRouteSegment:
    poses: np.ndarray
    start_index: int
    end_index: int
    target_error_m: float
    map_dir: Path


class LatentRoutePlanner:
    """Select forward-only route segments from saved map poses.

    The pose array is stored in the same order as the mapping traversal.  A
    pick mission may only advance through that sequence; it cannot invent a
    shortcut through a visually unrecorded aisle.
    """

    def __init__(
        self,
        map_dir: Path | str | None = None,
        *,
        max_target_error_m: float = 0.80,
        max_target_yaw_error_rad: float = 0.70,
    ) -> None:
        configured = map_dir or os.environ.get("WAREHOUSE_VJEPA_MAP")
        self.map_dir = Path(configured) if configured else DEFAULT_LATENT_MAP
        pose_path = self.map_dir / "poses.npy"
        if not pose_path.is_file():
            raise FileNotFoundError(
                f"latent route is missing {pose_path}; rebuild the V-JEPA map first"
            )
        poses = np.asarray(np.load(pose_path), dtype=np.float64)
        if poses.ndim != 2 or poses.shape[1] < 4 or len(poses) < 2:
            raise ValueError(f"invalid latent route pose array: {pose_path}")
        if not np.all(np.isfinite(poses[:, :4])):
            raise ValueError(f"latent route contains non-finite poses: {pose_path}")
        self.poses = poses[:, :4]
        self.max_target_error_m = float(max_target_error_m)
        self.max_target_yaw_error_rad = float(max_target_yaw_error_rad)

    @staticmethod
    def _target_xy_yaw(
        target_pose: list[float] | tuple[float, ...] | np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Accept semantic [x,y,yaw] or saved-map [x,y,z,yaw] poses."""
        target = np.asarray(target_pose, dtype=np.float64).reshape(-1)
        if len(target) == 3:
            return target[:2], float(target[2])
        if len(target) >= 4:
            return target[:2], float(target[3])
        raise ValueError("target pose must be [x,y,yaw] or [x,y,z,yaw]")

    def nearest_forward_index(
        self,
        target_pose: list[float] | tuple[float, ...] | np.ndarray,
        *,
        start_index: int = 0,
    ) -> tuple[int, float]:
        start = max(0, int(start_index))
        if start >= len(self.poses):
            raise ValueError("latent route start index is outside the saved map")
        target_xy, target_yaw = self._target_xy_yaw(target_pose)
        distances = np.linalg.norm(self.poses[start:, :2] - target_xy, axis=1)
        yaw_errors = np.abs(
            np.asarray(
                [wrap_angle(value - target_yaw) for value in self.poses[start:, 3]],
                dtype=np.float64,
            )
        )
        covered = (
            (distances <= self.max_target_error_m)
            & (yaw_errors <= self.max_target_yaw_error_rad)
        )
        if not np.any(covered):
            nearest = int(np.argmin(distances + 0.35 * yaw_errors))
            raise RuntimeError(
                "target pose is outside the recorded latent corridor: "
                f"nearest XY error={distances[nearest]:.2f} m, "
                f"yaw error={math.degrees(yaw_errors[nearest]):.1f} deg; "
                "record a denser mapping traversal before using this goal"
            )
        candidates = np.flatnonzero(covered)
        # Prefer the best pose within the first spatial visit to this target,
        # never a visually similar occurrence on a later return loop. A visit
        # is spatial rather than a fixed number of clips because Nav2 may spend
        # several seconds rotating in place before settling on the exact pose.
        spatial = distances <= self.max_target_error_m
        first = int(np.flatnonzero(spatial)[0])
        visit_end = len(distances)
        outside_run = 0
        leave_radius = self.max_target_error_m + 0.25
        for relative_index in range(first + 1, len(distances)):
            if distances[relative_index] > leave_radius:
                outside_run += 1
                if outside_run >= 3:
                    visit_end = relative_index - outside_run + 1
                    break
            else:
                outside_run = 0
        visit = candidates[(candidates >= first) & (candidates < visit_end)]
        if len(visit) == 0:
            # `covered` implies this should be unreachable, but retain a clear
            # failure if malformed thresholds are supplied by a caller.
            raise RuntimeError("recorded target visit has no heading-compatible pose")
        cost = distances[visit] + 0.35 * yaw_errors[visit]
        relative = int(visit[int(np.argmin(cost))])
        index = start + relative
        error = float(distances[relative])
        return index, error

    @staticmethod
    def _downsample(
        poses: np.ndarray,
        *,
        spacing_m: float,
        yaw_step_rad: float,
    ) -> np.ndarray:
        if len(poses) <= 2:
            return poses.copy()
        selected = [0]
        # Ignore camera-only / in-place yaw samples.  Retain a corner only
        # after real translation, based on the polyline geometry rather than
        # the recorded chassis yaw.  This prevents NavigateThroughPoses from
        # commanding reverse turns merely to revisit a mapping orientation.
        min_progress_m = min(0.35, float(spacing_m) * 0.35)
        for index in range(1, len(poses) - 1):
            pose = poses[index]
            delta = pose[:2] - poses[selected[-1], :2]
            translation = float(np.linalg.norm(delta))
            if translation < min_progress_m:
                continue

            next_delta = None
            for lookahead in range(index + 1, len(poses)):
                candidate = poses[lookahead, :2] - pose[:2]
                if float(np.linalg.norm(candidate)) >= min_progress_m:
                    next_delta = candidate
                    break

            corner = False
            if next_delta is not None:
                incoming = math.atan2(float(delta[1]), float(delta[0]))
                outgoing = math.atan2(
                    float(next_delta[1]), float(next_delta[0])
                )
                corner = abs(wrap_angle(outgoing - incoming)) >= yaw_step_rad

            if translation >= spacing_m or corner:
                selected.append(index)
        if selected[-1] != len(poses) - 1:
            selected.append(len(poses) - 1)
        return poses[np.asarray(selected, dtype=np.int64)].copy()

    def segment_to(
        self,
        target_pose: list[float] | tuple[float, ...] | np.ndarray,
        *,
        start_index: int = 0,
        # Aisle checkpoints constrain topology, not every metre of motion.
        # Wider spacing lets NavFn/MPPI follow one continuous curve instead of
        # repeatedly slowing for mapping samples that are not real stops.
        spacing_m: float = 2.50,
        yaw_step_rad: float = 0.65,
    ) -> LatentRouteSegment:
        end_index, error = self.nearest_forward_index(
            target_pose, start_index=start_index
        )
        raw = self.poses[int(start_index) : end_index + 1]
        route = self._downsample(
            raw,
            spacing_m=float(spacing_m),
            yaw_step_rad=float(yaw_step_rad),
        )
        return LatentRouteSegment(
            poses=route,
            start_index=int(start_index),
            end_index=end_index,
            target_error_m=error,
            map_dir=self.map_dir,
        )
