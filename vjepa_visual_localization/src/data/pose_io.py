"""Ground-truth pose loading and timestamp synchronization."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Pose:
    """A world-frame pose associated with a video timestamp."""

    timestamp: float
    x: float
    y: float
    z: float
    yaw: float

    def as_array(self) -> np.ndarray:
        """Return ``[x, y, z, yaw]`` as float64."""

        return np.asarray([self.x, self.y, self.z, self.yaw], dtype=np.float64)


@dataclass(frozen=True)
class PoseSample:
    """A synchronized pose and distance to its nearest measured timestamp."""

    pose: Pose
    source_time_error: float


def wrap_angle(angle: float) -> float:
    """Wrap an angle to ``[-pi, pi)``."""

    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class PoseSeries:
    """A sorted ground-truth trajectory with interpolation support."""

    def __init__(self, poses: list[Pose]) -> None:
        if not poses:
            raise ValueError("pose series is empty")
        ordered = sorted(poses, key=lambda pose: pose.timestamp)
        timestamps = np.asarray([pose.timestamp for pose in ordered], dtype=np.float64)
        if np.any(np.diff(timestamps) <= 0.0):
            raise ValueError("pose timestamps must be unique and strictly increasing")
        self._poses = tuple(ordered)
        self.timestamps = timestamps
        self.values = np.stack([pose.as_array() for pose in ordered])

    @classmethod
    def from_csv(cls, path: str | Path) -> "PoseSeries":
        """Load ``timestamp,x,y,z,yaw`` columns from a CSV file."""

        path = Path(path)
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            required = {"timestamp", "x", "y", "z", "yaw"}
            missing = required.difference(reader.fieldnames or ())
            if missing:
                raise ValueError(f"{path} is missing columns: {sorted(missing)}")
            poses = [
                Pose(
                    timestamp=float(row["timestamp"]),
                    x=float(row["x"]),
                    y=float(row["y"]),
                    z=float(row["z"]),
                    yaw=wrap_angle(float(row["yaw"])),
                )
                for row in reader
            ]
        return cls(poses)

    @property
    def start_time(self) -> float:
        return float(self.timestamps[0])

    @property
    def end_time(self) -> float:
        return float(self.timestamps[-1])

    def sample(
        self,
        timestamp: float,
        *,
        method: str = "interpolate",
        tolerance_sec: float = 0.2,
    ) -> PoseSample:
        """Synchronize a pose to ``timestamp``.

        ``source_time_error`` is the distance to the nearest actual pose row.
        It must be within ``tolerance_sec`` even when interpolation is used, so
        silent synchronization over large telemetry gaps is rejected.
        """

        if tolerance_sec < 0.0:
            raise ValueError("tolerance_sec must be non-negative")
        index = int(np.searchsorted(self.timestamps, timestamp, side="left"))
        neighbors = [candidate for candidate in (index - 1, index) if 0 <= candidate < len(self.timestamps)]
        nearest = min(neighbors, key=lambda item: abs(self.timestamps[item] - timestamp))
        source_error = abs(float(self.timestamps[nearest]) - timestamp)
        if source_error > tolerance_sec:
            raise ValueError(
                f"no pose within {tolerance_sec:.3f}s of video timestamp "
                f"{timestamp:.6f}; nearest error is {source_error:.6f}s"
            )

        if method == "nearest" or timestamp <= self.start_time or timestamp >= self.end_time:
            source = self._poses[nearest]
            return PoseSample(
                Pose(timestamp, source.x, source.y, source.z, source.yaw),
                source_error,
            )
        if method != "interpolate":
            raise ValueError(f"unsupported pose synchronization method: {method}")

        left = index - 1
        right = index
        t0, t1 = self.timestamps[left], self.timestamps[right]
        ratio = float((timestamp - t0) / (t1 - t0))
        p0, p1 = self.values[left], self.values[right]
        xyz = p0[:3] + ratio * (p1[:3] - p0[:3])
        yaw_delta = wrap_angle(float(p1[3] - p0[3]))
        yaw = wrap_angle(float(p0[3]) + ratio * yaw_delta)
        return PoseSample(
            Pose(timestamp, float(xyz[0]), float(xyz[1]), float(xyz[2]), yaw),
            source_error,
        )
