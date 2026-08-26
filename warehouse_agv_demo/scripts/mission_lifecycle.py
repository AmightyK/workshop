"""Pure mission-lifecycle gates shared by the warehouse ROS entry points."""

from __future__ import annotations

import math
import json
import os
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any


RECYCLE_SEQUENCE = (
    "DROP",
    "PARK",
    "CHARGING_HOME",
    "READY_FOR_NEXT_TASK",
)


def append_mission_event(payload: dict[str, Any]) -> None:
    """Append one state transition for runtime validation across subprocesses."""
    log_dir = Path(os.environ.get("WAREHOUSE_LOG_DIR", "/tmp/warehouse_agv_demo"))
    log_dir.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, sort_keys=True) + "\n"
    with (log_dir / "mission_states.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(line)


def planar_pose_error(
    actual_xy: Sequence[float], target_pose: Sequence[float]
) -> float:
    """Return the observed planar distance to a configured station pose."""
    if len(actual_xy) < 2 or len(target_pose) < 2:
        raise ValueError("actual and target poses must contain x and y")
    values = [float(actual_xy[0]), float(actual_xy[1]), float(target_pose[0]), float(target_pose[1])]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("actual and target poses must be finite")
    return math.hypot(values[0] - values[2], values[1] - values[3])


def require_home_pose(
    actual_xy: Sequence[float],
    target_pose: Sequence[float],
    tolerance_m: float,
) -> float:
    """Fail closed unless Nav2/Gazebo feedback proves the AGV is at home."""
    tolerance = float(tolerance_m)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("home position tolerance must be positive and finite")
    error = planar_pose_error(actual_xy, target_pose)
    if error > tolerance:
        raise RuntimeError(
            f"Charging/home gate failed: pose error {error:.3f} m exceeds "
            f"{tolerance:.3f} m"
        )
    return error


def contains_ordered_sequence(
    observed: Iterable[str], required: Sequence[str] = RECYCLE_SEQUENCE
) -> bool:
    """Return true only when every required state occurs in the given order."""
    remaining = iter(str(value) for value in required)
    try:
        expected = next(remaining)
    except StopIteration:
        return True
    for state in observed:
        if str(state) != expected:
            continue
        try:
            expected = next(remaining)
        except StopIteration:
            return True
    return False
