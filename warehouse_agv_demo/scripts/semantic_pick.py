#!/usr/bin/env python3
"""Compatibility entry point for the physical, camera-gated pickup pipeline.

The former implementation moved payloads through Gazebo's SetPose service.
Keeping this redirect means old launch habits now use the same no-teleport
mission as ``run_vqa_mission.sh --pick-only``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    mission = Path(__file__).with_name("vqa_mission.py")
    os.execv(
        sys.executable,
        [
            sys.executable,
            "-u",
            str(mission),
            "--pick-only",
            "--command",
            "Bring the blue box from Storage A to Packing Station",
            *sys.argv[1:],
        ],
    )


if __name__ == "__main__":
    main()
