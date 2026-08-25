#!/usr/bin/env python3
"""Audit one recorded warehouse mission against goal.md acceptance evidence."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from trajectory_evaluation import (
    compare_tracking,
    prefix_through_corner,
    summarize_tracking,
)


REQUIRED_SCENES = {
    "normal_driving",
    "human_1_encounter",
    "human_2_encounter",
    "shelf_approach",
    "pick_up_operation",
    "return_path",
}
REQUIRED_STATES = {
    "NAVIGATE_TO_SHELF",
    "SHELF_APPROACH",
    "ALIGN_PACKAGE",
    "GRASP_PACKAGE",
    "VERIFY_GRASP",
    "RETURN_TO_DROPOFF",
    "PLACE_PACKAGE",
    "MISSION_COMPLETE",
}
REQUIRED_BEHAVIOR_CASES = {
    "human_leaves_path",
    "human_continues_crossing",
    "vehicle_can_safely_pass",
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def check(name: str, passed: bool, evidence: object) -> dict:
    return {"requirement": name, "passed": bool(passed), "evidence": evidence}


def load_xy(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.asarray(np.load(path), dtype=np.float64)[:, :2]
    rows = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    names = rows.dtype.names or ()
    for x_name, y_name in (
        ("x", "y"), ("gazebo_x", "gazebo_y"), ("vjepa_x", "vjepa_y")
    ):
        if x_name in names and y_name in names:
            return np.column_stack([rows[x_name], rows[y_name]]).astype(np.float64)
    raise ValueError(f"could not find XY columns in {path}")


def inference_latency(samples: list[dict]) -> dict[str, float | int]:
    values = np.asarray(
        [
            float(row["inference_ms"])
            for row in samples
            if row.get("inference_ms") is not None
            and math.isfinite(float(row["inference_ms"]))
        ],
        dtype=np.float64,
    )
    if len(values) == 0:
        return {"count": 0, "p50_ms": math.inf, "p95_ms": math.inf}
    return {
        "count": len(values),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent-run", type=Path, required=True)
    parser.add_argument("--baseline-latent-run", type=Path)
    parser.add_argument("--behavior-log", type=Path, required=True)
    parser.add_argument("--reference-route", type=Path)
    parser.add_argument("--baseline-trajectory", type=Path)
    parser.add_argument("--candidate-trajectory", type=Path)
    parser.add_argument("--corner-x", type=float, default=-1.90)
    parser.add_argument("--corner-y", type=float, default=-2.30)
    parser.add_argument("--corner-outgoing-yaw", type=float, default=3.141592653589793)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    samples = read_jsonl(args.latent_run / "samples.jsonl")
    metrics = read_jsonl(args.latent_run / "latent_metrics.jsonl")
    behavior = read_jsonl(args.behavior_log)
    scenes = {str(row.get("scene")) for row in samples}
    states = {str(row.get("mission_state")) for row in samples}
    horizons = {int(row["horizon"]) for row in metrics}
    cases = {
        path.stem
        for path in (args.latent_run / "behavior_visualizations").glob("*.png")
    }
    decisions = {str(row.get("decision")) for row in behavior}
    scenarios = {str(row.get("scenario")) for row in behavior}
    reasons_complete = all(
        str(row.get("reason", "")).strip() for row in behavior
    )
    # The worker cylinder radius plus robot footprint is larger than this
    # contact-only threshold. WAIT at normal safety distance is not collision.
    collision_samples = 0
    for row in behavior:
        occupancy = row.get("occupancy") or []
        if occupancy and float(occupancy[0].get("separation_m", 99.0)) < 0.30:
            collision_samples += 1

    checks = [
        check("all critical V-JEPA scenes logged", REQUIRED_SCENES <= scenes, sorted(scenes)),
        check("z(t+1..3) evaluated", horizons == {1, 2, 3}, sorted(horizons)),
        check("behavior prediction visualizations", REQUIRED_BEHAVIOR_CASES <= cases, sorted(cases)),
        check("WAIT and PASS exercised with reasons", {"WAIT", "PASS"} <= decisions and reasons_complete, sorted(decisions)),
        check(
            "both human scenarios exercised",
            {
                "human_1_static_until_close",
                "human_2_continuous_crossing",
            }
            <= scenarios,
            sorted(scenarios),
        ),
        check("mission reached shelf, grasped, returned, and placed", REQUIRED_STATES <= states, sorted(states)),
        check("no human collision samples", collision_samples == 0, collision_samples),
        check(
            "latent plots and summary generated",
            (args.latent_run / "latent_prediction_metrics.png").is_file()
            and (args.latent_run / "summary.json").is_file(),
            str(args.latent_run),
        ),
    ]

    if args.baseline_latent_run is not None:
        baseline_samples = read_jsonl(
            args.baseline_latent_run / "samples.jsonl"
        )
        baseline_latency = inference_latency(baseline_samples)
        candidate_latency = inference_latency(samples)
        latency_limit = float(baseline_latency["p95_ms"]) * 1.05
        checks.append(
            check(
                "V-JEPA inference speed not worse than baseline",
                candidate_latency["count"] > 0
                and baseline_latency["count"] > 0
                and float(candidate_latency["p95_ms"]) <= latency_limit,
                {
                    "baseline": baseline_latency,
                    "candidate": candidate_latency,
                    "p95_limit_ms": latency_limit,
                },
            )
        )
    else:
        checks.append(
            check(
                "V-JEPA inference speed not worse than baseline",
                False,
                "baseline latent run not supplied",
            )
        )

    trajectory_arguments = (
        args.reference_route,
        args.baseline_trajectory,
        args.candidate_trajectory,
    )
    if any(value is not None for value in trajectory_arguments):
        if not all(value is not None for value in trajectory_arguments):
            parser.error(
                "reference-route, baseline-trajectory and candidate-trajectory "
                "must be supplied together"
            )
        kwargs = {
            "apex_xy": (args.corner_x, args.corner_y),
            "outgoing_yaw_rad": args.corner_outgoing_yaw,
        }
        route = prefix_through_corner(load_xy(args.reference_route), **kwargs)
        baseline_points = prefix_through_corner(
            load_xy(args.baseline_trajectory), **kwargs
        )
        candidate_points = prefix_through_corner(
            load_xy(args.candidate_trajectory), **kwargs
        )
        baseline = summarize_tracking(baseline_points, route, **kwargs)
        candidate = summarize_tracking(candidate_points, route, **kwargs)
        comparison = compare_tracking(baseline, candidate)
        checks.extend(
            [
                check(
                    "tracking performance not worse than baseline",
                    bool(comparison["tracking_not_worse"]),
                    {
                        "scope": "shared outbound prefix through 2 m after Shelf A corner",
                        "baseline": baseline.as_dict(),
                        "candidate": candidate.as_dict(),
                    },
                ),
                check(
                    "second corner overshoot significantly reduced",
                    bool(comparison["corner_significantly_reduced"]),
                    comparison,
                ),
            ]
        )
    else:
        checks.extend(
            [
                check("tracking performance not worse than baseline", False, "trajectory evidence not supplied"),
                check("second corner overshoot significantly reduced", False, "trajectory evidence not supplied"),
            ]
        )

    report = {
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
