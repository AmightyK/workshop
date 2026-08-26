#!/usr/bin/env python3
"""Asynchronously log V-JEPA mission latents and evaluate future rollouts."""

from __future__ import annotations

import argparse
import base64
import json
import math
import queue
import signal
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, String

from src.data.ros_image import image_message_to_rgb, message_timestamp_sec
from src.prediction.latent_predictor import (
    LatentEvaluation,
    LatentRollout,
    LatentRolloutPredictor,
    behavior_case,
    summarize_evaluations,
)
from src.utils.config import load_config


CRITICAL_SCENES = {
    "normal_driving",
    "human_1_encounter",
    "human_2_encounter",
    "shelf_approach",
    "pick_up_operation",
    "return_path",
}

MISSION_SCENES = {
    "NAVIGATE_TO_SHELF": "normal_driving",
    "SHELF_APPROACH": "shelf_approach",
    "ALIGN_PACKAGE": "pick_up_operation",
    "RAISE_LIFT": "pick_up_operation",
    "GRASP_PACKAGE": "pick_up_operation",
    "VERIFY_GRASP": "pick_up_operation",
    "RETURN_TO_DROPOFF": "return_path",
    "PLACE_PACKAGE": "return_path",
    "DROP": "return_path",
    "PARK": "return_path",
    "CHARGING_HOME": "return_path",
    "READY_FOR_NEXT_TASK": "return_path",
}


@dataclass(frozen=True)
class LogSample:
    timestamp: float
    frame_timestamp: float
    frame_time_error_s: float
    frame_rgb: np.ndarray
    latent: np.ndarray
    pose: tuple[float, float, float, float]
    scene: str
    mission_state: str
    behavior: dict[str, Any]
    inference_ms: float | None = None


def classify_scene(
    mission_state: str,
    behavior: dict[str, Any],
    pose: tuple[float, float, float, float] | None = None,
) -> str:
    scenario = str(behavior.get("scenario", ""))
    person = str(behavior.get("person_id", "none"))
    occupancy = [
        item for item in (behavior.get("occupancy") or [])
        if isinstance(item, dict)
    ]
    encounter = (
        str(behavior.get("decision", "PASS")) in {"WAIT", "REPLAN"}
        or behavior.get("behavior_case") is not None
        or any(
            bool(item.get("path_occupied"))
            or float(item.get("separation_m", math.inf)) <= 3.5
            for item in occupancy
        )
    )
    if encounter and (
        scenario == "human_1_static_until_close" or person == "random_worker_4"
    ):
        return "human_1_encounter"
    if encounter and (
        scenario == "human_2_continuous_crossing" or person == "random_worker_5"
    ):
        return "human_2_encounter"
    explicit = MISSION_SCENES.get(mission_state)
    if explicit is not None and explicit != "normal_driving":
        return explicit
    # A state publisher can exit between two V-JEPA inference strides. The
    # actual V-JEPA pose provides an observed fallback for shelf coverage,
    # avoiding an arbitrary state dwell solely to satisfy the logger.
    if pose is not None and mission_state in {
        "NAVIGATE_TO_SHELF",
        "SHELF_APPROACH",
        "PARSE_TASK",
    }:
        x, y = float(pose[0]), float(pose[1])
        if -7.3 <= x <= -3.0 and min(
            abs(y - shelf_y) for shelf_y in (-1.94, 1.06, 4.06)
        ) <= 1.5:
            return "shelf_approach"
    return explicit or "normal_driving"


def write_behavior_visualization(
    path: Path,
    frame_rgb: np.ndarray,
    predictions: list[LatentRollout],
    behavior: dict[str, Any],
    case_name: str,
) -> None:
    """Render current frame, three latent heatmaps, and planner decision."""
    canvas = np.full((620, 1280, 3), 24, dtype=np.uint8)
    current = cv2.resize(
        cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), (620, 430)
    )
    canvas[90:520, 20:640] = current
    cv2.putText(
        canvas, "CURRENT FRAME", (20, 65), cv2.FONT_HERSHEY_SIMPLEX,
        0.8, (255, 255, 255), 2,
    )
    by_horizon = {rollout.horizon: rollout for rollout in predictions}
    for row, horizon in enumerate((1, 2, 3)):
        rollout = by_horizon.get(horizon)
        if rollout is None:
            continue
        vector = rollout.predicted_latent[:256]
        padded = np.zeros(256, dtype=np.float32)
        padded[: len(vector)] = vector
        minimum, maximum = float(padded.min()), float(padded.max())
        scaled = np.zeros_like(padded, dtype=np.uint8)
        if maximum > minimum:
            scaled = np.asarray(
                255.0 * (padded - minimum) / (maximum - minimum),
                dtype=np.uint8,
            )
        heatmap = cv2.applyColorMap(scaled.reshape(16, 16), cv2.COLORMAP_VIRIDIS)
        heatmap = cv2.resize(heatmap, (580, 105), interpolation=cv2.INTER_NEAREST)
        top = 80 + row * 135
        canvas[top : top + 105, 675:1255] = heatmap
        cv2.putText(
            canvas, f"PREDICTED z(t+{horizon})", (675, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 240, 240), 2,
        )
    decision = str(behavior.get("decision", "PASS"))
    reason = str(behavior.get("reason", "no active person conflict"))[:90]
    cv2.putText(
        canvas, f"CASE: {case_name}", (20, 558), cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (0, 220, 255), 2,
    )
    cv2.putText(
        canvas, f"PLANNER: {decision} - {reason}", (20, 595),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError(f"could not write behavior visualization: {path}")


def write_metric_plot(path: Path, rows: list[LatentEvaluation]) -> None:
    """Write a dependency-free PNG plot for L1, cosine, and drift metrics."""
    canvas = np.full((780, 1280, 3), 250, dtype=np.uint8)
    metrics = (
        ("L1 LATENT ERROR", "l1_latent_error", False),
        ("COSINE SIMILARITY", "cosine_similarity", True),
        ("PREDICTION DRIFT ERROR", "prediction_drift_error", False),
    )
    colors = {1: (220, 80, 30), 2: (20, 150, 40), 3: (40, 60, 220)}
    for chart, (title, attribute, fixed_cosine) in enumerate(metrics):
        left, right = 100, 1230
        top, bottom = 55 + chart * 250, 245 + chart * 250
        cv2.rectangle(canvas, (left, top), (right, bottom), (80, 80, 80), 1)
        cv2.putText(
            canvas, title, (left, top - 15), cv2.FONT_HERSHEY_SIMPLEX,
            0.64, (25, 25, 25), 2,
        )
        values = [float(getattr(row, attribute)) for row in rows]
        low, high = ((-1.0, 1.0) if fixed_cosine else (0.0, max(values, default=1.0)))
        if math.isclose(low, high):
            high = low + 1.0
        for horizon in (1, 2, 3):
            group = [row for row in rows if row.horizon == horizon]
            points = []
            for index, row in enumerate(group):
                x = left + int((right - left) * index / max(1, len(group) - 1))
                value = float(getattr(row, attribute))
                y = bottom - int((bottom - top) * (value - low) / (high - low))
                points.append((x, max(top, min(bottom, y))))
            if len(points) >= 2:
                cv2.polylines(
                    canvas, [np.asarray(points, dtype=np.int32)], False,
                    colors[horizon], 2, cv2.LINE_AA,
                )
            cv2.putText(
                canvas, f"h={horizon}", (1010 + 70 * horizon, top + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, colors[horizon], 1,
            )
        cv2.putText(
            canvas, f"{low:.3f}", (15, bottom), cv2.FONT_HERSHEY_SIMPLEX,
            0.42, (50, 50, 50), 1,
        )
        cv2.putText(
            canvas, f"{high:.3f}", (15, top + 8), cv2.FONT_HERSHEY_SIMPLEX,
            0.42, (50, 50, 50), 1,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError(f"could not write latent metric plot: {path}")


class LatentLogWriter:
    def __init__(
        self,
        output_dir: Path,
        *,
        horizons: tuple[int, ...],
        velocity_alpha: float,
    ) -> None:
        self.output_dir = output_dir
        self.frame_dir = output_dir / "frames"
        self.latent_dir = output_dir / "latents"
        self.prediction_dir = output_dir / "predictions"
        self.visualization_dir = output_dir / "behavior_visualizations"
        for directory in (
            self.frame_dir, self.latent_dir, self.prediction_dir,
            self.visualization_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self.manifest_path = output_dir / "samples.jsonl"
        self.metrics_path = output_dir / "latent_metrics.jsonl"
        self.predictor = LatentRolloutPredictor(
            horizons=horizons, velocity_alpha=velocity_alpha
        )
        self.evaluations: list[LatentEvaluation] = []
        self.sample_count = 0
        self.visualized_cases: set[str] = set()

    @staticmethod
    def _append_json(path: Path, value: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, sort_keys=True) + "\n")

    def write(self, sample: LogSample) -> dict[str, Any]:
        index = self.sample_count
        frame_path = self.frame_dir / f"{index:06d}.jpg"
        latent_path = self.latent_dir / f"{index:06d}.npy"
        if not cv2.imwrite(
            str(frame_path), cv2.cvtColor(sample.frame_rgb, cv2.COLOR_RGB2BGR)
        ):
            raise RuntimeError(f"could not save raw frame: {frame_path}")
        np.save(latent_path, sample.latent.astype(np.float32))

        predictions, matured = self.predictor.observe(
            sample.latent, timestamp=sample.timestamp, scene=sample.scene
        )
        rollout_records = []
        for rollout in predictions:
            path = self.prediction_dir / (
                f"{index:06d}_z_t_plus_{rollout.horizon}.npy"
            )
            np.save(path, rollout.predicted_latent.astype(np.float32))
            rollout_records.append(
                {"horizon": rollout.horizon, "path": str(path.relative_to(self.output_dir))}
            )
        for evaluation in matured:
            self.evaluations.append(evaluation)
            self._append_json(self.metrics_path, evaluation.as_dict())

        record = {
            "sample_index": index,
            "timestamp": sample.timestamp,
            "frame_timestamp": sample.frame_timestamp,
            "frame_time_error_s": sample.frame_time_error_s,
            "vehicle_pose": list(sample.pose),
            "vehicle_pose_source": "/vjepa_localization/debug",
            "scene": sample.scene,
            "mission_state": sample.mission_state,
            "behavior_decision": sample.behavior,
            "inference_ms": sample.inference_ms,
            "raw_frame": str(frame_path.relative_to(self.output_dir)),
            "latent_vector": str(latent_path.relative_to(self.output_dir)),
            "latent_source": "/vjepa_latent (actual V-JEPA global embedding)",
            "rollouts": rollout_records,
        }
        self._append_json(self.manifest_path, record)

        case_name = sample.behavior.get("behavior_case")
        if case_name and case_name not in self.visualized_cases:
            write_behavior_visualization(
                self.visualization_dir / f"{case_name}.png",
                sample.frame_rgb,
                predictions,
                sample.behavior,
                str(case_name),
            )
            self.visualized_cases.add(str(case_name))
        self.sample_count += 1
        return {
            "timestamp": sample.timestamp,
            "scene": sample.scene,
            "source": "actual_vjepa_embedding_causal_temporal_dynamics",
            "model_contract": "V-JEPA embedding rollout; no RGB decoder",
            "horizons": [
                {
                    "step": rollout.horizon,
                    # Publish the actual predicted embedding so the live
                    # dashboard can project z(t+h) into the same PCA plane.
                    # This is predictor output, not a UI-generated point.
                    "predicted_latent_f16_b64": base64.b64encode(
                        rollout.predicted_latent.astype("<f2").tobytes()
                    ).decode("ascii"),
                    "latent_dimension": int(rollout.predicted_latent.size),
                    "latent_encoding": "float16_base64",
                    "predicted_drift_l2": float(
                        np.linalg.norm(
                            rollout.predicted_latent - rollout.origin_latent
                        )
                    ),
                }
                for rollout in predictions
            ],
            "matured_evaluations": [row.as_dict() for row in matured],
        }

    def close(self, dropped_samples: int) -> None:
        summary = summarize_evaluations(self.evaluations)
        summary.update(
            {
                "samples": self.sample_count,
                "dropped_samples": int(dropped_samples),
                "critical_scenes_observed": sorted(
                    {
                        row.scene for row in self.evaluations
                        if row.scene in CRITICAL_SCENES
                    }
                ),
                "behavior_cases_visualized": sorted(self.visualized_cases),
            }
        )
        (self.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_metric_plot(self.output_dir / "latent_prediction_metrics.png", self.evaluations)


class LatentPredictionMonitor(Node):
    """Pair existing V-JEPA outputs without entering the inference callback."""

    def __init__(self, config: dict[str, Any], output_dir: Path) -> None:
        super().__init__("vjepa_latent_prediction_monitor")
        self.output_dir = output_dir
        online = config.get("online", {})
        prediction = config.get("latent_prediction", {})
        horizons = tuple(int(value) for value in prediction.get("horizons", [1, 2, 3]))
        self.writer = LatentLogWriter(
            output_dir,
            horizons=horizons,
            velocity_alpha=float(prediction.get("velocity_alpha", 0.65)),
        )
        self.work_queue: queue.Queue[LogSample | None] = queue.Queue(
            maxsize=int(prediction.get("writer_queue_size", 8))
        )
        self.prediction_reports: queue.SimpleQueue[dict[str, Any]] = (
            queue.SimpleQueue()
        )
        self.worker = threading.Thread(target=self._writer_loop, daemon=True)
        self.worker.start()
        # Keep enough raw frames to recover the exact/nearest clip-center
        # image after GPU inference completes. Saving only the newest frame
        # mislabeled a future image as the latent's source observation.
        self.frames: deque[tuple[float, np.ndarray]] = deque(
            maxlen=max(256, int(prediction.get("frame_buffer_size", 4096)))
        )
        self.latest_latent: tuple[int, np.ndarray] | None = None
        self.latent_sequence = 0
        self.consumed_latent_sequence = -1
        self.mission_state = "NAVIGATE_TO_SHELF"
        self.behavior: dict[str, Any] = {
            "decision": "PASS", "reason": "normal driving",
            "person_id": "none", "scenario": "normal_driving",
        }
        self.previous_decisions: dict[str, str] = {}
        self.pending_behavior_case: str | None = None
        self.mission_events: list[dict[str, Any]] = []
        self.dropped_samples = 0
        self.closed = False
        self.prediction_publisher = self.create_publisher(
            String, "/vjepa/latent_prediction", 10
        )
        self.create_timer(0.05, self._publish_prediction_reports)
        self.create_subscription(
            Image,
            str(online.get("camera_topic", "/vjepa/camera/image_raw")),
            self._on_image,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Float32MultiArray,
            str(online.get("latent_topic", "/vjepa_latent")),
            self._on_latent,
            10,
        )
        self.create_subscription(
            String,
            str(online.get("debug_topic", "/vjepa_localization/debug")),
            self._on_debug,
            10,
        )
        self.create_subscription(
            String, "/warehouse/mission_state", self._on_mission_state, 10
        )
        self.create_subscription(
            String, "/warehouse/behavior_decision", self._on_behavior, 10
        )
        self.create_subscription(
            String,
            "/warehouse/behavior_observation",
            self._on_behavior_observation,
            10,
        )
        self.get_logger().info(
            f"Asynchronous latent rollout logger -> {output_dir} (horizons={horizons})"
        )

    def _on_image(self, message: Image) -> None:
        try:
            frame = image_message_to_rgb(message)
        except ValueError as error:
            self.get_logger().error(str(error))
            return
        timestamp = message_timestamp_sec(message)
        if self.frames and timestamp < self.frames[-1][0]:
            self.frames.clear()
        self.frames.append((timestamp, frame.copy()))

    def _on_latent(self, message: Float32MultiArray) -> None:
        latent = np.asarray(message.data, dtype=np.float32)
        if latent.size == 0 or not np.isfinite(latent).all():
            self.get_logger().warning("Ignoring empty/non-finite V-JEPA latent")
            return
        self.latent_sequence += 1
        self.latest_latent = (self.latent_sequence, latent)

    def _on_mission_state(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
            self.mission_state = str(payload.get("state", message.data))
            self.mission_events.append(payload)
        except json.JSONDecodeError:
            self.mission_state = message.data.strip() or self.mission_state
            self.mission_events.append(
                {"state": self.mission_state, "timestamp": None}
            )

    def _accept_behavior(self, payload: dict[str, Any]) -> None:
        person = str(payload.get("person_id", "none"))
        decision = str(payload.get("decision", "PASS"))
        previous = self.previous_decisions.get(person)
        occupancy = payload.get("occupancy") or []
        case_name = behavior_case(
            decision=decision,
            scenario=str(payload.get("scenario", "")),
            previous_decision=previous,
            path_occupied=any(
                bool(item.get("path_occupied"))
                for item in occupancy
                if isinstance(item, dict)
            ),
        )
        if case_name is not None:
            payload["behavior_case"] = case_name
            self.pending_behavior_case = case_name
        elif self.pending_behavior_case is not None:
            # Behavior arrives at 20 Hz while V-JEPA debug arrives at the
            # inference stride. Hold a transition label until one latent
            # sample has actually captured it.
            payload["behavior_case"] = self.pending_behavior_case
        self.previous_decisions[person] = decision
        self.behavior = payload

    def _on_behavior(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except json.JSONDecodeError:
            return
        self._accept_behavior(payload)

    def _on_behavior_observation(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except json.JSONDecodeError:
            return
        if payload.get("scenario") != "human_2_continuous_crossing":
            return
        occupancy = payload.get("occupancy") or []
        if not occupancy or not isinstance(occupancy[0], dict):
            return
        # Capture the individual prediction only during a real encounter.
        # The global decision callback still owns ordinary driving samples.
        if float(occupancy[0].get("separation_m", math.inf)) > 3.5:
            return
        self._accept_behavior(payload)

    def _on_debug(self, message: String) -> None:
        if not self.frames or self.latest_latent is None:
            return
        sequence, latent = self.latest_latent
        if sequence == self.consumed_latent_sequence:
            return
        try:
            debug = json.loads(message.data)
            timestamp = float(debug["timestamp"])
            pose_values = tuple(float(value) for value in debug["pose"][:4])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            self.get_logger().warning("Ignoring malformed V-JEPA debug sample")
            return
        if len(pose_values) != 4:
            return
        frame_timestamp, frame = min(
            self.frames, key=lambda item: abs(item[0] - timestamp)
        )
        behavior = dict(self.behavior)
        scene = classify_scene(self.mission_state, behavior, pose_values)
        sample = LogSample(
            timestamp=timestamp,
            frame_timestamp=frame_timestamp,
            frame_time_error_s=abs(frame_timestamp - timestamp),
            frame_rgb=frame.copy(),
            latent=latent.copy(),
            pose=pose_values,
            scene=scene,
            mission_state=self.mission_state,
            behavior=behavior,
            inference_ms=(
                float(debug["inference_ms"])
                if debug.get("inference_ms") is not None
                else None
            ),
        )
        try:
            self.work_queue.put_nowait(sample)
            self.consumed_latent_sequence = sequence
            if self.pending_behavior_case is not None:
                self.pending_behavior_case = None
                self.behavior.pop("behavior_case", None)
        except queue.Full:
            self.dropped_samples += 1

    def _writer_loop(self) -> None:
        while True:
            sample = self.work_queue.get()
            try:
                if sample is None:
                    return
                report = self.writer.write(sample)
                self.prediction_reports.put(report)
            except Exception as error:
                self.get_logger().error(f"Latent log write failed: {error}")
            finally:
                self.work_queue.task_done()

    def _publish_prediction_reports(self) -> None:
        """Publish async rollout evidence for the query-answer dashboard."""
        latest = None
        while True:
            try:
                latest = self.prediction_reports.get_nowait()
            except queue.Empty:
                break
        if latest is not None:
            self.prediction_publisher.publish(
                String(data=json.dumps(latest, sort_keys=True))
            )

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.work_queue.put(None)
        self.work_queue.join()
        self.worker.join(timeout=5.0)
        self.writer.close(self.dropped_samples)
        (self.output_dir / "mission_states.jsonl").write_text(
            "".join(
                json.dumps(event, sort_keys=True) + "\n"
                for event in self.mission_events
            ),
            encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/warehouse_experiment.yaml")
    parser.add_argument("--output", type=Path, default=None)
    args, ros_args = parser.parse_known_args()
    config = load_config(args.config)
    configured_root = Path(
        config.get("latent_prediction", {}).get(
            "output_root", "outputs/warehouse_latent_predictions"
        )
    )
    if not configured_root.is_absolute():
        configured_root = PROJECT_ROOT / configured_root
    run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or configured_root / run_name

    rclpy.init(args=ros_args)
    node = LatentPredictionMonitor(config, output)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
