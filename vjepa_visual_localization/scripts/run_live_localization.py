#!/usr/bin/env python3
"""Publish camera-only V-JEPA pose estimates from a rolling ROS image clip."""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, String

from scripts._common import encoder_from_config
from src.data.ros_image import image_message_to_rgb, message_timestamp_sec
from src.localization.global_localizer import GlobalVisualLocalizer
from src.localization.image_motion import CameraMotionEstimator
from src.localization.temporal_tracker import TemporalPoseTracker
from src.mapping.map_database import VisualMap
from src.retrieval.global_retriever import GlobalRetriever
from src.utils.config import load_config, section


@dataclass(frozen=True)
class BufferedFrame:
    timestamp: float
    rgb: np.ndarray


def validate_map_video_profile(visual_map: VisualMap, video: dict) -> None:
    """Reject map/query clip profiles that produce incomparable embeddings."""
    map_config = visual_map.metadata.get("config", {})
    map_video = map_config.get("video", {}) if isinstance(map_config, dict) else {}
    if not isinstance(map_video, dict) or not map_video:
        return

    mismatches: list[str] = []
    map_duration = float(map_video.get("clip_duration_sec", 2.0))
    query_duration = float(video.get("clip_duration_sec", 2.0))
    if not math.isclose(map_duration, query_duration, abs_tol=1e-9):
        mismatches.append(
            f"clip_duration_sec map={map_duration:g} query={query_duration:g}"
        )
    map_frames = int(map_video.get("num_sampled_frames", 16))
    query_frames = int(video.get("num_sampled_frames", 16))
    if map_frames != query_frames:
        mismatches.append(
            f"num_sampled_frames map={map_frames} query={query_frames}"
        )
    map_sampling = str(map_video.get("sampling", "uniform"))
    query_sampling = str(video.get("sampling", "uniform"))
    if map_sampling != query_sampling:
        mismatches.append(f"sampling map={map_sampling} query={query_sampling}")
    if mismatches:
        details = "; ".join(mismatches)
        raise RuntimeError(
            "V-JEPA map/query video profiles differ (" + details + "). "
            "Rebuild the latent map with the active config before live localization."
        )


LOW_LATENCY_IMAGE_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)

DDS_RESULT_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)


class LiveLocalizationNode(Node):
    """Use camera pixels only; this node never subscribes to odometry or GT."""

    def __init__(
        self,
        *,
        localizer: GlobalVisualLocalizer,
        camera_topic: str,
        pose_topic: str,
        debug_topic: str,
        latent_topic: str,
        frame_id: str,
        clip_duration_sec: float,
        num_sampled_frames: int,
        stride_sec: float,
        position_variance_m2: float,
        yaw_variance_rad2: float,
        temporal_tracker: TemporalPoseTracker | None,
        camera_forward_expansion_threshold: float,
        camera_min_fit_inlier_ratio: float,
        camera_min_essential_inlier_ratio: float,
        camera_max_forward_translation_z: float,
    ) -> None:
        super().__init__("vjepa_visual_localizer")
        self.localizer = localizer
        self.compute_host = socket.gethostname()
        self.camera_topic = camera_topic
        self.frame_id = frame_id
        self.clip_duration_sec = clip_duration_sec
        self.num_sampled_frames = num_sampled_frames
        self.sample_rate_fps = num_sampled_frames / clip_duration_sec
        self.stride_sec = stride_sec
        self.position_variance_m2 = position_variance_m2
        self.yaw_variance_rad2 = yaw_variance_rad2
        self.temporal_tracker = temporal_tracker
        self.camera_motion = CameraMotionEstimator(
            forward_expansion_threshold=camera_forward_expansion_threshold,
            min_fit_inlier_ratio=camera_min_fit_inlier_ratio,
            min_essential_inlier_ratio=camera_min_essential_inlier_ratio,
            max_forward_translation_z=camera_max_forward_translation_z,
        )
        self.frames: deque[BufferedFrame] = deque(maxlen=max(256, num_sampled_frames * 8))
        self.frames_lock = threading.Lock()
        self.inference_lock = threading.Lock()
        self.last_inference_timestamp = -math.inf
        self.pose_topic = pose_topic
        self.debug_topic = debug_topic
        self.latent_topic = latent_topic
        self.pose_publisher = self.create_publisher(
            PoseWithCovarianceStamped, pose_topic, DDS_RESULT_QOS
        )
        self.debug_publisher = self.create_publisher(
            String, debug_topic, DDS_RESULT_QOS
        )
        self.latent_publisher = self.create_publisher(
            Float32MultiArray, latent_topic, DDS_RESULT_QOS
        )
        self.camera_callback_group = ReentrantCallbackGroup()
        # KEEP_LAST(1) prevents an old queue from accumulating if image decode
        # briefly outruns the second executor thread.
        self.create_subscription(
            Image,
            camera_topic,
            self.on_image,
            LOW_LATENCY_IMAGE_QOS,
            callback_group=self.camera_callback_group,
        )
        self.get_logger().info(
            f"Camera-only V-JEPA localization: {camera_topic} -> {pose_topic} | "
            f"clip={clip_duration_sec:.2f}s, samples={num_sampled_frames} "
            f"({self.sample_rate_fps:.1f} FPS), stride={stride_sec:.2f}s"
        )

    def _select_clip(self, end_timestamp: float) -> np.ndarray | None:
        start_timestamp = end_timestamp - self.clip_duration_sec
        while self.frames and self.frames[0].timestamp < start_timestamp - 0.5:
            self.frames.popleft()
        if not self.frames or self.frames[0].timestamp > start_timestamp:
            return None
        frame_list = list(self.frames)
        timestamps = np.asarray([frame.timestamp for frame in frame_list])
        targets = np.linspace(
            start_timestamp,
            end_timestamp,
            self.num_sampled_frames,
            endpoint=False,
        )
        indices = np.abs(timestamps[:, None] - targets[None, :]).argmin(axis=0)
        return np.stack([frame_list[int(index)].rgb for index in indices])

    @staticmethod
    def _set_stamp(message: PoseWithCovarianceStamped, timestamp: float) -> None:
        seconds = math.floor(timestamp)
        message.header.stamp.sec = int(seconds)
        message.header.stamp.nanosec = int(round((timestamp - seconds) * 1e9))

    def on_image(self, message: Image) -> None:
        timestamp = message_timestamp_sec(message)
        if timestamp <= 0.0:
            timestamp = self.get_clock().now().nanoseconds * 1e-9
        try:
            rgb = image_message_to_rgb(message)
        except ValueError as error:
            self.get_logger().error(str(error))
            return
        with self.frames_lock:
            if self.frames and timestamp < self.frames[-1].timestamp:
                self.get_logger().warning(
                    "Camera time moved backwards; clearing rolling clip"
                )
                # A simulation restart is rare, so wait for any old-clock GPU
                # query before resetting its temporal/image-motion state.
                with self.inference_lock:
                    self.frames.clear()
                    self.last_inference_timestamp = -math.inf
                    self.camera_motion.reset()
                    if self.temporal_tracker is not None:
                        self.temporal_tracker.reset()
            self.frames.append(BufferedFrame(timestamp, rgb))
            if timestamp - self.last_inference_timestamp < self.stride_sec:
                return
            clip = self._select_clip(timestamp)
            if clip is None:
                return
            # Only one GPU inference may run. Other executor threads continue
            # filling the rolling clip with every incoming camera frame.
            if not self.inference_lock.acquire(blocking=False):
                return
            self.last_inference_timestamp = timestamp
        try:
            inference_started = time.perf_counter()
            try:
                result = self.localizer.localize(clip)
            except Exception as error:  # keep ROS alive for transient model/input failures
                self.get_logger().error(f"V-JEPA inference failed: {error}")
                return
            inference_ms = (time.perf_counter() - inference_started) * 1000.0

            center_timestamp = timestamp - self.clip_duration_sec / 2.0
            camera_motion = self.camera_motion.measure(rgb)
            camera_moving = camera_motion.translating
            camera_progress_scale = 0.0
            if camera_moving:
                # Normalize robust global expansion against the recorded
                # warehouse route. Pixel change is intentionally excluded:
                # a walking person can change many pixels while the AGV is
                # stationary and must not advance the latent sequence.
                expansion_strength = (
                    max(0.0, camera_motion.forward_expansion) / 0.07
                )
                camera_progress_scale = min(
                    1.25,
                    max(0.35, expansion_strength),
                )
            if self.temporal_tracker is not None:
                tracked = self.temporal_tracker.update(
                    result.retrieval,
                    timestamp=center_timestamp,
                    camera_moving=camera_moving,
                    camera_progress_scale=camera_progress_scale,
                )
                prediction = tracked.prediction
                tracking_state = tracked.state
                motion_credit = self.temporal_tracker.motion_credit
            else:
                tracked = None
                prediction = result.prediction
                tracking_state = "DISABLED"
                motion_credit = None
        finally:
            self.inference_lock.release()
        x, y, z, yaw = (float(value) for value in prediction.pose)
        pose_message = PoseWithCovarianceStamped()
        pose_message.header.frame_id = self.frame_id
        self._set_stamp(pose_message, center_timestamp)
        pose_message.pose.pose.position.x = x
        pose_message.pose.pose.position.y = y
        pose_message.pose.pose.position.z = z
        pose_message.pose.pose.orientation.z = math.sin(yaw / 2.0)
        pose_message.pose.pose.orientation.w = math.cos(yaw / 2.0)
        covariance = [0.0] * 36
        covariance[0] = self.position_variance_m2
        covariance[7] = self.position_variance_m2
        covariance[14] = self.position_variance_m2
        covariance[35] = self.yaw_variance_rad2
        pose_message.pose.covariance = covariance
        self.pose_publisher.publish(pose_message)

        # Publish the actual normalized global V-JEPA representation used by
        # retrieval.  The dashboard projects this vector with the same PCA
        # basis as the saved latent map; it is not a synthetic animation.
        query_embedding = np.asarray(result.query_embedding, dtype=np.float32).reshape(-1)
        latent_message = Float32MultiArray()
        latent_message.layout.dim = [
            MultiArrayDimension(
                label="vjepa_global_embedding",
                size=int(query_embedding.size),
                stride=int(query_embedding.size),
            )
        ]
        latent_message.data = query_embedding.astype(float).tolist()
        self.latent_publisher.publish(latent_message)

        debug = {
            "timestamp": center_timestamp,
            "pose": [x, y, z, yaw],
            "source_id": prediction.source_id,
            "pose_method": prediction.method,
            "tracking_state": tracking_state,
            "camera_motion_score": camera_motion.forward_expansion,
            "camera_pixel_change": camera_motion.pixel_change,
            "camera_motion_inlier_ratio": camera_motion.fit_inlier_ratio,
            "camera_essential_inlier_ratio": camera_motion.essential_inlier_ratio,
            "camera_translation_z": camera_motion.translation_z,
            "camera_motion_features": camera_motion.feature_count,
            "camera_moving": camera_moving,
            "camera_progress_scale": camera_progress_scale,
            "top1_similarity": float(result.retrieval.scores[0]),
            "confidence_margin": result.confidence_margin,
            "candidate_ids": result.retrieval.ids[:10].tolist(),
            "candidate_scores": result.retrieval.scores[:10].astype(float).tolist(),
            "latent_dimension": int(query_embedding.size),
            "clip_duration_sec": self.clip_duration_sec,
            "num_sampled_frames": self.num_sampled_frames,
            "sample_rate_fps": self.sample_rate_fps,
            "clip_center_delay_sec": self.clip_duration_sec / 2.0,
            "compute_host": self.compute_host,
            "rmw_implementation": os.environ.get(
                "RMW_IMPLEMENTATION", "rmw_fastrtps_cpp"
            ),
            "ros_domain_id": os.environ.get("ROS_DOMAIN_ID", "0"),
            "image_topic": self.camera_topic,
            "pose_topic": self.pose_topic,
            "latent_topic": self.latent_topic,
            "debug_topic": self.debug_topic,
            "inference_ms": inference_ms,
        }
        if tracked is not None:
            debug.update(
                {
                    "raw_pose": tracked.raw_prediction.pose.astype(float).tolist(),
                    "raw_source_id": tracked.raw_prediction.source_id,
                    "raw_jump_m": tracked.raw_jump_m,
                    "accepted_step_m": tracked.accepted_step_m,
                    "translation_gate_m": tracked.translation_gate_m,
                    "yaw_gate_rad": tracked.yaw_gate_rad,
                    "selected_rank": tracked.selected_rank,
                    "rejected_streak": tracked.rejected_streak,
                    "motion_credit": motion_credit,
                }
            )
        debug_message = String()
        debug_message.data = json.dumps(debug, sort_keys=True)
        self.debug_publisher.publish(debug_message)
        if tracked is not None and tracked.state == "HOLDING":
            self.get_logger().warning(
                f"[HOLDING] pose=({x:.2f}, {y:.2f}) | rejected raw jump="
                f"{tracked.raw_jump_m:.2f} m > gate={tracked.translation_gate_m:.2f} m | "
                f"streak={tracked.rejected_streak}"
            )
        elif (
            tracked is not None
            and tracked.state == "INITIALIZED"
            and tracked.selected_rank not in (None, 0)
        ):
            self.get_logger().warning(
                f"[INITIALIZED] route-start prior rejected ambiguous global "
                f"top1={tracked.raw_prediction.source_id} "
                f"score={tracked.raw_prediction.score:.3f}; selected "
                f"{prediction.source_id} rank={tracked.selected_rank} "
                f"pose=({x:.2f}, {y:.2f}, yaw={yaw:.2f})"
            )
        else:
            rank = tracked.selected_rank if tracked is not None else 0
            self.get_logger().info(
                f"[{tracking_state}] pose=({x:.2f}, {y:.2f}, yaw={yaw:.2f}) | "
                f"match={prediction.source_id} rank={rank} score={prediction.score:.3f} | "
                f"camera={'FORWARD' if camera_moving else 'TURN/STILL'} "
                f"progress={camera_progress_scale:.2f} "
                f"(expansion={camera_motion.forward_expansion:.3f}, "
                f"affine={camera_motion.fit_inlier_ratio:.2f}, "
                f"essential={camera_motion.essential_inlier_ratio:.2f}, "
                f"tz={camera_motion.translation_z:+.2f}, "
                f"pixels={camera_motion.pixel_change:.3f})"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--map", default="outputs/map")
    parser.add_argument("--camera-topic", default=None)
    parser.add_argument(
        "--no-temporal",
        action="store_true",
        help="disable prior-pose tracking and use independent frame retrieval",
    )
    args, ros_args = parser.parse_known_args()
    config = load_config(args.config)
    video = section(config, "video")
    retrieval = section(config, "retrieval")
    estimator = config.get("pose_estimator", {})
    online = config.get("online", {})
    temporal = config.get("temporal_tracking", {})
    visual_map = VisualMap.load(args.map)
    validate_map_video_profile(visual_map, video)
    candidate_pool = int(temporal.get("candidate_pool", 0))
    if candidate_pool <= 0:
        candidate_pool = len(visual_map.global_embeddings)
    localizer = GlobalVisualLocalizer(
        encoder_from_config(config),
        GlobalRetriever(visual_map),
        top_k=max(int(retrieval.get("global_top_k", 20)), candidate_pool),
        pose_method=str(estimator.get("method", "top1")),
        weighted_alpha=float(estimator.get("weighted_alpha", 10.0)),
        weighted_radius_m=float(estimator.get("weighted_radius_m", 2.0)),
    )
    temporal_tracker = None
    if bool(temporal.get("enabled", True)) and not args.no_temporal:
        temporal_tracker = TemporalPoseTracker(
            max_linear_speed_mps=float(temporal.get("max_linear_speed_mps", 1.25)),
            max_angular_speed_radps=float(temporal.get("max_angular_speed_radps", 2.4)),
            base_translation_gate_m=float(temporal.get("base_translation_gate_m", 0.65)),
            base_yaw_gate_rad=float(temporal.get("base_yaw_gate_rad", 0.55)),
            max_translation_gate_m=float(temporal.get("max_translation_gate_m", 2.2)),
            max_yaw_gate_rad=float(temporal.get("max_yaw_gate_rad", 2.8)),
            distance_penalty=float(temporal.get("distance_penalty", 0.04)),
            yaw_penalty=float(temporal.get("yaw_penalty", 0.015)),
            smoothing_alpha=float(temporal.get("smoothing_alpha", 0.82)),
            max_index_advance_per_sec=float(
                temporal.get("max_index_advance_per_sec", 4.0)
            ),
            forward_progress_bonus=float(
                temporal.get("forward_progress_bonus", 0.055)
            ),
            stationary_translation_gate_m=float(
                temporal.get("stationary_translation_gate_m", 0.18)
            ),
            initial_route_index=(
                int(temporal["initial_route_index"])
                if temporal.get("initial_route_index") is not None
                else None
            ),
            initial_route_window=int(temporal.get("initial_route_window", 0)),
            relocalization_frames=int(temporal.get("relocalization_frames", 4)),
            relocalization_cluster_radius_m=float(
                temporal.get("relocalization_cluster_radius_m", 1.2)
            ),
            relocalization_min_similarity=float(
                temporal.get("relocalization_min_similarity", 0.90)
            ),
            relocalization_max_jump_m=float(
                temporal.get("relocalization_max_jump_m", 4.0)
            ),
        )
    rclpy.init(args=ros_args)
    node = LiveLocalizationNode(
        localizer=localizer,
        camera_topic=args.camera_topic or str(online.get("camera_topic", "/camera")),
        pose_topic=str(online.get("pose_topic", "/vjepa_pose")),
        debug_topic=str(online.get("debug_topic", "/vjepa_localization/debug")),
        latent_topic=str(online.get("latent_topic", "/vjepa_latent")),
        frame_id=str(online.get("frame_id", "map")),
        clip_duration_sec=float(video.get("clip_duration_sec", 2.0)),
        num_sampled_frames=int(video.get("num_sampled_frames", 16)),
        stride_sec=float(video.get("stride_sec", 0.5)),
        position_variance_m2=float(online.get("position_variance_m2", 0.25)),
        yaw_variance_rad2=float(online.get("yaw_variance_rad2", 0.12)),
        temporal_tracker=temporal_tracker,
        camera_forward_expansion_threshold=float(
            temporal.get("camera_forward_expansion_threshold", 0.035)
        ),
        camera_min_fit_inlier_ratio=float(
            temporal.get("camera_min_fit_inlier_ratio", 0.14)
        ),
        camera_min_essential_inlier_ratio=float(
            temporal.get("camera_min_essential_inlier_ratio", 0.35)
        ),
        camera_max_forward_translation_z=float(
            temporal.get("camera_max_forward_translation_z", -0.75)
        ),
    )
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.remove_node(node)
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
