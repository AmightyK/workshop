#!/usr/bin/env python3
"""Move LiDAR-visible warehouse workers toward random safe aisle waypoints."""

from __future__ import annotations

import argparse
import fcntl
import functools
import math
import os
import random
import signal
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.int64_pb2 import Int64
from gz.msgs10.pose_pb2 import Pose
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.msgs10.twist_pb2 import Twist
from gz.transport13 import Node


WORLD = "world_demo"
# Mission reset / map parking is a one-shot command. The blocking variant can
# time out under full GUI/V-JEPA load while its request remains queued inside
# Gazebo; repeated late requests then keep snapping workers back to spawn and
# make a healthy VelocityControl stream look frozen.
SET_POSES_SERVICE = f"/world/{WORLD}/set_pose_vector"
ENABLE_TOPIC = "/warehouse/random_people/enabled"
RESET_TOPIC = "/warehouse/random_people/reset"
NEW_MISSION_TOPIC = "/warehouse/random_people/new_mission"
UPDATE_PERIOD = 1.0 / 30.0
MAX_YAW_SPEED_RADPS = 2.4
HUMAN_1_ACTIVATION_DISTANCE_M = 3.2
HUMAN_1_INITIAL_XY = (7.0, -10.0)
# Preserve the latest GitHub spawn at (7, -10), but keep the scripted person
# on the short camera-visible crossing.  Each mission traverses this segment
# once and parks just beyond the AGV lane; the next mission reverses direction.
# The old north endpoint at y=-2 made one trigger send the person through most
# of the warehouse, which looked like a flying/runaway worker.
HUMAN_1_WAYPOINTS = ((7.0, -10.8), (7.0, -7.0))
HUMAN_1_REARM_EACH_MISSION = True
HUMAN_1_CONTINUOUS = True
# The old -4.80 endpoint can settle against collision geometry at startup even
# while VelocityControl receives non-zero commands. A 15 cm inward offset is
# the nearest verified free pose and keeps the same crossing timing.
HUMAN_2_WAYPOINTS = ((-4.65, -4.2), (5.2, -4.2))
HUMAN_2_CONTINUOUS = True
WORKER_4_MODES = ("continuous", "proximity")
CAMERA_HORIZONTAL_FOV_RAD = 1.22
WORKER_BODY_RADIUS_M = 0.23
AGV_CROSSING_PRIORITY_RADIUS_M = 2.4
WORKER_CROSSING_COMMIT_RADIUS_M = 1.6


def worker_in_forward_camera_view(
    agv_xy: tuple[float, float] | None,
    agv_yaw: float | None,
    worker_xy: tuple[float, float],
) -> bool:
    """Geometric gate matching the AGV's forward RGB-camera field of view."""
    if agv_xy is None or agv_yaw is None:
        return False
    dx = worker_xy[0] - agv_xy[0]
    dy = worker_xy[1] - agv_xy[1]
    distance = math.hypot(dx, dy)
    if distance <= 1.0e-6:
        return True
    forward = math.cos(agv_yaw) * dx + math.sin(agv_yaw) * dy
    if forward <= 0.0:
        return False
    bearing = RandomPeopleController.shortest_angle(
        agv_yaw, math.atan2(dy, dx)
    )
    body_half_angle = math.asin(min(0.95, WORKER_BODY_RADIUS_M / distance))
    return abs(bearing) <= 0.5 * CAMERA_HORIZONTAL_FOV_RAD + body_half_angle


def worker_yields_to_agv_at_crossing(
    walker: "Walker", agv_xy: tuple[float, float] | None
) -> bool:
    """Stop an approaching worker outside a crossing already owned by AGV.

    A worker already inside the commit radius keeps moving so the AGV's
    predictive stop can let it clear. This ownership rule prevents the mutual
    WAIT deadlock that occurs when both agents stop inside the intersection.
    """
    if (
        not walker.yield_to_agv_at_crossing
        or walker.crossing_xy is None
        or walker.target is None
        or agv_xy is None
    ):
        return False
    crossing_x, crossing_y = walker.crossing_xy
    if math.hypot(agv_xy[0] - crossing_x, agv_xy[1] - crossing_y) > (
        AGV_CROSSING_PRIORITY_RADIUS_M
    ):
        return False
    worker_to_crossing = (crossing_x - walker.x, crossing_y - walker.y)
    worker_distance = math.hypot(*worker_to_crossing)
    if worker_distance < WORKER_CROSSING_COMMIT_RADIUS_M:
        return False
    worker_to_target = (walker.target[0] - walker.x, walker.target[1] - walker.y)
    crossing_is_ahead = (
        worker_to_crossing[0] * worker_to_target[0]
        + worker_to_crossing[1] * worker_to_target[1]
    ) > 0.0
    return crossing_is_ahead


def enforce_continuous_patrol_bounds(walker: "Walker") -> bool:
    """Reverse a continuous two-point patrol at or beyond either endpoint."""
    if not walker.continuous or len(walker.waypoints) != 2:
        return False
    first, second = walker.waypoints
    segment_x = second[0] - first[0]
    segment_y = second[1] - first[1]
    length_sq = segment_x * segment_x + segment_y * segment_y
    if length_sq <= 1.0e-9:
        return False
    progress = (
        (walker.x - first[0]) * segment_x
        + (walker.y - first[1]) * segment_y
    ) / length_sq
    if progress >= 1.0 and walker.target != first:
        walker.target_origin = (walker.x, walker.y)
        walker.target = first
        walker.wait_until = 0.0
        return True
    if progress <= 0.0 and walker.target != second:
        walker.target_origin = (walker.x, walker.y)
        walker.target = second
        walker.wait_until = 0.0
        return True
    return False


def acquire_controller_lock(lock_path: Path | None = None) -> TextIO:
    """Allow only one velocity publisher for the shared worker models."""
    if lock_path is None:
        log_dir = Path(os.environ.get("WAREHOUSE_LOG_DIR", "/tmp/warehouse_agv_demo"))
        partition = os.environ.get("GZ_PARTITION", "warehouse_agv_demo")
        safe_partition = "".join(
            character if character.isalnum() or character in "_.-" else "_"
            for character in partition
        )
        lock_path = log_dir / f"random_people_{safe_partition}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    stream = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close()
        raise RuntimeError(
            "another random_people controller already owns the worker models"
        ) from None
    stream.seek(0)
    stream.truncate()
    stream.write(f"{os.getpid()}\n")
    stream.flush()
    return stream


@dataclass
class Walker:
    name: str
    waypoints: tuple[tuple[float, float], ...]
    speed: float
    x: float
    y: float
    yaw: float = 0.0
    target: tuple[float, float] | None = None
    target_origin: tuple[float, float] | None = None
    wait_until: float = 0.0
    endpoint_wait_s: float = 0.0
    activation_distance_m: float | None = None
    activated: bool = True
    continuous: bool = False
    last_pose_update: float = 0.0
    pose_update_count: int = 0
    last_command_moving: bool = False
    crossing_xy: tuple[float, float] | None = None
    agv_clearance_before_return_m: float | None = None
    rearm_each_mission: bool = False
    completed_mission_generation: int = 0
    yield_to_agv_at_crossing: bool = False

    def choose_target(
        self, generator: random.Random, *, prefer_forward: bool = False
    ) -> None:
        choices = [point for point in self.waypoints if point != self.target]
        if prefer_forward:
            forward = [
                point for point in choices
                if abs(
                    RandomPeopleController.shortest_angle(
                        self.yaw,
                        math.atan2(point[1] - self.y, point[0] - self.x),
                    )
                )
                <= math.pi / 3.0
                and math.hypot(point[0] - self.x, point[1] - self.y) >= 0.25
            ]
            if forward:
                choices = forward
        self.target_origin = (self.x, self.y)
        self.target = generator.choice(choices)

    def reached_target(self) -> bool:
        """Accept both a close hit and an endpoint crossed between callbacks.

        Under a heavily loaded Gazebo GUI, two pose callbacks can straddle a
        waypoint without ever landing inside a tiny distance tolerance.  The
        old controller then kept walking toward the warehouse wall forever.
        Projecting onto the commanded segment makes endpoint handling robust
        to skipped samples.
        """
        if self.target is None:
            return False
        target_x, target_y = self.target
        if math.hypot(target_x - self.x, target_y - self.y) <= 0.15:
            return True
        if self.target_origin is None:
            return False
        origin_x, origin_y = self.target_origin
        segment_x, segment_y = target_x - origin_x, target_y - origin_y
        segment_length_sq = segment_x * segment_x + segment_y * segment_y
        if segment_length_sq <= 1e-9:
            return True
        progress = (
            (self.x - origin_x) * segment_x
            + (self.y - origin_y) * segment_y
        ) / segment_length_sq
        return progress >= 1.0


def hold_return_until_agv_clears(
    walker: Walker, agv_xy: tuple[float, float] | None
) -> bool:
    """Keep a crossing worker at its safe endpoint while the AGV passes.

    This gate applies only after the worker has completed a crossing. It does
    not alter predictive WAIT/PASS on the outbound traversal; it prevents an
    immediate U-turn from sending the same worker back into an AGV that is
    still stopped at or traversing the shared intersection.
    """
    if (
        walker.crossing_xy is None
        or walker.agv_clearance_before_return_m is None
        or agv_xy is None
    ):
        return False
    return math.hypot(
        agv_xy[0] - walker.crossing_xy[0],
        agv_xy[1] - walker.crossing_xy[1],
    ) <= walker.agv_clearance_before_return_m


class RandomPeopleController:
    def __init__(
        self, seed: int | None, speed_scale: float, worker4_mode: str
    ) -> None:
        self.node = Node()
        self.generator = random.Random(seed)
        self.enabled = True
        self.running = True
        self.last_enabled = True
        self.accept_pose_updates = True
        self.resume_poses: dict[str, tuple[float, float, float]] | None = None
        self.agv_xy: tuple[float, float] | None = None
        self.agv_yaw: float | None = None
        self.reset_requested = threading.Event()
        self.last_reset_request = -math.inf
        self.feedback_bootstrap_deadline = -math.inf
        self.mission_generation = 0
        self.last_mission_id: int | None = None
        self.walkers = [
            Walker(
                "random_worker_1",
                (
                    # The shelf centered near x=-9.84 protrudes into the old
                    # -12..-10 patrol and trapped the collision cylinder at
                    # (-9.3,-9.4). Keep this random patrol on the clear side.
                    (-8.0, -10.5), (-6.0, -9.2), (-4.0, -10.3),
                ),
                0.58 * speed_scale,
                -8.0,
                -10.0,
            ),
            Walker(
                "random_worker_2",
                # Stay in the open horizontal aisle at y=3.8. A northbound
                # segment along x=7 clips the cabinet corner near (7.1, 4.9),
                # where the collision proxy can become physically wedged.
                ((7.0, 3.8), (10.0, 3.8)),
                0.68 * speed_scale,
                7.0,
                3.8,
                yaw=math.pi / 2.0,
                continuous=True,
            ),
            Walker(
                "random_worker_3",
                (
                    (-7.0, 18.0), (-3.0, 19.0), (1.0, 18.0),
                    (5.0, 19.0), (9.0, 18.0),
                ),
                0.54 * speed_scale,
                -4.0,
                17.0,
            ),
            # These two workers visibly cross the main AGV routes. Worker 4
            # crosses the dock-to-cabinet corridor; worker 5 crosses the
            # northbound A/B/C approach. Nav2 must react to both as live LiDAR
            # obstacles instead of relying only on static-map avoidance.
            Walker(
                "random_worker_4",
                # Original GitHub camera-visible outbound scene, reused in
                # reverse on alternating missions.
                HUMAN_1_WAYPOINTS,
                0.62 * speed_scale,
                *HUMAN_1_INITIAL_XY,
                yaw=math.pi / 2.0,
                endpoint_wait_s=1.5 if worker4_mode == "proximity" else 0.0,
                activation_distance_m=(
                    HUMAN_1_ACTIVATION_DISTANCE_M
                    if worker4_mode == "proximity"
                    else None
                ),
                activated=worker4_mode != "proximity",
                continuous=(
                    HUMAN_1_CONTINUOUS if worker4_mode == "continuous" else False
                ),
                crossing_xy=HUMAN_1_INITIAL_XY,
                rearm_each_mission=(
                    HUMAN_1_REARM_EACH_MISSION
                    if worker4_mode == "proximity"
                    else False
                ),
            ),
            Walker(
                "random_worker_5",
                # A long open-floor patrol with both turnarounds well clear of
                # the AGV lane and warehouse collision geometry.
                # The east endpoint is 4.5 m beyond the shared crossing. The
                # previous x=1.0 endpoint was still inside the predictive
                # corridor; holding there while the AGV passed made both
                # agents wait forever. Walk fully clear, then hold the return.
                HUMAN_2_WAYPOINTS,
                0.56 * speed_scale,
                *HUMAN_2_WAYPOINTS[0],
                # Human #2 continuously patrols left-right. It decelerates
                # only as required to reverse its physical heading; there is
                # no endpoint dwell or mission-time trigger.
                continuous=HUMAN_2_CONTINUOUS,
                crossing_xy=(-1.5, -4.2),
                agv_clearance_before_return_m=4.0,
                yield_to_agv_at_crossing=True,
            ),
        ]
        self.walkers_by_name = {walker.name: walker for walker in self.walkers}
        self.initial_poses = {
            walker.name: (walker.x, walker.y, walker.yaw)
            for walker in self.walkers
        }
        self.velocity_publishers = {}
        # gz-transport's Python subscription wrapper does not guarantee the
        # lifetime of an anonymous callback after subscribe() returns. Keep
        # each partially-bound callback alive for the controller lifetime;
        # otherwise cmd_vel continues to be computed from the spawn pose and a
        # worker can walk straight past its endpoint into the warehouse wall.
        self.pose_callbacks = {}
        for walker in self.walkers:
            if walker.endpoint_wait_s > 0.0 or walker.continuous:
                # Scripted crossing workers start with one outbound pass. Their
                # long routes keep them moving while providing a large safe
                # interval before they can revisit the AGV intersection.
                walker.target = walker.waypoints[-1]
                walker.target_origin = (walker.x, walker.y)
            else:
                # Spawn facing a useful outbound waypoint. A random target
                # behind the person forced several seconds of in-place turn
                # under low simulation RTF and looked like a frozen worker.
                walker.choose_target(self.generator, prefer_forward=True)
            callback = functools.partial(self.on_worker_pose, walker.name)
            self.pose_callbacks[walker.name] = callback
            self.velocity_publishers[walker.name] = self.node.advertise(
                f"/model/{walker.name}/cmd_vel", Twist
            )

    def subscribe_feedback(self) -> None:
        """Subscribe only after Gazebo has created the world publishers.

        PosePublisher emits an initial state when a subscriber connects and
        then publishes updates as the entity changes. Subscribing before the
        models existed missed that initial state; the feedback fail-safe then
        held cmd_vel at zero, so no later pose change could break the cycle.
        """
        self.node.subscribe(Boolean, ENABLE_TOPIC, self.on_enable)
        self.node.subscribe(Boolean, RESET_TOPIC, self.on_reset)
        self.node.subscribe(Int64, NEW_MISSION_TOPIC, self.on_new_mission)
        self.node.subscribe(Pose, "/model/warehouse_agv/pose", self.on_agv_pose)
        for walker in self.walkers:
            self.node.subscribe(
                Pose,
                f"/model/{walker.name}/pose",
                self.pose_callbacks[walker.name],
            )

    def on_enable(self, message: Boolean) -> None:
        self.enabled = message.data

    def on_reset(self, message: Boolean) -> None:
        now = time.monotonic()
        if message.data and now - self.last_reset_request >= 0.5:
            # Gazebo Transport callbacks may run outside the controller loop.
            # Defer pose and route mutation to that single owning thread.
            self.last_reset_request = now
            self.reset_requested.set()

    def on_new_mission(self, message: Int64) -> None:
        """Arm exactly once for a unique mission ID.

        Gazebo discovery can make the five reliability copies arrive more than
        a second apart under load, so a time debounce cannot identify them
        reliably.  All copies now carry the same nanosecond mission ID and are
        idempotent regardless of their arrival spacing.
        """
        mission_id = int(message.data)
        if mission_id == 0 or mission_id == self.last_mission_id:
            return
        self.last_mission_id = mission_id
        self.mission_generation += 1
        print(
            f"Worker crossing mission generation {self.mission_generation} armed "
            f"(id={mission_id})",
            flush=True,
        )

    def on_agv_pose(self, pose: Pose) -> None:
        # The model pose topic also carries poses for every nested AGV link.
        # Only the model pose is expressed in the world frame.
        if pose.name != "warehouse_agv":
            return
        self.agv_xy = (float(pose.position.x), float(pose.position.y))
        orientation = pose.orientation
        self.agv_yaw = math.atan2(
            2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
            1.0 - 2.0 * (orientation.y**2 + orientation.z**2),
        )

    def on_worker_pose(self, name: str, pose: Pose) -> None:
        if pose.name != name:
            return
        if not self.accept_pose_updates:
            return
        walker = self.walkers_by_name[name]
        walker.x = float(pose.position.x)
        walker.y = float(pose.position.y)
        walker.last_pose_update = time.monotonic()
        walker.pose_update_count += 1
        orientation = pose.orientation
        walker.yaw = math.atan2(
            2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
            1.0 - 2.0 * (orientation.y**2 + orientation.z**2),
        )

    def wait_for_world(self) -> None:
        last_report = -math.inf
        required_pose_topics = {
            f"/model/{walker.name}/pose" for walker in self.walkers
        }
        while self.running:
            services = set(self.node.service_list())
            topics = set(self.node.topic_list())
            if (
                SET_POSES_SERVICE in services
                and required_pose_topics.issubset(topics)
            ):
                return
            now = time.monotonic()
            if now - last_report >= 2.0:
                missing_topics = len(required_pose_topics - topics)
                print(
                    "Waiting for Gazebo worker feedback: "
                    f"pose service={SET_POSES_SERVICE in services}, "
                    f"missing pose topics={missing_topics}",
                    flush=True,
                )
                last_report = now
            time.sleep(0.10)

    @staticmethod
    def shortest_angle(from_yaw: float, to_yaw: float) -> float:
        return (to_yaw - from_yaw + math.pi) % (2.0 * math.pi) - math.pi

    def set_poses_once(self, commands: list[tuple[str, float, float, float]]) -> bool:
        """Apply a batch of worker poses through Gazebo Transport."""
        request = Pose_V()
        for name, x, y, yaw in commands:
            pose = request.pose.add()
            pose.name = name
            pose.position.x = float(x)
            pose.position.y = float(y)
            pose.position.z = 0.0
            pose.orientation.z = math.sin(yaw / 2.0)
            pose.orientation.w = math.cos(yaw / 2.0)
        result = self.node.request(
            SET_POSES_SERVICE, request, Pose_V, Boolean, 1000
        )
        return bool(result and result[0] and len(result) > 1 and result[1].data)

    def publish_velocity(self, walker: Walker, linear: float, angular: float) -> None:
        command = Twist()
        command.linear.x = linear
        command.angular.z = angular
        walker.last_command_moving = abs(linear) >= 1e-4 or abs(angular) >= 1e-4
        self.velocity_publishers[walker.name].publish(command)

    def stop_people(self) -> None:
        for walker in self.walkers:
            self.publish_velocity(walker, 0.0, 0.0)

    def park_people(self) -> None:
        self.accept_pose_updates = False
        self.resume_poses = {
            walker.name: (walker.x, walker.y, walker.yaw) for walker in self.walkers
        }
        self.stop_people()
        if not self.set_poses_once(
            [
                (walker.name, 30.0 + 2.0 * index, 30.0, walker.yaw)
                for index, walker in enumerate(self.walkers)
            ]
        ):
            print("WARNING: failed to park people for static mapping", flush=True)

    def restore_people(self) -> None:
        poses = self.resume_poses or {
            walker.name: (walker.x, walker.y, walker.yaw) for walker in self.walkers
        }
        if not self.set_poses_once(
            [(name, *pose) for name, pose in poses.items()]
        ):
            print("WARNING: failed to restore people after static mapping", flush=True)
        for name, (x, y, yaw) in poses.items():
            walker = self.walkers_by_name[name]
            walker.x, walker.y, walker.yaw = x, y, yaw
        self.resume_poses = None
        self.accept_pose_updates = True

    def reset_people(self) -> None:
        """Return every worker to a fresh route at the start of a mission."""
        self.accept_pose_updates = False
        self.stop_people()
        commands = [
            (name, x, y, yaw)
            for name, (x, y, yaw) in self.initial_poses.items()
        ]
        if not self.set_poses_once(commands):
            print("WARNING: failed to reset people for new mission", flush=True)
        for walker in self.walkers:
            walker.x, walker.y, walker.yaw = self.initial_poses[walker.name]
            walker.target = None
            walker.target_origin = None
            walker.wait_until = 0.0
            walker.activated = walker.activation_distance_m is None
            if walker.rearm_each_mission:
                walker.activated = False
                walker.completed_mission_generation = self.mission_generation
            if walker.endpoint_wait_s > 0.0 or walker.continuous:
                walker.target = walker.waypoints[-1]
                walker.target_origin = (walker.x, walker.y)
            else:
                walker.choose_target(self.generator, prefer_forward=True)
        self.resume_poses = None
        self.accept_pose_updates = True
        print("Random people reset for new mission", flush=True)

    def update(self, now: float) -> None:
        for walker in self.walkers:
            # A velocity controller without fresh position feedback is an
            # open-loop runaway. Stop that one worker until its per-model pose
            # topic resumes; other patrols remain independent.
            if (
                walker.last_pose_update <= 0.0
                and now > self.feedback_bootstrap_deadline
            ) or (
                walker.last_command_moving
                and walker.last_pose_update > 0.0
                and now - walker.last_pose_update > 1.0
            ):
                self.publish_velocity(walker, 0.0, 0.0)
                continue
            if not walker.activated:
                if (
                    walker.rearm_each_mission
                    and walker.completed_mission_generation
                    >= self.mission_generation
                ):
                    self.publish_velocity(walker, 0.0, 0.0)
                    continue
                if self.agv_xy is None or walker.activation_distance_m is None:
                    self.publish_velocity(walker, 0.0, 0.0)
                    continue
                agv_distance = math.hypot(
                    walker.x - self.agv_xy[0], walker.y - self.agv_xy[1]
                )
                if (
                    agv_distance > walker.activation_distance_m
                    or not worker_in_forward_camera_view(
                        self.agv_xy,
                        self.agv_yaw,
                        (walker.x, walker.y),
                    )
                ):
                    self.publish_velocity(walker, 0.0, 0.0)
                    continue
                walker.activated = True
                print(
                    f"{walker.name} activated at AGV distance "
                    f"{agv_distance:.2f} m",
                    flush=True,
                )
            if walker.target is None:
                walker.choose_target(self.generator)
            enforce_continuous_patrol_bounds(walker)
            target_x, target_y = walker.target
            dx, dy = target_x - walker.x, target_y - walker.y
            distance = math.hypot(dx, dy)
            if worker_yields_to_agv_at_crossing(walker, self.agv_xy):
                self.publish_velocity(walker, 0.0, 0.0)
                continue
            if walker.reached_target():
                if walker.rearm_each_mission:
                    walker.completed_mission_generation = self.mission_generation
                    walker.activated = False
                    walker.wait_until = 0.0
                    walker.choose_target(self.generator)
                    self.publish_velocity(walker, 0.0, 0.0)
                    print(
                        f"{walker.name} completed mission crossing; next direction armed "
                        "for a future mission",
                        flush=True,
                    )
                    continue
                if hold_return_until_agv_clears(walker, self.agv_xy):
                    walker.wait_until = 0.0
                    self.publish_velocity(walker, 0.0, 0.0)
                    continue
                if walker.continuous:
                    walker.choose_target(self.generator)
                    target_x, target_y = walker.target
                    dx, dy = target_x - walker.x, target_y - walker.y
                    distance = math.hypot(dx, dy)
                    walker.wait_until = 0.0
                else:
                    if walker.wait_until == 0.0:
                        dwell = (
                            walker.endpoint_wait_s
                            if walker.endpoint_wait_s > 0.0
                            else self.generator.uniform(0.4, 1.8)
                        )
                        walker.wait_until = now + dwell
                    if now < walker.wait_until:
                        self.publish_velocity(walker, 0.0, 0.0)
                        continue
                    walker.wait_until = 0.0
                    walker.choose_target(self.generator)
                    target_x, target_y = walker.target
                    dx, dy = target_x - walker.x, target_y - walker.y
                    distance = math.hypot(dx, dy)

            heading_error = self.shortest_angle(walker.yaw, math.atan2(dy, dx))
            angular = max(
                -MAX_YAW_SPEED_RADPS,
                min(MAX_YAW_SPEED_RADPS, 2.0 * heading_error),
            )
            # Rotate first for large heading changes, then smoothly accelerate
            # along the person's local forward axis.
            linear = walker.speed * max(0.0, math.cos(heading_error))
            if abs(heading_error) > math.pi / 3.0:
                linear = 0.0
            # Do not freeze the worker in front of a stopped AGV: that made
            # both agents wait forever. The worker keeps crossing while the
            # dedicated pose-aware gate stops the AGV, then releases it as
            # soon as this moving person clears the retained path.
            self.publish_velocity(walker, linear, angular)

    def run(self) -> None:
        self.wait_for_world()
        if not self.running:
            return
        self.subscribe_feedback()
        # Initial poses come from the world file. Never choose or set a random
        # startup pose, which was the old one-frame teleport.
        self.stop_people()
        # PosePublisher may publish only on state change. Give each worker one
        # bounded open-loop second from its known world spawn so moving models
        # generate their first feedback; after that the stale-pose fail-safe
        # applies normally. The intentional stationary worker remains at zero.
        self.feedback_bootstrap_deadline = time.monotonic() + 1.0
        print(
            f"Random people active: {len(self.walkers)} workers, "
            f"control topics {ENABLE_TOPIC}, {RESET_TOPIC}, {NEW_MISSION_TOPIC}",
            flush=True,
        )
        last_health_report = -math.inf
        while self.running:
            started = time.monotonic()
            if self.reset_requested.is_set():
                self.reset_requested.clear()
                self.reset_people()
            if self.enabled != self.last_enabled:
                if self.enabled:
                    self.restore_people()
                    print("Random people resumed", flush=True)
                else:
                    self.park_people()
                    print("Random people parked for static mapping", flush=True)
                self.last_enabled = self.enabled

            now = time.monotonic()
            if self.enabled:
                self.update(now)
            if now - last_health_report >= 5.0:
                states = ", ".join(
                    f"{walker.name[-1]}=({walker.x:.2f},{walker.y:.2f})"
                    f"#{walker.pose_update_count}"
                    for walker in self.walkers
                )
                print(f"Worker pose feedback: {states}", flush=True)
                last_health_report = now
            time.sleep(max(0.0, UPDATE_PERIOD - (time.monotonic() - started)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--speed-scale", type=float, default=1.0)
    parser.add_argument(
        "--worker-4-mode",
        choices=WORKER_4_MODES,
        default="proximity",
        help=(
            "intentional proximity scenario (default), where worker 4 waits "
            "until the AGV is within 3.2 m; continuous is available for "
            "motion-only diagnostics"
        ),
    )
    args = parser.parse_args()
    controller_lock = acquire_controller_lock()
    controller = RandomPeopleController(
        args.seed, max(0.1, args.speed_scale), args.worker_4_mode
    )

    def stop(*_: object) -> None:
        controller.running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        controller.run()
    finally:
        controller.stop_people()
        controller_lock.close()


if __name__ == "__main__":
    main()
