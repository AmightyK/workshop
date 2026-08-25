#!/usr/bin/env python3
"""Move LiDAR-visible warehouse workers toward random safe aisle waypoints."""

from __future__ import annotations

import argparse
import math
import random
import signal
import subprocess
import threading
import time
from dataclasses import dataclass

from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.pose_pb2 import Pose
from gz.msgs10.twist_pb2 import Twist
from gz.transport13 import Node


WORLD = "world_demo"
SET_POSES_SERVICE = f"/world/{WORLD}/set_pose_vector/blocking"
ENABLE_TOPIC = "/warehouse/random_people/enabled"
RESET_TOPIC = "/warehouse/random_people/reset"
UPDATE_PERIOD = 1.0 / 30.0
MAX_YAW_SPEED_RADPS = 2.4


@dataclass
class Walker:
    name: str
    waypoints: tuple[tuple[float, float], ...]
    speed: float
    x: float
    y: float
    yaw: float = 0.0
    target: tuple[float, float] | None = None
    wait_until: float = 0.0
    endpoint_wait_s: float = 0.0
    activation_distance_m: float | None = None
    activated: bool = True
    continuous: bool = False

    def choose_target(self, generator: random.Random) -> None:
        choices = [point for point in self.waypoints if point != self.target]
        self.target = generator.choice(choices)


class RandomPeopleController:
    def __init__(self, seed: int | None, speed_scale: float) -> None:
        self.node = Node()
        self.generator = random.Random(seed)
        self.enabled = True
        self.running = True
        self.last_enabled = True
        self.accept_pose_updates = True
        self.resume_poses: dict[str, tuple[float, float, float]] | None = None
        self.agv_xy: tuple[float, float] | None = None
        self.reset_requested = threading.Event()
        self.last_reset_request = -math.inf
        self.node.subscribe(Boolean, ENABLE_TOPIC, self.on_enable)
        self.node.subscribe(Boolean, RESET_TOPIC, self.on_reset)
        # Subscribe to the compact per-model pose topics.  The full world
        # Pose_V contains hundreds of links and can build a callback backlog
        # while V-JEPA, Nav2 and Gazebo GUI are all active.  A stale yaw leaves
        # a worker rotating forever even though fresh cmd_vel is published.
        self.node.subscribe(Pose, "/model/warehouse_agv/pose", self.on_agv_pose)
        self.walkers = [
            Walker(
                "random_worker_1",
                (
                    (-12.0, -10.4), (-10.0, -9.3), (-8.0, -10.5),
                    (-6.0, -9.2), (-4.0, -10.3),
                ),
                0.58 * speed_scale,
                -8.0,
                -10.0,
            ),
            Walker(
                "random_worker_2",
                (
                    (1.5, 3.25), (3.5, 4.35), (5.5, 3.25),
                    (7.5, 4.35), (9.5, 3.25),
                ),
                0.68 * speed_scale,
                7.0,
                3.8,
                yaw=math.pi,
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
                ((7.0, -18.0), (7.0, -2.0)),
                0.62 * speed_scale,
                7.0,
                -13.2,
                yaw=math.pi / 2.0,
                endpoint_wait_s=1.5,
                # Start the crossing while both actors are still far apart.
                # The worker visibly walks in from the aisle instead of
                # appearing to react only when the AGV reaches the camera.
                activation_distance_m=7.2,
                activated=False,
            ),
            Walker(
                "random_worker_5",
                # A long open-floor patrol with both turnarounds well clear of
                # the AGV lane and warehouse collision geometry.
                ((-4.8, -4.2), (1.0, -4.2)),
                0.56 * speed_scale,
                -4.8,
                -4.2,
                # Pause off the AGV lane before walking back. An immediate
                # reversal made the worker re-enter while the vehicle was
                # resuming from the first safe stop.
                endpoint_wait_s=4.0,
            ),
        ]
        self.walkers_by_name = {walker.name: walker for walker in self.walkers}
        self.initial_poses = {
            walker.name: (walker.x, walker.y, walker.yaw)
            for walker in self.walkers
        }
        self.velocity_publishers = {}
        for walker in self.walkers:
            if walker.endpoint_wait_s > 0.0:
                # Scripted crossing workers start with one outbound pass. Their
                # long routes keep them moving while providing a large safe
                # interval before they can revisit the AGV intersection.
                walker.target = walker.waypoints[-1]
            else:
                walker.choose_target(self.generator)
            self.node.subscribe(
                Pose,
                f"/model/{walker.name}/pose",
                lambda pose, name=walker.name: self.on_worker_pose(name, pose),
            )
            self.velocity_publishers[walker.name] = self.node.advertise(
                f"/model/{walker.name}/cmd_vel", Twist
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

    def on_agv_pose(self, pose: Pose) -> None:
        # The model pose topic also carries poses for every nested AGV link.
        # Only the model pose is expressed in the world frame.
        if pose.name != "warehouse_agv":
            return
        self.agv_xy = (float(pose.position.x), float(pose.position.y))

    def on_worker_pose(self, name: str, pose: Pose) -> None:
        if pose.name != name:
            return
        if not self.accept_pose_updates:
            return
        walker = self.walkers_by_name[name]
        walker.x = float(pose.position.x)
        walker.y = float(pose.position.y)
        orientation = pose.orientation
        walker.yaw = math.atan2(
            2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
            1.0 - 2.0 * (orientation.y**2 + orientation.z**2),
        )

    def wait_for_world(self) -> None:
        deadline = time.monotonic() + 8.0
        while self.running and not all(
            publisher.has_connections()
            for publisher in self.velocity_publishers.values()
        ):
            # Gazebo Transport connection accounting can remain stale even
            # after every cmd_vel subscriber is visible. Publishing is safe
            # and the late subscriber receives the next 30 Hz command, so do
            # not leave all workers frozen forever on this advisory gate.
            if time.monotonic() >= deadline:
                print(
                    "Worker cmd_vel connection wait expired; starting live motion",
                    flush=True,
                )
                return
            time.sleep(0.10)

    @staticmethod
    def shortest_angle(from_yaw: float, to_yaw: float) -> float:
        return (to_yaw - from_yaw + math.pi) % (2.0 * math.pi) - math.pi

    @staticmethod
    def set_poses_once(commands: list[tuple[str, float, float, float]]) -> bool:
        """Move workers only when deliberately parking/resuming map capture."""
        pose_fields = []
        for name, x, y, yaw in commands:
            pose_fields.append(
                "pose { "
                f'name: "{name}" '
                f"position {{ x: {x:.9f} y: {y:.9f} z: 0 }} "
                "orientation { "
                f"z: {math.sin(yaw / 2.0):.9f} "
                f"w: {math.cos(yaw / 2.0):.9f} "
                "} }"
            )
        result = subprocess.run(
            [
                "gz", "service", "-s", SET_POSES_SERVICE,
                "--reqtype", "gz.msgs.Pose_V",
                "--reptype", "gz.msgs.Boolean",
                "--timeout", "3000",
                "--req", " ".join(pose_fields),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        return result.returncode == 0 and "data: true" in result.stdout

    def publish_velocity(self, walker: Walker, linear: float, angular: float) -> None:
        command = Twist()
        command.linear.x = linear
        command.angular.z = angular
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
            walker.wait_until = 0.0
            walker.activated = walker.activation_distance_m is None
            if walker.endpoint_wait_s > 0.0:
                walker.target = walker.waypoints[-1]
            else:
                walker.choose_target(self.generator)
        self.resume_poses = None
        self.accept_pose_updates = True
        print("Random people reset for new mission", flush=True)

    def update(self, now: float) -> None:
        for walker in self.walkers:
            if not walker.activated:
                if self.agv_xy is None or walker.activation_distance_m is None:
                    self.publish_velocity(walker, 0.0, 0.0)
                    continue
                agv_distance = math.hypot(
                    walker.x - self.agv_xy[0], walker.y - self.agv_xy[1]
                )
                if agv_distance > walker.activation_distance_m:
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
            target_x, target_y = walker.target
            dx, dy = target_x - walker.x, target_y - walker.y
            distance = math.hypot(dx, dy)
            if distance < 0.10:
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
        # Initial poses come from the world file. Never choose or set a random
        # startup pose, which was the old one-frame teleport.
        self.stop_people()
        print(
            f"Random people active: {len(self.walkers)} workers, "
            f"control topics {ENABLE_TOPIC}, {RESET_TOPIC}",
            flush=True,
        )
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
            time.sleep(max(0.0, UPDATE_PERIOD - (time.monotonic() - started)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--speed-scale", type=float, default=1.0)
    args = parser.parse_args()
    controller = RandomPeopleController(args.seed, max(0.1, args.speed_scale))

    def stop(*_: object) -> None:
        controller.running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        controller.run()
    finally:
        controller.stop_people()


if __name__ == "__main__":
    main()
