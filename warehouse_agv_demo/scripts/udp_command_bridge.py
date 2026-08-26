#!/usr/bin/env python3
"""Receive AIWaiter UDP actions and drive the local AGV through keyboard control.

The other machine (``AI_asr/src/robot_link``) sends one JSON datagram per interpreted turn:

    {"v": 1, "kind": "control", "action": {"type": "control", "verb": "STOP"},
     "session": "...", "seq": 4}

The network boundary is UDP only. ROS is used strictly on this machine, after the datagram has
been decoded, and all velocity commands go through ``/cmd_vel_keyboard``. Navigation is limited to
    the current demo task (Storage A pickup); the mission runner is an internal executor, never
    something the voice device needs to know or invoke.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import signal
import socket
import subprocess
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LOG = logging.getLogger("warehouse_udp_bridge")
WIRE_VERSION = 1
MAX_DATAGRAM = 1200
DEFAULT_BIND = "0.0.0.0:45455"
DEFAULT_TURN_DEG = 90.0
FORWARD_SPEED_MPS = 0.35
BACKWARD_SPEED_MPS = -0.22
TURN_MAX_RAD_S = 0.65
TURN_MIN_RAD_S = 0.20
TURN_TOLERANCE_RAD = math.radians(2.0)
CONTROL_VERBS = {"stop", "forward", "left", "right", "backward", "resume"}


def decode_packet(raw: bytes) -> dict[str, Any] | None:
    """Decode AIWaiter's v1 packet and normalize action verbs to lowercase."""
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("v") != WIRE_VERSION:
        return None
    kind = str(payload.get("kind", "")).lower()
    if kind not in {"control", "navigate", "ping", "ack"}:
        return None
    action = payload.get("action")
    if isinstance(action, str):
        action = {"type": "control", "verb": action}
    if isinstance(action, dict):
        action = dict(action)
        action_type = str(action.get("type", "")).lower()
        action["type"] = action_type
        if action_type == "control":
            action["verb"] = str(
                action.get("verb", action.get("action", ""))
            ).lower()
            angle = action.get(
                "angle_deg",
                action.get("angle_degrees", action.get("angle")),
            )
            if angle is not None:
                try:
                    action["angle_deg"] = min(360.0, max(1.0, float(angle)))
                except (TypeError, ValueError):
                    action.pop("angle_deg", None)
    payload["kind"] = kind
    payload["action"] = action
    return payload


class Deduper:
    """Bounded duplicate filter for sender repeats and retransmitted datagrams."""

    def __init__(self, window: int = 512) -> None:
        self._seen: set[tuple[str, int]] = set()
        self._order: deque[tuple[str, int]] = deque(maxlen=window)

    def duplicate(self, packet: dict[str, Any]) -> bool:
        session = str(packet.get("session", ""))
        try:
            sequence = int(packet.get("seq", 0))
        except (TypeError, ValueError):
            sequence = 0
        key = (session, sequence)
        if key in self._seen:
            return True
        if len(self._order) == self._order.maxlen:
            self._seen.discard(self._order[0])
        self._order.append(key)
        self._seen.add(key)
        return False


def ack_packet(packet: dict[str, Any]) -> bytes:
    """Build the ACK shape expected by AIWaiter/src/robot_link/sender.py."""
    try:
        sequence = int(packet.get("seq", 0) or 0)
    except (TypeError, ValueError):
        sequence = 0
    return json.dumps(
        {
            "kind": "ack",
            "session": str(packet.get("session", "")),
            "seq": sequence,
            "robot_id": str(packet.get("robot_id", "robo-1")),
            "v": WIRE_VERSION,
        },
        separators=(",", ":"),
    ).encode("utf-8")


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class TurnState:
    direction: int
    target_rad: float
    start_yaw: float | None
    last_yaw: float | None
    travelled_rad: float = 0.0
    deadline: float | None = None


class MissionProcess:
    """Own the one allowed automatic Storage-A pickup process."""

    def __init__(self, demo_dir: Path, *, dry_run: bool = False) -> None:
        self.demo_dir = demo_dir
        self.dry_run = dry_run
        self.process: subprocess.Popen | None = None
        self.paused = False

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start_pick_a(self, color: str = "blue", *, deliver: bool = True) -> None:
        color = str(color or "blue").lower()
        if color not in {"blue", "red", "green"}:
            raise ValueError("Storage A chỉ có màu blue, red hoặc green")
        self.cancel()
        command = [
            # run_demo.sh already owns Gazebo, bridge, V-JEPA and Nav2. Skip
            # pick_box.sh's repeated dependency startup checks on each command.
            str(self.demo_dir / "run_storage_pick.sh"),
            "--storage",
            "A",
            "--color",
            color,
        ]
        command.append("--deliver" if deliver else "--pick-only")
        if self.dry_run:
            LOG.info("[dry-run] %s", " ".join(command))
            return
        self.process = subprocess.Popen(
            command,
            cwd=str(self.demo_dir),
            start_new_session=True,
        )
        self.paused = False
        LOG.info("PICK A/%s started pid=%d", color, self.process.pid)

    def pause(self) -> bool:
        if not self.running or self.paused:
            return False
        assert self.process is not None
        try:
            os.killpg(self.process.pid, signal.SIGSTOP)
        except ProcessLookupError:
            self.process = None
            return False
        self.paused = True
        LOG.info("mission paused pid=%d", self.process.pid)
        return True

    def resume(self) -> bool:
        if not self.running or not self.paused:
            return False
        assert self.process is not None
        try:
            os.killpg(self.process.pid, signal.SIGCONT)
        except ProcessLookupError:
            self.process = None
            self.paused = False
            return False
        self.paused = False
        LOG.info("mission resumed pid=%d", self.process.pid)
        return True

    def cancel(self) -> bool:
        process = self.process
        if process is None or process.poll() is not None:
            self.process = None
            self.paused = False
            return False
        try:
            if self.paused:
                os.killpg(process.pid, signal.SIGCONT)
            os.killpg(process.pid, signal.SIGINT)
            process.wait(timeout=5.0)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        self.process = None
        self.paused = False
        LOG.info("mission cancelled")
        return True


class KeyboardUDPBridge:
    """ROS node + non-blocking UDP receiver for the local keyboard command path."""

    def __init__(self, bind: tuple[str, int], demo_dir: Path, *, dry_run: bool = False) -> None:
        import rclpy
        from geometry_msgs.msg import Twist
        from nav_msgs.msg import Odometry
        from rclpy.node import Node

        self._rclpy = rclpy
        self._Twist = Twist
        self.node = Node("aiwaiter_udp_keyboard_bridge")
        self.publisher = self.node.create_publisher(Twist, "/cmd_vel_keyboard", 10)
        self.node.create_subscription(Odometry, "/odom", self._on_odom, 20)
        self.node.create_timer(0.02, self._tick)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(bind)
        self.socket.setblocking(False)
        self.deduper = Deduper()
        self.mission = MissionProcess(demo_dir, dry_run=dry_run)
        self.linear = 0.0
        self.angular = 0.0
        self.override = False
        self.turn: TurnState | None = None
        self.odom_yaw: float | None = None
        LOG.info("UDP bridge listening on udp://%s:%d", bind[0], bind[1])

    def _on_odom(self, message) -> None:
        q = message.pose.pose.orientation
        sin_yaw = 2.0 * (q.w * q.z + q.x * q.y)
        cos_yaw = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.odom_yaw = math.atan2(sin_yaw, cos_yaw)

    def _send_velocity(self) -> None:
        if not self.override:
            return
        command = self._Twist()
        command.linear.x = float(self.linear)
        command.angular.z = float(self.angular)
        self.publisher.publish(command)

    def _tick(self) -> None:
        self._receive_all()
        self._advance_turn()
        self._send_velocity()

    def _receive_all(self) -> None:
        while True:
            try:
                raw, address = self.socket.recvfrom(MAX_DATAGRAM + 256)
            except BlockingIOError:
                return
            packet = decode_packet(raw)
            if packet is None:
                continue
            try:
                self.socket.sendto(ack_packet(packet), address)
            except OSError:
                pass
            if packet["kind"] in {"ping", "ack"} or self.deduper.duplicate(packet):
                continue
            try:
                self._handle(packet)
            except Exception:
                LOG.exception("failed to process UDP action")

    def _handle(self, packet: dict[str, Any]) -> None:
        action = packet.get("action")
        if not isinstance(action, dict):
            LOG.warning("UDP packet has no structured action; ignored")
            return
        action_type = str(action.get("type", "")).lower()
        if action_type == "control":
            self._control(action)
        elif action_type == "navigate":
            self._navigate(action)
        else:
            LOG.warning("unsupported action type %r", action_type)

    def _control(self, action: dict[str, Any]) -> None:
        verb = str(action.get("verb", "")).lower()
        if verb not in CONTROL_VERBS:
            LOG.warning("unsupported control verb %r", verb)
            return
        if verb == "stop":
            self.turn = None
            self.linear = self.angular = 0.0
            self.override = True
            self.mission.pause()
            LOG.info("STOP: keyboard zero hold")
        elif verb == "resume":
            self.turn = None
            self.linear = self.angular = 0.0
            self.mission.resume()
            self.override = False
            self.publisher.publish(self._Twist())
            LOG.info("RESUME: keyboard override released")
        elif verb == "forward":
            self.turn = None
            self.mission.pause()
            self.linear, self.angular, self.override = FORWARD_SPEED_MPS, 0.0, True
            LOG.info("FORWARD %.2f m/s", FORWARD_SPEED_MPS)
        elif verb == "backward":
            self.turn = None
            self.mission.pause()
            self.linear, self.angular, self.override = BACKWARD_SPEED_MPS, 0.0, True
            LOG.info("BACKWARD %.2f m/s", BACKWARD_SPEED_MPS)
        else:
            self.mission.pause()
            self._start_turn(1 if verb == "left" else -1, action.get("angle_deg"))
            LOG.info("%s turn requested", verb.upper())

    def _start_turn(self, direction: int, angle: Any) -> None:
        try:
            angle_deg = float(angle if angle is not None else DEFAULT_TURN_DEG)
        except (TypeError, ValueError):
            angle_deg = DEFAULT_TURN_DEG
        angle_deg = min(360.0, max(1.0, angle_deg))
        now = time.monotonic()
        self.turn = TurnState(
            direction=direction,
            target_rad=math.radians(angle_deg),
            start_yaw=self.odom_yaw,
            last_yaw=self.odom_yaw,
            deadline=(now + math.radians(angle_deg) / TURN_MAX_RAD_S)
            if self.odom_yaw is None
            else None,
        )
        self.linear = 0.0
        self.angular = direction * TURN_MAX_RAD_S
        self.override = True

    def _advance_turn(self) -> None:
        turn = self.turn
        if turn is None:
            return
        if turn.deadline is not None:
            if time.monotonic() >= turn.deadline:
                self.turn = None
                self.linear = self.angular = 0.0
            return
        if self.odom_yaw is not None and turn.last_yaw is not None:
            turn.travelled_rad += turn.direction * _wrap_angle(
                self.odom_yaw - turn.last_yaw
            )
            turn.travelled_rad = max(0.0, turn.travelled_rad)
            turn.last_yaw = self.odom_yaw
        remaining = turn.target_rad - turn.travelled_rad
        if remaining <= TURN_TOLERANCE_RAD:
            self.turn = None
            self.linear = self.angular = 0.0
            LOG.info("turn complete; holding keyboard zero until RESUME")
            return
        self.angular = turn.direction * min(
            TURN_MAX_RAD_S,
            max(TURN_MIN_RAD_S, remaining * 1.4),
        )

    def _navigate(self, action: dict[str, Any]) -> None:
        position = action.get("position") or {}
        token = str(position.get("token") or position.get("section") or "").upper()
        if token != "A":
            LOG.warning("only Storage A pickup is enabled; received %r", token)
            return
        task = str(action.get("task") or "fetch").lower()
        if task not in {"fetch", "fetch_hold"}:
            LOG.warning("only fetch/fetch_hold is enabled for Storage A; received %r", task)
            return
        color = str(position.get("color") or "blue").lower()
        self.turn = None
        self.linear = self.angular = 0.0
        self.override = False
        self.mission.start_pick_a(color, deliver=(task == "fetch"))

    def shutdown(self) -> None:
        self.override = False
        self.publisher.publish(self._Twist())
        self.mission.cancel()
        self.socket.close()
        self.node.destroy_node()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default=DEFAULT_BIND, help="UDP bind as HOST:PORT")
    parser.add_argument(
        "--demo-dir",
        default=str(Path(__file__).resolve().parents[1]),
        help="warehouse_agv_demo directory",
    )
    parser.add_argument("--dry-run", action="store_true", help="log pickup without launching it")
    args = parser.parse_args(argv)
    host, separator, port_text = str(args.bind).rpartition(":")
    if not separator:
        parser.error("--bind phải có dạng HOST:PORT")
    try:
        port = int(port_text)
    except ValueError:
        parser.error("UDP port không hợp lệ")
    import rclpy

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rclpy.init()
    bridge = KeyboardUDPBridge((host or "0.0.0.0", port), Path(args.demo_dir).expanduser().resolve(), dry_run=args.dry_run)
    try:
        rclpy.spin(bridge.node)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.shutdown()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
