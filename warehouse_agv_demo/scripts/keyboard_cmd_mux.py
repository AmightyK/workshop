#!/usr/bin/env python3
"""Give held-key teleop priority over Nav2 before collision monitoring."""

from __future__ import annotations

import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Bool


class KeyboardCmdMux(Node):
    """Select keyboard commands immediately, otherwise forward Nav2 output."""

    def __init__(self) -> None:
        super().__init__("keyboard_cmd_mux")
        self.manual_timeout = 0.12
        self.nav_timeout = 0.70
        self.manual_command = Twist()
        self.nav_command = Twist()
        self.manual_time = -float("inf")
        self.nav_time = -float("inf")
        self.person_stop = False
        self.publisher = self.create_publisher(Twist, "/cmd_vel_safety_input", 10)
        self.create_subscription(
            Twist, "/cmd_vel_keyboard", self._on_keyboard, 10
        )
        self.create_subscription(
            Twist, "/cmd_vel_smoothed", self._on_nav, 10
        )
        self.create_subscription(
            Bool, "/warehouse/person_stop", self._on_person_stop, 10
        )
        self.create_timer(0.02, self._publish_selected)
        self.get_logger().info(
            "Velocity priority: keyboard > Nav2; worker gate -> collision monitor"
        )

    def _on_keyboard(self, message: Twist) -> None:
        self.manual_command = message
        self.manual_time = time.monotonic()
        # Also publish in the callback so release and direction changes do not
        # wait for the next 50 Hz control tick.
        self.publisher.publish(Twist() if self.person_stop else message)

    def _on_nav(self, message: Twist) -> None:
        self.nav_command = message
        self.nav_time = time.monotonic()

    def _on_person_stop(self, message: Bool) -> None:
        was_stopped = self.person_stop
        self.person_stop = bool(message.data)
        if self.person_stop:
            # Publish the emergency transition immediately instead of waiting
            # for the next 50 Hz mux timer.
            self.publisher.publish(Twist())
        elif was_stopped:
            # The stored manual/Nav2 command remains intact and resumes now.
            self._publish_selected()

    def _publish_selected(self) -> None:
        if self.person_stop:
            self.publisher.publish(Twist())
            return
        now = time.monotonic()
        if now - self.manual_time <= self.manual_timeout:
            self.publisher.publish(self.manual_command)
        elif now - self.nav_time <= self.nav_timeout:
            self.publisher.publish(self.nav_command)
        else:
            self.publisher.publish(Twist())


def main() -> None:
    rclpy.init()
    node = KeyboardCmdMux()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.publisher.publish(Twist())
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
