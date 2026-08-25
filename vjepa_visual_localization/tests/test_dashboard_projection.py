from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

from scripts.localization_dashboard import (
    DashboardRenderer,
    LocalizationDashboardNode,
    PoseSample,
    STREAMING_HEIGHT,
    differential_keyboard_command,
    goal_status_array_has_active_goal,
    plan_for_display,
    project_pose_with_odometry,
)


def test_delayed_visual_pose_is_projected_by_relative_odometry() -> None:
    visual = PoseSample(10.0, 5.0, 2.0, 0.0, math.pi / 2.0)
    odom_at_visual_time = PoseSample(10.0, 1.0, 1.0, 0.0, 0.0)
    current_odom = PoseSample(11.0, 2.0, 1.0, 0.0, math.pi / 2.0)

    projected = project_pose_with_odometry(
        visual, odom_at_visual_time, current_odom
    )

    assert projected.timestamp == 11.0
    assert projected.x == 5.0
    assert projected.y == 3.0
    assert abs(abs(projected.yaw) - math.pi) < 1e-12


def test_once_computed_plan_stays_visible_for_active_nav2_goal() -> None:
    plan = ((1.0, 2.0), (3.0, 4.0))

    assert plan_for_display(
        plan, received_at=10.0, nav_goal_active=True, now=100.0
    ) == plan
    assert plan_for_display(
        plan, received_at=10.0, nav_goal_active=False, now=100.0
    ) == ()
    assert plan_for_display(
        plan, received_at=99.0, nav_goal_active=False, now=100.0
    ) == plan


def test_nav2_status_recognizes_only_live_goals() -> None:
    assert goal_status_array_has_active_goal(
        SimpleNamespace(status_list=[SimpleNamespace(status=2)])
    )
    assert not goal_status_array_has_active_goal(
        SimpleNamespace(status_list=[SimpleNamespace(status=4)])
    )


def test_map_render_contains_only_truth_gps_and_planning() -> None:
    drawn_polylines: list[tuple[tuple[tuple[float, float], ...], tuple[int, int, int]]] = []
    drawn_arrows: list[PoseSample] = []
    drawn_labels: list[str] = []

    class SpyRenderer(DashboardRenderer):
        @staticmethod
        def _polyline(image, points, convert, color, thickness):
            drawn_polylines.append((points, color))

        @staticmethod
        def _arrow(image, pose, color, radius=9):
            drawn_arrows.append(pose)

        @staticmethod
        def _line(image, text, x, y, **kwargs):
            drawn_labels.append(str(text))

    renderer = object.__new__(SpyRenderer)
    renderer.map_gray = np.zeros((20, 30), dtype=np.uint8)
    renderer.map_width = 30
    renderer.map_height = 20
    renderer.scale = 1.0
    renderer.origin_x = 0.0
    renderer.origin_y = 0.0
    renderer.resolution = 1.0
    renderer.regions = SimpleNamespace(centers=())
    truth = PoseSample(1.0, 2.0, 3.0, 0.0, 0.0)
    plan = ((0.0, 0.0), (1.0, 1.0))

    rendered = renderer.render_map({
        "astar_plan": plan,
        "truth_trail": ((0.0, 0.0), (2.0, 3.0)),
        "current_truth": truth,
    })

    assert rendered.shape == (20, 30, 3)
    assert [points for points, _ in drawn_polylines] == [plan, ((0.0, 0.0), (2.0, 3.0))]
    assert drawn_arrows == [truth]
    assert drawn_labels == []


def test_compact_streaming_layout_and_keyboard_teleop() -> None:
    class Publisher:
        def __init__(self, subscriptions: int = 0) -> None:
            self.subscriptions = subscriptions
            self.messages = []

        def get_subscription_count(self) -> int:
            return self.subscriptions

        def publish(self, message) -> None:
            self.messages.append(message)

    node = object.__new__(LocalizationDashboardNode)
    node.keyboard_mux_velocity = Publisher(subscriptions=1)
    node.keyboard_direct_velocity = Publisher()
    node.keyboard_active = False
    node.keyboard_linear = 0.0
    node.keyboard_angular = 0.0

    assert STREAMING_HEIGHT == 568
    assert node.handle_keyboard_key(ord("w")) is True
    command = node.keyboard_mux_velocity.messages[-1]
    assert command.linear.x == 1.00
    assert command.linear.y == 0.0
    assert command.angular.z == 0.0
    assert not node.keyboard_direct_velocity.messages

    node.refresh_keyboard_command()
    assert len(node.keyboard_mux_velocity.messages) == 2

    assert node.handle_keyboard_key(32) is True
    stop = node.keyboard_mux_velocity.messages[-1]
    assert stop.linear.x == 0.0
    assert stop.angular.z == 0.0

    assert differential_keyboard_command(
        forward=True, backward=False, left=False, right=True
    ) == (1.0, -0.75)
    assert node.update_held_keyboard({"w", "d"}) is True
    turn = node.keyboard_mux_velocity.messages[-1]
    assert turn.linear.x == 1.0
    assert turn.angular.z == -0.75
