#!/usr/bin/env python3
"""Live V-JEPA streaming-QA dashboard plus warehouse navigation map.

The visible estimate is the raw camera-only pose at the center of its rolling
clip. Gazebo truth is timestamp-aligned only for evaluation. Optional odometry
projection is disabled by default and, when explicitly enabled, remains
telemetry-only rather than being drawn as pure V-JEPA.
"""

from __future__ import annotations

import argparse
import base64
import ctypes
import ctypes.util
import functools
import hashlib
import json
import math
import os
import sys
import threading
import time
import unicodedata
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
WAREHOUSE_ROOT = WORKSPACE_ROOT / "warehouse_agv_demo"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import rclpy
from action_msgs.msg import GoalStatus, GoalStatusArray
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.transport13 import Node as GazeboNode
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from nav_msgs.msg import Odometry, Path as NavPath
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import Float32MultiArray, String

from src.data.ros_image import image_message_to_rgb, message_timestamp_sec

from src.evaluation.warehouse_context import (
    EntityPose,
    WarehouseRegionIndex,
    identify_forward_obstacle,
    scan_sector_min,
    wrap_angle,
)


TELEMETRY_WINDOW = "VL-JEPA Warehouse Streaming QA"
MAP_WINDOW = "Warehouse Map - Truth GPS & Planning"
QUESTIONS_WINDOW = "Warehouse Questions - 5 of 20"
STREAMING_WIDTH = 1440
STREAMING_HEIGHT = 568
# Questions window contains only the five cards. Its former 62 px header and
# 52 px footer duplicated information already visible in the cards.
QUESTIONS_WIDTH = 492
QUESTIONS_HEIGHT = 698
DASHBOARD_OBSTACLE_DISTANCE_M = 4.0
CAMERA_HORIZONTAL_FOV_RAD = 1.22
CAMERA_FORWARD_OFFSET_M = -0.1915625
CAMERA_PERSON_MAX_DISTANCE_M = 8.0

# Five questions selected from the existing twenty-question bank after the
# complete dock -> A01 blue-box pick -> Packing -> charging route was replayed.
# Their answers are deliberately recomputed from the current snapshot, rather
# than tied to a special waypoint or a memorized yes/no event.
WAREHOUSE_MODEL_ANSWER_CHOICES: dict[str, tuple[tuple[str, str], ...]] = {
    "q07": (
        ("none", "Chưa có vật cản gần"),
        ("left", "Vật cản ở bên trái"),
        ("center", "Vật cản ở chính giữa"),
        ("right", "Vật cản ở bên phải"),
    ),
    "q14": (
        ("clear", "Đủ rộng để tiếp tục"),
        ("narrow", "Hơi hẹp, cần giảm tốc"),
        ("blocked", "Không đủ khoảng trống an toàn"),
    ),
    "q13": (
        ("none", "Không phát hiện người ở phía trước"),
        ("far", "Có người ở phía xa"),
        ("near_clear", "Có người ở gần nhưng không chắn đường"),
        ("on_path_stationary", "Có người đứng yên phía trước robot"),
        ("on_path_moving", "Có người đang di chuyển phía trước robot"),
    ),
    "mission_state": (
        ("outbound", "Đang đi từ dock tới kệ"),
        ("align", "Đang căn chỉnh trước hộp"),
        ("grasp", "Đang gắp và kiểm tra hộp"),
        ("transport", "Đang chở hộp tới điểm giao"),
        ("finish", "Đang hoàn tất / chờ nhiệm vụ mới"),
    ),
    "q20": (
        ("stop", "Dừng và giữ vị trí"),
        ("straight", "Đi thẳng theo lộ trình"),
        ("left", "Cua trái và đi chậm"),
        ("right", "Cua phải và đi chậm"),
        ("replan", "Đổi sang đường khác"),
    ),
}

WAREHOUSE_ANSWER_CANDIDATES: dict[str, tuple[str, ...]] = {
    "q01": ("Robot đang đứng yên.", "Robot đang di chuyển chậm.", "Robot đang di chuyển nhanh."),
    "q02": ("Robot đi thẳng.", "Robot đang cua trái.", "Robot đang cua phải."),
    "q03": ("Tốc độ đang tăng.", "Tốc độ đang giữ tương đối đều.", "Tốc độ đang giảm."),
    "q04": ("Phía trước đang thoáng và nhìn được xa.", "Phía trước đang bị chặn gần."),
    "q05": ("Không có vật nào đang phình to rõ rệt.", "Có một vật phía trước đang lớn dần."),
    "q06": ("Vật đó đang đứng yên.", "Vật đó đang di chuyển.", "Chưa thấy vật đó tự di chuyển."),
    "q07": ("Vật cản gần nhất ở bên trái.", "Vật cản gần nhất ở chính giữa.", "Vật cản gần nhất ở bên phải."),
    "q08": ("Không, hướng đi hiện tại vẫn an toàn.", "Có, giữ nguyên hướng có thể dẫn đến va chạm."),
    "q09": ("Đi tiếp bình thường.", "Chậm lại và giữ khoảng cách.", "Dừng và chờ đường trống.", "Đổi sang đường đi khác."),
    "q10": ("Chưa cần né.", "Né sang trái.", "Né sang phải."),
    "q11": ("Lối đi phía trước đang rộng ra.", "Lối đi phía trước vẫn ổn định.", "Lối đi đang hẹp lại."),
    "q12": ("Robot đang giữ gần giữa lối.", "Robot đang lệch nhẹ sang trái.", "Robot đang lệch nhẹ sang phải."),
    "q13": (
        "Không phát hiện người ở phía trước.",
        "Có người ở phía xa.",
        "Có người ở gần nhưng không chắn đường.",
        "Có người đứng yên phía trước robot.",
        "Có người đang di chuyển phía trước robot.",
    ),
    "q14": (
        "Khoảng trống phía trước đủ rộng để đi qua.",
        "Khoảng trống phía trước hơi hẹp, robot nên giảm tốc.",
        "Phía trước không đủ khoảng trống để đi qua an toàn.",
    ),
    "q15": ("Phía trước tiếp tục là lối đi.", "Phía trước là một khúc cua trái.", "Phía trước là một khúc cua phải.", "Phía trước là lối cụt."),
    "q16": ("Chưa thấy vùng che khuất nguy hiểm.", "Vùng ngay sau vật cản đang bị che khuất.", "Hai bên góc kệ đang bị che khuất."),
    "q17": ("Lối đi sẽ tiếp tục mở ra phía trước.", "Vật phía trước sẽ lớn dần.", "Khung cảnh sẽ xoay dần sang trái.", "Khung cảnh sẽ xoay dần sang phải."),
    "q19": ("Có, khu vực này khá dễ nhầm với các nhịp kệ khác.", "Không, khu vực này có đặc điểm dễ nhận ra."),
    "q20": ("Dừng.", "Đi thẳng và giữ tốc độ hiện tại.", "Cua trái và đi chậm.", "Cua phải và đi chậm.", "Đổi sang đường khác."),
    "mission_state": (
        "Robot đang đi từ dock tới kệ.",
        "Robot đang căn chỉnh trước hộp.",
        "Robot đang gắp và kiểm tra hộp.",
        "Robot đang chở hộp tới điểm giao.",
        "Robot đang hoàn tất hoặc chờ nhiệm vụ mới.",
    ),
}

WAREHOUSE_ANSWER_DISTRACTORS: dict[str, tuple[str, ...]] = {
    "q01": ("Robot đang lùi chậm.", "Robot vừa bắt đầu tăng tốc.", "Robot đang giảm tốc.", "Robot đang xoay tại chỗ.", "Robot đang dừng lại."),
    "q02": ("Robot đang đi chếch trái.", "Robot đang đi chếch phải.", "Robot đang xoay trái tại chỗ.", "Robot đang xoay phải tại chỗ.", "Robot đang lùi thẳng."),
    "q03": ("Tốc độ đang tăng nhẹ.", "Tốc độ đang tăng nhanh.", "Tốc độ đang giảm nhẹ.", "Tốc độ đang giảm nhanh.", "Tốc độ đang dao động."),
    "q04": ("Phía trước chỉ bị che một phần.", "Phía trước bị chắn ở bên trái.", "Phía trước bị chắn ở chính giữa.", "Phía trước bị chắn ở bên phải.", "Tầm nhìn phía trước đang mở rộng.", "Tầm nhìn phía trước đang thu hẹp."),
    "q05": ("Có một vật đang lớn chậm ở bên trái.", "Có một vật đang lớn nhanh ở chính giữa.", "Có một vật đang lớn ở bên phải.", "Vật phía trước đang nhỏ dần.", "Kích thước vật phía trước gần như không đổi.", "Chỉ có mặt kệ đang trôi ngang."),
    "q06": ("Vật đó đang đi sang trái.", "Vật đó đang đi sang phải.", "Vật đó đang tiến lại gần.", "Vật đó đang đi ra xa.", "Chuyển động của vật chưa rõ ràng."),
    "q07": ("Chưa có vật cản gần.", "Vật cản nằm hơi lệch trái.", "Vật cản nằm hơi lệch phải.", "Có vật cản ở cả hai bên.", "Vật cản đang ở rất gần chính giữa."),
    "q08": ("Có nguy cơ va chạm ở bên trái.", "Có nguy cơ va chạm ở chính giữa.", "Có nguy cơ va chạm ở bên phải.", "Chưa có nguy cơ trong vài giây tới.", "Khoảng trống phía trước đang tăng.", "Nguy cơ đang giảm dần."),
    "q09": ("Đi tiếp nhưng giảm nhẹ tốc độ.", "Đi tiếp và tăng tốc.", "Giữ nguyên tốc độ.", "Dừng ngay.", "Lệch sang trái rồi đi tiếp.", "Lệch sang phải rồi đi tiếp."),
    "q10": ("Giữ nguyên hướng hiện tại.", "Né nhẹ sang trái.", "Né nhẹ sang phải.", "Lùi lại trước khi né.", "Dừng lại, chưa chọn hướng né."),
    "q11": ("Lối đi đang rộng ra bên trái.", "Lối đi đang rộng ra bên phải.", "Lối đi đang hẹp ở chính giữa.", "Chiều rộng lối đi thay đổi không đáng kể.", "Lối đi sắp mở rộng trở lại."),
    "q12": ("Robot đang sát mép trái.", "Robot đang sát mép phải.", "Robot đang trở về giữa lối.", "Robot đang đổi từ lệch trái sang lệch phải.", "Độ lệch hiện chưa rõ."),
    "q13": ("Có người ở xa đang đứng yên.", "Có người đang đi từ trái sang phải.", "Có người đang đi từ phải sang trái.", "Có người đang đi ra xa.", "Có người đang tiến lại gần."),
    "q14": (
        "Khoảng trống bên trái rộng hơn.",
        "Khoảng trống bên phải rộng hơn.",
        "Robot nên giữ gần giữa lối đi.",
        "Robot cần chờ lối phía trước mở ra.",
        "Chưa đủ quan sát để đánh giá bề rộng phía trước.",
    ),
    "q15": ("Phía trước có ngã rẽ trái.", "Phía trước có ngã rẽ phải.", "Phía trước là ngã tư.", "Khúc cua phía trước chưa nhìn rõ.", "Lối đi phía trước đang mở rộng."),
    "q16": ("Góc trái phía trước đang bị che khuất.", "Góc phải phía trước đang bị che khuất.", "Phía sau người đi bộ đang bị che khuất.", "Phía sau thùng hàng đang bị che khuất.", "Các khe giữa hai nhịp kệ đang bị che khuất."),
    "q17": ("Vật phía trước sẽ trôi sang trái.", "Vật phía trước sẽ trôi sang phải.", "Lối đi sẽ hẹp lại.", "Lối đi sẽ rộng ra.", "Một người sẽ đi ngang qua phía trước."),
    "q19": ("Khu vực này dễ nhầm vì các kệ lặp lại.", "Khu vực này dễ nhận nhờ biển chữ.", "Khu vực này dễ nhận nhờ ô màu.", "Khu vực này dễ nhận nhờ pad sàn.", "Cần thêm quan sát để phân biệt khu vực.", "Khu vực này không giống đoạn vừa đi qua."),
    "q20": ("Đi chậm và giữ giữa lối.", "Giảm tốc rồi dừng.", "Đi thẳng rồi lệch trái.", "Đi thẳng rồi lệch phải.", "Chờ thêm một quan sát.", "Tiếp tục nhưng giữ khoảng cách."),
    "mission_state": (
        "Robot vừa rời dock và đang tăng tốc.",
        "Robot đang chờ lối đi tới kệ thông thoáng.",
        "Robot đã tới kệ nhưng chưa bắt đầu căn chỉnh.",
        "Robot đang rời kệ với payload trên khay.",
        "Robot đã giao hàng và đang trở về trạm sạc.",
    ),
}


def answer_candidates(question_id: str) -> tuple[str, ...]:
    canonical = tuple(
        dict.fromkeys(
            WAREHOUSE_ANSWER_CANDIDATES.get(question_id, ())
            + WAREHOUSE_ANSWER_DISTRACTORS.get(question_id, ())
        )
    )
    variants: list[str] = []
    for answer in canonical:
        plain = answer.rstrip(".")
        lowered = plain[:1].lower() + plain[1:]
        variants.extend(
            (
                answer,
                f"Hiện tại, {lowered}.",
                f"Quan sát cho thấy {lowered}.",
                f"Dấu hiệu gần nhất cho thấy {lowered}.",
                f"Trong vài khung hình vừa qua, {lowered}.",
            )
        )
    return tuple(dict.fromkeys(variants))


def expanded_answer_bank() -> tuple[str, ...]:
    """Build a dense, fixed semantic answer corpus for the latent map."""
    return tuple(
        dict.fromkeys(
            answer
            for question_id in WAREHOUSE_ANSWER_CANDIDATES
            for answer in answer_candidates(question_id)
        )
    )


def answer_observation_ambiguity(
    question_id: str, snapshot: dict[str, Any]
) -> float:
    """Estimate semantic ambiguity from proximity to observed decision bands."""
    linear = abs(float(snapshot.get("linear_x", 0.0)))
    angular = abs(float(snapshot.get("angular_z", 0.0)))
    clearance = float(snapshot.get("front_clearance", math.inf))
    behavior = snapshot.get("behavior_decision") or {}

    def band(value: float, threshold: float, width: float) -> float:
        return max(0.0, 1.0 - abs(value - threshold) / max(width, 1.0e-6))

    if question_id in {"q01", "q03"}:
        return max(band(linear, 0.03, 0.05), band(linear, 0.22, 0.10))
    if question_id in {"q02", "q10", "q12", "q15", "q20"}:
        return band(angular, 0.12, 0.10)
    if question_id in {"q04", "q05", "q06", "q07", "q11", "q14", "q16", "q17"}:
        if not math.isfinite(clearance):
            return 0.15
        return max(band(clearance, 4.0, 0.9), band(clearance, 1.2, 0.45))
    if question_id in {"q08", "q09"}:
        if str(behavior.get("decision", "PASS")) in {"WAIT", "REPLAN"}:
            return 0.35
        if linear > 0.03 and math.isfinite(clearance):
            return band(clearance / linear, 3.0, 1.5)
        return 0.2
    if question_id == "q13":
        visual = snapshot.get("camera_person_detection") or {}
        confidence = float(visual.get("confidence", 0.0))
        return band(confidence, 0.28, 0.18)
    return 0.25


def live_model_answer_key(question_id: str, snapshot: dict[str, Any]) -> str:
    """Return the evidence-backed model answer state for a live snapshot."""
    linear = float(snapshot.get("linear_x", 0.0))
    angular = float(snapshot.get("angular_z", 0.0))
    clearance = float(snapshot.get("front_clearance", math.inf))
    obstacle = snapshot.get("obstacle")
    behavior = snapshot.get("behavior_decision") or {}
    decision = str(behavior.get("decision", "PASS")).upper()
    moving = abs(linear) >= 0.03 or abs(angular) >= 0.05

    if question_id == "q07":
        has_close_obstacle = obstacle is not None or (
            math.isfinite(clearance)
            and clearance <= DASHBOARD_OBSTACLE_DISTANCE_M
        )
        if not has_close_obstacle:
            return "none"
        bearing = getattr(obstacle, "bearing_rad", None)
        if bearing is None or abs(float(bearing)) < 0.25:
            return "center"
        return "left" if float(bearing) > 0.0 else "right"
    if question_id == "q14":
        if math.isfinite(clearance) and clearance < 1.2:
            return "blocked"
        if (
            obstacle is not None
            or (
                math.isfinite(clearance)
                and clearance <= DASHBOARD_OBSTACLE_DISTANCE_M
            )
        ):
            return "narrow"
        return "clear"
    if question_id == "q13":
        # Q3 is camera-gated. Planner occupancy and the old 360-degree
        # ``nearby_people`` list must never make a person behind / beside the
        # camera appear in this answer.
        camera_people = snapshot.get("camera_frustum_people") or ()
        visual = snapshot.get("camera_person_detection") or {}
        camera_age_ms = snapshot.get("camera_age_ms")
        if (
            not camera_people
            or not bool(visual.get("visible"))
            or (
                isinstance(camera_age_ms, (int, float))
                and float(camera_age_ms) > 750.0
            )
        ):
            return "none"
        bbox = visual.get("bbox")
        frame_size = visual.get("frame_size")
        matched_people = camera_people
        if bbox is not None and frame_size is not None:
            x, _, box_width, _ = (float(item) for item in bbox)
            frame_width = max(1.0, float(frame_size[0]))
            detection_x = (x + 0.5 * box_width) / frame_width
            half_fov = 0.5 * CAMERA_HORIZONTAL_FOV_RAD
            matched_people = tuple(
                item
                for item in camera_people
                if abs(
                    detection_x
                    - (
                        0.5
                        - math.tan(float(item[2]))
                        / (2.0 * math.tan(half_fov))
                    )
                )
                <= 0.10 + 0.5 * box_width / frame_width
            )
        if not matched_people:
            return "none"
        nearest_distance = float(matched_people[0][0])
        if nearest_distance >= 3.5 or bool(visual.get("far")):
            return "far"
        if not bool(visual.get("on_path")):
            return "near_clear"
        return (
            "on_path_moving"
            if bool(visual.get("moving"))
            else "on_path_stationary"
        )
    if question_id == "mission_state":
        mission = snapshot.get("mission_state") or {}
        state = str(mission.get("state", "WAITING")).upper()
        if state in {"NAVIGATE_TO_SHELF"}:
            return "outbound"
        if state == "PARSE_TASK":
            area = str(snapshot.get("area", "")).lower()
            return "align" if "kệ" in area or "shelf" in area else "finish"
        if state in {"SHELF_APPROACH", "RAISE_LIFT", "ALIGN_PACKAGE"}:
            return "align"
        if state in {"GRASP_PACKAGE", "VERIFY_GRASP"}:
            return "grasp"
        if state in {"RETURN_TO_DROPOFF"}:
            return "transport"
        return "finish"
    if question_id == "q20":
        if decision == "WAIT":
            return "stop"
        if decision == "REPLAN":
            return "replan"
        if not moving:
            return "stop"
        if angular >= 0.12:
            return "left"
        if angular <= -0.12:
            return "right"
        return "straight"
    raise KeyError(f"question {question_id!r} has no live model answer")


@functools.lru_cache(maxsize=32)
def dashboard_font(
    pixel_size: int, weight: str = "regular"
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a Unicode font for Vietnamese dashboard text."""
    regular_candidates = (
        os.environ.get("VJEPA_DASHBOARD_FONT", ""),
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/ubuntu/UbuntuSans[wdth,wght].ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    bold_candidates = (
        os.environ.get("VJEPA_DASHBOARD_FONT_BOLD", ""),
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
        "/usr/share/fonts/truetype/ubuntu/UbuntuSans[wdth,wght].ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    )
    candidates = bold_candidates if weight == "bold" else regular_candidates
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return ImageFont.truetype(candidate, pixel_size)
    return ImageFont.load_default(size=pixel_size)


def latent_evidence_profile(question_id: str) -> str:
    """Select which real latent evidence is relevant to the active question."""
    if question_id in {"q01", "q02", "q03", "q11", "q12", "mission_state"}:
        return "temporal_motion"
    if question_id in {"q08", "q09", "q10", "q15", "q17", "q20"}:
        return "future_rollout"
    if question_id == "q19":
        return "localization_match"
    return "current_observation"


LOW_LATENCY_IMAGE_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)
ACTIVE_NAV2_GOAL_STATES = frozenset({
    GoalStatus.STATUS_ACCEPTED,
    GoalStatus.STATUS_EXECUTING,
    GoalStatus.STATUS_CANCELING,
})


def gui_wait_ms(next_deadline: float, now: float) -> int:
    """Poll OpenCV only for the unspent part of the frame budget."""
    return max(1, int(math.ceil(max(0.0, next_deadline - now) * 1000.0)))


@dataclass(frozen=True)
class PoseSample:
    timestamp: float
    x: float
    y: float
    z: float
    yaw: float


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def gazebo_timestamp(message: Any) -> float:
    try:
        return float(message.header.stamp.sec) + float(message.header.stamp.nsec) * 1e-9
    except (AttributeError, TypeError):
        return 0.0


def ros_timestamp(message: PoseWithCovarianceStamped) -> float:
    return float(message.header.stamp.sec) + float(message.header.stamp.nanosec) * 1e-9


def ascii_text(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value).replace("đ", "d").replace("Đ", "D")
    return "".join(character for character in normalized if unicodedata.category(character) != "Mn")


def differential_keyboard_command(
    *, forward: bool, backward: bool, left: bool, right: bool
) -> tuple[float, float]:
    """Return simultaneous WASD linear/angular commands for a diff-drive base."""
    linear = 0.0
    angular = 0.0
    if forward != backward:
        linear = 1.0 if forward else -0.5
    if left != right:
        angular = 0.75 if left else -0.75
    return linear, angular


def goal_status_array_has_active_goal(message: GoalStatusArray) -> bool:
    """Return whether an action status array contains a live Nav2 goal."""
    return any(
        item.status in ACTIVE_NAV2_GOAL_STATES for item in message.status_list
    )


def plan_for_display(
    plan: tuple[tuple[float, float], ...],
    *,
    received_at: float,
    nav_goal_active: bool,
    now: float,
    startup_grace: float = 3.0,
) -> tuple[tuple[float, float], ...]:
    """Keep the once-computed A* path visible for the complete Nav2 goal."""
    if nav_goal_active or now - received_at <= startup_grace:
        return plan
    return ()


class X11KeyboardState:
    """Poll held keys, including simultaneous W+D, through XWayland/X11."""

    KEY_NAMES = ("w", "s", "a", "d", "space", "Escape")

    def __init__(self) -> None:
        self.library = None
        self.display = None
        self.keycodes: dict[str, int] = {}
        library_name = ctypes.util.find_library("X11")
        if not library_name or not os.environ.get("DISPLAY"):
            return
        library = ctypes.CDLL(library_name)
        library.XOpenDisplay.argtypes = [ctypes.c_char_p]
        library.XOpenDisplay.restype = ctypes.c_void_p
        library.XStringToKeysym.argtypes = [ctypes.c_char_p]
        library.XStringToKeysym.restype = ctypes.c_ulong
        library.XKeysymToKeycode.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        library.XKeysymToKeycode.restype = ctypes.c_ubyte
        library.XQueryKeymap.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        library.XQueryKeymap.restype = ctypes.c_int
        library.XCloseDisplay.argtypes = [ctypes.c_void_p]
        display = library.XOpenDisplay(None)
        if not display:
            return
        self.library = library
        self.display = display
        self.keycodes = {
            name: int(
                library.XKeysymToKeycode(
                    display, library.XStringToKeysym(name.encode("ascii"))
                )
            )
            for name in self.KEY_NAMES
        }

    @property
    def available(self) -> bool:
        return self.library is not None and self.display is not None

    def pressed(self) -> set[str]:
        if not self.available:
            return set()
        keymap = ctypes.create_string_buffer(32)
        self.library.XQueryKeymap(self.display, keymap)
        raw = keymap.raw
        return {
            name
            for name, code in self.keycodes.items()
            if code > 0 and raw[code >> 3] & (1 << (code & 7))
        }

    def close(self) -> None:
        if self.available:
            self.library.XCloseDisplay(self.display)
        self.library = None
        self.display = None


def project_pose_with_odometry(
    visual_pose: PoseSample,
    odom_at_visual_time: PoseSample,
    current_odom: PoseSample,
) -> PoseSample:
    """Bring a delayed visual pose to the latest odom time without truth data."""
    dx = current_odom.x - odom_at_visual_time.x
    dy = current_odom.y - odom_at_visual_time.y
    cos_origin = math.cos(odom_at_visual_time.yaw)
    sin_origin = math.sin(odom_at_visual_time.yaw)
    relative_x = cos_origin * dx + sin_origin * dy
    relative_y = -sin_origin * dx + cos_origin * dy
    cos_visual = math.cos(visual_pose.yaw)
    sin_visual = math.sin(visual_pose.yaw)
    return PoseSample(
        current_odom.timestamp,
        visual_pose.x + cos_visual * relative_x - sin_visual * relative_y,
        visual_pose.y + sin_visual * relative_x + cos_visual * relative_y,
        visual_pose.z,
        wrap_angle(
            visual_pose.yaw
            + wrap_angle(current_odom.yaw - odom_at_visual_time.yaw)
        ),
    )


def people_in_camera_frustum(
    *,
    agv_x: float,
    agv_y: float,
    agv_yaw: float,
    entities: tuple[EntityPose, ...],
) -> tuple[tuple[float, str, float], ...]:
    """Return workers geometrically inside the forward RGB-camera view.

    The camera is rigidly aligned with the robot's +X axis and uses the 1.22
    rad horizontal FOV declared in ``warehouse_agv/model.sdf``. The worker's
    collision radius is included so a partly visible body at an image edge is
    not discarded.
    """
    camera_x = agv_x + math.cos(agv_yaw) * CAMERA_FORWARD_OFFSET_M
    camera_y = agv_y + math.sin(agv_yaw) * CAMERA_FORWARD_OFFSET_M
    half_fov = 0.5 * CAMERA_HORIZONTAL_FOV_RAD
    visible: list[tuple[float, str, float]] = []
    for entity in entities:
        if not entity.name.startswith("random_worker_"):
            continue
        dx, dy = entity.x - camera_x, entity.y - camera_y
        distance = math.hypot(dx, dy)
        if distance <= 0.05 or distance > CAMERA_PERSON_MAX_DISTANCE_M:
            continue
        forward = math.cos(agv_yaw) * dx + math.sin(agv_yaw) * dy
        if forward <= 0.0:
            continue
        bearing = wrap_angle(math.atan2(dy, dx) - agv_yaw)
        body_half_angle = math.asin(min(0.95, 0.23 / distance))
        if abs(bearing) <= half_fov + body_half_angle:
            visible.append((distance, entity.name, bearing))
    return tuple(sorted(visible, key=lambda item: (item[0], item[1])))


class CameraPersonDetector:
    """Detect visible people from the latest RGB frame without world poses.

    OpenCV HOG runs on a latest-frame worker so camera callbacks never queue
    behind detection. Two consecutive positive frames are required, while a
    miss clears the visible result immediately. This deliberately favors no
    false person report when the camera cannot actually see one.
    """

    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.pending: tuple[np.ndarray, float] | None = None
        self.result: dict[str, Any] = {
            "visible": False,
            "confidence": 0.0,
            "bbox": None,
            "frame_size": None,
            "far": False,
            "on_path": False,
            "moving": False,
            "frame_timestamp": 0.0,
            "source": "camera_hog",
        }
        self.closed = False
        self.positive_streak = 0
        self.previous_center: tuple[float, float, float] | None = None
        self.motion_score = 0.0
        self.revision = 0
        self.thread = threading.Thread(
            target=self._run,
            name="camera-person-detector",
            daemon=True,
        )
        self.thread.start()

    def submit(self, frame_bgr: np.ndarray, timestamp: float) -> None:
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            return
        with self.condition:
            if self.closed:
                return
            # Only the newest camera frame matters. Replacing this slot avoids
            # a delayed detector reporting someone who has already left view.
            self.pending = (frame_bgr.copy(), float(timestamp))
            self.condition.notify()

    def snapshot(self, camera_timestamp: float) -> dict[str, Any]:
        with self.condition:
            value = dict(self.result)
        frame_timestamp = float(value.get("frame_timestamp", 0.0))
        if (
            camera_timestamp > 0.0
            and frame_timestamp > 0.0
            and camera_timestamp - frame_timestamp > 0.75
        ):
            value.update(visible=False, confidence=0.0, bbox=None)
        return value

    def close(self) -> None:
        with self.condition:
            self.closed = True
            self.pending = None
            self.condition.notify()
        self.thread.join(timeout=1.0)

    @staticmethod
    def _best_detection(
        hog: cv2.HOGDescriptor, frame_bgr: np.ndarray
    ) -> tuple[tuple[int, int, int, int] | None, float]:
        height, width = frame_bgr.shape[:2]
        scale_up = max(1.0, 540.0 / max(1, height))
        work = (
            cv2.resize(
                frame_bgr,
                (int(round(width * scale_up)), int(round(height * scale_up))),
                interpolation=cv2.INTER_LINEAR,
            )
            if scale_up > 1.01
            else frame_bgr
        )
        boxes, weights = hog.detectMultiScale(
            work,
            hitThreshold=0.0,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.04,
            groupThreshold=2.0,
            useMeanshiftGrouping=False,
        )
        candidates: list[tuple[float, tuple[int, int, int, int]]] = []
        for box, raw_score in zip(boxes, np.asarray(weights).reshape(-1)):
            x, y, box_width, box_height = (int(item) for item in box)
            original_box = (
                int(round(x / scale_up)),
                int(round(y / scale_up)),
                int(round(box_width / scale_up)),
                int(round(box_height / scale_up)),
            )
            _, original_y, original_width, original_height = original_box
            aspect = original_width / max(1, original_height)
            height_ratio = original_height / max(1, height)
            bottom_ratio = (original_y + original_height) / max(1, height)
            score = float(raw_score)
            if (
                score >= 0.85
                and 0.25 <= aspect <= 0.72
                and height_ratio >= 0.16
                and bottom_ratio >= 0.48
            ):
                candidates.append((score, original_box))
        if not candidates:
            return None, 0.0
        score, box = max(candidates, key=lambda item: item[0])
        return box, score

    def _run(self) -> None:
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        while True:
            with self.condition:
                while self.pending is None and not self.closed:
                    self.condition.wait()
                if self.closed:
                    return
                frame_bgr, timestamp = self.pending
                self.pending = None

            box, raw_score = self._best_detection(hog, frame_bgr)
            height, width = frame_bgr.shape[:2]
            if box is None:
                self.positive_streak = 0
                self.previous_center = None
                self.motion_score = 0.0
                value = {
                    "visible": False,
                    "confidence": 0.0,
                    "bbox": None,
                    "frame_size": (width, height),
                    "far": False,
                    "on_path": False,
                    "moving": False,
                    "frame_timestamp": timestamp,
                    "source": "camera_hog",
                }
            else:
                x, y, box_width, box_height = box
                center_x = (x + 0.5 * box_width) / max(1, width)
                center_y = (y + 0.5 * box_height) / max(1, height)
                previous = self.previous_center
                lateral_rate = 0.0
                if previous is not None:
                    previous_x, _, previous_timestamp = previous
                    delta_t = timestamp - previous_timestamp
                    if 0.05 <= delta_t <= 1.5:
                        lateral_rate = abs(center_x - previous_x) / delta_t
                self.previous_center = (center_x, center_y, timestamp)
                self.motion_score = 0.65 * self.motion_score + 0.35 * lateral_rate
                self.positive_streak += 1
                left = x / max(1, width)
                right = (x + box_width) / max(1, width)
                value = {
                    "visible": self.positive_streak >= 2,
                    "confidence": min(1.0, max(0.0, raw_score / 3.0)),
                    "bbox": box,
                    "frame_size": (width, height),
                    "far": box_height / max(1, height) < 0.30,
                    "on_path": left <= 0.58 and right >= 0.42,
                    "moving": self.motion_score >= 0.10,
                    "frame_timestamp": timestamp,
                    "source": "camera_hog",
                }
            with self.condition:
                self.revision += 1
                value["revision"] = self.revision
                self.result = value


class LocalizationDashboardNode(Node):
    """Join V-JEPA output and truth only in an external evaluation process."""

    def __init__(
        self,
        inventory: Path,
        *,
        camera_topic: str = "/vjepa/camera/image_raw",
        odom_projection_enabled: bool = False,
    ) -> None:
        super().__init__("vjepa_localization_dashboard")
        self.regions = WarehouseRegionIndex.from_inventory(inventory)
        self.odom_projection_enabled = odom_projection_enabled
        self.lock = threading.Lock()
        self.truth_history: deque[PoseSample] = deque(maxlen=12000)
        self.odom_history: deque[PoseSample] = deque(maxlen=12000)
        self.truth_trail: deque[tuple[float, float]] = deque(maxlen=2400)
        self.vjepa_trail: deque[tuple[float, float]] = deque(maxlen=1200)
        self.entities: tuple[EntityPose, ...] = ()
        self.current_truth: PoseSample | None = None
        self.stream_start_timestamp: float | None = None
        self.aligned_truth: PoseSample | None = None
        self.raw_vjepa_pose: PoseSample | None = None
        self.vjepa_pose: PoseSample | None = None
        self.projected_vjepa_pose: PoseSample | None = None
        self.relative_dx: float | None = None
        self.relative_dy: float | None = None
        self.position_error: float | None = None
        self.yaw_error: float | None = None
        self.raw_position_error: float | None = None
        self.raw_yaw_error: float | None = None
        self.time_delta: float | None = None
        self.front_clearance = math.inf
        self.command_linear = 0.0
        self.command_angular = 0.0
        self.keyboard_active = False
        self.keyboard_linear = 0.0
        self.keyboard_angular = 0.0
        self.debug: dict[str, Any] = {}
        self.camera_bgr: np.ndarray | None = None
        self.camera_timestamp = 0.0
        self.camera_person_detector = CameraPersonDetector()
        self.query_latent: np.ndarray | None = None
        self.query_latent_trail: deque[np.ndarray] = deque(maxlen=12)
        self.nav_status: dict[str, Any] = {}
        self.behavior_decision: dict[str, Any] = {
            "decision": "PASS",
            "reason": "waiting for predictive person planner",
            "person_id": "none",
            "scenario": "normal_driving",
        }
        self.mission_state: dict[str, Any] = {"state": "WAITING"}
        self.latent_prediction: dict[str, Any] = {}
        self.astar_plan: tuple[tuple[float, float], ...] = ()
        self.astar_plan_received = -math.inf
        self.nav_goal_active = {
            "navigate_to_pose": False,
            "navigate_through_poses": False,
        }
        self.sequence = 0

        self.create_subscription(
            PoseWithCovarianceStamped, "/vjepa_pose", self._on_vjepa, 20
        )
        self.create_subscription(String, "/vjepa_localization/debug", self._on_debug, 20)
        self.create_subscription(
            Float32MultiArray, "/vjepa_latent", self._on_latent, 20
        )
        self.create_subscription(
            Image, camera_topic, self._on_camera, LOW_LATENCY_IMAGE_QOS
        )
        self.create_subscription(
            String, "/nav/localization_status", self._on_nav_status, 20
        )
        self.create_subscription(
            String, "/warehouse/behavior_decision", self._on_behavior_decision, 20
        )
        self.create_subscription(
            String, "/warehouse/mission_state", self._on_mission_state, 20
        )
        self.create_subscription(
            String, "/vjepa/latent_prediction", self._on_latent_prediction, 20
        )
        self.create_subscription(NavPath, "/plan", self._on_plan, 10)
        # The controller follows this rounded A* path. It arrives just after
        # the raw planner output and therefore becomes the blue planning line
        # shown on the map for the rest of the active action.
        self.create_subscription(NavPath, "/plan_smoothed", self._on_plan, 10)
        self.create_subscription(
            GoalStatusArray,
            "/navigate_to_pose/_action/status",
            lambda message: self._on_action_status("navigate_to_pose", message),
            10,
        )
        self.create_subscription(
            GoalStatusArray,
            "/navigate_through_poses/_action/status",
            lambda message: self._on_action_status(
                "navigate_through_poses", message
            ),
            10,
        )
        if self.odom_projection_enabled:
            self.create_subscription(
                Odometry, "/odom", self._on_odom, qos_profile_sensor_data
            )
        self.create_subscription(LaserScan, "/scan", self._on_scan, qos_profile_sensor_data)
        self.create_subscription(Twist, "/cmd_vel", self._on_cmd_vel, 20)
        # A priority mux sends manual commands around the Nav2 smoother, then
        # through the collision monitor. A direct publisher is a fallback when
        # the dashboard is used without Nav2.
        self.keyboard_mux_velocity = self.create_publisher(
            Twist, "/cmd_vel_keyboard", 10
        )
        self.keyboard_direct_velocity = self.create_publisher(Twist, "/cmd_vel", 10)
        self.gz_node = GazeboNode()
        if not self.gz_node.subscribe(
            Pose_V, "/world/world_demo/pose/info", self._on_world
        ):
            raise RuntimeError("không thể subscribe Gazebo pose/info")
        projection_state = "enabled (telemetry only)" if odom_projection_enabled else "disabled"
        self.get_logger().info(
            f"Dashboard camera={camera_topic}; raw camera-only V-JEPA; "
            f"odom projection is {projection_state}"
        )

    def _refresh_live_vjepa_locked(self) -> None:
        raw = self.raw_vjepa_pose
        if raw is None:
            return
        if not self.odom_projection_enabled:
            self.projected_vjepa_pose = None
            return
        live = raw
        if self.odom_history:
            odom_at_visual_time = min(
                self.odom_history,
                key=lambda item: abs(item.timestamp - raw.timestamp),
            )
            current_odom = self.odom_history[-1]
            if abs(odom_at_visual_time.timestamp - raw.timestamp) <= 0.35:
                live = project_pose_with_odometry(
                    raw, odom_at_visual_time, current_odom
                )
        self.projected_vjepa_pose = live

    def _on_world(self, message: Pose_V) -> None:
        timestamp = gazebo_timestamp(message)
        if timestamp <= 0.0:
            timestamp = self.get_clock().now().nanoseconds * 1e-9
        truth = None
        entities: list[EntityPose] = []
        for pose in message.pose:
            if pose.name == "warehouse_agv":
                q = pose.orientation
                truth = PoseSample(
                    timestamp,
                    float(pose.position.x),
                    float(pose.position.y),
                    float(pose.position.z),
                    yaw_from_quaternion(q.x, q.y, q.z, q.w),
                )
            elif pose.name.startswith(("road_box_static_", "random_worker_")):
                entities.append(
                    EntityPose(pose.name, float(pose.position.x), float(pose.position.y))
                )
        if truth is None:
            return
        with self.lock:
            if self.stream_start_timestamp is None:
                self.stream_start_timestamp = timestamp
            elif timestamp < self.stream_start_timestamp:
                # A restarted Gazebo stream starts a fresh live QA rotation.
                self.stream_start_timestamp = timestamp
            self.truth_history.append(truth)
            self.current_truth = truth
            if (
                not self.truth_trail
                or math.hypot(
                    truth.x - self.truth_trail[-1][0],
                    truth.y - self.truth_trail[-1][1],
                )
                >= 0.05
            ):
                self.truth_trail.append((truth.x, truth.y))
            self.entities = tuple(entities)
            self._refresh_live_vjepa_locked()

    def _on_odom(self, message: Odometry) -> None:
        timestamp = (
            float(message.header.stamp.sec)
            + float(message.header.stamp.nanosec) * 1e-9
        )
        q = message.pose.pose.orientation
        odom = PoseSample(
            timestamp,
            float(message.pose.pose.position.x),
            float(message.pose.pose.position.y),
            float(message.pose.pose.position.z),
            yaw_from_quaternion(q.x, q.y, q.z, q.w),
        )
        with self.lock:
            self.odom_history.append(odom)
            self._refresh_live_vjepa_locked()

    def _on_scan(self, message: LaserScan) -> None:
        clearance = scan_sector_min(
            message.ranges,
            angle_min=float(message.angle_min),
            angle_increment=float(message.angle_increment),
            range_min=float(message.range_min),
            range_max=float(message.range_max),
            half_width_rad=0.65,
        )
        with self.lock:
            self.front_clearance = clearance

    def _on_cmd_vel(self, message: Twist) -> None:
        with self.lock:
            self.command_linear = float(message.linear.x)
            self.command_angular = float(message.angular.z)

    def _publish_keyboard_velocity(self, linear: float, angular: float) -> None:
        command = Twist()
        command.linear.x = float(linear)
        command.angular.z = float(angular)
        if self.keyboard_mux_velocity.get_subscription_count() > 0:
            self.keyboard_mux_velocity.publish(command)
        else:
            self.keyboard_direct_velocity.publish(command)

    def handle_keyboard_key(self, key: int) -> bool:
        """Translate an OpenCV window key into safe differential-drive teleop."""
        if key == 27:
            self.stop_keyboard()
            return False
        commands = {
            ord("w"): (1.00, 0.0), ord("W"): (1.00, 0.0),
            ord("s"): (-0.50, 0.0), ord("S"): (-0.50, 0.0),
            ord("a"): (0.0, 0.75), ord("A"): (0.0, 0.75),
            ord("d"): (0.0, -0.75), ord("D"): (0.0, -0.75),
            ord("q"): (0.65, 0.60), ord("Q"): (0.65, 0.60),
            ord("e"): (0.65, -0.60), ord("E"): (0.65, -0.60),
            ord("z"): (-0.40, -0.55), ord("Z"): (-0.40, -0.55),
            ord("c"): (-0.40, 0.55), ord("C"): (-0.40, 0.55),
            32: (0.0, 0.0),
            65362: (1.00, 0.0), 2490368: (1.00, 0.0),
            65364: (-0.50, 0.0), 2621440: (-0.50, 0.0),
            65361: (0.0, 0.75), 2424832: (0.0, 0.75),
            65363: (0.0, -0.75), 2555904: (0.0, -0.75),
        }
        command = commands.get(key)
        if command is None:
            return True
        self.keyboard_linear, self.keyboard_angular = command
        self.keyboard_active = command != (0.0, 0.0)
        self._publish_keyboard_velocity(*command)
        return True

    def refresh_keyboard_command(self) -> None:
        """Keep a selected manual command alive until Space or Esc is pressed."""
        if self.keyboard_active:
            self._publish_keyboard_velocity(
                self.keyboard_linear, self.keyboard_angular
            )

    def update_held_keyboard(self, pressed: set[str]) -> bool:
        """Publish a vehicle-style command from the currently held keys."""
        if "Escape" in pressed:
            self.stop_keyboard()
            return False
        linear, angular = differential_keyboard_command(
            forward="w" in pressed,
            backward="s" in pressed,
            left="a" in pressed,
            right="d" in pressed,
        )
        if "space" in pressed:
            linear, angular = 0.0, 0.0
        if linear == 0.0 and angular == 0.0:
            self.stop_keyboard()
            return True
        self.keyboard_linear = linear
        self.keyboard_angular = angular
        self.keyboard_active = True
        self._publish_keyboard_velocity(linear, angular)
        return True

    def stop_keyboard(self) -> None:
        if self.keyboard_active:
            self._publish_keyboard_velocity(0.0, 0.0)
        self.keyboard_active = False
        self.keyboard_linear = 0.0
        self.keyboard_angular = 0.0

    def _on_debug(self, message: String) -> None:
        try:
            value = json.loads(message.data)
        except json.JSONDecodeError:
            return
        with self.lock:
            self.debug = value

    def _on_camera(self, message: Image) -> None:
        try:
            rgb = image_message_to_rgb(message)
        except ValueError:
            return
        camera_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        camera_timestamp = message_timestamp_sec(message)
        with self.lock:
            self.camera_bgr = camera_bgr
            self.camera_timestamp = camera_timestamp
        self.camera_person_detector.submit(camera_bgr, camera_timestamp)

    def _on_latent(self, message: Float32MultiArray) -> None:
        latent = np.asarray(message.data, dtype=np.float32)
        if latent.ndim != 1 or latent.size < 2 or not np.isfinite(latent).all():
            return
        with self.lock:
            self.query_latent = latent
            self.query_latent_trail.append(latent.copy())

    def _on_nav_status(self, message: String) -> None:
        try:
            value = json.loads(message.data)
        except json.JSONDecodeError:
            return
        with self.lock:
            self.nav_status = value

    def _on_behavior_decision(self, message: String) -> None:
        try:
            value = json.loads(message.data)
        except json.JSONDecodeError:
            return
        with self.lock:
            self.behavior_decision = value

    def _on_mission_state(self, message: String) -> None:
        try:
            value = json.loads(message.data)
        except json.JSONDecodeError:
            value = {"state": message.data.strip() or "WAITING"}
        with self.lock:
            self.mission_state = value

    def _on_latent_prediction(self, message: String) -> None:
        try:
            value = json.loads(message.data)
        except json.JSONDecodeError:
            return
        with self.lock:
            self.latent_prediction = value

    def _on_plan(self, message: NavPath) -> None:
        points = tuple(
            (float(item.pose.position.x), float(item.pose.position.y))
            for item in message.poses
        )
        with self.lock:
            self.astar_plan = points
            self.astar_plan_received = time.monotonic()

    def _on_action_status(
        self, action_name: str, message: GoalStatusArray
    ) -> None:
        with self.lock:
            self.nav_goal_active[action_name] = (
                goal_status_array_has_active_goal(message)
            )

    def _on_vjepa(self, message: PoseWithCovarianceStamped) -> None:
        timestamp = ros_timestamp(message)
        q = message.pose.pose.orientation
        prediction = PoseSample(
            timestamp,
            float(message.pose.pose.position.x),
            float(message.pose.pose.position.y),
            float(message.pose.pose.position.z),
            yaw_from_quaternion(q.x, q.y, q.z, q.w),
        )
        with self.lock:
            if not self.truth_history:
                return
            truth = min(
                self.truth_history,
                key=lambda item: abs(item.timestamp - timestamp),
            )
            raw_dx = prediction.x - truth.x
            raw_dy = prediction.y - truth.y
            self.raw_vjepa_pose = prediction
            self.vjepa_pose = prediction
            self.aligned_truth = truth
            self.raw_position_error = math.hypot(raw_dx, raw_dy)
            self.raw_yaw_error = abs(wrap_angle(prediction.yaw - truth.yaw))
            self.relative_dx = raw_dx
            self.relative_dy = raw_dy
            self.position_error = self.raw_position_error
            self.yaw_error = self.raw_yaw_error
            self.time_delta = abs(prediction.timestamp - truth.timestamp)
            if (
                not self.vjepa_trail
                or math.hypot(
                    prediction.x - self.vjepa_trail[-1][0],
                    prediction.y - self.vjepa_trail[-1][1],
                )
                >= 0.05
            ):
                self.vjepa_trail.append((prediction.x, prediction.y))
            self._refresh_live_vjepa_locked()
            self.sequence += 1

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            current = self.current_truth
            prediction = self.vjepa_pose
            raw_prediction = self.raw_vjepa_pose
            projected_prediction = self.projected_vjepa_pose
            aligned = self.aligned_truth
            entities = self.entities
            clearance = self.front_clearance
            angular = self.command_angular
            linear = self.command_linear
            debug = dict(self.debug)
            nav_status = dict(self.nav_status)
            behavior_decision = dict(self.behavior_decision)
            mission_state = dict(self.mission_state)
            latent_prediction = dict(self.latent_prediction)
            camera_bgr = None if self.camera_bgr is None else self.camera_bgr.copy()
            camera_timestamp = self.camera_timestamp
            camera_person_detection = self.camera_person_detector.snapshot(
                camera_timestamp
            )
            query_latent = (
                None if self.query_latent is None else self.query_latent.copy()
            )
            query_latent_trail = tuple(
                item.copy() for item in self.query_latent_trail
            )
            # This demo computes A* once per goal, so /plan is intentionally
            # not republished. Keep that path for the action's whole lifetime;
            # the short grace covers the plan/status callback startup race.
            astar_plan = plan_for_display(
                self.astar_plan,
                received_at=self.astar_plan_received,
                nav_goal_active=any(self.nav_goal_active.values()),
                now=time.monotonic(),
            )
            snapshot = {
                "sequence": self.sequence
                + int(camera_person_detection.get("revision", 0)),
                "current_truth": current,
                "stream_elapsed_s": (
                    max(0.0, current.timestamp - self.stream_start_timestamp)
                    if current is not None and self.stream_start_timestamp is not None
                    else 0.0
                ),
                "aligned_truth": aligned,
                "vjepa": prediction,
                "raw_vjepa": raw_prediction,
                "projected_vjepa": projected_prediction,
                "relative_dx": self.relative_dx,
                "relative_dy": self.relative_dy,
                "position_error": self.position_error,
                "yaw_error": self.yaw_error,
                "raw_position_error": self.raw_position_error,
                "raw_yaw_error": self.raw_yaw_error,
                "time_delta": self.time_delta,
                "truth_trail": tuple(self.truth_trail),
                "vjepa_trail": tuple(self.vjepa_trail),
                "astar_plan": astar_plan,
                "front_clearance": clearance,
                "top1_similarity": debug.get("top1_similarity"),
                "confidence_margin": debug.get("confidence_margin"),
                "tracking_state": debug.get("tracking_state", "WAITING"),
                "camera_moving": debug.get("camera_moving", False),
                "camera_motion_score": debug.get("camera_motion_score"),
                "camera_pixel_change": debug.get("camera_pixel_change"),
                "camera_motion_inlier_ratio": debug.get("camera_motion_inlier_ratio"),
                "camera_progress_scale": debug.get("camera_progress_scale"),
                "motion_credit": debug.get("motion_credit"),
                "raw_jump_m": debug.get("raw_jump_m"),
                "accepted_step_m": debug.get("accepted_step_m"),
                "translation_gate_m": debug.get("translation_gate_m"),
                "selected_rank": debug.get("selected_rank"),
                "rejected_streak": debug.get("rejected_streak", 0),
                "nav_localization_state": nav_status.get("state", "WAITING"),
                "nav_localization_source": nav_status.get("source", "WAITING"),
                "nav_uses_gazebo_truth": nav_status.get("uses_gazebo_truth"),
                "nav_planner": nav_status.get("planner", "NAVFN_ASTAR"),
                "nav_correction_m": nav_status.get("correction_m"),
                "behavior_decision": behavior_decision,
                "mission_state": mission_state,
                "latent_prediction": latent_prediction,
                "camera_bgr": camera_bgr,
                "camera_timestamp": camera_timestamp,
                "camera_person_detection": camera_person_detection,
                "query_latent": query_latent,
                "query_latent_trail": query_latent_trail,
                "source_id": debug.get("source_id"),
                "latent_dimension": debug.get("latent_dimension"),
                "compute_host": debug.get("compute_host", "waiting"),
                "inference_ms": debug.get("inference_ms"),
                "result_pose_topic": debug.get("pose_topic", "/vjepa_pose"),
                "result_latent_topic": debug.get("latent_topic", "/vjepa_latent"),
                "odom_projection_enabled": self.odom_projection_enabled,
            }
        obstacle = None
        if current is not None:
            obstacle = identify_forward_obstacle(
                agv_x=current.x,
                agv_y=current.y,
                agv_yaw=current.yaw,
                entities=entities,
                lidar_clearance_m=clearance,
                detection_distance_m=DASHBOARD_OBSTACLE_DISTANCE_M,
                front_half_angle_rad=0.65,
            )
            snapshot["area"] = self.regions.describe(current.x, current.y)
        else:
            snapshot["area"] = "đang chờ Gazebo truth"
        snapshot["obstacle"] = obstacle
        snapshot["current_display_gap"] = (
            math.hypot(prediction.x - current.x, prediction.y - current.y)
            if prediction is not None and current is not None
            else None
        )
        snapshot["vjepa_age_ms"] = (
            max(0.0, current.timestamp - prediction.timestamp) * 1000.0
            if prediction is not None and current is not None
            else None
        )
        snapshot["linear_x"] = linear
        snapshot["angular_z"] = angular
        # Gazebo pose and camera stamps share simulation time even when this
        # standalone evaluator was started without use_sim_time. Prefer that
        # common clock so the on-screen latency never mixes wall and sim time.
        now = (
            current.timestamp
            if current is not None and current.timestamp > 0.0
            else 0.0
        )
        snapshot["camera_age_ms"] = (
            max(0.0, now - camera_timestamp) * 1000.0
            if camera_timestamp > 0.0 and now > 0.0
            else None
        )
        nearby_people: list[tuple[float, str]] = []
        people_ahead: list[tuple[float, str]] = []
        if current is not None:
            for entity in entities:
                if not entity.name.startswith("random_worker_"):
                    continue
                dx, dy = entity.x - current.x, entity.y - current.y
                distance = math.hypot(dx, dy)
                if distance <= 4.0:
                    nearby_people.append((distance, entity.name))
                forward = math.cos(current.yaw) * dx + math.sin(current.yaw) * dy
                lateral = abs(-math.sin(current.yaw) * dx + math.cos(current.yaw) * dy)
                if 0.0 < forward <= 3.5 and lateral <= 1.5:
                    people_ahead.append((distance, entity.name))
        snapshot["nearby_people"] = tuple(sorted(nearby_people))
        snapshot["people_ahead"] = tuple(sorted(people_ahead))
        snapshot["camera_frustum_people"] = (
            people_in_camera_frustum(
                agv_x=current.x,
                agv_y=current.y,
                agv_yaw=current.yaw,
                entities=entities,
            )
            if current is not None
            else ()
        )
        return snapshot

    def close(self) -> None:
        self.camera_person_detector.close()


class LatentProjector:
    """Project the saved map and live 1024-D V-JEPA vector into one PCA plane."""

    def __init__(self, map_directory: Path) -> None:
        embeddings = np.load(map_directory / "global_embeddings.npy").astype(np.float32)
        ids = np.load(map_directory / "ids.npy", allow_pickle=False)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-8)
        self.mean = embeddings.mean(axis=0)
        centered = embeddings - self.mean
        _, _, basis = np.linalg.svd(centered, full_matrices=False)
        self.basis = basis[:2]
        self.map_points = centered @ self.basis.T
        self.id_to_index = {str(value): index for index, value in enumerate(ids)}
        lower = np.quantile(self.map_points, 0.01, axis=0)
        upper = np.quantile(self.map_points, 0.99, axis=0)
        padding = np.maximum((upper - lower) * 0.12, 1e-3)
        self.lower = lower - padding
        self.upper = upper + padding
        self.dimension = int(embeddings.shape[1])

    def project(self, latent: np.ndarray | None) -> np.ndarray | None:
        if latent is None or latent.shape != (self.dimension,):
            return None
        vector = latent.astype(np.float32)
        vector /= max(float(np.linalg.norm(vector)), 1e-8)
        return (vector - self.mean) @ self.basis.T

    def selected(self, source_id: Any) -> np.ndarray | None:
        index = self.id_to_index.get(str(source_id))
        return None if index is None else self.map_points[index]

    def to_pixel(
        self, point: np.ndarray, x: int, y: int, width: int, height: int
    ) -> tuple[int, int]:
        normalized = (point - self.lower) / np.maximum(self.upper - self.lower, 1e-8)
        px = x + int(np.clip(normalized[0], 0.0, 1.0) * width)
        py = y + height - int(np.clip(normalized[1], 0.0, 1.0) * height)
        return px, py

    def draw(
        self,
        canvas: np.ndarray,
        bounds: tuple[int, int, int, int],
        query_latent: np.ndarray | None,
        source_id: Any,
        *,
        question_id: str,
        query_latent_trail: tuple[np.ndarray, ...] = (),
        latent_prediction: dict[str, Any] | None = None,
    ) -> None:
        x, y, width, height = bounds
        profile = latent_evidence_profile(question_id)
        profile_titles = {
            "temporal_motion": "TEMPORAL MOTION",
            "future_rollout": "FUTURE z(t+1..3)",
            "localization_match": "LOCALIZATION MATCH",
            "current_observation": "CURRENT OBSERVATION",
        }
        cv2.rectangle(canvas, (x, y), (x + width, y + height), (244, 244, 240), -1)
        cv2.rectangle(canvas, (x, y), (x + width, y + height), (92, 98, 108), 1)
        for point in self.map_points:
            cv2.circle(
                canvas,
                self.to_pixel(point, x + 18, y + 40, width - 36, height - 62),
                1,
                (145, 150, 155),
                -1,
                cv2.LINE_AA,
            )
        cv2.putText(
            canvas,
            f"V-JEPA LATENT - {question_id.upper()} - {profile_titles[profile]}",
            (x + 16, y + 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.53,
            (45, 48, 54),
            1,
            cv2.LINE_AA,
        )
        plot = (x + 18, y + 40, width - 36, height - 62)
        instant = self.project(query_latent)
        stable = self.selected(source_id)
        if profile == "temporal_motion":
            trail = [
                projected
                for projected in (
                    self.project(np.asarray(item, dtype=np.float32))
                    for item in query_latent_trail
                )
                if projected is not None
            ]
            if len(trail) >= 2:
                pixels = np.asarray(
                    [self.to_pixel(point, *plot) for point in trail],
                    dtype=np.int32,
                )
                cv2.polylines(
                    canvas, [pixels], False, (175, 110, 65), 2, cv2.LINE_AA
                )
        if instant is not None:
            cv2.circle(canvas, self.to_pixel(instant, *plot), 7, (45, 45, 235), -1, cv2.LINE_AA)
        if stable is not None and profile in {
            "current_observation",
            "localization_match",
        }:
            cv2.circle(canvas, self.to_pixel(stable, *plot), 7, (235, 90, 35), -1, cv2.LINE_AA)
        if instant is not None and stable is not None and profile in {
            "current_observation",
            "localization_match",
        }:
            cv2.line(
                canvas,
                self.to_pixel(instant, *plot),
                self.to_pixel(stable, *plot),
                (170, 135, 115),
                1,
                cv2.LINE_AA,
            )
        if profile == "future_rollout" and instant is not None:
            rollout_colors = {
                1: (35, 170, 235),
                2: (55, 180, 75),
                3: (200, 85, 155),
            }
            previous = instant
            for item in (latent_prediction or {}).get("horizons", []):
                if not isinstance(item, dict):
                    continue
                try:
                    step = int(item["step"])
                    dimension = int(item["latent_dimension"])
                    packed = base64.b64decode(
                        str(item["predicted_latent_f16_b64"]), validate=True
                    )
                    vector = np.frombuffer(packed, dtype="<f2").astype(np.float32)
                    if item.get("latent_encoding") != "float16_base64":
                        continue
                    if vector.size != dimension:
                        continue
                except (KeyError, TypeError, ValueError):
                    continue
                predicted = self.project(vector)
                if predicted is None:
                    continue
                color = rollout_colors.get(step, (90, 90, 90))
                cv2.arrowedLine(
                    canvas,
                    self.to_pixel(previous, *plot),
                    self.to_pixel(predicted, *plot),
                    color,
                    2,
                    cv2.LINE_AA,
                    tipLength=0.18,
                )
                pixel = self.to_pixel(predicted, *plot)
                cv2.circle(canvas, pixel, 7, color, -1, cv2.LINE_AA)
                cv2.putText(
                    canvas,
                    f"t+{step}",
                    (pixel[0] + 8, pixel[1] - 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    color,
                    1,
                    cv2.LINE_AA,
                )
                previous = predicted


def normalize_embedding(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(value))
    return value / norm if norm > 1.0e-8 else value.copy()


class SemanticTextEncoder:
    """Encode Vietnamese QA text, with a deterministic offline fallback.

    The preferred backend is a real multilingual sentence encoder. Feature
    hashing remains available so the controlled demo still starts without a
    network/cache; it is never presented as an official VL-JEPA checkpoint.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.requested_backend = str(config.get("backend", "auto")).lower()
        self.checkpoint = str(
            config.get("checkpoint", "intfloat/multilingual-e5-small")
        )
        self.device_name = str(config.get("device", "cpu"))
        self.allow_download = bool(config.get("allow_download", True))
        self.dimension = int(config.get("fallback_dimension", 384))
        self.backend = "uninitialized"
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None
        self._cache: dict[tuple[str, str], np.ndarray] = {}

    def _load(self) -> None:
        if self.backend != "uninitialized":
            return
        if self.requested_backend == "hash":
            self.backend = "feature_hash"
            return
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.checkpoint,
                local_files_only=not self.allow_download,
            )
            self._model = AutoModel.from_pretrained(
                self.checkpoint,
                local_files_only=not self.allow_download,
            )
            self._model.to(torch.device(self.device_name))
            self._model.eval()
            for parameter in self._model.parameters():
                parameter.requires_grad_(False)
            self._torch = torch
            self.dimension = int(self._model.config.hidden_size)
            self.backend = "multilingual_e5"
        except (ImportError, OSError, RuntimeError, ValueError):
            self.backend = "feature_hash"
            self._tokenizer = None
            self._model = None
            self._torch = None

    def _hash_encode(self, text: str, kind: str) -> np.ndarray:
        normalized = unicodedata.normalize("NFC", text.strip().lower())
        padded = f" {kind}:{normalized} "
        features = normalized.split()
        features.extend(
            padded[index : index + size]
            for size in (2, 3, 4)
            for index in range(max(0, len(padded) - size + 1))
        )
        vector = np.zeros(self.dimension, dtype=np.float32)
        for feature in features:
            digest = hashlib.blake2b(
                feature.encode("utf-8"), digest_size=8, person=b"qa-latent"
            ).digest()
            value = int.from_bytes(digest, "little")
            vector[value % self.dimension] += 1.0 if value & 1 else -1.0
        return normalize_embedding(vector)

    def _model_encode(self, texts: list[str], kind: str) -> np.ndarray:
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._torch is not None
        prefix = "query: " if kind == "query" else "passage: "
        batches: list[np.ndarray] = []
        for offset in range(0, len(texts), 64):
            encoded = self._tokenizer(
                [prefix + text for text in texts[offset : offset + 64]],
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            encoded = {
                key: value.to(self.device_name) for key, value in encoded.items()
            }
            with self._torch.inference_mode():
                output = self._model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(output.dtype)
            pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            values = pooled.detach().float().cpu().numpy().astype(np.float32)
            batches.append(
                np.stack([normalize_embedding(row) for row in values])
            )
        return np.concatenate(batches, axis=0)

    def encode(self, texts: tuple[str, ...], *, kind: str) -> np.ndarray:
        self._load()
        result: list[np.ndarray | None] = [None] * len(texts)
        missing_texts: list[str] = []
        missing_indices: list[int] = []
        for index, text in enumerate(texts):
            key = (kind, text)
            cached = self._cache.get(key)
            if cached is None:
                missing_texts.append(text)
                missing_indices.append(index)
            else:
                result[index] = cached
        if missing_texts:
            if self.backend == "multilingual_e5":
                encoded = self._model_encode(missing_texts, kind)
            else:
                encoded = np.stack(
                    [self._hash_encode(text, kind) for text in missing_texts]
                )
            for index, text, vector in zip(
                missing_indices, missing_texts, encoded, strict=True
            ):
                self._cache[(kind, text)] = vector
                result[index] = vector
        return np.stack([value for value in result if value is not None])


class QueryConditionedAnswerLatent:
    """Controlled-demo approximation of VL-JEPA answer-embedding inference."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        encoder: SemanticTextEncoder | None = None,
    ) -> None:
        self.encoder = encoder or SemanticTextEncoder(config)
        self.oracle_weight = float(config.get("oracle_weight", 0.92))
        self.uncertain_oracle_weight = float(
            config.get("uncertain_oracle_weight", 0.20)
        )
        self.query_weight = float(config.get("query_weight", 0.05))
        self.visual_weight = float(config.get("visual_weight", 0.03))
        total = self.oracle_weight + self.query_weight + self.visual_weight
        if total <= 0.0:
            raise ValueError("answer latent weights must have a positive sum")
        self.oracle_weight /= total
        self.query_weight /= total
        self.visual_weight /= total
        self.selective_threshold = float(
            config.get("selective_decode_cosine", 0.995)
        )
        self.manifold_neighbors = max(3, int(config.get("manifold_neighbors", 12)))
        self.manifold_temperature = max(
            1.0e-3, float(config.get("manifold_temperature", 0.035))
        )
        self.manifold_iterations = max(
            100, int(config.get("manifold_iterations", 700))
        )
        self.manifold_learning_rate = max(
            1.0, float(config.get("manifold_learning_rate", 80.0))
        )
        self.display_spring_stiffness = max(
            1.0, float(config.get("display_spring_stiffness", 30.0))
        )
        self.display_spring_damping = max(
            1.0, float(config.get("display_spring_damping", 11.0))
        )
        self.previous: dict[str, np.ndarray] = {}
        self.previous_answer: dict[str, str] = {}
        self.trails: dict[str, deque[np.ndarray]] = {}
        self.projections: dict[tuple[int, int], np.ndarray] = {}
        self.cached_report: dict[tuple[str, int], dict[str, Any]] = {}
        self.answer_space_embeddings: np.ndarray | None = None
        self.answer_space_points: np.ndarray | None = None
        self.answer_space_texts: tuple[str, ...] = ()
        self.answer_space_lower: np.ndarray | None = None
        self.answer_space_upper: np.ndarray | None = None

    def _ensure_answer_space(self) -> None:
        if self.answer_space_points is not None:
            return
        texts = expanded_answer_bank()
        embeddings = self.encoder.encode(texts, kind="answer")
        count = len(embeddings)
        neighbors = min(self.manifold_neighbors, count - 1)
        similarities = embeddings @ embeddings.T
        np.fill_diagonal(similarities, -np.inf)
        nearest = np.argpartition(similarities, -neighbors, axis=1)[:, -neighbors:]
        rows = np.arange(count)[:, None]
        local_similarity = similarities[rows, nearest]
        weights = np.exp(
            (local_similarity - local_similarity.max(axis=1, keepdims=True))
            / self.manifold_temperature
        ).astype(np.float64)
        probability = np.zeros((count, count), dtype=np.float64)
        probability[rows, nearest] = weights
        probability += probability.T
        probability /= max(float(probability.sum()), 1.0e-12)

        rng = np.random.default_rng(20260826)
        points = rng.normal(0.0, 1.0e-4, (count, 2))
        velocity = np.zeros_like(points)
        exaggeration_steps = min(120, self.manifold_iterations // 3)
        for iteration in range(self.manifold_iterations):
            difference = points[:, None, :] - points[None, :, :]
            squared_distance = np.sum(difference * difference, axis=2)
            student = 1.0 / (1.0 + squared_distance)
            np.fill_diagonal(student, 0.0)
            low_probability = student / max(float(student.sum()), 1.0e-12)
            exaggeration = 8.0 if iteration < exaggeration_steps else 1.0
            force = (exaggeration * probability - low_probability) * student
            gradient = 4.0 * (
                force.sum(axis=1, keepdims=True) * points - force @ points
            )
            momentum = 0.5 if iteration < exaggeration_steps else 0.8
            velocity = (
                momentum * velocity - self.manifold_learning_rate * gradient
            )
            points += velocity
            points -= points.mean(axis=0, keepdims=True)
            if iteration in {exaggeration_steps - 1, self.manifold_iterations // 2}:
                velocity *= 0.0
        points = points.astype(np.float32)
        for axis in range(2):
            pivot = int(np.argmax(np.abs(points[:, axis])))
            if points[pivot, axis] < 0.0:
                points[:, axis] *= -1.0
        points /= np.maximum(points.std(axis=0, keepdims=True), 1.0e-6)
        self.answer_space_embeddings = embeddings
        self.answer_space_points = points
        self.answer_space_texts = texts
        lower = np.quantile(self.answer_space_points, 0.01, axis=0)
        upper = np.quantile(self.answer_space_points, 0.99, axis=0)
        padding = np.maximum((upper - lower) * 0.10, 1.0e-3)
        self.answer_space_lower = lower - padding
        self.answer_space_upper = upper + padding

    def _project_answer_space(self, embeddings: np.ndarray) -> np.ndarray:
        self._ensure_answer_space()
        assert self.answer_space_embeddings is not None
        assert self.answer_space_points is not None
        values = np.asarray(embeddings, dtype=np.float32)
        similarities = values @ self.answer_space_embeddings.T
        neighbors = min(self.manifold_neighbors, len(self.answer_space_embeddings))
        nearest = np.argpartition(similarities, -neighbors, axis=1)[:, -neighbors:]
        rows = np.arange(len(values))[:, None]
        local_similarity = similarities[rows, nearest]
        weights = np.exp(
            (local_similarity - local_similarity.max(axis=1, keepdims=True))
            / self.manifold_temperature
        )
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1.0e-8)
        return np.sum(
            self.answer_space_points[nearest] * weights[:, :, None], axis=1
        )

    def _nearest_answer_space_point(
        self, embedding: np.ndarray
    ) -> tuple[np.ndarray, int]:
        self._ensure_answer_space()
        assert self.answer_space_embeddings is not None
        assert self.answer_space_points is not None
        similarities = self.answer_space_embeddings @ normalize_embedding(embedding)
        index = int(np.argmax(similarities))
        return self.answer_space_points[index].copy(), index

    def _visual_projection(
        self, visual: np.ndarray | None, output_dimension: int
    ) -> np.ndarray:
        if visual is None:
            return np.zeros(output_dimension, dtype=np.float32)
        value = np.asarray(visual, dtype=np.float32).reshape(-1)
        if value.size == 0 or not np.isfinite(value).all():
            return np.zeros(output_dimension, dtype=np.float32)
        key = (value.size, output_dimension)
        projection = self.projections.get(key)
        if projection is None:
            rng = np.random.default_rng(0x564C4A45 + value.size)
            projection = rng.standard_normal(key, dtype=np.float32)
            projection /= math.sqrt(float(value.size))
            self.projections[key] = projection
        return normalize_embedding(normalize_embedding(value) @ projection)

    def infer(
        self,
        *,
        question_id: str,
        question: str,
        policy_answer: str,
        stabilized_answer: str | None = None,
        visual_embedding: np.ndarray | None,
        sequence: int,
        ambiguity: float = 0.0,
    ) -> dict[str, Any]:
        cache_key = (question_id, int(sequence))
        cached = self.cached_report.get(cache_key)
        if cached is not None:
            return cached
        candidates = tuple(
            dict.fromkeys(
                (policy_answer,) + answer_candidates(question_id)
            )
        )
        candidate_vectors = self.encoder.encode(candidates, kind="answer")
        stabilized_text = stabilized_answer or policy_answer
        stabilized_vector = self.encoder.encode(
            (stabilized_text,), kind="answer"
        )[0]
        query_vector = self.encoder.encode((question,), kind="query")[0]
        visual_vector = self._visual_projection(
            visual_embedding, candidate_vectors.shape[1]
        )
        ambiguity = min(1.0, max(0.0, float(ambiguity)))
        effective_oracle = (
            (1.0 - ambiguity) * self.oracle_weight
            + ambiguity * self.uncertain_oracle_weight
        )
        non_oracle = max(0.0, 1.0 - effective_oracle)
        query_visual_total = max(self.query_weight + self.visual_weight, 1.0e-8)
        effective_query = non_oracle * self.query_weight / query_visual_total
        effective_visual = non_oracle * self.visual_weight / query_visual_total
        predicted = normalize_embedding(
            effective_oracle * candidate_vectors[0]
            + effective_query * query_vector
            + effective_visual * visual_vector
        )
        similarities = candidate_vectors @ predicted
        selected_index = int(np.argmax(similarities))

        previous = self.previous.get(question_id)
        semantic_cosine = (
            float(np.dot(previous, predicted)) if previous is not None else None
        )
        selected_answer = candidates[selected_index]
        decoded = (
            previous is None
            or semantic_cosine is None
            or semantic_cosine < self.selective_threshold
            or self.previous_answer.get(question_id) != selected_answer
        )
        self.previous[question_id] = predicted.copy()
        self.previous_answer[question_id] = selected_answer
        trail = self.trails.setdefault(question_id, deque(maxlen=20))
        trail.append(predicted.copy())
        self._ensure_answer_space()
        assert self.answer_space_points is not None
        assert self.answer_space_lower is not None
        assert self.answer_space_upper is not None
        answer_space_points = self.answer_space_points.copy()
        candidate_points = self._project_answer_space(candidate_vectors)
        predicted_point = self._project_answer_space(predicted[None, :])[0]
        stabilized_point, stabilized_anchor_index = (
            self._nearest_answer_space_point(stabilized_vector)
        )
        trajectory_points = self._project_answer_space(np.asarray(trail))
        report = {
            "question_id": question_id,
            "query_embedding": query_vector,
            "predicted_embedding": predicted,
            "candidate_embeddings": candidate_vectors,
            "candidate_answers": candidates,
            "candidate_similarities": similarities,
            "selected_index": selected_index,
            "selected_answer": selected_answer,
            "instant_reference_answer": policy_answer,
            "stabilized_answer": stabilized_text,
            "stabilized_embedding": stabilized_vector,
            "confidence": float(similarities[selected_index]),
            "prediction_matches_stabilized": selected_answer == stabilized_text,
            "prediction_matches_instant": selected_answer == policy_answer,
            "observation_ambiguity": ambiguity,
            "effective_oracle_weight": effective_oracle,
            "trajectory": tuple(item.copy() for item in trail),
            "answer_space_points_2d": answer_space_points,
            "answer_space_answers": self.answer_space_texts,
            "candidate_points_2d": candidate_points,
            "predicted_point_2d": predicted_point,
            "stabilized_point_2d": stabilized_point,
            "stabilized_anchor_index": stabilized_anchor_index,
            "trajectory_points_2d": trajectory_points,
            "answer_space_lower_2d": self.answer_space_lower.copy(),
            "answer_space_upper_2d": self.answer_space_upper.copy(),
            "display_spring_stiffness": self.display_spring_stiffness,
            "display_spring_damping": self.display_spring_damping,
            "semantic_cosine": semantic_cosine,
            "decoded": decoded,
            "backend": self.encoder.backend,
            "provenance": "vjepa_visual_plus_query_conditioned_policy_assisted_answer_embedding",
        }
        self.cached_report = {cache_key: report}
        return report


class AnswerLatentRenderer:
    """Render predicted and candidate answer embeddings for the active query."""

    @staticmethod
    def _stabilized_for_display(target: np.ndarray) -> np.ndarray:
        """Snap to each stabilized sample, just like the predicted marker."""
        return target.copy()

    def draw(
        self,
        canvas: np.ndarray,
        bounds: tuple[int, int, int, int],
        report: dict[str, Any] | None,
        question_id: str,
    ) -> None:
        x, y, width, height = bounds
        del question_id
        cv2.rectangle(canvas, (x, y), (x + width, y + height), (244, 244, 240), -1)
        cv2.rectangle(canvas, (x, y), (x + width, y + height), (92, 98, 108), 1)
        cv2.putText(
            canvas,
            "VL-JEPA ANSWER LATENT SPACE",
            (x + 16, y + 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.53,
            (45, 48, 54),
            1,
            cv2.LINE_AA,
        )
        if not report:
            cv2.putText(
                canvas,
                "Waiting for answer embeddings...",
                (x + 135, y + height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (90, 95, 105),
                1,
                cv2.LINE_AA,
            )
            return
        space = np.asarray(report["answer_space_points_2d"], dtype=np.float32)
        predicted = np.asarray(report["predicted_point_2d"], dtype=np.float32)
        stabilized_target = np.asarray(
            report["stabilized_point_2d"], dtype=np.float32
        )
        stabilized = self._stabilized_for_display(stabilized_target)
        trajectory = np.asarray(report["trajectory_points_2d"], dtype=np.float32)
        lower = np.asarray(report["answer_space_lower_2d"], dtype=np.float32)
        upper = np.asarray(report["answer_space_upper_2d"], dtype=np.float32)
        plot = (x + 24, y + 48, width - 48, height - 98)

        def pixel(point: np.ndarray) -> tuple[int, int]:
            normalized = (point - lower) / np.maximum(upper - lower, 1.0e-8)
            return (
                plot[0] + int(np.clip(normalized[0], 0.0, 1.0) * plot[2]),
                plot[1] + plot[3] - int(np.clip(normalized[1], 0.0, 1.0) * plot[3]),
            )

        for point in space:
            center = pixel(point)
            left, top = center[0] - 3, center[1] - 3
            right, bottom = center[0] + 4, center[1] + 4
            roi = canvas[top:bottom, left:right]
            overlay = roi.copy()
            cv2.circle(overlay, (3, 3), 2, (115, 121, 126), -1, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.30, roi, 0.70, 0.0, dst=roi)
        # Keep temporal history in report artifacts, but render exactly one
        # instant prediction marker so the public plot is unambiguous.
        cv2.circle(canvas, pixel(stabilized), 3, (235, 80, 35), -1, cv2.LINE_AA)
        cv2.circle(canvas, pixel(predicted), 3, (45, 45, 235), -1, cv2.LINE_AA)
        legend_y = y + height - 18
        cv2.circle(canvas, (x + 25, legend_y - 4), 3, (45, 45, 235), -1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            "INSTANT / PREDICTED",
            (x + 38, legend_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (45, 48, 54),
            1,
            cv2.LINE_AA,
        )
        cv2.circle(canvas, (x + 205, legend_y - 4), 3, (235, 80, 35), -1, cv2.LINE_AA)
        cv2.putText(
            canvas, "STABILIZED", (x + 218, legend_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (45, 48, 54), 1, cv2.LINE_AA,
        )
        cv2.circle(canvas, (x + 345, legend_y - 4), 2, (165, 169, 172), -1, cv2.LINE_AA)



class PreparedQA:
    """Rotate a fixed, answerable set of warehouse questions.

    The YAML file remains a pool of twenty questions.  A smaller configured
    subset is used for the public dashboard so the operator always sees five
    questions, regardless of the aisle, charging area, or obstacle
    state.  All twenty questions are still evaluated on every stream update,
    which keeps the pool available for later selection without making the
    visible question count depend on scene-gating heuristics.
    """

    def __init__(self, config_path: Path) -> None:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.questions = tuple(config["questions"])
        if not self.questions:
            raise ValueError("live question list is empty")
        self.question_by_id = {
            str(spec["id"]): spec for spec in self.questions
        }
        configured_ids = config.get("universal_question_ids")
        if configured_ids is None:
            configured_ids = [
                str(spec["id"])
                for spec in self.questions[: int(config.get("display_count", 10))]
            ]
        configured_ids = tuple(dict.fromkeys(str(item) for item in configured_ids))
        if not configured_ids:
            raise ValueError("universal question selection is empty")
        missing = tuple(
            question_id
            for question_id in configured_ids
            if question_id not in self.question_by_id
        )
        if missing:
            raise ValueError(
                "universal_question_ids contains unknown question id(s): "
                + ", ".join(missing)
            )
        self.universal_question_ids = configured_ids
        self.universal_questions = tuple(
            self.question_by_id[question_id] for question_id in configured_ids
        )
        if len(self.universal_questions) != 5:
            raise ValueError(
                "Warehouse Questions UI requires exactly five questions"
            )
        missing_choices = tuple(
            question_id
            for question_id in self.universal_question_ids
            if question_id not in WAREHOUSE_MODEL_ANSWER_CHOICES
        )
        if missing_choices:
            raise ValueError(
                "selected live question(s) have no model answer choices: "
                + ", ".join(missing_choices)
            )
        self.interval_sec = float(config.get("interval_sec", 3.5))
        self.stabilize_frames = max(1, int(config.get("stabilize_frames", 2)))
        self.answer_latent = QueryConditionedAnswerLatent(
            dict(config.get("answer_latent", {}))
        )
        self.started = time.monotonic()
        self.last_sequence = -1
        self.latest_stream_elapsed_s = 0.0
        self.instant: dict[str, str] = {}
        self.stable: dict[str, str] = {}
        self.candidate_key: dict[str, str] = {}
        self.candidate_count: dict[str, int] = {}
        self.latest_snapshot: dict[str, Any] | None = None
        self.active_slot: int | None = None
        self.active_spec: dict[str, Any] | None = None
        self.active_index = 0
        self.active_count = len(self.universal_questions)
        self.pool_count = len(self.questions)
        # The dashboard must be useful before the first DDS frame arrives.
        # Keep these answers separate from ``stable`` so normal temporal
        # stabilization semantics remain unchanged after streaming begins.
        self.default_answers = {
            str(spec["id"]): self._answer(
                str(spec["id"]),
                {
                    "area": "khu vực chưa xác định",
                    "linear_x": 0.0,
                    "angular_z": 0.0,
                    "front_clearance": math.inf,
                    "obstacle": None,
                    "behavior_decision": {"decision": "PASS", "occupancy": []},
                    "people_ahead": (),
                    "nearby_people": (),
                },
            )[1]
            for spec in self.questions
        }

    @staticmethod
    def _scope_is_relevant(scope: str, snapshot: dict[str, Any]) -> bool:
        """Gate a prepared question using current scene and world state."""
        area = str(snapshot.get("area", "")).lower()
        linear = abs(float(snapshot.get("linear_x", 0.0)))
        angular = abs(float(snapshot.get("angular_z", 0.0)))
        moving = linear >= 0.03 or angular >= 0.05
        clearance = float(snapshot.get("front_clearance", math.inf))
        obstacle = snapshot.get("obstacle")
        has_obstacle = obstacle is not None or (
            math.isfinite(clearance)
            and clearance <= DASHBOARD_OBSTACLE_DISTANCE_M
        )
        behavior = snapshot.get("behavior_decision") or {}
        decision = str(behavior.get("decision", "PASS"))
        occupied = any(
            bool(item.get("path_occupied"))
            for item in behavior.get("occupancy", [])
            if isinstance(item, dict)
        )
        has_people = bool(
            snapshot.get("camera_frustum_people")
            and (snapshot.get("camera_person_detection") or {}).get("visible")
        )
        in_service_area = "sạc" in area or "đóng gói" in area
        in_aisle = "kệ" in area or "hành lang" in area or "lối đi" in area
        in_corridor = "hành lang" in area or "lối đi" in area
        hazard = moving or has_obstacle or occupied or decision in {"WAIT", "REPLAN"}

        relevance = {
            "always": True,
            "motion": moving,
            "navigation": not in_service_area and in_aisle,
            "approaching_obstacle": has_obstacle and linear >= 0.03,
            "obstacle": has_obstacle,
            "hazard": hazard,
            "avoidance": has_obstacle or angular >= 0.12 or decision in {"WAIT", "REPLAN"},
            "aisle": in_aisle,
            "aisle_motion": in_aisle and moving,
            "people": has_people,
            "floor_marking": in_service_area or in_corridor,
            "shelf": "kệ" in area,
        }
        return relevance.get(scope, True)

    def _eligible_questions(
        self, snapshot: dict[str, Any] | None
    ) -> tuple[dict[str, Any], ...]:
        if snapshot is None:
            return self.questions
        eligible = tuple(
            spec
            for spec in self.questions
            if self._scope_is_relevant(str(spec.get("scope", "always")), snapshot)
        )
        return eligible or self.questions

    @staticmethod
    def _answer(question_id: str, snapshot: dict[str, Any]) -> tuple[str, str]:
        if question_id.startswith("q") and question_id[1:].isdigit():
            return PreparedQA._stream_answer(question_id, snapshot)
        if question_id == "area":
            area = ascii_text(str(snapshot["area"]))
            return area, f"Robot is in {area}."
        if question_id == "obstacle":
            obstacle = snapshot["obstacle"]
            if obstacle is None:
                return "clear", "No obstacle is detected in front."
            label = ascii_text(obstacle.label)
            return obstacle.name, f"Yes. {label} is {obstacle.clearance_m:.1f} m ahead."
        if question_id == "person":
            people = snapshot["people_ahead"] or snapshot["nearby_people"]
            if not people:
                return "no_person", "No person is close to the current route."
            distance, name = people[0]
            number = str(name).removeprefix("random_worker_")
            relation = "ahead" if snapshot["people_ahead"] else "nearby"
            return str(name), f"Worker {number} is {relation}, about {distance:.1f} m away."
        if question_id == "avoidance":
            behavior = snapshot.get("behavior_decision") or {}
            decision = str(behavior.get("decision", "PASS"))
            person = str(behavior.get("person_id", "none"))
            if person != "none" and decision in {"WAIT", "PASS", "REPLAN"}:
                reason = ascii_text(str(behavior.get("reason", "")))
                return f"{decision}:{person}", f"{decision}: {reason}"
            obstacle = snapshot["obstacle"]
            angular = float(snapshot["angular_z"])
            if obstacle is None:
                return "none", "Nothing now; the LiDAR corridor is clear."
            label = ascii_text(obstacle.label)
            if abs(angular) < 0.12:
                return f"wait:{obstacle.name}", f"Waiting for {label} to clear."
            direction = "left" if angular > 0.0 else "right"
            return f"{obstacle.name}:{direction}", f"Turning {direction} to avoid {label}."
        if question_id == "motion":
            linear = float(snapshot["linear_x"])
            angular = float(snapshot["angular_z"])
            if abs(linear) < 0.03 and abs(angular) < 0.05:
                return "stopped", "The robot is stopped."
            if abs(angular) >= 0.12:
                direction = "left" if angular > 0.0 else "right"
                return f"turn:{direction}", f"The robot is moving and turning {direction}."
            return "forward", "The robot is moving forward on the aisle."
        if question_id == "route":
            count = len(snapshot["astar_plan"])
            if count < 2:
                return "no_plan", "A* is waiting for the next route goal."
            return "plan_active", f"Yes. The active A* path has {count} poses."
        if question_id == "tracking":
            state = str(snapshot["tracking_state"])
            similarity = snapshot["top1_similarity"]
            if isinstance(similarity, (int, float)):
                return state, f"Temporal tracking is {state}; similarity is {similarity:.3f}."
            return state, f"Temporal tracking is {state}; waiting for a latent match."
        if question_id == "comparison":
            error = snapshot["position_error"]
            if not isinstance(error, (int, float)):
                return "waiting", "Waiting for timestamp-aligned Gazebo truth."
            band = "low" if error < 0.5 else "medium" if error < 1.5 else "high"
            return band, f"V-JEPA differs from Gazebo truth by {error:.2f} m."
        if question_id == "behavior_decision":
            behavior = snapshot.get("behavior_decision") or {}
            decision = str(behavior.get("decision", "WAIT"))
            person = str(behavior.get("person_id", "none"))
            reason = ascii_text(str(behavior.get("reason", "no reason received")))
            return f"{decision}:{person}", f"{decision} for {person}: {reason}"
        if question_id == "future_occupancy":
            behavior = snapshot.get("behavior_decision") or {}
            probability = behavior.get("collision_probability")
            free_window = behavior.get("predicted_free_space_window_s")
            ttc = behavior.get("time_to_collision_s")
            if not isinstance(probability, (int, float)) or not isinstance(
                free_window, (int, float)
            ):
                return "waiting", "Waiting for predicted person occupancy."
            ttc_text = "none" if not isinstance(ttc, (int, float)) else f"{ttc:.2f}s"
            occupied = any(
                bool(item.get("path_occupied"))
                for item in behavior.get("occupancy", [])
                if isinstance(item, dict)
            )
            key = "occupied" if occupied else "clear"
            return (
                key,
                f"Future path is {key}: risk={probability:.2f}, "
                f"TTC={ttc_text}, free window={free_window:.2f}s.",
            )
        if question_id == "latent_future":
            prediction = snapshot.get("latent_prediction") or {}
            horizons = prediction.get("horizons") or []
            steps = [
                int(item["step"])
                for item in horizons
                if isinstance(item, dict) and "step" in item
            ]
            if steps != [1, 2, 3]:
                return "waiting", "Waiting for V-JEPA z(t+1..3) rollouts."
            matured = prediction.get("matured_evaluations") or []
            if matured:
                mean_cosine = sum(
                    float(item["cosine_similarity"]) for item in matured
                ) / len(matured)
                return (
                    "rollout_123_evaluated",
                    "V-JEPA z(t+1), z(t+2), z(t+3) are active; "
                    f"latest matured cosine={mean_cosine:.3f}.",
                )
            return (
                "rollout_123_pending",
                "V-JEPA z(t+1), z(t+2), z(t+3) are active; awaiting actual latents.",
            )
        if question_id == "mission_state":
            key = live_model_answer_key("mission_state", snapshot)
            return key, {
                "outbound": "Robot đang đi từ dock tới kệ.",
                "align": "Robot đang căn chỉnh trước hộp.",
                "grasp": "Robot đang gắp và kiểm tra hộp.",
                "transport": "Robot đang chở hộp tới điểm giao.",
                "finish": "Robot đang hoàn tất hoặc chờ nhiệm vụ mới.",
            }[key]
        return "unsupported", "No prepared answer for this question."

    @staticmethod
    def _stream_answer(
        question_id: str, snapshot: dict[str, Any]
    ) -> tuple[str, str]:
        """Answer the prepared questions from current streaming evidence.

        This intentionally avoids object/color/timestamp assumptions. A later
        Storage B/C mission therefore receives answers about its current scene.
        """
        obstacle = snapshot.get("obstacle")
        clearance = float(snapshot.get("front_clearance", math.inf))
        linear = float(snapshot.get("linear_x", 0.0))
        angular = float(snapshot.get("angular_z", 0.0))
        behavior = snapshot.get("behavior_decision") or {}
        decision = str(behavior.get("decision", "PASS"))
        people = snapshot.get("people_ahead") or snapshot.get("nearby_people")
        has_front_obstacle = (
            obstacle is not None
            or (
                math.isfinite(clearance)
                and clearance <= DASHBOARD_OBSTACLE_DISTANCE_M
            )
        )
        obstacle_label = (
            str(obstacle.label)
            if obstacle is not None
            else "một vật cản"
        )
        predicted_person_speed = float(behavior.get("predicted_speed_mps", 0.0))
        occupied = any(
            bool(item.get("path_occupied"))
            for item in behavior.get("occupancy", [])
            if isinstance(item, dict)
        )
        moving = abs(linear) >= 0.03 or abs(angular) >= 0.05
        speed = "đứng yên" if not moving else (
            "đang di chuyển chậm" if abs(linear) < 0.22 else "đang di chuyển nhanh"
        )
        turn = "đi thẳng" if abs(angular) < 0.12 else (
            "đang cua trái" if angular > 0.0 else "đang cua phải"
        )
        if obstacle is None and not math.isfinite(clearance):
            obstacle_side = "chưa thấy vật cản gần"
        elif obstacle is None:
            obstacle_side = f"vật phía trước còn cách {clearance:.1f} m"
        else:
            bearing = obstacle.bearing_rad
            side = "chính giữa" if bearing is None or abs(bearing) < 0.25 else (
                "bên trái" if bearing > 0.0 else "bên phải"
            )
            obstacle_side = f"{side} ({obstacle.label})"
        q = {
            "q01": (
                f"motion:{speed}",
                "Robot đang đứng yên." if not moving
                else "Robot đang di chuyển chậm." if abs(linear) < 0.22
                else "Robot đang di chuyển nhanh.",
            ),
            "q02": (f"heading:{turn}", f"Robot {turn}."),
            "q03": ("steady", "Tốc độ đang giữ tương đối đều."),
            "q04": (
                "blocked" if has_front_obstacle else "open",
                f"Phía trước bị chặn gần bởi {obstacle_label}."
                if has_front_obstacle
                else "Phía trước đang thoáng và nhìn được xa.",
            ),
            "q05": (
                "obstacle" if has_front_obstacle else "none",
                f"Có, {obstacle_label} phía trước đang lớn dần."
                if has_front_obstacle and linear > 0.03
                else f"Có {obstacle_label} phía trước nhưng chưa thấy lớn nhanh."
                if has_front_obstacle
                else "Không có vật nào đang phình to rõ rệt.",
            ),
            "q06": (
                "tracked" if has_front_obstacle else "none",
                "Người đó đang di chuyển."
                if has_front_obstacle
                and obstacle is not None
                and obstacle.name.startswith("random_worker_")
                and predicted_person_speed > 0.08
                else "Người đó đang đứng yên."
                if has_front_obstacle
                and obstacle is not None
                and obstacle.name.startswith("random_worker_")
                else "Vật đó đang đứng yên; nó lớn dần vì robot tiến tới."
                if has_front_obstacle and linear > 0.03
                else "Chưa thấy vật đó tự di chuyển."
                if has_front_obstacle
                else "Không có vật đang phình to rõ rệt.",
            ),
            "q07": (f"obstacle:{obstacle_side}", f"Vật cản gần nhất: {obstacle_side}."),
            "q08": (
                "risk"
                if occupied
                or decision in {"WAIT", "REPLAN"}
                or (
                    has_front_obstacle
                    and (
                        clearance < 1.2
                        or (linear > 0.03 and clearance / linear <= 3.0)
                    )
                )
                else "clear",
                "Có, giữ nguyên hướng có thể dẫn đến va chạm."
                if occupied
                or decision in {"WAIT", "REPLAN"}
                or (
                    has_front_obstacle
                    and (
                        clearance < 1.2
                        or (linear > 0.03 and clearance / linear <= 3.0)
                    )
                )
                else "Không, hướng đi hiện tại vẫn an toàn.",
            ),
            "q09": (
                decision,
                "Dừng và chờ đường trống." if decision == "WAIT" else
                "Đổi sang đường đi khác." if decision == "REPLAN" else
                "Chậm lại và giữ khoảng cách." if has_front_obstacle and clearance < 2.0 else
                "Đi tiếp bình thường.",
            ),
            "q10": (
                "left" if angular > 0.12 else "right" if angular < -0.12 else "none",
                "Đang cua trái để tránh." if angular > 0.12 else
                "Đang cua phải để tránh." if angular < -0.12 else
                "Chưa cần né, tiếp tục giữ giữa lối.",
            ),
            "q11": (
                "narrow" if has_front_obstacle else "stable",
                "Lối đi đang hẹp lại vì có vật phía trước." if has_front_obstacle
                else "Lối đi phía trước vẫn rộng và ổn định.",
            ),
            "q12": (
                f"turn:{turn}",
                "Robot đang lệch nhẹ sang trái." if angular > 0.12
                else "Robot đang lệch nhẹ sang phải." if angular < -0.12
                else "Robot đang giữ gần giữa lối.",
            ),
            "q13": (
                live_model_answer_key("q13", snapshot),
                {
                    "none": "Không phát hiện người ở phía trước.",
                    "far": "Có người ở phía xa.",
                    "near_clear": "Có người ở gần nhưng không chắn đường.",
                    "on_path_stationary": "Có người đứng yên phía trước robot.",
                    "on_path_moving": "Có người đang di chuyển phía trước robot.",
                }[live_model_answer_key("q13", snapshot)],
            ),
            "q14": (
                "blocked" if clearance < 1.2 else "narrow" if has_front_obstacle else "clear",
                "Phía trước không đủ khoảng trống để đi qua an toàn."
                if clearance < 1.2
                else "Khoảng trống phía trước hơi hẹp, robot nên giảm tốc."
                if has_front_obstacle
                else "Khoảng trống phía trước đủ rộng để đi qua.",
            ),
            "q15": (
                "planned",
                "Phía trước là một khúc cua trái."
                if angular > 0.12
                else "Phía trước là một khúc cua phải."
                if angular < -0.12
                else "Phía trước tiếp tục là lối đi, chưa thấy lối cụt.",
            ),
            "q16": (
                "occluded" if has_front_obstacle else "unknown",
                "Vùng ngay sau vật cản có thể bị che khuất; giữ giới hạn an toàn." if has_front_obstacle
                else "Chưa thấy vùng che khuất nguy hiểm ở phía trước.",
            ),
            "q17": (
                "rollout",
                f"{obstacle_label.capitalize()} sẽ lớn dần khi robot tiến tới."
                if has_front_obstacle and linear > 0.03
                else "Khung cảnh sẽ xoay dần sang trái khi robot cua phải."
                if angular < -0.12
                else "Khung cảnh sẽ xoay dần sang phải khi robot cua trái."
                if angular > 0.12
                else "Lối đi sẽ tiếp tục mở ra phía trước.",
            ),
            "q19": (
                "ambiguous" if "kệ" in str(snapshot.get("area", "")) else "tracked",
                "Có, các nhịp kệ ở khu vực này khá giống nhau."
                if "kệ" in str(snapshot.get("area", ""))
                else "Không, khu vực này có đặc điểm khá dễ nhận ra.",
            ),
            "q20": (
                decision,
                "Dừng." if decision == "WAIT" else "Đổi sang đường khác." if decision == "REPLAN"
                else "Cua trái và đi chậm." if angular > 0.12
                else "Cua phải và đi chậm." if angular < -0.12
                else "Đi thẳng và giữ tốc độ hiện tại.",
            ),
        }
        return q[question_id]

    def update(self, snapshot: dict[str, Any]) -> None:
        sequence = int(snapshot["sequence"])
        if sequence == self.last_sequence:
            return
        self.last_sequence = sequence
        self.latest_snapshot = snapshot
        self.latest_stream_elapsed_s = float(snapshot.get("stream_elapsed_s", 0.0))
        for spec in self.questions:
            question_id = str(spec["id"])
            key, answer = self._answer(question_id, snapshot)
            self.instant[question_id] = answer
            if self.candidate_key.get(question_id) == key:
                self.candidate_count[question_id] = self.candidate_count.get(question_id, 0) + 1
            else:
                self.candidate_key[question_id] = key
                self.candidate_count[question_id] = 1
            if question_id not in self.stable or self.candidate_count[question_id] >= self.stabilize_frames:
                self.stable[question_id] = answer

    def active(self, now: float | None = None) -> dict[str, Any]:
        elapsed = self.latest_stream_elapsed_s
        if self.last_sequence < 0:
            elapsed = (time.monotonic() if now is None else now) - self.started
        slot = int(max(0.0, elapsed) / self.interval_sec)
        if self.active_spec is None or self.active_slot != slot:
            # Public rotation is deliberately independent of scene scope: the
            # selected five questions are valid in every warehouse location.
            self.active_index = slot % len(self.universal_questions)
            self.active_count = len(self.universal_questions)
            self.active_spec = self.universal_questions[self.active_index]
            self.active_slot = slot
        spec = self.active_spec
        index = self.active_index
        question_id = str(spec["id"])
        fallback_answer = self.default_answers.get(
            question_id, "Chưa có đủ quan sát để trả lời câu này."
        )
        instant_answer = self.instant.get(question_id, fallback_answer)
        stable_answer = self.stable.get(question_id, fallback_answer)
        answer_latent = None
        if self.latest_snapshot is not None and question_id in self.stable:
            answer_latent = self.answer_latent.infer(
                question_id=question_id,
                question=str(spec["text"]),
                policy_answer=instant_answer,
                stabilized_answer=stable_answer,
                visual_embedding=self.latest_snapshot.get("query_latent"),
                sequence=self.last_sequence,
                ambiguity=answer_observation_ambiguity(
                    question_id, self.latest_snapshot
                ),
            )
        display_questions = self.display_questions(
            active_id=question_id,
            active_index=index,
        )
        active_card = display_questions[index]
        return {
            "index": index,
            "count": self.active_count,
            "id": question_id,
            "question": str(spec["text"]),
            # Both public windows consume these exact two strings. ``stable``
            # remains the longer internal answer used by the latent adapter.
            "display_question": str(active_card["question"]),
            "display_answer": str(active_card["model_selected_answer"]),
            "display_answer_key": str(active_card["model_selected_key"]),
            "instant": instant_answer,
            "stable": stable_answer,
            "answer_latent": answer_latent,
            "pool_count": self.pool_count,
            "display_questions": display_questions,
        }

    def display_questions(
        self,
        *,
        active_id: str | None = None,
        active_index: int | None = None,
    ) -> tuple[dict[str, Any], ...]:
        """Return five live model question/answer cards rendered by OpenCV.

        The model answer is derived from the same current sensor/control
        snapshot used by the visible dashboard and changes continuously.
        """
        cards: list[dict[str, Any]] = []
        snapshot = self.latest_snapshot or {
            "linear_x": 0.0,
            "angular_z": 0.0,
            "front_clearance": math.inf,
            "obstacle": None,
            "behavior_decision": {"decision": "PASS"},
        }
        for index, spec in enumerate(self.universal_questions):
            question_id = str(spec["id"])
            fallback_answer = self.default_answers.get(
                question_id, "Chưa có đủ quan sát để trả lời câu này."
            )
            choices = tuple(
                {"key": key, "label": label}
                for key, label in WAREHOUSE_MODEL_ANSWER_CHOICES[question_id]
            )
            selected_key = live_model_answer_key(question_id, snapshot)
            selected_answer = next(
                (
                    str(choice["label"])
                    for choice in choices
                    if str(choice["key"]) == selected_key
                ),
                selected_key,
            )
            cards.append(
                {
                    "index": index,
                    "count": len(self.universal_questions),
                    "pool_count": self.pool_count,
                    "id": question_id,
                    "question": str(spec["text"]),
                    "instant": self.instant.get(question_id, fallback_answer),
                    "stable": self.stable.get(question_id, fallback_answer),
                    "choices": choices,
                    "model_selected_key": selected_key,
                    "model_selected_answer": selected_answer,
                    "stream_elapsed_s": float(
                        snapshot.get("stream_elapsed_s", 0.0)
                    ),
                    "area": str(snapshot.get("area", "đang chờ dữ liệu")),
                    "is_active": (
                        question_id == active_id
                        if active_id is not None
                        else index == active_index
                    ),
                }
            )
        return tuple(cards)


class DashboardRenderer:
    def __init__(
        self,
        map_yaml: Path,
        regions: WarehouseRegionIndex,
        latent_map: Path,
    ) -> None:
        with map_yaml.open(encoding="utf-8") as stream:
            metadata = yaml.safe_load(stream)
        image_path = Path(str(metadata["image"]))
        if not image_path.is_absolute():
            image_path = map_yaml.parent / image_path
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"không đọc được map image: {image_path}")
        self.map_gray = image
        self.resolution = float(metadata["resolution"])
        self.origin_x = float(metadata["origin"][0])
        self.origin_y = float(metadata["origin"][1])
        self.scale = min(0.82, 820.0 / image.shape[0])
        self.map_width = int(round(image.shape[1] * self.scale))
        self.map_height = int(round(image.shape[0] * self.scale))
        self.regions = regions
        # Keep the visual projector available for localization diagnostics;
        # the public streaming panel renders query-conditioned answer latents.
        self.visual_latent = LatentProjector(latent_map)
        self.answer_latent_renderer = AnswerLatentRenderer()

    def world_to_pixel(self, x: float, y: float) -> tuple[int, int]:
        column = (x - self.origin_x) / self.resolution
        row = self.map_gray.shape[0] - 1.0 - (y - self.origin_y) / self.resolution
        return int(round(column * self.scale)), int(round(row * self.scale))

    @staticmethod
    def _polyline(
        image: np.ndarray,
        points: tuple[tuple[float, float], ...],
        convert,
        color: tuple[int, int, int],
        thickness: int,
    ) -> None:
        if len(points) < 2:
            return
        pixels = np.asarray([convert(x, y) for x, y in points], dtype=np.int32)
        cv2.polylines(image, [pixels], False, color, thickness, cv2.LINE_AA)

    def _arrow(
        self,
        image: np.ndarray,
        pose: PoseSample,
        color: tuple[int, int, int],
        radius: int = 9,
    ) -> None:
        start = self.world_to_pixel(pose.x, pose.y)
        end = self.world_to_pixel(
            pose.x + 0.8 * math.cos(pose.yaw),
            pose.y + 0.8 * math.sin(pose.yaw),
        )
        cv2.circle(image, start, radius, color, -1, cv2.LINE_AA)
        cv2.arrowedLine(image, start, end, color, 3, cv2.LINE_AA, tipLength=0.35)

    @staticmethod
    def _line(
        image: np.ndarray,
        text: str,
        x: int,
        y: int,
        *,
        color: tuple[int, int, int] = (225, 225, 225),
        scale: float = 0.52,
        thickness: int = 1,
    ) -> None:
        value = str(text)
        if value.isascii():
            cv2.putText(
                image,
                value,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                color,
                thickness,
                cv2.LINE_AA,
            )
            return
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(rgb)
        draw = ImageDraw.Draw(pil_image)
        font = dashboard_font(max(11, int(round(31.0 * scale))))
        draw.text(
            (x, y),
            value,
            font=font,
            fill=(color[2], color[1], color[0]),
            anchor="ls",
            stroke_width=max(0, thickness - 1),
        )
        image[:] = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)

    @staticmethod
    def _pose_text(pose: PoseSample | None) -> str:
        if pose is None:
            return "waiting..."
        return f"x={pose.x:+.2f}  y={pose.y:+.2f}  yaw={math.degrees(pose.yaw):+.1f} deg"

    @staticmethod
    def avoidance_comment(snapshot: dict[str, Any]) -> str:
        obstacle = snapshot["obstacle"]
        angular = float(snapshot["angular_z"])
        if obstacle is None:
            return "CLEAR: no obstacle in front"
        label = ascii_text(obstacle.label)
        if abs(angular) >= 0.12:
            direction = "LEFT" if angular > 0.0 else "RIGHT"
            return f"AVOID: turning {direction} around {label}"
        return f"OBSTACLE: {label} at {obstacle.clearance_m:.2f} m"

    def render_streaming(
        self, snapshot: dict[str, Any], qa: dict[str, Any]
    ) -> np.ndarray:
        """Render the live streaming layout: camera left, latent plane right."""
        width, height = STREAMING_WIDTH, STREAMING_HEIGHT
        canvas = np.full((height, width, 3), (18, 22, 28), dtype=np.uint8)
        cv2.rectangle(canvas, (0, 0), (width, 82), (43, 37, 32), -1)
        self._line(
            canvas,
            f"Query [{qa['index'] + 1}/{qa['count']}]: {qa['display_question']}",
            24,
            32,
            color=(245, 245, 245),
            scale=0.62,
            thickness=2,
        )
        self._line(
            canvas,
            f"Answer: {qa['display_answer']}",
            24,
            64,
            color=(120, 235, 255),
            scale=0.58,
            thickness=2,
        )
        self._line(
            canvas,
            (
                f"DDS {snapshot['compute_host']} | "
                + (
                    f"infer {snapshot['inference_ms']:.0f} ms"
                    if isinstance(snapshot["inference_ms"], (int, float))
                    else "waiting for Orin"
                )
            ),
            1080,
            48,
            color=(100, 255, 180),
            scale=0.46,
            thickness=2,
        )

        camera_x, camera_y, camera_w, camera_h = 20, 98, 800, 450
        camera = snapshot["camera_bgr"]
        if camera is None:
            camera_view = np.full((camera_h, camera_w, 3), (30, 34, 40), dtype=np.uint8)
            self._line(
                camera_view,
                "Waiting for DDS /vjepa/camera/image_raw (640x360, 16:9)...",
                100,
                230,
                scale=0.56,
            )
        else:
            camera_view = cv2.resize(camera, (camera_w, camera_h), interpolation=cv2.INTER_AREA)
        canvas[camera_y : camera_y + camera_h, camera_x : camera_x + camera_w] = camera_view
        cv2.rectangle(
            canvas,
            (camera_x, camera_y),
            (camera_x + camera_w, camera_y + camera_h),
            (105, 112, 122),
            1,
        )
        cv2.rectangle(canvas, (camera_x, camera_y), (camera_x + 470, camera_y + 34), (18, 22, 28), -1)
        camera_age = snapshot["camera_age_ms"]
        age_text = (
            f"{camera_age:.0f} ms"
            if isinstance(camera_age, (int, float))
            else "waiting"
        )
        self._line(
            canvas,
            f"AGV CAMERA  16:9  |  latest frame: {age_text}  |  {snapshot['area']}",
            camera_x + 10,
            camera_y + 23,
            color=(235, 235, 235),
            scale=0.43,
            thickness=1,
        )

        latent_bounds = (840, 98, 580, 450)
        self.answer_latent_renderer.draw(
            canvas,
            latent_bounds,
            qa.get("answer_latent"),
            str(qa["id"]),
        )

        return canvas

    def render_questions(self, qa: dict[str, Any]) -> np.ndarray:
        """Render five live questions and the model-selected answers."""
        cards = tuple(qa.get("display_questions") or ())[:5]
        canvas = np.full(
            (QUESTIONS_HEIGHT, QUESTIONS_WIDTH, 3),
            (248, 248, 248),
            dtype=np.uint8,
        )
        text_items: list[
            tuple[str, int, int, int, tuple[int, int, int], bool]
        ] = []

        def text(
            value: str,
            x: int,
            y: int,
            size: int,
            color: tuple[int, int, int],
            bold: bool = False,
        ) -> None:
            text_items.append((str(value), x, y, size, color, bold))

        def wrap(
            value: str, max_width: int, size: int, bold: bool = False
        ) -> tuple[str, ...]:
            font = dashboard_font(size, "bold" if bold else "regular")
            words = str(value).split()
            if not words:
                return ("",)
            lines: list[str] = []
            current = words[0]
            for word in words[1:]:
                candidate = f"{current} {word}"
                width = (
                    font.getlength(candidate)
                    if hasattr(font, "getlength")
                    else font.getbbox(candidate)[2]
                )
                if width > max_width:
                    lines.append(current)
                    current = word
                else:
                    current = candidate
            lines.append(current)
            return tuple(lines)

        domain_labels = {
            "q07": "Nhận biết không gian",
            "q14": "Khả năng đi qua",
            "q13": "Tương tác người - robot",
            "mission_state": "Bộ nhớ trạng thái nhiệm vụ",
            "q20": "Lập kế hoạch hành động",
        }
        card_x, card_width, card_height, card_gap = 10, 472, 132, 7
        for index, card in enumerate(cards):
            x = card_x
            y = 5 + index * (card_height + card_gap)
            question_id = str(card["id"])
            selected_key = str(card["model_selected_key"])
            cv2.rectangle(
                canvas,
                (x, y),
                (x + card_width, y + card_height),
                (255, 255, 255),
                -1,
            )
            cv2.rectangle(
                canvas,
                (x, y),
                (x + card_width, y + card_height),
                (216, 219, 224),
                1,
            )
            cv2.rectangle(canvas, (x, y), (x + 4, y + card_height), (118, 189, 226), -1)
            text(f"Q{index + 1}", x + 14, y + 8, 12, (34, 39, 46), True)
            text(
                domain_labels.get(question_id, "World model"),
                x + 48,
                y + 10,
                9,
                (111, 118, 128),
            )
            for line_index, line in enumerate(
                wrap(str(card["question"]), card_width - 28, 11, True)[:2]
            ):
                text(
                    line,
                    x + 14,
                    y + 31 + line_index * 15,
                    11,
                    (28, 33, 40),
                    True,
                )

            choices = tuple(card.get("choices") or ())
            selected_label = str(
                card.get("model_selected_answer", card.get("stable", selected_key))
            )
            answer_bounds = (x + 14, y + 75, x + card_width - 14, y + 119)
            cv2.rectangle(
                canvas,
                answer_bounds[:2],
                answer_bounds[2:],
                (250, 241, 226),
                -1,
            )
            cv2.rectangle(
                canvas,
                answer_bounds[:2],
                answer_bounds[2:],
                (214, 139, 61),
                1,
            )
            for line_index, line in enumerate(
                wrap(selected_label, card_width - 56, 11, True)[:2]
            ):
                text(
                    line,
                    x + 28,
                    y + 84 + line_index * 14,
                    11,
                    (31, 75, 107),
                    True,
                )

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(rgb)
        draw = ImageDraw.Draw(pil_image)
        for value, x, y, size, color, bold in text_items:
            draw.text(
                (x, y),
                value,
                font=dashboard_font(size, "bold" if bold else "regular"),
                fill=color,
                anchor="lt",
            )
        return cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)

    def render_map(self, snapshot: dict[str, Any]) -> np.ndarray:
        """Render the bare map with Truth GPS and planning overlaid on it."""
        map_image = cv2.resize(
            self.map_gray,
            (self.map_width, self.map_height),
            interpolation=cv2.INTER_NEAREST,
        )
        canvas = cv2.cvtColor(map_image, cv2.COLOR_GRAY2BGR)
        self._polyline(canvas, snapshot["astar_plan"], self.world_to_pixel, (255, 170, 0), 3)
        self._polyline(canvas, snapshot["truth_trail"], self.world_to_pixel, (30, 30, 245), 2)

        current = snapshot["current_truth"]
        if current is not None:
            self._arrow(canvas, current, (30, 30, 245))
        return canvas


def serializable_snapshot(
    snapshot: dict[str, Any], qa: dict[str, Any] | None = None
) -> dict[str, Any]:
    def pose_value(pose: PoseSample | None) -> list[float] | None:
        if pose is None:
            return None
        return [pose.x, pose.y, pose.z, pose.yaw]

    obstacle = snapshot["obstacle"]
    value = {
        "area": snapshot["area"],
        "gazebo_truth": pose_value(snapshot["current_truth"]),
        "vjepa_pose": pose_value(snapshot["vjepa"]),
        "vjepa_raw_pose": pose_value(snapshot["raw_vjepa"]),
        "vjepa_odom_projected_pose": pose_value(snapshot["projected_vjepa"]),
        "odom_projection_enabled": snapshot["odom_projection_enabled"],
        "relative_dx": snapshot["relative_dx"],
        "relative_dy": snapshot["relative_dy"],
        "position_error_m": snapshot["position_error"],
        "yaw_error_rad": snapshot["yaw_error"],
        "raw_timestamp_aligned_position_error_m": snapshot["raw_position_error"],
        "raw_timestamp_aligned_yaw_error_rad": snapshot["raw_yaw_error"],
        "current_display_gap_m": snapshot["current_display_gap"],
        "tracking_state": snapshot["tracking_state"],
        "camera_moving": snapshot["camera_moving"],
        "camera_motion_score": snapshot["camera_motion_score"],
        "camera_pixel_change": snapshot["camera_pixel_change"],
        "camera_motion_inlier_ratio": snapshot["camera_motion_inlier_ratio"],
        "camera_progress_scale": snapshot["camera_progress_scale"],
        "motion_credit": snapshot["motion_credit"],
        "raw_jump_m": snapshot["raw_jump_m"],
        "accepted_step_m": snapshot["accepted_step_m"],
        "translation_gate_m": snapshot["translation_gate_m"],
        "selected_rank": snapshot["selected_rank"],
        "rejected_streak": snapshot["rejected_streak"],
        "nav_localization_state": snapshot["nav_localization_state"],
        "nav_localization_source": snapshot["nav_localization_source"],
        "nav_uses_gazebo_truth": snapshot["nav_uses_gazebo_truth"],
        "nav_planner": snapshot["nav_planner"],
        "nav_correction_m": snapshot["nav_correction_m"],
        "astar_plan_points": len(snapshot["astar_plan"]),
        "obstacle": obstacle.label if obstacle is not None else None,
        "comment": DashboardRenderer.avoidance_comment(snapshot),
        "latent_dimension": snapshot["latent_dimension"],
        "compute_host": snapshot["compute_host"],
        "inference_ms": snapshot["inference_ms"],
        "result_pose_topic": snapshot["result_pose_topic"],
        "result_latent_topic": snapshot["result_latent_topic"],
        "camera_age_ms": snapshot["camera_age_ms"],
        "vjepa_age_ms": snapshot["vjepa_age_ms"],
    }
    if qa is not None:
        value["prepared_qa"] = {
            "index": qa["index"],
            "question": qa["display_question"],
            "model_answer": qa["display_answer"],
            "pool_count": int(qa.get("pool_count", 20)),
            "display_count": int(qa.get("count", 0)),
            "questions": [
                {
                    "index": int(card["index"]),
                    "id": str(card["id"]),
                    "question": str(card["question"]),
                    "choices": [
                        str(choice["label"])
                        for choice in card.get("choices", ())
                    ],
                    "model_selected_key": str(card["model_selected_key"]),
                    "model_selected_answer": str(card["model_selected_answer"]),
                    "active": bool(card.get("is_active", False)),
                }
                for card in qa.get("display_questions", ())
            ],
        }
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inventory",
        type=Path,
        default=WAREHOUSE_ROOT / "config" / "inventory_locations.yaml",
    )
    parser.add_argument(
        "--map-yaml",
        type=Path,
        default=WAREHOUSE_ROOT / "maps" / "warehouse_lidar.yaml",
    )
    parser.add_argument(
        "--latent-map",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "autonomous_map_dense",
    )
    parser.add_argument(
        "--camera-topic",
        default="/vjepa/camera/image_raw",
        help="ROS 2 DDS image topic received by V-JEPA",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=PROJECT_ROOT / "configs" / "warehouse_live_questions.yaml",
    )
    parser.add_argument("--headless", action="store_true", help="print snapshots without GUI")
    parser.add_argument("--duration", type=float, default=0.0, help="optional run time for smoke tests")
    parser.add_argument("--refresh-hz", type=float, default=32.0)
    parser.add_argument(
        "--map-refresh-hz",
        type=float,
        default=5.0,
        help="redraw the heavier occupancy-map window independently of camera FPS",
    )
    parser.add_argument(
        "--odom-projection",
        action="store_true",
        help="add short-term wheel-odom projection to JSON telemetry only",
    )
    args, ros_args = parser.parse_known_args()
    if args.duration < 0.0 or args.refresh_hz <= 0.0 or args.map_refresh_hz <= 0.0:
        parser.error("duration must be non-negative and refresh rates must be positive")
    if not args.headless and not os.environ.get("DISPLAY"):
        parser.error("DISPLAY is not set; use --headless or run inside the desktop session")

    rclpy.init(args=ros_args)
    node = LocalizationDashboardNode(
        args.inventory.resolve(),
        camera_topic=args.camera_topic,
        odom_projection_enabled=args.odom_projection,
    )
    renderer = DashboardRenderer(
        args.map_yaml.resolve(), node.regions, args.latent_map.resolve()
    )
    qa_engine = PreparedQA(args.questions.resolve())
    started = time.monotonic()
    last_sequence = -1
    last_question_index = -1
    last_map_render = -math.inf
    cached_map: np.ndarray | None = None
    last_questions_render = -math.inf
    cached_questions: np.ndarray | None = None
    held_keyboard = X11KeyboardState() if not args.headless else None
    frame_period = 1.0 / args.refresh_hz
    next_frame_deadline = time.monotonic() + frame_period
    if not args.headless:
        cv2.namedWindow(TELEMETRY_WINDOW, cv2.WINDOW_NORMAL)
        cv2.namedWindow(MAP_WINDOW, cv2.WINDOW_NORMAL)
        # GUI_NORMAL removes Qt's image toolbar and RGB/coordinate status bar;
        # the window now shows only the five question cards.
        cv2.namedWindow(
            QUESTIONS_WINDOW,
            cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL,
        )
        cv2.resizeWindow(TELEMETRY_WINDOW, STREAMING_WIDTH, STREAMING_HEIGHT)
        cv2.resizeWindow(MAP_WINDOW, renderer.map_width, renderer.map_height)
        cv2.resizeWindow(QUESTIONS_WINDOW, QUESTIONS_WIDTH, QUESTIONS_HEIGHT)
        print(
            "[KEYBOARD] Hold W/S to move and A/D simultaneously to steer; "
            "A or D alone rotates in place. SPACE stops; ESC closes. "
            f"held-key polling={'X11' if held_keyboard and held_keyboard.available else 'fallback'}.",
            flush=True,
        )
    try:
        while rclpy.ok():
            # Process one blocking callback and then drain the ready queue.
            # Camera QoS depth=1 means this always converges to the newest
            # frame instead of replaying an accumulated video backlog.
            callback_deadline = time.monotonic() + min(0.006, 0.25 * frame_period)
            rclpy.spin_once(node, timeout_sec=min(0.003, frame_period))
            for _ in range(31):
                if time.monotonic() >= callback_deadline:
                    break
                rclpy.spin_once(node, timeout_sec=0.0)
            snapshot = node.snapshot()
            qa_engine.update(snapshot)
            qa = qa_engine.active()
            if int(qa["index"]) != last_question_index:
                print(
                    f"\n[VL-JEPA QUERY {qa['index'] + 1}/{qa['count']}] "
                    f"{qa['display_question']}\n"
                    f"[ANSWER] {qa['display_answer']}",
                    flush=True,
                )
                last_question_index = int(qa["index"])
            if args.headless:
                if snapshot["sequence"] != last_sequence:
                    print(
                        "[DASHBOARD] "
                        + json.dumps(serializable_snapshot(snapshot, qa), ensure_ascii=False),
                        flush=True,
                    )
                    last_sequence = int(snapshot["sequence"])
                time.sleep(1.0 / args.refresh_hz)
            else:
                cv2.imshow(TELEMETRY_WINDOW, renderer.render_streaming(snapshot, qa))
                now = time.monotonic()
                # This window is text-only and its selected answers are
                # stabilized. Redrawing PIL fonts at camera FPS wasted most of
                # the GUI frame budget, so refresh it independently at 5 Hz.
                if (
                    cached_questions is None
                    or now - last_questions_render >= 0.2
                ):
                    cached_questions = renderer.render_questions(qa)
                    last_questions_render = now
                cv2.imshow(QUESTIONS_WINDOW, cached_questions)
                if (
                    cached_map is None
                    or now - last_map_render >= 1.0 / args.map_refresh_hz
                ):
                    cached_map = renderer.render_map(snapshot)
                    last_map_render = now
                cv2.imshow(MAP_WINDOW, cached_map)
                now = time.monotonic()
                if next_frame_deadline < now:
                    # Rendering overran this frame. Drop the missed display
                    # slot immediately instead of adding another full-period
                    # wait and showing an increasingly old camera image.
                    next_frame_deadline = now
                key = cv2.waitKeyEx(gui_wait_ms(next_frame_deadline, now))
                next_frame_deadline += frame_period
                if held_keyboard and held_keyboard.available:
                    if not node.update_held_keyboard(held_keyboard.pressed()):
                        break
                else:
                    if key >= 0 and not node.handle_keyboard_key(key):
                        break
                    node.refresh_keyboard_command()
                if (
                    cv2.getWindowProperty(TELEMETRY_WINDOW, cv2.WND_PROP_VISIBLE) < 1
                    or cv2.getWindowProperty(MAP_WINDOW, cv2.WND_PROP_VISIBLE) < 1
                    or cv2.getWindowProperty(QUESTIONS_WINDOW, cv2.WND_PROP_VISIBLE) < 1
                ):
                    break
            if args.duration and time.monotonic() - started >= args.duration:
                break
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_keyboard()
        node.close()
        if held_keyboard is not None:
            held_keyboard.close()
        if not args.headless:
            cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
