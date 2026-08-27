from __future__ import annotations

import math
from types import SimpleNamespace
from pathlib import Path

import cv2
import numpy as np

from scripts.localization_dashboard import (
    AnswerLatentRenderer,
    DashboardRenderer,
    EntityPose,
    LocalizationDashboardNode,
    PoseSample,
    PreparedQA,
    QueryConditionedAnswerLatent,
    SemanticTextEncoder,
    STREAMING_HEIGHT,
    QUESTIONS_HEIGHT,
    QUESTIONS_WIDTH,
    answer_candidates,
    answer_observation_ambiguity,
    differential_keyboard_command,
    goal_status_array_has_active_goal,
    latent_evidence_profile,
    live_model_answer_key,
    plan_for_display,
    people_in_camera_frustum,
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


def test_query_answers_use_predictive_planner_and_real_latent_rollout_contract() -> None:
    snapshot = {
        "behavior_decision": {
            "decision": "WAIT",
            "reason": "collision probability 0.81 with TTC 1.25s",
            "person_id": "random_worker_5",
            "collision_probability": 0.81,
            "time_to_collision_s": 1.25,
            "predicted_free_space_window_s": 0.75,
            "occupancy": [{"path_occupied": True}],
        },
        "latent_prediction": {
            "source": "actual_vjepa_embedding_causal_temporal_dynamics",
            "horizons": [{"step": 1}, {"step": 2}, {"step": 3}],
            "matured_evaluations": [{"cosine_similarity": 0.94}],
        },
        "mission_state": {"state": "CHARGING_HOME"},
    }

    _, decision = PreparedQA._answer("behavior_decision", snapshot)
    _, occupancy = PreparedQA._answer("future_occupancy", snapshot)
    _, latent = PreparedQA._answer("latent_future", snapshot)
    _, state = PreparedQA._answer("mission_state", snapshot)

    assert decision.startswith("WAIT for random_worker_5")
    assert "risk=0.81" in occupancy and "TTC=1.25s" in occupancy
    assert "z(t+1), z(t+2), z(t+3)" in latent and "0.940" in latent
    assert state == "Robot đang hoàn tất hoặc chờ nhiệm vụ mới."


def test_each_question_selects_relevant_latent_evidence() -> None:
    assert latent_evidence_profile("q01") == "temporal_motion"
    assert latent_evidence_profile("q08") == "future_rollout"
    assert latent_evidence_profile("q17") == "future_rollout"
    assert latent_evidence_profile("mission_state") == "temporal_motion"
    assert latent_evidence_profile("q13") == "current_observation"


def test_each_question_has_real_candidate_competition() -> None:
    question_ids = tuple(f"q{index:02d}" for index in range(1, 18)) + (
        "mission_state",
        "q19",
        "q20",
    )
    counts = [len(answer_candidates(question_id)) for question_id in question_ids]
    assert min(counts) >= 40
    assert sum(counts) >= 800


def test_ambiguity_tracks_observed_decision_boundaries() -> None:
    near = answer_observation_ambiguity(
        "q04", {"front_clearance": 3.9, "linear_x": 0.2, "angular_z": 0.0}
    )
    clear = answer_observation_ambiguity(
        "q04", {"front_clearance": 7.0, "linear_x": 0.2, "angular_z": 0.0}
    )
    assert near > 0.8
    assert clear == 0.0


def test_stream_qa_loads_twenty_prepared_questions() -> None:
    qa = PreparedQA(
        Path(__file__).resolve().parents[2]
        / "vjepa_visual_localization/configs/warehouse_live_questions.yaml"
    )
    assert len(qa.questions) == 20
    assert qa.questions[0]["id"] == "q01"
    assert qa.questions[-1]["id"] == "q20"
    assert qa.interval_sec == 4.0
    assert all("scope" in question for question in qa.questions)
    assert qa.universal_question_ids == (
        "q07",
        "q14",
        "q13",
        "mission_state",
        "q20",
    )
    assert len(qa.universal_questions) == 5


def test_question_relevance_follows_area_and_observed_scene() -> None:
    qa = PreparedQA(
        Path(__file__).resolve().parents[2]
        / "vjepa_visual_localization/configs/warehouse_live_questions.yaml"
    )

    def eligible(snapshot: dict) -> set[str]:
        return {str(spec["id"]) for spec in qa._eligible_questions(snapshot)}

    base = {
        "linear_x": 0.0,
        "angular_z": 0.0,
        "front_clearance": math.inf,
        "obstacle": None,
        "behavior_decision": {"decision": "PASS", "occupancy": []},
        "people_ahead": (),
        "nearby_people": (),
        "camera_frustum_people": (),
        "camera_person_detection": {"visible": False},
    }
    charging = eligible({**base, "area": "khu sạc AGV"})
    assert {"q01", "mission_state", "q20"} <= charging
    assert {"q05", "q13", "q14", "q19"}.isdisjoint(charging)

    shelf = eligible({**base, "area": "khu kệ A", "linear_x": 0.25})
    assert {"q02", "q04", "q11", "q14", "q15", "q19"} <= shelf
    assert {"q05", "q13"}.isdisjoint(shelf)

    event = eligible({
        **base,
        "area": "khu kệ A",
        "linear_x": 0.25,
        "front_clearance": 1.5,
        "people_ahead": ((1.5, "random_worker_1"),),
        "camera_frustum_people": ((1.5, "random_worker_1", 0.0),),
        "camera_person_detection": {"visible": True},
    })
    assert {"q05", "q06", "q07", "q10", "q13", "q16"} <= event


def test_context_change_does_not_replace_question_inside_reading_interval() -> None:
    qa = PreparedQA(
        Path(__file__).resolve().parents[2]
        / "vjepa_visual_localization/configs/warehouse_live_questions.yaml"
    )
    clear = {
        "area": "khu kệ A",
        "linear_x": 0.25,
        "angular_z": 0.0,
        "front_clearance": math.inf,
        "obstacle": None,
        "behavior_decision": {"decision": "PASS", "occupancy": []},
        "people_ahead": (),
        "nearby_people": (),
    }
    qa.latest_snapshot = clear
    qa.last_sequence = 1
    qa.latest_stream_elapsed_s = 0.2
    first = qa.active()
    qa.latest_snapshot = {
        **clear,
        "front_clearance": 1.0,
        "people_ahead": ((1.0, "random_worker_1"),),
    }
    qa.latest_stream_elapsed_s = 3.8
    second = qa.active()
    assert first["id"] == second["id"] == "q07"


def test_observed_mode_answers_fixed_questions_from_current_stream_state() -> None:
    key, answer = PreparedQA._answer(
        "q09",
        {
            "linear_x": 0.0,
            "angular_z": 0.0,
            "front_clearance": 0.4,
            "obstacle": None,
            "behavior_decision": {"decision": "WAIT", "occupancy": []},
            "people_ahead": (),
            "nearby_people": (),
            "area": "khu kệ B",
        },
    )
    assert key == "WAIT"
    assert answer == "Dừng và chờ đường trống."


def test_lidar_obstacle_is_not_reported_clear_without_semantic_label() -> None:
    snapshot = {
        "linear_x": 0.25,
        "angular_z": 0.0,
        "front_clearance": 2.4,
        "obstacle": None,
        "behavior_decision": {"decision": "PASS", "occupancy": []},
        "people_ahead": (),
        "nearby_people": (),
        "area": "khu kệ A",
    }
    key, answer = PreparedQA._answer("q04", snapshot)
    assert key == "blocked"
    assert "một vật cản" in answer


def test_q14_reports_observed_forward_space_instead_of_floor_markings() -> None:
    base = {
        "linear_x": 0.2,
        "angular_z": 0.0,
        "obstacle": None,
        "behavior_decision": {"decision": "PASS", "occupancy": []},
        "people_ahead": (),
        "nearby_people": (),
        "area": "khu kệ A",
    }
    _, clear = PreparedQA._answer("q14", {**base, "front_clearance": 6.0})
    _, narrow = PreparedQA._answer("q14", {**base, "front_clearance": 2.0})
    _, blocked = PreparedQA._answer("q14", {**base, "front_clearance": 0.8})
    assert "đủ rộng" in clear
    assert "hơi hẹp" in narrow
    assert "không đủ khoảng trống" in blocked
    assert all("vạch" not in answer.lower() for answer in (clear, narrow, blocked))


def test_visible_answers_do_not_expose_internal_data_sources() -> None:
    snapshot = {
        "linear_x": 0.25,
        "angular_z": 0.0,
        "front_clearance": 2.4,
        "obstacle": None,
        "behavior_decision": {
            "decision": "PASS",
            "occupancy": [],
            "predicted_speed_mps": 0.0,
        },
        "people_ahead": (),
        "nearby_people": (),
        "area": "khu kệ A",
        "astar_plan": (),
    }
    forbidden = (
        "lidar",
        "planner",
        "nav2",
        "semantic",
        "rollout",
        "latent",
        "snapshot",
        "replan",
        "theo lệnh",
        "dữ liệu",
    )
    question_ids = tuple(f"q{index:02d}" for index in range(1, 18)) + (
        "mission_state",
        "q19",
        "q20",
    )
    for question_id in question_ids:
        _, answer = PreparedQA._answer(question_id, snapshot)
        lowered = answer.lower()
        assert all(word not in lowered for word in forbidden), answer


def test_dashboard_line_preserves_vietnamese_diacritics() -> None:
    accented = np.zeros((80, 500, 3), dtype=np.uint8)
    unaccented = accented.copy()
    DashboardRenderer._line(accented, "Phía trước có vật cản", 8, 45)
    DashboardRenderer._line(unaccented, "Phia truoc co vat can", 8, 45)
    assert np.any(accented)
    assert not np.array_equal(accented, unaccented)


def test_live_model_answers_change_with_current_environment() -> None:
    start = {
        "linear_x": 0.0,
        "angular_z": 0.0,
        "front_clearance": math.inf,
        "obstacle": None,
        "behavior_decision": {"decision": "PASS", "occupancy": []},
        "people_ahead": (),
        "nearby_people": (),
        "camera_frustum_people": (),
        "camera_person_detection": {"visible": False},
        "area": "khu sạc AGV",
        "mission_state": {"state": "WAITING"},
    }
    shelf_turn = {
        "linear_x": 0.20,
        "angular_z": -0.35,
        "front_clearance": 0.8,
        "obstacle": SimpleNamespace(bearing_rad=-0.5),
        "behavior_decision": {
            "decision": "PASS",
            "person_id": "random_worker_1",
            "predicted_speed_mps": 0.35,
            "occupancy": [{"path_occupied": True}],
        },
        "people_ahead": ((1.2, "random_worker_1"),),
        "nearby_people": (),
        "camera_frustum_people": ((1.2, "random_worker_1", 0.0),),
        "camera_person_detection": {
            "visible": True,
            "far": False,
            "on_path": True,
            "moving": True,
        },
        "area": "khu kệ A",
        "mission_state": {"state": "ALIGN_PACKAGE"},
    }
    packing = {
        "linear_x": 0.30,
        "angular_z": 0.0,
        "front_clearance": math.inf,
        "obstacle": None,
        "behavior_decision": {
            "decision": "PASS",
            "person_id": "random_worker_2",
            "predicted_speed_mps": 0.30,
            "occupancy": [{"path_occupied": False}],
        },
        "people_ahead": (),
        "nearby_people": ((2.0, "random_worker_2"),),
        "camera_frustum_people": ((2.0, "random_worker_2", 0.5),),
        "camera_person_detection": {
            "visible": True,
            "far": False,
            "on_path": False,
            "moving": True,
        },
        "area": "khu đóng gói",
        "mission_state": {"state": "RETURN_TO_DROPOFF"},
    }

    assert [
        live_model_answer_key(question_id, start)
        for question_id in ("q07", "q14", "q13", "mission_state", "q20")
    ] == ["none", "clear", "none", "finish", "stop"]
    assert [
        live_model_answer_key(question_id, shelf_turn)
        for question_id in ("q07", "q14", "q13", "mission_state", "q20")
    ] == ["right", "blocked", "on_path_moving", "align", "right"]
    assert [
        live_model_answer_key(question_id, packing)
        for question_id in ("q07", "q14", "q13", "mission_state", "q20")
    ] == ["none", "clear", "near_clear", "transport", "straight"]


def test_q13_ignores_people_outside_the_front_camera() -> None:
    snapshot = {
        "behavior_decision": {
            "decision": "WAIT",
            "person_id": "random_worker_1",
            "predicted_speed_mps": 0.4,
            "occupancy": [{"path_occupied": True}],
        },
        "people_ahead": (),
        "nearby_people": ((0.8, "random_worker_1"),),
        "camera_frustum_people": (),
        "camera_person_detection": {
            "visible": True,
            "far": False,
            "on_path": True,
            "moving": True,
        },
    }
    assert live_model_answer_key("q13", snapshot) == "none"


def test_q13_matches_pixel_detection_to_projected_worker_bearing() -> None:
    snapshot = {
        "camera_frustum_people": ((2.0, "random_worker_1", 0.45),),
        "camera_person_detection": {
            "visible": True,
            "bbox": (500, 80, 70, 220),
            "frame_size": (640, 360),
            "far": False,
            "on_path": False,
            "moving": False,
        },
    }
    assert live_model_answer_key("q13", snapshot) == "none"


def test_camera_frustum_uses_robot_heading_and_camera_fov() -> None:
    entities = (
        EntityPose("random_worker_front", 2.0, 0.0),
        EntityPose("random_worker_side", 0.0, 2.0),
        EntityPose("random_worker_behind", -2.0, 0.0),
    )
    visible = people_in_camera_frustum(
        agv_x=0.0,
        agv_y=0.0,
        agv_yaw=0.0,
        entities=entities,
    )
    assert [item[1] for item in visible] == ["random_worker_front"]


def test_five_cards_render_one_live_model_answer_each() -> None:
    engine = PreparedQA(
        Path(__file__).resolve().parents[2]
        / "vjepa_visual_localization/configs/warehouse_live_questions.yaml"
    )
    qa = engine.active(now=engine.started)
    renderer = object.__new__(DashboardRenderer)

    image = renderer.render_questions(qa)
    assert image.shape == (QUESTIONS_HEIGHT, QUESTIONS_WIDTH, 3)
    cards = qa["display_questions"]
    assert len(cards) == 5
    assert all(len(card["choices"]) >= 3 for card in cards)
    assert all(card["model_selected_key"] for card in cards)
    active_card = next(card for card in cards if card["is_active"])
    assert qa["display_question"] == active_card["question"]
    assert qa["display_answer"] == active_card["model_selected_answer"]
    assert qa["display_answer_key"] == active_card["model_selected_key"]
    assert not hasattr(renderer, "quiz_selected")
    assert not hasattr(renderer, "quiz_submitted")
    assert np.any(image)


def test_camera_query_cycles_through_the_same_five_question_cards() -> None:
    engine = PreparedQA(
        Path(__file__).resolve().parents[2]
        / "vjepa_visual_localization/configs/warehouse_live_questions.yaml"
    )
    engine.last_sequence = 0
    for index in range(5):
        engine.latest_stream_elapsed_s = index * engine.interval_sec
        qa = engine.active()
        active_card = next(
            card for card in qa["display_questions"] if card["is_active"]
        )
        assert qa["index"] == active_card["index"] == index
        assert qa["display_question"] == active_card["question"]
        assert qa["display_answer"] == active_card["model_selected_answer"]


def test_query_conditioned_answer_latent_selects_policy_candidate() -> None:
    encoder = SemanticTextEncoder({"backend": "hash", "fallback_dimension": 96})
    adapter = QueryConditionedAnswerLatent(
        {
            "oracle_weight": 0.92,
            "query_weight": 0.05,
            "visual_weight": 0.03,
            "selective_decode_cosine": 0.995,
        },
        encoder=encoder,
    )
    visual = np.linspace(-1.0, 1.0, 128, dtype=np.float32)
    blocked = adapter.infer(
        question_id="q04",
        question="Phía trước nhìn được xa hay bị chặn gần?",
        policy_answer="Phía trước đang bị chặn gần.",
        visual_embedding=visual,
        sequence=1,
    )
    clear = adapter.infer(
        question_id="q04",
        question="Phía trước nhìn được xa hay bị chặn gần?",
        policy_answer="Phía trước đang thoáng và nhìn được xa.",
        visual_embedding=-visual,
        sequence=2,
    )

    assert blocked["selected_answer"] == "Phía trước đang bị chặn gần."
    assert clear["selected_answer"] == "Phía trước đang thoáng và nhìn được xa."
    assert blocked["predicted_embedding"].shape == (96,)
    assert len(blocked["answer_space_points_2d"]) >= 200
    assert len(blocked["answer_space_points_2d"]) == len(
        blocked["answer_space_answers"]
    )
    anchor = blocked["stabilized_anchor_index"]
    assert np.allclose(
        blocked["stabilized_point_2d"],
        blocked["answer_space_points_2d"][anchor],
    )
    assert blocked["confidence"] > 0.8
    assert not np.allclose(
        blocked["predicted_embedding"], clear["predicted_embedding"]
    )
    assert blocked["provenance"].startswith("vjepa_visual_plus_query")


def test_prepared_qa_exposes_answer_latent_for_renderer() -> None:
    qa = PreparedQA(
        Path(__file__).resolve().parents[2]
        / "vjepa_visual_localization/configs/warehouse_live_questions.yaml"
    )
    qa.answer_latent = QueryConditionedAnswerLatent(
        {
            "oracle_weight": 0.92,
            "query_weight": 0.05,
            "visual_weight": 0.03,
        },
        encoder=SemanticTextEncoder({"backend": "hash", "fallback_dimension": 64}),
    )
    qa.update(
        {
            "sequence": 1,
            "stream_elapsed_s": 0.0,
            "linear_x": 0.2,
            "angular_z": 0.0,
            "front_clearance": 1.8,
            "obstacle": None,
            "behavior_decision": {
                "decision": "PASS",
                "occupancy": [],
                "predicted_speed_mps": 0.0,
            },
            "people_ahead": (),
            "nearby_people": (),
            "area": "khu kệ A",
            "query_latent": np.ones(128, dtype=np.float32),
        }
    )
    active = qa.active()
    report = active["answer_latent"]
    assert active["id"] == "q07"
    assert report is not None
    assert report["question_id"] == "q07"
    assert active["stable"] == report["selected_answer"]

    canvas = np.zeros((568, 1440, 3), dtype=np.uint8)
    AnswerLatentRenderer().draw(canvas, (840, 98, 580, 450), report, "q01")
    assert np.any(canvas[98:548, 840:1420])
    plot = canvas[146:450, 864:1396]
    red_mask = (
        (plot[:, :, 2] > 180)
        & (plot[:, :, 2] > plot[:, :, 1] * 1.5)
        & (plot[:, :, 2] > plot[:, :, 0] * 1.5)
    ).astype(np.uint8)
    components, _ = cv2.connectedComponents(red_mask)
    assert components - 1 == 1


def test_stabilized_marker_snaps_to_each_new_target() -> None:
    renderer = AnswerLatentRenderer()
    first = np.asarray((-0.8, 0.25), dtype=np.float32)
    second = np.asarray((0.9, -0.6), dtype=np.float32)

    assert np.array_equal(renderer._stabilized_for_display(first), first)
    assert np.array_equal(renderer._stabilized_for_display(second), second)


def test_instant_prediction_can_differ_until_stabilization_catches_up() -> None:
    qa = PreparedQA(
        Path(__file__).resolve().parents[2]
        / "vjepa_visual_localization/configs/warehouse_live_questions.yaml"
    )
    qa.answer_latent = QueryConditionedAnswerLatent(
        {"oracle_weight": 0.92, "query_weight": 0.05, "visual_weight": 0.03},
        encoder=SemanticTextEncoder({"backend": "hash", "fallback_dimension": 64}),
    )

    def update(sequence: int, clearance: float) -> dict:
        qa.update(
            {
                "sequence": sequence,
            "stream_elapsed_s": 4.0,
                "linear_x": 0.25,
                "angular_z": 0.0,
                "front_clearance": clearance,
                "obstacle": None,
                "behavior_decision": {
                    "decision": "PASS",
                    "occupancy": [],
                    "predicted_speed_mps": 0.0,
                },
                "people_ahead": (),
                "nearby_people": (),
                "area": "khu kệ A",
                "query_latent": np.ones(128, dtype=np.float32),
            }
        )
        return qa.active()

    clear = update(1, 5.0)
    transition = update(2, 1.5)
    assert clear["id"] == "q14"
    assert transition["answer_latent"]["selected_answer"].startswith(
        "Khoảng trống phía trước hơi hẹp"
    )
    assert transition["stable"].startswith("Khoảng trống phía trước đủ rộng")
    assert not np.allclose(
        transition["answer_latent"]["predicted_point_2d"],
        transition["answer_latent"]["stabilized_point_2d"],
    )

    update(3, 1.5)
    stabilized = update(4, 1.5)
    assert stabilized["stable"].startswith("Khoảng trống phía trước hơi hẹp")
