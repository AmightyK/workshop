from __future__ import annotations

import json

from pathlib import Path

from scripts.udp_command_bridge import (
    Deduper,
    MissionProcess,
    ack_packet,
    decode_packet,
)


def packet(action: dict, *, seq: int = 1) -> bytes:
    return json.dumps(
        {
            "v": 1,
            "kind": "control" if action.get("type") == "control" else "navigate",
            "action": action,
            "session": "test-session",
            "seq": seq,
            "robot_id": "robo-1",
        }
    ).encode()


def test_decode_normalizes_aiwaiter_control_and_angle() -> None:
    decoded = decode_packet(
        packet({"type": "control", "verb": "LEFT", "angle_deg": "45"})
    )
    assert decoded is not None
    assert decoded["action"] == {
        "type": "control",
        "verb": "left",
        "angle_deg": 45.0,
    }


def test_decode_accepts_string_action_variant() -> None:
    decoded = decode_packet(
        json.dumps(
            {
                "v": 1,
                "kind": "control",
                "action": "STOP",
                "session": "s",
                "seq": 2,
            }
        ).encode()
    )
    assert decoded is not None
    assert decoded["action"] == {"type": "control", "verb": "stop"}


def test_decode_rejects_bad_wire_version_and_unknown_kind() -> None:
    assert decode_packet(b'{"v":2,"kind":"control"}') is None
    assert decode_packet(b'{"v":1,"kind":"teleport"}') is None
    assert decode_packet(b"not-json") is None


def test_deduper_collapses_aiwaiter_repeats_by_session_and_sequence() -> None:
    deduper = Deduper()
    first = decode_packet(packet({"type": "control", "verb": "STOP"}, seq=7))
    second = decode_packet(packet({"type": "control", "verb": "STOP"}, seq=7))
    assert first is not None and second is not None
    assert deduper.duplicate(first) is False
    assert deduper.duplicate(second) is True


def test_ack_matches_aiwaiter_sender_schema() -> None:
    decoded = decode_packet(packet({"type": "control", "verb": "STOP"}, seq=9))
    assert decoded is not None
    ack = json.loads(ack_packet(decoded).decode())
    assert ack == {
        "kind": "ack",
        "session": "test-session",
        "seq": 9,
        "robot_id": "robo-1",
        "v": 1,
    }


def test_a_pick_uses_warm_inner_mission_without_pick_box_startup_wrapper(tmp_path: Path) -> None:
    mission = MissionProcess(tmp_path, dry_run=True)
    # Capture the dry-run command without launching ROS/Gazebo.
    import logging

    records: list[str] = []
    handler = logging.Handler()
    handler.emit = lambda record: records.append(record.getMessage())  # type: ignore[method-assign]
    logger = logging.getLogger("warehouse_udp_bridge")
    logger.addHandler(handler)
    try:
        mission.start_pick_a("blue", deliver=True)
    finally:
        logger.removeHandler(handler)
    assert any("run_storage_pick.sh" in line for line in records)
    assert all("pick_box.sh" not in line for line in records)
