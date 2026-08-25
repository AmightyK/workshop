from __future__ import annotations

from scripts.vjepa_image_relay import LatestFrameSlot


def test_latest_frame_slot_drops_superseded_frames() -> None:
    slot = LatestFrameSlot()

    first_sequence = slot.put("old")
    latest_sequence = slot.put("new")

    assert first_sequence == 1
    assert latest_sequence == 2
    assert slot.latest_after(0) == (2, "new")
    assert slot.latest_after(2) is None
