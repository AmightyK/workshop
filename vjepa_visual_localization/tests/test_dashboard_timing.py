from __future__ import annotations

from scripts.localization_dashboard import gui_wait_ms


def test_gui_wait_uses_only_remaining_frame_budget() -> None:
    assert gui_wait_ms(10.032, 10.020) == 13


def test_gui_wait_never_adds_a_full_delay_after_overrun() -> None:
    assert gui_wait_ms(10.020, 10.050) == 1
