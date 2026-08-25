"""Pure retry-budget helper shared by the physical grasp mission and tests."""

from __future__ import annotations


def grasp_attempts(max_retries: int) -> range:
    """Return one initial attempt plus exactly ``max_retries`` retries."""
    maximum = int(max_retries)
    if maximum < 0:
        raise ValueError("max_retries must not be negative")
    return range(1, maximum + 2)

