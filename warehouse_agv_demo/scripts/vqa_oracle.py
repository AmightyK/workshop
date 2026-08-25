#!/usr/bin/env python3
"""Deterministic stand-in for the Orin VQA service used by this simulation."""

from __future__ import annotations

import re
import unicodedata


COLOR_TERMS = {
    "blue": ("blue", "xanh duong", "xanh dương", "mau xanh"),
    "red": ("red", "do", "đỏ", "mau do", "màu đỏ"),
    "green": ("green", "xanh la", "xanh lá", "mau xanh la", "màu xanh lá"),
}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    return re.sub(r"\s+", " ", text)


def answer(command: str, config: dict) -> dict:
    """Return the structured answer that the unavailable Orin VQA would send."""
    normalized = normalize(command)
    matches = []
    for color, terms in COLOR_TERMS.items():
        if any(normalize(term) in normalized for term in terms):
            matches.append(color)
    if len(matches) != 1:
        raise ValueError(
            "VQA oracle needs exactly one target color: blue/xanh dương, "
            "red/đỏ, or green/xanh lá"
        )

    color = matches[0]
    zone_match = re.search(r"(?:storage|khu|ke)\s*([abc])\b", normalized)
    requested_location = (
        f"storage_{zone_match.group(1).upper()}" if zone_match else None
    )
    candidates = [
        (object_name, item)
        for object_name, item in config["objects"].items()
        if item["color"] == color
        and (requested_location is None or item["location"] == requested_location)
    ]
    if len(candidates) != 1:
        choices = ", ".join(
            f"{name} at {item['location']}/{item['slot']}"
            for name, item in candidates
        ) or "none"
        raise ValueError(
            f"VQA target is ambiguous or unavailable for color '{color}'. "
            f"Specify Storage A, B, or C; candidates: {choices}"
        )

    object_name, item = candidates[0]
    location = item["location"]
    slot_name = item["slot"]
    slot = config["stations"][location]["slots"][slot_name]

    return {
        "mode": "simulated_orin_vqa",
        "question": "Which requested box is visible, and where should the robot pick it?",
        "answer": {
            "object": object_name,
            "model": item["model"],
            "storage": location,
            "slot": slot_name,
            "pickup_anchor": slot["anchor"],
            "pickup_pose": slot["approach"],
            "destination": "packing_station",
            "destination_anchor": "dropoff_PACK01",
        },
        "confidence": 1.0,
    }
