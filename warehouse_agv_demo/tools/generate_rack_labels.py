#!/usr/bin/env python3
"""Generate lightweight rack signs and semantic locations for every rack."""

import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "models" / "rack_signs" / "model.sdf"
LOCATION_OUTPUT = ROOT / "config" / "inventory_locations.yaml"

# The first three racks already have the original hand-drawn A/B/C signs in
# storage_markers. Every remaining small rack gets the same size, mounting
# height, colors and two-sided presentation from this generator.
SMALL_RACKS = (
    ("A", "shelf", -4.91528, -0.690987, 0.0),
    ("B", "shelf_0", -4.91528, 2.30697, 0.0),
    ("C", "shelf_1", -4.91528, 5.30708, 0.0),
    ("D", "shelf_2", -4.91528, 8.34352, 0.0),
    ("E", "shelf_5", 5.10144, -0.690987, 0.0),
    ("F", "shelf_6", 5.10144, 2.30697, 0.0),
    ("G", "shelf_4", 5.10144, 5.30708, 0.0),
    ("H", "shelf_3", 5.10144, 8.34352, 0.0),
    ("I", "shelf_7", 12.8818, -21.2416, 0.0),
    ("J", "shelf_8", 12.8818, -19.0028, 0.0),
    ("K", "shelf_9", 12.8818, -16.4478, 0.0),
    ("L", "shelf_10", 12.8818, -14.1028, 0.0),
    # Three compact, separated racks replace each old 18 m shelf_big row.
    ("M", "shelf_11", -9.84177, -19.5598, 1.5708),
    ("N", "shelf_12", -9.84177, -13.5598, 1.5708),
    ("O", "shelf_13", -9.84177, -7.5598, 1.5708),
    ("P", "shelf_14", 5.69777, -18.9647, 1.5708),
    ("Q", "shelf_15", 5.69777, -12.9647, 1.5708),
    ("R", "shelf_16", 5.69777, -6.9647, 1.5708),
    ("S", "shelf_17", 0.094376, -18.9647, 1.5708),
    ("T", "shelf_18", 0.094376, -12.9647, 1.5708),
    ("U", "shelf_19", 0.094376, -6.9647, 1.5708),
    ("V", "shelf_20", -5.86284, -18.9647, 1.5708),
    ("W", "shelf_21", -5.86284, -12.9647, 1.5708),
    ("X", "shelf_22", -5.86284, -6.9647, 1.5708),
    ("Y", "shelf_23", 13.4821, 9.3190, 1.5708),
    ("Z", "shelf_24", 13.4821, 15.3190, 1.5708),
    ("AA", "shelf_25", 13.4821, 21.3190, 1.5708),
)

SMALL_COLUMNS = (-1.35, -0.45, 0.45, 1.35)

# Named strokes keep difficult letters such as I and K unambiguous. Values are
# local X/Z centers, box X/Z sizes and pitch (rotation in the sign plane).
STROKES = {
    "top": (0.00, 0.245, 0.46, 0.09, 0.0),
    "middle": (0.00, 0.000, 0.46, 0.09, 0.0),
    "bottom": (0.00, -0.245, 0.46, 0.09, 0.0),
    "left": (-0.22, 0.000, 0.10, 0.58, 0.0),
    "right": (0.22, 0.000, 0.10, 0.58, 0.0),
    "lower_left": (-0.22, -0.122, 0.10, 0.34, 0.0),
    "lower_right": (0.22, -0.122, 0.10, 0.34, 0.0),
    "center": (0.00, 0.000, 0.10, 0.58, 0.0),
    "upper_diagonal": (0.09, 0.125, 0.36, 0.08, -0.78),
    "lower_diagonal": (0.09, -0.125, 0.36, 0.08, 0.78),
    "upper_left": (-0.22, 0.122, 0.10, 0.34, 0.0),
    "upper_right": (0.22, 0.122, 0.10, 0.34, 0.0),
    "a_left": (-0.10, 0.000, 0.10, 0.58, 0.35),
    "a_right": (0.10, 0.000, 0.10, 0.58, -0.35),
    "rising_diagonal": (0.00, 0.000, 0.62, 0.08, -0.72),
    "falling_diagonal": (0.00, 0.000, 0.62, 0.08, 0.72),
    "upper_in_left": (-0.10, 0.125, 0.36, 0.08, 0.78),
    "upper_in_right": (0.10, 0.125, 0.36, 0.08, -0.78),
    "lower_in_left": (-0.10, -0.125, 0.36, 0.08, -0.78),
    "lower_in_right": (0.10, -0.125, 0.36, 0.08, 0.78),
}

GLYPHS = {
    "A": ("a_left", "a_right", "middle"),
    "D": ("top", "bottom", "left", "right"),
    "E": ("top", "middle", "bottom", "left"),
    "F": ("top", "middle", "left"),
    "G": ("top", "middle", "bottom", "left", "lower_right"),
    "H": ("left", "right", "middle"),
    "I": ("top", "bottom", "center"),
    "J": ("top", "right", "bottom", "lower_left"),
    "K": ("left", "upper_diagonal", "lower_diagonal"),
    "L": ("left", "bottom"),
    "M": ("left", "right", "upper_in_left", "upper_in_right"),
    "N": ("left", "right", "falling_diagonal"),
    "O": ("top", "bottom", "left", "right"),
    "P": ("top", "middle", "left", "upper_right"),
    "Q": ("top", "bottom", "left", "right", "lower_diagonal"),
    "R": ("top", "middle", "left", "upper_right", "lower_diagonal"),
    "S": ("top", "middle", "bottom", "upper_left", "lower_right"),
    "T": ("top", "center"),
    "U": ("left", "right", "bottom"),
    "V": ("a_left", "a_right"),
    "W": ("left", "right", "lower_in_left", "lower_in_right"),
    "X": ("rising_diagonal", "falling_diagonal"),
    "Y": ("upper_in_left", "upper_in_right", "center"),
    "Z": ("top", "bottom", "rising_diagonal"),
}
BLACK = "0.01 0.01 0.01 1"
BOARD = "0.96 0.96 0.93 1"


def material(color: str) -> str:
    return (
        f"<material><ambient>{color}</ambient><diffuse>{color}</diffuse>"
        "</material>"
    )


def glyph_visuals(label: str, side: str, face_y: float, mirror: bool) -> list[str]:
    result = []
    scale = 0.72 if len(label) > 1 else 1.0
    offsets = (0.0,) if len(label) == 1 else (-0.22, 0.22)
    for character_index, (character, offset) in enumerate(zip(label, offsets)):
        for stroke in GLYPHS[character]:
            x, z, width, height, pitch = STROKES[stroke]
            x = offset + x * scale
            z *= scale
            width *= scale
            height *= scale
            if mirror:
                x = -x
                pitch = -pitch
            result.append(
                f'      <visual name="{label}_{side}_{character_index}_{stroke}">'
                f'<pose>{x:.3f} {face_y:.3f} {z:.3f} 0 {pitch:.3f} 0</pose>'
                f'<geometry><box><size>{width:.3f} 0.035 {height:.3f}</size>'
                f'</box></geometry>{material(BLACK)}</visual>'
            )
    return result


def append_sign(
    lines: list[str], label: str, rack_x: float, rack_y: float, rack_yaw: float
) -> None:
    lines.extend((
        f'    <link name="storage_{label}_sign">',
        f'      <pose>{rack_x:.5f} {rack_y:.6f} 2.25 0 0 {rack_yaw:.4f}</pose>',
        '      <visual name="board"><geometry><box><size>1.40 0.07 0.76</size>'
        f'</box></geometry>{material(BOARD)}</visual>',
    ))
    lines.extend(glyph_visuals(label, "front", -0.055, False))
    lines.extend(glyph_visuals(label, "back", 0.055, True))
    lines.extend((
        '      <visual name="left_mount"><pose>-0.48 0 -0.62 0 0 0</pose>'
        '<geometry><box><size>0.06 0.06 0.50</size></box></geometry>'
        f'{material(BLACK)}</visual>',
        '      <visual name="right_mount"><pose>0.48 0 -0.62 0 0 0</pose>'
        '<geometry><box><size>0.06 0.06 0.50</size></box></geometry>'
        f'{material(BLACK)}</visual>',
        '    </link>',
    ))


def main() -> None:
    lines = [
        '<?xml version="1.0"?>',
        '<sdf version="1.9">',
        '  <model name="rack_zone_labels">',
        '    <static>true</static>',
    ]
    yaml_lines = [
        "# Generated semantic locations for every compact rack (A-AA).",
        "frame_id: map",
        "locations:",
    ]

    # A-C live in storage_markers; generate matching signs for every other rack.
    for label, _, rack_x, rack_y, rack_yaw in SMALL_RACKS[3:]:
        append_sign(lines, label, rack_x, rack_y, rack_yaw)

    location_count = 0
    for label, entity, rack_x, rack_y, rack_yaw in SMALL_RACKS:
        cosine, sine = math.cos(rack_yaw), math.sin(rack_yaw)
        for column, offset_x in enumerate(SMALL_COLUMNS):
            location = f"{label}{column:02d}"
            local_x, local_y = offset_x, -1.1
            approach_x = rack_x + cosine * local_x - sine * local_y
            approach_y = rack_y + sine * local_x + cosine * local_y
            approach_yaw = math.atan2(
                math.sin(rack_yaw + math.pi / 2.0),
                math.cos(rack_yaw + math.pi / 2.0),
            )
            yaml_lines.append(
                f"  {location}: {{rack: {entity}, zone: {label}, column: {column}, "
                f"x: {approach_x:.5f}, y: {approach_y:.6f}, "
                f"yaw: {approach_yaw:.4f}}}"
            )
            location_count += 1

    lines.extend(('  </model>', '</sdf>', ''))
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    LOCATION_OUTPUT.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    print(
        f"Generated {location_count} locations across A-AA and "
        f"{len(SMALL_RACKS) - 3} matching signs for D-AA"
    )


if __name__ == "__main__":
    main()
