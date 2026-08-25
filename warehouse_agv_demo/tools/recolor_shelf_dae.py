#!/usr/bin/env python3
"""Assign one solid material to each individual box in the MovAi shelf mesh."""

from __future__ import annotations

import copy
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


NS = "http://www.collada.org/2005/11/COLLADASchema"
ET.register_namespace("", NS)
Q = lambda name: f"{{{NS}}}{name}"

PALETTE = (
    ("blue", "0.04 0.28 0.88 1"),
    ("red", "0.88 0.06 0.05 1"),
    ("green", "0.04 0.58 0.18 1"),
    ("yellow", "0.96 0.65 0.02 1"),
    ("orange", "0.95 0.30 0.03 1"),
    ("brown", "0.48 0.22 0.07 1"),
    ("cyan", "0.02 0.62 0.75 1"),
    ("purple", "0.48 0.12 0.72 1"),
)


def add_materials(root: ET.Element) -> None:
    effects = root.find(Q("library_effects"))
    materials = root.find(Q("library_materials"))
    if effects is None or materials is None:
        raise RuntimeError("COLLADA material libraries not found")

    for name, rgba in PALETTE:
        effect = ET.SubElement(effects, Q("effect"), {"id": f"box_color_{name}-effect"})
        profile = ET.SubElement(effect, Q("profile_COMMON"))
        technique = ET.SubElement(profile, Q("technique"), {"sid": "common"})
        lambert = ET.SubElement(technique, Q("lambert"))
        diffuse = ET.SubElement(lambert, Q("diffuse"))
        ET.SubElement(diffuse, Q("color"), {"sid": "diffuse"}).text = rgba

        material = ET.SubElement(
            materials, Q("material"),
            {"id": f"box_color_{name}-material", "name": f"box_color_{name}"},
        )
        ET.SubElement(material, Q("instance_effect"), {"url": f"#box_color_{name}-effect"})


def split_box_mesh(root: ET.Element) -> int:
    geometry = root.find(f".//{Q('geometry')}[@id='Cube_026-mesh']")
    if geometry is None:
        raise RuntimeError("MovAi merged box geometry not found")
    mesh = geometry.find(Q("mesh"))
    triangles = mesh.find(f"{Q('triangles')}[@material='Material_001-material']") if mesh is not None else None
    if mesh is None or triangles is None:
        raise RuntimeError("Merged box triangle set not found")

    count = int(triangles.attrib["count"])
    inputs = triangles.findall(Q("input"))
    stride = max(int(item.attrib.get("offset", "0")) for item in inputs) + 1
    indices = [int(value) for value in triangles.find(Q("p")).text.split()]
    triangles_per_box = 12
    values_per_box = triangles_per_box * 3 * stride
    if count % triangles_per_box or len(indices) != count * 3 * stride:
        raise RuntimeError("Unexpected box topology; refusing an unsafe mesh rewrite")

    insertion_index = list(mesh).index(triangles)
    mesh.remove(triangles)
    box_count = count // triangles_per_box
    # Offset the palette by shelf row so adjacent cubes do not form simple stripes.
    order = (0, 3, 1, 6, 2, 4, 7, 5)
    for box_index in range(box_count):
        color_index = order[(box_index * 5 + box_index // 7) % len(order)]
        color_name = PALETTE[color_index][0]
        part = ET.Element(
            Q("triangles"),
            {"material": f"box_color_{color_name}-material", "count": str(triangles_per_box)},
        )
        for source_input in inputs:
            part.append(copy.deepcopy(source_input))
        start = box_index * values_per_box
        ET.SubElement(part, Q("p")).text = " ".join(map(str, indices[start:start + values_per_box]))
        mesh.insert(insertion_index + box_index, part)
    return box_count


def bind_materials(root: ET.Element) -> None:
    instance = root.find(f".//{Q('instance_geometry')}[@url='#Cube_026-mesh']")
    common = instance.find(f"{Q('bind_material')}/{Q('technique_common')}") if instance is not None else None
    if common is None:
        raise RuntimeError("Box instance material bindings not found")
    for name, _ in PALETTE:
        ET.SubElement(
            common, Q("instance_material"),
            {"symbol": f"box_color_{name}-material", "target": f"#box_color_{name}-material"},
        )


def main(path: Path) -> None:
    tree = ET.parse(path)
    root = tree.getroot()
    add_materials(root)
    box_count = split_box_mesh(root)
    bind_materials(root)
    tree.write(path, encoding="utf-8", xml_declaration=True)
    print(f"Assigned individual solid colors to {box_count} boxes in {path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} SHELF.dae")
    main(Path(sys.argv[1]))
