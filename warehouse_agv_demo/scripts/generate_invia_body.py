#!/usr/bin/env python3
"""Generate the smooth, open-top AGV body used by the Gazebo model."""

from __future__ import annotations

import math
from pathlib import Path


SEGMENTS_PER_CORNER = 10
ROBOT_SCALE = 0.70


def rounded_ring(half_x: float, half_y: float, radius: float, z: float):
    points = []
    corners = (
        (1.0, 1.0, 0.0),
        (-1.0, 1.0, math.pi / 2.0),
        (-1.0, -1.0, math.pi),
        (1.0, -1.0, 3.0 * math.pi / 2.0),
    )
    for sx, sy, start in corners:
        cx = sx * (half_x - radius)
        cy = sy * (half_y - radius)
        for step in range(SEGMENTS_PER_CORNER):
            angle = start + step * (math.pi / 2.0) / SEGMENTS_PER_CORNER
            points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle), z))
    return points


def add_strip(faces, first_a: int, first_b: int, count: int, reverse: bool = False):
    for i in range(count):
        j = (i + 1) % count
        if reverse:
            faces.append((first_a + i, first_b + i, first_b + j, first_a + j))
        else:
            faces.append((first_a + i, first_a + j, first_b + j, first_b + i))


def write_collada(
    path: Path,
    vertices,
    faces,
    material_name: str,
    ambient,
    diffuse,
    specular,
) -> None:
    triangles = []
    for face in faces:
        zero_based = [index - 1 for index in face]
        for index in range(1, len(zero_based) - 1):
            triangles.append((zero_based[0], zero_based[index], zero_based[index + 1]))

    normal_sums = [[0.0, 0.0, 0.0] for _ in vertices]
    for a, b, c in triangles:
        ax, ay, az = vertices[a]
        bx, by, bz = vertices[b]
        cx, cy, cz = vertices[c]
        ux, uy, uz = bx - ax, by - ay, bz - az
        vx, vy, vz = cx - ax, cy - ay, cz - az
        nx = uy * vz - uz * vy
        ny = uz * vx - ux * vz
        nz = ux * vy - uy * vx
        for vertex_index in (a, b, c):
            normal_sums[vertex_index][0] += nx
            normal_sums[vertex_index][1] += ny
            normal_sums[vertex_index][2] += nz
    normals = []
    for nx, ny, nz in normal_sums:
        length = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
        normals.append((nx / length, ny / length, nz / length))

    geometry_id = path.stem
    positions = " ".join(f"{value:.6f}" for vertex in vertices for value in vertex)
    normal_values = " ".join(f"{value:.6f}" for normal in normals for value in normal)
    indices = " ".join(
        f"{vertex_index} {vertex_index}"
        for triangle in triangles
        for vertex_index in triangle
    )
    xml = f'''<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
  <asset><unit meter="1" name="meter"/><up_axis>Z_UP</up_axis></asset>
  <library_effects>
    <effect id="{material_name}-effect"><profile_COMMON><technique sid="common"><phong>
      <emission><color>0 0 0 1</color></emission>
      <ambient><color>{ambient[0]} {ambient[1]} {ambient[2]} 1</color></ambient>
      <diffuse><color>{diffuse[0]} {diffuse[1]} {diffuse[2]} 1</color></diffuse>
      <specular><color>{specular[0]} {specular[1]} {specular[2]} 1</color></specular>
      <shininess><float>64</float></shininess>
    </phong></technique></profile_COMMON></effect>
  </library_effects>
  <library_materials><material id="{material_name}" name="{material_name}"><instance_effect url="#{material_name}-effect"/></material></library_materials>
  <library_geometries><geometry id="{geometry_id}-mesh" name="{geometry_id}"><mesh>
    <source id="{geometry_id}-positions"><float_array id="{geometry_id}-positions-array" count="{len(vertices) * 3}">{positions}</float_array><technique_common><accessor source="#{geometry_id}-positions-array" count="{len(vertices)}" stride="3"><param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/></accessor></technique_common></source>
    <source id="{geometry_id}-normals"><float_array id="{geometry_id}-normals-array" count="{len(normals) * 3}">{normal_values}</float_array><technique_common><accessor source="#{geometry_id}-normals-array" count="{len(normals)}" stride="3"><param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/></accessor></technique_common></source>
    <vertices id="{geometry_id}-vertices"><input semantic="POSITION" source="#{geometry_id}-positions"/></vertices>
    <triangles material="{material_name}-symbol" count="{len(triangles)}"><input semantic="VERTEX" source="#{geometry_id}-vertices" offset="0"/><input semantic="NORMAL" source="#{geometry_id}-normals" offset="1"/><p>{indices}</p></triangles>
  </mesh></geometry></library_geometries>
  <library_visual_scenes><visual_scene id="Scene" name="Scene"><node id="{geometry_id}" name="{geometry_id}"><instance_geometry url="#{geometry_id}-mesh"><bind_material><technique_common><instance_material symbol="{material_name}-symbol" target="#{material_name}"/></technique_common></bind_material></instance_geometry></node></visual_scene></library_visual_scenes>
  <scene><instance_visual_scene url="#Scene"/></scene>
</COLLADA>
'''
    path.write_text(xml, encoding="utf-8")


def main() -> None:
    # z is relative to base_link. The link itself is at model z=0.24 m.
    # Multiple tapered rings give the body its low, rounded inVia silhouette.
    outer_profiles = (
        (0.17, 0.145, 0.050, -0.17),
        (0.195, 0.165, 0.060, -0.125),
        (0.20, 0.171, 0.065, 0.040),
        (0.195, 0.165, 0.060, 0.125),
        (0.185, 0.158, 0.055, 0.210),
    )
    # The cavity follows the forward-offset lifting platform. Keeping it only
    # as long as the linkage removes the large unused pocket at the rear.
    inner_top = (0.160, 0.130, 0.035, 0.210)
    # The cavity floor is almost level with the base-link origin (model
    # z=0.245 m), deep enough to hide roughly 90% of the folded scissor pack.
    inner_floor = (0.145, 0.115, 0.030, 0.005)
    cavity_center_x = 0.0

    vertices = []
    faces = []
    ring_starts = []
    for profile in outer_profiles:
        ring_starts.append(len(vertices) + 1)
        vertices.extend(rounded_ring(*profile))

    count = 4 * SEGMENTS_PER_CORNER
    for lower, upper in zip(ring_starts, ring_starts[1:]):
        add_strip(faces, lower, upper, count)

    inner_top_start = len(vertices) + 1
    vertices.extend(
        (x + cavity_center_x, y, z) for x, y, z in rounded_ring(*inner_top)
    )
    inner_floor_start = len(vertices) + 1
    vertices.extend(
        (x + cavity_center_x, y, z) for x, y, z in rounded_ring(*inner_floor)
    )

    # Rounded top shoulder, inward-facing cavity wall and cavity floor.
    add_strip(faces, ring_starts[-1], inner_top_start, count)
    add_strip(faces, inner_top_start, inner_floor_start, count, reverse=True)

    bottom_center = len(vertices) + 1
    vertices.append((0.0, 0.0, outer_profiles[0][3]))
    floor_center = len(vertices) + 1
    vertices.append((cavity_center_x, 0.0, inner_floor[3]))
    for i in range(count):
        j = (i + 1) % count
        faces.append((bottom_center, ring_starts[0] + j, ring_starts[0] + i))
        faces.append((floor_center, inner_floor_start + i, inner_floor_start + j))

    vertices = [tuple(value * ROBOT_SCALE for value in vertex) for vertex in vertices]

    output = Path(__file__).resolve().parents[1] / "models/warehouse_agv/meshes/invia_body.obj"
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Generated by scripts/generate_invia_body.py",
        "mtllib invia_body.mtl",
        "o invia_picker_body",
        "usemtl graphite_body",
        "s 1",
    ]
    lines.extend(f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in vertices)
    lines.extend("f " + " ".join(str(index) for index in face) for face in faces)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    material = output.with_suffix(".mtl")
    material.write_text(
        "# Graphite inVia-style body material\n"
        "newmtl graphite_body\n"
        "Ka 0.030 0.034 0.040\n"
        "Kd 0.060 0.067 0.078\n"
        "Ks 0.420 0.420 0.450\n"
        "Ns 72.0\n"
        "d 1.0\n"
        "illum 2\n",
        encoding="utf-8",
    )
    write_collada(
        output.with_suffix(".dae"),
        vertices,
        faces,
        "graphite_body",
        (0.025, 0.029, 0.035),
        (0.055, 0.062, 0.074),
        (0.42, 0.42, 0.45),
    )

    # A separate, very thin black bumper wraps only the bottom perimeter. It
    # is intentionally not a second tall chassis section.
    bumper_vertices = []
    bumper_faces = []
    bumper_lower = rounded_ring(0.175, 0.150, 0.052, -0.175)
    bumper_upper = rounded_ring(0.205, 0.175, 0.062, -0.115)
    bumper_vertices.extend(bumper_lower)
    bumper_vertices.extend(bumper_upper)
    add_strip(bumper_faces, 1, count + 1, count)
    bumper_bottom_center = len(bumper_vertices) + 1
    bumper_vertices.append((0.0, 0.0, -0.175))
    bumper_top_center = len(bumper_vertices) + 1
    bumper_vertices.append((0.0, 0.0, -0.115))
    for i in range(count):
        j = (i + 1) % count
        bumper_faces.append((bumper_bottom_center, 1 + j, 1 + i))
        bumper_faces.append((bumper_top_center, count + 1 + i, count + 1 + j))
    bumper_vertices = [
        tuple(value * ROBOT_SCALE for value in vertex) for vertex in bumper_vertices
    ]

    bumper_output = output.with_name("invia_bumper.obj")
    bumper_lines = [
        "# Generated by scripts/generate_invia_body.py",
        "mtllib invia_bumper.mtl",
        "o invia_picker_bumper",
        "usemtl black_bumper",
        "s 1",
    ]
    bumper_lines.extend(
        f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in bumper_vertices
    )
    bumper_lines.extend(
        "f " + " ".join(str(index) for index in face) for face in bumper_faces
    )
    bumper_output.write_text("\n".join(bumper_lines) + "\n", encoding="utf-8")
    bumper_output.with_suffix(".mtl").write_text(
        "# Thin black perimeter bumper\n"
        "newmtl black_bumper\n"
        "Ka 0.008 0.009 0.011\n"
        "Kd 0.014 0.016 0.020\n"
        "Ks 0.240 0.240 0.260\n"
        "Ns 36.0\n"
        "d 1.0\n"
        "illum 2\n",
        encoding="utf-8",
    )
    write_collada(
        bumper_output.with_suffix(".dae"),
        bumper_vertices,
        bumper_faces,
        "black_bumper",
        (0.006, 0.007, 0.009),
        (0.012, 0.014, 0.018),
        (0.24, 0.24, 0.26),
    )

    # Red accent surfaces follow the tapered side profile instead of floating
    # rectangular bars. Both sides are included in a single mesh.
    stripe_vertices = []
    stripe_faces = []
    stripe_rows = (
        (-0.14, 0.150, 0.015),
        (-0.04, 0.168, 0.035),
        (0.07, 0.168, 0.055),
        (0.17, 0.158, 0.075),
    )
    for side in (1.0, -1.0):
        start = len(stripe_vertices) + 1
        for z, surface_y, center_x in stripe_rows:
            y = side * (surface_y + 0.004)
            stripe_vertices.extend(
                ((center_x - 0.020, y, z), (center_x + 0.020, y, z))
            )
        for row in range(len(stripe_rows) - 1):
            lower_left = start + row * 2
            lower_right = lower_left + 1
            upper_left = lower_left + 2
            upper_right = lower_left + 3
            face = (lower_left, upper_left, upper_right, lower_right)
            stripe_faces.append(face if side > 0 else tuple(reversed(face)))
    stripe_vertices = [
        tuple(value * ROBOT_SCALE for value in vertex) for vertex in stripe_vertices
    ]
    stripe_output = output.with_name("invia_red_stripe.dae")
    write_collada(
        stripe_output,
        stripe_vertices,
        stripe_faces,
        "red_accent",
        (0.50, 0.010, 0.015),
        (0.82, 0.018, 0.026),
        (0.45, 0.20, 0.20),
    )
    print(output)
    print(bumper_output)
    print(stripe_output)


if __name__ == "__main__":
    main()
