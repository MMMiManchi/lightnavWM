"""Draw a top-down map of each marked decision node on the walked route.

The floor is the shipped Matterport navmesh, used only as a picture of where
the robot was. It is not a new collection and it does not enter navigation.
Run with the Habitat 3.9 interpreter.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import habitat_sim
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
NODES = ROOT / "output/world_model_diagnostics/decision_nodes_20261103_01/decision_nodes.jsonl"
OUT = ROOT / "output/world_model_diagnostics/decision_nodes_20261103_01/bev"
SPLITS = {
    "formal_dev_20261028_01": ROOT / "output/world_model_data/formal_dev_20261028_01",
    "formal_fit_20261028_04": ROOT / "output/world_model_data/formal_fit_20261028_04",
}
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
MAP = 980
MARGIN = 36
PANEL = 460
FLOOR_Y = 0.75
SPLIT_Y = 1.2


def main() -> None:
    nodes = [json.loads(line) for line in NODES.open(encoding="utf-8")]
    poses = {}
    for root in SPLITS.values():
        poses.update(_poses(root))
    by_episode: dict[str, list[dict]] = defaultdict(list)
    for node in nodes:
        by_episode[node["episode_uid"]].append(node)
    for items in by_episode.values():
        items.sort(key=lambda item: int(item["step_index"]))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "episodes").mkdir(exist_ok=True)
    (OUT / "scenes").mkdir(exist_ok=True)
    meshes = {}
    pages = []
    for episode_uid, episode_nodes in sorted(by_episode.items()):
        scene = episode_nodes[0]["scene"]
        if scene not in meshes:
            meshes[scene] = _mesh(scene)
        path = poses[episode_uid]
        for band, path_band, node_band in _bands(path, episode_nodes):
            name = f"{scene}_{episode_uid.split('|')[-1]}_y{band:.1f}.png"
            _draw_episode(meshes[scene], path_band, node_band, OUT / "episodes" / name)
            pages.append(
                {
                    "kind": "episode",
                    "scene": scene,
                    "episode_id": episode_uid.split("|")[-1],
                    "split": episode_nodes[0]["split"],
                    "floor_y": band,
                    "nodes": len(node_band),
                    "file": f"episodes/{name}",
                }
            )
    scene_groups: dict[tuple[str, float], list] = defaultdict(list)
    scene_bands: dict[str, list[float]] = defaultdict(list)
    for episode_uid, episode_nodes in by_episode.items():
        scene = episode_nodes[0]["scene"]
        path = poses[episode_uid]
        for band, path_band, node_band in _bands(path, episode_nodes):
            canon = next((item for item in scene_bands[scene] if abs(item - band) < 0.8), None)
            if canon is None:
                scene_bands[scene].append(band)
                canon = band
            scene_groups[(scene, canon)].append((path_band, node_band, episode_uid))
    for (scene, band), groups in sorted(scene_groups.items()):
        name = f"{scene}_y{band:.1f}.png"
        _draw_scene(meshes[scene], groups, OUT / "scenes" / name)
        pages.append(
            {
                "kind": "scene",
                "scene": scene,
                "floor_y": band,
                "episodes": len(groups),
                "nodes": sum(len(nodes) for _path, nodes, _uid in groups),
                "file": f"scenes/{name}",
            }
        )
    (OUT / "pages.json").write_text(json.dumps(pages, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(pages)} maps to {OUT}")


def _poses(root: Path) -> dict[str, list[dict]]:
    labels = {}
    with (root / "privileged_labels.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            pose = row["pose"]
            labels[row["label_id"]] = {
                "position": [float(v) for v in pose["world_position_xyz_m"]],
                "yaw": float(pose["derived_yaw_rad"]),
            }
    grouped: dict[str, list[dict]] = defaultdict(list)
    with (root / "transitions.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            grouped[json.loads(line)["episode_uid"]].append(json.loads(line))
    episodes = {}
    for episode_uid, rows in grouped.items():
        rows.sort(key=lambda row: row["t"])
        sequence = [dict(labels[rows[0]["pose_before_id"]])]
        for row in rows:
            sequence.append(dict(labels[row["pose_after_id"]]))
        episodes[episode_uid] = sequence
    return episodes


def _mesh(scene: str):
    path = ROOT / "data/scene_datasets/mp3d" / scene / f"{scene}.navmesh"
    finder = habitat_sim.PathFinder()
    if not finder.load_nav_mesh(str(path)):
        raise RuntimeError(f"cannot load {path}")
    vertices = np.asarray(finder.build_navmesh_vertices(), dtype=np.float64)
    indices = np.asarray(finder.build_navmesh_vertex_indices(), dtype=np.int64).reshape(-1, 3)
    return vertices, indices


def _bands(path: list[dict], nodes: list[dict]):
    ys = [point["position"][1] for point in path]
    lo, hi = min(ys), max(ys)
    if hi - lo <= SPLIT_Y:
        yield ((lo + hi) / 2, path, nodes)
        return
    mid = (lo + hi) / 2
    lower_path = [point for point in path if point["position"][1] <= mid]
    upper_path = [point for point in path if point["position"][1] > mid]
    lower_nodes = [node for node in nodes if node["position"][1] <= mid]
    upper_nodes = [node for node in nodes if node["position"][1] > mid]
    if lower_nodes and lower_path:
        ys = [point["position"][1] for point in lower_path]
        yield ((min(ys) + max(ys)) / 2, lower_path, lower_nodes)
    if upper_nodes and upper_path:
        ys = [point["position"][1] for point in upper_path]
        yield ((min(ys) + max(ys)) / 2, upper_path, upper_nodes)


def _draw_episode(mesh, path: list[dict], nodes: list[dict], dest: Path) -> None:
    cloud = [(point["position"][0], point["position"][2]) for point in path]
    minx, minz, span = _bounds(cloud)
    pix = _projector(minx, minz, span)
    image = Image.new("RGB", (MAP + PANEL, MAP), (248, 247, 244))
    _paint_floor(image, mesh, pix, path, minx, minz, span)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(FONT, 16)
    small = ImageFont.truetype(FONT, 15)
    draw.line([pix(p[0], p[1]) for p in cloud], fill=(36, 84, 168), width=4)
    start = path[0]["position"]
    _dot(draw, pix(start[0], start[2]), 6, (20, 130, 70))
    draw.text((pix(start[0], start[2])[0] + 8, pix(start[0], start[2])[1] - 8), "start", fill=(20, 110, 60), font=small)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    over = ImageDraw.Draw(overlay)
    for index, node in enumerate(nodes, start=1):
        _wedge(over, pix, node, span)
    image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(image)
    for index, node in enumerate(nodes, start=1):
        _marker(draw, pix, node, str(index), font, span)
    _legend(draw, font, small)
    _episode_notes(draw, small, path, nodes)
    image.save(dest)


def _draw_scene(mesh, groups, dest: Path) -> None:
    cloud = [
        (point["position"][0], point["position"][2])
        for path, _nodes, _uid in groups
        for point in path
    ]
    minx, minz, span = _bounds(cloud)
    pix = _projector(minx, minz, span)
    flat = []
    for path, _nodes, _uid in groups:
        flat.extend(path)
    image = Image.new("RGB", (MAP + PANEL, MAP), (248, 247, 244))
    _paint_floor(image, mesh, pix, flat, minx, minz, span)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(FONT, 16)
    small = ImageFont.truetype(FONT, 15)
    colors = (
        (36, 84, 168),
        (20, 130, 130),
        (120, 70, 160),
        (150, 90, 30),
        (40, 110, 60),
        (160, 50, 90),
        (70, 70, 70),
        (180, 140, 20),
    )
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    over = ImageDraw.Draw(overlay)
    notes = []
    for group_index, (path, nodes, episode_uid) in enumerate(groups):
        color = colors[group_index % len(colors)]
        points = [(point["position"][0], point["position"][2]) for point in path]
        draw.line([pix(x, z) for x, z in points], fill=color, width=3)
        episode_id = episode_uid.split("|")[-1]
        for index, node in enumerate(nodes, start=1):
            _wedge(over, pix, node, span)
            notes.append((episode_id, index, node))
    image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(image)
    for episode_id, index, node in notes:
        _marker(draw, pix, node, f"{episode_id}-{index}", font, span)
    _legend(draw, font, small)
    text_y = 330
    draw.text((MAP + 16, text_y), "each walk has its own color", fill=(40, 40, 40), font=small)
    text_y += 24
    for group_index, (_path, nodes, episode_uid) in enumerate(groups):
        color = colors[group_index % len(colors)]
        episode_id = episode_uid.split("|")[-1]
        draw.text((MAP + 16, text_y), f"ep {episode_id}   {len(nodes)} nodes", fill=color, font=small)
        text_y += 22
    image.save(dest)


def _bounds(points: list[tuple[float, float]]) -> tuple[float, float, float]:
    xs = [point[0] for point in points]
    zs = [point[1] for point in points]
    pad = 2.5
    minx, maxx = min(xs) - pad, max(xs) + pad
    minz, maxz = min(zs) - pad, max(zs) + pad
    span = max(maxx - minx, maxz - minz, 12.0)
    cx, cz = (minx + maxx) / 2, (minz + maxz) / 2
    return cx - span / 2, cz - span / 2, span


def _projector(minx: float, minz: float, span: float):
    usable = MAP - 2 * MARGIN

    def pix(x: float, z: float) -> tuple[float, float]:
        u = MARGIN + (x - minx) / span * usable
        v = MARGIN + (minz + span - z) / span * usable
        return u, v

    return pix


def _paint_floor(image, mesh, pix, path, minx, minz, span) -> None:
    vertices, indices = mesh
    ys = [point["position"][1] for point in path]
    y_lo, y_hi = min(ys) - FLOOR_Y, max(ys) + FLOOR_Y
    draw = ImageDraw.Draw(image)
    order = []
    for tri in indices:
        pts = vertices[tri]
        cy = float(pts[:, 1].mean())
        if cy < y_lo or cy > y_hi:
            continue
        if float(pts[:, 0].max()) < minx or float(pts[:, 0].min()) > minx + span:
            continue
        if float(pts[:, 2].max()) < minz or float(pts[:, 2].min()) > minz + span:
            continue
        order.append((cy, pts))
    order.sort(key=lambda item: item[0])
    for cy, pts in order:
        shade = 214 + int(abs(math.sin(cy * 1.7)) * 24)
        draw.polygon([pix(p[0], p[2]) for p in pts], fill=(shade, shade - 4, shade - 10))


def _wedge(draw, pix, node: dict, span: float) -> None:
    reach = max(1.6, min(3.2, span * 0.045))
    origin = node["position"]
    yaw = float(node["yaw"])
    taken = node.get("taken_exit_id")
    for opening in node["openings"]:
        if taken is None:
            color = (120, 120, 120, 70)
        elif int(opening["exit_id"]) == int(taken):
            color = (30, 150, 70, 90)
        else:
            color = (190, 40, 40, 80)
        center = float(opening["bearing_left_rad"])
        half = float(opening["angular_width_rad"]) / 2
        samples = [center - half, center - half / 2, center, center + half / 2, center + half]
        polygon = [pix(origin[0], origin[2])]
        for bearing in samples:
            world = _aim(origin, yaw, bearing, reach)
            polygon.append(pix(world[0], world[1]))
        draw.polygon(polygon, fill=color)


def _marker(draw, pix, node: dict, label: str, font, span: float) -> None:
    origin = node["position"]
    u, v = pix(origin[0], origin[2])
    heading = max(0.8, 28.0 * span / (MAP - 2 * MARGIN))
    tip = _aim(origin, float(node["yaw"]), 0.0, heading)
    draw.line([(u, v), pix(tip[0], tip[1])], fill=(20, 20, 20), width=3)
    radius = 8
    draw.ellipse([u - radius, v - radius, u + radius, v + radius], fill=(225, 120, 20), outline=(255, 255, 255))
    draw.text((u + 10, v - 18), label, fill=(30, 30, 30), font=font)


def _aim(origin, yaw: float, bearing: float, distance: float) -> tuple[float, float]:
    forward = np.array([-math.sin(yaw), -math.cos(yaw)])
    left = np.array([-math.cos(yaw), math.sin(yaw)])
    direction = math.cos(bearing) * forward + math.sin(bearing) * left
    return (
        float(origin[0] + direction[0] * distance),
        float(origin[2] + direction[1] * distance),
    )


def _dot(draw, point, radius, fill) -> None:
    u, v = point
    draw.ellipse([u - radius, v - radius, u + radius, v + radius], fill=fill, outline=(255, 255, 255))


def _legend(draw, font, small) -> None:
    x, y = MAP + 16, 16
    draw.text((x, y), "where the node was marked", fill=(20, 20, 20), font=font)
    rows = (
        ((214, 210, 202), "walkable floor"),
        ((36, 84, 168), "walked path"),
        ((225, 120, 20), "marked node"),
        ((20, 20, 20), "heading"),
        ((30, 150, 70), "exit that was walked"),
        ((190, 40, 40), "exit that was seen, not walked"),
        ((120, 120, 120), "exit, walked direction unmatched"),
    )
    y += 32
    for color, text in rows:
        draw.rectangle([x, y + 3, x + 16, y + 17], fill=color)
        draw.text((x + 24, y), text, fill=(40, 40, 40), font=small)
        y += 24
    draw.text((x, y + 8), "Gray floor is the house navmesh.", fill=(80, 80, 80), font=small)
    draw.text((x, y + 28), "It is only a picture of the place.", fill=(80, 80, 80), font=small)


def _episode_notes(draw, font, path, nodes) -> None:
    scene = nodes[0]["scene"]
    episode = nodes[0]["episode_uid"].split("|")[-1]
    split = "fit" if "fit" in nodes[0]["split"] else "dev"
    y = 330
    draw.text((MAP + 16, y), f"{scene}", fill=(20, 20, 20), font=font)
    y += 22
    draw.text((MAP + 16, y), f"ep {episode}  {split}", fill=(20, 20, 20), font=font)
    y += 28
    for index, node in enumerate(nodes, start=1):
        sides = "+".join(opening["side"] for opening in node["openings"])
        taken = node.get("taken_exit_id")
        if taken is None:
            walked = "unmatched"
        else:
            walked = next(
                opening["side"] for opening in node["openings"] if int(opening["exit_id"]) == int(taken)
            )
        draw.text(
            (MAP + 16, y),
            f"{index}  step {node['step_index']}  {sides}  walked {walked}",
            fill=(40, 40, 40),
            font=font,
        )
        y += 22
        if y > MAP - 24:
            break


if __name__ == "__main__":
    main()
