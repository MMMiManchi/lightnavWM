#!/usr/bin/env python3
"""Time-ordered replay with the camera frame and a top-down map.

The map uses the rebaked Go2 navmesh. Green is the shortest path from the
robot to the goal at that step. Blue is the trajectory already walked. Orange
is the ten predicted waypoints. Red is the goal. The camera image also gets the
standard overlay: an azure trajectory ribbon, a mint affordance point (apos)
and a magenta object point (opos).

Run with the Habitat 3.9 interpreter. It only reads the saved trace and the
navmesh; it does not roll out a new episode.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import habitat_sim

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from lightnav.viz.render import draw_pointing_soft, draw_traj_ribbon, pointing_points
OUT = ROOT / "output/r2r_probe/error_screening_full_episode_20261021"
TRACE = ROOT / "output/r2r_probe/trace_20261020/steps.jsonl"
TIMELINE = OUT / "episode_timeline.csv"
PATHS = OUT / "expert_path_reconstruction.csv"
DATASET = ROOT / "data/datasets/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen.json.gz"
NAVMESH = OUT / "navmesh_go2w"
REPLAY = OUT / "episode_replay"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

HFOV_DEG = 120.0
CAM_HEIGHT_M = 0.50
RGB_W, RGB_H = 960, 540
MAP_SIZE = 620
FOOTER = 52
CANVAS_W = RGB_W + MAP_SIZE

FLOOR_COLORS = (
    (214, 214, 208),
    (190, 208, 220),
    (220, 206, 186),
    (196, 214, 196),
)
EVENT_TEXT = {
    "geo_rise": "geo rising",
    "rise_start": "rise starts",
    "forward_while_path_behind": "path behind, still forward",
    "forward_while_path_side": "path off to the side",
    "expert_stop_ignored": "expert says stop",
    "model_stop": "model STOP",
    "commanded_forward_but_barely_moved": "forward command, little motion",
    "entered_3m": "inside 3 m",
    "left_3m": "left 3 m",
}


def load_goals() -> dict[tuple[str, str], np.ndarray]:
    with gzip.open(DATASET, "rt") as handle:
        data = json.load(handle)
    goals = {}
    for episode in data["episodes"]:
        scene = episode["scene_id"].split("/")[-2]
        goals[(scene, str(episode["episode_id"]))] = np.asarray(
            episode["goals"][0]["position"], dtype=np.float64
        )
    return goals


def load_timeline() -> dict[tuple[str, str], list[dict]]:
    groups = defaultdict(list)
    with TIMELINE.open() as handle:
        for row in csv.DictReader(handle):
            groups[(row["scene"], row["episode_id"])].append(row)
    for rows in groups.values():
        rows.sort(key=lambda row: int(row["step"]))
    return groups


def load_trace() -> dict[tuple[str, str], dict[int, dict]]:
    groups = defaultdict(dict)
    with TRACE.open() as handle:
        for line in handle:
            row = json.loads(line)
            groups[(row["scene"], str(row["episode_id"]))][int(row["step"])] = row
    return groups


def load_paths() -> dict[tuple[str, str], dict[int, list[tuple[float, float, float]]]]:
    groups = defaultdict(dict)
    with PATHS.open() as handle:
        for row in csv.DictReader(handle):
            points = []
            if row["path_world_xyz"]:
                for item in row["path_world_xyz"].split(";"):
                    x, y, z = item.split(",")
                    points.append((float(x), float(y), float(z)))
            groups[(row["scene"], row["episode_id"])][int(row["step"])] = points
    return groups


def load_mesh(scene: str):
    finder = habitat_sim.PathFinder()
    path = NAVMESH / f"{scene}.navmesh"
    if not finder.load_nav_mesh(str(path)):
        raise RuntimeError(f"cannot load {path}")
    vertices = np.asarray(finder.build_navmesh_vertices(), dtype=np.float64)
    indices = np.asarray(finder.build_navmesh_vertex_indices(), dtype=np.int64).reshape(-1, 3)
    return vertices, indices


def world_xz(origin: np.ndarray, yaw: float, forward_m: float, lateral_m: float) -> np.ndarray:
    forward = np.array([-math.sin(yaw), -math.cos(yaw)])
    left = np.array([-math.cos(yaw), math.sin(yaw)])
    return np.array([origin[0], origin[2]]) + forward_m * forward + lateral_m * left


def view_bounds(points: list[tuple[float, float]]) -> tuple[float, float, float]:
    xs = [p[0] for p in points]
    zs = [p[1] for p in points]
    pad = 1.6
    minx, maxx = min(xs) - pad, max(xs) + pad
    minz, maxz = min(zs) - pad, max(zs) + pad
    span = max(maxx - minx, maxz - minz, 8.0)
    cx, cz = (minx + maxx) / 2, (minz + maxz) / 2
    return cx - span / 2, cz - span / 2, span


def make_projector(minx: float, minz: float, span: float, size: int, margin: int):
    usable = size - 2 * margin

    def pix(x: float, z: float) -> tuple[float, float]:
        # Overhead view: +Z is up. World +X is the robot's left when it faces -Z,
        # so plotting +X to the right mirrors camera left/right. Flip X.
        u = margin + (minx + span - x) / span * usable
        v = margin + (minz + span - z) / span * usable
        return u, v

    return pix


def draw_floor(draw: ImageDraw.ImageDraw, vertices, indices, pix, y_lo: float, y_hi: float) -> None:
    order = []
    for tri in indices:
        pts = vertices[tri]
        cy = float(pts[:, 1].mean())
        if cy < y_lo or cy > y_hi:
            continue
        order.append((cy, pts))
    order.sort(key=lambda item: item[0])
    for cy, pts in order:
        band = int(math.floor(cy * 2.0)) % len(FLOOR_COLORS)
        draw.polygon([pix(p[0], p[2]) for p in pts], fill=FLOOR_COLORS[band])


def draw_polyline(draw, pix, points, fill, width):
    if len(points) < 2:
        return
    draw.line([pix(p[0], p[1]) for p in points], fill=fill, width=width)


def overlay_camera(rgb: np.ndarray, step: dict) -> np.ndarray:
    """Azure ribbon plus mint apos and magenta opos, in the saved frame's pixels."""
    waypoints = step.get("waypoints_robot")
    if waypoints:
        rgb = draw_traj_ribbon(
            rgb, waypoints, hfov_deg=HFOV_DEG, cam_height=CAM_HEIGHT_M
        )
    frame = step.get("pointing_frame") or [rgb.shape[1], rgb.shape[0]]
    pointing = {
        "frame_size": frame,
        "apos_state": step.get("apos_state"),
        "apos_px": step.get("apos_px"),
        "opos_state": step.get("opos_state"),
        "opos_px": step.get("opos_px"),
    }
    points = pointing_points(pointing, rgb.shape[1], rgb.shape[0])
    if points:
        rgb = draw_pointing_soft(rgb, points)
    return rgb


def robot_marker(draw, pix, x: float, z: float, yaw: float, span: float, size: int) -> None:
    """Dot at the robot and a stick along its forward axis, same frame as the waypoints."""
    meters = max(0.7, 18.0 * span / (size - 56))
    forward = np.array([-math.sin(yaw), -math.cos(yaw)])
    tip = np.array([x, z]) + forward * meters
    draw.line([pix(x, z), pix(*tip)], fill=(10, 10, 10), width=4)
    u, v = pix(x, z)
    draw.ellipse([u - 5, v - 5, u + 5, v + 5], fill=(10, 10, 10), outline=(255, 255, 255))


def goal_marker(draw, pix, goal, font) -> None:
    u, v = pix(goal[0], goal[2])
    r = 7
    draw.ellipse([u - r, v - r, u + r, v + r], fill=(190, 30, 30), outline=(255, 255, 255))
    draw.line([(u - 12, v), (u + 12, v)], fill=(190, 30, 30), width=2)
    draw.line([(u, v - 12), (u, v + 12)], fill=(190, 30, 30), width=2)
    draw.text((u + 10, v - 16), "GOAL", fill=(160, 20, 20), font=font)


def wrap_text(text: str, font: ImageFont.ImageFont, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = word if not current else f"{current} {word}"
        if font.getlength(trial) <= width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


def event_line(raw: str) -> str:
    if not raw:
        return ""
    labels = []
    for name in raw.split("|"):
        if name in EVENT_TEXT:
            labels.append(EVENT_TEXT[name])
    return " | ".join(labels)


def render_episode(scene, eid, timeline, trace, paths, goal, mesh, dest, preview_steps):
    vertices, indices = mesh
    rows = timeline[(scene, eid)]
    traced = trace[(scene, eid)]
    episode_paths = paths.get((scene, eid), {})
    poses = []
    for row in rows:
        step = traced[int(row["step"])]
        origin = np.asarray(step["pose_xyz_before"], dtype=np.float64)
        poses.append((float(origin[0]), float(origin[2]), float(step["pose_yaw_before"]), float(origin[1])))
    cloud = [(p[0], p[1]) for p in poses]
    cloud.append((float(goal[0]), float(goal[2])))
    ys = [p[3] for p in poses] + [float(goal[1])]
    for points in episode_paths.values():
        for x, y, z in points:
            cloud.append((x, z))
            ys.append(y)
    minx, minz, span = view_bounds(cloud)
    pix = make_projector(minx, minz, span, MAP_SIZE, 28)
    y_lo, y_hi = min(ys) - 0.8, max(ys) + 0.8

    font = ImageFont.truetype(FONT, 16)
    small = ImageFont.truetype(FONT, 14)
    title = ImageFont.truetype(FONT, 22)
    instruction = wrap_text(rows[0]["instruction"].strip(), title, CANVAS_W - 28)
    line_h = 28
    header_h = 16 + line_h * len(instruction)
    background = Image.new("RGB", (MAP_SIZE, RGB_H), (246, 246, 244))
    draw = ImageDraw.Draw(background)
    draw_floor(draw, vertices, indices, pix, y_lo, y_hi)
    start_path = [(p[0], p[2]) for p in episode_paths.get(0, [])]
    draw_polyline(draw, pix, start_path, (24, 110, 64), 2)
    su, sv = pix(poses[0][0], poses[0][1])
    draw.ellipse([su - 5, sv - 5, su + 5, sv + 5], fill=(20, 140, 60), outline=(255, 255, 255))
    goal_marker(draw, pix, goal, small)
    draw.rectangle([0, 0, MAP_SIZE, 22], fill=(32, 32, 32))
    draw.text((8, 3), f"{scene}/{eid}", fill=(255, 255, 255), font=small)
    scale_m = 2.0 if span < 25 else 5.0
    scale_px = scale_m / span * (MAP_SIZE - 56)
    draw.line([(16, RGB_H - 18), (16 + scale_px, RGB_H - 18)], fill=(20, 20, 20), width=3)
    draw.text((16, RGB_H - 36), f"{scale_m:.0f} m", fill=(20, 20, 20), font=small)

    def frame_image(index: int) -> Image.Image:
        row = rows[index]
        step = traced[int(row["step"])]
        origin = np.asarray(step["pose_xyz_before"], dtype=np.float64)
        yaw = float(step["pose_yaw_before"])
        panel = background.copy()
        pen = ImageDraw.Draw(panel)
        walked = [(p[0], p[1]) for p in poses[: index + 1]]
        draw_polyline(pen, pix, walked, (30, 70, 170), 3)
        current = [(p[0], p[2]) for p in episode_paths.get(int(row["step"]), [])]
        draw_polyline(pen, pix, current, (40, 190, 80), 4)
        predicted = []
        for point in step.get("waypoints_robot") or []:
            xz = world_xz(origin, yaw, float(point[0]), float(point[1]))
            predicted.append((float(xz[0]), float(xz[1])))
        draw_polyline(pen, pix, predicted, (210, 90, 20), 3)
        if predicted:
            ou, ov = pix(*predicted[-1])
            pen.ellipse([ou - 3, ov - 3, ou + 3, ov + 3], fill=(210, 90, 20))
        robot_marker(pen, pix, float(origin[0]), float(origin[2]), yaw, span, MAP_SIZE)
        goal_marker(pen, pix, goal, small)

        native = np.asarray(Image.open(ROOT / row["frame_file"]).convert("RGB"))
        native = overlay_camera(native, step)
        rgb = Image.fromarray(native).resize((RGB_W, RGB_H), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (CANVAS_W, header_h + RGB_H + FOOTER), (24, 24, 24))
        canvas.paste(rgb, (0, header_h))
        canvas.paste(panel, (RGB_W, header_h))
        footer = ImageDraw.Draw(canvas)
        for line_index, line in enumerate(instruction):
            footer.text((14, 8 + line_index * line_h), line, fill=(255, 255, 255), font=title)
        body_bottom = header_h + RGB_H
        geo = float(row["geodesic_before_m"])
        euc = float(row["euclidean_before_m"])
        cmd_f = float(row["commanded_forward_m"])
        cmd_y = float(row["commanded_yaw_deg"])
        footer.text(
            (12, body_bottom + 6),
            f"step {row['step']}/{int(rows[-1]['step'])}   geo {geo:.2f} m   euc {euc:.2f} m"
            f"   cmd {cmd_f:.2f} m / {cmd_y:+.1f} deg"
            f"   apos {step.get('apos_state') or 'none'}   opos {step.get('opos_state') or 'none'}",
            fill=(240, 240, 240),
            font=font,
        )
        footer.text(
            (12, body_bottom + 28),
            "map: gray floor, dark green start path, green path now, blue walked, orange predicted, red GOAL"
            "    camera: azure path, mint apos, magenta opos",
            fill=(190, 190, 190),
            font=small,
        )
        note = event_line(row["events"])
        if note:
            footer.text((CANVAS_W - 620, body_bottom + 6), note[:78], fill=(255, 210, 120), font=small)
        return canvas

    if preview_steps is not None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        for index, row in enumerate(rows):
            if int(row["step"]) not in preview_steps:
                continue
            frame_image(index).save(dest.parent / f"{scene}_{eid}_step{int(row['step']):04d}.png")
        return len(rows)

    dest.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(dest, fps=8, codec="libx264", macro_block_size=1, quality=8) as writer:
        for index in range(len(rows)):
            writer.append_data(np.asarray(frame_image(index)))
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", default="", help="scene/id:step,scene/id:step writes PNG instead of MP4")
    args = parser.parse_args()
    preview = {}
    if args.preview:
        for item in args.preview.split(","):
            left, step = item.split(":")
            scene, eid = left.split("/")
            preview.setdefault((scene, eid), set()).add(int(step))

    timeline = load_timeline()
    trace = load_trace()
    paths = load_paths()
    goals = load_goals()
    meshes = {}
    index = []
    keys = list(preview) if preview else list(timeline)
    for scene, eid in keys:
        if scene not in meshes:
            meshes[scene] = load_mesh(scene)
        dest = REPLAY / f"{scene}_{eid}.mp4"
        frames = render_episode(
            scene,
            eid,
            timeline,
            trace,
            paths,
            goals[(scene, eid)],
            meshes[scene],
            dest if not preview else REPLAY / "preview" / f"{scene}_{eid}.png",
            preview.get((scene, eid)),
        )
        index.append(
            {
                "scene": scene,
                "episode_id": eid,
                "frames": frames,
                "fps": 8,
                "map": "go2_rebaked_navmesh",
                "video": str((REPLAY / f"{scene}_{eid}.mp4").relative_to(ROOT)),
            }
        )
        print(scene, eid, frames)
    if not preview:
        with (REPLAY / "index.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(index[0].keys()))
            writer.writeheader()
            writer.writerows(index)


if __name__ == "__main__":
    main()
