"""Replay every terrain clip of a terrain library on its tile (CPU MuJoCo, kinematic) and measure how
far the robot's collision geoms go into the terrain; optionally render some clips.

  uv run python scripts/terrain/check_tiles.py --lib data/library/dash_terrain_compact.pt --render 6
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
import torch
import tyro
from PIL import Image, ImageDraw, ImageFont

from supermimic.robots import get_robot


def tile_model(tile: np.ndarray, width=640, height=480) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(get_robot("dash").xml_path()))
    spec.visual.global_.offwidth, spec.visual.global_.offheight = width, height
    wb = spec.worldbody
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[50, 50, 0.1], rgba=[0.75, 0.77, 0.8, 1])
    for i, b in enumerate(np.asarray(tile).reshape(-1, 6)):
        wb.add_geom(name=f"tile_{i}", type=mujoco.mjtGeom.mjGEOM_BOX, pos=b[:3], size=b[3:], rgba=[0.55, 0.62, 0.72, 1])
    wb.add_light(pos=[0, 0, 10], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, diffuse=[0.7] * 3, castshadow=True)
    return spec.compile()


def penetration(m, d, qpos_seq, terrain_geoms):
    """Per frame: deepest robot-terrain contact (m, >0 = inside)."""
    out = np.zeros(len(qpos_seq))
    for t, q in enumerate(qpos_seq):
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        worst = 0.0
        for c in d.contact[: d.ncon]:
            if (c.geom1 in terrain_geoms) != (c.geom2 in terrain_geoms):
                worst = max(worst, -c.dist)
        out[t] = worst
    return out


def main(lib: Path = Path("data/library/dash_terrain_compact.pt"), render: int = 0, out: Path = Path("logs/terrain_tiles")) -> None:
    pack = torch.load(lib, map_location="cpu", weights_only=False)
    ref = pack["ref"].float().numpy()
    nj = get_robot("dash").num_joints
    models = {}
    rows = []
    rng = np.random.default_rng(0)
    tiles_rendered = set()
    for i, (name, st, ln, ti) in enumerate(zip(pack["names"], pack["starts"], pack["lengths"], pack["clip_tile"])):
        if ti < 0:
            continue
        if ti not in models:
            m = tile_model(pack["tiles"][ti])
            tg = {m.geom(f"tile_{k}").id for k in range(len(pack["tiles"][ti]))} | {m.geom("floor").id}
            models[ti] = (m, mujoco.MjData(m), tg)
        m, d, tg = models[ti]
        qpos = ref[st : st + ln, : 7 + nj]
        pen = penetration(m, d, qpos, tg)
        rows.append((name, pack["tile_names"][ti], pen.max(), np.mean(pen > 0.02)))
        if render and len(tiles_rendered) < render and ti not in tiles_rendered and rng.random() < 0.5:
            tiles_rendered.add(ti)
            render_clip(m, qpos, out / f"{name.removeprefix('dash_')}.mp4", f"{name.removeprefix('dash_')}  tile {pack['tile_names'][ti]}")
    pen = np.array([r[2] for r in rows])
    by = {}
    for r in rows:
        by.setdefault(r[1].split("_")[0], []).append(r[2])
    print(f"{len(rows)} terrain clips; max robot-terrain penetration per clip: p50 {np.median(pen) * 100:.1f} cm, p95 {np.percentile(pen, 95) * 100:.1f} cm, max {pen.max() * 100:.1f} cm")
    for k, v in by.items():
        v = np.array(v)
        print(f"  {k:7s} n={len(v):5d} p50 {np.median(v) * 100:.1f} cm  p95 {np.percentile(v, 95) * 100:.1f} cm  max {v.max() * 100:.1f} cm")
    worst = sorted(rows, key=lambda r: -r[2])[:5]
    for w in worst:
        print("  worst:", w[0], w[1], f"{w[2] * 100:.1f} cm")


def render_clip(m, qpos, path: Path, label: str, width=640, height=480):
    path.parent.mkdir(parents=True, exist_ok=True)
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, height, width)
    cam = mujoco.MjvCamera()
    cam.distance, cam.elevation, cam.azimuth = 2.6, -15.0, 90.0
    font = ImageFont.truetype("DejaVuSans.ttf", 15)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", "50", "-i", "-",
                           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22", str(path)], stdin=subprocess.PIPE)
    look = qpos[0, :3].copy()
    for q in qpos:
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        look = 0.9 * look + 0.1 * q[:3]
        cam.lookat[:] = [look[0], look[1], look[2] - 0.25]
        r.update_scene(d, cam)
        img = Image.fromarray(r.render())
        ImageDraw.Draw(img).text((8, 6), label, fill=(10, 10, 10), font=font)
        ff.stdin.write(img.tobytes())
    ff.stdin.close()
    ff.wait()


if __name__ == "__main__":
    tyro.cli(main)
