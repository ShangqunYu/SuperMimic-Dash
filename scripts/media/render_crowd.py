"""Hundreds of robots in one shot: a grid of cells, each a small diorama where one robot plays one clip on its own
terrain, with its reference motion as a green ghost. A cell is a raised square tile whose top is the lowest level
the robot's feet touch; its terrain is cut to the cell, from DEPTH below that (gaps and drops stay visible) to a
little above the highest foothold (walls the robot never reaches are trimmed). Every clip loops with a short
crossfade at a random phase. The camera starts close on the hero (the longest terrain clip, in the middle cell)
and zooms out steadily to the whole grid. Clips from crowd_picks.py; rollouts in the GPU env (nominal: no noise,
no DR) as in render_policy.py, in the contacts the checkpoint was trained in. A clip that fails here is replaced
by a spare of the same kind.

  python scripts/media/render_crowd.py ckpt picks.json out.mp4 [--cols 25 --rows 20] [--seconds 30] [--cell 5]
      [--width 1920 --height 1080]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter1d

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/parc"))
sys.path.insert(0, str(REPO / "scripts/media"))
from render_hq import DASH_XML, FPS, encode, resample, style  # noqa: E402
from render_policy import rollouts  # noqa: E402

from supermimic.terrain.terrains import load_terrains  # noqa: E402

FADE = 0.35  # s, loop crossfade
DEPTH = 0.6  # m, tile height above the ground: terrain shown below a cell's lowest foothold
GHOST = (0.1, 0.8, 0.3, 0.16)  # reference motion, as render_policy.py
TERRAIN_KINDS = ("teaser", "boxes", "platform", "stairs", "running", "us_stairs", "us_jumps")


def smoothstep(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def box(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return np.concatenate([(lo + hi) / 2, (hi - lo) / 2])


def cell_boxes(boxes, off, center, half, top):
    """Terrain boxes moved by ``off``, cut to the square cell, from -DEPTH up to ``top``."""
    out = []
    for b in boxes:
        c, s = np.asarray(b[:3]) + off, np.asarray(b[3:6])
        lo = np.maximum(c - s, [center[0] - half, center[1] - half, -DEPTH])
        hi = np.minimum(c + s, [center[0] + half, center[1] + half, top])
        if np.all(hi[:2] - lo[:2] > 0.02) and hi[2] > -DEPTH + 0.005:
            out.append(box(lo, hi))
    return out


def foot_levels(mr, dr, q):
    """Lowest and highest foothold (m) over a clip: per frame, the lower foot's sole."""
    feet = [mr.geom(f"{s}_foot_collision").id for s in "lr"]
    z = []
    for t in range(0, len(q), 2):
        dr.qpos[:] = q[t]
        mujoco.mj_kinematics(mr, dr)
        z.append(min(dr.geom_xpos[g][2] for g in feet) - mr.geom_size[feet[0]][0])
    return float(np.min(z)), float(np.max(z))


def build_model(boxes, w, h):
    spec = mujoco.MjSpec.from_file(DASH_XML)
    style(spec, [box([0, 0, -DEPTH - 0.1], [0.01, 0.01, -DEPTH])] + boxes, w, h)
    for g in spec.geoms:
        if g.type == mujoco.mjtGeom.mjGEOM_MESH:
            g.material = "dark" if any(k in g.meshname for k in ("foot", "dist_hip", "prox_shoulder")) else "light"
    m = spec.compile()
    m.geom_size[m.geom("ground").id] = [400, 400, 0.05]
    m.vis.map.zfar = 900 / m.stat.extent
    # brighter than render_hq: its light look comes from fog 5-14 m away, which would hide the crowd
    m.vis.headlight.ambient[:], m.vis.headlight.diffuse[:] = 0.52, 0.42
    m.light_diffuse[m.light("key").id], m.light_diffuse[m.light("fill").id] = 0.62, 0.25
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("picks")
    ap.add_argument("out")
    ap.add_argument("--cols", type=int, default=10)
    ap.add_argument("--rows", type=int, default=10)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--clip-seconds", type=float, default=14.0, help="longest part of a clip played (then it loops)")
    ap.add_argument("--cell", type=float, default=5.0)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hold", type=float, default=0.08, help="share of the video on the hero before the zoom-out starts")
    ap.add_argument("--title", default="{n} Dash robots, one policy: terrain-aware motion tracking teacher (v6), real time")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    n_cells = a.cols * a.rows
    picks = json.loads(Path(a.picks).read_text())
    lib = picks["lib"]
    cand = picks["cells"] + picks["spares"]
    names = [n for n, _ in cand]
    qs, starts, lens, tiles, fail_at = rollouts(a.ckpt, lib, names, int(a.clip_seconds * 50))
    ok = {n: (q, s, t) for n, q, s, t, f in zip(names, qs, starts, tiles, fail_at) if f is None}
    spares = [c for c in picks["spares"] if c[0] in ok]
    cells = []
    for n, kind in picks["cells"]:
        if n not in ok:
            alt = next((c for c in spares if c[1] == kind), spares[0] if spares else None)
            print(f"failed here: {n}; replaced by {alt}")
            if alt is None:
                continue
            spares.remove(alt)
            n, kind = alt
        cells.append((n, kind))
    cells = cells[:n_cells]
    rng.shuffle(cells)
    # the hero (longest terrain clip) in the middle cell
    hero = max((c for c in cells if c[1] in TERRAIN_KINDS), key=lambda c: len(ok[c[0]][0]))
    mid = (a.rows // 2) * a.cols + a.cols // 2
    k = cells.index(hero)
    cells[k], cells[mid] = cells[mid], cells[k]
    ref = torch.load(lib, map_location="cpu", weights_only=False, mmap=True)["ref"]
    ts = load_terrains(lib)
    mr = build_model([], 64, 64)  # robot only: kinematics for every robot (same materials and meshes as the scene)
    dr = mujoco.MjData(mr)
    P, half = a.cell, a.cell / 2 - 0.15
    boxes, plays = [], []
    for k, (n, kind) in enumerate(cells):
        q, s0, tile = ok[n]
        q = q.astype(float)
        center = np.array([(k % a.cols - (a.cols - 1) / 2) * P, (k // a.cols - (a.rows - 1) / 2) * P])
        lo, hi = q[:, :2].min(0), q[:, :2].max(0)
        if tile >= 0:
            base, high = foot_levels(mr, dr, q)
            fl = ts.terrains[int(tile)].floor
            base = fl if base < fl + 0.05 else base  # feet on the floor: the tile top is the floor
            floor = fl - base
            off = np.array([*(center - (lo + hi) / 2), -base])
            boxes += cell_boxes(ts.boxes[int(tile)], off, center, half, max(0.3, high - base + 0.6))
        else:
            floor, off = 0.0, np.array([*(center - (lo + hi) / 2), 0.0])
        if floor > -DEPTH + 0.005:  # the tile itself: the terrain's floor, or the ground of a flat clip
            boxes.append(box([center[0] - half, center[1] - half, -DEPTH], [center[0] + half, center[1] + half, min(floor, 0.0)]))
        Q, R = resample(q, 50, 1.0), resample(ref[s0 : s0 + len(q), : q.shape[1]].float().numpy().astype(float), 50, 1.0)
        n_f = min(len(Q), len(R))
        Q, R = Q[:n_f], R[:n_f]
        Q[:, :3] += off
        R[:, :3] += off
        plays.append((Q, R, 0 if k == mid else rng.randrange(n_f), kind, n))
    m = build_model(boxes, a.width, a.height)
    d0 = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d0)
    F, N = int(FADE * FPS), int(a.seconds * FPS)

    def drawn(play, f):
        """(frame, alpha) of a cell's robot at video frame f: the loop, and its previous end while fading out."""
        Q, _, phase, _, _ = play
        period = len(Q) - F
        u = (f + phase) % period
        return [(u, min(1.0, u / F))] + ([(u + period, 1.0 - u / F)] if u < F else [])

    # camera: from the hero (render_policy.py framing) out to the whole grid
    HQ = plays[mid][0]
    hero_xyz = np.array([HQ[drawn(plays[mid], f)[0][0], :3] for f in range(N)])
    hero_look = np.stack([gaussian_filter1d(hero_xyz[:, i], (0.6 if i < 2 else 1.0) * FPS, mode="nearest") for i in range(3)], 1) - [0, 0, 0.1]
    u0 = HQ[min(len(HQ) - 1, 3 * FPS), :2] - HQ[0, :2]
    az0 = float(np.degrees(np.arctan2(u0[1], u0[0])) + 110) if np.linalg.norm(u0) > 0.3 else 135.0
    grid_c = np.array([0.0, 0.0, 0.0])
    d1 = 0.62 * max(a.cols, a.rows * 16 / 9) * P

    def camera(f):
        s = smoothstep((f / max(N - 1, 1) - a.hold) / (1 - a.hold))
        w = smoothstep(1.7 * s)
        look = (1 - w) * hero_look[f] + w * grid_c
        return look, float(np.exp(np.log(3.4) + (np.log(d1) - np.log(3.4)) * s)), -12.0 - 26.0 * s, az0 + 35.0 * s, s

    r = mujoco.Renderer(m, a.height, a.width, max_geom=4 * 24 * n_cells + len(boxes) + 2000)
    geoms = r.scene.geoms  # views into the scene's geom buffer
    con = r._mjr_context
    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[[0, 1, 2]] = 1
    opt.sitegroup[:] = 0
    pert = mujoco.MjvPerturb()
    key = m.light("key").id
    kd = m.light_dir[key] / np.linalg.norm(m.light_dir[key])
    e = m.stat.extent
    sc = a.width / 1920
    font, font2 = ImageFont.truetype("DejaVuSans-Bold.ttf", int(30 * sc)), ImageFont.truetype("DejaVuSans.ttf", int(22 * sc))
    kinds = {p[3] for p in plays}
    lines = [a.title.format(n=len(plays)),
             "terrain: " + ", ".join(k.replace("us_", "US ") for k in TERRAIN_KINDS if k in kinds),
             "flat: " + ", ".join(sorted(k for k in kinds if k not in TERRAIN_KINDS)) + ".   Green: reference motion"]
    print(f"{len(plays)} robots, hero {plays[mid][4]} ({len(HQ) / FPS:.1f} s), {len(boxes)} terrain boxes, {N} frames", flush=True)

    def add(q, alpha, ghost=0.0):
        dr.qpos[:] = q
        mujoco.mj_kinematics(mr, dr)
        n0 = r.scene.ngeom
        mujoco.mjv_addGeoms(mr, dr, opt, pert, mujoco.mjtCatBit.mjCAT_DYNAMIC, r.scene)
        if ghost:
            for g in range(n0, r.scene.ngeom):
                geoms[g].rgba[:] = (*GHOST[:3], ghost * alpha)
        elif alpha < 1:
            for g in range(n0, r.scene.ngeom):
                geoms[g].rgba[3] = alpha

    def frames():
        for f in range(N):
            look, dist, el, az, s = camera(f)
            cam.lookat[:], cam.distance, cam.elevation, cam.azimuth = look, dist, el, az
            # shadows, fog and the near plane follow the zoom: sharp close up, the whole grid at the end
            clip = float(np.clip(1.6 * dist, 10.0, 160.0))
            con.shadowClip = clip
            con.fogStart, con.fogEnd = 1.6 * dist + 6.0, 7.0 * dist + 20.0  # a light haze on the far rows
            m.vis.map.znear = max(0.05, 0.02 * dist) / e
            m.light_pos[key] = look - clip * kd
            mujoco.mjv_updateScene(m, d0, opt, pert, cam, mujoco.mjtCatBit.mjCAT_STATIC, r.scene)
            for p in plays:
                for k, alpha in drawn(p, f):
                    add(p[1][k], alpha, ghost=GHOST[3] * (1 - 0.6 * s))  # fainter far out: the crowd stays robot-colored
                    add(p[0][k], alpha)
            r.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = 1
            im = Image.fromarray(r.render()).convert("RGBA")
            band = Image.new("RGBA", im.size, (0, 0, 0, 0))
            ImageDraw.Draw(band).rectangle([0, 0, a.width, int(118 * sc)], fill=(245, 247, 250, 170))
            im = Image.alpha_composite(im, band).convert("RGB")
            dw = ImageDraw.Draw(im)
            dw.text((int(32 * sc), int(24 * sc)), lines[0], fill=(30, 34, 40), font=font)
            for i, s in enumerate(lines[1:]):
                dw.text((int(32 * sc), int((66 + 30 * i) * sc)), s, fill=(60, 66, 76), font=font2)
            if f % 300 == 0:
                print(f"frame {f}/{N}", flush=True)
            yield np.asarray(im)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    encode(a.out, frames(), a.width, a.height)
    r.close()
    Path(a.out).with_suffix(".json").write_text(json.dumps([{"cell": k, "clip": p[4], "kind": p[3]} for k, p in enumerate(plays)], indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
