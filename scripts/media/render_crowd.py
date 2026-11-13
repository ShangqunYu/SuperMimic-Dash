"""100 robots in one shot: a 10 x 10 grid of cells, each a small diorama where one robot plays one clip on its own
terrain. A cell is a raised square tile whose top is the lowest level the robot's feet touch; its terrain is cut
to the cell, from DEPTH below that (gaps and drops stay visible) to a little above the highest foothold (walls the
robot never reaches are trimmed). Every clip loops with a short crossfade, at a random phase, while the camera flies
from a close view out to the whole grid. Clips from crowd_picks.py; rollouts in the GPU env (nominal: no noise,
no DR) as in render_policy.py, in the contacts the checkpoint was trained in. A clip that fails here is replaced
by a spare of the same kind.

  python scripts/media/render_crowd.py ckpt picks.json out.mp4 [--seconds 24] [--cell 5] [--width 1920 --height 1080]
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
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/parc"))
sys.path.insert(0, str(REPO / "scripts/media"))
from render_hq import DASH_XML, FPS, encode, resample, style  # noqa: E402
from render_policy import rollouts  # noqa: E402

from supermimic.terrain.terrains import load_terrains  # noqa: E402

G = 10  # grid side
FADE = 0.35  # s, loop crossfade
DEPTH = 0.6  # m, tile height above the ground: terrain shown below a cell's lowest foothold
TERRAIN_RGBA, GROUND_RGBA, ROBOT_RGBA = [0.92, 0.94, 0.98, 1], [0.62, 0.65, 0.72, 1], [0.92, 0.93, 0.95, 1]
# camera keyframes: (fraction of the video, look-at, distance, elevation, azimuth); MuJoCo free camera, azimuth 45
# looks from the (-x, -y) corner toward the grid center
KEYS = [(0.00, (-22.0, -21.5, 0.45), 8.0, -12.0, 40.0),
        (0.28, (-15.5, -14.0, 0.45), 12.0, -16.0, 34.0),
        (0.55, (-8.0, -7.0, 0.40), 24.0, -23.0, 42.0),
        (1.00, (0.0, 1.0, 0.0), 52.0, -31.0, 58.0)]


def camera(frac):
    for (f0, l0, d0, e0, a0), (f1, l1, d1, e1, a1) in zip(KEYS, KEYS[1:]):
        if frac <= f1:
            s = (frac - f0) / (f1 - f0)
            s = s * s * (3 - 2 * s)
            lerp = lambda x, y: x + (y - x) * s  # noqa: E731
            # distance in log space: a steady zoom
            return np.array([lerp(a, b) for a, b in zip(l0, l1)]), float(np.exp(lerp(np.log(d0), np.log(d1)))), lerp(e0, e1), lerp(a0, a1)
    return np.array(KEYS[-1][1]), KEYS[-1][2], KEYS[-1][3], KEYS[-1][4]


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


def build_model(boxes, w, h, extent_m):
    spec = mujoco.MjSpec.from_file(DASH_XML)
    style(spec, [box([0, 0, -DEPTH - 0.1], [0.01, 0.01, -DEPTH])] + boxes, w, h, robot_rgba=ROBOT_RGBA)
    spec.add_material(name="ground", textures=spec.material("terrain").textures, texuniform=True, texrepeat=[0.5, 0.5], rgba=GROUND_RGBA,
                      specular=0.05, shininess=0.1)
    spec.geom("ground").material = "ground"
    for g in spec.geoms:
        if g.type == mujoco.mjtGeom.mjGEOM_MESH:
            g.material = "dark" if any(k in g.meshname for k in ("foot", "dist_hip", "prox_shoulder")) else "light"
    m = spec.compile()
    m.geom_size[m.geom("ground").id] = [180, 180, 0.05]
    m.mat_rgba[m.material("terrain").id] = TERRAIN_RGBA
    e = m.stat.extent
    m.vis.map.znear, m.vis.map.zfar = 0.2 / e, 320 / e
    m.vis.map.fogstart, m.vis.map.fogend = 110 / e, 240 / e
    m.vis.map.shadowclip = extent_m / e
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("picks")
    ap.add_argument("out")
    ap.add_argument("--seconds", type=float, default=24.0)
    ap.add_argument("--clip-seconds", type=float, default=14.0, help="longest part of a clip played (then it loops)")
    ap.add_argument("--cell", type=float, default=5.0)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--title", default="100 Dash robots, one policy: terrain-aware motion tracking teacher (v6), real time")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    picks = json.loads(Path(a.picks).read_text())
    lib = picks["lib"]
    cand = picks["cells"] + picks["spares"]
    names = [n for n, _ in cand]
    qs, starts, lens, tiles, fail_at = rollouts(a.ckpt, lib, names, int(a.clip_seconds * 50))
    ok = {n: (q, t) for n, q, t, f in zip(names, qs, tiles, fail_at) if f is None}
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
    rng.shuffle(cells)
    ts = load_terrains(lib)
    mr = build_model([], 64, 64, extent_m=60.0)  # robot only: kinematics for every robot (same materials and meshes)
    dr = mujoco.MjData(mr)
    P, half = a.cell, a.cell / 2 - 0.15
    boxes, plays = [], []
    for k, (n, kind) in enumerate(cells[: G * G]):
        q, tile = ok[n]
        center = np.array([(k % G - (G - 1) / 2) * P, (k // G - (G - 1) / 2) * P])
        lo, hi = q[:, :2].min(0), q[:, :2].max(0)
        if tile >= 0:
            base, high = foot_levels(mr, dr, q.astype(float))
            fl = ts.terrains[int(tile)].floor
            base = fl if base < fl + 0.05 else base  # feet on the floor: the tile top is the floor
            floor = fl - base
            off = np.array([*(center - (lo + hi) / 2), -base])
            boxes += cell_boxes(ts.boxes[int(tile)], off, center, half, max(0.3, high - base + 0.6))
        else:
            floor, off = 0.0, np.array([*(center - (lo + hi) / 2), 0.0])
        if floor > -DEPTH + 0.005:  # the tile itself: the terrain's floor, or the ground of a flat clip
            boxes.append(box([center[0] - half, center[1] - half, -DEPTH], [center[0] + half, center[1] + half, min(floor, 0.0)]))
        Q = resample(q.astype(float), 50, 1.0)
        Q[:, :3] += off
        plays.append((Q, rng.randrange(len(Q)), kind, n))
    m = build_model(boxes, a.width, a.height, extent_m=60.0)
    d0 = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d0)
    r = mujoco.Renderer(m, a.height, a.width, max_geom=40000)
    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[[0, 1, 2]] = 1
    opt.sitegroup[:] = 0
    pert = mujoco.MjvPerturb()
    key = m.light("key").id
    kd = m.light_dir[key] / np.linalg.norm(m.light_dir[key])
    sc = a.width / 1920
    font, font2 = ImageFont.truetype("DejaVuSans-Bold.ttf", int(30 * sc)), ImageFont.truetype("DejaVuSans.ttf", int(22 * sc))
    kinds = sorted({p[2] for p in plays})
    sub = "terrain: " + ", ".join(k.replace("us_", "US ") for k in kinds if k in ("teaser", "boxes", "platform", "stairs", "running", "us_stairs", "us_jumps"))
    sub2 = "flat: " + ", ".join(k for k in kinds if k not in ("teaser", "boxes", "platform", "stairs", "running", "us_stairs", "us_jumps"))
    F = int(FADE * FPS)
    N = int(a.seconds * FPS)
    print(f"{len(plays)} robots, {len(boxes)} terrain boxes, {N} frames", flush=True)

    def frames():
        for f in range(N):
            look, dist, el, az = camera(f / max(N - 1, 1))
            cam.lookat[:], cam.distance, cam.elevation, cam.azimuth = look, dist, el, az
            m.light_pos[key] = look - 40 * kd
            mujoco.mjv_updateScene(m, d0, opt, pert, cam, mujoco.mjtCatBit.mjCAT_STATIC, r.scene)
            for Q, phase, _, _ in plays:
                period = len(Q) - F
                u = (f + phase) % period
                draw = [(u, min(1.0, u / F))]
                if u < F:
                    draw.append((u + period, 1.0 - u / F))  # the previous loop's end, fading out
                for k, alpha in draw:
                    dr.qpos[:] = Q[k]
                    mujoco.mj_kinematics(mr, dr)
                    n0 = r.scene.ngeom
                    mujoco.mjv_addGeoms(mr, dr, opt, pert, mujoco.mjtCatBit.mjCAT_DYNAMIC, r.scene)
                    if alpha < 1:
                        for g in range(n0, r.scene.ngeom):
                            r.scene.geoms[g].rgba[3] = alpha
            r.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = 1
            im = Image.fromarray(r.render()).convert("RGBA")
            band = Image.new("RGBA", im.size, (0, 0, 0, 0))
            ImageDraw.Draw(band).rectangle([0, 0, a.width, int(118 * sc)], fill=(245, 247, 250, 170))
            im = Image.alpha_composite(im, band).convert("RGB")
            dw = ImageDraw.Draw(im)
            dw.text((int(32 * sc), int(24 * sc)), a.title, fill=(30, 34, 40), font=font)
            for i, s in enumerate((sub, sub2)):
                dw.text((int(32 * sc), int((66 + 30 * i) * sc)), s, fill=(60, 66, 76), font=font2)
            if f % 120 == 0:
                print(f"frame {f}/{N}", flush=True)
            yield np.asarray(im)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    encode(a.out, frames(), a.width, a.height)
    r.close()
    Path(a.out).with_suffix(".json").write_text(json.dumps([{"cell": k, "clip": p[3], "kind": p[2]} for k, p in enumerate(plays)], indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
