"""Hundreds of robots in one shot: a grid of cells, each a small diorama where one robot plays one clip on its own
terrain, with its reference motion as a green ghost. A cell is a raised square tile whose top is the lowest level
the robot's feet touch; its terrain is cut to the cell, from DEPTH below that (gaps and drops stay visible) to a
little above the highest foothold (walls the robot never reaches are trimmed). Every clip loops with a short
crossfade at a random phase. Clips from crowd_picks.py; rollouts in the GPU env (nominal: no noise, no DR) as in
render_policy.py, in the contacts the checkpoint was trained in. A clip that fails here is replaced by a spare of
the same kind.

Camera: close on the hero (the longest terrain clip, in the middle cell), then out along a spiral over the grid
(stretched to its aspect) while rising slowly, so it passes over most cells, and a final pull-back to the whole
grid. Lighting as render_hq (key, fill, headlight; --levels for a tone pass on the frames).

  python scripts/media/render_crowd.py ckpt picks.json out.mp4 [--cols 25 --rows 20] [--seconds 70] [--cell 5]
      [--width 1920 --height 1080] [--cache rollouts.pkl] [--stills 2,20,60]
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
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
    x = np.clip(x, 0.0, 1.0)
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


def build_model(boxes, w, h, a):
    spec = mujoco.MjSpec.from_file(DASH_XML)
    style(spec, [box([0, 0, -DEPTH - 0.1], [0.01, 0.01, -DEPTH])] + boxes, w, h)
    spec.add_material(name="ground", textures=next(x for x in spec.materials if x.name == "terrain").textures, texuniform=True, texrepeat=[1.25, 1.25],
                      rgba=[a.ground_tint] * 3 + [1], specular=0.05, shininess=0.1)
    next(g for g in spec.geoms if g.name == "ground").material = "ground"
    for g in spec.geoms:
        if g.type == mujoco.mjtGeom.mjGEOM_MESH:
            g.material = "dark" if any(k in g.meshname for k in ("foot", "dist_hip", "prox_shoulder")) else "light"
    m = spec.compile()
    m.geom_size[m.geom("ground").id] = [400, 400, 0.05]
    m.vis.map.zfar = 900 / m.stat.extent
    m.vis.headlight.ambient[:], m.vis.headlight.diffuse[:] = a.head_ambient, a.head_diffuse
    key, fill = m.light("key").id, m.light("fill").id
    m.light_diffuse[key], m.light_specular[key], m.light_diffuse[fill] = a.key, 0.25, a.fill
    return m


def camera_path(N, hero_look, az0, a, P):
    """Per frame (look-at, distance, elevation, azimuth, zoom-out progress 0..1)."""
    t = np.arange(N) / FPS
    T = N / FPS
    t1, t2 = a.hold, T - a.zoom_s  # hero close-up until t1, spiral until t2, then the pull-back
    half_x, half_y = a.cols * P / 2, a.rows * P / 2
    r_end = 0.82 * min(half_x, half_y)
    sx = half_x / half_y if half_x > half_y else 1.0  # stretch the spiral to the grid's aspect
    sy = half_y / half_x if half_y > half_x else 1.0
    # spiral: r = b * theta / 2pi, at a speed ramping up to v; b sets the ring spacing so that r_end is reached at t2
    c0 = hero_look[0, :2]
    th, r, path = 0.0, 0.0, []
    s_total = None
    for b in np.linspace(8, 40, 200):  # ring spacing that reaches r_end on time at a steady pace
        s_total = np.pi * r_end**2 / b
        v = s_total / max(t2 - t1 - 2.0, 1.0)
        if v <= a.speed:
            break
    v = s_total / max(t2 - t1 - 2.0, 1.0)
    for k in range(N):
        if t[k] > t1:
            ramp = float(smoothstep((t[k] - t1) / 4.0))
            th += ramp * v / FPS / max(r, 6.0)
            r = min(r_end, b * th / (2 * np.pi))
        path.append((c0[0] + sx * r * np.cos(th), c0[1] + sy * r * np.sin(th), th, r))
    path = np.array(path)
    # look-at: hero, blending into the spiral; azimuth: from the hero framing into looking along the path, turned
    # 35 deg toward the center (the camera trails outside the ring, looking at the cells ahead and inside it)
    w_in = smoothstep((t - t1) / 4.0)
    look = np.zeros((N, 3))
    look[:, :2] = (1 - w_in)[:, None] * hero_look[:, :2] + w_in[:, None] * path[:, :2]
    look[:, 2] = (1 - w_in) * hero_look[:, 2] + w_in * 0.3
    tang = np.degrees(np.arctan2(sy * np.cos(path[:, 2]), -sx * np.sin(path[:, 2])))
    az_spiral = np.degrees(np.unwrap(np.radians(tang + 35.0)))
    az_spiral += 360.0 * np.round((az0 - az_spiral[0]) / 360.0)  # nearest turn to the hero framing
    az = (1 - w_in) * az0 + w_in * az_spiral
    rr = path[:, 3] / r_end
    dist = 3.4 + w_in * (a.dist_spiral[0] + (a.dist_spiral[1] - a.dist_spiral[0]) * rr - 3.4)
    el = -12.0 + w_in * (-8.0 - 10.0 * rr)
    # pull-back to the whole grid
    s = smoothstep((t - t2) / a.zoom_s)
    d1 = 0.88 * max(a.cols, a.rows * 16 / 9) * P  # the whole grid in frame
    look = (1 - s)[:, None] * look + s[:, None] * np.array([0.0, 0.0, 0.0])
    dist = np.exp((1 - s) * np.log(dist) + s * np.log(d1))
    el = (1 - s) * el + s * -45.0
    az = az + s * 25.0
    sm = lambda x, sig: gaussian_filter1d(x, sig * FPS, axis=0, mode="nearest")  # noqa: E731
    return sm(look, 0.4), sm(dist, 0.4), sm(el, 0.4), sm(az, 0.6), s


def tone(img, lo, hi, gamma):
    x = np.clip((img.astype(np.float32) / 255.0 - lo) / (hi - lo), 0.0, 1.0) ** gamma
    return (x * 255.0 + 0.5).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("picks")
    ap.add_argument("out")
    ap.add_argument("--cols", type=int, default=10)
    ap.add_argument("--rows", type=int, default=10)
    ap.add_argument("--seconds", type=float, default=70.0)
    ap.add_argument("--hold", type=float, default=3.0, help="s on the hero before the camera moves")
    ap.add_argument("--zoom-s", type=float, default=12.0, help="s of the final pull-back")
    ap.add_argument("--speed", type=float, default=4.5, help="max look-at speed along the spiral (m/s)")
    ap.add_argument("--dist-spiral", type=float, nargs=2, default=(8.0, 20.0), help="camera distance at the spiral's start / end (m)")
    ap.add_argument("--clip-seconds", type=float, default=14.0, help="longest part of a clip played (then it loops)")
    ap.add_argument("--cell", type=float, default=5.0)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--key", type=float, default=0.55, help="key light diffuse (render_hq's)")
    ap.add_argument("--fill", type=float, default=0.18)
    ap.add_argument("--head-ambient", type=float, default=0.35)
    ap.add_argument("--head-diffuse", type=float, default=0.35)
    ap.add_argument("--ground-tint", type=float, default=1.0)
    ap.add_argument("--levels", type=float, nargs=3, default=(0.0, 1.0, 1.0), help="tone: black, white, gamma")
    ap.add_argument("--cache", help="rollouts pickle: written on the first run, read after")
    ap.add_argument("--stills", help="comma-separated times (s): write these frames as PNG next to out, no video")
    ap.add_argument("--title", default="{n} Dash robots, one policy: terrain-aware motion tracking teacher (v6), real time")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    n_cells = a.cols * a.rows
    picks = json.loads(Path(a.picks).read_text())
    lib = picks["lib"]
    cand = picks["cells"] + picks["spares"]
    names = [n for n, _ in cand]
    if a.cache and Path(a.cache).exists():
        ok = pickle.loads(Path(a.cache).read_bytes())
    else:
        qs, starts, lens, tiles, fail_at = rollouts(a.ckpt, lib, names, int(a.clip_seconds * 50))
        ok = {n: (q, s, t) for n, q, s, t, f in zip(names, qs, starts, tiles, fail_at) if f is None}
        if a.cache:
            Path(a.cache).write_bytes(pickle.dumps(ok))
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
    mr = build_model([], 64, 64, a)  # robot only: kinematics for every robot (same materials and meshes as the scene)
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
    m = build_model(boxes, a.width, a.height, a)
    d0 = mujoco.MjData(m)
    mujoco.mj_forward(m, d0)  # world poses of the static geoms and the lights (mj_kinematics leaves lights unset)
    F, N = int(FADE * FPS), int(a.seconds * FPS)

    def drawn(play, f):
        """(frame, alpha) of a cell's robot at video frame f: the loop, and its previous end while fading out."""
        Q, _, phase, _, _ = play
        period = len(Q) - F
        u = (f + phase) % period
        return [(u, min(1.0, u / F))] + ([(u + period, 1.0 - u / F)] if u < F else [])

    HQ = plays[mid][0]
    hero_xyz = np.array([HQ[drawn(plays[mid], f)[0][0], :3] for f in range(N)])
    hero_look = np.stack([gaussian_filter1d(hero_xyz[:, i], (0.6 if i < 2 else 1.0) * FPS, mode="nearest") for i in range(3)], 1) - [0, 0, 0.1]
    u0 = HQ[min(len(HQ) - 1, 3 * FPS), :2] - HQ[0, :2]
    az0 = float(np.degrees(np.arctan2(u0[1], u0[0])) + 110) if np.linalg.norm(u0) > 0.3 else 135.0
    LOOK, DIST, EL, AZ, ZOOM = camera_path(N, hero_look, az0, a, P)

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

    def frame(f):
        look, dist = LOOK[f], float(DIST[f])
        cam.lookat[:], cam.distance, cam.elevation, cam.azimuth = look, dist, float(EL[f]), float(AZ[f])
        # shadows, haze and the near plane follow the camera distance; ground beyond the shadow box speckles (it
        # samples the clamped shadow-map edge), so the haze is complete by then
        clip = max(60.0, 3.0 * dist)
        con.shadowClip = clip
        con.fogStart, con.fogEnd = 1.8 * dist + 8.0, min(7.0 * dist + 30.0, clip)
        m.vis.map.znear = max(0.05, 0.02 * dist) / e
        m.light_pos[key] = look - clip * kd
        mujoco.mj_camlight(m, d0)  # the key light's world position: its shadow box follows the camera
        mujoco.mjv_updateScene(m, d0, opt, pert, cam, mujoco.mjtCatBit.mjCAT_STATIC, r.scene)
        for p in plays:
            for k, alpha in drawn(p, f):
                add(p[1][k], alpha, ghost=GHOST[3] * (1 - 0.6 * ZOOM[f]))  # fainter far out: the crowd stays robot-colored
                add(p[0][k], alpha)
        r.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = 1
        im = Image.fromarray(tone(r.render(), *a.levels)).convert("RGBA")
        band = Image.new("RGBA", im.size, (0, 0, 0, 0))
        ImageDraw.Draw(band).rectangle([0, 0, a.width, int(118 * sc)], fill=(248, 249, 251, 175))
        im = Image.alpha_composite(im, band).convert("RGB")
        dw = ImageDraw.Draw(im)
        dw.text((int(32 * sc), int(24 * sc)), lines[0], fill=(30, 34, 40), font=font)
        for i, s in enumerate(lines[1:]):
            dw.text((int(32 * sc), int((66 + 30 * i) * sc)), s, fill=(60, 66, 76), font=font2)
        return im

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if a.stills:
        for s in a.stills.split(","):
            f = min(N - 1, int(float(s) * FPS))
            frame(f).save(out.with_name(f"{out.stem}_t{float(s):g}.png"))
        r.close()
        return

    def frames():
        for f in range(N):
            if f % 600 == 0:
                print(f"frame {f}/{N}", flush=True)
            yield np.asarray(frame(f))

    encode(str(out), frames(), a.width, a.height)
    r.close()
    out.with_suffix(".json").write_text(json.dumps([{"cell": k, "clip": p[4], "kind": p[3]} for k, p in enumerate(plays)], indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
