"""Reference (kinematic, no physics) stair clips of a packed library on their terrains, two panels: full body
3/4 side view | close-up profile of the feet with the near terrain cut away. Feet are tinted per frame:
red = the foot capsule overlaps the step by > 1 cm, orange = standing with > 30% of the foot over a lower
step (hanging over the lip). Also --stats: the same foot checks over every non-aug stairs clip.

  python scripts/media/render_reference.py out_dir LIB --clips n1 n2 ... [--slow 0.5] [--evals "label=eval_edge.json" ...]
  python scripts/media/render_reference.py - LIB --stats
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "scripts/parc"))
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from scipy.ndimage import gaussian_filter1d  # noqa: E402

from render_hq import FPS, dash_model, resample  # noqa: E402
from render_strobe import section_cut  # noqa: E402

from supermimic.generator.metrics import FootMetrics  # noqa: E402
from supermimic.terrain.terrains import load_terrains  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("lib")
ap.add_argument("--clips", nargs="*", default=[])
ap.add_argument("--slow", type=float, default=0.5)
ap.add_argument("--stats", action="store_true")
ap.add_argument("--evals", nargs="*", default=[], help="label=eval_edge json: per-clip tracker outcome in the caption")
a = ap.parse_args()

pk = torch.load(a.lib, map_location="cpu", weights_only=False, mmap=True)
ts = load_terrains(a.lib)
fm = FootMetrics(None, "cpu")
RAD = fm.radius
names = list(pk["names"])


def solid_dist(ter, P):
    """Signed distance (m) from points P (..., 3) to the heightfield solid (union of cell columns from below),
    negative inside; searched over cells within 0.12 m."""
    sh = P.shape[:-1]
    P = P.reshape(-1, 3)
    k = int(np.ceil(0.12 / ter.dx))
    ij = np.round((P[:, :2] - ter.p0) / ter.dx).astype(int)
    off = np.stack(np.meshgrid(np.arange(-k, k + 1), np.arange(-k, k + 1), indexing="ij"), -1).reshape(-1, 2)
    c = ij[:, None] + off[None]
    ci = np.clip(c[..., 0], 0, ter.hf.shape[0] - 1)
    cj = np.clip(c[..., 1], 0, ter.hf.shape[1] - 1)
    h = ter.hf[ci, cj]  # (N, K)
    ctr = ter.p0 + c * ter.dx
    dxy = np.maximum(np.abs(P[:, None, :2] - ctr) - ter.dx / 2, 0)
    dz = np.maximum(P[:, None, 2] - h, 0)
    out = np.sqrt((dxy**2).sum(-1) + dz**2).min(1)
    own = ter.h(P[:, :2])
    inside = P[:, 2] < own
    if inside.any():
        # inside: the nearer of the way up (own top) and the way sideways (nearest column top below the point)
        lat = np.where(h < P[:, None, 2], np.sqrt((np.maximum(np.abs(P[:, None, :2] - ctr) - ter.dx / 2, 0) ** 2).sum(-1)) + ter.dx / 2, np.inf).min(1)
        out = np.where(inside, -np.minimum(own - P[:, 2], lat), out)
    return out.reshape(sh)


def foot_checks(Q, tile):
    """Per frame and foot: overlap (cm) of the foot capsule with the terrain, stance flag, share of the foot axis
    over ground > 3 cm below the foot's support, lowest sole height above the ground."""
    q = torch.as_tensor(Q, dtype=torch.float32)[None]
    ax = fm.axis(q[..., :3], q[..., 3:7], q[..., 7:25], 12)[0].numpy()  # (T, 2, 12, 3)
    T = len(Q)
    if tile < 0:
        z = np.zeros(ax.shape[:-1])
        g = z
        ov = np.maximum(RAD - ax[..., 2], 0).max(-1)
    else:
        ter = ts.terrains[tile]
        g = ter.h(ax[..., :2])
        ov = np.maximum(RAD - solid_dist(ter, ax), 0).max(-1)
    bottom = ax[..., 2] - RAD  # (T, 2, 12)
    sup = (bottom - g).min(-1)  # lowest sole point above the ground under it
    support_h = np.take_along_axis(g, (bottom - g).argmin(-1)[..., None], -1)[..., 0]
    mid = ax.mean(-2)
    v = np.zeros((T, 2))
    v[1:] = np.linalg.norm(mid[1:] - mid[:-1], axis=-1) * 50
    stance = (sup < 0.02) & (v < 0.25)
    hang = (g < support_h[..., None] - 0.03).mean(-1)
    return dict(overlap=ov * 100, stance=stance, hang=hang, gap=sup * 100)


def kind(n):
    return "aug" if n.startswith("aug/") else "parc" if n.startswith("parc/") else "mocap"


def clip(i):
    s, n = int(pk["starts"][i]), int(pk["lengths"][i])
    return pk["ref"][s : s + n, :25].float().numpy().astype(np.float64), int(pk["clip_tile"][i])


if a.stats:
    rows = {}
    for i, n in enumerate(names):
        Q, tile = clip(i)
        if tile < 0 or kind(n) == "aug" and i % 10:
            continue
        f = foot_checks(Q, tile)
        st = f["stance"]
        r = rows.setdefault(kind(n), [])
        r.append([f["overlap"].max(), (f["overlap"].max(-1) > 1).mean(), (f["hang"][st] > 0.3).mean() if st.any() else 0,
                  f["hang"][st].mean() if st.any() else 0])
    for k, r in rows.items():
        r = np.array(r)
        print(f"{k:6s} clips {len(r):5d}  max overlap p50 {np.median(r[:, 0]):.1f} cm  p90 {np.percentile(r[:, 0], 90):.1f} cm  "
              f"frames overlap>1cm {100 * r[:, 1].mean():.1f}%  stance feet >30% over a lip {100 * r[:, 2].mean():.1f}%  "
              f"mean overhang in stance {100 * r[:, 3].mean():.1f}%")
    sys.exit()

evals = []
for e in a.evals:
    lab, path = e.split("=", 1)
    evals.append((lab, {c["name"]: c for c in json.load(open(path))["clips"]}))

out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
W, H = 960, 720
font, font2, font3 = ImageFont.truetype("DejaVuSans-Bold.ttf", 26), ImageFont.truetype("DejaVuSans.ttf", 21), ImageFont.truetype("DejaVuSans.ttf", 19)
for ci, name in enumerate(a.clips):
    i = names.index(name)
    Q50, tile = clip(i)
    f = foot_checks(Q50, tile)
    Q = resample(Q50, 50, a.slow)
    src = np.minimum((np.arange(len(Q)) * a.slow / FPS * 50).round().astype(int), len(Q50) - 1)
    m0 = dash_model([], 64, 64)
    b = ts.boxes[tile]
    cut = section_cut(b, Q50, m0, mujoco.MjData(m0), margin=0.06)
    boxes = np.array([[0, 0, ts.terrains[tile].floor - 0.05, 0.01, 0.01, 0.05]] + [list(x) for x in b])
    boxes_cut = np.array([[0, 0, ts.terrains[tile].floor - 0.05, 0.01, 0.01, 0.05]] + [list(x) for x in cut])
    ms = [dash_model(boxes, W, H), dash_model(boxes_cut, W, H)]
    for m in ms:
        for g in range(m.ngeom):
            if m.geom(g).name.startswith("terrain_"):
                m.geom_matid[g] = -1
                m.geom_rgba[g] = (0.62, 0.68, 0.80, 1.0)
    feet = [{g for g in range(ms[0].ngeom) if ms[0].geom_bodyid[g] == ms[0].body(f"{s}_foot").id} for s in "lr"]
    u = Q50[-1, :2] - Q50[0, :2]
    head = np.degrees(np.arctan2(u[1], u[0]))
    fps_play = FPS / a.slow
    p = Q[:, :3]
    sm = lambda x, s: gaussian_filter1d(x, s * fps_play, mode="nearest")  # noqa: E731
    look_body = np.stack([sm(p[:, 0], 0.5), sm(p[:, 1], 0.5), sm(p[:, 2], 1.0) - 0.15], 1)
    d = mujoco.MjData(ms[0])
    fpos = []
    for q in Q:
        d.qpos[:] = q
        mujoco.mj_kinematics(ms[0], d)
        fpos.append((d.xpos[ms[0].body("l_foot").id] + d.xpos[ms[0].body("r_foot").id]) / 2)
    fpos = np.array(fpos)
    look_feet = np.stack([sm(fpos[:, 0], 0.35), sm(fpos[:, 1], 0.35), sm(fpos[:, 2], 0.5) + 0.05], 1)
    views = [(ms[0], look_body, 2.6, -14.0, head + 105), (ms[1], look_feet, 1.15, -4.0, head + 90)]
    rs = [mujoco.Renderer(m, H, W) for m in ms]
    ds = [mujoco.MjData(m) for m in ms]
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[[0, 1, 2]] = 1
    opt.sitegroup[:] = 0
    cam = mujoco.MjvCamera()
    rise = tread = None
    if "_us_r" in name:
        r_, t_ = name.split("_us_r")[1].split("_t")
        rise, tread = float(r_) / (10 if len(r_) == 3 else 1), float(t_) / (10 if len(t_) == 3 else 1)
    ter_txt = f"stairs edited to US size: riser {rise:g} cm, tread {tread:g} cm" if rise else "original capture stairs, shrunk x0.677 to Dash size (riser ~12 cm, tread ~17 cm)"
    outcome = []
    for lab, ev in evals:
        c = ev.get(name)
        if c:
            outcome.append(f"{lab}: " + (f"fell at {c['fail_step'] / 50:.1f} s" if c["failed"] else "tracked to the end"))
    ovmax = f["overlap"].max()
    st = f["stance"]
    print(f"{name}: tile {tile} {len(Q50) / 50:.1f}s  max overlap {ovmax:.1f} cm  frames >1cm {100 * (f['overlap'].max(-1) > 1).mean():.0f}%  "
          f"stance >30% over lip {100 * (f['hang'][st] > 0.3).mean():.0f}%  {outcome}", flush=True)
    path = out / f"{ci + 1:02d}_{name.removeprefix('dash_stairs_climbing_')}.mp4"
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{2 * W + 4}x{H + 110}", "-r", str(FPS), "-i", "-",
                           "-c:v", "libx264", "-preset", "slow", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)
    for t in range(len(Q)):
        k = src[t]
        canvas = Image.new("RGB", (2 * W + 4, H + 110), (245, 246, 248))
        dr = ImageDraw.Draw(canvas)
        dr.text((16, 10), f"Reference motion (kinematic, no physics): {name}", fill=(30, 34, 40), font=font)
        dr.text((16, 46), f"{ter_txt}.  {a.slow:g}x speed, t = {k / 50:.2f} s / {len(Q50) / 50:.1f} s", fill=(60, 66, 76), font=font2)
        dr.text((16, 78), "   ".join(outcome) if outcome else "", fill=(60, 66, 76), font=font3)
        tint = []
        for s in range(2):
            if f["overlap"][k, s] > 1.0:
                tint.append(((0.85, 0.12, 0.10, 1), f"{'LR'[s]} foot {f['overlap'][k, s]:.1f} cm into the step"))
            elif f["stance"][k, s] and f["hang"][k, s] > 0.3:
                tint.append(((0.95, 0.55, 0.05, 1), f"{'LR'[s]} foot standing {100 * f['hang'][k, s]:.0f}% over the lip"))
            else:
                tint.append((None, f"{'LR'[s]} foot ok" + (" (standing)" if f["stance"][k, s] else "")))
        for vi, (m, look, dist, elev, az) in enumerate(views):
            dd = ds[vi]
            dd.qpos[:] = Q[t]
            mujoco.mj_forward(m, dd)
            key = m.light("key").id
            m.light_pos[key] = look[t] - 20 * m.light_dir[key] / np.linalg.norm(m.light_dir[key])
            cam.lookat[:], cam.distance, cam.elevation, cam.azimuth = look[t], dist, elev, az
            rs[vi].update_scene(dd, cam, opt)
            for g in rs[vi].scene.geoms[: rs[vi].scene.ngeom]:
                if g.objtype == mujoco.mjtObj.mjOBJ_GEOM:
                    for s in range(2):
                        if g.objid in feet[s] and tint[s][0] is not None:
                            g.rgba[:] = tint[s][0]
            rs[vi].scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = 1
            im = Image.fromarray(rs[vi].render())
            di = ImageDraw.Draw(im)
            di.text((14, 10), "full body" if vi == 0 else "feet, close-up profile (near side cut away)", fill=(40, 44, 52), font=font2)
            if vi == 1:
                for s in range(2):
                    col = (170, 30, 25) if tint[s][0] and tint[s][0][1] < 0.3 else (190, 100, 0) if tint[s][0] else (40, 110, 60)
                    di.text((14, 40 + 28 * s), tint[s][1], fill=col, font=font2)
            canvas.paste(im, (vi * (W + 4), 110))
        ff.stdin.write(np.asarray(canvas).tobytes())
    ff.stdin.close()
    ff.wait()
    for r in rs:
        r.close()
    print(f"wrote {path} ({path.stat().st_size / 2**20:.1f} MiB)", flush=True)
