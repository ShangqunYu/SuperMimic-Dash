"""Side-by-side video of two course_demo runs with the same seed and combo (A left, B right) over one stretch of
the course (--zone up: start to the top of the stairs; down: the stairs down; all: start to the finish), slowed
by --slow, each panel with its own low tracking camera on the legs and its own clock (it starts as its robot
reaches the stretch). Feet turn orange while inside the step-lip
cylinder (scripts/eval_edge.py's edge zone: 5 cm + the foot capsule's radius around each lip).

  python scripts/media/course_sbs.py out.mp4 A_dir B_dir --labels "edge" "control" --pick V S K [--pick-b V S K] [--zone down]
                                    [--note "same seed"]
"""
import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MUJOCO_GL", "egl")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/generator"))
sys.path.insert(0, str(REPO / "scripts/parc"))
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from scipy.ndimage import gaussian_filter1d  # noqa: E402

from course_demo import LANDING, START, TREAD, course  # noqa: E402
from render_hq import FPS, dash_model, encode, resample  # noqa: E402

from supermimic.generator.metrics import FootMetrics, edge_penetration  # noqa: E402
from supermimic.terrain.height_scan import HeightScan  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("runs", nargs=2)
ap.add_argument("--labels", nargs=2, required=True)
ap.add_argument("--pick", type=int, nargs=3, required=True)
ap.add_argument("--pick-b", type=int, nargs=3, help="B's own run (same variant and speed; default: --pick)")
ap.add_argument("--note", default="same seed", help="how the two runs were chosen (caption)")
ap.add_argument("--zone", choices=("up", "down", "all"), default="up")
ap.add_argument("--slow", type=float, default=0.5)
ap.add_argument("--after", type=float, default=1.0)
ap.add_argument("--size", type=int, nargs=2, default=[960, 720])
ap.add_argument("--dist", type=float, default=1.9)
ap.add_argument("--elev", type=float, default=-8.0)
ap.add_argument("--look-dz", type=float, default=-0.3, help="m: camera target relative to the smoothed root height")
a = ap.parse_args()

w, h = a.size
trs = [torch.load(Path(d) / "traj.pt", weights_only=False) for d in a.runs]
assert trs[0]["combo"] == trs[1]["combo"]
picks = [tuple(a.pick), tuple(a.pick_b or a.pick)]
assert picks[0][:2] == picks[1][:2]
idx = [tr["combo"].index(pk) for tr, pk in zip(trs, picks)]
v, s, _ = a.pick
rise, gap, plat = trs[0]["variants"][v]
speed = trs[0]["speeds"][s]
ter, marks, finish = course(rise, gap, plat)
top = START + 5 * TREAD
d0 = marks["platform"] - 0.3 + LANDING
xa, xb, what = {"up": (-1e9, top + 0.3, "stairs up"), "down": (d0 - 1.0, marks["stairs down"], "stairs down"), "all": (-1e9, finish, "course")}[a.zone]

scan = HeightScan(SimpleNamespace(terrains=[ter]))
fm = FootMetrics(scan, "cpu")
runs, t0s, t1s = [], [], []
for tr, i in zip(trs, idx):
    q = tr["robot"][i].numpy().astype(np.float64)
    res = tr["results"][i]
    T = len(q)
    t_down = int(res["down_at_s"] * 50) if res["down_at_s"] is not None else T
    j = np.nonzero(q[:t_down, 0] > xa)[0]
    t0 = j[0] if len(j) else 0
    j = np.nonzero(q[:t_down, 0] > xb)[0]
    ok = len(j) > 0
    t1 = j[0] if ok else min(t_down, T - 1)
    Q = torch.as_tensor(q, dtype=torch.float32)[None]
    pts = fm.axis(Q[..., :3], Q[..., 3:7], Q[..., 7:], 12)[0]  # (T, 2, 12, 3)
    pen = edge_penetration(scan, pts.reshape(T, -1, 3), torch.zeros(T), 0.05 + fm.radius).reshape(T, 2, 12).amax(-1).numpy()
    fell = res["down_at_s"] is not None and not ok
    outcome = f"{what}: done" if ok else (f"{what}: fell ({res['down_at_s']:.1f} s)" if fell else f"{what}: not finished in time")
    runs.append(dict(q=q, ok=ok, outcome=outcome, pen=pen))
    t0s.append(t0)
    t1s.append(t1)
L = max(t1 - t0 for t0, t1 in zip(t0s, t1s)) + 25 + int(a.after * 50)
spans = [(max(0, t0 - 25), min(len(run["q"]), max(0, t0 - 25) + L)) for t0, run in zip(t0s, runs)]

m = dash_model(np.array([[0, 0, ter.floor - 0.05, 0.01, 0.01, 0.05]] + [list(b) for b in ter.boxes()]), w, h)
for g in range(m.ngeom):
    if m.geom(g).name.startswith("terrain_"):
        m.geom_matid[g] = -1
        m.geom_rgba[g] = (0.60, 0.66, 0.78, 1.0)
feet = [{g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body(f"{sd}_foot").id} for sd in "lr"]
key = m.light("key").id
kd = m.light_dir[key] / np.linalg.norm(m.light_dir[key])
r = mujoco.Renderer(m, h, w)
opt = mujoco.MjvOption()
opt.geomgroup[:] = 0
opt.geomgroup[[0, 1, 2]] = 1
opt.sitegroup[:] = 0
fps_sim = FPS / a.slow  # playback frames per simulated second

panels = []
for run, label, t1, (f0, f1) in zip(runs, a.labels, t1s, spans):
    Q = resample(run["q"][f0:f1], 50, a.slow)
    src = np.minimum(f0 + np.round(np.arange(len(Q)) * a.slow / FPS * 50).astype(int), f1 - 1)
    p = Q[:, :3]
    look = np.stack([gaussian_filter1d(p[:, 0], 0.4 * fps_sim, mode="nearest"), gaussian_filter1d(p[:, 1], 0.6 * fps_sim, mode="nearest"),
                     gaussian_filter1d(p[:, 2], 0.8 * fps_sim, mode="nearest") + a.look_dz], 1)
    vel = np.gradient(gaussian_filter1d(p[:, 0], 0.25 * fps_sim, mode="nearest")) * fps_sim
    inz = (run["pen"] > 0).any(-1)
    on_stretch = (run["q"][:, 0] > max(xa, START - 0.3) if a.zone != "all" else np.ones(len(inz), bool))
    panels.append(dict(Q=Q, src=src, look=look, vel=vel, label=label, run=run, at=t1, inz=inz, on=on_stretch, f0=f0, d=mujoco.MjData(m)))
T = max(len(pn["Q"]) for pn in panels)  # a shorter panel (its run ends) holds its last frame

cam = mujoco.MjvCamera()
cam.distance, cam.elevation, cam.azimuth = a.dist, a.elev, 95.0
bar = 76
W, H = 2 * w + 4, h + bar
font, font2, font3 = (ImageFont.truetype("DejaVuSans-Bold.ttf", h // 22), ImageFont.truetype("DejaVuSans.ttf", h // 32),
                      ImageFont.truetype("DejaVuSans.ttf", 22))
n_dn = int(np.ceil((5 * rise + plat) / 0.15 - 1e-6))
cap = (f"Stairs-only trackers + gen_v6 planner: stairs {rise * 100:.0f} cm x5 up, {(5 * rise + plat) / n_dn * 100:.0f} cm x{n_dn} down, 30 cm treads; "
       f"joystick {speed:.1f} m/s; {a.note}; DR + sensor noise; {a.slow:g}x speed",
       "orange foot: inside the step-edge zone (5 cm + the foot's radius around a step's edge), the zone the edge penalty acts on")


def frames():
    for t in range(T):
        canvas = Image.new("RGB", (W, H), (245, 246, 248))
        dr = ImageDraw.Draw(canvas)
        dr.text((16, 8), cap[0], fill=(40, 44, 52), font=font3)
        dr.text((16, 40), cap[1], fill=(190, 100, 0), font=font3)
        for k, pn in enumerate(panels):
            tk = min(t, len(pn["Q"]) - 1)
            d, j = pn["d"], pn["src"][tk]
            d.qpos[:] = pn["Q"][tk]
            mujoco.mj_forward(m, d)
            m.light_pos[key] = pn["look"][tk] - 20 * kd
            cam.lookat[:] = pn["look"][tk]
            r.update_scene(d, cam, opt)
            for g in r.scene.geoms[: r.scene.ngeom]:
                if g.objtype == mujoco.mjtObj.mjOBJ_GEOM:
                    for sd in range(2):
                        if g.objid in feet[sd] and pn["run"]["pen"][j, sd] > 0:
                            g.rgba[:] = (0.98, 0.50, 0.05, 1)
            r.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = 1
            im = Image.fromarray(r.render())
            di = ImageDraw.Draw(im)
            di.text((h // 30, h // 40), pn["label"], fill=(20, 90, 170) if k == 0 else (120, 60, 20), font=font)
            sel = pn["on"][pn["f0"] : j + 1]
            share = pn["inz"][pn["f0"] : j + 1][sel].mean() if sel.any() else 0.0
            di.text((h // 30, h // 40 + h // 16), f"robot {pn['vel'][tk]:.1f} m/s   on the stairs so far, a foot in the edge zone {100 * share:.0f}% of the time",
                    fill=(60, 66, 76), font=font2)
            if j >= pn["at"]:
                di.text((h // 30, h // 40 + h // 16 + h // 24), pn["run"]["outcome"], fill=(40, 120, 60) if pn["run"]["ok"] else (170, 50, 40), font=font2)
            canvas.paste(im, (k * (w + 4), bar))
        yield np.asarray(canvas)
    r.close()


encode(a.out, frames(), W, H)
print(f"wrote {a.out}: {T} frames, sim {[(f0 / 50, f1 / 50) for f0, f1 in spans]} s, {[pn['run']['outcome'] for pn in panels]}", flush=True)
