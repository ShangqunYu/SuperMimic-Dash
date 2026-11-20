"""Edge detector check: the step lips the edge penalty sees (metrics.edge_penetration) and how the feet meet them,
on one course_demo run (--course, physics rollout) or one library clip's reference motion on its terrain (--lib).

Video, two panels: a close-up side view of the feet along the lips by them (orthographic, near terrain cut away, only
the lips seen end-on, so each zone is a circle round its step corner) | a 3/4 view of the steps.
Orange tube: the edge zone, 5 cm around each lip the detector finds on the heightfield (a cell boundary where the
height jumps > 3 cm, the lip at the upper height); a thin orange line marks the lip itself. Blue / purple: the
left / right foot's collision capsule (r 1.25 cm); dots: the 12 capsule-axis points the detector tests, white outside the zone and
yellow to red by depth inside (a point is in when within 5 cm + the capsule radius of a lip, i.e. the capsule
touches the tube). Black line: the capsule's closest approach to the nearest lip, while in the zone. Bottom: depth
per foot over time, detector (lines) vs an exact capsule-to-lip distance (circles) with the lips taken from the
physics boxes' top edges instead of the heightfield (independent of the detector's row / column search).

  python scripts/media/render_edge_check.py out.mp4 --course RUN_DIR --pick V S K [--zone down] [--label control]
  python scripts/media/render_edge_check.py out.mp4 --lib LIB --clip NAME [--extent 0.7] [--still 5 9]  (--still: PNGs)
  python scripts/media/render_edge_check.py - --check RUN_DIR [RUN_DIR ...]   (detector vs exact over every run)
  python scripts/media/render_edge_check.py - --check-lib LIB   (... over a library's terrain clips, reference motion)
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
from render_strobe import section_cut  # noqa: E402

from supermimic.generator.metrics import FootMetrics, edge_penetration  # noqa: E402
from supermimic.terrain.height_scan import HeightScan  # noqa: E402
from supermimic.terrain.terrains import load_terrains  # noqa: E402

ZONE, STEP = 0.05, 0.03  # edge zone radius around a lip, height jump that makes a lip (as EdgeTerm)
fm = FootMetrics(None, "cpu")
R = ZONE + fm.radius  # axis points are tested against the zone inflated by the capsule radius


def hf_lips(ter):
    """The detector's lips: heightfield cell boundaries with a jump > STEP, at the upper height, merged along the
    boundary -> segments (N, 2, 3)."""
    segs = []
    for ax in (0, 1):
        h = ter.hf if ax == 0 else ter.hf.T
        jump = np.abs(np.diff(h, axis=0)) > STEP
        lip = np.maximum(h[1:], h[:-1])
        for i in np.nonzero(jump.any(1))[0]:
            c = ter.p0[ax] + (i + 0.5) * ter.dx
            row, j = jump[i], 0
            while j < len(row):
                if not row[j]:
                    j += 1
                    continue
                k = j
                while k + 1 < len(row) and row[k + 1] and abs(lip[i, k + 1] - lip[i, j]) < 1e-6:
                    k += 1
                o0, o1 = ter.p0[1 - ax] + (j - 0.5) * ter.dx, ter.p0[1 - ax] + (k + 0.5) * ter.dx
                ends = [[c, o0], [c, o1]] if ax == 0 else [[o0, c], [o1, c]]
                segs.append([[*ends[0], lip[i, j]], [*ends[1], lip[i, j]]])
                j = k + 1
    return np.array(segs, float).reshape(-1, 2, 3)


def box_lips(boxes, floor, ds=0.005):
    """Lips from the physics: top edges of the boxes where the surface just outside (highest box there, else the
    floor) is > STEP lower -> segments (N, 2, 3), ends to within ds / 2."""
    b = np.asarray(boxes, float).reshape(-1, 6)
    top = b[:, 2] + b[:, 5]

    def surface(x, y):
        inside = (np.abs(x[:, None] - b[None, :, 0]) < b[None, :, 3]) & (np.abs(y[:, None] - b[None, :, 1]) < b[None, :, 4])
        return np.where(inside, top[None], floor).max(1)

    segs = []
    for (cx, cy, _, hx, hy, _), z in zip(b, top):
        for ax in (0, 1):
            for sgn in (-1, 1):
                c = (cx, cy)[ax] + sgn * (hx, hy)[ax]
                lo, hi = (cy - hy, cy + hy) if ax == 0 else (cx - hx, cx + hx)
                n = max(1, int(np.ceil((hi - lo) / ds)))
                w = (hi - lo) / n
                s = lo + (np.arange(n) + 0.5) * w
                out = np.full(n, c + sgn * 1e-3)
                lip = z - (surface(out, s) if ax == 0 else surface(s, out)) > STEP
                d = np.diff(np.concatenate([[0], lip.astype(int), [0]]))
                for i0, i1 in zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]):
                    a0, a1 = lo + i0 * w, lo + i1 * w
                    segs.append([[c, a0, z], [c, a1, z]] if ax == 0 else [[a0, c, z], [a1, c, z]])
    return np.array(segs, float).reshape(-1, 2, 3)


def on_border(lips, ter, tol=0.003):
    """(N,) lips on the outer edge of the heightfield grid: the physics boxes end there (a drop to the floor), while
    heightfield lookups clamp (the border cells extend), so the detector has no lip there."""
    lo = ter.p0 - ter.dx / 2
    hi = ter.p0 + (np.array(ter.hf.shape) - 0.5) * ter.dx
    xy = lips[..., :2]
    return ((np.abs(xy - lo) < tol).all(1) | (np.abs(xy - hi) < tol).all(1)).any(-1)  # both ends on the same grid side


def seg_seg(p1, q1, p2, q2):
    """Closest points of segments p1q1 (..., 3) and p2q2 (..., 3) (Ericson 5.1.9) -> distance, point on 1, on 2."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = (d1 * d1).sum(-1), (d2 * d2).sum(-1), (d2 * r).sum(-1)
    c, b = (d1 * r).sum(-1), (d1 * d2).sum(-1)
    den = a * e - b * b
    s = torch.where(den > 1e-12, ((b * f - c * e) / den.clamp_min(1e-12)).clamp(0, 1), torch.zeros_like(den))
    t = (b * s + f) / e
    s = torch.where(t < 0, (-c / a).clamp(0, 1), torch.where(t > 1, ((b - c) / a).clamp(0, 1), s))
    t = ((b * s + f) / e).clamp(0, 1)
    c1, c2 = p1 + d1 * s[..., None], p2 + d2 * t[..., None]
    return (c1 - c2).norm(dim=-1), c1, c2


def depths(Q, ter, lips_exact):
    """Q (T, 25) -> detector depth per axis point (T, 2, 12), exact capsule depth per foot (T, 2), closest capsule
    axis point and lip point (T, 2, 3) each, capsule ends (T, 2, 2, 3)."""
    q = torch.as_tensor(Q, dtype=torch.float32)[None]
    T = len(Q)
    scan = HeightScan(SimpleNamespace(terrains=[ter]))
    pts = fm.axis(q[..., :3], q[..., 3:7], q[..., 7:25], 12)[0]
    det = edge_penetration(scan, pts.reshape(T, -1, 3), torch.zeros(T), R).reshape(T, 2, 12)
    ends = fm.ends_w(q[..., :3], q[..., 3:7], q[..., 7:25])[0]  # (T, 2, 2, 3)
    L = torch.as_tensor(lips_exact, dtype=torch.float32)
    ex, c1s, c2s = torch.zeros(T, 2), torch.zeros(T, 2, 3), torch.zeros(T, 2, 3)
    for t0 in range(0, T, 200):
        e = ends[t0 : t0 + 200, :, None]  # (t, 2, 1, 2, 3)
        d, c1, c2 = seg_seg(e[..., 0, :], e[..., 1, :], L[None, None, :, 0], L[None, None, :, 1])
        i = d.argmin(-1)
        ex[t0 : t0 + 200] = (R - d.gather(-1, i[..., None])[..., 0]).clamp_min(0)
        c1s[t0 : t0 + 200] = c1.gather(2, i[..., None, None].expand(-1, -1, 1, 3))[:, :, 0]
        c2s[t0 : t0 + 200] = c2.gather(2, i[..., None, None].expand(-1, -1, 1, 3))[:, :, 0]
    return det.numpy(), ex.numpy(), c1s.numpy(), c2s.numpy(), ends.numpy()


def point_seg(P, S):
    """Distance from points P (n, 3) to the nearest of segments S (m, 2, 3)."""
    p = torch.as_tensor(P, dtype=torch.float32)[:, None]
    a, b = (torch.as_tensor(S, dtype=torch.float32)[None, :, k] for k in (0, 1))
    t = (((p - a) * (b - a)).sum(-1) / ((b - a) ** 2).sum(-1).clamp_min(1e-12)).clamp(0, 1)
    return (p - a - t[..., None] * (b - a)).norm(dim=-1).min(1).values.numpy()


def lips_agree(A, B, ds=0.01):
    """Largest distance (m) from a point on lip set A (sampled every ds) to lip set B."""
    pts = [s[0] + (s[1] - s[0]) * w for s in A for w in np.linspace(0, 1, max(2, int(np.linalg.norm(s[1] - s[0]) / ds) + 1))]
    return float(point_seg(np.array(pts), B).max()) if pts else 0.0


def course_run(run_dir, pick):
    tr = torch.load(Path(run_dir) / "traj.pt", weights_only=False)
    i = tr["combo"].index(tuple(pick))
    rise, gap, plat = tr["variants"][pick[0]]
    ter, marks, finish = course(rise, gap, plat)
    return tr["robot"][i].numpy().astype(np.float64), tr["results"][i], ter, ter.boxes(), marks, (rise, gap, plat), tr["speeds"][pick[1]]


def compare(det, ex):
    """Per foot-frame: |detector - exact| (cm) and zone membership disagreement."""
    d = det.max(-1)
    err = np.abs(d - ex) * 100
    return err, (d > 0) != (ex > 0)


def check(dirs):
    for run_dir in dirs:
        tr = torch.load(Path(run_dir) / "traj.pt", weights_only=False)
        lips = {}
        errs, flips, inz = [], 0, 0
        for i, (v, s, k) in enumerate(tr["combo"]):
            if v not in lips:
                ter = course(*tr["variants"][v])[0]
                hl, bl = hf_lips(ter), box_lips(ter.boxes(), ter.floor)
                assert not on_border(bl, ter).any()
                lips[v] = (ter, bl)
                print(f"{run_dir} variant {v}: {len(hl)} heightfield lips, {len(bl)} box lips; heightfield -> box {1000 * lips_agree(hl, bl):.1f} mm, "
                      f"box -> heightfield {1000 * lips_agree(bl, hl):.1f} mm", flush=True)
            ter, bl = lips[v]
            Q = tr["robot"][i].numpy().astype(np.float64)
            res = tr["results"][i]
            Q = Q[: int(res["down_at_s"] * 50) if res["down_at_s"] is not None else len(Q)]
            det, ex = depths(Q, ter, bl)[:2]
            e, f = compare(det, ex)
            errs.append(e.ravel())
            flips += int(f.sum())
            inz += int((ex > 0).sum())
        e = np.concatenate(errs)
        print(f"{run_dir}: {len(e)} foot-frames, {inz} in the zone (exact); |detector - exact| mean {e.mean():.3f} cm, p99 {np.percentile(e, 99):.3f} cm, "
              f"max {e.max():.2f} cm; in/out disagree on {flips} ({100 * flips / len(e):.3f}%)", flush=True)


def check_lib(lib, n_max):
    pk = torch.load(lib, map_location="cpu", weights_only=False, mmap=True)
    ts = load_terrains(lib)
    lips, errs, flips, inz, worst, near_border = {}, [], 0, 0, (0.0, ""), np.inf
    ids = [i for i, t in enumerate(pk["clip_tile"]) if int(t) >= 0][:n_max]
    for i in ids:
        tile = int(pk["clip_tile"][i])
        if tile not in lips:
            ter = ts.terrains[tile]
            hl, bl = hf_lips(ter), box_lips(ts.boxes[tile], ter.floor)
            bd = on_border(bl, ter)
            lips[tile] = (ter, bl, max(lips_agree(hl, bl), lips_agree(bl[~bd], hl)), int(bd.sum()))
        ter, bl = lips[tile][:2]
        q = pk["ref"][int(pk["starts"][i]) : int(pk["starts"][i]) + int(pk["lengths"][i]), :2].float().numpy()
        bd = bl[on_border(bl, ter)]
        if len(bd):
            near_border = min(near_border, float(point_seg(np.concatenate([q, np.zeros((len(q), 1))], 1), bd * [1, 1, 0]).min()))
        s, n = int(pk["starts"][i]), int(pk["lengths"][i])
        det, ex = depths(pk["ref"][s : s + n, :25].float().numpy(), ter, bl)[:2]
        e, f = compare(det, ex)
        errs.append(e.ravel())
        flips += int(f.sum())
        inz += int((ex > 0).sum())
        worst = max(worst, (float(e.max()), pk["names"][i]))
    e = np.concatenate(errs)
    print(f"{lib}: {len(ids)} clips on {len(lips)} terrains; lips heightfield vs boxes within {1000 * max(v[2] for v in lips.values()):.1f} mm "
          f"(off the grid border; {sum(v[3] > 0 for v in lips.values())} terrains also have box lips on the border, the root comes "
          f"{near_border:.2f} m from one at the closest); "
          f"{len(e)} foot-frames, {inz} in the zone (exact); |detector - exact| mean {e.mean():.3f} cm, p99 {np.percentile(e, 99):.3f} cm, "
          f"max {e.max():.2f} cm ({worst[1]}); in/out disagree on {flips} ({100 * flips / len(e):.3f}%)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--course")
    ap.add_argument("--pick", type=int, nargs=3)
    ap.add_argument("--zone", choices=("up", "down"), default="down")
    ap.add_argument("--label", default="")
    ap.add_argument("--lib")
    ap.add_argument("--clip")
    ap.add_argument("--check", nargs="*")
    ap.add_argument("--check-lib", help="detector vs exact over the reference motion of a library's terrain clips")
    ap.add_argument("--max-clips", type=int, default=10**9)
    ap.add_argument("--slow", type=float, default=0.25)
    ap.add_argument("--size", type=int, nargs=2, default=[960, 720])
    ap.add_argument("--extent", type=float, default=0.6, help="m: height of the profile view")
    ap.add_argument("--still", type=float, nargs="*", help="write PNG frames at these playback times (s) instead of the video")
    a = ap.parse_args()
    if a.check:
        return check(a.check)
    if a.check_lib:
        return check_lib(a.check_lib, a.max_clips)

    if a.course:
        q50, res, ter, boxes, marks, (rise, gap, plat), speed = course_run(a.course, a.pick)
        top = START + 5 * TREAD
        d0 = marks["platform"] - 0.3 + LANDING
        xa, xb = (START - 0.6, top + 0.4) if a.zone == "up" else (d0 - 0.5, marks["stairs down"])
        t_end = int(res["down_at_s"] * 50) if res["down_at_s"] is not None else len(q50)
        x = q50[:t_end, 0]
        f0 = int(np.argmax(x > xa))
        f1 = int(np.argmax(x > xb)) if (x > xb).any() else t_end - 1
        n_dn = int(np.ceil((5 * rise + plat) / 0.15 - 1e-6))
        what = (f"{a.label + ' tracker' if a.label else 'tracker'} on the course (physics, DR + sensor noise), stairs "
                f"{'up ' + f'{rise * 100:.0f} cm x5' if a.zone == 'up' else 'down ' + f'{(5 * rise + plat) / n_dn * 100:.0f} cm x{n_dn}'}, "
                f"{speed:.1f} m/s")
        floor = ter.floor
    else:
        pk = torch.load(a.lib, map_location="cpu", weights_only=False, mmap=True)
        ts = load_terrains(a.lib)
        i = list(pk["names"]).index(a.clip)
        s, n = int(pk["starts"][i]), int(pk["lengths"][i])
        q50 = pk["ref"][s : s + n, :25].float().numpy().astype(np.float64)
        tile = int(pk["clip_tile"][i])
        ter, boxes = ts.terrains[tile], ts.boxes[tile]
        f0, f1 = 0, len(q50) - 1
        what = f"reference motion (no physics) {a.clip}"
        floor = ter.floor
    hl, bl = hf_lips(ter), box_lips(boxes, floor)
    agree = (lips_agree(hl, bl), lips_agree(bl[~on_border(bl, ter)], hl))
    print(f"{len(hl)} heightfield lips, {len(bl)} box lips; heightfield -> box {1000 * agree[0]:.1f} mm, box -> heightfield {1000 * agree[1]:.1f} mm", flush=True)

    f0 = max(0, f0 - 15)
    f1 = min(len(q50) - 1, f1 + 25)
    Qz = q50[f0 : f1 + 1]
    det, ex, c1, c2, ends = depths(Qz, ter, bl)
    err, flip = compare(det, ex)
    print(f"{len(Qz)} frames: |detector - exact| per foot mean {err.mean():.3f} cm, max {err.max():.2f} cm; in/out disagree on {int(flip.sum())} foot-frames; "
          f"a foot in the zone {100 * (ex > 0).any(-1).mean():.0f}% of frames (exact) / {100 * (det > 0).any((-1, -2)).mean():.0f}% (detector)", flush=True)

    w, h = a.size
    pw, ph = 2 * w + 4, 190  # plot strip
    bar = 112
    W, H = pw, bar + h + ph
    Q = resample(Qz, 50, a.slow)
    src = np.minimum(np.round(np.arange(len(Q)) * a.slow / FPS * 50).astype(int), len(Qz) - 1)
    fps_play = FPS / a.slow
    m0 = dash_model([], 64, 64)
    cut = section_cut(boxes, Qz, m0, mujoco.MjData(m0), margin=0.06)
    base = [[0, 0, floor - 0.05, 0.01, 0.01, 0.05]]
    ms = [dash_model(np.array(base + [list(b) for b in cut]), w, h), dash_model(np.array(base + [list(b) for b in boxes]), w, h)]
    for m in ms:
        for g in range(m.ngeom):
            if m.geom(g).name.startswith("terrain_"):
                m.geom_matid[g] = -1
                m.geom_rgba[g] = (0.62, 0.68, 0.80, 1.0)
    feet = [{g for g in range(ms[0].ngeom) if ms[0].geom_bodyid[g] == ms[0].body(f"{sd}_foot").id} for sd in "lr"]
    mid = ends.mean((1, 2))  # (T, 3) feet midpoint
    mid = np.stack([np.interp(np.arange(len(Q)) * a.slow / FPS * 50, np.arange(len(Qz)), mid[:, k]) for k in range(3)], 1)
    sm = lambda x, s: gaussian_filter1d(x, s * fps_play, mode="nearest")  # noqa: E731
    look = np.stack([sm(mid[:, 0], 0.35), sm(mid[:, 1], 0.6), sm(mid[:, 2], 0.5) + 0.02], 1)
    u = Qz[-1, :2] - Qz[0, :2]
    head = np.degrees(np.arctan2(u[1], u[0]))
    lat = np.array([-np.sin(np.radians(head)), np.cos(np.radians(head))])
    cut_at = float(min(e[:2] @ lat for e in ends.reshape(-1, 3))) - 0.06  # as section_cut (feet capsule ends; near side removed)
    ms[0].vis.global_.orthographic, ms[0].vis.global_.fovy = 1, a.extent  # profile: orthographic, fovy = view height (m)
    rs = [mujoco.Renderer(m, h, w, max_geom=20000) for m in ms]
    ds = [mujoco.MjData(m) for m in ms]
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[[0, 1, 2]] = 1
    opt.sitegroup[:] = 0
    cam = mujoco.MjvCamera()

    def trim(seg):
        """Lip segment on the kept side of the profile cut (None if wholly cut)."""
        s0, s1 = seg[0][:2] @ lat, seg[1][:2] @ lat
        if max(s0, s1) < cut_at:
            return None
        if min(s0, s1) >= cut_at:
            return seg
        w_ = (cut_at - s0) / (s1 - s0)
        p = seg[0] + w_ * (seg[1] - seg[0])
        return np.array([p, seg[1]]) if s0 < cut_at else np.array([seg[0], p])

    lips_profile = [s for s in (trim(s) for s in hl) if s is not None]

    def near(L, p, r):
        """Horizontal distance from p to lip segment L below r."""
        u = L[1, :2] - L[0, :2]
        t = np.clip((p[:2] - L[0, :2]) @ u / max(u @ u, 1e-12), 0, 1)
        return np.linalg.norm(p[:2] - L[0, :2] - t * u) < r

    # profile camera along the lips by the feet (heightfield lips run along x or y), from the right of the path
    run = {0: 0.0, 90: 0.0}
    for L in hl:
        if any(near(L, p, 0.3) for p in mid[:: max(1, len(mid) // 60)]):
            d = L[1, :2] - L[0, :2]
            run[0 if abs(d[0]) > abs(d[1]) else 90] += float(np.linalg.norm(d))
    az = max(run, key=run.get)
    lips_profile = [L for L in lips_profile if (abs(L[1, 0] - L[0, 0]) > abs(L[1, 1] - L[0, 1])) == (az == 0)]  # end-on only
    az = az if np.array([np.cos(np.radians(az)), np.sin(np.radians(az))]) @ lat > 0 else az + 180
    views = [(ms[0], 3.0, 0.0, az, True), (ms[1], 1.35, -28.0, head + 125, False)]

    def add(scn, typ, size, pos, rgba, mat=np.eye(3)):
        if scn.ngeom >= scn.maxgeom:
            return None
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, typ, np.asarray(size, float), np.asarray(pos, float), np.asarray(mat, float).ravel(), np.asarray(rgba, np.float32))
        scn.ngeom += 1
        return g

    def seg(scn, typ, width, p, q_, rgba):
        g = add(scn, typ, np.zeros(3), np.zeros(3), rgba)
        if g is not None:
            mujoco.mjv_connector(g, typ, width, np.asarray(p, float), np.asarray(q_, float))

    def ramp(dcm):
        t = min(1.0, dcm / 2.5)
        return (1.0, 0.85 - 0.75 * t, 0.15 - 0.05 * t, 1.0)

    font, font2, font3 = ImageFont.truetype("DejaVuSans-Bold.ttf", 24), ImageFont.truetype("DejaVuSans.ttf", 21), ImageFont.truetype("DejaVuSans.ttf", 18)
    cap = [f"Edge detector check: {what}; {a.slow:g}x speed",
           "orange tube: edge zone, 5 cm around each step lip the detector finds (height jump > 3 cm on the heightfield); thin orange line: the lip.  "
           "blue / purple: left / right foot collision capsule",
           "dots: the 12 points per foot the detector tests (white = outside, yellow -> red = deeper inside); a foot is in the zone when its capsule touches "
           "a tube.  black line: closest approach to the lip",
           f"lips from the heightfield vs the physics boxes' edges: within {1000 * max(agree):.1f} mm.  detector vs exact capsule-to-lip depth over this "
           f"stretch: max diff {err.max():.2f} cm, in/out disagree on {int(flip.sum())} of {flip.size} foot-frames"]
    LR = [(40, 110, 200), (140, 50, 190)]
    T50 = len(Qz)
    x0p, x1p, y0p, y1p = 70, pw - 30, 30, ph - 34
    ymax = max(3.0, float(np.ceil(max(det.max(), ex.max()) * 100)))
    plot = Image.new("RGB", (pw, ph), (250, 250, 251))
    dp = ImageDraw.Draw(plot)
    tx = lambda f: x0p + (x1p - x0p) * f / max(1, T50 - 1)  # noqa: E731
    ty = lambda v: y1p - (y1p - y0p) * min(v, ymax) / ymax  # noqa: E731
    for v in range(int(ymax) + 1):
        dp.line([(x0p, ty(v)), (x1p, ty(v))], fill=(225, 227, 232))
        dp.text((x0p - 40, ty(v) - 10), f"{v} cm", fill=(90, 96, 106), font=font3)
    for s_ in range(0, T50, 50):
        dp.text((tx(s_) - 8, y1p + 6), f"{(f0 + s_) / 50:.0f}s", fill=(90, 96, 106), font=font3)
    for sd in range(2):
        dd = det[:, sd].max(-1) * 100
        dp.line([(tx(f), ty(dd[f])) for f in range(T50)], fill=LR[sd], width=3)
        for f in range(0, T50, 2):
            if ex[f, sd] > 0 or dd[f] > 0:
                v = ex[f, sd] * 100
                dp.ellipse([tx(f) - 3, ty(v) - 3, tx(f) + 3, ty(v) + 3], outline=tuple(int(c * 0.55) for c in LR[sd]), width=1)
    dp.text((x0p, 4), "depth inside the edge zone per foot:  left foot (blue), right foot (purple); lines = detector, circles = exact geometry",
            fill=(40, 44, 52), font=font3)

    def frames():
        for t in range(len(Q)):
            k = src[t]
            canvas = Image.new("RGB", (W, H), (245, 246, 248))
            dr = ImageDraw.Draw(canvas)
            dr.text((16, 6), cap[0], fill=(30, 34, 40), font=font)
            for li, line in enumerate(cap[1:]):
                dr.text((16, 38 + 24 * li), line, fill=(150, 80, 0) if li < 2 else (40, 100, 60), font=font3)
            for vi, (m, dist, elev, az, prof) in enumerate(views):
                d = ds[vi]
                d.qpos[:] = Q[t]
                mujoco.mj_forward(m, d)
                key = m.light("key").id
                m.light_pos[key] = look[t] - 20 * m.light_dir[key] / np.linalg.norm(m.light_dir[key])
                cam.lookat[:], cam.distance, cam.elevation, cam.azimuth = look[t], dist, elev, az
                r = rs[vi]
                r.update_scene(d, cam, opt)
                scn = r.scene
                for g in scn.geoms[: scn.ngeom]:
                    if g.objtype == mujoco.mjtObj.mjOBJ_GEOM and (g.objid in feet[0] or g.objid in feet[1]):
                        g.rgba[3] = 0.12 if prof else 0.3
                for L in lips_profile if prof else hl:
                    if not near(L, look[t], 0.5 if prof else 3.0):  # profile: only lips by the feet
                        continue
                    seg(scn, mujoco.mjtGeom.mjGEOM_CYLINDER, ZONE, L[0], L[1], (1.0, 0.55, 0.10, 0.22))
                    seg(scn, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.0025, L[0], L[1], (0.95, 0.45, 0.0, 1.0))
                for sd in range(2):
                    inz = det[k, sd].max() > 0
                    seg(scn, mujoco.mjtGeom.mjGEOM_CAPSULE, fm.radius, ends[k, sd, 0], ends[k, sd, 1], (*(c / 255 for c in LR[sd]), 0.3 if inz else 0.2))
                    pts = ends[k, sd, 0][None] * (1 - np.linspace(0, 1, 12))[:, None] + ends[k, sd, 1][None] * np.linspace(0, 1, 12)[:, None]
                    for p, dp_ in zip(pts, det[k, sd]):
                        add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.0055, 0, 0], p, ramp(dp_ * 100) if dp_ > 0 else (0.97, 0.97, 0.97, 1.0))
                    if ex[k, sd] > 0:
                        seg(scn, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.0016, c1[k, sd], c2[k, sd], (0.05, 0.05, 0.05, 1.0))
                        add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.004, 0, 0], c2[k, sd], (0.05, 0.05, 0.05, 1.0))
                scn.flags[mujoco.mjtRndFlag.mjRND_FOG] = 1
                im = Image.fromarray(r.render())
                di = ImageDraw.Draw(im)
                di.text((14, 10), "feet, side view (orthographic, near side cut away)" if prof else "3/4 view", fill=(40, 44, 52), font=font2)
                if prof:
                    for sd in range(2):
                        dd, ee = det[k, sd].max() * 100, ex[k, sd] * 100
                        txt = (f"{'left' if sd == 0 else 'right'} foot: " + (f"IN the zone, detector {dd:.1f} cm deep, exact {ee:.1f} cm" if dd > 0 or ee > 0
                                                                           else "outside the zone"))
                        di.text((14, 42 + 28 * sd), txt, fill=LR[sd], font=font2)
                canvas.paste(im, (vi * (w + 4), bar))
            pl = plot.copy()
            dq = ImageDraw.Draw(pl)
            dq.line([(tx(k), y0p - 4), (tx(k), y1p + 2)], fill=(30, 30, 30), width=2)
            canvas.paste(pl, (0, bar + h))
            yield np.asarray(canvas)

    if a.still:
        want = {round(x * FPS) for x in a.still}
        for t, f in enumerate(frames()):
            if t in want:
                Image.fromarray(f).save(f"{a.out}.{t / FPS:.1f}s.png")
                want.discard(t)
            if not want:
                break
    elif a.out != "-":
        encode(a.out, frames(), W, H)
        for r in rs:
            r.close()
        print(f"wrote {a.out}: {len(Q)} frames ({len(Q) / FPS:.0f} s), sim {f0 / 50:.1f}-{f1 / 50:.1f} s", flush=True)


if __name__ == "__main__":
    main()
