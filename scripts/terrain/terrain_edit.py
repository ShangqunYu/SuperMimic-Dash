"""Terrain clips, step 2: fit clean terrain, then edit the retargeted motion to it.

Terrain (Dash scale, straight edges along the walking direction d):
  * boxes: the two-level profile of pilot_terrain.py, then the top lowered to ``box_height``;
  * stairs: a regular staircase (one rise, one run) fitted to all footholds and extended past
    both clip ends, so loop and jog clips (mid-staircase, few still footholds) get every step.
Motion edit (as in Generate-Track-Improve: foot offsets between contacts, then IK):
  * every stance phase of a foot gets a constant offset: dz = new - old height of the surface
    it stands on, ds = the slide along d that puts the whole foot on its tread (``margin``
    inside both edges; centred if the tread is shorter than the foot);
  * offsets blend over the swing between stance phases (smoothstep), the torso moves by the
    mean of both feet's offsets (arms ride along), and a damped least-squares leg IK puts each
    foot at its original pose plus its offset.
Scores against the edited terrain: foot/shin penetration, the fraction of each standing foot's
sole that is on a surface (support), and the IK residual.

  uv run python scripts/terrain/terrain_edit.py --only stairs
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
import tyro
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pilot_terrain import (  # noqa: E402
    CELL, FPS_IN, HUMAN_BONES, STEP, cluster_levels, profile_heightfield, recover, render_clip, stance_mask, to_qpos, track,
)

from supermimic.robots import get_robot  # noqa: E402

LEG = ("hip_yaw", "hip_roll", "hip_pitch", "knee_pitch", "ankle_pitch")


# ---------------------------------------------------------------- terrain models
class Profile:
    """h(s) piecewise constant along unit direction d through point c: heights[searchsorted(edges, s)]."""

    def __init__(self, c, d, edges, heights):
        self.c, self.d, self.edges, self.heights = np.asarray(c, float), np.asarray(d, float), np.asarray(edges, float), np.asarray(heights, float)

    def s(self, xy):
        return (xy - self.c) @ self.d

    def seg(self, s):
        return np.searchsorted(self.edges, s)

    def h(self, xy):
        return self.heights[self.seg(self.s(xy))]

    def tread(self, j):
        lo = self.edges[j - 1] if j > 0 else -np.inf
        hi = self.edges[j] if j < len(self.edges) else np.inf
        return lo, hi

    def grid(self, lo, hi):
        nx, ny = int(np.ceil((hi[0] - lo[0]) / CELL)) + 1, int(np.ceil((hi[1] - lo[1]) / CELL)) + 1
        X, Y = np.meshgrid(lo[0] + CELL * np.arange(nx), lo[1] + CELL * np.arange(ny))
        return self.h(np.stack([X, Y], -1)), (lo[0], lo[1])


def stance_phases(stance_foot: np.ndarray, min_len: int = 3):
    """bool (T,) -> list of (start, end) inclusive."""
    lab, n = ndimage.label(stance_foot)
    out = []
    for i in range(1, n + 1):
        idx = np.flatnonzero(lab == i)
        if len(idx) >= min_len:
            out.append((idx[0], idx[-1]))
    return out


def fit_stairs(pts, hs, fe, sp, foot_r, shin_r, stance):
    """Regular staircase: heights h0 + k*rise, edge between steps k-1 and k at e0 + k*run along d."""
    lab, levels = cluster_levels(hs, 0.04)
    counts = np.bincount(lab)
    lv = np.sort(levels[counts >= 4])
    diffs = np.diff(lv)
    small = diffs[diffs < 1.5 * diffs.min()] if len(diffs) else np.array([0.12])
    rise = float(np.median(small))
    h0 = 0.0
    for _ in range(3):  # h = h0 + k * rise, least squares on the level heights
        k = np.round((lv - h0) / rise)
        A = np.stack([np.ones_like(k), k], 1)
        h0, rise = np.linalg.lstsq(A, lv, rcond=None)[0]
    kf = np.round((hs - h0) / rise)  # step index of every foothold
    k_lo, k_hi = int(kf.min()), int(kf.max())
    # Direction and run: footholds on the steps advance one run per step, xy ~ a + k * b. The
    # floor and the top landing are left out (their footholds can be anywhere along them).
    mid = (kf > k_lo) & (kf < k_hi)
    if len(np.unique(kf[mid])) >= 2:
        A = np.stack([np.ones(mid.sum()), kf[mid]], 1)
        coef = np.linalg.lstsq(A, pts[mid], rcond=None)[0]
        b = coef[1]
        run0 = float(np.linalg.norm(b))
        d = b / run0
        c = coef[0]  # xy where step 0's footholds would be
    else:  # too few steps: direction from the lowest to the highest footholds, a typical run
        d = pts[kf == k_hi].mean(0) - pts[kf == k_lo].mean(0)
        d = d / np.linalg.norm(d)
        run0 = 0.18
        c = pts[kf == k_hi].mean(0) - (k_hi * run0 + 0.5 * run0) * d

    fs = (fe[..., :2] - c) @ d
    fz = fe[..., 2] - foot_r
    ss, sz = (sp[..., :2] - c) @ d, sp[..., 2] - shin_r
    standing = stance.any(-1)
    phases = [(f, a, z) for f in range(2) for a, z in stance_phases(standing[:, f])]
    s_all = np.concatenate([fs.ravel(), ss.ravel()])
    n_lo = min(k_lo - 3, int(np.floor(s_all.min() / max(run0, 0.05))) - 2)
    n_hi = max(k_hi + 3, int(np.ceil(s_all.max() / max(run0, 0.05))) + 2)
    ks = np.arange(max(n_lo, 1 if h0 < rise / 2 else n_lo), n_hi + 1)  # edges below step k=0 (the floor) are not needed

    def build(e0, run):
        edges = e0 + ks * run
        heights = np.concatenate([[h0 + (ks[0] - 1) * rise], h0 + ks * rise])
        return Profile(c, d, edges, np.maximum(heights, min(0.0, h0)))

    def cost(p):
        hf = p.heights[p.seg(fs)]
        pen = np.clip(hf - fz, 0, None).sum() + np.clip(p.heights[p.seg(ss)] - sz, 0, None).sum()
        over = 0.0  # how far standing feet stick out of their tread (fixed later by sliding, so cheap)
        for f, a, z in phases:
            sa = fs[a : z + 1, f].mean(0)
            j = int(np.argmin(np.abs(p.heights - np.median(fz[a : z + 1, f].min(-1)))))
            lo, hi = p.tread(j)
            over += max(0.0, lo - (sa.min() - foot_r)) + max(0.0, (sa.max() + foot_r) - hi)
        return pen + 0.2 * over * len(fs) / max(len(phases), 1)

    best = None
    for run in np.linspace(0.85 * run0, 1.2 * run0, 15):
        for e0 in np.linspace(-run, 0, 21):
            p = build(e0, run)
            cst = cost(p)
            if best is None or cst < best[0]:
                best = (cst, p, run)
    p = best[1]
    return p, {"rise": float(rise), "run": float(best[2]), "h0": float(h0), "steps": k_hi - k_lo, "e0": float(p.edges[0] - ks[0] * best[2]), "ks": ks}


def fit_box(pts, hs, lab, levels, fe, sp, foot_r, shin_r, stance, extent):
    prof = profile_heightfield(pts, lab, levels, fe, sp, foot_r, shin_r, stance, extent, stairs=False)
    if prof is None:
        return None
    H, origin, d, edges, heights = prof
    c = pts.mean(0)
    return Profile(c, d, edges, heights)


# ---------------------------------------------------------------- motion edit
def rot_between(v, w):
    """Rotation vector turning direction v onto direction w."""
    v, w = v / np.linalg.norm(v), w / np.linalg.norm(w)
    ax = np.cross(v, w)
    sn, cs = np.linalg.norm(ax), float(np.dot(v, w))
    return np.zeros(3) if sn < 1e-9 else ax / sn * np.arctan2(sn, cs)


def foot_offsets(fe, foot_r, stance, old: Profile, new: Profile, margin: float):
    """Per foot and frame: capsule-centre offset dc (T,2,3) and rotation offset (T,2,3, rotation vector).

    Each stance phase gets one constant offset: the foot is turned about its sideways axis so its
    sole is level (from the frames where heel and toe both rest, else all of the phase), placed
    at its surface's new height, and slid along d onto its tread (see main docstring). Swings
    blend between consecutive stance offsets with a smoothstep."""
    from scipy.spatial.transform import Rotation as Rot

    T = fe.shape[0]
    standing = stance.any(-1)
    dc = np.zeros((T, 2, 3))
    rv = np.zeros((T, 2, 3))
    for f in range(2):
        keys = []  # (start, end, dc, rotvec)
        for a, z in stance_phases(standing[:, f]):
            flat = stance[a : z + 1, f].all(-1)
            sel = np.arange(a, z + 1)[flat] if flat.sum() >= 3 else np.arange(a, z + 1)
            e0, e1 = np.median(fe[sel, f, 0], 0), np.median(fe[sel, f, 1], 0)
            cen = (e0 + e1) / 2
            v = e0 - e1
            vh = np.array([v[0], v[1], 0.0])
            r = rot_between(v, vh)  # level the sole
            R = Rot.from_rotvec(r).as_matrix()
            f0, f1 = cen + R @ (e0 - cen), cen + R @ (e1 - cen)
            # Surface it stands on: by the height of its lower (resting) end.
            sole = min(f0[2], f1[2]) - foot_r
            j = int(np.argmin(np.abs(old.heights - (np.median(fe[a : z + 1, f, :, 2].min(-1)) - foot_r))))
            sv = old.s(np.stack([f0[:2], f1[:2]]))
            smin, smax = sv.min() - foot_r, sv.max() + foot_r
            olo, ohi = old.tread(j)
            lo, hi = new.tread(j)
            # Move with the tread (deeper stairs push every later step further along d) ...
            if np.isfinite(olo) and np.isfinite(ohi):
                base = (lo + hi) / 2 - (olo + ohi) / 2
            elif np.isfinite(olo):
                base = lo - olo
            elif np.isfinite(ohi):
                base = hi - ohi
            else:
                base = 0.0
            smin, smax = smin + base, smax + base
            # ... then slide the least amount that puts the whole foot on it.
            lo, hi = lo + margin, hi - margin
            if smax - smin > hi - lo:
                ds = base + (lo + hi) / 2 - (smin + smax) / 2
            else:
                ds = base + max(0.0, lo - smin) - max(0.0, smax - hi)
            dz = new.heights[j] - sole  # level sole onto the surface
            keys.append((a, z, np.array([ds * new.d[0], ds * new.d[1], dz]), r))
        if not keys:
            continue
        for arr, k in ((dc, 2), (rv, 3)):
            arr[: keys[0][0], f] = keys[0][k]
            for i, key in enumerate(keys):
                a, z, o = key[0], key[1], key[k]
                arr[a : z + 1, f] = o
                if i + 1 < len(keys):
                    a2, o2 = keys[i + 1][0], keys[i + 1][k]
                    n = a2 - z - 1
                    if n > 0:
                        u = (np.arange(1, n + 1) / (n + 1))[:, None]
                        w = u * u * (3 - 2 * u)
                        arr[z + 1 : a2, f] = (1 - w) * o + w * o2
            arr[keys[-1][1] + 1 :, f] = keys[-1][k]
    return dc, rv


def leg_ik(m, qpos, dc, rv, lift=None, iters: int = 50, rot_w: float = 0.3, damping: float = 1e-3, smooth: float = 1e-3,
           knee_init: float = 0.25, tol: float = 1e-5):
    """Root += mean foot offset; legs solved so each foot is at its original pose turned by rv about
    its capsule centre and moved by dc.

    Per frame, damped least squares warm-started from the previous frame's solution plus the
    source motion's change, with a small pull (``smooth``) towards that warm start so redundant
    directions do not jump. The knee starts at
    least ``knee_init`` bent: at a straight knee the leg-length direction has no gradient, so a
    target the straight leg cannot reach left the knee stuck at 0 and then snapping (> 1 rad in
    one frame) once the solver escaped."""
    from scipy.spatial.transform import Rotation as Rot

    d = mujoco.MjData(m)
    feet = [m.body(f"{s}_foot").id for s in "lr"]
    gpos = [m.geom_pos[m.geom(f"{s}_foot_collision").id] for s in "lr"]
    legs = [[m.joint(f"{s}_{j}").id for j in LEG] for s in "lr"]
    qadr = [np.array([m.jnt_qposadr[j] for j in leg]) for leg in legs]
    dadr = [np.array([m.jnt_dofadr[j] for j in leg]) for leg in legs]
    rng = [m.jnt_range[leg] for leg in legs]
    T = len(qpos)
    tgt_p, tgt_R = np.zeros((T, 2, 3)), np.zeros((T, 2, 3, 3))
    for t in range(T):
        d.qpos[:] = qpos[t]
        mujoco.mj_kinematics(m, d)
        for f in range(2):
            Rb = d.xmat[feet[f]].reshape(3, 3)
            cen = d.xpos[feet[f]] + Rb @ gpos[f]
            R2 = Rot.from_rotvec(rv[t, f]).as_matrix() @ Rb
            tgt_R[t, f] = R2
            tgt_p[t, f] = cen + dc[t, f] - R2 @ gpos[f]
            if lift is not None:  # swing clearance: the foot only, not the torso
                tgt_p[t, f, 2] += lift[t, f]
    out = qpos.copy()
    out[:, :3] += dc.mean(1)
    resid = np.zeros((T, 2))
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    knee = LEG.index("knee_pitch")
    for t in range(T):
        if t > 0:  # warm start the legs from the previous solution's change
            for f in range(2):
                out[t, qadr[f]] += out[t - 1, qadr[f]] - qpos[t - 1, qadr[f]]
        prev = [out[t, qadr[f]].copy() for f in range(2)]
        for f in range(2):
            out[t, qadr[f][knee]] = max(out[t, qadr[f][knee]], knee_init)
        for _ in range(iters):
            d.qpos[:] = out[t]
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            step = 0.0
            for f in range(2):
                ep = tgt_p[t, f] - d.xpos[feet[f]]
                R = d.xmat[feet[f]].reshape(3, 3)
                Re = tgt_R[t, f] @ R.T
                er = 0.5 * np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])
                mujoco.mj_jacBody(m, d, jp, jr, feet[f])
                J = np.vstack([jp[:, dadr[f]], rot_w * jr[:, dadr[f]]])
                e = np.concatenate([ep, rot_w * er])
                q = out[t, qadr[f]]
                dq = np.linalg.solve(J.T @ J + (damping + smooth) * np.eye(len(q)), J.T @ e + smooth * (prev[f] - q))
                out[t, qadr[f]] = np.clip(q + dq, rng[f][:, 0], rng[f][:, 1])
                step = max(step, np.abs(dq).max())
            if step < tol:
                break
        d.qpos[:] = out[t]
        mujoco.mj_kinematics(m, d)
        for f in range(2):
            resid[t, f] = np.linalg.norm(tgt_p[t, f] - d.xpos[feet[f]])
    return out, resid


def slope_limit(w: np.ndarray, s: float) -> np.ndarray:
    """Smallest envelope >= w whose change per sample is at most s."""
    e = w.copy()
    for i in range(1, len(e)):
        e[i] = max(e[i], e[i - 1] - s)
    for i in range(len(e) - 2, -1, -1):
        e[i] = max(e[i], e[i + 1] - s)
    return e


def swing_lift(prof: Profile, fe, foot_r, stance, dt: float, base=None, clearance: float = 0.015, max_speed: float = 1.0):
    """(T,2) foot lift over each swing so the whole sole clears the terrain by ``clearance``.

    ``fe`` is the motion already lifted by ``base`` (zeros if None); returns base plus the extra
    lift still needed. Needed lift per frame (9 points along the sole) -> running max over 0.15 s
    -> Gaussian smoothing, never below the need. The total then changes by at most ``max_speed``
    (m/s), starting from 0 at lift-off and back to 0 at touchdown; zero while standing. Without
    this the lift switched on at up to 4 m/s (the toe often reaches the next riser in the first
    swing frame), which the leg IK followed with knee jumps of over 1 rad per frame."""
    from scipy.ndimage import gaussian_filter1d, maximum_filter1d

    u = np.linspace(0, 1, 9)[None, None, :, None]
    sole = fe[:, :, :1] * (1 - u) + fe[:, :, 1:] * u
    need = np.clip(prof.h(sole[..., :2]) + clearance - (sole[..., 2] - foot_r), 0, None).max(-1)  # (T,2)
    base = np.zeros_like(need) if base is None else base
    standing = stance.any(-1)
    lift = np.zeros_like(need)
    s = max_speed * dt
    for f in range(2):
        lab, n = ndimage.label(~standing[:, f])
        for i in range(1, n + 1):
            idx = np.flatnonzero(lab == i)
            w = need[idx, f]
            if w.max() <= 0:
                lift[idx, f] = base[idx, f]
                continue
            sm = base[idx, f] + np.maximum(gaussian_filter1d(maximum_filter1d(w, 9, mode="nearest"), 2, mode="constant"), w)
            k = np.arange(len(idx))
            lift[idx, f] = np.minimum(slope_limit(sm, s), np.minimum(k + 1, len(idx) - k) * s)
    return lift


# ---------------------------------------------------------------- scores
def scores(prof: Profile, fe, sp, foot_r, shin_r, stance, resid):
    hf = prof.h(fe[..., :2])
    foot_pen = hf - (fe[..., 2] - foot_r)
    shin_pen = prof.h(sp[..., :2]) - (sp[..., 2] - shin_r)
    # Support: sample the sole along each foot (heel..toe); a sample is supported if a surface is within 2 cm under it.
    u = np.linspace(0, 1, 9)[None, None, :, None]
    sole = fe[:, :, :1] * (1 - u) + fe[:, :, 1:] * u  # (T,2,9,3)
    gap = (sole[..., 2] - foot_r) - prof.h(sole[..., :2])
    frac = (np.abs(gap) < 0.02).mean(-1)  # (T,2)
    standing = stance.any(-1)
    fr = frac[standing]
    return {
        "foot_pen_max": float(foot_pen.max()),
        "foot_pen_frac_1cm": float(np.mean(foot_pen.max(axis=(1, 2)) > 0.01)),
        "shin_pen_max": float(shin_pen.max()),
        "support_p5": float(np.percentile(fr, 5)) if len(fr) else 1.0,
        "support_lt80": float(np.mean(fr < 0.8)) if len(fr) else 0.0,
        "ik_resid_max": float(resid.max()),
        "ik_resid_stance_max": float(resid[standing].max()) if standing.any() else 0.0,
    }


def joint_speed(q_src: np.ndarray, q_edit: np.ndarray, dt: float) -> dict:
    """Peak joint speeds of the edit against its source. ``speed_excess`` > 0 means the edit sped
    some joint up beyond 2x its own peak speed in the source plus 5 rad/s (IK jumps)."""
    v = (np.abs(np.diff(q_edit[:, 7:], axis=0)) / dt).max(0)
    v0 = (np.abs(np.diff(q_src[:, 7:], axis=0)) / dt).max(0)
    return {"joint_speed_max": float(v.max()), "src_joint_speed_max": float(v0.max()), "speed_excess": float((v - 2 * v0 - 5).max())}


def fmt(sc):
    return (f"pen {sc['foot_pen_max'] * 100:5.1f}cm ({sc['foot_pen_frac_1cm'] * 100:4.1f}%>1cm) shin {sc['shin_pen_max'] * 100:5.1f}cm  "
            f"support p5 {sc['support_p5'] * 100:3.0f}% (<80%: {sc['support_lt80'] * 100:4.1f}% fr)  stance ik {sc['ik_resid_stance_max'] * 100:.1f}cm"
            + (f"  joints {sc['joint_speed_max']:.0f} rad/s (src {sc['src_joint_speed_max']:.0f})" if "joint_speed_max" in sc else ""))


def main(raw: Path = Path("data/terrain_pilot/raw"), meta: Path = Path("data/terrain_pilot/pilot_meta.csv"), out: Path = Path("logs/terrain_edit"),
         save: Path = Path("data/terrain_pilot/edited"), box_height: float = 0.30, min_run: float = 0.22, margin: float = 0.01, render: bool = True, only: str = "",
         shard: int = 0, nshards: int = 1) -> None:
    (out / "clips").mkdir(parents=True, exist_ok=True)
    save.mkdir(parents=True, exist_ok=True)
    m = mujoco.MjModel.from_xml_path(str(get_robot("dash").xml_path()))
    info = {r["name"]: r for r in csv.DictReader(open(meta))}
    dt = STEP / FPS_IN
    rows = []
    for f in sorted(raw.glob(f"dash_*{only}*.npy" if only else "dash_*.npy"))[shard::nshards]:
        try:
            clip = f.stem.removeprefix("dash_")
            content = info.get(clip, {}).get("content", "")
            kind = "stairs" if "stair" in content else "box" if "box" in content or "50cm" in clip else None
            if kind is None:
                continue
            dat = np.load(f, allow_pickle=True).item()
            qpos = to_qpos(dat["q"])[::STEP]
            fe, sp, foot_r, shin_r = track(m, qpos)
            pts, hs, lab, levels, _, stance = recover(fe, foot_r, dt)
            lo = np.minimum(fe[..., :2].reshape(-1, 2).min(0), qpos[:, :2].min(0)) - 0.6
            hi = np.maximum(fe[..., :2].reshape(-1, 2).max(0), qpos[:, :2].max(0)) + 0.6
            if kind == "stairs":
                old, par = fit_stairs(pts, hs, fe, sp, foot_r, shin_r, stance)
                run = max(par["run"], min_run)
                new = Profile(old.c, old.d, par["e0"] + par["ks"] * run, old.heights)
                desc = f"stairs rise {par['rise'] * 100:.1f} cm, run {par['run'] * 100:.1f} -> {run * 100:.1f} cm"
            else:
                old = fit_box(pts, hs, lab, levels, fe, sp, foot_r, shin_r, stance, ((lo[0], hi[0]), (lo[1], hi[1])))
                if old is None:
                    print(f"{clip}: no box profile, skipped")
                    continue
                top = old.heights.max()
                new = Profile(old.c, old.d, old.edges, np.where(old.heights > top / 2, box_height, old.heights))
                desc = f"box {top * 100:.0f} -> {box_height * 100:.0f} cm"
            sc0 = scores(old, fe, sp, foot_r, shin_r, stance, np.zeros((len(qpos), 2)))
            dc, rv = foot_offsets(fe, foot_r, stance, old, new, margin)
            q2, resid = leg_ik(m, qpos, dc, rv)
            fe2, sp2, _, _ = track(m, q2)
            lift = np.zeros((len(qpos), 2))
            for _ in range(2):  # swing clearance on the edited motion, then re-solve
                lift = swing_lift(new, fe2, foot_r, stance, dt, lift)
                q2, resid = leg_ik(m, qpos, dc, rv, lift)
                fe2, sp2, _, _ = track(m, q2)
            stance2 = np.zeros_like(stance)
            for i in range(2):
                for e in range(2):
                    stance2[:, i, e] = stance_mask(fe2[:, i, e], dt, 0.12, 6) | stance[:, i, e]
            sc1 = scores(new, fe2, sp2, foot_r, shin_r, stance2, resid)
            sc1.update(joint_speed(qpos, q2, dt))
            print(f"{clip:42s} {desc}\n    before: {fmt(sc0)}\n    after : {fmt(sc1)}")
            rows.append({"clip": clip, "kind": kind, "terrain": desc, **{f"before_{k}": v for k, v in sc0.items()}, **{f"after_{k}": v for k, v in sc1.items()}})
            lo2 = np.minimum(fe2[..., :2].reshape(-1, 2).min(0), q2[:, :2].min(0)) - 0.6
            hi2 = np.maximum(fe2[..., :2].reshape(-1, 2).max(0), q2[:, :2].max(0)) + 0.6
            H, origin = new.grid(lo2, hi2)
            np.savez(save / f"dash_{clip}.npz", qpos=q2, fps=FPS_IN / STEP, c=new.c, d=new.d, edges=new.edges, heights=new.heights, kind=kind)
            if render:
                jn = list(dat["joint_names"])
                idx = {n: i for i, n in enumerate(jn)}
                bones = [(idx[a], idx[b]) for a, b in dat["skeleton"] if a in HUMAN_BONES and b in HUMAN_BONES]
                # The human overlay gets the same torso shift, so it stays on the robot.
                mocap = dat["mocap"][::STEP] + dc.mean(1)[:, None]
                render_clip(f"dash_{clip}", f"edited: {desc}", q2, mocap, bones, H, origin, new.d, f"after: {fmt(sc1)[:70]}", out / "clips" / f"{clip}.mp4")
        except Exception as e:  # keep going; the clip is reported as failed
            print(f"{f.stem}: FAILED {type(e).__name__}: {e}")
            rows.append({"clip": f.stem.removeprefix("dash_"), "kind": "failed", "terrain": repr(e)[:80]})
    if rows:
        name = out / f"scores{('_' + only) if only else ''}{f'_{shard:02d}' if nshards > 1 else ''}.csv"
        with open(name, "w") as fh:
            w = csv.DictWriter(fh, fieldnames=list(max(rows, key=len)), restval="")
            w.writeheader()
            w.writerows(rows)
        print("wrote", name)


if __name__ == "__main__":
    tyro.cli(main)
