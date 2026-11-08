"""Retarget PARC motion-terrain clips to Dash.

1. Froude scaling: positions and terrain scaled by s = Dash leg / human leg, time by sqrt(s), so
   ballistic phases still fall at g and a foot on a step stays on the (scaled) step.
2. Per-frame Gauss-Newton IK on the Dash model (MuJoCo Jacobians), warm started from the previous frame:
   - hips / knees / ankles / shoulders / elbows / hands to the scaled human joints, legs and arms
     shifted sideways by the hip / shoulder width difference;
   - foot capsule centre and heel->toe direction to the scaled human sole;
   - torso orientation to the human pelvis and chest;
   - stance (PARC's foot contact labels): the foot is locked in xy where it landed, and each capsule end
     the human has on the ground sits on the terrain;
   - no point of the feet, shins, thighs or hips below the terrain (one-sided);
   - small step regulariser; joint limits by clipping.

  python parc_retarget.py out_dir clip.pkl [clip.pkl ...]
"""
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parc_io import fk, load, qrot, skeleton  # noqa: E402

DASH_XML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../src/supermimic/assets/dash/xmls/dash_v1.xml")
HUMAN_LEG = 0.421546 + 0.409870
HUMAN_HIP_Y, HUMAN_SHOULDER_Y = 0.084887, 0.18311
HEEL, TOE = np.array([-0.0435, 0, -0.05]), np.array([0.1335, 0, -0.05])  # human foot box sole ends
SOLE_ON = 0.03  # a human sole end (human scale) this close to the terrain is "down"


class Terrain:
    def __init__(self, hf, min_point, dx):
        self.hf, self.p0, self.dx = hf, np.asarray(min_point, float), float(dx)

    def h(self, xy, r=0.0):
        """Highest terrain within r of xy (blocky cells centred on p0 + i dx)."""
        xy = np.atleast_2d(xy)
        best = np.full(len(xy), -np.inf)
        offs = [(0, 0)] if r == 0 else [(0, 0), (r, 0), (-r, 0), (0, r), (0, -r), (.7 * r, .7 * r), (-.7 * r, .7 * r), (.7 * r, -.7 * r), (-.7 * r, -.7 * r)]
        for ox, oy in offs:
            ij = np.round((xy + [ox, oy] - self.p0) / self.dx).astype(int)
            i = np.clip(ij[:, 0], 0, self.hf.shape[0] - 1); j = np.clip(ij[:, 1], 0, self.hf.shape[1] - 1)
            best = np.maximum(best, self.hf[i, j])
        return best

    def boxes(self, floor=None):
        """Merge equal-height cells into boxes (cx, cy, cz, hx, hy, hz) from z = floor up."""
        hf = self.hf
        floor = hf.min() - 0.5 if floor is None else floor
        used = np.zeros(hf.shape, bool); out = []
        for i in range(hf.shape[0]):
            for j in range(hf.shape[1]):
                if used[i, j] or hf[i, j] <= floor + 1e-6:
                    continue
                h = hf[i, j]; j2 = j
                while j2 + 1 < hf.shape[1] and not used[i, j2 + 1] and abs(hf[i, j2 + 1] - h) < 1e-6:
                    j2 += 1
                i2 = i
                while i2 + 1 < hf.shape[0] and np.all(np.abs(hf[i2 + 1, j:j2 + 1] - h) < 1e-6) and not used[i2 + 1, j:j2 + 1].any():
                    i2 += 1
                used[i:i2 + 1, j:j2 + 1] = True
                x0, x1 = self.p0[0] + (i - .5) * self.dx, self.p0[0] + (i2 + .5) * self.dx
                y0, y1 = self.p0[1] + (j - .5) * self.dx, self.p0[1] + (j2 + .5) * self.dx
                out.append([(x0 + x1) / 2, (y0 + y1) / 2, (h + floor) / 2, (x1 - x0) / 2, (y1 - y0) / 2, (h - floor) / 2])
        return np.array(out)


def quat_err(q_cur, q_tgt):
    """3-vector (world frame) rotating q_cur to q_tgt (wxyz)."""
    v = np.zeros(3)
    mujoco.mju_subQuat(v, q_tgt, q_cur)  # in q_cur's frame
    R = np.zeros(9); mujoco.mju_quat2Mat(R, q_cur)
    return R.reshape(3, 3) @ v


class Dash:
    def __init__(self, boxes=()):
        spec = mujoco.MjSpec.from_file(DASH_XML)
        for i, b in enumerate(boxes):
            spec.worldbody.add_geom(name=f"terrain_{i}", type=mujoco.mjtGeom.mjGEOM_BOX, pos=b[:3], size=b[3:])
        self.m = m = spec.compile()
        self.d = mujoco.MjData(self.m)
        self.terrain = {i for i in range(m.ngeom) if m.geom(i).name.startswith("terrain_")}
        self.bid = {m.body(i).name: i for i in range(m.nbody)}
        self.gid = {m.geom(i).name: i for i in range(m.ngeom)}
        mujoco.mj_forward(m, self.d)
        xp = self.d.xpos
        self.leg = np.linalg.norm(xp[self.bid["l_upper_leg"]] - xp[self.bid["l_lower_leg"]]) + np.linalg.norm(xp[self.bid["l_lower_leg"]] - xp[self.bid["l_foot"]])
        self.hip_y, self.shoulder_y = xp[self.bid["l_upper_leg"]][1], xp[self.bid["l_upper_arm"]][1]
        g = self.gid["l_foot_collision"]
        self.foot_r, self.foot_hl = m.geom_size[g][0], m.geom_size[g][1]
        self.jlo, self.jhi = m.jnt_range[1:, 0].copy(), m.jnt_range[1:, 1].copy()
        self.jacp, self.jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        self.foot_bodies = {self.bid["l_foot"], self.bid["r_foot"]}
        self.feet_pen = True

    def capsule(self, name):
        g = self.gid[name]
        ax = self.d.geom_xmat[g].reshape(3, 3)[:, 2]
        return self.d.geom_xpos[g].copy(), ax, self.m.geom_size[g][0], self.m.geom_size[g][1], self.m.geom_bodyid[g]

    def jp(self, p, body):
        mujoco.mj_jac(self.m, self.d, self.jacp, None, p, body)
        return self.jacp.copy()

    def jr(self, body):
        mujoco.mj_jacBody(self.m, self.d, None, self.jacr, body)
        return self.jacr.copy()


def human_targets(f, s, dash):
    m, t = load(f)
    names, pos, rot = fk(m)
    T = len(pos)
    ix = {n: i for i, n in enumerate(names)}
    P = pos * s
    terr = Terrain(t["hf"] * s, np.asarray(t["min_point"]) * s, t["dx"] * s)
    # pelvis yaw frame: lateral axis for the width corrections
    fwd = qrot(rot[:, 0], np.broadcast_to([1.0, 0, 0], (T, 3))); fwd[:, 2] = 0
    fwd /= np.linalg.norm(fwd, axis=1, keepdims=True) + 1e-9
    lat = np.stack([-fwd[:, 1], fwd[:, 0], np.zeros(T)], 1)  # +y (left) when facing +x
    dleg, darm = dash.hip_y - s * HUMAN_HIP_Y, dash.shoulder_y - s * HUMAN_SHOULDER_Y
    tg = {"lat": lat}
    for side, sg, hs in (("l", 1, "left"), ("r", -1, "right")):
        # hips at Dash's width; the feet keep the human footholds (the terrain was built around them),
        # knees in between
        off = sg * dleg * lat
        tg[f"{side}_hip"] = P[:, ix[f"{hs}_thigh"]] + off
        tg[f"{side}_knee"] = P[:, ix[f"{hs}_shin"]] + off / 2
        off = 0 * off
        tg[f"{side}_ankle"] = P[:, ix[f"{hs}_foot"]] + off
        fr = rot[:, ix[f"{hs}_foot"]]
        heel = (pos[:, ix[f"{hs}_foot"]] + qrot(fr, np.broadcast_to(HEEL, (T, 3)))) * s + off
        toe = (pos[:, ix[f"{hs}_foot"]] + qrot(fr, np.broadcast_to(TOE, (T, 3)))) * s + off
        u = toe - heel; u /= np.linalg.norm(u, axis=1, keepdims=True)
        tg[f"{side}_sole_mid"], tg[f"{side}_sole_dir"] = (heel + toe) / 2, u
        hh = terr.h(heel[:, :2] - off[:, :2]); ht = terr.h(toe[:, :2] - off[:, :2])  # human's own footing
        # a sole end is down when within SOLE_ON of the terrain and slower than 0.5 m/s (human scale; the
        # point, not the ankle, which swings over the toe at push-off)
        spd = lambda x: np.r_[0.0, np.linalg.norm(np.diff(x, axis=0), axis=1)] * m["fps"] / s
        tg[f"{side}_heel_gap"] = np.clip(heel[:, 2] - hh, 0, None); tg[f"{side}_toe_gap"] = np.clip(toe[:, 2] - ht, 0, None)
        tg[f"{side}_heel_h"], tg[f"{side}_toe_h"] = hh, ht
        tg[f"{side}_heel_down"] = (heel[:, 2] - hh < SOLE_ON * s) & (spd(heel) < 0.5)
        tg[f"{side}_toe_down"] = (toe[:, 2] - ht < SOLE_ON * s) & (spd(toe) < 0.5)
        # stance from the human kinematics (PARC's contact labels are offset on some clips)
        tg[f"{side}_contact"] = tg[f"{side}_heel_down"] | tg[f"{side}_toe_down"]
        aoff = sg * darm * lat
        tg[f"{side}_shoulder"] = P[:, ix[f"{hs}_upper_arm"]] + aoff
        tg[f"{side}_elbow"] = P[:, ix[f"{hs}_lower_arm"]] + aoff
        tg[f"{side}_hand"] = P[:, ix[f"{hs}_hand"]] + aoff
    wxyz = lambda q: np.concatenate([q[..., 3:], q[..., :3]], -1)
    tg["pelvis_q"], tg["chest_q"] = wxyz(rot[:, ix["pelvis"]]), wxyz(rot[:, ix["torso"]])
    other = [i for n, i in ix.items() if "foot" not in n]
    tg["other_contact"] = m["body_contacts"][:, other].max(1) > 0.5
    return tg, terr, T, float(m["fps"]) / np.sqrt(s)


W = dict(hip=3.0, knee=2.0, ankle=2.0, sole=6.0, sole_dir=3.0, shoulder=1.0, elbow=0.7, hand=0.7, rot=1.5,
         lock=30.0, ground=30.0, pen=100.0, smooth=0.1)


def residuals(dash, tg, t, terr, lock, q_prev, jac=True, pen=True):
    """Weighted residuals (and Jacobian) at dash.d.qpos."""
    m, d = dash.m, dash.d
    mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
    R, J = [], []
    jp = (lambda p, b: dash.jp(p, b)) if jac else (lambda p, b: np.zeros((3, m.nv)))
    def add(w, r, j):
        R.append(w * np.atleast_1d(r))
        if jac:
            J.append(w * np.atleast_2d(j))
    for side in "lr":
        for key, body, w in (("hip", "upper_leg", W["hip"]), ("knee", "lower_leg", W["knee"]), ("ankle", "foot", W["ankle"]),
                             ("shoulder", "upper_arm", W["shoulder"]), ("elbow", "lower_arm", W["elbow"])):
            b = dash.bid[f"{side}_{body}"]
            add(w, d.xpos[b] - tg[f"{side}_{key}"][t], jp(d.xpos[b], b))
        c, ax, r, hl, b = dash.capsule(f"{side}_lower_arm_collision")  # hand: far end of the forearm
        e = c + hl * ax if np.dot(ax, c - d.xpos[b]) > 0 else c - hl * ax
        add(W["hand"], e - tg[f"{side}_hand"][t], jp(e, b))
        c, ax, r, hl, b = dash.capsule(f"{side}_foot_collision")
        if np.dot(ax, d.xmat[b].reshape(3, 3)[:, 0]) < 0:  # heel -> toe along the foot's +x
            ax = -ax
        heel, toe = c - hl * ax, c + hl * ax
        Jh, Jt, Jc = jp(heel, b), jp(toe, b), jp(c, b)
        add(W["sole"], c - (tg[f"{side}_sole_mid"][t] + [0, 0, r]), Jc)
        add(W["sole_dir"], (toe - heel) - 2 * hl * tg[f"{side}_sole_dir"][t], Jt - Jh)
        for en, end, Je in (("heel", heel, Jh), ("toe", toe, Jt)):
            down = tg[f"{side}_{en}_down"][t]
            if down:  # at the human's height above the step the human stands on (a heel starting to lift
                # keeps lifting; the human's step, not the one under Dash's end, so an end on an edge is smooth)
                add(W["ground"], end[2] - r - tg[f"{side}_{en}_h"][t] - tg[f"{side}_{en}_gap"][t], Je[2])
                if (side, en) in lock:  # no sliding: the end stays where it touched down
                    add(W["lock"], end[:2] - lock[(side, en)], Je[:2])
    if pen:  # every robot geom out of the terrain boxes, along the contact normal
        mujoco.mj_collision(m, d)
        for c in d.contact[:d.ncon]:
            if c.dist >= 0 or (c.geom1 in dash.terrain) == (c.geom2 in dash.terrain):
                continue
            rob, sg = (c.geom2, 1.0) if c.geom1 in dash.terrain else (c.geom1, -1.0)
            if not (dash.feet_pen or m.geom_bodyid[rob] not in dash.foot_bodies):
                continue
            add(W["pen"], c.dist, (sg * c.frame[:3] @ jp(c.pos, m.geom_bodyid[rob])) if jac else None)
    tq = d.xquat[dash.bid["torso"]].copy()
    jr = dash.jr(dash.bid["torso"]) if jac else None
    add(W["rot"], -quat_err(tq, tg["pelvis_q"][t]), jr)
    add(W["rot"], -quat_err(tq, tg["chest_q"][t]), jr)
    if q_prev is not None:  # temporal smoothness on the joints
        Js = np.zeros((m.nv - 6, m.nv)); Js[:, 6:] = np.eye(m.nv - 6)
        add(W["smooth"], d.qpos[7:] - q_prev[7:], Js)
    r = np.concatenate(R)
    return (r, np.vstack(J)) if jac else r


def step(A, g, lam, qpos, dash):
    """Damped step; joints at a limit whose step points out of range are held (active set)."""
    nv = len(g); free = np.ones(nv, bool)
    for _ in range(3):
        Af = A[np.ix_(free, free)]
        dq = np.zeros(nv)
        dq[free] = -np.linalg.solve(Af + lam * np.diag(np.diag(Af) + 1e-6), g[free])
        q = qpos[7:]
        out = ((q <= dash.jlo + 1e-6) & (dq[6:] < 0)) | ((q >= dash.jhi - 1e-6) & (dq[6:] > 0))
        if not (out & free[6:]).any():
            break
        free[6:] &= ~out
    return dq


def solve_frame(dash, tg, t, terr, lock, q_prev, n_iter=15, pen=True):
    """Levenberg-Marquardt: a step is kept only if it lowers the cost; joint limits by clipping."""
    m, d = dash.m, dash.d
    lam = 1e-2
    r, Jm = residuals(dash, tg, t, terr, lock, q_prev, pen=pen)
    cost = r @ r
    for it in range(n_iter):
        g = Jm.T @ r; A = Jm.T @ Jm
        q0 = d.qpos.copy()
        while True:
            dq = step(A, g, lam, d.qpos, dash)
            mujoco.mj_integratePos(m, d.qpos, dq, 1.0)
            d.qpos[7:] = np.clip(d.qpos[7:], dash.jlo, dash.jhi)
            r_new = residuals(dash, tg, t, terr, lock, q_prev, jac=False, pen=pen)
            c_new = r_new @ r_new
            if c_new < cost or lam > 1e6:
                break
            d.qpos[:] = q0; lam *= 10
        if c_new >= cost:
            d.qpos[:] = q0
            break
        lam = max(lam / 3, 1e-6)
        done = cost - c_new < 1e-7 * (1 + cost)
        r, Jm = residuals(dash, tg, t, terr, lock, q_prev, pen=pen); cost = r @ r
        if done:
            break
    return float(np.sqrt(cost / len(r)))


def retarget(f):
    dash0 = Dash()
    s = dash0.leg / HUMAN_LEG
    tg, terr, T, fps = human_targets(f, s, dash0)
    dash = Dash(terr.boxes())
    m, d = dash.m, dash.d
    d.qpos[:] = m.qpos0
    hip_mid = (tg["l_hip"][0] + tg["r_hip"][0]) / 2
    d.qpos[:3] = hip_mid + [0, 0, 0.108]
    d.qpos[3:7] = tg["pelvis_q"][0]
    Q = np.zeros((T, m.nq)); err = np.zeros(T)
    lock = {}
    for t in range(T):
        for key in list(lock):
            if not tg[f"{key[0]}_{key[1]}_down"][t]:
                lock.pop(key)
        q_prev = Q[t - 1].copy() if t else None
        # free solve first (a warm start can sit behind a step edge the target has passed), then terrain
        solve_frame(dash, tg, t, terr, lock, q_prev, n_iter=60 if t == 0 else 15, pen=False)
        err[t] = solve_frame(dash, tg, t, terr, lock, q_prev, n_iter=60 if t == 0 else 15)
        new = False
        mujoco.mj_kinematics(m, d)
        for side in "lr":  # lock each sole end where it touched down
            c, ax, r, hl, b = dash.capsule(f"{side}_foot_collision")
            if np.dot(ax, d.xmat[b].reshape(3, 3)[:, 0]) < 0:
                ax = -ax
            for en, sg in (("heel", -1), ("toe", 1)):
                if tg[f"{side}_{en}_down"][t] and (side, en) not in lock:
                    lock[(side, en)] = (c + sg * hl * ax)[:2].copy(); new = True
        if new:
            err[t] = solve_frame(dash, tg, t, terr, lock, q_prev)
        Q[t] = d.qpos
    return dict(qpos=Q, fps=fps, hf=terr.hf, min_point=terr.p0, dx=terr.dx, scale=s, ik_err=err,
                contact=np.stack([tg["l_contact"], tg["r_contact"]], 1), other_contact=tg["other_contact"],
                down=np.stack([np.stack([tg[f"{sd}_heel_down"], tg[f"{sd}_toe_down"]], 1) for sd in "lr"], 1))


def main():
    out = sys.argv[1]; os.makedirs(out, exist_ok=True)
    for f in sys.argv[2:]:
        r = retarget(f)
        name = os.path.basename(f).replace(".pkl", "")
        np.savez(os.path.join(out, f"dash_{name}.npz"), source=f, **r)
        print(f"{name}: {len(r['qpos'])} frames, scale {r['scale']:.3f}, fps {r['fps']:.1f}, IK rms med {np.median(r['ik_err']):.3f} max {r['ik_err'].max():.3f}")


if __name__ == "__main__":
    main()
