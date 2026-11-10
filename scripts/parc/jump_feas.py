"""Can Dash do the jumps in a retarget? Per flight phase (both feet off for >= --min-flight s): flight time, CoM
travel, the takeoff speed a ballistic CoM needs, and the leg torques around takeoff and landing from inverse
dynamics (contacts solved as in qa_parc.support) relative to Dash's effort limits.

  python jump_feas.py out.csv retarget.npz [...] [--min-flight 0.12] [--window 0.2] [--effort hip=80 knee=120 ankle=45]

One row per flight, with the clip-wide torque and joint speed ratios repeated. ratio = |tau| / effort limit
(dash.py's, or --effort per motor group: hip covers hip yaw/roll/pitch and shoulder pitch); > 1 is beyond the
motors. az_fit_g / ah_fit_g: CoM vertical / horizontal acceleration in the air from a parabola fit, in g
(ballistic: -1 / 0). ghost_flight (inverse dynamics, smoothed over 7 frames) bleeds stance force into short
flights; prefer az_fit_g. Also reads flat-ground BONES npz (contact from sole height).
"""
import argparse
import csv
import os
import re
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parc_retarget import DASH_XML, Terrain
from qa_parc import support
from supermimic.assets.dash.dash import _ACTUATOR_GROUPS

G = 9.81
LEG = ("hip_yaw", "hip_roll", "hip_pitch", "knee_pitch", "ankle_pitch")


def runs(mask):
    """[start, end) of each run of True."""
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def load(f, m):
    """PARC retarget (qpos, contact, terrain) or a flat-ground tracking clip (BONES npz: body/joint arrays;
    a foot is in contact when a sole end is within 3 cm of the ground)."""
    z = np.load(f, allow_pickle=True)
    if "qpos" in z:
        return z["qpos"], 1.0 / float(z["fps"]), z["contact"].astype(bool), Terrain(z["hf"], z["min_point"], z["dx"])
    Q = np.concatenate([z["body_pos_w"][:, 0], z["body_quat_w"][:, 0], z["joint_pos"]], 1)
    d = mujoco.MjData(m); c = np.zeros((len(Q), 2), bool)
    for t, q in enumerate(Q):
        d.qpos[:] = q; mujoco.mj_kinematics(m, d)
        for k, s in enumerate("lr"):
            g = m.geom(f"{s}_foot_collision").id; r, hl = m.geom_size[g][:2]; ax = d.geom_xmat[g].reshape(3, 3)[:, 2]
            c[t, k] = min((d.geom_xpos[g] + sg * hl * ax)[2] - r for sg in (-1, 1)) < 0.03
    return Q, 1.0 / float(np.ravel(z["fps"])[0]), c, Terrain(np.zeros((2, 2)), [-100.0, -100.0], 200.0)


GROUP = {"hip": 0, "knee": 1, "ankle": 2, "arm": 3}  # --effort names -> _ACTUATOR_GROUPS index


def clip_row(f, m, min_flight, window, effort=None):
    Q, dt, c, terr = load(f, m)
    tau = []
    sc = support(m, Q, terr, dt, tau)
    tau = np.abs(np.array(tau))
    names = [m.joint(int(np.flatnonzero(m.jnt_dofadr == i)[0])).name for i in range(6, m.nv)]
    eff = [g[3] for g in _ACTUATOR_GROUPS]
    for k, v in (effort or {}).items():
        eff[GROUP[k]] = v
    lim = np.array([next(e for g, e in zip(_ACTUATOR_GROUPS, eff) if any(re.fullmatch(x, n) for x in g[0])) for n in names])
    ratio = tau / lim
    grp = {g: [i for i, n in enumerate(names) if n.endswith(g)] for g in LEG}
    leg = np.concatenate(list(grp.values()))
    d = mujoco.MjData(m); com = np.zeros((len(Q), 3))
    for t, q in enumerate(Q):
        d.qpos[:] = q; mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d); com[t] = d.subtree_com[1]
    qd = np.abs(np.diff(Q[:, 7:], axis=0)) / dt
    w = max(1, round(window / dt))
    out = []
    for a, b in runs(~c.any(1)):
        if a == 0 or b >= len(Q) or (b - a) * dt < min_flight:
            continue
        t0, t1 = a - 1, b  # last frame in contact, first frame back in contact
        T = (t1 - t0) * dt
        dxy = np.linalg.norm(com[t1, :2] - com[t0, :2]); dz = com[t1, 2] - com[t0, 2]
        vz = dz / T + G * T / 2; vh = dxy / T
        r = dict(flight_s=round(T, 3), com_travel=round(dxy, 3), com_dz=round(dz, 3), apex_rise=round(max(vz, 0) ** 2 / (2 * G), 3),
                 takeoff_v=round(float(np.hypot(vz, vh)), 2), takeoff_vz=round(vz, 2),
                 ghost_flight=round(float(sc["ghost"][a:b].mean()), 2), takeoff_frame=int(t0), landing_frame=int(t1))
        if b - a >= 4:  # CoM vertical acceleration in the air from a parabola fit (no filtering): -g if ballistic
            ts = np.arange(a, b) * dt
            r["az_fit_g"] = round(float(2 * np.polyfit(ts, com[a:b, 2], 2)[0] / G), 2)
            r["ah_fit_g"] = round(float(np.linalg.norm([2 * np.polyfit(ts, com[a:b, k], 2)[0] for k in (0, 1)]) / G), 2)
        for tag, sl in (("to", slice(max(t0 - w, 0), t0 + 1)), ("land", slice(t1, min(t1 + w + 1, len(Q))))):
            for g, ks in grp.items():
                if g in ("knee_pitch", "hip_pitch", "ankle_pitch"):
                    r[f"{tag}_{g}"] = round(float(ratio[sl][:, ks].max()), 2)
            r[f"{tag}_leg"] = round(float(ratio[sl][:, leg].max()), 2)
        out.append(r)
    row = dict(clip=os.path.relpath(f, "/media/shangqunyu/Dash/parc_dash_30cm") if f.startswith("/media") else f,
               seconds=round(len(Q) * dt, 2), n_flights=sum(1 for a, b in runs(~c.any(1)) if (b - a) * dt >= min_flight and a > 0 and b < len(Q)),
               leg_ratio_p95=round(float(np.percentile(ratio[3:-3][:, leg], 95)), 2), leg_ratio_max=round(float(ratio[3:-3][:, leg].max()), 2),
               frac_over=round(float((ratio[3:-3][:, leg] > 1).any(1).mean()), 3), qd_max=round(float(qd.max()), 1),
               ghost_p95=round(sc["ghost_p95"], 2))
    return [{**row, **r} for r in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("clips", nargs="+")
    ap.add_argument("--min-flight", type=float, default=0.12); ap.add_argument("--window", type=float, default=0.2)
    ap.add_argument("--effort", nargs="*", default=[], help="override effort limits (Nm), e.g. hip=80 knee=120 ankle=45")
    a = ap.parse_args()
    effort = {k: float(v) for k, v in (e.split("=") for e in a.effort)}
    m = mujoco.MjModel.from_xml_path(DASH_XML)
    rows = []  # clips without a flight add no rows
    for f in a.clips:
        try:
            rows += clip_row(f, m, a.min_flight, a.window, effort)
        except Exception as e:  # keep the shard going
            print("failed", f, e, file=sys.stderr)
    with open(a.out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    print("wrote", a.out, len(rows))


if __name__ == "__main__":
    main()
