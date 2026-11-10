"""QA of Dash retargets of PARC clips: penetration (MuJoCo contacts vs terrain boxes), stance foot slide,
unsupported ("ghost") force and joint torques from inverse dynamics, joint speeds.
  python qa_parc.py out.csv retarget.npz [...]"""
import csv, os, sys
import mujoco, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parc_retarget import DASH_XML, Terrain

def model(boxes):
    spec = mujoco.MjSpec.from_file(DASH_XML)
    for i, b in enumerate(boxes):
        spec.worldbody.add_geom(name=f"terrain_{i}", type=mujoco.mjtGeom.mjGEOM_BOX, pos=b[:3], size=b[3:])
    return spec.compile()

def support(m, Q, terr, dt, tau_out, mu=0.6, frac=0.75):
    """Inverse dynamics with the two sole ends of each foot as contacts (linearized cone), active when the
    end is within 2 cm of the terrain and the end point moves < 0.5 m/s. Ghost = root force the contacts
    cannot supply, in body weights."""
    import re
    from scipy.optimize import nnls
    from scipy.signal import savgol_filter
    from supermimic.assets.dash.dash import _ACTUATOR_GROUPS
    nv, T = m.nv, len(Q)
    v = np.zeros((T, nv))
    for t in range(T):
        a, b = max(t - 1, 0), min(t + 1, T - 1)
        mujoco.mj_differentiatePos(m, v[t], (b - a) * dt, Q[a], Q[b])
    qvel = savgol_filter(v, 7, 2, axis=0); qacc = savgol_filter(v, 7, 2, deriv=1, delta=dt, axis=0)
    jdof = np.arange(6, nv)
    names = [m.joint(int(np.flatnonzero(m.jnt_dofadr == i)[0])).name for i in jdof]
    lim = np.array([next(g[3] for g in _ACTUATOR_GROUPS if any(re.fullmatch(e, n) for e in g[0])) for n in names])
    feet = [m.geom(f"{s}_foot_collision").id for s in "lr"]
    E = np.array([[mu, 0, 1], [-mu, 0, 1], [0, mu, 1], [0, -mu, 1]]).T
    weight = 9.81 * m.body_subtreemass[1]
    m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
    d = mujoco.MjData(m); jacp = np.zeros((3, nv)); ghost = np.zeros(T)
    for t in range(T):
        d.qpos[:], d.qvel[:] = Q[t], qvel[t]
        mujoco.mj_forward(m, d); d.qacc[:] = qacc[t]; mujoco.mj_inverse(m, d)
        b = d.qfrc_inverse; cols = []
        for g in feet:
            r, hl = m.geom_size[g][:2]; ax = d.geom_xmat[g].reshape(3, 3)[:, 2]
            for sgn in (-1, 1):
                p = d.geom_xpos[g] + sgn * hl * ax - np.array([0.0, 0.0, r])
                mujoco.mj_jac(m, d, jacp, None, p, m.geom_bodyid[g])
                if p[2] - terr.h(p[None, :2])[0] < 0.02 and np.linalg.norm(jacp @ qvel[t]) < 0.5:
                    cols.append(jacp.T @ E)
        f = np.zeros(nv)
        if cols:
            A = np.concatenate(cols, 1)
            rows = np.vstack([1e3 * A[:6], A[jdof] / lim[:, None], 1e-3 * np.eye(A.shape[1])])
            rhs = np.concatenate([1e3 * b[:6], b[jdof] / lim, np.zeros(A.shape[1])])
            f = A @ nnls(rows, rhs, maxiter=2000)[0]
        ghost[t] = np.linalg.norm(b[:3] - f[:3]) / weight
        tau_out.append(b[jdof] - f[jdof])
    run = best = 0
    for g in ghost[3:-3] > frac:
        run = run + 1 if g else 0; best = max(best, run)
    return {"ghost_run": best * dt, "ghost_p95": float(np.percentile(ghost[3:-3], 95)), "ghost": ghost}


GROUPS = {"hip_roll": 50, "hip_pitch": 50, "knee_pitch": 80, "ankle_pitch": 35}


def main():
    rows = []
    for f in sys.argv[2:]:
        z = np.load(f, allow_pickle=True)
        terr = Terrain(z["hf"], z["min_point"], z["dx"])
        m = model(terr.boxes()); d = mujoco.MjData(m)
        tg = {m.geom(i).id for i in range(m.ngeom) if m.geom(i).name.startswith("terrain_")}
        Q, dt = z["qpos"], 1.0 / float(z["fps"])
        pen = np.zeros(len(Q)); slide = []
        feet = [m.geom(f"{s}_foot_collision").id for s in "lr"]
        prev = None
        for t, q in enumerate(Q):
            d.qpos[:] = q; mujoco.mj_forward(m, d)
            for c in d.contact[:d.ncon]:
                if (c.geom1 in tg) != (c.geom2 in tg):
                    pen[t] = max(pen[t], -c.dist)
            fc = np.array([d.geom_xpos[g][:2] for g in feet])
            if prev is not None:
                for k in range(2):
                    if z["contact"][t, k] and z["contact"][t - 1, k]:
                        slide.append(np.linalg.norm(fc[k] - prev[k]) / dt)
            prev = fc
        tau = []
        m0 = mujoco.MjModel.from_xml_path(DASH_XML)
        sc = support(m0, Q, terr, dt, tau)
        tau = np.abs(np.array(tau))
        jn = [m0.joint(int(np.flatnonzero(m0.jnt_dofadr == i)[0])).name for i in range(6, m0.nv)]
        qd = np.abs(np.diff(Q[:, 7:], axis=0)) / dt
        r = dict(clip=os.path.basename(f), seconds=round(len(Q) * dt, 2), pen_max=round(pen.max(), 3), pen_p95=round(np.percentile(pen, 95), 3),
                 slide_p95=round(np.percentile(slide, 95), 3) if slide else 0, ghost_run=round(sc["ghost_run"], 3), ghost_p95=round(sc["ghost_p95"], 2),
                 qd_max=round(qd.max(), 1), ik_err_p95=round(float(np.percentile(z["ik_err"], 95)), 3))
        for g, lim in GROUPS.items():
            ks = [i for i, n in enumerate(jn) if n.endswith(g)]
            a = tau[3:-3, ks].ravel()
            r[f"{g}_p95"] = round(np.percentile(a, 95)); r[f"{g}_max"] = round(a.max())
        rows.append(r); print(r)
    with open(sys.argv[1], "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
