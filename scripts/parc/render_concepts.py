"""Concept images of Dash on a PARC retarget for slides: perception (height scan, head camera frustum and its
depth image), decision (candidate and chosen footholds, swing paths, step timing) and motion control (ground
reaction forces from inverse dynamics, friction cones).

  python render_concepts.py out_dir retarget.npz [--see 24] [--plan 42] [--force 50] [--width 2400 --height 1150]
      [--see-ahead 1.4] [--depth-range 0.5 3.4] [--cut] [--plain-ground] [--inset left] [--aim-level]
      [--see-view -26 4.9 35 0.65] [--plan-view -24 3.3 15 0.5] [--force-view -14 3.2 25 0.1]   (elevation distance toward ahead)

--see/--plan/--force: the clip frame for each image (Dash fps).
"""
import argparse
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
import matplotlib
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import nnls
from scipy.signal import savgol_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parc_retarget import Terrain
from render_hq import dash_model
from render_strobe import section_cut

ACCENT = np.array([0.48, 0.10, 0.20, 1.0])  # maroon, the slide accent
CAND = np.array([0.20, 0.55, 0.45, 0.9])  # candidate footholds
INK = (40, 44, 52)


def add(scn, typ, size, pos, mat=np.eye(3), rgba=ACCENT):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, typ, np.asarray(size, float), np.asarray(pos, float), np.asarray(mat, float).ravel(), np.asarray(rgba, np.float32))
    g.emission, g.specular = 0.3, 0.1
    scn.ngeom += 1
    return g


def line(scn, a, b, width, rgba, typ=mujoco.mjtGeom.mjGEOM_CAPSULE):
    if scn.ngeom >= scn.maxgeom:
        return
    g = add(scn, typ, [0, 0, 0], [0, 0, 0], rgba=rgba)
    mujoco.mjv_connector(g, typ, width, np.asarray(a, float), np.asarray(b, float))
    g.emission = 0.3


def yaw_frame(q):
    w, x, y, z = q[3:7]
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    f = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    return f, np.array([-f[1], f[0], 0.0]), yaw


def project(glcam, p, w, h):
    """World point -> pixel for MuJoCo's OpenGL camera (scene.camera[0])."""
    pos, fwd, up = np.array(glcam.pos), np.array(glcam.forward), np.array(glcam.up)
    right = np.cross(fwd, up)
    v = np.asarray(p) - pos
    x, y, z = v @ right, v @ up, v @ fwd
    n = glcam.frustum_near
    hw = glcam.frustum_width or (glcam.frustum_top - glcam.frustum_bottom) / 2 * w / h  # 0: set by the viewport aspect
    l, r = glcam.frustum_center - hw, glcam.frustum_center + hw
    return (x * n / z - l) / (r - l) * w, (glcam.frustum_top - y * n / z) / (glcam.frustum_top - glcam.frustum_bottom) * h


class Scene:
    def __init__(self, z, w, h, cut=False, plain_ground=False):
        self.terr = Terrain(z["hf"], z["min_point"], z["dx"])
        self.boxes = self.terr.boxes()
        self.m_all = dash_model(self.boxes, w, h)  # the head camera sees all the terrain
        if cut:  # cutaway of the terrain between the side camera and the path, as in render_strobe
            self.boxes = section_cut(self.boxes, z["qpos"], self.m_all, mujoco.MjData(self.m_all))
        self.m = dash_model(self.boxes, w, h) if cut else self.m_all
        if plain_ground:  # untextured, darker ground below raised terrain: the walkway's edges and the gaps read
            g = self.m.geom("ground").id; self.m.geom_matid[g] = -1; self.m.geom_rgba[g] = [0.70, 0.73, 0.78, 1]
        self.m.vis.global_.fovy = 30.0
        self.d = mujoco.MjData(self.m)
        self.r = mujoco.Renderer(self.m, h, w, max_geom=20000)
        self.w, self.h = w, h
        self.full = mujoco.MjvOption(); self.full.geomgroup[:] = 0; self.full.geomgroup[[0, 1, 2]] = 1; self.full.sitegroup[:] = 0
        self.terrain_only = mujoco.MjvOption(); self.terrain_only.geomgroup[:] = 0; self.terrain_only.geomgroup[[0, 1]] = 1; self.terrain_only.sitegroup[:] = 0
        self.top = max(b[2] + b[5] for b in self.boxes)

    def pose(self, q):
        self.d.qpos[:] = q; mujoco.mj_forward(self.m, self.d)

    def cam(self, look, az, el, dist):
        c = mujoco.MjvCamera(); c.lookat[:], c.azimuth, c.elevation, c.distance = look, az, el, dist
        key = self.m.light("key").id; kd = self.m.light_dir[key] / np.linalg.norm(self.m.light_dir[key])
        self.m.light_pos[key] = np.asarray(look) - 20 * kd
        mujoco.mj_forward(self.m, self.d)  # the renderer reads d.light_xpos: a stale one leaves the shadow box elsewhere (acne)
        return c

    def shoot(self, cam, extra=None, opt=None, shadow=True):
        self.r.update_scene(self.d, cam, opt or self.full)
        self.r.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = 1
        self.r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = int(shadow)
        if extra:
            extra(self.r.scene)
        glcam = self.r.scene.camera[0]
        return Image.fromarray(self.r.render().copy()), glcam

    def surface(self, xy):
        return self.terr.h(xy)


def feet_now(m, d):
    out = {}
    for s in "lr":
        g = m.geom(f"{s}_foot_collision").id
        ax = d.geom_xmat[g].reshape(3, 3)[:, 2]; b = m.geom_bodyid[g]
        if np.dot(ax, d.xmat[b].reshape(3, 3)[:, 0]) < 0:
            ax = -ax
        out[s] = (d.geom_xpos[g].copy(), ax, m.geom_size[g][0], m.geom_size[g][1])
    return out


def side_cam(S, q, el, dist, ahead=0.6, toward=30.0, dz=-0.15):
    """Travel to screen right, the camera `toward` deg from the robot's left toward its front."""
    f, lft, yaw = yaw_frame(q)
    look = q[:3] + ahead * f + [0, 0, dz]
    # the camera sits at look - dist (cos az, sin az); on the robot's right: az = yaw + 90, rotated toward the front
    az = np.degrees(yaw) + 90 - toward
    return S.cam(look, az, el, dist)


def perception(S, Q, t, out, ahead=1.4, drange=(0.5, 3.4), view=(-26.0, 4.9, 35.0, 0.65), inset="left", aim_level=False):
    q = Q[t]; S.pose(q)
    f, lft, _ = yaw_frame(q)
    torso = S.d.xpos[S.m.body("torso").id]; R = S.d.xmat[S.m.body("torso").id].reshape(3, 3)
    eye = torso + R @ np.array([0.10, 0.0, 0.40])
    # aim at the terrain `ahead` m on, or (aim_level) at the height under the robot: over a gap the terrain aim jumps
    tgt = q[:3] * [1, 1, 0] + ahead * f + [0, 0, S.surface((q[:2] + (0 if aim_level else ahead) * f[:2])[None])[0]]
    fwd = tgt - eye; dist = np.linalg.norm(fwd); fwd /= dist
    fovy, aspect = 58.0, 1.6
    # height scan: a grid ahead of the robot, coloured by terrain height
    xs, ys = np.arange(0.15, 3.05, 0.065), np.arange(-0.39, 0.40, 0.065)
    P = np.array([q[:2] + a * f[:2] + b * lft[:2] for a in xs for b in ys])
    Z = S.surface(P)
    cmap = matplotlib.colormaps["viridis"]
    zn = (Z - Z.min()) / max(np.ptp(Z), 1e-6)
    up = np.cross(np.cross(fwd, [0, 0, 1]), fwd); up /= np.linalg.norm(up); rt = np.cross(fwd, up)
    th, tw = np.tan(np.radians(fovy / 2)), np.tan(np.radians(fovy / 2)) * aspect

    def extra(scn):
        for (x, y), zz, c in zip(P, Z, zn):
            add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.014, 0, 0], [x, y, zz + 0.012], rgba=[*cmap(0.15 + 0.8 * c)[:3], 1])
        add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.028, 0, 0], eye, rgba=ACCENT)  # head camera

    el, dist_, toward, ah = view
    cam = side_cam(S, q, el, dist_, ahead=ah, toward=toward)
    # the scan points cast no shadows (streaks on the walls): composite them from a shadowless pass
    img, _ = S.shoot(cam, lambda scn: add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.028, 0, 0], eye, rgba=ACCENT))
    pts, _ = S.shoot(cam, extra, shadow=False); bare, _ = S.shoot(cam, None, shadow=False)
    mask = (np.abs(np.asarray(pts, np.int16) - np.asarray(bare, np.int16)).max(-1) > 6)[..., None]
    img = Image.fromarray(np.where(mask, np.asarray(pts), np.asarray(img)))
    img.save(os.path.join(out, "perception_scene.png"))
    # the head camera's depth image (terrain only)
    hc = mujoco.MjvCamera(); hc.lookat[:] = tgt; hc.distance = dist
    hc.azimuth = np.degrees(np.arctan2(fwd[1], fwd[0])); hc.elevation = np.degrees(np.arcsin(fwd[2]))
    fovy0 = S.m_all.vis.global_.fovy; S.m_all.vis.global_.fovy = fovy
    dh, dw = S.h, int(S.h * aspect) if int(S.h * aspect) <= S.w else S.w
    rr = mujoco.Renderer(S.m_all, dh, dw); dd = mujoco.MjData(S.m_all); dd.qpos[:] = q; mujoco.mj_forward(S.m_all, dd)
    rr.update_scene(dd, hc, S.terrain_only); rr.enable_depth_rendering()
    depth = rr.render().copy(); rr.close()
    S.m_all.vis.global_.fovy = fovy0
    dn = np.clip((depth - drange[0]) / (drange[1] - drange[0]), 0, 1)
    gy, gx = np.gradient(depth)
    edge = np.clip(np.hypot(gx, gy) / (np.percentile(np.hypot(gx, gy), 99) + 1e-9), 0, 1)  # risers and step edges
    # shade by the surface normal from the depth (lit from above): walls read apart from the tops behind them
    th_ = np.tan(np.radians(fovy / 2))
    U, V = np.meshgrid((np.arange(dw) - dw / 2 + 0.5) / (dh / 2) * th_, (dh / 2 - np.arange(dh) - 0.5) / (dh / 2) * th_)
    P = np.dstack([U * depth, V * depth, depth])  # camera frame: x right, y up, z forward
    nrm = np.cross(np.gradient(P, axis=1), np.gradient(P, axis=0)); nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True) + 1e-12
    nrm *= np.where(nrm[..., 2:3] > 0, -1, 1)
    e = np.radians(hc.elevation); L = np.array([0, np.cos(e), np.sin(e)]) + [0.25, 0, -0.35]; L /= np.linalg.norm(L)
    shade = 0.6 + 0.4 * np.clip(nrm @ L, 0, 1)
    rgb = (matplotlib.colormaps["turbo"](1 - dn)[..., :3] * shade[..., None] * (1 - 0.55 * edge[..., None]) * 255).astype(np.uint8)
    dimg = Image.fromarray(rgb); dimg.save(os.path.join(out, "perception_depth.png"))
    # scene with the depth image inset (top right)
    comp = img.copy(); iw = int(S.w * 0.30); ih = int(iw / aspect)
    pad = int(S.w * 0.018); y0 = pad; x0 = pad if inset == "left" else S.w - pad - iw  # over empty floor or sky
    dr = ImageDraw.Draw(comp)
    dr.rectangle([x0 - 4, y0 - 4, x0 + iw + 3, y0 + ih + 3], fill=tuple(int(255 * c) for c in ACCENT[:3]))
    comp.paste(dimg.resize((iw, ih), Image.LANCZOS), (x0, y0))
    font = ImageFont.truetype("DejaVuSans-Bold.ttf", int(S.h * 0.032))
    dr.text((x0 + 14, y0 + 10), "depth", fill=(255, 255, 255), font=font)
    comp.save(os.path.join(out, "perception.png"))


def touchdowns(z, Q, m, t0, n):
    """The next n foot touchdowns after frame t0: (frame, side, foot capsule centre, heel->toe axis)."""
    d = mujoco.MjData(m); c = z["contact"]; ev = []
    for t in range(t0 + 1, len(Q)):
        for k, s in enumerate("lr"):
            if c[t, k] and not c[t - 1, k]:
                d.qpos[:] = Q[t]; mujoco.mj_kinematics(m, d)
                ft = feet_now(m, d)[s]; ev.append((t, s, ft[0], ft[1]))
    return ev[:n]


def decision(S, z, Q, t, out, n=2, candidates=False, view=(-24.0, 3.3, 15.0, 0.5), dz=-0.2):
    """The next n footholds (where and when) and each foot's swing path to its next one."""
    q = Q[t]; fps = float(z["fps"])
    ev = touchdowns(z, Q, S.m, t, n)
    last = {s: k for k, s, _, _ in reversed(ev)}  # each foot's path ends at its first shown touchdown
    d = mujoco.MjData(S.m); paths = {s: [] for s in last}
    for k in range(t, max(last.values()) + 1):
        d.qpos[:] = Q[k]; mujoco.mj_kinematics(S.m, d)
        for s, v in feet_now(S.m, d).items():
            if s in last and k <= last[s]:
                paths[s].append(v[0] - [0, 0, v[2]])
    S.pose(q)
    f, lft, _ = yaw_frame(q)
    cand = []
    if candidates:  # a grid on every surface ahead, kept off the edges (all four neighbours level)
        for a_ in np.arange(0.25, 2.9, 0.07):
            for b_ in np.arange(-0.30, 0.31, 0.07):
                p = q[:2] + a_ * f[:2] + b_ * lft[:2]
                h0 = S.surface(p[None])[0]
                nb = S.surface(np.array([p + e for e in ([0.06, 0], [-0.06, 0], [0, 0.06], [0, -0.06])]))
                if np.all(np.abs(nb - h0) < 1e-6):
                    cand.append([p[0], p[1], h0])
    labels = []

    def extra(scn):
        for p in cand:
            add(scn, mujoco.mjtGeom.mjGEOM_CYLINDER, [0.012, 0.002, 0], np.asarray(p) + [0, 0, 0.002], rgba=CAND)
        for s, pts in paths.items():
            pts = np.array(pts)
            for a_, b_ in zip(pts[:-1], pts[1:]):
                line(scn, a_, b_, 0.008, [0.35, 0.37, 0.42, 0.9])
        for k, s, c, ax in ev:
            h0 = S.surface(c[None, :2])[0]
            yaw = np.arctan2(ax[1], ax[0]); cy, sy = np.cos(yaw), np.sin(yaw)
            mat = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
            add(scn, mujoco.mjtGeom.mjGEOM_BOX, [0.095, 0.04, 0.004], [c[0], c[1], h0 + 0.005], mat, ACCENT)
            labels.append((np.array([c[0], c[1], h0 + 0.02]), f"+{(k - t) / fps:.2f} s"))
        base = q[:3] + [0, 0, 0.55]  # desired velocity
        g = add(scn, mujoco.mjtGeom.mjGEOM_ARROW, [0, 0, 0], [0, 0, 0], rgba=[0.25, 0.27, 0.32, 1])
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.022, base, base + 0.5 * f)

    el, dist, toward, ahead = view
    cam = side_cam(S, q, el, dist, ahead=ahead, toward=toward, dz=dz)
    img, glcam = S.shoot(cam, extra)
    dr = ImageDraw.Draw(img); font = ImageFont.truetype("DejaVuSans-Bold.ttf", int(S.h * 0.04))
    for p, s in labels:
        u, v = project(glcam, p, S.w, S.h)
        tw = dr.textlength(s, font=font)
        dr.text((u - tw / 2, v + S.h * 0.03), s, fill=tuple(int(255 * c) for c in ACCENT[:3]), font=font,
                stroke_width=4, stroke_fill=(255, 255, 255))
    img.save(os.path.join(out, "decision.png"))


def contact_forces(m, Q, dt, t, terr, mu=0.6):
    """Contact forces at frame t from inverse dynamics: the sole ends near the terrain and nearly still each
    take a force in a linearised friction cone (NNLS), as in qa_parc.support. Returns [(side, point, force)]."""
    T, nv = len(Q), m.nv
    v = np.zeros((T, nv))
    for k in range(T):
        a, b = max(k - 1, 0), min(k + 1, T - 1)
        mujoco.mj_differentiatePos(m, v[k], (b - a) * dt, Q[a], Q[b])
    qvel = savgol_filter(v, 7, 2, axis=0); qacc = savgol_filter(v, 7, 2, deriv=1, delta=dt, axis=0)
    m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
    d = mujoco.MjData(m); d.qpos[:], d.qvel[:] = Q[t], qvel[t]
    mujoco.mj_forward(m, d); d.qacc[:] = qacc[t]; mujoco.mj_inverse(m, d)
    m.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
    b = d.qfrc_inverse.copy()
    E = np.array([[mu, 0, 1], [-mu, 0, 1], [0, mu, 1], [0, -mu, 1]]).T
    jacp = np.zeros((3, nv)); pts, cols, sides = [], [], []
    for s, (c, ax, r, hl) in feet_now(m, d).items():
        g = m.geom(f"{s}_foot_collision").id
        for sgn in (-1, 1):
            p = c + sgn * hl * ax - [0, 0, r]
            mujoco.mj_jac(m, d, jacp, None, p, m.geom_bodyid[g])
            if p[2] - terr.h(p[None, :2])[0] < 0.02 and np.linalg.norm(jacp @ qvel[t]) < 0.5:
                pts.append(p); cols.append(jacp.T @ E); sides.append(s)
    if not pts:
        return [], d
    A = np.concatenate(cols, 1)
    lam = nnls(np.vstack([A[:6], 1e-3 * np.eye(A.shape[1])]), np.concatenate([b[:6], np.zeros(A.shape[1])]), maxiter=2000)[0]
    return [(s, p, E @ lam[4 * i:4 * i + 4]) for i, (s, p) in enumerate(zip(sides, pts))], d


def control(S, z, Q, t, out, mu=0.6, view=(-14.0, 3.2, 25.0, 0.1)):
    dt = 1.0 / float(z["fps"])
    F, _ = contact_forces(S.m, Q, dt, t, S.terr, mu)
    S.pose(Q[t])
    weight = 9.81 * S.m.body_subtreemass[1]
    com = S.d.subtree_com[1].copy()
    # one arrow per foot, at its centre of pressure
    feet = []
    for s in "lr":
        ps = [(p, f) for side, p, f in F if side == s]
        if ps:
            fz = np.array([max(f[2], 1e-6) for _, f in ps])
            feet.append((sum(w * p for w, (p, _) in zip(fz, ps)) / fz.sum(), sum(f for _, f in ps)))
    print("contact forces (body weights):", [round(float(np.linalg.norm(f) / weight), 2) for _, f in feet])
    grey = [0.30, 0.32, 0.37, 1]

    def cones(scn):
        for p, _ in feet:
            hc = 0.24; rr = mu * hc
            ring = [p + [rr * np.cos(a), rr * np.sin(a), hc] for a in np.linspace(0, 2 * np.pi, 33)]
            for k, a in enumerate(ring[:-1]):
                if k % 2 == 0:
                    line(scn, p, a, 0.0025, [0.35, 0.37, 0.42, 0.55])
                line(scn, a, ring[k + 1], 0.003, [0.35, 0.37, 0.42, 0.7])

    def arrows(scn):
        for p, fv in feet:
            g = add(scn, mujoco.mjtGeom.mjGEOM_ARROW, [0, 0, 0], [0, 0, 0], rgba=ACCENT)
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.026, p + [0, 0, 0.004], p + [0, 0, 0.004] + 0.42 * fv / weight)
        add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.04, 0, 0], com, rgba=grey)
        g = add(scn, mujoco.mjtGeom.mjGEOM_ARROW, [0, 0, 0], [0, 0, 0], rgba=grey)
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.02, com, com - [0, 0, 0.42])

    el, dist, toward, ahead = view
    cam = side_cam(S, Q[t], el, dist, ahead=ahead, toward=toward, dz=-0.2)
    img, _ = S.shoot(cam, lambda scn: (cones(scn), arrows(scn)))
    # x-ray pass: the arrows alone on black, pasted over the robot so the legs do not hide them
    none = mujoco.MjvOption(); none.geomgroup[:] = 0; none.sitegroup[:] = 0
    S.r.update_scene(S.d, cam, none)
    for f in (mujoco.mjtRndFlag.mjRND_SKYBOX, mujoco.mjtRndFlag.mjRND_FOG, mujoco.mjtRndFlag.mjRND_SHADOW):
        S.r.scene.flags[f] = 0
    arrows(S.r.scene)
    ov = S.r.render().astype(np.float32)
    a = np.clip(ov.max(-1, keepdims=True) / 60.0, 0, 1)
    over = np.clip(ov / np.maximum(a, 1e-3), 0, 255)
    base = np.asarray(img).astype(np.float32)
    Image.fromarray((base * (1 - a) + over * a).astype(np.uint8)).save(os.path.join(out, "motion_control.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("clip")
    ap.add_argument("--see", type=int, default=24); ap.add_argument("--plan", type=int, default=42); ap.add_argument("--force", type=int, default=50)
    ap.add_argument("--width", type=int, default=2400); ap.add_argument("--height", type=int, default=1150)
    ap.add_argument("--see-ahead", type=float, default=1.4, help="head camera aim, m ahead of the root")
    ap.add_argument("--depth-range", type=float, nargs=2, default=(0.5, 3.4), help="depth colour map range, m")
    ap.add_argument("--see-view", type=float, nargs=4, default=(-26.0, 4.9, 35.0, 0.65), help="elevation distance toward ahead")
    ap.add_argument("--force-view", type=float, nargs=4, default=(-14.0, 3.2, 25.0, 0.1), help="elevation distance toward ahead")
    ap.add_argument("--plan-view", type=float, nargs=4, default=(-24.0, 3.3, 15.0, 0.5), help="elevation distance toward ahead")
    ap.add_argument("--cut", action="store_true", help="cut away the terrain between the side camera and the path")
    ap.add_argument("--plain-ground", action="store_true", help="untextured darker ground (for terrain raised above it)")
    ap.add_argument("--aim-level", action="store_true", help="aim the head camera at the height under the robot")
    ap.add_argument("--inset", choices=("left", "right"), default="left", help="depth inset corner (top)")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    z = np.load(a.clip, allow_pickle=True); Q = z["qpos"]
    S = Scene(z, a.width, a.height, a.cut, a.plain_ground)
    perception(S, Q, a.see, a.out, a.see_ahead, a.depth_range, tuple(a.see_view), a.inset, a.aim_level)
    decision(S, z, Q, a.plan, a.out, view=tuple(a.plan_view))
    control(S, z, Q, a.force, a.out, view=tuple(a.force_view))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
