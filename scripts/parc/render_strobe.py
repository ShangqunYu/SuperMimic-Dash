"""Fixed side camera renders of Dash retargets for motion-sequence figures. The robot travels left to
right; the frame holds the whole path.

  python render_strobe.py out_dir clip.npz [...] [--width 3840] [--every 0.1] [--n 8]

Per clip, in out_dir/<clip>/:
  video.mp4           fixed camera, real time for Dash, 60 fps
  background.png      terrain only (no robot), same camera and light
  cutouts/t_X.XXs.png robot alone (RGBA, full canvas, so layers stack aligned) every --every s
  shadows/t_X.XXs.png the robot's shadow as a darkening layer (grey, multiply blend), same times
  sequence_solid.png  background + --n robots evenly spaced along the path, later on top, with shadows
  sequence_fade.png   same, earlier robots faded toward the background
Terrain between the camera and the path is cut away (section_cut).
"""
import argparse
import os
import subprocess
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parc_retarget import Terrain
from render_hq import FPS, dash_model, resample

FOVY, ELEV = 15.0, -6.0  # long lens: little perspective, step tops still visible


def framing(m, d, Q, aspect_wh):
    """Camera azimuth (travel to screen right), look-at point and distance that frame the whole path."""
    p = Q[:, :3]
    u = p[-1, :2] - p[0, :2]; u /= np.linalg.norm(u)
    lat = np.array([-u[1], u[0]])
    head = np.degrees(np.arctan2(u[1], u[0]))
    # extents of every robot body over the clip, along the path and vertically
    pts = []
    for q in Q[::4]:
        d.qpos[:] = q; mujoco.mj_kinematics(m, d)
        pts.append(d.xpos[1:].copy())
    pts = np.concatenate(pts)
    s = pts[:, :2] @ u; z = pts[:, 2]
    s0, s1 = s.min() - 0.35, s.max() + 0.35
    z0, z1 = z.min() - 0.25, z.max() + 0.30
    w, h = s1 - s0, z1 - z0
    if w / h > aspect_wh:
        h = w / aspect_wh
    else:
        w = h * aspect_wh
    c_s, c_z = (s0 + s1) / 2, (z0 + z1) / 2
    c_l = float(np.mean(pts[:, :2] @ lat))
    look = c_s * u + c_l * lat
    dist = (h / 2) / np.tan(np.radians(FOVY / 2))
    # the camera sits at look - dist (cos az, sin az): az = head + 90 puts it on the robot's right side
    return np.array([look[0], look[1], c_z]), head + 90.0, dist


def section_cut(boxes, Q, m, d, margin=0.3):
    """Drop terrain boxes that lie wholly between the camera and the path, and trim the ones that reach into
    that space, so foreground terrain never hides the robot (a cutaway). The camera is on the robot's right
    of the travel direction; the cut is `margin` beyond the feet."""
    p = Q[:, :2]; u = p[-1] - p[0]; u /= np.linalg.norm(u); lat = np.array([-u[1], u[0]])
    feet = []
    for q in Q[::4]:
        d.qpos[:] = q; mujoco.mj_kinematics(m, d)
        feet += [d.xpos[m.body(f"{s}_foot").id][:2] for s in "lr"]
    cut = float(np.min(np.array(feet) @ lat)) - margin
    out = []
    for b in boxes:
        cx, cy, cz, hx, hy, hz = b
        corners = np.array([[cx + sx * hx, cy + sy * hy] for sx in (-1, 1) for sy in (-1, 1)]) @ lat
        if corners.max() <= cut:
            continue
        if corners.min() < cut:  # trim along the box axis closest to the lateral direction
            ax = 0 if abs(lat[0]) >= abs(lat[1]) else 1
            sgn = np.sign(lat[ax])
            lo, hi = (cx - hx, cx + hx) if ax == 0 else (cy - hy, cy + hy)
            c0 = (cut - (np.array([cx, cy]) @ lat - lat[ax] * (cx if ax == 0 else cy))) / lat[ax]  # axis coord at the cut
            lo, hi = (max(lo, c0), hi) if sgn > 0 else (lo, min(hi, c0))
            if hi - lo < 1e-3:
                continue
            c, h = (lo + hi) / 2, (hi - lo) / 2
            b = (c, cy, cz, h, hy, hz) if ax == 0 else (cx, c, cz, hx, h, hz)
        out.append(b)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("clips", nargs="+")
    ap.add_argument("--width", type=int, default=3840); ap.add_argument("--height", type=int, default=1620)
    ap.add_argument("--every", type=float, default=0.1); ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    W, H = a.width, a.height
    for f in a.clips:
        name = os.path.basename(f).replace(".npz", "")
        od = os.path.join(a.out, name); os.makedirs(os.path.join(od, "cutouts"), exist_ok=True)
        os.makedirs(os.path.join(od, "shadows"), exist_ok=True)
        z = np.load(f, allow_pickle=True)
        Q = resample(z["qpos"], float(z["fps"]), 1.0)
        m0 = dash_model([], 64, 64)
        boxes = section_cut(Terrain(z["hf"], z["min_point"], z["dx"]).boxes(), Q, m0, mujoco.MjData(m0))
        m = dash_model(boxes, W, H)
        d = mujoco.MjData(m)
        look, az, dist = framing(m, d, Q, W / H)
        e = m.stat.extent
        m.vis.global_.fovy = FOVY
        m.vis.map.zfar = (dist + 60) / e
        m.vis.map.fogstart, m.vis.map.fogend = (dist + 6) / e, (dist + 25) / e
        key = m.light("key").id; kd = m.light_dir[key] / np.linalg.norm(m.light_dir[key])
        m.light_pos[key] = look - 20 * kd
        r = mujoco.Renderer(m, H, W)
        cam = mujoco.MjvCamera(); cam.lookat[:], cam.azimuth, cam.elevation, cam.distance = look, az, ELEV, dist
        full = mujoco.MjvOption(); full.geomgroup[:] = 0; full.geomgroup[[0, 1, 2]] = 1; full.sitegroup[:] = 0
        bgopt = mujoco.MjvOption(); bgopt.geomgroup[:] = 0; bgopt.geomgroup[[0, 1]] = 1; bgopt.sitegroup[:] = 0
        robot = np.flatnonzero(m.geom_group == 2)

        def shot(q, opt, seg=False):
            d.qpos[:] = q; mujoco.mj_forward(m, d)
            r.update_scene(d, cam, opt)
            r.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = 1
            if seg:
                r.enable_segmentation_rendering()
                s = r.render(); r.disable_segmentation_rendering()
                return s
            return r.render().copy()

        bg = shot(Q[0], bgopt)
        Image.fromarray(bg).save(os.path.join(od, "background.png"))
        bgf = bg.astype(np.float32) / 255

        def layers(t):
            img = shot(Q[t], full); s = shot(Q[t], full, seg=True)
            mask = ((s[..., 1] == mujoco.mjtObj.mjOBJ_GEOM) & np.isin(s[..., 0], robot)).astype(np.float32)
            alpha = np.clip(gaussian_filter(mask, 0.6) * 1.15, 0, 1)  # soft edge for the unantialiased mask
            shade = np.clip(img.astype(np.float32) / 255 / np.maximum(bgf, 1e-3), 0, 1).min(-1)
            shade = np.where(gaussian_filter(mask, 1.5) > 0.01, 1.0, shade)  # robot pixels: not shadow
            return img, alpha, shade

        # video
        ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                               "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                               os.path.join(od, "video.mp4")], stdin=subprocess.PIPE)
        for q in Q:
            ff.stdin.write(shot(q, full).tobytes())
        ff.stdin.close(); ff.wait()
        # cutouts and shadow layers on a time grid
        for t in range(0, len(Q), max(1, round(a.every * FPS))):
            img, alpha, shade = layers(t)
            tag = f"t_{t / FPS:05.2f}s.png"
            Image.fromarray(np.dstack([img, (alpha * 255).astype(np.uint8)])).save(os.path.join(od, "cutouts", tag))
            Image.fromarray((shade * 255).astype(np.uint8)).save(os.path.join(od, "shadows", tag))
        # sequences: n robots evenly spaced along the path
        prog = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(Q[:, :2], axis=0), axis=1))])
        ks = [int(np.searchsorted(prog, x)) for x in np.linspace(0, prog[-1], a.n)]
        ks[-1] = min(ks[-1], len(Q) - 1)
        L = [layers(k) for k in ks]
        shade = np.min([sh for _, _, sh in L], 0)
        for kind in ("solid", "fade"):
            out = bgf * shade[..., None]
            for i, (img, alpha, _) in enumerate(L):
                op = 1.0 if kind == "solid" else 0.35 + 0.65 * i / max(len(L) - 1, 1)
                al = (alpha * op)[..., None]
                out = out * (1 - al) + img.astype(np.float32) / 255 * al
            Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8)).save(os.path.join(od, f"sequence_{kind}.png"))
        with open(os.path.join(od, "sequence_times.txt"), "w") as fh:
            fh.write(" ".join(f"{k / FPS:.2f}" for k in ks) + "\n")
        r.close()
        print("wrote", od, "times", [round(k / FPS, 2) for k in ks], flush=True)


if __name__ == "__main__":
    main()
