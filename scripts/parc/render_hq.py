"""Presentation renders of Dash retargets of PARC clips: 1920x1080, 60 fps, MSAA, soft shadows, a
smoothed tracking camera. One video per clip.

  python render_hq.py out_dir retarget.npz [...] [--human] [--slow 0.5]

--human: side by side, PARC human on its terrain (left) | Dash on the scaled terrain (right).
--slow: playback speed (1 = real time for Dash).
"""
import argparse
import os
import re
import subprocess
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parc_io import HUMANOID_XML, load, skeleton
from parc_retarget import DASH_XML, Terrain

D = os.environ.get("PARC_DATA", "/media/shangqunyu/Dash/parc_src/parc_data/dec_release")  # huggingface.co/datasets/mxucg/PARC
W, H, FPS = 1920, 1080, 60
DARK, ALU = [0.13, 0.14, 0.15, 1], [0.70, 0.72, 0.75, 1]  # motor housings and soles | aluminum


def style(spec, boxes, w, h, robot_rgba=None):
    """Sky, grid-textured terrain and ground, key/fill lights, shadow and MSAA quality."""
    v = spec.visual
    v.global_.offwidth, v.global_.offheight = w, h
    v.quality.shadowsize, v.quality.offsamples = 16384, 8
    v.headlight.ambient, v.headlight.diffuse, v.headlight.specular = [0.35] * 3, [0.35] * 3, [0.05] * 3
    v.rgba.fog = [0.90, 0.92, 0.95, 1]
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.96, 0.97, 0.99], rgb2=[0.78, 0.83, 0.90], width=512, height=3072)
    spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[0.87, 0.88, 0.90], rgb2=[0.83, 0.84, 0.87], mark=mujoco.mjtMark.mjMARK_EDGE,
                     markrgb=[0.70, 0.72, 0.76], width=512, height=512)
    tex = [""] * 10; tex[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    spec.add_material(name="terrain", textures=tex, texuniform=True, texrepeat=[1.25, 1.25], specular=0.1, shininess=0.2, reflectance=0.0)
    spec.add_material(name="dark", rgba=DARK, specular=0.35, shininess=0.5)
    spec.add_material(name="light", rgba=robot_rgba or ALU, specular=0.9, shininess=0.9, reflectance=0.08)
    wb = spec.worldbody
    base = min(b[2] + b[5] for b in boxes) if len(boxes) else 0.0
    # a thin box, not a plane: MuJoCo planes show shadow acne away from the light center
    wb.add_geom(name="ground", type=mujoco.mjtGeom.mjGEOM_BOX, size=[40, 40, 0.05], pos=[0, 0, base - 0.05], material="terrain")
    for i, b in enumerate(boxes):
        if b[2] + b[5] <= base + 1e-6:  # the ground plane shows these; drawing both z-fights
            continue
        wb.add_geom(name=f"terrain_{i}", type=mujoco.mjtGeom.mjGEOM_BOX, pos=b[:3], size=b[3:], material="terrain")
    wb.add_light(name="key", pos=[0, 0, 25], dir=[0.45, 0.35, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                 diffuse=[0.55] * 3, specular=[0.2] * 3, castshadow=True)
    wb.add_light(name="fill", pos=[0, 0, 10], dir=[-0.5, -0.6, -0.8], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                 diffuse=[0.18] * 3, specular=[0, 0, 0], castshadow=False)


def distances(m, unit):
    """Clip, fog and shadow distances in meters (x unit, the character scale); MuJoCo takes them as fractions
    of the model extent. A far plane far beyond the scene leaves too little depth precision and the distant
    ground speckles; the fog fades the ground into the sky before that."""
    e = m.stat.extent
    m.vis.map.znear, m.vis.map.zfar = 0.05 * unit / e, 60 * unit / e
    m.vis.map.fogstart, m.vis.map.fogend = 5 * unit / e, 14 * unit / e
    # +-20 m around the key light's axis, which follows the robot: ground outside it but short of the fog
    # samples the clamped shadow map edge and speckles
    m.vis.map.shadowclip = 20 * unit / e
    return m


def dash_model(boxes, w, h):
    spec = mujoco.MjSpec.from_file(DASH_XML)
    style(spec, boxes, w, h)
    for g in spec.geoms:
        if g.type == mujoco.mjtGeom.mjGEOM_MESH:
            g.material = "dark" if re.search(r"foot|dist_hip|prox_shoulder", g.meshname) else "light"
    return distances(spec.compile(), 1.0)


def human_model(boxes, w, h, unit):
    xml = open(HUMANOID_XML).read()
    xml = re.sub(r'<geom name="floor"[^>]*/>', "", xml)
    xml = re.sub(r"<actuator>.*?</actuator>", "", xml, flags=re.S)
    xml = re.sub(r'(<joint name="(\w+?)_x" type="hinge"[^>]*/>)\s*<joint[^>]*_y"[^>]*/>\s*<joint[^>]*_z"[^>]*/>',
                 lambda mm: f'<joint name="{mm.group(2)}" type="ball"/>', xml)
    xml = re.sub(r'<joint name="(\w+_(?:knee|elbow))" type="hinge"[^>]*/>', r'<joint name="\1" type="ball"/>', xml)
    xml = re.sub(r'material="\w+"', "", xml)
    spec = mujoco.MjSpec.from_string(xml.replace('file="', 'file="' + os.path.dirname(HUMANOID_XML) + '/'))
    style(spec, boxes, w, h, robot_rgba=[0.83, 0.62, 0.45, 1])
    for g in spec.geoms:
        if g.name != "ground" and not g.name.startswith("terrain_"):
            g.material = "light"
    return distances(spec.compile(), unit)


def resample(q, fps, speed):
    """Free-joint qpos (wxyz root, then hinge/ball blocks) at 60 frames per second of playback."""
    T = len(q); t0 = np.arange(T) / fps
    t1 = np.arange(0, t0[-1] + 1e-9, speed / FPS)
    out = np.empty((len(t1), q.shape[1]))
    for k in range(q.shape[1]):
        out[:, k] = np.interp(t1, t0, q[:, k])
    out[:, 3:7] = Slerp(t0, Rotation.from_quat(q[:, [4, 5, 6, 3]]))(t1).as_quat()[:, [3, 0, 1, 2]]
    return out


def resample_balls(q, fps, speed, ball_adr):
    out = resample(q, fps, speed)
    T = len(q); t0 = np.arange(T) / fps; t1 = np.arange(0, t0[-1] + 1e-9, speed / FPS)
    for a in ball_adr:
        out[:, a:a + 4] = Slerp(t0, Rotation.from_quat(q[:, [a + 1, a + 2, a + 3, a]]))(t1).as_quat()[:, [3, 0, 1, 2]]
    return out


def camera_track(q, fps_play, height_off):
    """Look-at point: root xy smoothed over 0.5 s, root height smoothed over 1 s. Camera: on the robot's left,
    20 deg toward its front, following the heading smoothed over 1 s (clips turn around)."""
    p = q[:, :3]
    look = np.stack([gaussian_filter1d(p[:, 0], 0.5 * fps_play, mode="nearest"),
                     gaussian_filter1d(p[:, 1], 0.5 * fps_play, mode="nearest"),
                     gaussian_filter1d(p[:, 2], 1.0 * fps_play, mode="nearest") + height_off], 1)
    w, x, y, z = q[:, 3:7].T
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    head = np.degrees(gaussian_filter1d(np.unwrap(yaw), 1.0 * fps_play, mode="nearest"))
    return look, head + 70 + 180  # the camera sits at look - dist (cos az, sin az): 70 deg off the facing direction


def render(m, Q, look, az, dist, w, h, label=None):
    d = mujoco.MjData(m); r = mujoco.Renderer(m, h, w)
    cam = mujoco.MjvCamera(); cam.distance, cam.elevation = dist, -14.0
    az = np.broadcast_to(az, len(Q))
    opt = mujoco.MjvOption(); opt.geomgroup[:] = 0; opt.geomgroup[[0, 1, 2]] = 1; opt.sitegroup[:] = 0
    font = ImageFont.truetype("DejaVuSans.ttf", 34)
    key = m.light("key").id; kd = m.light_dir[key] / np.linalg.norm(m.light_dir[key])
    for t, q in enumerate(Q):
        m.light_pos[key] = look[t] - 20 * kd  # a directional light's shadow box is centered on its position
        d.qpos[:] = q; mujoco.mj_forward(m, d)
        cam.lookat[:] = look[t]; cam.azimuth = az[t]
        r.update_scene(d, cam, opt)
        r.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = 1
        img = r.render()
        if label:
            im = Image.fromarray(img); ImageDraw.Draw(im).text((36, 28), label, fill=(40, 44, 52), font=font); img = np.asarray(im)
        yield img
    r.close()


def encode(path, frames, w, h):
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-",
                           "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p", "-movflags", "+faststart", path],
                          stdin=subprocess.PIPE)
    for f in frames:
        ff.stdin.write(np.ascontiguousarray(f).tobytes())
    ff.stdin.close(); ff.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("clips", nargs="+")
    ap.add_argument("--human", action="store_true"); ap.add_argument("--slow", type=float, default=1.0)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    for f in a.clips:
        z = np.load(f, allow_pickle=True)
        s, fps = float(z["scale"]), float(z["fps"])
        Qd = resample(z["qpos"], fps, a.slow)
        look, az = camera_track(Qd, FPS, -0.15)
        name = os.path.basename(f).replace(".npz", "") + ("_side" if a.human else "") + (f"_x{a.slow:g}" if a.slow != 1 else "")
        td = Terrain(z["hf"], z["min_point"], z["dx"])
        if not a.human:
            encode(os.path.join(a.out, name + ".mp4"), render(dash_model(td.boxes(), W, H), Qd, look, az, 2.9, W, H), W, H)
        else:
            src = str(z["source"]); m, t = load(src if os.path.isabs(src) else os.path.join(D, src))
            th = Terrain(t["hf"], t["min_point"], t["dx"]); hm = human_model(th.boxes(), W // 2, H, 1 / s)
            names = skeleton()[0]; T = len(m["root_pos"])
            hq = np.zeros((T, hm.nq)); hq[:, :3] = m["root_pos"]; hq[:, 3:7] = m["root_rot"][:, [3, 0, 1, 2]]
            balls = []
            for j in range(1, hm.njnt):
                b = hm.body(hm.jnt_bodyid[j]).name; adr = hm.jnt_qposadr[j]; balls.append(adr)
                hq[:, adr:adr + 4] = m["joint_rot"][:, names.index(b) - 1][:, [3, 0, 1, 2]]
            Qh = resample_balls(hq, fps, a.slow, balls)[:len(Qd)]  # frame-aligned with Dash: the human plays at Dash time
            if len(Qh) < len(Qd):
                Qh = np.concatenate([Qh, np.repeat(Qh[-1:], len(Qd) - len(Qh), 0)])
            lh, _ = camera_track(Qh, FPS, -0.15 / s)
            fh = render(hm, Qh, lh, az, 2.9 / s, W // 2, H, "Human (PARC)")
            fd = render(dash_model(td.boxes(), W // 2, H), Qd, look, az, 2.9, W // 2, H, "Dash retarget")
            encode(os.path.join(a.out, name + ".mp4"), (np.concatenate([x, y], 1) for x, y in zip(fh, fd)), W, H)
        print("wrote", name, flush=True)


if __name__ == "__main__":
    main()
