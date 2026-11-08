"""Side by side: PARC human on its terrain | Dash retarget on the scaled terrain. Grid of clips.
  python render_parc.py out.mp4 retarget.npz [...]"""
import os, re, subprocess, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco, numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parc_io import HUMANOID_XML, load
from parc_retarget import DASH_XML, Terrain
W, H = 480, 360
font = ImageFont.truetype("DejaVuSans.ttf", 13)

def add_terrain(spec, boxes):
    wb = spec.worldbody
    for i, b in enumerate(boxes):
        wb.add_geom(name=f"terrain_{i}", type=mujoco.mjtGeom.mjGEOM_BOX, pos=b[:3], size=b[3:], rgba=[0.55, 0.62, 0.72, 1])
    wb.add_light(pos=[0, 0, 10], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, diffuse=[0.7] * 3, castshadow=True)
    spec.visual.global_.offwidth, spec.visual.global_.offheight = W, H

def human_model(boxes):
    xml = open(HUMANOID_XML).read()
    xml = re.sub(r'<geom name="floor"[^>]*/>', "", xml)
    xml = re.sub(r"<actuator>.*?</actuator>", "", xml, flags=re.S)
    # each body's hinge triplet -> one ball joint
    xml = re.sub(r'(<joint name="(\w+?)_x" type="hinge"[^>]*/>)\s*<joint[^>]*_y"[^>]*/>\s*<joint[^>]*_z"[^>]*/>',
                 lambda mm: f'<joint name="{mm.group(2)}" type="ball"/>', xml)
    xml = re.sub(r'<joint name="(\w+_(?:knee|elbow))" type="hinge"[^>]*/>', r'<joint name="\1" type="ball"/>', xml)
    spec = mujoco.MjSpec.from_string(xml.replace('file="', 'file="' + os.path.dirname(HUMANOID_XML) + '/'))
    add_terrain(spec, boxes)
    return spec.compile()

def dash_model(boxes):
    spec = mujoco.MjSpec.from_file(DASH_XML); add_terrain(spec, boxes); return spec.compile()

def frames(m, qposes, label, dist, lines=None):
    d = mujoco.MjData(m); r = mujoco.Renderer(m, H, W)
    cam = mujoco.MjvCamera(); cam.distance, cam.elevation, cam.azimuth = dist, -12.0, 100.0
    look = qposes[0][:3].copy(); out = []
    for t, q in enumerate(qposes):
        d.qpos[:len(q)] = q; mujoco.mj_forward(m, d)
        look = 0.9 * look + 0.1 * q[:3]; cam.lookat[:] = look
        r.update_scene(d, cam); img = Image.fromarray(r.render()); dr = ImageDraw.Draw(img)
        dr.text((6, 4), label, fill=(255, 255, 255), font=font)
        if lines: dr.text((6, 20), lines[t], fill=(255, 255, 255), font=font)
        out.append(np.asarray(img))
    return out

def clip_panels(npz):
    z = np.load(npz, allow_pickle=True)
    m, t = load(str(z["source"]))
    s = float(z["scale"])
    th = Terrain(t["hf"], t["min_point"], t["dx"]); td = Terrain(z["hf"], z["min_point"], z["dx"])
    hm = human_model(th.boxes()); dm = dash_model(td.boxes())
    from parc_io import skeleton
    names = skeleton()[0]; T = len(m["root_pos"])
    hq = np.zeros((T, hm.nq)); hq[:, :3] = m["root_pos"]; hq[:, 3:7] = m["root_rot"][:, [3, 0, 1, 2]]
    for j in range(1, hm.njnt):
        b = hm.body(hm.jnt_bodyid[j]).name
        hq[:, hm.jnt_qposadr[j]:hm.jnt_qposadr[j] + 4] = m["joint_rot"][:, names.index(b) - 1][:, [3, 0, 1, 2]]
    name = os.path.basename(str(z["source"])).replace(".pkl", "")[:40]
    c = z["contact"]
    lines = [f"stance L {'#' if c[i,0] else '-'} R {'#' if c[i,1] else '-'}" for i in range(len(c))]
    return frames(hm, hq, f"human  {name}", 3.6), frames(dm, z["qpos"], f"Dash (x{s:.2f}, time x{np.sqrt(s):.2f})", 3.6 * s, lines)

out, files = sys.argv[1], sys.argv[2:]
FPS = float(np.load(files[0])["fps"])  # Dash time: real time for the robot
panels = [p for f in files for p in clip_panels(f)]
NC = 2; NR = len(panels) // NC
L = max(len(p) for p in panels)
ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{NC*W}x{NR*H}", "-r", f"{FPS:.3f}", "-i", "-",
                       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22", out], stdin=subprocess.PIPE)
for t in range(L):
    cv = np.zeros((NR * H, NC * W, 3), np.uint8)
    for k, p in enumerate(panels):
        a, b = divmod(k, NC); cv[a*H:(a+1)*H, b*W:(b+1)*W] = p[min(t, len(p) - 1)]
    ff.stdin.write(cv.tobytes())
ff.stdin.close(); ff.wait(); print("wrote", out)
