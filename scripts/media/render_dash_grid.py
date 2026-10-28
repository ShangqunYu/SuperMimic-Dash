"""Render a Dash policy tracking N random clips at once, as one grid video (offscreen, no viewer).

Each robot is assigned one clip and replays it from its first frame, restarting the same clip when it
ends or the robot falls. Solid robot = policy in simulation, translucent blue = reference; a robot is
drawn red for 1 s after a fall. Each clip starts on its own grid cell and the robot then moves freely
(robots may cross); --treadmill instead keeps every reference root on its cell centre. Deterministic
evaluation settings (no noise, no DR).

  MUJOCO_GL=glfw uv run python scripts/media/render_dash_grid.py --seed 1 --out logs/dash_e1_sim/videos/grid_seed1.mp4

Also writes <out>.csv: cell (row, col) -> clip name, clip length, number of falls.
"""

from __future__ import annotations

import copy
import csv
import math
import os
import random
import subprocess
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw")
import mujoco
import numpy as np
import torch
import tyro
from PIL import Image, ImageDraw, ImageFont

from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.eval.benchmark import load_policy_from_checkpoint
from supermimic.robots import get_robot

def scene_model(xml: Path, width: int, height: int) -> mujoco.MjModel:
    """The robot plus a floor and lights. The renderer must be built from a model that owns the
    robot meshes (its GL context only uploads its own model's assets)."""
    spec = mujoco.MjSpec.from_file(str(xml))
    spec.visual.global_.offwidth = width
    spec.visual.global_.offheight = height
    spec.visual.quality.shadowsize = 8192
    spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[0.82, 0.84, 0.86], rgb2=[0.74, 0.76, 0.79], width=512, height=512)
    mat = spec.add_material(name="grid", texrepeat=[1, 1], texuniform=True)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    wb = spec.worldbody
    wb.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[100, 100, 0.1], material="grid", contype=0, conaffinity=0)
    wb.add_light(pos=[0, 0, 30], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, diffuse=[0.6, 0.6, 0.6], specular=[0.1, 0.1, 0.1], castshadow=True)
    wb.add_light(pos=[10, -10, 15], dir=[-1, 1, -1.5], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, diffuse=[0.25, 0.25, 0.25], castshadow=False)
    return spec.compile()


def recolor(model: mujoco.MjModel, rgba: tuple[float, float, float, float]) -> mujoco.MjModel:
    out = copy.deepcopy(model)
    out.geom_matid[:] = -1
    out.geom_rgba[:] = rgba
    return out


@torch.no_grad()
def simulate(ckpt: Path, files: list[str], seconds: float, fps: int, device: str):
    policy, cfg = load_policy_from_checkpoint(ckpt, device)
    robot = cfg.get("robot", "g1")
    n = len(files)
    env = make_env(
        SuperMimicEnvCfgArgs(
            npz_files=files, robot=robot, num_envs=n, future_frames=cfg["env"]["future_frames"], episode_length_s=1.0e5,
            action_noise_std=0.0, obs_noise=False, reset_noise=False, domain_randomization=False, sampling_mode="start",
            record_transitions=False, seed=0, anchor_obs=cfg["env"].get("anchor_obs", False),
        ),
        device=device,
    )
    try:
        cmd = env.command_manager.get_term("motion")
        lib = cmd.library
        # Env e replays clip e forever: every reset of e starts at its clip's first frame.
        cmd.forced_frame[:] = lib.clip_start[torch.arange(n, device=device)]
        names = [lib.clips[i].name for i in range(n)]
        lengths = [lib.clips[i].length for i in range(n)]
        start_xy = lib.ref(lib.clip_start[:n]).root_pos_w[:, :2].cpu().numpy()  # reference root at each clip's first frame
        obs, _ = env.reset()
        origins = env.scene.env_origins
        stride = max(1, round(1.0 / (env.step_dt * fps)))
        steps = int(seconds / env.step_dt)
        frames_robot, frames_ref, frames_fall = [], [], []
        falls = torch.zeros(n, dtype=torch.long, device=device)
        last_fall = torch.full((n,), -10_000, dtype=torch.long, device=device)
        for t in range(steps):
            obs, _, term, _, _ = env.step(policy(obs["policy"], obs["goal"]))
            falls += term.long()
            last_fall[term] = t
            if t % stride == 0:
                frames_robot.append(torch.cat([cmd.robot_root_pos_w - origins, cmd.robot_root_quat_w, cmd.robot_joint_pos], 1).cpu().numpy())
                frames_ref.append(torch.cat([cmd.root_pos_w - origins, cmd.root_quat_w, cmd.joint_pos], 1).cpu().numpy())
                frames_fall.append(((t - last_fall) * env.step_dt < 1.0).cpu().numpy())
        return robot, names, lengths, start_xy, falls.cpu().numpy(), np.stack(frames_robot), np.stack(frames_ref), np.stack(frames_fall)
    finally:
        env.close()


def render(robot_name: str, robot_q, ref_q, fall, start_xy, out: Path, fps: int, width: int, height: int, spacing: float, title: str, treadmill: bool = False) -> None:
    spec = get_robot(robot_name)
    base = scene_model(spec.xml_path(), width, height)
    models = {"robot": base, "ghost": recolor(base, (0.25, 0.5, 1.0, 0.2)), "fall": recolor(base, (0.9, 0.15, 0.1, 1.0))}
    data = mujoco.MjData(base)
    mujoco.mj_forward(base, data)

    T, n, _ = robot_q.shape
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    centers = np.array([[(i % cols - (cols - 1) / 2) * spacing, ((rows - 1) / 2 - i // cols) * spacing] for i in range(n)])

    renderer = mujoco.Renderer(base, height, width, max_geom=60 * 2 * n + 100)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = (0.0, -0.06 * rows * spacing, 0.3)
    cam.distance = 1.25 * max(rows, cols) * spacing + 4.0
    cam.elevation = -38.0
    cam.azimuth = 90.0
    opt = mujoco.MjvOption()
    pert = mujoco.MjvPerturb()
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 26)
    except OSError:
        font = ImageFont.load_default()

    out.parent.mkdir(parents=True, exist_ok=True)
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out)],
        stdin=subprocess.PIPE,
    )
    for t in range(T):
        scn = renderer.scene
        mujoco.mjv_updateScene(base, data, opt, pert, cam, mujoco.mjtCatBit.mjCAT_STATIC, scn)  # floor + lights only
        for e in range(n):
            # Free roaming: the clip's start lands on the cell centre. Treadmill: the current reference root does.
            shift = centers[e] - (ref_q[t, e, :2] if treadmill else start_xy[e])
            for key, q in (("ghost", ref_q[t, e]), ("fall" if fall[t, e] else "robot", robot_q[t, e])):
                m = models[key]
                data.qpos[:] = q
                data.qpos[:2] += shift
                mujoco.mj_kinematics(m, data)
                mujoco.mjv_addGeoms(m, data, opt, pert, mujoco.mjtCatBit.mjCAT_DYNAMIC, scn)
        img = Image.fromarray(renderer.render())
        ImageDraw.Draw(img).text((20, 14), f"{title}   t = {t / fps:5.1f} s   falls so far: {int(fall[: t + 1].any(0).sum())} robots", fill=(20, 20, 20), font=font)
        ff.stdin.write(img.tobytes())
    ff.stdin.close()
    ff.wait()


def main(
    ckpt: Path = Path("logs/dash_e1_sim/ckpt_latest.pt"),
    npz_dir: Path = Path("data/npz/dash_bones_seed"),
    out: Path = Path("logs/dash_e1_sim/videos/grid_seed0.mp4"),
    num_clips: int = 100,
    seed: int = 0,
    filter: str = "",
    seconds: float = 20.0,
    fps: int = 25,
    width: int = 1920,
    height: int = 1080,
    spacing: float = 1.6,
    treadmill: bool = False,
    device: str = "cuda:0",
) -> None:
    names = sorted(p.stem for p in npz_dir.glob("*.npz") if filter.lower() in p.stem.lower())
    picked = sorted(random.Random(seed).sample(names, min(num_clips, len(names))))
    files = [str(npz_dir / f"{s}.npz") for s in picked]
    robot, clip_names, lengths, start_xy, falls, robot_q, ref_q, fall = simulate(ckpt, files, seconds, fps, device)
    print(f"[render] {len(files)} clips, {robot_q.shape[0]} frames, {int((falls > 0).sum())} robots fell at least once")
    render(robot, robot_q, ref_q, fall, start_xy, out, fps, width, height, spacing, f"{ckpt.parent.name}  seed {seed}", treadmill)
    cols = math.ceil(math.sqrt(len(files)))
    with open(out.with_suffix(".csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell", "row", "col", "clip", "clip_seconds", "falls"])
        for e, (name, length, nf) in enumerate(zip(clip_names, lengths, falls)):
            w.writerow([e, e // cols, e % cols, name, f"{length / 50:.1f}", int(nf)])
    print(f"[render] wrote {out} and {out.with_suffix('.csv')}")


if __name__ == "__main__":
    tyro.cli(main)
