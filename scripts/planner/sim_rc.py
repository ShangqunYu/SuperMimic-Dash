"""Drive the motion-tracking policy with the motion-matching planner in simulation, from a scripted RC
command sequence, and render it (solid robot; text overlay: command, planner reference, actual velocity).

Nominal robot, no noise, fall-only termination. Segments are ``seconds:vx:vy:wz``.

  MUJOCO_GL=egl uv run python scripts/planner/sim_rc.py --out logs/planner/rc_demo.mp4
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
import torch
import tyro
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "media"))
from render_dash_grid import scene_model  # noqa: E402

from supermimic.envs import SuperMimicEnvCfgArgs, make_env  # noqa: E402
from supermimic.eval.benchmark import load_policy_from_checkpoint  # noqa: E402
from supermimic.interfaces.velocity import velocity_of  # noqa: E402
from supermimic.planner.motion_matching import MotionDatabase, MotionMatcher, PlannerCfg  # noqa: E402
from supermimic.robots import get_robot  # noqa: E402

DEFAULT_SEGMENTS = ["2:0:0:0", "6:0.6:0:0", "4:0.6:0:0.8", "3:0:0:0", "4:0:0:1.2", "4:0:0.4:0", "4:-0.4:0:0", "3:0:0:0"]


@torch.no_grad()
def simulate(ckpt: Path, db_path: Path, segments: list[str], start_clip: str, cfg: PlannerCfg):
    policy, pcfg = load_policy_from_checkpoint(ckpt, "cuda:0")
    env = make_env(
        SuperMimicEnvCfgArgs(
            npz_files=[f"data/npz/dash_bones_seed/{start_clip}.npz"], robot="dash", num_envs=1, future_frames=pcfg["env"]["future_frames"],
            episode_length_s=1.0e5, action_noise_std=0.0, obs_noise=False, reset_noise=False, domain_randomization=False,
            sampling_mode="start", record_transitions=True, replay_capacity=4, seed=0,
        ),
        device="cuda:0",
    )
    try:
        cmd_t = env.command_manager.get_term("motion")
        cmd_t.track_mask[:] = False  # fall-only termination; the env's own reference is ignored
        io = env.recorder_manager.get_term("replay").state_io
        db = MotionDatabase(db_path)
        mm = MotionMatcher(db, cfg)
        obs, _ = env.reset()
        rec = {k: [] for k in ("q", "cmd", "act", "ref", "clip", "jumps", "goal0")}
        falls, t_plan = 0, []
        for seg in segments:
            dur, *c = (float(x) for x in seg.split(":"))
            c = np.array(c, dtype=np.float32)
            for _ in range(int(round(dur / env.step_dt))):
                t0 = time.perf_counter()
                goal = mm.step(c)
                t_plan.append(time.perf_counter() - t0)
                g = torch.as_tensor(goal, device="cuda:0")[None]
                obs, _, term, _, _ = env.step(policy(obs["policy"], g))
                falls += int(term.sum())
                rec["q"].append(torch.cat([cmd_t.robot_root_pos_w - env.scene.env_origins, cmd_t.robot_root_quat_w, cmd_t.robot_joint_pos], 1)[0].cpu().numpy())
                rec["cmd"].append(c)
                rec["act"].append(velocity_of(io.read())[0].cpu().numpy())
                rec["ref"].append(mm.reference_velocity().copy())
                rec["clip"].append(mm.clip_name)
                rec["jumps"].append(mm.jumps)
                rec["goal0"].append(goal[: db.goal.shape[1]])
        out = {k: (np.stack(v) if k != "clip" else v) for k, v in rec.items()}
        t_plan = np.array(t_plan) * 1e3
        print(f"falls {falls}, planner jumps {mm.jumps}, planner step ms: mean {t_plan.mean():.2f} p99 {np.percentile(t_plan, 99):.2f} max {t_plan.max():.2f}")
        k = 0
        for seg in segments:
            n = int(round(float(seg.split(":")[0]) / env.step_dt))
            w = slice(k + n // 2, k + n)  # second half of the segment
            print(f"  cmd {seg:>16s}: ref {out['ref'][w].mean(0).round(2)}  actual {out['act'][w].mean(0).round(2)}")
            k += n
        return out
    finally:
        env.close()


def render(rec: dict, out: Path, fps: int = 50, width: int = 1280, height: int = 720, azimuth: float = 90.0) -> None:
    model = scene_model(get_robot("dash").xml_path(), width, height)
    data = mujoco.MjData(model)
    r = mujoco.Renderer(model, height, width)
    cam = mujoco.MjvCamera()
    cam.distance, cam.elevation, cam.azimuth = 3.0, -24.0, azimuth
    font = ImageFont.truetype("DejaVuSans.ttf", 24)
    q = rec["q"]
    out.parent.mkdir(parents=True, exist_ok=True)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
                           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    target = q[0, :3].copy()
    act = rec["act"][0].copy()
    stride = max(1, round(50 / fps))
    for t in range(0, len(q), stride):
        data.qpos[:] = q[t]
        mujoco.mj_forward(model, data)
        target = 0.92 * target + 0.08 * q[t, :3]
        cam.lookat[:] = [target[0], target[1], 0.30]
        r.update_scene(data, cam)
        act = 0.9 * act + 0.1 * rec["act"][t]
        img = Image.fromarray(r.render())
        d = ImageDraw.Draw(img)
        c, f = rec["cmd"][t], rec["ref"][t]
        d.text((20, 12), f"Dash: motion-matching planner + tracking policy   t = {t * 0.02:5.1f} s", fill=(20, 20, 20), font=font)
        d.text((20, 44), f"RC command  vx {c[0]:+.2f}  vy {c[1]:+.2f} m/s  wz {c[2]:+.2f} rad/s", fill=(20, 20, 140), font=font)
        d.text((20, 74), f"actual      vx {act[0]:+.2f}  vy {act[1]:+.2f} m/s  wz {act[2]:+.2f} rad/s", fill=(140, 20, 20), font=font)
        d.text((20, 104), f"playing     {rec['clip'][t].removeprefix('dash_')}", fill=(60, 60, 60), font=font)
        ff.stdin.write(img.tobytes())
    ff.stdin.close()
    ff.wait()
    print("wrote", out)


def main(
    ckpt: Path = Path("logs/dash_e2_hwdr/ckpt_latest.pt"),
    db: Path = Path("data/planner/dash_walk_mm.npz"),
    segments: list[str] = DEFAULT_SEGMENTS,
    out: Path = Path("logs/planner/rc_demo.mp4"),
    start_clip: str = "dash_idle_loop_003__A265",
    render_video: bool = True,
    planner: PlannerCfg = PlannerCfg(),
) -> dict:
    rec = simulate(ckpt, db, segments, start_clip, planner)
    if render_video:
        render(rec, out)
    return rec


if __name__ == "__main__":
    tyro.cli(main)
