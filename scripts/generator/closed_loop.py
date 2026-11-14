"""Generator + tracking teacher in physics, on the procedural scenes of rollout_generator.py (terrains neither
was trained on): every 0.2 s the generator plans 1.2 s from the robot's actual last 10 frames, the heightmap at
the robot and the joystick command; the plan is written into the env's reference (a packed library made here,
one clip per env on its scene's terrain, whose frames ahead of the tracked frame are overwritten) and the
teacher tracks it.

  python scripts/generator/closed_loop.py gen_ckpt teacher_ckpt out_dir [--samples 4] [--seconds 8] [--render]

Scores the robot's own motion with rollout_generator.score (feet in physics, root above the ground, past the
obstacle on the right level, speed). Terminations are off (a fall shows as the root dropping). Plays in the
teacher's contacts (its config). --render: robot + plan (green ghost) per scene, and grid.mp4.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rollout_generator import DX, REPLAN, VEL_HALFLIFE, X0, Y0, command_path, first_library, load, operator, scenes, score, standing_frame  # noqa: E402

from supermimic.envs import SuperMimicEnvCfgArgs, make_env  # noqa: E402
from supermimic.eval.benchmark import goal_env_kwargs, load_policy_from_checkpoint  # noqa: E402
from supermimic.generator.data import GEN_GRID, GRID_SHAPE, from_canonical, to_canonical  # noqa: E402
from supermimic.generator.metrics import FootMetrics  # noqa: E402
from supermimic.kinematics.rotations import heading_yaw, quat_inv, quat_mul, rotvec_from_quat  # noqa: E402
from supermimic.terrain.height_scan import HeightScan  # noqa: E402
from supermimic.terrain.terrains import Terrain, load_terrains, save_terrains, terrains_path  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DT = 0.02


def build_library(path: Path, terr, scene_of_clip, starts, frames: int, pad: float = 1.0) -> None:
    """One clip per env, `frames` long, every frame its start pose at rest; terrains as the sidecar."""
    n = len(scene_of_clip)
    ref = torch.zeros(n * frames, 49)
    ref[:, :25] = starts.cpu().repeat_interleave(frames, 0)
    torch.save({"ref": ref.half(), "names": [f"gen_loop_{k}" for k in range(n)], "starts": [k * frames for k in range(n)],
                "lengths": [frames] * n, "robot": "dash", "clip_tile": list(scene_of_clip), "terrain_names": [f"scene_{k}" for k in range(len(terr))]}, path)
    save_terrains(terrains_path(path), [f"scene_{k}" for k in range(len(terr))], [t.padded(pad) for t in terr])


def with_velocities(cur: torch.Tensor, plan: torch.Tensor) -> torch.Tensor:
    """cur (B, 25) robot now, plan (B, F, 25) -> (B, F, 49) library frames: velocities by differences."""
    seq = torch.cat([cur[:, None], plan], 1)
    lin = torch.gradient(seq[..., :3], spacing=DT, dim=1)[0]
    jv = torch.gradient(seq[..., 7:], spacing=DT, dim=1)[0]
    q = seq[..., 3:7]
    dq = quat_mul(q[:, 1:], quat_inv(q[:, :-1]))
    w = rotvec_from_quat(dq * torch.sign(dq[..., :1] + 1e-12)) / DT  # world frame, between frames (shortest way)
    ang = torch.cat([w[:, :1], 0.5 * (w[:, 1:] + w[:, :-1]) if w.shape[1] > 1 else w[:, :0], w[:, -1:]], 1)
    return torch.cat([seq[..., :25], lin, ang, jv], -1)[:, 1:]


def run_loop(model, diff, teacher: str, terr, tiles, start, cmd_v, seconds: float, workdir: Path, dev: str, steps: int = 20, guidance: float = 1.0,
             seed: int = 0, replan: int = REPLAN, halflife: float = VEL_HALFLIFE):
    """Generator + teacher in physics. terr: list of Terrain; tiles (n,) each env's terrain; start (n, 25) its start
    pose; cmd_v (n, 3) its joystick command (operator heading hold to yaw 0 applies), or a schedule (n, T, 4) per
    step: vx, vy, wz, the heading held where wz is 0. replan: frames between plans; halflife: the command path's
    velocity spring (rollout_generator.command_path). -> robot frames Q (n, T, 25), the
    tracked plan frame R (n, T, 25), robot velocities V (n, T, 6 + NJ) [root lin vel w, root ang vel w, joint vel]."""
    n = len(tiles)
    T = int(seconds * 50)
    lib = Path(workdir) / f"loop_library_{seed}.pt"
    build_library(lib, terr, tiles.tolist(), start, T + 2 * model.cfg.fut)
    load_terrains.cache_clear()  # cached by path: a rebuilt library must not reuse an earlier call's terrains
    policy, pcfg = load_policy_from_checkpoint(teacher, dev)
    env = make_env(SuperMimicEnvCfgArgs(npz_files=[str(lib)], robot="dash", num_envs=n, compact_library=True, **goal_env_kwargs(pcfg),
                                        episode_length_s=1e5, action_noise_std=0.0, obs_noise=False, reset_noise=False, domain_randomization=False,
                                        sampling_mode="start", seed=seed, root_height_threshold=1e3, root_tilt_threshold=1e3, record_transitions=False),
                   device=dev)
    cmd = env.command_manager.get_term("motion")
    L = cmd.library
    cmd.forced_frame[:] = L.clip_start[torch.arange(n, device=dev)]
    obs, _ = env.reset()
    cmd.forced_frame[:] = -1
    origins = env.scene.env_origins
    rd = cmd.robot.data

    def robot_now():
        return torch.cat([cmd.robot_root_pos_w - origins, cmd.robot_root_quat_w, cmd.robot_joint_pos], -1)

    H = model.cfg.hist
    buf = robot_now()[:, None].repeat(1, max(H, REPLAN + 1), 1)
    scan = HeightScan(SimpleNamespace(terrains=terr), GEN_GRID).to(dev)
    nm = model.norm
    g = torch.Generator(device=dev).manual_seed(seed)
    Q, R, V = [], [], []
    with torch.no_grad():
        for t in range(T):
            if t % replan == 0:
                cur = buf[:, -1]
                origin, yaw0 = cur[:, :3], heading_yaw(cur[:, 3:7])
                h = buf[:, -H:]
                hist = to_canonical(h[..., :3], h[..., 3:7], h[..., 7:], origin, yaw0)
                s49 = torch.zeros(n, 49, device=dev)
                s49[:, :7] = cur[:, :7]
                hmap = (scan(s49, tiles) - origin[:, 2:3]).reshape(n, *GRID_SHAPE)
                c = command_path(buf, operator(buf, cmd_v) if cmd_v.dim() == 2 else operator(buf, cmd_v[:, t, :3], cmd_v[:, t, 3]), halflife)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    x = diff.sample(model, nm.frames(hist), nm.hmap(hmap), nm.cmd(c), torch.ones(n, device=dev), steps=steps, guidance=guidance, generator=g).float()
                pos, quat, q = from_canonical(nm.frames_inv(x), origin, yaw0)
                frames = with_velocities(cur, torch.cat([pos, quat, q], -1))
                idx = cmd.frame[:, None] + torch.arange(1, frames.shape[1] + 1, device=dev)
                L._ref[idx.reshape(-1)] = frames.reshape(-1, 49).to(L._ref.dtype)
            obs, _, _, _, _ = env.step(policy(obs["policy"], obs["goal"]))
            now = robot_now()
            buf = torch.cat([buf[:, 1:], now[:, None]], 1)
            Q.append(now)
            R.append(L._ref[cmd.frame, :25].float())
            V.append(torch.cat([rd.root_link_lin_vel_w, rd.root_link_ang_vel_w, rd.joint_vel], -1))
    env.close()
    for f in (lib, terrains_path(lib)):  # working files (~70 MB per 1000 envs)
        f.unlink(missing_ok=True)
    return torch.stack(Q, 1), torch.stack(R, 1), torch.stack(V, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gen_ckpt")
    ap.add_argument("teacher")
    ap.add_argument("out")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--guidance", type=float, default=1.0, help="classifier-free guidance on the command")
    ap.add_argument("--steps", type=int, default=20, help="DDIM steps per plan")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0, help="generator sampling and sim seed")
    ap.add_argument("--replan", type=int, default=REPLAN, help="frames between plans")
    ap.add_argument("--halflife", type=float, default=VEL_HALFLIFE, help="s, the command path's velocity spring")
    a = ap.parse_args()
    dev = a.device
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model, diff, ck = load(a.gen_ckpt, dev)
    sc = scenes()
    terr = [Terrain(s.hf, np.array([X0, Y0]), DX) for s in sc]
    S = a.samples
    n = len(sc) * S
    tiles = torch.arange(len(sc), device=dev).repeat_interleave(S)
    st = standing_frame(first_library(ck), dev)
    start = st.repeat(n, 1)
    start[:, 2] += torch.as_tensor(np.array([t.h(np.zeros(2)) for t in terr]), dtype=torch.float32, device=dev).repeat_interleave(S)
    cmd_v = torch.as_tensor(np.array([s.cmd for s in sc]), dtype=torch.float32, device=dev).repeat_interleave(S, 0)
    Q, R, _ = run_loop(model, diff, a.teacher, terr, tiles, start, cmd_v, a.seconds, out, dev, steps=a.steps, guidance=a.guidance, seed=a.seed,
                       replan=a.replan, halflife=a.halflife)
    scan = HeightScan(SimpleNamespace(terrains=terr), GEN_GRID).to(dev)
    feet = FootMetrics(scan, dev)
    res, total = score(sc, terr, Q, tiles, feet, S)
    off = (Q[..., :2] - R[..., :2]).norm(dim=-1)
    for k, s in enumerate(sc):
        res[s.name]["robot_vs_plan_xy_max_m"] = off[k * S : (k + 1) * S].amax(1).cpu().numpy().round(2).tolist()
    (out / "results.json").write_text(json.dumps({"gen_ckpt": a.gen_ckpt, "teacher": a.teacher, "step": ck["step"], "guidance": a.guidance, "replan": a.replan, "halflife": a.halflife, "success": f"{total}/{n}", "scenes": res}, indent=1))
    torch.save({"robot": Q.cpu(), "plan": R.cpu(), "scenes": [s.name for s in sc], "samples": S}, out / "traj.pt")
    if a.render:
        render_all(sc, terr, Q[::S].cpu().numpy(), R[::S].cpu().numpy(), out)


def render_all(sc, terr, Q, R, out: Path, w=640, h=360):
    sys.path.insert(0, str(REPO / "scripts/parc"))
    sys.path.insert(0, str(REPO / "scripts/media"))
    from render_hq import FPS, dash_model, encode, resample
    from render_policy import render

    vids = []
    for k, s in enumerate(sc):
        boxes = np.array([[0, 0, terr[k].floor - 0.05, 0.01, 0.01, 0.05]] + [list(b) for b in terr[k].boxes()])
        m = dash_model(boxes, w, h)
        for g in range(m.ngeom):  # terrain in a plain colour: with the floor's checker texture, tops vanish into it
            if m.geom(g).name.startswith("terrain_"):
                m.geom_matid[g] = -1
                m.geom_rgba[g] = (0.60, 0.66, 0.78, 1.0)
        q, r = resample(Q[k].astype(np.float64), 50, 1.0), resample(R[k].astype(np.float64), 50, 1.0)
        p = out / f"{k:02d}.mp4"
        encode(str(p), render(m, q, r, w, h, [s.name]), w, h)
        vids.append(p)
    cols = 4
    rows = math.ceil(len(vids) / cols)
    pad = cols * rows - len(vids)
    inputs = sum([["-i", str(v)] for v in vids], []) + ["-f", "lavfi", "-i", f"color=c=white:s={w}x{h}:r={FPS}"] * pad
    filt = "".join(f"[{i}:v]" for i in range(len(vids) + pad))
    layout = "|".join(f"{c * w}_{r_ * h}" for r_ in range(rows) for c in range(cols))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", f"{filt}xstack=inputs={len(vids) + pad}:layout={layout}:shortest=1",
                    "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", str(out / "grid.mp4")], check=True)


if __name__ == "__main__":
    main()
