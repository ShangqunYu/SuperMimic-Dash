"""Benchmark a tracking policy on a terrain library (every clip from its start, on its tile) and
render a few clips from the GPU env (the env's compiled model holds the tiles).

  uv run python scripts/terrain/eval_terrain.py --ckpt logs/dash_e2_hwdr/ckpt_latest.pt
"""

from __future__ import annotations

import collections
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
import torch
import tyro
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.eval.benchmark import goal_env_kwargs, load_policy_from_checkpoint, run_benchmark, summarize


def group(name: str) -> str:
    n = name.removeprefix("dash_")
    for k in ("come_up", "come_down", "jump_on", "jump_off", "stairs_climbing_up", "stairs_climbing_down", "jump_over_obstacle", "avoid_obstacle", "jump_twice", "jump_ff"):
        if k in n:
            return k
    return "other"


@torch.no_grad()
def rollout_video(ckpt: Path, lib: Path, clips: list[str], out: Path, device: str, width=480, height=360, notes: list[str] | None = None, ghost: bool = True):
    """Policy rollouts of ``clips`` on their tiles, 3 per row; with ``ghost`` the reference is drawn
    as a translucent copy of the robot. ``notes`` are appended to each clip's label."""
    policy, pcfg = load_policy_from_checkpoint(ckpt, device)
    env = make_env(SuperMimicEnvCfgArgs(npz_files=[str(lib)], robot="dash", num_envs=len(clips), compact_library=True, **goal_env_kwargs(pcfg),
                                        episode_length_s=1.0e5, action_noise_std=0.0, obs_noise=False, reset_noise=False, domain_randomization=False,
                                        sampling_mode="start", seed=0), device=device)
    try:
        cmd = env.command_manager.get_term("motion")
        lib_ = cmd.library
        idx = [next(i for i, c in enumerate(lib_.clips) if c.name == n) for n in clips]
        cmd.forced_frame[:] = lib_.clip_start[torch.tensor(idx, device=device)]
        obs, _ = env.reset()
        cmd.forced_frame[:] = -1
        T = int(max(lib_.clips[i].length for i in idx))
        starts = [int(lib_.clip_start[i]) for i in idx]
        lens = [int(lib_.clips[i].length) for i in idx]
        origin = env.scene.env_origins.clone()
        tile_of = cmd.clip_tile.cpu().numpy()
        qs, alive = [], torch.ones(len(clips), dtype=torch.bool, device=device)
        for _ in range(T):
            q_prev = torch.cat([cmd.robot_root_pos_w - origin, cmd.robot_root_quat_w, cmd.robot_joint_pos], 1)
            obs, _, term, tout, _ = env.step(policy(obs["policy"], obs["goal"]))
            q = torch.cat([cmd.robot_root_pos_w - origin, cmd.robot_root_quat_w, cmd.robot_joint_pos], 1)
            done = term | tout
            # A finished env is auto-reset onto another clip: hold its last pose instead.
            q = torch.where((done | ~alive)[:, None], q_prev if not qs else torch.as_tensor(qs[-1], device=device), q)
            alive &= ~done
            qs.append(q.cpu().numpy())
            if not alive.any():
                break
        qs = np.stack(qs, 1)  # (E, T, nq), tile frame
    finally:
        env.close()
    from check_tiles import tile_model

    from supermimic.terrain.tiles import load_tiles

    tile_boxes = load_tiles(lib)
    font = ImageFont.truetype("DejaVuSans.ttf", 13)
    tiles = []
    nj = qs.shape[2] - 7
    pack_ref = torch.load(lib, map_location="cpu", weights_only=False)["ref"] if ghost else None
    for e, name in enumerate(clips):
        if ghost:
            st = starts[e]
            ref = pack_ref[st : st + lens[e], : 7 + nj].float().numpy()
        m = tile_model(tile_boxes[tile_of[e]] if tile_of[e] >= 0 else np.zeros((0, 6)), width, height)
        d = mujoco.MjData(m)
        dg = mujoco.MjData(m)
        r = mujoco.Renderer(m, height, width)
        vopt, pert = mujoco.MjvOption(), mujoco.MjvPerturb()
        cam = mujoco.MjvCamera()
        cam.distance, cam.elevation, cam.azimuth = 2.6, -15.0, 90.0
        frames = []
        look = qs[e, 0, :3].copy()
        for t in range(qs.shape[1]):
            d.qpos[: qs.shape[2]] = qs[e, t]
            mujoco.mj_forward(m, d)
            look = 0.9 * look + 0.1 * qs[e, t, :3]
            cam.lookat[:] = [look[0], look[1], look[2] - 0.25]
            r.update_scene(d, cam)
            if ghost:
                dg.qpos[: 7 + nj] = ref[min(t + 1, len(ref) - 1)]
                mujoco.mj_forward(m, dg)
                n0 = r.scene.ngeom
                mujoco.mjv_addGeoms(m, dg, vopt, pert, mujoco.mjtCatBit.mjCAT_DYNAMIC, r.scene)
                for g in range(n0, r.scene.ngeom):
                    r.scene.geoms[g].rgba[:] = (0.2, 0.8, 0.3, 0.3)
            img = Image.fromarray(r.render())
            label = name.removeprefix("dash_")[:52] + (f"  {notes[e]}" if notes else "")
            ImageDraw.Draw(img).text((6, 4), label, fill=(10, 10, 10), font=font)
            frames.append(np.asarray(img))
        tiles.append(frames)
    n = len(tiles)
    cols = 3
    rows = (n + cols - 1) // cols
    L = max(len(f) for f in tiles)
    out.parent.mkdir(parents=True, exist_ok=True)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{cols * width}x{rows * height}", "-r", "50", "-i", "-",
                           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(out)], stdin=subprocess.PIPE)
    for t in range(L):
        canvas = np.zeros((rows * height, cols * width, 3), dtype=np.uint8)
        for i, f in enumerate(tiles):
            canvas[(i // cols) * height : (i // cols + 1) * height, (i % cols) * width : (i % cols + 1) * width] = f[min(t, len(f) - 1)]
        ff.stdin.write(canvas.tobytes())
    ff.stdin.close()
    ff.wait()
    print("wrote", out)


def main(ckpt: Path = Path("logs/dash_e2_hwdr/ckpt_latest.pt"), lib: Path = Path("data/library/dash_terrain_compact.pt"), out: Path = Path("logs/terrain_eval"),
         device: str = "cuda:0", num_envs: int = 1024, video: list[str] | None = None, bench: bool = True) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if bench:
        policy, pcfg = load_policy_from_checkpoint(ckpt, device)
        n_clips = len(torch.load(lib, map_location="cpu", weights_only=False)["names"])
        res = run_benchmark(policy, [str(lib)], device=device, num_envs=num_envs, **goal_env_kwargs(pcfg), robot="dash", num_clips=n_clips, verbose=False)
        by = collections.defaultdict(list)
        for r in res:
            by[group(r.name)].append(r)
        print(f"{ckpt}: {summarize(res)['failed']}/{len(res)} failed")
        for g, rs in sorted(by.items()):
            print(f"  {g:22s} n={len(rs):5d} failed {sum(r.failed for r in rs):5d} ({100 * np.mean([r.failed for r in rs]):5.1f}%)  mean norm return {np.mean([r.norm_return for r in rs]):.3f}")
        json.dump({"ckpt": str(ckpt), "clips": [r.__dict__ for r in res]}, open(out / f"bench_{ckpt.parent.name}.json", "w"))
    if video:
        rollout_video(ckpt, lib, video, out / f"rollout_{ckpt.parent.name}.mp4", device)


if __name__ == "__main__":
    tyro.cli(main)
