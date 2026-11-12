"""Policy rollouts on their terrains (GPU env, nominal: no noise, no DR), rendered in the render_hq style with
the reference as a green ghost. One 60 fps video per clip.

  uv run python scripts/media/render_policy.py ckpt out_dir --lib LIB --clips name1 name2 ... [--labels ...] [--max-s 12]
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter1d

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/parc"))
from render_hq import FPS, dash_model, encode, resample  # noqa: E402
from render_strobe import section_cut  # noqa: E402

from supermimic.envs import SuperMimicEnvCfgArgs, make_env  # noqa: E402
from supermimic.eval.benchmark import goal_env_kwargs, load_policy_from_checkpoint  # noqa: E402
from supermimic.terrain.terrains import load_terrains  # noqa: E402
from mjlab.utils.lab_api.math import quat_apply_inverse  # noqa: E402


@torch.no_grad()
def rollouts(ckpt, lib, clips, max_steps, device="cuda:0", run_on=False):
    """With ``run_on`` the tracking terminations are off, so a clip plays to its end after the benchmark would
    have stopped it; ``fail_at`` is still when the benchmark criterion (root height error > 0.25 m or gravity-z
    difference > 0.8) first fires."""
    policy, pcfg = load_policy_from_checkpoint(ckpt, device)
    off = dict(root_height_threshold=1.0e3, root_tilt_threshold=1.0e3) if run_on else {}
    env = make_env(SuperMimicEnvCfgArgs(npz_files=[str(lib)], robot="dash", num_envs=len(clips), compact_library=True, **goal_env_kwargs(pcfg),
                                        episode_length_s=1.0e5, action_noise_std=0.0, obs_noise=False, reset_noise=False, domain_randomization=False,
                                        sampling_mode="start", seed=0, **off), device=device)
    try:
        cmd = env.command_manager.get_term("motion")
        L = cmd.library
        idx = [next(i for i, c in enumerate(L.clips) if c.name == n) for n in clips]
        cmd.forced_frame[:] = L.clip_start[torch.tensor(idx, device=device)]
        obs, _ = env.reset()
        cmd.forced_frame[:] = -1
        lens = [min(int(L.clips[i].length), max_steps) for i in idx]
        starts = [int(L.clip_start[i]) for i in idx]
        tiles = L.clip_tile[torch.tensor(idx, device=device)].cpu().numpy()
        origin = env.scene.env_origins.clone()
        state = lambda: torch.cat([cmd.robot_root_pos_w - origin, cmd.robot_root_quat_w, cmd.robot_joint_pos], 1)
        qs = [state().cpu().numpy()]
        alive = torch.ones(len(clips), dtype=torch.bool, device=device)
        fail_at = [None] * len(clips)
        for t in range(max(lens) - 1):
            obs, _, term, tout, _ = env.step(policy(obs["policy"], obs["goal"]))
            q = state()
            if run_on:
                g = cmd.robot.data.gravity_vec_w
                tilt = (quat_apply_inverse(cmd.root_quat_w, g)[:, 2] - quat_apply_inverse(cmd.robot_root_quat_w, g)[:, 2]).abs()
                bad = (((cmd.root_pos_w[:, 2] - cmd.robot_root_pos_w[:, 2]).abs() > 0.25) | (tilt > 0.8)) & cmd.track_mask
                for e in torch.nonzero(bad & alive).flatten().tolist():
                    if fail_at[e] is None and t + 1 < lens[e] - 1:
                        fail_at[e] = (t + 1) / 50
            for e in torch.nonzero(term & alive).flatten().tolist():
                if t + 1 < lens[e] - 1:
                    fail_at[e] = fail_at[e] if fail_at[e] is not None else (t + 1) / 50
            alive &= ~(term | tout)
            q = torch.where(alive[:, None], q, torch.as_tensor(qs[-1], device=device))  # auto-reset: hold the last pose
            qs.append(q.cpu().numpy())
            if not alive.any():
                break
    finally:
        env.close()
    qs = np.stack(qs, 1)
    return [qs[e, : lens[e]] for e in range(len(clips))], starts, lens, tiles, fail_at


def scene_boxes(ts, tile, Q):
    """Terrain boxes with the part between the camera (robot's right) and the path cut away, plus a tiny box
    whose top sets render_hq's ground at the terrain floor."""
    if tile < 0:
        return np.array([[0, 0, -0.05, 0.01, 0.01, 0.05]])
    floor = ts.terrains[tile].floor
    m0 = dash_model([], 64, 64)
    cut = section_cut(ts.boxes[tile], Q, m0, mujoco.MjData(m0)) if np.linalg.norm(Q[-1, :2] - Q[0, :2]) > 0.5 else list(ts.boxes[tile])
    return np.array([[0, 0, floor - 0.05, 0.01, 0.01, 0.05]] + [list(b) for b in cut])


def render(m, Q, R, w, h, lines, dist=3.2, elev=-12.0, fail_at=None):
    p = Q[:, :3]
    u = p[-1, :2] - p[0, :2]
    if np.linalg.norm(u) < 0.5:  # in place: the initial facing
        wq, x, y, z = Q[0, 3:7]
        u = np.array([1 - 2 * (y * y + z * z), 2 * (wq * z + x * y)])
    az = np.degrees(np.arctan2(u[1], u[0])) + 110  # travel to screen right, camera 20 deg toward the front
    look = np.stack([gaussian_filter1d(p[:, 0], 0.4 * FPS, mode="nearest"), gaussian_filter1d(p[:, 1], 0.4 * FPS, mode="nearest"),
                     gaussian_filter1d(p[:, 2], 0.8 * FPS, mode="nearest") - 0.1], 1)
    d, dg = mujoco.MjData(m), mujoco.MjData(m)
    r = mujoco.Renderer(m, h, w)
    cam = mujoco.MjvCamera()
    cam.distance, cam.elevation, cam.azimuth = dist, elev, az
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[[0, 1, 2]] = 1
    opt.sitegroup[:] = 0
    pert = mujoco.MjvPerturb()
    font = ImageFont.truetype("DejaVuSans-Bold.ttf", 30)
    font2 = ImageFont.truetype("DejaVuSans.ttf", 24)
    key = m.light("key").id
    kd = m.light_dir[key] / np.linalg.norm(m.light_dir[key])
    for t in range(len(Q)):
        m.light_pos[key] = look[t] - 20 * kd
        d.qpos[:] = Q[t]
        mujoco.mj_forward(m, d)
        cam.lookat[:] = look[t]
        r.update_scene(d, cam, opt)
        dg.qpos[:] = R[min(t, len(R) - 1)]
        mujoco.mj_forward(m, dg)
        n0 = r.scene.ngeom
        mujoco.mjv_addGeoms(m, dg, opt, pert, mujoco.mjtCatBit.mjCAT_DYNAMIC, r.scene)
        for g in range(n0, r.scene.ngeom):
            r.scene.geoms[g].rgba[:] = (0.1, 0.8, 0.3, 0.16)
        r.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = 1
        im = Image.fromarray(r.render())
        dr = ImageDraw.Draw(im)
        dr.text((32, 24), lines[0], fill=(30, 34, 40), font=font)
        off = float(np.linalg.norm(Q[t, :2] - R[min(t, len(R) - 1), :2]))
        for k, s in enumerate(lines[1:] + [f"horizontal offset from the reference: {off:.2f} m"]):
            dr.text((32, 66 + 32 * k), s, fill=(200, 30, 30) if k == len(lines) - 1 and off > 0.5 else (60, 66, 76), font=font2)
        if fail_at is not None and t / FPS >= fail_at:
            dr.text((32, 66 + 32 * len(lines) + 8), f"benchmark failure at {fail_at:.1f} s (shown running on)", fill=(200, 30, 30), font=font)
        yield np.asarray(im)
    r.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("out")
    ap.add_argument("--lib", default=str(REPO / "data/library/dash_flat_terrain_v4_parc_compact.pt"))
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+")
    ap.add_argument("--tag", default="teacher")
    ap.add_argument("--max-s", type=float, default=12.0)
    ap.add_argument("--run-on", action="store_true", help="no tracking terminations: show what happens after a failure")
    ap.add_argument("--only", type=int, nargs="*", help="render only these (1-based) clips")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    qs, starts, lens, tiles, fail_at = rollouts(a.ckpt, a.lib, a.clips, int(a.max_s * 50), run_on=a.run_on)
    ts = load_terrains(a.lib)
    ref = torch.load(a.lib, map_location="cpu", weights_only=False, mmap=True)["ref"]
    for e, name in enumerate(a.clips):
        if a.only and e + 1 not in a.only:
            continue
        nq = qs[e].shape[1]
        R = ref[starts[e] : starts[e] + lens[e], :nq].float().numpy()
        Q, Rr = resample(qs[e].astype(float), 50, 1.0), resample(R.astype(float), 50, 1.0)
        m = dash_model(scene_boxes(ts, int(tiles[e]), Q), a.width, a.height)
        label = a.labels[e] if a.labels else name.split("/")[-1].removeprefix("dash_")
        status = (f"fails at {fail_at[e]:.1f} s" if fail_at[e] is not None else "tracked to the end (passes the benchmark)")
        lines = [label, f"{a.tag}, real time, {lens[e] / 50:.1f} s, {status}", "green ghost: reference motion"]
        f = out / f"{e + 1:02d}_{name.split('/')[-1].removeprefix('dash_')}.mp4"
        encode(str(f), render(m, Q, Rr, a.width, a.height, lines, fail_at=fail_at[e] if a.run_on else None), a.width, a.height)
        print(f"wrote {f}  ({status})", flush=True)


if __name__ == "__main__":
    main()
