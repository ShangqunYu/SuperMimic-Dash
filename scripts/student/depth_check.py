"""Check the depth student's camera (envs/mdp/depth.py) in physics, on rollout_generator's 15 scenes.

  python scripts/student/depth_check.py teacher_ckpt out_dir [--bench 1024 4096]

The robot stands --gap m before each scene's first edge (the teacher tracks a standing reference) and the
RealSense renders.
Every pixel is checked against depth ray-marched through the scene's heightfield: pixels on the terrain
must agree, pixels nearer than the terrain are the robot's own body. Writes depth.png (render, march,
clean and noisy network input per scene) and check.json. --bench N...: env step time with and without the
camera at N envs.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/generator"))
from closed_loop import build_library  # noqa: E402
from rollout_generator import DX, X0, Y0, first_library, scenes, standing_frame  # noqa: E402

from supermimic.envs import SuperMimicEnvCfgArgs, make_env  # noqa: E402
from supermimic.envs.mdp.depth import DEPTH, SENSOR, _dropouts, corrupt_depth, process_depth  # noqa: E402
from supermimic.eval.benchmark import goal_env_kwargs, load_policy_from_checkpoint  # noqa: E402
from supermimic.terrain.terrains import Terrain, load_terrains, terrains_path  # noqa: E402

GEN_CKPT = REPO / "logs/gen_v6/ckpt_0040000.pt"  # only for the standing pose of its library


def approach_x(terr: Terrain, gap: float) -> float:
    """Start x `gap` m before the first height change along +x (0 on flat ground)."""
    xs = np.arange(0.0, 8.0, 0.02)
    hs = terr.h(np.stack([xs, np.zeros_like(xs)], -1))
    edge = np.flatnonzero(np.abs(hs - hs[0]) > 0.02)
    return max(float(xs[edge[0]]) - gap, 0.0) if len(edge) else 0.0


def scene_env(teacher: str, workdir: Path, n_per_scene: int, depth: bool, dev: str, gap: float | None = None, noisy: bool = False):
    """Envs standing at each scene's start, or `gap` m before its first edge; noisy: the student's training
    conditions (observation noise with the depth's noise, dropouts and latency, robot + hardware + camera DR,
    actuator delay)."""
    sc = scenes()
    terr = [Terrain(s.hf, np.array([X0, Y0]), DX) for s in sc]
    n = len(sc) * n_per_scene
    tiles = np.repeat(np.arange(len(sc)), n_per_scene)
    st = standing_frame(first_library(torch.load(GEN_CKPT, map_location="cpu", weights_only=False)), dev)
    start = st.repeat(n, 1)
    x0 = np.array([approach_x(terr[k], gap) if gap is not None else 0.0 for k in tiles])
    start[:, 0] += torch.as_tensor(x0, dtype=torch.float32, device=dev)
    start[:, 2] += torch.as_tensor(np.array([terr[k].h(np.array([x, 0.0])) for k, x in zip(tiles, x0)]), dtype=torch.float32, device=dev)
    lib = workdir / "depth_check_library.pt"
    build_library(lib, terr, tiles.tolist(), start, 400)
    load_terrains.cache_clear()
    policy, pcfg = load_policy_from_checkpoint(teacher, dev)
    env = make_env(SuperMimicEnvCfgArgs(npz_files=[str(lib)], robot="dash", num_envs=n, compact_library=True, **goal_env_kwargs(pcfg),
                                        episode_length_s=1e5, action_noise_std=0.0, obs_noise=noisy, reset_noise=False, domain_randomization=noisy,
                                        hardware_dr=noisy, command_delay=noisy, sampling_mode="start", seed=0, root_height_threshold=1e3,
                                        root_tilt_threshold=1e3, record_transitions=False, depth=depth), device=dev)
    cmd = env.command_manager.get_term("motion")
    cmd.forced_frame[:] = cmd.library.clip_start[torch.arange(n, device=dev)]
    obs, _ = env.reset()
    cmd.forced_frame[:] = -1
    return env, policy, obs, sc, terr, tiles, [lib, terrains_path(lib)]


def march(terr: Terrain, pos: np.ndarray, rot: np.ndarray, fovy: float, h: int, w: int, t_max: float = 4.0, dt: float = 0.004) -> np.ndarray:
    """Planar depth (h, w) of a MuJoCo camera (pos, rot: columns = camera axes; looks along -z) through the
    heightfield; 0 where nothing is hit within t_max. Rays are scaled to camera z = -1, so the marched
    parameter is the planar depth."""
    f = (h / 2) / math.tan(math.radians(fovy) / 2)
    u = (np.arange(w) + 0.5 - w / 2) / f
    v = -(np.arange(h) + 0.5 - h / 2) / f
    dirs = np.stack(np.broadcast_arrays(u[None, :], v[:, None], -np.ones((h, w))), -1) @ rot.T  # (h, w, 3) world
    ts = np.arange(0.05, t_max, dt)
    pts = pos + ts[:, None, None, None] * dirs[None]  # (T, h, w, 3)
    below = pts[..., 2] <= terr.h(pts[..., :2].reshape(-1, 2)).reshape(pts.shape[:3])
    hit = below.any(0)
    first = below.argmax(0)
    # Linear refinement between the last sample above and the first below.
    t1 = ts[first]
    t0 = np.maximum(t1 - dt, ts[0])
    p0 = pos + t0[..., None] * dirs
    p1 = pos + t1[..., None] * dirs
    g0 = p0[..., 2] - terr.h(p0[..., :2].reshape(-1, 2)).reshape(h, w)
    g1 = p1[..., 2] - terr.h(p1[..., :2].reshape(-1, 2)).reshape(h, w)
    frac = np.where(g0 - g1 > 1e-9, g0 / np.maximum(g0 - g1, 1e-9), 1.0)
    return np.where(hit, t0 + np.clip(frac, 0, 1) * (t1 - t0), 0.0)


def check(a, dev: str) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    env, policy, obs, sc, terr, tiles, tmp = scene_env(a.teacher, out, 1, True, dev, gap=a.gap, noisy=a.noisy)
    with torch.no_grad():
        for _ in range(25):
            obs, *_ = env.step(policy(obs["policy"], obs["goal"]))
    sensor = env.scene[SENSOR]
    raw = sensor.data.depth[..., 0].float().cpu().numpy()  # (n, h, w), 0 = no hit
    ci = sensor.camera_idx
    cam_pos = env.sim.data.cam_xpos[:, ci].cpu().numpy() - env.scene.env_origins.cpu().numpy()
    cam_rot = env.sim.data.cam_xmat[:, ci].reshape(-1, 3, 3).cpu().numpy()
    fovy = env.sim.model.cam_fovy.cpu().numpy()  # (ncam,), or (n, ncam) under camera DR
    fovy = fovy[:, ci] if fovy.ndim == 2 else np.full(len(raw), fovy[ci])
    n, h, w = raw.shape
    res, ana = {}, []
    for k in range(n):
        m = march(terr[tiles[k]], cam_pos[k], cam_rot[k], float(fovy[k]), h, w)
        ana.append(m)
        both = (raw[k] > 0) & (m > 0)
        body = both & (raw[k] < m - 0.05)
        ground = both & ~body
        err = np.abs(raw[k] - m)[ground]
        res[sc[tiles[k]].name] = {"terrain_px": int(ground.sum()), "body_px": int(body.sum()), "missed_px": int(((raw[k] > 0) != (m > 0)).sum()),
                                  "err_median_cm": round(float(np.median(err)) * 100, 2), "err_p95_cm": round(float(np.percentile(err, 95)) * 100, 2),
                                  "frac_err_gt_3cm": round(float((err > 0.03).mean()), 4), "cam_height_m": round(float(cam_pos[k, 2] - terr[tiles[k]].h(cam_pos[k, None, :2])[0]), 3)}
        print(sc[tiles[k]].name, res[sc[tiles[k]].name], flush=True)
    depth_obs = obs["depth"].reshape(n, DEPTH.out_height, DEPTH.out_width).cpu().numpy()
    d = torch.as_tensor(raw, device=dev)
    d = torch.where(d > 0, d, DEPTH.depth_range[1])
    noisy = process_depth(corrupt_depth(d), DEPTH)[:, 0].cpu().numpy()
    edges = _dropouts(d, dataclasses.replace(DEPTH, edge_drop=1.0, blob_prob=0.0)).float().cpu().numpy()
    for k in range(n):
        res[sc[tiles[k]].name]["edge_frac"] = round(float(edges[k].mean()), 4)
    print({sc[tiles[k]].name[:20]: res[sc[tiles[k]].name]["edge_frac"] for k in range(n)}, flush=True)
    sfx = "_noisy" if a.noisy else ""
    (out / f"check{sfx}.json").write_text(json.dumps({"fovy": fovy.round(2).tolist(), "render": [h, w], "noisy": a.noisy, "scenes": res}, indent=1))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = [("render (m)", raw, 0, 2.0), ("heightfield march (m)", np.stack(ana), 0, 2.0), ("edge pixels (can drop out)", edges, 0, 1),
            ("depth observation" + (" (noisy, delayed)" if a.noisy else ""), depth_obs, 0, 1), ("network input, noisy", noisy, 0, 1)]
    fig, ax = plt.subplots(n, len(cols), figsize=(16, 2.2 * n))
    for k in range(n):
        for j, (title, img, lo, hi) in enumerate(cols):
            ax[k, j].imshow(img[k], cmap="gray" if j == 2 else "turbo", vmin=lo, vmax=hi)
            ax[k, j].set_xticks([])
            ax[k, j].set_yticks([])
            if k == 0:
                ax[k, j].set_title(title, fontsize=10)
        ax[k, 0].set_ylabel(sc[tiles[k]].name[:22], fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"depth{sfx}.png", dpi=80)
    env.close()
    for f in tmp:
        f.unlink(missing_ok=True)


def bench(a, dev: str) -> None:
    out = Path(a.out)
    rows = {}
    for n in a.bench:
        for depth in (False, True):
            env, policy, obs, *_, tmp = scene_env(a.teacher, out, math.ceil(n / 15), depth, dev)
            with torch.no_grad():
                for _ in range(10):
                    obs, *_ = env.step(policy(obs["policy"], obs["goal"]))
                torch.cuda.synchronize()
                t = time.perf_counter()
                for _ in range(50):
                    obs, *_ = env.step(policy(obs["policy"], obs["goal"]))
                torch.cuda.synchronize()
            ms = (time.perf_counter() - t) / 50 * 1000
            rows[f"{env.num_envs} envs, depth {depth}"] = round(ms, 2)
            print(f"{env.num_envs} envs, depth {depth}: {ms:.2f} ms / step", flush=True)
            env.close()
            for f in tmp:
                f.unlink(missing_ok=True)
    (out / "bench.json").write_text(json.dumps(rows, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("teacher")
    ap.add_argument("out")
    ap.add_argument("--bench", type=int, nargs="*")
    ap.add_argument("--gap", type=float, default=0.8, help="m before each scene's first edge to stand")
    ap.add_argument("--noisy", action="store_true", help="the student's training conditions (noise, latency, DR, delay)")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    if a.bench:
        bench(a, a.device)
    else:
        check(a, a.device)


if __name__ == "__main__":
    main()
