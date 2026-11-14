"""PARC-style data augmentation: the generator + tracking teacher run in physics on random procedural terrains,
and the runs that succeed are kept as new reference clips (the robot's own motion, physically feasible for Dash).

  python scripts/generator/augment.py gen_ckpt teacher_ckpt out.pt [--envs 1024] [--rounds 4] [--seconds 12] [--seed 0]

Terrain families (each env its own terrain, all along +x, the robot starts at x = 0 facing +x): stairs up / down
(riser 8-19 cm, tread 24-36 cm, 3-8 steps), a box in the way (12-32 cm high, 20-50 cm deep, 0.6-2 m wide), a
platform up / down (8-32 cm), flat ground, on a 4 cm heightfield 15 m long; terrain runs vx 0.3-1.0 m/s (a quarter
with vy up to 0.3 m/s), flat runs any walking command (half turning); heading held by the operator where not turning
(rollout_generator.operator).
Kept: past the feature on the right level (flat: anywhere), no foot > 4 cm into the terrain, root 0.35 to 1.1 m (+ any drop) above
the ground, heading within 30 deg (unless turning), robot within 0.3 m of the plan throughout. The first --trim s
(1.0) are cut: every run starts from the same settled standing pose with the teacher lagging the first plans, and
5,000 copies of that start taught gen_v3 to hesitate from a standstill. Writes a packed library (out.pt,
names aug/<family>/<params>/<round>_<env>) with its terrains (out.terrains.npz) and out.json (kept per family).
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from closed_loop import run_loop  # noqa: E402
from rollout_generator import first_library, load, standing_frame  # noqa: E402

from supermimic.generator.data import GEN_GRID  # noqa: E402
from supermimic.generator.metrics import FootMetrics  # noqa: E402
from supermimic.kinematics.rotations import heading_yaw  # noqa: E402
from supermimic.terrain.height_scan import HeightScan  # noqa: E402
from supermimic.terrain.terrains import Terrain, save_terrains, terrains_path  # noqa: E402

FAMILIES = {"stairs_up": 0.22, "stairs_down": 0.17, "box": 0.26, "platform_up": 0.1, "platform_down": 0.1, "flat": 0.15}
# 4 cm cells (physics boxes and scans both come from this heightfield), 15 m along x: 12 s at up to 1 m/s
DX, X0, X1, Y0, Y1 = 0.04, -1.0, 14.0, -1.2, 1.2


def grid():
    x = X0 + DX * np.arange(round((X1 - X0) / DX) + 1)
    y = Y0 + DX * np.arange(round((Y1 - Y0) / DX) + 1)
    return np.meshgrid(x, y, indexing="ij")


def sample_scene(rng: np.random.Generator):
    """-> family, name params, heightfield, start ground, target top, x the root must pass (None: flat), command
    (vx, vy, wz; wz 0 = heading held). Terrain runs: vx 0.3-1.0 m/s, a quarter with vy up to 0.3 m/s; flat runs: any
    walking command, half turning."""
    X, Y = grid()
    fam = rng.choice(list(FAMILIES), p=list(FAMILIES.values()))
    vx = float(rng.uniform(0.3, 1.0))
    vy = float(rng.uniform(-0.3, 0.3)) if rng.random() < 0.25 else 0.0
    wz = 0.0
    if fam == "flat":
        vx, vy = float(rng.uniform(-0.4, 1.2)), float(rng.uniform(-0.4, 0.4))
        wz = float(rng.uniform(-0.8, 0.8)) if rng.random() < 0.5 else 0.0
        hf = np.zeros_like(X)
        par = "flat"
        past = None
    elif fam.startswith("stairs"):
        r, t, n, x0 = rng.uniform(0.08, 0.19), rng.uniform(0.24, 0.36), int(rng.integers(3, 9)), rng.uniform(0.8, 2.0)
        k = np.clip(np.floor((X - x0) / t) + 1, 0, n)
        hf = k * r if fam == "stairs_up" else (n - k) * r
        par = f"r{r * 100:.0f}_t{t * 100:.0f}_n{n}"
        past = x0 + n * t + 0.2
    elif fam == "box":
        h, d, w, x0 = rng.uniform(0.12, 0.32), rng.uniform(0.2, 0.5), rng.uniform(0.6, 2.0), rng.uniform(1.0, 2.5)
        hf = np.where((X >= x0) & (X < x0 + d) & (np.abs(Y) < w / 2), h, 0.0)
        par = f"h{h * 100:.0f}_d{d * 100:.0f}_w{w * 100:.0f}"
        past = x0 + d + 0.3
    else:
        h, x0 = rng.uniform(0.08, 0.32), rng.uniform(0.8, 2.0)
        hf = np.where(X >= x0, h, 0.0) if fam == "platform_up" else np.where(X < x0, h, 0.0)
        par = f"h{h * 100:.0f}"
        past = x0 + 0.3
    hf = hf.astype(np.float32)
    ter = Terrain(hf, np.array([X0, Y0]), DX)
    return fam, par, ter, float(ter.h(np.zeros(2))), float(ter.h(np.array([X.max() - 0.1, 0.0]))), past, (vx, vy, wz)


def keep_mask(terr, tiles, Q, R, past, top, start_ground, turning, dev):
    scan = HeightScan(SimpleNamespace(terrains=terr), GEN_GRID).to(dev)
    feet = FootMetrics(scan, dev)
    _, gap = feet.gap(Q[..., :3], Q[..., 3:7], Q[..., 7:], tiles)
    pen = (-gap).clamp_min(0).amax((1, 2)).cpu().numpy()
    xy = Q[..., :2].reshape(len(tiles), -1)
    ground = scan.lookup(xy[:, 0::2], xy[:, 1::2], tiles).cpu().numpy()  # (n, T)
    above = Q[..., 2].cpu().numpy() - ground
    drop = np.maximum(0.0, start_ground - top)
    yaw = heading_yaw(Q[..., 3:7])
    dev_deg = np.degrees(torch.atan2(torch.sin(yaw), torch.cos(yaw)).abs().amax(1).cpu().numpy())
    off = (Q[..., :2] - R[..., :2]).norm(dim=-1).amax(1).cpu().numpy()
    end_x = Q[:, -1, 0].cpu().numpy()
    ok = (pen < 0.04) & (above.min(1) > 0.35) & (above.max(1) < 1.1 + drop) & (np.abs(ground[:, -1] - top) < 0.02)
    flat = np.isnan(past)
    ok &= np.where(flat, True, end_x > np.nan_to_num(past)) & (turning | (dev_deg < 30)) & (off < 0.3)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gen_ckpt")
    ap.add_argument("teacher")
    ap.add_argument("out")
    ap.add_argument("--envs", type=int, default=1024)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trim", type=float, default=1.0, help="s cut from the start of each kept clip (the settled standing start, the teacher lagging the first plans)")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dev = a.device
    out = Path(a.out)
    work = out.with_suffix("")
    work.mkdir(parents=True, exist_ok=True)
    model, diff, ck = load(a.gen_ckpt, dev)
    st = standing_frame(first_library(ck), dev)
    rng = np.random.default_rng(a.seed)
    names, refs, terrains = [], [], []
    stats = collections.defaultdict(lambda: [0, 0])
    for rd in range(a.rounds):
        sc = [sample_scene(rng) for _ in range(a.envs)]
        terr = [s[2] for s in sc]
        tiles = torch.arange(a.envs, device=dev)
        start = st.repeat(a.envs, 1)
        start[:, 2] += torch.tensor([s[3] for s in sc], device=dev)
        cmd_v = torch.tensor([list(s[6]) for s in sc], device=dev)
        Q, R, V = run_loop(model, diff, a.teacher, terr, tiles, start, cmd_v, a.seconds, work, dev, seed=a.seed * 1000 + rd)
        ok = keep_mask(terr, tiles, Q, R, np.array([np.nan if s[5] is None else s[5] for s in sc]), np.array([s[4] for s in sc]),
                       np.array([s[3] for s in sc]), np.array([s[6][2] != 0 for s in sc]), dev)
        for e, s in enumerate(sc):
            stats[s[0]][0] += int(ok[e])
            stats[s[0]][1] += 1
            if ok[e]:
                names.append(f"aug/{s[0]}/{s[1]}_v{s[6][0] * 100:.0f}_{s[6][1] * 100:.0f}_{s[6][2] * 100:.0f}/{a.seed}_{rd}_{e}")
                refs.append(torch.cat([Q[e], V[e]], -1)[int(a.trim * 50) :].cpu())
                terrains.append(s[2].padded(1.0))
        print(f"round {rd}: kept {int(ok.sum())}/{a.envs}; so far " + ", ".join(f"{k} {v[0]}/{v[1]}" for k, v in sorted(stats.items())), flush=True)
    lens = [len(r) for r in refs]
    starts = np.cumsum([0] + lens[:-1]).tolist()
    torch.save({"ref": torch.cat(refs).half(), "names": names, "starts": starts, "lengths": lens, "robot": "dash",
                "clip_tile": list(range(len(names))), "terrain_names": names}, out)
    save_terrains(terrains_path(out), names, terrains)
    out.with_suffix(".json").write_text(json.dumps({"gen_ckpt": a.gen_ckpt, "teacher": a.teacher, "seconds": a.seconds,
                                                    "kept": {k: f"{v[0]}/{v[1]}" for k, v in sorted(stats.items())}, "clips": len(names)}, indent=1))
    print(f"{len(names)} clips ({sum(lens) / 50 / 3600:.2f} h) -> {out}")


if __name__ == "__main__":
    main()
