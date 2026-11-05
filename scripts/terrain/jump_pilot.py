"""Jump pilot: put a box under jumps that already clear it (no motion edit).

For each clip, every foot-sole and shin point of every frame is projected on the travel direction d
(start -> end of the root path). The clear height at s is the lowest such point over all frames;
the box goes where a ``depth``-long window has the most clearance, ``margin`` below it, capped at
``max_height``, ``width`` wide across the path.

Clips: the no-hands obstacle jumps (retargeted into data/jump_pilot/raw), the flat running jumps
and double jumps (dash_retarget/raw on the SSD), and the forward leaps that clear the most (screened
on the 50 fps library npz).

  uv run python scripts/terrain/jump_pilot.py
"""

from __future__ import annotations

import csv
import glob
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
import tyro

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pilot_terrain import CELL, HUMAN_BONES, render_clip, to_qpos, track  # noqa: E402

from supermimic.robots import get_robot  # noqa: E402

SSD_RAW = Path("/media/shangqunyu/Dash/dash_retarget/raw")
NPZ = Path("data/npz/dash_bones_seed")


def clearance_profile(qpos, fe, sp, foot_r, shin_r):
    d = qpos[-1, :2] - qpos[0, :2]
    d = d / max(np.linalg.norm(d), 1e-9)
    u = np.linspace(0, 1, 9)[None, None, :, None]
    sole = fe[:, :, :1] * (1 - u) + fe[:, :, 1:] * u
    s = np.concatenate([((sole[..., :2] - qpos[0, :2]) @ d).ravel(), ((sp[..., :2] - qpos[0, :2]) @ d).ravel()])
    z = np.concatenate([(sole[..., 2] - foot_r).ravel(), (sp[..., 2] - shin_r).ravel()])
    s0 = s.min()
    grid = s0 + CELL * np.arange(int((s.max() - s0) / CELL) + 1)
    clear = np.full(len(grid), 10.0)
    np.minimum.at(clear, np.clip(((s - s0) / CELL).astype(int), 0, len(grid) - 1), z)
    return d, grid, clear


def place_box(grid, clear, depth):
    k = max(1, int(round(depth / CELL)))
    win = np.array([clear[i : i + k].min() for i in range(len(grid) - k)]) if len(grid) > k else np.array([0.0])
    i = int(np.argmax(win))
    return grid[i], grid[i] + depth, float(win[i])


def box_grid(qpos, d, s_lo, s_hi, h, width):
    lo = qpos[:, :2].min(0) - 1.0
    hi = qpos[:, :2].max(0) + 1.0
    nx, ny = int(np.ceil((hi[0] - lo[0]) / CELL)) + 1, int(np.ceil((hi[1] - lo[1]) / CELL)) + 1
    X, Y = np.meshgrid(lo[0] + CELL * np.arange(nx), lo[1] + CELL * np.arange(ny))
    rel = np.stack([X - qpos[0, 0], Y - qpos[0, 1]], -1)
    s = rel @ d
    lat = rel @ np.array([-d[1], d[0]])
    # Centre the box across the path where the root crosses it.
    rs = (qpos[:, :2] - qpos[0, :2]) @ d
    t = int(np.argmin(np.abs(rs - (s_lo + s_hi) / 2)))
    lat0 = (qpos[t, :2] - qpos[0, :2]) @ np.array([-d[1], d[0]])
    H = np.where((s >= s_lo) & (s <= s_hi) & (np.abs(lat - lat0) <= width / 2), h, 0.0)
    return H, (lo[0], lo[1])


def load(path: Path):
    """-> qpos at 60 fps, optional mocap (+ joint names, skeleton)."""
    if path.suffix == ".npy":
        dat = np.load(path, allow_pickle=True).item()
        return to_qpos(dat["q"])[::2], dat
    d = np.load(path)
    q = np.concatenate([d["body_pos_w"][:, 0], d["body_quat_w"][:, 0], d["joint_pos"]], 1)
    return q, None


def screen_leaps(m, n_best: int, depth: float, min_clear: float = 0.0):
    files = [f for pat in ("dash_jump_ff_180_R_*", "dash_neutral_jump_ff_180_R_*") for f in glob.glob(str(NPZ / f"{pat}.npz")) if not f.endswith("_M.npz")]
    scored = []
    for f in files:
        q, _ = load(Path(f))
        fe, sp, fr, sr = track(m, q)
        d, grid, clear = clearance_profile(q, fe, sp, fr, sr)
        # Keep forward leaps: travel within 30 deg of the torso heading at the start.
        w, x, y, z = q[0, 3:7]
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        if np.dot(d, [np.cos(yaw), np.sin(yaw)]) < np.cos(np.radians(30)):
            continue
        scored.append((place_box(grid, clear, depth)[2], Path(f).stem))
    scored.sort(reverse=True)
    c = np.array([s for s, _ in scored])
    print(f"forward leaps screened: {len(c)} of {len(files)}; clear height for a {depth * 100:.0f} cm deep box: median {np.median(c) * 100:.1f} cm, "
          f">=8 cm {np.sum(c >= 0.08)}, >=10 cm {np.sum(c >= 0.10)}, >=12 cm {np.sum(c >= 0.12)}")
    keep = [(c_, n) for c_, n in scored if c_ >= min_clear] if min_clear > 0 else scored[:n_best]
    return [n for _, n in keep]


def main(out: Path = Path("logs/jump_pilot"), n_leaps: int = 18, margin: float = 0.02, width: float = 0.6, render: bool = True, group: str = "",
         obstacle_raw: Path = Path("data/jump_pilot/raw"), mirrors: bool = False, min_leap_clear: float = 0.0, save: Path | None = None) -> None:
    (out / "clips").mkdir(parents=True, exist_ok=True)
    m = mujoco.MjModel.from_xml_path(str(get_robot("dash").xml_path()))
    jobs = []  # (name, path, depth, max_height, group)
    for f in sorted(obstacle_raw.glob("dash_jump_over_obstacle*.npy")):
        if mirrors or not f.stem.endswith("_M"):
            jobs.append((f.stem, f, 0.15, 0.25, "obstacle"))
    for pat in ("dash_avoid_obstacle_jump_run_ff_180_R_*", "dash_jump_twice_R_*"):
        for f in sorted(SSD_RAW.glob(pat + ".npy")):
            if mirrors or not f.stem.endswith("_M"):
                jobs.append((f.stem, f, 0.15, 0.25, "run_jump"))
    for n in screen_leaps(m, n_leaps, 0.10, min_leap_clear):
        for nm in (n, n + "_M") if mirrors else (n,):
            f = SSD_RAW / f"{nm}.npy"
            jobs.append((nm, f if f.exists() else NPZ / f"{nm}.npz", 0.10, 0.12, "leap"))
    if group:
        jobs = [j for j in jobs if j[4] == group]
    rows = []
    for name, path, depth, hmax, group in jobs:
        qpos, dat = load(path)
        fe, sp, fr, sr = track(m, qpos)
        d, grid, clear = clearance_profile(qpos, fe, sp, fr, sr)
        s_lo, s_hi, c = place_box(grid, clear, depth)
        h = float(np.floor(min(hmax, c - margin) * 100) / 100)
        travel = float(np.linalg.norm(qpos[-1, :2] - qpos[0, :2]))
        rows.append({"clip": name, "group": group, "box_height": h, "box_depth": depth, "clear_above_box": c - h, "travel_m": travel})
        if save is not None:
            save.mkdir(parents=True, exist_ok=True)
            np.savez(save / f"{name}.npz", qpos=qpos, fps=60.0, kind="jump_" + group, c=qpos[0, :2], d=d, box_s=np.array([s_lo, s_hi]), box_height=h,
                     box_width=width, clear=c)
        print(f"{group:9s} {name.removeprefix('dash_'):46s} box {h * 100:4.0f} cm high x {depth * 100:.0f} cm deep, clear by {(c - h) * 100:.1f} cm, travel {travel:.2f} m")
        if render and h >= 0.03:
            H, origin = box_grid(qpos, d, s_lo, s_hi, h, width)
            mocap, bones = None, []
            if dat is not None and "mocap" in dat:
                jn = list(dat["joint_names"])
                idx = {k: i for i, k in enumerate(jn)}
                bones = [(idx[a], idx[b]) for a, b in dat["skeleton"] if a in HUMAN_BONES and b in HUMAN_BONES]
                mocap = dat["mocap"][::2]
            render_clip(name, f"{group}: box {h * 100:.0f} cm high, {depth * 100:.0f} cm deep (motion unedited)", qpos, mocap, bones, H, origin, d,
                        f"lowest foot/shin over the box: {c * 100:.1f} cm", out / "clips" / f"{group}__{name.removeprefix('dash_')}.mp4")
    with open(out / f"scores{('_' + jobs[0][4]) if group else ''}.csv", "w") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    tyro.cli(main)
