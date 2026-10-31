"""Build the motion-matching database (src/supermimic/planner/motion_matching.py) from the Dash library.

Clip selection:
  * name made only of neutral locomotion words (walk, turn, idle, start/stop, arc, sideways,
    backward, pace words, ...): no injured / hands-on-back / waving / jogging styles;
  * both the clip and its mirror tracked by the policy in the benchmark (not failed, normalized
    return >= ``min_return``);
  * up to ``takes`` best-tracked actor takes per motion, each with its mirror.

  uv run python scripts/planner/build_mm_db.py --out data/planner/dash_walk_mm.npz
"""

from __future__ import annotations

import collections
import json
import re
from pathlib import Path

import numpy as np
import torch
import tyro

from supermimic.assets.dash import dash as dash_asset
from supermimic.envs.mdp.motion_library import RefState, goal_features
from supermimic.kinematics.fk import Kinematics
from supermimic.kinematics.rotations import heading_yaw, quat_apply_inverse, yaw_quat
from supermimic.planner.motion_matching import FUTURE_FRAMES, MATCH_GROUPS, TRAJ_OFFSETS
from supermimic.robots import get_robot

WALK_WORDS = set(
    "walk walking turn idle change stop start step arc sideway ff r l loop cw right left to forward backward rotate "
    "slow very normal pace dir randdir long random direction in place".split()
)
JOG_WORDS = set("jog jogging run hurry fast".split())
# Relative importance of each matching group (normalized by its own std first).
GROUP_WEIGHTS = {"traj_pos": 1.0, "traj_dir": 1.0, "foot_pos": 0.75, "foot_vel": 0.5, "root_vel": 1.0}


def default_joint_pos(names) -> torch.Tensor:
    q = torch.zeros(len(names))
    for pat, v in dash_asset.KNEES_BENT_KEYFRAME.joint_pos.items():
        for i, n in enumerate(names):
            if re.fullmatch(pat, n):
                q[i] = v
    return q


def select_clips(names, bench: dict, takes: int, min_return: float, jog: bool) -> list[str]:
    allow = WALK_WORDS | (JOG_WORDS if jog else set())
    ok = lambda m: m in bench and not bench[m]["failed"] and bench[m]["norm_return"] >= min_return  # noqa: E731
    groups = collections.defaultdict(list)
    for n in names:
        if n.endswith("_M"):
            continue
        base = n[len("dash_") :].split("__")[0].lower()
        if not set(re.findall(r"[a-z]+", base)) <= allow or not (ok(n) and ok(n + "_M")):
            continue
        groups[re.sub(r"_\d{3}$", "", base)].append(n)
    return [m for g in groups.values() for n in sorted(g, key=lambda n: -bench[n]["norm_return"])[:takes] for m in (n, n + "_M")]


@torch.no_grad()
def clip_features(ref: torch.Tensor, nj: int, q0: torch.Tensor, fk: Kinematics, feet: torch.Tensor):
    """ref (T, 13+2NJ) -> goal (T,46), match (T,D) for one clip; trajectory offsets clamped at the clip end."""
    r = RefState(ref[:, 0:3], ref[:, 3:7], ref[:, 7 : 7 + nj], ref[:, 7 + nj : 10 + nj], ref[:, 10 + nj : 13 + nj], ref[:, 13 + nj :])
    goal = goal_features(r, q0)
    T = ref.shape[0]
    yaw = heading_yaw(r.root_quat_w)
    hq = yaw_quat(r.root_quat_w)
    traj_pos, traj_dir = [], []
    for o in TRAJ_OFFSETS:
        j = torch.clamp(torch.arange(T, device=ref.device) + o, max=T - 1)
        d = r.root_pos_w[j] - r.root_pos_w
        d[:, 2] = 0.0
        traj_pos.append(quat_apply_inverse(hq, d)[:, :2])
        dy = yaw[j] - yaw
        traj_dir.append(torch.stack([torch.cos(dy), torch.sin(dy)], -1))
    ang_b = quat_apply_inverse(r.root_quat_w, r.root_ang_vel_w)
    qvel = torch.cat([r.root_lin_vel_w, ang_b, r.joint_vel], -1)
    out = fk(torch.cat([r.root_pos_w, r.root_quat_w, r.joint_pos], -1), qvel)
    hq_b = hq[:, None, :].expand(-1, len(feet), 4)
    fpos = quat_apply_inverse(hq_b, out["pos"][:, feet] - r.root_pos_w[:, None, :]).reshape(T, -1)
    fvel = quat_apply_inverse(hq_b, out["lin_vel"][:, feet]).reshape(T, -1)
    lin_h = quat_apply_inverse(hq, r.root_lin_vel_w)[:, :2]
    root_vel = torch.cat([lin_h, r.root_ang_vel_w[:, 2:3]], -1)
    match = torch.cat([torch.cat(traj_pos, -1), torch.cat(traj_dir, -1), fpos, fvel, root_vel], -1)
    return goal, match


def main(
    library: Path = Path("data/library/dash_bones_seed_compact.pt"),
    benchmark: Path = Path("logs/dash_e2_hwdr/benchmark_ckpt_latest.json"),
    out: Path = Path("data/planner/dash_walk_mm.npz"),
    takes: int = 3,
    min_return: float = 0.9,
    jog: bool = False,
    device: str = "cuda:0",
) -> None:
    spec = get_robot("dash")
    nj = spec.num_joints
    pack = torch.load(library, map_location="cpu", weights_only=False)
    names = list(pack["names"])
    bench = {c["name"]: c for c in json.loads(benchmark.read_text())["clips"]}
    sel = select_clips(names, bench, takes, min_return, jog)
    idx = {n: i for i, n in enumerate(names)}
    q0 = default_joint_pos(spec.joint_names).to(device)
    fk = Kinematics(spec.load_model(), device=device)
    feet = fk.body_indices(["l_foot", "r_foot"])

    goals, matches, clip_of, clip_last, valid = [], [], [], [], []
    start = 0
    need = max(TRAJ_OFFSETS) + FUTURE_FRAMES
    for c, n in enumerate(sel):
        i = idx[n]
        s, length = int(pack["starts"][i]), int(pack["lengths"][i])
        ref = pack["ref"][s : s + length].to(device=device, dtype=torch.float32)
        g, m = clip_features(ref, nj, q0, fk, feet)
        goals.append(g.cpu().numpy())
        matches.append(m.cpu().numpy())
        clip_of.append(np.full(length, c))
        clip_last.append(start + length - 1)
        v = np.zeros(length, dtype=bool)
        v[: max(length - need, 0)] = True
        valid.append(v)
        start += length
    goal = np.concatenate(goals).astype(np.float32)
    match = np.concatenate(matches).astype(np.float32)
    valid = np.concatenate(valid)
    clip_of = np.concatenate(clip_of)

    # Per-group normalization over candidate frames: dims / (group std * sqrt(dims)) / weight.
    mean = match[valid].mean(0)
    scale = np.ones(match.shape[1], dtype=np.float32)
    o = 0
    for g, d in MATCH_GROUPS:
        sd = float(match[valid, o : o + d].std(0).mean())
        scale[o : o + d] = sd * np.sqrt(d) / GROUP_WEIGHTS[g]
        o += d

    # Start frame: the slowest valid frame of an idle clip.
    speed = np.linalg.norm(match[:, -3:-1], axis=-1) + 0.3 * np.abs(match[:, -1])
    idle = np.array(["idle" in sel[c].lower() and "turn" not in sel[c].lower() for c in clip_of]) & valid
    start_frame = int(np.flatnonzero(idle)[np.argmin(speed[idle])])

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, goal=goal.astype(np.float16), match=match, match_mean=mean, match_scale=scale, clip_of=clip_of.astype(np.int32),
        clip_last=np.array(clip_last, dtype=np.int64), valid=valid, clip_names=np.array(sel), start_frame=start_frame,
    )
    rv = match[valid, -3:]
    print(f"{len(sel)} clips, {len(goal)} frames ({len(goal) / 50 / 3600:.2f} h), {valid.sum()} candidates -> {out} ({out.stat().st_size / 1e6:.1f} MB)")
    print(f"start frame {start_frame} in {sel[clip_of[start_frame]]}")
    for k, nm in enumerate(("vx", "vy", "wz")):
        p = np.percentile(rv[:, k], [1, 5, 50, 95, 99])
        print(f"  {nm}: 1/5/50/95/99 pct {np.round(p, 2)}")


if __name__ == "__main__":
    tyro.cli(main)
