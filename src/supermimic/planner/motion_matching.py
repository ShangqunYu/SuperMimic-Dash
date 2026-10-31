"""Motion-matching planner: RC velocity command -> reference goal features for the motion-tracking policy.

numpy only (no torch), so the same file runs on the robot. The database is built offline by
``scripts/planner/build_mm_db.py`` from clean locomotion clips of the Dash library.

The tracking policy's goal input is K future reference frames of 46 features each
(``goal_features`` in envs/mdp/motion_library.py):
  [joint_pos - q0 (NJ), joint_vel (NJ), projected gravity (3), root ang vel, body frame (3),
   root lin vel, heading frame (3), root height (1)]
None of them is a global position or yaw, so the planner only chooses which mocap frames to
play: nothing is re-anchored to the robot's world pose.

Each control tick (50 Hz):
  1. A critically damped spring turns the command (vx, vy in the heading frame, wz world yaw
     rate) into the desired root trajectory over the next second, starting from the velocity of
     the frame being played.
  2. Every ``search_interval`` ticks, or when the playing clip is about to end, the database is
     searched for the frame whose [future root trajectory, foot positions/velocities, root
     velocity] best matches [desired trajectory, current pose]. Playback jumps there if it is
     better than continuing.
  3. A jump is smoothed by inertialization: the feature difference old - new at the jump decays
     with ``blend_halflife`` over the following frames.
  4. Output: goal features of the K frames from the playback frame on, then advance one frame.

Matching feature layout (per frame, in that frame's heading frame), see MATCH_GROUPS.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

DT = 0.02
TRAJ_OFFSETS = (10, 25, 50)  # frames ahead (0.2, 0.5, 1.0 s) of the trajectory features
FUTURE_FRAMES = 10  # K, the policy's goal window
# (name, dims); the database stores features in this order.
MATCH_GROUPS = (("traj_pos", 6), ("traj_dir", 6), ("foot_pos", 6), ("foot_vel", 6), ("root_vel", 3))
TRAJ_DIMS = 12  # traj_pos + traj_dir come first
GRAV = slice(36, 39)  # projected gravity inside a 46-d goal frame (NJ = 18)


@dataclass
class PlannerCfg:
    search_interval: int = 10  # ticks between searches (0.2 s)
    blend_halflife: float = 0.1  # s, inertialization decay
    vel_halflife: float = 0.3  # s, spring from current to commanded linear velocity
    yaw_halflife: float = 0.25  # s, same for yaw rate
    jump_margin: float = 0.02  # a jump must beat continuing by this much (normalized cost) ...
    jump_ratio: float = 0.2  # ... and by this fraction of the continuation cost
    same_clip_window: int = 20  # frames: ignore candidates this close to the playing frame


class MotionDatabase:
    def __init__(self, path: str | Path, use_tree: bool = True):
        d = np.load(path, allow_pickle=False)
        self.goal = d["goal"].astype(np.float32)  # (F, 46)
        self.match = d["match"].astype(np.float32)  # (F, D) raw
        self.mean = d["match_mean"].astype(np.float32)
        self.scale = d["match_scale"].astype(np.float32)  # per dim: group std / group weight
        self.clip_of = d["clip_of"].astype(np.int64)  # (F,)
        self.clip_last = d["clip_last"].astype(np.int64)  # (C,) last frame index (inclusive)
        self.valid = d["valid"].astype(bool)  # (F,) candidate: >= max(TRAJ_OFFSETS) + K frames left
        self.clip_names = [str(n) for n in d["clip_names"]]
        self.start_frame = int(d["start_frame"])  # a standing idle frame
        self.norm = (self.match - self.mean) / self.scale
        self.cand = np.nonzero(self.valid)[0]
        self.cand_norm = np.ascontiguousarray(self.norm[self.cand])
        # Exact nearest-neighbour search with a KD-tree when scipy is available (~1 ms instead of
        # ~50 ms brute force on 750k candidates); results are identical.
        self.tree = None
        if use_tree:
            try:
                from scipy.spatial import cKDTree

                self.tree = cKDTree(self.cand_norm)
            except ImportError:
                pass

    def nearest(self, q: np.ndarray, exclude) -> tuple[int, float]:
        """Best candidate (database frame, squared cost) not rejected by ``exclude(frames) -> bool mask``."""
        if self.tree is not None:
            for k in (16, 256):
                dist, idx = self.tree.query(q, k=k)
                ok = ~exclude(self.cand[idx])
                if ok.any():
                    i = int(np.argmax(ok))
                    return int(self.cand[idx[i]]), float(dist[i]) ** 2
        cost = ((self.cand_norm - q) ** 2).sum(-1)
        cost[exclude(self.cand)] = np.inf
        j = int(np.argmin(cost))
        return int(self.cand[j]), float(cost[j])

    def __len__(self) -> int:
        return self.goal.shape[0]


def spring_trajectory(v0: np.ndarray, w0: float, cmd: np.ndarray, cfg: PlannerCfg) -> np.ndarray:
    """Desired [traj_pos (6), traj_dir (6)] in the current heading frame.

    v0 (2,): current planar velocity in the heading frame; w0: current yaw rate; cmd: (vx, vy, wz)."""
    n = max(TRAJ_OFFSETS)
    t = np.arange(1, n + 1) * DT
    yv = 2.0 * np.log(2.0) / cfg.vel_halflife
    yw = 2.0 * np.log(2.0) / cfg.yaw_halflife
    v = cmd[None, :2] + (v0 - cmd[:2])[None, :] * ((1.0 + yv * t) * np.exp(-yv * t))[:, None]  # (n,2) heading frame at t
    w = cmd[2] + (w0 - cmd[2]) * (1.0 + yw * t) * np.exp(-yw * t)
    yaw = np.cumsum(w) * DT
    c, s = np.cos(yaw), np.sin(yaw)
    vw = np.stack([c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1]], -1)  # rotate into the current heading frame
    pos = np.cumsum(vw, 0) * DT
    idx = np.array(TRAJ_OFFSETS) - 1
    return np.concatenate([pos[idx].reshape(-1), np.stack([c[idx], s[idx]], -1).reshape(-1)]).astype(np.float32)


class MotionMatcher:
    def __init__(self, db: MotionDatabase, cfg: PlannerCfg | None = None):
        self.db = db
        self.cfg = cfg or PlannerCfg()
        self.reset()

    def reset(self, frame: int | None = None) -> None:
        self.frame = self.db.start_frame if frame is None else int(frame)
        self.offset = np.zeros(self.db.goal.shape[1], dtype=np.float32)
        self.offset_age = 0.0
        self.since_search = 0
        self.jumps = 0

    # ------------------------------------------------------------------ search
    def _query(self, cmd: np.ndarray) -> np.ndarray:
        cur = self.db.match[self.frame]
        rv = cur[-3:]  # (vx, vy, wz) of the playing frame
        q = cur.copy()
        q[:TRAJ_DIMS] = spring_trajectory(rv[:2], float(rv[2]), cmd, self.cfg)
        return (q - self.db.mean) / self.db.scale

    def _search(self, cmd: np.ndarray, forced: bool) -> None:
        db, cfg = self.db, self.cfg
        q = self._query(cmd)
        cur_clip = db.clip_of[self.frame]
        best, cost = db.nearest(q, lambda f: (db.clip_of[f] == cur_clip) & (np.abs(f - self.frame) < cfg.same_clip_window))
        if not forced:
            cont = float(((db.norm[self.frame] - q) ** 2).sum())
            if not cost < cont * (1.0 - cfg.jump_ratio) - cfg.jump_margin:
                return
        self._jump(best)

    def _jump(self, new: int) -> None:
        old_now = self._feat(self.frame, 0)
        self.frame = new
        self.offset = old_now - self.db.goal[new]
        self.offset_age = 0.0
        self.jumps += 1

    # ------------------------------------------------------------------ output
    def _decay(self, t: float) -> float:
        return float(np.exp(-np.log(2.0) * t / self.cfg.blend_halflife))

    def _feat(self, frame: int, k: int) -> np.ndarray:
        f = self.db.goal[min(frame + k, self.db.clip_last[self.db.clip_of[frame]])] + self.offset * self._decay(self.offset_age + k * DT)
        g = f[GRAV]
        f[GRAV] = g / max(float(np.linalg.norm(g)), 1e-6)
        return f

    def step(self, cmd) -> np.ndarray:
        """One 50 Hz tick: (vx, vy, wz) -> (K * 46,) goal features for the policy."""
        cmd = np.asarray(cmd, dtype=np.float32)
        last = self.db.clip_last[self.db.clip_of[self.frame]]
        forced = self.frame + FUTURE_FRAMES >= last
        if forced or self.since_search >= self.cfg.search_interval:
            self._search(cmd, forced)
            self.since_search = 0
        goal = np.concatenate([self._feat(self.frame, k) for k in range(FUTURE_FRAMES)])
        self.frame += 1
        self.offset_age += DT
        self.since_search += 1
        return goal

    @property
    def clip_name(self) -> str:
        return self.db.clip_names[self.db.clip_of[self.frame]]

    def reference_velocity(self) -> np.ndarray:
        """(vx, vy, wz) of the frame being played (for logging)."""
        return self.db.match[self.frame, -3:]
