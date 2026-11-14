"""Training windows for the terrain-aware motion generator, cut from a packed library on the GPU.

A window at frame t0 of a clip (t0 = the "current" frame):
  * history: frames t0-H+1 .. t0, clamped to the clip start (a clip's first frame repeats);
  * future: frames t0+1 .. t0+F, inside the clip;
  * frame features (``frame_dim(NJ)`` = 9 + NJ): root position and orientation (rot6d) in the canonical
    frame (origin at the current root, x along its heading, z up, so heights are relative to the current
    root), joint positions;
  * heightmap: terrain height minus the current root height on ``GEN_GRID`` (heading frame of the current
    root, like the policy scan, but reaching ``GEN_GRID.x[1]`` m ahead);
  * command: the smoothed "simulation root" (``ROOT_WINDOW``-frame centred mean of the torso's xy and
    unwrapped yaw, as the motion-matching planner) ``CMD_OFFSETS`` frames ahead, position and heading
    direction (cos, sin) in the canonical frame. At run time a joystick spring trajectory takes its place.

Clips: every terrain clip, plus flat locomotion (names made only of walking / jogging words, the
motion-matching vocabulary). Hold-out: 1 / ``val_every`` of the motions by name hash, with all their actor,
mirror and terrain variants (terrains are shared by many clips, so holding out terrains would take whole
categories). Unseen terrain geometry is tested separately.
"""

from __future__ import annotations

import re
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from mjlab.utils.lab_api.math import quat_from_matrix

from supermimic.kinematics.rotations import heading_yaw, quat_from_yaw, quat_mul, rot6d_from_quat
from supermimic.terrain.height_scan import HeightScan, ScanGrid
from supermimic.terrain.terrains import load_terrains

GEN_GRID = ScanGrid((-0.6, 3.0), (-0.8, 0.8), 0.05)  # 73 x 33
GRID_SHAPE = (73, 33)
CMD_OFFSETS = (20, 40, 60)
CMD_DIM = 4 * len(CMD_OFFSETS)
ROOT_WINDOW = 25
HMAP_NOMINAL = 0.65  # m, standing root height: flat ground reads 0 after normalisation
HMAP_SCALE = 0.5

LOCO_WORDS = set(
    "walk walking turn idle change stop start step arc sideway ff r l loop cw ccw right left to forward backward rotate "
    "slow very normal pace dir randdir long random direction in place jog jogging run hurry fast stance".split()
)


def frame_dim(nj: int) -> int:
    return 9 + nj


def is_locomotion(name: str) -> bool:
    base = name[len("dash_") :].split("__")[0].lower() if name.startswith("dash_") else None
    return base is not None and set(re.findall(r"[a-z]+", base)) <= LOCO_WORDS


def base_motion(name: str) -> str:
    """Clip name without actor / mirror suffix (BONES-SEED) or augmentation suffix (PARC)."""
    if name.startswith("parc/"):
        return re.sub(r"(_aug\d+|_flipped|_dm|_opt)+$", "", name.split("/")[-1])
    return re.sub(r"__A\d+(_M)?.*$", "", name)


def to_canonical(root_pos, root_quat, joints, origin, yaw0):
    """(..., T, 3/4/NJ) terrain-frame states -> (..., T, 9 + NJ) features in the frame (origin (..., 3), yaw0 (...))."""
    c, s = torch.cos(yaw0)[..., None], torch.sin(yaw0)[..., None]
    d = root_pos - origin[..., None, :]
    x = c * d[..., 0] + s * d[..., 1]
    y = -s * d[..., 0] + c * d[..., 1]
    q = quat_mul(quat_from_yaw(-yaw0)[..., None, :].expand_as(root_quat), root_quat)
    return torch.cat([x[..., None], y[..., None], d[..., 2:3], rot6d_from_quat(q), joints], -1)


def rot6d_to_matrix(r6: torch.Tensor) -> torch.Tensor:
    """(..., 6) first two columns, flattened row-major as rot6d_from_quat -> (..., 3, 3), Gram-Schmidt."""
    m = r6.reshape(*r6.shape[:-1], 3, 2)
    a = torch.nn.functional.normalize(m[..., 0], dim=-1)
    b = torch.nn.functional.normalize(m[..., 1] - (a * m[..., 1]).sum(-1, keepdim=True) * a, dim=-1)
    return torch.stack([a, b, torch.cross(a, b, dim=-1)], -1)


def from_canonical(feat, origin, yaw0):
    """Inverse of to_canonical: (..., T, 9 + NJ) -> root_pos, root_quat (wxyz), joints in the terrain frame."""
    c, s = torch.cos(yaw0)[..., None], torch.sin(yaw0)[..., None]
    p = feat[..., 0:3]
    x = c * p[..., 0] - s * p[..., 1] + origin[..., None, 0]
    y = s * p[..., 0] + c * p[..., 1] + origin[..., None, 1]
    z = p[..., 2] + origin[..., None, 2]
    q = quat_from_matrix(rot6d_to_matrix(feat[..., 3:9]))
    q = quat_mul(quat_from_yaw(yaw0)[..., None, :].expand_as(q), q)
    return torch.stack([x, y, z], -1), q, feat[..., 9:]


@dataclass
class Window:
    hist: torch.Tensor  # (B, H, D)
    fut: torch.Tensor  # (B, F, D)
    hmap: torch.Tensor  # (B, X, Y) m, terrain height - current root height
    cmd: torch.Tensor  # (B, CMD_DIM)
    origin: torch.Tensor  # (B, 3)
    yaw0: torch.Tensor  # (B,)
    tile: torch.Tensor  # (B,) terrain, -1 flat
    clip: torch.Tensor  # (B,) index into GenData.clip_names
    t0: torch.Tensor  # (B,) frame within the clip


class GenData:
    def __init__(self, library: str | Path, device: str = "cuda:0", hist: int = 10, fut: int = 60, val_every: int = 20, nj: int = 18):
        self.device = torch.device(device)
        self.H, self.F, self.nj = hist, fut, nj
        pack = torch.load(library, map_location="cpu", weights_only=False, mmap=True)
        names = list(pack["names"])
        starts, lens, tiles = np.asarray(pack["starts"]), np.asarray(pack["lengths"]), np.asarray(pack["clip_tile"])
        sel = [i for i in range(len(names)) if lens[i] >= fut + 2 and (tiles[i] >= 0 or is_locomotion(names[i]))]
        self.clip_names = [names[i] for i in sel]
        self.clip_tile = torch.as_tensor(tiles[sel], device=self.device)
        self.clip_len = torch.as_tensor(lens[sel], device=self.device)
        self.clip_start = torch.cumsum(self.clip_len, 0) - self.clip_len
        self.ref = torch.cat([pack["ref"][int(starts[i]) : int(starts[i] + lens[i])] for i in sel]).to(self.device)  # fp16
        self.terrain = tiles[sel] >= 0
        val = np.array([zlib.crc32(base_motion(n).encode()) % val_every == 0 for n in self.clip_names])
        self.is_val = val
        self._sim_root()
        self.scan = HeightScan(load_terrains(library), GEN_GRID).to(self.device)
        # per (split, kind): clips and cumulative counts of valid t0 (0 .. len - F - 1)
        self.pools = {}
        for split in ("train", "val"):
            for kind in ("terrain", "flat"):
                m = (val if split == "val" else ~val) & (self.terrain if kind == "terrain" else ~self.terrain)
                ids = torch.as_tensor(np.nonzero(m)[0], device=self.device)
                self.pools[(split, kind)] = (ids, torch.cumsum(self.clip_len[ids] - fut, 0))

    def _sim_root(self) -> None:
        """Smoothed planar root (xy, unwrapped yaw) per frame, each clip separately (edges hold their value)."""
        r = self.ref.float()
        n = r.shape[0]
        clip_of = torch.repeat_interleave(torch.arange(len(self.clip_len), device=self.device), self.clip_len)
        first, last = self.clip_start[clip_of], self.clip_start[clip_of] + self.clip_len[clip_of] - 1
        yaw = heading_yaw(r[:, 3:7]).double()
        d = torch.atan2(torch.sin(yaw.diff()), torch.cos(yaw.diff()))
        d = torch.cat([d.new_zeros(1), d])
        d[self.clip_start] = 0.0
        cum = d.cumsum(0)
        yaw_un = yaw[first] + cum - cum[first]
        h = ROOT_WINDOW // 2
        ar = torch.arange(n, device=self.device)
        xy = torch.zeros(n, 2, dtype=torch.float64, device=self.device)
        yw = torch.zeros(n, dtype=torch.float64, device=self.device)
        for o in range(-h, h + 1):
            j = torch.minimum(torch.maximum(ar + o, first), last)
            xy += r[j, :2].double()
            yw += yaw_un[j]
        self.sxy, self.syaw = (xy / ROOT_WINDOW).float(), (yw / ROOT_WINDOW).float()

    def num(self, split: str, kind: str) -> int:
        return int(self.pools[(split, kind)][1][-1]) if len(self.pools[(split, kind)][0]) else 0

    def sample(self, n: int, split: str = "train", terrain_frac: float = 0.5, generator: torch.Generator | None = None) -> Window:
        k = int(round(n * terrain_frac))
        clips, t0s = [], []
        for kind, m in (("terrain", k), ("flat", n - k)):
            ids, cum = self.pools[(split, kind)]
            if m == 0:
                continue
            u = torch.randint(int(cum[-1]), (m,), device=self.device, generator=generator)
            c = torch.searchsorted(cum, u, right=True)
            clips.append(ids[c])
            t0s.append(u - (cum[c] - (self.clip_len[ids[c]] - self.F)))
        return self.window(torch.cat(clips), torch.cat(t0s))

    def window(self, clip: torch.Tensor, t0: torch.Tensor) -> Window:
        """Windows at frame t0 (B,) of clips (B,), t0 + F < clip length."""
        s = self.clip_start[clip]
        g0 = s + t0
        hi = torch.maximum(g0[:, None] + torch.arange(-self.H + 1, 1, device=self.device), s[:, None])
        fi = g0[:, None] + torch.arange(1, self.F + 1, device=self.device)
        cur = self.ref[g0].float()
        origin, yaw0 = cur[:, 0:3], heading_yaw(cur[:, 3:7])
        nj = self.nj

        def feats(idx):
            r = self.ref[idx].float()
            return to_canonical(r[..., 0:3], r[..., 3:7], r[..., 7 : 7 + nj], origin, yaw0)

        tile = self.clip_tile[clip]
        hmap = (self.scan(cur, tile) - origin[:, 2:3]).reshape(-1, *GRID_SHAPE)
        ci = g0[:, None] + torch.as_tensor(CMD_OFFSETS, device=self.device)
        c, sn = torch.cos(yaw0)[:, None], torch.sin(yaw0)[:, None]
        d = self.sxy[ci] - origin[:, None, :2]
        dy = self.syaw[ci] - yaw0[:, None]
        cmd = torch.stack([c * d[..., 0] + sn * d[..., 1], -sn * d[..., 0] + c * d[..., 1], torch.cos(dy), torch.sin(dy)], -1).reshape(-1, CMD_DIM)
        return Window(feats(hi), feats(fi), hmap, cmd, origin, yaw0, tile, clip, t0)


class Normalizer(torch.nn.Module):
    """Per-dimension mean / std of frame features and commands (fixed after fit); heightmaps use a fixed map."""

    def __init__(self, d: int, cmd_dim: int = CMD_DIM):
        super().__init__()
        for k, n in (("fm", d), ("fs", d), ("cm", cmd_dim), ("cs", cmd_dim)):
            self.register_buffer(k, torch.zeros(n) if k.endswith("m") else torch.ones(n))

    @torch.no_grad()
    def fit(self, data: GenData, batches: int = 32, batch: int = 2048) -> None:
        f, c = [], []
        for _ in range(batches):
            w = data.sample(batch)
            f.append(torch.cat([w.hist, w.fut], 1).reshape(-1, w.fut.shape[-1]))
            c.append(w.cmd)
        f, c = torch.cat(f), torch.cat(c)
        self.fm.copy_(f.mean(0))
        self.fs.copy_(f.std(0).clamp_min(0.02))
        self.cm.copy_(c.mean(0))
        self.cs.copy_(c.std(0).clamp_min(0.02))

    def frames(self, x):
        return (x - self.fm) / self.fs

    def frames_inv(self, x):
        return x * self.fs + self.fm

    def cmd(self, c):
        return (c - self.cm) / self.cs

    @staticmethod
    def hmap(h):
        return (h + HMAP_NOMINAL) / HMAP_SCALE


__all__ = ["GenData", "Window", "Normalizer", "to_canonical", "from_canonical", "frame_dim", "GEN_GRID", "GRID_SHAPE", "CMD_DIM"]
