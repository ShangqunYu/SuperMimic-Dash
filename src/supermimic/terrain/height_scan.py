"""Height scans of the terrain around the robot, looked up exactly in the terrain heightfields.

Envs carry their terrain index as the last column of xi (-1 = flat ground; data/recorder.py), states are in
the terrain frame (envs/state_io.py), so the scan of any (imagined) state is a pure function of
(state, terrain): the root's xy and heading place a heading-aligned grid, and each point reads its cell of
the terrain's heightfield (supermimic/terrain/terrains.py; clamped to the edge cells outside it, which is
what the physics boxes reproduce). Heights are absolute in the terrain frame and may be negative (pits).

Two grids:
  * ``WM_GRID`` (world model dynamics and critic): x -0.35..0.85 m, y +-0.35 m, 5 cm; absolute height
    / ``SCAN_SCALE``, so flat ground scans identically zero.
  * ``POLICY_GRID`` (teacher policy, appended to the goal): x -0.25..1.25 m, y +-0.3 m, 5 cm; root height
    above each point minus ``POLICY_NOMINAL``, / ``SCAN_SCALE`` (``policy_terrain_obs``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from supermimic.kinematics.rotations import heading_yaw


@dataclass(frozen=True)
class ScanGrid:
    x: tuple[float, float]
    y: tuple[float, float]
    step: float

    def offsets(self) -> torch.Tensor:
        """(G, 2) grid points in the heading frame."""
        xs = torch.arange(self.x[0], self.x[1] + 1e-6, self.step)
        ys = torch.arange(self.y[0], self.y[1] + 1e-6, self.step)
        gx, gy = torch.meshgrid(xs, ys, indexing="ij")
        return torch.stack([gx.reshape(-1), gy.reshape(-1)], -1)

    @property
    def dim(self) -> int:
        return int(self.offsets().shape[0])


WM_GRID = ScanGrid((-0.35, 0.85), (-0.35, 0.35), 0.05)
POLICY_GRID = ScanGrid((-0.25, 1.25), (-0.30, 0.30), 0.05)
SCAN_SCALE = 0.5  # network input = height / SCAN_SCALE
SCAN_DIM = WM_GRID.dim
POLICY_SCAN_DIM = POLICY_GRID.dim
POLICY_NOMINAL = 0.65  # m, Dash's root height standing
POLICY_SCAN_NOISE = 0.02  # m, uniform half-width per point (with obs noise)


class HeightScan(nn.Module):
    def __init__(self, terrains=None, grid: ScanGrid = WM_GRID):
        super().__init__()
        self.register_buffer("offsets", grid.offsets(), persistent=False)
        for k in ("heights", "start", "shape", "p0", "dx"):  # lookup tables, set by set_terrains
            self.register_buffer(k, torch.zeros(1), persistent=False)
        self.set_terrains(terrains)

    def set_terrains(self, terrains) -> None:
        """Row 0 is flat ground (terrain -1); row t + 1 holds terrain t of a TerrainSet."""
        ts = [] if terrains is None else terrains.terrains
        dev = self.offsets.device
        heights = [np.zeros(1, np.float32)] + [t.hf.astype(np.float32).ravel() for t in ts]
        shape = [(1, 1)] + [t.hf.shape for t in ts]
        self.heights = torch.as_tensor(np.concatenate(heights), device=dev)
        self.start = torch.as_tensor(np.cumsum([0] + [len(h) for h in heights[:-1]]), dtype=torch.long, device=dev)
        self.shape = torch.as_tensor(np.array(shape), dtype=torch.long, device=dev)
        self.p0 = torch.as_tensor(np.array([(0.0, 0.0)] + [t.p0 for t in ts]), dtype=torch.float32, device=dev)
        self.dx = torch.as_tensor(np.array([1.0] + [t.dx for t in ts]), dtype=torch.float32, device=dev)

    def forward(self, s: torch.Tensor, tile: torch.Tensor) -> torch.Tensor:
        """s (..., 13+2NJ) terrain-frame states, tile (...,) terrain index (float or long, -1 = flat) ->
        (..., G) ground height at the grid points."""
        lead = s.shape[:-1]
        s = s.reshape(-1, s.shape[-1])
        yaw = heading_yaw(s[:, 3:7])
        c, sn = torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]
        ox, oy = self.offsets[None, :, 0], self.offsets[None, :, 1]
        x = s[:, 0:1] + c * ox - sn * oy  # (n, G)
        y = s[:, 1:2] + sn * ox + c * oy
        return self.lookup(x, y, tile.reshape(-1)).reshape(*lead, -1)

    def lookup(self, x: torch.Tensor, y: torch.Tensor, tile: torch.Tensor) -> torch.Tensor:
        """Ground height at terrain-frame points x, y (n, P) of terrains tile (n,) (-1 = flat)."""
        idx = (tile.round().long() + 1).clamp(0, self.start.shape[0] - 1)
        p0, dx, shp = self.p0[idx], self.dx[idx, None], self.shape[idx]
        i = torch.round((x - p0[:, 0:1]) / dx).long().clamp(min=0)
        j = torch.round((y - p0[:, 1:2]) / dx).long().clamp(min=0)
        i = torch.minimum(i, shp[:, 0:1] - 1)
        j = torch.minimum(j, shp[:, 1:2] - 1)
        return self.heights[self.start[idx, None] + i * shp[:, 1:2] + j]


def policy_terrain_obs(scan: HeightScan, s: torch.Tensor, tile: torch.Tensor, noise: bool = False) -> torch.Tensor:
    """Teacher policy terrain input: (root height above each POLICY_GRID point - nominal) / SCAN_SCALE."""
    rel = s[..., 2:3] - scan(s, tile)
    if noise:
        rel = rel + (torch.rand_like(rel) * 2 - 1) * POLICY_SCAN_NOISE
    return (rel - POLICY_NOMINAL) / SCAN_SCALE
