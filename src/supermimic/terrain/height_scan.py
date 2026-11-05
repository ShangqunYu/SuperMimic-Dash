"""Height scan of the terrain tiles around the robot, computed exactly from the tile boxes.

Only the world model and the critic see it (the policy stays blind). Envs carry their tile index
as the last column of xi (-1 = flat ground; data/recorder.py), states are in the tile frame
(envs/state_io.py), so the scan of any (imagined) state is a pure function of (state, tile).

Grid: a heading-aligned rectangle around the root, ``SCAN_X`` forward/back and ``SCAN_Y`` to the
sides, spacing ``SCAN_STEP``. Value: absolute ground height (tile frame, ground at z = 0), so on
flat ground the scan is identically zero and a model warm-started with zero weights on it is
unchanged there.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from supermimic.kinematics.rotations import heading_yaw

SCAN_X = (-0.35, 0.85)
SCAN_Y = (-0.35, 0.35)
SCAN_STEP = 0.05
SCAN_SCALE = 0.5  # network input = height / SCAN_SCALE


def scan_offsets() -> torch.Tensor:
    """(G, 3) grid points in the heading frame (z = 0)."""
    xs = torch.arange(SCAN_X[0], SCAN_X[1] + 1e-6, SCAN_STEP)
    ys = torch.arange(SCAN_Y[0], SCAN_Y[1] + 1e-6, SCAN_STEP)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    return torch.stack([gx.reshape(-1), gy.reshape(-1), torch.zeros(gx.numel())], -1)


SCAN_DIM = scan_offsets().shape[0]


class HeightScan(nn.Module):
    def __init__(self, tiles: list[np.ndarray] | None = None):
        super().__init__()
        self.register_buffer("offsets", scan_offsets(), persistent=False)
        self.register_buffer("boxes", torch.zeros(1, 1, 6), persistent=False)
        self.set_tiles(tiles or [])

    def set_tiles(self, tiles: list[np.ndarray]) -> None:
        """Row 0 is flat ground (tile -1); row i + 1 holds tile i, padded with empty boxes."""
        nb = max([1] + [len(t) for t in tiles])
        b = torch.zeros(len(tiles) + 1, nb, 6)
        b[..., 3:5] = -1.0  # negative half sizes: never contains a point
        for i, t in enumerate(tiles):
            b[i + 1, : len(t)] = torch.as_tensor(np.asarray(t, dtype=np.float32)).reshape(-1, 6)
        self.boxes = b.to(self.offsets.device)

    def forward(self, s: torch.Tensor, tile: torch.Tensor) -> torch.Tensor:
        """s (..., 13+2NJ) tile-frame states, tile (...,) tile index (float or long, -1 = flat) ->
        (..., G) ground height at the grid points."""
        lead = s.shape[:-1]
        s = s.reshape(-1, s.shape[-1])
        yaw = heading_yaw(s[:, 3:7])
        c, sn = torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]
        ox, oy = self.offsets[None, :, 0], self.offsets[None, :, 1]
        pts = torch.stack([s[:, 0:1] + c * ox - sn * oy, s[:, 1:2] + sn * ox + c * oy], -1)  # (n, G, 2)
        idx = (tile.reshape(-1).round().long() + 1).clamp(0, self.boxes.shape[0] - 1)
        bx = self.boxes[idx]  # (n, B, 6)
        d = (pts[:, :, None, :] - bx[:, None, :, 0:2]).abs()  # (n, G, B, 2)
        inside = (d <= bx[:, None, :, 3:5]).all(-1)
        top = (bx[..., 2] + bx[..., 5])[:, None, :]
        h = torch.where(inside, top, torch.zeros_like(top)).amax(-1).clamp_min(0.0)
        return h.reshape(*lead, -1)
