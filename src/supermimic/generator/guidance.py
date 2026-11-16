"""Guided sampling for the motion generator (no retraining): a Diffusion.sample x0_hook that corrects each late
denoising step's clean-plan estimate so the plan's soles stay out of the terrain.

On 17 / 18 cm stairs at 2 m/s gen_v6's plans put a sole > 3 cm into a step in 60 / 64 runs (10-11 cm at worst): the
swing foot cuts through the stair edge and the tracking policy has to improvise there. The cost is the squared depth
of the sole points (FootMetrics.sole: the foot capsules' end points) below the terrain within ``inflate`` m of them
(the max of the heightmap at the point and at 4 points that far away along x and y), so a toe that comes within
``inflate`` of a riser is pushed up over its top while a sole on a tread costs nothing. Gradient steps on the
normalized plan; with ``joints_only`` (default) only the joint angles move, the root's path and orientation stay the
generator's, so the swing leg folds instead of the whole body rising (which would lift the stance foot too).
"""

from __future__ import annotations

import torch

from supermimic.generator.data import Normalizer, from_canonical
from supermimic.generator.metrics import FootMetrics

ROOT_FEATURES = 9  # position (3) + rot6d (6) ahead of the joints in a canonical frame


class FootClearance:
    def __init__(self, norm: Normalizer, fm: FootMetrics, last: int = 4, iters: int = 5, step: float = 20.0, inflate: float = 0.03,
                 joints_only: bool = True):
        """last: guided denoising steps (the final ones); iters: gradient steps per guided step; step: gradient step
        size on the normalized plan."""
        self.norm, self.fm, self.last, self.iters, self.step, self.inflate, self.joints_only = norm, fm, last, iters, step, inflate, joints_only
        self.stats: list[tuple[float, float]] = []  # per final step: mean over envs of the worst depth (m) before / after

    def ground(self, s: torch.Tensor, tiles: torch.Tensor) -> torch.Tensor:
        """s (B, ..., 3) terrain-frame points -> (B, ...) highest ground within `inflate` of each."""
        B = s.shape[0]
        x, y = s[..., 0].reshape(B, -1), s[..., 1].reshape(B, -1)
        g = self.fm.scan.lookup(x, y, tiles)
        r = self.inflate
        for dx, dy in ((r, 0.0), (-r, 0.0), (0.0, r), (0.0, -r)):
            g = torch.maximum(g, self.fm.scan.lookup(x + dx, y + dy, tiles))
        return g.reshape(s.shape[:-1])

    def depth(self, x: torch.Tensor, origin, yaw0, tiles) -> torch.Tensor:
        """Normalized plan x (B, F, D) -> sole depth below the inflated ground (B, F, 2 feet, 2 ends), >= 0."""
        pos, quat, q = from_canonical(self.norm.frames_inv(x), origin, yaw0)
        s = self.fm.sole(pos, quat, q)
        return (self.ground(s.detach(), tiles) - s[..., 2]).clamp_min(0)

    def hook(self, origin: torch.Tensor, yaw0: torch.Tensor, tiles: torch.Tensor):
        """-> x0_hook for one Diffusion.sample call planning from origin (B, 3), yaw0 (B,) on terrains tiles (B,)."""

        def fix(x0: torch.Tensor, i: int, n: int) -> torch.Tensor:
            if i < n - self.last:
                return x0
            with torch.autocast("cuda", enabled=False), torch.enable_grad():
                x = x0.detach().float()
                d0 = None
                for _ in range(self.iters):
                    x.requires_grad_(True)
                    d = self.depth(x, origin, yaw0, tiles)
                    if d0 is None:
                        d0 = d.detach().flatten(1).amax(1)
                    (g,) = torch.autograd.grad((d**2).sum(), x)
                    if self.joints_only:
                        g[..., :ROOT_FEATURES] = 0
                    x = (x - self.step * g).detach()
                if i == n - 1:
                    d1 = self.depth(x, origin, yaw0, tiles).flatten(1).amax(1)
                    self.stats.append((float(d0.mean()), float(d1.mean())))
            return x.to(x0.dtype)

        return fix
