"""Foot-terrain quality of reference motion on its terrain (generated or mocap): penetration, floating
in stance, sliding in contact. Feet are Dash's foot capsules (r 1.25 cm), heights from the terrain heightfield."""

from __future__ import annotations

import math

import mujoco
import numpy as np
import torch

from supermimic.kinematics.fk import Kinematics
from supermimic.kinematics.rotations import quat_apply
from supermimic.robots import get_robot
from supermimic.terrain.height_scan import HeightScan

DT = 0.02


class FootMetrics:
    def __init__(self, scan: HeightScan, device="cuda:0"):
        m = get_robot("dash").load_model()
        self.scan = scan
        self.fk = Kinematics(m, device=device)
        self.feet = self.fk.body_indices(["l_foot", "r_foot"])
        ends = []
        for side in "lr":
            g = m.geom(f"{side}_foot_collision")
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, g.quat)
            axis = R.reshape(3, 3)[:, 2] * g.size[1]
            ends.append([g.pos + axis, g.pos - axis])
            self.radius = float(g.size[0])
        self.ends = torch.as_tensor(np.array(ends), dtype=torch.float32, device=device)  # (2 feet, 2 ends, 3) body frame

    def ends_w(self, root_pos, root_quat, joints):
        """(B, T, ...) -> capsule end centres (B, T, 2 feet, 2 ends, 3)."""
        out = self.fk(torch.cat([root_pos, root_quat, joints], -1))
        p, q = out["pos"][..., self.feet, :], out["quat"][..., self.feet, :]
        return p[..., None, :] + quat_apply(q[..., None, :].expand(*q.shape[:-1], 2, 4), self.ends.expand(*p.shape[:-1], 2, 3))

    def sole(self, root_pos, root_quat, joints):
        """(B, T, ...) -> sole points (B, T, 2 feet, 2 ends, 3): capsule end centres lowered by the radius."""
        e = self.ends_w(root_pos, root_quat, joints)
        return e - torch.tensor([0.0, 0.0, self.radius], device=e.device)

    def axis(self, root_pos, root_quat, joints, k: int = 12):
        """(B, T, ...) -> k points along each foot capsule's axis, end to end (B, T, 2 feet, k, 3): with the
        radius, spheres covering the capsule (the foot's collision volume)."""
        e = self.ends_w(root_pos, root_quat, joints)
        w = torch.linspace(0, 1, k, device=e.device)[:, None]
        return e[..., 0:1, :] * (1 - w) + e[..., 1:2, :] * w

    def gap(self, root_pos, root_quat, joints, tile):
        """-> sole points (B, T, 2, 2, 3), lowest sole point of each foot above the ground (B, T, 2)."""
        s = self.sole(root_pos, root_quat, joints)
        B, T = s.shape[:2]
        ground = self.scan.lookup(s[..., 0].reshape(B, -1), s[..., 1].reshape(B, -1), tile).reshape(B, T, 2, 2)
        return s, (s[..., 2] - ground).min(-1).values

    @torch.no_grad()
    def __call__(self, root_pos, root_quat, joints, tile) -> dict[str, float]:
        s, gap = self.gap(root_pos, root_quat, joints, tile)
        mid = s.mean(-2)  # (B, T, 2, 3)
        v = (mid[:, 1:] - mid[:, :-1]) / DT
        speed, hspeed = v.norm(dim=-1), v[..., :2].norm(dim=-1)
        g = gap[:, 1:]
        pen = (-gap).clamp_min(0).max(-1).values
        stance = speed < 0.1
        contact = g < 0.01
        return {
            "pen_mean_cm": float(pen.mean()) * 100,
            "pen_gt2cm": float((pen > 0.02).float().mean()),
            "float_stance_cm": float(g.clamp_min(0)[stance].mean()) * 100 if stance.any() else float("nan"),
            "skate_cm_s": float(hspeed[contact].mean()) * 100 if contact.any() else float("nan"),
        }


def edge_penetration(scan: HeightScan, p: torch.Tensor, tile: torch.Tensor, radius: float, step: float = 0.03) -> torch.Tensor:
    """Points p (n, P, 3) in the terrain frame of terrains tile (n,) -> (n, P) depth of each point inside a cylinder
    of `radius` around the nearest step lip ("Hiking in the Wild"'s edge cylinders, on a heightfield): a cell boundary
    where the height jumps by more than `step`, the lip on it at the upper height. Searched along x in the point's
    cell row and along y in its column, out to `radius`; a lip counts as running the whole boundary (box corners are
    rounded off)."""
    n, P, _ = p.shape
    idx = (tile.round().long() + 1).clamp(0, scan.start.shape[0] - 1)
    p0, dx = scan.p0[idx][:, None, None, :], scan.dx[idx][:, None, None]  # (n, 1, 1, 2), (n, 1, 1)
    k = int(math.ceil(radius / float(scan.dx[1:].min()))) + 1 if len(scan.dx) > 1 else 1
    off = torch.arange(-k, k + 1, device=p.device, dtype=p.dtype)
    z = p[..., 2:3]
    out = torch.zeros(n, P, device=p.device, dtype=p.dtype)
    for ax in (0, 1):
        c = p[..., ax : ax + 1]  # (n, P, 1)
        cells = p0[..., ax] + (torch.round((c - p0[..., ax]) / dx) + off) * dx  # (n, P, 2k + 1) cell centres
        other = p[..., 1 - ax : 2 - ax].expand_as(cells)
        qx, qy = (cells, other) if ax == 0 else (other, cells)
        h = scan.lookup(qx.reshape(n, -1), qy.reshape(n, -1), tile).reshape(n, P, -1)
        lip = torch.maximum(h[..., 1:], h[..., :-1])
        b = (cells[..., 1:] + cells[..., :-1]) / 2  # boundary coordinates
        d = torch.sqrt((c - b) ** 2 + (z - lip) ** 2)
        pen = torch.where((h[..., 1:] - h[..., :-1]).abs() > step, (radius - d).clamp_min(0), torch.zeros_like(d))
        out = torch.maximum(out, pen.amax(-1))
    return out
