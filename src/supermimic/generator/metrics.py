"""Foot-terrain quality of reference motion on its terrain (generated or mocap): penetration, floating
in stance, sliding in contact. Feet are Dash's foot capsules (r 1.25 cm), heights from the terrain heightfield."""

from __future__ import annotations

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

    def sole(self, root_pos, root_quat, joints):
        """(B, T, ...) -> sole points (B, T, 2 feet, 2 ends, 3): capsule end centres lowered by the radius."""
        out = self.fk(torch.cat([root_pos, root_quat, joints], -1))
        p, q = out["pos"][..., self.feet, :], out["quat"][..., self.feet, :]
        e = p[..., None, :] + quat_apply(q[..., None, :].expand(*q.shape[:-1], 2, 4), self.ends.expand(*p.shape[:-1], 2, 3))
        return e - torch.tensor([0.0, 0.0, self.radius], device=e.device)

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
