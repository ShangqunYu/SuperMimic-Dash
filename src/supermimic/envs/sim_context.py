"""Per-environment simulation context xi.

Base (4 + NJ): foot friction 1 + encoder bias NJ + torso COM offset 3.
With ``hardware=True`` the robot's hardware randomization is appended (4 NJ + NB):
kp scale NJ, kd scale NJ (both relative to the nominal gains), joint armature NJ,
joint friction loss NJ, body mass scale NB (relative to nominal; pseudo_inertia scales
inertia with it). Joint-indexed blocks follow the robot's joint order.
"""

from __future__ import annotations

import numpy as np
import torch

from supermimic.robots import xi_dim


def _per_env(field: torch.Tensor, n: int, batched_ndim: int) -> torch.Tensor:
    """Model fields are expanded per world (leading env axis) only once randomized."""
    return field if field.dim() == batched_ndim else field.expand(n, *field.shape)


class SimContext:
    def __init__(self, env, foot_geom_regex: str, com_body_name: str, entity_name: str = "robot", hardware: bool = False):
        self.env = env
        self.robot = env.scene[entity_name]
        self.hardware = hardware
        nj = len(self.robot.joint_names)
        nb = len(self.robot.body_names)
        self.xi_dim = xi_dim(nj, nb, hardware)
        local_ids, names = self.robot.find_geoms(foot_geom_regex)
        if len(local_ids) == 0:
            raise ValueError(f"no geoms match {foot_geom_regex}")
        self.foot_geom_global = int(self.robot.indexing.geom_ids[local_ids[0]].item())
        self.com_body_global = int(self.robot.indexing.body_ids[self.robot.body_names.index(com_body_name)].item())
        self.com_nominal = torch.as_tensor(
            np.array(env.sim.mj_model.body_ipos[self.com_body_global]), dtype=torch.float32, device=env.device
        )
        if hardware:
            mjm = env.sim.mj_model
            dev = env.device
            joint_ids = [int(j) for j in self.robot.indexing.joint_ids.tolist()]
            # Actuator of each joint, in joint order (each actuator drives exactly one joint).
            ctrl_of_joint = {int(mjm.actuator_trnid[c, 0]): c for c in self.robot.indexing.ctrl_ids.tolist()}
            missing = [self.robot.joint_names[i] for i, j in enumerate(joint_ids) if j not in ctrl_of_joint]
            if missing:
                raise ValueError(f"joints without an actuator: {missing}")
            self.ctrl_ids = torch.tensor([ctrl_of_joint[j] for j in joint_ids], device=dev)
            self.dof_ids = self.robot.indexing.joint_v_adr.to(dev)
            self.body_ids = self.robot.indexing.body_ids.to(dev)
            gain = torch.as_tensor(env.sim.get_default_field("actuator_gainprm"), device=dev)
            bias = torch.as_tensor(env.sim.get_default_field("actuator_biasprm"), device=dev)
            mass = torch.as_tensor(env.sim.get_default_field("body_mass"), device=dev)
            self.kp_nominal = gain[self.ctrl_ids, 0].float()
            self.kd_nominal = (-bias[self.ctrl_ids, 2]).float()
            self.mass_nominal = mass[self.body_ids].float()
            assert (self.kp_nominal > 0).all() and (self.kd_nominal > 0).all() and (self.mass_nominal > 0).all()

    def read(self) -> torch.Tensor:
        model = self.env.sim.model
        n = self.env.num_envs
        friction = _per_env(model.geom_friction, n, 3)[:, self.foot_geom_global, 0]
        com = _per_env(model.body_ipos, n, 3)[:, self.com_body_global] - self.com_nominal
        bias = self.robot.data.encoder_bias
        parts = [friction.reshape(n, 1), bias, com]
        if self.hardware:
            parts += [
                _per_env(model.actuator_gainprm, n, 3)[:, self.ctrl_ids, 0] / self.kp_nominal,
                -_per_env(model.actuator_biasprm, n, 3)[:, self.ctrl_ids, 2] / self.kd_nominal,
                _per_env(model.dof_armature, n, 2)[:, self.dof_ids],
                _per_env(model.dof_frictionloss, n, 2)[:, self.dof_ids],
                _per_env(model.body_mass, n, 2)[:, self.body_ids] / self.mass_nominal,
            ]
        xi = torch.cat([p.float() for p in parts], dim=-1)
        assert xi.shape == (n, self.xi_dim)
        return xi.clone()
