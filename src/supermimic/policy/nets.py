"""Policy (goal encoder + shared decoder) and critic (paper Table A.4)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from supermimic.world_model.dynamics import MLP
from supermimic.robots import goal_feat_dim, obs_dim, state_dim
from supermimic.robots import xi_dim as base_xi_dim
from supermimic.world_model.heads import obs_terms

LATENT_DIM = 32


@dataclass
class PolicyCfg:
    num_joints: int = 29  # = action dim; set from the robot spec by the trainer
    encoder_hidden: tuple[int, ...] = (2048, 1024, 512, 256)
    decoder_hidden: tuple[int, ...] = (2048, 1024, 512, 256)
    latent_dim: int = LATENT_DIM
    history: int = 10
    # Fixed per-term observation scales (divide): joint_pos, joint_vel, base_ang_vel, projected_gravity, actions.
    obs_term_scale: tuple[float, ...] = (1.0, 5.0, 2.0, 1.0, 1.0)
    # Fixed goal feature scales: joint_pos NJ, joint_vel NJ, grav 3, ang_vel_b 3, lin_vel_h 3, height 1.
    goal_term_scale: tuple[float, ...] = (1.0, 5.0, 1.0, 2.0, 1.0, 1.0)
    decoder_last_gain: float = 0.01
    goal_dim: int = 680  # K * (2 NJ + 10) (+9 with anchor_obs); set by the trainer
    # Velocity interface (paper Appendix B/G): encoder of the normalized command + observation history.
    velocity: bool = False
    velocity_hidden: tuple[int, ...] = (256, 256)
    cmd_scale: tuple[float, ...] = (1.5, 0.6, 1.8)  # divide (vx, vy, wz); set from VelocityCfg by the trainer

    @property
    def obs_dim(self) -> int:
        return self.history * obs_dim(self.num_joints)


@dataclass
class CriticCfg:
    hidden: tuple[int, ...] = (256, 128)


def _obs_scale_vector(cfg: PolicyCfg) -> torch.Tensor:
    parts = []
    for (name, d), s in zip(obs_terms(cfg.num_joints), cfg.obs_term_scale):
        parts.append(torch.full((cfg.history * d,), 1.0 / s))
    return torch.cat(parts)  # term-major layout matches the flattened history


def _goal_scale_vector(cfg: PolicyCfg, goal_dim: int) -> torch.Tensor:
    nj = cfg.num_joints
    sizes = (nj, nj, 3, 3, 3, 1)
    per_frame = torch.cat([torch.full((n,), 1.0 / s) for n, s in zip(sizes, cfg.goal_term_scale)])
    k, extra = divmod(goal_dim, goal_feat_dim(nj))
    return torch.cat([per_frame.repeat(k), torch.ones(extra)])


class MotionEncoder(nn.Module):
    def __init__(self, cfg: PolicyCfg, goal_dim: int | None = None):
        super().__init__()
        goal_dim = goal_dim or cfg.goal_dim
        self.mlp = MLP(goal_dim, cfg.encoder_hidden, cfg.latent_dim)
        self.register_buffer("scale", _goal_scale_vector(cfg, goal_dim))

    def forward(self, goal: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.mlp(goal * self.scale))


class VelocityEncoder(nn.Module):
    """z = tanh(MLP([cmd / cmd_scale, o_hist * obs_scale])) (Table A.4: normalized command + recent observations)."""

    def __init__(self, cfg: PolicyCfg):
        super().__init__()
        self.mlp = MLP(3 + cfg.obs_dim, cfg.velocity_hidden, cfg.latent_dim)
        self.register_buffer("cmd_scale", 1.0 / torch.tensor(cfg.cmd_scale, dtype=torch.float32))
        self.register_buffer("obs_scale", _obs_scale_vector(cfg))

    def forward(self, cmd: torch.Tensor, obs_hist: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.mlp(torch.cat([cmd * self.cmd_scale, obs_hist * self.obs_scale], dim=-1)))


class Decoder(nn.Module):
    def __init__(self, cfg: PolicyCfg):
        super().__init__()
        self.mlp = MLP(cfg.latent_dim + cfg.obs_dim, cfg.decoder_hidden, cfg.num_joints, last_gain=cfg.decoder_last_gain)
        self.register_buffer("scale", _obs_scale_vector(cfg))

    def forward(self, z: torch.Tensor, obs_hist: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([z, obs_hist * self.scale], dim=-1))


class Policy(nn.Module):
    """a = dec(enc_i(g), o_hist), one encoder per goal interface (plan Appendix B).

    "motion": g = K future reference frames. "velocity": g = (vx, vy, wz) command."""

    def __init__(self, cfg: PolicyCfg):
        super().__init__()
        self.cfg = cfg
        self.encoders = nn.ModuleDict({"motion": MotionEncoder(cfg)})
        if cfg.velocity:
            self.encoders["velocity"] = VelocityEncoder(cfg)
        self.decoder = Decoder(cfg)

    def latent(self, obs_hist: torch.Tensor, goal: torch.Tensor, interface: str = "motion") -> torch.Tensor:
        if interface == "velocity":
            return self.encoders["velocity"](goal, obs_hist)
        return self.encoders[interface](goal)

    def forward(self, obs_hist: torch.Tensor, goal: torch.Tensor, interface: str = "motion") -> torch.Tensor:
        return self.decoder(self.latent(obs_hist, goal, interface), obs_hist)

    def decode_frozen(self, z: torch.Tensor, obs_hist: torch.Tensor) -> torch.Tensor:
        """Decoder with stop-gradient on its weights: gradients reach z (and its encoder), not the
        decoder, which stays trained by motion tracking alone (Appendix B)."""
        params = {k: v.detach() for k, v in self.decoder.named_parameters()}
        params.update(dict(self.decoder.named_buffers()))
        return torch.func.functional_call(self.decoder, params, (z, obs_hist))


class Critic(nn.Module):
    """V(canonical state 13+2NJ, goal, xi, root error 4) -> scalar (one per interface).

    ``xi_dim`` defaults to the base 4+NJ; pass the world model's when xi carries hardware DR."""

    def __init__(self, cfg: CriticCfg, goal_dim: int | None = None, pcfg: PolicyCfg | None = None, xi_dim: int | None = None):
        super().__init__()
        pcfg = pcfg or PolicyCfg()
        goal_dim = goal_dim or pcfg.goal_dim
        nj = pcfg.num_joints
        self.mlp = MLP(state_dim(nj) + goal_dim + (xi_dim or base_xi_dim(nj)) + 4, cfg.hidden, 1)
        self.register_buffer("goal_scale", _goal_scale_vector(pcfg, goal_dim))

    def forward(self, feat: torch.Tensor, goal: torch.Tensor, xi_norm: torch.Tensor, e_root: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([feat, goal * self.goal_scale, xi_norm, e_root], dim=-1)).squeeze(-1)


class VelocityCritic(nn.Module):
    """V_vel(canonical state 13+2NJ, normalized command 3, xi) -> scalar."""

    def __init__(self, cfg: CriticCfg, pcfg: PolicyCfg, xi_dim: int):
        super().__init__()
        self.mlp = MLP(state_dim(pcfg.num_joints) + 3 + xi_dim, cfg.hidden, 1)
        self.register_buffer("cmd_scale", 1.0 / torch.tensor(pcfg.cmd_scale, dtype=torch.float32))

    def forward(self, feat: torch.Tensor, cmd: torch.Tensor, xi_norm: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([feat, cmd * self.cmd_scale, xi_norm], dim=-1)).squeeze(-1)
