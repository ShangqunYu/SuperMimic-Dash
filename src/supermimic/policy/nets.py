"""Policy (goal encoder + shared decoder) and critic (paper Table A.4)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from supermimic.world_model.dynamics import MLP
from supermimic.robots import goal_feat_dim, obs_dim, state_dim, xi_dim
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


class Decoder(nn.Module):
    def __init__(self, cfg: PolicyCfg):
        super().__init__()
        self.mlp = MLP(cfg.latent_dim + cfg.obs_dim, cfg.decoder_hidden, cfg.num_joints, last_gain=cfg.decoder_last_gain)
        self.register_buffer("scale", _obs_scale_vector(cfg))

    def forward(self, z: torch.Tensor, obs_hist: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([z, obs_hist * self.scale], dim=-1))


class Policy(nn.Module):
    """a = dec(enc_motion(g), o_hist). Extra goal interfaces can be registered later (plan Appendix B)."""

    def __init__(self, cfg: PolicyCfg):
        super().__init__()
        self.cfg = cfg
        self.encoders = nn.ModuleDict({"motion": MotionEncoder(cfg)})
        self.decoder = Decoder(cfg)

    def forward(self, obs_hist: torch.Tensor, goal: torch.Tensor, interface: str = "motion") -> torch.Tensor:
        return self.decoder(self.encoders[interface](goal), obs_hist)


class Critic(nn.Module):
    """V(canonical state 13+2NJ, goal, xi 4+NJ, root error 4) -> scalar (one per interface)."""

    def __init__(self, cfg: CriticCfg, goal_dim: int | None = None, pcfg: PolicyCfg | None = None):
        super().__init__()
        pcfg = pcfg or PolicyCfg()
        goal_dim = goal_dim or pcfg.goal_dim
        nj = pcfg.num_joints
        self.mlp = MLP(state_dim(nj) + goal_dim + xi_dim(nj) + 4, cfg.hidden, 1)
        self.register_buffer("goal_scale", _goal_scale_vector(pcfg, goal_dim))

    def forward(self, feat: torch.Tensor, goal: torch.Tensor, xi_norm: torch.Tensor, e_root: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([feat, goal * self.goal_scale, xi_norm, e_root], dim=-1)).squeeze(-1)
