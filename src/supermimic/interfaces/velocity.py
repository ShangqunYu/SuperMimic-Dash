"""Velocity interface (paper Appendix B, G; Table A.6): command g = (v_x, v_y, w_z).

v_x, v_y: planar root velocity in the heading (yaw-only) frame; w_z: world yaw rate.

* ``reference_commands``  the command a reference clip implies at every library frame: 25-frame
                          centred mean of the root velocity in the reference heading frame and of
                          the world yaw rate (clamped to the clip).
* ``velocity_reward_terms`` Table A.6 reward on a (13 + 2 NJ) state, dt-scaled like the tracking reward.
* ``fall_logits``         fall-only termination (root height, tilt) as logits; used analytically in
                          imagination (the learned termination head is reference-relative).
* ``CommandSampler``      reference commands or held uniform commands (10% zero), resampled
                          every ``resample_steps``; shared by the simulator and imagination.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from supermimic.kinematics.rotations import projected_gravity, quat_apply, quat_apply_inverse, yaw_quat
from supermimic.world_model.heads import split_state


@dataclass
class VelocityCfg:
    enabled: bool = False
    env_fraction: float = 0.5  # share of simulator envs driven by velocity commands
    batch_fraction: float = 0.5  # share of each imagination batch
    # Command ranges (Dash). Paper (G1): vx [-1.5, 2], vy [-1.32, 1.32], wz [-3.35, 3.35].
    # Dash defaults sit inside the 1st-99th percentiles of the reference commands of the Dash
    # BONES-SEED library: vx (-0.85, 1.71), vy (-0.76, 0.77), wz (-1.86, 1.86).
    vx_range: tuple[float, float] = (-0.8, 1.5)
    vy_range: tuple[float, float] = (-0.6, 0.6)
    wz_range: tuple[float, float] = (-1.8, 1.8)
    zero_prob: float = 0.1
    ref_prob: float = 0.5  # read the command from the reference clip each step
    resample_steps: int = 250
    ref_window: int = 25
    # Reward (Table A.6), each term x dt.
    std_lin_vel: float = 0.5
    var_yaw_rate: float = 0.5
    w_lin_vel: float = 2.0
    w_yaw_rate: float = 2.0
    w_action_rate: float = -0.3
    w_joint_limit: float = -2.0
    w_lateral_lean: float = -4.0
    w_roll_pitch_rate: float = -0.05
    dt: float = 0.02
    # Fall termination. Paper G1: pelvis below 0.35 m (of ~0.79) or 1 + u_z > 0.7; Dash's torso stands at 0.67 m.
    fall_height: float = 0.30
    fall_tilt: float = 0.7
    fall_temperature: tuple[float, float] = (0.02, 0.05)  # logit scale (m, tilt units) in imagination
    align_weight: float = 0.1
    encoder_hidden: tuple[int, ...] = (256, 256)

    @property
    def cmd_scale(self) -> tuple[float, float, float]:
        """Per-component normalisation for network inputs: the larger |range end|."""
        return tuple(max(abs(lo), abs(hi)) for lo, hi in (self.vx_range, self.vy_range, self.wz_range))  # type: ignore[return-value]

    @property
    def max_step_reward(self) -> float:
        return self.dt * (self.w_lin_vel + self.w_yaw_rate)


VELOCITY_TERM_NAMES = ("lin_vel", "yaw_rate", "action_rate", "joint_limit", "lateral_lean", "roll_pitch_rate")


@torch.no_grad()
def reference_commands(lib, window: int = 25, chunk: int = 1 << 22) -> torch.Tensor:
    """(num_frames, 3) command implied by the reference at every library frame."""
    nf = lib.num_frames
    dev = lib.device
    inst = torch.empty(nf, 3, device=dev, dtype=torch.float64)
    for i in range(0, nf, chunk):
        r = lib.ref(torch.arange(i, min(i + chunk, nf), device=dev))
        v_h = quat_apply_inverse(yaw_quat(r.root_quat_w), r.root_lin_vel_w)
        inst[i : i + len(v_h), 0:2] = v_h[:, 0:2].double()
        inst[i : i + len(v_h), 2] = r.root_ang_vel_w[:, 2].double()
    csum = torch.cat([torch.zeros(1, 3, device=dev, dtype=torch.float64), inst.cumsum(0)])
    f = torch.arange(nf, device=dev)
    clip = lib.frame_clip[f]
    half = window // 2
    lo = torch.maximum(f - half, lib.clip_start[clip])
    hi = torch.minimum(f + half, lib.clip_last[clip])
    out = (csum[hi + 1] - csum[lo]) / (hi - lo + 1).double()[:, None]
    return out.float()


def velocity_reward_terms(s_next: torch.Tensor, action: torch.Tensor, prev_action: torch.Tensor, cmd: torch.Tensor, soft_limits: torch.Tensor, cfg: VelocityCfg) -> dict[str, torch.Tensor]:
    """Per-term weighted, dt-scaled rewards (N,) plus 'total'. The root body is the torso (Dash) / pelvis (G1)."""
    _, quat, q, lin_w, ang_b, _ = split_state(s_next)
    v_h = quat_apply_inverse(yaw_quat(quat), lin_w)
    yaw_rate = quat_apply(quat, ang_b)[..., 2]  # world-frame yaw rate
    e_lin = ((v_h[..., 0:2] - cmd[..., 0:2]) ** 2).sum(-1)
    e_yaw = (yaw_rate - cmd[..., 2]) ** 2
    g = projected_gravity(quat)
    lo, hi = soft_limits[..., 0], soft_limits[..., 1]
    joint_limit = ((-(q - lo)).clamp(min=0.0) + (q - hi).clamp(min=0.0)).sum(-1)
    dt = cfg.dt
    terms = {
        "lin_vel": dt * cfg.w_lin_vel * torch.exp(-e_lin / cfg.std_lin_vel**2),
        "yaw_rate": dt * cfg.w_yaw_rate * torch.exp(-e_yaw / cfg.var_yaw_rate),
        "action_rate": dt * cfg.w_action_rate * ((action - prev_action) ** 2).sum(-1),
        "joint_limit": dt * cfg.w_joint_limit * joint_limit,
        # Sideways lean only: sin^2(roll) = (body-frame gravity y)^2 for any pitch with zero roll -> 0.
        "lateral_lean": dt * cfg.w_lateral_lean * g[..., 1] ** 2,
        "roll_pitch_rate": dt * cfg.w_roll_pitch_rate * (ang_b[..., 0:2] ** 2).sum(-1),
    }
    total = terms["lin_vel"]
    for k in VELOCITY_TERM_NAMES[1:]:
        total = total + terms[k]
    terms["total"] = total
    return terms


def velocity_of(s: torch.Tensor) -> torch.Tensor:
    """(…, 3) achieved (vx, vy in the heading frame, world yaw rate) of a state, comparable to a command."""
    _, quat, _, lin_w, ang_b, _ = split_state(s)
    v_h = quat_apply_inverse(yaw_quat(quat), lin_w)
    return torch.cat([v_h[..., 0:2], quat_apply(quat, ang_b)[..., 2:3]], dim=-1)


def fall_features(s: torch.Tensor) -> torch.Tensor:
    """[root height, 1 + u_z] (…, 2); u_z = body-frame gravity z (-1 upright)."""
    return torch.stack([s[..., 2], 1.0 + projected_gravity(s[..., 3:7])[..., 2]], dim=-1)


def fallen(s: torch.Tensor, cfg: VelocityCfg) -> torch.Tensor:
    f = fall_features(s)
    return (f[..., 0] < cfg.fall_height) | (f[..., 1] > cfg.fall_tilt)


def fall_logits(s: torch.Tensor, cfg: VelocityCfg) -> torch.Tensor:
    """Smooth version of ``fallen`` (logit > 0 exactly when fallen)."""
    f = fall_features(s)
    th, tt = cfg.fall_temperature
    return torch.maximum((cfg.fall_height - f[..., 0]) / th, (f[..., 1] - cfg.fall_tilt) / tt)


class CommandSampler:
    """Per-slot command state: reference-driven (read every step) or held uniform."""

    def __init__(self, n: int, ref_cmd: torch.Tensor, cfg: VelocityCfg, device):
        self.cfg = cfg
        self.ref_cmd = ref_cmd
        self.cmd = torch.zeros(n, 3, device=device)
        self.from_ref = torch.zeros(n, dtype=torch.bool, device=device)
        self.timer = torch.zeros(n, dtype=torch.long, device=device)
        lo = torch.tensor([cfg.vx_range[0], cfg.vy_range[0], cfg.wz_range[0]], device=device)
        hi = torch.tensor([cfg.vx_range[1], cfg.vy_range[1], cfg.wz_range[1]], device=device)
        self.lo, self.hi = lo, hi
        self.scale = torch.tensor(cfg.cmd_scale, device=device)

    def uniform(self, n: int) -> torch.Tensor:
        c = self.lo + torch.rand(n, 3, device=self.lo.device) * (self.hi - self.lo)
        return torch.where(torch.rand(n, 1, device=c.device) < self.cfg.zero_prob, torch.zeros_like(c), c)

    def resample(self, ids: torch.Tensor, frame: torch.Tensor) -> None:
        """New command source for slots ``ids`` (frame: their current library frames)."""
        n = ids.numel()
        if n == 0:
            return
        self.from_ref[ids] = torch.rand(n, device=self.cmd.device) < self.cfg.ref_prob
        self.cmd[ids] = torch.where(self.from_ref[ids, None], self.ref_cmd[frame], self.uniform(n))
        self.timer[ids] = 0

    def step(self, frame: torch.Tensor, clip_over: torch.Tensor | None = None) -> None:
        """Advance one control step: refresh reference commands, resample on the timer.

        clip_over: slots whose reference clip has ended; a reference command there switches to a
        held uniform one (the clip no longer implies a command)."""
        self.timer += 1
        due = self.timer >= self.cfg.resample_steps
        if due.any():
            self.resample(due.nonzero().squeeze(-1), frame[due])
        if clip_over is not None:
            stop = clip_over & self.from_ref
            if stop.any():
                ids = stop.nonzero().squeeze(-1)
                self.from_ref[ids] = False
                self.cmd[ids] = self.uniform(ids.numel())
        self.cmd = torch.where(self.from_ref[:, None], self.ref_cmd[frame], self.cmd)

    def normalized(self, cmd: torch.Tensor | None = None) -> torch.Tensor:
        return (self.cmd if cmd is None else cmd) / self.scale
