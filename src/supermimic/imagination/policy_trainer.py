"""Short-horizon actor–critic through the frozen world model (paper §3.3, Table A.5)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from supermimic.data.replay_buffer import ReplayBuffer
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.imagination.objective import gae_targets, policy_objective
from supermimic.imagination.rollout import ImagRollout, imagine, imagine_velocity
from supermimic.policy.nets import Critic, Policy, VelocityCritic
from supermimic.training.distributed import allreduce_grads
from supermimic.world_model.world_model import WorldModel


@dataclass
class PolicyTrainerCfg:
    horizon: int = 32
    batch_size: int = 4096
    history: int = 10
    future_frames: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 2e-4
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip_norm: float = 1.0
    grad_clip_value: float = 0.1
    critic_lr: float = 2e-4
    critic_grad_clip: float = 1.0
    critic_epochs: int = 5
    critic_minibatches: int = 4
    use_critic: bool = True
    obs_noise: bool = True
    compile: str | None = "default"
    anchor_obs: bool = False  # None | "default" | "reduce-overhead" | "max-autotune-no-cudagraphs"


class VelocityImagination:
    """What the trainer needs for velocity-interface rollouts."""

    def __init__(self, cfg, critic: VelocityCritic, ref_cmd: torch.Tensor, track_mask: torch.Tensor, sampler):
        self.cfg, self.critic, self.ref_cmd, self.track_mask, self.sampler = cfg, critic, ref_cmd, track_mask, sampler


class PolicyTrainer:
    def __init__(self, wm: WorldModel, policy: Policy, critic: Critic, lib: MotionLibrary, cfg: PolicyTrainerCfg, velocity: VelocityImagination | None = None):
        self.wm, self.policy, self.critic, self.lib, self.cfg = wm, policy, critic, lib, cfg
        self.velocity = velocity
        self.opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr, betas=cfg.betas)
        self.critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)
        from supermimic.imagination.compiled import ImagStep, VelStep

        self.step_fn = ImagStep(wm, lib.default_joint_pos, compile_mode=cfg.compile)
        if velocity is not None:
            self.vel_critic_opt = torch.optim.Adam(velocity.critic.parameters(), lr=cfg.critic_lr)
            self.vel_step_fn = VelStep(wm, lib.default_joint_pos, velocity.cfg, compile_mode=cfg.compile)

    def set_lr(self, lr: float) -> None:
        opts = [self.opt, self.critic_opt] + ([self.vel_critic_opt] if self.velocity is not None else [])
        for o in opts:
            for g in o.param_groups:
                g["lr"] = lr

    def _fit_critic(self, critic, opt, inputs: list[torch.Tensor], rewards: torch.Tensor, dhat_next: torch.Tensor) -> dict[str, float]:
        """Fit critic(*inputs) to the GAE targets of one rollout. inputs: (H+1, n, d) each, detached."""
        cfg = self.cfg
        H1, n = rewards.shape[0] + 1, rewards.shape[1]
        with torch.no_grad():
            values = critic(*[x.reshape(H1 * n, -1) for x in inputs]).reshape(H1, n)
            targets = gae_targets(rewards, dhat_next, values, cfg.gamma, cfg.gae_lambda)
            X = [x[:-1].reshape((H1 - 1) * n, -1) for x in inputs]
            Y = targets.reshape(-1)
        N = Y.shape[0]
        mb = N // cfg.critic_minibatches
        last = torch.zeros((), device=Y.device)
        for _ in range(cfg.critic_epochs):
            perm = torch.randperm(N, device=Y.device)
            for i in range(cfg.critic_minibatches):
                idx = perm[i * mb : (i + 1) * mb]
                closs = torch.nn.functional.mse_loss(critic(*[x[idx] for x in X]), Y[idx])
                opt.zero_grad(set_to_none=True)
                closs.backward()
                allreduce_grads(critic)
                torch.nn.utils.clip_grad_norm_(critic.parameters(), cfg.critic_grad_clip)
                opt.step()
                last = closs.detach()
        return {"critic_loss": float(last), "critic_value_mean": float(values.mean()), "critic_target_mean": float(targets.mean())}

    def _velocity_starts(self, buffer: ReplayBuffer, n: int):
        """Reference-command samples start in tracking envs (robot in phase with the clip, so the
        alignment target fits its state); held uniform commands start in velocity envs."""
        v = self.velocity
        hist = self.cfg.history
        n_ref = int(round(n * v.cfg.ref_prob))
        min_h = v.cfg.fall_height + 0.15
        parts, from_ref = [], []
        if n_ref > 0:
            parts.append(buffer.sample_starts(n_ref, hist, env_mask=v.track_mask, min_height=min_h))
            from_ref.append(torch.ones(n_ref, dtype=torch.bool, device=v.track_mask.device))
        if n - n_ref > 0:
            parts.append(buffer.sample_starts(n - n_ref, hist, env_mask=~v.track_mask, min_height=min_h))
            from_ref.append(torch.zeros(n - n_ref, dtype=torch.bool, device=v.track_mask.device))
        start = _cat_starts(parts)
        from_ref = torch.cat(from_ref)
        cmd0 = torch.where(from_ref[:, None], v.ref_cmd[start.frame], v.sampler.uniform(n))
        return start, cmd0, from_ref

    def _critic_inputs(self, ro: ImagRollout):
        H1, n, _ = ro.states.shape
        feat = self.wm.spec.canonical(ro.states.reshape(H1 * n, -1)).reshape(H1, n, -1)
        return feat, ro.goals, ro.e_root

    def update(self, buffer: ReplayBuffer, xi_norm_fn=None) -> dict[str, float]:
        cfg = self.cfg
        v = self.velocity
        n_vel = int(round(cfg.batch_size * v.cfg.batch_fraction)) if v is not None else 0
        start = buffer.sample_starts(cfg.batch_size - n_vel, cfg.history, env_mask=None if v is None else v.track_mask)
        xi_norm = self.wm.spec.normalize_xi(start.xi)

        # ---- actor: BPTT through the frozen world model ------------------------------------
        self.wm.requires_grad_(False)
        self.critic.requires_grad_(False)
        ro = imagine(self.wm, self.policy, self.lib, start, cfg.horizon, cfg.future_frames, obs_noise=cfg.obs_noise, step_fn=self.step_fn, anchor_obs=cfg.anchor_obs)
        feat, goals, e_root = self._critic_inputs(ro)
        if cfg.use_critic:
            v_H = self.critic(feat[-1], goals[-1], xi_norm, e_root[-1])
        else:
            v_H = None
        J = policy_objective(ro, v_H, cfg.gamma)
        loss = -J / cfg.horizon
        if n_vel > 0:
            vstart, cmd0, from_ref = self._velocity_starts(buffer, n_vel)
            vxi_norm = self.wm.spec.normalize_xi(vstart.xi)
            vro = imagine_velocity(self.wm, self.policy, self.lib, vstart, cmd0, from_ref, v.ref_cmd, cfg.horizon, self.vel_step_fn, cfg.future_frames, obs_noise=cfg.obs_noise)
            v.critic.requires_grad_(False)
            vfeat = self.wm.spec.canonical(vro.states.reshape(-1, vro.states.shape[-1])).reshape(*vro.states.shape[:2], -1)
            vv_H = v.critic(vfeat[-1], vro.cmds[-1], vxi_norm) if cfg.use_critic else None
            J_vel = policy_objective(vro, vv_H, cfg.gamma)
            loss = loss - J_vel / cfg.horizon + v.cfg.align_weight * vro.align_loss
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        allreduce_grads(self.policy)
        gn = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.grad_clip_norm)
        torch.nn.utils.clip_grad_value_(self.policy.parameters(), cfg.grad_clip_value)
        self.opt.step()
        self.wm.requires_grad_(True)
        self.critic.requires_grad_(True)
        if n_vel > 0:
            v.critic.requires_grad_(True)

        metrics = {
            "imag_objective": float(J.detach()),
            "imag_return_undisc": float((ro.survival[:-1] * ro.rewards).sum(0).mean().detach()),
            "imag_reward_mean": float(ro.rewards.mean().detach()),
            "imag_survival_H": float(ro.survival[-1].mean().detach()),
            "imag_dhat_mean": float(ro.dhat[1:].mean().detach()),
            "policy_grad_norm": float(gn),
            "action_abs_mean": float(ro.actions.abs().mean().detach()),
        }
        for k, val in ro.reward_terms.items():
            metrics[f"imag_r/{k}"] = val

        # ---- critics: fit to GAE targets of this rollout (targets fixed) --------------------
        if cfg.use_critic:
            H1, n, _ = feat.shape
            m = self._fit_critic(self.critic, self.critic_opt, [feat.detach(), goals.detach(), xi_norm[None].expand(H1, n, -1), e_root.detach()], ro.rewards.detach(), ro.dhat[1:].detach())
            metrics.update(m)
        if n_vel > 0:
            metrics.update(
                vel_imag_return_undisc=float((vro.survival[:-1] * vro.rewards).sum(0).mean().detach()),
                vel_imag_survival_H=float(vro.survival[-1].mean().detach()),
                vel_align_loss=float(vro.align_loss.detach()),
                vel_cmd_abs_mean=float(vro.cmds.abs().mean()),
            )
            for k, val in vro.reward_terms.items():
                metrics[f"vel_imag_r/{k}"] = val
            if cfg.use_critic:
                H1, n, _ = vfeat.shape
                m = self._fit_critic(v.critic, self.vel_critic_opt, [vfeat.detach(), vro.cmds.detach(), vxi_norm[None].expand(H1, n, -1)], vro.rewards.detach(), vro.dhat[1:].detach())
                metrics.update({f"vel_{k}": val for k, val in m.items()})
        return metrics


def _cat_starts(parts):
    from supermimic.data.replay_buffer import StartBatch

    if len(parts) == 1:
        return parts[0]
    return StartBatch(
        state=torch.cat([p.state for p in parts]),
        xi=torch.cat([p.xi for p in parts]),
        frame=torch.cat([p.frame for p in parts]),
        prev_action=torch.cat([p.prev_action for p in parts]),
        hist_states=torch.cat([p.hist_states for p in parts], dim=1),
        hist_actions=torch.cat([p.hist_actions for p in parts], dim=1),
    )
