"""Imagined rollouts through the world model (paper §3.3)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from supermimic.data.replay_buffer import StartBatch
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.imagination.compiled import REWARD_TERM_NAMES, ImagStep
from supermimic.policy.nets import Policy
from supermimic.world_model.heads import ReferenceFrame, analytic_observation, anchor_relative_obs, flatten_history_term_major
from supermimic.world_model.wm_trainer import reference_at
from supermimic.world_model.world_model import WorldModel



def obs_noise_half(nj: int) -> torch.Tensor:
    """Paper Table A.1 sensor noise (uniform half-widths) in the (3 NJ + 6)-d observation order."""
    return torch.tensor([0.01] * nj + [0.5] * nj + [0.2] * 3 + [0.05] * 3 + [0.0] * nj)


ANCHOR_NOISE_HALF = torch.tensor([0.25] * 3 + [0.05] * 6)


@dataclass
class ImagRollout:
    states: torch.Tensor  # (H+1, n, 13+2NJ)
    actions: torch.Tensor  # (H, n, NJ)
    rewards: torch.Tensor  # (H, n)      r_{t+1}
    dhat: torch.Tensor  # (H+1, n)    predicted termination prob of s_t (d_0 = 0)
    survival: torch.Tensor  # (H+1, n)    c_t
    goals: torch.Tensor  # (H+1, n, goal_dim)
    e_root: torch.Tensor  # (H+1, n, 4)
    frames: torch.Tensor  # (H+1, n)
    reward_terms: dict  # mean of each reward term over the rollout (diagnostics)


def imagine(
    wm: WorldModel,
    policy: Policy,
    lib: MotionLibrary,
    start: StartBatch,
    horizon: int,
    future_frames: int = 10,
    obs_noise: bool = True,
    step_fn: ImagStep | None = None,
    anchor_obs: bool = False,
) -> ImagRollout:
    device = start.state.device
    n = start.state.shape[0]
    q0 = lib.default_joint_pos
    step_fn = step_fn or ImagStep(wm, q0)
    noise_half = obs_noise_half(wm.num_joints).to(device)
    anchor_noise_half = ANCHOR_NOISE_HALF.to(device)
    ai = wm.heads.anchor_index

    def add_noise(o):
        if obs_noise:
            return o + (torch.rand_like(o) * 2 - 1) * noise_half
        return o

    def make_goal(frame_t, ref_t, anchor_pose):
        g = lib.goal_window(frame_t, future_frames)
        if not anchor_obs:
            return g
        rel = anchor_relative_obs(anchor_pose[..., 0:3], anchor_pose[..., 3:7], ref_t.body_pos_w[..., ai, :], ref_t.body_quat_w[..., ai, :])
        if obs_noise:
            rel = rel + (torch.rand_like(rel) * 2 - 1) * anchor_noise_half
        return torch.cat([g, rel], dim=-1)

    # Initial observation history from the logged states/actions (noise re-sampled per frame).
    hist = [add_noise(analytic_observation(start.hist_states[i], start.hist_actions[i], start.xi, q0)) for i in range(start.hist_states.shape[0])]
    s = start.state
    frame = start.frame.clone()
    a_prev = start.prev_action
    c = torch.ones(n, device=device)

    states = [s]
    actions, rewards = [], []
    dhats, survivals, frames = [torch.zeros(n, device=device)], [c], [frame]
    # All target frames of the rollout are known up front (frame + t, clamped at the clip end):
    # gather / FK the references for every step in one batched call.
    last = lib.last_frame_of(frame)
    frames_all = torch.minimum(frame[None, :] + torch.arange(horizon + 1, device=device)[:, None], last[None, :])  # (H+1, n)
    ref_all = reference_at(lib, frames_all.reshape(-1))

    def ref_at(t):
        sl = slice(t * n, (t + 1) * n)
        return ReferenceFrame(ref_all.body_pos_w[sl], ref_all.body_quat_w[sl], ref_all.body_lin_vel_w[sl], ref_all.body_ang_vel_w[sl], ref_all.root_pos_w[sl], ref_all.root_quat_w[sl])

    ref = ref_at(0)
    if anchor_obs:
        b0 = wm.heads.robot_bodies(s)
        anchor_pose = torch.cat([b0["pos"][..., ai, :], b0["quat"][..., ai, :]], dim=-1)
    else:
        anchor_pose = None
    goals = [make_goal(frame, ref, anchor_pose)]
    e_roots = [wm.heads.root_error_features(s, ref)]
    term_sum = torch.zeros(len(REWARD_TERM_NAMES), device=device)

    for t in range(horizon):
        obs_flat = flatten_history_term_major(torch.stack(hist, 0))
        a = policy(obs_flat, goals[-1])
        s_next, r, term_vec, logits, obs_clean, anchor_pose = step_fn(s, a, a_prev, start.xi, ref)
        d = torch.sigmoid(logits)
        c = c * (1.0 - d)
        term_sum += term_vec.detach().mean(-1)

        frame = frames_all[t + 1]
        ref = ref_at(t + 1)
        hist = hist[1:] + [add_noise(obs_clean)]
        a_prev = a
        s = s_next
        states.append(s)
        actions.append(a)
        rewards.append(r)
        dhats.append(d)
        survivals.append(c)
        goals.append(make_goal(frame, ref, anchor_pose))
        e_roots.append(wm.heads.root_error_features(s, ref))
        frames.append(frame)

    return ImagRollout(
        states=torch.stack(states),
        actions=torch.stack(actions),
        rewards=torch.stack(rewards),
        dhat=torch.stack(dhats),
        survival=torch.stack(survivals),
        goals=torch.stack(goals),
        e_root=torch.stack(e_roots),
        frames=torch.stack(frames),
        reward_terms={k: float(v / horizon) for k, v in zip(REWARD_TERM_NAMES, term_sum.tolist())},
    )


@dataclass
class VelRollout:
    states: torch.Tensor  # (H+1, n, 13+2NJ)
    actions: torch.Tensor  # (H, n, NJ)
    rewards: torch.Tensor  # (H, n)
    dhat: torch.Tensor  # (H+1, n)
    survival: torch.Tensor  # (H+1, n)
    cmds: torch.Tensor  # (H+1, n, 3) command in force at each state
    align_loss: torch.Tensor  # scalar: latent alignment MSE on reference-command samples (0 if none)
    reward_terms: dict


def imagine_velocity(
    wm: WorldModel,
    policy: Policy,
    lib: MotionLibrary,
    start: StartBatch,
    cmd0: torch.Tensor,
    from_ref: torch.Tensor,
    ref_cmd: torch.Tensor,
    horizon: int,
    step_fn,
    future_frames: int = 10,
    obs_noise: bool = True,
) -> VelRollout:
    """Velocity-interface rollout (Appendix B/G). The decoder runs with frozen weights, so the return
    gradient reaches only the velocity encoder. Samples with ``from_ref`` read their command from the
    reference clip every step and also get the latent alignment target enc_motion(goal) (no gradient)."""
    from supermimic.interfaces.velocity import VELOCITY_TERM_NAMES

    device = start.state.device
    n = start.state.shape[0]
    q0 = lib.default_joint_pos
    noise_half = obs_noise_half(wm.num_joints).to(device)

    def add_noise(o):
        return o + (torch.rand_like(o) * 2 - 1) * noise_half if obs_noise else o

    hist = [add_noise(analytic_observation(start.hist_states[i], start.hist_actions[i], start.xi, q0)) for i in range(start.hist_states.shape[0])]
    s, a_prev = start.state, start.prev_action
    last = lib.last_frame_of(start.frame)
    frames_all = torch.minimum(start.frame[None, :] + torch.arange(horizon + 1, device=device)[:, None], last[None, :])
    ref_ids = from_ref.nonzero().squeeze(-1)
    cmd = torch.where(from_ref[:, None], ref_cmd[frames_all[0]], cmd0)
    c = torch.ones(n, device=device)
    states, actions, rewards = [s], [], []
    dhats, survivals, cmds = [torch.zeros(n, device=device)], [c], [cmd]
    align = torch.zeros((), device=device)
    term_sum = torch.zeros(len(VELOCITY_TERM_NAMES), device=device)
    for t in range(horizon):
        obs_flat = flatten_history_term_major(torch.stack(hist, 0))
        z = policy.encoders["velocity"](cmd, obs_flat)
        if ref_ids.numel() > 0:
            with torch.no_grad():
                z_ref = policy.encoders["motion"](lib.goal_window(frames_all[t, ref_ids], future_frames))
            align = align + ((z[ref_ids] - z_ref) ** 2).mean()
        a = policy.decode_frozen(z, obs_flat)
        s_next, r, term_vec, logits, obs_clean = step_fn(s, a, a_prev, start.xi, cmd)
        d = torch.sigmoid(logits)
        c = c * (1.0 - d)
        term_sum += term_vec.detach().mean(-1)
        hist = hist[1:] + [add_noise(obs_clean)]
        a_prev, s = a, s_next
        cmd = torch.where(from_ref[:, None], ref_cmd[frames_all[t + 1]], cmd)
        states.append(s)
        actions.append(a)
        rewards.append(r)
        dhats.append(d)
        survivals.append(c)
        cmds.append(cmd)
    return VelRollout(
        states=torch.stack(states),
        actions=torch.stack(actions),
        rewards=torch.stack(rewards),
        dhat=torch.stack(dhats),
        survival=torch.stack(survivals),
        cmds=torch.stack(cmds),
        align_loss=align / horizon,
        reward_terms={k: float(v / horizon) for k, v in zip(VELOCITY_TERM_NAMES, term_sum.tolist())},
    )
