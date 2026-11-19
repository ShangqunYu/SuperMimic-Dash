"""One imagined step as a single compiled function (plan 3.6: launch-overhead control).

``ImagStep`` bundles world-model dynamics, analytic FK reward, termination logits and the
next noise-free observation into one call so ``torch.compile`` can fuse the hundreds of
small kernels of the per-body FK loop. Reference quantities are passed as plain tensors
(already gathered from the motion library) so the compiled graph has static shapes.
"""

from __future__ import annotations

import torch

from supermimic.world_model.heads import ReferenceFrame, analytic_observation
from supermimic.world_model.world_model import WorldModel

REWARD_TERM_NAMES = (
    "motion_global_root_pos",
    "motion_global_root_ori",
    "motion_body_pos",
    "motion_body_ori",
    "motion_body_lin_vel",
    "motion_body_ang_vel",
    "action_rate_l2",
    "joint_limit",
)


class ImagStep:
    """edge: an EdgeTerm (generator/metrics.py) subtracted from the reward, on the step's own FK of the feet (run
    eagerly on its own FK it cost ~2/3 of a teacher iteration). It is compiled as a separate graph: inside the step's
    graph the policy update went NaN at once (2026-11-19; the term alone compiled matches eager values and gradients).
    Its outputs (penalty, deepest point) come back as a (2, n) tensor, None without it."""

    def __init__(self, wm: WorldModel, default_joint_pos: torch.Tensor, compile_mode: str | None = None, edge=None):
        self.wm = wm
        self.q0 = default_joint_pos
        self.edge = edge
        if edge is not None:
            tracked = [int(i) for i in wm.heads.tracked_idx]
            self.feet = [tracked.index(int(f)) for f in edge.feet]
        self._fn = self._step
        self._edge_fn = edge.from_feet if edge is not None else None
        if compile_mode:
            mode = None if compile_mode == "default" else compile_mode
            self._fn = torch.compile(self._step, mode=mode, dynamic=False)
            if edge is not None:
                self._edge_fn = torch.compile(edge.from_feet, mode=mode, dynamic=False)

    def _step(self, s, a, a_prev, xi, ref_body_pos, ref_body_quat, ref_body_lin, ref_body_ang, ref_root_pos, ref_root_quat):
        ref = ReferenceFrame(ref_body_pos, ref_body_quat, ref_body_lin, ref_body_ang, ref_root_pos, ref_root_quat)
        s_next = self.wm.step(s, a, xi)
        bodies = self.wm.heads.robot_bodies(s_next)
        terms = self.wm.heads.reward(s_next, a, a_prev, ref, bodies)
        r = terms["total"]
        feet = None
        if self.edge is not None:
            f = self.feet
            feet = (bodies["pos"][:, f], bodies["quat"][:, f], bodies["lin_vel"][:, f], bodies["ang_vel"][:, f])
        term_vec = torch.stack([terms[k] for k in REWARD_TERM_NAMES], dim=0)  # (8, n)
        logits = self.wm.termination_logits(s_next, ref)
        obs_clean = analytic_observation(s_next, a, xi, self.q0)
        ai = self.wm.heads.anchor_index
        anchor_pose = torch.cat([bodies["pos"][..., ai, :], bodies["quat"][..., ai, :]], dim=-1)  # (n, 7)
        return s_next, r, term_vec, logits, obs_clean, anchor_pose, feet

    def __call__(self, s, a, a_prev, xi, ref: ReferenceFrame, tile=None):
        s_next, r, term_vec, logits, obs_clean, anchor_pose, feet = self._fn(
            s, a, a_prev, xi, ref.body_pos_w, ref.body_quat_w, ref.body_lin_vel_w, ref.body_ang_vel_w, ref.root_pos_w, ref.root_quat_w)
        edge_out = None
        if feet is not None:
            pen, deepest = self._edge_fn(*feet, tile)
            r = r - pen
            edge_out = torch.stack([pen.detach(), deepest])
        return s_next, r, term_vec, logits, obs_clean, anchor_pose, edge_out


class VelStep:
    """One imagined step of the velocity interface: dynamics, Table A.6 reward, analytic fall logits."""

    def __init__(self, wm: WorldModel, default_joint_pos: torch.Tensor, vcfg, compile_mode: str | None = None):
        self.wm = wm
        self.q0 = default_joint_pos
        self.vcfg = vcfg
        self._fn = self._step
        if compile_mode:
            mode = None if compile_mode == "default" else compile_mode
            self._fn = torch.compile(self._step, mode=mode, dynamic=False)

    def _step(self, s, a, a_prev, xi, cmd):
        from supermimic.interfaces.velocity import VELOCITY_TERM_NAMES, fall_logits, velocity_reward_terms

        s_next = self.wm.step(s, a, xi)
        terms = velocity_reward_terms(s_next, a, a_prev, cmd, self.wm.heads.soft_limits, self.vcfg)
        term_vec = torch.stack([terms[k] for k in VELOCITY_TERM_NAMES], dim=0)
        logits = fall_logits(s_next, self.vcfg)
        obs_clean = analytic_observation(s_next, a, xi, self.q0)
        return s_next, terms["total"], term_vec, logits, obs_clean

    def __call__(self, s, a, a_prev, xi, cmd):
        return self._fn(s, a, a_prev, xi, cmd)
