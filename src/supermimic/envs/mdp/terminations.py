"""Paper terminations: root (pelvis) height deviation and tilt; clip end as time-out.

Reference-relative terms apply to tracking envs only (``track_mask``); velocity-interface envs end
on a fall (``fall``, Table A.6) or the episode time-out, not at the end of their reference clip."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse

from supermimic.envs.mdp.commands import MultiClipMotionCommand

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def bad_root_height(env: ManagerBasedRlEnv, command_name: str, threshold: float = 0.25) -> torch.Tensor:
    cmd = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    return ((cmd.root_pos_w[:, 2] - cmd.robot_root_pos_w[:, 2]).abs() > threshold) & cmd.track_mask


def bad_root_tilt(env: ManagerBasedRlEnv, command_name: str, threshold: float = 0.8) -> torch.Tensor:
    cmd = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    g = cmd.robot.data.gravity_vec_w
    ref = quat_apply_inverse(cmd.root_quat_w, g)[:, 2]
    rob = quat_apply_inverse(cmd.robot_root_quat_w, g)[:, 2]
    return ((ref - rob).abs() > threshold) & cmd.track_mask


def clip_end(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    cmd = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    return cmd.clip_end & cmd.track_mask


def fall(env: ManagerBasedRlEnv, command_name: str, height: float = 0.30, tilt: float = 0.7) -> torch.Tensor:
    """Root below ``height`` or 1 + u_z > ``tilt`` (u_z: body-frame gravity z), untracked envs only."""
    cmd = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    g = cmd.robot.data.gravity_vec_w
    u_z = quat_apply_inverse(cmd.robot_root_quat_w, g)[:, 2]
    z = cmd.robot_root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return ((z < height) | (1.0 + u_z > tilt)) & ~cmd.track_mask
