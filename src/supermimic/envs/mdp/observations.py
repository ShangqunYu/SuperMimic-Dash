from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.managers.event_manager import requires_model_fields

from supermimic.envs.mdp.commands import MultiClipMotionCommand

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def goal_future_frames(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """(N, K*(2NJ+10)) future reference-frame features (plan D5)."""
    command = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    return command.goal


class policy_terrain:
    """(N, POLICY_SCAN_DIM) teacher terrain input: root height above the POLICY_GRID points of the env's
    terrain (supermimic/terrain/height_scan.py), the same function the imagined rollouts use."""

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        from supermimic.terrain.height_scan import POLICY_GRID, HeightScan

        self.command = cast(MultiClipMotionCommand, env.command_manager.get_term(cfg.params["command_name"]))
        self.scan = HeightScan(self.command.terrains, POLICY_GRID).to(env.device)

    def reset(self, env_ids=None) -> None:
        pass

    def __call__(self, env: ManagerBasedRlEnv, command_name: str, noise: bool = False) -> torch.Tensor:
        from supermimic.terrain.height_scan import policy_terrain_obs

        del command_name
        s = env.sim.data.qpos[:][:, :7].clone()
        s[:, :3] -= env.scene.env_origins
        return policy_terrain_obs(self.scan, s, self.command.clip_tile, noise)


@requires_model_fields("geom_pos", "geom_size", "geom_rbound", "geom_aabb")
def expand_terrain_fields(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None) -> None:
    """Startup no-op whose only job is to give the terrain slot geoms per-world fields (written at reset by
    MultiClipMotionCommand._place_on_terrain)."""
    del env, env_ids
