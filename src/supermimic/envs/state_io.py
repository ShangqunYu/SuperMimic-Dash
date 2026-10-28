"""Read / write the full simulator state (qpos | qvel) as one (13 + 2 NJ)-d tensor."""

from __future__ import annotations

import torch


class StateIO:
    def __init__(self, env):
        self.env = env
        self.sim = env.sim
        # States are stored relative to each env's origin so that they are directly comparable
        # with the (origin-free) motion library and translation-invariant across envs.
        self.origins = env.scene.env_origins.clone()  # (N, 3)
        self.nq = int(self.sim.data.qpos.shape[-1])
        self.nv = int(self.sim.data.qvel.shape[-1])
        # A single free-floating robot and nothing else in the scene: qpos = 7 + NJ, qvel = 6 + NJ.
        assert self.nq == self.nv + 1, (self.nq, self.nv)
        self.num_joints = self.nq - 7
        self.state_dim = self.nq + self.nv

    def read(self) -> torch.Tensor:
        """(N, 13 + 2 NJ) copy of [qpos | qvel]."""
        # Index with [:] to get plain tensors out of mjlab's TorchArray proxies (torch.cat on the
        # proxies recurses inside their __torch_function__).
        s = torch.cat([self.sim.data.qpos[:], self.sim.data.qvel[:]], dim=-1).clone()
        s[:, 0:3] -= self.origins
        return s

    @torch.no_grad()
    def write(self, env_ids: torch.Tensor, state: torch.Tensor, forward: bool = True) -> None:
        """Overwrite qpos/qvel of ``env_ids`` and refresh derived quantities."""
        state = state.to(self.sim.data.qpos.dtype).clone()
        state[:, 0:3] += self.origins[env_ids]
        self.sim.data.qpos[env_ids] = state[:, : self.nq]
        self.sim.data.qvel[env_ids] = state[:, self.nq :]
        if hasattr(self.sim.data, "qacc_warmstart"):
            self.sim.data.qacc_warmstart[env_ids] = 0.0
        if forward:
            self.sim.forward()
