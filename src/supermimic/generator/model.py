"""Terrain-aware motion generator: a transformer denoiser (x0 prediction) over the future frames of a window,
conditioned on the history frames, the heightmap and the command (supermimic/generator/data.py).

Tokens: [noise level, command, heightmap patches (CNN, stride 8), history frames, future frames], ``patch``
consecutive frames per frame token; the future tokens carry the noise level as well and are read out as the
predicted clean frames. The command
is dropped (``cmd_drop``) in training so it can be guided or left out at sampling time.

Diffusion: cosine schedule over ``T`` steps, the network predicts x0, DDIM sampling with any number of steps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from supermimic.generator.data import CMD_DIM, GRID_SHAPE, Normalizer


@dataclass
class GenCfg:
    frame_dim: int = 27
    hist: int = 10
    fut: int = 60
    d: int = 384
    layers: int = 8
    heads: int = 6
    dropout: float = 0.1
    cmd_drop: float = 0.1
    T: int = 1000
    patch: int = 2  # frames per token (hist and fut must divide)


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    f = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    a = t.float()[:, None] * f[None]
    return torch.cat([torch.cos(a), torch.sin(a)], -1)


class Denoiser(nn.Module):
    def __init__(self, cfg: GenCfg):
        super().__init__()
        self.cfg = cfg
        d = cfg.d
        self.norm = Normalizer(cfg.frame_dim)
        P = cfg.patch
        assert cfg.hist % P == 0 and cfg.fut % P == 0
        self.frame_in = nn.Linear(cfg.frame_dim * P, d)
        self.kind = nn.Parameter(torch.zeros(2, d))  # history, future
        self.pos = nn.Parameter(torch.randn((cfg.hist + cfg.fut) // P, d) * 0.02)
        self.hmap_cnn = nn.Sequential(
            nn.Conv2d(1, 32, 5, 2, 2), nn.GELU(), nn.Conv2d(32, 64, 3, 2, 1), nn.GELU(), nn.Conv2d(64, d, 3, 2, 1)
        )
        with torch.no_grad():
            n_hm = self.hmap_cnn(torch.zeros(1, 1, *GRID_SHAPE)).flatten(2).shape[-1]
        self.hm_pos = nn.Parameter(torch.randn(n_hm, d) * 0.02)
        self.cmd_in = nn.Sequential(nn.Linear(CMD_DIM + 1, d), nn.GELU(), nn.Linear(d, d))
        self.t_in = nn.Sequential(nn.Linear(256, d), nn.GELU(), nn.Linear(d, d))
        layer = nn.TransformerEncoderLayer(d, cfg.heads, 4 * d, cfg.dropout, activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, cfg.layers, enable_nested_tensor=False)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, cfg.frame_dim * P))

    def forward(self, x_t, t, hist, hmap, cmd, cmd_on):
        """All normalised: x_t (B, F, D) noisy future, t (B,) step, hist (B, H, D), hmap (B, X, Y), cmd (B, C),
        cmd_on (B,) 1 = command given -> (B, F, D) predicted clean future."""
        te = self.t_in(timestep_embedding(t, 256))
        c = self.cmd_in(torch.cat([cmd * cmd_on[:, None], cmd_on[:, None]], -1))
        hm = self.hmap_cnn(hmap[:, None]).flatten(2).transpose(1, 2) + self.hm_pos
        B, F, D = x_t.shape
        P = self.cfg.patch
        H = hist.shape[1] // P
        h = self.frame_in(hist.reshape(B, H, P * D)) + self.kind[0] + self.pos[:H]
        f = self.frame_in(x_t.reshape(B, F // P, P * D)) + self.kind[1] + self.pos[H:] + te[:, None]
        z = self.enc(torch.cat([te[:, None], c[:, None], hm, h, f], 1))
        return self.out(z[:, -(F // P) :]).reshape(B, F, D)


class Diffusion:
    def __init__(self, T: int = 1000, device="cpu"):
        self.T = T
        s = 0.008
        x = torch.linspace(0, T, T + 1, dtype=torch.float64)
        ab = torch.cos(((x / T) + s) / (1 + s) * math.pi / 2) ** 2
        ab = ab / ab[0]
        self.ab = ab[1:].clamp(1e-5, 1.0).float().to(device)  # alpha_bar at steps 0 .. T-1

    def q_sample(self, x0, t, noise):
        a = self.ab[t][:, None, None]
        return a.sqrt() * x0 + (1 - a).sqrt() * noise

    @torch.no_grad()
    def sample(self, model: Denoiser, hist, hmap, cmd, cmd_on, steps: int = 20, guidance: float = 1.0, eta: float = 0.0, generator=None):
        """DDIM from pure noise; inputs normalised as for forward. guidance > 1: classifier-free on the command."""
        B, F, D = hist.shape[0], model.cfg.fut, model.cfg.frame_dim
        x = torch.randn(B, F, D, device=hist.device, generator=generator)
        ts = torch.linspace(self.T - 1, 0, steps, device=hist.device).round().long()
        for i, t in enumerate(ts):
            tt = t.expand(B)
            x0 = model(x, tt, hist, hmap, cmd, cmd_on)
            if guidance != 1.0:
                x0u = model(x, tt, hist, hmap, cmd, torch.zeros_like(cmd_on))
                x0 = x0u + guidance * (x0 - x0u)
            a = self.ab[t]
            if i == len(ts) - 1:
                x = x0
                break
            ap = self.ab[ts[i + 1]]
            eps = (x - a.sqrt() * x0) / (1 - a).sqrt()
            sig = eta * ((1 - ap) / (1 - a) * (1 - a / ap)).sqrt()
            x = ap.sqrt() * x0 + (1 - ap - sig**2).clamp_min(0).sqrt() * eps
            if eta > 0:
                x = x + sig * torch.randn(x.shape, device=x.device, generator=generator)
        return x
