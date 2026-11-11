"""Dynamics MLP f_omega (paper Table A.4: MLP (2048, 2048, 2048)) and the shared MLP / input normaliser."""

from __future__ import annotations

import torch
import torch.nn as nn

ACTIVATIONS = {"silu": nn.SiLU, "gelu": nn.GELU, "elu": nn.ELU}


class MLP(nn.Module):
    """(Linear [-> LayerNorm] -> act)* -> Linear. Official SuperMimic: LayerNorm + GELU everywhere."""

    def __init__(self, in_dim: int, hidden: tuple[int, ...], out_dim: int, act: str = "silu", zero_last: bool = False, last_gain: float | None = None, layer_norm: bool = False):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), *([nn.LayerNorm(h)] if layer_norm else []), ACTIVATIONS[act]()]
            d = h
        last = nn.Linear(d, out_dim)
        if zero_last:
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        elif last_gain is not None:
            nn.init.orthogonal_(last.weight, gain=last_gain)
            nn.init.zeros_(last.bias)
        layers.append(last)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RunningNorm(nn.Module):
    """Running mean / std of an input, (x - mean) / (std + eps) (rsl_rl's EmpiricalNormalization, as in the
    official code). ``update`` folds in a batch of fresh data; every buffer changes in place, so compiled
    graphs that read them stay valid. Identity until the first update."""

    def __init__(self, dim: int, eps: float = 1e-2):
        super().__init__()
        self.eps = eps
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("std", torch.ones(dim))
        self.register_buffer("count", torch.zeros((), dtype=torch.float64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.std + self.eps)

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.reshape(-1, x.shape[-1]).float()
        n = torch.tensor(float(x.shape[0]), dtype=torch.float64, device=x.device)
        mean_x = x.mean(0)
        var_x = x.var(0, unbiased=False)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            # Global batch statistics: the global mean, then the local variances around it.
            local_n, local_mean = n.clone(), mean_x
            torch.distributed.all_reduce(n)
            s = mean_x * float(local_n)
            torch.distributed.all_reduce(s)
            mean_x = s / float(n)
            v = float(local_n) * (var_x + (local_mean - mean_x).square())
            torch.distributed.all_reduce(v)
            var_x = v / float(n)
        self.count += n
        rate = (n / self.count).float()
        delta = mean_x - self.mean
        self.mean += rate * delta
        self.var += rate * (var_x - self.var + delta * (mean_x - self.mean))
        self.std.copy_(self.var.sqrt())


class DynamicsMLP(nn.Module):
    """delta = f(canonical(s), a, xi_norm). Zero-initialised last layer (zero output = the constant-velocity
    prior at init). With ``input_norm`` the state features and the action pass running normalisers first."""

    def __init__(self, feat_dim: int, act_dim: int, xi_dim: int, out_dim: int, hidden=(2048, 2048, 2048), act: str = "silu", layer_norm: bool = False, input_norm: bool = False):
        super().__init__()
        self.mlp = MLP(feat_dim + act_dim + xi_dim, tuple(hidden), out_dim, act=act, zero_last=True, layer_norm=layer_norm)
        self.input_norm = input_norm
        if input_norm:
            self.feat_norm = RunningNorm(feat_dim)
            self.action_norm = RunningNorm(act_dim)

    def forward(self, feat: torch.Tensor, action: torch.Tensor, xi_norm: torch.Tensor) -> torch.Tensor:
        if self.input_norm:
            feat, action = self.feat_norm(feat), self.action_norm(action)
        return self.mlp(torch.cat([feat, action, xi_norm], dim=-1))

    def update_normalization(self, feat: torch.Tensor, action: torch.Tensor) -> None:
        if self.input_norm:
            self.feat_norm.update(feat)
            self.action_norm.update(action)
