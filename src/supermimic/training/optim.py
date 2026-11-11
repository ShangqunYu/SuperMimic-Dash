"""Optimizer bundle: plain Adam, or the official SuperMimic split (Muon on hidden Linear weights, Adam on the rest)."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


def hidden_linear_weights(modules: Sequence[nn.Module]) -> list[torch.Tensor]:
    """Weights of the Linear layers strictly between the first and the last Linear of every nn.Sequential
    (the official code's Muon set: square-ish hidden matrices, not the input / output layers)."""
    out, seen = [], set()
    for module in modules:
        for seq in module.modules():
            if not isinstance(seq, nn.Sequential):
                continue
            for lin in [m for m in seq if isinstance(m, nn.Linear)][1:-1]:
                if id(lin.weight) not in seen:
                    seen.add(id(lin.weight))
                    out.append(lin.weight)
    return out


class Optimizers:
    """One optimizer step over ``modules``. With ``muon``: torch.optim.Muon (Nesterov momentum 0.95, weight decay
    ``muon_weight_decay``, lr matched to AdamW's update RMS) on ``hidden_linear_weights``, Adam on the rest, both at
    ``lr``. Exposes the Adam-like API the trainers use (zero_grad, step, set_lr, state_dict)."""

    def __init__(self, modules: Sequence[nn.Module], lr: float, betas: tuple[float, float] = (0.9, 0.999), weight_decay: float = 0.0, muon: bool = False, muon_weight_decay: float = 0.1):
        params, seen = [], set()
        for m in modules:
            for p in m.parameters():
                if id(p) not in seen:
                    seen.add(id(p))
                    params.append(p)
        hidden = hidden_linear_weights(modules) if muon else []
        hidden_ids = {id(p) for p in hidden}
        rest = [p for p in params if id(p) not in hidden_ids]
        self.opts: list[torch.optim.Optimizer] = [torch.optim.Adam(rest, lr=lr, betas=betas, weight_decay=weight_decay)]
        if hidden:
            self.opts.append(torch.optim.Muon(hidden, lr=lr, weight_decay=muon_weight_decay, adjust_lr_fn="match_rms_adamw"))
        self.params = params

    def zero_grad(self, set_to_none: bool = True) -> None:
        for o in self.opts:
            o.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for o in self.opts:
            o.step()

    def set_lr(self, lr: float) -> None:
        for o in self.opts:
            for g in o.param_groups:
                g["lr"] = lr

    def state_dict(self) -> dict:
        return {"opts": [o.state_dict() for o in self.opts]}

    def load_state_dict(self, sd: dict) -> None:
        sds = sd["opts"] if "opts" in sd else [sd]  # a plain Adam state dict (checkpoints before the bundle)
        assert len(sds) == len(self.opts), f"checkpoint has {len(sds)} optimizers, this run {len(self.opts)}"
        for o, s in zip(self.opts, sds):
            o.load_state_dict(s)
