"""Train the terrain-aware motion generator (supermimic/generator) on a packed library.

  python scripts/generator/train_generator.py --run gen_v0 [--steps 300000] [--batch 512] [--resume PATH]

Logs to logs/<run>/ (TensorBoard tb/, ckpt_<step>.pt, ckpt_latest.pt). Every --eval-every steps the EMA
model samples held-out windows (half terrain) and their foot quality is compared with the mocap's.
"""

from __future__ import annotations

import copy
import dataclasses
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as Fn
import tyro
from torch.utils.tensorboard import SummaryWriter

from supermimic.generator.data import GenData, from_canonical
from supermimic.generator.metrics import FootMetrics
from supermimic.generator.model import Denoiser, Diffusion, GenCfg

REPO = Path(__file__).resolve().parents[2]


@dataclass
class Args:
    run: str = "gen_v0"
    library: str = str(REPO / "data/library/dash_flat_terrain_v5_parc_compact.pt")
    model: GenCfg = field(default_factory=GenCfg)
    steps: int = 300_000
    batch: int = 512
    lr: float = 2e-4
    min_lr_frac: float = 0.05
    warmup: int = 2000
    weight_decay: float = 0.01
    vel_weight: float = 1.0  # loss on frame-to-frame differences (x VEL_GAIN, normalised units)
    terrain_frac: float = 0.5
    hist_noise: float = 0.0  # std (normalised units) added to the history frames in training
    ema: float = 0.999
    log_every: int = 200
    eval_every: int = 5000
    ckpt_every: int = 20000
    eval_windows: int = 512
    sample_steps: int = 20
    device: str = "cuda:0"
    seed: int = 0
    resume: str | None = None
    compile: bool = True


VEL_GAIN = 10.0


def loss_fn(model: Denoiser, diff: Diffusion, w, a: Args, generator=None):
    n = model.norm
    x0, hist = n.frames(w.fut), n.frames(w.hist)
    if a.hist_noise > 0:
        hist = hist + a.hist_noise * torch.randn_like(hist)
    hm, cmd = n.hmap(w.hmap), n.cmd(w.cmd)
    B = x0.shape[0]
    dev = x0.device
    cmd_on = (torch.rand(B, device=dev, generator=generator) >= model.cfg.cmd_drop).float() if model.training else torch.ones(B, device=dev)
    t = torch.randint(diff.T, (B,), device=dev, generator=generator)
    xt = diff.q_sample(x0, t, torch.randn(x0.shape, device=dev, generator=generator))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        x0h = model(xt, t, hist, hm, cmd, cmd_on).float()
    lx = Fn.mse_loss(x0h, x0)
    last = n.frames(w.hist[:, -1:])
    lv = Fn.mse_loss(VEL_GAIN * torch.cat([last, x0h], 1).diff(dim=1), VEL_GAIN * torch.cat([last, x0], 1).diff(dim=1))
    return lx, lv, t


@torch.no_grad()
def evaluate(ema: Denoiser, diff: Diffusion, data: GenData, feet: FootMetrics, a: Args) -> dict[str, float]:
    g = torch.Generator(device=a.device).manual_seed(1234)
    out = {}
    # denoising loss by noise level
    ema.eval()
    w = data.sample(4096, "val", a.terrain_frac, generator=g)
    lx, lv, t = loss_fn(ema, diff, w, a, generator=g)
    out["val/loss_x"], out["val/loss_v"] = float(lx), float(lv)
    # samples: foot quality vs the mocap of the same windows, command path following
    w = data.sample(a.eval_windows, "val", 0.5, generator=g)
    n = ema.norm
    on = torch.ones(w.cmd.shape[0], device=w.cmd.device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        x = diff.sample(ema, n.frames(w.hist), n.hmap(w.hmap), n.cmd(w.cmd), on, steps=a.sample_steps, generator=g).float()
    gen = n.frames_inv(x)
    k = a.eval_windows // 2
    for name, fut in (("gen", gen), ("mocap", w.fut)):
        for kind, sl in (("terrain", slice(0, k)), ("flat", slice(k, None))):
            pos, quat, q = from_canonical(fut[sl], w.origin[sl], w.yaw0[sl])
            for m, v in feet(pos, quat, q, w.tile[sl]).items():
                out[f"{kind}_{name}/{m}"] = v
            out[f"{kind}_{name}/end_xy_err_m"] = float((fut[sl, -1, :2] - w.fut[sl, -1, :2]).norm(dim=-1).mean())
            out[f"{kind}_{name}/end_z_err_m"] = float((fut[sl, -1, 2] - w.fut[sl, -1, 2]).abs().mean())
    return out


def main(a: Args) -> None:
    torch.manual_seed(a.seed)
    log = REPO / "logs" / a.run
    log.mkdir(parents=True, exist_ok=True)
    data = GenData(a.library, a.device, hist=a.model.hist, fut=a.model.fut)
    print(f"clips {len(data.clip_names)} ({int(data.terrain.sum())} terrain), frames {data.ref.shape[0]:,}; windows train terrain "
          f"{data.num('train', 'terrain'):,} flat {data.num('train', 'flat'):,}, val terrain {data.num('val', 'terrain'):,} flat {data.num('val', 'flat'):,}", flush=True)
    model = Denoiser(a.model).to(a.device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.99), weight_decay=a.weight_decay)
    step = 0
    if a.resume:
        ck = torch.load(a.resume, map_location=a.device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        ema = copy.deepcopy(model)
        ema.load_state_dict(ck["ema"])
    else:
        model.norm.fit(data)
        ema = copy.deepcopy(model)
    ema.eval().requires_grad_(False)
    print(f"model {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M parameters", flush=True)
    diff = Diffusion(a.model.T, a.device)
    feet = FootMetrics(data.scan, a.device)
    train_model = torch.compile(model) if a.compile else model
    writer = SummaryWriter(str(log / "tb"))

    def lr_at(s):
        if s < a.warmup:
            return a.lr * (s + 1) / a.warmup
        p = min(1.0, (s - a.warmup) / max(1, a.steps - a.warmup))
        return a.lr * (a.min_lr_frac + (1 - a.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * p)))

    def save(tag):
        ck = {"step": step, "cfg": dataclasses.asdict(a.model), "args": dataclasses.asdict(a), "model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict()}
        torch.save(ck, log / f"ckpt_{tag}.pt")

    t_last, acc = time.time(), [0.0, 0.0, 0]
    while step < a.steps:
        model.train()
        for gr in opt.param_groups:
            gr["lr"] = lr_at(step)
        w = data.sample(a.batch, "train", a.terrain_frac)
        lx, lv, _ = loss_fn(train_model, diff, w, a)
        loss = lx + a.vel_weight * lv
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - a.ema)
        step += 1
        acc[0] += float(lx.detach())
        acc[1] += float(lv.detach())
        acc[2] += 1
        if step % a.log_every == 0:
            dt = time.time() - t_last
            t_last = time.time()
            writer.add_scalar("train/loss_x", acc[0] / acc[2], step)
            writer.add_scalar("train/loss_v", acc[1] / acc[2], step)
            writer.add_scalar("train/grad_norm", float(gn), step)
            writer.add_scalar("train/lr", lr_at(step), step)
            print(f"step {step} loss_x {acc[0] / acc[2]:.4f} loss_v {acc[1] / acc[2]:.4f} gn {float(gn):.2f} lr {lr_at(step):.2e} {a.log_every / dt:.1f} it/s", flush=True)
            acc = [0.0, 0.0, 0]
        if step % a.eval_every == 0 or step == a.steps:
            ev = evaluate(ema, diff, data, feet, a)
            for k, v in ev.items():
                writer.add_scalar(k, v, step)
            print(f"eval {step}: " + " ".join(f"{k} {v:.3f}" for k, v in ev.items()), flush=True)
        if step % a.ckpt_every == 0 or step == a.steps:
            save(f"{step:07d}")
            save("latest")
    writer.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
