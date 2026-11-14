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

from mjlab.utils.lab_api.math import quat_from_matrix

from supermimic.generator.data import GEN_GRID, GRID_SHAPE, GenData, from_canonical, rot6d_to_matrix
from supermimic.generator.metrics import FootMetrics
from supermimic.generator.model import Denoiser, Diffusion, GenCfg

REPO = Path(__file__).resolve().parents[2]


@dataclass
class Args:
    run: str = "gen_v0"
    library: list[str] = field(default_factory=lambda: [str(REPO / "data/library/dash_flat_terrain_v5_parc_compact.pt")])
    terrain_share: list[float] | None = None  # per library: share of the terrain windows (default: by window count)
    init: str | None = None  # warm start: model + EMA weights (and normaliser) of this checkpoint, fresh optimiser
    model: GenCfg = field(default_factory=GenCfg)
    steps: int = 300_000
    batch: int = 512
    lr: float = 2e-4
    min_lr_frac: float = 0.05
    warmup: int = 2000
    weight_decay: float = 0.01
    vel_weight: float = 1.0  # loss on frame-to-frame differences (x VEL_GAIN, normalised units)
    foot_weight: float = 0.0  # loss on foot sole positions (FK of the prediction vs the mocap's, / FOOT_SCALE)
    pen_weight: float = 0.0  # penalty on predicted soles more than PEN_SLACK below the heightmap (/ PEN_SCALE)
    terrain_frac: float = 0.5
    terrain_balance: float = 0.0  # terrain windows weighted by (windows of their category)^-this (0.5: sqrt balancing)
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
FOOT_SCALE = 0.1  # m
PEN_SLACK, PEN_SCALE = 0.01, 0.02  # m: mocap feet sit up to 1-3 cm into the terrain


class GeoLoss:
    """Foot geometry of predicted clean frames in the canonical frame: sole positions vs the mocap's (squared), and
    soles below the heightmap (linear). Weighted per sample by alpha_bar(t): little weight at high noise."""

    def __init__(self, device):
        self.feet = FootMetrics(None, device)

    def soles(self, f):
        q = quat_from_matrix(rot6d_to_matrix(f[..., 3:9]))
        return self.feet.sole(f[..., :3], q, f[..., 9:])  # (B, F, 2, 2, 3)

    def __call__(self, pred, true, hmap, wt):
        sp = self.soles(pred)
        with torch.no_grad():
            sg = self.soles(true)
        foot = (((sp - sg) / FOOT_SCALE) ** 2).mean((1, 2, 3, 4))
        x0, y0 = GEN_GRID.x[0], GEN_GRID.y[0]
        i = ((sp[..., 0] - x0) / GEN_GRID.step).round().long().clamp(0, GRID_SHAPE[0] - 1)
        j = ((sp[..., 1] - y0) / GEN_GRID.step).round().long().clamp(0, GRID_SHAPE[1] - 1)
        b = torch.arange(sp.shape[0], device=sp.device)[:, None, None, None].expand_as(i)
        # lowest of the 3 x 3 cells around: a sole within a cell of an edge is not penalised (mocap feet stand next
        # to steps, and a nearest-cell lookup there reads the step's height)
        hmin = -Fn.max_pool2d(-hmap[:, None], 3, 1, 1)[:, 0]
        h = hmin[b, i, j]
        pen = (torch.relu(h - sp[..., 2] - PEN_SLACK) / PEN_SCALE).mean((1, 2, 3))
        return (wt * foot).mean(), (wt * pen).mean()


def loss_fn(model: Denoiser, diff: Diffusion, w, a: Args, generator=None, geo: GeoLoss | None = None):
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
    if geo is None:
        return lx, lv, t, None, None
    lf, lp = geo(n.frames_inv(x0h), w.fut, w.hmap, diff.ab[t])
    return lx, lv, t, lf, lp


@torch.no_grad()
def evaluate(ema: Denoiser, diff: Diffusion, data: GenData, feet: FootMetrics, a: Args, geo: GeoLoss | None = None) -> dict[str, float]:
    g = torch.Generator(device=a.device).manual_seed(1234)
    out = {}
    # denoising loss by noise level
    ema.eval()
    w = data.sample(4096, "val", a.terrain_frac, generator=g)
    lx, lv, t, lf, lp = loss_fn(ema, diff, w, a, generator=g, geo=geo)
    out["val/loss_x"], out["val/loss_v"] = float(lx), float(lv)
    if lf is not None:
        out["val/loss_foot"], out["val/loss_pen"] = float(lf), float(lp)
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
    data = GenData(a.library, a.device, hist=a.model.hist, fut=a.model.fut, terrain_balance=a.terrain_balance, terrain_share=a.terrain_share)
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
    elif a.init:
        ck = torch.load(a.init, map_location=a.device, weights_only=False)
        model.load_state_dict(ck["model"])
        ema = copy.deepcopy(model)
        ema.load_state_dict(ck["ema"])
        print(f"warm start from {a.init} (step {ck['step']})", flush=True)
    else:
        model.norm.fit(data)
        ema = copy.deepcopy(model)
    ema.eval().requires_grad_(False)
    print(f"model {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M parameters", flush=True)
    diff = Diffusion(a.model.T, a.device)
    feet = FootMetrics(data.scan, a.device)
    train_model = torch.compile(model) if a.compile else model
    geo = GeoLoss(a.device) if a.foot_weight > 0 or a.pen_weight > 0 else None
    geo_train = torch.compile(geo.__call__) if geo is not None and a.compile else geo  # FK: 65 -> 6 ms per batch
    writer = SummaryWriter(str(log / "tb"))

    def lr_at(s):
        if s < a.warmup:
            return a.lr * (s + 1) / a.warmup
        p = min(1.0, (s - a.warmup) / max(1, a.steps - a.warmup))
        return a.lr * (a.min_lr_frac + (1 - a.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * p)))

    def save(tag):
        ck = {"step": step, "cfg": dataclasses.asdict(a.model), "args": dataclasses.asdict(a), "model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict()}
        torch.save(ck, log / f"ckpt_{tag}.pt")

    t_last, acc = time.time(), [0.0, 0.0, 0, 0.0, 0.0]
    while step < a.steps:
        model.train()
        for gr in opt.param_groups:
            gr["lr"] = lr_at(step)
        w = data.sample(a.batch, "train", a.terrain_frac)
        lx, lv, _, lf, lp = loss_fn(train_model, diff, w, a, geo=geo_train)
        loss = lx + a.vel_weight * lv
        if geo is not None:
            loss = loss + a.foot_weight * lf + a.pen_weight * lp
            acc[3] += float(lf.detach())
            acc[4] += float(lp.detach())
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
            geo_s = ""
            if geo is not None:
                writer.add_scalar("train/loss_foot", acc[3] / acc[2], step)
                writer.add_scalar("train/loss_pen", acc[4] / acc[2], step)
                geo_s = f" loss_foot {acc[3] / acc[2]:.4f} loss_pen {acc[4] / acc[2]:.4f}"
            print(f"step {step} loss_x {acc[0] / acc[2]:.4f} loss_v {acc[1] / acc[2]:.4f}{geo_s} gn {float(gn):.2f} lr {lr_at(step):.2e} {a.log_every / dt:.1f} it/s", flush=True)
            acc = [0.0, 0.0, 0, 0.0, 0.0]
        if step % a.eval_every == 0 or step == a.steps:
            ev = evaluate(ema, diff, data, feet, a, geo)
            for k, v in ev.items():
                writer.add_scalar(k, v, step)
            print(f"eval {step}: " + " ".join(f"{k} {v:.3f}" for k, v in ev.items()), flush=True)
        if step % a.ckpt_every == 0 or step == a.steps:
            save(f"{step:07d}")
            save("latest")
    writer.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
