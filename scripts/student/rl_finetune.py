"""RL fine-tuning of the depth student: PPO with a DAgger term (as PHP, arXiv 2602.15827), in physics, from a DAgger
checkpoint.

  python scripts/student/rl_finetune.py gen_ckpt teacher_ckpt student_ckpt run_dir [--envs 2048] [--rounds 100]
      [--steps 24] [--anneal 50] [--dagger-floor 0.1] [--dagger-kl 0.01] [--critic-warmup 3] [--eval-every 4]
      [dagger.py's scene flags: --courses --course-share --families --flat-share --vmax --sidestep]
      [--tall-stairs 0.5] [--gap-max 0.5] [--edge 5 --foothold 1 --aux 1] [--cmd 4] [--lam0 1] [--resume]

Rounds as dagger.py's (--envs robots for --seconds on its scenes / courses, the generator replanning every 0.2 s from
each robot's motion, the student's training conditions, no resets; a robot is done once down), but the student
drives every env, sampling a Gaussian around its action (a learned std per joint, from --std0 = 0.05; PHP's 0.01
at our 2048 envs: KL 0.01 per update is ~3e-4 rad of change, and the policy gradient's signal to noise falls with
the std). The student's camera timing (depth_refresh_steps / depth_latency_steps) is its checkpoint's. Every
--steps steps one update, in two phases with an Adam each. PPO (clip 0.2, gamma 0.99, GAE 0.95) on those steps with
the env's reward, the teacher's own tracking reward (Table A.2) against the generator's plan; its lr is fixed for
--adaptive-after updates, then adapted to a KL of 0.01 (PHP's order). Then --dagger-steps steps of the DAgger loss to the teacher's
action on windows from a replay of the last --keep rounds (as dagger.py's), its lr adapted to a KL of --dagger-kl x
lam, lam linear from --lam0 to --dagger-floor over --anneal rounds after the warm-up (PHP: the DAgger term must stay
on). Separate, as the DAgger term's policy change in the PPO step set the KL and held the shared lr at its floor
(rl_v1, rl_v2), while the PPO gradient set its direction (at std 0.1, ~20x the DAgger term's). The
critic (an MLP) sees the teacher's inputs (proprio history, goal frames, terrain scan); in the first --critic-warmup
rounds only it trains. --tall-stairs / --gap-max: courses.py's share of stairs with 15-19 cm risers and its widest
gap. Foot placement (--edge, --foothold; geometry from the robot's pose, the env has no contact sensor): penalties
for a sole below the highest ground within --reach (a toe against a riser) and for a foot in contact straddling a
> 3 cm height step (a stair edge, half over a gap). --cmd: a penalty per unit of the robot's speed along the joystick
falling short of (1 - --cmd-tol) of it (relative; the tracking reward hardly charges for stalling, as the generator
replans from wherever the robot is: rl_v2 stopped before slow stairs). --aux: an auxiliary head predicting the
teacher's terrain scan (POLICY_GRID, noise-free) from the student's features, trained with the DAgger term, so the
depth pathway keeps where the edges are. A checkpoint of this script as the student also gives the action std and
the critic; --lam0 0.1 starts at the DAgger floor.
Writes run_dir/ckpt_XXX.pt (loaded by eval_student / course_demo as DAgger's), ckpt_latest.pt and log.jsonl.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/generator"))
sys.path.insert(0, str(REPO / "scripts/student"))
import courses  # noqa: E402
from augment import FAMILIES  # noqa: E402
from dagger import Replay, close_round, open_round, replan, sample_scenes, still_up  # noqa: E402
from eval_student import evaluate, robot_frame  # noqa: E402
from rollout_generator import REPLAN, first_library, load, operator, standing_frame  # noqa: E402

from supermimic.eval.benchmark import goal_env_kwargs, load_policy_from_checkpoint  # noqa: E402
from supermimic.generator.metrics import FootMetrics  # noqa: E402
from supermimic.kinematics.rotations import heading_yaw  # noqa: E402
from supermimic.student.model import Student, StudentCfg, StudentRunner  # noqa: E402
from supermimic.terrain.height_scan import POLICY_GRID, POLICY_NOMINAL, SCAN_SCALE, HeightScan  # noqa: E402

LOG2PI = math.log(2 * math.pi)


class Critic(nn.Module):
    """Value of the teacher's inputs, normalized by running mean / variance (updated from collected steps only)."""

    def __init__(self, d: int, hidden=(512, 256, 128)):
        super().__init__()
        layers, w = [], d
        for h in hidden:
            layers += [nn.Linear(w, h), nn.LayerNorm(h), nn.GELU()]
            w = h
        self.net = nn.Sequential(*layers, nn.Linear(w, 1))
        self.register_buffer("mean", torch.zeros(d))
        self.register_buffer("var", torch.ones(d))
        self.register_buffer("count", torch.tensor(1e-4))

    @torch.no_grad()
    def update_norm(self, x: torch.Tensor) -> None:
        if not len(x):
            return
        m, v, c = x.mean(0), x.var(0, unbiased=False), len(x)
        tot = self.count + c
        d = m - self.mean
        self.var.copy_((self.var * self.count + v * c + d**2 * self.count * c / tot) / tot)
        self.mean.add_(d * c / tot)
        self.count.copy_(tot)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(((x - self.mean) / (self.var.sqrt() + 1e-5)).clamp(-10, 10))[..., 0]


def log_prob(mu, log_std, x):
    return (-0.5 * ((x - mu) / log_std.exp()) ** 2 - log_std - 0.5 * LOG2PI).sum(-1)


def foot_terms(fm: FootMetrics, now: torch.Tensor, tiles: torch.Tensor, reach: float):
    """Robot frames now (n, 25) -> per env: `edge`, the depth (m, summed over the sole points) below the highest
    ground within `reach` (a toe against a riser, a sole catching a step's edge), and `straddle`, the number of
    feet in contact (lowest sole point < 2 cm above the ground under it) whose toe and heel stand on ground more than
    3 cm apart in height (on a stair edge, half over a gap). Geometry only: the env has no contact sensor."""
    s = fm.sole(now[:, None, :3], now[:, None, 3:7], now[:, None, 7:])[:, 0]  # (n, 2 feet, 2 ends, 3)
    n = len(s)
    x, y = s[..., 0].reshape(n, -1), s[..., 1].reshape(n, -1)
    g0 = fm.scan.lookup(x, y, tiles)
    g = g0
    for dx, dy in ((reach, 0.0), (-reach, 0.0), (0.0, reach), (0.0, -reach)):
        g = torch.maximum(g, fm.scan.lookup(x + dx, y + dy, tiles))
    g0, g = g0.reshape(n, 2, 2), g.reshape(n, 2, 2)
    z = s[..., 2]
    edge = (g - z - 0.01).clamp_min(0).sum((1, 2))
    stance = (z - g0).amin(-1) < 0.02
    straddle = (stance & (g0.amax(-1) - g0.amin(-1) > 0.03)).float().sum(1)
    return edge, straddle


def cmd_shortfall(buf: torch.Tensor, joy: torch.Tensor, dt: float, tol: float) -> torch.Tensor:
    """Robot frames buf (n, >= REPLAN + 1, 25), joystick (n, 3) -> per env, how far the robot's velocity along the
    joystick's (mean over the last REPLAN steps, heading frame) falls short of (1 - tol) of it, as a share of the
    joystick's speed, in [0, 1]; 0 where the joystick asks for < 0.1 m/s."""
    cur, prev = buf[:, -1], buf[:, -REPLAN - 1]
    yaw = heading_yaw(cur[:, 3:7])
    c, s = torch.cos(yaw), torch.sin(yaw)
    d = (cur[:, :2] - prev[:, :2]) / (REPLAN * dt)
    v = torch.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], -1)
    j2 = (joy[:, :2] ** 2).sum(-1)
    along = (v * joy[:, :2]).sum(-1) / j2.clamp_min(1e-6)
    return torch.where(j2 > 0.01, (1 - tol - along).clamp(0, 1), torch.zeros_like(along))


def windows(P, D, e, t, cfg: StudentCfg):
    """Sample (env e, step t) -> its proprio window (B, n_proprio, .) and depth-frame window as indices into the
    unique frames used (inv (B, n_depth)) and those frames (U, depth_h, depth_w) in [0, 1]; windows start padded with
    the first input, as StudentRunner's."""
    pi = (t[:, None] - cfg.n_proprio + 1 + torch.arange(cfg.n_proprio, device=t.device)).clamp_min(0)
    k = t // cfg.depth_every
    ki = (k[:, None] - cfg.n_depth + 1 + torch.arange(cfg.n_depth, device=t.device)).clamp_min(0)
    fid = e[:, None] * D.shape[1] + ki
    u, inv = torch.unique(fid, return_inverse=True)
    return P[e[:, None], pi], inv, D.reshape(-1, *D.shape[2:])[u].float() / 255


@torch.no_grad()
def refresh_tokens(runner: StudentRunner, D, t: int) -> None:
    """After an actor update: re-encode the depth frames in the runner's window (as of step t, the next to act) with
    the new weights, so the acting policy is the trained one."""
    c = runner.m.cfg
    k = (t - 1) // c.depth_every
    ki = (k - c.n_depth + 1 + torch.arange(c.n_depth, device=D.device)).clamp_min(0)
    n = len(D)
    runner.D = runner.m.encode_depth(D[:, ki].reshape(-1, c.depth_h, c.depth_w).float() / 255).reshape(n, c.n_depth, -1)


def update(student, log_std, critic, opts, ch, P, D, t0, lam, train_actor, replay: Replay, st, a):
    """One update. The critic and PPO on a chunk (ch: per (env, step) tensors of --steps steps from t0), the actor by
    opts["rl"] at st["lr"], adapted to this phase's KL. Then --dagger-steps steps of the DAgger loss (+ --aux x the
    terrain loss) on windows drawn from the replay of earlier rounds, by opts["dag"] (a chunk is one 0.5 s slice of
    every robot's round: fitting the teacher on it alone chases that slice and forgets the rest), its lr st["lr_dag"]
    adapted to that phase's KL (--dagger-kl x lam) on the PPO phase's last minibatch; the terrain head (which does not
    act) by opts["aux"] at a fixed lr. In the warm-up (not train_actor) only the critic, and the terrain head on the
    frozen trunk. -> stats."""
    cfg = student.cfg
    n, S = ch["alive"].shape
    idx = ch["alive"].flatten().nonzero()[:, 0]
    if len(idx) < 2:
        return {}
    e, i = idx // S, idx % S
    t = t0 + i
    flat = {k: v.reshape(n * S, *v.shape[2:])[idx] for k, v in ch.items() if k != "alive"}
    adv = flat["adv"]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    ret = flat["adv"] + flat["val"]
    log_std_old = log_std.detach().clone()
    out = {k: [] for k in ("vloss", "surr", "dagger", "dagger_chunk", "kl", "kl_dag", "clip", "ratio0", "aux")}
    student.train()
    for ep in range(a.epochs):
        # Minibatches by env (all its steps in the chunk): windows of one env share most depth frames, ~4x fewer
        # to encode than for samples drawn across all envs.
        grp = torch.empty(n, dtype=torch.long, device=idx.device)
        grp[torch.randperm(n, device=idx.device)] = torch.arange(n, device=idx.device) * a.minibatches // n
        g_s = grp[e]
        for j in range(a.minibatches):
            mb = (g_s == j).nonzero()[:, 0]
            if not len(mb):
                continue
            v = critic(flat["cobs"][mb])
            vc = flat["val"][mb] + (v - flat["val"][mb]).clamp(-a.clip, a.clip)
            vloss = torch.max((v - ret[mb]) ** 2, (vc - ret[mb]) ** 2).mean()
            opts["c"].zero_grad(set_to_none=True)
            vloss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            opts["c"].step()
            out["vloss"].append(vloss.item())
            if not train_actor:
                continue
            Pw, inv, frames = windows(P, D, e[mb], t[mb], cfg)
            mu = student(Pw, student.encode_depth(frames)[inv])
            lp = log_prob(mu, log_std, flat["act"][mb])
            ratio = (lp - flat["logp"][mb]).exp()
            if ep == 0 and not out["ratio0"]:  # before any actor step: collection and training agree?
                out["ratio0"].append((ratio - 1).abs().mean().item())
            A = adv[mb]
            surr = -torch.min(ratio * A, ratio.clamp(1 - a.clip, 1 + a.clip) * A).mean()
            ent = (log_std + 0.5 * (1 + LOG2PI)).sum()
            opts["rl"].zero_grad(set_to_none=True)
            (surr - a.entropy * ent).backward()
            nn.utils.clip_grad_norm_([*student.parameters(), log_std], 1.0)
            opts["rl"].step()
            with torch.no_grad():
                log_std.clamp_(math.log(a.std_min), math.log(0.5))
                s0, s1 = log_std_old.exp(), log_std.exp()
                kl = (log_std - log_std_old + (s0**2 + (flat["mu"][mb] - mu.detach()) ** 2) / (2 * s1**2) - 0.5).sum(-1).mean().item()
            if st["updates"] < a.adaptive_after:
                pass
            elif kl > 2 * a.kl:  # rsl_rl's adaptive schedule, bounded
                st["lr"] = max(a.lr_min, st["lr"] / 1.5)
            elif kl < a.kl / 2:
                st["lr"] = min(a.lr_max, st["lr"] * 1.5)
            for gr in opts["rl"].param_groups:
                gr["lr"] = st["lr"]
            out["surr"].append(surr.item())
            out["dagger_chunk"].append(F.mse_loss(mu.detach(), flat["a_tch"][mb]).item())
            out["kl"].append(kl)
            out["clip"].append(((ratio - 1).abs() > a.clip).float().mean().item())
    if replay.idx is not None and (train_actor or a.aux):
        if train_actor:
            with torch.no_grad():
                mu_p = student(Pw, student.encode_depth(frames)[inv])
        for gr in opts["dag"].param_groups:
            gr["lr"] = st["lr_dag"]
        for _ in range(a.dagger_steps if train_actor else a.epochs * a.minibatches):
            Pd, Dd, Ad, *Xd = replay.batch(a.dagger_batch)
            if not train_actor:  # warm-up: fit the new terrain head on the frozen trunk
                with torch.no_grad():
                    h = student.trunk(Pd, student.encode_depth(Dd).reshape(len(Pd), cfg.n_depth, -1))
                aux = F.mse_loss(student.aux(h), Xd[0].float() * 0.01 / SCAN_SCALE)
                opts["aux"].zero_grad(set_to_none=True)
                aux.backward()
                opts["aux"].step()
                out["aux"].append(aux.item())
                continue
            tok_d = student.encode_depth(Dd).reshape(len(Pd), cfg.n_depth, -1)
            if a.aux:
                mu_d, aux_d = student(Pd, tok_d, aux=True)
                aux = F.mse_loss(aux_d, Xd[0].float() * 0.01 / SCAN_SCALE)
            else:
                mu_d, aux = student(Pd, tok_d), torch.zeros((), device=Pd.device)
            dag = F.mse_loss(mu_d, Ad)
            for o in (opts["dag"], opts["aux"]) if a.aux else (opts["dag"],):
                o.zero_grad(set_to_none=True)
            (dag + a.aux * aux).backward()
            nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            for o in (opts["dag"], opts["aux"]) if a.aux else (opts["dag"],):
                o.step()
            out["dagger"].append(dag.item())
            if a.aux:
                out["aux"].append(aux.item())
        if train_actor:
            with torch.no_grad():
                d = student(Pw, student.encode_depth(frames)[inv]) - mu_p
                kl = (d**2 / (2 * log_std.exp() ** 2)).sum(-1).mean().item()
            if kl > 2 * a.dagger_kl * lam:
                st["lr_dag"] = max(a.lr_min, st["lr_dag"] / 1.5)
            elif kl < a.dagger_kl * lam / 2:
                st["lr_dag"] = min(a.lr_max, st["lr_dag"] * 1.5)
            out["kl_dag"].append(kl)
    student.eval()
    return {k: float(np.mean(v)) for k, v in out.items() if v}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gen_ckpt")
    ap.add_argument("teacher")
    ap.add_argument("student")
    ap.add_argument("run")
    ap.add_argument("--envs", type=int, default=2048)
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--steps", type=int, default=24, help="env steps per update")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--minibatches", type=int, default=4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95, help="GAE lambda")
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--entropy", type=float, default=0.001)
    ap.add_argument("--kl", type=float, default=0.01, help="target KL of the adaptive actor lr")
    ap.add_argument("--lr", type=float, default=1e-6, help="PPO lr, fixed for --adaptive-after updates")
    ap.add_argument("--adaptive-after", type=int, default=0, help="updates before the PPO lr adapts to the KL")
    ap.add_argument("--lr-min", type=float, default=1e-8)
    ap.add_argument("--lr-max", type=float, default=1e-4)
    ap.add_argument("--critic-lr", type=float, default=3e-4)
    ap.add_argument("--std0", type=float, default=0.05, help="initial action std")
    ap.add_argument("--std-min", type=float, default=0.005)
    ap.add_argument("--dagger-lr", type=float, default=1e-6, help="initial lr of the DAgger phase")
    ap.add_argument("--dagger-kl", type=float, default=0.01, help="target KL of the DAgger phase at lam 1")
    ap.add_argument("--dagger-steps", type=int, default=8, help="DAgger steps per update")
    ap.add_argument("--dagger-floor", type=float, default=0.1)
    ap.add_argument("--lam0", type=float, default=1.0, help="DAgger KL factor after the warm-up, annealed to --dagger-floor")
    ap.add_argument("--dagger-batch", type=int, default=2048, help="replay windows per DAgger step")
    ap.add_argument("--keep", type=int, default=8, help="rounds in the DAgger replay")
    ap.add_argument("--anneal", type=float, default=50, help="rounds (after the warm-up) for the DAgger weight to reach its floor")
    ap.add_argument("--critic-warmup", type=int, default=3, help="rounds in which only the critic trains")
    ap.add_argument("--edge", type=float, default=0.0, help="penalty weight per m of sole below the highest ground within --reach")
    ap.add_argument("--reach", type=float, default=0.04)
    ap.add_argument("--foothold", type=float, default=0.0, help="penalty weight per foot in contact straddling a > 3 cm height step")
    ap.add_argument("--cmd", type=float, default=0.0, help="penalty weight per unit of relative speed short of (1 - --cmd-tol) of the joystick's")
    ap.add_argument("--cmd-tol", type=float, default=0.5)
    ap.add_argument("--aux", type=float, default=0.0, help="weight of an auxiliary loss (with the DAgger term): predict the teacher's terrain scan (POLICY_GRID)")
    ap.add_argument("--sidestep", type=float, default=0.15)
    ap.add_argument("--families", default=None)
    ap.add_argument("--courses", action="store_true")
    ap.add_argument("--course-share", type=float, default=1.0)
    ap.add_argument("--flat-share", type=float, default=0.2)
    ap.add_argument("--vmax", type=float, default=2.0)
    ap.add_argument("--tall-stairs", type=float, default=0.0, help="share of course stairs with 15-19 cm risers")
    ap.add_argument("--gap-max", type=float, default=0.4, help="widest course gap (m)")
    ap.add_argument("--eval-every", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    if a.families:
        w = {k: float(v) for k, v in (kv.split("=") for kv in a.families.split(","))}
        assert set(w) <= set(FAMILIES), w
        FAMILIES.clear()
        FAMILIES.update({k: v / sum(w.values()) for k, v in w.items()})
    courses.TALL_SHARE, courses.GAP = a.tall_stairs, (courses.GAP[0], a.gap_max)
    dev = a.device
    # Acting (eval, no grad) would take nn.TransformerEncoder's fused fast path: ~4e-4 per action off the training
    # forward (float64: 4e-7), a 14 % ratio error at std 0.01.
    torch.backends.mha.set_fastpath_enabled(False)
    run = Path(a.run)
    run.mkdir(parents=True, exist_ok=True)
    gen, diff, gck = load(a.gen_ckpt, dev)
    st = standing_frame(first_library(gck), dev)
    teacher, pcfg = load_policy_from_checkpoint(a.teacher, dev)
    ck0 = torch.load(a.student, map_location=dev, weights_only=False)
    env_kw = goal_env_kwargs(pcfg)
    for k in ("depth_refresh_steps", "depth_latency_steps"):  # the student's camera timing (dagger.py's flags)
        if ck0.get("env_kwargs", {}).get(k) is not None:
            env_kw[k] = ck0["env_kwargs"][k]
    cfg = StudentCfg(**{**ck0["cfg"], "aux_dim": POLICY_GRID.dim if a.aux else ck0["cfg"].get("aux_dim", 0)})
    student = Student(cfg).to(dev)
    missing, unexpected = student.load_state_dict(ck0["model"], strict=False)
    assert all(k.startswith("aux.") for k in missing + unexpected), (missing, unexpected)
    student.eval()
    log_std = nn.Parameter(torch.full((cfg.act_dim,), math.log(a.std0), device=dev))
    critic = None
    opts = {"rl": torch.optim.Adam([*student.parameters(), log_std], lr=a.lr),
            "dag": torch.optim.Adam([p for k, p in student.named_parameters() if not k.startswith("aux.")], lr=a.dagger_lr),
            "aux": torch.optim.Adam(student.aux.parameters(), lr=3e-4) if a.aux else None}
    if "critic" in ck0:  # an rl_finetune checkpoint: keep its action std and critic (warmed up again for a new reward)
        log_std.data.copy_(ck0["log_std"].to(dev))
        critic = Critic(ck0["critic_dim"]).to(dev)
        critic.load_state_dict(ck0["critic"])
        opts["c"] = torch.optim.Adam(critic.parameters(), lr=a.critic_lr)
        print(f"student, std and critic from {a.student}", flush=True)
    state = {"lr": a.lr, "lr_dag": a.dagger_lr, "updates": 0}
    rd0 = 0
    latest = run / "ckpt_latest.pt"
    if a.resume and latest.exists():
        ck = torch.load(latest, map_location=dev, weights_only=False)
        student.load_state_dict(ck["model"])
        log_std.data.copy_(ck["log_std"].to(dev))
        critic = Critic(ck["critic_dim"]).to(dev)
        critic.load_state_dict(ck["critic"])
        opts["c"] = torch.optim.Adam(critic.parameters(), lr=a.critic_lr)
        for k, v in ck["opts"].items():
            opts[k].load_state_dict(v)
        state, rd0 = ck["state"], ck["round"] + 1
        print(f"resumed after round {ck['round']}", flush=True)
    n, T, S, de = a.envs, int(a.seconds * 50), a.steps, cfg.depth_every
    assert T % S == 0 and T % de == 0, (T, S, de)
    per_round = T // S
    rng = np.random.default_rng([a.seed, rd0])
    lib = run / "round_library.pt"
    replay = Replay(a.keep, n, T, cfg, dev, aux_dim=cfg.aux_dim if a.aux else 0)  # restarts empty on --resume
    fm = None
    for rd in range(rd0, a.rounds):
        t_start = time.perf_counter()
        sc, sched = sample_scenes(rng, a, n, T)
        sched = torch.as_tensor(sched, device=dev)
        env, cmd, obs, scan, tiles, origins = open_round(sc, st, gen, env_kw, lib, T, a.seed * 1000 + rd, dev)
        # make_env's configure_torch_backends turns cudnn autotuning on: with a different number of unique depth
        # frames in every minibatch it re-tunes the CNN each time (4x slower updates). And TF32 (error in the action).
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.fp32_precision = torch.backends.cudnn.fp32_precision = "ieee"
        fm = FootMetrics(scan, dev) if fm is None else fm
        fm.scan = scan
        pscan = HeightScan(SimpleNamespace(terrains=[s_[2] for s_ in sc]), POLICY_GRID).to(dev) if a.aux else None
        X_r = torch.zeros(n, T, cfg.aux_dim, dtype=torch.int8, device=dev) if a.aux else None
        dt = env.step_dt
        dc = obs["policy"].shape[-1] + obs["goal"].shape[-1]
        if critic is None:
            critic = Critic(dc).to(dev)
            opts["c"] = torch.optim.Adam(critic.parameters(), lr=a.critic_lr)
        buf = robot_frame(cmd, origins)[:, None].repeat(1, max(gen.cfg.hist, REPLAN + 1), 1)
        g = torch.Generator(device=dev).manual_seed(a.seed * 1000 + rd)
        runner = StudentRunner(student, n, dev)
        P = torch.zeros(n, T, cfg.proprio_dim + cfg.cmd_dim, device=dev)
        D = torch.zeros(n, T // de, cfg.depth_h, cfg.depth_w, dtype=torch.uint8, device=dev)
        A_r = torch.zeros(n, T, cfg.act_dim, dtype=torch.float16, device=dev)
        alive_r = torch.zeros(n, T, dtype=torch.bool, device=dev)
        up = torch.ones(n, dtype=torch.bool, device=dev)
        train_actor = rd >= a.critic_warmup and replay.idx is not None
        rec = {"round": rd, "reward": 0.0, "alive_steps": 0, "edge": 0.0, "foothold": 0.0, "cmd": 0.0, "stall": 0.0}
        stats, t_upd = [], 0.0
        for t0 in range(0, T, S):
            ch = {"act": torch.zeros(n, S, cfg.act_dim, device=dev), "mu": torch.zeros(n, S, cfg.act_dim, device=dev),
                  "a_tch": torch.zeros(n, S, cfg.act_dim, device=dev), "logp": torch.zeros(n, S, device=dev),
                  "val": torch.zeros(n, S, device=dev), "cobs": torch.zeros(n, S, dc, device=dev),
                  "alive": torch.zeros(n, S, dtype=torch.bool, device=dev)}
            rew = torch.zeros(n, S, device=dev)
            done = torch.zeros(n, S, dtype=torch.bool, device=dev)
            with torch.no_grad():
                for i in range(S):
                    t = t0 + i
                    joy = operator(buf, sched[:, t, :3], sched[:, t, 3])
                    if t % REPLAN == 0:
                        replan(gen, diff, cmd, buf, joy, scan, tiles, g)
                    ch["a_tch"][:, i] = teacher(obs["policy"], obs["goal"])
                    co = torch.cat([obs["policy"], obs["goal"]], -1)
                    critic.update_norm(co[up])
                    ch["cobs"][:, i] = co
                    ch["val"][:, i] = critic(co)
                    depth = (obs["depth"] * 255).round()
                    P[:, t] = torch.cat([obs["proprio"], joy], -1)
                    if t % de == 0:
                        D[:, t // de] = depth.reshape(n, cfg.depth_h, cfg.depth_w).byte()
                    mu = runner.act(obs["proprio"], joy, depth / 255)
                    x = mu + log_std.exp() * torch.randn_like(mu)
                    ch["mu"][:, i], ch["act"][:, i], ch["logp"][:, i] = mu, x, log_prob(mu, log_std, x)
                    ch["alive"][:, i] = alive_r[:, t] = up
                    A_r[:, t] = ch["a_tch"][:, i]
                    if a.aux:  # target: the teacher's terrain input at this step, noise-free, as int8 cm
                        s49 = torch.zeros(n, 49, device=dev)
                        s49[:, :7] = buf[:, -1, :7]
                        rel = buf[:, -1, 2:3] - pscan(s49, tiles)
                        X_r[:, t] = ((rel - POLICY_NOMINAL) / 0.01).round().clamp(-127, 127).to(torch.int8)
                    obs, r, *_ = env.step(x)
                    now = robot_frame(cmd, origins)
                    buf = torch.cat([buf[:, 1:], now[:, None]], 1)
                    up_new = up & still_up(now, scan, tiles)
                    edge, straddle = foot_terms(fm, now, tiles, a.reach)
                    # from 1 s on (the window starts with the standing frame repeated)
                    short = cmd_shortfall(buf, joy, dt, a.cmd_tol) * (t >= 50)
                    rew[:, i] = (r - dt * (a.edge * edge + a.foothold * straddle + a.cmd * short)) * up
                    rec["edge"] += float((edge * up).sum())
                    rec["foothold"] += float((straddle * up).sum())
                    rec["cmd"] += float((short * up).sum())
                    rec["stall"] += float(((short >= 1 - a.cmd_tol) & up).sum())
                    done[:, i] = up & ~up_new
                    up = up_new
                last_val = critic(torch.cat([obs["policy"], obs["goal"]], -1))
            adv = torch.zeros(n, S, device=dev)
            gae = torch.zeros(n, device=dev)
            for i in reversed(range(S)):  # a robot's last step is terminal once down; the round's end bootstraps
                nv = last_val if i == S - 1 else ch["val"][:, i + 1]
                keep = (~done[:, i]).float()
                gae = rew[:, i] + a.gamma * keep * nv - ch["val"][:, i] + a.gamma * a.lam * keep * gae
                adv[:, i] = gae
            ch["adv"] = adv
            rec["reward"] += float(rew.sum())
            rec["alive_steps"] += int(ch["alive"].sum())
            k = max(0, state["updates"] - a.critic_warmup * per_round)
            lam = max(a.dagger_floor, a.lam0 * (1 - k / (a.anneal * per_round))) if train_actor else 1.0
            t1 = time.perf_counter()
            s = update(student, log_std, critic, opts, ch, P, D, t0, lam, train_actor, replay, state, a)
            if train_actor and t0 + S < T:
                refresh_tokens(runner, D, t0 + S)
            t_upd += time.perf_counter() - t1
            state["updates"] += 1
            stats.append({**s, "lam": lam})
            del ch
        close_round(env, lib)
        replay.add({"P": P.half(), "D": D, "A": A_r, "alive": alive_r, "X": X_r})
        del A_r, alive_r, X_r
        fam = np.array([s_[0] for s_ in sc])
        past = np.array([np.nan if s_[5] is None else s_[5] for s_ in sc])
        ok = up.cpu().numpy() & (np.isnan(past) | (buf[:, -1, 0].cpu().numpy() > np.nan_to_num(past)))
        rec.update({k: round(float(np.mean([s_[k] for s_ in stats if k in s_])), 5) for k in ("vloss", "surr", "dagger", "dagger_chunk", "kl", "kl_dag", "clip", "ratio0", "aux", "lam")
                    if any(k in s_ for s_ in stats)})
        rec["reward"] = round(rec["reward"] / max(rec["alive_steps"], 1), 5)
        rec["edge"] = round(rec["edge"] / max(rec["alive_steps"], 1), 5)  # m of sole below nearby ground, per step
        rec["foothold"] = round(rec["foothold"] / max(rec["alive_steps"], 1), 5)  # straddling feet per step
        rec["cmd"] = round(rec["cmd"] / max(rec["alive_steps"], 1), 5)  # relative speed shortfall per step
        rec["stall"] = round(rec["stall"] / max(rec["alive_steps"], 1), 5)  # share of steps not moving along the joystick
        rec.update({"up": round(float(up.float().mean()), 3), "pass": {f: round(float(ok[fam == f].mean()), 2) for f in sorted(set(fam))},
                    "std": round(float(log_std.exp().mean()), 4), "lr": state["lr"], "lr_dag": state["lr_dag"], "updates": state["updates"],
                    "round_s": round(time.perf_counter() - t_start), "update_s": round(t_upd)})
        ck = {"model": student.state_dict(), "cfg": student.config(), "round": rd, "start": st.cpu(), "env_kwargs": env_kw, "args": vars(a),
              "log_std": log_std.detach().cpu(), "critic": critic.state_dict(), "critic_dim": dc,
              "opts": {k: o.state_dict() for k, o in opts.items() if o is not None}, "state": state, "init": a.student}
        torch.save(ck, latest)
        if a.eval_every and (rd + 1) % a.eval_every == 0:
            t2 = time.perf_counter()
            torch.save(ck, run / f"ckpt_{rd:03d}.pt")
            res, total, n_ev = evaluate(student, run / "eval", st, env_kw, seed=rd, dev=dev)
            rec["eval"] = f"{total}/{n_ev}"
            rec["eval_scenes"] = {k: v["success"] for k, v in res.items()}
            res, total, n_ev = evaluate(student, run / "eval_fast", st, env_kw, seed=rd, dev=dev, suite="fast")
            rec["eval_fast"] = f"{total}/{n_ev}"
            rec["eval_fast_scenes"] = {k: f"{v['success']} @ {v['crossing_m_s']}" for k, v in res.items()}
            rec["eval_s"] = round(time.perf_counter() - t2)
        with open(run / "log.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
