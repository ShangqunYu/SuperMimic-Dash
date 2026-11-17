"""PPO fine-tuning of the tracking teacher in physics, in the closed loop where it demonstrates for the student.

    python scripts/student/teacher_rl.py GEN_CKPT TEACHER_CKPT RUN_DIR [--edge-vol 33] [--foothold 0] [--anchor 0]
      [--courses --course-share 0.5 --tall-stairs 0.5 --gap-max 0.6 --vmax 2.5] [--resume]

Rounds as rl_finetune.py's: per round --envs robots on dagger.py's random terrains and courses (12 s), the generator
replanning every 0.2 s from each robot's actual motion, the teacher tracking the plan on the student's robot (camera
mass, robot + hardware DR, observation noise, actuator delay; nothing rendered); a robot that is down stops counting.
Reward: the env's (the teacher's own tracking reward, Table A.2, against the plan) minus, per second,

  --edge-vol x sum over the foot points of (depth inside a cylinder of --edge-radius around a step lip)
             x (the point's speed + 1e-6)

("Hiking in the Wild", arXiv 2601.07718, Eq. 1; InstinctLab's volume_points_penetration on GreedyconcatEdgeCylinder
virtual obstacles, weight 4 on 100 points per foot: 33 on our 12), simplified for our heightfields: a lip is a cell boundary with a
height jump above --edge-step (metrics.edge_penetration; their mesh-edge detection is not needed), the foot volume
is --edge-points spheres along Dash's foot capsule (its collision geometry; the cylinder radius grows by the
capsule's 1.25 cm), point speeds by finite difference over one step. A planted foot near a lip costs nothing; a
foot moving through the zone (a toe scuffing the nose, a foot slipping off it) does. --foothold: a penalty per foot
in contact straddling a > 3 cm step (rl_finetune's; the role of their feet_at_plane). --anchor: KL to the initial
teacher on the visited states.
PPO as rl_finetune.py's: a fresh critic (an MLP on the teacher's inputs: proprio history, goal frames, terrain scan),
trained alone for --critic-warmup rounds; actor lr adapted to a KL of --kl per minibatch.
Writes run_dir/ckpt_XXX.pt in the teacher's checkpoint format (config + policy: closed_loop.py, course_demo.py and
dagger.py take it as the teacher), ckpt_latest.pt (+ the PPO state, for --resume) and log.jsonl.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/generator"))
sys.path.insert(0, str(REPO / "scripts/student"))
import courses  # noqa: E402
from augment import FAMILIES  # noqa: E402
from dagger import close_round, open_round, replan, sample_scenes, still_up  # noqa: E402
from eval_student import robot_frame  # noqa: E402
from rl_finetune import LOG2PI, Critic, foot_terms, log_prob  # noqa: E402
from rollout_generator import REPLAN, first_library, load, operator, standing_frame  # noqa: E402

from supermimic.eval.benchmark import goal_env_kwargs, load_policy_from_checkpoint  # noqa: E402
from supermimic.generator.metrics import FootMetrics, edge_penetration  # noqa: E402


def update(policy, log_std, critic, opts, ch, dop, train_actor, st, a):
    """The critic and (after the warm-up) PPO on a chunk (ch: per (env, step) tensors of --steps steps), minibatches
    over (env, step); the actor lr st["lr"] adapted to the KL. -> stats."""
    n, S = ch["alive"].shape
    idx = ch["alive"].flatten().nonzero()[:, 0]
    if len(idx) < 2:
        return {}
    flat = {k: v.reshape(n * S, *v.shape[2:])[idx] for k, v in ch.items() if k != "alive"}
    adv = flat["adv"]
    ret = adv + flat["val"]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    log_std_old = log_std.detach().clone()
    out = {k: [] for k in ("vloss", "surr", "anchor", "kl", "clip", "ratio0")}
    for ep in range(a.epochs):
        for mb in torch.randperm(len(idx), device=idx.device).chunk(a.minibatches):
            co = flat["cobs"][mb]
            v = critic(co)
            vc = flat["val"][mb] + (v - flat["val"][mb]).clamp(-a.clip, a.clip)
            vloss = torch.max((v - ret[mb]) ** 2, (vc - ret[mb]) ** 2).mean()
            opts["c"].zero_grad(set_to_none=True)
            vloss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            opts["c"].step()
            out["vloss"].append(vloss.item())
            if not train_actor:
                continue
            mu = policy(co[:, :dop], co[:, dop:])
            ratio = (log_prob(mu, log_std, flat["act"][mb]) - flat["logp"][mb]).exp()
            if ep == 0 and not out["ratio0"]:  # before any actor step: collection and training agree?
                out["ratio0"].append((ratio - 1).abs().mean().item())
            A = adv[mb]
            surr = -torch.min(ratio * A, ratio.clamp(1 - a.clip, 1 + a.clip) * A).mean()
            ent = (log_std + 0.5 * (1 + LOG2PI)).sum()
            loss = surr - a.entropy * ent
            if a.anchor:
                anc = ((mu - flat["mu_ref"][mb]) ** 2 / (2 * log_std.detach().exp() ** 2)).sum(-1).mean()
                loss = loss + a.anchor * anc
                out["anchor"].append(anc.item())
            opts["pi"].zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_([*policy.parameters(), log_std], 1.0)
            opts["pi"].step()
            with torch.no_grad():
                log_std.clamp_(math.log(a.std_min), math.log(0.5))
                s0, s1 = log_std_old.exp(), log_std.exp()
                kl = (log_std - log_std_old + (s0**2 + (flat["mu"][mb] - mu.detach()) ** 2) / (2 * s1**2) - 0.5).sum(-1).mean().item()
            if kl > 2 * a.kl:  # rsl_rl's adaptive schedule, bounded
                st["lr"] = max(a.lr_min, st["lr"] / 1.5)
            elif kl < a.kl / 2:
                st["lr"] = min(a.lr_max, st["lr"] * 1.5)
            for gr in opts["pi"].param_groups:
                gr["lr"] = st["lr"]
            out["surr"].append(surr.item())
            out["kl"].append(kl)
            out["clip"].append(((ratio - 1).abs() > a.clip).float().mean().item())
    return {k: float(np.mean(v)) for k, v in out.items() if v}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gen_ckpt")
    ap.add_argument("teacher", help="a teacher checkpoint (or one of this script's: also its action std and critic)")
    ap.add_argument("run")
    ap.add_argument("--envs", type=int, default=2048)
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--steps", type=int, default=24, help="env steps per update")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--minibatches", type=int, default=4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95, help="GAE lambda")
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--entropy", type=float, default=0.0)
    ap.add_argument("--kl", type=float, default=0.01, help="target KL of the adaptive actor lr")
    ap.add_argument("--lr", type=float, default=1e-6, help="initial actor lr (1e-5: Adam's first steps overshoot the KL ~50x)")
    ap.add_argument("--lr-min", type=float, default=1e-8)
    ap.add_argument("--lr-max", type=float, default=1e-4)
    ap.add_argument("--critic-lr", type=float, default=3e-4)
    ap.add_argument("--std0", type=float, default=0.05, help="initial action std")
    ap.add_argument("--std-min", type=float, default=0.01)
    ap.add_argument("--anchor", type=float, default=0.0, help="weight of the KL to the initial teacher")
    ap.add_argument("--critic-warmup", type=int, default=3, help="rounds in which only the critic trains")
    ap.add_argument("--edge-vol", type=float, default=0.0, help="weight of the edge penetration x speed penalty")
    ap.add_argument("--edge-radius", type=float, default=0.05, help="radius of the keep-out cylinder around a lip (m)")
    ap.add_argument("--edge-step", type=float, default=0.03, help="height jump that makes a cell boundary a lip (m)")
    ap.add_argument("--edge-points", type=int, default=12, help="points along each foot capsule")
    ap.add_argument("--foothold", type=float, default=0.0, help="penalty weight per foot in contact straddling a > 3 cm height step")
    ap.add_argument("--sidestep", type=float, default=0.25)
    ap.add_argument("--families", default="flat=0.35,stairs_up=0.2,stairs_down=0.12,box=0.15,platform_up=0.1,platform_down=0.08")
    ap.add_argument("--courses", action="store_true")
    ap.add_argument("--course-share", type=float, default=0.5)
    ap.add_argument("--flat-share", type=float, default=0.2)
    ap.add_argument("--vmax", type=float, default=2.5)
    ap.add_argument("--tall-stairs", type=float, default=0.5, help="share of course stairs with 15-19 cm risers")
    ap.add_argument("--gap-max", type=float, default=0.6, help="widest course gap (m)")
    ap.add_argument("--save-every", type=int, default=5)
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
    run = Path(a.run)
    run.mkdir(parents=True, exist_ok=True)
    gen, diff, gck = load(a.gen_ckpt, dev)
    st = standing_frame(first_library(gck), dev)
    ck0 = torch.load(a.teacher, map_location="cpu", weights_only=False)
    policy, pcfg = load_policy_from_checkpoint(a.teacher, dev)
    ref = load_policy_from_checkpoint(a.teacher, dev)[0].requires_grad_(False) if a.anchor else None
    env_kw = goal_env_kwargs(pcfg)
    log_std = nn.Parameter(torch.full((pcfg["policy"]["num_joints"],), math.log(a.std0), device=dev))
    opts = {"pi": torch.optim.Adam([*policy.parameters(), log_std], lr=a.lr)}
    critic = None
    if "teacher_rl" in ck0:  # this script's checkpoint: keep its action std and critic
        log_std.data.copy_(ck0["teacher_rl"]["log_std"].to(dev))
        critic = Critic(ck0["teacher_rl"]["critic_dim"]).to(dev)
        critic.load_state_dict(ck0["teacher_rl"]["critic"])
        opts["c"] = torch.optim.Adam(critic.parameters(), lr=a.critic_lr)
        print(f"teacher, std and critic from {a.teacher}", flush=True)
    state = {"lr": a.lr, "updates": 0}
    rd0 = 0
    latest = run / "ckpt_latest.pt"
    if a.resume and latest.exists():
        ck = torch.load(latest, map_location=dev, weights_only=False)
        policy.load_state_dict(ck["policy"])
        r_ = ck["teacher_rl"]
        log_std.data.copy_(r_["log_std"].to(dev))
        critic = Critic(r_["critic_dim"]).to(dev)
        critic.load_state_dict(r_["critic"])
        opts["c"] = torch.optim.Adam(critic.parameters(), lr=a.critic_lr)
        for k, v in r_["opts"].items():
            opts[k].load_state_dict(v)
        state, rd0 = r_["state"], r_["round"] + 1
        print(f"resumed after round {r_['round']}", flush=True)
    n, T, S = a.envs, int(a.seconds * 50), a.steps
    assert T % S == 0, (T, S)
    rng = np.random.default_rng([a.seed, rd0])
    lib = run / "round_library.pt"
    fm = None
    for rd in range(rd0, a.rounds):
        t_start = time.perf_counter()
        sc, sched = sample_scenes(rng, a, n, T)
        sched = torch.as_tensor(sched, device=dev)
        env, cmd, obs, scan, tiles, origins = open_round(sc, st, gen, env_kw, lib, T, a.seed * 1000 + rd, dev, depth=False)
        # Exact float32 (no TF32): collection and training forwards agree (ratio0).
        torch.backends.cuda.matmul.fp32_precision = torch.backends.cudnn.fp32_precision = "ieee"
        fm = FootMetrics(scan, dev) if fm is None else fm
        fm.scan = scan
        dt = env.step_dt
        dop = obs["policy"].shape[-1]
        dc = dop + obs["goal"].shape[-1]
        if critic is None:
            critic = Critic(dc).to(dev)
            opts["c"] = torch.optim.Adam(critic.parameters(), lr=a.critic_lr)
        buf = robot_frame(cmd, origins)[:, None].repeat(1, max(gen.cfg.hist, REPLAN + 1), 1)
        g = torch.Generator(device=dev).manual_seed(a.seed * 1000 + rd)
        up = torch.ones(n, dtype=torch.bool, device=dev)
        now = buf[:, -1]
        pts_prev = fm.axis(now[:, :3], now[:, 3:7], now[:, 7:], a.edge_points).reshape(n, -1, 3)
        train_actor = rd >= a.critic_warmup
        rec = {"round": rd, "reward": 0.0, "track": 0.0, "alive_steps": 0, "edge_vol": 0.0, "edge_hit": 0.0, "edge_depth": 0.0, "edge": 0.0, "foothold": 0.0}
        stats, t_upd = [], 0.0
        for t0 in range(0, T, S):
            ch = {"act": torch.zeros(n, S, log_std.shape[0], device=dev), "mu": torch.zeros(n, S, log_std.shape[0], device=dev),
                  "logp": torch.zeros(n, S, device=dev), "val": torch.zeros(n, S, device=dev), "cobs": torch.zeros(n, S, dc, device=dev),
                  "alive": torch.zeros(n, S, dtype=torch.bool, device=dev)}
            if ref is not None:
                ch["mu_ref"] = torch.zeros(n, S, log_std.shape[0], device=dev)
            rew = torch.zeros(n, S, device=dev)
            done = torch.zeros(n, S, dtype=torch.bool, device=dev)
            with torch.no_grad():
                for i in range(S):
                    t = t0 + i
                    joy = operator(buf, sched[:, t, :3], sched[:, t, 3])
                    if t % REPLAN == 0:
                        replan(gen, diff, cmd, buf, joy, scan, tiles, g)
                    co = torch.cat([obs["policy"], obs["goal"]], -1)
                    critic.update_norm(co[up])
                    ch["cobs"][:, i] = co
                    ch["val"][:, i] = critic(co)
                    mu = policy(obs["policy"], obs["goal"])
                    if ref is not None:
                        ch["mu_ref"][:, i] = ref(obs["policy"], obs["goal"])
                    x = mu + log_std.exp() * torch.randn_like(mu)
                    ch["mu"][:, i], ch["act"][:, i], ch["logp"][:, i] = mu, x, log_prob(mu, log_std, x)
                    ch["alive"][:, i] = up
                    obs, r, *_ = env.step(x)
                    now = robot_frame(cmd, origins)
                    buf = torch.cat([buf[:, 1:], now[:, None]], 1)
                    up_new = up & still_up(now, scan, tiles)
                    pts = fm.axis(now[:, :3], now[:, 3:7], now[:, 7:], a.edge_points).reshape(n, -1, 3)
                    pen = edge_penetration(scan, pts, tiles, a.edge_radius + fm.radius, a.edge_step)
                    vol = (pen * ((pts - pts_prev).norm(dim=-1) / dt + 1e-6)).sum(-1)
                    pts_prev = pts
                    edge, straddle = foot_terms(fm, now, tiles, 0.04)  # rl_finetune's, logged
                    rew[:, i] = (r - dt * (a.edge_vol * vol + a.foothold * straddle)) * up
                    rec["track"] += float((r * up).sum())
                    rec["edge_vol"] += float((vol * up).sum())
                    rec["edge_hit"] += float(((pen.amax(-1) > 0) & up).sum())
                    rec["edge_depth"] += float((pen.amax(-1) * up).sum())
                    rec["edge"] += float((edge * up).sum())
                    rec["foothold"] += float((straddle * up).sum())
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
            t1 = time.perf_counter()
            stats.append(update(policy, log_std, critic, opts, ch, dop, train_actor, state, a))
            t_upd += time.perf_counter() - t1
            state["updates"] += 1
            del ch
        close_round(env, lib)
        fam = np.array([s_[0] for s_ in sc])
        past = np.array([np.nan if s_[5] is None else s_[5] for s_ in sc])
        ok = up.cpu().numpy() & (np.isnan(past) | (buf[:, -1, 0].cpu().numpy() > np.nan_to_num(past)))
        rec.update({k: round(float(np.mean([s_[k] for s_ in stats if k in s_])), 5) for k in ("vloss", "surr", "anchor", "kl", "clip", "ratio0")
                    if any(k in s_ for s_ in stats)})
        for k in ("reward", "track", "edge_vol", "edge_hit", "edge_depth", "edge", "foothold"):  # per alive step
            rec[k] = round(rec[k] / max(rec["alive_steps"], 1), 5)
        rec.update({"up": round(float(up.float().mean()), 3), "pass": {f: round(float(ok[fam == f].mean()), 2) for f in sorted(set(fam))},
                    "std": round(float(log_std.exp().mean()), 4), "lr": state["lr"], "updates": state["updates"],
                    "round_s": round(time.perf_counter() - t_start), "update_s": round(t_upd)})
        ck = {"config": ck0["config"], "policy": policy.state_dict(), "iteration": ck0.get("iteration"), "init": a.teacher, "round": rd, "args": vars(a)}
        if a.save_every and (rd + 1) % a.save_every == 0:
            torch.save(ck, run / f"ckpt_{rd:03d}.pt")
        ck["teacher_rl"] = {"log_std": log_std.detach().cpu(), "critic": critic.state_dict(), "critic_dim": dc, "round": rd, "state": state,
                            "opts": {k: o.state_dict() for k, o in opts.items()}}
        torch.save(ck, latest)
        with open(run / "log.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
