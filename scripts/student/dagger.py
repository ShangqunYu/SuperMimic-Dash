"""DAgger for the depth student (supermimic/student/model.py) from the generator + v6 teacher, in physics.

  python scripts/student/dagger.py gen_ckpt teacher_ckpt run_dir [--envs 2048] [--rounds 40] [--seconds 12]
      [--iters 1500] [--batch 1024] [--keep 12] [--eval-every 5] [--seed 0] [--resume] [--families flat=0.35,...]
      [--courses [--flat-share 0.2] [--vmax 2.0]] [--init student_ckpt --beta0 0.25]

Each round: --envs robots on augment.py's random terrains (stairs up / down, a box, a platform up / down, flat)
with its varying joystick (command_schedule), in the student's training conditions (observation noise with the
depth's noise, dropouts and latency; robot, hardware and camera DR; the 0-25 ms actuator delay). As in
closed_loop.run_loop, every 0.2 s the generator plans from the robot's actual motion, the heightmap and the
operator's joystick, and the teacher's action tracking that plan is the label at every step. Who acts: the
teacher in a share beta of the envs (round 0: all; then 0.5^round), the student in the rest, so the data covers
the states the student itself reaches. A robot is counted down (and its later steps dropped) once its root is
within 0.3 m of the ground or tilted past 60 deg; there are no resets within a round.

Stored per step: the student's inputs (proprio group + joystick; the depth frame every 5th step, as uint8) and the
teacher's action; the last --keep rounds are trained on (MSE to the teacher's action, windows ending at any live
step). Every --eval-every rounds: eval_student.evaluate on the 15 scenes (noisy). Writes run_dir/ckpt_XXX.pt,
ckpt_latest.pt and log.jsonl. --courses: courses.py's obstacle courses at up to 2 m/s instead of augment.py's
single features at up to 1 m/s; the evals then add the fast suite (eval_student --suite fast).
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
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/generator"))
sys.path.insert(0, str(REPO / "scripts/student"))
from augment import FAMILIES, command_schedule, sample_scene  # noqa: E402
from courses import course_schedule, sample_course  # noqa: E402
from closed_loop import build_library, with_velocities  # noqa: E402
from eval_student import evaluate, robot_frame, student_env  # noqa: E402
from rollout_generator import REPLAN, VEL_HALFLIFE, command_path, first_library, load, operator, standing_frame  # noqa: E402

from supermimic.eval.benchmark import goal_env_kwargs, load_policy_from_checkpoint  # noqa: E402
from supermimic.generator.data import GEN_GRID, GRID_SHAPE, from_canonical, to_canonical  # noqa: E402
from supermimic.kinematics.rotations import heading_yaw  # noqa: E402
from supermimic.student.model import Student, StudentCfg, StudentRunner  # noqa: E402
from supermimic.terrain.height_scan import HeightScan  # noqa: E402
from supermimic.terrain.terrains import terrains_path  # noqa: E402

DOWN_HEIGHT = 0.3  # m, root above the ground
DOWN_UP_Z = 0.5  # cos 60 deg, the torso's up axis


class Replay:
    """The last `keep` rounds, each (n envs, T steps), on the GPU."""

    def __init__(self, keep: int, n: int, T: int, cfg: StudentCfg, dev: str):
        de = cfg.depth_every
        self.P = torch.zeros(keep, n, T, cfg.proprio_dim + cfg.cmd_dim, dtype=torch.float16, device=dev)
        self.D = torch.zeros(keep, n, T // de, cfg.depth_h, cfg.depth_w, dtype=torch.uint8, device=dev)
        self.A = torch.zeros(keep, n, T, cfg.act_dim, dtype=torch.float16, device=dev)
        self.alive = torch.zeros(keep, n, T, dtype=torch.bool, device=dev)
        self.cfg, self.keep, self.next, self.idx = cfg, keep, 0, None

    def add(self, d: dict) -> None:
        k = self.next % self.keep
        self.P[k], self.D[k], self.A[k], self.alive[k] = d["P"], d["D"], d["A"], d["alive"]
        self.next += 1
        self.idx = self.alive.flatten().nonzero()[:, 0]

    def batch(self, B: int):
        c = self.cfg
        i = self.idx[torch.randint(len(self.idx), (B,), device=self.idx.device)]
        _, n, T = self.alive.shape
        r, e, t = i // (n * T), (i // T) % n, i % T
        pi = (t[:, None] - c.n_proprio + 1 + torch.arange(c.n_proprio, device=t.device)).clamp_min(0)
        k = t // c.depth_every
        ki = (k[:, None] - c.n_depth + 1 + torch.arange(c.n_depth, device=t.device)).clamp_min(0)
        return self.P[r[:, None], e[:, None], pi].float(), self.D[r[:, None], e[:, None], ki].float() / 255, self.A[r, e, t].float()


@torch.no_grad()
def collect(gen, diff, teacher, env_kw, student: Student, beta: float, rng, st, a, rd: int, dev: str):
    """One round: -> data (P, D, A, alive) and stats."""
    n, T = a.envs, int(a.seconds * 50)
    if a.courses:
        sc = [sample_course(rng, a.flat_share) for _ in range(n)]
        sched = np.stack([course_schedule(rng, s[0] == "flat", T, a.sidestep, a.vmax) for s in sc])
    else:
        sc = [sample_scene(rng) for _ in range(n)]
        sched = np.stack([command_schedule(rng, s[0], T, a.sidestep) for s in sc])
    sched = torch.as_tensor(sched, device=dev)
    terr = [s[2] for s in sc]
    tiles = torch.arange(n, device=dev)
    start = st.repeat(n, 1)
    start[:, 2] += torch.tensor([s[3] for s in sc], device=dev)
    lib = Path(a.run) / "round_library.pt"
    build_library(lib, terr, tiles.tolist(), start, T + 2 * gen.cfg.fut)
    env, cmd, obs = student_env(lib, n, a.seed * 1000 + rd, True, env_kw, dev)
    L = cmd.library
    origins = env.scene.env_origins
    scan = HeightScan(SimpleNamespace(terrains=terr), GEN_GRID).to(dev)
    H = gen.cfg.hist
    buf = robot_frame(cmd, origins)[:, None].repeat(1, max(H, REPLAN + 1), 1)
    g = torch.Generator(device=dev).manual_seed(a.seed * 1000 + rd)
    use_teacher = torch.rand(n, device=dev, generator=g) < beta
    runner = StudentRunner(student, n, dev)
    c = student.cfg
    P = torch.zeros(n, T, c.proprio_dim + c.cmd_dim, dtype=torch.float16, device=dev)
    D = torch.zeros(n, T // c.depth_every, c.depth_h, c.depth_w, dtype=torch.uint8, device=dev)
    A = torch.zeros(n, T, c.act_dim, dtype=torch.float16, device=dev)
    alive = torch.ones(n, T, dtype=torch.bool, device=dev)
    up = torch.ones(n, dtype=torch.bool, device=dev)
    err = torch.zeros(n, device=dev)
    t_plan = 0.0
    for t in range(T):
        joy = operator(buf, sched[:, t, :3], sched[:, t, 3])
        if t % REPLAN == 0:  # as closed_loop.run_loop
            t0 = time.perf_counter()
            cur = buf[:, -1]
            origin, yaw0 = cur[:, :3], heading_yaw(cur[:, 3:7])
            h = buf[:, -H:]
            hist = to_canonical(h[..., :3], h[..., 3:7], h[..., 7:], origin, yaw0)
            s49 = torch.zeros(n, 49, device=dev)
            s49[:, :7] = cur[:, :7]
            hmap = (scan(s49, tiles) - origin[:, 2:3]).reshape(n, *GRID_SHAPE)
            cpath = command_path(buf, joy, VEL_HALFLIFE)
            nm = gen.norm
            with torch.autocast("cuda", dtype=torch.bfloat16):
                x = diff.sample(gen, nm.frames(hist), nm.hmap(hmap), nm.cmd(cpath), torch.ones(n, device=dev), steps=20, guidance=1.0, generator=g).float()
            pos, quat, q = from_canonical(nm.frames_inv(x), origin, yaw0)
            frames = with_velocities(cur, torch.cat([pos, quat, q], -1))
            idx = cmd.frame[:, None] + torch.arange(1, frames.shape[1] + 1, device=dev)
            L._ref[idx.reshape(-1)] = frames.reshape(-1, 49).to(L._ref.dtype)
            t_plan += time.perf_counter() - t0
        a_t = teacher(obs["policy"], obs["goal"])
        depth = (obs["depth"] * 255).round()
        a_s = runner.act(obs["proprio"], joy, depth / 255)
        P[:, t] = torch.cat([obs["proprio"], joy], -1).half()
        if t % c.depth_every == 0:
            D[:, t // c.depth_every] = depth.reshape(n, c.depth_h, c.depth_w).byte()
        A[:, t] = a_t.half()
        alive[:, t] = up
        err += ((a_s - a_t) ** 2).mean(-1) * up
        obs, *_ = env.step(torch.where(use_teacher[:, None], a_t, a_s))
        now = robot_frame(cmd, origins)
        buf = torch.cat([buf[:, 1:], now[:, None]], 1)
        above = now[:, 2] - scan.lookup(now[:, :1], now[:, 1:2], tiles)[:, 0]
        qx, qy = now[:, 4], now[:, 5]
        up &= (above > DOWN_HEIGHT) & (1 - 2 * (qx**2 + qy**2) > DOWN_UP_Z)  # world z of the torso's up axis
    env.close()
    for f in (lib, terrains_path(lib)):
        f.unlink(missing_ok=True)
    fam = np.array([s[0] for s in sc])
    past = np.array([np.nan if s[5] is None else s[5] for s in sc])
    end_x = buf[:, -1, 0].cpu().numpy()
    ok = up.cpu().numpy() & (np.isnan(past) | (end_x > np.nan_to_num(past)))
    ut = use_teacher.cpu().numpy()
    stats = {"beta": beta, "teacher_envs": int(ut.sum()), "plan_s": round(t_plan, 1),
             "student_up": round(float(up.cpu().numpy()[~ut].mean()), 3) if (~ut).any() else None,
             "teacher_up": round(float(up.cpu().numpy()[ut].mean()), 3) if ut.any() else None,
             "student_pass": {f: round(float(ok[(fam == f) & ~ut].mean()), 2) for f in sorted(set(fam)) if ((fam == f) & ~ut).any()},
             "teacher_pass": {f: round(float(ok[(fam == f) & ut].mean()), 2) for f in sorted(set(fam)) if ((fam == f) & ut).any()},
             "action_mse_student_envs": round(float((err / alive.sum(1).clamp_min(1))[~use_teacher].mean()), 4) if (~ut).any() else None,
             "live_steps": int(alive.sum())}
    return {"P": P, "D": D, "A": A, "alive": alive}, stats


def train(student: Student, opt, replay: Replay, iters: int, B: int, step0: int, a) -> tuple[float, int]:
    student.train()
    tot = 0.0
    for i in range(iters):
        k = step0 + i + 1
        lr = a.lr * min(1.0, k / a.warmup) * (0.1 + 0.45 * (1 + math.cos(math.pi * min(1.0, k / (a.rounds * a.iters)))))  # cosine to 0.1 x
        for gr in opt.param_groups:
            gr["lr"] = lr
        Pw, Dw, tgt = replay.batch(B)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tok = student.encode_depth(Dw.reshape(-1, 1, *Dw.shape[-2:])).reshape(B, Dw.shape[1], -1)
            pred = student(Pw, tok)
        loss = F.mse_loss(pred.float(), tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        tot += loss.item()
    student.eval()
    return tot / iters, step0 + iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gen_ckpt")
    ap.add_argument("teacher")
    ap.add_argument("run")
    ap.add_argument("--envs", type=int, default=2048)
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--iters", type=int, default=1500, help="gradient steps per round")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--keep", type=int, default=12, help="rounds kept for training")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--beta-decay", type=float, default=0.5, help="share of envs the teacher drives: beta0 x decay^round")
    ap.add_argument("--sidestep", type=float, default=0.15, help="flat: share of joystick segments that are pure sidesteps")
    ap.add_argument("--families", default=None, help="terrain family weights as augment.py's, e.g. flat=0.35,stairs_up=0.2 (others 0)")
    ap.add_argument("--courses", action="store_true", help="courses.py obstacle courses (1-4 features in a row) and its fast joystick")
    ap.add_argument("--flat-share", type=float, default=0.2, help="--courses: share of flat envs")
    ap.add_argument("--vmax", type=float, default=2.0, help="--courses: top joystick vx (m/s)")
    ap.add_argument("--init", default=None, help="student checkpoint to start from (its input normalization is kept)")
    ap.add_argument("--beta0", type=float, default=1.0, help="share of envs the teacher drives in round 0 (then x beta-decay per round)")
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    if a.families:
        w = {k: float(v) for k, v in (kv.split("=") for kv in a.families.split(","))}
        assert set(w) <= set(FAMILIES), w
        FAMILIES.clear()
        FAMILIES.update({k: v / sum(w.values()) for k, v in w.items()})
    dev = a.device
    run = Path(a.run)
    run.mkdir(parents=True, exist_ok=True)
    gen, diff, gck = load(a.gen_ckpt, dev)
    st = standing_frame(first_library(gck), dev)
    teacher, pcfg = load_policy_from_checkpoint(a.teacher, dev)
    env_kw = goal_env_kwargs(pcfg)
    cfg = StudentCfg()
    student = Student(cfg).to(dev)
    opt = torch.optim.AdamW(student.parameters(), lr=a.lr, weight_decay=0.01)
    T = int(a.seconds * 50)
    replay = Replay(a.keep, a.envs, T, cfg, dev)
    rd0, step = 0, 0
    if a.init:
        student.load_state_dict(torch.load(a.init, map_location=dev, weights_only=False)["model"])
        print(f"initialized from {a.init}", flush=True)
    if a.resume and (run / "ckpt_latest.pt").exists():
        ck = torch.load(run / "ckpt_latest.pt", map_location=dev, weights_only=False)
        student.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        rd0, step = ck["round"] + 1, ck["step"]
        print(f"resumed after round {ck['round']} (the replay restarts empty)", flush=True)
    rng = np.random.default_rng([a.seed, rd0])
    print(f"student: {sum(p.numel() for p in student.parameters()) / 1e6:.2f} M parameters", flush=True)
    for rd in range(rd0, a.rounds):
        t0 = time.perf_counter()
        beta = a.beta0 * a.beta_decay**rd
        data, stats = collect(gen, diff, teacher, env_kw, student, beta, rng, st, a, rd, dev)
        if rd == 0 and not a.resume and not a.init:
            student.set_normalization(data["P"][data["alive"]])
        replay.add(data)
        del data
        t1 = time.perf_counter()
        loss, step = train(student, opt, replay, a.iters, a.batch, step, a)
        t2 = time.perf_counter()
        ck = {"model": student.state_dict(), "cfg": student.config(), "opt": opt.state_dict(), "round": rd, "step": step, "start": st.cpu(),
              "env_kwargs": env_kw, "args": vars(a)}
        torch.save(ck, run / "ckpt_latest.pt")
        rec = {"round": rd, "loss": round(loss, 5), "collect_s": round(t1 - t0), "train_s": round(t2 - t1), **stats}
        if a.eval_every and (rd + 1) % a.eval_every == 0:
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
