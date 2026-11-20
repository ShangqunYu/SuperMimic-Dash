"""Generator + v6 teacher on one assembled course, for a showcase video: stairs up onto a raised deck, a gap in the
deck, a jump up onto a higher platform, stairs back down to the ground, then flat run-out. The course is 2 m wide
(sides drop to the ground); the joystick holds vx at each speed (heading hold, closed_loop.run_loop).

  python scripts/generator/course_demo.py gen_ckpt teacher out [--speeds 1.5 2.0] [--samples 8] [--dr]
  python scripts/generator/course_demo.py --render out [--pick VARIANT SPEED SAMPLE] [--ghost] [--size 1920 1080]
  python scripts/generator/course_demo.py --rescore out  (results.json and traj.pt's results from the current scoring)

Variants: stair riser up x gap width x platform height (all combinations of --rises, --gaps, --platforms). Per run:
the furthest feature passed while up (root > 0.3 m above the lowest ground within 0.3 m along x, tilt < 60 deg),
success (reached the finish line without going down; the run-out after it does not count) and the speed from x 0.5 m
to the finish line. Failures: the feature the robot went down before, or "timeout" (still up, short of the finish). Results in
out/results.json, trajectories in out/traj.pt. --render: the fastest successful run per speed (or --pick) as a
tracking-camera video, robot speed and the joystick overlaid; --ghost adds the generator's plan.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from closed_loop import REPO, build_library, run_loop  # noqa: E402
from rollout_generator import first_library, load, standing_frame  # noqa: E402

from supermimic.generator.data import GEN_GRID  # noqa: E402
from supermimic.terrain.height_scan import HeightScan  # noqa: E402
from supermimic.terrain.terrains import Terrain  # noqa: E402

DX, X0, X1, Y0, Y1 = 0.04, -1.0, 22.0, -2.0, 2.0
HALF_W = 1.0  # course half width (the generator's heightmap reaches 0.8 m sideways, the teacher's scan 0.3 m)
TREAD, LANDING, START = 0.30, 1.5, 1.5


def course(rise: float, gap: float, platform: float, n_up: int = 5, rise_down: float = 0.15):
    """-> heightfield on the X0..X1 x Y0..Y1 grid, {feature: x where it is behind the robot}, finish x."""
    x = X0 + DX * np.arange(round((X1 - X0) / DX) + 1)
    y = Y0 + DX * np.arange(round((Y1 - Y0) / DX) + 1)
    X, Y = np.meshgrid(x, y, indexing="ij")
    deck = n_up * rise
    top = START + n_up * TREAD  # end of the stairs up
    g0 = top + LANDING
    g1 = g0 + gap
    p0 = g1 + LANDING  # platform edge
    hi = deck + platform
    d0 = p0 + LANDING  # first step down
    n_dn = math.ceil(hi / rise_down - 1e-6)
    hf = np.clip(np.floor((X - START) / TREAD) + 1, 0, n_up) * rise
    hf = np.where((X >= g0) & (X < g1), 0.0, hf)
    hf = np.where(X >= p0, hi, hf)
    hf = np.where(X >= d0, hi - np.clip(np.floor((X - d0) / TREAD) + 1, 0, n_dn) * hi / n_dn, hf)
    hf = np.where(np.abs(Y) <= HALF_W, hf, 0.0)
    end = d0 + (n_dn - 1) * TREAD  # ground reached
    marks = {"stairs up": top + 0.3, "gap": g1 + 0.3, "platform": p0 + 0.3, "stairs down": end + 0.3}
    return Terrain(hf.astype(np.float32), np.array([X0, Y0]), DX), marks, end + 0.8


def score(Q, tiles, terr, marks, finish, dev):
    """Q (n, T, 25) -> per run dict."""
    scan = HeightScan(SimpleNamespace(terrains=terr), GEN_GRID).to(dev)
    x, y = Q[..., 0], Q[..., 1]
    ground = torch.stack([scan.lookup(x + dx, y, tiles) for dx in (-0.3, 0.0, 0.3)]).amin(0)
    qx, qy = Q[..., 4], Q[..., 5]
    down = ((Q[..., 2] - ground) < 0.3) | ((1 - 2 * (qx**2 + qy**2)) < 0.5)
    T = Q.shape[1]
    t_down = torch.where(down.any(1), down.float().argmax(1), torch.full_like(tiles, T))
    out = []
    for e in range(len(tiles)):
        k, td = int(tiles[e]), int(t_down[e])
        xs = x[e, :td].cpu().numpy()
        reach = float(xs.max()) if len(xs) else 0.0
        passed = [f for f, m in marks[k].items() if reach > m]
        i0 = np.argmax(xs > 0.5) if (xs > 0.5).any() else None
        i1 = np.argmax(xs > finish[k]) if (xs > finish[k]).any() else None
        ok = i1 is not None
        out.append({"success": ok, "passed": passed, "down_at_s": None if td == T else round(td / 50, 2), "reach_x": round(reach, 2),
                    "course_m_s": round((finish[k] - 0.5) / ((i1 - i0) * 0.02), 2) if i0 is not None and i1 is not None else None,
                    "finish_s": round(i1 / 50, 2) if i1 is not None else None})
    return out


def run(a):
    dev = "cuda:0"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    variants = list(itertools.product(a.rises, a.gaps, a.platforms))
    built = [course(r, g, p) for r, g, p in variants]
    terr = [b[0] for b in built]
    names = [f"riser {r * 100:.0f} cm, gap {g * 100:.0f} cm, platform {p * 100:.0f} cm" for r, g, p in variants]
    V, NS, S = len(variants), len(a.speeds), a.samples
    combo = [(v, s, k) for v in range(V) for s in range(NS) for k in range(S)]
    n = len(combo)
    tiles = torch.tensor([c[0] for c in combo], device=dev)
    cmd_v = torch.tensor([[a.speeds[c[1]], 0.0, 0.0] for c in combo], device=dev)
    model, diff, ck = load(a.gen_ckpt, dev)
    start = standing_frame(first_library(ck), dev).repeat(n, 1)
    if a.student:
        Q = run_student(a, terr, tiles, start, cmd_v, out, dev)
        R = Q.clone()
        R[..., 2] -= 50.0  # no plan: a ghost would sit far below the floor
    else:
        guide = {"step": a.guide_step, "iters": a.guide_iters, "last": a.guide_last, "inflate": a.guide_inflate, "joints_only": not a.guide_root} if a.foot_guide else None
        Q, R, _ = run_loop(model, diff, a.teacher, terr, tiles, start, cmd_v, a.seconds, out, dev, seed=a.seed, dr=a.dr, delay=a.dr, foot_guide=guide)
    res = score(Q, tiles, terr, [b[1] for b in built], [b[2] for b in built], dev)
    meta = {"gen_ckpt": a.gen_ckpt, "teacher": a.teacher, "student": a.student, "dr": a.dr, "seed": a.seed,
            "foot_guide": a.foot_guide and {"step": a.guide_step, "iters": a.guide_iters, "last": a.guide_last, "inflate": a.guide_inflate, "root": a.guide_root}}
    report(out, meta, names, a.speeds, combo, res)
    torch.save({"robot": Q.cpu(), "plan": R.cpu(), "variants": variants, "names": names, "speeds": a.speeds, "samples": S, "combo": combo, "results": res,
                "marks": [b[1] for b in built], "finish": [b[2] for b in built], "dr": a.dr, "student": a.student}, out / "traj.pt")


def report(out: Path, meta: dict, names, speeds, combo, res):
    table = {}
    for (v, s, k), r in zip(combo, res):
        table.setdefault(names[v], {}).setdefault(f"{speeds[s]}", []).append(r)
    summary = {}
    for nm, by in table.items():
        for sp, rs in by.items():
            v = [r["course_m_s"] for r in rs if r["success"]]
            fails = {}
            for r in rs:
                if not r["success"]:
                    nxt = "timeout" if r["down_at_s"] is None else next((f for f in ("stairs up", "gap", "platform", "stairs down") if f not in r["passed"]), "end")
                    fails[nxt] = fails.get(nxt, 0) + 1
            summary.setdefault(nm, {})[sp] = {"success": f"{len(v)}/{len(rs)}", "course_m_s": round(float(np.mean(v)), 2) if v else None, "failed_at": fails}
            print(f"{nm} | joystick {sp} m/s: {len(v)}/{len(rs)} @ {summary[nm][sp]['course_m_s']} m/s, failed at {fails}", flush=True)
    (out / "results.json").write_text(json.dumps({**meta, "summary": summary, "runs": table}, indent=1))


def rescore(out: Path, dev: str = "cuda:0"):
    tr = torch.load(out / "traj.pt", weights_only=False)
    built = [course(*v) for v in tr["variants"]]
    tiles = torch.tensor([c[0] for c in tr["combo"]], device=dev)
    res = score(tr["robot"].to(dev), tiles, [b[0] for b in built], [b[1] for b in built], [b[2] for b in built], dev)
    meta = {k: v for k, v in json.loads((out / "results.json").read_text()).items() if k not in ("summary", "runs")}
    report(out, meta, tr["names"], tr["speeds"], tr["combo"], res)
    tr["results"] = res
    torch.save(tr, out / "traj.pt")


@torch.no_grad()
def run_student(a, terr, tiles, start, cmd_v, out: Path, dev: str):
    """The depth student alone (joystick + proprio + depth; no generator, no teacher). -> robot frames (n, T, 25)."""
    sys.path.insert(0, str(REPO / "scripts/student"))
    from eval_student import robot_frame, student_env
    from rollout_generator import operator

    from supermimic.student.model import StudentRunner, load_student
    from supermimic.terrain.terrains import terrains_path

    ck = torch.load(a.student, map_location=dev, weights_only=False)
    stu = load_student(a.student, dev)
    T = int(a.seconds * 50)
    lib = out / "student_library.pt"
    build_library(lib, terr, tiles.tolist(), start, T + 200)
    env, cmd, obs = student_env(lib, len(tiles), a.seed, a.dr, ck["env_kwargs"], dev)
    runner = StudentRunner(stu, len(tiles), dev)
    origins = env.scene.env_origins
    buf = robot_frame(cmd, origins)[:, None]
    Q = []
    for _ in range(T):
        obs, *_ = env.step(runner.act(obs["proprio"], operator(buf, cmd_v), obs["depth"]))
        buf = robot_frame(cmd, origins)[:, None]
        Q.append(buf[:, 0])
    env.close()
    for f in (lib, terrains_path(lib)):
        f.unlink(missing_ok=True)
    return torch.stack(Q, 1)


def render_runs(a):
    os.environ.setdefault("MUJOCO_GL", "egl")
    sys.path.insert(0, str(REPO / "scripts/parc"))
    import mujoco
    from PIL import Image, ImageDraw, ImageFont
    from render_hq import FPS, dash_model, encode, resample
    from scipy.ndimage import gaussian_filter1d

    out = Path(a.out)
    tr = torch.load(out / "traj.pt", weights_only=False)
    combo, res = tr["combo"], tr["results"]
    if a.pick:
        picks = [combo.index(tuple(a.pick))]
    else:  # per speed, the fastest successful run (any variant)
        picks = []
        for s in range(len(tr["speeds"])):
            ok = [i for i, c in enumerate(combo) if c[1] == s and res[i]["success"]]
            if ok:
                picks.append(max(ok, key=lambda i: res[i]["course_m_s"]))
    w, h = a.size
    font, font2 = ImageFont.truetype("DejaVuSans-Bold.ttf", h // 30), ImageFont.truetype("DejaVuSans.ttf", h // 40)
    for i in picks:
        v, s, k = combo[i]
        rise, gap, plat = tr["variants"][v]
        ter = course(rise, gap, plat)[0]
        boxes = np.array([[0, 0, ter.floor - 0.05, 0.01, 0.01, 0.05]] + [list(b) for b in ter.boxes()])
        m = dash_model(boxes, w, h)
        for g in range(m.ngeom):  # terrain in a plain colour: with the floor's checker texture, tops vanish into it
            if m.geom(g).name.startswith("terrain_"):
                m.geom_matid[g] = -1
                m.geom_rgba[g] = (0.60, 0.66, 0.78, 1.0)
        q50 = tr["robot"][i].numpy().astype(np.float64)
        stop = res[i]["finish_s"] or res[i]["down_at_s"] or len(q50) / 50  # a failure ends --tail s after the fall
        end = min(len(q50), int((stop + a.tail) * 50))
        if res[i]["success"]:
            outcome = f"result: whole course at {res[i]['course_m_s']:.2f} m/s"
        else:
            where = next((mk for mk in ("stairs up", "gap", "platform", "stairs down") if mk not in res[i]["passed"]), "the end")
            outcome = (f"result: fell at the {where} ({res[i]['down_at_s']:.1f} s)" if res[i]["down_at_s"] else
                       f"result: still up at the time limit, {res[i]['reach_x']:.1f} m along the course")
        Q, Rp = resample(q50[:end], 50, 1.0), resample(tr["plan"][i].numpy().astype(np.float64)[:end], 50, 1.0)
        p = Q[:, :3]
        look = np.stack([gaussian_filter1d(p[:, 0], 0.4 * FPS, mode="nearest"), gaussian_filter1d(p[:, 1], 0.6 * FPS, mode="nearest"),
                         gaussian_filter1d(p[:, 2], 0.8 * FPS, mode="nearest") - 0.1], 1)
        vel = np.gradient(gaussian_filter1d(p[:, 0], 0.25 * FPS, mode="nearest")) * FPS
        d, dg = mujoco.MjData(m), mujoco.MjData(m)
        r = mujoco.Renderer(m, h, w)
        cam = mujoco.MjvCamera()
        cam.distance, cam.elevation, cam.azimuth = a.dist, -12.0, 110.0  # travel to screen right, 20 deg toward the front
        opt = mujoco.MjvOption()
        opt.geomgroup[:] = 0
        opt.geomgroup[[0, 1, 2]] = 1
        opt.sitegroup[:] = 0
        pert = mujoco.MjvPerturb()
        key = m.light("key").id
        kd = m.light_dir[key] / np.linalg.norm(m.light_dir[key])
        who = "depth student (one network: depth camera + proprioception)" if tr.get("student") else "motion generator + tracking policy"
        head = f"Dash: {who}"
        sub = f"stairs {rise * 100:.0f} cm up, {gap * 100:.0f} cm gap, {plat * 100:.0f} cm platform, stairs down"
        sub += "   (domain randomization + sensor noise)" if tr["dr"] else ""

        def frames():
            for t in range(len(Q)):
                m.light_pos[key] = look[t] - 20 * kd
                d.qpos[:] = Q[t]
                mujoco.mj_forward(m, d)
                cam.lookat[:] = look[t]
                r.update_scene(d, cam, opt)
                if a.ghost:
                    dg.qpos[:] = Rp[t]
                    mujoco.mj_forward(m, dg)
                    n0 = r.scene.ngeom
                    mujoco.mjv_addGeoms(m, dg, opt, pert, mujoco.mjtCatBit.mjCAT_DYNAMIC, r.scene)
                    for g in range(n0, r.scene.ngeom):
                        r.scene.geoms[g].rgba[:] = (0.1, 0.8, 0.3, 0.16)
                r.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = 1
                im = Image.fromarray(r.render())
                dr = ImageDraw.Draw(im)
                dr.text((h // 30, h // 40), head, fill=(30, 34, 40), font=font)
                dr.text((h // 30, h // 40 + h // 22), sub, fill=(60, 66, 76), font=font2)
                dr.text((h // 30, h // 40 + 2 * h // 22), f"joystick {tr['speeds'][s]:.1f} m/s   robot {vel[t]:.1f} m/s", fill=(60, 66, 76), font=font2)
                dr.text((h // 30, h // 40 + 3 * h // 22), outcome, fill=(40, 120, 60) if res[i]["success"] else (170, 50, 40), font=font2)
                if a.ghost:
                    dr.text((h // 30, h // 40 + 4 * h // 22), "green ghost: the generator's plan", fill=(40, 140, 70), font=font2)
                yield np.asarray(im)
            r.close()

        f = out / f"course_v{v}_{tr['speeds'][s]}ms_{k}{'_ghost' if a.ghost else ''}.mp4"
        encode(str(f), frames(), w, h)
        print(f"wrote {f}: {tr['names'][v]}, joystick {tr['speeds'][s]} m/s, {res[i]}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gen_ckpt", nargs="?")
    ap.add_argument("teacher", nargs="?")
    ap.add_argument("out")
    ap.add_argument("--speeds", type=float, nargs="+", default=[1.0, 1.5, 2.0])
    ap.add_argument("--rises", type=float, nargs="+", default=[0.10, 0.14])
    ap.add_argument("--gaps", type=float, nargs="+", default=[0.3, 0.5])
    ap.add_argument("--platforms", type=float, nargs="+", default=[0.2, 0.3])
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dr", action="store_true", help="robot + hardware DR, observation noise, 0-25 ms actuator delay (student: and the depth "
                    "camera's noise, dropouts, latency and pose)")
    ap.add_argument("--foot-guide", action="store_true", help="guided sampling keeping the plan's soles out of the terrain (generator/guidance.py)")
    ap.add_argument("--guide-step", type=float, default=20.0)
    ap.add_argument("--guide-iters", type=int, default=5)
    ap.add_argument("--guide-last", type=int, default=4, help="guided (final) denoising steps")
    ap.add_argument("--guide-inflate", type=float, default=0.03, help="m: soles kept above the highest ground this close")
    ap.add_argument("--guide-root", action="store_true", help="the correction may move the root too (default: joints only)")
    ap.add_argument("--student", help="run this depth student alone instead of the generator + teacher (gen_ckpt still gives the start pose)")
    ap.add_argument("--render", action="store_true", help="render runs from out/traj.pt")
    ap.add_argument("--rescore", action="store_true", help="re-score out/traj.pt (results.json and traj.pt rewritten)")
    ap.add_argument("--pick", type=int, nargs=3, metavar=("VARIANT", "SPEED", "SAMPLE"))
    ap.add_argument("--ghost", action="store_true")
    ap.add_argument("--size", type=int, nargs=2, default=[1920, 1080])
    ap.add_argument("--dist", type=float, default=4.0)
    ap.add_argument("--tail", type=float, default=1.0, help="s shown after the finish line")
    a = ap.parse_args()
    if a.render:
        render_runs(a)
    elif a.rescore:
        rescore(Path(a.out))
    else:
        run(a)


if __name__ == "__main__":
    main()
