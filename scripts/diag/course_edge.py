"""Physics edge contact on course_demo runs while the robot is up, per speed: stairs-up zone (0.3 m before the first
riser to the top) and stairs-down zone (0.3 m before the first step down to 0.3 m past the bottom). Each argument
is a run prefix whose seeds are <prefix>_s0, _s1, ... (course_demo out dirs). Same units as scripts/eval_edge.py:
share of steps with a foot point inside a lip cylinder (5 cm + the foot capsule's radius), edge_vol per step (sum
of depth x point speed), mean deepest depth, and how far the robot's feet are from the plan's (capsule centres,
mean over both feet). --key plan: the generator's plan instead of the robot.

  python scripts/diag/course_edge.py logs/stairs_eval/course_edge_50k logs/stairs_eval/course_ctrl_50k [--key plan]
"""
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "generator"))
from course_demo import LANDING, START, TREAD, course  # noqa: E402

from supermimic.generator.metrics import FootMetrics, edge_penetration  # noqa: E402
from supermimic.terrain.height_scan import HeightScan  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--key", choices=("robot", "plan"), default="robot")
a = ap.parse_args()
dev = "cuda:0"
fm = None
for run in a.runs:
    acc = {}
    for d in sorted(Path(run).parent.glob(Path(run).name + "_s*")):
        if not (d / "traj.pt").exists():
            continue
        t = torch.load(d / "traj.pt", weights_only=False)
        cs = [course(*v) for v in t["variants"]]
        scan = HeightScan(SimpleNamespace(terrains=[c[0] for c in cs])).to(dev)
        if fm is None:
            fm = FootMetrics(scan, dev)
        fm.scan = scan
        Q = t[a.key].to(dev)
        n, T = Q.shape[:2]
        tiles = torch.tensor([c[0] for c in t["combo"]], device=dev)
        pts = fm.axis(Q[..., :3], Q[..., 3:7], Q[..., 7:], 12).reshape(n, T, -1, 3)
        pen = edge_penetration(scan, pts.reshape(n * T, -1, 3), tiles[:, None].expand(n, T).reshape(-1), 0.05 + fm.radius).reshape(n, T, -1)
        speed = torch.zeros_like(pen)
        speed[:, 1:] = (pts[:, 1:] - pts[:, :-1]).norm(dim=-1) / 0.02
        vol = (pen * (speed + 1e-6)).sum(-1)
        R = t["plan"].to(dev)
        fp = fm.axis(R[..., :3], R[..., 3:7], R[..., 7:], 12).reshape(n, T, 2, 12, 3).mean(-2)
        gap = (pts.reshape(n, T, 2, 12, 3).mean(-2) - fp).norm(dim=-1).mean(-1)  # (n, T)
        ground = scan.lookup(Q[..., 0].reshape(n, -1), Q[..., 1].reshape(n, -1), tiles).reshape(n, T)
        up = torch.cummin((Q[..., 2] - ground > 0.3).float(), 1).values.bool()
        for k, (v, sp, _) in enumerate(t["combo"]):
            marks = cs[v][1]
            top = START + 5 * TREAD
            d0 = marks["platform"] - 0.3 + LANDING  # first step down
            x = Q[k, :, 0]
            for zone, m in (("up", up[k] & (x > START - 0.3) & (x < top)), ("down", up[k] & (x > d0 - 0.3) & (x < marks["stairs down"]))):
                c = acc.setdefault((zone, t["speeds"][sp]), [0, 0.0, 0.0, 0.0, 0.0])
                c[0] += int(m.sum())
                c[1] += float(((pen[k].amax(-1) > 0) & m).sum())
                c[2] += float((vol[k] * m).sum())
                c[3] += float((pen[k].amax(-1) * m).sum())
                c[4] += float((gap[k] * m).sum())
    for (zone, sp), (cnt, hit, v_, dep, fg) in sorted(acc.items()):
        print(f"{Path(run).name:20s} {a.key:5s} stairs {zone:4s} {sp} m/s: steps {cnt:6d}  hit {hit / max(cnt, 1):.3f}  edge_vol {v_ / max(cnt, 1):.4f}  "
              f"depth {100 * dep / max(cnt, 1):.2f} cm  feet from plan {100 * fg / max(cnt, 1):.1f} cm", flush=True)
