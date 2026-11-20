"""Outcomes of course_demo runs per run prefix (seeds <prefix>_s0, _s1, ...): per riser and speed, successes, falls
and timeouts (course_demo.score), the mean course speed of the successes, and stalls: runs whose robot, while up and
short of the finish, advanced < 0.15 m in some 1.5 s window. --speeds limits the speeds shown.

  python scripts/diag/course_summary.py logs/stairs_eval/course_edge_50k logs/stairs_eval/plan_sel8_edge [--speeds 0.5 1.0]
"""
import argparse
from pathlib import Path

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--speeds", type=float, nargs="*")
a = ap.parse_args()
for run in a.runs:
    tab = {}
    for d in sorted(Path(run).parent.glob(Path(run).name + "_s*")):
        if not (d / "traj.pt").exists():
            continue
        t = torch.load(d / "traj.pt", weights_only=False)
        for i, ((v, sp, _), r) in enumerate(zip(t["combo"], t["results"])):
            if a.speeds and t["speeds"][sp] not in a.speeds:
                continue
            x = t["robot"][i, :, 0].numpy()
            end = int(r["finish_s"] * 50) if r["finish_s"] else int(r["down_at_s"] * 50) if r["down_at_s"] else len(x)
            stall = end > 75 and bool(((x[75:end] - x[: end - 75]) < 0.15).any())
            c = tab.setdefault((t["variants"][v][0], t["speeds"][sp]), np.zeros(6))
            c += [1, r["success"], r["down_at_s"] is not None and not r["success"], r["down_at_s"] is None and not r["success"], r["course_m_s"] or 0, stall]
    tot = np.zeros(6)
    for (rise, sp), c in sorted(tab.items()):
        tot += c
        print(f"{Path(run).name:22s} riser {100 * rise:4.0f} cm {sp} m/s: success {int(c[1]):3d}/{int(c[0])}  falls {int(c[2]):2d}  timeouts {int(c[3]):2d}  "
              f"speed {c[4] / max(c[1], 1):.2f} m/s  stalls {int(c[5])}")
    print(f"{Path(run).name:22s} all: success {int(tot[1])}/{int(tot[0])}  falls {int(tot[2])}  timeouts {int(tot[3])}  stalls {int(tot[5])}", flush=True)
