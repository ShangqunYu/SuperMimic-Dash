"""Edge contact of a packed library's own reference motion (its clips on their terrains), by kind: share of frames
with a foot point inside a lip cylinder (r 0.05 + capsule), edge_vol per frame (sum depth x point speed), mean
deepest depth. Same units as scripts/eval_edge.py (physics) and EdgeTerm (imagination, before dt x weight).

  python scripts/diag/ref_edge.py LIBRARY.pt
"""
import sys
from types import SimpleNamespace

import numpy as np
import torch

from supermimic.generator.metrics import FootMetrics, edge_penetration
from supermimic.terrain.height_scan import HeightScan
from supermimic.terrain.terrains import load_terrains

dev = "cuda:0"
lib = sys.argv[1]
pk = torch.load(lib, map_location="cpu", weights_only=False, mmap=True)
ts = load_terrains(lib)
scan = HeightScan(SimpleNamespace(terrains=ts.terrains)).to(dev)
fm = FootMetrics(scan, dev)
acc = {}
for i, (nm, t) in enumerate(zip(pk["names"], pk["clip_tile"])):
    if t < 0:
        continue
    kind = nm.split("/")[1] if nm.startswith("aug/") else ("parc" if nm.startswith("parc/") else "mocap")
    s, n = int(pk["starts"][i]), int(pk["lengths"][i])
    Q = pk["ref"][s : s + n, :25].float().to(dev)[None]
    pts = fm.axis(Q[..., :3], Q[..., 3:7], Q[..., 7:], 12).reshape(n, -1, 3)
    tile = torch.full((n,), int(t), device=dev)
    pen = edge_penetration(scan, pts, tile, 0.05 + fm.radius)
    sp = torch.zeros_like(pen)
    sp[1:] = (pts[1:] - pts[:-1]).norm(dim=-1) * 50
    a = acc.setdefault(kind, np.zeros(5))
    a += [n, float((pen.amax(-1) > 0).sum()), float((pen * (sp + 1e-6)).sum()), float(pen.amax(-1).sum()), 1]
for k, (n, hit, vol, dep, c) in sorted(acc.items()):
    print(f"{lib.split('/')[-1]:28s} {k:12s} clips {int(c):5d} frames {int(n):8d}  hit {hit / n:.3f}  edge_vol {vol / n:.4f}  depth {100 * dep / n:.2f} cm")
