"""Cut the first SECONDS of every clip of a packed library (terrains sidecar copied alongside).

  python scripts/generator/trim_library.py in.pt out.pt [--seconds 1.0]
"""

import argparse
import shutil

import torch

from supermimic.terrain.terrains import terrains_path

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("dst")
ap.add_argument("--seconds", type=float, default=1.0)
a = ap.parse_args()
p = torch.load(a.src, map_location="cpu", weights_only=False)
k = int(a.seconds * 50)
refs = [p["ref"][s + k : s + n] for s, n in zip(p["starts"], p["lengths"])]
lens = [len(r) for r in refs]
p["ref"] = torch.cat(refs)
p["lengths"] = lens
p["starts"] = [sum(lens[:i]) for i in range(len(lens))] if len(lens) < 1000 else torch.cumsum(torch.tensor([0] + lens[:-1]), 0).tolist()
torch.save(p, a.dst)
shutil.copy(terrains_path(a.src), terrains_path(a.dst))
print(f"{len(lens)} clips, {sum(lens) / 50 / 3600:.2f} h -> {a.dst}")
