"""Track every clip of a packed library in physics with a checkpoint; per kind: failures, mean normalised return,
foot contact with step lips (run_benchmark edge=True). Writes <out>.json.

  python scripts/eval_edge.py CKPT LIBRARY OUT.json [--randomize]
"""
import json
import sys

import torch

from supermimic.eval.benchmark import goal_env_kwargs, load_policy_from_checkpoint, run_benchmark

ckpt, lib, out = sys.argv[1:4]
rand = "--randomize" in sys.argv
policy, cfg = load_policy_from_checkpoint(ckpt, "cuda:0")
n = len(torch.load(lib, map_location="cpu", weights_only=False, mmap=True)["names"])
res = run_benchmark(policy, [lib], device="cuda:0", num_envs=n, num_clips=n, robot="dash", randomize=rand, edge=True, verbose=False, **goal_env_kwargs(cfg))
kinds = {}
for r in res:
    k = r.name.split("/")[1] if r.name.startswith("aug/") else ("parc" if r.name.startswith("parc/") else ("flat" if "stairs" not in r.name and "_us_r" not in r.name else "mocap"))
    a = kinds.setdefault(k, {"clips": 0, "failed": 0, "ret": 0.0, "steps": 0, "edge_hit": 0.0, "edge_vol": 0.0, "edge_depth": 0.0})
    a["clips"] += 1
    a["failed"] += int(r.failed)
    a["ret"] += r.norm_return
    a["steps"] += r.steps
    for e in ("edge_hit", "edge_vol", "edge_depth"):
        a[e] += r.terms[e] * r.steps
for k, a in sorted(kinds.items()):
    print(f"{ckpt.split('/')[-2]:24s} {k:12s} clips {a['clips']:4d} failed {a['failed']:3d} ret {a['ret'] / a['clips']:.3f}  hit {a['edge_hit'] / a['steps']:.3f}  edge_vol {a['edge_vol'] / a['steps']:.4f}  depth {100 * a['edge_depth'] / a['steps']:.2f} cm")
json.dump({"ckpt": ckpt, "library": lib, "randomize": rand, "kinds": kinds, "clips": [r.__dict__ for r in res]}, open(out, "w"), indent=1)
