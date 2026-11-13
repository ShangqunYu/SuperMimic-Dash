"""Benchmark a terrain-library checkpoint by clip group: every PARC clip, every v3 mocap terrain clip, 2,000
random flat clips (the same ones in every library built on v4_compact), and with a v4 or later library
the US stairs (12-20 cm risers) and the 25-31 cm box jumps.

  python scripts/eval_groups.py out.json ckpt.pt [--lib PATH] [--only parc,flat2k] [--drift-m 1.0] [--randomize]
      [--contact-timeconst S]

Per group: failures by the paper rule (root height / tilt), and failures or passes whose root got more than
--drift-m from the reference horizontally (eval/benchmark.py). Also PARC by motion type, US stairs by riser
and direction, box jumps by height, and every clip's result under "clips". Contacts as the checkpoint was
trained (env.contact_timeconst) unless --contact-timeconst (0.02: the model's default).
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

from supermimic.eval.benchmark import goal_env_kwargs, load_policy_from_checkpoint, run_benchmark, summarize

sys.path.insert(0, str(Path(__file__).resolve().parent / "parc"))
from teacher_filter import clip_type  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
US_JUMP = re.compile(r"_h(\d+)_d\d+_w\d+$")
US_STAIR = re.compile(r"_us_r(\d+)_t\d+$")


def groups(lib: Path) -> tuple[dict[str, list[int]], list[str]]:
    pack = torch.load(lib, map_location="cpu", weights_only=False, mmap=True)
    names, tile = list(pack["names"]), np.asarray(pack["clip_tile"])
    parc = [i for i, n in enumerate(names) if n.startswith("parc/")]
    terr = [i for i, n in enumerate(names) if tile[i] >= 0 and not n.startswith("parc/") and not US_STAIR.search(n) and not US_JUMP.search(n)]
    flat = [i for i in range(len(names)) if tile[i] < 0]
    flat = sorted(np.random.default_rng(0).choice(flat, 2000, replace=False).tolist())
    g = {"parc": parc, "terrain_v3": terr, "flat2k": flat}
    us_st = [i for i, n in enumerate(names) if US_STAIR.search(n)]
    if us_st:
        g |= {"us_stairs": us_st, "us_jumps": [i for i, n in enumerate(names) if US_JUMP.search(n)]}
    return g, names


def breakdown(res, key, drift_m) -> dict[str, str]:
    """'failed / failed or drifted / clips' per key(name) (None: skip)."""
    c = collections.defaultdict(lambda: [0, 0, 0])
    for r in res:
        k = key(r.name)
        if k is not None:
            c[k][0] += r.failed
            c[k][1] += r.failed or r.max_xy_err > drift_m
            c[k][2] += 1
    return {k: f"{a}/{b}/{n}" for k, (a, b, n) in sorted(c.items())}


def stair_key(n):
    r = US_STAIR.search(n).group(1)
    return f"{'up' if 'up_start' in n else 'down'} {int(r) / 10 if len(r) == 3 else int(r):g} cm"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("ckpt")
    ap.add_argument("--lib", default=str(REPO / "data/library/dash_flat_terrain_v4_parc_compact.pt"))
    ap.add_argument("--only", help="comma-separated groups")
    ap.add_argument("--drift-m", type=float, default=1.0)
    ap.add_argument("--randomize", action="store_true", help="training randomization (noise, DR) instead of nominal")
    ap.add_argument("--contact-timeconst", type=float, help="every geom's contact time constant (s); default: the checkpoint's")
    a = ap.parse_args()
    g, names = groups(Path(a.lib))
    if a.only:
        g = {k: v for k, v in g.items() if k in a.only.split(",")}
    policy, cfg = load_policy_from_checkpoint(a.ckpt, "cuda:0")
    ids = [i for idx in g.values() for i in idx]
    kw = goal_env_kwargs(cfg)
    if a.contact_timeconst is not None:
        kw["contact_timeconst"] = a.contact_timeconst
    res = run_benchmark(policy, [a.lib], device="cuda:0", num_envs=1024, robot="dash", num_clips=len(names), verbose=False, randomize=a.randomize,
                        clip_ids=ids, **kw)
    out, k = {"ckpt": a.ckpt, "lib": a.lib, "drift_m": a.drift_m, "contact_timeconst": kw["contact_timeconst"], "groups": {}, "clips": {}}, 0
    for name, idx in g.items():
        r = res[k : k + len(idx)]
        k += len(idx)
        s = summarize(r, a.drift_m)
        s.pop("terms", None)
        passed = np.array([x.max_xy_err for x in r if not x.failed])
        s["pass_drift_p50_p90_max"] = [round(float(np.percentile(passed, q)), 3) for q in (50, 90, 100)] if len(passed) else None
        s["failed_names"] = [x.name for x in r if x.failed]
        out["groups"][name] = s
        out["clips"] |= {x.name: {"failed": x.failed, "fail_step": x.fail_step, "steps": x.steps, "max_xy_err": round(x.max_xy_err, 4), "norm_return": round(x.norm_return, 4)} for x in r}
        line = (f"{name}: {s['failed']}/{s['clips']} failed, {s['failed_or_drift']} failed or drift > {a.drift_m:g} m, mean {s['mean_norm_return']:.4f}, "
                f"step-weighted {s['step_weighted_norm_return']:.4f}, pass drift p50/p90/max {s['pass_drift_p50_p90_max']}")
        print(line, flush=True)
        if name == "parc":
            s["by_type"] = breakdown(r, clip_type, a.drift_m)
        elif name == "us_stairs":
            s["by_riser"] = breakdown(r, stair_key, a.drift_m)
        elif name == "us_jumps":
            s["by_height_cm"] = breakdown(r, lambda n: US_JUMP.search(n).group(1), a.drift_m)
        for key in ("by_type", "by_riser", "by_height_cm"):
            if key in s:
                print(f"  {key} (failed / failed or drift / clips): " + ", ".join(f"{t} {v}" for t, v in s[key].items()), flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
