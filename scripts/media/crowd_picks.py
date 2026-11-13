"""Pick the clips for render_crowd.py: terrain clips the teacher passed in an eval_groups.py run, and expressive
flat BONES-SEED clips (dance, jumps, crawling, combat, gestures, ...) benchmarked here. Only clips tracked closely
(no failure, low drift), one variant per motion (actor and mirror copies dropped), path within --extent m.

  python scripts/media/crowd_picks.py ckpt groups.json out.json [--lib PATH] [--extent 4.4] [--seconds 14]

groups.json: eval_groups.py output for the same checkpoint and library. out.json: {"cells": [[name, kind], ...]
(100), "spares": [...]} in category order; render_crowd.py fills its grid from it.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch

from supermimic.eval.benchmark import goal_env_kwargs, load_policy_from_checkpoint, run_benchmark

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/parc"))
from teacher_filter import clip_type  # noqa: E402

# (label, name regex, count): terrain kinds are clip_type() names plus us_stairs / us_jumps
TERRAIN = [("teaser", None, 12), ("boxes", None, 7), ("platform", None, 7), ("stairs", None, 5), ("running", None, 5),
           ("us_stairs", None, 7), ("us_jumps", None, 7)]
FLAT = [("dance", r"dancing_routine", 14), ("jump", r"^dash_jump_", 5), ("spin", r"spin", 2), ("crouch", r"^dash_crouch", 3),
        ("kneel", r"^dash_kneeling", 2), ("all fours", r"all_fours", 3), ("crawl", r"crawl", 1), ("sit", r"^dash_sitting", 2),
        ("combat", r"combat", 3), ("gesture", r"wave|salute|bicep|highfive", 5), ("injured", r"^dash_injured", 3),
        ("proud", r"proud", 2), ("panic", r"panic", 2), ("get up", r"stand_up_lying", 1), ("hurry", r"^dash_hurry", 2)]
US_JUMP = re.compile(r"_h(\d+)_d\d+_w\d+$")


def base_motion(name: str) -> str:
    """The motion without actor / mirror (BONES-SEED) or augmentation (PARC) suffixes: one cell per motion. A
    BONES-SEED take number stays (dance routine takes are different parts of the routine)."""
    if not name.startswith("parc/"):
        return re.sub(r"__A\d+(_M)?$", "", name)
    return re.sub(r"(_aug\d+|_flipped|_dm|_opt)+$", "", name.split("/")[-1])


def terrain_kind(n: str) -> str:
    return clip_type(n) if n.startswith("parc/") else "us_stairs" if "_us_r" in n else "us_jumps" if US_JUMP.search(n) else "v3"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("groups")
    ap.add_argument("out")
    ap.add_argument("--lib", default=str(REPO / "data/library/dash_flat_terrain_v5_parc_compact.pt"))
    ap.add_argument("--extent", type=float, default=4.4, help="max root path extent (m) over the played part")
    ap.add_argument("--seconds", type=float, default=14.0, help="longest part of a clip played")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    pack = torch.load(a.lib, map_location="cpu", weights_only=False, mmap=True)
    names, lens, starts, tile = list(pack["names"]), np.asarray(pack["lengths"]), np.asarray(pack["starts"]), np.asarray(pack["clip_tile"])
    ref = pack["ref"]
    idx = {n: i for i, n in enumerate(names)}
    T = int(a.seconds * 50)

    def path(i):
        return ref[starts[i] : starts[i] + min(lens[i], T), :3].float().numpy()

    def extent(i):
        xy = path(i)[:, :2]
        return float((xy.max(0) - xy.min(0)).max())

    def take(pool, k, used):
        out = []
        for n in pool:
            b = base_motion(n)
            if b not in used:
                used.add(b)
                out.append(n)
            if len(out) == k:
                break
        return out

    cells, spares, used = [], [], set()
    # terrain: passed, close tracking, above-median return, visible relief, fits a cell
    res = json.loads(Path(a.groups).read_text())["clips"]
    by_kind: dict[str, list[tuple[str, float]]] = {}
    for n, r in res.items():
        i = idx[n]
        if tile[i] < 0 or r["failed"] or r["max_xy_err"] > 0.4:
            continue
        by_kind.setdefault(terrain_kind(n), []).append((n, r["norm_return"]))
    for kind, _, k in TERRAIN:
        c = by_kind.get(kind, [])
        med = np.median([x[1] for x in c])
        pool = [n for n, ret in c if ret >= med]
        rng.shuffle(pool)
        pool = [n for n in pool if extent(idx[n]) <= a.extent and np.ptp(path(idx[n])[:, 2]) >= (0.15 if kind != "us_jumps" else 0.0)]
        got = take(pool, k + 3, used)
        cells += [[n, kind] for n in got[:k]]
        spares += [[n, kind] for n in got[k:]]
        print(f"terrain {kind}: {len(c)} passed, {len(pool)} candidates, took {min(k, len(got))}", flush=True)
    # flat: candidates by name, benchmarked here
    flat = [i for i in range(len(names)) if tile[i] < 0 and lens[i] >= 200]
    cand: dict[str, list[int]] = {}
    for label, rx, k in FLAT:
        pool = [i for i in flat if re.search(rx, names[i])]
        rng.shuffle(pool)
        cand[label] = [i for i in pool[: 6 * k + 20] if extent(i) <= a.extent][: 4 * k + 8]
    ids = sorted({i for v in cand.values() for i in v})
    policy, cfg = load_policy_from_checkpoint(a.ckpt, "cuda:0")
    print(f"benchmarking {len(ids)} flat candidates", flush=True)
    out = run_benchmark(policy, [a.lib], device="cuda:0", num_envs=1024, robot="dash", num_clips=len(names), verbose=False, clip_ids=ids, **goal_env_kwargs(cfg))
    ok = {r.name: r.norm_return for r in out if not r.failed and r.max_xy_err < 0.4}
    for label, _, k in FLAT:
        pool = sorted((names[i] for i in cand[label] if names[i] in ok), key=lambda n: -ok[n])
        got = take(pool, k + 2, used)
        cells += [[n, label] for n in got[:k]]
        spares += [[n, label] for n in got[k:]]
        print(f"flat {label}: {len(cand[label])} candidates, {len(pool)} passed, took {min(k, len(got))}", flush=True)
    # short categories: fill from the spares
    while len(cells) < 100 and spares:
        cells.append(spares.pop(0))
    Path(a.out).write_text(json.dumps({"ckpt": a.ckpt, "lib": a.lib, "cells": cells, "spares": spares}, indent=1))
    print(f"{len(cells)} cells, {len(spares)} spares -> {a.out}")


if __name__ == "__main__":
    main()
