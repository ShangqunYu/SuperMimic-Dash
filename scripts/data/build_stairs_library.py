"""Stairs-only tracking library (stairs up / down, plus flat walking for the run-up and landings), with a held-out
eval pack of stairs clips.

  python scripts/data/build_stairs_library.py --out data/library/dash_stairs_v1.pt

Stairs: every "stairs" clip of the base library (mocap stairs + PARC stairs, generator/data.py terrain_category) and
a sample of the physics-augmented stairs runs (generator + teacher in physics, scripts/generator/augment.py:
procedural stairs, riser 8-19 cm, tread 24-36 cm, 0.3-1 m/s). Flat: locomotion clips of the base library
(generator/data.py is_locomotion) up to --flat-share of the hours. Held out: stairs clips whose base motion hashes
to 0 mod --val-every (as GenData), written to <out stem>_eval.pt with their terrains.
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np
import torch
import tyro

from supermimic.generator.data import base_motion, is_locomotion, terrain_category
from supermimic.terrain.terrains import load_terrains, save_terrains, terrains_path

FPS = 50
AUG = Path("data/library/gen_aug")


def main(
    out: Path,
    base: Path = Path("data/library/dash_flat_terrain_v5_parc_compact.pt"),
    aug_up: tuple[str, ...] = ("aug3a", "aug4a"),
    aug_down: tuple[str, ...] = ("aug3a", "aug4a", "aug4b"),
    n_up: int = 2000,
    n_down: int = 1700,
    flat_share: float = 0.2,
    val_every: int = 20,
    seed: int = 0,
) -> None:
    rng = np.random.default_rng(seed)
    packs: dict[Path, dict] = {}

    def pack(p: Path) -> dict:
        if p not in packs:
            packs[p] = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
        return packs[p]

    def clips_of(p: Path, keep) -> list[tuple[Path, int]]:
        pk = pack(p)
        return [(p, i) for i, (n, t) in enumerate(zip(pk["names"], pk["clip_tile"])) if keep(n, int(t))]

    stairs = clips_of(base, lambda n, t: t >= 0 and terrain_category(n) == "stairs")
    for fam, srcs, n_want in (("stairs_up", aug_up, n_up), ("stairs_down", aug_down, n_down)):
        cand = [c for s in srcs for c in clips_of(AUG / f"{s}.pt", lambda n, t, fam=fam: n.startswith(f"aug/{fam}/"))]
        pick = rng.choice(len(cand), size=min(n_want, len(cand)), replace=False)
        stairs += [cand[i] for i in sorted(pick)]

    def name(c):
        return pack(c[0])["names"][c[1]]

    def secs(c):
        return int(pack(c[0])["lengths"][c[1]]) / FPS

    held = [zlib.crc32((name(c) if name(c).startswith("aug/") else base_motion(name(c))).encode()) % val_every == 0 for c in stairs]
    train_stairs = [c for c, h in zip(stairs, held) if not h]
    eval_stairs = [c for c, h in zip(stairs, held) if h]

    flat = clips_of(base, lambda n, t: t < 0 and is_locomotion(n))
    flat = [flat[i] for i in rng.permutation(len(flat))]
    stairs_h = sum(secs(c) for c in train_stairs) / 3600
    want = flat_share / (1 - flat_share) * stairs_h * 3600
    picked, tot = [], 0.0
    for c in flat:
        if tot >= want:
            break
        picked.append(c)
        tot += secs(c)

    def write(path: Path, clips: list[tuple[Path, int]]) -> dict:
        dtype = pack(base)["ref"].dtype
        refs, names, tile, t_names, terrains, seen = [], [], [], [], [], {}
        for p, i in clips:
            pk = pack(p)
            s, n = int(pk["starts"][i]), int(pk["lengths"][i])
            refs.append(pk["ref"][s : s + n].to(dtype))
            names.append(pk["names"][i])
            t = int(pk["clip_tile"][i])
            if t < 0:
                tile.append(-1)
                continue
            if (p, t) not in seen:
                ts = load_terrains(p)
                seen[(p, t)] = len(terrains)
                terrains.append(ts.terrains[t])
                t_names.append(ts.names[t])
            tile.append(seen[(p, t)])
        lengths = [len(r) for r in refs]
        starts = np.cumsum([0] + lengths[:-1]).tolist()
        torch.save({"ref": torch.cat(refs), "names": names, "starts": starts, "lengths": lengths, "robot": pack(base)["robot"],
                    "clip_tile": tile, "terrain_names": t_names}, path)
        tset = save_terrains(terrains_path(path), t_names, terrains)
        cats: dict[str, list[float]] = {}
        for nm, n, t in zip(names, lengths, tile):
            k = "flat" if t < 0 else (nm.split("/")[1] if nm.startswith("aug/") else "stairs_mocap_parc" if nm.startswith("parc/") else "stairs_mocap")
            cats.setdefault(k, [0, 0.0])
            cats[k][0] += 1
            cats[k][1] += n / FPS / 3600
        info = {"clips": len(names), "hours": round(sum(lengths) / FPS / 3600, 2), "terrains": len(terrains), "num_slots": tset.num_slots,
                "by_kind": {k: {"clips": v[0], "hours": round(v[1], 2)} for k, v in cats.items()}}
        print(f"saved {path}: {json.dumps(info)}", flush=True)
        return info

    out.parent.mkdir(parents=True, exist_ok=True)
    ev = out.with_name(out.stem + "_eval.pt")
    info = {"train": write(out, train_stairs + picked), "eval": write(ev, eval_stairs),
            "args": {"base": str(base), "aug_up": aug_up, "aug_down": aug_down, "n_up": n_up, "n_down": n_down, "flat_share": flat_share,
                     "val_every": val_every, "seed": seed}}
    out.with_suffix(".json").write_text(json.dumps(info, indent=1))


if __name__ == "__main__":
    tyro.cli(main)
