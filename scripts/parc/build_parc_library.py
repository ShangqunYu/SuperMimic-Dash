"""Append PARC terrain retargets to a packed library, each clip with its own terrain heightfield.

  python scripts/parc/build_parc_library.py --base data/library/dash_flat_terrain_v3_compact.pt \
      --clips /media/shangqunyu/Dash/parc_dash_30cm/qa/feasible_loose_peak5.txt \
      --out data/library/dash_flat_terrain_v3_parc_compact.pt

The base library's terrains (its ``.terrains.npz``, or its box tiles rasterised; supermimic/terrain/terrains.py)
come first, then one terrain per distinct PARC heightfield, padded by terrains.PAD of edge cells. PARC clips stay
in their heightfield's frame (the retarget's world frame) and are resampled to 50 fps like the other terrain
clips. Writes the pack and its ``.terrains.npz`` sidecar.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import torch
import tyro

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "terrain"))
from build_terrain_library import FPS, resample  # noqa: E402

from supermimic.robots import get_robot  # noqa: E402
from supermimic.terrain.terrains import Terrain, load_terrains, save_terrains, terrains_path  # noqa: E402

PARC_ROOT = Path("/media/shangqunyu/Dash/parc_dash_30cm")


def main(base: Path, clips: Path, out: Path, root: Path = PARC_ROOT) -> None:
    spec = get_robot("dash")
    pack = torch.load(base, map_location="cpu", weights_only=False)
    assert pack.get("robot") == spec.name
    ts = load_terrains(base)
    names_t = list(ts.names) if ts is not None else []
    terrains = list(ts.terrains) if ts is not None else []
    seen: dict[str, int] = {}
    names, refs, clip_tile = [], [], []
    rel = [ln.strip() for ln in open(clips) if ln.strip()]
    for k, c in enumerate(rel):
        z = np.load(root / c)
        hf = np.asarray(z["hf"], dtype=np.float32)
        p0 = np.asarray(z["min_point"], dtype=float)
        key = hashlib.md5(np.round(hf, 4).tobytes() + np.round(p0, 4).tobytes() + str(hf.shape).encode()).hexdigest()
        if key not in seen:
            seen[key] = len(terrains)
            terrains.append(Terrain(hf, p0, float(z["dx"])).padded())
            names_t.append(f"parc_{key[:10]}")
        ref = resample(np.asarray(z["qpos"], dtype=float), float(z["fps"]))
        assert ref.shape[1] == spec.state_dim, (c, ref.shape)
        names.append("parc/" + c.removesuffix(".npz"))
        refs.append(ref.astype(np.float32))
        clip_tile.append(seen[key])
        if k % 500 == 0:
            print(f"{k}/{len(rel)}", flush=True)
    lengths = list(pack["lengths"]) + [len(r) for r in refs]
    ref = torch.cat([pack["ref"], torch.from_numpy(np.concatenate(refs, 0)).to(pack["ref"].dtype)], 0)
    all_names = list(pack["names"]) + names
    tile = list(pack.get("clip_tile") or [-1] * len(pack["names"])) + clip_tile
    starts = np.cumsum([0] + lengths[:-1]).tolist()
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"ref": ref, "names": all_names, "starts": starts, "lengths": lengths, "robot": spec.name, "clip_tile": tile, "terrain_names": names_t}, out)
    tset = save_terrains(terrains_path(out), names_t, terrains)
    nb = [len(b) for b in tset.boxes]
    parc_s = sum(len(r) for r in refs) / FPS
    print(f"saved {out}: {len(all_names)} clips, {ref.shape[0] / FPS / 3600:.2f} h; PARC {len(names)} clips {parc_s / 3600:.2f} h on "
          f"{len(seen)} terrains; {len(terrains)} terrains total, boxes per terrain max {max(nb)} median {int(np.median(nb))}")


if __name__ == "__main__":
    tyro.cli(main)
