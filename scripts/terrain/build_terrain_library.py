"""Pack the terrain clips into a compact motion library with terrain tiles.

Each clip passing QA (data/terrain/manifest.csv) is moved into its tile's frame by a yaw rotation +
planar shift, resampled 60 -> 50 fps, and given velocities (finite differences; world-frame angular
velocity from the SO(3) central difference), the same 13 + 2 NJ reference layout as build_library.py.

Tiles (supermimic/terrain/tiles.py; boxes in the tile frame, ground at z = 0, x = walking direction):
  * "box30": every box clip. Edge at x = 0, the 0.30 m box over x in [0, 4] m, 4 m wide.
  * "stairs_<clip>": one per stairs clip (each has its own rise/run), solid steps, 3 m wide,
    the top landing 3 m long.
  * "jump_h<cm>_d<cm>": the box under jump clips, centred at x = 0, 0.6 m wide; clips with the same
    box share it.
With ``--flat`` the flat library is prepended (its clips get tile -1) for mixed training.

  uv run python scripts/terrain/build_terrain_library.py --out data/library/dash_terrain_compact.pt
  uv run python scripts/terrain/build_terrain_library.py --flat data/library/dash_bones_seed_compact.pt --out data/library/dash_flat_terrain_compact.pt
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
import tyro
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp

from supermimic.robots import get_robot
from supermimic.terrain.tiles import tiles_path

FPS = 50.0
BOX_HEIGHT = 0.30
BOX_LEN, BOX_WIDTH = 4.0, 4.0
STAIR_WIDTH, LANDING = 3.0, 3.0


def resample(qpos: np.ndarray, fps_in: float) -> np.ndarray:
    """qpos (T, 7+NJ) wxyz -> 50 fps reference (T', 13 + 2 NJ)."""
    T = len(qpos)
    t_in = np.arange(T) / fps_in
    t = np.minimum(np.arange(0.0, t_in[-1] + 1e-9, 1.0 / FPS), t_in[-1])
    pos = np.stack([np.interp(t, t_in, qpos[:, k]) for k in range(3)], 1)
    jp = np.stack([np.interp(t, t_in, qpos[:, k]) for k in range(7, qpos.shape[1])], 1)
    rot = Slerp(t_in, Rot.from_quat(qpos[:, [4, 5, 6, 3]]))(t)
    q = rot.as_quat()[:, [3, 0, 1, 2]]
    q = q * np.sign(q[:, :1] + 1e-12)
    dt = 1.0 / FPS
    lin = np.gradient(pos, dt, axis=0)
    jv = np.gradient(jp, dt, axis=0)
    ang = np.zeros_like(pos)
    if len(t) >= 3:
        ang[1:-1] = (rot[2:] * rot[:-2].inv()).as_rotvec() / (2 * dt)
        ang[0], ang[-1] = (rot[1] * rot[0].inv()).as_rotvec() / dt, (rot[-1] * rot[-2].inv()).as_rotvec() / dt
    return np.concatenate([pos, q, jp, lin, ang, jv], 1)


def to_tile_frame(qpos: np.ndarray, origin_xy: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Rotate/shift qpos so that origin_xy -> 0 and direction d -> +x."""
    yaw = np.arctan2(d[1], d[0])
    c, s = np.cos(-yaw), np.sin(-yaw)
    out = qpos.copy()
    xy = qpos[:, :2] - origin_xy
    out[:, 0], out[:, 1] = c * xy[:, 0] - s * xy[:, 1], s * xy[:, 0] + c * xy[:, 1]
    r = Rot.from_euler("z", -yaw) * Rot.from_quat(qpos[:, [4, 5, 6, 3]])
    out[:, 3:7] = r.as_quat()[:, [3, 0, 1, 2]]
    return out


def box(x0, x1, y0, y1, z0, z1):
    return [(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2, (x1 - x0) / 2, (y1 - y0) / 2, (z1 - z0) / 2]


def main(manifest: Path = Path("data/terrain/manifest.csv"), out: Path = Path("data/library/dash_terrain_compact.pt"), flat: Path | None = None) -> None:
    spec = get_robot("dash")
    tiles: dict[str, list] = {"box30": [box(0.0, BOX_LEN, -BOX_WIDTH / 2, BOX_WIDTH / 2, 0.0, BOX_HEIGHT)]}
    names, refs, clip_tile = [], [], []
    rows = [r for r in csv.DictReader(open(manifest)) if r["pass"] == "1"]
    for r in rows:
        z = np.load(r["file"])
        qpos, fps = z["qpos"], float(z["fps"])
        kind = str(z["kind"])
        d = np.asarray(z["d"], dtype=float)
        if kind == "box":
            edges = z["edges"]
            o = z["c"] + edges[0] * d
            key = "box30"
        elif kind == "stairs":
            o = np.asarray(z["c"], dtype=float)
            key = f"stairs_{r['clip']}"
            edges, heights = z["edges"], z["heights"]
            bxs = []
            for j in range(1, len(heights)):
                h = float(heights[j])
                if h <= 0.005:
                    continue
                x1 = float(edges[j]) if j < len(edges) else float(edges[-1]) + LANDING
                bxs.append(box(float(edges[j - 1]), x1, -STAIR_WIDTH / 2, STAIR_WIDTH / 2, 0.0, h))
            tiles[key] = bxs
        else:  # jump_*: box under the jump
            s_lo, s_hi = z["box_s"]
            h, w = float(z["box_height"]), float(z["box_width"])
            rs = (qpos[:, :2] - z["c"]) @ d
            t = int(np.argmin(np.abs(rs - (s_lo + s_hi) / 2)))
            perp = np.array([-d[1], d[0]])
            lat0 = (qpos[t, :2] - z["c"]) @ perp
            o = z["c"] + (s_lo + s_hi) / 2 * d + lat0 * perp
            depth = float(s_hi - s_lo)
            key = f"jump_h{round(h * 100)}_d{round(depth * 100)}"
            tiles.setdefault(key, [box(-depth / 2, depth / 2, -w / 2, w / 2, 0.0, h)])
        ref = resample(to_tile_frame(qpos, o, d), fps)
        names.append(f"dash_{r['clip']}")
        refs.append(ref.astype(np.float32))
        clip_tile.append(key)
    tile_names = list(tiles)
    tile_idx = [tile_names.index(k) for k in clip_tile]
    lengths = [len(x) for x in refs]
    ref = torch.from_numpy(np.concatenate(refs, 0)).to(torch.float16)
    assert ref.shape[1] == spec.state_dim, (ref.shape, spec.state_dim)
    if flat is not None:
        fp = torch.load(flat, map_location="cpu", weights_only=False)
        names = list(fp["names"]) + names
        tile_idx = [-1] * len(fp["names"]) + tile_idx
        lengths = list(fp["lengths"]) + lengths
        ref = torch.cat([fp["ref"], ref], 0)
    starts = np.cumsum([0] + lengths[:-1]).tolist()
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"ref": ref, "names": names, "starts": starts, "lengths": lengths, "robot": spec.name,
                "clip_tile": tile_idx, "tile_names": tile_names, "tiles": [np.asarray(tiles[k], dtype=np.float32) for k in tile_names]}, out)
    # Sidecar read by the env config to build the tiles without loading the whole pack.
    np.savez(tiles_path(out), names=np.array(tile_names), **{f"tile_{i}": np.asarray(tiles[k], dtype=np.float32) for i, k in enumerate(tile_names)})
    nt = sum(t >= 0 for t in tile_idx)
    tf = sum(l for l, t in zip(lengths, tile_idx) if t >= 0)
    print(f"saved {out}: {len(names)} clips ({nt} terrain, {tf / FPS / 3600:.2f} h), {ref.shape[0] / FPS / 3600:.2f} h total, {len(tile_names)} tiles")
    by = {}
    for k in clip_tile:
        g = k.split("_")[0]
        by[g] = by.get(g, 0) + 1
    print("terrain clips by tile type:", by)


if __name__ == "__main__":
    tyro.cli(main)
