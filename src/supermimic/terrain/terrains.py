"""Terrains as heightfields, one per terrain clip (or shared by many), and their physics boxes.

A terrain is a blocky heightfield ``hf`` (nx, ny) with cell (i, j) centred on ``p0 + (i, j) dx`` in the
terrain frame (= the env frame: every env's origin stays put, the terrain moves into its world). Outside
the grid the edge cells extend (lookups clamp), so heightfields are stored padded by ``PAD`` metres of
edge cells on every side, and the physics reproduces exactly the clamped lookup within that pad.
Heights may be negative (pits); ``floor`` = the lowest cell is the env's floor box top, and every cell
above it becomes part of a merged box from the floor up (``boxes``).

Index -1 is flat ground (z = 0 everywhere). The sidecar ``<library stem>.terrains.npz`` holds them all
(build with ``save_terrains``). Libraries built before (``.tiles.npz``, boxes on a z = 0 floor) load
through ``tiles_to_terrain``, rasterised at ``TILE_DX``.

Each env carries ``num_slots`` box geoms plus its floor (supermimic/terrain/tiles.py); at reset
``MultiClipMotionCommand`` writes the sampled terrain's boxes into its own world.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PAD = 3.0  # m of edge cells around every heightfield
TILE_DX = 0.02  # raster cell of box tiles
PARK_Z = -100.0  # unused slot boxes sit here


@dataclass
class Terrain:
    hf: np.ndarray  # (nx, ny) cell heights
    p0: np.ndarray  # (2,) centre of cell (0, 0)
    dx: float

    @property
    def floor(self) -> float:
        return float(self.hf.min())

    def h(self, xy: np.ndarray) -> np.ndarray:
        """Ground height at xy (..., 2), clamping to the edge cells."""
        ij = np.round((np.asarray(xy, float) - self.p0) / self.dx).astype(int)
        i = np.clip(ij[..., 0], 0, self.hf.shape[0] - 1)
        j = np.clip(ij[..., 1], 0, self.hf.shape[1] - 1)
        return self.hf[i, j]

    def padded(self, pad: float = PAD) -> Terrain:
        k = int(np.ceil(pad / self.dx))
        return Terrain(np.pad(self.hf, k, mode="edge"), self.p0 - k * self.dx, self.dx)

    def boxes(self) -> np.ndarray:
        """(B, 6) boxes (centre xyz, half sizes) from the floor up to every cell above it; equal-height
        cells merge greedily (runs along y, then whole runs along x)."""
        hf, floor = self.hf, self.floor
        used = hf <= floor + 1e-6
        out = []
        nx, ny = hf.shape
        for i in range(nx):
            for j in range(ny):
                if used[i, j]:
                    continue
                h = hf[i, j]
                j2 = j
                while j2 + 1 < ny and not used[i, j2 + 1] and abs(hf[i, j2 + 1] - h) < 1e-6:
                    j2 += 1
                i2 = i
                while i2 + 1 < nx and not used[i2 + 1, j : j2 + 1].any() and np.all(np.abs(hf[i2 + 1, j : j2 + 1] - h) < 1e-6):
                    i2 += 1
                used[i : i2 + 1, j : j2 + 1] = True
                x0, x1 = self.p0[0] + (i - 0.5) * self.dx, self.p0[0] + (i2 + 0.5) * self.dx
                y0, y1 = self.p0[1] + (j - 0.5) * self.dx, self.p0[1] + (j2 + 0.5) * self.dx
                out.append([(x0 + x1) / 2, (y0 + y1) / 2, (h + floor) / 2, (x1 - x0) / 2, (y1 - y0) / 2, (h - floor) / 2])
        return np.asarray(out, dtype=np.float32).reshape(-1, 6)


def tiles_to_terrain(boxes: np.ndarray, dx: float = TILE_DX, pad: float = PAD) -> Terrain:
    """Box tile on a z = 0 floor (centre xyz, half sizes) -> heightfield of its tops, padded with floor."""
    b = np.asarray(boxes, dtype=float).reshape(-1, 6)
    lo = (b[:, :2] - b[:, 3:5]).min(0) - pad
    hi = (b[:, :2] + b[:, 3:5]).max(0) + pad
    n = np.ceil((hi - lo) / dx).astype(int)
    p0 = lo + dx / 2
    xs, ys = p0[0] + dx * np.arange(n[0]), p0[1] + dx * np.arange(n[1])
    hf = np.zeros(n, dtype=np.float32)
    for cx, cy, cz, hx, hy, hz in b:
        ix = (xs >= cx - hx) & (xs < cx + hx)
        iy = (ys >= cy - hy) & (ys < cy + hy)
        hf[np.ix_(ix, iy)] = np.maximum(hf[np.ix_(ix, iy)], cz + hz)
    return Terrain(hf, p0, dx)


@dataclass
class TerrainSet:
    names: list[str]
    terrains: list[Terrain]
    boxes: list[np.ndarray]  # physics boxes per terrain

    def __len__(self) -> int:
        return len(self.terrains)

    @property
    def num_slots(self) -> int:
        return max([1] + [len(b) for b in self.boxes])

    def slot_arrays(self, num_slots: int | None = None):
        """Row t + 1 for terrain t, row 0 flat: slot centres (T+1, K, 3), half sizes (T+1, K, 3), floor top
        (T+1,). Unused slots are small boxes parked at PARK_Z."""
        k = num_slots or self.num_slots
        n = len(self) + 1
        pos = np.zeros((n, k, 3), np.float32)
        pos[..., 2] = PARK_Z
        size = np.full((n, k, 3), 0.01, np.float32)
        floor = np.zeros(n, np.float32)
        for t, (ter, b) in enumerate(zip(self.terrains, self.boxes)):
            assert len(b) <= k, (self.names[t], len(b), k)
            pos[t + 1, : len(b)] = b[:, :3]
            size[t + 1, : len(b)] = b[:, 3:]
            floor[t + 1] = ter.floor
        return pos, size, floor


def terrains_path(library: str | Path) -> Path:
    p = Path(library)
    return p.with_name(p.stem + ".terrains.npz")


def save_terrains(path: str | Path, names: list[str], terrains: list[Terrain]) -> TerrainSet:
    """Write the sidecar (heightfields as given, so pad them first) with their physics boxes."""
    boxes = [t.boxes() for t in terrains]
    np.savez(
        path,
        names=np.array(names),
        heights=np.concatenate([t.hf.astype(np.float32).ravel() for t in terrains]) if terrains else np.zeros(0, np.float32),
        shape=np.array([t.hf.shape for t in terrains], dtype=np.int64).reshape(-1, 2),
        p0=np.array([t.p0 for t in terrains], dtype=np.float32).reshape(-1, 2),
        dx=np.array([t.dx for t in terrains], dtype=np.float32),
        boxes=np.concatenate(boxes) if boxes else np.zeros((0, 6), np.float32),
        box_count=np.array([len(b) for b in boxes], dtype=np.int64),
    )
    return TerrainSet(list(names), list(terrains), boxes)


def read_terrains(path: str | Path) -> TerrainSet:
    with np.load(path) as f:
        z = {k: f[k] for k in f.files}  # read each array once (indexing an NpzFile re-reads it)
    names, terrains, boxes = [str(n) for n in z["names"]], [], []
    o = b = 0
    for (nx, ny), p0, dx, nb in zip(z["shape"], z["p0"], z["dx"], z["box_count"]):
        terrains.append(Terrain(z["heights"][o : o + nx * ny].reshape(nx, ny).copy(), p0.astype(float), float(dx)))
        boxes.append(z["boxes"][b : b + nb].copy())
        o += nx * ny
        b += nb
    return TerrainSet(names, terrains, boxes)


@functools.lru_cache(maxsize=4)
def load_terrains(library: str | Path) -> TerrainSet | None:
    """Terrains of a packed library (``.terrains.npz``, or an older ``.tiles.npz`` converted), None if flat.
    Cached: the env config, the command and the world model all ask."""
    if not str(library).endswith(".pt"):
        return None
    p = terrains_path(library)
    if p.exists():
        return read_terrains(p)
    from supermimic.terrain.tiles import load_tiles

    tiles = load_tiles(library)
    if tiles is None:
        return None
    terrains = [tiles_to_terrain(t) for t in tiles]
    return TerrainSet([f"tile_{i}" for i in range(len(tiles))], terrains, [t.boxes() for t in terrains])
