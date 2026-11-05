"""Terrain tiles for terrain clips (boxes, stairs, obstacles), as static boxes in the scene.

Every terrain clip is stored in its tile's frame (scripts/terrain/build_terrain_library.py): a tile
is a list of axis-aligned boxes (centre xyz, half sizes) around the tile origin, ground at z = 0.
Many clips share a tile (all 0.30 m step-ups use one), and since every env is its own MuJoCo world,
any number of envs can use the same tile at once: an env playing a terrain clip just has its origin
moved to that tile (MultiClipMotionCommand).

World layout: flat clips spawn at the origin (they stay within ~9 m of their start); tiles sit on a
grid from x = ``TILE_X0`` on, ``TILE_SPACING`` apart, all on one big floor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry, TerrainOutput

TILE_X0 = 30.0
TILE_SPACING = 16.0
TILE_COLS = 8
FLOOR_THICKNESS = 1.0


def tiles_path(library: str | Path) -> Path:
    """Sidecar file next to a packed library holding its tiles (written by build_terrain_library.py)."""
    p = Path(library)
    return p.with_name(p.stem + ".tiles.npz")


def load_tiles(library: str | Path) -> list[np.ndarray] | None:
    """Tiles of a packed library, or None for a library without terrain."""
    p = tiles_path(library)
    if not str(library).endswith(".pt") or not p.exists():
        return None
    z = np.load(p)
    return [z[f"tile_{i}"] for i in range(len(z["names"]))]


def tile_origins(n: int) -> np.ndarray:
    """(n, 3) world origins of n tiles."""
    i = np.arange(n)
    return np.stack([TILE_X0 + TILE_SPACING * (i // TILE_COLS), TILE_SPACING * (i % TILE_COLS - (TILE_COLS - 1) / 2), np.zeros(n)], 1)


def world_extent(n: int) -> tuple[float, float, float, float]:
    """(x0, x1, y0, y1) covering the flat spawn area and all tiles, with margin."""
    o = tile_origins(max(n, 1))
    m = TILE_SPACING
    return min(-20.0, o[:, 0].min() - m), max(20.0, o[:, 0].max() + m), min(-20.0, o[:, 1].min() - m), max(20.0, o[:, 1].max() + m)


@dataclass(kw_only=True)
class ClipTilesTerrainCfg(SubTerrainCfg):
    """One sub-terrain holding the whole world: floor + every tile's boxes.

    Use with TerrainGeneratorCfg(num_rows=1, num_cols=1, size=world size); the generator places the
    patch corner at -size/2, so geometry is offset by +size/2 to keep the world origin fixed."""

    tiles: list[np.ndarray] = field(default_factory=list)  # per tile: (B, 6) centre xyz + half sizes
    color: tuple[float, float, float, float] = (0.55, 0.62, 0.72, 1.0)

    def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
        del difficulty, rng
        body = spec.body("terrain")
        sx, sy = self.size
        off = np.array([sx / 2, sy / 2, 0.0])  # patch-corner frame -> world-centred frame
        geoms = []
        floor = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=(sx / 2, sy / 2, FLOOR_THICKNESS / 2), pos=(sx / 2, sy / 2, -FLOOR_THICKNESS / 2))
        floor.rgba = (0.45, 0.47, 0.5, 1.0)
        geoms.append(TerrainGeometry(geom=floor, color=(0.45, 0.47, 0.5, 1.0)))
        for o, boxes in zip(tile_origins(len(self.tiles)), self.tiles):
            for b in np.asarray(boxes, dtype=float).reshape(-1, 6):
                g = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=tuple(b[3:]), pos=tuple(off + o + b[:3]), rgba=self.color)
                geoms.append(TerrainGeometry(geom=g, color=self.color))
        return TerrainOutput(origin=off.copy(), geometries=geoms)  # flat clips spawn at the world origin


def terrain_entity_cfg(tiles: list[np.ndarray] | None):
    """Scene terrain: the mjlab plane without tiles, else one generator patch holding floor + tiles."""
    from mjlab.terrains import TerrainEntityCfg
    from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

    if not tiles:
        return TerrainEntityCfg(terrain_type="plane")
    x0, x1, y0, y1 = world_extent(len(tiles))
    size = (2 * max(abs(x0), abs(x1)), 2 * max(abs(y0), abs(y1)))  # the generator centres the patch on the origin
    gen = TerrainGeneratorCfg(size=size, num_rows=1, num_cols=1, curriculum=False, color_scheme="none", seed=0,
                              sub_terrains={"clip_tiles": ClipTilesTerrainCfg(tiles=list(tiles))})
    return TerrainEntityCfg(terrain_type="generator", terrain_generator=gen, textures=(), materials=())
