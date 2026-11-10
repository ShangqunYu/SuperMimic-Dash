"""Scene terrain for terrain libraries: per env, a floor box and ``num_slots`` box slots.

Every env is its own MuJoCo world, so each env holds its own terrain: at reset MultiClipMotionCommand
writes the boxes of the sampled clip's terrain (supermimic/terrain/terrains.py) into that env's slots
and moves its floor to the terrain's lowest cell (per-world geom size, bounds and pose; static geom
poses live in mjData.geom_xpos, which the physics never recomputes). Unused slots are parked far below.
Env origins stay where mjlab puts them (all at the world origin with this one-patch terrain), and every
clip is stored in its terrain's frame. Flat clips (terrain -1) see the floor at z = 0 and no boxes.

Older libraries store box tiles (``<stem>.tiles.npz``: per tile, boxes on a z = 0 floor); they load
through terrains.load_terrains.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry, TerrainOutput

FLOOR_HALF = 30.0  # m; flat clips travel at most ~9 m from their start
FLOOR_THICKNESS = 1.0
WORLD_SIZE = 2 * FLOOR_HALF + 4.0
FLOOR_NAME = "terrain_floor"
SLOT_NAME = "terrain_slot_{}"


def tiles_path(library: str | Path) -> Path:
    """Sidecar of an older packed library holding its box tiles (build_terrain_library.py)."""
    p = Path(library)
    return p.with_name(p.stem + ".tiles.npz")


def load_tiles(library: str | Path) -> list[np.ndarray] | None:
    """Box tiles of an older packed library, or None."""
    p = tiles_path(library)
    if not str(library).endswith(".pt") or not p.exists():
        return None
    z = np.load(p)
    return [z[f"tile_{i}"] for i in range(len(z["names"]))]


@dataclass(kw_only=True)
class SlotTerrainCfg(SubTerrainCfg):
    """One sub-terrain: the floor (top at z = 0) and ``num_slots`` parked boxes, rewritten per env.

    Use with TerrainGeneratorCfg(num_rows=1, num_cols=1, size=(WORLD_SIZE, WORLD_SIZE)); the generator places
    the patch corner at -size/2, so geometry is offset by +size/2 to keep the world origin fixed."""

    num_slots: int = 1
    color: tuple[float, float, float, float] = (0.55, 0.62, 0.72, 1.0)

    def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
        del difficulty, rng
        from supermimic.terrain.terrains import PARK_Z

        body = spec.body("terrain")
        sx, sy = self.size
        off = np.array([sx / 2, sy / 2, 0.0])  # patch-corner frame -> world-centred frame
        floor_rgba = (0.45, 0.47, 0.5, 1.0)
        floor = body.add_geom(name=FLOOR_NAME, type=mujoco.mjtGeom.mjGEOM_BOX, size=(FLOOR_HALF, FLOOR_HALF, FLOOR_THICKNESS / 2),
                              pos=tuple(off + (0.0, 0.0, -FLOOR_THICKNESS / 2)), rgba=floor_rgba)
        geoms = [TerrainGeometry(geom=floor, color=floor_rgba)]
        for k in range(self.num_slots):
            g = body.add_geom(name=SLOT_NAME.format(k), type=mujoco.mjtGeom.mjGEOM_BOX, size=(0.01, 0.01, 0.01), pos=tuple(off + (0.0, 0.0, PARK_Z)), rgba=self.color)
            geoms.append(TerrainGeometry(geom=g, color=self.color))
        return TerrainOutput(origin=off.copy(), geometries=geoms)  # every env spawns at the world origin


def terrain_entity_cfg(num_slots: int | None):
    """Scene terrain: the mjlab plane for flat libraries, else floor + ``num_slots`` box slots."""
    from mjlab.terrains import TerrainEntityCfg
    from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

    if not num_slots:
        return TerrainEntityCfg(terrain_type="plane")
    gen = TerrainGeneratorCfg(size=(WORLD_SIZE, WORLD_SIZE), num_rows=1, num_cols=1, curriculum=False, color_scheme="none", seed=0,
                              sub_terrains={"slots": SlotTerrainCfg(num_slots=num_slots)})
    return TerrainEntityCfg(terrain_type="generator", terrain_generator=gen, textures=(), materials=())
