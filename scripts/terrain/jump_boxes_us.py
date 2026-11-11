"""Box jump-over variants: the obstacle and running jumps of ``jump_pilot.py`` (motion unedited) over boxes
of several heights, depths and widths, wherever the motion clears the box by ``margin``.

Only these clips clear a 27-32 cm box (screened 2026-11-11): the 7 no-hands obstacle jumps and 3 of the 4
running obstacle jumps (mirrors included); forward broad jumps (91) clear < 27 cm.

  uv run python scripts/terrain/jump_boxes_us.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import mujoco
import numpy as np
import tyro

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jump_pilot import clearance_profile, place_box  # noqa: E402
from pilot_terrain import track  # noqa: E402

from supermimic.robots import get_robot  # noqa: E402


def main(src: Path = Path("data/terrain/edited_jumps"), save: Path = Path("data/terrain/edited_jumps_us"), out: Path = Path("data/terrain/jumps_us.csv"),
         heights: tuple[float, ...] = (0.25, 0.28, 0.31), depths: tuple[float, ...] = (0.15, 0.25, 0.35), widths: tuple[float, ...] = (0.4, 1.0),
         margin: float = 0.02, patterns: tuple[str, ...] = ("dash_jump_over_obstacle*", "dash_avoid_obstacle_jump_run*")) -> None:
    m = mujoco.MjModel.from_xml_path(str(get_robot("dash").xml_path()))
    save.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in sorted(p for pat in patterns for p in src.glob(pat + ".npz")):
        z = np.load(f)
        qpos = z["qpos"]
        fe, sp, fr, sr = track(m, qpos)
        d, grid, clear = clearance_profile(qpos, fe, sp, fr, sr)
        n = 0
        for depth in depths:
            s_lo, s_hi, c = place_box(grid, clear, depth)
            for h in heights:
                if c - margin < h:
                    continue
                for w in widths:
                    name = f"{f.stem}_h{round(h * 100)}_d{round(depth * 100)}_w{round(w * 100)}"
                    np.savez(save / f"{name}.npz", qpos=qpos, fps=float(z["fps"]), kind=str(z["kind"]), c=qpos[0, :2], d=d, box_s=np.array([s_lo, s_hi]),
                             box_height=h, box_width=w, clear=c)
                    rows.append({"clip": name.removeprefix("dash_"), "group": "jump_obstacle_us", "pass": 1, "file": str(save / f"{name}.npz"),
                                 "terrain": f"box {h * 100:.0f} cm high, {depth * 100:.0f} cm deep, {w * 100:.0f} cm wide", "seconds": f"{len(qpos) / float(z['fps']):.2f}"})
                    n += 1
        print(f"{f.stem:55s} clear over 15/25/35 cm: {', '.join(f'{place_box(grid, clear, x)[2] * 100:.0f}' for x in depths)} cm -> {n} variants")
    with open(out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"wrote {out}: {len(rows)} clips")


if __name__ == "__main__":
    tyro.cli(main)
