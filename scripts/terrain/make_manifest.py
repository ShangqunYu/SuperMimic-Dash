"""Terrain clips, step 3: QA the edited box/stairs clips and write the manifest build_terrain_library.py reads.

Pass (scores from terrain_edit.py, ``after_*`` columns):
  * foot penetration <= 4 cm, shin penetration <= 3 cm, stance IK error <= 3 cm;
  * < 80% sole support on <= 25% of standing frames;
  * no edit-made joint jumps: ``speed_excess`` <= 0 (each joint's peak speed within 2x its peak in
    the source + 5 rad/s); skipped for score files written before the check existed;
Stair jogging is left out (too few footholds for a clean staircase fit).
Jump clips (jump_pilot.py: box placed under the unedited motion) are copied from ``jumps_from``.

  uv run python scripts/terrain/make_manifest.py --scores logs/terrain_edit_v2 --out data/terrain/manifest_v2.csv
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import tyro

FIELDS = ["clip", "group", "pass", "file", "terrain", "seconds"]


def passes(r: dict) -> bool:
    f = lambda k: float(r[f"after_{k}"])
    ok = f("foot_pen_max") <= 0.04 and f("shin_pen_max") <= 0.03 and f("ik_resid_stance_max") <= 0.03 and f("support_lt80") <= 0.25
    if r.get("after_speed_excess", "") != "":
        ok = ok and f("speed_excess") <= 0.0
    return ok


def main(scores: Path = Path("logs/terrain_edit_v2"), edited: Path | None = None, meta: Path = Path("data/terrain/meta.csv"),
         jumps_from: Path = Path("data/terrain/manifest.csv"), out: Path = Path("data/terrain/manifest_v2.csv")) -> None:
    edited = edited or Path(str(scores).replace("logs/terrain_edit", "data/terrain/edited"))
    group = {r["name"]: r["content"] for r in csv.DictReader(open(meta))}
    rows, seen = [], set()
    for sf in sorted(scores.glob("scores_*.csv")):
        for r in csv.DictReader(open(sf)):
            if r["kind"] == "failed" or r["clip"] in seen or not r.get("after_ik_resid_stance_max") or "jog" in r["clip"]:
                continue
            seen.add(r["clip"])
            f = edited / f"dash_{r['clip']}.npz"
            z = np.load(f)
            rows.append({"clip": r["clip"], "group": group.get(r["clip"], ""), "pass": int(passes(r)), "file": str(f), "terrain": r["terrain"],
                         "seconds": f"{len(z['qpos']) / float(z['fps']):.2f}"})
    jumps = [r for r in csv.DictReader(open(jumps_from)) if "edited_jumps" in r["file"]]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows + jumps)
    n_pass = sum(r["pass"] for r in rows)
    print(f"wrote {out}: {len(rows)} edited clips ({n_pass} pass) + {len(jumps)} jump clips ({sum(r['pass'] == '1' for r in jumps)} pass)")


if __name__ == "__main__":
    tyro.cli(main)
