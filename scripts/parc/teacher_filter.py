"""Keep the PARC retargets the terrain teacher can track: the physics QA lists (qa_parc.py) keep clips that
are infeasible for Dash in practice, and the benchmark's height/tilt rule passes a policy that walks past
a move it cannot do. A clip is dropped if the teacher failed it in the benchmark or in a standalone replay,
if its root got more than --max-xy from the reference horizontally, or if it is a vault or jump (most of
those are skipped rather than done).

  python scripts/parc/teacher_filter.py --bench logs/ab_eval/teacher_v4b_150k.json \
      --drift logs/ab_eval/teacher_v4b_150k_drift_parc.json --out scripts/parc/lists/teacher_keep_1m.txt

--bench: eval_groups.py output (``parc_failed_names``); --drift: per-clip standalone replays
(``fail_at``, ``xy_max``). Writes the kept clips in the QA-list format (path under the PARC root, .npz) for
build_parc_library.py --clips, and <out>.dropped.csv with each dropped clip's reason.
"""

from __future__ import annotations

import collections
import csv
import json
from pathlib import Path

import tyro

SKIP_TYPES = ("vaulting", "jumping")


def clip_type(name: str) -> str:
    """parc/<dir>/...: initial_aug/<type>/..., *_boxes, *_siggraph_terrain (the PARC teaser terrains)."""
    p = name.removeprefix("parc/").split("/")
    if p[0] == "initial_aug":
        return p[1]
    return "boxes" if p[0].endswith("_boxes") else "teaser" if p[0].endswith("_siggraph_terrain") else p[0]


def main(bench: Path, drift: Path, out: Path, max_xy: float = 1.0) -> None:
    failed = set(json.loads(bench.read_text())["parc_failed_names"])
    rep = json.loads(drift.read_text())
    keep, dropped = [], []
    for n, v in rep.items():
        t = clip_type(n)
        why = ("benchmark failure" if n in failed else "replay failure" if v["fail_at"] is not None else
               f"drift {v['xy_max']:.2f} m" if v["xy_max"] > max_xy else f"type {t}" if t in SKIP_TYPES else None)
        (dropped.append((n, t, why)) if why else keep.append(n))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(n.removeprefix("parc/") + ".npz\n" for n in sorted(keep)))
    with open(str(out) + ".dropped.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["clip", "type", "reason"])
        w.writerows(sorted(dropped))
    tot, kt = collections.Counter(clip_type(n) for n in rep), collections.Counter(clip_type(n) for n in keep)
    print(f"kept {len(keep)}/{len(rep)}: " + ", ".join(f"{t} {kt[t]}/{tot[t]}" for t in sorted(tot)))


if __name__ == "__main__":
    tyro.cli(main)
