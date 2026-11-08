"""Per PARC clip: duration, stance-height changes the feet make, hand-on-terrain, flight time. Human scale."""
import csv, glob, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parc_io import fk, load, qrot, skeleton
from parc_retarget import HEEL, TOE, Terrain
D = os.environ.get("PARC_DATA", "/media/shangqunyu/Dash/parc_src/parc_data/dec_release")  # huggingface.co/datasets/mxucg/PARC
sk = skeleton(); names = sk[0]; ix = {n: i for i, n in enumerate(names)}
files = sorted(glob.glob(f"{D}/**/*.pkl", recursive=True))
part, nparts = int(sys.argv[1]), int(sys.argv[2])
rows = []
for f in files[part::nparts]:
    try:
        m, t = load(f)
    except Exception:
        continue
    if t is None or "root_pos" not in m:
        continue
    _, pos, rot = fk(m, sk); T = len(pos); fps = m["fps"]
    terr = Terrain(t["hf"], t["min_point"], t["dx"])
    stance_h, air = [], np.ones(T, bool)
    for hs in ("left", "right"):
        k = ix[f"{hs}_foot"]
        sole = [pos[:, k] + qrot(rot[:, k], np.broadcast_to(p, (T, 3))) for p in (HEEL, TOE)]
        gap = np.minimum(*[s[:, 2] - terr.h(s[:, :2]) for s in sole])
        v = np.r_[0, np.linalg.norm(np.diff(pos[:, k], axis=0), axis=1) * fps]
        st = (gap < 0.03) & (v < 0.5)
        air &= gap > 0.03
        stance_h.append(np.where(st, np.minimum(sole[0][:, 2], sole[1][:, 2]), np.nan))
    sh = np.nanmin(np.stack(stance_h), 0)  # lowest stance foot height per frame
    lv = sh[~np.isnan(sh)]
    jumps = np.abs(np.diff(lv)) if len(lv) > 1 else np.zeros(1)
    hand = 0.0
    for hs in ("left", "right"):
        hp = pos[:, ix[f"{hs}_hand"]]
        hand = max(hand, float(np.mean(hp[:, 2] - terr.h(hp[:, :2], 0.05) < 0.06)))
    # longest flight
    best = cur = 0
    for a in air:
        cur = cur + 1 if a else 0; best = max(best, cur)
    rows.append(dict(file=os.path.relpath(f, D), subset=os.path.relpath(f, D).split("/")[0], seconds=T / fps,
                     stance_rise=float(jumps.max()), level_range=float(lv.max() - lv.min()) if len(lv) else 0.0,
                     hand_frac=hand, flight=best / fps, root_drop=float(pos[:, 0, 2].max() - pos[:, 0, 2].min())))
with open(f"survey_{part}.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(part, len(rows))
