"""Batch QA for q-only Dash retargets: one CSV row per clip.

Usage: python batch_qa.py <processed_xml> <raw_dir> <clip_list.csv> <out.csv>
"""
import csv
import multiprocessing as mp
import os
import sys

import mujoco
import numpy as np

XML, RAW, LIST, OUT = sys.argv[1:5]
_m = _d = None


def init():
  global _m, _d
  _m = mujoco.MjModel.from_xml_path(XML)
  _d = mujoco.MjData(_m)


def qa(row):
  m, d = _m, _d
  name = os.path.splitext(os.path.basename(row["move_soma_uniform_path"]))[0]
  out = {"clip": name, "body_position": row["content_body_position"]}
  try:
    D = np.load(os.path.join(RAW, f"dash_{name}.npy"), allow_pickle=True).item()
  except Exception as e:
    return {**out, "error": f"load: {e}"}
  q, fps = D["q"], D["fps"]
  if not np.isfinite(q).all():
    return {**out, "error": "nan"}
  feet = {s: [m.body(f"{s}_foot_heel").id, m.body(f"{s}_foot_toe").id] for s in "lr"}
  lo, hi = m.jnt_range[1:, 0], m.jnt_range[1:, 1]
  FP = {s: np.empty((len(q), 2, 3)) for s in "lr"}
  body_minz = np.empty(len(q))
  for t, x in enumerate(q):
    d.qpos[:] = np.r_[x[:3], x[[6, 3, 4, 5]], x[7:]]
    mujoco.mj_kinematics(m, d)
    for s in "lr":
      FP[s][t] = d.xpos[feet[s]]
    body_minz[t] = d.xpos[2:, 2].min()  # skip world and floor-free root origin
  j = q[:, 7:]
  near = (j < lo + 0.02) | (j > hi - 0.02)
  qd = np.abs(np.diff(j, axis=0)) * fps
  jumps = int((qd > np.maximum(4 * np.percentile(qd, 99, axis=0), 6)).any(1).sum())
  foot_minz, sl = 1e9, []
  for s in "lr":
    P = FP[s]
    foot_minz = min(foot_minz, P[:, :, 2].min())
    down = P[:, :, 2].max(1) < 0.01
    v = np.linalg.norm(np.diff(P[:, :, :2].mean(1), axis=0), axis=1) * fps
    sl.append(v[down[1:] & down[:-1]])
  sl = np.concatenate(sl) * 100
  return {
    **out,
    "seconds": round(len(q) / fps, 2),
    "at_limit_pct": round(near.any(1).mean() * 100, 1),
    "worst_limit_joint_pct": round(near.mean(0).max() * 100, 1),
    "max_qd": round(float(qd.max()), 1),
    "jumps": jumps,
    "foot_minz_mm": round(foot_minz * 1000, 1),
    "body_minz_mm": round(float(body_minz.min()) * 1000, 1),
    "root_z_min": round(float(q[:, 2].min()), 3),
    "slide_p50": round(float(np.percentile(sl, 50)), 2) if len(sl) else "",
    "slide_p95": round(float(np.percentile(sl, 95)), 2) if len(sl) else "",
    "error": "",
  }


if __name__ == "__main__":
  rows = list(csv.DictReader(open(LIST)))
  with mp.Pool(12, initializer=init) as pool:
    res = pool.map(qa, rows, chunksize=16)
  keys = list(next(r for r in res if not r["error"]).keys())
  with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    w.writerows(res)
  print(f"wrote {len(res)} rows, {sum(bool(r['error']) for r in res)} errors")
