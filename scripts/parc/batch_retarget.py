"""Retarget a list of PARC clips to Dash, one shard per process; resumable (skips finished clips).

  python batch_retarget.py list.txt out_dir part nparts

list.txt holds clip paths relative to PARC_DATA; each result goes to out_dir/<same path>.npz.
Failures are logged to out_dir/failed_<part>.txt and skipped.
"""
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parc_retarget import retarget

D = os.environ.get("PARC_DATA", "/media/shangqunyu/Dash/parc_src/parc_data/dec_release")  # huggingface.co/datasets/mxucg/PARC


def main():
    lst, out, part, nparts = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    clips = [l.strip() for l in open(lst) if l.strip()][part::nparts]
    t0, done = time.time(), 0
    for i, rel in enumerate(clips):
        dst = os.path.join(out, rel.replace(".pkl", ".npz"))
        if os.path.exists(dst):
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        try:
            r = retarget(os.path.join(D, rel))
            np.savez(dst + ".tmp.npz", source=rel, **r)
            os.replace(dst + ".tmp.npz", dst)
            done += 1
            print(f"[{part}] {i + 1}/{len(clips)} {rel}: {len(r['qpos'])} frames, IK rms med {np.median(r['ik_err']):.3f}, "
                  f"{(time.time() - t0) / done:.1f} s/clip", flush=True)
        except Exception:
            with open(os.path.join(out, f"failed_{part}.txt"), "a") as fh:
                fh.write(rel + "\n" + traceback.format_exc() + "\n")
            print(f"[{part}] FAILED {rel}", flush=True)


if __name__ == "__main__":
    main()
