"""Load every raw .npy and npz; list files that fail to load or hold bad data."""
import multiprocessing as mp, os, sys
import numpy as np
D = "/media/simon/Crucial X10/dash_retarget"

def check(path):
    try:
        if path.endswith(".npy"):
            q = np.load(path, allow_pickle=True).item()["q"]
            ok = q.ndim == 2 and q.shape[1] == 25 and len(q) > 1 and np.isfinite(q).all()
        else:
            a = np.load(path)
            ok = a["fps"][0] == 50 and all(np.isfinite(a[k]).all() for k in a.files)
            n = len(a["joint_pos"]); ok = ok and n > 1 and all(len(a[k]) == n for k in a.files if k != "fps")
            ok = ok and np.abs(a["body_pos_w"][:, 0, 2]).max() < 3  # torso height sane
        return None if ok else path
    except Exception:
        return path

if __name__ == "__main__":
    files = [e.path for sub in ("raw", "npz") for e in os.scandir(f"{D}/{sub}")]
    with mp.Pool(12) as p:
        bad = [b for b in p.imap_unordered(check, files, chunksize=64) if b]
    print(len(files), "checked,", len(bad), "bad")
    open(sys.argv[1], "w").write("\n".join(sorted(bad)))
