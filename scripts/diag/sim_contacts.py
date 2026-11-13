"""Inside the GPU simulator: foot-terrain penetration, contacts per foot (floor vs terrain boxes), constraint counts
and overflow flags per env, for a policy playing the given library clips (nominal, terminations off).

  python scripts/diag/sim_contacts.py ckpt clip [clip ...] [--lib PATH]

Contacts as the checkpoint was trained (env.contact_timeconst). Env SOLREF=<timeconst> (passed as
contact_timeconst) and DMIN/DMAX=<solimp> override every geom before the sim is built; SPEC=7 sets
mjENBL_SPECULATIVE (MuJoCo 4; MuJoCo Warp 4.0.1 rejects it). Deepest contacts (> 2.5 cm) are listed with
their normal (from geom[0] to geom[1]).
"""

import collections
import os
import sys

import mjlab.sim.sim as mjsim
import numpy as np
import torch

from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.eval.benchmark import goal_env_kwargs, load_policy_from_checkpoint

_apply = mjsim.MujocoCfg.apply


def _apply_overrides(self, model):
    _apply(self, model)
    if os.environ.get("DMAX"):
        model.geom_solimp[:, 0] = float(os.environ.get("DMIN", os.environ["DMAX"]))
        model.geom_solimp[:, 1] = float(os.environ["DMAX"])
    if os.environ.get("SPEC"):
        model.opt.enableflags |= 1 << int(os.environ["SPEC"])


mjsim.MujocoCfg.apply = _apply_overrides


def main():
    args = sys.argv[1:]
    lib = "data/library/dash_flat_terrain_v4_parc_compact.pt"
    if "--lib" in args:
        k = args.index("--lib")
        lib = args[k + 1]
        del args[k : k + 2]
    ckpt, clips = args[0], args[1:]
    dev = "cuda:0"
    policy, pcfg = load_policy_from_checkpoint(ckpt, dev)
    kw = goal_env_kwargs(pcfg)
    if os.environ.get("SOLREF"):
        kw["contact_timeconst"] = float(os.environ["SOLREF"])
    env = make_env(SuperMimicEnvCfgArgs(npz_files=[lib], robot="dash", num_envs=len(clips), compact_library=True, **kw,
                                        episode_length_s=1e5, action_noise_std=0.0, obs_noise=False, reset_noise=False, domain_randomization=False,
                                        sampling_mode="start", seed=0, root_height_threshold=1e3, root_tilt_threshold=1e3, record_transitions=False),
                   device=dev)
    cmd = env.command_manager.get_term("motion")
    L = cmd.library
    idx = [next(i for i, c in enumerate(L.clips) if c.name == n) for n in clips]
    cmd.forced_frame[:] = L.clip_start[torch.tensor(idx, device=dev)]
    obs, _ = env.reset()
    cmd.forced_frame[:] = -1
    mj = env.sim.mj_model
    names = [mj.geom(g).name for g in range(mj.ngeom)]
    ter = np.array(["terrain" in n for n in names])
    foot = np.array(["foot_collision" in n for n in names])
    wd = env.sim.wp_data
    print("njmax", wd.njmax, "naconmax", wd.naconmax, "foot geom solref", mj.geom_solref[foot][0], "solimp", mj.geom_solimp[foot][0], "margin", mj.geom_margin[foot][0])
    lens = [int(L.clips[i].length) for i in idx]
    stats = {e: {"pen": [], "nefc": [], "ncon": [], "ovf": 0, "deep": [], "floor": [], "slot": []} for e in range(len(clips))}
    with torch.no_grad():
        for t in range(max(lens) - 1):
            obs, _, _, _, _ = env.step(policy(obs["policy"], obs["goal"]))
            n = int(wd.nacon.numpy()[0])
            dist, geom, wid = wd.contact.dist.numpy()[:n], wd.contact.geom.numpy()[:n], wd.contact.worldid.numpy()[:n]
            nefc, ovf = wd.nefc.numpy(), wd.overflow.numpy()
            for e in range(len(clips)):
                if t >= lens[e] - 2:
                    continue
                sel = wid == e
                g, dd = geom[sel], dist[sel]
                ft = (ter[g[:, 0]] & foot[g[:, 1]]) | (ter[g[:, 1]] & foot[g[:, 0]])
                stats[e]["pen"].append(float(-dd[ft].min()) if ft.any() else 0.0)
                for side in ("l_foot", "r_foot"):
                    for kind in ("floor", "slot"):
                        mk = np.array([(side in names[a] and kind in names[b]) or (side in names[b] and kind in names[a]) for a, b in g], dtype=bool)
                        if mk.any():
                            stats[e][kind].append((int(mk.sum()), float(-dd[mk].min())))
                if ft.any() and -dd[ft].min() > 0.025:
                    k = np.flatnonzero(sel)[ft][np.argmin(dd[ft])]
                    normal = wd.contact.frame.numpy()[k][0].round(2).tolist()
                    stats[e]["deep"].append((t / 50, float(-dd[ft].min()), normal, names[geom[k][0]], names[geom[k][1]]))
                stats[e]["nefc"].append(int(nefc[e]))
                stats[e]["ncon"].append(int(sel.sum()))
                stats[e]["ovf"] |= int(ovf[e])
            wd.overflow.zero_()
    env.close()
    for e, c in enumerate(clips):
        s = stats[e]
        p = np.array(s["pen"])
        print(f"{c.split('/')[-1]}: foot-terrain penetration max {p.max() * 100:.1f} cm, p99 {np.percentile(p, 99) * 100:.1f} cm, steps > 2 cm "
              f"{(p > 0.02).sum()}/{len(p)}; nefc max {max(s['nefc'])} (njmax {wd.njmax}); contacts max {max(s['ncon'])}; overflow flags {s['ovf']:#x}")
        for kind in ("floor", "slot"):
            if s[kind]:
                a = np.array(s[kind])
                counts = dict(sorted(collections.Counter(a[:, 0].astype(int).tolist()).items()))
                print(f"   on {kind}: foot-steps {len(a)}, contacts per foot {counts}, depth p50 {np.median(a[:, 1]) * 100:.2f} "
                      f"p90 {np.percentile(a[:, 1], 90) * 100:.2f} max {a[:, 1].max() * 100:.2f} cm")
        for x in s["deep"][:8]:
            print(f"   t={x[0]:.2f}s depth {x[1] * 100:.1f} cm normal {x[2]} {x[3]} / {x[4]}")


if __name__ == "__main__":
    main()
