"""Benchmark evaluation (paper Appendix F.1 / plan 5.1).

Every clip is played once from its first frame with the deterministic policy:
no observation/action noise, no reset noise, no domain randomization (foot friction 1.0),
deterministic physics. A clip *fails* if the training termination (root height /
tilt vs reference) fires before the clip ends. Normalised return = sum of per-step
rewards / (max step reward * number of steps).

The paper's rule never checks where the robot is horizontally, so a policy that skips a move (walks past
a vault) still passes; each result also records its largest root horizontal distance from the reference,
and ``summarize`` counts the clips that failed or drifted farther than ``drift_m``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.policy.nets import LEGACY_POLICY_CFG, Policy, PolicyCfg


@dataclass
class ClipResult:
    name: str
    frames: int
    steps: int
    norm_return: float
    failed: bool
    fail_step: int  # -1 if not failed
    terms: dict | None = None  # per-term mean reward per step (dt-scaled)
    max_xy_err: float = 0.0  # largest root horizontal distance from the reference over the episode (m)


def load_policy_from_checkpoint(path: str | Path, device: str) -> tuple[Policy, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    pc = {**LEGACY_POLICY_CFG, **ck["config"]["policy"]}
    pcfg = PolicyCfg(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in pc.items()})
    policy = Policy(pcfg).to(device)
    policy.load_state_dict(ck["policy"])
    policy.eval()
    return policy, ck["config"]


def goal_env_kwargs(config: dict) -> dict:
    """SuperMimicEnvCfgArgs fields that set a checkpoint's goal layout, from its saved config (older
    checkpoints lack the newer fields: consecutive goal frames, no root orientation, no terrain scan)."""
    e = config["env"]
    return dict(future_frames=e["future_frames"], goal_stride=e.get("goal_stride", 1), goal_root_ori=e.get("goal_root_ori", False),
                anchor_obs=e.get("anchor_obs", False), policy_scan=e.get("policy_scan", False))


@torch.no_grad()
def run_benchmark(policy: Policy, npz_files: list[str], device: str = "cuda:0", num_envs: int | None = None, future_frames: int = 10, verbose: bool = True, anchor_obs: bool = False, robot: str = "g1", num_clips: int | None = None, randomize: bool = False, goal_stride: int = 1, goal_root_ori: bool = False, policy_scan: bool = False, clip_ids: list[int] | None = None) -> list[ClipResult]:
    """``npz_files`` may be one packed .pt library; then ``num_clips`` (its clip count) sizes the env batch.
    ``randomize`` turns on the training randomization (observation noise, sim and hardware DR; same seed for every policy).
    ``clip_ids``: evaluate only these library clips (results in that order)."""
    packed = len(npz_files) == 1 and str(npz_files[0]).endswith(".pt")
    n_total = len(clip_ids) if clip_ids is not None else (num_clips if packed and num_clips else len(npz_files))
    num_envs = min(num_envs or 1024, n_total)
    args = SuperMimicEnvCfgArgs(
        npz_files=npz_files,
        robot=robot,
        num_envs=num_envs,
        future_frames=future_frames,
        episode_length_s=1.0e5,  # clip end decides
        action_noise_std=0.0,
        obs_noise=randomize,
        reset_noise=False,
        domain_randomization=randomize,
        hardware_dr=randomize,
        sampling_mode="start",
        record_transitions=False,
        seed=0,
        anchor_obs=anchor_obs,
        goal_stride=goal_stride,
        goal_root_ori=goal_root_ori,
        policy_scan=policy_scan,
        compact_library=packed,
    )
    env = make_env(args, device=device)
    try:
        cmd = env.command_manager.get_term("motion")
        lib = cmd.library
        queue = torch.as_tensor(clip_ids if clip_ids is not None else range(lib.num_clips), dtype=torch.long, device=device)
        n_clips = int(queue.numel())
        clip_at = lambda pos: queue[torch.clamp(pos, max=n_clips - 1)]  # noqa: E731
        max_r = 0.1  # paper reward upper bound per step (dt * 5)

        # Clip queue: env e plays queue entries e, e+N, e+2N, ...
        current = torch.arange(num_envs, device=device)
        nxt = current + num_envs
        cmd.forced_frame[:] = lib.clip_start[clip_at(current)]
        obs, _ = env.reset()
        cmd.forced_frame[:] = lib.clip_start[clip_at(nxt)]
        active = torch.ones(num_envs, dtype=torch.bool, device=device)
        ret_sum = torch.zeros(num_envs, device=device)
        steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        term_names = list(env.reward_manager.active_terms)
        term_sum = torch.zeros(num_envs, len(term_names), device=device)
        xy_max = torch.zeros(num_envs, device=device)
        results: dict[int, ClipResult] = {}
        remaining = n_clips
        while remaining > 0:
            act = policy(obs["policy"], obs["goal"])
            obs, rew, term, tout, _ = env.step(act)
            ret_sum += rew
            term_sum += env.reward_manager.term_rewards
            steps += 1
            # done envs were reset inside step: their state is the next clip's
            xy = torch.linalg.vector_norm(cmd.root_pos_w[:, :2] - cmd.robot_root_pos_w[:, :2], dim=-1)
            xy_max = torch.where(term | tout, xy_max, torch.maximum(xy_max, xy))
            done = (term | tout) & active
            if done.any():
                ids = done.nonzero().squeeze(-1)
                for e in ids.tolist():
                    ci = int(current[e])
                    if ci < n_clips and ci not in results:
                        st = int(steps[e])
                        li = int(queue[ci])
                        results[ci] = ClipResult(
                            name=lib.clips[li].name,
                            frames=lib.clips[li].length,
                            steps=st,
                            norm_return=float(ret_sum[e]) / (max_r * max(st, 1)),
                            failed=bool(term[e]),
                            fail_step=st if bool(term[e]) else -1,
                            terms={n: float(term_sum[e, i]) / max(st, 1) for i, n in enumerate(term_names)},
                            max_xy_err=float(xy_max[e]),
                        )
                        remaining -= 1
                        if verbose and (len(results) % 50 == 0 or len(results) == n_clips):
                            print(f"  evaluated {len(results)}/{n_clips}")
                # advance the queue for these envs (the auto-reset already used forced_frame)
                current[ids] = nxt[ids]
                nxt[ids] = nxt[ids] + num_envs
                finished = current[ids] >= n_clips
                active[ids[finished]] = False
                cmd.forced_frame[ids] = lib.clip_start[clip_at(nxt[ids])]
                ret_sum[ids] = 0.0
                term_sum[ids] = 0.0
                xy_max[ids] = 0.0
                steps[ids] = 0
        return [results[i] for i in range(n_clips)]
    finally:
        env.close()


def summarize(results: list[ClipResult], drift_m: float = 1.0) -> dict:
    """``failed_or_drift``: clips that failed or whose root got more than ``drift_m`` from the reference horizontally."""
    n = len(results)
    failed = sum(r.failed for r in results)
    mean_ret = sum(r.norm_return for r in results) / max(n, 1)
    # Length-weighted return (paper Fig. 3a uses the benchmark aggregate; weighting by steps).
    tot_steps = sum(r.steps for r in results)
    w_ret = sum(r.norm_return * r.steps for r in results) / max(tot_steps, 1)
    out = {"clips": n, "failed": failed, "failed_or_drift": sum(r.failed or r.max_xy_err > drift_m for r in results), "drift_m": drift_m,
           "mean_norm_return": mean_ret, "step_weighted_norm_return": w_ret}
    if results and results[0].terms:
        out["terms"] = {k: sum(r.terms[k] * r.steps for r in results) / max(tot_steps, 1) for k in results[0].terms}
    return out


def save_results(results: list[ClipResult], path: str | Path) -> None:
    Path(path).write_text(json.dumps({"summary": summarize(results), "clips": [asdict(r) for r in results]}, indent=1))
