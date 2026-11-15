"""Depth student alone in physics on rollout_generator's 15 scenes: joystick + proprioception + forward depth in,
actions out, no generator or teacher. Scored as the generator + teacher loop (rollout_generator.score), so the
numbers compare with closed_loop.py's (gen_v6 + v6 teacher: 310 / 360 clean, 294-296 with DR + delay).

  python scripts/student/eval_student.py student_ckpt out_dir [--samples 8] [--seconds 12] [--seed 0] [--clean] [--render]

Default conditions are the student's training ones: observation noise (with the depth's noise, dropouts and
latency), robot + hardware + camera DR and the 0-25 ms actuator delay; --clean turns all of that off.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/generator"))
from closed_loop import build_library, render_all  # noqa: E402
from rollout_generator import DX, X0, Y0, operator, scenes, score  # noqa: E402

from supermimic.envs import SuperMimicEnvCfgArgs, make_env  # noqa: E402
from supermimic.generator.data import GEN_GRID  # noqa: E402
from supermimic.generator.metrics import FootMetrics  # noqa: E402
from supermimic.student.model import Student, StudentRunner, load_student  # noqa: E402
from supermimic.terrain.height_scan import HeightScan  # noqa: E402
from supermimic.terrain.terrains import Terrain, load_terrains, terrains_path  # noqa: E402

def student_env(lib: Path, n: int, seed: int, noisy: bool, env_kw: dict, dev: str):
    """Env on a closed_loop-style library (one standing clip per env), depth on, terminations off; env_kw: the
    teacher's goal layout and contacts (goal_env_kwargs; the student ignores the goal, the teacher reads it); noisy:
    the student's training conditions."""
    load_terrains.cache_clear()  # cached by path: a rebuilt library must not reuse an earlier call's terrains
    env = make_env(SuperMimicEnvCfgArgs(npz_files=[str(lib)], robot="dash", num_envs=n, compact_library=True, **env_kw, episode_length_s=1e5,
                                        action_noise_std=0.0, obs_noise=noisy, reset_noise=False, domain_randomization=noisy, hardware_dr=noisy,
                                        command_delay=noisy, depth=True, sampling_mode="start", seed=seed, root_height_threshold=1e3,
                                        root_tilt_threshold=1e3, record_transitions=False), device=dev)
    cmd = env.command_manager.get_term("motion")
    cmd.forced_frame[:] = cmd.library.clip_start[torch.arange(n, device=dev)]
    obs, _ = env.reset()
    cmd.forced_frame[:] = -1
    return env, cmd, obs


def robot_frame(cmd, origins) -> torch.Tensor:
    """(n, 25) [root pos (env frame), quat, joints]."""
    return torch.cat([cmd.robot_root_pos_w - origins, cmd.robot_root_quat_w, cmd.robot_joint_pos], -1)


@torch.no_grad()
def evaluate(student: Student, out: Path, start: torch.Tensor, env_kw: dict, samples: int = 8, seconds: float = 12.0, seed: int = 0,
             noisy: bool = True, render: bool = False, dev: str = "cuda:0") -> tuple[dict, int, int]:
    """start (25,): the standing frame (root at x = y = 0, yaw 0); env_kw: student_env's. -> per-scene results,
    successes, runs."""
    out.mkdir(parents=True, exist_ok=True)
    sc = scenes()
    terr = [Terrain(s.hf, np.array([X0, Y0]), DX) for s in sc]
    S = samples
    n = len(sc) * S
    tiles = torch.arange(len(sc), device=dev).repeat_interleave(S)
    st = start.repeat(n, 1)
    st[:, 2] += torch.as_tensor(np.array([t.h(np.zeros(2)) for t in terr]), dtype=torch.float32, device=dev).repeat_interleave(S)
    T = int(seconds * 50)
    lib = out / f"eval_library_{seed}.pt"
    build_library(lib, terr, tiles.tolist(), st, T + 200)
    env, cmd, obs = student_env(lib, n, seed, noisy, env_kw, dev)
    cmd_v = torch.as_tensor(np.array([s.cmd for s in sc]), dtype=torch.float32, device=dev).repeat_interleave(S, 0)
    runner = StudentRunner(student, n, dev)
    origins = env.scene.env_origins
    buf = robot_frame(cmd, origins)[:, None]
    Q = []
    for _ in range(T):
        joy = operator(buf, cmd_v)
        obs, *_ = env.step(runner.act(obs["proprio"], joy, obs["depth"]))
        now = robot_frame(cmd, origins)
        buf = now[:, None]
        Q.append(now)
    env.close()
    for f in (lib, terrains_path(lib)):
        f.unlink(missing_ok=True)
    Q = torch.stack(Q, 1)
    scan = HeightScan(SimpleNamespace(terrains=terr), GEN_GRID).to(dev)
    with contextlib.redirect_stdout(io.StringIO()):  # score prints every scene's details; results.json has them
        res, total = score(sc, terr, Q, tiles, FootMetrics(scan, dev), S)
    torch.save({"robot": Q.cpu(), "scenes": [s.name for s in sc], "samples": S}, out / "traj.pt")
    if render:
        hidden = Q[::S].clone()
        hidden[..., 2] -= 50.0  # no reference here: the renderer's ghost goes far below the floor
        render_all(sc, terr, Q[::S].cpu().numpy(), hidden.cpu().numpy(), out)
    return res, total, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("student")
    ap.add_argument("out")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clean", action="store_true", help="no noise, latency, DR or delay")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    ck = torch.load(a.student, map_location="cpu", weights_only=False)
    student = load_student(a.student, a.device)
    res, total, n = evaluate(student, Path(a.out), ck["start"].to(a.device), ck["env_kwargs"], a.samples, a.seconds, a.seed, not a.clean, a.render, a.device)
    print(f"success {total}/{n}")
    (Path(a.out) / "results.json").write_text(json.dumps({"student": a.student, "noisy": not a.clean, "seed": a.seed, "success": f"{total}/{n}", "scenes": res}, indent=1))


if __name__ == "__main__":
    main()
