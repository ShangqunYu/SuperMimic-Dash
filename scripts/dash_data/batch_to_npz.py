"""Batched mjlab.scripts.csv_to_npz for Dash: many clips per GPU forward pass.

Same resampling (lerp/slerp onto the output clock), velocities (torch.gradient,
SO(3) central difference) and forward kinematics as csv_to_npz, but each env
replays a different clip. Reads q-only mocap_retargeting .npy files.

Usage: uv run python batch_to_npz.py <raw_dir> <clip_names.txt> <out_dir>
       [--batch 256] [--limit N]
"""

import argparse
import os

import numpy as np
import torch

from mjlab.scene import Scene
from mjlab.scripts.csv_to_npz import DASH_JOINT_NAMES
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.tasks.tracking.config.dash.env_cfgs import dash_flat_tracking_env_cfg
from mjlab.utils.lab_api.math import axis_angle_from_quat, quat_conjugate, quat_mul

OUT_FPS = 50.0
FIELDS = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w")
FIELDS += ("body_lin_vel_w", "body_ang_vel_w")


def slerp(a: torch.Tensor, b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
  """Batched lab_api quat_slerp: shortest path, q1 when nearly parallel."""
  d = (a * b).sum(-1, keepdim=True)
  b = torch.where(d < 0, -b, b)
  d = d.abs()
  angle = torch.acos(d.clamp(-1, 1))
  eps = torch.finfo(a.dtype).eps * 4.0
  same = (torch.abs(d - 1.0) < eps) | (angle < eps)
  s = torch.sin(angle).clamp_min(eps)
  t = t[:, None]
  out = (a * torch.sin((1 - t) * angle) + b * torch.sin(t * angle)) / s
  out = torch.where(same, a, out)
  out = torch.where(t == 0, a, out)
  return torch.where(t == 1, b, out)


def resample(path: str) -> dict[str, torch.Tensor]:
  """csv_to_npz.MotionLoader on a mocap_retargeting q array, vectorized."""
  d = np.load(path, allow_pickle=True).item()
  q = torch.from_numpy(np.asarray(d["q"])).to(torch.float32)
  in_fps = float(d["fps"])
  pos, rot, dof = q[:, :3], q[:, [6, 3, 4, 5]], q[:, 7:]  # xyzw -> wxyz
  rot = rot / torch.linalg.norm(rot, dim=1, keepdim=True)
  n = q.shape[0]
  duration = (n - 1) / in_fps
  dt = 1.0 / OUT_FPS
  times = torch.arange(0, duration, dt, dtype=torch.float32)
  phase = times / duration
  i0 = (phase * (n - 1)).floor().long()
  i1 = torch.minimum(i0 + 1, torch.tensor(n - 1))
  w = phase * (n - 1) - i0
  pos = pos[i0] * (1 - w[:, None]) + pos[i1] * w[:, None]
  dof = dof[i0] * (1 - w[:, None]) + dof[i1] * w[:, None]
  rot = slerp(rot[i0], rot[i1], w)
  q_rel = quat_mul(rot[2:], quat_conjugate(rot[:-2]))
  omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
  return {
    "pos": pos,
    "rot": rot,
    "dof": dof,
    "lin_vel": torch.gradient(pos, spacing=dt, dim=0)[0],
    "ang_vel": torch.cat([omega[:1], omega, omega[-1:]]),
    "dof_vel": torch.gradient(dof, spacing=dt, dim=0)[0],
  }


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("raw_dir")
  p.add_argument("clip_list")
  p.add_argument("out_dir")
  p.add_argument("--batch", type=int, default=256)
  p.add_argument("--limit", type=int, default=0)
  args = p.parse_args()
  os.makedirs(args.out_dir, exist_ok=True)

  names = [ln.strip() for ln in open(args.clip_list) if ln.strip()]
  if args.limit:
    names = names[: args.limit]
  # One directory scan each: per-file lookups are slow on exFAT with ~70k files.
  done = set(os.listdir(args.out_dir))
  names = [n for n in names if f"dash_{n}.npz" not in done]
  raw_sizes = {e.name: e.stat().st_size for e in os.scandir(args.raw_dir)}
  # Longest first so each batch pads little.
  sizes = {n: raw_sizes[f"dash_{n}.npy"] for n in names}
  names.sort(key=sizes.__getitem__, reverse=True)
  print(f"{len(names)} clips to convert")

  B = min(args.batch, len(names))
  cfg = dash_flat_tracking_env_cfg().scene
  cfg.num_envs = B
  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = 1.0 / OUT_FPS
  scene = Scene(cfg, device="cuda:0")
  sim = Simulation(num_envs=B, cfg=sim_cfg, model=scene.compile(), device="cuda:0")
  scene.initialize(sim.mj_model, sim.model, sim.data)
  robot = scene["robot"]
  jidx = robot.find_joints(DASH_JOINT_NAMES, preserve_order=True)[0]
  origins = scene.env_origins.clone()

  for start in range(0, len(names), B):
    chunk = names[start : start + B]
    clips = [resample(f"{args.raw_dir}/dash_{n}.npy") for n in chunk]
    lens = [c["pos"].shape[0] for c in clips]
    T = max(lens)

    def stack(key: str) -> torch.Tensor:
      # Pad each clip by holding its last frame; pad the batch with clip 0.
      rows = [
        torch.cat([c[key], c[key][-1:].expand(T - len(c[key]), -1)]) for c in clips
      ]
      rows += [rows[0]] * (B - len(rows))
      return torch.stack(rows, 1).to("cuda:0")  # (T, B, ...)

    m = {k: stack(k) for k in clips[0]}
    log: dict[str, list[torch.Tensor]] = {k: [] for k in FIELDS}
    for t in range(T):
      root = robot.data.default_root_state.clone()
      root[:, 0:3] = m["pos"][t]
      root[:, :2] += origins[:, :2]
      root[:, 3:7] = m["rot"][t]
      root[:, 7:10] = m["lin_vel"][t]
      root[:, 10:] = m["ang_vel"][t]
      robot.write_root_state_to_sim(root)
      jp = robot.data.default_joint_pos.clone()
      jv = robot.data.default_joint_vel.clone()
      jp[:, jidx] = m["dof"][t]
      jv[:, jidx] = m["dof_vel"][t]
      robot.write_joint_state_to_sim(jp, jv)
      sim.forward()
      scene.update(sim.mj_model.opt.timestep)
      log["joint_pos"].append(robot.data.joint_pos.clone())
      log["joint_vel"].append(robot.data.joint_vel.clone())
      bp = robot.data.body_link_pos_w.clone()
      bp[..., :2] -= origins[:, None, :2]
      log["body_pos_w"].append(bp)
      log["body_quat_w"].append(robot.data.body_link_quat_w.clone())
      log["body_lin_vel_w"].append(robot.data.body_link_lin_vel_w.clone())
      log["body_ang_vel_w"].append(robot.data.body_link_ang_vel_w.clone())

    out = {k: torch.stack(v, 1).cpu().numpy() for k, v in log.items()}  # (B, T, ...)
    for b, (n, L) in enumerate(zip(chunk, lens, strict=True)):
      path = f"{args.out_dir}/dash_{n}.npz"
      np.savez(path + ".tmp.npz", fps=[OUT_FPS], **{k: v[b, :L] for k, v in out.items()})
      os.replace(path + ".tmp.npz", path)
    print(f"{start + len(chunk)}/{len(names)} done (T={T})", flush=True)


if __name__ == "__main__":
  main()
