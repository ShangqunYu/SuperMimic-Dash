"""Export a SuperMimic policy to ONNX for deployment: one file per motion clip, or one velocity-command policy.

--interface velocity (checkpoints trained with --velocity.enabled):

  inputs : obs (1, obs_dim)    same proprioceptive history as below
           command (1, 3)      (vx, vy) m/s in the heading frame, wz rad/s (world yaw rate), raw units
  outputs: actions (1, NJ)     as below

  uv run --with onnxruntime python scripts/export_onnx.py --interface velocity --ckpt logs/<run>/ckpt_latest.pt

--interface motion (default):

Same interface as mjlab_dash's tracking export (MotionTrackingOnPolicyRunner):

  inputs : obs (1, obs_dim)    proprioceptive history, the env's "policy" group
           time_step (1, 1)    control steps since the clip started (50 Hz), clamped at the clip end
  outputs: actions (1, NJ)     joint target = default_joint_pos + action_scale * actions
           joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w
                               the reference at time_step (for logging / initialization)

The policy's goal input (K future reference frames) depends only on the reference, so it is
precomputed per frame and stored in the file; the robot only supplies obs and time_step.

obs layout (term-major, each term's history oldest -> newest, 10 frames):
  joint_pos - default (NJ) | joint_vel (NJ) | base_ang_vel (3, body frame) | projected_gravity (3) | last action (NJ)
No observation scaling or clipping on the robot side: the network applies its own scales.

  uv run --with onnxruntime python scripts/export_onnx.py --ckpt logs/dash_e2_hwdr/ckpt_latest.pt \
      --clips dash_idle_loop_003__A265 dash_walk_forward_stop_002__A026
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
import torch
import tyro
from torch import nn

from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.eval.benchmark import load_policy_from_checkpoint


class OnnxMotionPolicy(nn.Module):
    def __init__(self, policy: nn.Module, goal: torch.Tensor, ref: dict[str, torch.Tensor]):
        super().__init__()
        self.policy = policy
        self.register_buffer("goal", goal)
        for k, v in ref.items():
            self.register_buffer(k, v)
        self.time_step_total = int(goal.shape[0])

    def forward(self, obs: torch.Tensor, time_step: torch.Tensor):
        t = torch.clamp(time_step.long().squeeze(-1), min=0, max=self.time_step_total - 1)
        return (
            self.policy(obs, self.goal[t]),
            self.joint_pos[t],
            self.joint_vel[t],
            self.body_pos_w[t],
            self.body_quat_w[t],
            self.body_lin_vel_w[t],
            self.body_ang_vel_w[t],
        )


class OnnxVelocityPolicy(nn.Module):
    def __init__(self, policy: nn.Module):
        super().__init__()
        self.policy = policy

    def forward(self, obs: torch.Tensor, command: torch.Tensor):
        return self.policy(obs, command, "velocity")


# Step-test commands for the velocity export check (vx, vy, wz).
CHECK_COMMANDS = [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0), (-0.5, 0.0, 0.0), (0.0, 0.4, 0.0), (0.0, 0.0, 1.0), (0.6, 0.0, 0.8)]

OUTPUTS = ["actions", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"]


def _csv(v) -> str:
    if isinstance(v, (list, tuple)):
        return ",".join(";".join(f"{y:.6g}" for y in x) if isinstance(x, (list, tuple)) else (f"{x:.6g}" if isinstance(x, float) else str(x)) for x in v)
    return str(v)


def metadata(env, ckpt: Path, clip: str, frames: int, cfg: dict) -> dict:
    """Keys follow mjlab_dash's get_base_metadata, plus SuperMimic-specific ones."""
    robot = env.scene["robot"]
    mjm = env.sim.mj_model
    ctrl = {int(mjm.actuator_trnid[c, 0]): c for c in robot.indexing.ctrl_ids.tolist()}
    ctrl_ids = [ctrl[int(j)] for j in robot.indexing.joint_ids.tolist()]
    om = env.observation_manager
    names = list(om.active_terms["policy"])
    term_cfgs = [om.get_term_cfg("policy", n) for n in names]
    act = env.action_manager.get_term("joint_pos")
    scale = act._scale[0].cpu().tolist() if isinstance(act._scale, torch.Tensor) else act._scale
    cmd = env.command_manager.get_term("motion")
    return {
        "run_path": f"{ckpt.parent.name}/{ckpt.name}",
        "motion_name": clip,
        "motion_frames": frames,
        "control_dt": env.step_dt,
        "joint_names": list(robot.joint_names),
        "joint_stiffness": mjm.actuator_gainprm[ctrl_ids, 0].tolist(),
        "joint_damping": (-mjm.actuator_biasprm[ctrl_ids, 2]).tolist(),
        "default_joint_pos": robot.data.default_joint_pos[0].cpu().tolist(),
        "command_names": list(env.command_manager.active_terms),
        "observation_names": names,
        "observation_terms_scale": [1.0] * len(names),
        "observation_terms_flatten_history_dim": [True] * len(names),
        "observation_terms_history_length": [cfg["env"].get("history_length", 10)] * len(names),
        "observation_terms_clip": [[float("-inf"), float("inf")] for _ in term_cfgs],
        "observation_layout": "term-major; each term's history oldest->newest",
        "action_scale": scale,
        "anchor_body_name": cmd.cfg.anchor_body_name,
        "body_names": list(cmd.cfg.body_names),
        "future_frames": cfg["env"]["future_frames"],
    }


@torch.no_grad()
def export_velocity(ckpt: Path, npz_dir: Path, out_dir: Path, check_steps: int, start_clip: str = "dash_idle_loop_003__A265") -> None:
    """Export obs + command -> actions and run a closed-loop step test in simulation (no DR, no noise)."""
    from supermimic.interfaces.velocity import velocity_of

    policy, cfg = load_policy_from_checkpoint(ckpt, "cuda:0")
    assert cfg["policy"].get("velocity"), "checkpoint has no velocity encoder"
    n = len(CHECK_COMMANDS)
    v = cfg["velocity"]
    env = make_env(
        SuperMimicEnvCfgArgs(
            npz_files=[str(npz_dir / f"{start_clip}.npz")], robot=cfg.get("robot", "g1"), num_envs=n, future_frames=cfg["env"]["future_frames"],
            episode_length_s=1.0e5, action_noise_std=0.0, obs_noise=False, reset_noise=False, domain_randomization=False,
            sampling_mode="start", record_transitions=True, replay_capacity=4, seed=0, fall_height=v["fall_height"], fall_tilt=v["fall_tilt"],
        ),
        device="cuda:0",
    )
    try:
        cmd = env.command_manager.get_term("motion")
        cmd.track_mask[:] = False  # velocity envs: fall-only termination, no clip end
        obs, _ = env.reset()
        cpu_policy, _ = load_policy_from_checkpoint(ckpt, "cpu")
        model = OnnxVelocityPolicy(cpu_policy).eval()
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "velocity.onnx"
        torch.onnx.export(model, (torch.zeros(1, obs["policy"].shape[1]), torch.zeros(1, 3)), str(path), export_params=True,
                          opset_version=18, input_names=["obs", "command"], output_names=["actions"], dynamic_axes={}, dynamo=False)
        meta = metadata(env, ckpt, "velocity", 0, cfg)
        for k in ("motion_name", "motion_frames", "anchor_body_name", "body_names", "future_frames"):
            meta.pop(k)
        meta.update(interface="velocity", command_names=["vx", "vy", "wz"], command_frame="vx, vy: m/s in the heading (yaw-only) frame; wz: rad/s world yaw rate",
                    command_range_vx=list(v["vx_range"]), command_range_vy=list(v["vy_range"]), command_range_wz=list(v["wz_range"]))
        m = onnx.load(str(path))
        for k, val in meta.items():
            m.metadata_props.append(onnx.StringStringEntryProto(key=k, value=_csv(val)))
        onnx.save(m, str(path))
        (out_dir / "velocity.json").write_text(json.dumps(meta, indent=1))
        print(f"[export] {path}")

        try:
            import onnxruntime as ort
        except ImportError:
            print("[check] onnxruntime not installed; run with `uv run --with onnxruntime` to verify")
            return
        sess = ort.InferenceSession(str(path))
        io = env.recorder_manager.get_term("replay").state_io
        c = torch.tensor(CHECK_COMMANDS, device=env.device)
        err, falls, achieved, k = 0.0, np.zeros(n, dtype=int), torch.zeros(n, 3, device=env.device), 0
        for t in range(check_steps):
            a_cpu = cpu_policy(obs["policy"].cpu(), c.cpu(), "velocity").numpy()
            o = obs["policy"].cpu().numpy()
            for i in range(n):
                a = sess.run(["actions"], {"obs": o[i : i + 1], "command": c[i : i + 1].cpu().numpy()})[0]
                err = max(err, float(np.abs(a - a_cpu[i : i + 1]).max()))
            obs, _, term, _, _ = env.step(policy(obs["policy"], c, "velocity"))
            falls += term.cpu().numpy()
            if t >= check_steps - 100:  # last 2 s
                achieved += velocity_of(io.read())
                k += 1
        achieved /= max(k, 1)
        for i, cc in enumerate(CHECK_COMMANDS):
            print(f"[check] cmd {cc}: achieved ({achieved[i, 0]:+.2f}, {achieved[i, 1]:+.2f}, {achieved[i, 2]:+.2f}) over the last 2 s, falls {falls[i]}")
        print(f"[check] max |a_onnx - a_torch| = {err:.2e} over {check_steps} steps")
        assert err < 1e-4, "ONNX output differs from the PyTorch policy"
    finally:
        env.close()


@torch.no_grad()
def main(
    ckpt: Path = Path("logs/dash_e2_hwdr/ckpt_latest.pt"),
    interface: str = "motion",
    clips: list[str] = ["dash_idle_loop_003__A265", "dash_walk_forward_stop_002__A026", "dash_walk_ff_stop_360_R_slow_001__A445"],
    npz_dir: Path = Path("data/npz/dash_bones_seed"),
    out_dir: Path | None = None,
    check_steps: int = 300,
) -> None:
    out_dir = out_dir or ckpt.parent / "onnx"
    if interface == "velocity":
        return export_velocity(ckpt, npz_dir, out_dir, check_steps)
    assert interface == "motion", interface
    out_dir.mkdir(parents=True, exist_ok=True)
    policy, cfg = load_policy_from_checkpoint(ckpt, "cuda:0")
    files = [str(npz_dir / f"{c}.npz") for c in clips]
    n = len(files)
    env = make_env(
        SuperMimicEnvCfgArgs(
            npz_files=files, robot=cfg.get("robot", "g1"), num_envs=n, future_frames=cfg["env"]["future_frames"],
            history_length=cfg["env"].get("history_length", 10), episode_length_s=1.0e5, action_noise_std=0.0,
            obs_noise=False, reset_noise=False, domain_randomization=False, sampling_mode="start",
            record_transitions=False, seed=0, anchor_obs=cfg["env"].get("anchor_obs", False),
        ),
        device="cuda:0",
    )
    assert not cfg["env"].get("anchor_obs", False), "anchor_obs needs the robot's world pose; not supported here"
    try:
        cmd = env.command_manager.get_term("motion")
        lib = cmd.library
        K = cfg["env"]["future_frames"]
        cmd.forced_frame[:] = lib.clip_start[:n]
        obs, _ = env.reset()
        cpu_policy, _ = load_policy_from_checkpoint(ckpt, "cpu")
        models = []
        for i, clip in enumerate(clips):
            assert lib.clips[i].name == clip, (lib.clips[i].name, clip)
            frames = torch.arange(int(lib.clip_start[i]), int(lib.clip_last[i]) + 1, device=lib.device)
            r, b = lib.ref(frames), lib.bodies(frames)
            ref = {"joint_pos": r.joint_pos, "joint_vel": r.joint_vel, "body_pos_w": b.pos_w, "body_quat_w": b.quat_w, "body_lin_vel_w": b.lin_vel_w, "body_ang_vel_w": b.ang_vel_w}
            model = OnnxMotionPolicy(cpu_policy, lib.goal_window(frames, K).float().cpu(), {k: v.float().cpu() for k, v in ref.items()}).eval()
            path = out_dir / f"{clip}.onnx"
            torch.onnx.export(model, (torch.zeros(1, obs["policy"].shape[1]), torch.zeros(1, 1)), str(path), export_params=True,
                              opset_version=18, input_names=["obs", "time_step"], output_names=OUTPUTS, dynamic_axes={}, dynamo=False)
            m = onnx.load(str(path))
            meta = metadata(env, ckpt, clip, len(frames), cfg)
            for k, v in meta.items():
                m.metadata_props.append(onnx.StringStringEntryProto(key=k, value=_csv(v)))
            onnx.save(m, str(path))
            (out_dir / f"{clip}.json").write_text(json.dumps(meta, indent=1))
            models.append((path, model))
            print(f"[export] {path}  ({len(frames)} frames, {len(frames) * env.step_dt:.1f} s)")

        # Check: closed loop in simulation, the exported model (onnxruntime, CPU) given only obs + time_step
        # must reproduce the PyTorch policy given the env's own goal observation.
        try:
            import onnxruntime as ort
        except ImportError:
            print("[check] onnxruntime not installed; run with `uv run --with onnxruntime` to verify")
            return
        sessions = [ort.InferenceSession(str(p)) for p, _ in models]
        err, falls = np.zeros(n), np.zeros(n, dtype=int)
        for _ in range(check_steps):
            # Reference on CPU: mjlab enables TF32 on the GPU, which alone differs by ~1e-3.
            a_cpu = cpu_policy(obs["policy"].cpu(), obs["goal"].cpu()).numpy()
            t = (cmd.frame[:n] - lib.clip_start[:n]).cpu().numpy()
            o = obs["policy"].cpu().numpy()
            for i, s in enumerate(sessions):
                a = s.run(["actions"], {"obs": o[i : i + 1], "time_step": np.array([[t[i]]], dtype=np.float32)})[0]
                err[i] = max(err[i], float(np.abs(a - a_cpu[i : i + 1]).max()))
            obs, _, term, _, _ = env.step(policy(obs["policy"], obs["goal"]))
            falls += term.cpu().numpy()
        for i, clip in enumerate(clips):
            print(f"[check] {clip}: max |a_onnx - a_torch| = {err[i]:.2e} over {check_steps} steps, falls {falls[i]}")
        assert err.max() < 1e-4, "ONNX output differs from the PyTorch policy"
    finally:
        env.close()


if __name__ == "__main__":
    tyro.cli(main)
