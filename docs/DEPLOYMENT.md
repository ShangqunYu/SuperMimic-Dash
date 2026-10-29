# Dash SuperMimic policy: deployment handoff

Written 2026-10-29 on the lab server, for whoever (human or Claude) runs this policy on the real
Dash robot. It is self-contained: you do not need the training machine to read it.

## What this is

- **Repo:** github.com/ShangqunYu/SuperMimic-Dash, branch `dash-port`.
- **Method:** a reproduction of SuperMimic, a motion-tracking policy trained with a learned
  world model (the policy learns in imagination, not with PPO), ported from the Unitree G1 to
  Dash, our 18-DOF humanoid.
- **Training data:** 64,103 BONES-SEED mocap clips retargeted to Dash (129 h, mirrors
  included). That means one policy for all of them, unlike mjlab_dash's one-clip-per-policy
  BeyondMimic setup.
- **For deployment, each ONNX file bundles the policy with one reference clip.** You give it
  the robot's proprioception and the step count; it returns joint targets. This is the same
  input/output contract as mjlab_dash's tracking export (`MotionTrackingOnPolicyRunner`).
  **The observation vector is different**, though, so the robot-side observation builder must
  change (see below).

## Files

The ONNX files are attached to the GitHub release `dash_e2_hwdr`
(github.com/ShangqunYu/SuperMimic-Dash/releases/tag/dash_e2_hwdr). Each `.onnx` has a `.json`
next to it with the same metadata in readable form.

| file | motion | length | benchmark score |
|---|---|---|---|
| `dash_idle_loop_003__A265.onnx` | standing idle, small sway | 9.6 s (481 frames) | 0.994 |
| `dash_walk_forward_stop_002__A026.onnx` | walk forward, then stop | 9.1 s (456 frames) | 0.975 |
| `dash_walk_ff_stop_360_R_slow_001__A445.onnx` | slow walk with a full turn to the right | 9.2 s (458 frames) | 0.965 |

Suggested order: idle first, on a gantry or harness, then the forward walk, then the turn.

Other clips can be exported on the training machine (clip names are the npz stems in
`data/npz/dash_bones_seed/`):

```bash
uv run --with onnxruntime python scripts/export_onnx.py \
    --ckpt logs/dash_e2_hwdr/ckpt_latest.pt --clips <clip> [<clip> ...]
```

## ONNX interface

**Inputs, both float32:**
- `obs` `(1, 600)`: proprioceptive history, built as described below.
- `time_step` `(1, 1)`: control steps since the clip started (0, 1, 2, ...), at 50 Hz.
  - Inside the model it is clamped to `[0, frames-1]`.
  - After the last frame, the reference holds its final pose.

**Outputs:**
- `actions` `(1, 18)`: the raw policy action.
  - Joint target = `default_joint_pos + action_scale * actions`, sent to joint PD.
  - Use the gains `joint_stiffness` / `joint_damping` from the metadata:
    - kp 80, kd 2: all leg joints and shoulder_pitch
    - kp 40, kd 1: shoulder_roll, shoulder_yaw, elbow
  - `action_scale` is 0.25 for every joint, the same as mjlab_dash.
  - No action clipping was used in training.
- `joint_pos`, `joint_vel` `(1, 18)`, and `body_pos_w`, `body_quat_w` (wxyz), `body_lin_vel_w`,
  `body_ang_vel_w` `(1, 11, ·)`:
  - These are the reference at `time_step`, for logging and for checking the start pose.
  - The policy does not need them back.
  - Bodies are in the order of the `body_names` metadata, with the torso first.

**Control rate:** 50 Hz (`control_dt` 0.02). Physics was simulated at 200 Hz (decimation 4).

**Joint order** (action order = observation order = model order):

```
l_hip_yaw l_hip_roll l_hip_pitch l_knee_pitch l_ankle_pitch
l_shoulder_pitch l_shoulder_roll l_shoulder_yaw l_elbow_pitch
r_hip_yaw r_hip_roll r_hip_pitch r_knee_pitch r_ankle_pitch
r_shoulder_pitch r_shoulder_roll r_shoulder_yaw r_elbow_pitch
```

**`default_joint_pos`:** 0 for every joint, except:
- hip_pitch −0.349
- knee 0.698
- ankle −0.349
- elbow −0.8

## Building `obs` (600 = 10 history frames × 60)

Five terms per frame, 60 values in total:

| term | dims | definition |
|---|---|---|
| `joint_pos` | 18 | measured joint position − `default_joint_pos` |
| `joint_vel` | 18 | measured joint velocity (rad/s) |
| `base_ang_vel` | 3 | torso angular velocity in the torso (IMU) frame, rad/s |
| `projected_gravity` | 3 | gravity unit vector (0, 0, −1) rotated into the torso frame; ≈ (0, 0, −1) when upright |
| `actions` | 18 | the raw `actions` output of the previous step (0 before the first step) |

**Layout is term-major, not frame-major:**

```
obs = [ joint_pos[t-9] ... joint_pos[t]          # 10 × 18
        joint_vel[t-9] ... joint_vel[t]          # 10 × 18
        base_ang_vel[t-9] ... base_ang_vel[t]    # 10 × 3
        projected_gravity[t-9] ... [t]           # 10 × 3
        actions[t-9] ... actions[t] ]            # 10 × 18   (oldest first in every block)
```

**Rules:**
- No scaling and no clipping on the robot side; the network applies its own scales.
- **At start, fill every history slot with the first observation.** This is mjlab's reset
  behaviour, so the history is never zero-padded.
- `actions` in the history is the *raw* network output, not the joint target.
- Training had no base linear velocity and no world position. The policy needs no state
  estimator: IMU and joint encoders only.

**Per-step loop:**

```
t = 0; fill history with obs_now (actions term = 0)
every 20 ms:
    read joints + IMU -> append frame to history (actions term = last raw action)
    actions = onnx(obs=history, time_step=t)
    q_target = default_joint_pos + 0.25 * actions  -> PD
    t += 1
```

## How the policy was trained (run `dash_e2_hwdr`)

- **Run:** 100k iterations, about 9.7 h on one RTX PRO 6000, finished 2026-10-29.
- **Clips:** all 64,103.
- **Observations:** proprioception only, as above.
- **Randomization the policy has seen:**
  - **Paper DR, per episode:**
    - foot friction 0.3–1.2
    - joint encoder bias ±0.01 rad
    - observation noise:
      - joint_pos ±0.01
      - joint_vel ±0.5
      - ang_vel ±0.2
      - gravity ±0.05
    - action execution noise, std 0.1 (in action units, before the 0.25 scale)
    - reset pose and velocity noise
  - **Hardware DR, taken from mjlab_dash's `hardware_dr.py`:**
    - body mass ×0.88–1.05, with inertia scaled along
    - kp ×0.8–1.2 and kd ×0.7–1.3, sampled per joint
    - armature:
      - hips/shoulder_pitch 0.01–0.03
      - knee 0.02–0.05
      - ankle 0.01–0.03
      - arms 0.008–0.01
    - joint friction:
      - hips 0–2.5 Nm
      - knee/ankle 0–1.5 Nm
      - arms 0–2.0 Nm
    - torso COM offset:
      - x ±0.06
      - y ±0.025
      - z ±0.03
- **NOT randomized: actuator or communication latency.** This was a deliberate choice for this
  run. The world model takes no action history, so a delay would make its targets
  inconsistent. **If the robot oscillates or buzzes, latency is the first suspect.** The fix
  would be to train with `--env.command-delay` plus a previous-action input to the world
  model, which isn't implemented yet.
- **No pushes**, and no `torque_near_limit` reward (mjlab_dash's tracking task has one).
- **Collision:** forearm–thigh mesh hulls are active in simulation.

## How good it is (simulation)

Benchmark: every one of the 64,103 clips played from start to end once, nominal robot, no noise.

| | sim-only run (`dash_e1_sim`) | this policy (`dash_e2_hwdr`) |
|---|---|---|
| clips failed (fell) | 498 (0.78%) | 490 (0.76%) |
| mean tracking score (0-1) | 0.887 | 0.882 |

Failures by motion type (`dash_e2_hwdr`):
- walk 0.03%
- jump 0.22%
- turn 0.12%
- run 0.45%
- **floor sitting 13.6%:** most sit-down clips fail within the first 10–20 steps

Avoid sitting and lying clips on hardware.

The ONNX files were checked in closed loop in simulation for 600 steps each: they matched the
PyTorch policy to 2e-6 given only `obs` and `time_step`, with no falls. This checks the time
alignment and the goal precomputation. It does **not** check the robot-side observation
builder; that is the first thing to unit-test on the deployment side, for example by replaying
a logged sim rollout.

## Known risks on hardware, in order of likelihood

1. **Observation layout mismatch.** Term-major history, projected gravity, and the raw action
   in the history. A layout bug looks like immediate violent motion. Test with the robot
   hanging first.
2. **Latency** (not trained), see above.
3. **Start pose.** Start from the clip's first reference frame. Use the `joint_pos` output at
   `time_step` 0 to interpolate there slowly before starting the clock. Idle and walk clips
   start near the default standing pose.
4. **COM / backward lean.** mjlab_dash found that the real robot leans backward and pins its
   tracking COM x at −0.06; this policy saw x uniform in ±0.06.
5. **Effort limits and armature** in the model are estimates (see mjlab_dash
   `dash_constants.py`).

## Pointers

- **Training repo notes:** `CLAUDE.md` (port status, dataset, run history).
- **Exporter:** `scripts/export_onnx.py`.
- **Observation definitions:** `src/supermimic/envs/tracking_env_cfg.py` (`policy_terms`).
- **Robot constants:** `src/supermimic/assets/dash/`, vendored from mjlab_dash 0d56f20.
- **mjlab_dash** (github.com/ShangqunYu/mjlab_dash): the BeyondMimic-style Dash tracking task
  whose ONNX policy already ran on the robot. Its deploy code is the starting point; only the
  observation builder and the action scale need checking.
