# SuperMimic-Dash: handoff notes

This repo is a from-scratch SuperMimic reproduction for the Unitree G1 (see
README.md). **The goal now is to port it to Dash**, our own 18-DOF humanoid,
and train a SuperMimic-style tracking foundation model on our BONES-SEED
retargets. Training will run on a machine with an RTX 6000 Pro (96 GB).

The port is in progress on branch `dash-port` (pushed to origin 2026-10-28). See "Port status" below. The data-pipeline
scripts that produced the dataset are in `scripts/dash_data/`.

Always use `uv run`. Prefix with `env -u PYTHONPATH` if the shell has ROS or
other Python packages on its path.

## Related repos (all ShangqunYu / DARoSLab, all pushed)

- **`mjlab_dash`** (github.com/ShangqunYu/mjlab_dash): an mjlab fork with the
  Dash robot and a working BeyondMimic-style single-clip tracking task (PPO).
  A policy from it has already run on the real robot. Relevant pieces:
  - `src/mjlab/asset_zoo/robots/dash/`: `dash_constants.py` (actuators,
    `DASH_ACTION_SCALE`, `KNEES_BENT_KEYFRAME`, `get_dash_robot_cfg`) and
    `xmls/dash_v1.xml` plus `assets_v1/` meshes.
  - `src/mjlab/tasks/tracking/config/dash/env_cfgs.py`: the Dash tracking
    config, a good reference for what Dash needs:
    - **Tracked bodies:** torso (anchor), l/r_dist_hip, l/r_lower_leg,
      l/r_foot, l/r_upper_arm, l/r_lower_arm.
    - **Forearm–thigh collision:** mesh hulls plus explicit pairs, because
      mocap drives the hands into the thighs.
    - **`torque_near_limit` reward:** threshold 0.8.
    - **Base COM x range:** (-0.1, -0.02), because the real robot leaned
      backward.
    - **Ends of the limbs** (`l/r_foot`, `l/r_lower_arm`) are used for the
      end-effector termination.
    - **No state estimator:** the actor gets no `base_lin_vel` and no
      `motion_anchor_pos_b`. The critic keeps them.
  - `src/mjlab/tasks/velocity/config/dash/hardware_dr.py`: hardware domain
    randomization (COM, mass, PD gains, armature, friction).
  - `src/mjlab/scripts/csv_to_npz.py`: the reference converter that our npz
    files match.
  - It pins mujoco-warp ~=3.11. This repo pins ~=3.16 and torch 2.14/cu130,
    so **don't depend on the fork**. Vendor the Dash asset and constants into
    this repo instead.
- **`mocap_retargeting`** (github.com/DARoSLab/mocap_retargeting, run with
  `pixi run`): per-frame SQP IK retargeter. Dash/BONES-SEED setup:
  - `conf/task/dash/soma.yaml` and `assets/robots/dash/dash_soma_mocap.xml`.
  - `soma_ground_margin: 0` stops the feet hovering.
  - Batch options in `retarget_headless.py`: `mocap_list_file`,
    `kin_output_dir`, `save_mocap`, `skip_existing`, and atomic saves.

## Dash robot facts

- **Bodies (19, in model order):** torso, l_hip, l_dist_hip, l_upper_leg,
  l_lower_leg, l_foot, l_prox_shoulder, l_dist_shoulder, l_upper_arm,
  l_lower_arm, then the same 9 for the right side. The torso is the root and
  also carries the IMU.
- **Joints (18, in model order = action order):**
  - `{l,r}_hip_yaw`, `hip_roll`, `hip_pitch`, `knee_pitch`, `ankle_pitch`
  - `{l,r}_shoulder_pitch`, `shoulder_roll`, `shoulder_yaw`, `elbow_pitch`
  - All of the left side comes first. nq=25, nv=24.
- **No wrists and no ankle roll.** `lower_arm` is the arm's end effector.
- **PD gains:**
  - 80/2 on every leg joint and on shoulder_pitch (it shares the hip motor).
  - 40/1 on shoulder_roll, shoulder_yaw and elbow.
- **Effort limits and armature are estimates.** See `dash_constants.py`.

## Dataset (BONES-SEED → Dash), built 2026-10-26/27

Everything is on the external SSD "Crucial X10" (exFAT), under
`/media/simon/Crucial X10/dash_retarget/`. Copy `npz/` to local NVMe for
training: exFAT is very slow at opening tens of thousands of files in one
folder.

- **Source:** BONES-SEED SOMA uniform BVH (120 fps), Locomotion package,
  clips without props.
  - Kneeling, crawling, sitting on the floor and lying were kept.
  - Box, wall, stairs, ladder, chair and other prop clips were dropped.
  - 34,692 originals plus 34,660 mirrors (`*_M`).
- **`raw/dash_<clip>.npy`:** 69,352 retargets at 120 fps. A dict with
  - `q`: (T, 25) = root pos, quat **xyzw**, then the 18 joints
  - `fps`, `skeleton`, `joint_names`
  - No mocap array.
- **No foot-slide post-processing.** This is deliberate (median stance slide
  is about 5 cm/s); physics and RL absorb it, as SONIC does.
- **QA:** `batch_qa.py` writes `qa.csv`.
  - A clip is flagged for any of:
    - one-frame joint jumps
    - joint speed > 30 rad/s
    - foot below −5 mm, or body below −20 mm
  - 5,249 flagged clips (7.6%) are listed in `clips_flagged.txt`. Floor
    motions are flagged most: about 38% of sitting and 27% of crouching.
  - 64,103 passing clips are listed in `clips_pass_qa.txt`: 129 h, 32,069
    originals plus 32,034 mirrors.
  - Joint limits being hit is mostly expected and not filtered: shoulder roll
    at −0.17 (arms at the sides), straight knee, and hip pitch at −1.57 or
    +0.52.
- **`npz/dash_<clip>.npz`:** all 64,103 passing clips at 50 fps, in mjlab
  csv_to_npz format, all float32:
  - `fps` = [50]
  - `joint_pos` / `joint_vel`: (T, 18)
  - `body_pos_w`: (T, 19, 3); `body_quat_w`: (T, 19, 4) wxyz
  - `body_lin_vel_w` / `body_ang_vel_w`: (T, 19, 3)
  - The root (torso) is body index 0.
  - Made by `batch_to_npz.py` (many clips per GPU pass). It matches mjlab's
    `csv_to_npz` to within float32 rounding.
  - About 27 GB of data.
  - **Every file was load-checked after a filesystem repair** (see below).
- This repo's `motion_library.py` and `scripts/data/build_library.py` read
  exactly these keys. They take only root pose + joints and recompute bodies
  with FK. So our npz should feed `build_library.py` directly once FK and the
  robot spec are Dash.

## Port status (2026-10-27, lab server)

Lab server: 2× RTX PRO 6000 Max-Q, SSD at `/media/shangqunyu/Crucial X10`, `uv` at
`~/.local/bin/uv` (not on PATH). Local data (gitignored): `data/npz/dash_bones_seed/`
(64,103 clips, load-checked), `data/library/dash_bones_seed_compact.pt` (23.26M frames,
129.25 h, 2.28 GB fp16), LAFAN1 G1 data in `data/raw/lafan1`, `data/npz/lafan1` so the
G1 regression tests run.

- **Done, steps 1-4.**
  - `src/supermimic/robots.py`: `RobotSpec` registry (`g1`, `dash`). Every size follows
    from NJ: state 13+2NJ, xi 4+NJ, obs 3NJ+6, goal frame 2NJ+10 (Dash 49/22/60/46).
    Select with `--robot dash` (`TrainConfig.robot`); checkpoints and packed
    libraries record the robot and refuse a mismatch.
  - `src/supermimic/assets/dash/`: vendored XML, meshes and constants from
    mjlab_dash 0d56f20. Forearm-thigh hulls always on. Actuator command delay
    (0-25 ms) only with `--env.command-delay` (see below).
  - Hardware DR (`--env.hardware-dr`, off by default): mass, PD gains, armature, joint
    friction, wider COM. COM x is mjlab_dash's walking range (-0.06, 0.06) (its tracking
    task pins -0.06, d819b37; we keep the range by choice). **All of it enters xi**
    (2026-10-28): kp/kd scale, armature, joint friction per joint and mass scale per body
    are appended, xi 22 -> 113 (`envs/sim_context.py`, `StateSpecCfg.xi_hw_bodies`, set
    by the training loop). Reason: the policy learns only in imagination, so a parameter
    the world model cannot see is averaged out and the policy never trains against its
    variation. Checked by `tests/test_sim_context.py`.
  - Command delay (0-25 ms) is a separate flag, `--env.command-delay`, off: a delay
    makes s' depend on earlier actions, which the world model does not take as input.
  - Tests cover Dash FK vs MuJoCo, npz vs FK, and the analytic reward/obs/termination vs
    the simulator (parametrized g1/dash).
- **Observations.** Open-loop PD on the reference joints falls within ~0.5 s (kp=80
  sag). The reference sits outside the 0.9 soft joint limits on 39% of frames; the
  joint_limit penalty averages 3% of the max tracking reward per step (G1: 0.5%).
- **Not ported:** `scripts/media/render_rollout.py` (G1 XML), `eval_push.py` (G1 clip).
- **Multi-GPU on the lab server.** NCCL's default P2P path hangs on the first
  collective (PCIe, no NVLink). `NCCL_P2P_DISABLE=1` works (66 MB all-reduce 2.1 ms).
  `torchrun --standalone` also resolves the host to its Tailscale name and cannot
  connect; pass `--master-addr 127.0.0.1`. Decision: train on one GPU.
- **Run `dash_e1_sim`** (started 2026-10-27 ~20:15, GPU 0): sim-only (paper xi DR, no
  hardware DR), all 64,103 clips, 4096 envs, 100k iterations (cosine decay over 78k),
  ~3.6 it/s ≈ 8 h. Log `logs/dash_e1_sim.out`, checkpoints `logs/dash_e1_sim/`.
  Finished 2026-10-28 05:06 (8.9 h): train return plateaued at ~0.72 from ~50k,
  wm nse1 0.0036 / nse8 0.0047. Under training noise + DR it falls in 0.4-0.9% of
  episodes. Not deployable (no hardware DR; trained with the sampler bug below).
- **Reset-sampler fix (2026-10-28).** With raw discounted returns-to-go (paper, D14) the
  last bin of every clip has a low G_b only because few steps remain; with 64k short
  Dash clips (median 5.2 s) the hard term piled onto clip endings: in `dash_e1_sim`
  half of the clip-end episodes lasted <= 4 steps and the mean episode was 2.4 s
  (4.8 s expected with no falls). `MultiClipMotionCommandCfg.return_norm="horizon"`
  (default) divides R_t by 1 - gamma^h_t, h_t = steps the return could cover (episode
  steps left, plus the unreached clip frames after a fall). Same checkpoint, fresh
  sampler: clip-end median 4 -> 134 steps, mean episode 120 -> 220 steps (uniform:
  236). `return_norm="none"` restores the paper behaviour. Note the critic bootstrap
  of D14 was never wired (`bootstrap_fn` is None): time-outs drop their last 100 steps.
- The lab server is shared: other users' jobs can take both GPUs (seen 2026-10-28),
  which slowed `dash_e1_sim` from 3.6 to 2.6 it/s after ~60k.

## Original port plan

Work on a branch. Keep G1 working, and run `uv run pytest tests -q` after
each step.

1. **Robot-spec refactor.** "29" is hard-coded across about 10 files: all the
   files below that reference the G1 or 29 joints. Derive these from one robot
   spec instead: joint count, action dim, state/observation layouts, noise
   vectors, ξ (sim context) = 1 friction + NJ encoder bias + 3 COM, tracked
   body names, root body, and XML path.
   - `world_model/state_spec.py` (`NJ = 29`, xi offset/scale)
   - `policy/nets.py` (`ACT_DIM`, sizes)
   - `world_model/heads.py` (`OBS_TERMS`)
   - `world_model/dynamics.py`
   - `world_model/world_model.py`
   - `imagination/rollout.py` (`OBS_NOISE_HALF`)
   - `data/recorder.py`
   - `envs/mdp/motion_library.py`
   - `envs/mdp/rewards.py`
   - `envs/sim_context.py`
   - `envs/tracking_env_cfg.py` (G1 robot cfg, `G1_ACTION_SCALE`, tracked
     link names)
   - `data/motion_convert.py` (`g1_xml_path` / `load_g1_model`)
   - `training/loop.py`
2. **Vendor Dash.** Copy `dash_v1.xml`, the meshes and the relevant parts of
   `dash_constants.py` into this repo.
   - The FK (`kinematics/g1_fk.py`) is generic: it is built from any
     single-tree MjModel with hinge joints, so it should work for Dash
     unchanged. Verify it against MuJoCo, the way `tests/test_g1_fk.py` does.
3. **Dash env config.**
   - Tracked bodies as listed above.
   - `DASH_ACTION_SCALE`.
   - Termination per the paper: |Δh| > 0.25 m or tilt > 0.8. The thresholds
     may need loosening for floor motions.
   - Consider the forearm–thigh hull pairs from `mjlab_dash`.
4. **Build the compact library** from `npz/` with `build_library.py`, then
   run a short smoke training run.
5. **Later:**
   - Hardware DR beyond ξ (PD gains, armature, mass) for sim-to-real.
   - ONNX export.
   - Deployment. The paper's observations are proprioception-only, which
     already matches Dash having no state estimator.

## Machine caveats (original workstation)

- **Unstable RAM.** It showed bit flips: file hashes changing between reads,
  and a kernel Oops in exFAT writeback that hung the SSD. Validate batch
  outputs by loading them back, and use atomic writes and resumable steps.
- **SSD repair needs a newer fsck.** After that crash the SSD's exFAT needed
  repair. Ubuntu 22.04's exfatprogs 1.1.3 cannot fix "more clusters are
  allocated"; 1.2.5 built from source did.
- **Retargeter memory leak.** The retarget processes leak about 0.1 GB/min
  each. `run_loop.sh` restarts them every 12 minutes.
- **Hard-coded paths.** The scripts in `scripts/dash_data/` use the
  `/media/simon/Crucial X10/...` paths and the mocap_retargeting checkout.
  Adjust them on a new machine.
