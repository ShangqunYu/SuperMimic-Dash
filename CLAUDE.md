# SuperMimic-Dash: handoff notes

This repo is a from-scratch SuperMimic reproduction for the Unitree G1 (see
README.md). **The goal now is to port it to Dash**, our own 18-DOF humanoid,
and train a SuperMimic-style tracking foundation model on our BONES-SEED
retargets. Training will run on a machine with an RTX 6000 Pro (96 GB).

Nothing in `src/` has been changed for Dash yet. The only Dash addition so far
is `scripts/dash_data/`, which holds the data-pipeline scripts that produced
the dataset described below.

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

## Port plan (agreed, not started)

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
