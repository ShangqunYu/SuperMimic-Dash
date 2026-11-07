# SuperMimic-Dash: handoff notes

This repo is a from-scratch SuperMimic reproduction for the Unitree G1 (see
README.md). **The goal now is to port it to Dash**, our own 18-DOF humanoid,
and train a SuperMimic-style tracking foundation model on our BONES-SEED
retargets. Training will run on a machine with an RTX 6000 Pro (96 GB).

The port is in progress on branch `dash-port` (pushed to origin 2026-10-28). See "Port status" below; for running a policy on the robot see `docs/DEPLOYMENT.md`. The data-pipeline
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

Everything is on the external SSD "Crucial X10" (exFAT; relabelled `Dash` on the lab server), under
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

## Terrain dataset (2026-11-05, lab server)

Goal: rough-terrain tracking, in the style of Generate-Track-Improve (arXiv
2609.31577, which also starts from BONES-SEED). BONES-SEED has no terrain
geometry, only prop labels, so the terrain is recovered from Dash's footholds.

- **Retarget:** the same mocap_retargeting setup (pixi at `~/.pixi/bin/pixi`),
  unchanged. Its floor constraint is only `z >= 0`, so box and stair footholds
  pass through.
- **Data:** 1,634 prop clips plus mirrors, in `data/terrain/raw/` (q at
  120 fps, plus the scaled human `mocap`).
  - Box, stairs (walking only) and no-hands obstacle jumps.
  - Running jumps, double jumps and forward leaps come from the flat set on
    the SSD.
- **`scripts/terrain/pilot_terrain.py`:**
  - Foothold detection: a foot capsule end that is still.
  - Heights clustered into levels.
  - Straight-edge profile along the walking direction, scoring and rendering.
- **`scripts/terrain/terrain_edit.py`:** fits clean terrain and edits the
  motion onto it (GTI's "foot offsets between contacts, then IK"):
  - **Box:** two-level profile, top lowered to 0.30 m. The real "50 cm" box
    is 0.33–0.40 m at Dash scale.
  - **Stairs:** regular rise and run fitted to all footholds, extended past
    the clip ends. Run deepened to at least 22 cm; the scaled tread is
    15.5–20 cm against Dash's 18 cm foot, so feet hung over step edges.
  - **Each stance phase:** levelled, set at surface height and slid fully
    onto its tread. Offsets blend over swings, and the torso gets the mean
    foot offset.
  - **Leg IK:** damped least squares on the 5 leg joints. Each frame starts
    with the knee bent at least 0.25 rad, with a small pull towards the warm
    start; up to 50 iterations.
  - **Swing-clearance lift**, then the legs are re-solved. The total lift
    ramps at most 1 m/s, from 0 at lift-off back to 0 at touchdown.
  - **IK fix (2026-11-06):** the first version made joint jumps in about 1,100
    passing clips, up to 1.45 rad in one frame (stairs up 5.4x their source's
    peak speed). Causes:
    - the lift switched on at up to 4 m/s, because the toe reaches the next
      riser in the first swing frame;
    - at a straight knee the IK has no gradient along the leg, so the knee
      stuck at 0, then snapped.
  - **Stair jogging is dropped:** too few footholds, 18 clips.
- **`scripts/terrain/jump_pilot.py`:**
  - Box placed under unedited jumps where every foot and shin point clears
    it by 2 cm.
  - Obstacle jumps: 25 cm box (the feet clear about 50 cm). Leaps: 8–12 cm,
    forward leaps only (travel within 30° of the heading).
- **Result (v2, fixed IK):** `data/terrain/manifest_v2.csv` (written by
  `scripts/terrain/make_manifest.py`), 2,774 edited clips plus 194 jumps pass
  (2,968). Edited clips are in `data/terrain/edited_v2/`; the library is
  `data/library/dash_terrain_v2_compact.pt` (36 tiles). Worst per-clip joint
  speed, stairs up: 36.5 -> 16.8 rad/s.
  - **Old (jumpy) set:** `data/terrain/manifest.csv`, `data/terrain/edited/`,
    `dash_terrain_compact.pt` and `dash_flat_terrain_compact.pt`; 3,019 pass.
    `dash_t1_blind` was trained on it.
  - QA thresholds (`make_manifest.py`):
    - no edit-made joint jumps: each joint's peak speed ≤ 2x its peak in the
      source + 5 rad/s (v2 only)
    - foot penetration ≤ 4 cm
    - shin penetration ≤ 3 cm
    - stance IK error ≤ 3 cm
    - < 80% sole support on ≤ 25% of standing frames
  - Passing clips by type:
    - box come up 1,645, come down 1,086, jump on/off 68
    - stairs 26
    - jumps 194
  - Boxes dominate, so oversample stairs and jumps.
  - Outputs:
    - `data/terrain/edited/*.npz`: `qpos` at 60 fps; profile `c`, `d`,
      `edges`, `heights`, where `h(xy) = heights[searchsorted(edges, (xy - c) @ d)]`.
    - `data/terrain/edited_jumps/*.npz`: `qpos`, `box_s`, `box_height`,
      `box_width`, `d`, `c`.
  - Not trimmed yet; the user may trim later.
- **Depth camera:** `realsense_depth` in mjlab_dash `dash_constants.py`.
- **In simulation:**
  - **`scripts/terrain/build_terrain_library.py`:**
    - Puts each clip into its tile's frame (yaw plus a planar shift),
      resamples to 50 fps and packs a compact library (optionally prepending
      the flat pack, `--flat`).
    - The pack carries `clip_tile`, `tile_names` and `tiles`, plus a
      sidecar `<pack>.tiles.npz`.
    - 38 tiles:
      - `box30`, shared by 2,799 clips
      - 26 `stairs_<clip>`
      - 11 `jump_h<cm>_d<cm>`
  - **`src/supermimic/terrain/tiles.py`:**
    - `ClipTilesTerrainCfg`: one mjlab generator patch holding the floor plus
      all tile boxes.
    - Tiles sit on a grid from x = 30 m, 16 m apart. Flat clips spawn at the
      world origin; they travel at most 8.6 m.
    - The env config switches to it when `load_tiles(npz_files[0])` finds
      the sidecar.
  - **`MultiClipMotionCommand._place_on_terrain`:**
    - At reset, an env's `scene.env_origins` row moves to the sampled clip's
      tile, or back to its flat origin. Each env is its own MuJoCo world, so
      any number of envs share a tile.
    - `StateIO` reads the origins live.
  - **Checks:**
    - `scripts/terrain/check_tiles.py` (CPU replay on the tiles): robot
      penetration p50 1.3 cm, max 4.6 cm.
    - `scripts/terrain/eval_terrain.py` (benchmark plus rollout video):
      dash_e2_hwdr, the blind flat policy, fails 2,679 of 3,019 terrain
      clips. Come-up fails 99.9%, come-down 82%, stairs 96%, forward leaps
      27%.
  - Rendering from `env.sim.mj_model` with `qpos` gives an exploded robot. Use
    the robot XML plus the tile (`check_tiles.tile_model`) in the tile frame
    instead.
- **Blind tracker, terrain-aware world model** (2026-11-05, user's decision: the
  policy stays blind, a later depth-conditioned generator plans the references):
  - The policy learns only in imagination, so the world model needs the terrain,
    or it averages a step-up with a fall. The world model and the critic get a
    height scan; the policy never does. ONNX export is unchanged.
  - xi carries the env's tile index as an extra last column (`data/recorder.py`,
    `terrain_xi`; -1 = flat). `StateSpec.height_scan` is set by the training loop
    when the library has tiles. `normalize_xi` drops the column. `ctx_dim` = xi
    plus the scan.
  - `terrain/height_scan.py`: a heading-aligned grid around the root, x from -0.35
    to 0.85 m, y ±0.35 m, 5 cm spacing (375 points). Each value is the absolute
    ground height in the tile frame, computed exactly from the tile boxes, so it is
    0 on flat ground. `WorldModel.context(s, xi)` = [normalised xi, scan / 0.5],
    recomputed every imagined step.
  - Warm start (`init_from`): zero input columns are inserted into the first layer
    of the dynamics and the critic, so they start exactly as before. The WM Adam
    state is then fresh, and the first updates raise WM nse1 from about 0.003 to
    0.07; it recovers to about 0.01 within 400 iterations. Sampler bins of a
    `--flat` mixed library are e2's as a prefix and are copied.
  - `--env.terrain-ratio`: this share of "return" resets is drawn uniformly from
    terrain bins. Log keys `wm/online_nse_{1,8}step_terrain`.
  - Mixed library `data/library/dash_flat_terrain_compact.pt`: 67,122 clips,
    132.2 h (3,019 terrain clips, 2.96 h).
  - **Run `dash_t1_blind`** (started 2026-11-05 ~14:00, GPU 0, shared with other
    users' jobs, ~2 it/s): init from dash_e2_hwdr, mixed library, hardware DR,
    terrain ratio 0.25, lr 1e-4, policy batch 2048, 40k iterations. Log
    `logs/dash_t1_blind.out`. Finished 18:25 (4.4 h). WM nse1 0.0055, terrain
    0.012. It spiked to 0.011 / 0.025 around 16k, then recovered.
  - **Terrain benchmark** (`eval_terrain.py`, 3,019 clips) failures per checkpoint:
    e2 2,679; 8k 1,680; 16k 1,247; 24k 255; 32k 116; **40k 58** (mean score 0.733
    -> 0.814). At 40k:
    - step up 17/1,647, step down 1/1,074, leaps 7/166, jump off 4/48;
    - jump on 9/20, stairs down 4/18, **stairs up 8/8 (still all fail)**.
    - Step-up was learned between 16k and 32k. Before that, failures came at the
      weight transfer, with the reference torso 11-17 cm past the edge.
  - **Flat benchmark** (all 64,103 clips from the packed flat library, same
    protocol for both; scratchpad `flat_bench.py`, results in
    `logs/terrain_eval/flat_*.json`):
    - e2: 498 failures, mean 0.8823. t1 40k: **100 failures**, mean 0.8812.
    - 438 clips fixed (mostly floor sitting), 40 newly failing (sitting, reach
      jump, jog).
    - So flat tracking did not degrade.
  - **Run `dash_t2_blind_v2`** (started 2026-11-06 ~12:05, GPU 1): fine-tune of t1 40k on
    the fixed-IK mixed library `data/library/dash_flat_terrain_v2_compact.pt` (67,071
    clips, 2,968 terrain), same flags as t1, 20k iterations. Log `logs/dash_t2_blind_v2.out`.
    `init_from` from a checkpoint with more sampler bins than the library (another terrain
    set) keeps only the leading flat bins.
  - Finished 13:55 (1.8 h, 3.2 it/s). Benchmarks (`logs/terrain_eval/v2/`,
    `flat_*`, `dr_*`; "rand" = `run_benchmark(randomize=True)`: obs noise, sim and hardware
    DR, same seed for every policy):

    | | terrain v2 | terrain v2 rand | flat | flat rand |
    |---|---|---|---|---|
    | t1 40k | 56, 0.80 | 109, 0.80 | 100, 0.881 | 142, 0.867 |
    | t2 20k | 32, 0.82 | 41, 0.85 | 174, 0.884 | 218, 0.869 |

    (failures, mean score). t2 is better on terrain (step up 12 -> 2, stairs down 5 -> 2;
    stairs up still 7/8 fail) and on flat motions other than floor sitting and leaning
    idles, which account for the extra flat failures (sitting 52 -> 120, rand 60 -> 146).
  - Train return fell from ~0.5 to ~0 during t2 (t1 stayed ~0.4). It only measures the
    hardest bins: the sampler gives ~100% of its weight to the top 1% of bins. Over all
    flat bins t2's bin returns match t1's (mean -0.03).
  - The restart at lr 1e-4 (t1 ended at 3e-6) costs flat tracking mid-run: at 8k flat mean
    0.864 (115 failures), recovered by 20k.
- **Physics cleanup test (DSMS, 2026-11-06).** GTI cleans each reference with "Shooting
  for Contact" (contact-implicit multiple shooting in MuJoCo, IPOPT; public repo, run from a
  scratchpad clone with a micromamba env, not vendored). Receding horizon on stairs-up clip 101
  in the training model (PD servos, effort limits), ~32 min per 4.9 s clip on 8 CPU threads.
  - Paper (soft) contact tracks the reference within 4-6 cm at the root, but the training
    contact's open-loop replay drifts 0.5 m by 2 s. The training contact tracks poorly (12-15 cm).
  - Torque: with both the old and the fixed reference, the ankles are at the 35 Nm limit
    10-16% of the time, hip roll/pitch (50 Nm) 6-31%, knees (80 Nm) 2-16%, mostly in stance.
    The optimizer barely penalises torque (Rtau 1e-6), so this is an upper bound on need.
  - Videos `logs/terrain_eval/dsms_up101*_torque.mp4`.
  - **Those peaks are not real demand (2026-11-06).** The 180 Nm knee peak is the optimizer's
    target zig-zagging past the clip end (horizon > clip); the uncapped 170-190 Nm peaks are
    shins wedged on step edges. Excluding both, p99 is hip ~65-72, knee ~70, ankle ~75 Nm.
- **Reference dynamics check (scratchpad `dsms/ref_id.py`):** inverse dynamics of the
  reference with line-foot contacts (two points per capsule foot, friction cone), contact forces
  chosen to minimise sum (tau/limit)^2. Hip roll needs only 10-16 Nm p95; ankle p95 ~30 Nm on
  every clip including flat walking; stairs up hip pitch/knee p95 38-50 Nm.
  - **Edit bug found:** stairs clips 101/103 (A301) need a ~body-weight "ghost" force for
    0.12-0.18 s windows: both feet 4-9 cm above any tread while the torso holds height. Cause:
    the retarget's feet slide 0.15-0.2 m/s on stairs (human stands on the forefoot, heel up; Dash's
    rigid line foot rolls), so `recover`'s stance (robot foot end still, < 0.12 m/s) ends early;
    the "swing" lift then raises the loaded foot over the next edge. Human foot contacts
    (mocap ankle/ball still) are missed as stance on 11-35% of planted frames across stairs and
    box clips.
- **Terrain edit v3 (2026-11-06, fixes the above):** stance = robot foot end still OR human
  ankle/ball still (only where the robot foot end is within 3 cm of a recovered surface; humans
  pause feet mid-step); in each stance the resting end is held still (`slide_correction`); sole
  support is scored on flat-foot frames only; new QA `support_check` (inverse dynamics, line-foot
  contacts): reject when > 3/4 of the body weight is unsupported for > 0.1 s; stance IK bound
  3 -> 4 cm (a locked trailing foot at full leg extension ends ~3 cm out of reach at lift-off).
  - **58% of the v2 passing clips float** > 0.1 s by this check (come down 75%, come up 48%,
    stairs up 4/8). t1/t2 were trained on them.
  - v3: `data/terrain/manifest_v3.csv` 2,208 edited + 194 jumps pass, all float-free; stairs up
    12/12, stairs down 14/20, come up 1,393, come down 719 (v2 float-free: 4, 6, 850, 259).
    Libraries `dash_terrain_v3_compact.pt` (2,402 clips, 38 tiles), `dash_flat_terrain_v3_compact.pt`.
  - Box step-downs still drop at ~0.3-0.6 g for ~0.3 s (the retarget loses the trailing toe's
    contact on the box edge); allowed (below 3/4).
  - Reference inverse dynamics, stairs up v3 (p95 / max Nm): hip roll 14-16 / 21-31, hip pitch
    35-77 / 108-170, knee 47-62 / 80-89, ankle 31-35 / 36-41; unsupported force p95 37-69 N
    (walking 48). Hip pitch rises vs v2 because the legs now carry what the ghost force did.
    Tried: torso following only the placement offsets (not the lock): hip pitch max 175 -> 132
    but stance IK error 5 cm; not kept.
  - t2 on v3 terrain: 31/2,402 failures, stairs up 12/12 fail. **Run `dash_t3_blind_v3`**
    (GPU 1): fine-tune of t2 20k on the v3 mixed library, same flags, 20k iterations.

## Port status (2026-10-27, lab server)

Lab server: 2× RTX PRO 6000 Max-Q, SSD relabelled `Dash` (3.6 TB exFAT, 2026-11-04) at `/media/shangqunyu/Dash`, `uv` at
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
- **Run `dash_e2_hwdr`** (2026-10-28 18:30 to 2026-10-29 04:15, GPU 0): `dash_e1_sim` plus
  `--env.hardware-dr` (hardware xi 113), fixed sampler, no command delay. Train return dipped
  to ~0 at 12-25k and settled at 0.33-0.45; it is not comparable to e1 (harder DR, no more
  free 1-step episodes). wm nse1 0.0025-0.0034. **Benchmark** (all 64,103 clips, nominal, no
  noise; `scripts/eval_benchmark.py`, results in `logs/<run>/benchmark_ckpt_latest.json`):
  e1 498 falls (0.78%), mean score 0.887; e2 490 (0.76%), 0.882; 442 clips fail under both.
  Floor sitting fails 13.6% (3,128 clips), lying 2/7; everything else < 0.5%. Not yet
  measured: the benchmark with hardware DR on, which is what e2 is for.
- **ONNX export** (2026-10-29): `scripts/export_onnx.py`, one file per clip with the policy
  and that clip's precomputed goal features; mjlab_dash interface (obs, time_step -> actions
  + reference). Three clips exported from `dash_e2_hwdr` (idle, forward walk, slow turn),
  attached to the GitHub release `dash_e2_hwdr`. **Deployment handoff: `docs/DEPLOYMENT.md`.**
- **Velocity interface** (2026-10-30, `--velocity.enabled`, paper Appendix B/G; code in
  `interfaces/velocity.py`, `imagination/rollout.py:imagine_velocity`, `policy/nets.py`):
  - Command (vx, vy) m/s in the heading frame + wz world yaw rate. New `VelocityEncoder` (cmd +
    the 600-d obs history -> 32-d latent, MLP 256x256), shared decoder run with **frozen weights**
    for velocity rollouts (`Policy.decode_frozen`), own `VelocityCritic`, latent alignment MSE
    (weight 0.1) to `enc_motion(goal).detach()` on reference-command rollouts.
  - Sim: the last half of the envs (`track_mask` False on the motion command) follow commands:
    fall-only termination (torso < 0.30 m or 1+u_z > 0.7; Dash stands at 0.67 m), no clip-end
    time-out, reset to upright library frames, excluded from the reset-sampler statistics and from
    the world model's termination-head loss. Commands: 50% the reference clip's (25-frame centred
    mean, read each step), else held uniform (10% zero), resampled every 250 steps.
  - Imagination: each batch is half motion, half velocity. Reference-command velocity rollouts start
    in *tracking* envs (robot in phase with the clip, so the alignment target matches its state);
    uniform-command rollouts start in velocity envs. Fall termination is analytic (sigmoid logits).
  - Dash command ranges (uniform): vx (-0.8, 1.5), vy (-0.6, 0.6), wz (-1.8, 1.8), inside the
    1-99% of the library's reference commands (vx -0.85..1.71, vy +-0.77, wz +-1.86).
  - Warm start: `--init-from <ckpt>` loads world model, policy (velocity encoder new), critic and
    sampler state. Export: `scripts/export_onnx.py --interface velocity` (obs + command -> actions)
    with a closed-loop step test. Tests: `tests/test_velocity.py`.
  - Run `dash_e3_vel` (warm start from e2, stopped at 46k): velocity error 0.11 m/s / 0.11 rad/s,
    no falls, 1 m/s step test 0.99 m/s, **but the gait looks unnatural (both arms held up and
    forward)**; the tracking train return also fell to < 0 from 22k (not benchmarked). Superseded
    by the motion-matching planner below for RC control.
- **Motion-matching planner for RC control** (2026-10-30, SONIC-style: a kinematic planner feeds
  reference motion to the unchanged tracking policy `dash_e2_hwdr`; code
  `src/supermimic/planner/motion_matching.py`, numpy only so it runs on the robot):
  - The goal features hold no global position/yaw, so the planner only chooses mocap frames.
  - Database `data/planner/dash_walk_mm.npz` (`scripts/planner/build_mm_db.py`): neutral
    locomotion clips (walk/turn/idle/start/stop/arc/sideways/backward words only, no jog or styles),
    clip and mirror both benchmarked >= 0.9 by e2, best 3 takes per *numbered* motion
    (walk_forward_loop_001 and _002 are separate motions): 2420 clips, 4.97 h, 150 MB. Speed
    coverage (1-99%): vx -0.61..0.91, vy +-0.57, wz +-1.62. v1 (`_v1.npz`, 3 takes per motion
    family, 730 clips) had too few straight-walking loops: during straight walking it hopped between
    arc/turn clips every 0.2 s and the reference speed sagged once per step (the "hitch" the user saw).
  - Matching features use a smoothed "simulation root" (25-frame centred mean of torso xy and
    unwrapped yaw); the raw torso yaws with every step, which made straight walking look curved.
  - Runtime: spring trajectory from the command, search every 10 ticks (or at clip end) over
    [future root traj 0.2/0.5/1.0 s, foot pos/vel, root vel], jump only if the best cost is below
    0.8 x the continuation cost - 0.02 (`jump_ratio`, `jump_margin`), inertialized feature offsets
    (half-life 0.1 s). Exact KD-tree search (scipy cKDTree, brute-force fallback): step 0.4 ms mean,
    6 ms p99, ~30 ms worst case, so run the search off the control loop on the robot.
  - Sim (`scripts/planner/sim_rc.py`, 34 s scripted sequence, e2 policy): 0 falls; 0.5 s-mean robot
    speed during steady 0.6 m/s walking min 0.62, std 0.039 (v1: 0.47, 0.056). Command errors: forward
    0.6 -> 0.71 (database dominated by ~0.7 m/s natural pace), turn in place 1.2 -> 1.47 rad/s,
    sideways/backward within 0.05. Heavier speed weights fix accuracy but bring back the hopping;
    playback time-warping would be the clean fix (not done).
  - Export: `scripts/export_onnx.py --interface goal` -> `goal.onnx` (obs (1,600) + goal (1,460)
    -> actions), checked closed loop with the planner (3e-6). Tests: `tests/test_planner.py`.
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
