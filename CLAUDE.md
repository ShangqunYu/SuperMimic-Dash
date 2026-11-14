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
- **Effort limits:** leg motors at the hardware peak with raised current since 2026-11-10: hip group (hip
  yaw/roll/pitch, shoulder_pitch) 80, knee 120, ankle 45 Nm (were estimates 50 / 80 / 35; results dated before
  then used those). Arm limit 30 Nm and armature are still estimates, see `dash_constants.py`.

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
    (GPU 1, finished 2026-11-07 ~01:00, 3 h): fine-tune of t2 20k on the v3 mixed library, same
    flags, 20k iterations.
    - v3 terrain: **20/2,402 failures** (t2 31; 16 fixed, 5 new, all leaps/jumps), mean 0.852 -> 0.864.
      Stairs up still **12/12 fail**, but further in: mean progress at failure 61% -> 70% of the clip.
      Failure mode (rollout `logs/terrain_eval/v3/rollout_dash_t3_blind_v3.mp4`): no fall; the robot
      stalls on the first steps while the reference climbs on, until the root error terminates.
      Stairs up are 12 of 66,505 clips (~0.1% of resets at terrain ratio 0.25).
    - Flat (64,103 clips): **134 failures**, mean 0.8883 (t2 174 / 0.8842; t1 100 / 0.8812).

## Teacher with terrain input (2026-11-10, lab server)

Teacher-student plan (user's decision): the teacher policy sees an exact terrain scan and trains in
imagination; a depth student (mjlab_dash `depth.py` pipeline) will be distilled from it in the sim (DAgger).
- **Terrains are heightfields** (`terrain/terrains.py`): per terrain `hf`, cell (i, j) centred on
  `p0 + (i, j) dx`, lookups clamp to edge cells, stored padded by 3 m of edge cells, negative heights
  allowed (pits; the old `clamp_min(0)` is gone). Physics boxes = equal-height cells merged greedily from the
  lowest cell (the env's floor top) up. Old box-tile libraries (`.tiles.npz`) load rasterised at 2 cm.
- **Per-env terrain in the sim** (`terrain/tiles.py`, `MultiClipMotionCommand._place_on_terrain`): every env
  has a floor box + `num_slots` box slots (max boxes over the set; 66 for PARC). At reset the sampled clip's
  boxes are written into that env's world: per-world `geom_size/rbound/aabb/pos` (expanded by the startup
  event `expand_terrain_fields`) and `data.geom_xpos` directly, since mujoco_warp computes static geom poses
  only once at `put_data`. Env origins all stay at 0; clips are stored in their terrain's frame. The tile
  grid (`tile_origins`) is gone, so `scripts/terrain/eval_terrain.py` / `check_tiles.py` still assume the old
  layout and need updating before use on the new library.
- **Scans** (`terrain/height_scan.py`): `WM_GRID` (world model and critic, as before: absolute height / 0.5)
  and `POLICY_GRID` (x -0.25..1.25, y +-0.3, 5 cm, 403 points; root height above each point - 0.65, / 0.5,
  +-2 cm noise with obs noise). `--env.policy-scan` appends it to the goal (env obs term
  `policy_terrain`, same function in `imagine`), `PolicyCfg.terrain_dim` keeps the goal scales right.
  Not with the velocity interface.
- **Library** `data/library/dash_flat_terrain_v3_parc_compact.pt` (`scripts/parc/build_parc_library.py`):
  v3 mixed library + PARC `feasible_loose_peak5` (4,681 clips, 3.56 h, 4,644 distinct terrains, 50 fps):
  71,186 clips, 135.1 h, 4,682 terrains (median 9 boxes).
- Checks (scratchpad `teacher/env_test.py`, 16 envs): slots equal each terrain's boxes, policy scan equals
  the CPU heightfield lookup, reference replay foot penetration <= 2.4 cm. Smoke run (256 envs, 300 it,
  from scratch, hardware DR, terrain ratio 0.4, policy scan): runs, ~3 it/s at that size, 4.5 GB host RAM.
- 2026-11-10: a `read_terrains` bug (re-reading the 64 MB npz array per terrain) took >100 GB host RAM and
  systemd-oomd killed the GNOME terminal running Claude (and with it the user's desktop session). Run memory
  heavy tests under `systemd-run --user --scope -p MemoryMax=32G -p MemorySwapMax=0`.

## Alignment with the official SuperMimic code (2026-11-10)

The authors released their code on 2026-11-10 (github.com/HybridRobotics/supermimic; its env base config is in a
private mjlab fork, so some env details are unverifiable). Same as ours: 4096 envs, UTD 0.4, 1 WM + 1 policy
update, 10x2048 8-step WM windows, H = 32, gamma/lambda, reward terms, network sizes, reset-sampler mix
0.5/0.4/0.1. Adopted from it (all new defaults, each with a switch back):
- **Goal**: `env.goal_stride` 5 (10 frames over 0.9 s instead of 0.18 s) and `env.goal_root_ori` (each goal
  frame's reference root orientation in the robot root frame, rot6d, +6K dims after the frame features; IMU
  attitude only). In imagination it is computed from the predicted root, so BPTT flows through it. Motion ONNX
  export then takes a third input `root_quat` (IMU attitude, yaw aligned to the reference at time_step 0); the
  planner `goal` interface refuses such checkpoints.
- **World model**: `state_spec.delta="accel"` (6 + NJ outputs, velocity increments only; poses integrate
  v + c dv, c = 5/8 for 4 substeps), `state_spec.loss="official"` (Bryson weights 1 cm, 0.02 rad, 0.1 rad,
  3.14 rad/s on root pos / root rot / qj / qd, no root velocity terms, mean over 12 + 2 NJ channels; our old sum
  kept the grad clip always active). The logged `nse` stays the old normalised error, so runs compare.
  `world_model.termination="analytic"`: sigmoid((max(|dh|/0.25, |dtilt|/0.8) - 1)/0.05), no BCE.
- **Networks**: Linear -> LayerNorm -> GELU (`act`, `layer_norm` on world model, policy, critic) and running
  mean/std input normalisers (`input_norm`: WM state features + action, policy obs history + goal), updated each
  env step from sim-collected data only (`training/loop.py`), never in imagination.
- **Optimizer**: `training/optim.py` `Optimizers`: torch.optim.Muon (weight decay 0.1, AdamW-matched lr) on the
  hidden Linear weights, Adam on the rest (`wm_trainer.muon`, `policy_trainer.muon`). `schedule.lr_schedule`
  "constant" (official code; the paper says cosine).
- **Critic**: bootstrap value and GAE targets clamped to max_step_reward / (1 - gamma) = 10.
- **Reset sampler memory**: `memory_mode="cumulative"` (sum over the run of hard^2, official); our old EMA
  forgot within ~100 env steps (a likely cause of t2 losing floor sitting).
- Kept ours: proprio actor with noise and 10-frame history (official default is the privileged noise-free
  state), per-reset DR + hardware DR, terminal transitions in the replay, MC + horizon-normalised sampler
  returns, the heading frame (official uses the body frame), policy loss / H.
- Old checkpoints: `load_policy_from_checkpoint` fills missing policy fields with `LEGACY_POLICY_CFG`;
  `eval.benchmark.goal_env_kwargs(config)` gives scripts the checkpoint's goal layout (old: stride 1, no root ori).
- Smoke run (256 envs, 300 it, teacher flags): WM nse8 0.024 vs 0.34 with the old code, imag return 0.95 vs 0.44.
- **A/B, 10k iterations each** (2026-11-10, `dash_ab_new` = db4ddb4 on GPU 0, `dash_ab_old` = c05ea41 from a
  worktree on GPU 1; both from scratch, 4096 envs, PARC library, hardware DR, terrain ratio 0.4, policy scan,
  total 100k so the old cosine barely decays). Benchmark (`run_benchmark(clip_ids=...)`, nominal, no noise;
  scratchpad `ab_bench.py`, results `logs/ab_eval/`), failures and mean score:

  | | PARC (4,681) | v3 terrain (2,402) | flat sample (2,000) |
  |---|---|---|---|
  | old | 2,589, 0.244 | 1,440, 0.571 | 85, 0.623 |
  | new | **1,396, 0.461** | **68, 0.839** | **48, 0.847** |

  PARC failures new / old: running 208 / 611 (of 1,643), platform 166 / 376 (604), stairs 60 / 147 (198), vault
  56 / 131 (228), jumps 15 / 10 (26), PARC-generated 891 / 1,314 (1,982). v3 box come-up 29 / 1,299. WM at 10k:
  nse8 0.064 / 0.101, terrain nse8 0.16 / 0.19.
- **Blind vs terrain-seeing policy, 10k each** (2026-11-11, `dash_ab_blind`: same code and flags as `dash_ab_new`
  but no `--env.policy-scan`; world model and critic still get the scan; GPU 1, from scratch). Failures, mean score:

  | | PARC (4,681) | v3 terrain (2,402) | flat sample (2,000) |
  |---|---|---|---|
  | seeing (`dash_ab_new`) | 1,396, 0.461 | 68, 0.839 | 48, 0.847 |
  | blind (`dash_ab_blind`) | 1,855, 0.464 | 50, 0.830 | 27, 0.863 |

  PARC failures blind / seeing: jumps 25 / 15 (of 26), stairs 102 / 60 (198), platform 224 / 166 (604), running
  306 / 208 (1,643), vault 61 / 56 (228), PARC-generated 1,137 / 891 (1,982); 1,131 fail under both, 724 only
  blind, 265 only seeing. The scan pays off on PARC (big level changes, gaps, jumps); on v3 terrain and flat
  motions blind is as good or better (one seed each, so ~20-clip differences are within noise).
- **Direction (2026-11-11): terrain-seeing teacher.** Targets: jumping over a small box, stairs up and down.
  Stairs are where the scan helped most (PARC stairs 60 vs 102 failures of 198); box jump-over is too thin to
  tell (v3 jump-over + avoid-obstacle 22 clips: seeing 11, blind 8 failures). A scan teacher keeps every deploy
  layout open (elevation map + teacher, depth student tracker, end-to-end depth student); a blind tracker only
  fits generator -> blind tracker. **Run `dash_teacher`**: resume of `dash_ab_new` 10k -> 100k, same flags, GPU 0,
  started 2026-11-11, 2.9 it/s, log `logs/dash_teacher.out`; benchmark at 30k (`logs/ab_eval/teacher_30k.json`).
  - Library coverage for the targets is thin: stairs have two riser heights (v3 11-12.6 cm, 26 clips on 13
    terrains; PARC 13.5 cm, 198 clips), all about 0.7-0.87 m total climb; box jump-over is 14 clips on one 25 cm
    box (+ 14 avoid-obstacle clips on 3 terrains). v3 stair clips replay cleanly on their terrain (kinematic
    penetration <= 2.1 cm); the stairs-up failures are long climbs failing late (step 110-275 of 230-314).

## US stairs and 25-31 cm box jumps (2026-11-11, lab server)

User targets: climb regular US building stairs (IBC commercial: riser <= 7 in / 17.8 cm, tread >= 11 in /
27.9 cm; IRC residential: <= 19.7 cm, >= 25.4 cm) and jump over a 25-30 cm box.
- **The stair mocap is US stairs shrunk to Dash scale** (x0.677): v3 risers 11-12.6 cm, treads ~17 cm. Real
  stairs are ~1.45x that in both. Dash: leg (hip to ankle) 0.56 m, ~0.6 m to the sole, 40 kg, capsule foot
  18 cm long; 17.8 cm is ~30% of its leg (like a 26 cm step for an adult). It already steps onto 30 cm boxes.
- **`terrain_edit.py` options** (`stairs_us.sh` batches them): `--rise/--run` rebuild the fitted staircase
  (steps keep their indices), `--slow` resamples the source first (stairs_us: run / 0.172, the source's speed
  along the stairs), `--reach` (`reach_root`: the torso moves by the smoothed unreached part of the standing
  feet's targets), `--follow-support` (`support_offset`: torso offset = the standing foot's at mid single
  support, both feet's mean at long double support, linear between), torque use per joint in the scores.
  - Pilot (up_start 101, 17.8/27.9): mean-offset torso alone -> stance IK 5.7 cm, 200 ms unsupported, hip
    p99 1.5x limit. + reach: hip p99 2.7x; the high-torque frames were single support with the COM 19 cm ahead
    of the standing foot (swing feet pulled the torso forward). Standing-only reach + follow-support + 1.6x
    slower: QA passes, hip over its 80 Nm limit 3.2% of the time (the trained 12.4 cm clip: 6.3%), COM 3 cm
    from the support. Inverse-dynamics hip p99 is still 1.2-2.5x the limit on the A301 takes.
  - The joint-speed QA compared a slowed edit with the slowed source; now the source at its own speed.
- **Batch**: 32 walking stair takes (up_start 6, down_stop 6, down_loop 4, + mirrors) x risers 12 / 15 /
  17.8 / 20 cm x treads 25.4 / 27.9 / 32 cm = 384 edits (`data/terrain/edited_us`, scores
  `logs/terrain_edit_us`), **306 pass**: up 12/12 at 12-15 cm, 10-11/12 at 17.8 cm, 3-4/12 at 20 cm (stance
  IK > 4 cm, penetration); down 14-18/20 everywhere.
- **Box jumps** (`jump_boxes_us.py`): only 10 source motions clear >= 27 cm (7 no-hands obstacle jumps, 3 of 4
  running obstacle jumps; the 91 forward broad jumps clear < 27 cm). 336 clips: boxes 25 / 28 / 31 cm high,
  15 / 25 / 35 cm deep, 40 / 100 cm wide, wherever the unedited motion clears by 2 cm.
- **Library v4**: `data/terrain/manifest_v4.csv` (v3 + US stairs + jumps) -> `dash_flat_terrain_v4_compact.pt`
  (67,147 clips, 3,044 terrain) -> `dash_flat_terrain_v4_parc_compact.pt` (+ PARC; 71,828 clips, 136.2 h,
  5,006 terrains). Kinematic replay: stairs max penetration p50 1.5-1.8 cm, max 3.4 cm; jumps 0.
- `init_from` now copies sampler bins by clip name when the library changed (520,835 of 525,557 matched).
- **Run `dash_teacher_v4`** (GPU 0, started 2026-11-11 ~11:15): init_from `dash_teacher` 18k (which resumed
  `dash_ab_new` 10k on v3+PARC), v4 library, same flags (hardware DR, policy scan, terrain ratio 0.4,
  constant LR), 100k iterations. Log `logs/dash_teacher_v4.out`.
- Benchmarks (scratchpad `ab_bench.py --lib`, groups by name; first three as in the A/B):

  | | PARC (4,681) | v3 terrain (2,402) | flat sample (2,000) | US stairs (306) | US box jumps (336) |
  |---|---|---|---|---|---|
  | dash_teacher 18k (v3 lib) | 1,172, 0.494 | 30, 0.860 | 40, 0.877 | 186, 0.622 | 142, 0.572 |
  | dash_teacher_v4 10k (18k + 10k on v4) | 1,088, 0.480 | 27, 0.843 | 33, 0.877 | 65, 0.796 | 10, 0.703 |
  | dash_teacher_v4 30k | 908, 0.511 | 8, 0.879 | 13, 0.893 | 8, 0.839 | 3, 0.745 |
  | dash_teacher_v4 60k | 781, 0.523 | 6, 0.890 | 5, 0.897 | 2, 0.866 | 7, 0.779 |
  | dash_teacher_v4 88k | 723, 0.543 | 20, 0.891 | 9, 0.908 | 8, 0.872 | 0, 0.788 |
  | **dash_teacher_v4 100k (final)** | 710, 0.541 | 8, 0.901 | 4, 0.909 | 1, 0.895 | 0, 0.796 |

  US stair failures by riser (12 / 15 / 17.8 / 20 cm), 18k -> v4 10k: up 19/36 -> 9, 22/36 -> 18, 29/31 -> 13,
  11/11 -> 6; down 20/50 -> 0, 21/46 -> 1, 26/48 -> 8, 38/48 -> 10. Box jumps by height (25 / 28 / 31 cm):
  35/120 -> 3, 44/108 -> 1, 63/108 -> 6. Stairs up stay the weak spot; failures by source take (all sizes):
  A300 001 3/13, 002 8/22, 003 8/18; A301 101 5/18, 102 11/23, 103 11/20 (102/103 have the highest reference
  hip torque).
  At v4 30k: stairs down 0 failures at every riser; up 0/36, 1/36, 3/31, 4/11 (12 / 15 / 17.8 / 20 cm). Box
  jumps 0/120, 1/108, 2/108. Training return 0.600 at 30.8k (2.8 it/s).
  At 60k: stairs fail only 2/11 up at 20 cm (0 at 17.8 cm and below); box jumps 0/120, 2/108, 5/108.
  At 88k: PARC and mean scores still rise (train_ret per 10k: 0.551, 0.577, 0.586, 0.590, 0.593, 0.596, 0.596,
  0.600, 0.606), but failures on the near-solved sets churn between checkpoints (constant LR): vs 60k, 19 new
  come_up_50cm_box failures, 7 new stairs-down, 7 sitting clips; 0 box-jump failures. Pick the final checkpoint by
  benchmark, not by recency.
  The run finished at 100k (2026-11-11 ~20:50, 9.8 h). 100k is the best or within 0.002 of the best checkpoint in
  every group: stairs fail 1/306 (one 17.8 cm climb), box jumps 0/336. Use `logs/dash_teacher_v4/ckpt_0100000.pt`
  as the teacher. Rollout videos (60k): `logs/teacher_videos/v4_60k/` via `scripts/media/render_policy.py`.
- **Run `dash_teacher_v4b`** (GPU 0, started 2026-11-11 ~21:10): train_ret was flat over v4's last 20k
  (0.606 / 0.606 / 0.605 per 10k), so resume v4 100k with a cosine LR decay instead of more constant LR:
  `--resume .../dash_teacher_v4/ckpt_0100000.pt --schedule.total-iterations 150000 --schedule.lr-schedule cosine
  --schedule.lr-decay-start 100000 --schedule.lr-decay-fraction 1.0` (2e-4 -> 6e-6 over 50k; same other flags).
  Log `logs/dash_teacher_v4b.out`; benchmarks every 10k in `logs/ab_eval/teacher_v4b_*k.json`.
  Finished 2026-11-12 ~02:00 (4.8 h). The decay helped far more than more constant LR (PARC failures per 10k:
  constant -13, decay -35 to -89):

  | | PARC (4,681) | v3 terrain (2,402) | flat sample (2,000) | US stairs (306) | US box jumps (336) |
  |---|---|---|---|---|---|
  | v4 100k (start) | 710, 0.541 | 8, 0.901 | 4, 0.909 | 1, 0.895 | 0, 0.796 |
  | v4b 110k | 675, 0.540 | 10, 0.895 | 2, 0.909 | 1, 0.883 | 0, 0.781 |
  | v4b 120k | 612, 0.559 | 13, 0.896 | 1, 0.912 | 0, 0.902 | 0, 0.812 |
  | v4b 130k | 551, 0.563 | 9, 0.908 | 1, 0.917 | 1, 0.919 | 0, 0.811 |
  | v4b 140k | 462, 0.587 | 0, 0.917 | 0, 0.925 | 0, 0.930 | 0, 0.834 |
  | **v4b 150k (final)** | **443, 0.592** | 1, 0.918 | 0, 0.928 | 0, 0.930 | 0, 0.840 |

  **Teacher: `logs/dash_teacher_v4b/ckpt_0150000.pt`.** All US stairs and box jumps pass; PARC 9.5% fail.
- **The benchmark misses horizontal drift (2026-11-12).** "Failed" is root height error > 0.25 m or tilt > 0.8, so a
  robot that skips a move but stays upright passes (video: `logs/teacher_videos/v4b_150k/successes/03_*`, a vault
  where the robot never vaults and ends 2.9 m behind). Max horizontal root offset of every 150k pass (scratchpad
  `drift_check.py`, standalone rollouts with terminations off; `logs/ab_eval/teacher_v4b_150k_drift_*.json`):

  | | passes | max offset > 0.5 m | > 1 m | median max offset |
  |---|---|---|---|---|
  | v3 terrain | 2,401 | 0 | 0 | 0.08 m |
  | US stairs | 306 | 0 | 0 | 0.13 m |
  | US box jumps | 336 | 2 | 0 | 0.11 m |
  | PARC | 4,243 | 2,207 | 1,170 | 0.52 m |

  PARC by kind (passes with > 1 m): vaulting 182/211, jumping 22/24 (median 5.4 m: the jumps are skipped),
  mixed terrain 433/1,029, boxes 208/612, running 256/1,616, platform 68/556, stairs 1/195. The offset grows to
  the clip end (final/max p50 1.0): the robot falls behind and does not catch up. Counting > 1 m as a failure, PARC
  succeeds 66% (not 90.5%); our own mocap sets are unaffected. Training terminates on the same height/tilt rule, so
  skipping a hard move is never punished by termination, only by lower reward.
  Standalone re-runs of the 444 benchmark failures: 411 fail again (33 borderline); 29 fail in the first 15% of the
  clip (reference start likely infeasible).
- **Render cutaway bug (fixed 2026-11-12, physics unaffected):** `render_strobe.section_cut` kept views of
  `d.xpos` (every stored foot position = the last frame's) and trimmed diagonal cuts at the box centre, so terrain
  under the robot vanished in 5 of the first 12 v4b_150k videos (robot "walking on air"). The 12 retarget
  strobe/grid clips were unaffected (checked). MuJoCo `d.xpos[i]` is a live view: copy before storing.
- **Drift-aware benchmark and termination (2026-11-12).** `eval/benchmark.py` records each clip's max root horizontal
  distance from the reference; `summarize` adds `failed_or_drift` (> 1 m). `scripts/eval_groups.py` is the grouped
  benchmark (was scratchpad `ab_bench.py`; same groups, plus PARC by type, stairs by riser, jumps by height,
  per-clip results). `--env.root-xy-threshold M` ends tracking episodes at M m horizontal distance in the sim
  (`bad_root_xy`) and in imagination (`world_model.xy_threshold` follows it: a third analytic soft threshold);
  default off. Console `ends ...` / TensorBoard `train/end_*`: share of episode ends per termination term.
  150k teacher, v4 library (`logs/ab_eval/teacher_v4b_150k_groups.json`; failed / failed or drift > 1 m / clips):
  PARC 428 / 1,638 / 4,681 (vaulting 18/200/228, jumping 2/24/26, teaser 197/638/1,234, boxes 133/346/748, running
  23/300/1,643, platform 52/126/604, stairs 3/4/198); v3 terrain 1/1; flat 0/5 of 2,000 (pass drift p50 0.10 m,
  p90 0.21 m: a 1 m rule barely touches flat clips); US stairs 0/0; box jumps 0/0. PARC failures 428 here vs 443
  in the earlier run: batch composition moves borderline clips.
- **PARC filtered by the teacher**: `scripts/parc/teacher_filter.py` -> `scripts/parc/lists/teacher_keep_1m.txt`
  (3,033 of 4,681: no benchmark or replay failure, drift <= 1 m, no vaulting/jumping; `.dropped.csv` has each
  reason) -> `dash_flat_terrain_v5_parc_compact.pt` (base v4_compact; 70,180 clips, 135.0 h, 3,359 terrains).
  150k teacher on v5 (`teacher_v4b_150k_v5lib.json`): PARC 3 / 60 / 3,033, mean 0.671, pass drift p50 0.41 m p90
  0.80 m (kept by this teacher's results, so near zero by construction; compare return and drift, not failures).
  The rest as on v4. `init_from` bin matching by name: 178 clip names occur twice (flat and terrain copy), only
  the last copy's bins carry over (610 bins start fresh).
- **Run `dash_teacher_v5`** (GPU 0, started 2026-11-12 ~15:15): init_from v4b 150k, v5 library, same flags plus
  `--env.root-xy-threshold 1.0`, 50k iterations, cosine `--schedule.lr-decay-start 10000 --schedule.lr-decay-fraction
  1.0` (2e-4 for 10k, then down to 6e-6). Log `logs/dash_teacher_v5.out`; benchmarks every 10k on the v5 library in
  `logs/ab_eval/teacher_v5_*k.json`. Early: the xy rule ends ~6% of episodes (height 10%, tilt 1%).
  Finished 2026-11-12 ~20:35 (5.3 h, 2.6-2.9 it/s alongside another user's GPU jobs). v5 library, nominal
  (failed / failed or drift > 1 m, mean return; PARC also pass drift p50 / p90):

  | | PARC kept (3,033) | v3 terrain (2,402) | flat sample (2,000) | US stairs (306) | US box jumps (336) |
  |---|---|---|---|---|---|
  | v4b 150k (start) | 3 / 60, 0.671; 0.41 / 0.80 m | 0 / 0, 0.918 | 0 / 5, 0.928 | 0 / 0, 0.930 | 0 / 0, 0.840 |
  | v5 10k | 134 / 177, 0.613; 0.38 / 0.64 | 6 / 7, 0.903 | 2 / 17, 0.910 | 0 / 0, 0.916 | 0 / 0, 0.792 |
  | v5 20k | 140 / 190, 0.617; 0.41 / 0.70 | 13 / 13, 0.901 | 4 / 16, 0.908 | 0 / 0, 0.910 | 0 / 0, 0.798 |
  | v5 30k | 105 / 140, 0.639; 0.37 / 0.59 | 3 / 3, 0.919 | 0 / 8, 0.917 | 0 / 0, 0.906 | 5 / 5, 0.810 |
  | v5 40k | 96 / 114, 0.650; 0.36 / 0.59 | 0 / 1, 0.932 | 1 / 5, 0.922 | 0 / 0, 0.932 | 0 / 0, 0.830 |
  | **v5 50k (final)** | 74 / 89, 0.661; 0.36 / 0.58 | 0 / 0, **0.935** | 0 / 3, 0.926 | 0 / 0, **0.935** | 0 / 0, 0.841 |

  The 2e-4 restart cost precision everywhere for the first ~25k (as v4's constant LR); the decay recovered it.
  150k's 3 kept-PARC failures are low by construction (the list is its own passes). On all 4,681 PARC clips (v4
  library, `teacher_v5_50k_v4lib.json`; same groups as `teacher_v4b_150k_groups.json`): 150k 428 / 1,638, 0.593 ->
  50k 717 / 1,038, 0.576. Kept clips 3 / 55 -> 70 / 86 (return 0.671 -> 0.661; on the 2,958 both pass, return
  -0.004, drift p90 0.80 -> 0.58 m); dropped clips (not trained on) 425 / 1,583 -> 647 / 952: fewer silent skips, more
  honest falls. By type, failed / failed or drift: vaulting 54/210 (was 18/200), jumping 19/23 (2/24), teaser 322/392
  (197/638), boxes 177/214 (133/346), running 77/115 (23/300), platform 63/79 (52/126), stairs 5/5 (3/4). Most new
  kept-clip failures are fast PARC running clips the 150k trailed by 0.5-0.9 m. **Teacher:
  `logs/dash_teacher_v5/ckpt_0050000.pt`.** Videos: `logs/teacher_videos/v5_50k/` (`compare_150k` /
  `compare_v5_50k`: same clips, both teachers; `new_failures`).
- **Feet sink 2-5 cm into terrain boxes (found 2026-11-12, user saw it in a video).** Real, not rendering (the foot
  visual mesh and its collision capsule, r 1.2 cm, share the lowest point). `scripts/diag/sim_contacts.py` reads the
  contacts inside the GPU sim: no overflow (nefc <= 32 of njmax 300), contact normals vertical (sinking into box
  tops), same solref/solimp on floor and slot boxes, slot rbound/aabb written at reset. The reference foot is <= 0.7 cm
  deep, so it is not imitation. Cause: MuJoCo soft contacts (solref 0.02 s at dt 5 ms) overshoot ~ impact speed x
  time constant at touchdown; PARC running / step-downs land harder than flat jogging. v5 50k max depth: running
  clip 5.0 cm, boxes 4.7, teaser 2.7, flat jog 1.6 (p90 on flat 0.5). With `SOLREF=0.01` (stiffest MuJoCo advises
  at this dt): 3.5, 1.0, 1.2, 0.8. Adding solimp 0.95/0.99 trips the line-search iteration limit (overflow 0x400).
  **MuJoCo 4.0.1** (branch `mujoco4`: mjlab main c855b316, MuJoCo / MuJoCo Warp 4.0.1; our env runs unchanged):
  same depths with defaults; MuJoCo Warp rejects `mjENBL_SPECULATIVE` ("unsupported"), and the IPC flag is for flex
  (deformables). Speculative contacts would only stop tunnelling through thin geoms within a step anyway.
- **Stiffer contacts: `--env.contact-timeconst S`** (default None: the model's 0.02 s) sets every geom's solref[0]
  in the compiled model (`ContactMujocoCfg`, sim only). Saved in the checkpoint config: `goal_env_kwargs` passes it,
  so benchmarks, renders and diagnostics play a checkpoint in the contacts it was trained in;
  `eval_groups.py --contact-timeconst` overrides (0.02 = default physics). **Run `dash_teacher_v6`** (GPU 0,
  2026-11-12 22:44 to 2026-11-13 03:30, 4.8 h, 2.9 it/s): init_from v5 50k, same flags as v5 plus
  `--env.contact-timeconst 0.01`, 50k iterations, cosine from the start (`--schedule.lr-decay-start 0`: same library,
  so no 2e-4 phase). Log `logs/dash_teacher_v6.out`. World model one-step error at the start 0.119 (it learned the
  softer contacts), 0.038 by 800 iterations. v5 library, contacts 0.01 (failed / failed or drift > 1 m, mean return):

  | | PARC kept (3,033) | v3 terrain | flat sample | US stairs | US box jumps |
  |---|---|---|---|---|---|
  | v5 50k, contacts 0.02 (its own) | 74 / 89, 0.661 | 0 / 0, 0.935 | 0 / 3, 0.926 | 0 / 0, 0.935 | 0 / 0, 0.841 |
  | v5 50k, contacts 0.01 (start) | 68 / 86, 0.664 | 0 / 0, 0.935 | 3 / 6, 0.924 | 0 / 0, 0.936 | 0 / 0, 0.838 |
  | v6 10k | 111 / 141, 0.649 | 1 / 1, 0.928 | 5 / 19, 0.912 | 0 / 0, 0.924 | 0 / 0, 0.815 |
  | v6 20k | 110 / 125, 0.657 | 3 / 3, 0.931 | 4 / 10, 0.915 | 0 / 0, 0.924 | 0 / 0, 0.819 |
  | v6 30k | 70 / 83, 0.670 | 0 / 0, 0.944 | 5 / 12, 0.920 | 0 / 0, 0.928 | 0 / 0, 0.840 |
  | v6 40k | 67 / 75, 0.686 | 0 / 1, 0.949 | 4 / 8, 0.925 | 0 / 0, 0.935 | 0 / 0, 0.852 |
  | **v6 50k** | **63 / 68, 0.689** | 0 / 0, **0.952** | 4 / 6, 0.927 | 0 / 0, **0.940** | 0 / 0, **0.859** |

  Training randomization (`--randomize`, `*_rand.json`): v5 50k under 0.01 PARC 363 / 519, 0.559 (running 211 / 330),
  terrain 5 / 5, 0.926, flat 4 / 9, 0.914; v6 36k 200 / 368, 0.599; **v6 50k 169 / 342, 0.609**, terrain 1 / 1, 0.943,
  flat 4 / 8, 0.917, stairs 0.931, jumps 1 / 1, 0.826. The fine-tune matters most under randomization: untuned, the v5
  teacher fails twice as many PARC clips in stiff contacts with DR. Foot depth (`scripts/diag/sim_contacts.py`, max / p90 in
  contact), v5 50k in 0.02 -> v6 50k in 0.01: PARC running_006 3.5 / 1.9 -> 1.5 / 0.6 cm, teaser_3269 2.8 / 0.8 ->
  0.9 / 0.25, boxes_456 4.7 / 2.3 -> 1.3 / 0.4, flat jog 1.6 / 0.5 -> 0.6 / 0.15. **Teacher:
  `logs/dash_teacher_v6/ckpt_0050000.pt`** (plays in 0.01 contacts: its config carries it). Videos (same clips, both
  teachers): `logs/teacher_videos/v6_50k/{v5_50k_soft,v6_50k_stiff}/`.
- **Sampler trap: clip-tail bins (found in dash_teacher_v6, fixed 2026-11-13).** Training return fell from 0.60 to
  0.29 at 37.6k with no change in episode length or ends: one bin, the last 6 frames of the flat kneeling clip
  `dash_leaning_idle_003__A246_M`, went to bin return -2.9 (38k) ... -9.2 (50k) and took the whole hard share
  (softmax temperature 25) and 66% of the memory share: ~25% of all resets for 12k iterations. An episode started a
  few steps before a clip's end is mostly the reset transient (previous action zero: action-rate and joint-limit
  penalties; the kneeling pose needs a large first action), and "horizon" normalization multiplies a 2-5 step return
  up to 1 / (1 - gamma) = 100x. Not specific to stiff contacts: v5's estimate of the same bin was +5.2 and drifting
  down. Wider: 35,461 bins (6.8%) start < 25 frames from their clip's end; at v6 30k the hardest bins were PARC
  running clip tails, at 36k 88% of the memory share sat on such bins (v5 50k: one bin held 55%). Fix
  (`commands.py`): returns covering < `min_horizon` (25) steps are not scored, and bins that cannot be scored keep out
  of the hard and memory shares whatever value they carry (so init_from a checkpoint with stale tail values is safe).
  Test `test_env_smoke.py::test_clip_tail_bins_not_scored` (old sampler: the tail bin got 50% of resets). v6 was
  trained with the old sampler; its benchmarks still improved through the trap (it only wasted samples). A run with
  the fixed sampler may train hard clips better: untested.
- **Crowd videos (2026-11-13):** `scripts/media/crowd_picks.py` picks clips the teacher tracks closely (half terrain from
  an eval_groups.py run, half expressive BONES-SEED flat clips benchmarked there: dance, jumps, crawling, combat,
  gestures, injured gaits, ...; `--cells N`, `--per-motion` copies by other actors / PARC augmentations) ->
  `scripts/media/render_crowd.py` rolls them out in one GPU env batch (`--cache` keeps them) and draws them all in
  one scene: a `--cols` x `--rows` grid of 5 m raised tiles, each with its clip's terrain cut to the tile and the
  reference as a green ghost; clips loop with a crossfade. Camera: close on the hero (longest terrain clip, middle
  tile), an outward spiral over the grid (stretched to its aspect), a final pull-back to the whole grid; shadow box,
  haze and near plane follow the camera distance. `--stills t1,t2` writes single frames for quick looks. Lights as
  render_hq; the scene's MjData needs mj_forward (or mj_camlight): mj_kinematics leaves light directions at zero, so
  only the headlight shone (the first crowd videos were dark and flat for that reason). v6 50k, 500 robots (25 x 20):
  `logs/crowd/crowd500_v6_spiral.mp4` (70 s, ~6 min on GPU 1; `_share.mp4` is a 28 MB 2-pass copy for chat).
- Not modelled: stair nosing (2-3 cm overhang; a heightfield can't hold it). More box jump-over mocap (standing
  and running approaches) would add the most variety; the box set reuses 10 motions.

## Terrain-aware motion generator (2026-11-14, lab server)

Plan step 2 (user's decision 2026-11-14): a kinematic generator that sees the heightmap (sim only, privileged),
the v6 teacher tracks its output, and later one depth network on the robot is distilled from generator + teacher
(DAgger in the sim: the sim renders depth, no world model needed). Command: joystick velocity turned into a
desired root path (as the motion-matching planner). Dash's computer is an i7 UP Xtreme (CPU only); it only has
to run the final student.
- **Data** (`src/supermimic/generator/data.py`, `GenData`, all on the GPU): v5 library, every terrain clip plus flat
  locomotion (names made only of walk/jog/turn/idle/... words, the motion-matching vocabulary): 34,402 clips, 56.9 h
  (terrain 5,911 clips, 5.7 h). Window at frame t0: 10 history frames (clamped at the clip start), 60 future
  frames (1.2 s); frame = root position + rot6d in the canonical frame (origin at the current root, x along its
  heading, heights relative to it) + 18 joint positions (27-d); heightmap `GEN_GRID` x -0.6..3.0 m, y +-0.8 m,
  5 cm (73 x 33), terrain height minus current root height; command = the smoothed sim root (25-frame centred
  mean) 20/40/60 frames ahead, xy + heading (cos, sin) in the canonical frame (12-d). Batches half terrain.
  Hold-out: motions whose base name hashes to 0 mod 20 (all actor/mirror/terrain variants; holding out terrain ids
  took 40% of terrain windows, since terrains are shared): val windows terrain 88k, flat 543k.
- **Model** (`generator/model.py`): transformer denoiser, x0 prediction, cosine schedule T = 1000, DDIM sampling;
  tokens = [noise level, command, 50 heightmap patches (3 stride-2 convs), history, future], 2 frames per token;
  d 384, 8 layers, 6 heads, 14.9 M parameters; command dropped 10% (classifier-free guidance possible). Loss: x0
  MSE + 1.0 x MSE of 10 x frame differences (normalised units). `HeightScan.lookup` (point lookups) was
  factored out of `HeightScan.forward` for it.
- **Metrics** (`generator/metrics.py`): foot capsule soles vs the heightfield: penetration, floating in stance,
  sliding in contact. Mocap reference (val windows): terrain penetration mean 0.37 cm, > 2 cm 2.4% of frames.
- **Training** `scripts/generator/train_generator.py` (torch.compile, bf16, AdamW 2e-4 cosine, EMA 0.999, batch 512;
  12 it/s on GPU 0 shared with another user's job). Every 5k steps the EMA model samples 512 val windows (20 DDIM
  steps) and logs foot quality next to the mocap's. Run **`gen_v0`** (200k steps, ~4.6 h, started 2026-11-14),
  log `logs/gen_v0.out`, checkpoints `logs/gen_v0/ckpt_*.pt`.
- **Kinematic test on unseen terrain** `scripts/generator/rollout_generator.py ckpt out [--render]`: procedural
  stairs up (10/14/18 cm x 6, 28 cm treads) and down, 20/30 cm boxes in the way, 30/50 cm platforms up and down,
  flat walk / jog / turn / sidestep; constant joystick command (velocity springs to it, half-life 0.3 s), the
  generator replans every 0.2 s from its own last 10 frames. Success per rollout: no foot > 3 cm into the terrain,
  contact sliding < 0.4 m/s, root 0.35-0.4 to 1.1 m above the ground, past the obstacle on the right level, speed in
  range (turning within 50% of the commanded yaw rate). An untrained model scores 0/30.
- **Closed loop** `scripts/generator/closed_loop.py gen_ckpt teacher_ckpt out [--render]`: same scenes in physics; a
  packed library is written per run (one clip per env on its scene's terrain, sidecar terrains), the generator plans
  every 0.2 s from the robot's actual last 10 frames and its plan (velocities by differences) overwrites the env
  reference ahead of the tracked frame (`library._ref`); the teacher tracks it. Scored on the robot's own motion
  with the same rules; terminations off. ~25 s for 15 scenes on GPU 1. Both take `--guidance` (classifier-free on
  the command) and default to 12 s scenes. `scripts/generator/eval_checkpoints.sh RUN` runs both on every 20k
  checkpoint as it appears (summary `logs/gen_eval/RUN.summary`).
- **Success rules calibrated on mocap** (scratchpad `calib*.py`): mocap on its own terrain has max foot penetration
  p90 2-3 cm (US stairs, come up 50 cm), so the limit is 4 cm; human stair / platform clips move at 0.15-0.26 m/s,
  so scenes run 12 s; "sliding" (horizontal speed of a sole point within 1 cm of the ground) reads ~0.5 m/s on mocap
  flat walking itself (low swing clearance; heel-toe roll) and is only reported, not scored.
- **gen_v0 at 20k** (val: terrain penetration mean 1.2 cm, mocap 0.37): scenes 5/60 kinematic, 11/60 closed loop
  (8 s rules: boxes 7/8 and 30 cm platform up 3/4 in the loop, the teacher stepping where the plan walks through).
  Guidance 2 / 3: 9 / 0 of 60 (heading drifts; worse). Library windows with true inputs (scratchpad `indist.py`):
  the heightmap is used for level changes (stairs / 50 cm come-up: plans climb the right height; a flat map gives
  flat plans), but 28-30% of plans put a foot > 4 cm into the terrain (mocap 0%), and on the US box jumps a flat
  map changes nothing (plans walk into the box: p90 17.5 cm). Joint-space MSE does not pin the feet to the terrain.
- **Foot geometry losses** (`--foot-weight`, `--pen-weight`, `GeoLoss` in train_generator.py, off by default):
  differentiable FK of the predicted clean frames in the canonical frame; sole positions vs the mocap's
  (squared, / 0.1 m) and soles more than 1 cm below the heightmap (linear, / 2 cm; lowest of the 3 x 3 cells around,
  since a nearest-cell lookup next to a step reads the step and mocap feet stand there), each weighted by
  alpha_bar(t). Checks: mocap vs itself 0 / 0.005, mocap + 2 cm noise 0.06 / 0.12, 5 cm down pen 1.2. Compiled
  6 ms per batch (eager 65 ms; first compile ~3 min). Run **`gen_v1`** (GPU 1, from scratch, weights 1.0 / 1.0, else
  as gen_v0), log `logs/gen_v1.out`.
- **Scene results** (12 s, rules above; the floating limit adds the drop of a step-down, since the body crosses a
  ledge before it descends), kinematic / closed loop with v6: gen_v0 20k 5 / 16, 40k 12 / 15 (stopped at 40k);
  **gen_v1 20k 32 / 44** of 60 (held-out terrain penetration mean 0.09 cm vs gen_v0 40k 1.04, mocap 0.37). gen_v1 20k
  in the loop: flat 14/16, stairs up 10 / 14 cm 8/8 (18 cm: reaches 0.9-1.08 m of 1.08 but too slowly), stairs down
  6/8, platform 30 cm up + down 8/8, box 20 cm 4/4, box 30 cm 1/4 (falls), platform up 50 cm 0/4 (stops at the
  wall), platform down 50 cm 3/4. Kinematic plans still walk through the boxes and the 50 cm wall, and clip stair
  edges (10-16 cm "penetration" at the riser; the loss's 3 x 3 slack). Terrain windows by skill
  (`terrain_category`): PARC running 25%, stairs 20%, 50 cm box up 19% / down 15% (clips start at the box, slow),
  box jump-over 8%.
- **Heading drift (gen_v1 40k):** the scene command was a body-frame velocity, so nothing restored the heading: at
  40k the closed-loop robot turned 100-150 deg away from every drop (and ended behind its start), at 20k it drifted
  15-70 deg; the kinematic rollouts drifted less. Scenes now use a heading-holding operator (`operator`: where the
  scene asks for no turning, wz = 2 x heading error, capped at 1 rad/s, as a person would steer) and report the max
  heading deviation; 8 samples per scene (4 was too noisy: 20k -> 40k moved 44 -> 31 / 60 in the loop while held-out
  losses kept improving). Old summaries: `logs/gen_eval/*.summary.4samples_noheadinghold`.
- **`--terrain-balance B`**: terrain windows weighted by (windows of their skill)^-B (0.5: box jump-over 7.9 -> 9.9%
  of terrain windows). Run **`gen_v2`** (GPU 0): gen_v1 + `--terrain-frac 0.75 --terrain-balance 0.5` (box
  jump-over 4.0 -> 7.4% of all windows; flat 50 -> 25%), log `logs/gen_v2.out`.

## PARC terrain data (2026-11-08, lab server)

Source with real terrain, replacing fitted terrain. PARC (Xu, Shi, Yin, Peng, SIGGRAPH 2025; MIT):
`huggingface.co/datasets/mxucg/PARC` `dec_release.zip` (3.9 GB), unzipped at
`/media/shangqunyu/Dash/parc_src/parc_data/dec_release`; repo (character XML, file format) at
`/media/shangqunyu/Dash/parc_src/PARC`.
- Content: 26,396 clips, 29 h at 30 fps (human time). Blocky heightfield, 0.4 m cells.
  - `initial_aug`: augmentations of the 14 min of mocap (stairs, platform, running, gap jumps,
    vaulting, climbing).
  - `*_iter_*`: motions PARC generated and recorded in physics.
  - ~40% of clips use the hands on terrain (wall climbing); Dash can't.
  - No hands and stance-level change <= 0.30 m (Dash scale): 7,003 clips, 6.6 h.
- `scripts/parc/`: `parc_io.py` (load, numpy FK), `parc_retarget.py`, `qa_parc.py`,
  `render_parc.py` (human | Dash side by side), `survey.py`.
- `parc_retarget.py`:
  - Froude scaling: motion and terrain x s = 0.677 (leg ratio), time x sqrt(s), so falls stay at g
    and footholds stay on their steps.
  - Per-frame Levenberg-Marquardt IK, warm started, with:
    - joint and sole targets; hips at Dash's width; feet keep the human footholds;
    - stance per sole end (human end within 3 cm of terrain and slower than 0.5 m/s), each end
      locked in xy where it touched down, at the human's height above the human's step;
    - MuJoCo contact distances vs the terrain boxes;
    - joint limits by an active set.
  - Each frame is solved first without terrain, then with it (a warm start can sit behind a step
    edge).
  - Joint speed bound `QD_MAX` 30 rad/s (Dash time) as per-frame joint bounds in the active set.
    Without it, when the thigh goes past Dash's 90 deg hip flexion limit (high-knee running,
    vaulting), the knee flips straight in one frame to reach the foot target (up to 140 rad/s;
    the human knees peak at ~21 rad/s in Dash time, p90 over clips).
  - ~0.006 s/frame on one core.
- QA of 8 demo clips: penetration <= 1.3 cm, joint speed 17-37 rad/s (one platform clip 66).
  - Unsupported force p95 0.9-2.2 body weights; flat BONES jogging retargets give 0.5-1.6.
  - In this check, a sole end supports only if it is within 2 cm of the terrain and moving slower
    than 0.5 m/s; gating on ankle speed instead dropped running stance.
  - Videos: `logs/parc/parc_dash_{1,2}.mp4`.
- Batch (2026-11-08): `batch_retarget.py list out part nparts` (resumable; 16 processes, ~5 min
  for 7,003 clips), QA with `qa_parc.py` in 16 shards (~2 min).
  - Selection: hands on terrain in <= 5% of frames and largest stance-level change <= 0.30 m at Dash
    scale (`survey.py` columns `hand_frac`, `stance_rise` x 0.677): 7,003 clips. 0 retarget failures.
  - Output `/media/shangqunyu/Dash/parc_dash_30cm/<PARC relative path>.npz` (7.1 GB); QA table
    `qa/qa_all.csv` (column `keep`), kept list `keep.txt`.
  - Kept if pen_max <= 2 cm, IK err p95 <= 0.2, ghost p95 <= 2.5 body weights, knee torque p95 <= 100 Nm:
    6,022 clips, 4.76 h Dash time. By source:

    | Source | Clips kept | Hours kept |
    |---|---|---|
    | running | 1,821 / 2,035 | 1.73 |
    | vaulting (no hands) | 971 / 1,341 | 0.90 |
    | platform | 801 / 955 | 0.45 |
    | stairs | 199 / 200 | 0.21 |
    | jumping | 46 / 62 | 0.05 |
    | PARC-generated (boxes, siggraph terrain) | 2,184 / 2,408 | 1.40 |

  - Torque demand is still above Dash's limits on many clips: median knee p95 76 Nm (limit 80), median
    knee peak 175 Nm. Flat BONES jogging gives knee p95 56-73 and peak 63-89 Nm; BONES jumps peak
    61-141 Nm. Peak joint speed sits at the 30 rad/s bound in most running, platform and vaulting
    clips (flat BONES: 6-14 rad/s).
  - Not yet in training: the sim only has box and stair tiles; PARC terrain is a 0.4 m heightfield.
- `jump_feas.py out.csv clip.npz ...` (2026-11-10): one row per flight phase (both feet off >= 0.12 s): flight time,
  CoM travel and rise, ballistic takeoff speed, CoM accel from a parabola fit (`az_fit_g`), leg torque / effort
  limit in the 0.2 s before takeoff (`to_*`) and after landing (`land_*`). Also reads flat BONES npz. The
  inverse-dynamics ghost force in the air is smoothing bleed (7-frame savgol over ~7-frame flights): ignore it.
  - Results in `qa/flights.csv` (6,022 kept clips, 17,169 flights) and `qa/flights_bones400.csv` (400 flat BONES
    jog/jump clips as the baseline; a tracker follows those). PARC needs far more than BONES: takeoff speed median
    2.9-3.8 m/s (BONES 1.7-2.2), takeoff torque ratio median 1.6-2.3 (BONES 1.2-1.4), clip leg-torque p95 ratio
    median 0.96 (BONES 0.56). Ankles (35 Nm) bind first. Big flights (travel >= 0.9 m or rise >= 0.15 m: 2,300
    clips) median 3.9 m/s and 2.9x limit at takeoff. Froude scaling keeps the human's relative effort; parkour
    humans are relatively stronger than Dash.
  - `qa/feasible_strict.txt` (every flight within takeoff_v <= 3.0, to_leg <= 1.8, land_leg <= 2.0, about the BONES
    p90): 1,129 clips, 0.74 h. `qa/feasible_loose.txt` (3.5 / 2.2 / 2.4): 1,866 clips, 1.22 h. Jumping and
    vaulting nearly all fail; stairs and PARC-generated terrain clips survive best.
  - Raised peak limits (2026-11-10, `--effort hip=80 knee=120 ankle=45`: more motor current): `qa/flights_peak.csv`,
    `qa/flights_bones400_peak.csv`. Same thresholds on the new ratios (the reference-vs-tracker bias is a property
    of the method, not the limits): `qa/feasible_strict_peak.txt` 1,569 clips 1.03 h, `feasible_loose_peak.txt`
    2,547 clips 1.71 h (both with the 3.0 m/s takeoff cap, which 68% of flights exceed). Flights over the torque
    bars drop from 45% to 15% (takeoff) and 24% to 4% (landing). The ankle (+29% only) binds on about two thirds
    of flights. Big flights pass the torque bars 13% -> 39%.
  - Takeoff cap raised to 5.0 m/s (hardware can go faster; no flight exceeds 4.6, so torque alone filters):
    `qa/feasible_strict_peak5.txt` (1.8 / 2.0) 3,663 clips 2.72 h, `qa/feasible_loose_peak5.txt` (2.2 / 2.4)
    4,681 clips 3.58 h. Strict passes 39% of big flights, loose 59%. These are the training lists. dash.py now
    holds the peak limits, so jump_feas without `--effort` gives the same ratios.
- `render_hq.py out_dir clip.npz ... [--human] [--slow 0.5]`: presentation renders (1920x1080, 60 fps,
  8x MSAA, 16k shadow map, aluminum robot, camera on the robot's front-left following its smoothed heading).
  - Shadow/fog notes: a directional light's shadow box is centered on its position, so the key light follows
    the robot; MuJoCo planes get shadow acne, so the ground is a thin box. Clip, fog and shadow
    distances are set in meters (`distances()`); MuJoCo takes them as fractions of each model's extent.
  - Six no-hands clips (2026-11-08): `logs/parc/hq/` (Dash only and `_side` with the human).
- `render_strobe.py out_dir clip.npz ... [--n 8] [--every 0.1]`: fixed side camera (15 deg lens, robot
  travels left to right, frame fits the whole path) for motion-sequence figures. Per clip: 4K video,
  background, robot cutouts (RGBA, full canvas) and shadow layers every 0.1 s, and composites
  `sequence_{solid,fade}.png`. Terrain on the camera side of the path is cut away (`section_cut`).
  Straight-path clips: `survey`/QA filtered, lateral deviation < 0.25 m. Outputs in `logs/parc/strobe/`
  (32 clips). Smooth-clip pick: median forward speed >= 1.3 m/s, max joint accel <= ~650 rad/s^2, no
  backtracking, heading within ~20 deg of the path (else the robot turns toward or away from the camera).
  Mocap-derived `initial_aug` clips are the smoothest; PARC-generated ones pause and jitter more.
- `render_concepts.py out_dir clip.npz [--see 24 --plan 42 --force 50]`: slide concept images on one clip:
  perception (height scan + head-camera depth inset), decision (the next two footholds with timing and
  the swing paths to them, close-up at the stairs; `candidates=True` adds a candidate-foothold grid), motion control (per-foot ground reaction force at the center of pressure
  from inverse dynamics, friction cones, CoM and gravity; arrows drawn in an x-ray pass over the robot).
  Set `m.light_pos` then call `mj_forward` before rendering: the renderer reads `d.light_xpos`, and a
  stale one puts the shadow box elsewhere (acne). Outputs: `logs/parc/concepts/` (stairs_001 aug36).
  Gap version (`logs/parc/concepts_gap/`, boxes_1274_1_opt_flipped_dm: walkway, 0.54 m gap, +0.28 m platform):
  `--cut --plain-ground --inset right --see 32 --aim-level --see-ahead 1.5 --depth-range 0.8 3.0
  --see-view -20 3.6 5 0.8 --plan 41 --plan-view -18 3.3 5 0.6 --force 55 --force-view -18 3.1 5 -0.45`.
  `--aim-level` keeps the head camera aim at the robot's height (aiming at the terrain jumps into the pit).
  Push-off frames give no forces there (toe hangs over the edge, heel only); 55 is the landing (1.7 bw).
  Scan points are composited from a shadowless pass (their shadows streak the walls; alpha < 1 does not help).
  `--still 49 --still-view -14 3.4 5 0.1`: the robot alone mid-stride over the gap (`still_49.png`). Scene flags
  persist across `update_scene`: `shoot` re-enables the skybox the x-ray pass turns off.

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
