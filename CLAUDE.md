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
- **The "50 cm box" clips are on a 30 cm box** (scaled to Dash in retargeting; the names keep the human height):
  come_up / come_down / jump_on relief 0.30 m, US box jumps 0.25-0.31 m. The 50 cm platform scenes are beyond the
  data: gen_v1 40k fails the climb at 0.15 / 0.3 / 0.6 m/s and from 0.3 m away (the 30 cm control 8/8;
  scratchpad `speed_probe.py`). With heading hold and 8 samples: gen_v1 20k 63 / 84, 40k 69 / 86 of 120
  (kinematic / loop; the 13 in-data scenes 81 / 104 in the loop at 40k); gen_v2 20k 55 / 68 (flat turning and
  sidestep 0/8: flat share 25%). Renders: terrain boxes drawn in a plain colour (the floor's checker texture hid
  their tops); `logs/gen_eval/gen_v1_40k_loop_render/grid.mp4`.
- **gen_v1 60k** (heading hold, 8 samples): 62 / 87 of 120; flat now 32/32 but stairs up 4, 0, 0 of 8 (12 of the
  16 14/18 cm runs reach the top or the last step at 0.1-0.15 m/s, too slow for x > 3 m in 12 s; 3/8 of the 10 cm
  runs stall at the bottom). Checkpoints swing scene by scene while the learning rate is high. gen_v2 40k 53 / 72
  (stopped: no flat turning / sidestep, boxes not better).
- **PARC-style augmentation** `scripts/generator/augment.py gen teacher out.pt [--envs 1024 --rounds 8]`: random
  terrains per env (stairs up / down: riser 8-19 cm, tread 24-36 cm, 3-8 steps; box 12-32 cm high, 20-50 cm deep,
  0.6-2 m wide; platform up / down 8-32 cm; 4 cm heightfield, x -1..14 m, y +-1.2 m, 1 m pad), vx 0.3-1.0 m/s with
  heading hold, 12 s through `closed_loop.run_loop` (generator + teacher in physics; now returns robot velocities
  too). Kept: past the feature on the right level, no foot > 4 cm in, root 0.35 to 1.1 m (+ drop) above the ground,
  heading within 30 deg, robot within 0.3 m of the plan. Kept clips are the robot's own motion (sim root / joint
  velocities) as a packed library + terrains sidecar (names `aug/<family>/<params>_v<cm/s>/<seed>_<round>_<env>`).
  Test (gen_v1 60k, 128 envs, 47 s): 92 kept (box 23/39, stairs up 20/31, down 17/20, platform up 13/16, down 19/22);
  kept clips' foot penetration p50 0.9, max 3.7 cm; renders look right.
- **Bug fixed (2026-11-14): `load_terrains` is lru-cached by path**, so rebuilding a library at the same path in one
  process reused the first call's terrains: augmentation rounds 1-7 ran physics on round 0's terrains (kept 2% vs 62%
  in round 0; that output was deleted). `run_loop` now writes `loop_library_<seed>.pt` and clears the cache.
- **gen_v1 instability:** train loss bottomed at ~45k (x 0.043) and rose to 0.068 by 80k at lr 1.4-1.7e-4 (val 0.043 ->
  0.048; scenes 80k 36 / 73 of 120), recovering as the cosine decays (0.051 at 90k). Fine-tunes use a lower lr.
- **Augmentation 1** (`data/library/gen_aug/aug1_v1_60k.pt`, gen_v1 60k + v6, 8 x 1024 envs, ~14 min on GPU 0): 5,023
  clips kept, 16.7 h (box 1,240/2,467, stairs up 994/2,043, stairs down 1,215/1,675, platform up 678/995, down
  896/1,012). gen_v1 100k: 64 / 88 of 120 (recovered). Run **`gen_v3`** (GPU 0): `--init gen_v1 100k`, libraries v5 +
  aug1 with `--terrain-share 0.5 0.5`, foot / pen 1.0, lr 5e-5 (warmup 1k), 60k steps; log `logs/gen_v3.out`.
- **Final gen_v1 200k: 69 / 83 of 120** (kinematic / loop; reached level on stairs up 10/14/18 cm: 6, 5, 1 of 8). gen_v1
  loop by checkpoint: 84, 86, 87, 73, 88, 82, 77, 78, 82, 83 (20k ... 200k). **gen_v3 (aug1 fine-tune) 60k: 69 / 64**:
  box 30 cm and platform 30 cm up 8/8, but stairs up reached 1, 0, 1 of 8, stairs down 10 cm 3/8, sidestep 0/8.
  Cause: every aug1 clip starts from the same settled standing pose (env reset) with the teacher lagging the first
  plans; from that pose in the loop, gen_v3 creeps (stairs: 0.44-0.67 m in 3 s vs gen_v1 0.8-1.1 m), or stands still
  (sidestep; half the stairs-down runs). The kinematic test (library standing frame) does not show it. Fix:
  `augment.py --trim 1.0` (first second of each kept clip cut) and `scripts/generator/trim_library.py` for aug1
  (`aug1_v1_60k_trim1s.pt`, 15.35 h). Augmentation 2 (`aug2_v1_200k.pt`, gen_v1 200k, seed 1, flat family and vy):
  running.
- **Closed-loop test noise** (`closed_loop.py --seed`): gen_v1 200k scores 83 / 90 / 91 of 120 on seeds 0 / 1 / 2 (single
  scenes move by up to 4 of 8, e.g. stairs up 10 cm 4/5/8, box 30 cm 5/7/8). Compare models on 3 seeds; +-5 in the
  total is noise. Augmentation 2 (`aug2_v1_200k.pt`, trimmed): 5,874 clips, 17.95 h (box 1,349/2,144, flat
  1,252/1,252, stairs up 916/1,794, down 1,040/1,373, platform up 604/849, down 713/780). Run **`gen_v4`** (GPU 0):
  `--init gen_v1 200k`, libraries v5 + aug1 trimmed + aug2 with `--terrain-share 0.6 0.2 0.2`, lr 5e-5, 40k steps.
- **gen_v4 20k, 3 seeds: 249 / 360** (gen_v1 200k 264): sidestep 0/24 (it sidesteps at ~0.2 m/s for a 0.4 command:
  outside the 30% speed rule), terrain slightly better (stairs 10 cm 20 vs 17, box 30 cm 22 vs 20).
- **Stall at the top of tall stairs** (both models, 14 / 18 cm x 6): about one step per second up to the last step,
  then the *plan* stays put for 3-6 s (plan = robot). Not the mocap (US stair clips keep 0.26 m/s to their end) nor a
  lack of tall stairs in the augmentation (307 kept climbs >= 90 cm). The generator alone reaches the top 8/8 but its
  soles cut the stair edges by 12-18 cm on the way: in physics the last riser blocks the robot and, replanned from that
  state, the plan stalls. The penetration loss's 3 x 3 minimum on the 5 cm heightmap leaves ~5 cm of slack at every
  edge. **`--pen-exact`**: soles checked against each terrain's own heightfield (2-4 cm cells; gradient through the sole
  height). On data: clean clips 0.0023 (3 x 3 min 0.0034), soles 6 cm forward (into risers) 0.073 (0.012), 5 cm down
  1.26 (1.16). Run **`gen_v5`** (GPU 1) = gen_v4 + `--pen-exact --pen-weight 2.0`.
- **gen_v4 40k, 3 seeds: 230 / 360** (20k 249, gen_v1 200k 264): stairs up 10 cm 22/24 (best), but walk + turn left 8/24
  (5 of 8 seed-0 runs turn *right*, yaw -0.48 for +0.6 commanded) and sidestep 0/24 (0.04-0.18 m/s for 0.4). Forward
  walking fine. Fine-tuning on the robot's own executed motion weakens command following (it mostly continues its
  current motion); from a standstill with a turn command the plan follows the robot's first drift. Run **`gen_v6`**
  (GPU 0) = gen_v1 200k + `--pen-exact --pen-weight 2.0`, mocap only (v5 library), lr 5e-5, 40k: the edge fix alone.
- **gen_v5 / gen_v6 at 40k, 3 seeds (loop, of 24 per scene; reached level in brackets):**

  | | total | turn | sidestep | stairs up 10 / 14 / 18 cm | platform up 30 | platform down 30 / 50 | box 30 |
  |---|---|---|---|---|---|---|---|
  | gen_v1 200k | **264** | 17 | 24 | 17 (19) / 2 (19) / 0 (3) | 19 | 24 / 21 | 20 |
  | gen_v4 40k (aug) | 230 | 8 | 0 | 22 / 2 / 0 | 17 | 23 / 20 | 18 |
  | gen_v5 40k (aug + exact) | 215 | 5 | 0 | 20 (23) / **16 (24)** / 1 (**16**) | **23** | 10 / 2 | 20 |
  | gen_v6 40k (exact, mocap) | 250 | **24** | 19 | 12 (18) / 0 (5) / 0 (1) | 13 | 23 / 22 | 17 |

  The stair stall is fixed only by both together (exact penetration + augmented data); the exact loss on mocap alone
  does not climb. gen_v5 also slows at drops (kinematic platform down 0.32 m/s for 0.6: learned hesitation).
- **Why augmentation kills turning and sidestepping (2026-11-14):** kinematically gen_v4 / v5 turn and sidestep 8/8 at
  the commanded speeds; only in physics (history = the robot's own motion) they go straight. Every aug1 / aug2 run
  held one command for 11 s, so the aug windows hardly turn or sidestep (turning > 0.3 rad within 1.2 s: aug 5-9% of
  windows, mocap flat 26%; sideways > 0.2 m: 5-16% vs 36%; scratchpad `cmd_predict.py`): physics-looking history
  came to mean "keep going straight". **`augment.py --vary`**: the joystick changes every 2-4 s
  (`command_schedule`: flat stand 15%, turns at a set rate, steering to new headings up to 90 deg away, sidesteps to
  0.5 m/s, backwards; terrain speed changes, sidesteps to 0.3 m/s and headings up to 20 deg off the course),
  `operator` takes a target heading, `run_loop` takes a per-step schedule (n, T, 4). New keep rules: heading within
  30 deg of the target once settled, flat runs must follow each command segment (`follows`, one miss allowed), box
  and platform-down runs at least 0.4 of the commanded speed, the strip widened to y +-2 m (kept within 2.4 m).
  Test (gen_v6, 128 envs): flat windows turn 32%, backwards 10%, stand 17%. **aug3a** (gen_v5: stairs up / box /
  platform up / stairs down) and **aug3b** (gen_v6: flat 0.7, platform down, stairs down), 8 x 1024 envs each.
- **Root disk full (2026-11-14 08:45, shared 913 GB volume):** both aug3 runs died writing their per-round loop
  library (ENOSPC). Freed only own scratch: per-round `loop_library_*` files (augment work dirs, gen_eval dirs), the
  DSMS micromamba env in the scratchpad, `/tmp/torchinductor_shangqunyu`. Terrain sidecars are now written with
  `np.savez_compressed` (aug2's 1,113 MB -> 2 MB, identical on read; the three existing aug sidecars rewritten).
  Check `df -h /` before big writes; the disk is near full from all users. `run_loop` now deletes its working library.
- **aug3a** (`data/library/gen_aug/aug3a.pt`, gen_v5 40k, --vary, seed 11): 6,038 clips, 18.45 h (box 1,688/2,450,
  platform up 1,179/1,251, stairs down 1,243/1,604, stairs up 1,928/2,887). **aug3b** (`aug3b.pt`, gen_v6 40k, seed 12):
  4,935 clips, 15.08 h (flat 2,793/5,810, platform down 1,132/1,189, stairs down 1,010/1,193). aug3b flat windows: turning
  > 0.3 rad 32%, sideways > 0.2 m 24%, backwards 8%, standing 25% (scratchpad `cmd_by_family.py`). Runs **gen_v7** (GPU 0,
  init gen_v6 40k) and **gen_v8** (GPU 1, init gen_v5 40k): libraries v5 + aug3a + aug3b, `--terrain-share 0.5 0.2 0.3`,
  `--pen-exact --pen-weight 2`, lr 5e-5, 40k steps (aug1 / aug2 left out).
  - **First try was launched without `--foot-weight 1.0`** (default 0): the exact penetration term alone lifted every
    plan ~8 cm on all terrain (root z error at zero noise +7 to +11 cm, kinematic scenes 29-34 / 120; logged loss_foot
    0.31 vs 0.02 because it is logged unweighted). Discarded (logs `gen_v7.out.foot0`, `gen_v8.out.foot0`);
    train_generator.py now refuses `--pen-weight` without `--foot-weight`. Relaunched 2026-11-14 ~10:10 with foot 1.0.
- **gen_v7 / gen_v8 40k, 3 seeds: 236 / 202** (old scoring): gen_v7 climbs (stairs 18 cm reached level 20/24, platform
  up 30 cm 24) and turns again (16), but sidestep 0/24 and platform down 50 cm 3/24; gen_v8 (from gen_v5) worse.
- **Stalls came from the command path, not the models (2026-11-14):** replanned offline from gen_v7's recorded robot
  history (scratchpad `replan_probe.py`): from a standstill the plan reaches the commanded 1.2 s point but moves only
  ~0.1 m/s over its first 0.2 s; the robot executes only that and lags it, the next plan starts from the slower robot,
  and the speed settles near zero (sidestep stops after ~3 s with plans still asking for 0.3 m; 50 cm drop approached
  at 0.1-0.3 m/s and stalled at the edge). `command_path`'s spring (velocity from the robot's to the joystick's,
  half-life `VEL_HALFLIFE` 0.3 s) let plans start late. Physics-augmented data makes it worse (the robot lags every
  plan and the generator learns that lag). `closed_loop.py --halflife` / `--replan` (`augment.py --halflife`).
- **Turn scene scoring fix:** speed was the chord over the second half; at 0.6 rad/s for 6 s the chord is ~half the
  path, so turning at the commanded rate failed the speed rule (turning less passed). Now path length with the root
  sampled every 0.5 s. All runs rescored (scratchpad `rescore_agg.py`). **Closed loop, 3 seeds, new scoring (of 24;
  reached level in brackets):**

  | | total | turn | sidestep | stairs up 10 / 14 / 18 cm | platform up 30 | platform down 50 | box 30 |
  |---|---|---|---|---|---|---|---|
  | gen_v1 200k, half-life 0.3 | 271 | 24 | 24 | 17 / 2 / 0 (19 / 19 / 3) | 19 | 21 | 20 |
  | gen_v1 200k, 0.15 | 311 | 20 | 24 | 22 / 23 / 15 | 17 | 24 | 22 |
  | gen_v5 40k, 0.3 / 0.15 | 216 / 262 | 6 / 2 | 0 / 5 | 20 / 22 / 20 (0.15) | 23 | 2 / 7 | 22 |
  | gen_v6 40k, 0.3 | 250 | 24 | 19 | 12 / 0 / 0 (18 / 5 / 1) | 13 | 22 | 17 |
  | gen_v6 40k, 0.15 / **0.1** / 0.05 | 299 / **310** / 314 | 24 / **22** / 14 | 24 | 21 / 23 / 22 (0.1) | 12 | 24 | 18 |
  | gen_v7 40k, 0.3 / 0.15 / 0.1 / 0.05 | 243 / 296 / 296 / 290 | 23 / 22 / 19 / 13 | 0 | 23 / 23 / 22 (0.15) | 24 | 16 / 18 / 20 | 23 |

  Replan every 0.4 s (`--replan 20`) instead: gen_v6 275, gen_v7 239 (turning drops). **`VEL_HALFLIFE` now 0.1**
  (default for the kinematic test, closed loop and augment). gen_v6's platform-up misses are a deep crouch (root
  0.37-0.40 m over the edge vs the 0.4 rule, all 24 reach the top); gen_v7 (aug-trained) steps up tall (0.44-0.6).
  gen_v7 sidesteps at 0.15 m/s for 0.4 even at 0.05 (the aug3b sidesteps were slow: v6 at 0.3, `follows` passes
  0.16 m/s for 0.4). platform up 50 cm: 0 for every model. **aug4a** (gen_v7 40k, terrain families as aug3a, seed 13)
  and **aug4b** (gen_v6 40k, families as aug3b, seed 14): `--vary --halflife 0.1 --follow-tol 0.67` (flat segments
  within 0.08 + 20% sideways), 8 x 1024 envs. aug4a 7,086 clips (box 1,974/2,425, platform up 1,168/1,200, stairs
  down 1,629/1,728, stairs up 2,315/2,839); aug4b 6,116 (flat 3,747/5,790, platform down 1,155/1,173, stairs down
  1,214/1,229). Runs **gen_v9** (init gen_v6 40k) / **gen_v10** (init gen_v7 40k): v5 + aug4a + aug4b, 0.5/0.2/0.3,
  as gen_v7. **40k, 3 seeds (half-life 0.1): 299 / 296** (gen_v6 310): turn 24 / 24, platform up 30 cm 21 / 22, stairs
  23-22-22, but **sidestep 0 / 0** again and platform down 50 cm 20 / 18.
- **Why every aug fine-tune loses the sidestep:** gen_v9 starts at 0.3 m/s, then its own plans slow to 0.13-0.17 m/s
  within ~4 s and the robot tracks them within 1 cm (gen_v6: plans 0.40-0.45, robot 0.38-0.43); kinematically gen_v9
  sidesteps at 0.43. Pure sidestep windows (lateral >= 0.3 m at 1.2 s, forward < 0.15 m; scratchpad
  `side_windows.py`): mocap flat 5.9% (history 0.49 m/s, future 0.61), aug4b flat 2.0% (0.26 / 0.35), aug3b 0.8%:
  with physics-looking history a fast sustained sidestep is almost unseen. **`augment.py --sidestep S`**: that share of
  flat segments a pure sidestep (vx within 0.1, |vy| 0.25-0.6 m/s); 0.25 gives 28% of segments (turning 22%).
  **aug5b** = aug4b with `--sidestep 0.25` (seed 15). Runs **gen_v11** (init gen_v6: v5 + aug4a + aug5b, 0.5/0.2/0.3)
  and **gen_v12** (init gen_v6: v5 + aug4a only, 0.6/0.4: is the flat aug needed at all?).
- **aug5b**: 6,300 clips (flat 3,878/5,735, platform down 1,241/1,254, stairs down 1,181/1,203); pure sidesteps 14.4%
  of its flat windows (history 0.33 m/s, future 0.38). **gen_v11 / gen_v12 40k, 3 seeds: 297 / 295** (gen_v6 310):
  sidestep 0 / 0 (0.14-0.22 / 0.11 m/s), platform up 30 cm 24 / 23, turn 24 / 24, platform down 50 cm 14 / 18.
  gen_v11 guidance 1.25 / 1.5: 298 / 302 (sidestep still 0, drop 50 cm 15 / 19).
- **The aug models' sidestep, closed loop:** offline replans from the recorded history reproduce the executed plans
  (scratchpad `probe_check.py`: lateral -0.275 vs -0.273 m/s at 4 s). gen_v11's plans ask for 0.25-0.28 m/s (for 0.4)
  and the teacher executes 0.12-0.22: each plan starts ~1 cm ahead of the robot and the next plan starts from the robot,
  so the lag is never made up (gen_v6: plans 0.40-0.45, robot 0.38-0.43). Not the data share (gen_v12 has no flat aug
  and is slowest), not the operator's heading hold (wz 0 or reversed changes nothing), not the scene edge (lookups
  clamp: flat). The aug models also yaw 6-9 deg into the sidestep (gen_v6 2-3). Any physics-aug fine-tune so far gains
  platform up / box / turn and loses the sidestep. **Best generator now: gen_v6 40k at half-life 0.1 (310 / 360).**
- **gen_v6 under the teacher's training randomization** (`closed_loop.py --dr`: robot + hardware DR, observation
  noise, no action noise; `--delay`: 0-25 ms actuator delay, which the v6 teacher never trained with), 3 seeds:
  310 -> 294 (dr) / 296 (dr + delay); the delay costs nothing beyond DR. The loss is stairs up 18 cm (22 -> 10 / 9 of
  24): mostly slower climbing (end x 2.5-3.1 m vs 3.2-3.4 m, the scene's line is 3.0 m), 3-4 of 24 not reaching the
  top. Every other scene is within noise.
- **GenData takes several libraries** (terrains appended, tiles offset; `--library A B`) and `terrain_share` (each
  library's share of the terrain windows; `--terrain-share 0.5 0.5`); `--init ckpt` warm-starts model + EMA.
- **`--terrain-balance B`**: terrain windows weighted by (windows of their skill)^-B (0.5: box jump-over 7.9 -> 9.9%
  of terrain windows). Run **`gen_v2`** (GPU 0): gen_v1 + `--terrain-frac 0.75 --terrain-balance 0.5` (box
  jump-over 4.0 -> 7.4% of all windows; flat 50 -> 25%), log `logs/gen_v2.out`.

## Depth student (2026-11-14, lab server)

Plan step 3 (user's decisions 2026-11-14): one network on the robot, trained by DAgger in the physics sim from
gen_v6 (half-life 0.1) + the v6 teacher; no world model (it cannot render depth). Student: a small transformer
over a fixed window (depth tokens, one per 10 Hz frame, ~2 s; proprio + joystick + last action tokens, ~0.5 s),
chosen over a GRU because DAgger then trains on shuffled self-contained windows (no stale hidden states) and RL
fine-tuning later sees a feed-forward policy. RL fine-tuning (task reward + a stay-near-teacher term, critic
warm-up first) is a possible later step if the DAgger student falls short of the teacher.
- **Demonstrator under the student's training conditions**: gen_v6 + v6 teacher with robot + hardware DR, obs noise
  and the 0-25 ms actuator delay: 294-296 / 360 vs 310 (see the generator section). Train the student with DR and
  delay on.
- **Camera** (`assets/dash/dash.py`, `realsense=True`; off by default, the teachers trained without its 92 g):
  the RealSense D435 + jig of mjlab_dash commit db83fd6, camera `robot/realsense_depth`, 50 deg below
  horizontal, fovy 58.9 deg. Standing, the lens is 0.66 m up and sees the ground from ~0.1 to ~1.8 m ahead;
  ~3-7 % of the image is the robot's own arms / legs (the real lens sees them too).
- **Pipeline** (`envs/mdp/depth.py`, env arg `depth=True`, Dash only): ported from mjlab_dash `depth.py` (parkour
  Go1 settings): render 106x60 (the 424x240 stream / 4), crop 60 / 46 of 424 columns, resize to 64x48, clip
  0-2 m, scale to [0, 1]; `process_depth` is the same code on the robot. With `obs_noise`: uniform 0-2 cm noise,
  0 closer than 0.12 m, dropouts, 10 Hz frames held between refreshes (staggered per env), 80-140 ms latency
  (obs delay buffer; a placeholder, measure on the robot). With `domain_randomization`: camera mount +-3 mm,
  +-0.6-1.1 deg, fovy +-1 deg per env.
- **Dropouts (new here, rates are guesses until compared with recorded robot depth)**: pixels at occluding edges
  (second difference of inverse depth > 0.06 /m along a row or column: zero on any plane however grazing, so
  floors, treads and risers never count; drop edges, step lips seen from above, box sides and the robot's
  outline do) lost with p 0.3; with p 0.3 a frame loses 1-3 blobs of 3-10 px (render pixels). A first version
  thresholded the depth jump (5 cm) and flagged all distant flat floor (adjacent rows differ by ~10 cm at 2 m).
- **Check** (`scripts/student/depth_check.py teacher out [--noisy] [--bench N...]`): robots stand 0.8 m before
  each of rollout_generator's 15 scenes' first edge; each rendered pixel vs depth ray-marched through the
  heightfield: p95 error <= 0.21 cm, none > 3 cm, clean and with full DR (per-env camera pose and fovy). The
  renderer gives planar depth (as the D435), and mujoco_warp's BVH refit reads per-world geom_size / geom_xpos,
  so the per-env terrain box slots render. Images: `logs/student/depth_check/depth{,_noisy}.png`.
- **Cost**: env step 18.9 -> 40.2 ms at 4110 envs (17.1 -> 22.2 at 1035); the camera renders every step though
  each env takes a frame every 5th. Rendering only every 5th step (synchronized phases) would recover most of it.

- **Student** (`student/model.py`, 0.88 M parameters): CNN per 64x48 depth frame -> a token (encoded once when the
  frame arrives); 3-layer transformer (128 wide, 4 heads) over a readout token, 20 depth tokens (one per 5 steps:
  2 s) and 25 proprio + joystick tokens (the env's `proprio` group, history-free, + the operator's vx, vy, wz:
  0.5 s); out the 18 joint-position actions. Proprio normalization frozen from round 0's data.
- **DAgger** (`scripts/student/dagger.py`): per round 2048 envs on augment.py's random terrains with its varying
  joystick (12 s), all of the student's training conditions on; the generator replans every 0.2 s from the robot's
  actual motion and the teacher's action is the label at every step; the teacher drives a share 0.5^round of
  the envs (round 0: all). A robot counts as down (later steps dropped) when its root is within 0.3 m of the
  ground or tilted past 60 deg; no resets within a round. Trains on the last 12 rounds (uint8 depth; ~11 GB GPU):
  1500 steps per round, batch 1024 windows, AdamW 3e-4, cosine to 0.1x. A round: ~127 s collecting (84 s of it
  the generator's 60 plans at 2048 envs, 20 DDIM steps) + 45 s training; 40 rounds ~2 h.
- **Eval** (`scripts/student/eval_student.py ckpt out [--clean]`): the student alone (joystick + proprio + depth;
  no generator, no teacher) on rollout_generator's 15 scenes, scored as closed_loop.py, default noisy.
- **dagger_v1** (augment.py's family mix, flat 15%): eval per 5 rounds 35, 64, 90, 98, 97, 93, 96, 98 / 120 (flat
  command following came last: round 10 walked 0.4 m/s for 0.6, turned the wrong way, sidestepped 0.15-0.25 m/s;
  fixed by round 20 without changing the mix). Student-driven envs up 96 % by the end, action MSE to the
  teacher 0.048. Final, 3 seeds (noisy): **284 / 360** vs the generator + teacher's 294 under the same DR.
  Equal or better on flat, stairs down, drops and 10 cm stairs; short on stairs up 14 cm (15 vs 20 of 24), 18 cm
  (6 vs 10; reached the top 22 / 24, slow) and platform up 30 cm (14 vs 17).
- **dagger_v2** (flat 0.35, stairs up 0.2, down 0.12, box 0.15, platform up 0.1, down 0.08; sidestep segments 0.25;
  seed 1): ahead early (54 vs 35 at round 5). 3 seeds (eval seeds 100-102, noisy): round 25 **292**, 35 290, 40 291
  / 360, i.e. the demonstrator's 294: DAgger has converged on it. Round 25 vs generator + teacher: stairs up 18 cm
  15 vs 10, sidestep 22 vs 23, platform up 30 cm 11 vs 17 (round 40: 21, but sidestep 14). **Best student then:
  `logs/student/dagger_v2/ckpt_024.pt`.** Clips: `logs/student/eval/v2_024_render/`.
- **Eval noise**: the same checkpoint and eval seed gave 98 and then 107 / 120 (GPU physics is not bit-reproducible),
  so +-5-10 per 120 between runs; compare on 3 seeds and treat < ~10 / 360 as a tie.
- **CPU** (eager PyTorch, this Threadripper 9960X, one robot): transformer 1.4 ms + depth encode 0.6 ms every 5th step
  at 1 thread (0.9 + 0.5 ms at 4). The UP Xtreme's i7 is slower per core, but the 20 ms step leaves a wide margin.
- **Speed (2026-11-15).** The v2 student crawled in the videos because its commands did (terrain vx <= 1 m/s); the
  generator + teacher cross the scenes at 1.3-2 m/s (scratchpad `speed_probe.py`). **`dagger.py --courses`**
  (`scripts/student/courses.py`): 28 m courses of 1-4 features in a row (stairs, box, platform, rough blocks, slopes,
  gaps up to 40 cm), joystick vx 0.3-2 m/s (half the segments >= 1 m/s), 20% flat. **Fast suite** (`eval_student.py
  --suite fast`, `courses.fast_scenes`): the terrain scenes at 1.5 m/s plus unseen rough blocks 0-6/12/20 cm, slopes
  10/15 deg, gaps 20/40 cm, 7 s; success = past the line, never tilted past 60 deg, standing at the end.
- **dagger_v3** (`--courses`, init v2 round 25, beta0 0.25, seed 2, 30 rounds). 3 seeds, slow suite (of 360) / fast
  suite (of 480): v2 r25 292 / 266; v3 r15 228 / 406, **r20 265 / 403**, r25 249 / 405, r30 256 / 402. Fast scenes
  crossed at 1.1-1.8 m/s. Lost on the slow suite: stairs down 10 / 14 cm at 0.5 m/s 24 / 24 -> 17 / 9 (rushes down in a
  deep crouch, root 0.22-0.39 m above the ground, below the 0.4 rule), sidestep 22 -> 15, stairs up 18 cm 15 -> 11;
  gained platform up 30 cm 11 -> 16. **dagger_v4** (2026-11-15 12:50-14:20, GPU 0): `--courses --course-share 0.5`
  (half the envs on courses, half on v2's families at <= 1 m/s), init v3 r20, beta0 0.25, seed 3, 30 rounds; log
  `logs/student_dagger_v4.out`. 3 seeds, slow / fast: r15 281 / 402, r20 285 / 403, r25 281 / 379, **r30 296 / 397**
  (v2 292 / 266, v3 403 at best): slow stairs down back to 24 / 23 of 24, platform up 30 cm 20 (best yet). Weak spots
  left: stairs up 18 cm (slow 14, fast 4 of 24), rough blocks 0-20 cm 11, platform up 50 cm 0 (beyond the data).
  Best student then: `logs/student/dagger_v4/ckpt_029.pt`.
- **Course demo** (`scripts/generator/course_demo.py`, generator + v6 teacher, clean physics unless `--dr`): a 2 m wide
  course of 5 stairs up (10 / 14 cm) onto a deck, a 30 / 50 cm gap in the deck, a 20 / 30 cm platform up, stairs down
  (<= 15 cm), run-out; joystick vx held. 8 variants x 8 runs, success (whole course, never down) / mean speed over the
  course: 1.5 m/s 62 / 64 @ 1.5; 2.0 m/s 62 / 64 @ 1.9; 2.5 m/s 63 / 64 @ 2.3; 3.0 m/s 60 / 64 @ 2.7 (the 14 cm / 50 cm
  / 30 cm variant 8 / 8 @ 2.66, best 2.76). At 1.0 m/s most runs do not finish in 10 s and some stall or fall at the
  gap (a 50 cm gap needs a stride from speed). With robot + hardware DR, obs noise and the 0-25 ms delay: 1.5 / 2.0 /
  2.5 m/s 51 / 47 / 41 of 64 (falls spread over stairs up, stairs down and the 30 cm platform). Videos (1080p,
  `--render --pick V S K [--ghost]`): `logs/course_demo/clean/course_v7_2.0ms_6.mp4`,
  `logs/course_demo/fast/course_v7_3.0ms_7{,_ghost}.mp4`.
  The v4 r30 student alone (`--student ckpt`; with `--dr` also the depth noise, dropouts, latency and camera DR), of 64
  at 1.5 / 2.0 / 2.5 m/s: clean 38 / 49 / 41, DR 33 / 24 / 3 (generator + teacher: 62 / 62 / 63, DR 51 / 47 / 41).
  Clean failures are mostly at the gap: 50 cm is wider than its training gaps (<= 40 cm) and here the gap is in a
  raised deck; at 1.5 m/s the 50 cm gap fails more (no run-up). Under DR at 2.5 m/s (beyond its 2 m/s training
  joystick) it falls on the stairs up. Videos: `logs/course_demo/student_v4_clean/course_v7_2.0ms_1.mp4` (14 / 50 /
  30 cm, 1.92 m/s), `logs/course_demo/student_v4_dr/course_v4_2.0ms_5.mp4` (14 / 30 / 20 cm, DR, 1.95 m/s).
  17 / 18 cm risers (`--rises 0.17 0.18`, `logs/course_demo/tall_*`), of 64 at 1.5 / 2.0 / 2.5 m/s: generator +
  teacher clean 59 / 45 / 56, DR 48 / 37 / 29; student clean 51 / 36 / 22, DR 25 / 5 / 0. The added falls are
  nearly all on the stairs up. The DR student trips at the 1st-2nd riser (x 1.7-2.1 m, 1.3-1.4 s in, still
  accelerating), the clean one at the 3rd-4th. Depth ablation at 2 m/s on 17 / 18 cm, student `--dr` with one part
  of the depth corruption off (scratchpad `depth_ablation.py`, `logs/course_demo/abl_*`), of 64: all on 5, no
  latency 10, no noise / dropouts 8, neither 21 (no DR at all 36; generator + teacher DR 37, clean 45). So the
  student loses more than the teacher to the robot DR (36 -> 21 vs 45 -> 37), and the depth noise and latency
  compound (either alone off recovers little).
- **RL fine-tuning** (user's decision 2026-11-15: keep the 80-140 ms / 10 Hz depth, PPO with a DAgger term that
  decays; as PHP, arXiv 2602.15827: 60-80 ms delay, 30 Hz depth, lam_D max(0.1, 1 - k / (K/2)), DAgger coef 10,
  lr fixed then KL-adaptive). `scripts/student/rl_finetune.py`: dagger.py's rounds (no resets, done once down) with
  the student acting everywhere (Gaussian, learned std from 0.1), reward = the env's tracking reward against the
  generator's plan (the teacher's Table A.2 terms, x dt: ~0.085 / step of 0.1 max), critic an MLP on the teacher's
  inputs (policy history + goal + scan, 1523). Pitfalls found in smoke tests: StudentRunner caches each depth
  frame's CNN token for 2 s, so after an actor update the acting policy used stale tokens (first-minibatch
  |ratio - 1| 0.04-0.12; fixed by re-encoding the window after each update, plus cudnn TF32 off: 0.016); PPO at a
  fixed 5e-5 moves the policy to KL ~0.2 per update at std 0.1 (clip fraction 0.6), hence the adaptive lr from
  update 150, bounded 5e-6..1e-4. Update cost: the transformer, not the CNN (3.2 s per update of 2048 x 24 steps,
  5 epochs x 4 env-grouped minibatches), but make_env's `configure_torch_backends` turns cudnn autotuning on and
  a new unique-frame count per minibatch re-tuned the CNN every step (12 s per update): off after each env build.
  First try (`logs/student/rl_v1_chunkdagger/`): the DAgger term on the PPO chunk itself. The robots of a round
  are in step (no resets), so a chunk is one 0.5 s slice of every round; fitting the teacher on it chased that
  slice: after one round of actor updates (25, lam 0.99, lr 5e-5) the slow suite fell to 76 / 120 (stairs up
  14 / 18 cm 0 / 8, stairs down 14 cm 2 / 8). Now the DAgger term draws windows from a replay of the last 8 rounds
  (dagger.Replay), PPO stays on the chunk. **rl_v1** (GPU 0, from v4 r30, seed 4): v4's mix (`--courses
  --course-share 0.5`, v2 families) with `--vmax 2.5 --tall-stairs 0.5 --gap-max 0.5`, 100 rounds, critic-only
  rounds 0-2, lam_D to 0.1 by round 53; log `logs/student_rl_v1.out`, run `logs/student/rl_v1/`.
  Result (2026-11-15/16, 7 h): the adaptive lr sat at its 5e-6 floor from update 150 on (KL ~0.016 even there);
  training share up 0.86 -> 0.90, reward 0.0875 -> 0.0888 / step. 3 seeds, slow / fast (v4 r30 296 / 397):
  r43 286 / 423, r63 280 / **432**, r71 287 / 421, r83 272 / 427, r99 261 / 414. Gained at speed: stairs up 18 cm
  4 -> 7-10, rough blocks 0-20 cm 11 -> 16-22, stairs up / down 14 cm, gaps; lost at 0.5 m/s: sidestep 20 -> 0-17
  (r99 0 / 24), stairs up 10 cm 23 -> 14-20. Tall-stairs course, DR, 1.5 / 2.0 m/s of 64: v4 25 / 5, r63 20 / 13,
  r83 26 / 14 (stairs-up falls at 2 m/s 52 -> 31; more falls at the 50 cm gap, reached at ~1 m/s after slowing on
  the stairs; generator + teacher 48 / 37). Clips: `logs/course_demo/tall_rl1_083_dr/*.mp4` (2 successes, a trip
  on the stairs, a fall into the gap).
- **The generator's plans cut through stair edges** (scratchpad `foot_clip.py`: FootMetrics sole points vs the
  terrain under them, tall-stairs course at 2 m/s, stairs-up zone x 1.3-3.2 m): the plan's sole goes > 3 cm into a
  step in 56 / 64 (clean) and 60 / 64 (DR) runs, worst depth 10-11 cm on average (flat run-up control: 3 / 64,
  0.9 cm). The teacher's robot stays out (worst ~3-4 cm, the physics stops it), so the teacher partly ignores its
  reference at the edges, and the RL tracking reward pays for following a colliding plan.
  By phase (scratchpad `foot_clip2.py`): 17 % of the plan's *stance* frames on the stairs have a sole > 2 cm inside a
  step (7 cm deep), 10 % of swing frames: mostly misplaced footholds (toe into the riser / foot below the tread),
  not swing clearance.
- **Foot guidance at sampling time does not help** (2026-11-16; `generator/guidance.py` FootClearance, a
  `Diffusion.sample` `x0_hook`; `closed_loop.run_loop(foot_guide=...)`, `course_demo.py --foot-guide`; off by
  default). Gradient steps on each late denoising step's x0 against the squared sole depth below the terrain (optionally
  the max within `--guide-inflate` m). Tall-stairs course, generator + v6 teacher, 2 m/s, of 64 (clean): no
  guidance 48; joints only (legs fold, the root height stays the plan's) 39 / 41 (inflate 3 / 0 cm; the body sank
  0.633 -> 0.622 m on the stairs, 22 stairs-up falls); root + joints 48 (3 cm) / 54 (0 cm). Under DR, 3 seeds, of
  192: no guidance 139 (1.5 m/s) / 108 (2 m/s); root + joints, no inflation, last 4 steps x 5 iterations 132 / 92;
  all 20 steps x 1 iteration - / 83. The guided plans are clean (worst depth 11 -> 0.1-3 cm) but the teacher tracks
  them worse: it was trained on the generator's / mocap's motions and compensates for their foothold errors itself;
  a corrected plan is out of its distribution. Planning ~2.6x slower. Fixing footholds would need the tracker
  retrained on such plans (or the generator's data / training); the student-side ideas (RL stumble / foothold
  rewards, an auxiliary terrain-prediction loss) do not depend on it.
- **rl_v2** (2026-11-16, GPU 0; user's go-ahead for the student-side ideas): from rl_v1 r71 (its std and critic
  too), `--lam0 0.1` (DAgger weight at the floor from the start), `--lr 5e-6 --adaptive-after 0`, 60 rounds,
  critic warm-up 3 (the aux head fits on the frozen trunk meanwhile), `--gap-max 0.6`, rest as rl_v1. New terms:
  `--edge 5` (per m of sole below the highest ground within 4 cm: toe against a riser) and `--foothold 1` (per foot
  in contact straddling a > 3 cm step), both x dt like the env's reward (tracking ~0.088 / step); smoke-test rates
  at r71: 0.004 m and 0.023 straddling feet per step. `--aux 1`: a head on the readout features predicting the
  teacher's noise-free POLICY_GRID scan (403, int8 cm targets in the replay; MSE in the teacher's input units,
  0.11 at the start = ~17 cm RMS). Log `logs/student_rl_v2.out`, run `logs/student/rl_v2/`.
  Result: the aux head learned the terrain (MSE 0.11 -> 0.016, ~6 cm RMS) but the foot terms barely moved
  (straddling feet / step 0.052 -> ~0.045, edge 0.009 -> ~0.0065); lr at its 5e-6 floor again (KL ~0.015). 3 seeds,
  slow / fast (rl_v1 r71 287 / 421): r31 274 / 408, r47 260 / **426** (fast stairs up 18 cm 16 / 24, best yet),
  r59 239 / 420. Tall-stairs course DR, 1.5 / 2.0 m/s of 64 (rl_v1 r71 25 / 8, r83 26 / 14; generator + teacher
  48 / 37): r31 27 / 7, r47 21 / 15, r59 **35** / 13. **Reward hacking at slow speed**: r59's slow stairs up
  10 / 14 cm fell to 4 / 6 of 24 with no falls; 13 of 24 runs on 10 cm stairs stop within 0.9 m of the start,
  before the first riser. The foot penalties make nearing a step costly, and the tracking reward hardly charges
  for stalling, since the generator replans from wherever the robot is every 0.2 s (a robot standing still trails
  each plan by ~10 cm at 0.5 m/s). Any further RL needs a progress / command-following term (and smaller foot
  weights), and probably its own lr control apart from the DAgger + aux terms.
- **PHP's timing, from its code** (github.com/amazon-far/php_parkour, Holosoma pinned at 4a0bf04; checked
  2026-11-16): policy 50 Hz (IsaacSim 200 Hz, `control_decimation=4`; inference `rl_rate=50`). Depth in training
  is rendered every policy step into a rolling buffer and read `latency_frame (3, 4)` steps back (60-80 ms); no
  camera frame rate or frame hold is modeled (the paper's "30 Hz" is the robot's camera; on hardware the policy
  takes the latest frame from shared memory each step). 58x87 input, D435i on the torso pitched 27 deg. Ours:
  a frame every 5 steps (10 Hz), 4-7 steps late (80-140 ms), so up to 220 ms old. PHP's student is memoryless:
  `DepthOnlyFCBackbone58x87Small` (3 conv + GAP -> 32, num_frames 1) on the single delayed image, concatenated with
  this step's proprio + one-hot command (`history_length=1`), MLP 2048-1024-512-256-128; no GRU, no frame stack.
  PPO `init_noise_std=0.01` (ours 0.1), 2 epochs x 96 minibatches, `dagger_loss_coef 10`, `dagger_end_epoch 10000`.
- **dagger_v5** (2026-11-16, GPU 0; user's choice: PHP's camera timing, the last 10 frames): camera a frame every
  step, 3-4 steps late (60-80 ms; `--depth-refresh 1 --depth-latency 3 4`, saved in env_kwargs); student tokens
  every step, 10 of them (0.2 s; `--depth-every 1 --n-depth 10`), proprio window unchanged (25 steps). Init rl_v1
  r71 (slot embeddings interpolated by frame age, `model.adapt_slots`), beta0 0.5, seed 6, 30 rounds, `--keep 8`
  (a round's depth is now 3.8 GB on the GPU), v4's mix + `--vmax 2.5 --tall-stairs 0.5 --gap-max 0.6`. Smoke test
  (512 envs): collection time unchanged by the per-step depth processing; the r71 warm start's action MSE under
  the new timing 0.18 -> 0.10 after 100 iterations. Log `logs/student_dagger_v5.out`.
  Result (1.8 h): round 1 dipped (student up 0.84 -> 0.60, the replay held one round at warm-up lr) and recovered
  by round 2; student up 0.87 -> 0.90, action MSE (student envs) 0.10 -> 0.075 by round 29, still falling. 3 seeds,
  slow / fast: r19 **304** / 408, r24 **300 / 425**, r29 293 / 419 (v4 296 / 397, rl_v1 r71 287 / 421, rl_v2 r47
  260 / 426): pure DAgger with the fresher camera matches the RL runs at speed without their slow-speed losses
  (sidestep 19-21 of 24, stairs up 10 cm 22-23). Tall-stairs course DR, 1.5 / 2.0 m/s of 64: r19 30 / 13, r24 28 /
  **22**, r29 35 / 19 (v4 25 / 5, rl_v1 r83 26 / 14, rl_v2 r59 35 / 13; generator + teacher 48 / 37). **Best
  student now: `logs/student/dagger_v5/ckpt_024.pt`** (valid only if the robot's camera path achieves ~60-80 ms with
  a fresh frame each step: run the D435 at 60 fps and measure).
  Continued to round 49 (`--resume --rounds 50 --lr 1e-4`: the cosine horizon grows with --rounds, so the lr was
  lowered to resume near 4e-5; the replay restarts empty, no dip; log `logs/student_dagger_v5b.out`): converged.
  Action MSE flat at ~0.073; 3 seeds, slow / fast: r39 304 / 416, r44 291 / 413, r49 292 / 411 (r24 300 / 425),
  all within noise. Tall-stairs course DR, 1.5 / 2.0 m/s: r39 33 / 23, r44 35 / 20, r49 **39 / 25** (r24 28 / 22,
  r29 35 / 19); stairs up remains the failure (30-35 of 64 at 2 m/s vs the teacher's 15).
- **RL fine-tune reworked** (2026-11-16, commit cf869e9; user's go-ahead after v5 converged). PPO and the DAgger
  term now step in two phases per update, each with its own Adam and its lr adapted to its own KL (PPO 0.01 per
  minibatch as before; DAgger `--dagger-kl` x lam, measured on the PPO phase's last minibatch before / after its
  `--dagger-steps` steps). In rl_v1 / v2's single step the DAgger change set the KL and pinned the lr at its floor,
  while the PPO gradient set the direction (at std 0.1 ~20x the DAgger term's): the DAgger floor did not hold the
  slow-speed skills. The terrain head trains with the DAgger term but by its own Adam (3e-4; it does not act).
  `--cmd`: penalty x dt per unit of relative speed along the joystick below (1 - `--cmd-tol` 0.5) of it (0.2 s
  mean, heading frame, from 1 s into the round), against rl_v2's stalling; v5 r49 under it: ~0.03 / step, 1-2 % of
  steps stalled. The camera timing comes from the student checkpoint's env_kwargs. **Acting vs training forward**:
  in eval / no_grad nn.TransformerEncoder takes its fused fast path, ~4e-4 per action off the training path
  (float64 reference: 4e-7), and TF32 adds as much; |ratio - 1| at an update's first minibatch was 0.14 at std 0.01
  (the 0.016 at std 0.1 in rl_v1 was the same). Both off in rl_finetune (`torch.backends.mha
  .set_fastpath_enabled(False)`, fp32_precision "ieee" after each env build): 2e-5. Std: PHP's 0.01 tried in the
  smoke test (128 envs): KL 0.01 per update then allows ~3e-4 rad of change (PPO lr ~3e-8) and the policy
  gradient's signal / noise falls with the std, so 0.05 (PPO lr settles ~1.5e-7, DAgger ~1.5e-6, clip fraction
  0.18). **rl_v3** (queued 2026-11-16 by scratchpad `launch_rl3.sh` until a GPU has 60 GB free; the other lab
  user's jobs held both): from dagger_v5 r49 (fresh critic, aux head), 60 rounds, critic warm-up 3, `--edge 2.5
  --foothold 0.5` (half of rl_v2's) `--cmd 4 --aux 1`, DAgger KL 0.01 -> 0.005 over 25 rounds (`--dagger-floor 0.5
  --anneal 25`), keep 8, seed 8, v5's scene mix. Log `logs/student_rl_v3.out`, run `logs/student/rl_v3/`.
  Result (GPU 1, 18:21-23:03, ~4.4 min / round on the shared GPU; a Warp CUDA error 700 in mjlab's sensor context
  at round 55's env build, GPU nearly full, resumed with `--resume`, log `logs/student_rl_v3b.out`): stable, PPO KL
  ~0.0137 / clip 0.17 throughout, DAgger KL on its 0.01 -> 0.005 target, aux MSE 0.026 -> 0.0084 (~4.4 cm RMS);
  training share up flat at ~0.90, stalled steps flat at ~1.8 %, action MSE to the teacher flat at ~0.066 (the
  policy stayed near its DAgger solution). 3 seeds, slow / fast (v5 r49 292 / 411): r23 292 / 419, r35 **296 /
  424**, r55 294 / 415, r59 287 / 420; no slow-speed loss this time (sidestep 21-23 of 24, stairs up 10 cm 20-24),
  fast stairs up 18 cm 7 -> 8-12. Tall-stairs course DR, 1.5 / 2.0 m/s of 64 (v5 r49 39 / 25; generator + teacher
  48 / 37): r23 38 / 29, r35 34 / **30**, r55 38 / 23, r59 **40** / 28; stairs-up falls at 2 m/s 35 -> 22-29.
  Gains of +3-5 / 64 at 2 m/s are about one binomial SD (~4): the fixes work (no hacking, no forgetting) but at
  this trust region (std 0.05, KL 0.01) 60 rounds moved the policy little.
- **Edge penalty from "Hiking in the Wild"** (2026-11-17; arXiv 2601.07718, code in InstinctLab, not instinct_rl:
  `volume_points_penetration`, weight -4, sum over a 10x5x2 grid of volume points per ankle link of (inside an edge
  cylinder) x (|v| + 1e-6) x depth; edges = mesh edges with dihedral > 70 deg, cylinders r 0.05; plus `feet_at_plane`
  -0.1). Port (commit f2e075f, `generator/metrics.py` `edge_penetration` / `EdgeTerm`): lips are heightfield cell
  boundaries with a jump > 3 cm, at the upper height, searched along x in the point's row and y in its column (box
  corners rounded, concave corners missed); 12 points along each foot capsule's axis, cylinder r 0.05 + the capsule's
  1.25 cm; weight 33 (~4 x 100 points / 12). Physics-side metric: scratchpad `edge_metric.py` on course_demo traj.pt
  (share of steps with a foot point in a cylinder, edge_vol per step; v6 on the tall course's stairs up: ~0.41, 0.06).
  - **PPO teacher fine-tune in the closed loop** (`scripts/student/teacher_rl.py`: rl_finetune's PPO on the v6
    teacher, depth off): at std 0.05 / KL 0.01 the 8.7M MLP barely moved (0.15 % weight change in 6 rounds); at std
    0.1 / KL 0.02 (edge_v2 / ctrl_v2, 20 rounds) both runs lost the sidestep (21 -> 0) and stairs, edge term or not.
    Abandoned.
  - **In imagination** (commit 2bb8c4c, `--policy-trainer.edge-vol 33`: reward minus the term on world-model states,
    gradients through FK; ~1 it/s vs 3 without). **dash_teacher_v7_edge / v7_ctrl**: from v6 50k, 12k it, lr 3e-5
    (policy, critic, world model), v6's flags otherwise (scratchpad `launch_v7.sh`). Imagined edge hit 18 % -> 6.7 %,
    train return unchanged (~0.63). 3 seeds (scratchpad `eval_trl.sh`, `logs/teacher_rl/eval/<tag>_*`), ctrl / edge
    at 3k, 6k, 12k (v6): tall course DR 1.5 m/s 121, 134, 128 / 132, 139, 134 (133) of 192; **2.0 m/s 104, 81, 88 /
    115, 113, 117 (103), stairs-up falls 78, 99, 86 / 55, 61, 60 (74)**; 15 scenes DR + delay 303, 294, 282 / 275,
    291, 298 (295) of 360; stairs up 18 cm 15, 12, 4 / 9, 12, 17 (9); **platform up 30 cm 15, 11, 14 / 4, 9, 6 (13)**;
    course edge_vol at 1.5 m/s 0.033, 0.032, 0.034 / 0.022, 0.020, 0.020 (0.033), stairs-up hit 0.41 / 0.31 at 12k.
    The term transfers (-40 % edge contact, consistent at every checkpoint), fast stairs up improves, platform up
    30 cm pays. **Best teacher now: `logs/dash_teacher_v7_edge/ckpt_0012000.pt`** (no student distilled from it yet).

## Stairs-only tracker with the edge penalty (2026-11-19, lab server)

User's decision: narrow the scope to stairs up / down and train a tracker from scratch with the edge penalty on,
next to an identical control without it, to see whether the tracker keeps its feet off stair lips. Gaps left for
a later round (no labelled gap clips; PARC gap jumps would have to be found by searching terrains for pits).
- **Edge term compiled** (`imagination/compiled.py` ImagStep `edge=`): the step returns the feet from the reward's
  own FK and `EdgeTerm.from_feet` runs as its own compiled graph: 2.9 it/s with the term (eager 1.0, without it
  3.0), imagined edge_hit / edge_vol equal to eager's. Compiled *inside* the step's graph the policy update went NaN
  at the first update (the term compiled alone matches eager values and gradients; not pinned down further).
- **Library `data/library/dash_stairs_v1.pt`** (`scripts/data/build_stairs_library.py`): v5's stairs clips (mocap
  286, PARC 194; 0.81 h), physics-augmented stairs runs sampled from aug3a / aug4a (up 1,907, 5.83 h) and aug3a /
  aug4a / aug4b (down 1,623, 4.96 h), flat locomotion 2.9 h (20 %); 14.5 h, 4,010 terrains, 14 box slots (v5: 66).
  Held out (base-motion hash mod 20, as GenData): `dash_stairs_v1_eval.pt`, 216 clips (mocap 46, aug up 93, down 77).
- **The references' own edge contact** (`scripts/diag/ref_edge.py`; share of frames with a foot point in a 5 cm
  lip cylinder, edge_vol per frame): mocap stairs 0.75 / 0.018 (eval 0.86 / 0.033: human feet sit on the lips,
  2-3 cm deep), PARC 0.36 / 0.027, aug up 0.28 / 0.016, aug down 0.25 / 0.019 (the v6 robot's own placements).
- **Tracking the held-out pack in physics** (`scripts/eval_edge.py`: `run_benchmark(edge=True)`, nominal physics;
  failed / mean return / edge hit / edge_vol):

  | | mocap stairs (46) | aug stairs up (93) | aug stairs down (77) |
  |---|---|---|---|
  | v6 50k | 0 / 0.934 / 0.82 / 0.064 | 10 / 0.843 / 0.33 / 0.016 | 15 / 0.861 / 0.21 / 0.020 |
  | v7_ctrl 12k | 0 / 0.936 / 0.85 / 0.063 | 14 / 0.843 / 0.34 / 0.016 | 17 / 0.860 / 0.21 / 0.018 |
  | v7_edge 12k | 0 / 0.891 / **0.32 / 0.004** | 17 / 0.834 / 0.25 / 0.008 | 6 / 0.857 / 0.19 / 0.012 |

  The fine-tuned edge tracker already leaves the mocap's lip footholds (-94 % edge_vol for -0.045 return). On the
  aug climbs every tracker lags the reference (v6: max root distance median 1.1 m on stairs up, 0.35 m down; mocap
  0.11 m): it climbs slower than its own recorded runs and never catches up on the flat top (global root term
  0.5 x exp(-d^2 / 0.3^2) gives nothing back once ~0.5 m behind). Failures there are falls after lagging.
- **From-scratch runs** (`scripts/train_stairs.sh`): v6's flags on `dash_stairs_v1.pt`, 50k iterations (lr decay
  from 10k), `logs/dash_stairs_edge` (`--policy-trainer.edge-vol 33`, GPU 1) and `logs/dash_stairs_ctrl` (GPU 0),
  2.9-3.0 it/s each, ~4.7 h. `scripts/eval_stairs.sh` evaluates every 10k checkpoint and runs the course at 50k
  (`logs/stairs_eval/`). Held-out pack, failed / mean return / edge hit / edge_vol, nominal (DR within 0-3 fails,
  0.01-0.04 return):

  | | mocap stairs (46) | aug stairs up (93) | aug stairs down (77) |
  |---|---|---|---|
  | ctrl 10k | 6 / 0.720 / 0.887 / 0.071 | 5 / 0.845 / 0.295 / 0.015 | 0 / 0.877 / 0.246 / 0.018 |
  | edge 10k | 22 / 0.623 / 0.432 / 0.012 | 1 / 0.830 / 0.084 / 0.002 | 1 / 0.857 / 0.082 / 0.002 |
  | edge 20k | 7 / 0.742 / 0.391 / 0.006 | 0 / 0.876 / 0.053 / 0.001 | 1 / 0.890 / 0.071 / 0.001 |
  | ctrl 50k | 0 / 0.902 / 0.843 / 0.061 | 0 / 0.948 / 0.250 / 0.012 | 0 / 0.947 / 0.249 / 0.018 |
  | **edge 50k** | 0 / 0.870 / **0.271 / 0.003** | 0 / 0.934 / **0.032 / 0.0002** | 0 / 0.928 / **0.050 / 0.0003** |

  The edge run's mocap failures at 10-20k were transient (0-1 from 30k on). The 46 held-out mocap clips are only
  two human stairs-down takes (`down_stop_R_001__A300`, `down_loop_R_102__A301`), mirrored and refitted to 12 US
  stair sizes; no held-out human stairs up.
- **The mocap references are clean apart from shallow step overlap** (`scripts/media/render_reference.py --stats`;
  foot capsule vs the heightfield solid): frames with a foot > 1 cm into a step: mocap 11.9 % (train) / 9.5 %
  (eval), worst per clip p50 1.6 cm, p90 2.6 cm; aug 0.2 %; PARC 0.0 %. Standing with > 30 % of the foot over a
  lower step: mocap 0.1 % of stance frames, aug 4 %, PARC 6 %. Human feet sit near the nosing (edge zone) but on
  the tread. Videos (two panels: body | feet profile, feet red / orange when overlapping / overhanging):
  `render_reference.py out LIB --clips ... --evals "label=logs/stairs_eval/edge_10k.json"`.
- **Course at 50k** (stairs-only `course_demo.py`: 5 steps up of 10 / 14 / 17 cm, 30 cm treads, 3 m flat deck,
  steps of <= 15 cm down, `--gaps 0 --platforms 0 --seconds 24 --dr`, gen_v6 plans, 3 seeds x 8 = 24 per cell).
  **Scoring changed** (2026-11-20, `course_demo.py --rescore` re-scores a traj.pt): success = reached the finish
  line without going down (before: also still up at the end of `--seconds`; at 1.5-2 m/s x 24 s the robots ran
  ~30 m and fell off the end of the world, scoring 0 successes); failures split into falls (by feature) and
  "timeout" (still up, short of the finish). Earlier course numbers in this file used the old rule.

  | success (falls, timeouts) of 24 | 0.5 m/s | 1.0 | 1.5 | 2.0 |
  |---|---|---|---|---|
  | ctrl 10 / 14 / 17 cm | 24 / 24 / 24 | 24 / 22 (2) / 22 (2) | 23 / 20 / 11 | 20 / 4 / 1 |
  | edge 10 / 14 / 17 cm | 24 / 23 (0, 1) / **16 (0, 8)** | 24 / 23 (1) / 24 | 17 / 17 / 18 | 11 / 9 / 10 |

  At 0.5-1 m/s (the stairs data's range is 0.3-1 m/s) 140 vs 134 of 144, edge falls 1 vs 4; at 1.5-2 m/s (outside
  it) 79 vs 82. **Edge timeouts are generator stalls**: in 6 of the edge tracker's 72 runs at 0.5 m/s the plan
  itself stops (robot and plan stand in a split stance for 2-6 s, mostly on the deck 0.4-1.7 m past the top; never
  with the control). The generator replans from the robot's state, so the edge tracker's gait leads it into a
  stop; not a tracking failure. Video: `logs/stairs_eval/course_edge_50k_s0/course_v2_0.5ms_0_ghost.mp4`.
- **Edge contact on the course** (`scripts/diag/course_edge.py`, while up; hit / edge_vol per step):

  | | up 0.5 m/s | up 1.0 | down 0.5 | down 1.0 | up 1.5-2.0 | down 1.5-2.0 |
  |---|---|---|---|---|---|---|
  | ctrl robot | 0.29 / 0.011 | 0.34 / 0.032 | 0.84 / 0.059 | 0.81 / 0.092 | 0.32-0.38 / 0.06 | 0.74-0.78 / 0.11 |
  | **edge robot** | **0.03 / 0.001** | **0.07 / 0.003** | **0.30 / 0.001** | **0.42 / 0.009** | 0.17-0.20 / 0.02-0.03 | 0.56-0.61 / 0.04-0.06 |
  | plan (ctrl runs) | 0.38 / 0.030 | 0.47 / 0.078 | 0.79 / 0.069 | 0.68 / 0.074 | | |
  | plan (edge runs) | 0.25 / 0.017 | 0.40 / 0.044 | 0.44 / 0.014 | 0.47 / 0.032 | | |

  The generator plans footholds on the lips (it learned them from the mocap) and the edge tracker overrides them;
  its plans' contact is lower only because the plan restarts from the robot's state. Per run (17 cm), edge hit
  varies a lot (down 1.0 m/s p10 / p50 / p90: edge 0.18 / 0.48 / 0.60, ctrl 0.68 / 0.84 / 0.95): pick median runs
  for videos. Side by side, each tracker's median run (`scripts/media/course_sbs.py`, feet orange in the zone):
  `logs/stairs_eval/sbs/` (`render.sh`).
- **The edge tracker's timeouts are a slower walk, not stalls alone**: 17 cm at 0.5 m/s with `--seconds 40`, 3
  seeds x 16: edge 48/48 finish (median 23.5 s, p90 25.6, max 26.1), ctrl 47/48 (1 fall; 20.7 / 21.9 / 23.4).
  Section speeds edge vs ctrl: flat start 0.49 / 0.49, stairs up 0.35 / 0.35, deck 0.42 / 0.54, stairs down
  0.43 / 0.48 m/s. The 24 s limit at 0.5 m/s leaves 4 s of slack on the 9.8 m course.
- **Edge-aware planning at sampling time does not move the feet** (2026-11-20; `generator/guidance.py`
  `EdgeClearance` (x0_hook: gradient on the plan's foot-axis points out of the lip cylinders, horizontal only,
  joints only, last 4 steps x 5) and `select_plans` (best of N samples by lip contact, among those travelling >= 0.9
  x the median distance); `course_demo.py --edge-guide / --edge-select N`; `logs/stairs_eval/plan_guidance.sh`).
  Stairs-only course, 0.5 / 1 m/s, 3 seeds, 144 runs per row; hit = share of steps with a foot in the zone:

  | | success (falls, timeouts) | plan hit up / down | robot hit up / down | robot feet from plan |
  |---|---|---|---|---|
  | ctrl, plain | 140 (4, 0) | 0.38-0.47 / 0.68-0.79 | 0.29-0.34 / 0.81-0.84 | 4-8 cm |
  | ctrl, best of 8 | 142 (1, 1) | 0.35-0.41 / 0.60-0.79 | 0.29-0.35 / 0.81-0.83 | 4-8 cm |
  | ctrl, guided | 144 (0, 0) | **0.08-0.11 / 0.13-0.15** | 0.28-0.39 / 0.78-0.84 | **6-9 cm** |
  | edge, plain | 134 (1, 9) | 0.25-0.40 / 0.44-0.47 | 0.03-0.07 / 0.30-0.42 | 4-8 cm |
  | edge, best of 8 | 133 (1, 10) | 0.24-0.36 / 0.38-0.45 | 0.02-0.07 / 0.21-0.46 | 4-8 cm |
  | edge, guided | 135 (0, 9) | 0.06-0.10 / 0.08-0.10 | 0.03-0.06 / 0.22-0.41 | 5-8 cm |

  Guidance makes the plans nearly edge-free (worst depth per plan 1.3 -> 0.4 cm) but the control tracker puts its
  feet on the lips anyway, further from the plan; best of 8 only lowers the plan's contact by a third (8 samples
  differ little in their footholds) and changes nothing downstream. Where the feet land is the tracker's own habit
  (as with the sole-clearance guidance above); edge avoidance has to be trained into the tracker, which the edge
  run did. Wall time per run (48 envs x 24 s, shared GPUs): best of 8 ~3 min, guided ~4.3 min.

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
