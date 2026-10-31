"""Motion-matching planner: database goal features equal the env's, spring trajectory, matcher bookkeeping."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
DASH_NPZ = ROOT / "data/npz/dash_bones_seed"
CLIP = "dash_walk_forward_stop_002__A026"


def _builder():
    spec = importlib.util.spec_from_file_location("build_mm_db", ROOT / "scripts/planner/build_mm_db.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_db_goal_features_match_env():
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env
    from supermimic.kinematics.fk import Kinematics
    from supermimic.robots import get_robot

    if not (DASH_NPZ / f"{CLIP}.npz").exists():
        pytest.skip("Dash npz not copied")
    b = _builder()
    env = make_env(SuperMimicEnvCfgArgs(npz_files=[str(DASH_NPZ / f"{CLIP}.npz")], robot="dash", num_envs=1, seed=0), device="cuda:0")
    try:
        lib = env.command_manager.get_term("motion").library
        spec = get_robot("dash")
        q0 = b.default_joint_pos(spec.joint_names).cuda()
        torch.testing.assert_close(q0, lib.default_joint_pos)
        f = torch.arange(lib.clip_start[0], lib.clip_last[0] + 1, device="cuda:0")
        fk = Kinematics(spec.load_model(), device="cuda:0")
        goal, match = b.clip_features(lib.ref_state(f), spec.num_joints, q0, fk, fk.body_indices(["l_foot", "r_foot"]))
        k = 10
        win = lib.goal_window(f[:50], k).reshape(50, k, -1)
        for j in range(k):
            torch.testing.assert_close(win[:, j], goal[j : j + 50], atol=1e-5, rtol=1e-5)
        # root_vel features (smoothed simulation root) agree with the velocity interface's reference
        # command (25-frame centred mean of the heading-frame velocity / world yaw rate) away from the clip ends.
        from supermimic.interfaces.velocity import reference_commands

        ref_cmd = reference_commands(lib, window=25)[f]
        inner = slice(30, -30)
        torch.testing.assert_close(match[inner, -3:], ref_cmd[inner], atol=0.05, rtol=0.0)
        assert ref_cmd[inner, 0].max() > 0.5  # the clip does walk
    finally:
        env.close()


def test_spring_trajectory():
    from supermimic.planner.motion_matching import DT, TRAJ_OFFSETS, PlannerCfg, spring_trajectory

    cfg = PlannerCfg()
    z = spring_trajectory(np.zeros(2), 0.0, np.zeros(3, dtype=np.float32), cfg)
    np.testing.assert_allclose(z[:6], 0.0, atol=1e-7)
    np.testing.assert_allclose(z[6:].reshape(3, 2), [[1, 0]] * 3, atol=1e-7)
    # Already at the commanded forward speed: straight line at that speed.
    v = spring_trajectory(np.array([0.5, 0.0]), 0.0, np.array([0.5, 0.0, 0.0], dtype=np.float32), cfg)
    np.testing.assert_allclose(v[:6].reshape(3, 2), [[0.5 * o * DT, 0.0] for o in TRAJ_OFFSETS], atol=1e-5)
    # Turning at the commanded rate: heading at offset o is wz * o * dt.
    w = spring_trajectory(np.zeros(2), 1.0, np.array([0.0, 0.0, 1.0], dtype=np.float32), cfg)
    ang = np.arctan2(w[6:].reshape(3, 2)[:, 1], w[6:].reshape(3, 2)[:, 0])
    np.testing.assert_allclose(ang, [o * DT for o in TRAJ_OFFSETS], atol=1e-5)
    # Starting from rest, a forward command moves forward, less than the steady-state distance.
    s = spring_trajectory(np.zeros(2), 0.0, np.array([0.6, 0.0, 0.0], dtype=np.float32), cfg)
    assert 0 < s[4] < 0.6 * 50 * DT and abs(s[5]) < 1e-7


DB = ROOT / "data/planner/dash_walk_mm.npz"


@pytest.mark.skipif(not DB.exists(), reason="planner database not built")
def test_matcher_stays_in_valid_frames_and_follows_command():
    from supermimic.planner.motion_matching import FUTURE_FRAMES, MotionDatabase, MotionMatcher

    db = MotionDatabase(DB)
    mm = MotionMatcher(db)
    rv = []
    for t in range(600):
        c = (0.6, 0.0, 0.0) if t >= 100 else (0.0, 0.0, 0.0)
        g = mm.step(c)
        assert g.shape == (FUTURE_FRAMES * db.goal.shape[1],) and np.isfinite(g).all()
        assert mm.frame <= db.clip_last[db.clip_of[mm.frame]]
        grav = g.reshape(FUTURE_FRAMES, -1)[:, 36:39]
        np.testing.assert_allclose(np.linalg.norm(grav, axis=-1), 1.0, atol=1e-5)
        rv.append(mm.reference_velocity().copy())
    rv = np.array(rv)
    assert np.abs(rv[50:100]).max() < 0.3  # standing
    assert abs(rv[400:, 0].mean() - 0.6) < 0.15  # walking at the commanded speed
