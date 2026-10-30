"""Velocity interface: reference commands, reward quantities vs the simulator, fall termination, frozen decoder."""

from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
DASH_NPZ = ROOT / "data/npz/dash_bones_seed"
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture(scope="module")
def env():
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env
    from supermimic.envs.mdp.motion_library import discover_npz

    if not DASH_NPZ.exists():
        pytest.skip("Dash npz not copied")
    files = [str(f) for f in discover_npz(DASH_NPZ, names=["dash_walk_forward_stop_002__A026", "dash_walk_ff_stop_360_R_slow_001__A445", "dash_idle_loop_003__A265"])]
    env = make_env(SuperMimicEnvCfgArgs(npz_files=files, robot="dash", num_envs=64, replay_capacity=16, seed=3), device="cuda:0")
    cmd = env.command_manager.get_term("motion")
    cmd.track_mask[32:] = False  # second half: velocity-interface envs
    env.reset()
    yield env
    env.close()


def test_reference_commands_match_naive_window(env):
    from supermimic.interfaces.velocity import reference_commands
    from supermimic.kinematics.rotations import quat_apply_inverse, yaw_quat

    lib = env.command_manager.get_term("motion").library
    c = reference_commands(lib, window=25)
    assert c.shape == (lib.num_frames, 3)
    for f in [0, 5, 100, int(lib.clip_last[0]), int(lib.clip_start[1]) + 7, lib.num_frames - 1]:
        clip = int(lib.frame_clip[f])
        lo, hi = max(f - 12, int(lib.clip_start[clip])), min(f + 12, int(lib.clip_last[clip]))
        r = lib.ref(torch.arange(lo, hi + 1, device=lib.device))
        v = quat_apply_inverse(yaw_quat(r.root_quat_w), r.root_lin_vel_w)[:, :2].mean(0)
        torch.testing.assert_close(c[f, :2], v, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(c[f, 2], r.root_ang_vel_w[:, 2].mean(), atol=1e-4, rtol=1e-4)
    # The forward walk moves forward on average.
    i = [cl.name for cl in lib.clips].index("dash_walk_forward_stop_002__A026")
    walk = c[lib.clip_start[i] : lib.clip_last[i] + 1]
    assert walk[:, 0].max() > 0.5 and walk[:, 0].min() > -0.1  # walks forward, then stops
    idle = [cl.name for cl in lib.clips].index("dash_idle_loop_003__A265")
    assert c[lib.clip_start[idle] : lib.clip_last[idle] + 1].abs().max() < 0.2


def test_velocity_quantities_match_simulator(env):
    from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat

    from supermimic.interfaces.velocity import VelocityCfg, velocity_of, velocity_reward_terms

    robot = env.scene["robot"]
    io = env.recorder_manager.get_term("replay").state_io
    for _ in range(5):
        env.step(0.3 * torch.randn(env.num_envs, robot.num_joints if hasattr(robot, "num_joints") else 18, device=env.device))
    s = io.read()
    d = robot.data
    v = velocity_of(s)
    v_h = quat_apply_inverse(yaw_quat(d.root_link_quat_w), d.root_link_lin_vel_w)
    torch.testing.assert_close(v[:, :2], v_h[:, :2], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(v[:, 2], d.root_link_ang_vel_w[:, 2], atol=1e-4, rtol=1e-4)
    # Lean / roll-pitch-rate terms read the same quantities the simulator reports.
    cfg = VelocityCfg()
    z = torch.zeros(env.num_envs, 18, device=env.device)
    r = velocity_reward_terms(s, z, z, v.clone(), d.soft_joint_pos_limits[0], cfg)
    torch.testing.assert_close(r["lateral_lean"], cfg.dt * cfg.w_lateral_lean * d.projected_gravity_b[:, 1] ** 2, atol=1e-6, rtol=1e-4)
    torch.testing.assert_close(r["roll_pitch_rate"], cfg.dt * cfg.w_roll_pitch_rate * (d.root_link_ang_vel_b[:, :2] ** 2).sum(-1), atol=1e-6, rtol=1e-4)
    # A perfectly tracked command earns the full tracking terms.
    torch.testing.assert_close(r["lin_vel"] + r["yaw_rate"], torch.full_like(r["lin_vel"], cfg.max_step_reward))


def test_fall_termination_and_masks(env):
    from supermimic.interfaces.velocity import VelocityCfg, fallen

    cmd = env.command_manager.get_term("motion")
    io = env.recorder_manager.get_term("replay").state_io
    tm = env.termination_manager
    cfg = VelocityCfg()
    saw_fall = saw_track_term = False
    for _ in range(150):
        _, _, term, _, _ = env.step(2.0 * torch.randn(env.num_envs, 18, device=env.device))  # violent: everything falls
        fall = tm.get_term("fall")
        assert not fall[cmd.track_mask].any()
        assert not (tm.get_term("root_height") | tm.get_term("root_tilt") | tm.get_term("clip_end"))[~cmd.track_mask].any()
        saw_fall |= bool(fall.any())
        saw_track_term |= bool((tm.get_term("root_height") | tm.get_term("root_tilt")).any())
    assert saw_fall and saw_track_term
    # The analytic predicate agrees with the simulator term on a fresh, non-reset state.
    s = io.read()
    pred = fallen(s, cfg) & ~cmd.track_mask
    from supermimic.envs.mdp.terminations import fall as fall_term

    torch.testing.assert_close(pred, fall_term(env, "motion", cfg.fall_height, cfg.fall_tilt))


def test_decode_frozen_reaches_encoder_only():
    from supermimic.policy.nets import Policy, PolicyCfg

    cfg = PolicyCfg(num_joints=18, goal_dim=10 * 46, velocity=True, encoder_hidden=(32,), decoder_hidden=(32,))
    p = Policy(cfg)
    obs = torch.randn(4, cfg.obs_dim)
    z = p.encoders["velocity"](torch.randn(4, 3), obs)
    a = p.decode_frozen(z, obs)
    torch.testing.assert_close(a, p.decoder(z, obs))
    a.sum().backward()
    assert all(q.grad is None for q in p.decoder.parameters())
    assert all(q.grad is None for q in p.encoders["motion"].parameters())
    assert any(q.grad is not None and q.grad.abs().sum() > 0 for q in p.encoders["velocity"].parameters())
