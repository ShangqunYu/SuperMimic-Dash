"""xi with Dash's hardware randomization: layout, ranges, and agreement with the simulator model."""

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
    files = [str(f) for f in discover_npz(DASH_NPZ, limit=2)]
    env = make_env(SuperMimicEnvCfgArgs(npz_files=files, robot="dash", num_envs=256, replay_capacity=16, hardware_dr=True, seed=5), device="cuda:0")
    env.reset()
    yield env
    env.close()


def _blocks(xi, nj, nb):
    sizes = {"friction": 1, "bias": nj, "com": 3, "kp": nj, "kd": nj, "armature": nj, "joint_friction": nj, "mass": nb}
    out, i = {}, 0
    for k, n in sizes.items():
        out[k] = xi[:, i : i + n]
        i += n
    assert i == xi.shape[1]
    return out


def test_hardware_xi_layout_and_ranges(env):
    from supermimic.robots import DASH
    from supermimic.world_model.state_spec import StateSpec, StateSpecCfg

    ctx = env.recorder_manager.get_term("replay").ctx
    nj, nb = DASH.num_joints, DASH.num_bodies
    xi = ctx.read()
    assert xi.shape == (env.num_envs, 113) == (env.num_envs, StateSpec(StateSpecCfg(xi_hw_bodies=nb), "cpu", nj).xi_dim)
    b = _blocks(xi, nj, nb)
    eps = 1e-4
    assert ((b["friction"] >= 0.3 - eps) & (b["friction"] <= 1.2 + eps)).all()
    assert (b["bias"].abs() <= 0.01 + eps).all()
    for axis, (lo, hi) in {0: (-0.06, 0.06), 1: (-0.025, 0.025), 2: (-0.03, 0.03)}.items():
        assert ((b["com"][:, axis] >= lo - eps) & (b["com"][:, axis] <= hi + eps)).all(), axis
    assert ((b["kp"] >= 0.8 - eps) & (b["kp"] <= 1.2 + eps)).all()
    assert ((b["kd"] >= 0.7 - eps) & (b["kd"] <= 1.3 + eps)).all()
    assert ((b["armature"] >= 0.008 - eps) & (b["armature"] <= 0.05 + eps)).all()
    assert ((b["joint_friction"] >= -eps) & (b["joint_friction"] <= 2.5 + eps)).all()
    assert ((b["mass"] >= 0.88 - 1e-3) & (b["mass"] <= 1.05 + 1e-3)).all()
    # Randomized per env and per element, not constant.
    for k in ("kp", "kd", "armature", "joint_friction", "mass", "com"):
        assert b[k].std(0).min() > 1e-4, k  # arm armature spans only 0.008-0.01
        assert b[k].std(1).mean() > 1e-3, k
    # Normalised xi is O(1).
    xi_n = StateSpec(StateSpecCfg(xi_hw_bodies=nb), xi.device, nj).normalize_xi(xi)
    assert xi_n.abs().max() < 3.0


def test_hardware_xi_matches_model_by_name(env):
    """Spot-check the joint/actuator/body mapping against named lookups in the compiled model."""
    import mujoco

    from supermimic.robots import DASH

    ctx = env.recorder_manager.get_term("replay").ctx
    b = _blocks(ctx.read(), DASH.num_joints, DASH.num_bodies)
    mjm = env.sim.mj_model
    m = env.sim.model
    for ji, jname in enumerate(DASH.joint_names):
        jid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, f"robot/{jname}")
        dof = int(mjm.jnt_dofadr[jid])
        act = [a for a in range(mjm.nu) if int(mjm.actuator_trnid[a, 0]) == jid]
        assert len(act) == 1, jname
        a = act[0]
        torch.testing.assert_close(b["armature"][:, ji], m.dof_armature[:, dof].float())
        torch.testing.assert_close(b["joint_friction"][:, ji], m.dof_frictionloss[:, dof].float())
        torch.testing.assert_close(b["kp"][:, ji] * float(mjm.actuator_gainprm[a, 0]), m.actuator_gainprm[:, a, 0].float())
    # Mass: the expected per-env total follows from the scales.
    nominal = torch.tensor([mjm.body_mass[mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, f"robot/{n}")] for n in DASH.body_names()], device=b["mass"].device)
    body_ids = [mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, f"robot/{n}") for n in DASH.body_names()]
    torch.testing.assert_close((b["mass"] * nominal.float()).sum(1), m.body_mass[:, body_ids].float().sum(1))
