"""Torch FK must reproduce MuJoCo's xpos/xquat and body-origin velocities exactly."""

from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from supermimic.data.motion_convert import load_g1_model
from supermimic.kinematics.fk import Kinematics
from supermimic.kinematics.rotations import quat_apply_inverse, quat_from_rotvec, rotvec_from_quat
from supermimic.robots import DASH, get_robot

DASH_NPZ = Path(__file__).resolve().parents[1] / "data/npz/dash_bones_seed"


def _random_state(model, rng, n):
    qpos = np.zeros((n, model.nq))
    qvel = rng.normal(size=(n, model.nv)) * np.array([1.0] * 3 + [2.0] * 3 + [3.0] * (model.nv - 6))
    qpos[:, :3] = rng.normal(size=(n, 3))
    q = rng.normal(size=(n, 4))
    qpos[:, 3:7] = q / np.linalg.norm(q, axis=-1, keepdims=True)
    lo, hi = model.jnt_range[1:, 0], model.jnt_range[1:, 1]
    qpos[:, 7:] = rng.uniform(lo, hi, size=(n, model.nq - 7))
    return qpos, qvel


def _mujoco_reference(model, qpos, qvel):
    data = mujoco.MjData(model)
    n = qpos.shape[0]
    nb = model.nbody - 1
    pos = np.zeros((n, nb, 3))
    quat = np.zeros((n, nb, 4))
    lin = np.zeros((n, nb, 3))
    ang = np.zeros((n, nb, 3))
    res = np.zeros(6)
    for i in range(n):
        data.qpos[:] = qpos[i]
        data.qvel[:] = qvel[i]
        mujoco.mj_forward(model, data)
        pos[i] = data.xpos[1:]
        quat[i] = data.xquat[1:]
        for b in range(1, model.nbody):
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, b, res, 0)  # XBODY = frame origin (BODY would be the COM)
            ang[i, b - 1] = res[:3]
            lin[i, b - 1] = res[3:]
    return pos, quat, lin, ang


def _signed_quat_err(q, q_ref):
    sign = np.sign(np.sum(q * q_ref, axis=-1, keepdims=True))
    sign[sign == 0] = 1
    return np.abs(q * sign - q_ref).max()


@pytest.mark.parametrize("robot", ["g1", "dash"])
@pytest.mark.parametrize("dtype,atol_pos,atol_vel", [(torch.float64, 1e-10, 1e-9), (torch.float32, 2e-5, 2e-4)])
def test_fk_matches_mujoco(robot, dtype, atol_pos, atol_vel):
    torch.backends.cuda.matmul.allow_tf32 = False
    model = get_robot(robot).load_model()
    nb = model.nbody - 1
    rng = np.random.default_rng(0)
    qpos, qvel = _random_state(model, rng, 64)
    pos_ref, quat_ref, lin_ref, ang_ref = _mujoco_reference(model, qpos, qvel)

    fk = Kinematics(model, device="cpu", dtype=dtype)
    out = fk(torch.as_tensor(qpos, dtype=dtype), torch.as_tensor(qvel, dtype=dtype))
    assert out["pos"].shape == (64, nb, 3) and out["quat"].shape == (64, nb, 4)
    np.testing.assert_allclose(out["pos"].numpy(), pos_ref, atol=atol_pos)
    assert _signed_quat_err(out["quat"].numpy(), quat_ref) < atol_pos * 10
    np.testing.assert_allclose(out["ang_vel"].numpy(), ang_ref, atol=atol_vel)
    np.testing.assert_allclose(out["lin_vel"].numpy(), lin_ref, atol=atol_vel)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_fk_cuda_matches_cpu():
    torch.backends.cuda.matmul.allow_tf32 = False
    model = load_g1_model()
    rng = np.random.default_rng(1)
    qpos, qvel = _random_state(model, rng, 256)
    fk_cpu = Kinematics(model, "cpu", torch.float32)
    fk_gpu = Kinematics(model, "cuda", torch.float32)
    a = fk_cpu(torch.as_tensor(qpos, dtype=torch.float32), torch.as_tensor(qvel, dtype=torch.float32))
    b = fk_gpu(torch.as_tensor(qpos, dtype=torch.float32, device="cuda"), torch.as_tensor(qvel, dtype=torch.float32, device="cuda"))
    for k in a:
        np.testing.assert_allclose(a[k].numpy(), b[k].cpu().numpy(), atol=5e-5)


def test_dash_sim_spec_has_same_kinematics():
    """The simulated Dash spec (forearm-thigh hulls added) must not change the kinematic tree."""
    from supermimic.assets.dash.dash import get_spec

    plain, sim = DASH.load_model(), get_spec().compile()
    assert plain.nbody == sim.nbody and plain.nq == sim.nq and sim.npair == 2
    np.testing.assert_array_equal(plain.body_pos, sim.body_pos)
    np.testing.assert_array_equal(plain.jnt_axis, sim.jnt_axis)
    np.testing.assert_allclose(plain.body_mass, sim.body_mass)


@pytest.mark.skipif(not DASH_NPZ.exists(), reason="Dash npz not copied")
def test_dash_npz_bodies_match_fk():
    """Stored Dash link states == FK of the stored root pose + joints (body axis = model order)."""
    files = sorted(DASH_NPZ.glob("*.npz"))[::5000][:8]
    fk = Kinematics(DASH.load_model(), "cpu", torch.float64)
    for f in files:
        d = np.load(f)
        bp, bq, blv, bav = (d[k].astype(np.float64) for k in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"))
        q = torch.as_tensor(bq[:, 0])
        qpos = torch.cat([torch.as_tensor(bp[:, 0]), q, torch.as_tensor(d["joint_pos"], dtype=torch.float64)], -1)
        ang_b = quat_apply_inverse(q, torch.as_tensor(bav[:, 0]))
        qvel = torch.cat([torch.as_tensor(blv[:, 0]), ang_b, torch.as_tensor(d["joint_vel"], dtype=torch.float64)], -1)
        out = fk(qpos, qvel)
        assert out["pos"].shape[1] == bp.shape[1] == 19, f.name
        np.testing.assert_allclose(out["pos"].numpy(), bp, atol=1e-4, err_msg=f.name)
        assert _signed_quat_err(out["quat"].numpy(), bq) < 1e-4, f.name
        np.testing.assert_allclose(out["ang_vel"].numpy(), bav, atol=1e-3, err_msg=f.name)
        np.testing.assert_allclose(out["lin_vel"].numpy(), blv, atol=1e-3, err_msg=f.name)


def test_fk_is_differentiable():
    model = load_g1_model()
    rng = np.random.default_rng(2)
    qpos, qvel = _random_state(model, rng, 4)
    fk = Kinematics(model, "cpu", torch.float64)
    qp = torch.tensor(qpos, dtype=torch.float64, requires_grad=True)
    qv = torch.tensor(qvel, dtype=torch.float64, requires_grad=True)
    out = fk(qp, qv)
    loss = out["pos"].square().sum() + out["lin_vel"].square().sum() + out["quat"].square().sum()
    loss.backward()
    assert torch.isfinite(qp.grad).all() and torch.isfinite(qv.grad).all()
    assert qp.grad.abs().sum() > 0 and qv.grad.abs().sum() > 0


def test_rotvec_round_trip():
    v = torch.randn(1000, 3, dtype=torch.float64) * 2.5
    v[0] = 0.0
    v[1] = torch.tensor([1e-9, 0, 0])
    q = quat_from_rotvec(v)
    np.testing.assert_allclose(q.norm(dim=-1).numpy(), 1.0, atol=1e-12)
    v2 = rotvec_from_quat(q)
    # Wrap v into (-pi, pi] for the comparison.
    n = v.norm(dim=-1, keepdim=True)
    ang = torch.remainder(n, 2 * np.pi)
    ang = torch.where(ang > np.pi, ang - 2 * np.pi, ang)
    wrapped = torch.where(n > 0, v / n.clamp_min(1e-30) * ang, v)
    np.testing.assert_allclose(v2.numpy(), wrapped.numpy(), atol=1e-9)
    # Gradient finite at zero.
    v0 = torch.zeros(1, 3, dtype=torch.float64, requires_grad=True)
    quat_from_rotvec(v0).sum().backward()
    assert torch.isfinite(v0.grad).all()
