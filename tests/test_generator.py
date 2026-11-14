"""Motion generator: canonical frame round trip, heightmaps equal the terrain lookup, window bounds, model and
sampler shapes."""

from pathlib import Path

import numpy as np
import pytest
import torch

from supermimic.generator.data import GEN_GRID, GRID_SHAPE, from_canonical, rot6d_to_matrix, to_canonical
from supermimic.generator.model import Denoiser, Diffusion, GenCfg
from supermimic.kinematics.rotations import matrix_from_quat, quat_normalize, rot6d_from_quat

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "data/library/dash_flat_terrain_v5_parc_compact.pt"


def test_canonical_round_trip():
    g = torch.Generator().manual_seed(0)
    B, T, nj = 4, 7, 18
    pos = torch.randn(B, T, 3, generator=g)
    quat = quat_normalize(torch.randn(B, T, 4, generator=g))
    q = torch.randn(B, T, nj, generator=g)
    origin, yaw0 = torch.randn(B, 3, generator=g), torch.rand(B, generator=g) * 6 - 3
    f = to_canonical(pos, quat, q, origin, yaw0)
    p2, q2, j2 = from_canonical(f, origin, yaw0)
    torch.testing.assert_close(p2, pos, atol=1e-5, rtol=0)
    torch.testing.assert_close(matrix_from_quat(q2), matrix_from_quat(quat), atol=1e-5, rtol=0)
    torch.testing.assert_close(j2, q)
    torch.testing.assert_close(rot6d_to_matrix(rot6d_from_quat(quat)), matrix_from_quat(quat), atol=1e-5, rtol=0)
    # the origin frame itself maps to zero position and identity heading
    f0 = to_canonical(origin[:, None], quat[:, :1], q[:, :1], origin, torch.zeros(B))
    torch.testing.assert_close(f0[..., :3], torch.zeros(B, 1, 3))


def test_model_and_sampler_shapes():
    cfg = GenCfg(d=64, layers=2, heads=2, hist=4, fut=8)
    m = Denoiser(cfg)
    B = 3
    hist, x = torch.randn(B, 4, cfg.frame_dim), torch.randn(B, 8, cfg.frame_dim)
    hmap, cmd, on = torch.randn(B, *GRID_SHAPE), torch.randn(B, 12), torch.ones(B)
    assert m(x, torch.randint(1000, (B,)), hist, hmap, cmd, on).shape == x.shape
    out = Diffusion(cfg.T).sample(m.eval(), hist, hmap, cmd, on, steps=3, guidance=1.5)
    assert out.shape == x.shape and torch.isfinite(out).all()


@pytest.mark.skipif(not torch.cuda.is_available() or not LIB.exists(), reason="needs CUDA and the v5 library")
def test_windows_match_library():
    from supermimic.generator.data import GenData
    from supermimic.terrain.terrains import load_terrains

    d = GenData(LIB, "cuda:0")
    w = d.sample(256, generator=torch.Generator(device="cuda:0").manual_seed(0))
    s = d.clip_start[w.clip]
    assert bool((w.t0 + d.F < d.clip_len[w.clip]).all())
    r = d.ref[s[:, None] + w.t0[:, None] + torch.arange(1, d.F + 1, device="cuda:0")].float()
    pos, _, j = from_canonical(w.fut, w.origin, w.yaw0)
    torch.testing.assert_close(pos, r[..., :3], atol=1e-4, rtol=0)
    torch.testing.assert_close(j, r[..., 7:25])
    assert float(w.hist[:, -1, :3].abs().max()) == 0.0
    ts = load_terrains(LIB)
    off = GEN_GRID.offsets().numpy()
    for b in range(0, 128, 8):  # terrain half
        o, y, t = w.origin[b].cpu().numpy(), float(w.yaw0[b]), int(w.tile[b])
        assert t >= 0
        xy = o[:2] + np.stack([np.cos(y) * off[:, 0] - np.sin(y) * off[:, 1], np.sin(y) * off[:, 0] + np.cos(y) * off[:, 1]], -1)
        np.testing.assert_allclose((ts.terrains[t].h(xy) - o[2]).reshape(GRID_SHAPE), w.hmap[b].cpu().numpy(), atol=1e-4)
    assert bool((w.tile[128:] == -1).all()) and float((w.hmap[128:] + w.origin[128:, 2:3, None]).abs().max()) == 0.0
