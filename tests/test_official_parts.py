"""Pieces adopted from the official SuperMimic code: running input normalisers, the Muon / Adam optimizer split,
the analytic soft termination."""

import torch


def test_running_norm_matches_batch_statistics():
    from supermimic.world_model.dynamics import RunningNorm

    torch.manual_seed(0)
    x = torch.randn(3000, 5) * torch.tensor([1.0, 2.0, 5.0, 0.1, 3.0]) + torch.tensor([0.0, 1.0, -2.0, 0.5, 10.0])
    n = RunningNorm(5)
    for chunk in x.split(700):  # uneven batches
        n.update(chunk)
    torch.testing.assert_close(n.mean, x.mean(0), atol=1e-4, rtol=0)
    torch.testing.assert_close(n.std, x.std(0, unbiased=False), atol=1e-4, rtol=1e-4)
    y = n(x)
    torch.testing.assert_close(y.mean(0), torch.zeros(5), atol=1e-4, rtol=0)
    assert int(n.count) == 3000


def test_optimizers_put_muon_on_hidden_linears():
    from supermimic.training.optim import Optimizers, hidden_linear_weights
    from supermimic.world_model.dynamics import MLP

    m = MLP(10, (32, 32, 32), 4, act="gelu", layer_norm=True)
    hidden = hidden_linear_weights([m])
    lin = [x for x in m.net if isinstance(x, torch.nn.Linear)]
    assert [id(p) for p in hidden] == [id(lin[1].weight), id(lin[2].weight)]
    opt = Optimizers([m], 1e-3, muon=True)
    assert len(opt.opts) == 2 and isinstance(opt.opts[1], torch.optim.Muon)
    adam_ids = {id(p) for g in opt.opts[0].param_groups for p in g["params"]}
    assert id(lin[0].weight) in adam_ids and id(lin[3].weight) in adam_ids and id(lin[1].weight) not in adam_ids
    before = [p.detach().clone() for p in m.parameters()]
    loss = m(torch.randn(16, 10)).square().mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    assert all(not torch.equal(a, b) for a, b in zip(before, m.parameters()) if b.ndim == 2)
    sd = opt.state_dict()
    opt2 = Optimizers([m], 1e-3, muon=True)
    opt2.load_state_dict(sd)


def test_analytic_termination_is_soft_threshold():
    from types import SimpleNamespace

    from supermimic.world_model.world_model import WorldModel, WorldModelCfg

    cfg = WorldModelCfg()
    wm = SimpleNamespace(cfg=cfg, term_head=None)
    feats = torch.tensor([[0.25, 0.0, 0.0], [0.0, -0.8, 0.0], [0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.1, 0.4, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 3.0]])
    p = torch.sigmoid(WorldModel._term_logits(wm, feats))
    torch.testing.assert_close(p[:2], torch.full((2,), 0.5))
    assert p[2] < 1e-8 and p[3] > 0.999 and p[4] < 1e-4
    assert p[5] < 1e-8 and p[6] < 1e-8  # horizontal distance off by default
    wm.cfg = WorldModelCfg(xy_threshold=1.0)
    p = torch.sigmoid(WorldModel._term_logits(wm, feats))
    torch.testing.assert_close(p[5], torch.tensor(0.5))
    assert p[2] < 1e-8 and p[6] > 0.999
