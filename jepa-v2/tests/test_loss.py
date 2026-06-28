"""Factored loss — group invariance, speed remap, and the dimension-independence
property that the v0.1.0 fix restored (invariance must be mean-over-dims)."""
import torch

from jepa_v2.config import LossConfig
from jepa_v2.loss import _group_invariance, _remap_speed, factored_loss


def test_group_invariance_zero_when_members_identical():
    z = torch.randn(4, 8)
    z[1] = z[0]
    z[3] = z[2]
    g = torch.tensor([0, 0, 1, 1])
    assert _group_invariance(z, g).item() < 1e-6


def test_group_invariance_dimension_independent():
    """mean-over-dims: same per-dim distance gives the same value for any D.
    The buggy sum-over-dims version would scale linearly with D."""
    vals = []
    for d in (4, 32, 96):
        z = torch.zeros(2, d)
        z[1] = 0.5  # points at 0 and 0.5 -> centroid 0.25 -> sq dist 0.0625 per elem
        vals.append(_group_invariance(z, torch.tensor([0, 0])).item())
    # mean-over-dims => same value for every D (buggy sum version would scale with D)
    assert all(abs(v - 0.0625) < 1e-6 for v in vals)
    assert abs(vals[0] - vals[-1]) < 1e-6


def test_remap_speed_default_identity():
    lvl = torch.tensor([0, 1, 2, 3])
    assert torch.equal(_remap_speed(lvl), lvl)


def test_factored_loss_finite_and_differentiable():
    torch.manual_seed(0)
    B = 64
    z_sem = torch.randn(B, 16, requires_grad=True)
    z_speed = torch.randn(B, 8, requires_grad=True)
    prog = torch.arange(B) // 4
    lvl = torch.arange(B) % 4
    out = factored_loss(z_sem, z_speed, prog, lvl, LossConfig())
    assert torch.isfinite(out.total)
    out.total.backward()
    assert z_sem.grad is not None and torch.isfinite(z_sem.grad).all()
    for k in ("sem_inv", "speed_inv", "cross", "sem_var", "speed_var"):
        assert k in out.parts


def test_sem_invariance_zero_when_program_views_aligned():
    torch.manual_seed(0)
    B = 32
    prog = torch.arange(B) // 4
    lvl = torch.arange(B) % 4
    base = torch.randn(B // 4, 16)
    z_sem = base[prog]                       # identical across a program's views
    z_speed = torch.randn(B, 8)
    out = factored_loss(z_sem, z_speed, prog, lvl, LossConfig())
    assert out.parts["sem_inv"] < 1e-5


def test_speed_invariance_zero_when_level_views_aligned():
    torch.manual_seed(0)
    B = 32
    prog = torch.arange(B) // 4
    lvl = torch.arange(B) % 4
    base = torch.randn(4, 8)                  # one centroid per -O level
    z_speed = base[lvl]
    z_sem = torch.randn(B, 16)
    out = factored_loss(z_sem, z_speed, prog, lvl, LossConfig())
    assert out.parts["speed_inv"] < 1e-5
