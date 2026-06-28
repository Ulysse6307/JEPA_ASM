"""VICReg building blocks — including regression guards for the normalization
bug fixed in v0.1.0 (covariance must divide by D, not D(D-1))."""
import torch

from jepa_v2.vicreg import (
    covariance_term, cross_covariance_term, off_diagonal, variance_term,
)


def test_off_diagonal_count():
    m = torch.arange(16).reshape(4, 4).float()
    assert off_diagonal(m).numel() == 4 * 3  # D(D-1)


def test_variance_constant_input_collapses_to_one():
    z = torch.ones(64, 8)
    # std = sqrt(0 + eps) = 0.01 -> hinge relu(1 - 0.01) = 0.99
    assert variance_term(z).item() > 0.98


def test_variance_unit_input_near_zero():
    torch.manual_seed(0)
    z = torch.randn(8192, 8)  # ~unit std per dim
    assert variance_term(z).item() < 0.05


def test_covariance_decorrelated_near_zero():
    torch.manual_seed(0)
    z = torch.randn(8192, 6)
    assert covariance_term(z).item() < 0.05


def test_covariance_normalized_by_D_not_D_times_Dm1():
    """The fix: sum(offdiag^2)/D. The buggy version used .mean() = /(D(D-1))."""
    torch.manual_seed(0)
    n, d = 20000, 4
    base = torch.randn(n, 1)
    # cols 0,1 strongly correlated; 2,3 independent
    z = torch.cat([base, base + 0.01 * torch.randn(n, 1),
                   torch.randn(n, 1), torch.randn(n, 1)], dim=1)
    zc = z - z.mean(0, keepdim=True)
    cov = (zc.T @ zc) / (n - 1)
    expected_sum_over_d = off_diagonal(cov).pow(2).sum() / d
    buggy_mean = off_diagonal(cov).pow(2).mean()           # /(d(d-1))
    got = covariance_term(z)
    assert torch.allclose(got, expected_sum_over_d, atol=1e-5)
    # sum/d is (d-1)x the mean version -> must be clearly larger
    assert got.item() > buggy_mean.item() * 2.0


def test_cross_covariance_independent_low_identical_high():
    torch.manual_seed(0)
    a = torch.randn(8192, 5)
    b = torch.randn(8192, 3)
    indep = cross_covariance_term(a, b).item()
    same = cross_covariance_term(a[:, :3], a[:, :3]).item()
    assert indep < 0.05
    assert same > indep
