"""VICReg building blocks — the mandatory anti-collapse regularizer.

Ported from v1 (../jepa-ir/src/jepa_ir/train/vicreg.py). Comparing two encodings
admits a degenerate solution (encoder -> constant -> zero invariance loss). VICReg
forbids it with three terms:

  invariance  : MSE(z_a, z_b)                       pulls matched views together
  variance    : hinge keeping each dim's std >= 1   forbids dimensional collapse
  covariance  : push off-diagonal covariance -> 0   decorrelates dimensions

Reference: Bardes, Ponce, LeCun, "VICReg" (2022). We expose the individual terms
(not just the paired loss) because the factored objective in loss.py applies
variance/covariance to whole batches and invariance to GROUPS, not clean pairs.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def off_diagonal(m: torch.Tensor) -> torch.Tensor:
    """Off-diagonal elements of a square matrix as a flat vector."""
    n, _ = m.shape
    return m.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def variance_term(z: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Hinge keeping each feature's std >= 1 (forbids dimensional collapse)."""
    std = torch.sqrt(z.var(dim=0) + eps)
    return torch.mean(F.relu(1.0 - std))


def covariance_term(z: torch.Tensor) -> torch.Tensor:
    """Mean squared off-diagonal covariance (decorrelates dims within a block)."""
    n, _d = z.shape
    z = z - z.mean(dim=0, keepdim=True)
    cov = (z.T @ z) / max(n - 1, 1)
    return off_diagonal(cov).pow(2).mean()


def cross_covariance_term(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Mean squared cross-covariance between two blocks a:[B,Da], b:[B,Db].

    This is the DISENTANGLEMENT term: minimizing it forces z_sem ⟂ z_speed so the
    two heads carry independent information.
    """
    n = a.size(0)
    a = a - a.mean(dim=0, keepdim=True)
    b = b - b.mean(dim=0, keepdim=True)
    cross = (a.T @ b) / max(n - 1, 1)  # [Da, Db]
    return cross.pow(2).mean()
