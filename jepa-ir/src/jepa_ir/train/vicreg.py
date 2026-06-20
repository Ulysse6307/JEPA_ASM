"""VICReg loss — the mandatory anti-collapse regularizer.

Comparing two encodings admits a degenerate solution: encoder -> constant vector
-> zero invariance loss. VICReg forbids it with three terms:

  invariance  : MSE(z_a, z_b)                       pulls the two views together
  variance    : hinge keeping each dim's std >= 1    forbids dimensional collapse
  covariance  : push off-diagonal covariance -> 0    decorrelates dimensions

Reference: Bardes, Ponce, LeCun, "VICReg" (2022). The variance + covariance terms
are what make collapse impossible, which is exactly why the project mandates this
style of regularization.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..config import VICRegConfig


@dataclass
class VICRegOutput:
    total: torch.Tensor
    invariance: torch.Tensor
    variance: torch.Tensor
    covariance: torch.Tensor

    def item_dict(self) -> dict[str, float]:
        return {
            "total": float(self.total.detach()),
            "inv": float(self.invariance.detach()),
            "var": float(self.variance.detach()),
            "cov": float(self.covariance.detach()),
        }


def _off_diagonal(m: torch.Tensor) -> torch.Tensor:
    """Return the off-diagonal elements of a square matrix as a flat vector."""
    n, _ = m.shape
    return m.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def _variance_term(z: torch.Tensor, eps: float) -> torch.Tensor:
    # std along the batch dimension for each feature; hinge at 1.
    std = torch.sqrt(z.var(dim=0) + eps)
    return torch.mean(F.relu(1.0 - std))


def _covariance_term(z: torch.Tensor) -> torch.Tensor:
    # eb_jepa convention: mean of squared off-diagonal covariances (CovarianceLoss).
    n, _d = z.shape
    z = z - z.mean(dim=0, keepdim=True)
    cov = (z.T @ z) / (n - 1)
    return _off_diagonal(cov).pow(2).mean()


def vicreg_loss(
    z_a: torch.Tensor, z_b: torch.Tensor, cfg: VICRegConfig
) -> VICRegOutput:
    """Compute VICReg over a batch of paired projected embeddings.

    z_a, z_b : [B, D]  (B = batch graphs, D = embedding_dim)
    Needs B >= 2 for variance/covariance to be meaningful.
    """
    inv = F.mse_loss(z_a, z_b)

    var = _variance_term(z_a, cfg.eps) + _variance_term(z_b, cfg.eps)

    cov = _covariance_term(z_a) + _covariance_term(z_b)

    total = cfg.sim_coeff * inv + cfg.std_coeff * var + cfg.cov_coeff * cov
    return VICRegOutput(total=total, invariance=inv, variance=var, covariance=cov)