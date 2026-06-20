"""Optimization predictor: predict emb(O3) from emb(OX), in latent space.

This is a JEPA-style predictor (no decoder): given the embedding of a program
compiled at a lower optimization level (-O0/-O1/-O2), predict the embedding the
SAME program would have at -O3. Target and prediction are both vectors; the loss
is a plain MSE in latent space.

    predictor( encoder(OX) )  ->  z_hat   ;   loss = MSE(z_hat, encoder(O3))

The encoder stays FROZEN — only this small MLP is trained. The point is to learn
"the effect of optimization in latent space" without ever regenerating code.

A residual formulation is used (predict the DELTA to add to the input embedding):
emb(OX) and emb(O3) are already close, so predicting the residual is easier and
makes the identity baseline (predict zero delta) explicit.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class OptPredictor(nn.Module):
    def __init__(self, dim: int, hidden: int = 512, residual: bool = True):
        super().__init__()
        self.residual = residual
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, z_in: torch.Tensor) -> torch.Tensor:
        out = self.net(z_in)
        return z_in + out if self.residual else out