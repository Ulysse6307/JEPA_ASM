"""JEPA model: one shared encoder, two encodings, VICReg on the latent.

Per the project spec, aligned on the eb_jepa reference repo:
  * The SAME encoder encodes the masked context view (A) and the full target
    view (B). This is *joint embedding* — both views go through identical weights
    (no separate target network, no decoder).
  * VICReg regularizes the encoder's latent output DIRECTLY — there is no
    projector/expander head. This is the eb_jepa convention: the anti-collapse
    terms act on the very representation we deliver, not on a throwaway
    projection. (The original image VICReg paper used a projector; eb_jepa drops
    it, and so do we.)

Note on stop-gradient: vanilla VICReg-JEPA lets gradients flow through both
branches (the variance/covariance terms prevent collapse, so no stop-grad or EMA
target is required). eb_jepa does the same — no EMA target encoder.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..config import ModelConfig
from .encoder import IRGraphEncoder


class JEPAModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = IRGraphEncoder(cfg)

    def encode(self, batch) -> torch.Tensor:  # noqa: ANN001
        """Public API: graph batch -> embedding [B, embedding_dim].

        This is the deliverable used at inference (no masking)."""
        return self.encoder(batch)

    def forward(self, context_batch, target_batch):  # noqa: ANN001
        """Return (z_context, z_target): the latent embeddings of both views.

        VICReg is computed on these directly (no projector)."""
        z_a = self.encoder(context_batch)   # masked view
        z_b = self.encoder(target_batch)    # full view
        return z_a, z_b