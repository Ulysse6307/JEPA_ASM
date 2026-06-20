"""The GNN encoder: program graph -> embedding.

This is the deliverable. Trained from scratch (no pretrained GNN), it maps a
lossless 3-relation IR graph to a fixed-size vector.

Node input assembly:
  h0 = [ opcode_embed(opcode)  OR  mask_token   ;   struct_features ]
       + structural_positional_encoding
  * masked nodes (mask flag True) take a single LEARNED mask token in place of
    the opcode embedding, so the encoder knows where the holes are.
  * struct features of masked nodes are zeroed by the masker; we still append
    them (zeros) to keep the vector layout identical between views.

Message passing:
  For each of the three relations we keep a separate GraphConv. Each layer sums
  the per-relation messages, so all three relations propagate jointly (the graph
  is never reduced to a single relation). Residual + LayerNorm between layers.

Pooling:
  Mean + max pooling concatenated, then a linear projection to embedding_dim.

Positional/structural encoding:
  We add a cheap degree-based structural encoding (per-relation in/out degree,
  log1p-scaled) so otherwise-identical opcodes in different structural positions
  get distinguishable initial features. (A full Laplacian PE can be swapped in
  later; degree PE is fast and batch-friendly.)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv
from torch_geometric.utils import degree

from ..config import EDGE_TYPES, ModelConfig


def _edge_key(et: str) -> str:
    return f"edge_index_{et}"


class StructuralPE(nn.Module):
    """Per-relation degree encoding -> projected to hidden_dim and added to h0."""

    def __init__(self, hidden_dim: int, num_relations: int):
        super().__init__()
        # 2 (in/out) * num_relations degree features
        self.lin = nn.Linear(2 * num_relations, hidden_dim)

    def forward(self, num_nodes: int, edge_indices: list[torch.Tensor], device) -> torch.Tensor:  # noqa: ANN001
        feats = []
        for ei in edge_indices:
            if ei.numel() == 0:
                indeg = torch.zeros(num_nodes, device=device)
                outdeg = torch.zeros(num_nodes, device=device)
            else:
                outdeg = degree(ei[0], num_nodes=num_nodes).to(device)
                indeg = degree(ei[1], num_nodes=num_nodes).to(device)
            feats.append(torch.log1p(indeg))
            feats.append(torch.log1p(outdeg))
        x = torch.stack(feats, dim=1)  # [N, 2*R]
        return self.lin(x)


class IRGraphEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.edge_types = cfg.edge_types

        self.opcode_embed = nn.Embedding(
            cfg.opcode_vocab_size, cfg.opcode_embed_dim, padding_idx=0
        )
        # single learned mask token, same dim as the opcode embedding
        self.mask_token = nn.Parameter(torch.zeros(cfg.opcode_embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        in_dim = cfg.opcode_embed_dim + cfg.num_struct_features
        self.input_proj = nn.Linear(in_dim, cfg.hidden_dim)
        self.pe = StructuralPE(cfg.hidden_dim, len(self.edge_types))

        # one conv per relation, per layer
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(cfg.num_layers):
            layer = nn.ModuleDict(
                {et: GraphConv(cfg.hidden_dim, cfg.hidden_dim) for et in self.edge_types}
            )
            self.convs.append(layer)
            self.norms.append(nn.LayerNorm(cfg.hidden_dim))

        self.dropout = nn.Dropout(cfg.dropout)
        # pooled (mean||max) -> embedding
        self.out_proj = nn.Linear(2 * cfg.hidden_dim, cfg.embedding_dim)

    # ------------------------------------------------------------------ #
    def _assemble_input(self, batch) -> torch.Tensor:  # noqa: ANN001
        op = self.opcode_embed(batch.x_opcode)            # [N, De]
        if hasattr(batch, "mask") and batch.mask is not None:
            m = batch.mask.bool()
            op = op.clone()
            op[m] = self.mask_token
        h = torch.cat([op, batch.x_struct], dim=1)        # [N, De+S]
        h = self.input_proj(h)                            # [N, H]
        edge_indices = [getattr(batch, _edge_key(et)) for et in self.edge_types]
        h = h + self.pe(int(batch.num_nodes), edge_indices, h.device)
        return h

    def forward_nodes(self, batch) -> torch.Tensor:  # noqa: ANN001
        """Return per-node embeddings [N, H] (before pooling)."""
        h = self._assemble_input(batch)
        edge_indices = {et: getattr(batch, _edge_key(et)) for et in self.edge_types}
        for conv_layer, norm in zip(self.convs, self.norms):
            msg = torch.zeros_like(h)
            for et in self.edge_types:
                ei = edge_indices[et]
                if ei.numel() > 0:
                    msg = msg + conv_layer[et](h, ei)
            h = norm(h + self.dropout(F.relu(msg)))       # residual + norm
        return h

    def forward(self, batch) -> torch.Tensor:  # noqa: ANN001
        """Return graph-level embeddings [B, embedding_dim]."""
        from torch_geometric.nn import global_max_pool, global_mean_pool

        h = self.forward_nodes(batch)
        b = batch.batch if hasattr(batch, "batch") else torch.zeros(
            h.size(0), dtype=torch.long, device=h.device
        )
        pooled = torch.cat([global_mean_pool(h, b), global_max_pool(h, b)], dim=1)
        return self.out_proj(pooled)                      # [B, embedding_dim]