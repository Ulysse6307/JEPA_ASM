"""Masking: produce the two JEPA views from one graph.

JEPA here (no decoder):
  * view B (target)  = the FULL graph, encoded as-is.
  * view A (context) = the SAME graph with a subset of nodes MASKED. A masked
    node keeps its position in the graph and all its edges (structure is NOT
    removed), but its *content* (opcode + struct features) is replaced by a
    learned mask token, and a boolean `mask` flag marks it.

The point (per project spec): we do NOT predict the missing content. Masking is
a *constraint* — forced to produce, from a partial graph, an embedding coherent
with the full graph's embedding, the encoder must learn real program structure.

Two masking strategies:
  * block_masking=True  : mask whole basic blocks (harder, more structural).
  * block_masking=False : mask random individual nodes.

A deterministic per-call RNG (seeded by `generator`) keeps runs reproducible and
lets the dataset draw a fresh mask each epoch.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..config import MaskConfig, OPCODE_PAD
from .convert import IRData


@dataclass
class MaskedView:
    """A masked context view paired with its target (full) view.

    Both share the SAME graph topology; only the masked node contents differ.
    """
    context: IRData      # view A: masked
    target: IRData       # view B: full
    mask: torch.Tensor   # bool [N], True where masked


def _select_masked_nodes(
    data: IRData, cfg: MaskConfig, generator: torch.Generator
) -> torch.Tensor:
    n = int(data.num_nodes)
    if n <= cfg.min_kept_nodes:
        return torch.zeros(n, dtype=torch.bool)  # too small: mask nothing

    n_mask = int(round(cfg.mask_ratio * n))
    n_mask = min(n_mask, n - cfg.min_kept_nodes)
    n_mask = max(n_mask, 1)

    mask = torch.zeros(n, dtype=torch.bool)
    if cfg.block_masking and hasattr(data, "node_block"):
        # mask whole blocks until we reach ~n_mask nodes
        blocks = data.node_block  # long [N]
        uniq = blocks.unique()
        perm = uniq[torch.randperm(uniq.numel(), generator=generator)]
        chosen = 0
        for b in perm:
            sel = blocks == b
            cnt = int(sel.sum())
            if chosen + cnt > n - cfg.min_kept_nodes and chosen > 0:
                continue
            mask |= sel
            chosen += cnt
            if chosen >= n_mask:
                break
        if mask.all():  # safety: never mask everything
            keep = torch.randperm(n, generator=generator)[: cfg.min_kept_nodes]
            mask[keep] = False
    else:
        perm = torch.randperm(n, generator=generator)[:n_mask]
        mask[perm] = True
    return mask


def mask_graph(
    data: IRData, cfg: MaskConfig, generator: torch.Generator | None = None
) -> MaskedView:
    """Build (context, target) views. `data` is left untouched (we clone)."""
    if generator is None:
        generator = torch.Generator().manual_seed(0)

    mask = _select_masked_nodes(data, cfg, generator)

    # target = full graph, with an all-False mask flag
    target = data.clone()
    target.mask = torch.zeros(int(data.num_nodes), dtype=torch.bool)

    # context = clone where masked nodes get sentinel opcode + zeroed struct
    context = data.clone()
    context.x_opcode = context.x_opcode.clone()
    context.x_struct = context.x_struct.clone()
    context.x_opcode[mask] = OPCODE_PAD          # sentinel; model swaps in mask token
    context.x_struct[mask] = 0.0
    context.mask = mask

    # Optionally cut the edges incident to masked nodes -> a real structural hole.
    # Nodes stay (so the mask token still marks the position) but lose their links,
    # forcing the encoder to infer the embedding from the surviving subgraph.
    if cfg.mask_edges:
        from ..config import EDGE_TYPES

        for et in EDGE_TYPES:
            key = f"edge_index_{et}"
            ei = getattr(context, key)
            if ei.numel() == 0:
                continue
            src_ok = ~mask[ei[0]]
            dst_ok = ~mask[ei[1]]
            keep = src_ok & dst_ok          # keep edges with BOTH endpoints unmasked
            setattr(context, key, ei[:, keep].contiguous())

    return MaskedView(context=context, target=target, mask=mask)