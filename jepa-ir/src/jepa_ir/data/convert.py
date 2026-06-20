"""Convert a ProgramGraph into a PyG Data object.

Design: nodes are homogeneous (all instructions); the three relations are kept
as separate edge_index tensors on one Data object. This is simpler than
HeteroData (we have a single node type) and batches cleanly because every
per-relation edge_index is offset consistently by PyG's default collation when
we register them as edge attributes via `__inc__`.

Node features fed to the model are split into two parts:
  x_opcode : Long tensor [N]      -> embedded by the model (categorical)
  x_struct : Float tensor [N, S]  -> concatenated after the opcode embedding

We keep them separate (rather than pre-concatenating) so the model owns the
opcode embedding table and the mask token can replace the *opcode embedding*
cleanly for masked nodes.
"""
from __future__ import annotations

import torch
from torch_geometric.data import Data

from ..config import EDGE_TYPES, OPCODE_TO_ID, OPCODE_UNK
from ..graph.schema import ProgramGraph


def _edge_attr_key(etype: str) -> str:
    return f"edge_index_{etype}"


def node_blocks(graph: ProgramGraph) -> torch.Tensor:
    """Return a Long tensor [N] giving the basic-block index of each node.
    Used by block-level masking."""
    return torch.tensor([nd.block for nd in graph.nodes], dtype=torch.long)


def node_features(graph: ProgramGraph) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (x_opcode [N] long, x_struct [N, S] float)."""
    n = graph.num_nodes
    x_opcode = torch.empty(n, dtype=torch.long)
    x_struct = torch.zeros(n, len(_STRUCT_BUILDERS), dtype=torch.float)
    # precompute a normalization constant for operand counts
    max_ops = max((len(nd.operands) for nd in graph.nodes), default=1) or 1
    for nd in graph.nodes:
        x_opcode[nd.idx] = OPCODE_TO_ID.get(nd.opcode, OPCODE_UNK)
        for j, fn in enumerate(_STRUCT_BUILDERS):
            x_struct[nd.idx, j] = fn(nd, max_ops)
    return x_opcode, x_struct


# Structural feature builders, in the order declared by config.STRUCT_FEATURES.
def _f_is_terminator(nd, _max):  # noqa: ANN001
    return 1.0 if nd.is_terminator else 0.0


def _f_is_memory(nd, _max):  # noqa: ANN001
    return 1.0 if nd.is_memory_op else 0.0


def _f_num_operands_norm(nd, max_ops):  # noqa: ANN001
    return len(nd.operands) / max_ops


def _f_produces_value(nd, _max):  # noqa: ANN001
    return 1.0 if nd.produces_value else 0.0


_STRUCT_BUILDERS = (
    _f_is_terminator,
    _f_is_memory,
    _f_num_operands_norm,
    _f_produces_value,
)


def _edges_to_tensor(pairs: list[tuple[int, int]]) -> torch.Tensor:
    if not pairs:
        return torch.empty(2, 0, dtype=torch.long)
    return torch.tensor(pairs, dtype=torch.long).t().contiguous()


class IRData(Data):
    """Data subclass that knows how to increment per-relation edge indices when
    PyG concatenates graphs into a batch."""

    def __inc__(self, key, value, *args, **kwargs):  # noqa: ANN001
        if key.startswith("edge_index_"):
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):  # noqa: ANN001
        if key.startswith("edge_index_"):
            return 1  # concatenate along the edge dimension
        return super().__cat_dim__(key, value, *args, **kwargs)


def program_graph_to_data(graph: ProgramGraph) -> IRData:
    x_opcode, x_struct = node_features(graph)
    data = IRData()
    data.x_opcode = x_opcode
    data.x_struct = x_struct
    data.node_block = node_blocks(graph)
    data.num_nodes = graph.num_nodes
    for et in EDGE_TYPES:
        setattr(data, _edge_attr_key(et), _edges_to_tensor(graph.edges[et]))
    data.name = graph.name
    return data