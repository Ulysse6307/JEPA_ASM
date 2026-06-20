"""In-memory representation of a lossless program graph.

A ProgramGraph is framework-agnostic (no torch here) so the builder stays
testable without heavy deps. Conversion to PyG lives in jepa_ir.data.

Nodes are LLVM instructions. Edges come in three typed relations:
  control : block-level CFG, lifted onto the terminator/entry instructions
  data    : SSA def -> use (value produced by node A is an operand of node B)
  memory  : program order between memory-touching instructions (side effects)
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..config import EDGE_TYPES


@dataclass
class Node:
    """One LLVM instruction."""
    idx: int                      # node index within the graph (0-based, stable)
    opcode: str                   # llvm opcode, e.g. "load", "br", "add"
    block: int                    # index of the containing basic block
    result_name: str | None       # SSA name it defines (e.g. "%3"), or None
    operands: list[str] = field(default_factory=list)  # operand value names it reads
    is_terminator: bool = False
    is_memory_op: bool = False
    produces_value: bool = False
    text: str = ""                # raw IR line (debug / inspection)


@dataclass
class ProgramGraph:
    """A single program (typically one function) as a typed multigraph.

    edges[etype] is a list of (src_idx, dst_idx) pairs for that relation.
    """
    nodes: list[Node] = field(default_factory=list)
    edges: dict[str, list[tuple[int, int]]] = field(
        default_factory=lambda: {et: [] for et in EDGE_TYPES}
    )
    name: str = ""                # function name, for provenance

    # -- convenience ------------------------------------------------------- #
    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    def num_edges(self, etype: str | None = None) -> int:
        if etype is None:
            return sum(len(v) for v in self.edges.values())
        return len(self.edges[etype])

    def add_edge(self, etype: str, src: int, dst: int) -> None:
        if etype not in self.edges:
            raise KeyError(f"unknown edge type {etype!r}; expected one of {EDGE_TYPES}")
        self.edges[etype].append((src, dst))

    def summary(self) -> str:
        per = ", ".join(f"{et}={len(self.edges[et])}" for et in EDGE_TYPES)
        return f"<ProgramGraph {self.name!r} nodes={self.num_nodes} edges[{per}]>"