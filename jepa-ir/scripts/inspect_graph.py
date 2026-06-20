#!/usr/bin/env python
"""Inspect the 3-relation graph built from a single C/C++ file.

Usage:
    python scripts/inspect_graph.py examples/sum_array.c
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jepa_ir.graph import build_graph_from_source  # noqa: E402


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    path = Path(sys.argv[1])
    is_cpp = path.suffix in (".cpp", ".cc", ".cxx")
    graphs = build_graph_from_source(path.read_text(), is_cpp=is_cpp)
    for g in graphs:
        print(g.summary())
        print("  nodes:")
        for nd in g.nodes:
            print(f"    [{nd.idx:3d}] blk{nd.block} {nd.opcode:16s}"
                  f" def={str(nd.result_name):18s} ops={nd.operands}")
        for et in ("control", "data", "memory"):
            print(f"  {et:8s} ({len(g.edges[et])}): {g.edges[et]}")
        print()


if __name__ == "__main__":
    main()