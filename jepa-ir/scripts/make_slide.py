#!/usr/bin/env python
"""Make a single slide: C code -> LLVM IR -> 3-relation graph, side by side.

For oral presentation. Three panels on one figure:
  left   : the C source
  middle : the (cleaned) LLVM IR
  right  : the program graph with the 3 colored relations

Usage:
    python scripts/make_slide.py examples/demo_oral.c --out slide.png
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402

from jepa_ir.graph import build_graph_from_source  # noqa: E402
from jepa_ir.ir import compile_to_ir  # noqa: E402

EDGE_STYLE = {
    "control": dict(color="#d62728", style="solid",  rad=0.0,   label="control (controle)"),
    "data":    dict(color="#1f77b4", style="solid",  rad=0.12,  label="data (donnees)"),
    "memory":  dict(color="#2ca02c", style="dashed", rad=-0.18, label="memory (effets)"),
}


def clean_ir(ir: str) -> str:
    keep = []
    for ln in ir.splitlines():
        s = ln.strip()
        if not s or s.startswith((";", "source_", "target", "attributes", "!")):
            continue
        if "tbaa" in ln:
            ln = ln.split(", !tbaa")[0]          # drop metadata tail
        keep.append(ln)
    return "\n".join(keep)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--out", default="slide.png")
    args = ap.parse_args()
    path = Path(args.path)

    code = path.read_text().rstrip()
    ir = clean_ir(compile_to_ir(path))
    g = build_graph_from_source(code)[0]

    fig = plt.figure(figsize=(18, 7))
    fig.suptitle("code  →  LLVM IR  →  graphe 3-relations",
                 fontsize=17, weight="bold")

    mono = {"family": "monospace", "fontsize": 11, "va": "top"}

    # panel 1: C code
    ax1 = fig.add_subplot(1, 3, 1); ax1.axis("off")
    ax1.set_title("1. code C", fontsize=13, weight="bold", loc="left")
    ax1.text(0.0, 0.95, code, transform=ax1.transAxes, **mono)

    # panel 2: IR
    ax2 = fig.add_subplot(1, 3, 2); ax2.axis("off")
    ax2.set_title("2. LLVM IR (clang -emit-llvm)", fontsize=13, weight="bold", loc="left")
    ax2.text(0.0, 0.98, ir, transform=ax2.transAxes,
             **{**mono, "fontsize": 8.5})

    # panel 3: graph
    ax3 = fig.add_subplot(1, 3, 3); ax3.axis("off")
    ax3.set_title("3. graphe (1 noeud = 1 instruction)", fontsize=13,
                  weight="bold", loc="left")

    G = nx.DiGraph()
    for nd in g.nodes:
        G.add_node(nd.idx, label=f"{nd.idx}: {nd.opcode}")
    for et, pairs in g.edges.items():
        for s, d in pairs:
            G.add_edge(s, d, rel=et)
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    except Exception:
        pos = nx.spring_layout(G, seed=1, k=2.0, iterations=300)

    colors = ["#c7e9c0" if nd.is_memory_op else
              "#fbb4ae" if nd.is_terminator else "#dddddd" for nd in g.nodes]
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=1400,
                           edgecolors="#333", ax=ax3)
    nx.draw_networkx_labels(G, pos, labels=nx.get_node_attributes(G, "label"),
                            font_size=8, ax=ax3)
    for et, st in EDGE_STYLE.items():
        edges = [(s, d) for s, d, r in G.edges(data="rel") if r == et]
        if edges:
            nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=st["color"],
                                   style=st["style"], width=2.0, arrowsize=16,
                                   connectionstyle=f"arc3,rad={st['rad']}", ax=ax3)
    legend = [mpatches.Patch(color=s["color"], label=s["label"]) for s in EDGE_STYLE.values()]
    ax3.legend(handles=legend, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.08))

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(args.out, dpi=130)
    print(f"[slide] saved -> {args.out}  ({g.summary()})")


if __name__ == "__main__":
    main()