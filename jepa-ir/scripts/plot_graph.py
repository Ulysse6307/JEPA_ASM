#!/usr/bin/env python
"""Visualize the 3-relation ProgramGraph built from a C/C++ file (or .ll).

Draws instruction nodes (labelled by opcode) with the three edge relations in
distinct colors:
    control  -> red
    data     -> blue
    memory   -> green

Usage:
    python scripts/plot_graph.py examples/sum_array.c --out graph_sum_array.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402

from jepa_ir.graph import build_graph_from_source, build_graph_from_ir  # noqa: E402

EDGE_STYLE = {
    "control": dict(color="#d62728", style="solid",  label="control (flot de controle)"),
    "data":    dict(color="#1f77b4", style="solid",  label="data (flot de donnees)"),
    "memory":  dict(color="#2ca02c", style="dashed", label="memory (ordre des effets)"),
}


def main() -> None:
    p = argparse.ArgumentParser(description="Plot the 3-relation program graph")
    p.add_argument("path", help="C/C++ source (.c/.cpp) or LLVM IR (.ll)")
    p.add_argument("--out", default="program_graph.png")
    p.add_argument("--index", type=int, default=0, help="which function (if many)")
    args = p.parse_args()

    path = Path(args.path)
    if path.suffix == ".ll":
        graphs = build_graph_from_ir(path.read_text())
    else:
        graphs = build_graph_from_source(
            path.read_text(), is_cpp=path.suffix in (".cpp", ".cc", ".cxx")
        )
    g = graphs[args.index]
    print(g.summary())

    # build a networkx DiGraph for layout; keep edges grouped by relation
    G = nx.DiGraph()
    for nd in g.nodes:
        G.add_node(nd.idx, label=f"{nd.idx}\n{nd.opcode}")
    for et, pairs in g.edges.items():
        for s, d in pairs:
            G.add_edge(s, d, rel=et)

    # layout: hierarchical-ish via graphviz if available, else spring
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    except Exception:
        pos = nx.spring_layout(G, seed=0, k=1.5, iterations=200)

    fig, ax = plt.subplots(figsize=(13, 10))
    fig.suptitle(f"ProgramGraph 3-relations : {g.name}", fontsize=15, weight="bold")

    # color nodes: memory ops greenish, terminators reddish, else grey
    node_colors = []
    for nd in g.nodes:
        if nd.is_memory_op:
            node_colors.append("#c7e9c0")
        elif nd.is_terminator:
            node_colors.append("#fbb4ae")
        else:
            node_colors.append("#dddddd")

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=900,
                           edgecolors="#333", linewidths=1.0, ax=ax)
    nx.draw_networkx_labels(G, pos, labels=nx.get_node_attributes(G, "label"),
                            font_size=7, ax=ax)

    # draw each relation separately with its style (curved to separate parallels)
    for et, style in EDGE_STYLE.items():
        edges = [(s, d) for s, d, r in G.edges(data="rel") if r == et]
        if not edges:
            continue
        rad = {"control": 0.0, "data": 0.12, "memory": -0.18}[et]
        nx.draw_networkx_edges(
            G, pos, edgelist=edges, edge_color=style["color"],
            style=style["style"], width=1.8, alpha=0.8, arrowsize=14,
            connectionstyle=f"arc3,rad={rad}", ax=ax,
        )

    legend = [mpatches.Patch(color=s["color"], label=s["label"])
              for s in EDGE_STYLE.values()]
    legend += [
        mpatches.Patch(color="#c7e9c0", label="noeud memoire (load/store/call)"),
        mpatches.Patch(color="#fbb4ae", label="noeud terminator (br/ret)"),
    ]
    ax.legend(handles=legend, loc="upper left", fontsize=9)
    ax.axis("off")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.out, dpi=130)
    print(f"[plot] saved -> {args.out}")


if __name__ == "__main__":
    main()