"""Build graphs for the PREDICTOR pool at all 4 opt levels (O0/O1/O2/O3).

Anti-leakage: only files in the 'predictor' pool (disjoint from the encoder pool)
are used. For each program we compile at -O0/-O1/-O2/-O3, build the graph, and
keep only programs that built at ALL FOUR levels (aligned rows). The encoder is
NOT needed here — we cache raw graphs; embeddings come later, once the encoder is
trained, via precompute_embeddings on this cache.

Output (.pt):
    {"opts": ["-O0",...], "graphs": {"-O0":[IRData...], ...},
     "names": [...], "n": N, "subsplit": ["train"/"val"/"test", ...]}

Usage:
    python scripts/build_predictor_graphs.py --sources <corpus> --glob '**/*.c' \
        --n 60000 --out ../../data/predictor_graphs.pt
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch  # noqa: E402

from jepa_ir.data.convert import program_graph_to_data  # noqa: E402
from jepa_ir.data.splits import pool_of, subsplit  # noqa: E402
from jepa_ir.graph import build_graph_from_ir  # noqa: E402
from jepa_ir.ir import compile_to_ir, IRCompileError  # noqa: E402

OPTS = ["-O0", "-O1", "-O2", "-O3"]


def _build(args):
    path, opt = args
    try:
        graphs = build_graph_from_ir(compile_to_ir(path, opt_level=opt))
    except (IRCompileError, Exception):  # noqa: BLE001
        return None
    if not graphs:
        return None
    return program_graph_to_data(max(graphs, key=lambda g: g.num_nodes))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", required=True)
    ap.add_argument("--glob", default="**/*.c")
    ap.add_argument("--n", type=int, default=60000, help="cap on predictor-pool files")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    allf = sorted(Path(args.sources).glob(args.glob))
    files = [f for f in allf if pool_of(f.name) == "predictor"][: args.n]
    print(f"[predbuild] {len(files)} predictor-pool files (of {len(allf)} scanned)")

    tasks = [(str(f), o) for f in files for o in OPTS]
    print(f"[predbuild] compiling {len(files)} x {len(OPTS)} = {len(tasks)} on {args.workers} workers...")
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        built = list(ex.map(_build, tasks, chunksize=16))

    graphs = {o: [] for o in OPTS}
    names, subs = [], []
    kept = 0
    for i, f in enumerate(files):
        row = built[i * len(OPTS):(i + 1) * len(OPTS)]
        if any(r is None for r in row):
            continue
        for k, o in enumerate(OPTS):
            graphs[o].append(row[k])
        names.append(f.name)
        subs.append(subsplit(f.name))   # deterministic train/val/test
        kept += 1
    print(f"[predbuild] {kept} programs built at ALL {len(OPTS)} levels")
    from collections import Counter
    print(f"[predbuild] subsplit: {dict(Counter(subs))}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"opts": OPTS, "graphs": graphs, "names": names,
                "subsplit": subs, "n": kept}, out)
    print(f"[predbuild] saved -> {out}")


if __name__ == "__main__":
    main()