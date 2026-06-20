#!/usr/bin/env python
"""Pre-compute embeddings at -O0/-O1/-O2/-O3 for N programs, into a cache.

Runs LOCALLY (needs clang). For each program, compile at every opt level, build
the graph, encode with the FROZEN encoder, and store the resulting vectors. The
cache (just float arrays) can then be shipped to Dalia where the predictor trains
on B200 without any clang/IR toolchain.

Output cache (.pt) holds a dict:
    {"opts": ["-O0","-O1","-O2","-O3"],
     "emb": {"-O0": [N,D], ... },        # aligned rows = same program
     "dim": D, "n": N}

Usage:
    python scripts/precompute_embeddings.py \
        --sources ../../data/anghabench/sample_100k --n 8000 \
        --ckpt ../../runs_from_dalia/job_76092_encoder.pt \
        --out ../../data/emb_o0123_8k.pt
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch_geometric.data import Batch  # noqa: E402

from jepa_ir.config import ModelConfig  # noqa: E402
from jepa_ir.data.convert import program_graph_to_data  # noqa: E402
from jepa_ir.graph import build_graph_from_ir  # noqa: E402
from jepa_ir.ir import compile_to_ir, IRCompileError  # noqa: E402
from jepa_ir.model import IRGraphEncoder  # noqa: E402

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


@torch.no_grad()
def encode(enc, datas, bs=256):
    out = []
    for s in range(0, len(datas), bs):
        out.append(enc(Batch.from_data_list(datas[s:s + bs])).numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", required=True)
    ap.add_argument("--glob", default="*.c")
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = blob.get("cfg", ModelConfig())
    enc = IRGraphEncoder(cfg); enc.load_state_dict(blob["encoder"]); enc.eval()

    files = [str(p) for p in sorted(Path(args.sources).glob(args.glob))[: args.n]]
    print(f"[precomp] compiling {len(files)} progs x {len(OPTS)} levels on {args.workers} workers...")
    tasks = [(f, o) for f in files for o in OPTS]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        built = list(ex.map(_build, tasks, chunksize=16))

    # keep programs that built at ALL four levels (aligned rows)
    by_opt = {o: [] for o in OPTS}
    kept = 0
    for i in range(len(files)):
        row = built[i * len(OPTS):(i + 1) * len(OPTS)]
        if any(r is None for r in row):
            continue
        for k, o in enumerate(OPTS):
            by_opt[o].append(row[k])
        kept += 1
    print(f"[precomp] {kept} programs compiled at ALL {len(OPTS)} levels")

    emb = {o: encode(enc, by_opt[o]) for o in OPTS}
    D = emb["-O0"].shape[1]
    torch.save({"opts": OPTS, "emb": emb, "dim": D, "n": kept}, args.out)
    print(f"[precomp] saved embeddings cache -> {args.out}  (n={kept}, dim={D})")
    print(f"[precomp] ship to Dalia, then: train_predictor.py --emb-cache <path>")


if __name__ == "__main__":
    main()