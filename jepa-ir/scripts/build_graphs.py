#!/usr/bin/env python
"""Pre-build the 3-relation graphs from C/C++ sources into a .pt cache.

Run this LOCALLY (where clang exists). Does the clang-dependent half of the
pipeline (source -> IR -> graph) once and serializes the graphs, so a GPU machine
with no clang (Dalia) can train from the cache.

Anti-leakage: pass --pool {encoder,predictor,heldout} to keep ONLY the files that
belong to that deterministic pool (see jepa_ir.data.splits). Pools are disjoint by
filename hash, so a program used to train the encoder can never end up in the
predictor's data or in the final held-out set.

Usage:
    # 200k graphs from the ENCODER pool, drawn from the full corpus
    python scripts/build_graphs.py --sources <full_corpus> --glob '**/*.c' \
        --pool encoder --n 200000 --out data/graphs_encoder_200k.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jepa_ir.config import MaskConfig  # noqa: E402
from jepa_ir.data import IRGraphDataset  # noqa: E402
from jepa_ir.data.dataset import _compile_and_build  # noqa: E402
from jepa_ir.data.splits import POOLS, pool_of  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Pre-build graphs into a .pt cache")
    p.add_argument("--sources", required=True, help="dir of C/C++ sources (corpus)")
    p.add_argument("--glob", default="**/*.c")
    p.add_argument("--pool", choices=POOLS, default=None,
                   help="keep only files of this deterministic pool (anti-leakage)")
    p.add_argument("--n", type=int, default=None, help="cap on kept files")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--out", required=True, help="output .pt cache path")
    args = p.parse_args()

    src = Path(args.sources)
    all_files = sorted(src.glob(args.glob))
    if args.pool:
        files = [f for f in all_files if pool_of(f.name) == args.pool]
        print(f"[build] pool={args.pool}: {len(files)}/{len(all_files)} files match")
    else:
        files = all_files
    if args.n is not None:
        files = files[: args.n]
    if not files:
        raise SystemExit("no files selected")

    # compile + build in parallel via the dataset's worker
    import os
    from concurrent.futures import ProcessPoolExecutor

    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)
    print(f"[build] building {len(files)} files on {workers} workers...")
    data_list, n_ok, n_fail, done = [], 0, 0, 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for graphs in ex.map(_compile_and_build, [str(f) for f in files], chunksize=16):
            done += 1
            if graphs is None:
                n_fail += 1
            else:
                data_list.extend(graphs)
                n_ok += 1
            if done % 2000 == 0:
                print(f"  {done}/{len(files)} ({n_ok} ok, {n_fail} fail, {len(data_list)} graphs)")

    ds = IRGraphDataset.from_data_list(data_list, MaskConfig())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.save_cache(out)
    print(f"[build] {len(ds)} graphs cached -> {out} (pool={args.pool})")


if __name__ == "__main__":
    main()