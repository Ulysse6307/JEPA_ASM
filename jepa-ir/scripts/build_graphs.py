#!/usr/bin/env python
"""Pre-build the 3-relation graphs from C/C++ sources into a .pt cache.

Run this LOCALLY (where clang exists). It does the clang-dependent half of the
pipeline (source -> IR -> graph) once, and serializes the graphs. The cache can
then be shipped to a GPU machine with NO clang/LLVM toolchain (e.g. Dalia) and
loaded with `IRGraphDataset.from_cache(...)`.

Usage:
    python scripts/build_graphs.py --sources data/anghabench/sample \
        --glob '*.c' --out data/anghabench/graphs_10k.pt

Then ship the single .pt file and train against it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jepa_ir.config import MaskConfig  # noqa: E402
from jepa_ir.data import IRGraphDataset  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Pre-build graphs into a .pt cache")
    p.add_argument("--sources", required=True, help="dir of C/C++ sources")
    p.add_argument("--glob", default="**/*.c")
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument("--out", required=True, help="output .pt cache path")
    args = p.parse_args()

    ds = IRGraphDataset.from_sources(
        args.sources, MaskConfig(), glob=args.glob, max_files=args.max_files
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.save_cache(out)
    print(f"[build_graphs] {len(ds)} graphs cached -> {out}")
    print("[build_graphs] train clang-free with:")
    print(f"  python scripts/train.py --graph-cache {out}")


if __name__ == "__main__":
    main()