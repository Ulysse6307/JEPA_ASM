#!/usr/bin/env python
"""Distribution of embedding similarity O0 vs O3, at scale (parallel compile).

For N programs, compute emb(O0) and emb(O3) (compilation parallelized across
processes; encoding batched), then report the full distribution of:
  cos(O0_i, O3_i)            same program, O0 vs O3
  cos(O0_i, O0_j) i!=j       O0 vs O0 (different programs)
  cos(O3_i, O3_j) i!=j       O3 vs O3 (different programs)
  cos(O0_i, O3_j) i!=j       O0 vs O3 (different programs)  [the reference]
plus L2 versions. Tells us whether the encoder ties O0/O3 of the same program
together (good signal for a downstream distance loss) or treats them as strangers
/ clusters them by optimization level.

Usage:
    python scripts/measure_o0_o3.py --sources data/anghabench/sample_100k \
        --n 5000 --ckpt ../runs_from_dalia/job_76092_encoder.pt \
        --out ../runs_from_dalia/o0_o3_5k.png
"""
from __future__ import annotations

import argparse
import sys
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


def _build(args):
    """Worker: (path, opt) -> IRData or None. Module-level for ProcessPool."""
    path, opt = args
    try:
        graphs = build_graph_from_ir(compile_to_ir(path, opt_level=opt))
    except (IRCompileError, Exception):  # noqa: BLE001
        return None
    if not graphs:
        return None
    return program_graph_to_data(max(graphs, key=lambda g: g.num_nodes))


def load_encoder(ckpt):
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    enc = IRGraphEncoder(blob.get("cfg", ModelConfig()))
    enc.load_state_dict(blob["encoder"]); enc.eval()
    return enc


@torch.no_grad()
def encode_all(enc, datas, bs=256):
    """Encode a list of IRData (some None) -> array [n, D] keeping only non-None
    indices; returns (emb, keep_mask)."""
    keep = [i for i, d in enumerate(datas) if d is not None]
    out = []
    for s in range(0, len(keep), bs):
        chunk = [datas[i] for i in keep[s:s + bs]]
        out.append(enc(Batch.from_data_list(chunk)).numpy())
    emb = np.concatenate(out) if out else np.empty((0, 0))
    return emb, keep


def stats(name, x):
    return f"  {name:28s}: {x.mean():.3f} ± {x.std():.3f}"


def main():
    import os
    from concurrent.futures import ProcessPoolExecutor

    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", required=True)
    ap.add_argument("--glob", default="*.c")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--pairs", type=int, default=200000, help="random cross-pairs to sample")
    ap.add_argument("--out", default="o0_o3_dist.png")
    args = ap.parse_args()

    files = [str(p) for p in sorted(Path(args.sources).glob(args.glob))[: args.n]]
    print(f"[o0_o3] compiling {len(files)} programs x2 (O0,O3) on {args.workers} workers...")
    tasks = [(f, "-O0") for f in files] + [(f, "-O3") for f in files]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        built = list(ex.map(_build, tasks, chunksize=16))
    nf = len(files)
    d0, d3 = built[:nf], built[nf:]
    # keep only programs where BOTH O0 and O3 built
    both = [i for i in range(nf) if d0[i] is not None and d3[i] is not None]
    d0 = [d0[i] for i in both]; d3 = [d3[i] for i in both]
    print(f"[o0_o3] {len(both)} programs compiled at BOTH levels")

    enc = load_encoder(args.ckpt)
    o0, _ = encode_all(enc, d0)
    o3, _ = encode_all(enc, d3)
    n = o0.shape[0]
    o0n = o0 / (np.linalg.norm(o0, axis=1, keepdims=True) + 1e-9)
    o3n = o3 / (np.linalg.norm(o3, axis=1, keepdims=True) + 1e-9)

    rng = np.random.default_rng(0)

    def sample_cross(A, B, same_index_forbidden):
        i = rng.integers(0, n, args.pairs)
        j = rng.integers(0, n, args.pairs)
        if same_index_forbidden:
            m = i != j; i, j = i[m], j[m]
        return np.sum(A[i] * B[j], axis=1)

    same = np.sum(o0n * o3n, axis=1)                       # O0_i vs O3_i
    o0o0 = sample_cross(o0n, o0n, True)
    o3o3 = sample_cross(o3n, o3n, True)
    o0o3_diff = sample_cross(o0n, o3n, True)

    print(f"\n========= COSINE sur {n} programmes ({args.pairs} paires aléatoires) =========")
    print(stats("O0<->O3 MEME programme", same))
    print(stats("O0<->O0 progs differents", o0o0))
    print(stats("O3<->O3 progs differents", o3o3))
    print(stats("O0<->O3 progs differents", o0o3_diff))
    print()
    margin = same.mean() - o0o3_diff.mean()
    print(f"  marge (meme-prog - cross) = {margin:+.3f}")
    if same.mean() > max(o0o0.mean(), o3o3.mean(), o0o3_diff.mean()):
        print("  -> VERDICT: meme programme = la paire la PLUS proche => l'encodeur RELIE O0 et O3. BON signal.")
    else:
        print("  -> VERDICT: meme programme PAS la plus proche => signal faible/absent.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(f"Distribution cosinus O0/O3 — {n} programmes", fontsize=14, weight="bold")
    ax.hist(same, bins=60, alpha=0.7, color="#2ca02c", density=True, label=f"O0-O3 MEME prog ({same.mean():.2f})")
    ax.hist(o0o3_diff, bins=60, alpha=0.5, color="#d62728", density=True, label=f"progs differents ({o0o3_diff.mean():.2f})")
    ax.axvline(same.mean(), color="#2ca02c", ls="--"); ax.axvline(o0o3_diff.mean(), color="#d62728", ls="--")
    ax.set_xlabel("cosine similarity"); ax.set_ylabel("densite"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(args.out, dpi=130)
    print(f"\n[o0_o3] figure -> {args.out}")


if __name__ == "__main__":
    main()