#!/usr/bin/env python
"""Sweep the mask ratio: how does encode(masked) ~ encode(full) degrade as we
hide more of the graph? Answers "close but not overlapping" precisely.

For each mask ratio in a sweep, on the SAME held-out programs, we report:
  * cos(masked, full) of the same program  (1.0 = overlap; <1 = distinct but close)
  * retrieval top-1 (does masked still point to its own full?)
A good encoder stays high on retrieval while cos drops below 1.0 as masking grows
— i.e. masked and full become *distinct* yet the program identity *survives*.

Usage:
    python scripts/probe_masking_sweep.py \
        --full-corpus data/anghabench/AnghaBench-XXX --exclude data/anghabench/sample \
        --n 100 --ratios 0.3 0.5 0.7 0.9 \
        --ckpt ../runs_from_dalia/job_75686/encoder_final.pt \
        --out ../runs_from_dalia/probe_masking_sweep.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch_geometric.data import Batch  # noqa: E402

from jepa_ir.config import MaskConfig, ModelConfig  # noqa: E402
from jepa_ir.data.convert import program_graph_to_data  # noqa: E402
from jepa_ir.data.masking import mask_graph  # noqa: E402
from jepa_ir.graph import build_graph_from_ir  # noqa: E402
from jepa_ir.ir import compile_to_ir, IRCompileError  # noqa: E402
from jepa_ir.model import IRGraphEncoder  # noqa: E402


def load_encoder(ckpt):
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    enc = IRGraphEncoder(blob.get("cfg", ModelConfig()))
    enc.load_state_dict(blob["encoder"]); enc.eval()
    return enc


def train_names(d: Path):
    s = set()
    for f in d.glob("*.c"):
        nm = f.name
        if "_" in nm and nm.split("_", 1)[0].isdigit():
            nm = nm.split("_", 1)[1]
        s.add(nm)
    return s


def build_one(path):
    try:
        graphs = build_graph_from_ir(compile_to_ir(path))
    except Exception:  # noqa: BLE001
        return None
    return program_graph_to_data(max(graphs, key=lambda g: g.num_nodes)) if graphs else None


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-corpus", required=True)
    ap.add_argument("--exclude", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--ratios", type=float, nargs="+", default=[0.3, 0.5, 0.7, 0.9])
    ap.add_argument("--mask-edges", action="store_true",
                    help="also cut edges of masked nodes (match edge-mask training)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", default="probe_masking_sweep.png")
    args = ap.parse_args()

    enc = load_encoder(args.ckpt)
    tn = train_names(Path(args.exclude))

    # collect held-out graphs + their FULL embeddings once
    datas, full_emb = [], []
    scanned = 0
    for f in Path(args.full_corpus).rglob("*.c"):
        if f.name in tn:
            continue
        scanned += 1
        d = build_one(f)
        if d is None:
            continue
        datas.append(d)
        full_emb.append(enc(Batch.from_data_list([d])).squeeze(0).numpy())
        if len(datas) >= args.n or scanned > args.n * 50:
            break
    full = np.stack(full_emb); n = len(datas)
    fn = full / (np.linalg.norm(full, axis=1, keepdims=True) + 1e-9)
    print(f"[sweep] {n} held-out programs")

    rows = []
    for r in args.ratios:
        cfg = MaskConfig(mask_ratio=r, block_masking=True, mask_edges=args.mask_edges)
        gen = torch.Generator().manual_seed(args.seed)
        masked = []
        for d in datas:
            v = mask_graph(d, cfg, gen)
            masked.append(enc(Batch.from_data_list([v.context])).squeeze(0).numpy())
        masked = np.stack(masked)
        intra = np.array([cos(masked[i], full[i]) for i in range(n)])
        mn = masked / (np.linalg.norm(masked, axis=1, keepdims=True) + 1e-9)
        sim = mn @ fn.T
        top1 = float((sim.argmax(1) == np.arange(n)).mean())
        overlap = int((intra > 0.999).sum())
        rows.append((r, intra.mean(), intra.std(), top1, overlap))
        print(f"  mask={r:.0%}  cos(masque,complet)={intra.mean():.3f}  "
              f"retrieval_top1={top1*100:.0f}%  overlap(cos>.999)={overlap}/{n}")

    # plot two curves vs mask ratio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rs = [x[0] * 100 for x in rows]
    coss = [x[1] for x in rows]
    tops = [x[3] * 100 for x in rows]

    fig, ax1 = plt.subplots(figsize=(9, 6))
    fig.suptitle("Robustesse au masquage : masque vs complet (held-out)",
                 fontsize=13, weight="bold")
    c1 = "#ff7f0e"
    ax1.plot(rs, coss, "o-", color=c1, lw=2.5, ms=8, label="cos(masque, complet)")
    ax1.set_xlabel("% du graphe masque")
    ax1.set_ylabel("cosine masque vs complet", color=c1)
    ax1.tick_params(axis="y", labelcolor=c1)
    ax1.set_ylim(0, 1.05)
    ax1.axhline(1.0, ls=":", color=c1, alpha=0.4)
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    c2 = "#2ca02c"
    ax2.plot(rs, tops, "s--", color=c2, lw=2.5, ms=8, label="retrieval top-1 (%)")
    ax2.set_ylabel("retrieval top-1 (%)", color=c2)
    ax2.tick_params(axis="y", labelcolor=c2)
    ax2.set_ylim(0, 105)

    for r, c, t in zip(rs, coss, tops):
        ax1.annotate(f"{c:.2f}", (r, c), textcoords="offset points", xytext=(0, 8),
                     color=c1, fontsize=9, ha="center")
        ax2.annotate(f"{t:.0f}%", (r, t), textcoords="offset points", xytext=(0, -16),
                     color=c2, fontsize=9, ha="center")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(args.out, dpi=130)
    print(f"[sweep] figure -> {args.out}")


if __name__ == "__main__":
    main()