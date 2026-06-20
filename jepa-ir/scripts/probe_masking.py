#!/usr/bin/env python
"""JEPA consistency probe: does encode(MASKED graph) ~ encode(FULL graph)?

This is THE test that validates the training objective directly. JEPA's loss pulls
the embedding of a masked view toward the embedding of the full view. We check, on
HELD-OUT programs (not seen during training), whether the LEARNED encoder
(encoder_final.pt) actually achieves this — while keeping different programs apart.

Metrics (same "close but not overlapping" logic):
  * cos(masked(P), full(P))          same program  -> should be HIGH
  * cos(masked(Pi), full(Pj)) i!=j   different prog -> should be LOWER
  * retrieval top-1: for each masked(P), is full(P) its nearest full neighbor?
  If masked(P) retrieves its OWN full over all others, the representation survives
  the hole = the encoder learned the program structure, not the surface.

Usage:
    python scripts/probe_masking.py \
        --full-corpus data/anghabench/AnghaBench-XXX \
        --exclude data/anghabench/sample \
        --n 100 \
        --ckpt ../runs_from_dalia/job_75686/encoder_final.pt \
        --out ../runs_from_dalia/probe_masking.png
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


def load_encoder(ckpt_path: str) -> IRGraphEncoder:
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = blob.get("cfg", ModelConfig())
    enc = IRGraphEncoder(cfg)
    enc.load_state_dict(blob["encoder"])
    enc.eval()
    return enc


def _train_names(exclude_dir: Path) -> set[str]:
    """Original file names present in the train sample (prefixed NNNNNNN_<name>)."""
    names = set()
    for f in exclude_dir.glob("*.c"):
        stem = f.name
        # strip leading "NNNNNNN_" index prefix the sampler added
        if "_" in stem and stem.split("_", 1)[0].isdigit():
            stem = stem.split("_", 1)[1]
        names.add(stem)
    return names


def build_one(path: Path):
    """Compile (default O1, as training) and build the largest function's graph."""
    try:
        ir = compile_to_ir(path)
        graphs = build_graph_from_ir(ir)
    except (IRCompileError, Exception):  # noqa: BLE001
        return None
    if not graphs:
        return None
    return program_graph_to_data(max(graphs, key=lambda g: g.num_nodes))


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-corpus", required=True, help="dir with all .c (held-out source)")
    ap.add_argument("--exclude", required=True, help="train sample dir to exclude")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--ckpt", required=True, help="encoder_final.pt (TRAINED weights)")
    ap.add_argument("--mask-ratio", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", default="probe_masking.png")
    args = ap.parse_args()

    print(f"[probe] loading TRAINED encoder: {args.ckpt}")
    enc = load_encoder(args.ckpt)
    mask_cfg = MaskConfig(mask_ratio=args.mask_ratio, block_masking=True)

    print("[probe] indexing train names to exclude (held-out split)...")
    train_names = _train_names(Path(args.exclude))

    # walk the full corpus, keep files whose name is NOT in train
    full_emb, masked_emb, names = [], [], []
    gen = torch.Generator().manual_seed(args.seed)
    scanned = 0
    for f in Path(args.full_corpus).rglob("*.c"):
        if f.name in train_names:
            continue                       # skip training programs -> held-out
        scanned += 1
        data = build_one(f)
        if data is None:
            continue
        # full view embedding
        full_e = enc(Batch.from_data_list([data])).squeeze(0).numpy()
        # masked view embedding (same masking as training)
        view = mask_graph(data, mask_cfg, gen)
        masked_e = enc(Batch.from_data_list([view.context])).squeeze(0).numpy()
        full_emb.append(full_e); masked_emb.append(masked_e); names.append(f.name)
        if len(names) >= args.n:
            break
        if scanned > args.n * 50:          # safety bound
            break

    full = np.stack(full_emb); masked = np.stack(masked_emb)
    n = len(names)
    print(f"[probe] {n} HELD-OUT programs embedded (full + masked@{args.mask_ratio})")

    def cos(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    intra = np.array([cos(masked[i], full[i]) for i in range(n)])
    inter = np.array([cos(masked[i], full[j]) for i in range(n) for j in range(n) if i != j])

    fn = full / (np.linalg.norm(full, axis=1, keepdims=True) + 1e-9)
    mn = masked / (np.linalg.norm(masked, axis=1, keepdims=True) + 1e-9)
    sim = mn @ fn.T                        # masked (rows) vs full (cols)
    nn1 = sim.argmax(axis=1)
    top1 = float((nn1 == np.arange(n)).mean())
    top5 = float(np.mean([i in sim[i].argsort()[-5:] for i in range(n)]))

    print("\n================ RESULTATS (held-out) ================")
    print(f"  cos(masque, complet) MEME prog       : {intra.mean():.3f} ± {intra.std():.3f}  (haut = bon)")
    print(f"  cos(masque, complet) progs DIFFERENTS: {inter.mean():.3f} ± {inter.std():.3f}  (bas = bon)")
    print(f"  -> marge (intra - inter)             : {intra.mean() - inter.mean():+.3f}")
    print(f"  retrieval top-1 (complet du meme prog le + proche): {top1*100:.1f}%  (hasard={100/n:.1f}%)")
    print(f"  retrieval top-5                                   : {top5*100:.1f}%")
    overlap = int((intra > 0.999).sum())
    print(f"  paires quasi-identiques (cos>0.999)  : {overlap}/{n}  (0 = proches SANS chevaucher)")
    print("======================================================\n")

    # PCA: masked/full pairs joined by a line
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    allv = np.vstack([full, masked]); allc = allv - allv.mean(0, keepdims=True)
    u, s, _ = np.linalg.svd(allc, full_matrices=False)
    proj = u[:, :2] * s[:2]; pf, pm = proj[:n], proj[n:]

    fig, ax = plt.subplots(figsize=(9, 8))
    for i in range(n):
        ax.plot([pf[i, 0], pm[i, 0]], [pf[i, 1], pm[i, 1]], color="#ccc", lw=0.6, zorder=1)
    ax.scatter(pf[:, 0], pf[:, 1], s=30, c="#2ca02c", label="complet", zorder=2)
    ax.scatter(pm[:, 0], pm[:, 1], s=30, c="#ff7f0e", label="masque (30%)", zorder=2)
    ax.set_title(
        f"JEPA: encode(masque) vs encode(complet) — {n} progs held-out\n"
        f"cos meme-prog={intra.mean():.2f}  vs autres={inter.mean():.2f}  "
        f"retrieval top-1={top1*100:.0f}%", fontsize=11)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(args.out, dpi=130)
    print(f"[probe] figure -> {args.out}")
    print("  (traits gris courts = masque proche de son complet)")


if __name__ == "__main__":
    main()