#!/usr/bin/env python
"""Train the JEPA-IR encoder on a directory of C/C++ sources.

Usage:
    python scripts/train.py --sources examples/mini_corpus --epochs 50 \
        --batch-size 32 --ckpt-dir checkpoints

The trained encoder (the deliverable) is saved to <ckpt-dir>/encoder_final.pt.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# allow running from repo root without installing
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch  # noqa: E402

from jepa_ir.config import Config  # noqa: E402
from jepa_ir.data import IRGraphDataset, collate_views  # noqa: E402
from jepa_ir.train import build_device, train  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Train JEPA-IR encoder")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--sources", help="dir of C/C++ sources (needs clang)")
    src.add_argument("--graph-cache", help="pre-built graph .pt cache (no clang)")
    p.add_argument("--glob", default="**/*.c", help="glob for source files")
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=None,
                   help="DataLoader workers (parallel batch prep; ~8 keeps GPU fed)")
    p.add_argument("--lr", type=float, default=1e-3)
    # model size (defaults = config: hidden=128, layers=4)
    p.add_argument("--hidden-dim", type=int, default=None, help="GNN layer width")
    p.add_argument("--num-layers", type=int, default=None, help="message-passing rounds")
    p.add_argument("--embedding-dim", type=int, default=None, help="final embedding size")
    p.add_argument("--mask-ratio", type=float, default=0.30)
    p.add_argument("--no-block-masking", action="store_true")
    p.add_argument("--mask-edges", action="store_true",
                   help="also drop edges incident to masked nodes (real structural "
                        "hole; forces the encoder to use opcodes, not just topology)")
    # VICReg coefficients (defaults = eb_jepa 1/1/1). Set --std-coeff 0 --cov-coeff 0
    # for the no-VICReg ablation (only invariance) → should COLLAPSE.
    p.add_argument("--sim-coeff", type=float, default=None, help="invariance weight")
    p.add_argument("--std-coeff", type=float, default=None, help="variance weight (0 disables)")
    p.add_argument("--cov-coeff", type=float, default=None, help="covariance weight (0 disables)")
    p.add_argument("--ckpt-dir", default="checkpoints")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--diag-every", type=int, default=5,
                   help="run collapse diagnostics (PCA PNG + metrics) every N "
                        "epochs; 0 to disable")
    args = p.parse_args()

    cfg = Config()
    if args.hidden_dim is not None:
        cfg.model.hidden_dim = args.hidden_dim
    if args.num_layers is not None:
        cfg.model.num_layers = args.num_layers
    if args.embedding_dim is not None:
        cfg.model.embedding_dim = args.embedding_dim
    cfg.train.epochs = args.epochs
    cfg.train.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.train.num_workers = args.num_workers
    cfg.train.lr = args.lr
    cfg.train.ckpt_dir = args.ckpt_dir
    cfg.train.device = args.device
    cfg.train.seed = args.seed
    cfg.mask.mask_ratio = args.mask_ratio
    cfg.mask.block_masking = not args.no_block_masking
    cfg.mask.mask_edges = args.mask_edges
    # override VICReg coeffs if given (None = keep config default)
    if args.sim_coeff is not None:
        cfg.vicreg.sim_coeff = args.sim_coeff
    if args.std_coeff is not None:
        cfg.vicreg.std_coeff = args.std_coeff
    if args.cov_coeff is not None:
        cfg.vicreg.cov_coeff = args.cov_coeff
    print(f"[train] VICReg coeffs: sim={cfg.vicreg.sim_coeff} "
          f"std={cfg.vicreg.std_coeff} cov={cfg.vicreg.cov_coeff}")

    if args.graph_cache:
        ds = IRGraphDataset.from_cache(args.graph_cache, cfg.mask, seed=args.seed)
    else:
        ds = IRGraphDataset.from_sources(
            args.sources, cfg.mask, glob=args.glob,
            max_files=args.max_files, seed=args.seed,
        )
    device = build_device(args.device)
    print(f"[train] {len(ds)} graphs | device={device} | "
          f"batch={args.batch_size} epochs={args.epochs} diag_every={args.diag_every}")
    train(ds, cfg, collate_fn=collate_views, device=device,
          diag_every=args.diag_every)
    print(f"[train] done. encoder -> {Path(args.ckpt_dir)/'encoder_final.pt'}")
    print(f"[train] collapse diagnostics (PCA PNGs + metrics) -> "
          f"{Path(args.ckpt_dir)/'diagnostics'}")


if __name__ == "__main__":
    main()