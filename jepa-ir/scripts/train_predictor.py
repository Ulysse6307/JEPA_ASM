#!/usr/bin/env python
"""Train the optimization predictor: emb(OX) -> emb(O3), MSE in latent space.

Frozen encoder. For each program we build embeddings at the input opt level(s)
(-O0/-O1/-O2) and the target -O3, then train a small MLP to map input->target.

CRUCIAL baseline: the IDENTITY (predict emb(OX) unchanged). emb(OX) and emb(O3)
are already close, so the predictor is only useful if it BEATS identity MSE.

Usage:
    python scripts/train_predictor.py \
        --sources data/anghabench/sample_100k --n 5000 \
        --ckpt ../../runs_from_dalia/job_76092_encoder.pt \
        --in-opts -O0 -O1 -O2 --epochs 200
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
import torch.nn as nn  # noqa: E402
from torch_geometric.data import Batch  # noqa: E402

from jepa_ir.config import ModelConfig  # noqa: E402
from jepa_ir.data.convert import program_graph_to_data  # noqa: E402
from jepa_ir.graph import build_graph_from_ir  # noqa: E402
from jepa_ir.ir import compile_to_ir, IRCompileError  # noqa: E402
from jepa_ir.model import IRGraphEncoder, OptPredictor  # noqa: E402


def _build(args):
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
    return enc, blob.get("cfg", ModelConfig())


@torch.no_grad()
def encode(enc, datas, bs=256):
    out = []
    for s in range(0, len(datas), bs):
        out.append(enc(Batch.from_data_list(datas[s:s + bs])).numpy())
    return np.concatenate(out)


def build_embeddings(args):
    """Either load a precomputed embedding cache (clang-free, Dalia) or compile+
    encode from sources (local). Returns (emb_dict, dim)."""
    if args.emb_cache:
        blob = torch.load(args.emb_cache, weights_only=False)
        print(f"[pred] loaded embedding cache {args.emb_cache} "
              f"(n={blob['n']}, dim={blob['dim']})")
        return blob["emb"], blob["dim"]
    # fallback: compile + encode locally (needs clang)
    in_opts = [f"-O{d.lstrip('-O')}" for d in args.in_opts]
    opts = sorted(set(in_opts + ["-O3"]))
    files = [str(p) for p in sorted(Path(args.sources).glob(args.glob))[: args.n]]
    enc, cfg = load_encoder(args.ckpt)
    print(f"[pred] compiling {len(files)} progs x {len(opts)} levels...")
    tasks = [(f, o) for f in files for o in opts]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        built = list(ex.map(_build, tasks, chunksize=16))
    by_opt = {o: [] for o in opts}
    for i in range(len(files)):
        row = built[i * len(opts):(i + 1) * len(opts)]
        if any(r is None for r in row):
            continue
        for k, o in enumerate(opts):
            by_opt[o].append(row[k])
    return {o: encode(enc, by_opt[o]) for o in opts}, cfg.embedding_dim


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--emb-cache", help="precomputed embeddings .pt (no clang; Dalia)")
    src.add_argument("--sources", help="dir of C sources (local; needs clang)")
    ap.add_argument("--glob", default="*.c")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--ckpt", help="encoder ckpt (only needed with --sources)")
    ap.add_argument("--in-opts", nargs="+", default=["0", "1", "2"])
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--patience", type=int, default=20, help="early-stop patience (evals)")
    ap.add_argument("--device", default="auto", help="auto|cuda|mps|cpu")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default="predictor.pt")
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device("cuda") if torch.cuda.is_available() else (
            torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))
    else:
        device = torch.device(args.device)
    print(f"[pred] device = {device}")

    emb, dim = build_embeddings(args)
    in_opts = [f"-O{d.lstrip('-O')}" for d in args.in_opts]
    z3 = emb["-O3"]
    X = torch.tensor(np.concatenate([emb[o] for o in in_opts]), dtype=torch.float32)
    Y = torch.tensor(np.concatenate([z3 for _ in in_opts]), dtype=torch.float32)

    n = X.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    n_val = n // 5
    vi, ti = perm[:n_val], perm[n_val:]
    Xtr, Ytr = X[ti].to(device), Y[ti].to(device)
    Xva, Yva = X[vi].to(device), Y[vi].to(device)

    id_mse = nn.functional.mse_loss(Xva, Yva).item()
    print(f"\n[pred] {n} pairs | IDENTITY baseline val MSE = {id_mse:.4f}  (must beat)")

    model = OptPredictor(dim, hidden=args.hidden, residual=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    bs = args.batch_size
    best, best_state, since = float("inf"), None, 0
    ntr = Xtr.shape[0]
    for ep in range(args.epochs):
        model.train()
        order = torch.randperm(ntr, device=device)
        for s in range(0, ntr, bs):
            idx = order[s:s + bs]
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(Xtr[idx]), Ytr[idx])
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vmse = nn.functional.mse_loss(model(Xva), Yva).item()
        if vmse < best - 1e-5:
            best, since = vmse, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
        if ep % 25 == 0 or since == 0:
            print(f"  epoch {ep:4d} | val MSE {vmse:.4f} | best {best:.4f}")
        if since >= args.patience:
            print(f"  early stop @ epoch {ep} (no improve for {args.patience} evals)")
            break

    print("\n========== RESULTAT ==========")
    print(f"  identity (copier emb(OX))  val MSE = {id_mse:.4f}")
    print(f"  predictor (meilleur)       val MSE = {best:.4f}")
    print(f"  -> gain du predictor = {(id_mse - best) / id_mse * 100:+.1f}%")
    if best < id_mse * 0.95:
        print("  -> Le predictor BAT l'identite : il a appris l'effet de l'optimisation. BON.")
    else:
        print("  -> Le predictor ne bat pas l'identite.")

    torch.save({"predictor": best_state, "dim": dim, "hidden": args.hidden,
                "id_mse": id_mse, "best_mse": best}, args.out)
    print(f"\n[pred] saved -> {args.out}")


if __name__ == "__main__":
    main()