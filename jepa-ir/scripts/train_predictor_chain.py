"""Train the CHAIN predictor: emb(O_k) -> emb(O_{k+1}), one optimization step.

The predictor learns "advance one optimization level":
    predictor(emb(O0)) ~ emb(O1)
    predictor(emb(O1)) ~ emb(O2)
    predictor(emb(O2)) ~ emb(O3)
so applying it repeatedly walks O0 -> O1 -> O2 -> O3 in latent space.

Anti-leakage: works on the PREDICTOR-pool graph cache (disjoint from the encoder
pool), with a deterministic per-program train/val/test split baked into the cache.
The frozen encoder turns graphs into embeddings; only the small MLP is trained.

Final evaluation (printed):
  * val/test MSE of the one-step predictor vs the IDENTITY baseline (copy input)
  * distance ordering check: mean dist(O0,O3) > dist(O1,O3) > dist(O2,O3)
    (more optimized = closer to O3 — must hold)
  * compositional check: predictor applied 3x to emb(O0) vs emb(O3)

Usage:
    python scripts/train_predictor_chain.py \
        --graph-cache ../../data/predictor_graphs_60k.pt \
        --encoder ../../runs_from_dalia/<encoder>.pt \
        --epochs 3000 --out ../../runs_from_dalia/predictor_chain.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch_geometric.data import Batch  # noqa: E402

from jepa_ir.config import ModelConfig  # noqa: E402
from jepa_ir.model import IRGraphEncoder, OptPredictor  # noqa: E402

OPTS = ["-O0", "-O1", "-O2", "-O3"]


def load_encoder(ckpt):
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    enc = IRGraphEncoder(blob.get("cfg", ModelConfig()))
    enc.load_state_dict(blob["encoder"]); enc.eval()
    return enc, blob.get("cfg", ModelConfig()).embedding_dim


@torch.no_grad()
def encode(enc, datas, device, bs=512):
    out = []
    for s in range(0, len(datas), bs):
        b = Batch.from_data_list(datas[s:s + bs]).to(device)
        out.append(enc(b).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-cache", required=True, help="predictor-pool graphs (4 opts)")
    ap.add_argument("--encoder", required=True, help="frozen encoder ckpt")
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="predictor_chain.pt")
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device("cuda") if torch.cuda.is_available() else (
            torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))
    else:
        device = torch.device(args.device)
    print(f"[chain] device = {device}")

    blob = torch.load(args.graph_cache, weights_only=False)
    graphs, subs = blob["graphs"], blob["subsplit"]
    n = blob["n"]
    print(f"[chain] {n} programs, 4 opt levels, splits={dict(__import__('collections').Counter(subs))}")

    enc, dim = load_encoder(args.encoder)
    enc = enc.to(device)
    # embed every program at every level (frozen encoder)
    emb = {o: encode(enc, graphs[o], device) for o in OPTS}   # each [n, dim]
    subs = np.array(subs)

    # CHAIN pairs: (O0->O1), (O1->O2), (O2->O3) per program
    steps = [("-O0", "-O1"), ("-O1", "-O2"), ("-O2", "-O3")]

    def make(split):
        m = subs == split
        X = np.concatenate([emb[a][m] for a, _ in steps])
        Y = np.concatenate([emb[b][m] for _, b in steps])
        return (torch.tensor(X, dtype=torch.float32).to(device),
                torch.tensor(Y, dtype=torch.float32).to(device))

    Xtr, Ytr = make("train")
    Xva, Yva = make("val")
    Xte, Yte = make("test")
    print(f"[chain] pairs: train {Xtr.shape[0]}, val {Xva.shape[0]}, test {Xte.shape[0]}")

    id_val = nn.functional.mse_loss(Xva, Yva).item()
    print(f"[chain] IDENTITY val MSE = {id_val:.4f}  (predictor must beat)")

    model = OptPredictor(dim, hidden=args.hidden, residual=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    bs, best, best_state, since = args.batch_size, float("inf"), None, 0
    ntr = Xtr.shape[0]
    for ep in range(args.epochs):
        model.train()
        order = torch.randperm(ntr, device=device)
        for s in range(0, ntr, bs):
            idx = order[s:s + bs]
            opt.zero_grad()
            nn.functional.mse_loss(model(Xtr[idx]), Ytr[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vmse = nn.functional.mse_loss(model(Xva), Yva).item()
        if vmse < best - 1e-5:
            best, since = vmse, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
        if ep % 50 == 0 or since == 0:
            print(f"  ep {ep:4d} | val MSE {vmse:.4f} | best {best:.4f}")
        if since >= args.patience:
            print(f"  early stop @ {ep}")
            break

    model.load_state_dict(best_state); model.eval()

    # ---- FINAL EVALUATION on TEST (never seen) ----
    print("\n==================== EVAL (test set) ====================")
    with torch.no_grad():
        te_mse = nn.functional.mse_loss(model(Xte), Yte).item()
    id_te = nn.functional.mse_loss(Xte, Yte).item()
    print(f"  one-step predictor test MSE = {te_mse:.4f}")
    print(f"  identity baseline    test MSE = {id_te:.4f}")
    print(f"  -> gain = {(id_te - te_mse) / id_te * 100:+.1f}%")

    # distance ordering (cosine dist = 1 - cos) on TEST programs, to O3
    mte = subs == "test"
    def cos(a, b):
        a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
        b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
        return np.sum(a * b, axis=1)
    z3 = emb["-O3"][mte]
    print("\n  --- distance a O3 (1 - cos), doit DECROITRE O0>O1>O2 ---")
    for o in ["-O0", "-O1", "-O2"]:
        d = 1 - cos(emb[o][mte], z3)
        print(f"    dist({o}, O3) = {d.mean():.4f} ± {d.std():.4f}")

    # compositional: predictor^3 on O0 vs O3
    with torch.no_grad():
        z = torch.tensor(emb["-O0"][mte], dtype=torch.float32).to(device)
        for _ in range(3):
            z = model(z)
        z = z.cpu().numpy()
    d_chain = 1 - cos(z, z3)
    d_raw = 1 - cos(emb["-O0"][mte], z3)
    print("\n  --- test compositionnel : predictor^3(O0) vs O3 ---")
    print(f"    dist(O0 brut, O3)        = {d_raw.mean():.4f}")
    print(f"    dist(predictor^3(O0), O3) = {d_chain.mean():.4f}")
    print(f"    -> {'RAPPROCHE de O3 (chaine marche)' if d_chain.mean() < d_raw.mean() else 'PAS plus proche'}")

    torch.save({"predictor": best_state, "dim": dim, "hidden": args.hidden,
                "test_mse": te_mse, "id_test_mse": id_te}, args.out)
    print(f"\n[chain] saved -> {args.out}")


if __name__ == "__main__":
    main()