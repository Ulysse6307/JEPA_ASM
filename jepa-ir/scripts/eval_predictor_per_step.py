"""Per-transition evaluation of the chain predictor on the TEST set.

The global +50.8% averages 3 transitions (O0->O1, O1->O2, O2->O3). Since O1/O2/O3
are nearly co-located in latent space, the easy transitions inflate the average.
This script breaks it down: for EACH transition separately, on the held-out test
set, it reports:
  - identity MSE (copy input)         : how hard the transition is
  - predictor MSE                     : how well the predictor does
  - gain %                            : (id - pred) / id
  - cos(predicted, true target)       : 1.0 = perfect
  - cos(input, true target)           : the identity baseline in cosine

Usage:
    python scripts/eval_predictor_per_step.py \
        --graph-cache ../../data/predictor_graphs_60k.pt \
        --encoder ../../runs_from_dalia/encoder_77085.pt \
        --predictor ../../runs_from_dalia/predictor_chain_final.pt
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
from jepa_ir.model import IRGraphEncoder, OptPredictor  # noqa: E402

OPTS = ["-O0", "-O1", "-O2", "-O3"]
STEPS = [("-O0", "-O1"), ("-O1", "-O2"), ("-O2", "-O3")]


@torch.no_grad()
def encode(enc, datas, device, bs=512):
    out = []
    for s in range(0, len(datas), bs):
        out.append(enc(Batch.from_data_list(datas[s:s + bs]).to(device)).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-cache", required=True)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--predictor", required=True)
    args = ap.parse_args()
    device = torch.device("cpu")

    # encoder
    eb = torch.load(args.encoder, map_location="cpu", weights_only=False)
    enc = IRGraphEncoder(eb.get("cfg", ModelConfig())); enc.load_state_dict(eb["encoder"]); enc.eval()
    dim = eb.get("cfg", ModelConfig()).embedding_dim

    # predictor
    pb = torch.load(args.predictor, map_location="cpu", weights_only=False)
    pred = OptPredictor(dim, hidden=pb.get("hidden", 1024), residual=True)
    pred.load_state_dict(pb["predictor"]); pred.eval()

    # data: TEST subset only
    blob = torch.load(args.graph_cache, weights_only=False)
    subs = np.array(blob["subsplit"])
    mte = subs == "test"
    print(f"[eval] test programs: {mte.sum()}")
    emb = {o: encode(enc, [g for g, m in zip(blob["graphs"][o], mte) if m], device) for o in OPTS}

    def mse(a, b): return float(((a - b) ** 2).mean())
    def cosd(a, b):
        a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
        b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
        return float((a * b).sum(1).mean())

    print("\n================ VALIDATION PAR TRANSITION (test set) ================")
    print(f"{'transition':12} {'MSE id':>9} {'MSE pred':>9} {'gain':>7} {'cos(in,cible)':>14} {'cos(pred,cible)':>16}")
    with torch.no_grad():
        for a, b in STEPS:
            X = emb[a]; Y = emb[b]
            P = pred(torch.tensor(X, dtype=torch.float32)).numpy()
            id_mse, pr_mse = mse(X, Y), mse(P, Y)
            gain = (id_mse - pr_mse) / id_mse * 100 if id_mse > 0 else 0
            print(f"{a+'->'+b:12} {id_mse:9.4f} {pr_mse:9.4f} {gain:+6.1f}% "
                  f"{cosd(X,Y):14.3f} {cosd(P,Y):16.3f}")

    print("\n  Lecture :")
    print("   - MSE id grande  = transition DIFFICILE (input loin de la cible)")
    print("   - cos(pred,cible) proche de 1 = le predictor reconstruit bien la cible")
    print("   - compare cos(pred,cible) vs cos(in,cible) : le predictor doit faire MIEUX")


if __name__ == "__main__":
    main()