#!/usr/bin/env python
"""Build a comparison figure: WITH VICReg vs WITHOUT (ablation).

Reads the [collapse] metric lines from two run logs and produces:
  1. metric curves over epochs (emb_std, effective rank, |corr|, PC1)
  2. final PCA scatter side by side

Usage:
    python scripts/compare_runs.py \
        --with-log  runs_from_dalia/job_75686/jepa-ir_75686.out \
        --without-log runs_from_dalia/job_75735/jepa-ablation_75735.out \
        --with-pca  runs_from_dalia/job_75686/diagnostics/pca_final.png \
        --without-pca runs_from_dalia/job_75735/diagnostics/pca_final.png \
        --out runs_from_dalia/comparison_vicreg.png
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# parse lines like:
# [collapse] n=2048 dim=128 | std=1.0689 eff_rank=50.7/128 |corr|=0.105 | PCA top5=[0.04, ...]
_RE = re.compile(
    r"std=([0-9.eE+-]+).*?eff_rank=([0-9.]+)/.*?\|corr\|=([0-9.]+).*?top5=\[([0-9.]+)"
)


def parse_metrics(log_path: str):
    stds, ranks, corrs, pc1s = [], [], [], []
    for line in Path(log_path).read_text().splitlines():
        if "[collapse]" not in line:
            continue
        m = _RE.search(line)
        if not m:
            continue
        stds.append(float(m.group(1)))
        ranks.append(float(m.group(2)))
        corrs.append(float(m.group(3)))
        pc1s.append(float(m.group(4)))
    return stds, ranks, corrs, pc1s


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--with-log", required=True)
    p.add_argument("--without-log", required=True)
    p.add_argument("--with-pca", required=True)
    p.add_argument("--without-pca", required=True)
    p.add_argument("--out", default="comparison_vicreg.png")
    args = p.parse_args()

    w = parse_metrics(args.with_log)
    wo = parse_metrics(args.without_log)
    # x axis: snapshots are every 5 epochs (0,5,10,... + final)
    xs_w = list(range(0, 5 * len(w[0]), 5))
    xs_wo = list(range(0, 5 * len(wo[0]), 5))

    fig = plt.figure(figsize=(13, 9))
    fig.suptitle("VICReg empeche l'effondrement de l'espace latent",
                 fontsize=16, weight="bold")

    # --- top: THE key curve, emb_std over epochs ---
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(xs_w, w[0], "o-", color="#2a7", label="AVEC VICReg", lw=2.5, ms=6)
    ax.plot(xs_wo, wo[0], "s--", color="#c33", label="SANS VICReg (ablation)", lw=2.5, ms=6)
    ax.axhline(1.0, color="#2a7", ls=":", alpha=0.4)
    ax.set_title("Ecart-type moyen des embeddings  (→0 = collapse total)",
                 fontsize=12)
    ax.set_xlabel("epoch")
    ax.set_ylabel("emb_std")
    ax.legend(fontsize=11, loc="center right")
    ax.grid(alpha=0.3)
    # annotate the two endpoints
    ax.annotate(f"{w[0][-1]:.2f}  (sain)", (xs_w[-1], w[0][-1]),
                textcoords="offset points", xytext=(-70, 8), color="#2a7", weight="bold")
    ax.annotate(f"{wo[0][-1]:.4f}  (effondre)", (xs_wo[-1], wo[0][-1]),
                textcoords="offset points", xytext=(-90, 12), color="#c33", weight="bold")

    # --- bottom: final PCA side by side ---
    for j, (path, label, color) in enumerate([
        (args.with_pca, "AVEC VICReg — nuage etale (sain)", "#2a7"),
        (args.without_pca, "SANS VICReg — tout au meme point", "#c33"),
    ]):
        ax = fig.add_subplot(2, 2, 3 + j)
        ax.imshow(mpimg.imread(path))
        ax.set_title(label, fontsize=12, color=color, weight="bold")
        ax.axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(args.out, dpi=120)
    print(f"[compare] saved -> {args.out}")


if __name__ == "__main__":
    main()