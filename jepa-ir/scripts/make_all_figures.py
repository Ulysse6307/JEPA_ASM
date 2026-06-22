#!/usr/bin/env python
"""Generate all presentation figures from the real numbers (logged results).

Produces, in --out-dir:
  fig_predictor_gain.png      : predictor vs identity baseline (test MSE)
  fig_distance_order.png      : dist(O0/O1/O2 -> O3), must decrease
  fig_compositional.png       : predictor^3(O0) vs O0 raw, distance to O3
  fig_encoder_collapse.png    : effective rank & emb_std over epochs (no collapse)
  fig_vicreg_ablation.png     : with vs without VICReg (emb_std)
  fig_retrieval.png           : mask retrieval over training

All numbers are hard-coded from the actual run logs (cited in the report).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

GREEN, RED, BLUE, ORANGE = "#2ca02c", "#d62728", "#1f77b4", "#ff7f0e"


def fig_predictor_gain(out):
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(["Baseline\n(identité)", "Predictor\nchaîne"], [0.3414, 0.1678],
                  color=[RED, GREEN], width=0.6, edgecolor="#333")
    ax.set_ylabel("MSE test (latent)")
    ax.set_title("Predictor O0→O1→O2→O3 : −50.8 % d'erreur\nvs recopier l'entrée",
                 fontsize=12, weight="bold")
    for b, v in zip(bars, [0.3414, 0.1678]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}",
                ha="center", fontsize=11, weight="bold")
    ax.set_ylim(0, 0.40)
    ax.annotate("", xy=(1, 0.175), xytext=(1, 0.34),
                arrowprops=dict(arrowstyle="->", color="#333", lw=2))
    ax.text(1.15, 0.255, "−50.8 %", color=GREEN, fontsize=12, weight="bold")
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


def fig_distance_order(out):
    levels = ["O0", "O1", "O2"]
    d = [0.2102, 0.0208, 0.0034]
    err = [0.2288, 0.0888, 0.0368]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(levels, d, yerr=err, capsize=6, color=[RED, ORANGE, GREEN],
           edgecolor="#333", width=0.6)
    ax.set_ylabel("distance à O3   (1 − cosinus)")
    ax.set_title("Plus le code est optimisé,\nplus il est proche de O3 (held-out)",
                 fontsize=12, weight="bold")
    for i, v in enumerate(d):
        ax.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=11, weight="bold")
    ax.set_ylim(0, 0.30)
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


def fig_compositional(out):
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(["O0 brut", "predictor³(O0)"], [0.2102, 0.0998],
                  color=[RED, GREEN], width=0.6, edgecolor="#333")
    ax.axhline(0.0, color="#999", ls=":")
    ax.set_ylabel("distance à O3   (1 − cosinus)")
    ax.set_title("Chaîne compositionnelle :\nappliquer le predictor 3× rapproche de O3",
                 fontsize=12, weight="bold")
    for b, v in zip(bars, [0.2102, 0.0998]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.006, f"{v:.3f}",
                ha="center", fontsize=11, weight="bold")
    ax.set_ylim(0, 0.27)
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


def fig_encoder_collapse(out):
    # run FINAL 77085 (encodeur propre, pool encoder, 50 epochs)
    ep = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    rank = [33.3, 25.9, 27.5, 28.1, 28.7, 30.2, 30.1, 30.3, 30.5, 29.3, 29.4]
    std = [1.12, 1.35, 1.35, 1.35, 1.37, 1.32, 1.32, 1.36, 1.33, 1.38, 1.29]
    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(ep, rank, "o-", color=BLUE, lw=2.5, label="rang effectif")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("rang effectif / 128", color=BLUE)
    ax1.tick_params(axis="y", labelcolor=BLUE); ax1.set_ylim(0, 64)
    ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(ep, std, "s--", color=GREEN, lw=2.5, label="emb_std")
    ax2.set_ylabel("écart-type des embeddings", color=GREEN)
    ax2.tick_params(axis="y", labelcolor=GREEN); ax2.set_ylim(0, 1.6)
    fig.suptitle("Espace latent stable : rang effectif ~30/128, std ~1.3\n(aucun collapse — std loin de 0)",
                 fontsize=12, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig(out, dpi=140); plt.close(fig)


def fig_vicreg_ablation(out):
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(["AVEC VICReg", "SANS VICReg\n(ablation)"], [1.07, 0.0003],
                  color=[GREEN, RED], width=0.6, edgecolor="#333")
    ax.set_ylabel("écart-type des embeddings (emb_std)")
    ax.set_title("VICReg empêche l'effondrement\n(sans lui : std → 0)",
                 fontsize=12, weight="bold")
    ax.text(0, 1.09, "1.07  (sain)", ha="center", fontsize=11, weight="bold", color=GREEN)
    ax.text(1, 0.05, "0.0003\n(effondré)", ha="center", fontsize=11, weight="bold", color=RED)
    ax.set_ylim(0, 1.25)
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="../../runs_from_dalia/figures")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    fig_predictor_gain(out / "fig_predictor_gain.png")
    fig_distance_order(out / "fig_distance_order.png")
    fig_compositional(out / "fig_compositional.png")
    fig_encoder_collapse(out / "fig_encoder_collapse.png")
    fig_vicreg_ablation(out / "fig_vicreg_ablation.png")
    print(f"[figures] saved 5 figures -> {out}")


if __name__ == "__main__":
    main()