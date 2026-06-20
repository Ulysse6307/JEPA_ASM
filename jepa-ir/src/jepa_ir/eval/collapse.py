"""Collapse diagnostics for the latent space.

The JEPA + VICReg objective is supposed to *forbid* collapse (encoder -> constant
/ low-rank vector). This module lets us SEE whether that holds, by looking at the
distribution of embeddings, not just the training loss.

We report, over a batch of embeddings Z [N, D]:

  * std_mean        : mean per-dimension std. Near 0 => collapse.
  * effective_rank  : exp(entropy of normalized singular values). The sharpest
                      collapse signal: D means full spread, ->1 means the cloud
                      lives on a line/point. (Roy & Vetterli, 2007.)
  * mean_abs_corr   : mean |off-diagonal correlation|. High => dimensions
                      redundant (a soft collapse VICReg's covariance term fights).
  * pca_explained   : variance ratio of the top PCA components. If PC1 explains
                      ~100%, the cloud is essentially 1D => collapse.

Plus a 2D PCA scatter saved as PNG (headless Agg backend, works on Dalia).

All heavy ops are numpy; torch tensors are detached/moved to CPU first.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np


@dataclass
class CollapseReport:
    n: int
    dim: int
    std_mean: float
    effective_rank: float
    mean_abs_corr: float
    pca_explained: list[float]   # variance ratio of top-k components
    pca_png: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        top = ", ".join(f"{r:.2f}" for r in self.pca_explained[:5])
        return (
            f"[collapse] n={self.n} dim={self.dim} | std={self.std_mean:.4f} "
            f"eff_rank={self.effective_rank:.1f}/{self.dim} "
            f"|corr|={self.mean_abs_corr:.3f} | PCA top5=[{top}]"
        )

    @property
    def looks_collapsed(self) -> bool:
        """Heuristic flag. Effective rank near 1 or near-zero std => collapse."""
        return self.effective_rank < 2.0 or self.std_mean < 1e-3


# --------------------------------------------------------------------------- #
# metrics                                                                      #
# --------------------------------------------------------------------------- #

def _effective_rank(z: np.ndarray) -> float:
    # singular values of the centered matrix
    zc = z - z.mean(axis=0, keepdims=True)
    if zc.shape[0] < 2:
        return 1.0
    sv = np.linalg.svd(zc, compute_uv=False)
    sv = sv[sv > 1e-12]
    if sv.size == 0:
        return 1.0
    p = sv / sv.sum()
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


def _mean_abs_corr(z: np.ndarray) -> float:
    if z.shape[0] < 2 or z.shape[1] < 2:
        return 0.0
    # guard against zero-variance dims
    std = z.std(axis=0)
    keep = std > 1e-8
    zz = z[:, keep]
    if zz.shape[1] < 2:
        return 0.0
    c = np.corrcoef(zz, rowvar=False)
    d = c.shape[0]
    off = c[~np.eye(d, dtype=bool)]
    return float(np.mean(np.abs(off)))


def pca_2d(z: np.ndarray, k: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Return (projected [N, k], explained_variance_ratio [min(N,D)])."""
    zc = z - z.mean(axis=0, keepdims=True)
    # SVD-based PCA (no sklearn dependency)
    u, s, _vt = np.linalg.svd(zc, full_matrices=False)
    var = (s ** 2)
    ratio = var / var.sum() if var.sum() > 0 else var
    proj = u[:, :k] * s[:k]
    return proj, ratio


def collapse_metrics(z: np.ndarray) -> CollapseReport:
    z = np.asarray(z, dtype=np.float64)
    n, d = z.shape
    std_mean = float(z.std(axis=0).mean())
    eff_rank = _effective_rank(z)
    corr = _mean_abs_corr(z)
    _, ratio = pca_2d(z, k=min(d, 2))
    return CollapseReport(
        n=n, dim=d, std_mean=std_mean, effective_rank=eff_rank,
        mean_abs_corr=corr, pca_explained=[float(r) for r in ratio[:10]],
    )


# --------------------------------------------------------------------------- #
# plotting (headless)                                                          #
# --------------------------------------------------------------------------- #

def plot_pca(z: np.ndarray, out_png: str | Path, title: str = "") -> str:
    import matplotlib
    matplotlib.use("Agg")  # headless: no display needed (works on Dalia)
    import matplotlib.pyplot as plt

    proj, ratio = pca_2d(z, k=2)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(proj[:, 0], proj[:, 1], s=8, alpha=0.5)
    pc1 = ratio[0] * 100 if len(ratio) > 0 else 0.0
    pc2 = ratio[1] * 100 if len(ratio) > 1 else 0.0
    ax.set_xlabel(f"PC1 ({pc1:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pc2:.1f}% var)")
    er = _effective_rank(np.asarray(z, dtype=np.float64))
    ax.set_title(f"{title}\neff_rank={er:.1f}/{z.shape[1]}".strip())
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    out_png = str(out_png)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return out_png


# --------------------------------------------------------------------------- #
# end-to-end: collect embeddings from a model + loader                         #
# --------------------------------------------------------------------------- #

def collect_embeddings(model, loader, device, *, max_graphs: int | None = None):  # noqa: ANN001
    """Run the encoder over (context, target) batches and return TARGET-view
    embeddings as a numpy array [N, embedding_dim]. We embed the FULL (target)
    view — that is the deliverable representation we care about not collapsing.
    """
    import torch

    model.eval()
    chunks = []
    seen = 0
    with torch.no_grad():
        for batch in loader:
            # loader may yield (context, target) or a single batch
            target = batch[1] if isinstance(batch, (tuple, list)) else batch
            target = target.to(device)
            emb = model.encode(target).detach().cpu().numpy()
            chunks.append(emb)
            seen += emb.shape[0]
            if max_graphs is not None and seen >= max_graphs:
                break
    model.train()
    if not chunks:
        return np.empty((0, 0))
    return np.concatenate(chunks, axis=0)


def mask_quality(model, loader, device, *, max_graphs: int | None = 2000):  # noqa: ANN001
    """Measure how well encode(masked) matches encode(full) — the JEPA goal —
    on the diagnostic loader (which yields (context=masked, target=full) pairs).

    Returns dict with:
      cos_intra : mean cos(masked_i, full_i)        same program  -> want HIGH
      cos_inter : mean cos(masked_i, full_j) i!=j   diff programs -> want LOWER
      retr_top1 : fraction where full_i is masked_i's nearest full neighbor
    """
    import numpy as _np
    import torch as _t

    model.eval()
    masked_chunks, full_chunks, seen = [], [], 0
    with _t.no_grad():
        for batch in loader:
            if not isinstance(batch, (tuple, list)):
                break  # need (context, target) pairs; single-batch loader can't
            ctx, tgt = batch
            masked_chunks.append(model.encode(ctx.to(device)).cpu().numpy())
            full_chunks.append(model.encode(tgt.to(device)).cpu().numpy())
            seen += masked_chunks[-1].shape[0]
            if max_graphs is not None and seen >= max_graphs:
                break
    model.train()
    if not masked_chunks:
        return None
    m = _np.concatenate(masked_chunks); f = _np.concatenate(full_chunks)
    n = m.shape[0]
    if n < 2:
        return None
    mn = m / (_np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)
    fn = f / (_np.linalg.norm(f, axis=1, keepdims=True) + 1e-9)
    sim = mn @ fn.T
    cos_intra = float(_np.diag(sim).mean())
    off = sim[~_np.eye(n, dtype=bool)]
    cos_inter = float(off.mean())
    retr_top1 = float((sim.argmax(1) == _np.arange(n)).mean())
    return {"cos_intra": cos_intra, "cos_inter": cos_inter, "retr_top1": retr_top1, "n": n}


def run_diagnostics(
    model, loader, device, *, out_dir: str | Path, tag: str = "",
    max_graphs: int | None = 2000, make_plot: bool = True,
) -> CollapseReport:
    """Collect embeddings, compute metrics, optionally save a PCA PNG.

    Returns the CollapseReport (also printed). `tag` (e.g. 'epoch010') names the
    PNG so successive snapshots are kept side by side.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    z = collect_embeddings(model, loader, device, max_graphs=max_graphs)
    if z.shape[0] < 2:
        rep = CollapseReport(n=z.shape[0], dim=z.shape[1] if z.ndim == 2 else 0,
                             std_mean=0.0, effective_rank=1.0, mean_abs_corr=0.0,
                             pca_explained=[])
        print(rep.summary(), "(too few graphs for diagnostics)")
        return rep

    rep = collapse_metrics(z)
    if make_plot:
        png = out_dir / f"pca_{tag or 'latest'}.png"
        rep.pca_png = plot_pca(z, png, title=f"latent PCA {tag}")
    print(rep.summary() + (f" -> {rep.pca_png}" if rep.pca_png else ""))
    if rep.looks_collapsed:
        print("  ⚠️  WARNING: latent space looks COLLAPSED (low rank / std).")

    # mask-quality: how well masked(P) matches full(P) — tracked per snapshot
    mq = mask_quality(model, loader, device, max_graphs=max_graphs)
    if mq is not None:
        print(f"  [mask] cos(masque,complet)={mq['cos_intra']:.3f} "
              f"vs autres={mq['cos_inter']:.3f}  "
              f"retrieval_top1={mq['retr_top1']*100:.0f}%  (n={mq['n']})")
    return rep