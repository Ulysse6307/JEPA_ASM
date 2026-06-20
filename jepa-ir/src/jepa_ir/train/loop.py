"""JEPA training loop with VICReg.

Each step:
  1. draw a batch of (masked context, full target) view pairs
  2. encode both with the shared encoder, project
  3. VICReg loss (invariance + variance + covariance)
  4. step

We log a collapse monitor: the mean per-dimension std of the embeddings. If
anti-collapse works it stays well above 0; if it trended to 0 the model would be
collapsing (it should not, thanks to VICReg).
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..config import Config
from ..model import JEPAModel
from .vicreg import vicreg_loss


def build_device(spec: str = "auto") -> torch.device:
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(spec)


@torch.no_grad()
def _embedding_std(model: JEPAModel, batch) -> float:  # noqa: ANN001
    emb = model.encode(batch)
    return float(emb.std(dim=0).mean())


def train(
    dataset,
    cfg: Config,
    *,
    collate_fn,
    device: torch.device | None = None,
    on_log=None,
    diag_every: int = 0,
    diag_dir: str | None = None,
) -> JEPAModel:
    """Train a JEPAModel on a dataset of MaskedViews. Returns the trained model.

    `collate_fn` must turn a list[MaskedView] into (context_batch, target_batch).
    `on_log` optional callback(step, metrics_dict).
    `diag_every` > 0 runs the collapse diagnostics (PCA PNG + metrics) every
    `diag_every` epochs (and at the end) into `diag_dir` (defaults to
    <ckpt_dir>/diagnostics). This is how we watch the latent space for collapse.
    """
    device = device or build_device(cfg.train.device)
    torch.manual_seed(cfg.train.seed)

    model = JEPAModel(cfg.model).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        collate_fn=collate_fn,
        drop_last=True,        # VICReg needs batch >= 2; drop ragged tail
    )

    ckpt_dir = Path(cfg.train.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    diag_path = Path(diag_dir) if diag_dir else ckpt_dir / "diagnostics"

    # a fixed loader for diagnostics (no shuffle) so PCA snapshots are comparable
    def _run_diag(tag: str):
        if diag_every <= 0:
            return
        from ..eval import run_diagnostics

        diag_loader = DataLoader(
            dataset, batch_size=cfg.train.batch_size, shuffle=False,
            num_workers=0, collate_fn=collate_fn,
        )
        run_diagnostics(model, diag_loader, device, out_dir=diag_path, tag=tag)

    step = 0
    model.train()
    for epoch in range(cfg.train.epochs):
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)
        for context, target in loader:
            context = context.to(device)
            target = target.to(device)

            z_a, z_b = model(context, target)
            out = vicreg_loss(z_a, z_b, cfg.vicreg)

            opt.zero_grad(set_to_none=True)
            out.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            if step % cfg.train.log_every == 0:
                metrics = out.item_dict()
                metrics["epoch"] = epoch
                metrics["emb_std"] = _embedding_std(model, target)
                model.train()
                msg = (
                    f"e{epoch:03d} s{step:05d} | loss {metrics['total']:.3f} "
                    f"(inv {metrics['inv']:.3f} var {metrics['var']:.3f} "
                    f"cov {metrics['cov']:.4f}) | emb_std {metrics['emb_std']:.4f}"
                )
                print(msg)
                if on_log:
                    on_log(step, metrics)
            step += 1

        # checkpoint per epoch (encoder is the deliverable; save it standalone too)
        torch.save(
            {"model": model.state_dict(), "encoder": model.encoder.state_dict(),
             "cfg": cfg.model, "epoch": epoch},
            ckpt_dir / f"jepa_epoch{epoch:03d}.pt",
        )

        # periodic collapse diagnostics (PCA + metrics)
        if diag_every > 0 and (epoch % diag_every == 0):
            _run_diag(f"epoch{epoch:03d}")

    # final encoder-only checkpoint + final diagnostics
    torch.save({"encoder": model.encoder.state_dict(), "cfg": cfg.model},
               ckpt_dir / "encoder_final.pt")
    _run_diag("final")
    return model