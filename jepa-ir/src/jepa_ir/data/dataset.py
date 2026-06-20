"""Dataset of program graphs + a collator that batches the two JEPA views.

The dataset holds a list of IRData (one per function). Each __getitem__ draws a
FRESH random mask (so the model sees different holes every epoch) and returns a
MaskedView. The collator batches the context views and target views separately
into two PyG Batches that are node-aligned (same node count, same order), which
is what the JEPA loss needs to compare A and B node-for-node or graph-for-graph.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch

from ..config import MaskConfig
from ..graph import build_graph_from_ir
from ..ir import compile_to_ir, IRCompileError
from .convert import IRData, program_graph_to_data
from .masking import MaskedView, mask_graph


def _compile_and_build(path: str):
    """Worker: one C/C++ file -> list[IRData] (or None on failure).

    Module-level so ProcessPoolExecutor can pickle it. Each call shells out to a
    fresh clang, so calls are fully independent and parallelize cleanly.
    """
    try:
        ir = compile_to_ir(path)
        return [program_graph_to_data(g) for g in build_graph_from_ir(ir)]
    except Exception:  # noqa: BLE001 - skip bad units (non-compiling tail)
        return None


class IRGraphDataset(Dataset):
    """Holds pre-built IRData graphs and yields masked views.

    Construct from already-built graphs (`from_data_list`) or from a directory of
    C/C++ sources (`from_sources`).
    """

    def __init__(self, data_list: list[IRData], mask_cfg: MaskConfig, seed: int = 0):
        self.data_list = data_list
        self.mask_cfg = mask_cfg
        self._base_seed = seed
        self._epoch = 0

    # -- constructors ------------------------------------------------------ #
    @classmethod
    def from_data_list(cls, data_list, mask_cfg, seed=0):  # noqa: ANN001
        return cls(list(data_list), mask_cfg, seed)

    @classmethod
    def from_cache(cls, cache_path, mask_cfg, seed=0):  # noqa: ANN001
        """Load pre-built graphs from a .pt cache produced by `save_cache`.

        This is how training runs WITHOUT clang (e.g. on Dalia): graphs are
        compiled+built once locally, serialized, and only the graphs travel to
        the GPU machine. No IR toolchain needed at train time.
        """
        import torch

        from .convert import IRData  # noqa: F401  (needed for unpickling)

        blob = torch.load(cache_path, weights_only=False)
        data_list = blob["data_list"] if isinstance(blob, dict) else blob
        print(f"[dataset] loaded {len(data_list)} pre-built graphs from {cache_path}")
        return cls(list(data_list), mask_cfg, seed)

    def save_cache(self, cache_path) -> None:  # noqa: ANN001
        """Serialize the built graphs to a .pt file for later clang-free loading."""
        import torch

        torch.save({"data_list": self.data_list}, cache_path)
        print(f"[dataset] saved {len(self.data_list)} graphs -> {cache_path}")

    @classmethod
    def from_sources(
        cls,
        source_dir: str | Path,
        mask_cfg: MaskConfig,
        *,
        glob: str = "**/*.c",
        seed: int = 0,
        max_files: int | None = None,
        verbose: bool = True,
        workers: int | None = None,
    ) -> "IRGraphDataset":
        """Compile every matching source file and build graphs from it.

        Files that fail to compile or build are skipped (counted, not fatal) —
        real corpora always have a non-compiling tail.

        `workers` parallelizes compilation across processes (each clang call is
        independent). Defaults to os.cpu_count()-1. Set workers=1 for serial.
        Progress is reported every ~500 files when verbose.
        """
        import os

        source_dir = Path(source_dir)
        files = sorted(source_dir.glob(glob))
        if max_files is not None:
            files = files[:max_files]
        if not files:
            raise RuntimeError(f"no files match {glob} in {source_dir}")

        if workers is None:
            workers = max(1, (os.cpu_count() or 2) - 1)

        data_list: list[IRData] = []
        n_ok = n_fail = 0

        if workers <= 1:
            for i, f in enumerate(files):
                graphs = _compile_and_build(str(f))
                if graphs is None:
                    n_fail += 1
                else:
                    data_list.extend(graphs)
                    n_ok += 1
                if verbose and (i + 1) % 500 == 0:
                    print(f"[dataset] {i+1}/{len(files)} files "
                          f"({n_ok} ok, {n_fail} fail, {len(data_list)} graphs)")
        else:
            from concurrent.futures import ProcessPoolExecutor

            with ProcessPoolExecutor(max_workers=workers) as ex:
                done = 0
                for graphs in ex.map(_compile_and_build,
                                     [str(f) for f in files], chunksize=16):
                    done += 1
                    if graphs is None:
                        n_fail += 1
                    else:
                        data_list.extend(graphs)
                        n_ok += 1
                    if verbose and done % 500 == 0:
                        print(f"[dataset] {done}/{len(files)} files "
                              f"({n_ok} ok, {n_fail} fail, {len(data_list)} graphs)")

        if verbose:
            print(f"[dataset] built {len(data_list)} graphs from {n_ok} files "
                  f"({n_fail} failed) in {source_dir} [workers={workers}]")
        if not data_list:
            raise RuntimeError(f"no graphs built from {source_dir}")
        return cls(data_list, mask_cfg, seed)

    # -- epoch-aware masking ---------------------------------------------- #
    def set_epoch(self, epoch: int) -> None:
        """Vary the mask RNG per epoch so holes differ across epochs."""
        self._epoch = epoch

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> MaskedView:
        # deterministic but epoch- and index-dependent seed
        seed = self._base_seed + self._epoch * 1_000_003 + idx
        gen = torch.Generator().manual_seed(seed)
        return mask_graph(self.data_list[idx], self.mask_cfg, gen)


def collate_views(views: list[MaskedView]) -> tuple[Batch, Batch]:
    """Collate a list of MaskedView into (context_batch, target_batch).

    Both batches preserve node alignment: graph k occupies the same node slots in
    both, so a graph-level pooled embedding from each batch corresponds 1:1.
    """
    context = Batch.from_data_list([v.context for v in views])
    target = Batch.from_data_list([v.target for v in views])
    return context, target