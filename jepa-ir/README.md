# JEPA-IR

Self-supervised **GNN encoder for program representations**, trained from scratch
with a **JEPA** objective on **lossless LLVM-IR graphs**.

The deliverable is the *encoder* and the quality of the embeddings it produces —
not a classifier, not a decoder.

```
code (C/C++)  →  LLVM IR  →  lossless 3-relation graph  →  GNN encoder  →  embedding
```

## The graph keeps three relations at once (lossless)

Unlike a plain CFG or AST, none of these is discarded:

1. **control flow** — CFG edges between basic blocks
2. **data flow** — SSA def→use edges
3. **memory ordering** — program order among side-effecting instructions

See [`src/jepa_ir/graph/builder.py`](src/jepa_ir/graph/builder.py).

## Training (JEPA, no decoder)

1. **Mask** part of the graph (whole basic blocks or random nodes). Masked nodes
   keep their position and edges; only their *content* is replaced by a **learned
   mask token** + a `mask` flag.
2. Encode the **masked** graph → `embedding_A`.
3. Encode the **full** graph → `embedding_B`.
4. Loss compares A and B **in latent space** — no reconstruction, the target is a
   *vector*, never the graph.

### Anti-collapse (mandatory)

Comparing two encodings admits the degenerate solution *encoder → constant*.
We forbid it with **VICReg** (invariance + variance + covariance terms): see
[`src/jepa_ir/train/vicreg.py`](src/jepa_ir/train/vicreg.py). The variance hinge
keeps each embedding dimension's std ≥ 1; the covariance term decorrelates dims.

## Layout

| Path | Role |
|---|---|
| `src/jepa_ir/ir/` | C/C++ → LLVM IR via `clang -emit-llvm` |
| `src/jepa_ir/graph/` | LLVM IR → 3-relation `ProgramGraph` (parsed with llvmlite) |
| `src/jepa_ir/data/` | `ProgramGraph` → PyG `Data` + masking → two JEPA views |
| `src/jepa_ir/model/` | GNN encoder (mask token, structural PE) + projector |
| `src/jepa_ir/train/` | JEPA loop + VICReg loss + collapse monitor |
| `src/jepa_ir/config.py` | graph schema + all hyperparameters (single source of truth) |
| `scripts/` | `train.py`, `inspect_graph.py` |
| `tests/` | end-to-end pipeline smoke tests |

## Setup

```bash
python3.13 -m venv .venv
.venv/bin/pip install -e .          # or: pip install llvmlite torch torch-geometric
```

Requires a `clang` on PATH (macOS: `xcode-select --install`) for the IR step.
Set `$CLANG` to override the binary. The graph builder works on raw `.ll` too
(no clang needed) via `build_graph_from_ir`.

## Quickstart

```bash
# inspect the 3-relation graph of one file
python scripts/inspect_graph.py examples/sum_array.c

# train on a directory of C sources (needs clang)
python scripts/train.py --sources examples/mini_corpus --glob '*.c' \
    --epochs 50 --batch-size 32 --ckpt-dir checkpoints

# run the tests
PYTHONPATH=src .venv/bin/python -m pytest tests/ -v
```

## Two-phase workflow: local build, remote train (no clang on the GPU box)

The `code → IR → graph` half needs **clang**; the `graph → GNN → embedding` half
needs a **GPU**. These rarely live on the same machine (the IDRIS *Dalia* cluster
has B200 GPUs but no clang). So we split the pipeline at the graph boundary:

```
LOCAL (has clang)                          DALIA (has GPU, no clang)
  AnghaBench .c                              loads graphs.pt
    → clang → IR → ProgramGraph     ──┐       → GNN + JEPA + VICReg (GPU)
    → cache as graphs.pt            rsync      → encoder + PCA collapse PNGs
```

```bash
# --- LOCAL: build the graph cache once (clang) ---
python scripts/fetch_anghabench.py --out data/anghabench --sample 10000
python scripts/build_graphs.py --sources data/anghabench/sample \
    --glob '*.c' --out data/graphs.pt

# --- ship code + cache to your OWN dir on Dalia (must be on OpenGate) ---
rsync -avz --exclude .venv --exclude data src scripts pyproject.toml \
    dalia:/lustre/work/vivatech-jepacode/$USER/jepa-ir/
rsync -avz data/graphs.pt dalia:/lustre/work/vivatech-jepacode/$USER/data/graphs.pt

# --- DALIA: submit the GPU job (clang-free, trains from the cache) ---
ssh dalia 'cd /lustre/work/vivatech-jepacode/$USER/jepa-ir && sbatch scripts/dalia_train.slurm'
ssh dalia 'squeue -u $USER'

# --- pull the diagnostics back to inspect collapse ---
rsync -avz dalia:/lustre/work/vivatech-jepacode/$USER/runs/ ./runs_from_dalia/
```

## Watching for latent collapse

Training emits collapse diagnostics every few epochs (see
[`src/jepa_ir/eval/collapse.py`](src/jepa_ir/eval/collapse.py)): a **2D PCA
scatter PNG** of the embeddings plus metrics — **effective rank** (sharpest
signal; →1 means collapse), mean per-dim std, mean |correlation|, and PCA
explained-variance. A `looks_collapsed` flag fires if rank→1 or std→0. This is
how we verify the VICReg regularizer is actually keeping the latent space spread
out, not just trusting the loss curve.

Training prints a collapse monitor each log step:

```
e000 s00000 | loss 43.30 (inv 0.160 var 1.323 cov 6.24) | emb_std 0.256
e010 s00020 | loss 41.10 (inv 0.026 var 1.406 cov 5.29) | emb_std 0.632
                        ▲ invariance falls (JEPA)              ▲ std rises (no collapse)
```

The trained encoder is saved to `checkpoints/encoder_final.pt`. Load it and call
`JEPAModel.encode(batch)` (or `IRGraphEncoder.forward`) to get embeddings.

## Corpus

Dev uses tiny inline examples. For scale, the recommended source corpus is
**AnghaBench** (~1M standalone-compilable C functions) — it compiles to IR in
isolation, which is exactly what the IR step needs. Compilation is embarrassingly
parallel and is meant to run on the GPU cluster (Slurm array job) when scaling up;
local dev runs on CPU.

## Status

All pipeline stages implemented and smoke-tested (5/5 tests pass). Next steps:
ingest AnghaBench at scale, add Laplacian positional encoding, downstream probes
to evaluate embedding quality.