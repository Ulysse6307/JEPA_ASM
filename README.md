<h1 align="center">JEPA-IR</h1>

<p align="center">
  <b>A self-supervised GNN encoder for programs — and latent prediction of compiler optimization.</b><br>
  <i>Learn a semantic embedding of code from a lossless LLVM-IR graph, with a JEPA objective and no decoder.</i>
</p>

<p align="center">
  <img alt="self-supervised" src="https://img.shields.io/badge/training-self--supervised-2ca02c">
  <img alt="JEPA" src="https://img.shields.io/badge/objective-JEPA%20(no%20decoder)-1f77b4">
  <img alt="anti-collapse" src="https://img.shields.io/badge/anti--collapse-VICReg-ff7f0e">
  <img alt="hardware" src="https://img.shields.io/badge/trained%20on-B200%20(IDRIS%20Dalia)-d62728">
  <img alt="python" src="https://img.shields.io/badge/python-3.10%2B-blue">
</p>

---

## TL;DR

> **Today, knowing whether a compiler optimization speeds up a program requires compiling *and* running it.**
> We predict the *effect* of optimization **in latent space, without execution** — and we demonstrate it: on the hard
> jump `-O0 → -O1`, our predictor reconstructs the optimized embedding (cosine **0.79 → 0.92**) on code it has never seen.

The pipeline:

```
C/C++  →  LLVM IR  →  lossless 3-relation graph  →  GNN encoder (JEPA)  →  embedding
                                                                              │
                                                       latent predictor ──────┘
                                                       emb(O_k)  →  emb(O_{k+1})
```

Two contributions:

1. **A self-supervised program encoder.** With *no labels*, it maps a program to a vector capturing its semantics. The input is neither an AST nor a plain CFG, but an **IR graph that keeps all three relations at once** — control flow, data flow, **and memory-effect ordering**.
2. **A latent predictor of optimization.** A small network learns to "step one optimization level forward" *in latent space*: `predictor(emb(O0)) ≈ emb(O1)`. Chained, it predicts optimization's effect **without recompiling or running** anything.

---

## Why this, and why it's not trivial

Choosing which optimizations to apply to a program (the **phase-ordering problem**) is open and expensive: to know if a pass sequence helps, you must compile **and execute** it — across thousands of variants. Compiler heuristics (`-O1/-O2/-O3`) are frozen compromises, not tailored to *your* code.

**What's missing is a representation of the program you can reason over *before* executing.** That representation is exactly what this encoder produces.

### Why JEPA — and not an autoencoder

- **No reconstruction, no decoder.** The target is a *vector*, never the graph. We avoid the cost and artifacts of code generation entirely.
- **The learning signal comes from masking.** We hide part of the graph (nodes **and** edges) and force the encoder to produce an embedding consistent with the full graph. This forces it to learn the program's *real structure*, not its surface.
- **Explicit anti-collapse (VICReg).** Comparing two encodings admits the degenerate "encoder → constant vector" solution. Variance + covariance terms forbid it — and we *prove* this by ablation (cutting VICReg collapses the space, `emb_std 1.07 → 0.0003`).

---

## The graph keeps three relations at once (lossless)

Unlike a plain CFG or AST, **none of these is discarded** — this is what separates JEPA-IR's input from existing representations (ProGraML, inst2vec):

| Relation | Meaning | Edges |
|---|---|---|
| **control flow** | the CFG between basic blocks | terminator/entry instructions |
| **data flow** | SSA def → use (a value produced by A is read by B) | per-value |
| **memory ordering** | program order among side-effecting instructions | between memory ops |

Each node is one LLVM instruction. See [`src/jepa_ir/graph/builder.py`](src/jepa_ir/graph/builder.py) and the [`ProgramGraph` schema](src/jepa_ir/graph/schema.py).

---

## The encoder (the deliverable)

A relational GNN trained **from scratch** (no pretrained model) — [`src/jepa_ir/model/encoder.py`](src/jepa_ir/model/encoder.py):

- **Per-relation message passing.** One `GraphConv` per relation per layer; messages are summed so all three relations propagate **jointly** — the graph is never reduced to a single relation. Residual + LayerNorm between layers.
- **Learned mask token.** Masked nodes keep their position and edges; only their *content* (opcode) is replaced by a single learned mask token + flag, so the encoder knows where the holes are.
- **Structural positional encoding.** A cheap per-relation degree encoding distinguishes otherwise-identical opcodes in different structural positions.
- **Pooling.** Mean ‖ max, then a linear projection to a **128-dim embedding** (reference run: `hidden=256`, `6 layers`, ~2.5 M params).

### Training (JEPA, no decoder)

1. **Mask** part of the graph (random nodes + their edges).
2. Encode the **masked** graph → `embedding_A`.
3. Encode the **full** graph → `embedding_B`.
4. Loss compares A and B **in latent space** — VICReg = `invariance (MSE A↔B) + variance + covariance`.

---

## Results

> All numbers below come from **real logs** (IDRIS *Dalia* GPU B200 runs + local eval), not estimates.
> Methodology is **leak-proof**: the corpus is split by hash into 3 disjoint pools (encoder / predictor / held-out) with a deterministic 70/15/15 train/val/test split — the encoder never saw the programs that evaluate the predictor.

### 1. The encoder learns, and does **not** collapse

<p align="center"><img src="docs/figures/fig_encoder_collapse.png" width="48%"> <img src="docs/figures/fig_vicreg_ablation.png" width="48%"></p>

- Effective rank of the latent space **grows** over training (~35–50 / 128); PC1 ≈ 5 % → an isotropic cloud, the opposite of collapse.
- **Decisive ablation:** cutting VICReg collapses the space immediately (`emb_std 1.07 → 0.0003`). Proof-by-contrast that the regularizer is what prevents collapse — it isn't decorative.

### 2. Masking is non-trivial (a discovered pitfall)

We found that masking *content only* let the encoder cheat via topology (mask 100 % of opcodes → cos **0.999** — it was ignoring content). By masking **edges too**, the encoder is forced to use content (cos drops to **0.48**) → a genuinely informative representation.

On the reference run, the masked graph recovers its own full embedding at **cos 0.995**, and retrieves its own full counterpart among 2048 candidates **58 % of the time** (chance ≈ 0.05 %).

### 3. Latent prediction of optimization works

<p align="center"><img src="docs/figures/fig_distance_order.png" width="48%"> <img src="docs/figures/fig_compositional.png" width="48%"></p>

On the **held-out test set** (encoder `77085` + chained predictor on the disjoint predictor pool):

- **Distance order is respected:** `dist(O0,O3)=0.210 > dist(O1,O3)=0.021 > dist(O2,O3)=0.003` — the more optimized a program, the closer its embedding to the optimum.
- **Compositionality:** applying the predictor 3× to `emb(O0)` genuinely moves it toward `emb(O3)` (distance 0.210 → 0.100).

**The strongest, honest result is per-transition** — the global "−50.8 %" average is *misleading* (it mixes one real success with trivial transitions where the predictor can only match the identity):

| Transition | cos(input, target) | cos(predicted, target) | verdict |
|---|---|---|---|
| **O0 → O1** | 0.795 | **0.915** | the only real jump — **predictor excels** |
| O1 → O2 | 0.982 | 0.983 | near-identity, marginal gain |
| O2 → O3 | 0.997 | 0.995 | nothing to predict; a single residual MLP slightly overshoots |

### 4. A scientific finding (not a failure)

Why is `O1 ≈ O2 ≈ O3` in latent space? We checked the IR directly: on **800 AnghaBench programs, the IR of -O2 and -O3 is identical 100 % of the time** — clang saturates at -O2. AnghaBench functions are *short and isolated*, so the inter-procedural inlining / vectorization that -O3 adds has nothing to bite on. This is a **measured result about the diminishing returns of optimization on isolated code**, not a limitation of the representation. Details: [`docs/limitation_non_bijective.md`](docs/limitation_non_bijective.md).

---

## Applications

| Use case | Contribution |
|---|---|
| **Phase-ordering** | score optimization sequences without executing them |
| **Compiler auto-tuning** | pick `-Ox` / passes adapted to *this* code |
| **Pre-filtering** | discard optimizations with no predicted gain before real testing |
| **Reusable representation** | a "universal" code embedding for downstream tasks |

Beyond compilation: binary similarity, clone/vulnerability detection, semantic code search — any task that benefits from a semantic embedding of a program.

---

## Repository layout

| Path | Role |
|---|---|
| [`src/jepa_ir/ir/`](src/jepa_ir/ir/) | C/C++ → LLVM IR via `clang -emit-llvm` |
| [`src/jepa_ir/graph/`](src/jepa_ir/graph/) | LLVM IR → 3-relation `ProgramGraph` (parsed with `llvmlite`) |
| [`src/jepa_ir/data/`](src/jepa_ir/data/) | `ProgramGraph` → PyG `Data` + masking → two JEPA views |
| [`src/jepa_ir/model/`](src/jepa_ir/model/) | GNN encoder (mask token, structural PE) + projector + predictor |
| [`src/jepa_ir/train/`](src/jepa_ir/train/) | JEPA loop + VICReg loss + collapse monitor |
| [`src/jepa_ir/eval/`](src/jepa_ir/eval/) | collapse diagnostics (effective rank, PCA, retrieval) |
| [`src/jepa_ir/config.py`](src/jepa_ir/config.py) | graph schema + all hyperparameters (single source of truth) |
| [`scripts/`](scripts/) | build graphs, train, train predictor, eval, make figures |
| [`docs/`](docs/) | results, per-step analysis, corpus limitation, pitch |
| [`tests/`](tests/) | end-to-end pipeline smoke tests |

---

## Setup

```bash
python3.13 -m venv .venv
.venv/bin/pip install -e .          # or: pip install llvmlite torch torch-geometric
```

Requires a `clang` on PATH (macOS: `xcode-select --install`) for the IR step. Set `$CLANG` to override the binary. The graph builder also works on raw `.ll` (no clang) via `build_graph_from_ir`.

## Quickstart

```bash
# inspect the 3-relation graph of one file
python scripts/inspect_graph.py examples/sum_array.c

# train the encoder on a directory of C sources (needs clang)
python scripts/train.py --sources examples/mini_corpus --glob '*.c' \
    --mask-edges --mask-ratio 0.15 --hidden-dim 256 --num-layers 6 \
    --epochs 50 --batch-size 32 --ckpt-dir checkpoints

# regenerate the result figures (from the logged numbers)
python scripts/make_all_figures.py --out-dir docs/figures

# run the tests
PYTHONPATH=src .venv/bin/python -m pytest tests/ -v
```

The trained encoder is saved to `checkpoints/encoder_final.pt`. Load it and call `JEPAModel.encode(batch)` (or `IRGraphEncoder.forward`) to get embeddings.

---

## Two-phase workflow: local build, remote train

The `code → IR → graph` half needs **clang**; the `graph → GNN → embedding` half needs a **GPU**. They rarely live on the same machine (the IDRIS *Dalia* cluster has B200 GPUs but no clang). So we split the pipeline at the graph boundary:

```
LOCAL (has clang)                          DALIA (has GPU, no clang)
  AnghaBench .c                              loads graphs.pt
    → clang → IR → ProgramGraph     ──┐       → GNN + JEPA + VICReg (GPU)
    → cache as graphs.pt            rsync      → encoder + PCA collapse PNGs
```

```bash
# LOCAL: build the graph cache once (clang)
python scripts/fetch_anghabench.py --out data/anghabench --sample 10000
python scripts/build_graphs.py --sources data/anghabench/sample --glob '*.c' --out data/graphs.pt

# ship to the cluster, then submit the clang-free GPU job
sbatch scripts/dalia_train.slurm
```

---

## Corpus

Dev uses tiny inline examples. At scale, the source corpus is **AnghaBench** (~1M standalone-compilable C functions) — it compiles to IR in isolation, exactly what the IR step needs. Compilation is embarrassingly parallel (Slurm array job when scaling up).

## Limitations & next steps (scientific honesty)

- The predictor's real win is **O0→O1**; `O1≈O2≈O3` because **clang's IR is identical at -O2/-O3 on this corpus** (measured, not assumed).
- The IR is **deterministic** — JEPA learns a *representation* (like BERT on text), not randomness; the non-trivial prediction is the *predictor* (guessing emb(O1) without compiling).
- No real **execution time** (the corpus isn't executable) → we predict the *structural* effect of optimization, not speedup in seconds.
- **Next step:** a corpus of complete, executable programs (e.g. **ExeBench**) where -O3 makes a real difference and embeddings can be tied to real runtimes.

---

<p align="center"><i>See <a href="docs/pitch.md">docs/pitch.md</a> for the full technical pitch and <a href="docs/results_mask15.md">docs/results_mask15.md</a> for the complete run report.</i></p>
