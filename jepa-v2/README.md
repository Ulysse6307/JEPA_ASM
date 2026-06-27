<h1 align="center">JEPA-v2 — Factored Program Embedding</h1>

<p align="center">
  <b>A self-supervised program encoder on <a href="https://github.com/ChrisCummins/ProGraML">ProgramML</a> graphs,
  whose embedding is split into two disentangled sub-spaces.</b>
</p>

---

## The idea

We learn a program embedding `z` that is **factored** into two concatenated blocks:

```
z = [  z_sem  |  z_speed  ]
```

| Block | Pulled together | Pushed apart | Captures |
|---|---|---|---|
| **`z_sem`** | the 4 `-O` levels of the **same source** | different sources | *what the code does* (invariant to optimization) |
| **`z_speed`** | the **same `-O` level** across different sources | different `-O` levels | *the optimization / "speed" profile* (invariant to the program) |

So for one source compiled at O0/O1/O2/O3: `z_sem` is ~identical across the four,
while `z_speed` separates them by level.

## Pipeline

```
ExeBench C  ──clang -O{0,1,2,3}──►  LLVM IR  ──ProgramML──►  graph (control+data+call)
   ──to_pyg──►  GNN encoder (trained from scratch)  ──►  z = [z_sem | z_speed]
```

- **Self-supervised, no manual labels.** The `-O` level is used only to *group*
  views (positives/negatives); it is never a classification target.
- **No masking** in v2 (unlike v1 / `../jepa-ir`): the learning signal is the
  factored invariance structure across `-O` levels and across programs.
- **Anti-collapse: VICReg** on each block separately, plus a **cross-covariance**
  term that forces `z_sem ⟂ z_speed` (the disentanglement).

## Why v2 (vs `../jepa-ir`)

1. **ProgramML** as the input representation — the standard GNN-on-IR graph,
   comparable to the literature, instead of the v1 hand-built graph.
2. v1 proved that on **AnghaBench** clang saturates at -O2 (O1≈O2≈O3 have an
   *identical* IR graph), so a speed sub-space is impossible there. v2 uses
   **ExeBench** (functions with real deps → inter-procedural optimization has
   material) and **gates on a probe** that O2 ≠ O3 before any training.

## Working on the shared pod

The B200 pod is **shared**. We isolate by environment (a dedicated venv under
`/workspace/jepa-v2`), and we never throttle the GPU. The RunPod SSH proxy is
quirky (no scp, no `%` in printf, commands via stdin) — use `scripts/pod.sh`:

```bash
scripts/pod.sh run 'python3 -c "import programl; print(1)"'
scripts/pod.sh put scripts/probe_exebench.py
scripts/pod.sh shell
```

## Status

Bootstrapping. ProgramML environment being pinned (see `pyproject.toml` notes);
ExeBench probe (the go/no-go gate) is the next milestone.
