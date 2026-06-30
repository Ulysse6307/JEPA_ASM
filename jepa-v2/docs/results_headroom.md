# Headroom proof — where is the performance actually left on the table?

> Measured 2026-06-30 with `scripts/probe_headroom.py` on Apple clang 17 / M3 Pro
> (modern, strong auto-vectorizer). 6 compute-bound whole-program kernels, each run
> 6× per `-O` level, MIN wall-clock reported. Reproduce: `python3 scripts/probe_headroom.py`.

## TL;DR

- **`-O3` over `-O2` is flat** (median **1.01×**, max 1.04×, **0/6** kernels beat 5%).
  There is essentially **no wall-clock headroom between O2 and O3** — confirming the
  Step-1 gate (O2≈O3 graphs) at the *runtime* level too.
- **The real headroom is the aggressive, semantics-relaxing rewrites the compiler
  conservatively declines.** `-Ofast` (fast-math) reached **4.42×** on an FP-reduction
  kernel; geomean across kernels **1.31×**. The compiler *can* do it — it just won't,
  without permission, because it must guarantee strict IEEE semantics.

## Results (ms, min of 6 runs)

| kernel | -O0 | -O1 | -O2 | -O3 | -Ofast | O3/O2 | Ofast/O2 |
|---|---|---|---|---|---|---|---|
| matmul     | 175.1 | 49.1  | 13.2  | 12.8  | 12.9  | 1.03 | 1.02 |
| saxpy_dot  | 497.8 | 300.1 | 246.4 | 246.6 | **55.7** | 1.00 | **4.42** |
| mandelbrot | 241.9 | 99.0  | 99.7  | 95.8  | 104.7 | 1.04 | 0.95 |
| nbody      | 289.1 | 65.9  | 66.0  | 65.8  | 57.3  | 1.00 | 1.15 |
| stencil    | 53.4  | 13.6  | 10.4  | 10.2  | 10.2  | 1.02 | 1.00 |
| sieve      | 345.5 | 92.3  | 106.6 | 106.8 | 106.4 | 1.00 | 1.00 |

Summary vs `-O2`:

| target | median | geomean | max | kernels > 5% |
|---|---|---|---|---|
| `-O3`    | 1.011 | 1.015 | 1.041 | 0 / 6 |
| `-Ofast` | 1.021 | 1.311 | **4.424** | 2 / 6 |

## Interpretation (what this means for the thesis)

1. **The naive framing "O3 beats O2" is false** — on both graphs (the gate) and
   wall-clock (here), O2 and O3 are the same. A product whose pitch is "we pick O3
   for you" has no headroom to sell.

2. **The true framing — the deck's actual claim — is validated and quantified.** The
   compiler "plays it safe ... won't attempt the aggressive rewrites that could win."
   Concretely: clang refuses to vectorize / reassociate the floating-point reduction
   in `saxpy_dot` at `-O2`/`-O3` because that changes IEEE results; granted permission
   (`-Ofast`) it is **4.4× faster**. That 4.4× is *performance left on the table for
   safety*, not for lack of capability.

3. **So the product is a judgment problem, not a flag picker.** The value is deciding
   *when* a semantics-relaxing or aggressive transform is (a) safe for this code and
   (b) worth it — exactly the kind of context-dependent decision a learned model over
   the data-flow graph can make and a fixed heuristic cannot. This is where
   `z_speed` / the optimization-profile representation earns its keep.

## Caveats / next

- Apple clang 17 on arm64 (NEON). x86 + a different compiler will shift absolute
  numbers; the qualitative split (O3≈O2, headroom lives in aggressive/fast-math) is
  expected to hold and should be re-run on the training target (x86 + clang) once the
  pod is back.
- Next: expand the kernel set, add `-march=native` and PGO variants, and — the real
  milestone — show the *learned* model proposing one such rewrite, verified correct,
  that beats `-O3`.
