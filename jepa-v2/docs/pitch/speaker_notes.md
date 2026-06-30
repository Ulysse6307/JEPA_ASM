# Speaker notes — "The Compiler That Learns"

Target ~3.5 min. Don't read the slides — say these. **Bold** = the line that must land.
Transition slides (3, 6) go fast. Time budget in brackets.

---

**1 · Title** [10s]
"We're building the compiler that *learns*. An AI that optimizes machine code by
understanding how data flows through it — and to our knowledge, we're the first to
apply JEPA, LeCun's self-supervised approach, to assembly." → set the tone, move on.

**2 · The stakes** [20s]
"Think about the hardest code out there — trading engines, simulations, databases.
Every cycle of speed and every byte matters, and **correctness is never negotiable —
right down to the metal**. These teams spend fortunes chasing performance."

**3 · The hidden truth** [15s · fast, punchy]
"Here's the thing: **performance isn't decided in your source code — it's decided in
the assembly the compiler produces.** So why not let the compiler handle it? That's
exactly the problem."

**4 · The problem** [25s]
"Today's compilers run on **hand-written heuristics** — rules an expert tuned years
ago, applied in a fixed order, with cost models that drift further from the hardware
every chip generation. They play it safe, so they **leave performance on the table** —
and worst of all, **they never learn**. Every new architecture means more manual
tuning, by hand."

**5 · Our insight** [25s]
"Two bets. First, **represent code as a graph of data flow** — what depends on what.
That's the real structure that drives performance; unlike an LLM, we never touch text.
Second, **learn on that graph with JEPA** — a world-model approach. It's barely been
explored outside biology, and **never on assembly. We're first.**"

**6 · How it works** [15s · fast]
"The pipeline: assembly to a data-flow graph — control, data, call edges — through a
GNN encoder trained from scratch, no labels. Out comes a **factored embedding: what
the code does, and how it's optimized.**"

**7 · Proof it works** [30s · slow down, this is the demo]
"And it works. On held-out programs, the model **cleanly separates the two**: one
half captures meaning and ignores the optimization level; the other captures the
optimization profile and ignores which program it is. Gap of 0.89, decorrelated — you
can see it in the PCA. **Self-supervised, no labels, trained from scratch on one GPU.
It's on GitHub, reproducible, with a passing test suite.**" (← lean on "it's real".)

**8 · Real science** [20s]
"We do real science, not demo theater. **We gate before we train** — we honestly
measured where today's compiler saturates and reported the ceiling instead of hiding
it. And we push the representation hard: careful tuning took the model from using a
handful of dimensions to **72 of 96 — on the same data.**"

**9 · Go-to-market** [20s]
"How we reach customers — the **GitGuardian playbook**: scan open-source repos, flag
code we can radically optimize, reach out automatically with a **free demo on their
own code**. Irrefutable proof turns open source into qualified inbound."

**10 · Business model** [25s · the money slide, be crisp]
"Who pays: **teams whose compute bill *is* the business** — HFT, databases, HPC, ML
infra. Pricing is **value-based — we take a cut of the compute we save.** The unit
economics are beautiful: optimizing a hot loop costs cents, the speedup recurs
forever. **5% off a $2M compute bill is $100k saved; we keep $25k a year per
workload — at ~90% margin.**"

**11 · Vision** [15s]
"The encoder is the foundational brick. The end state: **a universal, AI-driven
compiler** — any source, any language, optimal machine code for any hardware. A world
model that gets smarter with every program it sees."

**12 · Why now** [15s]
"Why now: JEPA world models are just crossing into new sciences and **nobody's
touched assembly**; every new chip makes hand-tuned compilers more unsustainable; and
distributed GPU markets make training cheap for a small team."

**13 · Execution** [15s]
"We run lean — **massive compute at a fraction of the cost** via distributed cloud,
and we structure training as a curriculum, basics up to complex architectures. Today,
a single GPU trains the whole encoder end-to-end."

**14 · Q&A 1/2** [skip unless asked — these are for live Q&A]
If asked *"does it make code faster yet?"* → **"Not yet — we've proven the
representation. Next milestone is a first measured speedup on a real benchmark."**
If asked *correctness* → **"We propose, the toolchain proves — every rewrite is gated
by the compiler's equivalence checks."**

**15 · Q&A 2/2** [skip unless asked]
If asked *moat* → **"We learn on the graph, not text. The durable moat is a
proprietary corpus of (graph → measured speedup) pairs — not a weekend clone."**

**16 · Close** [15s]
"So: we've taken this from idea to a **proven, reproducible encoder** in 36 hours.
With pre-seed, the next step is the **first real speedup on real code.** Let's make
the compiler learn. Thank you."

---

## Cheat sheet — numbers to never get wrong
- z_sem gap **0.89**, z_speed program-silhouette **−0.91** (disentanglement holds).
- Capacity: **3 → 72** of 96 dims after tuning.
- Gate: **O0→O1 changes 100%** of graphs; **O2≈O3 identical** (we report the ceiling).
- Corpus: **~8k** ExeBench functions, **one B200 GPU**, **24 tests pass**.
- Business: **$2M bill → 5% → $100k saved → $25k/yr per workload, ~90% margin.**

## If you only have 90 seconds
Slides **2 → 4 → 5 → 7 → 10 → 16** (stakes, problem, insight, proof, money, ask).
