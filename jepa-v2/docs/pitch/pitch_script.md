# Investor pitch — what to say (verbatim script)

~3.5–4 min. Speak it, don't read it. Pause on the **bold** lines. Numbers in the
cheat-sheet at the bottom — never get them wrong.

---

### 1 · Title
"Hi — we're building **the compiler that learns.** Today, the software that runs the
world is optimized by compilers that haven't fundamentally changed in decades. We're
teaching an AI to do it better — and we're the first to bring Yann LeCun's JEPA
approach to machine code."

### 2 · The stakes
"Think about the hardest code on earth — trading engines, simulations, databases.
There, **every cycle of speed and every byte of memory is money**, and correctness is
never up for negotiation. These companies pour fortunes into performance — and they're
still leaving a huge amount on the table."

### 3 · The hidden truth
"Because here's the thing investors miss: **performance isn't decided in the source
code. It's decided in the assembly the compiler produces.** That layer is where the
real money is — and today it's run on autopilot."

### 4 · The problem
"Compilers today are **frozen, hand-written heuristics** — rules an expert tuned years
ago, in a fixed order, with cost models that get less accurate with every new chip.
They play it safe, so they leave performance behind — and critically, **they never
learn.** Every new architecture means more manual tuning. That's a treadmill, and it's
getting steeper."

### 5 · Our insight
"Our edge is two bets. One: **we represent code as a graph of how data actually
flows** — not as text like an LLM. That's the structure that drives performance. Two:
**we learn from it with JEPA**, a self-supervised world-model approach. It's just
reaching new sciences like biology — and **nobody has applied it to assembly. We're
first.**"

### 6 · How it works
"Concretely: we take the assembly, turn it into a data-flow graph, and run it through
a graph neural network we trained from scratch — **with zero labels.** Out comes an
embedding split in two: **what the code does, and how it's optimized.**"

### 7 · Proof it works  ← slow down, this is your traction
"And it works. On programs the model has never seen, it **cleanly separates those two
things** — one half is the program's meaning, blind to the optimization level; the
other is the optimization profile, blind to which program it is. You can see it right
here. **This is self-supervised, trained from scratch on a single GPU, it's on GitHub,
it reproduces from one command, and it ships with a passing test suite.** In 36 hours,
we went from idea to a working, proven core."

### 8 · We do real science
"Two things that should give you confidence in this team. **We gate before we train** —
we honestly measured the limits of today's compiler and reported the ceiling instead
of hiding it. And we debug deep: careful work on the objective took the model from
using a handful of dimensions to **72 out of 96 — on the exact same data.** That's the
kind of rigor machine-level correctness demands."

### 9 · Go-to-market
"Getting customers is the easy part, and we have a proven playbook — **the GitGuardian
motion.** We scan open-source repos, find code we can radically speed up, and reach out
automatically offering a **free optimization of their own code.** Proof, not promises —
that's how you turn open source into a pipeline of qualified leads."

### 10 · Business model  ← the money slide, be crisp and confident
"And the model is beautiful. Our customers are **teams whose compute bill *is* their
business** — high-frequency trading, databases, HPC, ML infrastructure. We price on
value: **we take a cut of the compute we save.** Optimizing a hot loop costs us cents;
the speedup recurs forever. **Five percent off a two-million-dollar compute bill is a
hundred thousand saved — we keep twenty-five thousand a year, per workload, at ninety
percent margin.**"

### 11 · The vision
"This encoder is the foundational brick of something much bigger: **a universal,
AI-driven compiler.** Any source, any language, compiled optimally for any hardware —
CPU, GPU, custom silicon. A world model for code that gets smarter with everything it
sees."

### 12 · Why now
"And the timing is now. **JEPA world models are crossing into new domains and assembly
is wide open.** Every new chip generation makes hand-tuned compilers more
unsustainable. And cheap distributed compute means a small team like us can actually
train this."

### 13 · Execution
"We're built to run lean — **massive compute at a fraction of hyperscaler cost**, and a
curriculum that teaches the model from basic algorithms up to complex architectures.
Today a single GPU trains the whole thing end to end."

### 14 · (Q&A — only if asked)
- *"Does it make code faster yet?"* → "**Not yet — and we won't pretend otherwise.**
  We've proven the representation. The next milestone is a first measured speedup on a
  real benchmark, and that's exactly what this round funds."
- *"How do you guarantee correctness?"* → "**We propose, the toolchain proves.** Every
  rewrite is checked by the compiler's own equivalence verification. We never ship
  something unverified."

### 15 · (Q&A — only if asked)
- *"Why won't Google or an LLM lab crush you?"* → "Because we learn on the **graph, not
  text**, and the durable moat is a **proprietary dataset of graph-to-measured-speedup
  pairs** that compounds with every customer. That's not something you reproduce in a
  weekend."

### 16 · Close / the ask
"So — in 36 hours we turned a bold idea into a **proven, reproducible core.** We're
raising **pre-seed to take it from disentanglement to the first real speedup on
production code**, and to lock in the data moat. **Let's make the compiler learn.**
Thank you — we'd love your questions."

---

## Numbers — never get these wrong
- Disentanglement: z_sem **gap 0.89**; z_speed ignores the program (**−0.91**).
- Capacity after tuning: **3 → 72** of 96 dimensions.
- Honesty gate: O0→O1 changes **100%** of graphs; O2≈O3 are **identical**.
- Scale: **~8,000** functions · **one B200 GPU** · **24 tests pass** · on GitHub.
- Money: **$2M bill → 5% → $100k saved → $25k/yr per workload → ~90% margin.**

## 90-second version
Slides **2 → 4 → 5 → 7 → 10 → 16**: stakes, problem, insight, proof, money, ask.
Drop 6, 11, 12, 13; keep Q&A in your back pocket.
