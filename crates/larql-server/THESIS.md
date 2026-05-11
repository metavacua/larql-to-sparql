# THESIS

## What this is for

`larql-server` is a **reference implementation** of inference under the
LARQL paradigm: model-as-database, training-as-compilation,
inference-as-graph-walk. It is not trying to compete with vLLM, SGLang,
or TGI on adoption; it is trying to demonstrate, in working code, what
production inference looks like when you take those theses seriously.

The expected and intended outcome is that the ideas demonstrated here
propagate into production-grade serving stacks. **The reference
implementation succeeds when its ideas are no longer unique to it.**

## Success measured in citations, not stars

If `larql-server` is a reference implementation, then "winning" doesn't
look like adoption — it looks like **diffusion**. Concretely, success
looks like:

- vLLM ships `/v1/describe` (or an equivalent indexed-knowledge query
  endpoint).
- SGLang adds expert-level sharding for CPU.
- TGI exposes patches as a first-class API.
- `llama.cpp`'s server gains a vindex loader.
- A serving-stack design doc at Anthropic, Google DeepMind, or a
  research lab cites the LARQL papers.

The reference server having 50 users while the *ideas* show up in five
production stacks is a complete win. None of those outcomes require
this codebase to have meaningful market share.

## What follows from this framing

### The roadmap is a demonstration sequence, not a product backlog.

Each item exists to make a paradigm claim concretely visible.

- **N5 (federated knowledge graph)** isn't a feature. It's an existence
  proof that "if you treat models as databases, you can federate them,
  and here's what that looks like running."
- **F-FLY (multi-host deployment)** isn't a deployment milestone. It's
  evidence that "CPU-first MoE serving works on commodity hardware at
  production tok/s" — a measurement that's hard to argue with once
  published.

The reference implementation's job is to make claims **unreplicable on
vibes**. People have to engage with the working artefact, not a
position paper.

### Parity items are legitimacy markers, not adoption blockers.

Working OpenAI compatibility is here so that when a vLLM contributor
reads the codebase, they see a serious system that handles the boring
stuff — not a research toy that punted on the hard bits. Sessions,
streaming, structured output, LoRA hot-loading — these aren't here
because users demand them; they're here so that the paradigm work is
**citable** by serving-stack engineers.

That's the difference between "interesting research prototype" and
"reference architecture for the next generation."

### Engineering decisions are evaluated for legibility, not raw speed.

"Is this clean enough that someone porting it to vLLM can read it?"
matters more than "is this the absolute fastest implementation?"

- The Q1 cleanup pass (modular `routes/expert/`, centralised
  `env_flags`, lifted magic literals, slim `main.rs`) is more
  important under this frame, not less. **Readability is now a
  primary feature, because the artefact's job is to be read and
  copied.**
- The 2026-04-27 F0 paper trail (CPU vs Metal MoE divergence, what
  was tried, what didn't help, where the bug actually localised) is
  there for whoever next debugs a similar divergence — in this
  codebase or any other. Reference implementations carry their
  forensics.
- Marking shipped work with **measurements attached** in
  `ROADMAP.md → Completed` (cos-similarity, tok/s, RSS, latency
  histograms) is the same instinct: a number someone can reproduce
  is harder to dismiss than a bullet point.

### Demonstrability beats feature scope.

Better five paradigm-distinctive capabilities each shipped with
measurement, video, and clean reference code than fifteen capabilities
in various states of done.

The video series ("I added a 769th expert to GPT-OSS, it's Python";
the Shannon experiments at ~/chris-source/chris-experiments/SHANNON_SYNTHESIS.md; the
WASM-in-FFN demos) is the same artefact at different scales: each
major capability lands as **claim → measurement → code that proves
the claim**. The research, the videos, and the server are three faces
of the same demonstration project.

## Historical precedent

The most influential systems software often *was* reference
implementations:

- **Plan 9** wasn't trying to beat Unix in market share; it was
  demonstrating ideas (everything-is-a-file pushed to its conclusion,
  per-process namespaces) that then showed up in Linux containers, in
  9P, in WSL.
- **The Burrows–Wheeler transform** shipped in `bzip2` first and then
  showed up everywhere, including in ML tokenisers via SentencePiece.
- **Bret Victor's** work on direct manipulation isn't a product. The
  ideas propagate because the demos are too clear to ignore.
- **Scuttlebutt / SSB** isn't competing with Twitter for users; the
  protocol and the patterns flow into other federated systems.
- **mcp-cli** at 1.9k stars (one of this author's other projects) does
  exactly what you'd want from a reference: people use it, fork it,
  build their own versions, and the patterns spread.

When the ideas are right, the reference implementation's job is just
to **exist legibly enough to be copied** — and the diffusion happens
whether the reference ever scales or not.

## Strategic implication

Prioritise legibility and demonstrability over feature scope. Better
to ship five paradigm-distinctive capabilities each with a measurement,
a video, and clean reference code than fifteen capabilities in various
states of done.

The ROADMAP discipline — marking items shipped *with measurements
attached* — points in this direction. Lean further into it.

## See also

- `README.md` — developer-facing entry point. Describes what the
  server does and how to use it.
- `ROADMAP.md` — current state, parity vs paradigm tracks, completed
  work with measurements.
- `docs/server-spec.md` — wire-format and endpoint reference (for
  anyone porting endpoints to another stack).
- `~/chris-source/chris-experiments/SHANNON_SYNTHESIS.md` — research thesis at the
  information-theoretic level: bits per token, slot-bits as
  factual-confidence readout, in-context decay, entropy-aligned
  measurement of the substrate this server exposes.
