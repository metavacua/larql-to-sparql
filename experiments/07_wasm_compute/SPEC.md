# Experiment 07 — Spec

**Chris Hay · LARQL Project · Apr 2026 (v2)**

Embedding deterministic compute into neural-model pipelines — at compile
time, at inference time, or as a sandboxed runtime module.

This spec describes the **architecture as it stands after the experiment
arc**. For the historical phase-1/2/3 proposal and the standalone
sub-specs, see [`archive/`](archive/).

---

## 1. Thesis

A language model's accuracy on formal problems is bounded by its ability
to compute exactly. Three distinct paths can eliminate that bound:

| Tier | Where compute runs | What it handles |
|---|---|---|
| **Tier 3 — AOT static** | Standard weight matmuls | One known answer per compiled edge |
| **Tier 2 — AOT compiled** | Standard weight matmuls | Prompt-triggered answers, resolved at build time |
| **Tier 1 — Runtime dispatch** | WASM module or Rust kernel | Any computation, any input, any forward pass |

Tiers 2 and 3 are production today (via `larql compile` + `COMPILE CURRENT
INTO MODEL`). Tier 1 is the research bet — the hook point is designed,
but the forward-pass integration has not shipped.

---

## 2. Architecture

### 2.1 The residual stream as a programmable data bus

The transformer forward pass is a dataflow processor:

- **Residual stream** (dim=512 on v11, 2560 on Gemma 3 4B): register file.
- **FFN slots** (2048/layer on v11, 10240/layer on Gemma): instruction set.
- **Gate vectors**: instruction decoder (dot-product against FFN input).
- **Layer index**: program counter (fixed, sequential, no jumps).

Three slot types coexist in each FFN layer. The rest of the network
cannot distinguish between them because all three contribute additively
to the same residual.

| Slot type | Defined | What it does |
|---|---|---|
| **Frozen** | Training time | Knowledge, language, reasoning (trained weights) |
| **Compiled edge** | Compile time | Fixed (gate, up, down) triple via `install_edge` |
| **WASM gate** | Runtime | Arbitrary callable via forward hook (research, not production) |

### 2.2 The compiled-edge install primitive

The convention that a compiled answer follows when landing into weights:

```text
gate[slot, :]  ← trigger̂ × g_norm × gate_scale
up[slot,   :]  ← trigger̂ × u_norm
down[:, slot]  ← write × (d_norm / ‖write‖) × alpha_mul
```

- `trigger`: unit direction in FFN-input space (residual capture or entity embedding).
- `write`: the answer direction (token embedding, or synthetic tag for downstream relay).
- `g_norm`, `u_norm`, `d_norm`: reference L2 norms of the original slot, so the
  installed edge sits in the same magnitude regime as trained slots.
- `gate_scale`: 30.0 fires the gate strongly (vs typical ~1–5 frozen preacts).
- `alpha_mul`: 1.0 for tag-style writes, 10.0 for token-embedding writes
  routed through the LM head.

**Implementation:** `crates/larql-cli/src/commands/extraction/compile_cmd/edge.rs`.
Seven tests including the magnitude-preservation invariant. Called by both
compile modes (`single` — CLI-driven prompt/answer; `patch` — vindex-driven).

### 2.3 Bounded compute (Tier 1, native half)

Deterministic Rust kernels with hard-capped cost, used at compile time
to resolve an answer string that the caller then converts to a token
and installs as an edge.

**Implementation:** `crates/model-compute/` with feature `native` (default).

| Kernel | Examples | Bound |
|---|---|---|
| `arithmetic` | `sum(1..101) → "5050"`, `factorial(10) → "3628800"` | ≤10⁸ iter; factorial ≤20 |
| `datetime` | `days_between('2026-01-01', '2026-04-16') → "105"` | O(1) per op, chrono gregorian range |

Future kernels (`regex`, `units`, `graph`) slot in via the `Kernel` trait
and `KernelRegistry::register()`. Not in V1 because no demo needs them yet.

### 2.4 WASM compute (Tier 1, sandboxed half)

Wasmtime-hosted compute modules with fuel/memory caps, for problems too
input-variable or too expensive for native kernels (CP-SAT, symbolic
algebra, SAT/SMT).

**Implementation:** `crates/model-compute/` with feature `wasm`. Optional
dep; users who only need arithmetic don't pay the wasmtime compile cost.

Guest ABI (matches `solver/src/lib.rs`):

| Export | Signature | Purpose |
|---|---|---|
| `alloc` | `(u32) → i32` | reserve input buffer |
| `solve` | `(i32 ptr, u32 len) → u32` | run, return status (0/2 ok, 1 infeasible) |
| `solution_ptr` | `() → i32` | pointer to output buffer |
| `solution_len` | `() → u32` | length of output buffer |

Every call creates a fresh `Store` with explicit fuel and memory caps —
no state bleeds between calls, and exceeding limits errors rather than
wedges the host.

### 2.5 Synthetic-tag ABI (research)

For research work where a runtime WASM gate needs to pass a value to a
downstream frozen edge via the residual stream, using real token
embeddings as triggers fails — natural residual content accidentally
triggers the edge (see archived `WASM_GATE_ARCHITECTURE.md` §2.3).

The solution is to use Gram-Schmidt-orthogonalised synthetic directions
as the inter-stage calling convention. Orthogonal to baseline residual
at each target layer and to each other, they don't fire spuriously.

**Status:** validated on v11 in the archived experiments. Not in
production — no forward-hook infrastructure exists in `larql-inference`
and no current use case needs it.

---

## 3. Substrate invariants

Residual persistence (cosine between a write at L and the FFN input at
L+k) is substrate-dependent, not primitive-dependent. Known profiles:

| Substrate | Per-hop cos | Usable hops | Refresh needed |
|---|---|---|---|
| v11 TinyModel (dim=512, 20 layers) | ~0.99 | 8+ | No |
| Gemma 3 4B Instruct (dim=2560, 34 layers) | ~0.48 | ~3 | Yes, relay edge at L15 recovers 0.10 → 0.62 |

Any multi-hop program needs to budget for its substrate. Programs deeper
than `usable_hops` need a refresh-edge primitive (trigger = tag, write =
tag) to re-amplify the signal.

---

## 4. Out of scope for this spec

- **Forward-pass dispatch for WASM gates.** The hook point is designed
  (`LayerAction::Compute(problem)` in the archived architecture doc) but
  nothing in `larql-inference` implements it. Research path, may never
  ship.
- **A unified LQL surface for compile-time computation.** `INSERT` is
  specified as structured `(entity, relation, target)` constellation
  installs per the LQL spec; a `(prompt, answer)` form would break that.
  Deferred until real multi-source usage shapes the right syntax. Today
  the CLI (`larql compile --prompt ... --answer ...`) is sufficient.
- **Loader/tokenizer/forward-pass code.** Lives in `larql-models`,
  `larql-inference`, etc. The compute crates are deliberately model-host
  agnostic.

---

## 5. Where the code lives

| Concern | Path |
|---|---|
| `install_edge` primitive | `crates/larql-cli/src/commands/extraction/compile_cmd/edge.rs` |
| Compile CLI (prompt/answer + vindex-patch) | `crates/larql-cli/src/commands/extraction/compile_cmd/` |
| Native kernels | `crates/model-compute/src/native/` |
| WASM host runtime | `crates/model-compute/src/wasm/` |
| CP-SAT solver (guest) | `experiments/07_wasm_compute/solver/` |
| CP-SAT Rust demo | `crates/model-compute/examples/cpsat_scheduling.rs` |
| CP-SAT Python forward-pass demo | `experiments/07_wasm_compute/wasm_solver_demo_v11.py` |
| Persistence sweep data | `persistence_sweep.py` |

See [`RESULTS.md`](RESULTS.md) for findings and
[`ROADMAP.md`](ROADMAP.md) for what's next.
