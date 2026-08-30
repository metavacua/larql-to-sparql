# ApolloEngine — Specification

**Status:** ✅ Shipped. Research engine; bench-only (no `larql run`
wiring). W1-GPU integration is deferred (Apollo's forward pass
already skips most layers, so the dispatch route gives a smaller
relative win than for the cached-K/V engines).

> **2026-05-24 — Apollo implements [`RetrievalEngine`], not
> [`KvEngine`].** The per-step contract diverges from the K/V cache
> engines (no per-token K/V append, no `FfnBackend` dispatch — state
> is `injection_delta` + `boundary_residual` + token list). The
> sibling trait extraction in `larql-inference::kv_engine` moves
> Apollo off the K/V trait surface; the harness wraps it in
> [`AnyEngine::Retrieval`] alongside the K/V engines. Store misses
> surface as `EngineError::RetrievalMiss { reason }` (recoverable;
> the accuracy harness reports `served_rate < 1.0`). See
> [`kv-engine-unification.md`](./kv-engine-unification.md) and the
> 2026-05-24 entry in [`larql-kv/ROADMAP.md`](../../../larql-kv/ROADMAP.md).

**Audience:** LARQL contributors.

---

## 1. Purpose

`ApolloEngine` is a retrieval-injection engine: it doesn't manage
its own K/V cache. Instead it consults an external **constellation
store** of pre-captured **boundary residuals** at layer
`crystal_layer` (default 30) and **injects** the closest match
into the forward residual stream, then runs only `crystal_layer..
num_layers` (≈ 4 layers on Gemma 3 4B) instead of the full stack.

The end-to-end speedup is ~8.5× per step when the store has a hit
— bypassing 30 of 34 layers — at the cost of being task-level
accurate (not bit-identical, not even bounded-KL).

Use case: "compile" a known task or document set into a
constellation store, then serve it at sub-step latency. The store
is built offline by capturing residuals at `crystal_layer` from
running representative prompts through `StandardEngine`; serving
just does the lookup + injection + short tail.

The engine is **not** a K/V cache — it's a residual cache. Don't
expect token-level accuracy; expect task-level recall.

---

## 2. Contract

### 2.1 Accuracy contract

> When the store has a hit (cosine to the query residual ≥ `coef`
> in the configured metric), the injected forward pass produces a
> top-1 that matches the original capture's top-1 at task level —
> e.g. "what is the capital of France" → " Paris" — with cosine on
> the injected residual ≥ 0.97 to the original.

There is **no** KL bound vs `StandardEngine` and no claim of
bit-identity. The contract is task-level: it lands on the same
top-1 / top-K with high cosine on the injected residual.

### 2.2 Memory contract

> Persistent state: the constellation store (a
> `larql_apollo::ConstellationStore` of boundary residuals + task
> labels). Engine-side memory is `O(1)` — it holds a borrowed
> reference / handle to the store; the store itself sizes with the
> number of compiled tasks.

For a 1,000-task store on Gemma 3 4B at `crystal_layer=30`:
- 1,000 × hidden_dim (3072) × 4 B = 12 MB of boundary residuals
- + task labels + index = ~14 MB total

The store is **shared across requests** — many engines can read
the same `Arc<ConstellationStore>` concurrently.

### 2.3 Forward-pass shape

```
on each decode step:
  - run layers [0, crystal_layer) normally → residual_at_crystal
  - query store(residual_at_crystal) → best_match
  - if cos(residual_at_crystal, best_match) ≥ coef:
      residual_at_crystal := residual_at_crystal + coef * (best_match - residual_at_crystal)
  - run layers [crystal_layer, num_layers) over modified residual
  - sample next token from final_logits
```

When the store has no hit (cos < threshold), Apollo falls through
to the full-stack forward pass and behaves like `StandardEngine`
for that step.

---

## 3. CLI selector

```text
apollo:layer=25,coef=8.0,top_k=12
```

| Param | Meaning | Default |
|---|---|---|
| `layer` | `crystal_layer` — where in the stack to inject | 30 |
| `coef` | injection coefficient + similarity threshold | 4.0 |
| `top_k` | how many candidates to consider before picking best | 8 |

The `larql-apollo` crate is the offline tool that captures
constellations and builds stores; `ApolloEngine` is the runtime
that consumes them.

---

## 4. Implementation

| Concern | Location |
|---|---|
| Engine struct + `RetrievalEngine` impl (NOT `KvEngine` — see the 2026-05-24 note above) | `crates/larql-kv/src/engines/apollo/engine.rs` |
| Store schema | `crates/larql-apollo/` |
| `forward_from_layer` (run-tail) | **does not exist as written** — the shipped surface is `AsyncComputeBackend::forward_from_layer_async` (`crates/larql-compute/src/async_compute_backend/cpu.rs`) |
| Residual capture (offline) | `crates/larql-apollo::capture` |

Apollo composes with `StandardEngine`'s `KvHandle` underneath when
the store misses — the back-end K/V cache continues to grow during
fall-through steps so subsequent on-hit steps still see a coherent
prior. The engine doesn't try to maintain its own K/V; it
piggybacks on the production cache for K/V continuity and only
intervenes at the residual layer.

---

## 5. Performance (Gemma 3 4B, M3 Max, 2026-05-17)

| Path | Per-step latency | Throughput |
|---|---:|---:|
| Store hit (4-layer tail) | ~1.1 ms | ~900 tok/s ceiling, single-task |
| Store miss (full forward) | ~9.4 ms | matches `standard` |

The "store hit" number is the upper bound; real workloads mix hits
and misses, so observed tok/s on `larql bench --engine apollo:...`
depends on store coverage of the bench prompts.

W1-GPU is not wired today — Apollo's hot path is already short
(4 layers), so per-layer state-dump dispatch buys less than for
the cached-K/V engines. P1 follow-up: dispatch the 4-layer tail as
a single fused Metal kernel (same shape as `standard`'s coarse
prefill).

---

## 6. Non-goals

- **Token-level accuracy.** Apollo is task-level by design.
  Diverges from `StandardEngine` even when the store hits.
- **General-purpose serving.** The store has to be built offline
  for the workload. Apollo is not a drop-in for arbitrary chats.
- **Cross-architecture transferability.** Constellation stores are
  per-architecture and per-checkpoint — residual geometry doesn't
  port across models. Rebuild the store after every model swap.
- **Per-token compression.** That's `turbo_quant`. Apollo
  compresses **whole-task computation**, not per-token state.

---

## 7. P1 follow-ups (from ROADMAP / experiments)

- **Multi-layer injection.** Today's spec injects only at
  `crystal_layer`. Experiments 11+ suggest combined fact@L10 +
  passage@L17 injection improves passage-recitation fidelity. The
  engine currently exposes only single-layer injection.
- **Fused Metal tail.** The 4-layer tail still walks layer-by-
  layer. A `coarse_tail_from_layer` dispatch surface would close
  most of the remaining latency.
- **Hit-rate metrics in `EngineInfo`.** Today the engine doesn't
  surface the fraction of decode steps that took the on-hit path
  vs fall-through — making it hard to interpret bench results.
