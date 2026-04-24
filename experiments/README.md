# Experiments

Hypothesis-driven experiments using the vindex Python bindings.
Each directory tests one idea. Results go in `results/`.

## Setup

```bash
cd crates/larql-python
maturin develop --release
```

```python
import larql
vindex = larql.load_vindex("output/gemma3-4b-v2.vindex")
```

---

## Foundations

### 01 — Gate Synthesis
Can you synthesise a gate vector from scratch and have it match a forward pass residual?
Compare heuristic synthesis vs captured residual.

### 02 — Manifold Dimensionality
What is the true rank of the knowledge manifold? SVD of all gate vectors from knowledge layers.
If 99% variance in 15D, compress 71 GB to 416 MB.

### 03 — Build Knowledge Layer
Can you construct L14–27 from Wikidata triples? Embed entities, assign to layers by relation type,
write gate+down vectors. Run INFER — does "The capital of France is" produce Paris?

---

## Compilation & Insertion

### 04 — Constellation Insert
Trace-guided multi-feature knowledge injection with alpha sweeping.

### 06 — Backprop Insert
Unfaithful CoT and geometric interpretation validation. Backprop-based edge insertion.

### 09 — Compiled Model
Gate activation test — write structured data directly into FFN format.

### 11 — Passage Memorisation
Minimal-representation passage storage: how few edges can reproduce a verbatim passage?

### 12 — Article Compilation
Proof of concept: compile a Wikipedia-style article as fact+passage edges on v10c.

### 13 — CoT Reformulation
Frozen FFN + attention fine-tune with CoT + rehearsal mix. End-to-end on v10c.

### 14 — Vindex Compilation
Gemma 3-4B: 10/10 retrieval with decoy-aware refine and basic balancer.

---

## Language & Syntax

### 05 — Syntax Circuit Routing
Sub-centroid routing with template variants. Syntax band insertion.

### 08 — Templates, Slots & Chaining
Template funnel writes canonical answers; sub-band ablation + horn heterogeneous chaining.

### 20 — Free Monoids & Poincaré
Permutation diffuseness analysis (F2, variant iii-full).

---

## Compute & WASM

### 07 — WASM Compute Engine
Embedding deterministic solvers (arithmetic, algebra, constraint satisfaction) directly in the
inference path. Phase 1: token-level interception. Phase 2: residual-level dispatch. Phase 3: WASM
runtime in Rust.

---

## Architecture & Models

### 10 — Architecture Search
LARQL-Tiny architecture search — training and evaluation harness.

### 10b — Variable Argument Dispatch
Digit-position dispatch with K=9 slots at L30. Probe confirms N is linearly decodable from L5
residual (R²=0.887). Extraction circuit = single gate at L5.

### 15 — v11 Model
Probe training on Gemma L10 REL-slot residuals. Knowledge-first tokenizer validation.

---

## Attention & Inference

### 16 — Attention Head Cache
W_Q × W_K^T SVD analysis on all attention heads. Precomputed constants for static heads.

### 17 — Speculation Error
Attention output sensitivity to residual perturbation.

### 18 — Transformer Recutting
FSM artifacts for Gemma 3 4B — Phase A. Residual stream recutting experiments.

---

## Routing & Geometry

### 19 — Routing Vocabulary
Routing vocabulary discovery (corrected methodology, addresses DUNDER bias).

### 21 — Softmax Cliff
Softmax phase transition at ~1142 tokens. Pre-RoPE Q/K routing, residual cliff analysis,
attention bottleneck investigation.

### 22 — KV Cache Routing
K/V pre-injection, per-position routing, argmax injection, aggregate routing at scale.
Includes synthetic K/V injection probes.

### 23 — Cross-Model Routing
Cross-model routing alignment (1B reads 4B K-space), RSA similarity, GEGLU wall bypass,
FFN foreign call, format unlocking. LQL combined results.

### 24 — Routing Geometry
Geometric routing through residual space: dark dimension atlas, compass navigation,
hierarchical routing, sparse semantic index, 1D superposition capacity.
