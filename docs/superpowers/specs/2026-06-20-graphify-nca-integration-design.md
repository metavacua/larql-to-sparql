# Design: Graphify + NCA Integration — Codebase-to-Language-Model Pipeline

**Date:** 2026-06-20  
**Branch:** `feat/graphify-nca-integration`  
**Status:** Design — approved, pending implementation plan

---

## 1. Vision

Transform any codebase into a deployable language model usable by standard LLM tooling
(ollama, tabby, goose, openrouter, NCA, pi-agent, and equivalents) without requiring
pretrained weights, cloud access, or a training loop.

The pipeline is:

```
any codebase
  └─ larql extract-codebase .   →  codebase.vindex  (BitNet basis, inference-capable)
       └─ larql export --format gguf  →  model.gguf  (loadable by ollama/tabby/etc.)
```

Cross-validation with graphify (Python reference implementation) is available via
`larql extract-graphify` + `larql graph-diff`. graphify is a reference tool with preserved
git provenance; it is not a runtime dependency of the finished pipeline.

NCA (`metavacua/native-cli-ai`, forked for larql integration) uses the resulting model
as its local inference backend. Cloud providers (Anthropic, MiniMax) are explicit
opt-in flags, never defaults.

---

## 2. Mathematical Foundations

### 2.1 Unit-Asserted Graphs Are Implicit Weight Matrices

A knowledge graph where every positive assertion carries confidence = 1.0 has an
adjacency matrix A where A[i,j] ∈ {0, 1}. Symmetric row-column normalisation
`A_norm = D^{-½} A D^{-½}` produces unit vectors. Every absent edge is weight 0
(open-world assumption). Every present edge is weight +1.

This is already a valid transformer weight matrix. No training is required. No
pretrained base is required. Weight derivation is making the implicit explicit.

### 2.2 BitNet Trit Encoding Is the Natural Basis

Weights drawn from {+1, 0, −1} map directly:
- Positive assertion present → +1
- Assertion absent → 0
- Explicit negation → −1

larql already has trit encoding in `larql-vindex` (BitNet PR #159). The
codebase-to-model pipeline uses this existing infrastructure.

### 2.3 God Nodes Correspond to Superposition

High-degree nodes (graphify's "god nodes") appear in so many relation contexts that
their representations must be compressed across multiple dimensions — exactly the
superposition hypothesis from mechanistic interpretability. God-node detection in
graphify is implicitly finding the nodes that would be superposition-packed in the
weight matrix. The trit encoding's signed dimension handles this: a god node gets
multiple non-zero entries in its weight row, with sign derived from in/out degree
asymmetry.

### 2.4 Change of Basis Connects to All Model Families

BitNet is the natural starting basis for unit-asserted graphs. Other model families
are reachable via basis transforms:

| Target basis | Transform | Notes |
|---|---|---|
| Dense f16 | cast + scale from trit | Standard transformer format |
| Q4_K / Q8_K | quantise f16 | larql already has kquant |
| Spectral | eigenvectors of graph Laplacian L = D − A | Global structure in fewer dims |
| MoE | decompose by relation type into expert FFNs | Relation → routing key |

The `BasisTransform` trait (Section 5.3) is the extension point for future families.
"Change of basis" is not a metaphor — it is the mathematical operation connecting
the unit-weight representation to any other weight family.

### 2.5 Monoid Structure of Path Traversal

WALK over a unit graph forms a monoid: identity = self-loop, composition = edge
following, associativity = path concatenation. LQL `WALK "A" VIA ["calls","imports"]`
is curried monoid composition — two single-hop operations chained. Each relation type
is a curried linear map. This is why relation-specific projection matrices in the
attention heads are the correct architectural choice.

---

## 3. Resource Tiers and Compile-Time Enforcement

All code is assigned to a resource tier. The tier determines which compile target it
must build for and what CI jobs validate it.

| Tier | Resources allowed | Compile target | CI validation |
|---|---|---|---|
| 0 — Pure compute | None | `wasm32v1-none` ✓ | `cargo build --target wasm32v1-none` |
| 1 — Filesystem | Read/write files, mmap | native | standard `cargo build` |
| 2 — Network | TCP / HTTP / gRPC | native | standard `cargo build` |
| 3 — Full OS | Processes, signals, env | native | standard `cargo build` |

`wasm32v1-none` has no WASI — no system calls, no filesystem, no network. Code that
compiles for this target provably has zero OS/IO/network dependencies. This is
enforced by the linker, not by convention.

`model-compute` already demonstrates this pattern (`features = ["native", "wasm"]`).
The design extends it systematically across the workspace.

### 3.1 Tier 0 Crates (wasm32v1-none safe)

These crates must build cleanly for `wasm32v1-none` in CI:

| Crate | Contents |
|---|---|
| `model-compute` (wasm feature) | Bounded kernels — already compliant |
| `larql-core` (after io split) | In-memory Graph, algorithms, schema, Edge/Node types |
| `larql-lql-core` (new split) | Lexer, parser, AST, pure evaluator |
| `larql-codebase-core` (new) | Weight derivation, adjacency normalisation, basis transforms, superposition encoding, architecture sizing |

### 3.2 Tier 1 Crates (filesystem, no network)

| Crate | Contents |
|---|---|
| `larql-core-io` (new split from larql-core) | `.larql.json` file read/write |
| `larql-vindex` | Vindex lifecycle, mmap, GGUF reader |
| `larql-codebase` (new) | tree-sitter file scan, AST extraction, vindex write, GGUF export |
| `larql-lql` | LQL executor with vindex IO (retains USE REMOTE stub that errors if called offline) |

### 3.3 Tier 2 Crates (network)

| Crate | Contents |
|---|---|
| `larql-server` | HTTP + gRPC server |
| `larql-inference` (network parts) | USE REMOTE client, remote FFN expert shards |
| NCA `HttpProvider` | HTTP provider for larql-graph-serve / ollama / cloud |

### 3.4 LQL Offline Subset

Operations available with no network and no pretrained weights:

```
DESCRIBE, WALK, SELECT, RELATIONS, STATS    — Tier 0+1
INSERT, DELETE, UPDATE (patch overlay)      — Tier 1
USE MODEL <local-path>                      — Tier 1
INFER (on codebase vindex, BitNet)          — Tier 0+1
COMPILE                                     — Tier 1
─────────────────────────────────────────── offline boundary
USE REMOTE                                  — Tier 2 (network, explicit)
larql pull / larql publish                  — Tier 2 (HuggingFace, explicit)
```

### 3.5 Required SourceType Addition

`larql-core/src/core/enums.rs` `SourceType` enum needs an `Ast` variant before
codebase extraction can tag edges correctly:

```rust
pub enum SourceType {
    Parametric, Document, Installed, Wikidata, Manual,
    Ast,        // ← new: edges derived from static AST analysis
    #[default]
    Unknown,
}
```

---

## 4. Architecture

```
any codebase
     │
     ├── larql extract-codebase .          [Tier 1+0, Rust, tree-sitter]
     │       AST → triples → adjacency → trit weights → codebase.vindex
     │       --basis f16|spectral|moe      [change-of-basis variants]
     │
     ├── larql extract-graphify graph.json [Tier 1, cross-validation]
     │       NetworkX node-link → .larql.json
     │
     │       larql graph-diff a.larql.json b.larql.json   [Tier 0]
     │           → consensus.larql.json (agree=1.0, diverge=0.6+flag)
     │
     ├── larql export --format gguf <vindex>   [Tier 1]
     │       → model.gguf
     │       ollama / tabby / openrouter / goose / NCA / pi-agent
     │
     └── larql graph-serve <vindex|.larql.json>   [Tier 2]
             /v1/chat/completions  /v1/describe  /v1/walk  /v1/select
             NCA --provider larql  [Phase 1 HTTP bridge]
             NCA VindexProvider    [Phase 2 direct larql_core link]
```

**Nothing in the existing stack changes.** `larql extract-index`, `larql build`
(Vindexfile FROM path), `larql serve`, the LQL parser/executor, all existing
crates — untouched. All new capabilities are additive.

---

## 5. New Crate: `larql-codebase`

Positioned after `larql-core` and `larql-vindex` in the dependency chain.
`larql-cli` gains the new commands; `larql-codebase` never imports `larql-cli`.

```
larql-codebase-core   [Tier 0]
      ↓
larql-codebase        [Tier 1, depends on larql-codebase-core + larql-vindex]
      ↓
larql-cli             [Tier 3, gains extract-codebase, extract-graphify, graph-diff, export]
```

### 5.1 Supported Languages (starting set)

Rust, Python, TypeScript/JavaScript. Each is a module under
`larql-codebase/src/languages/`. Additional languages are additive with no
architectural changes.

### 5.2 Node Kinds and Relation Types

**Node kinds:** `function`, `struct`, `trait`, `enum`, `impl`, `module`, `file`,
`class`, `interface`, `type_alias`

**Relation types (structural subset shared with graphify):**

`calls`, `imports`, `contains`, `inherits`, `implements`, `defines`, `references`,
`method`, `field`, `parameter_type`, `return_type`

Each edge: `(subject: label, relation, object: label, confidence: 1.0,
source: SourceType::Ast, metadata: {kind, source_file, line})`

### 5.3 Weight Derivation Pipeline (`larql-codebase-core`, Tier 0)

1. **Adjacency matrix A**: A[i,j] = 1 if any edge i→j exists (direction preserved).
   Entities assigned indices from sorted label list for determinism.

2. **Symmetric normalisation**: `A_norm = D^{-½} A D^{-½}`. Each row is a unit
   vector representing the entity's normalised neighbourhood.

3. **Relation-type projections**: For each relation type r, build A_r (edges of
   type r only). A_r becomes an attention head weight matrix — one head per
   relation type, grouped if relation count exceeds head budget.

4. **God-node detection**: degree > μ + 3σ on the degree distribution (flagged in
   edge metadata `is_god_node: true`). These nodes receive superposition encoding:
   multiple non-zero entries in their weight row, sign from in/out degree asymmetry.

5. **Trit quantisation (BitNet basis)**:
   `W_trit = sign(A_norm)` → {+1, 0, −1}.
   Written using existing `bitnet_writer` in `larql-vindex`.

6. **Architecture sizing from graph statistics**:

   | Parameter | Derivation |
   |---|---|
   | `hidden_dim` | ⌈log₂(n_entities)⌉ × 64, rounded to next power of 2 |
   | `n_layers` | graph diameter (longest shortest path, capped at 32) |
   | `n_heads` | number of distinct relation types |
   | `vocab_size` | n_entities + n_relation_types + 4 special tokens |
   | `max_seq_len` | max path length in graph |

### 5.4 BasisTransform Trait (`larql-codebase-core`)

```rust
pub trait BasisTransform: Send + Sync {
    fn name(&self) -> &str;
    fn transform(&self, adjacency: &AdjacencyRepr) -> WeightRepr;
}

pub struct BitNetBasis;    // default: sign(A_norm) → trit
pub struct F16Basis;       // cast + scale
pub struct SpectralBasis;  // eigenvectors of L = D − A
pub struct MoEBasis;       // decompose by relation type
```

`larql extract-codebase --basis <name>` selects the transform at extract time.

---

## 6. New Commands

### 6.1 `larql extract-codebase <dir> [opts]`

**Tier:** 1 (filesystem read) + 0 (weight derivation)  
**Output:** `<dir>-codebase.vindex` by default, or `--out <path>`  
**Options:** `--lang rust,python,ts` (default: auto-detect), `--basis bitnet|f16|spectral|moe`  
**Location:** `larql-cli/src/commands/extraction/extract_codebase_cmd.rs`

Produces an inference-capable vindex with BitNet weights derived from the codebase
AST. Supports full LQL including INFER (BitNet forward pass).

### 6.2 `larql extract-graphify <graph.json> [--out <path.larql.json>]`

**Tier:** 1 (filesystem read/write)  
**Output:** `.larql.json` compact edge format  
**Location:** `larql-cli/src/commands/extraction/extract_graphify_cmd.rs`

Reads graphify's NetworkX node-link JSON. Applies φ:
- node.label → entity name
- edge.relation → relation string (vocabulary identical, no mapping needed)
- node.kind → `INSERT entity, "has_kind", kind` (metadata triple)
- node.source_file → `INSERT entity, "defined_in", source_file` (metadata triple)

Used for cross-validation against `larql extract-codebase` output.

### 6.3 `larql graph-diff <a> <b> [--out consensus.larql.json]`

**Tier:** 0 (pure computation on in-memory Graph structs)  
**Input:** two `.larql.json` files or vindexes  
**Location:** `larql-cli/src/commands/extraction/graph_diff_cmd.rs`  
**Core logic:** `larql-core` (Tier 0)

Confidence scoring:
- Triple in both inputs: confidence = 1.0
- Triple in A only: confidence = 0.6, metadata `{divergence: "a_only"}`
- Triple in B only: confidence = 0.6, metadata `{divergence: "b_only"}`

Outputs consensus graph + divergence report. Run iteratively to converge the two
extraction implementations toward agreement.

### 6.4 `larql export --format gguf <vindex> [--out model.gguf] [--arch bitnet]`

**Tier:** 1 (filesystem read/write)  
**Location:** `larql-cli/src/commands/primary/export_cmd.rs`  
**GGUF writer:** uses existing `larql_models::loading::gguf::writer::GgufWriter` (`larql-models/src/loading/gguf/writer.rs`), which already provides `to_bytes()` and `write_to_file()`. No new writer needed.

Writes a GGUF file with BitNet weights using the **I2_S** quantisation type:
2-bit signed, strided block layout (128-element blocks / 32 bytes; byte `p` packs
elements `{p, p+32, p+64, p+96}` at bit-shifts 6/4/2/0; unsigned code {0,1,2} →
ternary {-1,0,+1}; zero tensor packs to `0x55`). This is the Microsoft BitNet b1.58
layout, already implemented in `larql-models/src/loading/gguf/loader.rs` (PR #156
decode fix, PR #159 end-to-end inference, both merged). After export:

```bash
ollama create my-codebase -f Modelfile   # Modelfile: FROM ./model.gguf
ollama run my-codebase
```

GGUF metadata written:

| Field | Value |
|---|---|
| `general.architecture` | `"bitnet"` |
| `general.name` | vindex metadata name |
| `tokenizer.ggml.tokens` | entity labels + relation types + special tokens |
| `tokenizer.ggml.model` | `"gpt2"` |
| `<arch>.embedding_length` | hidden_dim from weight derivation |
| `<arch>.block_count` | n_layers |
| `<arch>.attention.head_count` | n_heads |

### 6.5 `larql graph-serve <path> [--port 8282]`

**Tier:** 2 (network — HTTP server)  
**Input:** `.larql.json` or vindex  
**Location:** `larql-cli/src/commands/primary/graph_serve_cmd.rs`

Serves browse routes from a `.larql.json` (no weights required) or full inference
routes from a vindex with weights. Reuses existing route handlers from `larql-server`.

| Route | Weights required |
|---|---|
| `GET /v1/describe` | No |
| `GET /v1/walk` | No |
| `GET /v1/select` | No |
| `GET /v1/stats` | No |
| `POST /v1/chat/completions` | Yes |
| `POST /v1/completions` | Yes |

---

## 7. NCA Integration

### 7.1 Provider Hierarchy (cloud-optional)

`larql-core/src/engine/http_provider.rs` gains:

```rust
pub fn larql_graph(port: u16) -> Self {
    Self::new(format!("http://localhost:{port}/v1"), "codebase")
}
```

Default provider resolution order (no flags):
1. `~/.nca/config.toml` declared provider
2. `larql graph-serve` on port 8282 (local, no cloud)
3. ollama on port 11434 (local, no cloud)
4. Error — explicit `--provider anthropic|minimax` required for cloud

| Flag | Backend | Tier |
|---|---|---|
| *(default)* | config → larql → ollama | 1–2 |
| `--provider larql` | `larql graph-serve` (local) | 2 |
| `--provider ollama --model <n>` | ollama (local) | 2 |
| `--provider anthropic` | Anthropic API | 2 (cloud, explicit) |
| `--provider minimax` | MiniMax API | 2 (cloud, explicit) |

### 7.2 Phase 1 — HTTP Bridge

NCA `run` command uses `HttpProvider::larql_graph(8282)`. Workflow:

```
nca run --prompt "..."
  → HttpProvider::larql_graph → larql graph-serve <codebase.vindex>
  → /v1/chat/completions (BitNet inference, local)
```

### 7.3 Phase 2 — Direct Integration (native)

New `larql-core/src/engine/vindex_provider.rs`:

```rust
pub struct VindexProvider {
    graph: Arc<Graph>,
    weights: Option<BitNetWeights>,
}
```

- Context retrieval: zero-latency `graph.describe()` / `graph.walk()` (Tier 0)
- Reasoning: BitNet forward pass if weights present (Tier 0)
- Loading from disk: `Graph::from_json_value()` on startup (Tier 1, one-time)

`nca index build` updated to write the vindex path into `cli-index.json` so
`nca run` auto-loads `VindexProvider` when a workspace vindex exists.

---

## 8. CI Enforcement

New matrix entries in `.github/workflows/`:

```yaml
- name: Tier 0 wasm32v1-none build
  run: |
    rustup target add wasm32v1-none
    cargo build --target wasm32v1-none -p larql-codebase-core
    cargo build --target wasm32v1-none -p larql-lql-core
    cargo build --target wasm32v1-none -p larql-core
    cargo build --target wasm32v1-none -p model-compute --features wasm
```

Failure is a compile error, not a lint finding. A hidden OS/network dependency in a
Tier 0 crate breaks the build immediately.

---

## 9. Out of Scope for This Design

- Semantic labelling of graph communities (requires LLM — explicitly deferred)
- Multi-codebase federated vindex (separate design)
- GPU acceleration of BitNet forward pass on codebase vindexes
- Automatic Vindexfile generation from `.larql.json` (intermediate Vindexfile path still
  valid; this design adds a direct path that bypasses it)
- NCA tool execution (file edit, shell) — NCA's existing tool layer is unchanged

---

## 10. Open Questions (not blocking implementation)

- **Spectral basis convergence**: Graph diameter as `n_layers` may be too deep for
  large codebases. A cap (32 layers) is specified; empirical tuning needed.
- **GGUF BitNet quant type**: Confirmed **I2_S** (Microsoft strided block layout,
  PRs #156 + #159 merged). The existing `bitnet_writer`/`bitnet_loader` in
  `larql-vindex/src/extract/` handle the vindex-side encoding; the GGUF export
  writes the same I2_S bytes in the GGUF container format.
- **`larql-lql-core` split scope**: The LQL executor boundary (which parts are pure
  vs. IO-dependent) needs a file-by-file audit of `larql-lql/src/executor/` at
  implementation time.
