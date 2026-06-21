# LARQL Sub-project Decomposition & OSS Alternatives Audit

**Date**: 2026-06-20  
**Scope**: Structural decomposition of the LARQL workspace into independent sub-projects/toolchains, and a comparative audit against Apache-2.0/MIT-licensed OSS alternatives — identifying where LARQL is reinventing something that already exists.

---

## 1. What LARQL Is

LARQL is a Rust research substrate built on a single thesis: **the model IS the database**. It decompiles transformer weights into a queryable format (vindex) and provides LQL to browse, mutate, and run inference over that format. The ultimate aim is frontier-scale inference (100B–1T+ param MoE models) on consumer hardware without GPU, via sparse weight retrieval.

It is explicitly **not** a production inference engine, not a competitor to Ollama/vLLM/llama.cpp in the commercial sense — but must match within 10% of llama.cpp on baseline benchmarks before claiming deltas from novel techniques.

---

## 2. Crate Inventory by Sub-project

The workspace decomposes into **8 natural sub-projects** plus **3 toolchain/protocol layers**:

### Sub-project A — Knowledge Graph Engine (`larql-core`)
**What it does**: Typed weighted property graph (entities + edges with confidence scores). Algorithms: BFS/DFS traversal, PageRank, A\*, Dijkstra, connected components, merge, diff, filter, walk, JSON/msgpack/CSV/packed IO, checkpoint log.

**External deps**: `reqwest`, `rmp-serde`, `serde_json`

**Novel?**: No — this is a generic graph library with domain-specific IO. The confidence-weighted edge model and knowledge-graph IO conventions are LARQL-specific but the algorithms are textbook.

**Note on PageRank**: petgraph 0.8.3 (confirmed latest as of 2026-06-20 via docs.rs) includes `petgraph::algo::page_rank`. The earlier claim that petgraph lacks PageRank is incorrect — it has it.

---

### Sub-project B — Model Format (`larql-models`, `larql-vindex`, `larql-vindex-spec`)
**What it does**: 
- `larql-models`: Architecture definitions (Gemma, Granite, etc.), safetensors loading, tensor key mappings
- `larql-vindex`: The vindex on-disk format — mmap storage, gate vector HNSW index, extract pipeline (safetensors→vindex), Q4K/Q6K/Q8K quant pack/unpack, canonical analysis (whitening, covariance, Hilbertian residual, regime classification), insert/delete mutations, clustering, vindexfile I/O
- `larql-vindex-spec`: Public on-disk format contract (types + JSON Schema, zero larql deps)

**External deps**: `safetensors`, `memmap2`, `ndarray`, `tokenizers`, `hf-hub`, `rayon`, `sha2`, `base64`

**Novel?**: The vindex format itself is genuinely novel — there is no OSS equivalent of "model weights as a queryable typed graph with gate-vector KNN index". The HNSW index is a custom implementation. The canonical/Hilbertian analysis pipeline is original research.

---

### Sub-project C — Compute Backends (`larql-compute`, `larql-compute-metal`)
**What it does**:
- `larql-compute`: `ComputeBackend` trait + CPU impl. BLAS f32 matmul (OpenBLAS/Accelerate), hand-written C kernels for Q4K/Q6K/Q8K matvec (ARM NEON `vdotq_s32`), GEGLU activation, fused causal attention, Cholesky/ridge solve
- `larql-compute-metal`: Apple Silicon Metal GPU backend — custom MSL shaders for Q4K matmul, fused attention, RoPE, MoE dispatch. Uses `metal-rs` + `objc`

**External deps**: `ndarray`, `rayon`, `blas-src`/`openblas-src`, `metal`, `objc`

**Novel?**: The Q4K/Q6K kernels are a reimplementation of patterns from llama.cpp (MIT). The Metal shaders are original MSL work, though they lag llama.cpp in two identified areas: flash attention (D-ATTN-MTG) and `simdgroup_matrix` prefill matmul (D-PREFILL-MM2). The compute trait abstraction is well-designed.

---

### Sub-project D — Inference Engine (`larql-inference`, `larql-kv`)
**What it does**:
- `larql-inference`: Full transformer forward pass (GQA attention, FFN sparse retrieval via WalkFfn, MoE dispatch, multi-modal prefix, layer graph, residual tracing, tokenizer, async pipeline)
- `larql-kv`: Pluggable KV-cache engines — Standard, MarkovResidual, TurboQuant, UnlimitedContext, BoundaryKV, Apollo (retrieval engine)

**External deps**: `tokenizers`, `wasmtime`, `tokio`, `tonic`, `minijinja`, `half`, `memmap2`

**Novel?**: The WalkFfn / sparse FFN retrieval is the central research invention — no OSS equivalent. The KV cache engines (MarkovResidual, TurboQuant, BoundaryKV) are novel research implementations with no OSS equivalent.

---

### Sub-project E — Query Language (`larql-lql`)
**What it does**: LQL (Lazarus Query Language) parser + executor + REPL. Custom hand-written lexer and recursive descent parser. Statements: USE, DESCRIBE, SELECT, INSERT, DELETE, UPDATE, MERGE, WALK, INFER, COMPILE, EXPLAIN, STATS, DIFF, SHOW. Uses `rustyline` for REPL history/completion.

**External deps**: `rustyline`, `serde_json`, `reqwest`

**Novel?**: LQL semantics are genuinely novel and LARQL-specific. The implementation (hand-written lexer + parser) is unsurprising but is maintenance work.

---

### Sub-project F — Quantum / Hilbert Formalism (`larql-hilbert`)
**What it does**: NOT a space-filling curve. Implements: complex structure formalization, unitary ops, single-qubit / Bloch sphere, Born rule, n-qubit (GHZ/W states), entanglement entropy bipartition, NQubitLM, NRegister, LIMLL (Linear Logic Model) fragment. Foundation for quantum language models (QLM) in the QLM-ROADMAP.

**External deps**: `ndarray`, `num-complex`

**Novel?**: Original research (Rosko 2025, arXiv:2511.21296). The LIMLL formalism tying linear logic to transformer architecture is novel. No OSS equivalent.

---

### Sub-project G — CLI (`larql-cli`)
**What it does**: Top-level `larql` binary. All subcommands: `extract`, `run`, `convert`, `pull`, `list`, `serve`, `bench`, `dev` (diagnostics/probes), `shannon`, `parity`, quantize, accuracy. Multi-modal `--image` support.

**External deps**: `clap` 4 (MIT/Apache), `indicatif` 0.17 (MIT)

**Novel?**: CLI dispatch layer only. Uses best-in-class OSS for CLI parsing and progress bars.

---

### Sub-project H — Knowledge Pipeline (`knowledge/`)
**What it does**: Python package for reference databases and probe labels. WordNet integration, feature probe via MLX, AST analysis (tree-sitter), vindex label ingest.

**External deps**: `numpy`, `nltk` (Apache-2.0), `mlx`/`mlx-lm` (MIT, Apple Silicon), `tree-sitter` (MIT), `tokenizers` (Apache-2.0)

**Novel?**: The probe methodology (gate-vector labeling via WordNet relations) is novel. All Python dependencies are best-in-class OSS.

---

### Toolchain Layer 1 — Server / Router (`larql-server`, `larql-router`, `larql-router-protocol`)
**What it does**: HTTP+gRPC server (axum+tonic), layer-sharding router with QUIC transport (quinn), Prometheus metrics, OpenAPI (utoipa), self-assembling distributed expert grid.

**External deps**: `axum`, `tonic`, `prost`, `prometheus`, `quinn`, `rustls`, `tower`, `utoipa`

**Novel?**: Uses excellent existing OSS throughout. The distributed expert grid protocol is novel but built on standard gRPC.

---

### Toolchain Layer 2 — Boundary Codec (`larql-boundary`)
**What it does**: Confidence-gated residual compression codec. Compresses transformer final-layer residuals to int8 (2× compression) with a gate that falls back to bf16 when the boundary is fragile. Used by BoundaryKV engine.

**External deps**: `serde`, `serde_json`

**Novel?**: Genuinely novel. The three-phase architecture (codec / metadata / gate) and the accuracy contract (top-1 preservation, not MSE) are research contributions.

---

### Toolchain Layer 3 — WASM Compute (`model-compute`, `larql-experts`)
**What it does**:
- `model-compute`: Portable bounded-cost compute — native Rust kernels + optional wasmtime-hosted WASM modules with fuel/memory caps
- `larql-experts`: 20 WASM expert modules compiled to cdylib (arithmetic, dijkstra, graph, markov, sql, statistics, geometry, trig, finance, hash, etc.)

**External deps**: `wasmtime`, `evalexpr`, `wat`

**Novel?**: The WASM-as-replaceable-kernel pattern is novel. Uses wasmtime (Apache-2.0) correctly. Designed for extraction to a sibling repo.

---

## 3. OSS Alternatives Comparison Table

| Subsystem | LARQL DIY | OSS Alternative | License | OSS Maturity (2026) | Replace? | Recommendation |
|---|---|---|---|---|---|---|
| **Graph storage + standard algorithms** | `larql-core` — typed edge graph, BFS/DFS/PageRank/A\*/Dijkstra/CC | `petgraph` 0.8.3 | MIT/Apache-2.0 | **Production-grade**; used by cargo, rustc. Has: BFS/DFS, Dijkstra, A\*, Bellman-Ford, PageRank (`petgraph::algo::page_rank` — confirmed 2026), connected components, SCC, topo sort, min spanning tree. Gap: custom confidence-edge model, LARQL domain IO. | **Yes, partial** | Use petgraph as graph backing store; thin wrapper for typed confidence-edges. All standard traversal/algo code in `algo/` (~500 LOC) can be removed. Merge/diff/walk/IO stays in LARQL. |
| **HNSW vector index** | `larql-vindex` — custom HNSW with random projection (64D), ~300 LOC | `usearch` 2.25.2 (Apache-2.0, verified crates.io 2026), `instant-distance` 0.6.1 (MIT), `hnswlib-rs` (Apache-2.0) | Apache-2.0 / MIT | `usearch`: C++ core with Rust bindings, production ANN, actively maintained (Unum, latest 2.25.2, released 2026-05-02). `instant-distance`: pure Rust HNSW, last crates.io release 0.6.1 (June 2023) with no new releases since — GitHub shows some activity, but release cadence is very low. `hnswlib-rs`: pure Rust port of hnswlib (MIT/Apache-2.0). | **Possible but low ROI** | Custom HNSW is tightly coupled to vindex mmap format and gate-vector semantics. Adapter layer cost likely exceeds ~300 LOC custom code. Keep custom unless HNSW correctness/perf gaps emerge at 31B+ model scale. If replacing, prefer `usearch` (actively maintained) over `instant-distance`. |
| **Transformer inference (dense path)** | `larql-inference` — full custom forward pass | `candle` (HF, Apache-2.0), `burn` (tracel-ai, Apache-2.0) | Apache-2.0 | `candle`: production-grade, supports llama.cpp quantized types (Q4K family) via GGUF, GQA, MoE. `burn`: general ML framework with 4-bit PTQ. Both assume dense tensor forward passes. | **No** | The *sparse FFN retrieval orchestration* via gate-vector KNN (WalkFfn) is the central research invention — candle/burn have no concept of this. The dense-kernel substrate (matmul, attention) could in principle be borrowed from candle, but the retrofit cost to a research substrate exceeds the gain. The inference architecture IS the research. Keep custom. |
| **Quantized CPU matmul kernels** | `larql-compute` — C kernels for Q4K/Q6K/Q8K (ARM NEON), BLAS f32 | llama.cpp kernels (MIT), `ggml-rs` bindings | MIT | llama.cpp's `ggml_vec_dot_q4_K_q8_K` family are the reference Q4K kernels, battle-tested and faster (0.95 ms vs comparable on M3 Max). LARQL's kernels are a reimplementation of the same approach. | **Borrow, don't replace** | Consider adapting llama.cpp's Q4K/Q6K dot product C functions directly (MIT licensed). The kernel math is identical; the only difference is call convention. This would reduce maintenance of LARQL-specific C kernel code and gain any llama.cpp kernel improvements for free. |
| **Metal GPU shaders** | `larql-compute-metal` — custom MSL shaders | llama.cpp Metal kernels (MIT) | MIT | llama.cpp has `kernel_flash_attn_ext_vec_reduce` (flash attention) and `kernel_mul_mm_*` (`simdgroup_matrix` prefill matmul) — both identified as gaps in `docs/llama-cpp-comparison.md`. | **Adapt gap kernels** | For flash attention (D-ATTN-MTG) and simdgroup_matrix prefill (D-PREFILL-MM2), study or adapt llama.cpp's MIT-licensed MSL shaders rather than designing from scratch. The doc already identifies exactly which kernels to study. |
| **LQL query language parser** | `larql-lql` — hand-written lexer + recursive descent parser | `sqlparser-rs` 0.62.0 / `datafusion-sqlparser-rs` (Apache-2.0, 65M downloads), `pest` 2.7 (MIT/Apache-2.0), `nom` 7.1 (MIT) | MIT/Apache-2.0 | `sqlparser-rs` 0.62.0 (May 2026, 86 versions, 65M downloads) is mature SQL AST with extensible Dialect trait (161 methods). `pest` (PEG grammar files) suits divergent custom grammars. `nom` (combinator) for hand-rolled control. | **Optional migration** | Hand-written parser works fine for a research project. If LQL syntax stays SQL-like, `sqlparser-rs` is the lowest-risk foundation (ASF-governed). If LQL syntax diverges (WALK/DESCRIBE/INFER semantics), `pest` gives a cleaner formal grammar than hand-written recursive descent. |
| **REPL** | `larql-lql/repl.rs` — uses `rustyline` | `rustyline` 15 (MIT) | MIT | Already using the right crate (rustyline = readline-compatible, cross-platform). | **Already OSS** | No change needed. |
| **KV cache engines** | `larql-kv` — Standard, MarkovResidual, TurboQuant, UnlimitedContext, BoundaryKV, Apollo | None equivalent | — | No OSS project implements markov-residual KV compression, turbo-quant, or boundary-gated KV. | **Keep custom** | These ARE the research. Not replaceable. |
| **Residual boundary codec** | `larql-boundary` — int8/bf16 compression with confidence gate | None equivalent | — | Novel: Exp 43/44, confidence-gated top-1 preservation contract. No OSS equivalent. | **Keep custom** | Novel research. |
| **Quantum / n-qubit formalism** | `larql-hilbert` — LIMLL, n-qubit, Born rule, NQubitLM | `qrusty` 0.1, `quantrs` 0.1, `spinach` | Various | Existing quantum crates target circuit simulation (Qiskit-equivalents). LARQL's formalism (LIMLL, admissibility bounds, Σ⁰₁/Π⁰₂ query shapes) is tied to the Rosko 2025 arXiv formalism. | **Keep custom** | Original research. Existing crates don't implement LIMLL or the transformer↔quantum bridge. |
| **WASM compute / expert dispatch** | `model-compute` + `larql-experts` (20 cdylib modules) | `wasmtime` (Apache-2.0) already used | Apache-2.0 | Pattern (fuel-capped WASM sandbox + hot-swap experts) is novel. wasmtime is the correct foundation. | **Already OSS** | wasmtime is correct. The expert pattern is novel. |
| **Model loading (safetensors)** | `larql-models` — architecture configs, safetensors deserialization | `safetensors` 0.7 (Apache-2.0) already used | Apache-2.0 | Already using huggingface/safetensors. | **Already OSS** | No change needed. |
| **HuggingFace Hub model pull** | `larql-vindex` (pull cmd) | `hf-hub` 0.5 (Apache-2.0) already used | Apache-2.0 | Already using hf-hub. | **Already OSS** | No change needed. |
| **Tokenization** | `larql-inference/src/tokenizer.rs` — thin wrapper | `tokenizers` 0.21 (Apache-2.0) already used | Apache-2.0 | HuggingFace tokenizers. | **Already OSS** | No change needed. |
| **gRPC protocol** | `larql-router-protocol` | `tonic` 0.13 + `prost` 0.13 (MIT/Apache) already used | MIT/Apache | Standard gRPC stack for Rust. | **Already OSS** | No change needed. |
| **HTTP server** | `larql-server` | `axum` 0.8 (MIT) already used | MIT | Best-in-class async HTTP for Rust. | **Already OSS** | No change needed. |
| **QUIC transport** | `larql-router-protocol` (optional) | `quinn` 0.11 (MIT/Apache) already used | MIT/Apache | Production QUIC for Rust. | **Already OSS** | No change needed. |
| **Python bindings** | `larql-python` | `pyo3` 0.24 + `numpy` (MIT/Apache) already used | MIT/Apache | Best-in-class Python↔Rust bridge. | **Already OSS** | No change needed. |
| **Metrics** | `larql-router` | `prometheus` 0.13 (Apache-2.0) already used | Apache-2.0 | Standard. | **Already OSS** | No change needed. |
| **OpenAPI docs** | `larql-server` | `utoipa` 5 (MIT/Apache) already used | MIT/Apache | Good for Rust axum integration. | **Already OSS** | No change needed. |

---

## 4. Where LARQL Is Genuinely Reinventing the Wheel

### High Priority — Immediate OSS Wins

**A. Graph algorithms → petgraph**

`larql-core` implements BFS/DFS traversal, Dijkstra, A\*, PageRank, connected components, merge — all of which are in `petgraph` 0.8.3 (MIT/Apache-2.0). Verified on docs.rs 2026: petgraph includes `petgraph::algo::page_rank` (damping factor, iterative, convergence). petgraph is used by rustc and cargo; it is not going anywhere.

The real gap: larql-core's edge model is confidence-weighted, typed, and carries JSON/msgpack IO — it would need a thin wrapper layer over petgraph's `Graph<N, E, Directed>`. The algorithmic core (~500 LOC in `algo/`) could largely be deleted.

**Estimate**: ~2–3 days of refactor to adopt petgraph as graph backing store. Reduces algorithmic maintenance surface and gains better-tested traversal primitives.

---

**B. Quantized CPU kernels — consider borrowing from llama.cpp**

The Q4K/Q6K/Q8K matmul kernels in `larql-compute/src/cpu/ops/` implement the same ggml super-block format as llama.cpp and use the same `vdotq_s32` ARM NEON intrinsics. They are not inferior — they're a clean reimplementation. But:
- llama.cpp (MIT) already has the reference implementations
- Any improvements to llama.cpp's kernel (e.g. stride-32 variants, AArch64 assembly) could be borrowed
- Maintaining C kernel files in Rust workspaces is friction

**Estimate**: Optional. The kernels are stable and correct. Evaluate if llama.cpp's kernel evolution diverges significantly from LARQL's.

---

**C. Metal flash attention — borrow llama.cpp's MSL shader**

This is already identified in `docs/llama-cpp-comparison.md`. The llama.cpp `kernel_flash_attn_ext_vec_reduce` shader (MIT licensed) could be studied or adapted as the basis for D-ATTN-MTG. Writing it from scratch risks repeating the TG-count regression of the 2026-05-01 attempt.

**Estimate**: D-ATTN-MTG already planned. Suggest using llama.cpp's flash-attention shader as the design reference and adapting to LARQL's Metal buffer conventions.

---

### Medium Priority — Consider on Growth

**D. LQL parser → pest**

The hand-written lexer (`lexer.rs`) and recursive-descent parser (`parser/`) are currently maintainable. But as LQL adds features (complex WALK expressions, subqueries, piping), a formal grammar in `pest` would be cleaner. No urgency while LQL's syntax is stable.

**Estimate**: ~3–5 days if LQL grammar grows to need it. Low priority now.

---

**E. HNSW → instant-distance (long-term)**

The custom HNSW (`larql-vindex/src/index/compute/hnsw.rs`, ~300 LOC) is minimal and coupled to the gate-vector format. If performance or correctness gaps emerge (e.g. for larger models with many more gate vectors), `usearch` (Apache-2.0, C++ core with Rust bindings, actively maintained by Unum, version 2.25.2 released 2026-05-02) is the preferred replacement — production-grade and battle-tested. `instant-distance` (MIT, 0.6.1) is pure-Rust but has had no new crates.io release since June 2023; prefer usearch.

**Estimate**: Low priority unless HNSW becomes a bottleneck.

---

### No-change (Keep Custom — Novel Research)

The following are **correctly custom** and should not be replaced:

| Subsystem | Why it stays custom |
|---|---|
| `larql-inference` + WalkFfn | IS the research thesis. Dense-path frameworks (candle, burn) can't express it. |
| `larql-kv` engines | Markov-residual, TurboQuant, BoundaryKV, Apollo are the KV research track. |
| `larql-boundary` | Novel confidence-gated codec, calibrated per-model. |
| `larql-hilbert` | LIMLL + n-qubit formalism ties to arXiv:2511.21296. No OSS equivalent. |
| `larql-vindex` (format layer) | The vindex model-as-database concept is the core invention. |
| `larql-experts` | WASM expert dispatch pattern is novel; wasmtime is correctly the OSS foundation. |
| `larql-router-protocol` | gRPC spec atop tonic/prost — correct OSS use, novel protocol design. |

---

## 5. What's Already Using Best-in-Class OSS

LARQL's server and protocol layers make excellent OSS choices throughout:

| Area | Crate Used | License | Notes |
|---|---|---|---|
| Async runtime | `tokio` | MIT | Correct |
| HTTP server | `axum` | MIT | Correct |
| gRPC | `tonic` + `prost` | MIT/Apache | Correct |
| QUIC | `quinn` | MIT/Apache | Correct |
| TLS | `rustls` | MIT/Apache | Correct |
| Metrics | `prometheus` | Apache-2.0 | Correct |
| OpenAPI | `utoipa` | MIT/Apache | Correct |
| Python FFI | `pyo3` + `numpy` | MIT/Apache | Correct |
| Safetensors | `safetensors` | Apache-2.0 | Correct |
| Tokenization | `tokenizers` | Apache-2.0 | Correct |
| HuggingFace Hub | `hf-hub` | Apache-2.0 | Correct |
| WASM runtime | `wasmtime` | Apache-2.0 | Correct |
| REPL | `rustyline` | MIT | Correct |
| Dense f32 matmul | OpenBLAS / Accelerate | BSD/Apple | Correct |
| msgpack | `rmp-serde` | MIT | Correct |
| Template engine | `minijinja` | Apache-2.0 | Correct |

---

## 6. Priority Action Items

| Priority | Action | Est. Effort | Impact |
|---|---|---|---|
| **P1** | Adopt `petgraph` 0.8.3 as graph backing store in `larql-core`; petgraph covers ALL standard algorithms including PageRank, so keep only LARQL-specific domain ops (merge/diff/walk + typed confidence-edge IO) | 2–3 days | Removes ~500 LOC algorithmic maintenance. petgraph has battle-tested traversal + PageRank. |
| **P2** | Use llama.cpp's flash-attention MSL shader as design reference for D-ATTN-MTG | 0 extra effort | Avoids repeating the TG-count regression. Direct the D-ATTN-MTG session to start by reading llama.cpp's `kernel_flash_attn_ext_vec_reduce`. |
| **P3** | Evaluate llama.cpp Q4K kernel (MIT) and candle's `k_quants.rs` (MIT/Apache-2.0) for parity with LARQL's C kernels | 1 day audit | candle has SIMD-dispatched AVX2/NEON/WASM Q4K vec_dot (confirmed 2026). If LARQL's kernels lag in stride or SIMD variant coverage, borrow the kernel math from either source. |
| **P4** | Consider `pest` grammar if LQL syntax grows significantly | 3–5 days (future) | Cleaner grammar evolution. Not urgent while LQL is stable. |
| **P5** | Re-evaluate HNSW if gate-vector count scales to 31B+ models | 2–3 days (future) | `instant-distance` as drop-in if custom HNSW becomes a bottleneck. |

---

## 7. Sub-project Extraction Roadmap

The AGENTS.md explicitly notes which crates are designed to be extracted from the LARQL monorepo:

| Crate | Status | Extraction condition |
|---|---|---|
| `model-compute` | **Extraction planned** (noted in AGENTS.md) | "extract to sibling repo once interface stabilises" |
| `larql-vindex-spec` | **Extract-ready today** — zero `larql-*` deps | Could be published independently as the vindex format contract |
| `larql-boundary` | Extract-ready (no vindex/inference deps) | Could be a standalone codec library |
| `larql-hilbert` | Extract-ready (only `ndarray` + `num-complex`) | QLM/LIMLL formalism could be its own crate |
| `larql-core` | Extractable if petgraph adopted | Generic knowledge-graph crate, useful without LARQL |

**Not extractable** (deeply coupled to LARQL's vindex/inference architecture): `larql-compute`, `larql-compute-metal`, `larql-inference`, `larql-kv`, `larql-vindex`, `larql-lql`, `larql-server`, `larql-router`.

---

## 8. Summary

LARQL is a well-designed research substrate. Most of its "DIY" choices are **justified** because the core inventions (vindex, WalkFfn/sparse FFN retrieval, novel KV engines, boundary codec, quantum formalism) have no OSS equivalent — they are the research. The server/protocol/tokenization/serialization stack already uses excellent OSS throughout.

The one area of genuine wheel-reinvention is `larql-core`'s graph algorithms. `petgraph` covers the algorithmic core and is battle-tested; adopting it would reduce maintenance surface without sacrificing any novel functionality.

The quantized kernel and Metal shader work are correctly custom but could borrow more directly from llama.cpp's MIT-licensed code for the identified gaps (flash attention, simdgroup_matrix prefill).

Everything else stays custom for good reason.
