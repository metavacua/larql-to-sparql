# larql-vindex

The queryable model format. Decompile, browse, edit, and recompile neural networks.

## What is a Vindex?

A vindex (vector index) is a directory containing a transformer model's weights reorganised for queryability. The model IS the database — each weight matrix is stored once in its optimal format.

```rust
use larql_vindex::*;

// Load (readonly base)
let index = VectorIndex::load_vindex(&path, &mut SilentLoadCallbacks)?;
let mut patched = PatchedVindex::new(index);

// Query — which features fire for "France"?
let hits = patched.gate_knn(layer, &query, 10);  // 2.7ms/layer at full dim

// Walk — multi-layer feature scan
let trace = patched.walk(&query, &layers, 10);

// Mutate via patch overlay (base files never modified)
patched.insert_feature(layer, feature, gate_vec, meta);
patched.set_down_vector(layer, feature, down_vec);

// Apply a saved patch
let patch = VindexPatch::load("medical.vlp")?;
patched.apply_patch(patch);

// Bake patches into a new clean VectorIndex (in-memory)
let baked = patched.bake_down();
baked.save_vindex(&output_path, &mut config)?;

// Or bake the constellation into the canonical down_weights.bin
// via COMPILE INTO VINDEX (see larql-lql) — produces a real
// standalone vindex with no overlay needed at load time, and the
// inserted facts survive a future COMPILE INTO MODEL safetensors
// export because the bytes are sitting in the standard
// down_proj tensors that the manifest references.
```

### Layering note: gate vs down overrides

`PatchedVindex` stores the two kinds of override in different places:

- **Gate vectors** (`insert_feature`, `update_feature_meta`) live in
  `overrides_gate` / `overrides_meta` on the patch overlay. The
  `gate_vectors.bin` on disk is never touched.
- **Down vectors** (`set_down_vector`) are forwarded to the underlying
  base index's `down_overrides` HashMap. The `down_weights.bin` on
  disk is never touched at runtime.

This asymmetry is intentional and load-bearing for `COMPILE INTO
VINDEX`. The dense FFN inference path reads gate scores from
`gate_vectors.bin`; baking norm-matched override gates there would
produce moderate dense activations that combined with the override
down vectors would blow up the residual stream. Keeping the source's
weak free-slot gate at the inserted index keeps the dense activation
small, so `small_activation × poseidon_vector` per layer accumulates
into the validated multi-layer constellation effect. See
`patch/core.rs` for the full doc on `PatchedVindex`.

### Refine pass (`patch/refine.rs`)

`refine_gates(inputs, decoy_residuals) -> RefineResult` orthogonalises
each patched gate against the other patched gates at the same layer,
plus any decoy residuals supplied by the caller. Pure Gram-Schmidt over
`Array1<f32>` slices — no model dependency, no forward passes. The
result carries the refined gates plus per-fact `retained_norm`
statistics.

This is the load-bearing fix for cross-fact bleed and is called by
INSERT's batch refine pass at install time.
Refining is per-layer (facts at different layers can't interfere
through the FFN math). Decoy residuals are layer-scoped — the caller is
responsible for capturing them at the correct depth, which is exactly
what `larql_inference::capture_decoy_residuals` does. Validated against
synthetic constellations by the unit tests in `patch/refine.rs`; the
end-to-end Gemma 3 4B reproduction lives in
`larql-lql/examples/refine_demo.rs`.

## The Headline

A 1T model in 10.9 GB on a laptop.

```
Model          Full Inference RAM    Vindex Infer RAM    Ratio
Gemma 3 4B              7 GB              1.3 GB          5x
Llama 3 8B             15 GB              2.2 GB          7x
Llama 3 70B           130 GB              4.9 GB         27x
Llama 3 405B          754 GB              8.6 GB         88x
DeepSeek V3          1250 GB             10.9 GB        115x
Kimi-K2              1863 GB             10.9 GB        171x
```

Vindex inference uses mmap: only 1 layer of gate vectors + 1 layer of attention
weights are resident at a time. The rest stays on disk until touched.

## Features

- **Extract** from safetensors, GGUF, or MLX models (streaming — no full model load)
- **Gate KNN** via BLAS matmul, Q4 matvec (CPU/Metal/CUDA), or HNSW approximate search
- **Walk** across all layers with down-meta annotation
- **Readonly base** — base vindex files are never modified after extraction
- **Patch overlay** — all mutations go through PatchedVindex (INSERT/DELETE/UPDATE)
- **Patches** — stackable, reversible knowledge diffs (.vlp files)
- **Vindexfile** — declarative model builds (FROM + PATCH + INSERT, like Dockerfile)
- **HuggingFace Hub** — download and publish vindexes (`hf://user/repo` URI scheme)
- **Split weight files** — gate, up, down, attn, norms, lm_head (no duplication)
- **Zero-copy mmap** — gate vectors sliced directly from disk, no heap allocation
- **Binary down_meta** — compact binary format (no JSONL)
- **f16 storage** — halves file sizes with negligible accuracy loss
- **MoE support** — Mixtral, DeepSeek (experts as contiguous features)
- **Layer bands** — per-family boundaries (Gemma, Llama, Qwen, etc.)
- **Checksums** — SHA256 integrity verification for all binary files
- **Provenance** — model source, timestamp, version tracking
- **LM head KNN** — top-K token lookup via single BLAS gemv against output projection
- **Adaptive residency** — pin hot layers in memory, stream cold ones. More memory = faster. Smooth gradient vs llama.cpp's all-or-nothing cliff

## Crate Structure

```
larql-vindex/src/
├── lib.rs                      Crate root + re-exports
├── error.rs                    VindexError
├── describe.rs                 DescribeEdge, LabelSource
├── mmap_util.rs                madvise-optimized mmap helper
│
├── config/                     Configuration types
│   ├── types.rs                VindexConfig, ExtractLevel, LayerBands, MoeConfig
│   └── dtype.rs                StorageDtype (f32/f16), encode/decode/write_floats
│
├── index/                      In-memory KNN engine (zero-copy mmap)
│   ├── types.rs                FeatureMeta, GateIndex trait, WalkHit, WalkTrace
│   ├── core.rs                 VectorIndex struct + Clone + constructors (new, new_mmap)
│   ├── loaders.rs              load_gates, load_down_meta (NDJSON readers)
│   ├── gate.rs                 Gate KNN dispatch (brute-force, batched, HNSW, Q4)
│   ├── gate_trait.rs           impl GateIndex for VectorIndex
│   ├── accessors.rs            feature_meta, gate_vector(s), warmup, total_*
│   ├── walk.rs                 Feature-major down/up vectors, interleaved, Q4
│   ├── attn.rs                 Attention weight loaders (Q8, Q4_K, Q4)
│   ├── lm_head.rs              LM-head loaders + KNN (f32 + Q4)
│   ├── hnsw.rs                 HNSW graph index (random projection, exact rescoring)
│   ├── mutate.rs               set/delete features, save to disk
│   ├── router.rs               MoE expert router
│   └── residency.rs            Adaptive layer pinning (memory budget → performance)
│
├── format/                     Vindex file I/O
│   ├── load.rs                 load_vindex, load_embeddings, load_tokenizer
│   ├── down_meta.rs            Binary down_meta read/write
│   ├── weights/
│   │   ├── mod.rs              Re-exports
│   │   ├── write.rs            write_model_weights, WeightSource, StreamingWeights
│   │   └── load.rs             load_model_weights, find_tokenizer_path
│   ├── checksums.rs            SHA256 computation + verification
│   ├── huggingface.rs          HuggingFace Hub download/publish
│   └── quant/mod.rs            Re-exports from larql_models::quant
│
├── extract/                    Build pipeline (model → vindex)
│   ├── build.rs                build_vindex coordinator + BuildContext + 6 stages
│   ├── build_helpers.rs        chrono_now, build_whole_word_vocab,
│   │                           compute_gate_top_tokens, compute_offset_direction,
│   │                           run_clustering_pipeline, ClusterData
│   ├── streaming.rs            Streaming extraction (mmap, no full model load)
│   ├── callbacks.rs            IndexBuildCallbacks trait
│   └── build_from_vectors.rs   Build from pre-extracted NDJSON
│
├── patch/                      Patch system
│   ├── format.rs               VindexPatch, PatchOp, PatchDownMeta + base64
│   ├── overlay.rs              PatchedVindex (queries, mutators, walk, bake_down)
│   ├── overlay_apply.rs        apply_patch, remove_patch, rebuild_overrides
│   ├── overlay_gate_trait.rs   impl GateIndex for PatchedVindex
│   ├── knn_store.rs            L0 KnnStore (arch-B residual-key KNN)
│   ├── knn_store_io.rs         KnnStore .lknn save / load (f16 keys)
│   └── refine.rs               Gate refine pass (Gram-Schmidt orthogonalisation
│                               of patched gates + optional decoy residuals)
│
├── storage/                    Storage engine + L2 MEMIT cycles
│   ├── engine.rs               StorageEngine (PatchedVindex + epoch + memit_store)
│   ├── epoch.rs                Monotonic mutation counter
│   ├── status.rs               CompactStatus snapshot
│   └── memit_store.rs          MemitStore + MemitFact + memit_solve +
│                               MemitSolveResult (vanilla closed-form, BLAS-batched)
│
├── clustering/                 Relation discovery
│   ├── kmeans.rs               k-means clustering (BLAS via larql-compute)
│   ├── labeling.rs             Pattern detection, TF-IDF labels
│   ├── categories.rs           Entity category word lists
│   ├── pair_matching/
│   │   ├── mod.rs              Re-exports
│   │   ├── database.rs         RelationDatabase + Wikidata/WordNet loaders
│   │   └── labeling.rs         label_clusters_from_pairs / _from_outputs
│   └── probe.rs                Probe label loading
│
└── vindexfile/                 Declarative model builds
    ├── mod.rs                  Build executor (FROM + PATCH + INSERT → bake_down)
    └── parser.rs               Vindexfile parser (FROM, PATCH, INSERT, DELETE, etc.)
```

All matrix operations go through `larql-compute` (BLAS on CPU, Metal GPU planned for gate KNN).

## MEMIT decomposition (`storage/memit_store.rs`)

`memit_solve` is the vanilla closed-form MEMIT decomposition that
populates `MemitStore` during `COMPACT MAJOR`. It wraps the generic
`larql_compute::cpu::ops::linalg::ridge_decomposition_solve` with the
MEMIT interpretation:

```rust
use larql_vindex::{memit_solve, MemitFact, MemitStore};

let solve = memit_solve(&keys, &targets, lambda)?;
//   solve.delta_w           — (d, d) weight update
//   solve.decomposed[i]     — ΔW @ k_i   (one row per fact)
//   solve.reconstruction_cos[i] — cos(ΔW k_i, t_i)
//   solve.max_off_diagonal  — cross-template interference
//   solve.frobenius_norm    — ‖ΔW‖_F

let facts: Vec<MemitFact> = /* package decomposed pairs */;
store.add_cycle(layer, facts, solve.frobenius_norm,
                min_cos, solve.max_off_diagonal);
```

This is **vanilla** MEMIT — no covariance whitening. Cross-template
bleed grows with N when keys share a dominant direction (the canonical-
form template case from exp 8). For production weight edits with C⁻¹
whitening + per-fact optimised target deltas (the validated v11 200/200
pipeline), use `larql-inference::forward::memit`.

| Run | Command |
|-----|---------|
| Demo | `cargo run --release -p larql-vindex --example demo_memit_solve` |
| Bench | `cargo bench -p larql-vindex --bench memit_solve` |

## Compute Integration

| Module | Operation | Backend |
|--------|-----------|---------|
| gate.rs | Gate KNN f32 (matmul_transb) | CPU BLAS |
| gate.rs | Gate KNN Q4 (q4_matvec) | Any ComputeBackend |
| gate.rs | Adaptive KNN (pinned → Q4 → f32) | Any ComputeBackend |
| gate.rs | Gate walk (gemv) | CPU BLAS |
| gate.rs | Batch gate scores (matmul_transb) | CPU BLAS |
| hnsw.rs | Random projection (matmul) | CPU BLAS |
| hnsw.rs | Dot product (graph traversal) | CPU BLAS |
| walk.rs | LM head KNN (matmul_transb) | CPU BLAS |
| kmeans.rs | Similarity matrix (matmul_transb) | CPU BLAS |
| router.rs | MoE routing (matmul) | CPU BLAS |

## Supported Architectures

| Family | Models | FFN Type | Notes |
|--------|--------|----------|-------|
| Gemma 4 | Gemma 4 31B/E2B | Gated (GeGLU) | Per-layer head_dim, K=V, V-norm, partial RoPE, PLE, KV sharing |
| Gemma 3 | Gemma 3 (4B-27B) | Gated (GeGLU) | QK-norm, sliding window, dual RoPE |
| Gemma 2 | Gemma 2 (2B-27B) | Gated (GeGLU) | Softcapping, QK-norm |
| Llama | Llama 2/3 (7B-405B) | Gated (SiLU) | GQA, RoPE scaling |
| Mistral | Mistral 7B | Gated (SiLU) | Sliding window |
| Mixtral | Mixtral 8x7B/8x22B | MoE (8 experts) | PerExpert format |
| Qwen | Qwen 2/2.5/3 | Gated (SiLU) | Attention bias, QK-norm |
| Phi | Phi 2/3 | Gated | |
| DeepSeek | DeepSeek V2/V3 | MoE (shared + routed) | MLA, YaRN |
| Granite | Granite | Gated (SiLU) | Scaling multipliers |
| StarCoder2 | StarCoder2 | Standard (GELU) | LayerNorm, bias, non-gated FFN |
| GPT-OSS | GPT-OSS | MoE (PackedMxfp4) | MXFP4 packed experts |
| GPT-2 | GPT-2 | Dense (GELU) | |

## File Layout

```
model.vindex/
├── gate_vectors.bin        W_gate per layer (f32/f16 KNN index)
├── gate_vectors_q4.bin     W_gate Q4_0 (7x smaller, for Q4 KNN)
├── embeddings.bin          W_embed matrix
├── down_meta.bin           Per-feature output metadata (binary)
├── attn_weights.bin        Q, K, V, O per layer
├── up_weights.bin          W_up per layer
├── down_weights.bin        W_down per layer
├── norms.bin               LayerNorm parameters
├── lm_head.bin             Output projection
├── interleaved.bin         gate|up|down packed per layer (optional)
├── interleaved_q4.bin      Q4_0 quantized version (optional, 7x smaller)
├── interleaved_q4k.bin     Q4_K gate/up + Q6_K down (when quant=q4k)
├── interleaved_q4k_manifest.json  Per-tensor offsets for interleaved_q4k.bin
├── attn_weights_q4k.bin    Q4_K Q/K/O + Q6_K V (when quant=q4k)
├── attn_weights_q4k_manifest.json Per-tensor offsets for attn_weights_q4k.bin
├── ple_weights.bin         Per-Layer Embedding tensors at f16 (Gemma 4 E2B only)
├── index.json              Config, layer bands, provenance, checksums, quant format
├── tokenizer.json          Tokenizer
├── relation_clusters.json  Discovered relation types
├── feature_labels.json     Probe-confirmed labels
└── weight_manifest.json    Weight file → offset mapping
```

## Extract Levels

| Level | Size (f16) | Enables |
|-------|-----------|---------|
| Browse | ~3 GB | DESCRIBE, WALK, SELECT |
| Inference | ~6 GB | + INFER |
| All | ~8.5 GB | + COMPILE |

## Streaming Quantisation (`--quant q4k`)

`build_vindex_streaming` can quantise model weights inline as it reads
the safetensors shards, skipping the f32 intermediate entirely. Pass
`QuantFormat::Q4k` (or `--quant q4k` on the CLI) to emit Ollama-
compatible blocks:

- Q/K/O/gate/up → Q4_K (148 bytes per 256 values)
- V/down → Q6_K (210 bytes per 256 values)

Output files: `attn_weights_q4k.bin` + `interleaved_q4k.bin` with
per-tensor manifests. `VindexConfig.quant = Q4k` in `index.json` so
loaders can dispatch on config.

When `quant != None`, `--level browse` is implicitly promoted to
`--level all` — the Q4_K writer emits all of attention, FFN, norms,
and `lm_head` in one pass, and a browse-only Q4k vindex would be
incoherent.

### Per-Layer Embeddings (Gemma 4 E2B)

E2B's Per-Layer Embedding tensors don't go through Q4_K because the
per-super-block (d, dmin) calibration destroys embedding-style tensors
— one outlier row per super-block pulls the scale, zeroing the other
255 cells. The noise then compounds across 35 layers' additive PLE
contributions. Instead they land in `ple_weights.bin` at **f16**:

- `per_layer_model_projection.weight`  (~27 MB at f16)
- `embed_tokens_per_layer.weight`      (~4.7 GB at f16 on E2B)
- `layers.N.per_layer_input_gate.weight` + `per_layer_projection.weight`

Load dequantises to f32 at mmap time and inserts into `weights.tensors`.
`larql_inference::forward::ple::precompute_per_layer_inputs` and
`apply_per_layer_embedding` then work unchanged.

### E2B caveats worth knowing

- **Cross-layer KV sharing** (`num_kv_shared_layers=20`): layers 15-34
  reuse K/V computed by the last unshared sliding / global layer. The
  Q4 forward path threads a `kv_cache` through the loop to honour this.
- **Double-wide MLP** (`use_double_wide_mlp=True`): half the layers
  ship with `intermediate=12288` while the model-wide config reports
  6144. `VectorIndex::num_features(layer)` is the authoritative
  per-layer FFN width; don't read `weights.intermediate_size` in any
  dequant / forward code.
- **Final-logit softcap** (`final_logit_softcapping=30.0`): preserved
  through `VindexModelConfig.final_logit_softcapping`. Missing it lets
  `logits_to_predictions` peak on the wrong token — there is no "fail
  loudly" mode for a dropped softcap, only a silent accuracy hit.

## Testing

```bash
cargo test -p larql-vindex                                                      # 106 tests (lib + 1 integration + doc)

# Demos (synthetic fixtures, no model download needed)
cargo run -p larql-vindex --example demo_features                               # Feature showcase (build, KNN, patches, MoE, f16)
cargo run --release -p larql-vindex --example mmap_demo                         # mmap RAM behaviour + scaling table
cargo run --release -p larql-vindex --example q4k_demo                          # Streaming Q4_K showcase: size comparison, file layout, dequant round-trip
cargo run --release -p larql-vindex --example demo_memit_solve                  # MEMIT closed-form decomposition + MemitStore round-trip

# Criterion benches (run with --quick for a fast sweep, omit for full sample)
cargo bench  -p larql-vindex --bench vindex_ops                                 # KNN, walk, save/load, mutate, MoE
cargo bench  -p larql-vindex --bench vindex_scaling                             # Production dims (CPU)
cargo bench  -p larql-vindex --features metal --bench vindex_scaling            # Production dims (Metal)
cargo bench  -p larql-vindex --bench memit_solve                                # Ridge decomposition throughput
cargo bench  -p larql-vindex --bench extract_throughput                         # Streaming extract: f32 vs Q4K write-path time
cargo bench  -p larql-vindex --bench q4k_vs_f32                                 # Per-layer attn retrieval: mmap memcpy vs mmap + dequant

# Streaming build (one-shot, skips f32 intermediate)
larql extract-index <model> -o <vindex> --quant q4k                             # Q4_K/Q6_K attn + FFN + norms + lm_head in one pass

# Multi-tier build pipeline (post-hoc, uses larql-compute quantizers on an
# already-extracted f32 vindex — kept for backwards compatibility)
cargo run --release -p larql-vindex --example build_q4k_weights -- <vindex>     # Q4_K/Q6_K attn + FFN
cargo run --release -p larql-vindex --example build_attn_q8 -- <vindex>         # Q8 attention (fallback)
cargo run --release -p larql-vindex --example build_interleaved -- <vindex>     # Pack gate|up|down
cargo run --release -p larql-vindex --example build_down_features -- <vindex>   # Feature-major transpose
cargo run --release -p larql-vindex --example build_up_features -- <vindex>     # f16 → f32 decode
cargo run --release -p larql-vindex --example build_gate_q4 -- <vindex>         # Q4 gate vectors
cargo run --release -p larql-vindex --example build_lm_head_q4 -- <vindex>      # Q4 logits projection
```

### Bench measurements (typical machine, synthetic Gemma-like fixture)

| Bench | Operation | Time |
|---|---|---|
| `extract_throughput` | streaming extract, f32 | ~37 ms |
| `extract_throughput` | streaming extract, **Q4K** | ~22 ms (1.67× faster; output is ~3× smaller so disk I/O dominates) |
| `q4k_vs_f32` | f32 per-layer Q retrieval (mmap → Vec<f32>) | ~880 µs |
| `q4k_vs_f32` | **Q4K** per-layer Q retrieval (mmap → dequant → Vec<f32>) | ~3.3 ms (3.7× slower per-layer to save 6.26× on disk) |

Test coverage (104 tests):
- Construction, dimensions, layer counts, feature counts
- Gate KNN: brute-force, f32, Q4 via compute backend, top-K ordering
- Gate walk: BLAS gemv path matches brute-force KNN
- Walk: multi-layer tracing, metadata annotation
- LM head KNN: top-K token lookup via matmul_transb
- HNSW: enable/disable, integration with VectorIndex, valid results
- Q4 gate: load round-trip, data slice correctness, Q4 vs f32 top-1 match
- Mutation: set gate vectors, metadata, patch overlay
- Patching: apply, revert, bake down
- Binary serialization: checksums, dtype, config
- MoE: expert-scoped queries, multiple experts per layer
- Streaming extraction: safetensors mmap, one layer at a time
- Adaptive residency: pin/evict, budget enforcement, auto_pin, pin_range, adaptive dispatch

## Benchmarks

Criterion benches live in `benches/`. Run with `cargo bench -p
larql-vindex` (full sample) or `-- --quick` (5-iter sweep). HTML
reports go to `target/criterion/`.

### Core operations (`benches/vindex_ops.rs`, M3 Max, synthetic dims)

| Operation | Time |
|---|---|
| `gate_knn_per_layer / 1024f×256h` | **24 µs** |
| `gate_knn_per_layer / 4096f×512h` | 445 µs |
| `gate_knn_per_layer / 10240f×2560h` (Gemma production) | **2.78 ms** |
| `walk_all_layers / 8L×1024f×256h` | 221 µs |
| `walk_all_layers / 8L×10240f×2560h` (8L Gemma band) | 22.7 ms |
| `feature_meta_lookup` (per call) | ~245 ns |
| `mutate / set_meta_plus_gate` | 301 ns |
| `save_load / save_gate_vectors` | 2.01 ms |
| `save_load / save_down_meta` | 462 µs |
| `save_load / load_vindex` | 261 µs |
| `moe_scaling / 8x_experts` (vs 1x baseline) | 17.6× for 8× features (sub-linear) |

### Production dimensions (M3 Max, synthetic data)

| Model | Features | Hidden | f32 BLAS | Q4 CPU | Q4 Metal | Speedup | Walk 14L |
|---|---|---|---|---|---|---|---|
| Gemma 3 4B | 10,240 | 2,560 | 2.7ms | 0.96ms | **0.50ms** | 5x | 7.0ms |
| Llama 3 8B | 14,336 | 4,096 | 15.7ms | 2.1ms | **0.95ms** | 17x | 15.2ms |
| Llama 3 70B | 28,672 | 8,192 | 98.3ms | 8.2ms | **1.31ms** | **75x** | 63.1ms |

Vindex provides Q4 gate data. Compute crate scores it. Same interface, any backend.

### HNSW vs brute-force (dim=2560)

| Features | Brute | HNSW | Winner |
|---|---|---|---|
| 1,024 | 0.18ms | 0.14ms | HNSW |
| 4,096 | 2.3ms | 1.9ms | HNSW |
| 10,240 | 2.6ms | 1.7ms | HNSW |
| 28,672 | 18.8ms | 15.2ms | HNSW |

### Memory (mmap, 34L × 4096 × 2560)

| Metric | Value |
|---|---|
| Cold KNN (first access) | 0.39ms |
| Warm KNN (paged) | 0.37ms |
| Page fault overhead | 0.02ms |
| Zero-copy mmap | true (0 bytes heap) |

### Adaptive residency (simulated 70B, M3 Max Metal)

```
Budget    Pinned   KNN/layer   Walk 48L    tok/s
stream     0/80     0.28ms      13.4ms      75      ← 0 MB pinned
200 MB    14/80     0.28ms      13.4ms      75
500 MB    35/80     0.28ms      13.3ms      75
all       80/80     0.29ms      13.8ms      72      ← all pinned

llama.cpp 70B:
40GB VRAM  all                              8-12    ← needs ALL weights
24GB VRAM  partial                          2-3     ← PCIe cliff
CPU only                                    1-2
```

On unified memory (Apple Silicon), mmap is effectively pinned — the gradient
is flat because there's no PCIe bottleneck. On discrete GPU systems,
pinned layers skip PCIe transfers and the gradient steepens.

## Design Principles

1. **Readonly base** — binary files on disk are never modified after extraction
2. **Patch overlay** — all mutations via in-memory PatchedVindex
3. **Zero-copy mmap** — gate vectors are sliced from the file, not loaded to heap
4. **One file per matrix type** — gate, attn, up, down stored separately
5. **Streaming extraction** — processes one layer at a time (~2 GB peak for 120B models)
6. **All compute through larql-compute** — BLAS dispatch, no raw ndarray .dot() calls
7. **Adaptive residency** — pin hot layers in memory budget, stream cold ones from mmap
8. **Format-agnostic storage** — vindex stores raw quantized bytes, compute dequants at inference

## Documentation

| Doc | Content |
|-----|---------|
| [PERFORMANCE.md](PERFORMANCE.md) | Benchmark data, scaling projections, compute integration |
| [ROADMAP.md](ROADMAP.md) | Planned features, completed items |
| [docs/vindex-format.md](docs/vindex-format.md) | File format specification, directory layout, manifest schemas |
| [docs/compute-integration.md](docs/compute-integration.md) | How vindex stores data and compute consumes it |
| [docs/adr/001](docs/adr/001-weights-as-database.md) | Transformer weights as queryable database |
| [docs/adr/002](docs/adr/002-quantization-strategy.md) | Ollama-compatible Q4_K/Q6_K quantization |
| [docs/adr/003](docs/adr/003-mmap-zero-copy.md) | Mmap zero-copy architecture |
| [docs/adr/004](docs/adr/004-three-storage-tiers.md) | Three-tier weight storage (f32, Q8, Q4_K) |
| [docs/adr/005](docs/adr/005-patch-overlay.md) | Patch overlay for editable knowledge |
| [docs/adr/006](docs/adr/006-hnsw-index.md) | HNSW graph index for sub-linear KNN |
| [docs/adr/007](docs/adr/007-interleaved-layout.md) | Interleaved weight layout (TLB optimization) |
| [docs/adr/008](docs/adr/008-quantizer-source-of-truth.md) | Single source of truth for quantizers |

## Status

```
Tests:      146 passing (41 clustering + 7 HNSW + 98 main)
Warnings:   0 (build)
Formats:    f32, Q8_0, Q4_K, Q6_K, Q4_0
Models:     Gemma 2/3/4, Llama, Mistral, Mixtral, Qwen, Phi, DeepSeek, Granite, StarCoder2, GPT-OSS, GPT-2
```

## License

Apache-2.0
