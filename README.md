# LARQL

The model IS the database. Query neural network weights like a graph database. No GPU required.

LARQL decompiles transformer models into a queryable format called a **vindex** (vector index), then provides **LQL** (Lazarus Query Language) to browse, edit, and recompile the model's knowledge.

```sql
larql> USE "gemma3-4b.vindex";
Using: gemma3-4b.vindex (34 layers, 348.2K features, relations: 512 types)

larql> DESCRIBE "France";
France
  Edges (L14-27):
    capital     → Paris              1436.9  L27  (probe)
    language    → French               35.2  L24  (probe)
    continent   → Europe               14.4  L25  (probe)
    borders     → Spain                13.3  L18  (probe)

larql> INSERT INTO EDGES (entity, relation, target)
   ...   VALUES ("John Coyle", "lives-in", "Colchester");
Inserted 1 edge. Feature F8821@L26 allocated.

larql> INFER "The capital of France is" TOP 3;
  1. Paris                (97.91%)
  2. the                  (0.42%)
  3. a                    (0.31%)
```

## Quick Start

```bash
# Build
cargo build --release

# Pull a pre-built vindex from HuggingFace
larql pull hf://chrishayuk/gemma-3-4b-it-vindex

# List what's cached
larql list

# Run it — one-shot or chat
larql run gemma-3-4b-it-vindex "The capital of France is"
larql run gemma-3-4b-it-vindex          # drops into chat mode

# Or extract locally — inference-ready at f16 by default
larql extract google/gemma-3-4b-it -o gemma3-4b.vindex
larql run gemma3-4b.vindex "Einstein is known for"
```

`larql extract` defaults to `--level inference` (full local forward
pass) stored at f16. No flags needed for the common case.

<details>
<summary>Extract tiers and options</summary>

```bash
# Browse-only — gate KNN + embeddings, no forward pass (~3 GB for 4B)
larql extract google/gemma-3-4b-it -o gemma3-4b.vindex --level browse

# Attention-only — client-side slice for `run --ffn URL` (Act 2 demo)
larql extract google/gemma-3-4b-it -o gemma3-4b.attn.vindex --level attention

# Inference (default) — full local forward pass
larql extract google/gemma-3-4b-it -o gemma3-4b.vindex
larql extract google/gemma-3-4b-it -o gemma3-4b.vindex --level inference

# All — +lm_head +COMPILE extras (largest)
larql extract google/gemma-3-4b-it -o gemma3-4b.vindex --level all

# Q4_K/Q6_K inline (Ollama-compatible, smallest disk footprint)
larql extract google/gemma-3-4b-it -o gemma3-4b.vindex --quant q4k

# Maximum size reduction on Q4K — drop gate_vectors.bin, rebuild from
# interleaved_q4k.bin at load (~1.6 s cost on 4B, ~12 s on 31B)
larql extract google/gemma-3-4b-it -o gemma3-4b.vindex \
  --quant q4k --drop-gate-vectors

# Uniform Q4_K on FFN — gate + up + down all Q4_K (default stores
# down as Q6_K). ~30 MB/layer smaller, ~1.5–1.7× faster decode down
# matmul. Adds ~1.5 % softmax drift; top-1 / top-5 preserved.
larql extract google/gemma-4-31b-it -o gemma4-31b.vindex \
  --quant q4k --down-q4k

# Opt out of f16 (rarely wanted — doubles file sizes)
larql extract google/gemma-3-4b-it -o gemma3-4b.vindex --f32

# Convert from GGUF instead of extracting from safetensors
larql convert gguf-to-vindex model.gguf -o model.vindex
```

`extract-index` is kept as a backwards-compatible alias of `extract`.

</details>

### Serve it over HTTP + gRPC

```bash
larql serve gemma3-4b.vindex --port 8080
```

### Run attention locally, FFN on another machine

```bash
# Extract once, then carve deployment slices with `larql slice`.
# Either --preset or --parts a,b,c works; `--dry-run` previews.
larql extract google/gemma-4-31b-it -o gemma4-31b.vindex --quant q4k

# Client slice (7.4 GB for 31B Q4_K — attn + embed + norms + tokenizer)
larql slice gemma4-31b.vindex --preset client -o gemma4-31b.client.vindex

# Server slice (27 GB — gate + interleaved FFN + down_meta, no attention)
larql slice gemma4-31b.vindex --preset server -o gemma4-31b.server.vindex

# Server (holds the FFN half):
larql serve gemma4-31b.server.vindex --port 8080 --ffn-only

# Client (laptop — runs attention locally, FFN over HTTP):
larql run gemma4-31b.client.vindex --ffn http://server.local:8080 \
  "The capital of France is"
```

Other presets: `browse` (DESCRIBE/WALK only, no forward pass), `router`
(MoE router only, ADR-0003), `all` (full clone). See `larql slice --help`
for the explicit part list.

**3-tier topology (ADR-0008).** When laptop RAM matters, split the
embedding table out to its own server:

```bash
# Attention-only client (no embed, no FFN — ~310 MB on 4B, 10× smaller than `client`)
larql slice gemma3-4b.vindex --preset attn -o gemma3-4b.attn.vindex

# Embed server slice (embed + tokenizer; paired with ADR-0008 embed-server)
larql slice gemma3-4b.vindex --preset embed -o gemma3-4b.embed.vindex
```

The 3-tier client + embed server + FFN server split unlocks the
"laptop in ~1 GB" version of the dense-remote topology for small
models. Full rationale in
[`docs/adr/0007-vindex-distribution.md`](docs/adr/0007-vindex-distribution.md)
and [`docs/adr/0008-embed-server.md`](docs/adr/0008-embed-server.md).

### Publish to HuggingFace — full + slices + collections

`larql publish` combines `slice` + `hf publish` and adds HuggingFace
**collections**: one run uploads six sibling repos and files them into
three nested collections (model / family / library) for discovery.

```bash
# One command. Six repos (full + client + attn + embed + server + browse).
# Three collections (model / family / library).
larql publish gemma4-31b.vindex --repo chrishayuk/gemma-4-31b-it-vindex

# Preview without touching HF
larql publish gemma4-31b.vindex --repo chrishayuk/gemma-4-31b-it-vindex --dry-run
```

**Skip-if-unchanged.** Each upload compares the local SHA256 against the
remote `lfs.oid`. Files that already match skip the transfer. Re-publishing
a ~27 GB server slice where nothing changed re-uploads only the manifest —
not 27 GB of weights. Override with `--force-upload`.

**Streaming + progress.** Uploads stream the file (no 27 GB-into-RAM pre-read)
and report live progress via a per-file bar. An interrupted run picks up
on the next invocation: completed files skip via SHA, the interrupted
file re-uploads.

Flags: `--no-full`, `--slices client,server`, `--collections model,family`,
`--model-title`, `--family`, `--library-title`, `--slice-repo-template`,
`--force-upload`, `--dry-run`. Requires `HF_TOKEN` or
`~/.huggingface/token`.

### Pull with slice awareness

`larql pull` mirrors `publish` on the download side: pick a specific
sibling, pull them all, or pull a whole collection. Each file gets an
indicatif progress bar; hf-hub resumes interrupted downloads from the
`.incomplete` partial on the next run.

```bash
# Plain pull — the full vindex. Shows a hint at the end listing
# any `-client` / `-attn` / `-embed` / `-server` / `-browse` siblings
# that exist on HF.
larql pull chrishayuk/gemma-4-31b-it-vindex

# Pull just the client slice (laptop side of `run --ffn URL`)
larql pull chrishayuk/gemma-4-31b-it-vindex --preset client

# Pull full + every default sibling in one command
larql pull chrishayuk/gemma-4-31b-it-vindex --all-slices

# Pull every dataset in an HF collection — works on the collection URL
# from larql publish or the slug alone.
larql pull --collection chrishayuk/gemma-4-31b-it-larql-vindex-abc123
```

**Bounding server RSS.** `--ffn-only` skips the eager gate warmup at
startup (55 GB → 5.6 GB on 31B Q4_K). For steady-state bounds, layer
each of these on as needed:

```bash
larql serve gemma4-31b.vindex --port 8080 --ffn-only \
  --layers 0-19                    \  # hard bound: this shard serves only layers 0-19
  --max-gate-cache-layers 4        \  # LRU cap on decoded f16 gate heap
  --release-mmap-after-request        # madvise(DONTNEED) post-request (Linux strict)
```

`--layers` is the reliable hard bound on both Linux and macOS.
`--release-mmap-after-request` is strict on Linux, advisory on Darwin.
See `docs/adr/0005-ffn-service-memory-bounds.md` for the measured
ceilings under each combination.

### Query via LQL

```bash
larql repl
larql lql 'USE "gemma3-4b.vindex"; DESCRIBE "France";'
larql lql 'USE "hf://chrishayuk/gemma-3-4b-it-vindex"; DESCRIBE "France";'
```

### Research / interpretability tools

All under `larql dev <subcmd>` (weight extraction, QK rank analysis,
OV→gate projection, circuit discovery, trajectory tracing, 20+ others):

```bash
larql dev --help
larql dev walk --prompt "The capital of France is" --index gemma3-4b.vindex --predict
```

Legacy invocation `larql walk …` still works and transparently trampolines
to `larql dev walk …`.

## What is a Vindex?

A vindex is a directory containing a model's weights reorganised for queryability. Gate vectors become a KNN index. Embeddings become token lookups. Down projections become edge labels. The model IS the database.

```
gemma3-4b.vindex/
  gate_vectors.bin         # W_gate rows (KNN index, 3.3 GB)
  embeddings.bin           # W_embed matrix (token lookup, 2.5 GB)
  down_meta.bin            # Per-feature output metadata (binary)
  index.json               # Config, layer bands, provenance
  tokenizer.json           # Tokenizer
  relation_clusters.json   # Discovered relation types
  feature_labels.json      # Probe-confirmed labels
```

Three extraction levels:

| Level | CLI Flag | LQL Syntax | Size (f16) | Enables |
|-------|----------|-----------|-----------|---------|
| Browse | `--level browse` (default) | `EXTRACT MODEL ... INTO ...` | ~3 GB | DESCRIBE, WALK, SELECT |
| Inference | `--level inference` | `... WITH INFERENCE` | ~6 GB | + INFER |
| All | `--level all` | `... WITH ALL` | ~10 GB | + COMPILE |

Add `--f16` to halve file sizes with negligible accuracy loss.

## Architecture

Two crate families. LARQL-specific crates own the vindex + LQL + server stack;
portable `model-*` crates carry primitives that any neural-model compiler
(LARQL, TinyModel, others) can consume.

```
# LARQL-specific
larql-models      Model config, architecture traits, weight loading, quant/dequant
    ↓
larql-vindex      Vindex lifecycle: extract, load, query, mutate, patch, save
    ↓
larql-core        Graph algorithms, merge, diff
larql-inference   Forward pass, BLAS-fused attention, Metal GPU, WalkFfn
    ↓
larql-lql         LQL parser, executor, REPL, USE REMOTE client
    ↓
larql-server      HTTP/gRPC server: serve vindexes over the network
larql-cli         CLI commands (extract-index, build, serve, repl, convert, hf, verify)

# Portable (no LARQL deps; extract to sibling repo later)
model-compute         bounded compute: native kernels (default) + wasmtime (opt-in)
```

The portable crate never imports `larql-*`. Flow is one-way: LARQL consumes
it (e.g. compile-time resolution of `sum(1..100)` via `model_compute::native`).
See [crates/model-compute/README.md](crates/model-compute/README.md).

### larql-vindex

Owns the vindex lifecycle. Streaming extraction (mmap, no full model load), KNN via BLAS matmul,
zero-copy mmap loading, split weight files, readonly base with patch overlay, clustering, f16 storage.

```rust
// Load (readonly base)
let index = VectorIndex::load_vindex(&path, &mut cb)?;
let patched = PatchedVindex::new(index);

// Query
let hits = patched.gate_knn(layer, &query, 10);  // 0.008ms/layer
let trace = patched.walk(&query, &layers, 10);    // multi-layer scan

// Mutate (patch overlay — base files never modified)
patched.insert_feature(layer, feature, gate_vec, meta);
patched.apply_patch(VindexPatch::load("edits.vlp")?);
```

### larql-lql

LQL parser and executor. 20+ statement types across 5 categories:

- **Lifecycle**: EXTRACT, COMPILE, DIFF, USE
- **Browse**: WALK, DESCRIBE, SELECT, EXPLAIN WALK
- **Inference**: INFER, EXPLAIN INFER
- **Mutation**: INSERT, DELETE, UPDATE, MERGE
- **Patches**: BEGIN PATCH, SAVE PATCH, APPLY PATCH, SHOW PATCHES, REMOVE PATCH
- **Introspection**: SHOW RELATIONS/LAYERS/FEATURES/MODELS/PATCHES, STATS

## LQL Reference

See [docs/specs/lql-spec.md](docs/specs/lql-spec.md) for the full language specification and [docs/lql-guide.md](docs/lql-guide.md) for a quick start guide.

### Key Statements

```sql
-- Decompile a model
EXTRACT MODEL "google/gemma-3-4b-it" INTO "gemma3-4b.vindex" WITH ALL;

-- Browse knowledge (no GPU needed)
USE "gemma3-4b.vindex";
DESCRIBE "France";                      -- verbose by default: [relation] labels, also-tokens
DESCRIBE "Einstein" ALL LAYERS;
DESCRIBE "France" BRIEF;                -- compact view
WALK "The capital of France is" TOP 10;

-- Run inference (needs model weights in vindex)
INFER "The capital of France is" TOP 5 COMPARE;

-- Trace the residual stream (decomposed forward pass)
TRACE "The capital of France is" FOR "Paris";
TRACE "The capital of France is" DECOMPOSE LAYERS 22-27;
TRACE "The capital of France is" SAVE "france.trace";

-- Edit knowledge (auto-patch: base files never modified)
INSERT INTO EDGES (entity, relation, target)
    VALUES ("John Coyle", "lives-in", "Colchester");
-- "Auto-patch started (use SAVE PATCH to persist)"

-- Insert with all knobs (multi-layer constellation, validated regime)
INSERT INTO EDGES (entity, relation, target)
    VALUES ("Atlantis", "capital-of", "Poseidon")
    AT LAYER 24
    CONFIDENCE 0.95
    ALPHA 0.30;

-- Patches (lightweight, shareable knowledge diffs)
BEGIN PATCH "medical.vlp";
INSERT INTO EDGES (entity, relation, target)
    VALUES ("aspirin", "treats", "headache");
SAVE PATCH;
APPLY PATCH "medical.vlp";

-- Bake the patches into a fresh standalone vindex (instant on APFS:
-- weight files are hardlinked from source, only down_weights.bin gets
-- the override columns rewritten in place).
COMPILE CURRENT INTO VINDEX "gemma3-4b-medical.vindex";

-- Or recompile back to standard HuggingFace / GGUF format. The
-- constellation is in the standard down_proj tensors, so loading in
-- Transformers or GGUF runtimes Just Works — no special loader code.
COMPILE CURRENT INTO MODEL "edited/" FORMAT safetensors;
```

## Patches

Patches are lightweight JSON files (.vlp) that capture INSERT/DELETE/UPDATE operations. They overlay an immutable base vindex without modifying it.

```sql
-- Create a patch
BEGIN PATCH "medical-knowledge.vlp";
INSERT INTO EDGES (entity, relation, target)
    VALUES ("aspirin", "side_effect", "bleeding");
SAVE PATCH;

-- Apply patches (stackable, reversible)
APPLY PATCH "medical-knowledge.vlp";
APPLY PATCH "fix-hallucinations.vlp";
SHOW PATCHES;
REMOVE PATCH "fix-hallucinations.vlp";

-- Extract diff between two vindexes as a patch
DIFF "base.vindex" "edited.vindex" INTO PATCH "changes.vlp";
```

A single fact is ~10 KB. A 1,000-fact domain patch is ~10 MB. Compared to the full model at 8 GB, that's 1/800th the size. No fine-tuning, no GPU, no retraining.

The base vindex is always readonly. INSERT/DELETE/UPDATE automatically create a patch overlay. Edits are never written to base files.

## Vindexfile

Declarative model builds. Like a Dockerfile for model knowledge.

```dockerfile
# Vindexfile
FROM hf://chrishayuk/gemma-3-4b-it-vindex
PATCH hf://medical-ai/drug-interactions@2.1.0
PATCH ./patches/company-facts.vlp
INSERT ("Acme Corp", "headquarters", "London")
LABELS hf://chrishayuk/gemma-3-4b-it-labels@latest
EXPOSE browse inference
```

```bash
larql build .                          # build from Vindexfile
larql build . --stage prod             # named stage
larql build . --output custom.vindex   # custom output path
```

## Model Support

Input formats: **safetensors** (HuggingFace), **GGUF** (llama.cpp, dequantized to f32), **MLX** (Apple, same safetensors layout).

| Family | Models | FFN Type |
|--------|--------|----------|
| Gemma | Gemma 2/3/4 (2B-31B) | Gated (GeGLU) |
| Llama | Llama 2/3 (7B-405B) | Gated (SiLU) |
| Mistral | Mistral 7B | Gated (SiLU) |
| Mixtral | Mixtral 8x7B, 8x22B | MoE (8 experts) |
| Qwen | Qwen 2/2.5 (0.5B-72B) | Gated (SiLU) |
| Phi | Phi 2/3 (2.7B-14B) | Gated |
| DeepSeek | DeepSeek V2/V3 | MoE (shared + routed) |
| GPT-OSS | GPT-OSS-120B | MoE (128 experts, MXFP4) |
| GPT-2 | GPT-2 (117M-1.5B) | Dense (GELU) |

Dense and full-precision MoE models support all operations (DESCRIBE, WALK, INFER). MXFP4-quantized MoE models (GPT-OSS) can be extracted and served but DESCRIBE/WALK produce noisy results due to 4-bit weight precision — use INFER for accurate knowledge queries. See [operations spec](docs/specs/vindex-operations-spec.md) for details.

## Benchmarks

### Vindex Operations

| Operation | Latency |
|---|---|
| Gate KNN (per layer) | 0.008ms |
| Walk (34 layers) | 0.3ms |
| Feature lookup | <1ns |
| Save gates (8 MB) | 1.1ms |
| Load vindex | 8ms |
| Mutate (meta + gate) | 617ns |

### Inference Engine (Gemma 3 4B, Apple Silicon M3 Max)

| Operation | Latency | tok/s |
|---|---|---|
| **GPU Q4K decode (Metal, 34L, KV cache)** | **15.6ms** | **64** |
| Walk prediction (CPU, no attention) | 33ms | 30 |
| INFER walk (CPU, with attention, mmap FFN) | 517ms | 1.9 |
| INFER dense (CPU, all matmul) | 535ms | 1.9 |
| DESCRIBE (knowledge browse) | 33ms | — |

GPU decode per-stage breakdown:

| Component | Time | % of total |
|---|---|---|
| GPU forward (34 layers, Q4K/Q6K) | 14.1ms | 86% |
| LM head (Q4_0 synthesized from f16 embeddings) | 2.0ms | 12% |
| Embed + norm + detokenize | <0.1ms | <1% |

CPU walk breakdown:

| Component | Time | % of total |
|---|---|---|
| Logits (262K vocab gemv) | 221ms | 41% |
| FFN × 34 layers (walk) | 194ms | 36% |
| Attention × 34 layers | 84ms | 16% |

Walk is **faster than dense** (517ms vs 535ms). GPU Q4K decode is **16× faster** than CPU walk. FFN down projection in walk reads from mmap'd vindex (zero-copy BLAS). Walk only needs ~3.5GB of model weights (attention + embeddings), not 16.6GB. No quantization. See [docs/ffn-graph-layer.md](docs/ffn-graph-layer.md) for architecture and [docs/inference-engine.md](docs/inference-engine.md) for engine details.

## Residual Stream Trace

Capture the complete record of inference — every layer, every contribution, queryable.

```sql
-- LQL: answer trajectory through all layers
larql> TRACE "The capital of France is" FOR "Paris";
  Layer   Rank     Prob      Attn       FFN      Who
    L22     50    0.002     +22.2     +34.4   BOTH ↑
    L23     10    0.024     -16.9     +55.9    FFN ↑
    L24      1    0.714    +105.7     +24.4   BOTH ↑  ← phase transition
    L25      1    0.997      +4.3     +94.4    FFN ↑
    L26      1    0.999     +83.1     +18.7   BOTH ↑

-- Attn vs FFN decomposition at the phase transition
larql> TRACE "The capital of France is" DECOMPOSE LAYERS 22-27;

-- Persist for later analysis
larql> TRACE "The capital of France is" SAVE "france.trace";
```

```python
# Python: same trace, programmatic access
import larql

wm = larql.WalkModel("gemma3-4b.vindex")
t = wm.trace("The capital of France is")
t.answer_trajectory("Paris")   # rank, prob, attn/ffn logits per layer
t.top_k(24)                    # [('Paris', 0.714), ...]
t.save("trace.bin")            # mmap'd store
```

### Tiered Context (infinite context without KV cache)

| Storage | Per window | 370K tokens | vs KV cache |
|---|---|---|---|
| Boundary residual | 10 KB | 18.9 MB | 3,100x |
| Tier 4 int8 (bit-perfect) | 58 KB | 110 MB | 511x |
| KV cache | ~30 MB | 56,000 MB | 1x |

```python
from larql._native import BoundaryWriter, BoundaryStore

# Write boundary residuals — one per 200-token window
writer = BoundaryWriter("context.bndx", hidden_size=2560, window_size=200)
writer.append(token_offset=0, window_tokens=200, residual=boundary_vec)
writer.finish()

# Mmap'd read — OS pages on demand, RSS ≈ one boundary
store = BoundaryStore("context.bndx")
store.residual(42)  # zero-copy from mmap
```

See [docs/residual-trace.md](docs/residual-trace.md) for the full writeup.

## Documentation

| Doc | Description |
|---|---|
| [docs/specs/lql-spec.md](docs/specs/lql-spec.md) | LQL language specification (v0.3) |
| [docs/specs/vindex-format-spec.md](docs/specs/vindex-format-spec.md) | Vindex file format specification (v0.3, ~98% implemented) |
| [docs/specs/vindex-operations-spec.md](docs/specs/vindex-operations-spec.md) | Vindex operations, API, patches (~98% implemented) |
| [docs/specs/vindex-ecosystem-spec.md](docs/specs/vindex-ecosystem-spec.md) | Distributed hosting, HuggingFace, Vindexfile (~85% implemented) |
| [docs/lql-guide.md](docs/lql-guide.md) | LQL quick start guide |
| [docs/cli.md](docs/cli.md) | CLI reference |
| [docs/inference-engine.md](docs/inference-engine.md) | Inference engine — BLAS-fused attention, Metal GPU, auto-calibration |
| [docs/ffn-graph-layer.md](docs/ffn-graph-layer.md) | FFN graph layer — mmap walk faster than dense (517ms vs 535ms), all 34 layers |
| [docs/walk-boundary-sweep.md](docs/walk-boundary-sweep.md) | Walk boundary sweep — correctness proof across all layer boundaries |
| [docs/residual-trace.md](docs/residual-trace.md) | Residual stream trace — decomposition, storage, tiered context |
| [docs/specs/trace-format-spec.md](docs/specs/trace-format-spec.md) | Trace file format specification (.bin, .bndx, .ctxt) |

## Building & Testing

(Needs Openblas under Linux)
```bash
cargo build --release                    # optimised build
cargo build --release --features metal   # with Metal GPU backend
cargo test                               # all tests across all crates
cargo test -p larql-inference            # inference engine tests (109 tests)
cargo test -p larql-inference --features metal  # + Metal GPU tests (115 tests)
cargo test -p larql-lql                  # LQL parser + executor tests (272 tests)
cargo test -p larql-vindex               # vindex storage + patch tests (104 tests)

# Inference engine examples
cargo run --release -p larql-inference --example attention_demo    # fused attention demo
cargo run --release -p larql-inference --example bench_attention   # attention benchmarks
cargo run --release -p larql-inference --example backend_demo --features metal   # backend demo
cargo run --release -p larql-inference --example bench_backend --features metal  # backend benchmarks
cargo run --release -p larql-inference --example bench_inference   # full inference benchmarks

# Vindex tools (build once, enables mmap walk)
cargo run --release -p larql-vindex --example convert_gates_f32 -- path/to/vindex   # f16→f32 gate vectors
cargo run --release -p larql-vindex --example build_down_features -- path/to/vindex  # feature-major down vectors
cargo run --release -p larql-vindex --example build_up_features -- path/to/vindex    # feature-major up vectors

# Server (walk inference over HTTP)
cargo run --release -p larql-server -- path/to/vindex --port 8080

# Vindex and LQL demos (synthetic — run in CI)
cargo run -p larql-vindex --example demo_features                    # vindex feature showcase
cargo run --release -p larql-vindex --example mmap_demo              # mmap RAM behaviour + scaling table
cargo run --release -p larql-vindex --example q4k_demo               # streaming Q4_K: size ratio, manifests, dequant round-trip
cargo run --release -p larql-vindex --example demo_memit_solve       # MEMIT decomposition + MemitStore round-trip
cargo run -p larql-lql --example parser_demo                         # parser demo (24/24 statements)
cargo run -p larql-lql --example lql_demo                            # LQL spec compliance (61/61)
cargo run --release -p larql-lql --example compact_demo              # LSM storage tier walkthrough

# Model-dependent demos (require real vindex, skip gracefully otherwise)
cargo run --release -p larql-lql --example compile_demo              # end-to-end COMPILE INTO VINDEX on real Gemma 4B
cargo run --release -p larql-lql --example refine_demo               # 10-fact INSERT + COMPILE (exp 14 reproduction, 10/10 retrieval)
cargo run --release -p larql-lql --example trace_demo                # TRACE residual decomposition on real Gemma 4B

# Criterion benches (use --quick for a fast sweep, omit for full sample sizes)
cargo bench -p larql-lql    --bench parser               # parse_single × 18 + parse_batch
cargo bench -p larql-lql    --bench executor             # SELECT, SHOW, DELETE, UPDATE, patch lifecycle
cargo bench -p larql-lql    --bench compile              # COMPILE INTO VINDEX bake cost
cargo bench -p larql-vindex --bench vindex_ops           # KNN, walk, save/load, mutate, MoE
cargo bench -p larql-vindex --bench vindex_scaling       # production-dim KNN (Gemma/Llama/Mixtral)
cargo bench -p larql-vindex --bench memit_solve          # ridge decomposition throughput
cargo bench -p larql-vindex --bench extract_throughput   # streaming extract: f32 vs Q4K write-path
cargo bench -p larql-vindex --bench q4k_vs_f32           # per-layer attn retrieval: f32 memcpy vs Q4K dequant
cargo bench -p larql-compute --bench matmul              # CPU/Metal matmul backends
```

The `compile_demo` example proves the full flow on a real Gemma 4B
vindex: `INSERT Atlantis → Poseidon`, `COMPILE CURRENT INTO VINDEX`,
then `USE` the compiled vindex in a fresh session and verify
`INFER "The capital of Atlantis is" → Pose 56.91%` and
`INFER "The capital of France is" → Paris 67.34%` (neighbour
preserved). The constellation is baked into `down_weights.bin`
column-wise — no overlay or sidecar needed at load time.

Bench HTML reports go to `target/criterion/`. The `parser` bench
parses 100 mixed statements in ~78 µs (1.28 M stmts/s); `vindex_ops`
runs production-sized Gemma 4B gate KNN in ~2.78 ms/layer; `compile`
runs `COMPILE INTO VINDEX` in ~1.84 ms (no patches) to 2.41 ms (with
`down_weights.bin`).

## License

Apache-2.0
