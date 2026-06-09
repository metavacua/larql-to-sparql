# Step 2b — Design notes for the `Qwen35Weights` vindex adapter

Investigation from session 2026-05-16 (post PR #149) so a fresh session
can pick up Step 2b cold.

## Target API

```rust
// In crates/larql-inference/src/attention/qwen35_load_vindex.rs (NEW)
pub fn load_qwen35_weights_from_vindex(
    vindex_dir: &std::path::Path,
) -> Result<crate::attention::qwen35_forward::Qwen35Weights, LoadError>;
```

## Field map: Qwen35Weights → vindex artifacts

`Qwen35Weights` is defined at
`crates/larql-inference/src/attention/qwen35_forward.rs:98`. Fields:

| Field | Source | Approach |
|---|---|---|
| `embed: ArcArray2<f32>` | `embeddings.bin` | f32 reader (existing `load_vindex_embeddings`) |
| `embed_quant: Option<QuantTensor>` | none in vindex today | leave `None` for first cut |
| `layers: Vec<Qwen35FullLayerWeights>` | per-layer assembly — see below | new code |
| `final_norm: Arc<[f32]>` | `norms.bin` → `norm.weight` | existing norms reader |
| `lm_head: ArcArray2<f32>` | 0×0 placeholder | placeholder (lazy path) |
| `lm_head_quant: Option<QuantTensor>` | `lm_head_q4.bin` | `QuantTensor::from_raw(read_to_vec, TYPE_Q6_K, vocab, hidden)` |
| `ffn_dim: usize` | `index.json::model_config::moe_intermediate_size` | direct |
| `backend: Option<Arc<dyn ComputeBackend>>` | none | `None` |

### Qwen35FullLayerWeights (per-layer assembly)

```rust
pub struct Qwen35FullLayerWeights {
    pub block: Qwen35LayerWeights,   // Linear(DeltaNetLayerWeights) | Attention(Qwen35AttentionLayerWeights)
    pub attn_post_norm: Arc<[f32]>,
    // Dense SwiGLU FFN slots — 0×0 placeholders for MoE layers.
    pub ffn_gate: ArcArray2<f32>,
    pub ffn_up: ArcArray2<f32>,
    pub ffn_down: ArcArray2<f32>,
    pub ffn_gate_quant: Option<QuantTensor>,
    pub ffn_up_quant: Option<QuantTensor>,
    pub ffn_down_quant: Option<QuantTensor>,
    // MoE FFN — populated for every layer on qwen35moe.
    pub moe: Option<Qwen35MoeFfnWeights>,
}
```

For Qwen 3.6 35B-A3B (every layer is MoE):
- Set all 6 dense/quant FFN slots to placeholders (`Array2::zeros((0,0))` / `None`).
- Populate `moe: Some(Qwen35MoeFfnWeights { ... })` from
  `layers/layer_{LL}.weights`.

For block (per layer):

```rust
if arch.is_linear_attention_layer(layer) {
    Qwen35LayerWeights::Linear(DeltaNetLayerWeights { ... })
} else {
    Qwen35LayerWeights::Attention(Qwen35AttentionLayerWeights { ... })
}
```

### DeltaNetLayerWeights (linear layers — 30 of 40)

`crates/larql-inference/src/attention/deltanet_block.rs:53`. Fields:

| Field | Vindex source | Format | Note |
|---|---|---|---|
| `attn_norm: Arc<[f32]>` | `norms.bin` `layers.{L}.attn_norm.weight` | f32 vector | hidden=2048 |
| `attn_qkv: ArcArray2<f32>` | 0×0 placeholder | — | use lazy path |
| `attn_gate: ArcArray2<f32>` | 0×0 placeholder | — | use lazy path |
| `ssm_conv1d: ArcArray2<f32>` | `norms.bin` `layers.{L}.ssm_conv1d.weight` | f32 2-D flattened | reshape `[d_conv=4, conv_dim]` |
| `ssm_dt: Arc<[f32]>` | `norms.bin` `layers.{L}.ssm_dt.weight` | f32 vector | `n_v_heads` |
| `ssm_a: Arc<[f32]>` | `norms.bin` `layers.{L}.ssm_a.weight` | f32 vector | `n_v_heads` |
| `ssm_beta: ArcArray2<f32>` | `deltanet_weights_q4k.bin` | Q4_K → f32 | dequant; no lazy slot |
| `ssm_alpha: ArcArray2<f32>` | `deltanet_weights_q4k.bin` | Q4_K → f32 | dequant; no lazy slot |
| `ssm_norm: Arc<[f32]>` | `norms.bin` `layers.{L}.ssm_norm.weight` | f32 vector | `head_v_dim` |
| `ssm_out: ArcArray2<f32>` | 0×0 placeholder | — | use lazy path |
| `attn_qkv_quant: Option<QuantTensor>` | `deltanet_weights_q4k.bin` | Q4_K mmap | **lazy path** |
| `attn_gate_quant: Option<QuantTensor>` | `deltanet_weights_q4k.bin` | Q4_K mmap | **lazy path** |
| `ssm_out_quant: Option<QuantTensor>` | `deltanet_weights_q4k.bin` | Q4_K mmap | **lazy path** |

Key insight: `ssm_alpha` and `ssm_beta` have **no `*_quant` variant** in
the struct. Either dequant them at load (cheap — [32, 2048] = 65K f32
each × 30 linear layers = 7.5 MB total) or add `_quant` variants. For
the first cut, **dequant at load**.

### QuantTensor construction from vindex bytes

`crates/larql-models/src/quant/lazy.rs` exposes:

```rust
pub fn from_raw(data: Vec<u8>, tensor_type: u32, rows: usize, cols: usize)
    -> Result<Self, ModelError>;

pub fn from_mmap_region(mmap: Arc<memmap2::Mmap>, byte_offset: usize,
    byte_len: usize, tensor_type: u32, rows: usize, cols: usize)
    -> Result<Self, ModelError>;
```

Vindex storage today holds `bytes::Bytes`, NOT `Arc<Mmap>`. Two
options for zero-copy:

**Option A (recommended for first cut):** use `from_raw` with a
`view.as_slice().to_vec()` copy at load time. ~542 MB of DeltaNet
copy on the Qwen 3.6 35B-A3B vindex; one-time cost. Simple.

**Option B (RAM-optimal, defer):** extend `QuantBacking` (currently
`Heap(Arc<[u8]>) | Mmap(Arc<memmap2::Mmap>)`) with a new
`Bytes(bytes::Bytes)` variant. Or expose the underlying
`Arc<Mmap>` from `VindexStorage` via a new method. This is a
follow-up RAM optimisation, not blocking Step 2b's correctness.

Use `from_raw` for Step 2b. File a follow-up arc for the Bytes-
backed `QuantTensor`.

### Q6_K vs Q4_K formats

`deltanet_q4k_layer_data` returns a `format` tag string (e.g.
`"Q4_K"`). Map to the GGML tensor type constants via
`larql_models::quant::ggml::{TYPE_Q4_K, TYPE_Q6_K}` (see
`crates/larql-vindex/src/quant/registry.rs::lookup`).

### MoE PerExpert from `layers/layer_{LL}.weights`

`crates/larql-vindex/src/format/weights/write_layers.rs`:
- `parse_layer_weights_header(bytes)` returns
  `(format, num_entries, inter, hidden, offsets)` where each
  offset entry is `(gate_up_off, gate_up_bytes, down_off, down_bytes)`.
- Per-expert layout: `gate_up` is `[2*inter, hidden]` Q4_K
  (interleaved `[gate rows | up rows]`); `down` is
  `[hidden, padded_inter]` Q4_K.

For `Qwen35MoeFfnWeights`:

```rust
pub struct Qwen35MoeFfnWeights {
    pub router: QuantTensor,        // [num_experts, hidden] — from norms.bin? gate_vectors.bin?
    pub gate_exps: QuantTensor,     // [num_experts * expert_ffn_dim, hidden]
    pub up_exps: QuantTensor,       // [num_experts * expert_ffn_dim, hidden]
    pub down_exps: QuantTensor,     // [num_experts * hidden, expert_ffn_dim]
    pub shexp_gate: Option<QuantTensor>,
    pub shexp_up: Option<QuantTensor>,
    pub shexp_down: Option<QuantTensor>,
    pub num_experts: usize,
    pub top_k: usize,
}
```

This struct expects **packed** per-projection tensors (`gate_exps`
concatenating all 256 expert gate rows into one big
`[256 * inter, hidden]` matrix), not 256 separate `gate_proj`
QuantTensors. Two options:

1. Build a **packed Vec<u8>** by concatenating the 256 expert
   gate bytes (and similarly for up + down), then call
   `QuantTensor::from_raw(packed, TYPE_Q4_K, num_experts * inter, hidden)`.
   This matches the consumer's expectation.

2. Add a `Vec<QuantTensor>` per-expert alternate path. More
   invasive on the forward side.

**Recommended:** option 1. The vindex `layer_LL.weights` already
stores experts contiguously (entry table → entry data → next entry).
The packed Vec is essentially `gate_up_bytes` from each expert in
order. Verify the stride math vs `quantize_dense_entry`'s
`[2 * inter, hidden]` layout — gate rows must come before up rows
per expert.

The **router** tensor:
`crates/larql-vindex/src/format/weights/write_q4k/norms.rs:73-89`
writes it into `norms.bin` (as f32, flattened) under
`arch.moe_router_key(layer)` only for `is_hybrid_moe`. Qwen35moe
returns `false` for is_hybrid_moe so this isn't fired — **bug:
the router is currently NOT being written for qwen35moe**.

**Step 2b prerequisite:** fix the router write. Either:
- Loosen the `is_hybrid_moe()` guard in `norms.rs:73` to
  `is_moe() && expert_format() != PackedMxfp4` so qwen35moe
  PerExpert also writes the router.
- OR write the router into the per-layer file.

Recommend the first — keeps `norms.bin` as the single source for
small-scalars-per-layer.

This is a **separate small PR** before Step 2b can complete.

## Scope correction (2026-05-16, post PR #157 investigation)

**The 455 LoC remaining estimate was wrong.** Reading
`crates/larql-inference/src/attention/qwen35_load.rs:62-374`
revealed that the existing `load_qwen35_weights(weights, arch)`
function is **data-source agnostic** — it takes a generic
`&ModelWeights` reference and walks it via three helpers
(`get_vec`, `get_tensor`, `weights.quant_tensors.get(key)`). Those
helpers don't care whether `weights` was populated by
`larql_models::load_gguf(...)` (GGUF mmap) or by
`larql_vindex::load_model_weights_q4k_shard(...)` (vindex
manifest).

So the **real work for 2b** is not per-layer assembly logic
(already exists, reusable) — it's just **populating
`weights.quant_tensors` and `weights.tensors` from vindex bytes**
before calling `load_qwen35_weights`. Three concrete bridges:

1. **DeltaNet matmul bridge** (~50 LoC): for each linear layer L,
   pull the 5 (data, fmt) tuples from
   `idx.deltanet_q4k_layer_data(L)`, build a `QuantTensor` per
   matmul-class tensor (`attn_qkv` / `attn_gate` / `ssm_out`)
   inserted into `weights.quant_tensors`, and dequant the two
   smaller matmuls (`ssm_alpha` / `ssm_beta`) into
   `weights.tensors` (they have no `*_quant` slot in
   `DeltaNetLayerWeights`).

2. **Standard attn bridge** — **blocked by 2b.3a, then ~30 LoC**.
   The existing `attn_q4k_layer_data(layer)` accessor at
   `crates/larql-vindex/src/index/storage/vindex_storage/mmap_storage.rs:484`
   uses `let base = layer * ATTN_TENSORS_PER_LAYER` index arithmetic.
   For qwen35moe with `full_attention_interval=4` the manifest is
   **sparse** — only 10 of 40 layers carry Q/K/V/O entries (layers
   3, 7, 11, …, 39), so `attn_q4k_layer_data(3)` returns layer 11's
   bytes (manifest index 12-15) instead of layer 3's (manifest
   index 0-3).
   The storage drops keys + shapes when parsing the JSON manifest
   (only keeps `Vec<(offset, length, format)>`), so a
   key-prefix lookup like the DeltaNet accessor isn't directly
   possible without extending the in-memory manifest.

   **Task 2b.3a (~40 LoC)** — extend `attn_q4k_manifest` to also
   carry `key: String` + `shape: Vec<usize>` per entry (parallel
   to PR #149's + #160's DeltaNet storage). Add a new
   `attn_q4k_sparse_layer_data(layer)` accessor that filters
   manifest entries by `layers.{L}.self_attn.{q,k,v,o}_proj.weight`
   prefix. Don't break the existing dense-manifest callers
   (Gemma 3, Gemma 4 dense, …); they continue using
   `attn_q4k_layer_data` unchanged.

   **Task 2b.3b (~30 LoC)** — bridge body that consumes
   `attn_q4k_sparse_layer_data` and populates
   `weights.quant_tensors` under
   `arch.attn_{q,k,v,o}_key(L)`. Same pattern as PR #161's
   DeltaNet bridge.

3. **MoE PerExpert bridge** (~80 LoC): parse each
   `layers/layer_LL.weights` via
   `parse_layer_weights_header`, then for each of 256 experts
   construct the packed `Qwen35MoeFfnWeights::{gate,up,down}_exps`
   QuantTensors. The expert-byte concatenation order needs to
   match `quantize_dense_entry`'s layout — gate-rows then up-rows
   per expert (NOT all-gates then all-ups). Plus the router from
   `weights.vectors`.

The `load_qwen35_weights_from_vindex` orchestrator in PR #157's
stub becomes: call the three bridges, then invoke
`load_qwen35_weights(&weights, &*arch)` and propagate its result.

### Q4_K dequant primitive

For the `ssm_alpha` / `ssm_beta` dense slots, use
`larql_models::quant::ggml::dequantize(bytes, TYPE_Q4_K, n_elems)`
(or the appropriate per-type wrapper). The vindex format tag
string `"Q4_K"` maps to `TYPE_Q4_K = 12` via
`crates/larql-vindex/src/quant/registry.rs::lookup`.

Total: ~160 LoC across three bridges + orchestrator. One focused
PR. The previous 2b.2/2b.3/2b.4/2b.5 split is collapsed by this
finding.



| Phase | Scope | LoC | Status |
|---|---|---:|---|
| 2b.0a | Router write **gate** fix in `write_q4k/norms.rs` | ~10 | ✅ Shipped in PR #152 |
| 2b.0b | Router **key** fix in `qwen35.rs::moe_router_key` | ~5 | ❌ See note below |
| 2b.1 | Norms reader: extract DeltaNet small tensors from `norms.bin` per layer | ~0 | ✅ No-op — existing reader works |
| 2b.2 | Per-layer assembly: `DeltaNetLayerWeights` from vindex bytes | ~120 | pending |
| 2b.3 | Per-layer assembly: `Qwen35AttentionLayerWeights` for full-attn layers | ~80 | pending |
| 2b.4 | MoE per-layer: parse `layer_LL.weights` → pack 256 experts into `Qwen35MoeFfnWeights` | ~150 | pending |
| 2b.5 | Top-level: `load_qwen35_weights_from_vindex` orchestrator | ~50 | pending |
| 2b.6 | Unit test: load `/tank/ai/Qwen/Qwen3.6-35B-A3B-vindex`, dimension-check the produced struct | ~50 | pending |

Total remaining ~455 LoC (was ~540).

### 2b.0b — router key still doesn't reach the writer (smoke result)

PR #152's gate change is correct — the `if arch.is_moe() &&
expert_format() != PackedMxfp4` block at `write_q4k/norms.rs:114`
now fires for qwen35moe. But the smoke conversion on the live
21 GB Qwen3.6-35B-A3B GGUF (output at
`/tank/ai/Qwen/Qwen3.6-35B-A3B-vindex-v2/`) still produces a
`weight_manifest.json` with **zero** entries matching
`mlp.gate.weight` or `gate.weight`.

Root cause: a key naming mismatch between the arch handler and the
GGUF loader.
- `crates/larql-models/src/architectures/qwen35.rs:340`:
  `moe_router_key(layer) → "layers.{L}.mlp.gate.weight"` (HF style).
- The GGUF tensor is `blk.{L}.ffn_gate_inp.weight`.
- `crates/larql-models/src/loading/gguf.rs:113` (the
  `normalize_gguf_key` remap table) has `("ffn_gate.",
  "mlp.gate_proj.")` but **no entry for `ffn_gate_inp.`**. So
  `normalize_gguf_key("blk.0.ffn_gate_inp.weight")` returns
  `"layers.0.ffn_gate_inp.weight"`, not the
  `"layers.0.mlp.gate.weight"` the arch returns.

`source.get_tensor(arch.moe_router_key(layer))` therefore returns
`None` and the write is silently skipped.

Two clean fixes:

1. **Update qwen35.rs** to return the loader-canonical key
   `"layers.{L}.ffn_gate_inp.weight"` from `moe_router_key`.
   Smallest delta. Mixtral already does the per-arch trick
   (returns `block_sparse_moe.gate.weight` directly). No global
   remap change.

2. **Add a remap entry** `("ffn_gate_inp.", "mlp.gate.")` to
   `gguf.rs:113`. Affects all MoE arches whose GGUFs ship
   `ffn_gate_inp` — Mixtral, qwen35moe, future MoE variants.
   Cleaner long-term but risks per-arch regressions; needs an
   audit before landing.

**Recommend option 1** for the next PR — minimum blast radius.

Verification: after the fix, re-run the smoke conversion and
check that
`grep -c 'ffn_gate_inp.weight' weight_manifest.json` (or
`'mlp.gate.weight'` if option 2 chosen) returns 40 (one per MoE
layer).

### 2b.1 simplification — existing reader already does it

`crates/larql-vindex/src/format/weights/load.rs:534` already loads
every `kind::VECTOR` entry from `norms.bin` (via
`weight_manifest.json`) into a `HashMap<String, Vec<f32>>` keyed by
the full tensor name. The DeltaNet small tensors emitted by
`write_q4k/norms.rs` — `layers.{L}.ssm_norm.weight`,
`layers.{L}.ssm_dt.weight`, `layers.{L}.ssm_a.weight`,
`layers.{L}.ssm_conv1d.weight` — are all VECTOR-kind, so they land
in the map for free.

Once task 2b.0b lands, the router will also be a VECTOR entry the
existing reader picks up automatically. **Implication for Step
2b.2/2b.4:** consume those vectors via `vectors.get(key)`. No new
reader code is needed.

## Step 2c — server dispatch (scope correction, post-2b green)

Investigation while opening Step 2c found the original ~100 LoC
estimate was wrong. The server's `run_chat_completion` at
`crates/larql-server/src/routes/openai/chat.rs:695` dispatches
through `larql_inference::layer_graph::generate_with_sampling` —
the standard transformer multi-token generate driver that calls
`predict_q4k_hidden_with_cache` per token and threads a `KvCache`
through.

qwen35 needs a **parallel generate driver** with three distinct
shape changes:

1. Per-token forward is `qwen35_forward_step(token, weights,
   dn_dims, attn_dims, hybrid_cache, eps)` — a different
   signature than the generic forward.
2. Cache is `DeltaNetHybridCache { kv_cache, dn_state,
   layer_kinds }`, not a plain `KvCache`. The DeltaNet linear
   layers carry per-layer matrix-valued recurrent state instead
   of K/V slabs.
3. Sampling + EOS + greedy + constrained-masking glue all need
   to be replicated. About half of `generate_with_sampling`'s
   ~200 LoC.

Refined estimate: **~150-200 LoC** for `qwen35_generate_with_sampling`
+ the arch-family dispatch in `run_chat_completion`. The dispatch
itself is ~10 LoC:

```rust
let result = if matches!(arch_family, "qwen35" | "qwen35moe") {
    qwen35_generate_with_sampling(...)
} else if let Some(schema) = constrained_schema {
    generate_constrained_streaming_sampled(...)
} else {
    generate_with_sampling(...)
};
```

The hard part is `qwen35_generate_with_sampling` itself —
materially a duplicate of `generate_with_sampling` with the three
shape changes. Could potentially factor a generic
"per-token-forward + sampling" template if both paths converge on
a trait, but that's premature; copy-and-modify first, factor
later.

### Dependencies

- Step 2b ✅ green (PR #167 smoke validated the Qwen35Weights
  produced by `load_qwen35_weights_from_vindex` is structurally
  complete).
- `qwen35_forward_step` ✅ exists and is battle-tested via
  `real_gguf_qwen35_bench`.
- `DeltaNetHybridCache` ✅ exists in
  `crates/larql-inference/src/attention/qwen35_block.rs`.

Nothing new required; just glue.

### Smoke gate

Live HTTP request:

```
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.6-35b-a3b",
    "messages": [{"role": "user", "content": "Hi"}]
  }'
```

against `larql serve /tank/ai/Qwen/Qwen3.6-35B-A3B-vindex-v4`.
Success = HTTP 200 + non-empty completion text + no panic. Full
parity vs llama.cpp is a separate change.

## Step 2c — driver readiness (post-#170 investigation)

Concrete implementation checklist for `qwen35_generate_with_sampling`,
based on the existing `real_gguf_qwen35_token_diff_vs_llama_cpp`
loop (`crates/larql-inference/src/attention/qwen35_load.rs:2641`)
and the in-tree primitives:

### Inputs

```rust
pub fn qwen35_generate_with_sampling(
    weights: &Qwen35Weights,
    tokenizer: &tokenizers::Tokenizer,
    prompt_ids: &[u32],
    max_tokens: usize,
    sampling: SamplingConfig,
    eos: &EosConfig,
) -> GenerateResult
```

The server's `run_chat_completion` (`chat.rs:738`) builds
`prompt_ids` + `sampling` + `eos` already and threads
`max_tokens` through. The driver dispatch arm becomes:

```rust
let qwen35_weights = larql_inference::attention::qwen35_load_vindex
    ::load_qwen35_weights_from_vindex(model.vindex_dir())?;
qwen35_generate_with_sampling(
    &qwen35_weights, &model.tokenizer, &prompt_ids,
    max_tokens, sampling, &eos,
)
```

### Construction

1. **`DeltaNetDims`** from the arch (`weights.arch.config()` exposes
   the SSM fields after PR #167's loader fix): `hidden`,
   `head_v_dim = ssm_state_size`, `n_v_heads = ssm_dt_rank`,
   `n_k_heads = ssm_group_count`, `d_conv = ssm_conv_kernel`.
2. **`Qwen35AttentionDims`**: `hidden`, `n_head`, `head_dim`,
   `n_head_kv` from `arch.config()`. Use the `rope_dimension_sections`
   when present (4-section MRoPE).
3. **`DeltaNetHybridCache::allocate(layer_kinds, kv_dim, d_conv,
   conv_dim, head_v_dim, n_v_heads)`** — `layer_kinds[i] =
   arch.is_linear_attention_layer(i)`, `kv_dim =
   attn_dims.kv_dim()`, `conv_dim = dn_dims.conv_dim()`.

### Loop

```rust
let mut cache = DeltaNetHybridCache::allocate(...);
let mut all_tokens = Vec::with_capacity(prompt_ids.len() + max_tokens);

// Prefill — feed every prompt token; only the final logits matter.
let mut last_logits = None;
for &tok in prompt_ids {
    last_logits = Some(qwen35_forward_step(
        tok, weights, &dn_dims, &attn_dims, &mut cache, 1e-6,
    ));
}

// Decode loop.
let mut detok = Detokenizer::new(tokenizer);
let mut out: Vec<(String, f64)> = Vec::with_capacity(max_tokens);
for _ in 0..max_tokens {
    let logits = last_logits.take().expect("prefill produced logits");
    let (tid, prob) = sample_from_logits(&logits, &sampling);
    let text = detok.push(tid);
    if eos.is_eos(tid, &text) { break; }
    out.push((text, prob));
    last_logits = Some(qwen35_forward_step(
        tid, weights, &dn_dims, &attn_dims, &mut cache, 1e-6,
    ));
}
GenerateResult { tokens: out, prompt_token_count: prompt_ids.len() }
```

### Glue dependencies

- `SamplingConfig` + `EosConfig`: re-use the existing
  `larql_inference::layer_graph::{SamplingConfig, EosConfig}` types.
- `Detokenizer`: re-use `larql_inference::tokenizer::Detokenizer`.
- `GenerateResult`: re-use the existing struct.
- `sample_from_logits`: extract from `generate_streaming` or
  re-implement as a small helper (greedy = argmax; sampling =
  top-k + top-p + temperature + sampler).

### Server-side wiring

Replace the guard in `chat.rs:739-769` (added by PR #170) with the
actual dispatch arm. Drop the `index` / `backend` / `cached_layers`
parameters — the qwen35 driver doesn't use them (CPU-only forward,
no KNN lm_head).

Add `qwen35_weights` to `LoadedModel` (or load on-demand per
request — re-loading 60 GB per chat is unworkable, so cache it on
first use behind an `OnceLock` keyed by vindex path).

### Estimated breakdown

| Sub-task | LoC | Risk |
|---|---:|---|
| 2c.a `qwen35_generate_with_sampling` body | ~120 | Medium (sampling + EOS + detok integration) |
| 2c.b `sample_from_logits` helper extraction | ~40 | Low (greedy is argmax; sampling is mechanical) |
| 2c.c `Qwen35Weights` caching in `LoadedModel` | ~40 | Low (`OnceLock` keyed by vindex path) |
| 2c.d Server dispatch arm replacing guard | ~15 | Low (mechanical replacement of PR #170's stub) |
| 2c.e Live HTTP smoke | ~50 | Low (curl against `larql serve`) |

Total ~265 LoC across 5 sub-tasks. The first runtime-meaningful
end state is HTTP 200 with non-empty completion text from
`/v1/chat/completions` against the v4 vindex.

## Out of scope (still)

- Step 2c — server dispatch routing `qwen35*` arches.
- Forward parity vs llama.cpp on vindex-loaded weights.
- 40 GB `gate_vectors.bin` size optimisation.
- `QuantBacking::Bytes` zero-copy variant (option B above).
- DeepSeek V4 Flash MLA — still parked.

## Reference paths

- Live vindex: `/tank/ai/Qwen/Qwen3.6-35B-A3B-vindex/`
- Source GGUF: `/tank/ai/Qwen/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`
- `Qwen35Weights` struct: `crates/larql-inference/src/attention/qwen35_forward.rs:98`
- `DeltaNetLayerWeights` struct: `crates/larql-inference/src/attention/deltanet_block.rs:53`
- `Qwen35MoeFfnWeights` struct: `crates/larql-inference/src/attention/qwen35_forward.rs:51`
- `QuantTensor::from_raw`: `crates/larql-models/src/quant/lazy.rs:81`
- `parse_layer_weights_header`: `crates/larql-vindex/src/format/weights/write_layers.rs:241`
- DeltaNet storage reader: `crates/larql-vindex/src/index/storage/deltanet.rs`
- Existing GGUF loader: `crates/larql-inference/src/attention/qwen35_load.rs:62`
  (`load_qwen35_weights`) — use as semantic reference for how each
  field is populated from a GGUF mmap; the vindex loader should
  produce structurally equivalent output.
