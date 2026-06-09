## Context

DSv4-Flash inference today (`crates/larql-inference/src/attention/dsv4_*`)
dequantizes every weight to f32 at load time. `DsV4LayerWeightStorage`
(`dsv4_storage.rs`) and `FfnStorage` hold `Array2/Array3<f32>`. The
GGUF reader (`dsv4_gguf_reader.rs::read_dsv4_layer_tensors_from_gguf`)
reads the raw Q4_K bytes and immediately calls
`larql_models::quant::ggml::dequantize()` to produce `Vec<f32>`, which
`dsv4_storage_build.rs::build_layer_storage` reshapes into the arrays.
Because the f32 expansion is ~1.1 TB for the full model, the only
viable forward is `dsv4_streaming_model_forward_cached`, which loads +
dequantizes each layer per token and drops it.

The completed GPU push (PRs #339–#367) threaded
`Option<&dyn ComputeBackend>` through every matmul. The 2026-05-25
RTX 4090 bench (`dsv4_bench_cpu_vs_cuda`) showed 0.98× — the streaming
per-token dequant dominates; matmul placement is irrelevant.

qwen35 already solved the same problem with `QuantTensor`
(`larql-models/src/quant/lazy.rs`): weights stay quantized
(`Arc<[u8]>` or `Arc<Mmap>`); `matvec` / `matmul` run per-row fused
Q4_K×Q8_K dot kernels with no full dequant; `expert_slice` gives
zero-copy MoE expert views. qwen35 runs 35B-A3B at ~14.7 tok/s with
this.

## Goals / Non-Goals

**Goals:**
- Hold DSv4 large matmul weights as resident `QuantTensor`s (~161 GB,
  fits a 256 GB host) instead of f32 (~1.1 TB).
- A resident forward that loads weights once, eliminating per-token
  reload/dequant — the actual decode speedup.
- Preserve the existing f32 path as a fallback and as the streaming
  path for model-exceeds-RAM.
- Enable FFN-on-CPU / attention-on-GPU within one process.

**Non-Goals:**
- vindex conversion / on-disk format change. v1 builds `QuantTensor`
  directly from GGUF bytes. (Zero-copy `from_mmap_region` is a later
  optimization.)
- A new quantized matvec CUDA kernel. FFN runs on CPU via the existing
  `QuantTensor` kernels; attention runs on GPU via the existing f32
  `dot_proj_gpu` after dequant (or stays f32-resident on device).
- Changing the HCA / mHC / indexer math. Only the weight
  representation and matmul dispatch change.

## Decisions

**D1 — Dual storage (`Option<QuantTensor>` + f32), not a hard swap.**
Mirror qwen35's `ffn_gate_quant: Option<QuantTensor>` + `ffn_gate:
ArcArray2` pattern. Rationale: keeps every existing test green
(f32 path untouched), lets unsupported tensor formats fall back, and
makes the port incrementally mergeable. Alternative (replace f32
outright) would break the streaming path and every existing
real-GGUF test at once.

**D2 — Build `QuantTensor` from GGUF bytes directly, skip vindex.**
`QuantTensor::from_raw(bytes, tensor_type, rows, cols)` accepts the
GGUF tensor bytes the reader already has. Rationale: DSv4 has a
working GGUF reader; vindex adds a second on-disk format + a
conversion step for no v1 benefit. The mmap zero-copy win
(`from_mmap_region`) can come later without changing the forward.

**D3 — Per-callsite dispatch helper.** Introduce a small dispatch
(`quant ? qt.matmul(x) : dot_proj_gpu(x, w, backend)`) at each DSv4
matmul site rather than overloading `dot_proj_gpu` to accept
`QuantTensor`. Rationale: `dot_proj_gpu` lives in `larql-compute` and
is f32-only by contract; `QuantTensor` lives in `larql-models`.
Keeping the dispatch in `larql-inference` (which depends on both)
avoids a layering inversion.

**D4 — Resident forward as a new entry point, streaming retained.**
Add a non-streaming forward that takes pre-built resident layer
storage; keep `dsv4_streaming_model_forward_cached` for
model-exceeds-RAM. Rationale: the resident path is the speedup but
assumes the quantized set fits RAM; not every host can.

**D5 — Phasing (P1–P4), each independently mergeable.**
- **P1** dual storage + loader: add `Option<QuantTensor>` fields;
  loader builds them; f32 fallback retained; forward still uses f32.
  No behavior change; lands green.
- **P2** quant-aware dispatch: per-callsite `quant ? qt : f32`. FFN
  now runs on resident Q4_K on CPU. Tolerance parity test vs f32.
- **P3** resident forward: load all layers once; bench. Speedup here.
- **P4** hybrid GPU: attention → `Some(cuda)` (weight cache from
  PR #368 earns its keep), FFN → `None`/CPU.

**D6 — Tolerance-based parity, not bit-exact.** Per-row Q4_K×Q8_K
dot ≠ f32 dequant + BLAS reduction order. Add a relative-tolerance
parity test (qwen35 precedent) and a greedy-token-equality test.

## Risks / Trade-offs

- **Quant formats DSv4-Flash actually uses** → confirm every tensor is
  Q4_K/Q5_K/Q6_K/Q8_0/F32 (the formats `QuantTensor` dispatches). Any
  MXFP4 or unusual format must take the f32 fallback (D1 covers this);
  audit in P1.
- **Numeric drift breaks a downstream test that assumed f32** →
  mitigate with tolerance-based parity (D6); run the existing
  cached/prefill equivalence tests under the quant path with relaxed
  tolerance.
- **Resident RAM assumption** → 161 GB needs a ≥192 GB host; on
  smaller hosts the resident forward is unusable. Mitigation: keep the
  streaming path (D4); gate the resident path on an explicit caller
  choice, not a silent default.
- **MoE `expert_slice` semantics** → verify the DSv4 expert tensor
  layout (`[n_expert, n_ff_exp, n_embd]`) maps to `expert_slice`'s
  expected `[n_expert*rows, cols]` packing; may need a reshape/adapter.
- **P4 attention-on-GPU still pays htod for the input + dtoh for the
  output per call** → the PR #368 weight cache removes the weight
  htod; input/output are small. Acceptable; revisit if profiling says
  otherwise.

## Migration Plan

- P1 is additive (new fields default `None`); no migration.
- P2/P3 add new forward entry points; existing `dsv4_generate` /
  streaming callers keep working. A feature flag or explicit
  constructor selects resident-quant vs streaming-f32.
- Rollback: each phase is a standalone PR; reverting P-n leaves P-(n−1)
  working.

## Open Questions

- Does DSv4-Flash use any tensor format outside Q4_K/Q5_K/Q6_K/Q8_0?
  (Audit the GGUF tensor types in P1.)
- For attention-on-GPU (P4): dequant-to-device-once (f32 resident on
  GPU, use PR #368 cache) vs a quantized device matvec? Start with
  dequant-to-device — simpler, attention weights are only a few GB.
- Should the resident forward own the `QuantTensor`s or borrow them
  from a model-level store shared across requests? (Server concern;
  defer to P3.)
