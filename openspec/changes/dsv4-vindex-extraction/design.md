# Design — dsv4-vindex-extraction

## Context (from the gap analysis)

- **Extraction entry**: `larql-vindex/src/extract/build.rs` (`build_vindex`),
  6 stages, generic loop over layers calling `arch` methods + a
  `dyn WeightSource`. MoE handled via `arch.is_moe()` + per-expert keys.
- **Format**: `larql-vindex/src/format/filenames.rs` (file constants);
  attn in `attn_weights_q4k.bin` (Q4_K Q/K/O, Q6_K V); MoE experts +
  shared-expert + lm_head shards; `index.json` config.
- **Config**: `VindexModelConfig` (`larql-vindex/src/config/model.rs`) —
  attention/MoE/Gemma/DeltaNet fields; no DSv4 concepts.
- **Capabilities**: `format/weights/capabilities.rs` rejects `uses_mla()`
  (V2/V3); V4 unhandled.
- **Arch trait**: `ModelArchitecture` (`larql-models/src/config.rs`) —
  tensor-key methods + behavior flags. `DeepSeekV4Arch` is a ~43-line
  stub.
- **DSv4 tensors**: `larql-models/src/architectures/deepseek_v4_tensors.rs`
  (37 `DsV4TensorKind`s) — the authoritative key set, already used by the
  GGUF inference loaders.

## Goals / Non-Goals

**Goal**: produce a vindex from a DSv4-Flash GGUF that faithfully carries
every weight DSv4 inference needs, in a form a future DSv4 vindex *reader*
(separate change) can load into the existing `DsV4LayerWeightStorage` /
resident path.

**Non-goals**: the vindex *reader* / serving (separate change — this is
extraction only); precomputing indexer top-k masks (runtime-dynamic);
SVD/clustering compression of DSv4 weights (store Q4_K as-is, mirroring
the GGUF); changing any non-DSv4 arch.

## Decisions

### D1 — Reuse the GGUF tensor schema as the extraction source of truth

`deepseek_v4_tensors.rs::DsV4TensorKind` + `tensor_name_of` already map
every DSv4 weight to its GGUF name and are battle-tested by the inference
loaders. Extraction reads via the same names rather than inventing new
key methods on `ModelArchitecture`. `DeepSeekV4Arch` gains only the
methods the generic pipeline strictly needs (family, is_moe, expert
counts, the MoE/shared keys); the DSv4-specific tensors are written by a
**dedicated DSv4 extraction module**, not the generic Q/K/V/O writer.
**Alternative considered**: extend `ModelArchitecture` with ~15 DSv4
key-methods so the generic pipeline handles it — rejected: pollutes the
trait with arch-specific surface and still can't represent grouped-O /
HCA structurally.

### D2 — New DSv4 weight files, Q4_K passthrough (no recompression)

DSv4's GGUF weights are already Q4_K/Q6_K. Extraction copies the
quantized bytes into DSv4 vindex shards (mirrors the resident loader's
"keep the bytes" approach), not dequant+recompress. New files:
- `dsv4_attn.bin` (+ manifest): low-rank Q (`q_a`,`q_b`), latent KV
  (`kv_latent`), grouped O (`output_a`,`output_b`), + the small f32
  norms/sinks inline.
- `dsv4_hca.bin`: compressor (`attn_compress_*`) + indexer
  (`indexer.*`), per layer, gated by `compress_ratio`.
- `dsv4_mhc.bin`: the `hc_attn/ffn/head_*` bookends.
- MoE experts/shared-expert/router → existing generic shards.

### D3 — Per-layer variant metadata in the config

`VindexModelConfig` gains a `dsv4: Option<DsV4VindexMeta>` (n_hc, indexer
head dims, FP8-KV flag, YARN config); `VindexLayerInfo` gains
`compress_ratio: u8` (0/1/4) so the reader dispatches the right attention
variant per layer. Other arches leave these `None`/0.

### D4 — Capabilities: allow V4 extraction

`capabilities.rs` distinguishes V4 from V2/V3: V2/V3 are rejected as
classic MLA; V4 is allowed (its weights are extractable; the
attention-variant dispatch is the reader's concern). Add a V4 branch +
the supported extract levels.

### D5 — Round-trip is the correctness anchor

Extraction has no model output to validate against on its own; the gate
is a **byte/shape round-trip**: every DSv4 tensor read from the GGUF
appears in the vindex with the expected shape + quant type, and a
reload-from-vindex (a thin test reader, ahead of the full serving reader)
reconstructs `DsV4LayerWeightStorage` whose weights equal the GGUF-loaded
ones. Full output parity comes with the serving-reader change.

## Risks / Unknowns (front-loaded into Phase 1)

1. **Grouped-O + low-rank/latent attention storage** (D2 `dsv4_attn.bin`)
   — the single biggest net-new piece; Phase 1 proves it round-trips
   before the rest.
2. **HCA/indexer/mHC manifest design** — ensure the format cleanly
   represents per-layer-variable tensors (compress layers vs indexer
   layers vs none).
3. **FP8 KV** — decide store-FP8-bytes vs dequant; affects the reader.
4. **Re-evaluation gate**: after Phase 1 (config + attn round-trip), the
   remaining ~600 LoC (HCA/indexer/mHC + reader) can be weighed against
   the GGUF-direct alternative (#392) with real data on the effort.
