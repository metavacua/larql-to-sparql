# Vindex v1 Conformance Contract

**Date:** 2026-07-31
**Status:** Implemented (structural + corruption contract); perf benchmark protocol is a documented follow-up
**Companion:** [`format-spec.md`](format-spec.md) (byte layouts), [`larql-vindex-spec`](../../larql-vindex-spec/SPEC.md) (manifest/provenance contract), [`operations-spec.md`](operations-spec.md)

This document is the *conformance* face of the format spec: what a
loader must do when handed a v1 vindex — especially a damaged one.
It assembles what already exists (the zero-dep spec crate, per-shard
digests, provenance hardening) with the corruption-test suite and the
byte-order golden vectors that pin the contract in CI.

The enforcing tests live in
`crates/larql-vindex/tests/conformance_v1_*.rs` plus the colocated
unit suites referenced per artifact below.

## 1. The artifact set

A v1 vindex is a directory. Core artifacts (per
[`format-spec.md`](format-spec.md) §3):

| Artifact | Role | Loader |
|---|---|---|
| `index.json` | manifest / `VindexConfig` (+ spec-crate `VindexManifest` view) | `format/load.rs::load_vindex_config`, `load_vindex` |
| `gate_vectors.bin` | W_gate, f32/f16 | `format/load.rs` (mmap) + `index/storage/gate_accessors`, `gate_store` |
| `down_meta.bin` | per-feature output metadata | `format/down_meta/read.rs` (`mmap_binary` primary, `read_binary` legacy) |
| `embeddings.bin`, `tokenizer.json` | embeddings / tokenizer | `format/load.rs` |
| `interleaved_kquant.bin` + `interleaved_kquant_manifest.json` | Q4_K/Q6_K FFN slab | `index/storage/ffn_store/interleaved_kquant.rs` |
| `down_features_kquant.bin` + manifest | feature-major down sidecar | same module |
| `attn_weights_kquant.bin` + manifest, `norms.bin`, `lm_head*.bin` | attention/norms/LM head | `index/storage/attn.rs`, `format/weights/load/` |
| `.vlp` patch files | JSON diffs with base64-LE embedded vectors | `patch/format.rs`, `patch/overlay_apply.rs` |
| `knn_store.bin` / `.lknn` | arch-B residual-key KNN store | `patch/knn_store_io.rs` |

The spec crate (`larql-vindex-spec`) additionally requires provenance
(`source` with `base_model_sha`, `base_safetensors_sha256`,
`extractor_sha`) and a `checksums` map covering every referenced
`.bin`; `VindexManifest::validate_self_consistency` rejects wrong
spec versions, ambiguous layer slots, and unchecksummed layer files.
Full digest verification is `larql verify`
(`format-spec.md` §11.2).

## 2. The error-not-panic guarantee

**Contract:** feeding corrupt bytes to any v1 loader yields exactly
one of:

1. **`Err`** (`VindexError::Parse`/`Io`; `.lknn` returns `String`) —
   for artifacts whose corruption makes the whole load meaningless;
2. **a documented safe decline** — `None` from a per-layer accessor,
   after which the walk ladder falls back to another FFN backend;

and **never** a panic, an out-of-bounds read, an
attacker-sized allocation, or silently wrong data.

Per artifact, with the enforcing code and the pinning tests:

### index.json (`conformance_v1_index.rs`)

- Malformed JSON, missing required fields, wrong field types, unknown
  `dtype`/`quant` tags → `Err` (serde; no defaults on structural
  fields).
- **Out-of-range `layers[].layer` → `Err`** — the 2026-07-30 H4 fix
  (`format/load.rs::check_layer_in_bounds`), pinned as contract by
  `out_of_range_layer_entry_is_error_not_panic` (previously a panic
  on the `gate_slices[layer]` write).
- The on-disk `version` field is parsed but not gated (loader-domain
  evolution); the spec-level gate is `vindex_spec_version` in the
  spec crate.

### gate_vectors.bin

- Layer geometry from `index.json` is validated lazily at every
  access: `gate_accessors::gate_vector`, `gate_store` re-check
  `byte_end <= mmap.len()` and return `None` — never slice
  unchecked. Q4-gate and attention views go through
  `mmap_storage.rs::checked_view` (checked add + length bound);
  corruption tests colocated in `mmap_storage.rs`
  (`*_rejects_out_of_bounds*`, `gate_q4_layer_data_rejects_zero_or_overflow`).

### down_meta.bin (`conformance_v1_down_meta.rs`)

- Both readers validate magic, version, and **every header count
  against the actual file size with checked arithmetic** before any
  allocation: truncated header/records, absurd `num_layers`/
  `num_features`/`top_k` (the legacy `read_binary` allocation bomb —
  §3 LOW, fixed 2026-07-31 in `format/down_meta/read.rs`), record-size
  overflow → `Err`.
- A present-but-corrupt `down_meta.bin` fails the **whole**
  `load_vindex` (no silent JSONL fallback).

### interleaved_kquant.bin + manifest (`conformance_v1_kquant.rs`)

- Manifest is typed-deserialised (`Q4kManifestEntry`; a missing
  `format` field is a parse error, not a silent default) and every
  format tag must exist in `quant::registry` → else `Err`.
- Declared byte ranges are re-checked at access time by
  `checked_view`: truncated slab, `offset`/`length` overflow, or a
  short manifest → the per-layer accessor **declines** (`None`) and
  the ladder falls back. This is the documented decline case: the
  slab loaders are best-effort inside `load_vindex`
  (`format/load.rs` discards their errors), so slab-level corruption
  degrades the backend rather than failing the load. Calling the
  loader directly surfaces the `Err`.
- **H1 padded-stride rule:** rows are padded to the 256-element
  super-block; `shape[1]` in the manifest is the *padded* width and
  decoders must stride by it (`kquant_decode.rs`, non-aligned
  fixture in `kquant_cache.rs::q4k_ffn_layer_decodes_row_padded_non_aligned_slabs`;
  writer-level pin `writer_pads_rows_to_super_block`).

### down_features sidecar (same test file)

- `.bin` without its manifest → `Err` (no stride fallback exists).
- Manifest entry without `shape[1]` → `Err`.
- Out-of-bounds ranges → per-layer decline.

### .vlp patches (`conformance_v1_patches.rs`)

- Corrupt JSON / unknown `op` tag → `Err` at `VindexPatch::load`.
- Embedded vectors are base64 over **explicit-LE** f32 bytes
  (`patch/format.rs`, M4 fix — no unaligned transmutes). Invalid
  chars, non-f32-aligned payloads, and truncated base64 (including
  the silently-droppable 4-char-chunk prefix) → `Err`.
- **All-or-nothing application** (M5 fix):
  `PatchedVindex::try_apply_patch` decode-validates every embedded
  vector before applying anything; one corrupt vector rejects the
  patch with zero overrides applied. Unit grid in
  `patch/format.rs` + `patch/overlay_apply.rs` tests.

### knn_store.bin / .lknn (`conformance_v1_patches.rs` + `patch/knn_store_io.rs` tests)

- Bad magic / unknown version / truncation anywhere → `Err`.
- Header counts are untrusted: capacity hints are bounded by the
  remaining bytes and `meta_len` is validated before its buffer is
  allocated (fixed 2026-07-31) — a corrupt count is a fast read
  error, not a multi-GB reserve.

### Vindexfile (build input, `vindexfile/parser.rs` tests)

Not an on-disk artifact of the vindex itself, but part of the same
conformance pass (audit §3 LOW): quote-aware tuple parsing
(`INSERT ("Acme, Inc", …)`), DELETE condition form requires all three
keys exactly once (missing/unknown/duplicate keys → `Err`), and
INSERT with no free feature slot is an error instead of silently
overwriting feature 0.

## 3. Byte-order contract (cross-platform)

All v1 binary formats are **little-endian on disk**, written and read
via explicit `to_le_bytes`/`from_le_bytes` (or codecs built on them:
`format/le_floats.rs`, the `.vlp` base64 embedding, `down_meta`,
`.lknn` f16 keys). Decoders must also tolerate **unaligned** source
buffers (mmap slices, base64 payloads) — no `&[u8] → &[f32]`
reinterpret casts.

Round-trip equality cannot catch an endianness regression (a
consistently-BE pair round-trips fine), so
`conformance_v1_golden_le.rs` pins the **exact bytes** of each codec
against hand-written golden vectors: `le_floats`, `.vlp` base64
(`[1.0] → "AACAPw=="`), a complete golden `down_meta.bin`, and a
complete golden `.lknn`. Any target producing different bytes fails
these tests. No big-endian CI runner exists (all supported targets
are LE); the golden vectors are the standing guard.

## 4. Benchmark protocol

What conformance SHOULD cover beyond correctness:

- **Numerical parity (exists):** the walk-vs-dense parity suite —
  ROADMAP item 20, 2026-07-30 — pins walk output against dense
  `WeightFfn` for the gemv/exact/full-mmap/interleaved paths
  (`larql-inference/src/vindex/walk_ffn/{exact,full_mmap,sparse_gemv}.rs`
  et al.), plus base+delta and shortlist parity tests. Correctness
  claims about a v1 vindex backend route through that suite.
- **Performance protocol (documented follow-up, not yet built):** a
  standard perf harness per artifact/backend — tok/s and per-layer
  walk latency at fixed K on pinned fixtures (dense vs walk vs Q4K
  paths), run on a thermally sane machine (see the standing
  thermal/power caveats) with the spin-pool defaults recorded. Until
  it exists, perf claims in READMEs must carry the
  HNSW-vs-brute/which-N qualifiers (audit item 23). This document
  deliberately does not fake numbers for it.

## 5. Known residual gaps (tracked)

Recorded here so the contract doesn't overclaim; all are outside the
corruption classes pinned above:

- `format/load.rs` best-effort loaders (`let _ =`) mean slab-level
  corruption degrades silently at `load_vindex` granularity — use
  `LARQL_VINDEX_DESCRIBE=1` to surface the chosen backend.
- Duplicate `layers[].layer` entries in `index.json` are not
  deduplicated; `synthesize_gate_from_q4k` can still panic on a
  crafted duplicate-layer manifest.
- `read_q4k_manifest` does not cross-check declared `length` against
  `expected_bytes(shape)` the way the attention loader does.
- `weight_manifest.json` offsets in `format/weights/load/{f32,q4k}.rs`
  use unchecked adds.
