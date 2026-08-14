# Roadmap — larql-vindex

For shipped work, see [CHANGELOG.md](CHANGELOG.md).

## Current state (verified 2026-08-04)

**Tests.** 1298 lib tests passing (`cargo test -p larql-vindex --lib`).
Crate-local checks are wired through `make larql-vindex-ci`: fmt, clippy
`-D warnings`, tests, example compile checks, bench compile/tests, and coverage
policy. All four self-contained examples (`mmap_demo`, `demo_memit_solve`,
`q4k_demo`, `walker_demo`) run end-to-end; the rest compile clean via
`cargo check --examples`.

**Layout.** 196 source files across `cache`, `clustering`, `config`, `engine`,
`extract`, `format`, `index`, `patch`, `quant`, `trie`, `vindexfile`, `walker`.
The decomposition history — which round split which monolith — is in
[`CHANGELOG.md`](CHANGELOG.md); the invariants that came out of it are:

- `VectorIndex` is four typed substores (`GateStore`, `FfnStore`,
  `ProjectionStore`, `MetadataStore`), so adding a field is one edit in one
  store.
- `index/core/` and `index/types/` hold one capability impl / one trait per
  sibling file.
- `extract/streaming/stages/` holds one extraction stage per sibling file.
- `format/weights/write_q4k/` holds one emitted artefact per sibling file.

**Storage formats — 5**: f32, f16, Q4_0, Q4_K/Q6_K (Ollama-compatible), Q8,
FP4/FP8. Quant dispatch runs through `quant::registry`, so adding the next
K-quant is one table entry plus codec functions (~3-file edit). Block sizes
flow through `larql_models::quant::ggml::K_QUANT_BLOCK_ELEMS`.
`LEGACY_BLOCK_Q4_K_STRIDE` names the 148-byte historical bug shape.

**Runtime behaviour.** Mmap zero-copy with adaptive residency. HNSW graph index
wired into `gate_knn` (opt-in via `--hnsw`). Q4_K dequant cache LRU-bounded via
`--max-q4k-cache-layers`. Patch system for editable knowledge
(`PatchedVindex` overlay).

**Extraction.** Streaming extract checkpoints + auto-resume — phase-level
progress recorded to `.extract_checkpoint.json`; gate and down_meta phases
auto-skip on a compatible checkpoint. Vindexfile `FROM hf://…` resolves through
the same resolver `larql run` and `larql extract` use. Stage labels are
centralised in `extract::stage_labels` (15 labels; a typo at any site is a
compile error).

**Constants.** Filename literals live in `format::filenames` (252+ occurrences
collapsed to one constant module). `DEFAULT_C_SCORE` is lifted onto
`index::types` so the patch overlay fallback and the vindexfile builder share
one default.

**Coverage.** Enforced: 90% per file and 90% aggregate, with 45 debt baselines
that may only ratchet upward. `make larql-vindex-coverage-summary` /
`make larql-vindex-coverage-html` (cargo-llvm-cov) enforce both.

**CI.** `.github/workflows/larql-vindex.yml` runs format, check, examples,
clippy, tests, and bench compile/tests on Linux, Windows, and macOS. Coverage
policy runs on Ubuntu. `LARQL_EXTRACT_STRICT=1` is set in this workflow, so an
unrecognised checkpoint tensor fails the build. The bench rig is daemon-aware
(`make bench-vindex-scaling` refuses if `larql-server` / `larql-router` are
running on the host).

**Large-file debt — re-measured, and the list has changed.** Five files are now
≥800 LOC:

| File | LOC | Note |
|---|--:|---|
| `format/huggingface/download/mod.rs` | 1329 | Grew and became a directory since the last count |
| `index/storage/vindex_storage/mmap_storage.rs` | 1187 | From the `VindexStorage` migration |
| `patch/overlay.rs` | 1071 | **New to this list** |
| `extract/build/mod.rs` | 862 | |
| `format/weights/write_f32_tests.rs` | 815 | Test file |

Six files previously tracked here have since been split and are gone from the
list: `walker/vector_extractor.rs`, `format/huggingface/publish/lfs.rs`,
`index/types/ffn_row.rs`, `extract/build_helpers.rs`,
`index/storage/gate_accessors.rs`, `format/huggingface/discovery.rs`. Tracked
under P1 "Residual large-file debt" below.

---

## Open defects

Both raised by the 2026-05-28 whole-codebase review; neither is fixed.

- **P1 — NaN `partial_cmp().unwrap()`** at `router:107`, `lm_head:322`,
  `gate_store:330`. Route through a shared NaN-safe top-K/sort helper. This is
  a workspace-wide cleanup (~10 sites across vindex/core/cli/python) —
  `larql-core` has five and `larql-cli` one.
- **Low — implicit 4-byte alignment** in the `*const f32` reinterprets in
  `decode_floats` / `decode_gate_vector`; the invariant is enforced only by
  caller offset arithmetic, not by the helper.

---

## P0: Active

### Modularity + magic-literal debt

**Status**: Mostly closed. Only the large-file decomposition bullet
remains open as of 2026-05-09.

Closed during the 2026-05-09 review (verified against the tree, not
just the audit doc):

- [x] Architecture-specific extraction literals — dense clustering
  routes through `LayerBands::for_family(...).knowledge` via
  `extract::build::knowledge_layer_range`. No hard-coded layer ranges
  remain in the extraction pipeline.
- [x] Vindex file layout literals — production paths fully routed
  through `format::filenames`. The 2026-05-09 sweep added
  `ROUTER_WEIGHTS_BIN` (was the last stray production literal in
  `extract/streaming.rs`). Test fixtures keep literals deliberately to
  pin the wire contract.
- [x] Stringly typed `ffn_layout` — already a typed
  `Option<FfnLayout>` enum (`config/index.rs`).
- [x] Algorithm parameters lifted — extraction batch sizes,
  relation-cluster cap, k-means iterations live in
  `extract::constants`; HNSW build parameters in
  `config::hnsw::HnswBuildConfig::{LAYER, EXPERT}`.
- [x] `GateIndex` split — narrower capability traits (`GateLookup`,
  `PatchOverrides`, `NativeFfnAccess`, `QuantizedFfnAccess`,
  `Fp4FfnAccess`, `FfnRowAccess`) with `GateIndex` retained as the
  compatibility composition for existing trait-object consumers.

Still open:

- [x] Large-file decomposition — **closed 2026-05-09**. Last file
  (`format/weights/write_q4k/mod.rs` 734 L) split into one sibling per
  emitted artefact (attn / ffn / moe_layers / norms / ple / lm_head),
  orchestrator down to 318 L. Closed today: `publish.rs` (997),
  `streaming.rs` (832), `load.rs` (817), `core.rs` (755), `types.rs`
  (715), `streaming/stages.rs` (644), `write_q4k/mod.rs` (734). No
  non-test file in vindex now exceeds the soft 600-LOC threshold.

**Acceptance bar:** no remaining production filename/layout magic
strings for vindex-owned files (met), extraction remains model-family
agnostic (met — see P1), `GateIndex` split into narrower capability
traits (met), and no new module grows past the current large-file
debt without a split plan.

## P1: Active

### Architecture-independent extraction and weight writing

**Status**: Closed for current architectures (2026-05-09). Reopen when a
non-standard attention contract (MLA, MQA-with-shared-rotary, etc.) is
landed in `larql-models` and needs writer support.

The extraction stack now preserves architecture facts from
`ModelArchitecture` end-to-end, and a single capability helper gates
unsupported attention layouts before any output is written.

Work items:

- [x] Audit f32/Q4K writer entry points and loader surfaces for implicit
  standard-attention assumptions. Both writers (`write_model_weights` and
  `write_model_weights_q4k`) call `ensure_standard_attention_supported`
  on entry; the check lives in one place at
  `format/weights/capabilities.rs`.
- [x] Replace `extract/build_from_vectors.rs` model-name heuristics —
  audit (2026-05-09) found no `contains("gemma")` / `contains("llama")`
  string heuristics remain. The path routes through `arch.family()` and
  `LayerBands::for_family`.
- [x] Add an architecture capability check **before any partial write**.
  Added `ensure_extract_level_supported` (2026-05-09) wired into both
  `build_vindex` and `build_vindex_streaming`. Browse-level extracts of
  MLA architectures still succeed (no attention written); Attention /
  Inference / All tiers fail with a targeted `UnsupportedArchitecture`
  error before the output directory is created.
- [x] Centralise remaining protocol-like tensor/manifest tags. Quant tags
  flow through `quant::registry`; file-kind strings through
  `format::filenames`; capability surfaces through
  `SURFACE_F32_WEIGHT_WRITER` / `SURFACE_Q4K_WEIGHT_WRITER` /
  `SURFACE_EXTRACT_PIPELINE` constants.
- [ ] Extend f32/Q4K weight writers beyond standard Q/K/V/O when a concrete
  non-standard architecture contract is added. Won't fix until a
  `larql-models` MLA implementation lands.
- [x] Tests proving unsupported attention layouts are rejected before any
  partial vindex write — `build_inference_rejects_mla_before_writing`
  asserts `read_dir(output_dir).is_empty()` after the failure;
  `extract_level_*_rejects_mla` cover Attention/Inference/All and
  `extract_level_browse_passes_for_mla` covers the no-attention path.
- [x] Fixture tests that prove unknown/custom families do not inherit
  Gemma/Llama defaults through string matching — extended
  `unknown_family_does_not_inherit_known_bands_by_string_prefix` in
  `config::compliance` to compare lookalikes (`gemma3-clone`,
  `llamafied`) against canonical bands and prove the layer-count
  fallback is structurally distinct.

Acceptance bar (met 2026-05-09): vector-only and model-backed extracts
agree on family, embedding scale, layer bands, and required tensor
coverage; unsupported attention layouts fail before any file is written;
no string-prefix inheritance of curated band tables.

### Residual large-file debt — reopened 2026-08-04
**Impact**: Code navigability + future split cost
**Effort**: Small–Medium per file
**Status**: Reopened. The round-6 pass closed this on 2026-05-10 with "no file
outside the won't-split list is ≥800 LOC" — that is no longer true. See
[`CHANGELOG.md`](CHANGELOG.md) 2026-05-10 for what was split and why
`mmap_storage.rs` was deliberately kept whole.

Four files now breach, three of them new:

- [ ] `format/huggingface/download/mod.rs` — **1329 L**, up from the 676 L it
  was left at by the round-6 two-way split. The largest regression, and the
  same file the round-6 entry already flagged as "still over 600 LOC by user
  direction". It is network-bound and hard to mock, which is why it carries a
  64% coverage baseline; a split should separate the pure helpers from the HF
  API surface rather than slicing by line count.
- [ ] `patch/overlay.rs` — **1071 L**, new to the list. Grew through the
  2026-05-16 `gate_knn` optimization pass, which added the LayerGateCache code
  paths.
- [ ] `extract/build/mod.rs` — **862 L**, new to the list.
- [ ] `format/weights/write_f32_tests.rs` — **815 L**, new to the list. A test
  file, so the navigability cost is real but the coverage risk is not; lowest
  priority of the four.

`index/storage/vindex_storage/mmap_storage.rs` (1187 L) remains on the
won't-split list by the original 2026-05-10 reasoning — a 12-method trait impl
that fragments without modularity gain. Nothing here changes that.

**Why it regressed**: the acceptance bar was a one-time measurement, not a
gate. If this matters, the fix is a CI check that fails on a new file ≥800 LOC
outside a declared exemption list — otherwise it will regress again.

### HF LFS multipart upload for files >5 GB
**Impact**: `larql publish` fails on any vindex with a single file >5 GB
(Granite 4.1 30B `gate_vectors.bin` is 16 GB, `interleaved_kquant.bin`
is also 16 GB). Today users have to drop down to a Python escape hatch
(`huggingface_hub.HfApi().upload_folder(...)`) for the big files.
**Effort**: 1–2 days
**Status**: Pending (2026-05-17)

`format/huggingface/publish/lfs/stream.rs::stream_put_with_progress`
does a single `reqwest::blocking::Body::sized(...)` PUT to the
batch-returned signed URL. HF's LFS batch endpoint refuses single-PUT
on files >5 GB with `400 Bad Request: "You need to configure your
repository to enable upload of files > 5GB"`. The fix is to extend the
LFS protocol implementation to honour the multipart-response shape:

1. **Batch request** — keep as-is, but pass the `transfer` flag for
   multipart in the request body when local size exceeds the threshold
   (HF's `lfs/objects/batch` response then carries an `actions.upload`
   object with `parts: [{href, headers, ...}]` instead of a single
   `href` + `header`).
2. **`upload.actions.upload.parts` parsing** — extend
   `lfs/batch.rs::parse_batch_response` to handle both shapes (single
   `href` for ≤5 GB, parts list for larger). Map into a new
   `UploadAction::Multipart { parts: Vec<PartUrl> }` variant.
3. **Chunked streaming PUT** — in `lfs/stream.rs`, when the action is
   `Multipart`, stream the file in `part_size`-byte chunks (typically
   the per-part size HF returns is 5 GB; AWS S3 multipart max is also
   5 GB per part / 10 000 parts), PUT each part with its signed URL
   concurrently (parallelism = 3–4 to match the HF Xet client),
   collect the response `ETag`s.
4. **Multipart complete** — POST the collected ETags to the
   `complete` URL HF returns in `actions.upload.completionUrl` (or
   equivalent in the batch response — confirm exact JSON shape from
   `https://github.com/git-lfs/git-lfs/blob/main/docs/api/batch.md`'s
   multipart extension).
5. **Verify + commit pointer** — unchanged.

Acceptance bar:
- `larql publish output/granite-4.1-30b-q4k.vindex --repo
  chrishayuk/granite-4.1-30b-q4k-vindex --slices none` completes
  without the manual Python escape hatch.
- Existing single-PUT path stays bit-identical for files ≤5 GB
  (parity test on a synthetic 100 MB vindex).
- Resume-on-retry: if a part PUT fails after the batch step, retry
  that part only (current single-PUT code already restarts the whole
  file).
- Progress callbacks fire per-part so `PublishCallbacks` sees the
  same `bytes_sent` granularity it does today.

References:
- HF LFS batch endpoint docs:
  https://github.com/git-lfs/git-lfs/blob/main/docs/api/batch.md
- `huggingface_hub` Rust SDK doesn't exist; the Python SDK switched
  to Xet (block-level CAS, 64 MB xorbs, global dedup) — a future
  "Option B" once `xet-core` is a polished crates.io dep, but LFS
  multipart is the right portable next step.
- Diagnosed and worked around 2026-05-17 during Granite 4.1 30B
  publish: 3B + 8B uploaded fine through the existing path; 30B's
  16 GB `gate_vectors.bin` tripped the limit. Resume command in
  conversation history; xet-staged chunks in
  `~/.cache/huggingface/xet/` already content-addressed so the
  manual Python resume picks up where the failed `larql publish`
  left off.

### Coverage round-7 (review finding 2026-05-10, re-checked 2026-08-04)
**Impact**: Per-file ratchet — files below the 90% default
**Effort**: Small per file
**Status**: Active, and the count has moved the wrong way

The debt list was 40 entries when this was raised; `coverage-policy.json` now
carries **45**. The aggregate floor is met (enforced at 90%), so the ratchet is
holding the line but not advancing it.

Highest-leverage targets, taken from the current policy file rather than the
2026-05-10 snapshot:

- [ ] `format/weights/write_kquant/moe_layers.rs` — **34.0%**, still the single
  biggest tractable win. (This file was listed here as
  `write_q4k/moe_layers.rs`; the directory was renamed, the debt was not paid.)
- [ ] `quant/convert_q4k.rs` — 55.0%
- [ ] `format/weights/load/q4k.rs` — 57.0%
- [ ] `config/compliance.rs` — 57.9%
- [ ] `format/weights/write_f32.rs` — 58.0%
- [ ] `format/weights/write_kquant/norms.rs` — 60.0%
- [ ] `format/load.rs` — 62.0%
- [ ] `engine/core.rs` — 82.0%

**Acceptance bar**: each listed file moves to ≥90% line coverage or carries a
documented rationale (e.g. error-path branches that require a real S3 outage to
exercise).

### Cached layer decode for template-fixed layers (L0–12) — parked
**Impact**: 155+ tok/s decode (skip 13 of 21 layers)
**Effort**: Medium
**Status**: ⏸ Parked — depends on upstream work that isn't ready yet.
Don't start until the prerequisite lands. Keep `CachedLayerGraph` in
`larql-inference` as the integration point.

### Layer-level resume within an incomplete phase
**Impact**: A run interrupted at gate-layer-30-of-34 today re-runs
all 34 layers; layer-level resume would skip 30
**Effort**: Medium
**Status**: Forward-looking — phase-level resume now in place
(2026-04-25 round-3); the layer-level extension needs mid-phase file
truncation to the last clean layer boundary, which is more delicate
than the phase flag.

## P2: Forward-looking

### Expert-level sharding protocol
**Impact**: Unlocks > 256-expert MoE sharding-within-layer
**Effort**: Medium
**Status**: Forward-looking

Today `larql-router` shards by layer, not by expert ID within a
layer. For DeepSeek-V4-class models (1K+ experts) experts need to
shard across servers. Add an `ExpertRoute` message type to
`larql-router-protocol` and wire `GridState` dispatch.

### Q5_K / Q3_K / BF16 quant additions
**Effort**: Small per format (≈ 3 files thanks to the registry)
**Status**: Not yet needed — add when a target model demands it

Path: implement codec functions in `larql-models/src/quant/ggml/`,
add one entry to `QUANT_FORMATS` in `quant::registry`, add match arm
in `larql-compute::backend::quant_matvec`. Verified by the round-2
audit.

### Multi-model vindex
**Status**: Research

Store features from multiple models in one vindex. Compare
representations across architectures.

### Incremental extraction
**Status**: Research

Add new layers / features to an existing vindex without full rebuild.

---

## Won't fix

- **`detect.rs` (1391 L) split** in `larql-models` — cohesive single
  entry point dispatching to 12 architectures. Splitting fragments
  without modularity gain. Reconsider when a second detection system
  emerges (auto-discovery from model ID, multi-modal config).

---

## History

Completed entries previously kept here have been moved to
[`CHANGELOG.md`](CHANGELOG.md), reverse-chronological by date. Active
P0/P1/P2 items above; once a row lands it migrates to the changelog.
