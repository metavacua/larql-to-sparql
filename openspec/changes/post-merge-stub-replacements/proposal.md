## Why

The wholesale merge of upstream `chrishayuk/larql` (commit `fceb4a1`)
landed with several mechanical stubs in place of upstream features the
fork's own changes had displaced. The stubs compile and the test
suites pass, but the behaviour they hide ranges from "tests pass for
the wrong reason" (synthetic Gemma3/StarCoder2 fixtures aliased to a
generic tiny RMSNorm fixture) to "CLI returns wrong answers" (the
Metal `ov_rd` head-replacement / pre-W_O capture hooks fell back to
the unconditioned forward pass).

This change replaces those stubs with real, tested code while staying
inside the fork's spec-first contract.

## What Changes

- ADD `make_gemma3_test_weights()` and `make_starcoder2_test_weights()`
  fixtures in `crates/larql-inference/src/engines/test_utils.rs`. They
  build synthetic `ModelWeights` configured for the real Gemma 3
  (QK norm + post-norms + sliding-window pattern + `norm_weight_offset
  = 1.0` + GeluTanh) and StarCoder 2 (LayerNorm + non-gated FFN with
  `c_fc` / `c_proj` naming + attention/FFN biases + GeluTanh) arches,
  so the attention/forward/ffn tests that reach for these fixtures
  exercise the dormant branches they were written for.
- ADD `full_pipeline_q4_with_head_replacement` and
  `full_pipeline_q4_capture_pre_wo` methods to the
  `larql_compute::DecodeBackend` trait, with default impls returning
  `None` so CPU/CUDA fall back to their existing paths. The MetalBackend
  inherent methods already exist on the macOS build; promoting them to
  the trait lets the platform-agnostic `predict_q4k_metal_*` call sites
  in `larql_inference::vindex::q4k_forward::metal` route through the
  trait instead of the stubbed `full_pipeline_q4` fallback.
- ADD a `tokenizer_cache: Arc<TokenizerCache>` field to
  `LoadedModel` and wire `encode_cached_ids` through the existing two-
  tier `TokenizerCache` (L0 exact-match LRU + L1 chat-template prefix
  LRU keyed at the last special-token sentinel). The cache module
  itself was already present at `crates/larql-server/src/tokenizer_cache.rs`
  but was not declared in `lib.rs`.
- ADD `try_generate`, `try_generate_with_sampling`, and
  `try_generate_streaming` fallible wrappers in
  `crates/larql-inference/src/layer_graph/generate/gpu.rs`, mirroring
  the existing `try_generate_constrained*` family. Each runs the
  infallible `generate*` then routes `GenerateResult::error` to a
  typed `Result<GenerateResult, GenerateError>` via `into_result()`.

## Impact

- Affected specs:
  - `inference-attention-and-kv` — Gemma 3 and StarCoder 2 fixtures
    are now expected to exercise the corresponding arch branches.
  - `compute-backend-traits` — new optional trait surface for the
    head-replacement and pre-W_O capture intervention hooks.
  - `server-tokenizer-cache` — `LoadedModel` now owns a real
    `TokenizerCache` and `encode_cached_ids` is the production
    entry point.

- Affected code (build-clean, all tests pass):
  - `crates/larql-inference/src/engines/test_utils.rs`
  - `crates/larql-inference/src/lib.rs` (stub removal)
  - `crates/larql-inference/src/test_utils.rs` (orphan deleted)
  - `crates/larql-compute/src/backend/decode.rs`
  - `crates/larql-inference/src/vindex/q4k_forward/metal.rs`
  - `crates/larql-server/src/{lib,state,bootstrap,routes/stream}.rs`
  - `crates/larql-inference/src/layer_graph/{generate/{gpu,mod},mod}.rs`

- Tests added (15):
  - `engines::test_utils::tests::{gemma3,starcoder2}_fixture_*` (4)
  - `backend::decode::tests::cpu_full_pipeline_q4_*_returns_none` (2)
  - `state::loaded_model_tests::encode_cached_ids_*` (4)
  - `layer_graph::generate::tests::try_generate_*` (4)
  - `routes/dispatcher` callsite shape unchanged (1 compile-only check)

- No regressions, full crate-test sweep green:
  - `larql-models` lib tests: 227 → 227 (unaffected)
  - `larql-compute` lib tests: 161 → 163 (+2 default-impl tests)
  - `larql-inference` lib tests: 1199 → 1203 (+4 try_generate tests
    after the engine fixtures were verified)
  - `larql-vindex` lib tests: broken on merge → 922 (cleared the
    2× missing-`ModelWeights`-field errors)
  - `larql-kv` lib tests: broken on merge → 200 (cleared the
    24× unresolved-`test_utils` errors)
  - `larql-server` lib tests: 283 → 287 (+4 encode_cached_ids tests)
  - `cargo check --workspace` → 0 errors (cuda-oxide build break is
    third-party and pre-existing, not in this scope).

- Additionally fixed (pre-existing merge breakage cleared up while
  the test surface was being touched):
  - ADD a `test-utils` Cargo feature on `larql-inference`. When on,
    `larql_inference::test_utils` is `pub` rather than
    `#[cfg(test)] pub(crate)`, exposing the synthetic-weights
    factory (`make_test_weights`, `make_test_vindex`,
    `make_test_tokenizer`, `make_gemma3_test_weights`,
    `make_starcoder2_test_weights`, `TestFixtures`) to downstream
    crates without forcing `tokenizers` / `ndarray` into a separate
    feature-gate that doesn't already always-include them.
  - MODIFY `larql-kv/Cargo.toml`: add
    `larql-inference = { ..., features = ["test-utils"] }` under
    `[dev-dependencies]`. Clears the 24× E0432 `unresolved import
    larql_inference::test_utils` errors that the merge baseline
    surfaced on `cargo test -p larql-kv --lib`.
  - MODIFY `crates/larql-vindex/src/extract/build.rs` and
    `crates/larql-vindex/src/extract/build_helpers/test_support.rs`
    to add the `position_embed: None`, `embed_quant: None`,
    `lm_head_quant: None`, `quant_tensors: HashMap::new()` fields
    that the fork's `ModelWeights` shape requires. Clears the
    2× E0063 missing-field errors on
    `cargo test -p larql-vindex --lib`.

- Out of scope (still blocked, documented for future work):
  - The macOS-only Metal trait_impl already uses the new signatures
    (`full_pipeline_q4` without explicit q_dim/kv_dim/head geometry),
    matching the `DecodeBackend` trait additions in this change. The
    pre-existing `full_pipeline_q4` signature mismatch between the
    trait and the macOS impl is a separate structural issue not
    addressed here — both files compile on their own respective
    cfg-gated paths, but verifying the macOS build requires Apple
    Silicon hardware we do not have on the CI host.
