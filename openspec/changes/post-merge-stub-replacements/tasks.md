## 1. Synthetic Gemma 3 / StarCoder 2 fixtures

- [x] 1.1 Add `rand_mat_seeded` helper that draws f32 values in
      `[-scale, +scale]` from an LCG seeded by a caller-supplied u64,
      so the new fixtures get independent random matrices without
      sharing the existing `make_test_weights` `rng_state`.
- [x] 1.2 Implement `make_gemma3_test_weights()` that routes through
      the `Gemma3Arch` impl: `attn_q_norm_key` / `attn_k_norm_key`
      vectors at `[HEAD_DIM]`, pre/post FFN norm vectors, all four
      layer norms per layer with `norm_weight_offset = 1.0` semantics
      (saved weights are deltas — zeros = unit norm at runtime),
      `residual_multiplier = 0.5` to exercise the non-1.0 branch in
      `run_ffn`.
- [x] 1.3 Implement `make_starcoder2_test_weights()`: LayerNorm with
      `norm_weight_offset = 0`, non-gated FFN with `c_fc` / `c_proj`
      naming via `arch.ffn_up_key` / `arch.ffn_down_key`, attention
      Q/K/V/O biases, FFN up/down biases, `attention_multiplier = 2.0`
      to exercise the non-1.0 attention scaling branch.
- [x] 1.4 Add unit tests verifying each fixture has the right shape
      and arch-discriminant keys.
- [x] 1.5 Remove the stub aliases from `crates/larql-inference/src/lib.rs`
      (the `pub(crate) mod test_utils { ... }` block that re-exported
      `make_gemma3_test_weights` / `make_starcoder2_test_weights` as
      passthroughs to the generic `make_test_weights`).
- [x] 1.6 Delete the orphan `crates/larql-inference/src/test_utils.rs`
      (it was never wired into `lib.rs` and held stale `ModelWeights`
      shape without the `embed_quant` / `lm_head_quant` /
      `quant_tensors` fields).

## 2. Metal Q4K head-replacement / pre-W_O capture trait surface

- [x] 2.1 Add `full_pipeline_q4_with_head_replacement` to the
      `DecodeBackend` trait with the same signature as the macOS
      `MetalBackend` inherent method: `(layers, x, hidden, inter,
      seq_len, use_qk_norm, softcap, target_layer, target_head,
      replacement_delta) -> Option<Vec<f32>>`. Default impl returns
      `None` so non-Metal backends route to the CPU intervention.
- [x] 2.2 Add `full_pipeline_q4_capture_pre_wo` symmetrically:
      `(layers, x, hidden, inter, seq_len, use_qk_norm, softcap,
      target_layer, target_head) -> Option<Vec<f32>>`.
- [x] 2.3 Rewrite `predict_q4k_metal_with_replaced_head_residual_delta`
      and `predict_q4k_metal_capture_pre_wo` in
      `crates/larql-inference/src/vindex/q4k_forward/metal.rs` to
      call the new trait methods. Remove the `FIXME(merge)` markers.
- [x] 2.4 Add unit tests that the CPU backend returns `None` from
      both new methods (proves the default impls fire on the correct
      branch when Metal isn't available).

## 3. TokenizerCache wiring on `LoadedModel`

- [x] 3.1 Declare `pub mod tokenizer_cache;` in
      `crates/larql-server/src/lib.rs` (the module file already
      exists; only the `mod` declaration was missing).
- [x] 3.2 Add `pub tokenizer_cache: Arc<crate::tokenizer_cache::TokenizerCache>`
      to `LoadedModel`. Initialise via `TokenizerCache::from_env()`
      in the production constructor at `bootstrap.rs:342` and via
      `TokenizerCache::new(0, 0)` in the two test-only constructors.
- [x] 3.3 Replace the `encode_cached_ids` shim with a real two-tier
      lookup: L0 full-text hit returns immediately; L1 partial hit
      cold-encodes the suffix after the last special-token sentinel
      and concatenates, then promotes the merged result back to L0;
      full miss cold-encodes and inserts into both tiers.
- [x] 3.4 Add unit tests for L0 hit (identical ids on repeat), L1
      partial hit (chat-template prefix), eviction at capacity (L0=1
      with two distinct prompts), and the disabled-cache path (L0=0,
      L1=0).

## 4. `try_generate*` fallible wrappers

- [x] 4.1 Add `try_generate(weights, tokenizer, token_ids, max_tokens,
      index, backend, cached_layers, layer_range) -> Result<GenerateResult,
      GenerateError>` in `gpu.rs` — runs `generate` then routes
      `error: Option<GenerateError>` through `into_result()`.
- [x] 4.2 Add `try_generate_with_sampling` taking explicit
      `SamplingConfig` and `EosConfig`. Same wrapping pattern.
- [x] 4.3 Add `try_generate_streaming<F: FnMut(u32, &str, f64)>` for
      the streaming case.
- [x] 4.4 Re-export the new fns from
      `crates/larql-inference/src/layer_graph/generate/mod.rs` and
      from `crates/larql-inference/src/layer_graph/mod.rs` so callers
      hit them at the public `larql_inference::layer_graph::*` path.
- [x] 4.5 Add tests at the `GenerateResult::into_result` boundary:
      `empty_success` → `Ok(_)`, `empty_error(typed)` → `Err(typed)`,
      partial-tokens-on-error preserves the typed variant.
- [x] 4.6 Add compile-only function-pointer tests for the streaming
      and non-streaming wrapper signatures so a future signature
      churn fails fast.

## 5. Cross-crate test breakage cleanup (pre-existing merge fallout)

- [x] 5.1 Add a `test-utils` feature to `larql-inference/Cargo.toml`
      (empty feature set; the fixture code uses already-required
      deps).
- [x] 5.2 Re-gate the `crate::test_utils` re-export module in
      `crates/larql-inference/src/lib.rs` from `#[cfg(test)]
      pub(crate)` to `#[cfg(any(test, feature = "test-utils"))] pub`.
      Documentation comment updated to describe the new surface.
- [x] 5.3 Add
      `larql-inference = { path = "../larql-inference", features =
      ["test-utils"] }` under `[dev-dependencies]` in
      `crates/larql-kv/Cargo.toml`.
- [x] 5.4 Add the four missing `ModelWeights` fields
      (`position_embed`, `embed_quant`, `lm_head_quant`,
      `quant_tensors`) to the two literal initialisers in
      `crates/larql-vindex/src/extract/build.rs` and
      `crates/larql-vindex/src/extract/build_helpers/test_support.rs`.

## 6. Validation

- [x] 6.1 `cargo check --workspace` → 0 errors.
- [x] 6.2 `cargo test -p larql-models --lib` → 227 pass.
- [x] 6.3 `cargo test -p larql-compute --lib` → 163 pass (+2
      default-impl tests on top of the 161 baseline).
- [x] 6.4 `cargo test -p larql-inference --lib` → 1203 pass (+4
      `try_generate` tests on top of the 1199 baseline).
- [x] 6.5 `cargo test -p larql-vindex --lib` → 922 pass (broken on
      merge, now green).
- [x] 6.6 `cargo test -p larql-kv --lib` → 200 pass (broken on
      merge, now green).
- [x] 6.7 `cargo test -p larql-server --lib` → 287 pass (+4
      `encode_cached_ids` tests on top of the 283 baseline).
