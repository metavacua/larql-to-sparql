//! Autoregressive generation with a CPU [`KvCache`].
//!
//! Two-phase decoder:
//!
//! 1. **Prefill.** Run a full forward pass over the prompt: per layer,
//!    attention (capturing post-RoPE K and post-V-norm V into the
//!    [`KvCache`]) → FFN → per-layer embedding (PLE, Gemma-4) →
//!    layer-scalar (Gemma-4). PLE and layer-scalar are no-ops on
//!    archs that don't define those keys (Gemma-3, TinyModel, etc.).
//! 2. **Decode.** For each new token: embed it as a single row,
//!    precompute the single-token PLE input, run decode-step attention
//!    (Q of new token attends against cached K/V + the new token's
//!    own K/V), FFN, PLE, layer-scalar, next layer. At end of layer
//!    stack, logits → argmax → next token. Streams tokens to a
//!    caller-supplied callback.
//!
//! This is **not** a full re-implementation of the prefill path — the
//! prefill reuses `predict_with_ffn` verbatim. Only the decode step
//! has new code, gated to single-token inputs where per-step cost is
//! O(cached_len) instead of O(cached_len²).
//!
//! Works with any [`FfnBackend`] — local `WalkFfn`, `RemoteWalkBackend`
//! (FFN over HTTP), etc.
//!
//! Lifted from `larql-inference::forward::kv_generate` in 2026-05-16.
//! These loops drive every engine's `prefill` / `decode_step` impl via
//! [`generate_with_engine`]; [`generate_cached_backend`] is retained as
//! the parity oracle for the unification migration (see
//! `larql-inference/docs/specs/kv-engine-unification.md` §8.7).

// Every fn in this file except `last_row_as_2d` and `is_stop_token_str`
// takes `&larql_inference::tokenizers::Tokenizer` (not(wasm32)-gated
// upstream) and/or `&dyn FfnBackend` / `&ModelWeights` /
// `&larql_vindex::VectorIndex` used only by those native callers — no
// portable pocket beyond the two pure helpers below.
#[cfg(not(target_arch = "wasm32"))]
use larql_inference::attention::{
    run_attention_block_decode_step_backend, run_attention_with_kv_backend,
};
#[cfg(not(target_arch = "wasm32"))]
use larql_inference::ffn::FfnBackend;
#[cfg(not(target_arch = "wasm32"))]
use larql_inference::forward::hooks::{LayerHook, NoopHook};
#[cfg(not(target_arch = "wasm32"))]
use larql_inference::forward::layer::apply_layer_scalar;
#[cfg(not(target_arch = "wasm32"))]
use larql_inference::forward::ple::{apply_per_layer_embedding, precompute_per_layer_inputs};
#[cfg(not(target_arch = "wasm32"))]
use larql_inference::forward::{
    embed_tokens_pub, hidden_to_raw_logits, logits_to_predictions_pub, run_ffn,
};
#[cfg(not(target_arch = "wasm32"))]
use larql_inference::ModelWeights;
use ndarray::Array2;

#[cfg(not(target_arch = "wasm32"))]
use crate::cache::KvCache;

mod kv_run;

pub use kv_run::{kv_decode_step_run, kv_prefill_run};

/// Stream autoregressive generation with a KV cache.
///
/// `on_token` receives `(token_id, decoded_string)` for each generated
/// token as it arrives (including the first, which comes out of the
/// prefill step).
///
/// Returns the concatenated generated IDs. Stops on EOS or when
/// `max_new_tokens` have been produced.
#[cfg(not(target_arch = "wasm32"))]
pub fn generate_cached<F>(
    weights: &ModelWeights,
    tokenizer: &larql_inference::tokenizers::Tokenizer,
    ffn: &dyn FfnBackend,
    prompt_ids: &[u32],
    max_new_tokens: usize,
    mut on_token: F,
) -> Vec<u32>
where
    F: FnMut(u32, &str),
{
    generate_cached_bounded(
        weights,
        tokenizer,
        ffn,
        prompt_ids,
        max_new_tokens,
        None,
        None,
        &mut on_token,
    )
}

/// Variant of [`generate_cached`] that runs Q/K/V/O projections on a
/// GPU `ComputeBackend` when provided. GQA softmax stays on CPU.
#[cfg(not(target_arch = "wasm32"))]
#[allow(clippy::too_many_arguments)]
pub fn generate_cached_backend<F>(
    weights: &ModelWeights,
    tokenizer: &larql_inference::tokenizers::Tokenizer,
    ffn: &dyn FfnBackend,
    prompt_ids: &[u32],
    max_new_tokens: usize,
    backend: Option<&dyn larql_compute::ComputeBackend>,
    window: Option<usize>,
    mut on_token: F,
) -> Vec<u32>
where
    F: FnMut(u32, &str),
{
    generate_cached_bounded(
        weights,
        tokenizer,
        ffn,
        prompt_ids,
        max_new_tokens,
        window,
        backend,
        &mut on_token,
    )
}

/// Sliding-window (Markov-residual-bounded) variant of
/// [`generate_cached`]. Keeps only the last `window` positions of K/V
/// per layer — older tokens drop off the back of the cache and are no
/// longer attendable. Memory stays O(num_layers × window × kv_dim)
/// regardless of total generation length. Pass `window = None` for
/// unbounded growth.
#[cfg(not(target_arch = "wasm32"))]
pub fn generate_cached_with_window<F>(
    weights: &ModelWeights,
    tokenizer: &larql_inference::tokenizers::Tokenizer,
    ffn: &dyn FfnBackend,
    prompt_ids: &[u32],
    max_new_tokens: usize,
    window: Option<usize>,
    mut on_token: F,
) -> Vec<u32>
where
    F: FnMut(u32, &str),
{
    generate_cached_bounded(
        weights,
        tokenizer,
        ffn,
        prompt_ids,
        max_new_tokens,
        window,
        None,
        &mut on_token,
    )
}

#[cfg(not(target_arch = "wasm32"))]
#[allow(clippy::too_many_arguments)]
fn generate_cached_bounded(
    weights: &ModelWeights,
    tokenizer: &larql_inference::tokenizers::Tokenizer,
    ffn: &dyn FfnBackend,
    prompt_ids: &[u32],
    max_new_tokens: usize,
    window: Option<usize>,
    backend: Option<&dyn larql_compute::ComputeBackend>,
    on_token: &mut dyn FnMut(u32, &str),
) -> Vec<u32> {
    generate_cached_hooked_inner(
        weights,
        tokenizer,
        ffn,
        prompt_ids,
        max_new_tokens,
        window,
        backend,
        &mut NoopHook,
        on_token,
    )
}

/// Hook-aware autoregressive generation on the CPU KV-cache path.
///
/// Same prefill + decode loop as [`generate_cached`], but fires
/// [`LayerHook`] callbacks at every layer of every step (prefill **and**
/// every decode step):
///
/// - `on_pre_layer` — residual entering the layer.
/// - `on_post_attention(&mut h)` — post-attention residual; mutating it
///   here changes what the layer's FFN sees.
/// - `on_post_layer(&mut h)` — full-layer output; mutating it here
///   changes what the **next** layer sees.
///
/// The Metal-fast `layer_graph::generate::gpu::generate*` path is
/// hook-free by design (the kernel pipeline is fused; threading hooks
/// through it would force per-layer kernel splits even when no hook is
/// registered, so we keep the fast path fast). When you need hooks
/// during multi-token generation use this CPU path instead — typically
/// 5–20× slower than the Metal path on the same model, but every
/// primitive in [`larql_inference::forward::hooks`] works end-to-end.
///
/// The `on_attention_weights` and `on_ffn_activation` callbacks do
/// **not** fire on this path — the production decode kernels don't
/// capture those intermediates. Use
/// [`larql_inference::forward::trace_forward_full_hooked`] for a single
/// forward pass when you need them.
#[cfg(not(target_arch = "wasm32"))]
#[allow(clippy::too_many_arguments)]
pub fn generate_cached_hooked<F>(
    weights: &ModelWeights,
    tokenizer: &larql_inference::tokenizers::Tokenizer,
    ffn: &dyn FfnBackend,
    prompt_ids: &[u32],
    max_new_tokens: usize,
    window: Option<usize>,
    backend: Option<&dyn larql_compute::ComputeBackend>,
    hook: &mut dyn LayerHook,
    mut on_token: F,
) -> Vec<u32>
where
    F: FnMut(u32, &str),
{
    generate_cached_hooked_inner(
        weights,
        tokenizer,
        ffn,
        prompt_ids,
        max_new_tokens,
        window,
        backend,
        hook,
        &mut on_token,
    )
}

/// Drive autoregressive generation through any [`crate::KvEngine`].
///
/// This is the engine-trait-based equivalent of [`generate_cached_backend`]:
/// same prefill → sample → decode loop → sample → ... shape, but the
/// per-stage forward passes are delegated to `engine.prefill` /
/// `engine.decode_step`. Sampling, tokenizer decoding, and EOS detection
/// remain centralized here so every engine produces a stream with
/// identical sampling semantics.
///
/// Parity contract: with `engine = StandardEngine::new(window)`, the
/// returned `Vec<u32>` is bit-identical to
/// `generate_cached_backend(weights, tokenizer, ffn, prompt, max,
/// backend, window, ...)`. This is the parity gate for the unification
/// migration (see `larql-inference/docs/specs/kv-engine-unification.md` §8.4).
#[cfg(not(target_arch = "wasm32"))]
pub fn generate_with_engine<F>(
    engine: &mut crate::AnyEngine,
    weights: &ModelWeights,
    tokenizer: &larql_inference::tokenizers::Tokenizer,
    ffn: &dyn FfnBackend,
    prompt_ids: &[u32],
    max_new_tokens: usize,
    mut on_token: F,
) -> Vec<u32>
where
    F: FnMut(u32, &str),
{
    if max_new_tokens == 0 || prompt_ids.is_empty() {
        return Vec::new();
    }

    // ── Phase 1: prefill ──
    let last_hidden = match engine.prefill(weights, ffn, prompt_ids) {
        Ok(h) => h,
        Err(_) => return Vec::new(),
    };

    // Sample first new token from the prefill-end hidden state.
    let first = match argmax_next_token(weights, tokenizer, &last_hidden) {
        Some(t) => t,
        None => return Vec::new(),
    };
    on_token(first.0, &first.1);

    let mut generated = Vec::with_capacity(max_new_tokens);
    generated.push(first.0);
    if is_stop_token_str(&first.1) {
        return generated;
    }
    if max_new_tokens == 1 {
        return generated;
    }

    // ── Phase 2: decode loop ──
    let mut current_id = first.0;
    for _step in 1..max_new_tokens {
        let h_step = match engine.decode_step(weights, ffn, current_id) {
            Ok(h) => h,
            Err(_) => break,
        };
        let (id, tok_str) = match argmax_next_token(weights, tokenizer, &h_step) {
            Some(t) => t,
            None => break,
        };
        on_token(id, &tok_str);
        generated.push(id);
        if is_stop_token_str(&tok_str) {
            break;
        }
        current_id = id;
    }

    generated
}

/// Like [`generate_with_engine`] but drives the engine's **resident-weights
/// quant** path (`prefill_resident` / `decode_step_resident`), threading the
/// `index` so a backend with a Q4K-direct attention kernel
/// (`LARQL_Q4K_DIRECT_ATTN`) reads packed bytes instead of `weights.tensors`.
///
/// Takes `&ModelWeights` (immutable) — the caller must have already made the
/// client weights f32-resident (so no lazy dequant / `&mut` is needed), which
/// lets `ffn` borrow the same `&weights` concurrently (the moe-shards path's
/// `RemoteMoeFfn`). With `LARQL_Q4K_DIRECT_ATTN` unset, the backend ignores the
/// index and runs the f32 path — output is identical to [`generate_with_engine`].
#[cfg(not(target_arch = "wasm32"))]
#[allow(clippy::too_many_arguments)]
pub fn generate_with_engine_resident<F>(
    engine: &mut crate::AnyEngine,
    weights: &ModelWeights,
    tokenizer: &larql_inference::tokenizers::Tokenizer,
    ffn: &dyn FfnBackend,
    index: &larql_inference::larql_vindex::VectorIndex,
    prompt_ids: &[u32],
    max_new_tokens: usize,
    mut on_token: F,
) -> Vec<u32>
where
    F: FnMut(u32, &str),
{
    if max_new_tokens == 0 || prompt_ids.is_empty() {
        return Vec::new();
    }

    let last_hidden = match engine.prefill_resident(weights, ffn, index, prompt_ids) {
        Ok(h) => h,
        Err(_) => return Vec::new(),
    };

    let first = match argmax_next_token_resident(weights, tokenizer, index, &last_hidden) {
        Some(t) => t,
        None => return Vec::new(),
    };
    on_token(first.0, &first.1);

    let mut generated = Vec::with_capacity(max_new_tokens);
    generated.push(first.0);
    if is_stop_token_str(&first.1) || max_new_tokens == 1 {
        return generated;
    }

    let mut current_id = first.0;
    for _step in 1..max_new_tokens {
        let h_step = match engine.decode_step_resident(weights, ffn, index, current_id) {
            Ok(h) => h,
            Err(_) => break,
        };
        let (id, tok_str) = match argmax_next_token_resident(weights, tokenizer, index, &h_step) {
            Some(t) => t,
            None => break,
        };
        on_token(id, &tok_str);
        generated.push(id);
        if is_stop_token_str(&tok_str) {
            break;
        }
        current_id = id;
    }

    generated
}

/// Multi-modal-capable peer of [`generate_with_engine`]. Same shape;
/// the only difference is the prefill input: pre-built initial hidden
/// state (e.g. from `larql_compute::forward::embed_plan` on an
/// `EmbeddingPlan` mixing `Tokens` and `Precomputed` chunks) instead of
/// a token-id slice.
///
/// **Contract: caller MUST verify `engine.supports_multimodal()` returns
/// true BEFORE calling this function** (see ADR-0023). The default
/// `prefill_from_hidden` impl panics on engines that don't support
/// MM; the capability check is the contract and the panic is
/// defense-in-depth. For text-only inputs on any engine, use
/// `generate_with_engine` instead.
///
/// `max_new_tokens` accounting is independent of `initial_hidden.nrows()`
/// — the budget counts only newly *decoded* tokens, never prefill rows
/// (which may include vision/audio embeddings that aren't tokens at all).
/// Pinned by the `max_tokens_independent_of_hidden_rows` test below.
///
/// Bit-identity contract: feeding the single-Tokens-chunk plan
/// `EmbeddingPlan::from_tokens(prompt_ids)` through `embed_plan` then
/// this function produces the same token stream as
/// `generate_with_engine(engine, ..., prompt_ids, max_new_tokens, on_token)`
/// — same sampling, same EOS, same callback shape. Pinned by the
/// `wrapper_text_only_plan_matches_generate_with_engine` test below.
#[cfg(not(target_arch = "wasm32"))]
pub fn generate_with_engine_from_hidden<F>(
    engine: &mut crate::AnyEngine,
    weights: &ModelWeights,
    tokenizer: &larql_inference::tokenizers::Tokenizer,
    ffn: &dyn FfnBackend,
    initial_hidden: &Array2<f32>,
    max_new_tokens: usize,
    mut on_token: F,
) -> Vec<u32>
where
    F: FnMut(u32, &str),
{
    if max_new_tokens == 0 || initial_hidden.nrows() == 0 {
        return Vec::new();
    }

    // ── Phase 1: prefill from pre-built hidden state ──
    // Panics if engine doesn't support MM; capability check is upstream
    // per ADR-0023. An Err return (e.g. BackendFailure on dispatch) is
    // mapped to an empty stream, matching `generate_with_engine`'s
    // post-refactor behaviour.
    let last_hidden = match engine.prefill_from_hidden(weights, ffn, initial_hidden) {
        Ok(h) => h,
        Err(_) => return Vec::new(),
    };

    // ── Sample first new token (identical to generate_with_engine) ──
    let first = match argmax_next_token(weights, tokenizer, &last_hidden) {
        Some(t) => t,
        None => return Vec::new(),
    };
    on_token(first.0, &first.1);

    let mut generated = Vec::with_capacity(max_new_tokens);
    generated.push(first.0);
    if is_stop_token_str(&first.1) {
        return generated;
    }
    if max_new_tokens == 1 {
        return generated;
    }

    // ── Phase 2: decode loop (verbatim from generate_with_engine; ADR-0023
    // scoped-out: decode-loop embedding stays text-token-based) ──
    let mut current_id = first.0;
    for _step in 1..max_new_tokens {
        let h_step = match engine.decode_step(weights, ffn, current_id) {
            Ok(h) => h,
            Err(_) => break,
        };
        let (id, tok_str) = match argmax_next_token(weights, tokenizer, &h_step) {
            Some(t) => t,
            None => break,
        };
        on_token(id, &tok_str);
        generated.push(id);
        if is_stop_token_str(&tok_str) {
            break;
        }
        current_id = id;
    }

    generated
}

#[cfg(not(target_arch = "wasm32"))]
#[allow(clippy::too_many_arguments)]
fn generate_cached_hooked_inner(
    weights: &ModelWeights,
    tokenizer: &larql_inference::tokenizers::Tokenizer,
    ffn: &dyn FfnBackend,
    prompt_ids: &[u32],
    max_new_tokens: usize,
    window: Option<usize>,
    backend: Option<&dyn larql_compute::ComputeBackend>,
    hook: &mut dyn LayerHook,
    on_token: &mut dyn FnMut(u32, &str),
) -> Vec<u32> {
    if max_new_tokens == 0 || prompt_ids.is_empty() {
        return Vec::new();
    }

    // ── Phase 1: prefill ──
    let (last_hidden, mut cache) = match kv_prefill_run(
        larql_inference::WeightsView::dense(weights),
        ffn,
        prompt_ids,
        window,
        backend,
        hook,
    ) {
        Ok(t) => t,
        Err(_) => return Vec::new(),
    };

    let first = match argmax_next_token(weights, tokenizer, &last_hidden) {
        Some(t) => t,
        None => return Vec::new(),
    };
    on_token(first.0, &first.1);

    let mut generated = Vec::with_capacity(max_new_tokens);
    generated.push(first.0);
    if is_stop_token_str(&first.1) {
        return generated;
    }
    if max_new_tokens == 1 {
        return generated;
    }

    let mut current_id = first.0;
    for _step in 1..max_new_tokens {
        let h_step = match kv_decode_step_run(weights, ffn, &mut cache, current_id, backend, hook) {
            Ok(h) => h,
            Err(_) => break,
        };
        let (id, tok_str) = match argmax_next_token(weights, tokenizer, &h_step) {
            Some(t) => t,
            None => break,
        };
        on_token(id, &tok_str);
        generated.push(id);
        if is_stop_token_str(&tok_str) {
            break;
        }
        current_id = id;
    }

    generated
}

fn last_row_as_2d(h: &Array2<f32>) -> Array2<f32> {
    let seq_len = h.shape()[0];
    let hidden = h.shape()[1];
    let mut out = Array2::<f32>::zeros((1, hidden));
    out.row_mut(0).assign(&h.row(seq_len - 1));
    out
}

#[cfg(not(target_arch = "wasm32"))]
fn argmax_next_token(
    weights: &ModelWeights,
    tokenizer: &larql_inference::tokenizers::Tokenizer,
    h_single: &Array2<f32>,
) -> Option<(u32, String)> {
    // lm_head (vocab projection) dominates this call — time it for the
    // decode-stage split (`LARQL_DECODE_STAGES=1`).
    let _t_lmhead = std::time::Instant::now();
    let result = logits_to_predictions_pub(weights, h_single, tokenizer, 1, 1.0);
    larql_inference::decode_stages::record_lmhead(_t_lmhead.elapsed().as_nanos());
    let id = *result.token_ids.first()?;
    let (decoded, _) = result.predictions.first()?.clone();
    Some((id, decoded))
}

/// `LARQL_Q4K_LM_HEAD=1` routes the resident-path lm_head matvec through
/// the vindex's Q4_K lm_head view (synthesised from f16 embeddings at load
/// for tied-embedding models) instead of the f32 row-parallel sgemv. On a
/// 262K-vocab head this drops lm_head bandwidth ~4× (e.g. 2.95 GB → 0.42 GB
/// per step on Gemma 4 26B-A4B). **Default on** (`LARQL_Q4K_LM_HEAD=0` opts
/// out); falls back to the f32 path when no Q4_K head view exists.
///
/// Delegates to `larql_compute::options`, an env-var/OnceLock-backed
/// module; its only caller, `argmax_next_token_resident`, is native
/// anyway.
#[cfg(not(target_arch = "wasm32"))]
fn q4k_lm_head_enabled() -> bool {
    larql_compute::options::q4k_lm_head_enabled()
}

/// Resident-path argmax: like [`argmax_next_token`] but with the vindex at
/// hand, so the lm_head matvec can run Q4_K-direct under
/// `LARQL_Q4K_LM_HEAD=1`. Falls back to the f32 path when the flag is off
/// or the vindex has no Q4_K lm_head view (untied model without
/// `lm_head_q4.bin`).
#[cfg(not(target_arch = "wasm32"))]
fn argmax_next_token_resident(
    weights: &ModelWeights,
    tokenizer: &larql_inference::tokenizers::Tokenizer,
    index: &larql_inference::larql_vindex::VectorIndex,
    h_single: &Array2<f32>,
) -> Option<(u32, String)> {
    if q4k_lm_head_enabled() && index.vocab_size > 0 {
        if let Some(q4_bytes) = index.storage.lm_head_kquant_view() {
            let _t_lmhead = std::time::Instant::now();
            // Argmax-only fast path: skips the full-vocab softmax + top-k
            // temporaries (scaling/softcap/temperature are monotone, so the
            // selected token is identical to the softmax route).
            let result = larql_inference::forward::predict::q4_lm_head_argmax(
                weights,
                h_single,
                q4_bytes.as_ref(),
                index.vocab_size,
                tokenizer,
            );
            larql_inference::decode_stages::record_lmhead(_t_lmhead.elapsed().as_nanos());
            if let Some((id, decoded)) = result {
                return Some((id, decoded));
            }
        }
    }
    argmax_next_token(weights, tokenizer, h_single)
}

fn is_stop_token_str(s: &str) -> bool {
    matches!(
        s,
        "<eos>"
            | "</s>"
            | "<|endoftext|>"
            | "<|im_end|>"
            | "<|end_of_turn|>"
            | "<end_of_turn>"
            | "<|end_of_text|>"
            | "<|eom_id|>"
            | "<|eot_id|>"
    )
}

/// Autoregressive generation where a caller-supplied closure can mask the raw
/// logits before each argmax step.
///
/// `mask_fn(generated_ids, logits)` is called after computing logits for each
/// new token. It may modify `logits` in place (e.g. set unwanted token positions
/// to `f32::NEG_INFINITY`) before the argmax is applied. Returning without
/// modification gives the same result as unconstrained generation.
///
/// Useful for grammar-constrained generation: the caller tracks the partial
/// output and restricts the vocabulary to tokens valid at each position.
#[cfg(not(target_arch = "wasm32"))]
pub fn generate_cached_constrained<F, M>(
    weights: &ModelWeights,
    tokenizer: &larql_inference::tokenizers::Tokenizer,
    ffn: &dyn FfnBackend,
    prompt_ids: &[u32],
    max_new_tokens: usize,
    mut mask_fn: M,
    mut on_token: F,
) -> Vec<u32>
where
    F: FnMut(u32, &str),
    M: FnMut(&[u32], &mut Vec<f32>),
{
    if max_new_tokens == 0 || prompt_ids.is_empty() {
        return Vec::new();
    }

    let num_layers = weights.num_layers;
    let mut cache = KvCache::with_layers(num_layers);

    let mut h = embed_tokens_pub(weights, prompt_ids);
    // Per-Layer Embedding inputs for Gemma-4 archs — same per-layer
    // sequence as `kv_prefill_run` (empty Vec / no-ops elsewhere).
    let ple_inputs = precompute_per_layer_inputs(weights, &h, prompt_ids);
    for layer in 0..num_layers {
        let (h_post_attn, k_rope, v) = match run_attention_with_kv_backend(
            larql_inference::WeightsView::dense(weights),
            &h,
            layer,
            None,
            None,
        ) {
            Some(t) => t,
            None => return Vec::new(),
        };
        cache.layers[layer] = Some((k_rope, v));
        let (h_post_ffn, _) = run_ffn(weights, &h_post_attn, layer, ffn, false);
        let mut h_out =
            apply_per_layer_embedding(weights, &h_post_ffn, layer, ple_inputs.get(layer));
        apply_layer_scalar(weights, &mut h_out, layer);
        h = h_out;
    }
    cache.next_position = prompt_ids.len();

    let last_hidden = last_row_as_2d(&h);
    let mut logits = hidden_to_raw_logits(weights, &last_hidden);
    let mut generated: Vec<u32> = Vec::with_capacity(max_new_tokens);
    mask_fn(&generated, &mut logits);
    let (first_id, first_str) = match masked_argmax(&logits, tokenizer) {
        Some(t) => t,
        None => return Vec::new(),
    };
    on_token(first_id, &first_str);
    generated.push(first_id);
    if is_stop_token_str(&first_str) || max_new_tokens == 1 {
        return generated;
    }

    let mut current_id = first_id;
    for _step in 1..max_new_tokens {
        let h_new = embed_tokens_pub(weights, &[current_id]);
        let abs_position = cache.next_position;
        // PLE inputs are per-token — recompute for this single-token
        // decode step, same sequence as `kv_decode_step_run`.
        let ple_inputs = precompute_per_layer_inputs(weights, &h_new, &[current_id]);
        let mut h_step = h_new;
        for layer in 0..num_layers {
            let kv_entry = cache.layers[layer].as_ref();
            let (h_post_attn, new_kv) = match run_attention_block_decode_step_backend(
                larql_inference::WeightsView::dense(weights),
                &h_step,
                layer,
                kv_entry,
                abs_position,
                None,
            ) {
                Some(t) => t,
                None => return generated,
            };
            cache.layers[layer] = Some(new_kv);
            let (h_post_ffn, _) = run_ffn(weights, &h_post_attn, layer, ffn, false);
            let mut h_out =
                apply_per_layer_embedding(weights, &h_post_ffn, layer, ple_inputs.get(layer));
            apply_layer_scalar(weights, &mut h_out, layer);
            h_step = h_out;
        }
        cache.next_position += 1;

        let mut logits = hidden_to_raw_logits(weights, &h_step);
        mask_fn(&generated, &mut logits);
        let (id, tok_str) = match masked_argmax(&logits, tokenizer) {
            Some(t) => t,
            None => break,
        };
        on_token(id, &tok_str);
        generated.push(id);
        if is_stop_token_str(&tok_str) {
            break;
        }
        current_id = id;
    }

    generated
}

#[cfg(not(target_arch = "wasm32"))]
fn masked_argmax(
    logits: &[f32],
    tokenizer: &larql_inference::tokenizers::Tokenizer,
) -> Option<(u32, String)> {
    let (idx, _) = logits
        .iter()
        .enumerate()
        .filter(|(_, &v)| !v.is_nan())
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))?;
    let id = idx as u32;
    let decoded = tokenizer.decode(&[id], true).ok()?;
    Some((id, decoded))
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_inference::ffn::WeightFfn;
    use larql_inference::test_utils::{make_test_tokenizer, make_test_weights};

    #[test]
    fn generate_cached_returns_token_ids() {
        let weights = make_test_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let ffn = WeightFfn { weights: &weights };
        let mut decoded_tokens: Vec<String> = Vec::new();
        let ids = generate_cached(&weights, &tokenizer, &ffn, &[0u32, 1], 3, |_id, text| {
            decoded_tokens.push(text.to_string())
        });
        assert!(ids.len() <= 3, "should generate at most 3 tokens");
        assert_eq!(
            ids.len(),
            decoded_tokens.len(),
            "callback called once per token"
        );
    }

    #[test]
    fn generate_cached_with_window_limits_cache() {
        let weights = make_test_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let ffn = WeightFfn { weights: &weights };
        let ids =
            generate_cached_with_window(&weights, &tokenizer, &ffn, &[0u32], 4, Some(2), |_, _| {});
        assert!(ids.len() <= 4);
    }

    // ── generate_with_engine coverage ─────────────────────────────────────
    //
    // Synthetic engine that returns deterministic hidden states to drive
    // the helper through each branch: empty inputs, max_new_tokens=0,
    // max_new_tokens=1, normal multi-step generation, prefill failure,
    // decode failure.

    struct StubEngine {
        cache: Option<KvCache>,
        fail_prefill: bool,
        fail_decode_after: Option<usize>,
        decode_count: usize,
    }

    impl crate::KvEngine for StubEngine {
        fn name(&self) -> &str {
            "stub"
        }
        fn info(&self) -> crate::EngineInfo {
            crate::EngineInfo {
                name: "stub".into(),
                description: "test fixture".into(),
                backend: "cpu".into(),
                config: String::new(),
            }
        }
        fn prefill(
            &mut self,
            weights: &ModelWeights,
            ffn: &dyn FfnBackend,
            token_ids: &[u32],
        ) -> Result<Array2<f32>, larql_inference::kv_engine::EngineError> {
            if self.fail_prefill {
                return Err(larql_inference::kv_engine::EngineError::BackendFailure {
                    details: "test stub: fail_prefill set".into(),
                });
            }
            let (hidden, cache) = kv_prefill_run(
                larql_inference::WeightsView::dense(weights),
                ffn,
                token_ids,
                None,
                None,
                &mut NoopHook,
            )?;
            self.cache = Some(cache);
            Ok(hidden)
        }
        fn decode_step(
            &mut self,
            weights: &ModelWeights,
            ffn: &dyn FfnBackend,
            token_id: u32,
        ) -> Result<Array2<f32>, larql_inference::kv_engine::EngineError> {
            self.decode_count += 1;
            if let Some(limit) = self.fail_decode_after {
                if self.decode_count > limit {
                    return Err(larql_inference::kv_engine::EngineError::BackendFailure {
                        details: "test stub: fail_decode_after exceeded".into(),
                    });
                }
            }
            let cache = self.cache.as_mut().ok_or_else(|| {
                larql_inference::kv_engine::EngineError::InvariantViolation {
                    what: "decode_step called before prefill".into(),
                }
            })?;
            kv_decode_step_run(weights, ffn, cache, token_id, None, &mut NoopHook)
        }
        // MM support: drive `generate_with_engine_from_hidden`. We can't
        // recover the original tokens from a pre-built hidden state, so the
        // stub seeds a fresh cache from synthetic ids `0..nrows`. That's a
        // valid populated cache (the from-hidden tests assert control flow
        // — break arms, EOS, max-tokens budget — not bit-parity with a
        // real embed).
        fn supports_multimodal(&self) -> bool {
            true
        }
        fn prefill_from_hidden(
            &mut self,
            weights: &ModelWeights,
            ffn: &dyn FfnBackend,
            initial_hidden: &Array2<f32>,
        ) -> Result<Array2<f32>, larql_inference::kv_engine::EngineError> {
            if self.fail_prefill {
                return Err(larql_inference::kv_engine::EngineError::BackendFailure {
                    details: "test stub: fail_prefill set".into(),
                });
            }
            let ids: Vec<u32> = (0..initial_hidden.nrows() as u32).collect();
            let (hidden, cache) = kv_prefill_run(
                larql_inference::WeightsView::dense(weights),
                ffn,
                &ids,
                None,
                None,
                &mut NoopHook,
            )?;
            self.cache = Some(cache);
            Ok(hidden)
        }
        // Resident-weights path: drive `generate_with_engine_resident`. The
        // stub ignores `index` and reuses the f32 prefill/decode bodies —
        // enough to exercise the wrapper's prefill/decode/break arms.
        fn prefill_resident(
            &mut self,
            weights: &ModelWeights,
            ffn: &dyn FfnBackend,
            _index: &larql_inference::larql_vindex::VectorIndex,
            token_ids: &[u32],
        ) -> Result<Array2<f32>, larql_inference::kv_engine::EngineError> {
            self.prefill(weights, ffn, token_ids)
        }
        fn decode_step_resident(
            &mut self,
            weights: &ModelWeights,
            ffn: &dyn FfnBackend,
            _index: &larql_inference::larql_vindex::VectorIndex,
            token_id: u32,
        ) -> Result<Array2<f32>, larql_inference::kv_engine::EngineError> {
            self.decode_step(weights, ffn, token_id)
        }
        fn memory_bytes(&self) -> usize {
            0
        }
    }

    fn fresh_stub() -> StubEngine {
        StubEngine {
            cache: None,
            fail_prefill: false,
            fail_decode_after: None,
            decode_count: 0,
        }
    }

    #[test]
    fn generate_with_engine_empty_prompt_returns_empty() {
        let weights = make_test_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let ffn = WeightFfn { weights: &weights };
        let mut eng = crate::AnyEngine::Kv(Box::new(fresh_stub()));
        let out = generate_with_engine(&mut eng, &weights, &tokenizer, &ffn, &[], 5, |_, _| {});
        assert!(out.is_empty());
    }

    #[test]
    fn generate_with_engine_zero_max_returns_empty() {
        let weights = make_test_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let ffn = WeightFfn { weights: &weights };
        let mut eng = crate::AnyEngine::Kv(Box::new(fresh_stub()));
        let out = generate_with_engine(
            &mut eng,
            &weights,
            &tokenizer,
            &ffn,
            &[0u32, 1],
            0,
            |_, _| {},
        );
        assert!(out.is_empty());
    }

    #[test]
    fn generate_with_engine_max_one_returns_single_token() {
        let weights = make_test_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let ffn = WeightFfn { weights: &weights };
        let mut eng = crate::AnyEngine::Kv(Box::new(fresh_stub()));
        let out = generate_with_engine(
            &mut eng,
            &weights,
            &tokenizer,
            &ffn,
            &[0u32, 1],
            1,
            |_, _| {},
        );
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn generate_with_engine_multi_step_fires_callback_per_token() {
        let weights = make_test_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let ffn = WeightFfn { weights: &weights };
        let mut eng = crate::AnyEngine::Kv(Box::new(fresh_stub()));
        let mut callbacks = 0usize;
        let out = generate_with_engine(
            &mut eng,
            &weights,
            &tokenizer,
            &ffn,
            &[0u32, 1],
            4,
            |_, _| callbacks += 1,
        );
        assert_eq!(out.len(), callbacks);
        assert!(out.len() <= 4);
    }

    #[test]
    fn generate_with_engine_prefill_failure_returns_empty() {
        let weights = make_test_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let ffn = WeightFfn { weights: &weights };
        let mut stub = fresh_stub();
        stub.fail_prefill = true;
        let mut eng = crate::AnyEngine::Kv(Box::new(stub));
        let out = generate_with_engine(&mut eng, &weights, &tokenizer, &ffn, &[0u32], 3, |_, _| {});
        assert!(out.is_empty());
    }

    #[test]
    fn generate_with_engine_decode_failure_breaks_loop_early() {
        let weights = make_test_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let ffn = WeightFfn { weights: &weights };
        let mut stub = fresh_stub();
        stub.fail_decode_after = Some(1);
        let mut eng = crate::AnyEngine::Kv(Box::new(stub));
        let out = generate_with_engine(
            &mut eng,
            &weights,
            &tokenizer,
            &ffn,
            &[0u32, 1],
            5,
            |_, _| {},
        );
        assert!(
            out.len() <= 2,
            "should break after decode failure, got {} tokens",
            out.len()
        );
    }

    #[test]
    fn generate_cached_backend_cpu() {
        let weights = make_test_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let ffn = WeightFfn { weights: &weights };
        let ids = generate_cached_backend(
            &weights,
            &tokenizer,
            &ffn,
            &[2u32, 3],
            2,
            None,
            None,
            |_, _| {},
        );
        assert!(ids.len() <= 2);
    }

    #[test]
    fn generate_cached_constrained_restricts_tokens() {
        let weights = make_test_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let ffn = WeightFfn { weights: &weights };
        let allowed: std::collections::HashSet<u32> = (0u32..8).collect();
        let ids = generate_cached_constrained(
            &weights,
            &tokenizer,
            &ffn,
            &[0u32],
            3,
            |_generated, logits| {
                for (id, logit) in logits.iter_mut().enumerate() {
                    if !allowed.contains(&(id as u32)) {
                        *logit = f32::NEG_INFINITY;
                    }
                }
            },
            |_, _| {},
        );
        for &id in &ids {
            assert!(
                allowed.contains(&id),
                "generated token {id} outside allowed set"
            );
        }
    }

    #[test]
    fn generate_cached_empty_prompt() {
        let weights = make_test_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let ffn = WeightFfn { weights: &weights };
        let ids = generate_cached(&weights, &tokenizer, &ffn, &[], 2, |_, _| {});
        assert!(ids.len() <= 2);
    }

    // ── generate_cached_hooked ────────────────────────────────────────────────

    // The unhooked and hooked decode paths are mathematically equivalent
    // under NoopHook, but BLAS reduction order can drift call-to-call on
    // Windows OpenBLAS — observed argmax flipping after the first decode
    // step. Linux/macOS BLAS implementations are bit-stable enough for
    // this assertion to hold, so we keep the coverage there.
    #[cfg(not(windows))]
    #[test]
    fn generate_cached_hooked_with_noop_matches_baseline() {
        let weights = make_test_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let ffn = WeightFfn { weights: &weights };

        let baseline = generate_cached(&weights, &tokenizer, &ffn, &[0u32, 1, 2], 4, |_, _| {});

        let hooked = generate_cached_hooked(
            &weights,
            &tokenizer,
            &ffn,
            &[0u32, 1, 2],
            4,
            None,
            None,
            &mut NoopHook,
            |_, _| {},
        );

        assert_eq!(baseline, hooked, "noop hook must not change generated ids");
    }

    #[test]
    fn generate_cached_hooked_record_fires_during_prefill_and_decode() {
        struct CountHook {
            calls: std::collections::HashMap<usize, usize>,
        }
        impl LayerHook for CountHook {
            fn on_post_layer(&mut self, layer: usize, _h: &mut Array2<f32>) {
                *self.calls.entry(layer).or_insert(0) += 1;
            }
        }

        let weights = make_test_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let ffn = WeightFfn { weights: &weights };
        let max_new = 3usize;
        let mut hook = CountHook {
            calls: std::collections::HashMap::new(),
        };

        let _ = generate_cached_hooked(
            &weights,
            &tokenizer,
            &ffn,
            &[0u32, 1],
            max_new,
            None,
            None,
            &mut hook,
            |_, _| {},
        );

        for layer in 0..weights.num_layers {
            let count = *hook.calls.get(&layer).unwrap_or(&0);
            assert!(
                count >= 1,
                "hook should fire at least once per layer (got {count} for layer {layer})"
            );
            assert!(
                count <= max_new,
                "hook fires at most max_new times per layer (got {count} for layer {layer})"
            );
        }
    }

    #[test]
    fn generate_cached_hooked_steer_changes_output() {
        use larql_inference::forward::SteerHook;
        use ndarray::Array1;

        let weights = make_test_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let ffn = WeightFfn { weights: &weights };
        let prompt = vec![1u32, 2, 3];

        let baseline = generate_cached(&weights, &tokenizer, &ffn, &prompt, 4, |_, _| {});

        let v = Array1::from_vec(
            (0..weights.hidden_size)
                .map(|i| (i as f32 + 1.0) * 0.1)
                .collect(),
        );
        let mut steer = SteerHook::new().add(0, v, 5.0);

        let steered = generate_cached_hooked(
            &weights,
            &tokenizer,
            &ffn,
            &prompt,
            4,
            None,
            None,
            &mut steer,
            |_, _| {},
        );

        if !baseline.is_empty() && !steered.is_empty() {
            assert_ne!(
                baseline, steered,
                "steering with α=5 must change generated tokens"
            );
        }
    }

    // ── Gemma-4 PLE arch coverage (regression test for issue #98) ──
    //
    // Before this PR, `kv_prefill_run` and `kv_decode_step_run` called
    // `run_attention*` + `run_ffn` directly, skipping the
    // `apply_per_layer_embedding` and `apply_layer_scalar` steps that
    // `run_layer_with_ffn` performs. On Gemma-4 (`gemma-4-E4B-it`),
    // the missing PLE contribution compounded across decode steps and
    // produced garbage (`ッケッケTobchal的存在` after a correct first
    // token). These tests pin both phases through the synthetic E2B-like
    // fixture so any future regression that drops PLE / layer_scalar
    // from the cached path fails locally rather than at the user's
    // terminal.

    /// `kv_prefill_run` must execute cleanly on a PLE arch — the
    /// fixture's PLE keys + projection tensors / norms / gates must be
    /// reachable from the prefill loop without dimension mismatch or
    /// panic. Value-level pins live in `tests/dispatch_parity.rs` (PLE
    /// parity vs the dispatch helpers, bit-exact); this test asserts
    /// finiteness + correct hidden-dim shape.
    #[test]
    fn kv_prefill_run_works_on_synthetic_e2b_ple_arch() {
        let weights = larql_inference::test_utils::make_synthetic_e2b_like_weights();
        let ffn = WeightFfn { weights: &weights };
        let prompt = [0u32, 1, 2];
        let (last_hidden, cache) = kv_prefill_run(
            larql_inference::WeightsView::dense(&weights),
            &ffn,
            &prompt,
            None,
            None,
            &mut NoopHook,
        )
        .expect("PLE-arch prefill should not fail");
        assert_eq!(last_hidden.shape(), &[1, weights.hidden_size]);
        assert!(
            last_hidden.iter().all(|v| v.is_finite()),
            "prefill output must be finite"
        );
        assert_eq!(cache.next_position, prompt.len());
    }

    /// `kv_decode_step_run` must execute cleanly on a PLE arch for at
    /// least three successive steps. Issue #98's signature was: step 1
    /// looks fine, steps 2+ degrade. Driving three steps exercises the
    /// per-decode-step PLE recompute (`precompute_per_layer_inputs(..,
    /// &[token_id])`) under the same code path that produced the
    /// regression.
    #[test]
    fn kv_decode_step_run_works_for_multiple_steps_on_synthetic_e2b_ple_arch() {
        let weights = larql_inference::test_utils::make_synthetic_e2b_like_weights();
        let ffn = WeightFfn { weights: &weights };
        let prompt = [0u32, 1];
        let (_h_prefill, mut cache) = kv_prefill_run(
            larql_inference::WeightsView::dense(&weights),
            &ffn,
            &prompt,
            None,
            None,
            &mut NoopHook,
        )
        .expect("PLE-arch prefill should not fail");

        for step in 0..3 {
            let h_step = kv_decode_step_run(&weights, &ffn, &mut cache, 0u32, None, &mut NoopHook)
                .unwrap_or_else(|e| panic!("decode step {step} failed: {e:?}"));
            assert_eq!(h_step.shape(), &[1, weights.hidden_size]);
            assert!(
                h_step.iter().all(|v| v.is_finite()),
                "decode step {step} output must be finite"
            );
        }
        assert_eq!(cache.next_position, prompt.len() + 3);
    }

    /// `generate_cached_constrained` carries its own inline prefill +
    /// decode loops (it cannot reuse `kv_prefill_run` because of the
    /// mask hook), so those loops must run the same per-layer sequence
    /// — attention → FFN → PLE → layer_scalar. With a no-op mask the
    /// token stream must match `generate_cached` (which drives the
    /// oracle loops) exactly; on the E2B fixture a constrained path
    /// that drops PLE / layer_scalar computes different logits and
    /// diverges.
    #[cfg(not(windows))]
    #[test]
    fn generate_cached_constrained_matches_generate_cached_on_ple_arch() {
        let weights = larql_inference::test_utils::make_synthetic_e2b_like_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let ffn = WeightFfn { weights: &weights };
        let prompt = [0u32, 1, 2];
        const MAX_NEW_TOKENS: usize = 6;

        let mut toks_oracle: Vec<u32> = Vec::new();
        generate_cached(
            &weights,
            &tokenizer,
            &ffn,
            &prompt,
            MAX_NEW_TOKENS,
            |id, _| toks_oracle.push(id),
        );
        let mut toks_constrained: Vec<u32> = Vec::new();
        generate_cached_constrained(
            &weights,
            &tokenizer,
            &ffn,
            &prompt,
            MAX_NEW_TOKENS,
            |_, _| {},
            |id, _| toks_constrained.push(id),
        );
        assert!(
            !toks_oracle.is_empty(),
            "oracle generation must emit tokens"
        );
        assert_eq!(
            toks_oracle, toks_constrained,
            "no-op-mask constrained generation must match generate_cached"
        );
    }

    // ─── Phase 1d.3b: generate_with_engine_from_hidden contracts ─────────
    //
    // Two pins:
    //   1. Bit-identity: a single-Tokens-chunk plan run through
    //      embed_plan → generate_with_engine_from_hidden produces the
    //      same token stream as generate_with_engine(tokens). This is
    //      the analog of the dispatch-level bit-identity test, applied
    //      one layer up. If they diverge, something in the wrapper
    //      silently dropped a flag/callback/state vs the original.
    //   2. max_tokens accounting independent of initial_hidden.nrows().
    //      The contract is "max_new_tokens counts decoded tokens only,
    //      never prefill rows" — this catches the off-by-one risk that
    //      would manifest as captions being one token short, or vision
    //      tokens being counted against the user's --max-tokens budget.

    #[test]
    fn wrapper_text_only_plan_matches_generate_with_engine() {
        use crate::engines::standard::StandardEngine;
        use crate::AnyEngine;
        use larql_compute::forward::{embed_plan, EmbeddingPlan};
        let weights = make_test_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let ffn = WeightFfn { weights: &weights };
        let tokens = [0u32, 1, 2, 3];
        let max_new = 4usize;

        // Path A: text path. Post kv-engine-retrieval-trait-split,
        // engines are wrapped in AnyEngine for uniform dispatch.
        let mut engine_a = AnyEngine::Kv(Box::new(StandardEngine::new(None)));
        let mut emitted_a: Vec<(u32, String)> = Vec::new();
        let ids_a = generate_with_engine(
            &mut engine_a,
            &weights,
            &tokenizer,
            &ffn,
            &tokens,
            max_new,
            |id, s| emitted_a.push((id, s.to_string())),
        );

        // Path B: single-Tokens-chunk plan → embed_plan → wrapper.
        let mut engine_b = AnyEngine::Kv(Box::new(StandardEngine::new(None)));
        let plan = EmbeddingPlan::from_tokens(&tokens);
        let initial_hidden = embed_plan(&weights, &plan);
        let mut emitted_b: Vec<(u32, String)> = Vec::new();
        let ids_b = generate_with_engine_from_hidden(
            &mut engine_b,
            &weights,
            &tokenizer,
            &ffn,
            &initial_hidden,
            max_new,
            |id, s| emitted_b.push((id, s.to_string())),
        );

        assert_eq!(
            ids_a, ids_b,
            "text path and from-hidden wrapper must produce identical token streams \
             on a single-Tokens-chunk plan"
        );
        assert_eq!(
            emitted_a, emitted_b,
            "streaming callback must fire identically across paths \
             (same id + same decoded text per token)"
        );
    }

    #[test]
    fn wrapper_max_tokens_independent_of_hidden_rows() {
        // Build a hidden state with MORE rows than max_new_tokens to
        // confirm the budget isn't accidentally tangled with prefill
        // length. If it were, generation would terminate early (or not
        // at all) depending on the off-by-one's direction.
        use crate::engines::standard::StandardEngine;
        use larql_compute::forward::{embed_plan, EmbeddingPlan};
        let weights = make_test_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let ffn = WeightFfn { weights: &weights };

        let prefill_rows = 7usize;
        let max_new = 3usize;
        let tokens: Vec<u32> = (0u32..prefill_rows as u32).collect();
        let plan = EmbeddingPlan::from_tokens(&tokens);
        let initial_hidden = embed_plan(&weights, &plan);
        assert_eq!(initial_hidden.nrows(), prefill_rows);

        let mut engine = crate::AnyEngine::Kv(Box::new(StandardEngine::new(None)));
        let ids = generate_with_engine_from_hidden(
            &mut engine,
            &weights,
            &tokenizer,
            &ffn,
            &initial_hidden,
            max_new,
            |_, _| {},
        );

        // Wrapper may terminate early on EOS (stop-token in the
        // synthetic stream), but must NEVER exceed max_new even though
        // initial_hidden has more rows than max_new.
        assert!(
            ids.len() <= max_new,
            "wrapper decoded {} tokens but max_new_tokens={max_new}; \
             prefill rows ({prefill_rows}) leaked into the token budget",
            ids.len(),
        );
    }

    #[test]
    fn wrapper_zero_hidden_rows_returns_empty() {
        use crate::engines::standard::StandardEngine;
        let weights = make_test_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let ffn = WeightFfn { weights: &weights };
        let empty = Array2::<f32>::zeros((0, weights.hidden_size));
        let mut engine = crate::AnyEngine::Kv(Box::new(StandardEngine::new(None)));
        let ids = generate_with_engine_from_hidden(
            &mut engine,
            &weights,
            &tokenizer,
            &ffn,
            &empty,
            5,
            |_, _| {},
        );
        assert!(ids.is_empty(), "zero-row hidden should yield empty stream");
    }

    // ── generate_with_engine_from_hidden break-arm coverage ────────────────
    //
    // The from-hidden wrapper duplicates the prefill→sample→decode loop of
    // `generate_with_engine`, including the early-return / break arms:
    // zero-max budget, prefill failure, max_new=1, and decode failure
    // mid-loop. These mirror the `generate_with_engine_*` stub tests above
    // but drive the from-hidden code path (engine.prefill_from_hidden +
    // engine.decode_step). The StubEngine seeds a fresh cache from
    // synthetic ids, so the assertions are on control flow, not parity.

    fn hidden_for(weights: &ModelWeights, rows: usize) -> Array2<f32> {
        let mut h = Array2::<f32>::zeros((rows, weights.hidden_size));
        for r in 0..rows {
            for c in 0..weights.hidden_size {
                h[[r, c]] = ((r * 7 + c * 3) % 11) as f32 * 0.01 - 0.05;
            }
        }
        h
    }

    #[test]
    fn from_hidden_zero_max_returns_empty() {
        let weights = make_test_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let ffn = WeightFfn { weights: &weights };
        let h = hidden_for(&weights, 3);
        let mut eng = crate::AnyEngine::Kv(Box::new(fresh_stub()));
        let out = generate_with_engine_from_hidden(
            &mut eng,
            &weights,
            &tokenizer,
            &ffn,
            &h,
            0,
            |_, _| {},
        );
        assert!(out.is_empty());
    }

    #[test]
    fn from_hidden_prefill_failure_returns_empty() {
        let weights = make_test_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let ffn = WeightFfn { weights: &weights };
        let h = hidden_for(&weights, 3);
        let mut stub = fresh_stub();
        stub.fail_prefill = true;
        let mut eng = crate::AnyEngine::Kv(Box::new(stub));
        let out = generate_with_engine_from_hidden(
            &mut eng,
            &weights,
            &tokenizer,
            &ffn,
            &h,
            4,
            |_, _| {},
        );
        assert!(out.is_empty());
    }

    #[test]
    fn from_hidden_max_one_returns_single_token() {
        let weights = make_test_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let ffn = WeightFfn { weights: &weights };
        let h = hidden_for(&weights, 2);
        let mut eng = crate::AnyEngine::Kv(Box::new(fresh_stub()));
        let out = generate_with_engine_from_hidden(
            &mut eng,
            &weights,
            &tokenizer,
            &ffn,
            &h,
            1,
            |_, _| {},
        );
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn from_hidden_decode_failure_breaks_loop_early() {
        let weights = make_test_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let ffn = WeightFfn { weights: &weights };
        let h = hidden_for(&weights, 2);
        let mut stub = fresh_stub();
        stub.fail_decode_after = Some(1);
        let mut eng = crate::AnyEngine::Kv(Box::new(stub));
        let out = generate_with_engine_from_hidden(
            &mut eng,
            &weights,
            &tokenizer,
            &ffn,
            &h,
            5,
            |_, _| {},
        );
        assert!(
            out.len() <= 2,
            "should break after decode failure, got {} tokens",
            out.len()
        );
    }

    #[test]
    fn from_hidden_multi_step_fires_callback_per_token() {
        let weights = make_test_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let ffn = WeightFfn { weights: &weights };
        let h = hidden_for(&weights, 2);
        let mut eng = crate::AnyEngine::Kv(Box::new(fresh_stub()));
        let mut callbacks = 0usize;
        let out = generate_with_engine_from_hidden(
            &mut eng,
            &weights,
            &tokenizer,
            &ffn,
            &h,
            4,
            |_, _| callbacks += 1,
        );
        assert_eq!(out.len(), callbacks);
        assert!(out.len() <= 4);
    }

    // ── generate_with_engine_resident coverage ─────────────────────────────
    //
    // The resident wrapper drives `engine.prefill_resident` /
    // `engine.decode_step_resident`, threading the `index`. With
    // `LARQL_Q4K_DIRECT_ATTN` unset (default), the CPU backend ignores the
    // index and runs the f32 path, so a StandardEngine over f32 test
    // weights + a Q4K vindex exercises the full happy path. The stub drives
    // the zero-max / prefill-failure / max-one / decode-failure break arms.

    #[test]
    fn resident_happy_path_via_standard_engine() {
        use crate::engines::standard::StandardEngine;
        use larql_inference::ffn::NullFfn;
        use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
        let weights = make_test_q4k_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let index = make_test_q4k_vindex(&weights);
        let ffn = NullFfn;
        let mut engine = crate::AnyEngine::Kv(Box::new(StandardEngine::new(None)));
        let mut callbacks = 0usize;
        let out = generate_with_engine_resident(
            &mut engine,
            &weights,
            &tokenizer,
            &ffn,
            &index,
            &[0u32, 1, 2],
            4,
            |_, _| callbacks += 1,
        );
        assert_eq!(out.len(), callbacks, "callback fires once per token");
        assert!(out.len() <= 4);
    }

    #[test]
    fn resident_zero_max_returns_empty() {
        use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
        let weights = make_test_q4k_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let index = make_test_q4k_vindex(&weights);
        let ffn = WeightFfn { weights: &weights };
        let mut eng = crate::AnyEngine::Kv(Box::new(fresh_stub()));
        let out = generate_with_engine_resident(
            &mut eng,
            &weights,
            &tokenizer,
            &ffn,
            &index,
            &[0u32, 1],
            0,
            |_, _| {},
        );
        assert!(out.is_empty());
    }

    #[test]
    fn resident_empty_prompt_returns_empty() {
        use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
        let weights = make_test_q4k_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let index = make_test_q4k_vindex(&weights);
        let ffn = WeightFfn { weights: &weights };
        let mut eng = crate::AnyEngine::Kv(Box::new(fresh_stub()));
        let out = generate_with_engine_resident(
            &mut eng,
            &weights,
            &tokenizer,
            &ffn,
            &index,
            &[],
            4,
            |_, _| {},
        );
        assert!(out.is_empty());
    }

    #[test]
    fn resident_prefill_failure_returns_empty() {
        use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
        let weights = make_test_q4k_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let index = make_test_q4k_vindex(&weights);
        let ffn = WeightFfn { weights: &weights };
        let mut stub = fresh_stub();
        stub.fail_prefill = true;
        let mut eng = crate::AnyEngine::Kv(Box::new(stub));
        let out = generate_with_engine_resident(
            &mut eng,
            &weights,
            &tokenizer,
            &ffn,
            &index,
            &[0u32],
            3,
            |_, _| {},
        );
        assert!(out.is_empty());
    }

    #[test]
    fn resident_max_one_returns_single_token() {
        use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
        let weights = make_test_q4k_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let index = make_test_q4k_vindex(&weights);
        let ffn = WeightFfn { weights: &weights };
        let mut eng = crate::AnyEngine::Kv(Box::new(fresh_stub()));
        let out = generate_with_engine_resident(
            &mut eng,
            &weights,
            &tokenizer,
            &ffn,
            &index,
            &[0u32, 1],
            1,
            |_, _| {},
        );
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn resident_decode_failure_breaks_loop_early() {
        use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
        let weights = make_test_q4k_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let index = make_test_q4k_vindex(&weights);
        let ffn = WeightFfn { weights: &weights };
        let mut stub = fresh_stub();
        stub.fail_decode_after = Some(1);
        let mut eng = crate::AnyEngine::Kv(Box::new(stub));
        let out = generate_with_engine_resident(
            &mut eng,
            &weights,
            &tokenizer,
            &ffn,
            &index,
            &[0u32, 1],
            5,
            |_, _| {},
        );
        assert!(
            out.len() <= 2,
            "should break after decode failure, got {} tokens",
            out.len()
        );
    }
}
