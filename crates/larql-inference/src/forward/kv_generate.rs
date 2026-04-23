//! Autoregressive generation with CPU KV cache.
//!
//! Two-phase decoder:
//!
//! 1. **Prefill.** Run a full forward pass over the prompt via
//!    `predict_with_ffn` (which already handles all Gemma 3 / Gemma 4
//!    specifics — QK norm, V norm, cross-layer KV sharing, PLE, layer
//!    scalar). During the pass, capture post-RoPE K and post-V-norm V
//!    per layer into a [`KvCache`].
//! 2. **Decode.** For each new token: embed it as a single row, run
//!    the decode-step attention (Q of new token attends against
//!    cached K/V + the new token's own K/V), FFN, next layer. At end
//!    of layer stack, logits → argmax → next token. Streams tokens
//!    to a caller-supplied callback.
//!
//! This is **not** a full re-implementation of the prefill path — the
//! prefill reuses `predict_with_ffn` verbatim. Only the decode step
//! has new code, gated to single-token inputs where per-step cost is
//! O(cached_len) instead of O(cached_len²).
//!
//! Works with any [`FfnBackend`] — local `WalkFfn`, `RemoteWalkBackend`
//! (FFN over HTTP), etc.

use ndarray::Array2;

use crate::attention::{
    run_attention_block_decode_step_backend, run_attention_with_kv_backend, KvCache,
};
use crate::ffn::FfnBackend;
use crate::forward::{embed_tokens_pub, logits_to_predictions_pub, run_ffn};
use crate::forward::predict::hidden_to_raw_logits;
use crate::model::ModelWeights;

/// Stream autoregressive generation with a KV cache.
///
/// `on_token` receives `(token_id, decoded_string)` for each generated
/// token as it arrives (including the first, which comes out of the
/// prefill step).
///
/// Returns the concatenated generated IDs. Stops on EOS or when
/// `max_new_tokens` have been produced.
pub fn generate_cached<F>(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    ffn: &dyn FfnBackend,
    prompt_ids: &[u32],
    max_new_tokens: usize,
    mut on_token: F,
) -> Vec<u32>
where
    F: FnMut(u32, &str),
{
    generate_cached_bounded(
        weights, tokenizer, ffn, prompt_ids, max_new_tokens, None, None, &mut on_token,
    )
}

/// Variant of [`generate_cached`] that runs Q/K/V/O projections on a
/// GPU `ComputeBackend` when provided. GQA softmax stays on CPU.
pub fn generate_cached_backend<F>(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
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
        weights, tokenizer, ffn, prompt_ids, max_new_tokens, window, backend, &mut on_token,
    )
}

/// Sliding-window (Markov-residual-bounded) variant of
/// [`generate_cached`]. Keeps only the last `window` positions of K/V
/// per layer — older tokens drop off the back of the cache and are no
/// longer attendable. Memory stays O(num_layers × window × kv_dim)
/// regardless of total generation length. Pass `window = None` for
/// unbounded growth.
pub fn generate_cached_with_window<F>(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
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
        weights, tokenizer, ffn, prompt_ids, max_new_tokens, window, None, &mut on_token,
    )
}

#[allow(clippy::too_many_arguments)]
fn generate_cached_bounded(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    ffn: &dyn FfnBackend,
    prompt_ids: &[u32],
    max_new_tokens: usize,
    window: Option<usize>,
    backend: Option<&dyn larql_compute::ComputeBackend>,
    on_token: &mut dyn FnMut(u32, &str),
) -> Vec<u32> {
    if max_new_tokens == 0 || prompt_ids.is_empty() {
        return Vec::new();
    }

    // ── Phase 1: prefill — full forward pass capturing K/V per layer ──
    let num_layers = weights.num_layers;
    let mut cache = match window {
        Some(w) => KvCache::with_window(num_layers, w),
        None => KvCache::with_layers(num_layers),
    };

    let mut h = embed_tokens_pub(weights, prompt_ids);
    for layer in 0..num_layers {
        let (h_post_attn, k_rope, v) =
            match run_attention_with_kv_backend(weights, &h, layer, backend) {
                Some(t) => t,
                None => return Vec::new(),
            };
        cache.layers[layer] = Some((k_rope, v));
        // Apply the window bound immediately — if prompt is longer
        // than the window, attention during later decode steps only
        // sees the last W positions of the prompt.
        cache.clip_layer(layer);
        let (h_out, _) = run_ffn(weights, &h_post_attn, layer, ffn, false);
        h = h_out;
    }
    // After prefill, the "next" absolute position is prompt_len.
    // Clipping shortens the cache rows but does NOT change the next
    // token's absolute position — new K gets RoPE at prompt_len
    // regardless of how many older positions were evicted.
    cache.next_position = prompt_ids.len();

    // Sample first new token from the prefill-end hidden state.
    let last_hidden = last_row_as_2d(&h);
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
        let h_new = embed_tokens_pub(weights, &[current_id]);

        let abs_position = cache.next_position;
        let mut h_step = h_new;
        for layer in 0..num_layers {
            let kv_entry = cache.layers[layer].as_ref();
            let (h_post_attn, new_kv) = match run_attention_block_decode_step_backend(
                weights, &h_step, layer, kv_entry, abs_position, backend,
            ) {
                Some(t) => t,
                None => return generated,
            };
            cache.layers[layer] = Some(new_kv);
            // Sliding window — evict the oldest row(s) if we've
            // exceeded `max_window`. No-op when unbounded.
            cache.clip_layer(layer);
            let (h_out, _) = run_ffn(weights, &h_post_attn, layer, ffn, false);
            h_step = h_out;
        }
        // Increment absolute position for the next iteration.
        cache.next_position += 1;

        // h_step is [1, hidden] — project to logits and argmax.
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

fn argmax_next_token(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    h_single: &Array2<f32>,
) -> Option<(u32, String)> {
    // `logits_to_predictions_pub` does final norm + lm_head + softmax +
    // top-k. We ask for top-1 and decode. Emits PredictResult with
    // `token_ids` parallel to `predictions`.
    let result = logits_to_predictions_pub(weights, h_single, tokenizer, 1, 1.0);
    let id = *result.token_ids.first()?;
    let (decoded, _) = result.predictions.first()?.clone();
    Some((id, decoded))
}

fn is_stop_token_str(s: &str) -> bool {
    matches!(
        s,
        "<eos>" | "</s>" | "<|endoftext|>" | "<|im_end|>"
            | "<|end_of_turn|>" | "<end_of_turn>"
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
pub fn generate_cached_constrained<F, M>(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
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

    // ── Prefill ──
    let mut h = embed_tokens_pub(weights, prompt_ids);
    for layer in 0..num_layers {
        let (h_post_attn, k_rope, v) =
            match run_attention_with_kv_backend(weights, &h, layer, None) {
                Some(t) => t,
                None => return Vec::new(),
            };
        cache.layers[layer] = Some((k_rope, v));
        let (h_out, _) = run_ffn(weights, &h_post_attn, layer, ffn, false);
        h = h_out;
    }
    cache.next_position = prompt_ids.len();

    // ── First token from prefill ──
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

    // ── Decode loop ──
    let mut current_id = first_id;
    for _step in 1..max_new_tokens {
        let h_new = embed_tokens_pub(weights, &[current_id]);
        let abs_position = cache.next_position;
        let mut h_step = h_new;
        for layer in 0..num_layers {
            let kv_entry = cache.layers[layer].as_ref();
            let (h_post_attn, new_kv) = match run_attention_block_decode_step_backend(
                weights, &h_step, layer, kv_entry, abs_position, None,
            ) {
                Some(t) => t,
                None => return generated,
            };
            cache.layers[layer] = Some(new_kv);
            let (h_out, _) = run_ffn(weights, &h_post_attn, layer, ffn, false);
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

/// Argmax over a (possibly masked) logit vector — returns `(token_id, decoded)`.
fn masked_argmax(logits: &[f32], tokenizer: &tokenizers::Tokenizer) -> Option<(u32, String)> {
    let (idx, _) = logits
        .iter()
        .enumerate()
        .filter(|(_, &v)| !v.is_nan())
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))?;
    let id = idx as u32;
    let decoded = tokenizer.decode(&[id], true).ok()?;
    Some((id, decoded))
}
