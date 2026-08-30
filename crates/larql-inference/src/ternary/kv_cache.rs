//! KV-cached prefill and single-token decode.
//!
//! Split out of the former single-file `ternary.rs`; [`super`] shows
//! how the pieces compose.

// Siblings resolve through the parent's re-exports.
use super::*;
// Shared with `predict`, which owns the full-sequence forward.
use super::predict::{ffn_forward_after_input_norm, scaled_dot_product_attention_gqa};
use larql_compute::cpu::ops::ternary_matvec::{
    matvec_i2s_a8_f32_into, matvec_i2s_a8_into, quantize_activation_i8,
};
use ndarray::{Array1, Array2, ArrayView2};

/// Per-layer K/V projections accumulated across all positions seen
/// so far.  Held by the caller across decode steps.
#[derive(Clone)]
pub struct BitnetKvCache {
    /// `k[layer]` is `[past_len, n_kv_heads * head_dim]` f32.
    /// Always RoPE-applied at the position the row represents.
    pub k: Vec<Array2<f32>>,
    /// `v[layer]` is `[past_len, n_kv_heads * head_dim]` f32.
    pub v: Vec<Array2<f32>>,
    /// Number of positions accumulated so far.  Equal to
    /// `k[0].shape()[0]` for every layer; tracked separately so we
    /// can construct an empty cache without choosing a layer count.
    pub seq_len: usize,
}

impl BitnetKvCache {
    /// Empty cache sized for `n_layers`.  Each per-layer `k`/`v`
    /// starts with zero rows; rows are appended one-at-a-time as
    /// decode_step or prefill runs.
    pub fn new(n_layers: usize, n_kv_heads: usize, head_dim: usize) -> Self {
        let kv_width = n_kv_heads * head_dim;
        Self {
            k: (0..n_layers)
                .map(|_| Array2::zeros((0, kv_width)))
                .collect(),
            v: (0..n_layers)
                .map(|_| Array2::zeros((0, kv_width)))
                .collect(),
            seq_len: 0,
        }
    }
}

/// Run the full prompt through every layer, accumulating K/V into a
/// fresh cache.  Returns the cache + raw logits at the last position
/// (caller decides sampling / softmax / top-K).
///
/// Equivalent to `predict_bitnet` minus the top-K extraction.
pub fn prefill(model: &BitnetModel, token_ids: &[u32]) -> (BitnetKvCache, Vec<f32>) {
    let n_layers = model.layers.len();
    let mut cache = BitnetKvCache::new(n_layers, model.n_kv_heads, model.head_dim);
    if token_ids.is_empty() {
        let vocab = model.lm_head.shape()[0];
        return (cache, vec![0.0; vocab]);
    }
    let logits = run_full_forward(model, token_ids, Some(&mut cache), None);
    (cache, logits)
}

/// Append one new token to an existing cache and return the logits
/// for that position.  Caller picks the sampling strategy.
///
/// Internally: position = cache.seq_len; the new token's Q sees
/// causal-masked attention against the full cached K/V plus its own
/// row.
pub fn decode_step(model: &BitnetModel, cache: &mut BitnetKvCache, new_token: u32) -> Vec<f32> {
    let position = cache.seq_len;
    let hidden = model.embed.shape()[1];
    let head_dim = model.head_dim;
    let n_q_heads = model.n_q_heads;
    let n_kv_heads = model.n_kv_heads;
    let kv_width = n_kv_heads * head_dim;
    let q_width = n_q_heads * head_dim;
    debug_assert_eq!(q_width, hidden, "hidden = n_q_heads * head_dim");

    // 1. Embed the new token.
    let mut h = Array1::<f32>::zeros(hidden);
    let row = model.embed.row(new_token as usize % model.embed.shape()[0]);
    for (dst, &src) in h.iter_mut().zip(row.iter()) {
        *dst = src * model.embed_scale;
    }

    let mut x_norm = vec![0.0f32; hidden];
    let mut q = vec![0.0f32; q_width];
    let mut k = vec![0.0f32; kv_width];
    let mut v = vec![0.0f32; kv_width];
    let mut attn_pool = vec![0.0f32; hidden];
    let mut attn_pool_norm = vec![0.0f32; hidden];
    let mut attn_out = vec![0.0f32; hidden];
    let mut ffn_x_norm = vec![0.0f32; hidden];
    let mut ffn_gate = vec![0.0f32; model.layers[0].ffn.gate.rows];
    let mut ffn_up = vec![0.0f32; model.layers[0].ffn.up.rows];
    let mut ffn_hid = vec![0.0f32; model.layers[0].ffn.gate.rows];
    let mut ffn_out_row = vec![0.0f32; hidden];

    for (layer_idx, layer) in model.layers.iter().enumerate() {
        // a. attn_norm.
        rmsnorm_into(
            h.as_slice().unwrap(),
            &layer.attn_norm,
            model.eps,
            &mut x_norm,
        );

        // b. Q/K/V projections. Q/K/V share x_norm — quantise once (A8).
        let (x_i8, x_scale) = quantize_activation_i8(&x_norm);
        matvec_i2s_a8_into(&layer.attn_q, &x_i8, x_scale, &mut q).expect("attn_q shape");
        matvec_i2s_a8_into(&layer.attn_k, &x_i8, x_scale, &mut k).expect("attn_k shape");
        matvec_i2s_a8_into(&layer.attn_v, &x_i8, x_scale, &mut v).expect("attn_v shape");

        // c. RoPE on the new token's Q + K only.  The cached K
        //    already carries RoPE for positions 0..position-1.
        let q_arr = Array2::from_shape_vec((1, q_width), q.clone()).expect("q shape");
        let k_arr = Array2::from_shape_vec((1, kv_width), k.clone()).expect("k shape");
        let q_rotated = larql_compute::attention::rope::apply_rope_partial_at(
            &q_arr,
            n_q_heads,
            head_dim,
            model.rope_base,
            1.0,
            position,
        );
        let k_rotated = larql_compute::attention::rope::apply_rope_partial_at(
            &k_arr,
            n_kv_heads,
            head_dim,
            model.rope_base,
            1.0,
            position,
        );

        // d. Append K/V rows to the per-layer cache.  ndarray has no
        //    cheap append, so we rebuild — cache growth is O(n) total
        //    across n_layers per decode_step which is fine for our
        //    workloads (max_new_tokens typically ≤ 256, hidden ≤ 4k).
        let new_k_row = k_rotated.row(0).to_owned();
        let new_v_row = Array1::from(v.clone());
        cache.k[layer_idx] = stack_one_row(&cache.k[layer_idx], &new_k_row);
        cache.v[layer_idx] = stack_one_row(&cache.v[layer_idx], &new_v_row);

        // e. Causal-masked GQA attention: new Q vs cached K/V (which
        //    now includes our just-appended row).
        let q_view = q_rotated.row(0);
        attention_decode_into(
            q_view.as_slice().unwrap(),
            cache.k[layer_idx].view(),
            cache.v[layer_idx].view(),
            n_q_heads,
            n_kv_heads,
            head_dim,
            &mut attn_pool,
        );

        // f. Sub-norm + O projection.
        rmsnorm_into(
            &attn_pool,
            &layer.attn_sub_norm,
            model.eps,
            &mut attn_pool_norm,
        );
        matvec_i2s_a8_f32_into(&layer.attn_o, &attn_pool_norm, &mut attn_out)
            .expect("attn_o shape");

        // g. Residual + FFN + residual.
        for (dst, &src) in h.iter_mut().zip(attn_out.iter()) {
            *dst += src;
        }
        rmsnorm_into(
            h.as_slice().unwrap(),
            &layer.ffn.ffn_norm,
            model.eps,
            &mut ffn_x_norm,
        );
        ffn_forward_after_input_norm(
            &layer.ffn,
            &ffn_x_norm,
            model.eps,
            &mut ffn_gate,
            &mut ffn_up,
            &mut ffn_hid,
            &mut ffn_out_row,
        );
        for (dst, &src) in h.iter_mut().zip(ffn_out_row.iter()) {
            *dst += src;
        }
    }

    cache.seq_len += 1;

    // h_final = output_norm(h)
    let mut h_final = vec![0.0f32; hidden];
    rmsnorm_into(
        h.as_slice().unwrap(),
        &model.output_norm,
        model.eps,
        &mut h_final,
    );
    let h_arr = Array1::from(h_final);
    model.lm_head.dot(&h_arr).to_vec()
}

/// Generate up to `max_new_tokens` greedily from `prompt`.  Stops
/// early if `stop_token` is produced.  Returns the raw token-id
/// stream (caller decodes for surface form).
///
/// Backwards-compat shim around [`generate_sampled`] with
/// [`SamplingConfig::greedy`] — byte-for-byte identical output to
/// callers built before sampling existed.
pub fn generate(
    model: &BitnetModel,
    tokenizer: &larql_vindex::tokenizers::Tokenizer,
    prompt_token_ids: &[u32],
    max_new_tokens: usize,
    stop_token: Option<u32>,
) -> Vec<u32> {
    let _ = tokenizer; // unused on the greedy path; kept for API stability
    generate_sampled(
        model,
        prompt_token_ids,
        max_new_tokens,
        crate::layer_graph::generate::SamplingConfig::greedy(),
        stop_token,
    )
}

/// Generate up to `max_new_tokens` from `prompt` using a configurable
/// sampler (temperature / top-k / top-p / repetition penalties /
/// seedable RNG).  See [`SamplingConfig`] for the knobs.
///
/// `stop_token` halts generation before the token would be emitted
/// (mirrors [`generate`]).  EOS detection beyond a single token id
/// (stop strings, multiple EOS ids) belongs in
/// [`generate_streaming_bitnet`] which threads the full
/// [`EosConfig`].
pub fn generate_sampled(
    model: &BitnetModel,
    prompt_token_ids: &[u32],
    max_new_tokens: usize,
    sampling: crate::layer_graph::generate::SamplingConfig,
    stop_token: Option<u32>,
) -> Vec<u32> {
    if prompt_token_ids.is_empty() || max_new_tokens == 0 {
        return Vec::new();
    }
    let mut sampler = crate::layer_graph::generate::Sampler::new(sampling);
    let (mut cache, last_logits) = prefill(model, prompt_token_ids);
    let mut generated = Vec::with_capacity(max_new_tokens);

    let Some(mut next) = sampler.sample_with_history(&last_logits, &generated) else {
        return generated;
    };
    for _ in 0..max_new_tokens {
        if let Some(stop) = stop_token {
            if next == stop {
                break;
            }
        }
        generated.push(next);
        let logits = decode_step(model, &mut cache, next);
        match sampler.sample_with_history(&logits, &generated) {
            Some(t) => next = t,
            None => break,
        }
    }
    generated
}
/// Decode-time attention: one Q-row vs the full cached K/V history.
///
/// `q` is `[n_q_heads * head_dim]`, `k` and `v` are `[seq_len,
/// n_kv_heads * head_dim]`.  Result is written to `out` (length
/// `n_q_heads * head_dim`).
fn attention_decode_into(
    q: &[f32],
    k: ArrayView2<f32>,
    v: ArrayView2<f32>,
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    out: &mut [f32],
) {
    let seq_len = k.shape()[0];
    debug_assert_eq!(v.shape()[0], seq_len);
    let scale = 1.0 / (head_dim as f32).sqrt();
    let groups = n_q_heads / n_kv_heads.max(1);

    for o in out.iter_mut() {
        *o = 0.0;
    }

    for h_q in 0..n_q_heads {
        let h_kv = h_q / groups.max(1);
        let q_off = h_q * head_dim;
        let kv_off = h_kv * head_dim;

        // Scores over the full cached K (no causal mask — position
        // is at the end, attends to all of 0..seq_len-1 + itself,
        // which equals the full prefix).
        let mut scores = vec![0.0f32; seq_len];
        for (j, score) in scores.iter_mut().enumerate() {
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q[q_off + d] * k[(j, kv_off + d)];
            }
            *score = dot * scale;
        }

        // Stable softmax.
        let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for s in scores.iter_mut() {
            *s = (*s - max_score).exp();
            sum += *s;
        }
        if sum > 0.0 {
            for s in scores.iter_mut() {
                *s /= sum;
            }
        }

        // out[q_head] = Σ_j w[j] * v[j, kv_head]
        for (j, &w) in scores.iter().enumerate() {
            if w == 0.0 {
                continue;
            }
            for d in 0..head_dim {
                out[q_off + d] += w * v[(j, kv_off + d)];
            }
        }
    }
}

/// Append one row to a 2D ndarray.  ndarray has no built-in append;
/// we rebuild and copy.
fn stack_one_row(prev: &Array2<f32>, new_row: &Array1<f32>) -> Array2<f32> {
    let cols = prev.shape()[1];
    debug_assert_eq!(new_row.len(), cols);
    let new_rows = prev.shape()[0] + 1;
    let mut out = Array2::<f32>::zeros((new_rows, cols));
    if !prev.is_empty() {
        out.slice_mut(ndarray::s![..new_rows - 1, ..]).assign(prev);
    }
    out.row_mut(new_rows - 1).assign(new_row);
    out
}

#[cfg(test)]
pub(super) fn argmax(logits: &[f32]) -> u32 {
    let mut best_idx = 0u32;
    let mut best = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best {
            best = v;
            best_idx = i as u32;
        }
    }
    best_idx
}

/// Shared-spine forward used by both `prefill` (when `cache=Some`) and
/// the legacy single-shot `predict_bitnet`.  When `cache` is `Some`,
/// per-layer K/V are pushed in (after RoPE) so a subsequent
/// `decode_step` can extend the sequence.
///
/// Returns logits at the last position only.
pub(super) fn run_full_forward(
    model: &BitnetModel,
    token_ids: &[u32],
    mut cache: Option<&mut BitnetKvCache>,
    mut residuals: Option<&mut Vec<(usize, Vec<f32>)>>,
) -> Vec<f32> {
    let seq_len = token_ids.len();
    let hidden = model.embed.shape()[1];
    let head_dim = model.head_dim;
    let n_q_heads = model.n_q_heads;
    let n_kv_heads = model.n_kv_heads;
    debug_assert_eq!(n_q_heads * head_dim, hidden);

    let mut h = Array2::<f32>::zeros((seq_len, hidden));
    for (i, &tok) in token_ids.iter().enumerate() {
        let row = model.embed.row(tok as usize % model.embed.shape()[0]);
        let mut h_row = h.row_mut(i);
        for (dst, &src) in h_row.iter_mut().zip(row.iter()) {
            *dst = src * model.embed_scale;
        }
    }

    let mut x_norm = Array2::<f32>::zeros((seq_len, hidden));
    let mut q = Array2::<f32>::zeros((seq_len, n_q_heads * head_dim));
    let mut k = Array2::<f32>::zeros((seq_len, n_kv_heads * head_dim));
    let mut v = Array2::<f32>::zeros((seq_len, n_kv_heads * head_dim));
    let mut attn_pool = Array2::<f32>::zeros((seq_len, hidden));
    let mut attn_pool_norm = Array2::<f32>::zeros((seq_len, hidden));
    let mut attn_out = Array2::<f32>::zeros((seq_len, hidden));
    let mut ffn_x_norm = Array2::<f32>::zeros((seq_len, hidden));
    let mut ffn_gate = vec![0.0f32; model.layers[0].ffn.gate.rows];
    let mut ffn_up = vec![0.0f32; model.layers[0].ffn.up.rows];
    let mut ffn_hid = vec![0.0f32; model.layers[0].ffn.gate.rows];
    let mut ffn_out_row = vec![0.0f32; hidden];

    for (layer_idx, layer) in model.layers.iter().enumerate() {
        for i in 0..seq_len {
            rmsnorm_into(
                h.row(i).as_slice().unwrap(),
                &layer.attn_norm,
                model.eps,
                x_norm.row_mut(i).as_slice_mut().unwrap(),
            );
        }
        for i in 0..seq_len {
            // Q/K/V share x_norm.row(i) — quantise to int8 once (A8) and reuse.
            let (x_i8, x_scale) = quantize_activation_i8(x_norm.row(i).as_slice().unwrap());
            matvec_i2s_a8_into(
                &layer.attn_q,
                &x_i8,
                x_scale,
                q.row_mut(i).as_slice_mut().unwrap(),
            )
            .expect("attn_q shape");
            matvec_i2s_a8_into(
                &layer.attn_k,
                &x_i8,
                x_scale,
                k.row_mut(i).as_slice_mut().unwrap(),
            )
            .expect("attn_k shape");
            matvec_i2s_a8_into(
                &layer.attn_v,
                &x_i8,
                x_scale,
                v.row_mut(i).as_slice_mut().unwrap(),
            )
            .expect("attn_v shape");
        }

        let q_rot =
            larql_compute::attention::rope::apply_rope(&q, n_q_heads, head_dim, model.rope_base);
        let k_rot =
            larql_compute::attention::rope::apply_rope(&k, n_kv_heads, head_dim, model.rope_base);

        attn_pool.fill(0.0);
        scaled_dot_product_attention_gqa(
            q_rot.view(),
            k_rot.view(),
            v.view(),
            n_q_heads,
            n_kv_heads,
            head_dim,
            attn_pool.view_mut(),
        );

        // If a cache is being built, capture the prefill K/V for
        // this layer (post-RoPE for K, pre-anything for V).
        if let Some(c) = cache.as_deref_mut() {
            c.k[layer_idx] = k_rot.clone();
            c.v[layer_idx] = v.clone();
        }

        for i in 0..seq_len {
            rmsnorm_into(
                attn_pool.row(i).as_slice().unwrap(),
                &layer.attn_sub_norm,
                model.eps,
                attn_pool_norm.row_mut(i).as_slice_mut().unwrap(),
            );
            matvec_i2s_a8_f32_into(
                &layer.attn_o,
                attn_pool_norm.row(i).as_slice().unwrap(),
                attn_out.row_mut(i).as_slice_mut().unwrap(),
            )
            .expect("attn_o shape");
        }
        h += &attn_out;

        for i in 0..seq_len {
            rmsnorm_into(
                h.row(i).as_slice().unwrap(),
                &layer.ffn.ffn_norm,
                model.eps,
                ffn_x_norm.row_mut(i).as_slice_mut().unwrap(),
            );
            ffn_forward_after_input_norm(
                &layer.ffn,
                ffn_x_norm.row(i).as_slice().unwrap(),
                model.eps,
                &mut ffn_gate,
                &mut ffn_up,
                &mut ffn_hid,
                &mut ffn_out_row,
            );
            for (dst, &src) in h.row_mut(i).iter_mut().zip(ffn_out_row.iter()) {
                *dst += src;
            }
        }

        // Capture the last-token residual at this layer for walk
        // inference's KNN-store override.  Mirrors what the dense
        // `WalkFfn::take_residuals` produces — same semantic position
        // (post-FFN-residual at the last prompt token).
        if let Some(r) = residuals.as_deref_mut() {
            r.push((layer_idx, h.row(seq_len - 1).to_vec()));
        }
    }

    if let Some(c) = cache {
        c.seq_len = seq_len;
    }

    let last_h = h.row(seq_len - 1).to_owned();
    let mut h_final = vec![0.0f32; hidden];
    rmsnorm_into(
        last_h.as_slice().unwrap(),
        &model.output_norm,
        model.eps,
        &mut h_final,
    );
    let h_arr = Array1::from(h_final);
    model.lm_head.dot(&h_arr).to_vec()
}

// ─────────────────────────────────────────────────────────────────────────────
//  load_bitnet_model — construct a BitnetModel from a `--keep-quant` vindex
// ─────────────────────────────────────────────────────────────────────────────
