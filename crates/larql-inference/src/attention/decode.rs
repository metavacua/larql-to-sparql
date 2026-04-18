//! Decode-step attention — GQA for a single new token against a
//! growing KV cache.
//!
//! Prefill does full O(seq²) attention and returns K/V per layer. Decode
//! runs one token at a time with O(cached_len) attention: Q for the new
//! token attends against [K_cache | K_new] and [V_cache | V_new], with
//! no causal mask needed (the new query is at the end and can see every
//! cached position + itself).
//!
//! See `predict::generate_cached` for the prefill→decode driver.

use ndarray::Array2;

use super::SharedKV;
use super::rope::apply_rope_partial_at;

/// Per-layer K/V cache. Can grow unbounded or be clamped to a fixed
/// sliding window (Markov-residual-bounded strategy — keep the last W
/// positions' K/V, evict older). When bounded, attention becomes
/// "look at the last W tokens" — identical to StreamingLLM / sliding
/// window approaches.
///
/// Memory: O(num_layers × window × kv_dim × 4 bytes) when bounded,
/// O(num_layers × seq_len × kv_dim × 4 bytes) when unbounded.
#[derive(Clone, Debug, Default)]
pub struct KvCache {
    /// One entry per layer. `None` for layers that reuse another
    /// layer's K/V (Gemma 4 cross-layer sharing).
    pub layers: Vec<Option<SharedKV>>,
    /// When `Some(W)`, each layer's K/V is clipped to the last W
    /// positions after every append — the "bounded" part of the
    /// Markov Residual Bounded strategy. `None` = unbounded growth.
    pub max_window: Option<usize>,
    /// Absolute token position of the NEXT token to be appended.
    /// Used for RoPE: a new token's K needs RoPE at its true absolute
    /// position, not its row index in the clipped cache. Starts at 0
    /// and increments per append (not per eviction).
    pub next_position: usize,
}

impl KvCache {
    /// Unbounded cache — grows with every decode step.
    pub fn with_layers(num_layers: usize) -> Self {
        Self {
            layers: vec![None; num_layers],
            max_window: None,
            next_position: 0,
        }
    }

    /// Bounded (Markov-residual-bounded) — keeps only the last
    /// `window` positions per layer. Memory stays O(window).
    pub fn with_window(num_layers: usize, window: usize) -> Self {
        Self {
            layers: vec![None; num_layers],
            max_window: if window == 0 { None } else { Some(window) },
            next_position: 0,
        }
    }

    /// Number of cached positions for a given layer. Returns 0 if the
    /// layer has no cache yet.
    pub fn cached_len(&self, layer: usize) -> usize {
        self.layers
            .get(layer)
            .and_then(|opt| opt.as_ref())
            .map(|(k, _)| k.shape()[0])
            .unwrap_or(0)
    }

    /// Apply the window bound to a layer's cache: if the cache has more
    /// than `max_window` rows, drop the oldest rows (keeping the tail).
    /// No-op when unbounded or under the limit.
    pub fn clip_layer(&mut self, layer: usize) {
        let window = match self.max_window {
            Some(w) => w,
            None => return,
        };
        let Some(Some((k, v))) = self.layers.get_mut(layer) else {
            return;
        };
        let rows = k.shape()[0];
        if rows <= window { return; }
        let start = rows - window;
        let k_slice = k.slice(ndarray::s![start..rows, ..]).to_owned();
        let v_slice = v.slice(ndarray::s![start..rows, ..]).to_owned();
        *k = k_slice;
        *v = v_slice;
    }
}

/// GQA attention for a single decode step.
///
/// `q_new`: `[1, num_q * head_dim]` — Q for the new token only.
/// `k_full`: `[total_len, num_kv * head_dim]` — K_cache concatenated
/// with the new token's K_rope. Same for `v_full`.
///
/// Returns `[1, num_q * head_dim]` attention output for the new token.
/// No causal mask — the new token naturally sees everything, and the
/// cache only grew by 1 at the end.
#[allow(clippy::too_many_arguments)]
pub fn gqa_attention_decode_step(
    q_new: &Array2<f32>,
    k_full: &Array2<f32>,
    v_full: &Array2<f32>,
    num_q: usize,
    head_dim: usize,
    reps: usize,
    scale: f64,
    softcap: Option<f32>,
) -> Array2<f32> {
    let total_len = k_full.shape()[0];
    let mut out = Array2::<f32>::zeros((1, num_q * head_dim));
    let scale_f32 = scale as f32;

    let mut scores = vec![0.0f32; total_len];
    for h in 0..num_q {
        let kv_h = h / reps;
        let q_off = h * head_dim;
        let kv_off = kv_h * head_dim;

        let q_row = q_new.slice(ndarray::s![0, q_off..q_off + head_dim]);
        let k_block = k_full.slice(ndarray::s![.., kv_off..kv_off + head_dim]);
        let raw: ndarray::Array1<f32> = k_block.dot(&q_row);
        for i in 0..total_len {
            let mut s = raw[i] * scale_f32;
            if let Some(cap) = softcap {
                s = (s / cap).tanh() * cap;
            }
            scores[i] = s;
        }
        // Softmax
        let max_val = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f64;
        for s in scores.iter_mut() {
            let e = ((*s - max_val) as f64).exp();
            *s = e as f32;
            sum += e;
        }
        let inv_sum = (1.0 / sum) as f32;
        for s in scores.iter_mut() {
            *s *= inv_sum;
        }
        // Weighted sum of V
        let v_block = v_full.slice(ndarray::s![.., kv_off..kv_off + head_dim]);
        let scores_view = ndarray::ArrayView1::from(&scores[..]);
        let weighted_v = v_block.t().dot(&scores_view);
        for d in 0..head_dim {
            out[[0, q_off + d]] = weighted_v[d];
        }
    }
    out
}

/// Run the attention block for one decode step using an incremental KV
/// cache. `h_new` is the `[1, hidden]` residual for the new token.
/// `kv_entry` is the layer's existing `(K_cache, V_cache)` or `None` on
/// first step. `abs_position` is the new token's absolute RoPE
/// position — the caller must pass its true position in the original
/// sequence, NOT the clipped cache length (those differ under a
/// sliding window). Returns the updated `(h_post_attn, new_kv)`.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
pub fn run_attention_block_decode_step(
    weights: &crate::model::ModelWeights,
    h_new: &Array2<f32>,
    layer: usize,
    kv_entry: Option<&SharedKV>,
    abs_position: usize,
) -> Option<(Array2<f32>, SharedKV)> {
    use crate::forward::{add_bias, dot_proj};
    use crate::residual::{rms_norm_heads, rms_norm_heads_no_weight};

    let arch = &*weights.arch;
    let head_dim = arch.head_dim_for_layer(layer);
    let num_q = arch.num_q_heads_for_layer(layer);
    let num_kv = arch.num_kv_heads_for_layer(layer);
    let reps = num_q / num_kv;
    let scale = if arch.attention_multiplier() != 1.0 {
        arch.attention_multiplier() as f64
    } else {
        arch.attention_scale_for_layer(layer)
    };
    let norm_offset = arch.norm_weight_offset();
    let position = abs_position;

    let h_norm = crate::forward::apply_norm(
        weights, h_new, &arch.input_layernorm_key(layer), norm_offset,
    );

    let w_q = weights.tensors.get(&arch.attn_q_key(layer))?;
    let w_o = weights.tensors.get(&arch.attn_o_key(layer))?;
    let mut q_full = dot_proj(&h_norm, w_q);
    if let Some(bias) = arch.attn_q_bias_key(layer).and_then(|k| weights.vectors.get(&k)) {
        add_bias(&mut q_full, bias);
    }

    let qk_offset = weights.arch.qk_norm_weight_offset();
    let qk_norm_off = if qk_offset != 0.0 { qk_offset } else { norm_offset };
    let q_normed = match arch.attn_q_norm_key(layer).and_then(|k| weights.vectors.get(&k)) {
        Some(norm_w) => rms_norm_heads(&q_full, norm_w, num_q, head_dim, qk_norm_off),
        None => q_full,
    };
    let layer_rope_base = arch.rope_base_for_layer(layer);
    let rotary_frac = arch.rotary_fraction_for_layer(layer);
    let q_rope = apply_rope_partial_at(&q_normed, num_q, head_dim, layer_rope_base, rotary_frac, position);

    // New token's K, V — RoPE'd at `position`, then appended to cache.
    let w_k = weights.tensors.get(&arch.attn_k_key(layer))?;
    let v_from_k = !weights.tensors.contains_key(&arch.attn_v_key(layer));
    let w_v = if v_from_k { w_k } else { weights.tensors.get(&arch.attn_v_key(layer))? };

    let mut k_full_new = dot_proj(&h_norm, w_k);
    let mut v_full_new = dot_proj(&h_norm, w_v);
    if let Some(bias) = arch.attn_k_bias_key(layer).and_then(|k| weights.vectors.get(&k)) {
        add_bias(&mut k_full_new, bias);
    }
    if let Some(bias) = arch.attn_v_bias_key(layer).and_then(|k| weights.vectors.get(&k)) {
        add_bias(&mut v_full_new, bias);
    }
    if arch.has_v_norm() {
        v_full_new = rms_norm_heads_no_weight(&v_full_new, num_kv, head_dim);
    }
    let k_normed = match arch.attn_k_norm_key(layer).and_then(|k| weights.vectors.get(&k)) {
        Some(norm_w) => rms_norm_heads(&k_full_new, norm_w, num_kv, head_dim, qk_norm_off),
        None => k_full_new,
    };
    let k_new_rope = apply_rope_partial_at(&k_normed, num_kv, head_dim, layer_rope_base, rotary_frac, position);

    // Concatenate cache + new along seq axis.
    let (k_concat, v_concat) = match kv_entry {
        Some((k_cached, v_cached)) => {
            let kv_dim = num_kv * head_dim;
            let total = k_cached.shape()[0] + 1;
            let mut k_out = Array2::<f32>::zeros((total, kv_dim));
            let mut v_out = Array2::<f32>::zeros((total, kv_dim));
            k_out.slice_mut(ndarray::s![..k_cached.shape()[0], ..]).assign(k_cached);
            v_out.slice_mut(ndarray::s![..v_cached.shape()[0], ..]).assign(v_cached);
            k_out.slice_mut(ndarray::s![k_cached.shape()[0].., ..]).assign(&k_new_rope);
            v_out.slice_mut(ndarray::s![v_cached.shape()[0].., ..]).assign(&v_full_new);
            (k_out, v_out)
        }
        None => (k_new_rope, v_full_new),
    };

    let softcap = arch.attn_logit_softcapping();
    let attn_out = gqa_attention_decode_step(
        &q_rope, &k_concat, &v_concat,
        num_q, head_dim, reps, scale, softcap,
    );

    let mut attn_projected = dot_proj(&attn_out, w_o);
    if let Some(bias) = arch.attn_o_bias_key(layer).and_then(|k| weights.vectors.get(&k)) {
        add_bias(&mut attn_projected, bias);
    }

    let res_mult = arch.residual_multiplier();
    let h_post_attn = if arch.has_post_norms() {
        let normed = crate::forward::apply_norm(
            weights, &attn_projected, &arch.post_attention_layernorm_key(layer), norm_offset,
        );
        if res_mult != 1.0 { h_new + &(&normed * res_mult) } else { h_new + &normed }
    } else if res_mult != 1.0 {
        h_new + &(&attn_projected * res_mult)
    } else {
        h_new + &attn_projected
    };

    Some((h_post_attn, (k_concat, v_concat)))
}
