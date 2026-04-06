//! Attention computation — GQA, RoPE, causal masking.

use ndarray::Array2;

/// Apply Rotary Position Embeddings to Q or K.
/// Uses split-half pairing: (x[i], x[i + half_dim]).
/// This matches MLX traditional=False and HuggingFace's default.
/// x: (seq_len, num_heads * head_dim)
pub fn apply_rope(
    x: &Array2<f32>,
    num_heads: usize,
    head_dim: usize,
    rope_base: f64,
) -> Array2<f32> {
    let seq_len = x.shape()[0];
    let mut out = x.clone();

    let half_dim = head_dim / 2;
    let inv_freq: Vec<f64> = (0..half_dim)
        .map(|i| 1.0 / rope_base.powf(2.0 * i as f64 / head_dim as f64))
        .collect();

    for pos in 0..seq_len {
        for h in 0..num_heads {
            let offset = h * head_dim;
            for i in 0..half_dim {
                let theta = pos as f64 * inv_freq[i];
                let cos_t = theta.cos() as f32;
                let sin_t = theta.sin() as f32;

                let x0 = x[[pos, offset + i]];
                let x1 = x[[pos, offset + half_dim + i]];

                out[[pos, offset + i]] = x0 * cos_t - x1 * sin_t;
                out[[pos, offset + half_dim + i]] = x0 * sin_t + x1 * cos_t;
            }
        }
    }
    out
}

/// Per-head attention weights for the last token position.
/// `weights[head]` = vec of attention scores over all positions.
pub struct AttentionWeights {
    /// Per-head attention distribution for the last sequence position.
    /// `heads[h][j]` = attention weight from last token to position j.
    pub heads: Vec<Vec<f32>>,
}

/// Run the full attention block for a layer: norm → Q/K/V projection → bias →
/// QK norm → RoPE → GQA attention → O projection → bias → residual add.
///
/// Shared implementation used by both forward.rs and trace/capture.rs.
/// Returns (h_post_attn, attn_projected_pre_residual, optional_attention_weights).
#[allow(clippy::too_many_arguments)]
pub fn run_attention_block(
    weights: &crate::model::ModelWeights,
    h: &Array2<f32>,
    layer: usize,
    capture_attention: bool,
) -> Option<(Array2<f32>, Array2<f32>, Option<AttentionWeights>)> {
    use crate::forward::{dot_proj, add_bias};
    use crate::residual::rms_norm_heads;

    let arch = &*weights.arch;
    let head_dim = weights.head_dim;
    let num_q = weights.num_q_heads;
    let num_kv = weights.num_kv_heads;
    let reps = num_q / num_kv;
    let scale = if arch.attention_multiplier() != 1.0 {
        arch.attention_multiplier() as f64
    } else {
        arch.attention_scale()
    };
    let seq_len = h.shape()[0];
    let norm_offset = arch.norm_weight_offset();

    // Input norm
    let h_norm = crate::forward::apply_norm(weights, h, &arch.input_layernorm_key(layer), norm_offset);

    // Q/K/V projections
    let w_q = weights.tensors.get(&arch.attn_q_key(layer))?;
    let w_k = weights.tensors.get(&arch.attn_k_key(layer)).unwrap();
    let w_v = weights.tensors.get(&arch.attn_v_key(layer)).unwrap();
    let w_o = weights.tensors.get(&arch.attn_o_key(layer)).unwrap();

    let mut q_full = dot_proj(&h_norm, w_q);
    let mut k_full = dot_proj(&h_norm, w_k);
    let mut v_full = dot_proj(&h_norm, w_v);

    // Attention bias (Qwen2/2.5)
    if let Some(bias) = arch.attn_q_bias_key(layer).and_then(|k| weights.vectors.get(&k)) {
        add_bias(&mut q_full, bias);
    }
    if let Some(bias) = arch.attn_k_bias_key(layer).and_then(|k| weights.vectors.get(&k)) {
        add_bias(&mut k_full, bias);
    }
    if let Some(bias) = arch.attn_v_bias_key(layer).and_then(|k| weights.vectors.get(&k)) {
        add_bias(&mut v_full, bias);
    }

    // Per-head QK norm (Qwen2/Granite)
    let qk_offset = weights.arch.qk_norm_weight_offset();
    let qk_norm_off = if qk_offset != 0.0 { qk_offset } else { norm_offset };
    let q_normed = match arch.attn_q_norm_key(layer).and_then(|k| weights.vectors.get(&k)) {
        Some(norm_w) => rms_norm_heads(&q_full, norm_w, num_q, head_dim, qk_norm_off),
        None => q_full,
    };
    let k_normed = match arch.attn_k_norm_key(layer).and_then(|k| weights.vectors.get(&k)) {
        Some(norm_w) => rms_norm_heads(&k_full, norm_w, num_kv, head_dim, qk_norm_off),
        None => k_full,
    };

    // RoPE
    let layer_rope_base = arch.rope_base_for_layer(layer);
    let q_rope = apply_rope(&q_normed, num_q, head_dim, layer_rope_base);
    let k_rope = apply_rope(&k_normed, num_kv, head_dim, layer_rope_base);

    // GQA attention
    let softcap = arch.attn_logit_softcapping();
    let (attn_out, attn_weights) = gqa_attention_with_weights(
        &q_rope, &k_rope, &v_full, num_q, head_dim, reps, scale, seq_len,
        capture_attention, softcap,
    );

    // O projection
    let mut attn_projected = dot_proj(&attn_out, w_o);
    if let Some(bias) = arch.attn_o_bias_key(layer).and_then(|k| weights.vectors.get(&k)) {
        add_bias(&mut attn_projected, bias);
    }

    // Residual connection
    let res_mult = arch.residual_multiplier();
    let h_post_attn = if arch.has_post_norms() {
        let normed = crate::forward::apply_norm(
            weights, &attn_projected, &arch.post_attention_layernorm_key(layer), norm_offset,
        );
        if res_mult != 1.0 { h + &(&normed * res_mult) } else { h + &normed }
    } else if res_mult != 1.0 {
        h + &(&attn_projected * res_mult)
    } else {
        h + &attn_projected
    };

    Some((h_post_attn, attn_projected, attn_weights))
}

/// GPU-accelerated attention block. Same as `run_attention_block` but routes
/// Q/K/V/O projections through the ComputeBackend (Metal, CUDA, or CPU).
/// If backend is None, falls back to CPU BLAS.
pub fn run_attention_block_gpu(
    weights: &crate::model::ModelWeights,
    h: &Array2<f32>,
    layer: usize,
    capture_attention: bool,
    backend: Option<&dyn larql_compute::ComputeBackend>,
) -> Option<(Array2<f32>, Array2<f32>, Option<AttentionWeights>)> {
    use larql_compute::dot_proj_gpu;
    use crate::forward::add_bias;
    use crate::residual::rms_norm_heads;

    let arch = &*weights.arch;
    let head_dim = weights.head_dim;
    let num_q = weights.num_q_heads;
    let num_kv = weights.num_kv_heads;
    let reps = num_q / num_kv;
    let scale = if arch.attention_multiplier() != 1.0 {
        arch.attention_multiplier() as f64
    } else {
        arch.attention_scale()
    };
    let seq_len = h.shape()[0];
    let norm_offset = arch.norm_weight_offset();

    let h_norm = crate::forward::apply_norm(weights, h, &arch.input_layernorm_key(layer), norm_offset);

    let w_q = weights.tensors.get(&arch.attn_q_key(layer))?;
    let w_k = weights.tensors.get(&arch.attn_k_key(layer)).unwrap();
    let w_v = weights.tensors.get(&arch.attn_v_key(layer)).unwrap();
    let w_o = weights.tensors.get(&arch.attn_o_key(layer)).unwrap();

    // Q/K/V projections — GPU-accelerated
    let mut q_full = dot_proj_gpu(&h_norm, w_q, backend);
    let mut k_full = dot_proj_gpu(&h_norm, w_k, backend);
    let mut v_full = dot_proj_gpu(&h_norm, w_v, backend);

    if let Some(bias) = arch.attn_q_bias_key(layer).and_then(|k| weights.vectors.get(&k)) {
        add_bias(&mut q_full, bias);
    }
    if let Some(bias) = arch.attn_k_bias_key(layer).and_then(|k| weights.vectors.get(&k)) {
        add_bias(&mut k_full, bias);
    }
    if let Some(bias) = arch.attn_v_bias_key(layer).and_then(|k| weights.vectors.get(&k)) {
        add_bias(&mut v_full, bias);
    }

    let qk_offset = weights.arch.qk_norm_weight_offset();
    let qk_norm_off = if qk_offset != 0.0 { qk_offset } else { norm_offset };
    let q_normed = match arch.attn_q_norm_key(layer).and_then(|k| weights.vectors.get(&k)) {
        Some(norm_w) => rms_norm_heads(&q_full, norm_w, num_q, head_dim, qk_norm_off),
        None => q_full,
    };
    let k_normed = match arch.attn_k_norm_key(layer).and_then(|k| weights.vectors.get(&k)) {
        Some(norm_w) => rms_norm_heads(&k_full, norm_w, num_kv, head_dim, qk_norm_off),
        None => k_full,
    };

    let layer_rope_base = arch.rope_base_for_layer(layer);
    let q_rope = apply_rope(&q_normed, num_q, head_dim, layer_rope_base);
    let k_rope = apply_rope(&k_normed, num_kv, head_dim, layer_rope_base);

    let softcap = arch.attn_logit_softcapping();
    let (attn_out, attn_weights) = gqa_attention_with_weights(
        &q_rope, &k_rope, &v_full, num_q, head_dim, reps, scale, seq_len,
        capture_attention, softcap,
    );

    // O projection — GPU-accelerated
    let mut attn_projected = dot_proj_gpu(&attn_out, w_o, backend);
    if let Some(bias) = arch.attn_o_bias_key(layer).and_then(|k| weights.vectors.get(&k)) {
        add_bias(&mut attn_projected, bias);
    }

    let res_mult = arch.residual_multiplier();
    let h_post_attn = if arch.has_post_norms() {
        let normed = crate::forward::apply_norm(
            weights, &attn_projected, &arch.post_attention_layernorm_key(layer), norm_offset,
        );
        if res_mult != 1.0 { h + &(&normed * res_mult) } else { h + &normed }
    } else if res_mult != 1.0 {
        h + &(&attn_projected * res_mult)
    } else {
        h + &attn_projected
    };

    Some((h_post_attn, attn_projected, attn_weights))
}

/// Grouped-query attention with causal masking (no weight capture).
///
/// q: (seq, num_q * head_dim), k: (seq, num_kv * head_dim), v: same as k
#[allow(clippy::too_many_arguments)]
pub fn gqa_attention(
    q: &Array2<f32>,
    k: &Array2<f32>,
    v: &Array2<f32>,
    num_q: usize,
    head_dim: usize,
    reps: usize,
    scale: f64,
    seq_len: usize,
) -> Array2<f32> {
    let (out, _) = gqa_attention_with_weights(q, k, v, num_q, head_dim, reps, scale, seq_len, false, None);
    out
}

/// GQA attention that optionally captures per-head attention weights for the last token.
/// `softcap`: if Some(cap), apply tanh(scores/cap)*cap before softmax (Gemma2).
///
/// BLAS-fused attention: uses `gemv` (matrix-vector multiply via Accelerate/AMX)
/// for the Q·K dot products and softmax·V accumulation, but never allocates the
/// full [seq, seq] attention matrix. Per query position `qi`:
///   1. `scores = K[0..=qi] @ Q[qi]` — one BLAS gemv
///   2. scale + softcap + two-pass softmax on the scores vector
///   3. `output = V[0..=qi]^T @ softmax_scores` — one BLAS gemv
///
/// Memory: O(seq) temporary per position, vs O(seq²) for the materialized path.
/// At seq=6 this is negligible; at seq=512+ the savings are significant.
#[allow(clippy::too_many_arguments)]
pub fn gqa_attention_with_weights(
    q: &Array2<f32>,
    k: &Array2<f32>,
    v: &Array2<f32>,
    num_q: usize,
    head_dim: usize,
    reps: usize,
    scale: f64,
    seq_len: usize,
    capture: bool,
    softcap: Option<f32>,
) -> (Array2<f32>, Option<AttentionWeights>) {
    let mut out = Array2::<f32>::zeros((seq_len, num_q * head_dim));
    let mut captured_heads: Vec<Vec<f32>> = if capture {
        Vec::with_capacity(num_q)
    } else {
        Vec::new()
    };

    let scale_f32 = scale as f32;
    let last_pos = seq_len - 1;

    // Reusable buffer for softmax scores (avoids per-position allocation)
    let mut scores_buf = vec![0.0f32; seq_len];

    for h in 0..num_q {
        let kv_h = h / reps;
        let q_off = h * head_dim;
        let kv_off = kv_h * head_dim;

        for qi in 0..seq_len {
            let causal_len = qi + 1; // positions 0..=qi

            // ── BLAS gemv: compute all causal scores at once ──
            // scores[0..=qi] = K[0..=qi, kv_off..kv_off+hd] @ Q[qi, q_off..q_off+hd]
            let q_row = q.slice(ndarray::s![qi, q_off..q_off + head_dim]);
            let k_block = k.slice(ndarray::s![0..causal_len, kv_off..kv_off + head_dim]);
            let raw_scores = k_block.dot(&q_row); // [causal_len] via BLAS gemv

            // ── Scale + softcap ──
            for i in 0..causal_len {
                let mut s = raw_scores[i] * scale_f32;
                if let Some(cap) = softcap {
                    s = (s / cap).tanh() * cap;
                }
                scores_buf[i] = s;
            }

            // ── Two-pass softmax with f64 accumulation ──
            let max_val = scores_buf[..causal_len]
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f64;
            for i in 0..causal_len {
                let e = ((scores_buf[i] - max_val) as f64).exp();
                scores_buf[i] = e as f32;
                sum += e;
            }
            let inv_sum = (1.0 / sum) as f32;
            for i in 0..causal_len {
                scores_buf[i] *= inv_sum;
            }

            // ── Capture last-token attention weights ──
            if capture && qi == last_pos {
                let mut captured = vec![0.0f32; seq_len];
                captured[..causal_len].copy_from_slice(&scores_buf[..causal_len]);
                captured_heads.push(captured);
            }

            // ── BLAS gemv: weighted V accumulation ──
            // output[qi] = V[0..=qi, kv_off..kv_off+hd]^T @ softmax_scores[0..=qi]
            let v_block = v.slice(ndarray::s![0..causal_len, kv_off..kv_off + head_dim]);
            let scores_view = ndarray::ArrayView1::from(&scores_buf[..causal_len]);
            let weighted_v = v_block.t().dot(&scores_view); // [head_dim] via BLAS gemv

            // Write output
            for d in 0..head_dim {
                out[[qi, q_off + d]] = weighted_v[d];
            }
        }
    }

    let weights = if capture {
        Some(AttentionWeights {
            heads: captured_heads,
        })
    } else {
        None
    };

    (out, weights)
}

/// Same as run_attention_block_gpu but also returns K (after RoPE) and V for KV cache.
pub fn run_attention_with_kv(
    weights: &crate::model::ModelWeights,
    h: &Array2<f32>,
    layer: usize,
) -> Option<(Array2<f32>, Array2<f32>, Array2<f32>)> {
    use crate::forward::{apply_norm, add_bias, dot_proj};
    use crate::residual::rms_norm_heads;

    let arch = &*weights.arch;
    let (hd, nq, nkv) = (weights.head_dim, weights.num_q_heads, weights.num_kv_heads);
    let reps = nq / nkv;
    let scale = if arch.attention_multiplier() != 1.0 { arch.attention_multiplier() as f64 } else { arch.attention_scale() };
    let seq_len = h.shape()[0];
    let norm_off = arch.norm_weight_offset();

    let h_norm = apply_norm(weights, h, &arch.input_layernorm_key(layer), norm_off);
    let (wq, wk, wv, wo) = (
        weights.tensors.get(&arch.attn_q_key(layer))?,
        weights.tensors.get(&arch.attn_k_key(layer))?,
        weights.tensors.get(&arch.attn_v_key(layer))?,
        weights.tensors.get(&arch.attn_o_key(layer))?,
    );

    let (mut q, mut k, mut v) = (dot_proj(&h_norm, wq), dot_proj(&h_norm, wk), dot_proj(&h_norm, wv));
    for (proj, bias_fn) in [(&mut q, arch.attn_q_bias_key(layer) as Option<String>),
                             (&mut k, arch.attn_k_bias_key(layer)),
                             (&mut v, arch.attn_v_bias_key(layer))] {
        if let Some(b) = bias_fn.and_then(|key| weights.vectors.get(&key)) { add_bias(proj, b); }
    }

    let qk_off = if arch.qk_norm_weight_offset() != 0.0 { arch.qk_norm_weight_offset() } else { norm_off };
    let q = match arch.attn_q_norm_key(layer).and_then(|k| weights.vectors.get(&k)) {
        Some(w) => rms_norm_heads(&q, w, nq, hd, qk_off), None => q,
    };
    let k = match arch.attn_k_norm_key(layer).and_then(|k| weights.vectors.get(&k)) {
        Some(w) => rms_norm_heads(&k, w, nkv, hd, qk_off), None => k,
    };

    let rb = arch.rope_base_for_layer(layer);
    let q_r = apply_rope(&q, nq, hd, rb);
    let k_r = apply_rope(&k, nkv, hd, rb);

    let (attn_out, _) = gqa_attention_with_weights(
        &q_r, &k_r, &v, nq, hd, reps, scale, seq_len, false, arch.attn_logit_softcapping());
    let mut o = dot_proj(&attn_out, wo);
    if let Some(b) = arch.attn_o_bias_key(layer).and_then(|k| weights.vectors.get(&k)) { add_bias(&mut o, b); }

    let rm = arch.residual_multiplier();
    let h_out = if arch.has_post_norms() {
        let n = apply_norm(weights, &o, &arch.post_attention_layernorm_key(layer), norm_off);
        if rm != 1.0 { h + &(&n * rm) } else { h + &n }
    } else if rm != 1.0 { h + &(&o * rm) } else { h + &o };

    Some((h_out, k_r, v))
}

/// Q4 attention projection: single projection via Q4 matvec through ComputeBackend.
///
/// Replaces `dot_proj_gpu(h, w, backend)` when Q4 weight data is available.
/// Returns [seq_len, out_dim] f32 result.
///
/// The caller provides raw Q4 bytes for the weight matrix and the number of
/// output rows (num_rows = out_dim for projections, since Q4 layout is [out, in]).
pub fn q4_attention_proj(
    h: &Array2<f32>,
    q4_data: &[u8],
    num_rows: usize,
    hidden: usize,
    backend: &dyn larql_compute::ComputeBackend,
) -> Option<Array2<f32>> {
    if !backend.has_q4() { return None; }
    let seq_len = h.shape()[0];
    let mut out = Array2::<f32>::zeros((seq_len, num_rows));

    for s in 0..seq_len {
        let x_row = h.row(s);
        let x_slice = x_row.as_slice()?;
        let (q8_x, q8_scales) = larql_compute::cpu::q4::quantize_to_q8(x_slice);
        let scores = backend.q4_matvec(q4_data, &q8_x, &q8_scales, num_rows, hidden)?;
        let mut out_row = out.row_mut(s);
        for j in 0..num_rows { out_row[j] = scores[j]; }
    }
    Some(out)
}
