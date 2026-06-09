//! `run_attention_block_prefill_q4k_direct` — multi-row direct Q4_K × Q8_K
//! matvec for the attention Q/K/V/O projections during prefill.
//!
//! Mirrors [`super::block::run_attention_block_with_kv_out`] (the multi-row
//! prefill function that backs `kv_cache=None` and `cached_len=0` cases)
//! but reads Q/K/V/O weights as quantised bytes from the vindex and does
//! per-row Q4_K × Q8_K matvec instead of f32 BLAS GEMV against
//! `weights.tensors`.
//!
//! Together with [`super::q4k_direct::run_attention_block_decode_step_q4k_direct`]
//! and [`crate::ffn::Q4kDirectFfn`], this closes the loop on the "drop the
//! f32 dequant cache" arc — every path that previously read
//! `weights.tensors.get(attn_q_key)` etc. now has a direct-matvec equivalent.
//!
//! Per-row matvec is `q4k_q8k_matvec_into` (rayon-parallel AVX2). Running
//! N matvecs serially is slower than a single GEMM at large N because each
//! call pays a rayon-dispatch overhead — at seq=128 the per-row loop is
//! ~10-20× slower than the BLAS GEMM the f32 cache enabled. The RAM win
//! (≈12 GB on Gemma 3 4B) is the trade-off; a batched multi-row kernel is
//! the next-arc to close the prefill speed gap.

use larql_compute::cpu::ops::q4k_q8k_dot::{
    q4k_q8k_matmul_into, q4k_q8k_matvec_into, q6k_q8k_matvec_into, quantize_x_to_q8k,
};
use larql_vindex::VectorIndex;
use ndarray::Array2;

use super::SharedKV;
use crate::attention::gqa::gqa_attention;
use crate::attention::rope::apply_rope_partial;
use crate::forward::add_bias;
use crate::residual::{rms_norm_heads, rms_norm_heads_no_weight};

/// Per-row Q4_K/Q6_K dispatch with format-aware kernel selection.
fn matvec_row(
    out: &mut [f32],
    q8k_x: &larql_compute::cpu::ops::q4k_q8k_dot::Q8KActivation,
    w_bytes: &[u8],
    w_fmt: &str,
    rows: usize,
    cols: usize,
) {
    match w_fmt {
        "Q4_K" => q4k_q8k_matvec_into(out, q8k_x, w_bytes, rows, cols),
        "Q6_K" => q6k_q8k_matvec_into(out, q8k_x, w_bytes, rows, cols),
        other => panic!("unsupported attn weight format for q4k-direct prefill: {other}"),
    }
}

/// Multi-row matmul `[N, cols] × [rows, cols]^T = [N, rows]`.
///
/// Q4_K weights take the batched [`q4k_q8k_matmul_into`] path — one
/// rayon dispatch parallelising over the N input rows. Q6_K weights
/// still go through per-row [`q6k_q8k_matvec_into`] (no batched Q6_K
/// kernel yet — typically only the V projection on full-attention
/// layers; a separate arc).
///
/// Bit-exact against the per-row loop for Q4_K — see
/// `q4k_q8k_matmul_into_matches_per_row_matvec_loop_bit_exact` in
/// `larql-compute/src/cpu/ops/q4k_q8k_dot.rs`.
fn matmul_per_row(
    x_normed: &Array2<f32>,
    w_bytes: &[u8],
    w_fmt: &str,
    rows: usize,
    cols: usize,
) -> Array2<f32> {
    let n = x_normed.nrows();
    let mut out = Array2::<f32>::zeros((n, rows));

    // Quantise every input row to Q8K once. Same work either path —
    // the batched kernel needs them as a slice, the per-row path
    // discards them after each iteration.
    let q8ks: Vec<_> = (0..n)
        .map(|r| {
            let row_in = x_normed
                .row(r)
                .as_slice()
                .map(<[f32]>::to_vec)
                .unwrap_or_else(|| x_normed.row(r).iter().copied().collect());
            quantize_x_to_q8k(&row_in)
        })
        .collect();

    match w_fmt {
        "Q4_K" => {
            // Batched: one rayon dispatch over the N input rows.
            // The output is already laid out `[n * rows + r]` —
            // matches the `Array2<f32>` row-major layout for shape
            // `(n, rows)`, so we can pass `out.as_slice_mut()` straight
            // through.
            let out_slice = out
                .as_slice_mut()
                .expect("freshly-allocated row-major Array2 has contiguous slice");
            q4k_q8k_matmul_into(out_slice, &q8ks, w_bytes, rows, cols);
        }
        "Q6_K" => {
            for (r, q8k) in q8ks.iter().enumerate() {
                let mut out_flat = vec![0.0f32; rows];
                q6k_q8k_matvec_into(&mut out_flat, q8k, w_bytes, rows, cols);
                out.row_mut(r)
                    .iter_mut()
                    .zip(out_flat.iter())
                    .for_each(|(o, v)| *o = *v);
            }
        }
        other => panic!("unsupported attn weight format for q4k-direct prefill: {other}"),
    }
    out
}

/// Multi-row prefill attention with direct Q4_K × Q8_K Q/K/V/O matvec.
///
/// Returns `(h_post_attn, (k_rope, v_final))`. The K/V are the post-norm
/// post-RoPE views ready to be stashed in a `KvCache` snapshot by the
/// caller (same semantics as `run_attention_block_with_kv_out`'s return).
///
/// Pre-conditions enforced by the caller:
/// - `h.shape() == [seq_len, hidden]` with `seq_len >= 1`.
/// - `hidden % 256 == 0`.
/// - `(num_q * head_dim) % 256 == 0`.
/// - `arch.is_hybrid_moe() == false` (MoE prefill keeps the cache path).
/// - No `shared_kv` donor (cross-layer K/V share keeps the cache path).
/// - The layer's vindex Q/K/V/O bytes are present.
#[allow(clippy::type_complexity)]
pub fn run_attention_block_prefill_q4k_direct(
    weights: &crate::model::ModelWeights,
    index: &VectorIndex,
    h: &Array2<f32>,
    layer: usize,
) -> Option<(Array2<f32>, SharedKV)> {
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
    let hidden = h.ncols();
    let seq_len = h.nrows();
    let q_dim = num_q * head_dim;
    let kv_dim = num_kv * head_dim;

    let h_norm =
        crate::forward::apply_norm(weights, h, &arch.input_layernorm_key(layer), norm_offset);

    let attn = index.attn_q4k_layer_data(layer)?;

    // Q projection.
    let mut q_full = matmul_per_row(&h_norm, attn[0].0, attn[0].1, q_dim, hidden);
    if let Some(bias) = arch
        .attn_q_bias_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        add_bias(&mut q_full, bias);
    }

    let qk_offset = arch.qk_norm_weight_offset();
    let qk_norm_off = if qk_offset != 0.0 {
        qk_offset
    } else {
        norm_offset
    };
    let q_normed = match arch
        .attn_q_norm_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        Some(norm_w) => rms_norm_heads(&q_full, norm_w, num_q, head_dim, qk_norm_off),
        None => q_full,
    };
    let layer_rope_base = arch.rope_base_for_layer(layer);
    let rotary_frac = arch.rotary_fraction_for_layer(layer);
    let q_rope = apply_rope_partial(&q_normed, num_q, head_dim, layer_rope_base, rotary_frac);

    // K projection.
    let mut k_full = matmul_per_row(&h_norm, attn[1].0, attn[1].1, kv_dim, hidden);
    if let Some(bias) = arch
        .attn_k_bias_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        add_bias(&mut k_full, bias);
    }

    // V projection (Q6_K on Gemma 3; Q4_K on most others).
    let mut v_full = matmul_per_row(&h_norm, attn[2].0, attn[2].1, kv_dim, hidden);
    if let Some(bias) = arch
        .attn_v_bias_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        add_bias(&mut v_full, bias);
    }
    if arch.has_v_norm() {
        v_full = rms_norm_heads_no_weight(&v_full, num_kv, head_dim);
    }

    let k_normed = match arch
        .attn_k_norm_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        Some(norm_w) => rms_norm_heads(&k_full, norm_w, num_kv, head_dim, qk_norm_off),
        None => k_full,
    };
    let k_rope = apply_rope_partial(&k_normed, num_kv, head_dim, layer_rope_base, rotary_frac);

    // GQA with causal masking — multi-row, full seq × seq attention.
    let attn_out = gqa_attention(
        &q_rope, &k_rope, &v_full, num_q, head_dim, reps, scale, seq_len,
    );

    // O projection.
    let mut attn_projected = matmul_per_row(&attn_out, attn[3].0, attn[3].1, hidden, q_dim);
    if let Some(bias) = arch
        .attn_o_bias_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        add_bias(&mut attn_projected, bias);
    }

    // Residual + post-attn-norm (Gemma 3's `has_post_norms` adds an extra
    // norm before adding to the residual).
    let res_mult = arch.residual_multiplier();
    let h_post_attn = if arch.has_post_norms() {
        let normed = crate::forward::apply_norm(
            weights,
            &attn_projected,
            &arch.post_attention_layernorm_key(layer),
            norm_offset,
        );
        if res_mult != 1.0 {
            h + &(&normed * res_mult)
        } else {
            h + &normed
        }
    } else if res_mult != 1.0 {
        h + &(&attn_projected * res_mult)
    } else {
        h + &attn_projected
    };

    Some((h_post_attn, (k_rope, v_full)))
}
