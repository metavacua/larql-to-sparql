//! `run_attention_block_decode_step_q4k_direct` — direct Q4_K × Q8_K matvec
//! for the attention Q/K/V/O projections on the decode step.
//!
//! Mirrors [`run_attention_block_decode_step_backend`](super::decode::run_attention_block_decode_step_backend)
//! but reads the Q/K/V/O weights as quantised bytes from the vindex and
//! dispatches to the AVX2 kernels in `larql_compute::cpu::ops::q4k_q8k_dot`
//! instead of going through `weights.tensors` + BLAS GEMV. Same speedup
//! mechanism as [`crate::ffn::Q4kDirectFfn`] but for attention.
//!
//! ## Alignment requirements
//!
//! The Q8_K activation quantiser requires the input length to be a multiple
//! of 256. Two activation streams enter the kernels:
//!
//! - **`h_norm`** for Q/K/V: shape `[1, hidden]`. `hidden % 256 == 0` is
//!   required. Gemma 3 4B (hidden=2560) is aligned; 1B (hidden=1152) is not.
//! - **`attn_out`** for O: shape `[1, num_q * head_dim]`. That product must
//!   also be a multiple of 256. Gemma 3 4B has `num_q * head_dim = 8 * 256 = 2048`
//!   which is aligned.
//!
//! Callers gate this path on the alignment check; non-aligned models fall
//! back to [`super::decode::run_attention_block_decode_step`] over the
//! dequant cache.

use larql_compute::cpu::ops::q4k_q8k_dot::{
    q4k_q8k_matvec_into, q6k_q8k_matvec_into, quantize_x_to_q8k,
};
use larql_vindex::VectorIndex;
use ndarray::Array2;

use super::SharedKV;
use crate::attention::decode::gqa_attention_decode_step;
use crate::attention::rope::apply_rope_partial_at;

/// Matvec a single row of f32 input against a Q4_K- or Q6_K-encoded weight
/// matrix. `q8k_x` MUST be the Q8_K-quantised input row (its `qs.len()` ==
/// `cols`). The output buffer is sized `rows` and is fully overwritten.
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
        other => panic!("unsupported attn weight format for q4k-direct path: {other}"),
    }
}

/// Decode-step attention with direct Q4_K × Q8_K matvec for Q, K, V, and O
/// projections. Returns `(h_post_attn, (k_concat, v_concat))` — the same
/// signature as [`super::decode::run_attention_block_decode_step`].
///
/// Pre-conditions enforced by the caller:
/// - `h_new.shape() == [1, hidden]`
/// - `hidden % 256 == 0`
/// - `(num_q * head_dim) % 256 == 0`
/// - The vindex has Q4_K or Q6_K bytes for the layer's Q, K, V, O.
#[allow(clippy::type_complexity)]
pub fn run_attention_block_decode_step_q4k_direct(
    weights: &crate::model::ModelWeights,
    index: &VectorIndex,
    h_new: &Array2<f32>,
    layer: usize,
    kv_entry: Option<&SharedKV>,
    abs_position: usize,
) -> Option<(Array2<f32>, SharedKV)> {
    use crate::forward::add_bias;
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
    let hidden = h_new.ncols();
    let q_dim = num_q * head_dim;
    let kv_dim = num_kv * head_dim;

    let h_norm = crate::forward::apply_norm(
        weights,
        h_new,
        &arch.input_layernorm_key(layer),
        norm_offset,
    );

    let attn = index.attn_q4k_layer_data(layer)?;

    // Q8K-quantise h_norm once and reuse for Q, K, V (all three share the
    // same activation row).
    let h_norm_flat = h_norm.as_slice().expect("h_norm contiguous");
    let h_q8k = quantize_x_to_q8k(h_norm_flat);

    // Q projection.
    let mut q_flat = vec![0.0f32; q_dim];
    matvec_row(&mut q_flat, &h_q8k, attn[0].0, attn[0].1, q_dim, hidden);
    let mut q_full = Array2::from_shape_vec((1, q_dim), q_flat).expect("q shape");
    if let Some(bias) = arch
        .attn_q_bias_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        add_bias(&mut q_full, bias);
    }

    let qk_offset = weights.arch.qk_norm_weight_offset();
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
    let q_rope = apply_rope_partial_at(
        &q_normed,
        num_q,
        head_dim,
        layer_rope_base,
        rotary_frac,
        position,
    );

    // K projection.
    let mut k_flat = vec![0.0f32; kv_dim];
    matvec_row(&mut k_flat, &h_q8k, attn[1].0, attn[1].1, kv_dim, hidden);
    let mut k_full_new = Array2::from_shape_vec((1, kv_dim), k_flat).expect("k shape");

    // V projection. Some archs share K with V (no separate V tensor); in
    // that case `attn[2]` will be the same Q4_K bytes as `attn[1]`.
    let mut v_flat = vec![0.0f32; kv_dim];
    matvec_row(&mut v_flat, &h_q8k, attn[2].0, attn[2].1, kv_dim, hidden);
    let mut v_full_new = Array2::from_shape_vec((1, kv_dim), v_flat).expect("v shape");

    if let Some(bias) = arch
        .attn_k_bias_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        add_bias(&mut k_full_new, bias);
    }
    if let Some(bias) = arch
        .attn_v_bias_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        add_bias(&mut v_full_new, bias);
    }
    if arch.has_v_norm() {
        v_full_new = rms_norm_heads_no_weight(&v_full_new, num_kv, head_dim);
    }
    let k_normed = match arch
        .attn_k_norm_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        Some(norm_w) => rms_norm_heads(&k_full_new, norm_w, num_kv, head_dim, qk_norm_off),
        None => k_full_new,
    };
    let k_new_rope = apply_rope_partial_at(
        &k_normed,
        num_kv,
        head_dim,
        layer_rope_base,
        rotary_frac,
        position,
    );

    // Concatenate cache + new along seq axis.
    let (k_concat, v_concat) = match kv_entry {
        Some((k_cached, v_cached)) => {
            let total = k_cached.shape()[0] + 1;
            let mut k_out = Array2::<f32>::zeros((total, kv_dim));
            let mut v_out = Array2::<f32>::zeros((total, kv_dim));
            k_out
                .slice_mut(ndarray::s![..k_cached.shape()[0], ..])
                .assign(k_cached);
            v_out
                .slice_mut(ndarray::s![..v_cached.shape()[0], ..])
                .assign(v_cached);
            k_out
                .slice_mut(ndarray::s![k_cached.shape()[0].., ..])
                .assign(&k_new_rope);
            v_out
                .slice_mut(ndarray::s![v_cached.shape()[0].., ..])
                .assign(&v_full_new);
            (k_out, v_out)
        }
        None => (k_new_rope, v_full_new),
    };

    let softcap = arch.attn_logit_softcapping();
    let attn_out = gqa_attention_decode_step(
        &q_rope, &k_concat, &v_concat, num_q, head_dim, reps, scale, softcap,
    );

    // O projection — activation is `attn_out` with dim `q_dim`. Requires
    // `q_dim % 256 == 0` for Q8_K quantisation.
    let attn_out_flat = attn_out.as_slice().expect("attn_out contiguous");
    let attn_out_q8k = quantize_x_to_q8k(attn_out_flat);
    let mut o_flat = vec![0.0f32; hidden];
    matvec_row(
        &mut o_flat,
        &attn_out_q8k,
        attn[3].0,
        attn[3].1,
        hidden,
        q_dim,
    );
    let mut attn_projected = Array2::from_shape_vec((1, hidden), o_flat).expect("o shape");
    if let Some(bias) = arch
        .attn_o_bias_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        add_bias(&mut attn_projected, bias);
    }

    let res_mult = arch.residual_multiplier();
    let h_post_attn = if arch.has_post_norms() {
        let normed = crate::forward::apply_norm(
            weights,
            &attn_projected,
            &arch.post_attention_layernorm_key(layer),
            norm_offset,
        );
        if res_mult != 1.0 {
            h_new + &(&normed * res_mult)
        } else {
            h_new + &normed
        }
    } else if res_mult != 1.0 {
        h_new + &(&attn_projected * res_mult)
    } else {
        h_new + &attn_projected
    };

    Some((h_post_attn, (k_concat, v_concat)))
}
