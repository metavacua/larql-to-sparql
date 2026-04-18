//! CPU attention block — full layer attention computation.
//!
//! norm → Q/K/V projection → bias → V-norm → QK-norm → RoPE → GQA → O projection → residual.
//! Supports KV sharing (reuse K/V from a source layer).

use ndarray::Array2;
use super::{AttentionWeights, SharedKV};
use super::rope::apply_rope_partial;
use super::gqa::gqa_attention_with_weights;

/// Run the full attention block. Returns (h_post_attn, attn_projected, optional_weights).
#[allow(clippy::too_many_arguments)]
pub fn run_attention_block(
    weights: &crate::model::ModelWeights,
    h: &Array2<f32>,
    layer: usize,
    capture_attention: bool,
) -> Option<(Array2<f32>, Array2<f32>, Option<AttentionWeights>)> {
    run_attention_block_shared(weights, h, layer, capture_attention, None)
}

/// Run attention with optional shared K/V, returning K/V for caching.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
pub fn run_attention_block_with_kv_out(
    weights: &crate::model::ModelWeights,
    h: &Array2<f32>,
    layer: usize,
    capture_attention: bool,
    shared_kv: Option<&SharedKV>,
) -> Option<(Array2<f32>, Array2<f32>, Option<AttentionWeights>, Array2<f32>, Array2<f32>)> {
    run_attention_block_core(weights, h, layer, capture_attention, shared_kv)
}

/// Run attention with optional shared K/V (discards K/V output).
#[allow(clippy::too_many_arguments)]
pub fn run_attention_block_shared(
    weights: &crate::model::ModelWeights,
    h: &Array2<f32>,
    layer: usize,
    capture_attention: bool,
    shared_kv: Option<&SharedKV>,
) -> Option<(Array2<f32>, Array2<f32>, Option<AttentionWeights>)> {
    let (h_post, attn_proj, attn_w, _, _) =
        run_attention_block_core(weights, h, layer, capture_attention, shared_kv)?;
    Some((h_post, attn_proj, attn_w))
}

/// Core attention block implementation.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
fn run_attention_block_core(
    weights: &crate::model::ModelWeights,
    h: &Array2<f32>,
    layer: usize,
    capture_attention: bool,
    shared_kv: Option<&SharedKV>,
) -> Option<(Array2<f32>, Array2<f32>, Option<AttentionWeights>, Array2<f32>, Array2<f32>)> {
    use crate::forward::{dot_proj, add_bias};
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
    let seq_len = h.shape()[0];
    let norm_offset = arch.norm_weight_offset();

    // Layer-0 stage dumps, paired with the Metal side via
    // LARQL_CPU_STAGE_DUMP=<dir>. Scoped to layer 0 for noise budget.
    let stage_dump = if layer == 0 { std::env::var("LARQL_CPU_STAGE_DUMP").ok() } else { None };
    let dump_f32 = |name: &str, arr: &Array2<f32>| {
        if let Some(ref dir) = stage_dump {
            let slice = arr.as_slice().unwrap_or(&[]);
            let bytes: Vec<u8> = slice.iter().flat_map(|v| v.to_le_bytes()).collect();
            let _ = std::fs::write(format!("{dir}/cpu_L0_{name}.f32"), &bytes);
        }
    };

    // Input norm
    let h_norm = crate::forward::apply_norm(weights, h, &arch.input_layernorm_key(layer), norm_offset);
    dump_f32("norm_out", &h_norm);

    // Q projection (always from current hidden state)
    let w_q = weights.tensors.get(&arch.attn_q_key(layer))?;
    let w_o = weights.tensors.get(&arch.attn_o_key(layer)).unwrap();
    let mut q_full = dot_proj(&h_norm, w_q);
    if let Some(bias) = arch.attn_q_bias_key(layer).and_then(|k| weights.vectors.get(&k)) {
        add_bias(&mut q_full, bias);
    }
    dump_f32("q_out_raw", &q_full);

    // QK norm on Q
    let qk_offset = weights.arch.qk_norm_weight_offset();
    let qk_norm_off = if qk_offset != 0.0 { qk_offset } else { norm_offset };
    let q_normed = match arch.attn_q_norm_key(layer).and_then(|k| weights.vectors.get(&k)) {
        Some(norm_w) => rms_norm_heads(&q_full, norm_w, num_q, head_dim, qk_norm_off),
        None => q_full,
    };
    dump_f32("q_out_after_qk_norm", &q_normed);

    // RoPE on Q
    let layer_rope_base = arch.rope_base_for_layer(layer);
    let rotary_frac = arch.rotary_fraction_for_layer(layer);
    let q_rope = apply_rope_partial(&q_normed, num_q, head_dim, layer_rope_base, rotary_frac);

    // K/V: either from shared cache or computed fresh
    let (k_rope, v_final) = if let Some((cached_k, cached_v)) = shared_kv {
        (cached_k.clone(), cached_v.clone())
    } else {
        let w_k = weights.tensors.get(&arch.attn_k_key(layer)).unwrap();
        let v_from_k = !weights.tensors.contains_key(&arch.attn_v_key(layer));
        let w_v = if v_from_k { w_k } else { weights.tensors.get(&arch.attn_v_key(layer)).unwrap() };

        let mut k_full = dot_proj(&h_norm, w_k);
        let mut v_full = dot_proj(&h_norm, w_v);

        if let Some(bias) = arch.attn_k_bias_key(layer).and_then(|k| weights.vectors.get(&k)) {
            add_bias(&mut k_full, bias);
        }
        if let Some(bias) = arch.attn_v_bias_key(layer).and_then(|k| weights.vectors.get(&k)) {
            add_bias(&mut v_full, bias);
        }

        if arch.has_v_norm() {
            v_full = rms_norm_heads_no_weight(&v_full, num_kv, head_dim);
        }

        let k_normed = match arch.attn_k_norm_key(layer).and_then(|k| weights.vectors.get(&k)) {
            Some(norm_w) => rms_norm_heads(&k_full, norm_w, num_kv, head_dim, qk_norm_off),
            None => k_full,
        };

        let k_r = apply_rope_partial(&k_normed, num_kv, head_dim, layer_rope_base, rotary_frac);
        (k_r, v_full)
    };

    dump_f32("q_out_after_rope", &q_rope);

    // GQA attention
    let softcap = arch.attn_logit_softcapping();
    let (attn_out, attn_weights) = gqa_attention_with_weights(
        &q_rope, &k_rope, &v_final, num_q, head_dim, reps, scale, seq_len,
        capture_attention, softcap,
    );
    dump_f32("attn_out", &attn_out);

    // O projection
    let mut attn_projected = dot_proj(&attn_out, w_o);
    if let Some(bias) = arch.attn_o_bias_key(layer).and_then(|k| weights.vectors.get(&k)) {
        add_bias(&mut attn_projected, bias);
    }
    dump_f32("o_out", &attn_projected);

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

    Some((h_post_attn, attn_projected, attn_weights, k_rope, v_final))
}
