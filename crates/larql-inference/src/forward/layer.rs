//! Layer dispatch — runs attention + FFN + PLE + layer_scalar for a single layer.
//!
//! Orchestrates the per-layer computation: attention (with optional KV sharing),
//! FFN, per-layer embeddings, and layer scalar multiplication.

use ndarray::Array2;
use crate::attention::{AttentionWeights, SharedKV};
use crate::ffn::FfnBackend;
use crate::model::ModelWeights;
use crate::residual::rms_norm;
use super::apply_norm;
use super::ple::{apply_per_layer_embedding};

/// Public wrapper for run_attention — used by diagnostic/capture tooling.
pub fn run_attention_public(weights: &ModelWeights, h: &Array2<f32>, layer: usize) -> Option<Array2<f32>> {
    run_attention(weights, h, layer)
}

/// Run attention for a single layer. Returns the post-attention residual.
pub(super) fn run_attention(weights: &ModelWeights, h: &Array2<f32>, layer: usize) -> Option<Array2<f32>> {
    let (h_post_attn, _) = run_attention_inner(weights, h, layer, false, None)?;
    Some(h_post_attn)
}

/// Run attention with optional per-head weight capture and shared K/V.
pub(super) fn run_attention_inner(
    weights: &ModelWeights,
    h: &Array2<f32>,
    layer: usize,
    capture_attention: bool,
    shared_kv: Option<&SharedKV>,
) -> Option<(Array2<f32>, Option<AttentionWeights>)> {
    let (h_post_attn, _attn_projected, attn_weights) =
        crate::attention::run_attention_block_shared(weights, h, layer, capture_attention, shared_kv)?;
    Some((h_post_attn, attn_weights))
}

/// Run attention returning post-processed K/V for caching (KV sharing source layers).
pub(super) fn run_attention_with_kv_cache(
    weights: &ModelWeights,
    h: &Array2<f32>,
    layer: usize,
) -> Option<(Array2<f32>, SharedKV)> {
    let (h_post_attn, _, _, k_rope, v_final) =
        crate::attention::run_attention_block_with_kv_out(weights, h, layer, false, None)?;
    Some((h_post_attn, (k_rope, v_final)))
}

/// Run FFN for a single layer using the given backend. Returns the post-FFN residual.
pub fn run_ffn(
    weights: &ModelWeights,
    h_post_attn: &Array2<f32>,
    layer: usize,
    ffn: &dyn FfnBackend,
    capture_activation: bool,
) -> (Array2<f32>, Option<Array2<f32>>) {
    let norm_offset = weights.arch.norm_weight_offset();
    let arch = &*weights.arch;

    // Layer-0 stage dumps (LARQL_CPU_STAGE_DUMP=<dir>) — matches the
    // Metal `LARQL_METAL_DUMP_LAYERS` convention. Lets us diff per-stage
    // intermediates between CPU and Metal for the first layer.
    let stage_dump_dir = if layer == 0 { std::env::var("LARQL_CPU_STAGE_DUMP").ok() } else { None };
    let dump_f32 = |name: &str, arr: &Array2<f32>| {
        if let Some(ref dir) = stage_dump_dir {
            let slice = arr.as_slice().unwrap_or(&[]);
            let bytes: Vec<u8> = slice.iter().flat_map(|v| v.to_le_bytes()).collect();
            let _ = std::fs::write(format!("{dir}/cpu_L0_{name}.f32"), &bytes);
        }
    };
    dump_f32("h_post_attn", h_post_attn);

    let pre_ffn_key = if arch.has_post_norms() {
        arch.pre_feedforward_layernorm_key(layer)
    } else {
        Some(arch.post_attention_layernorm_key(layer))
    };
    let h_ffn = match pre_ffn_key {
        Some(key) => apply_norm(weights, h_post_attn, &key, norm_offset),
        None => rms_norm(h_post_attn, None, norm_offset),
    };
    dump_f32("ffn_norm_out", &h_ffn);

    let (ffn_out, activation) = if capture_activation {
        let (out, act) = ffn.forward_with_activation(layer, &h_ffn);
        (out, Some(act))
    } else {
        (ffn.forward(layer, &h_ffn), None)
    };
    dump_f32("ffn_out_raw", &ffn_out);

    let res_mult = arch.residual_multiplier();
    let h_out = if arch.has_post_norms() {
        let normed = match arch.post_feedforward_layernorm_key(layer) {
            Some(key) => apply_norm(weights, &ffn_out, &key, norm_offset),
            None => rms_norm(&ffn_out, None, norm_offset),
        };
        if res_mult != 1.0 {
            h_post_attn + &(&normed * res_mult)
        } else {
            h_post_attn + &normed
        }
    } else if res_mult != 1.0 {
        h_post_attn + &(&ffn_out * res_mult)
    } else {
        h_post_attn + &ffn_out
    };

    (h_out, activation)
}

/// Apply per-layer scalar multiplier if present (e.g., Gemma 4 layer_scalar).
pub(super) fn apply_layer_scalar(weights: &ModelWeights, h: &mut Array2<f32>, layer: usize) {
    if let Some(key) = weights.arch.layer_scalar_key(layer) {
        if let Some(scalars) = weights.vectors.get(&key) {
            if let Some(&scalar) = scalars.first() {
                if scalar != 1.0 {
                    *h *= scalar;
                }
            }
        }
    }
}

/// Run a single transformer layer with the given FFN backend.
///
/// Handles: attention → FFN → per-layer embedding → layer_scalar.
/// All four steps are needed for Gemma 4 correctness. Exposed `pub` so
/// alternate forward drivers (notably `vindex::predict_q4k`) get the same
/// sequence as `predict_with_temperature` without duplicating logic.
#[allow(clippy::type_complexity)]
pub fn run_layer_with_ffn(
    weights: &ModelWeights,
    h: &Array2<f32>,
    layer: usize,
    ffn: &dyn FfnBackend,
    capture_activation: bool,
    ple_input: Option<&Array2<f32>>,
    shared_kv: Option<&SharedKV>,
) -> Option<(Array2<f32>, Option<Array2<f32>>, Option<SharedKV>)> {
    let (h_post_attn, kv_out) = if shared_kv.is_some() {
        (run_attention_inner(weights, h, layer, false, shared_kv)?.0, None)
    } else {
        let (h_pa, kv) = run_attention_with_kv_cache(weights, h, layer)?;
        (h_pa, Some(kv))
    };
    let (h_post_ffn, activation) = run_ffn(weights, &h_post_attn, layer, ffn, capture_activation);
    let mut h_out = apply_per_layer_embedding(weights, &h_post_ffn, layer, ple_input);
    apply_layer_scalar(weights, &mut h_out, layer);
    Some((h_out, activation, kv_out))
}

/// Run a single transformer layer, optionally capturing attention weights.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
pub(super) fn run_layer_with_capture(
    weights: &ModelWeights,
    h: &Array2<f32>,
    layer: usize,
    ffn: &dyn FfnBackend,
    capture_activation: bool,
    capture_attention: bool,
    ple_input: Option<&Array2<f32>>,
    shared_kv: Option<&SharedKV>,
) -> Option<(Array2<f32>, Option<Array2<f32>>, Option<AttentionWeights>, Option<SharedKV>)> {
    let (h_post_attn, attn_weights) = run_attention_inner(weights, h, layer, capture_attention, shared_kv)?;
    let kv_out = None;
    let (h_post_ffn, activation) = run_ffn(weights, &h_post_attn, layer, ffn, capture_activation);
    let mut h_out = apply_per_layer_embedding(weights, &h_post_ffn, layer, ple_input);
    apply_layer_scalar(weights, &mut h_out, layer);
    Some((h_out, activation, attn_weights, kv_out))
}
