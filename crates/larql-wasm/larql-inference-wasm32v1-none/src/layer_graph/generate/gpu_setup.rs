//! Shared setup for GPU/vindex-backed generation paths.

use super::types::GenerateError;
use crate::layer_graph::pipeline_layer::{kv_cache_shapes_for_arch, DEFAULT_GPU_KV_CACHE_MAX_SEQ};
use crate::model::ModelWeights;
use larql_compute::backend::Capability;
use larql_compute::{prelude::*, FullPipelineLayer};
use std::ops::Range;

/// True when the model has at least two layers and any per-layer
/// attention parameter differs from layer 0. Catches Gemma 4 31B's
/// sliding/global geometry alternation, the canonical heterogeneous
/// case (50 sliding-attention layers at head_dim=256/16-kv plus 10
/// global-attention layers at head_dim=512/4-kv).
pub(crate) fn has_heterogeneous_attention(weights: &ModelWeights) -> bool {
    let arch = &*weights.arch;
    let n = weights.num_layers;
    if n < 2 {
        return false;
    }
    let l0_head = arch.head_dim_for_layer(0);
    let l0_kv = arch.num_kv_heads_for_layer(0);
    let l0_q = arch.num_q_heads_for_layer(0);
    let l0_rope = arch.rope_base_for_layer(0);
    let l0_rotary = arch.rotary_fraction_for_layer(0);
    let l0_sliding = arch.is_sliding_window_layer(0);
    (1..n).any(|l| {
        arch.head_dim_for_layer(l) != l0_head
            || arch.num_kv_heads_for_layer(l) != l0_kv
            || arch.num_q_heads_for_layer(l) != l0_q
            || arch.rope_base_for_layer(l) != l0_rope
            || arch.rotary_fraction_for_layer(l) != l0_rotary
            || arch.is_sliding_window_layer(l) != l0_sliding
    })
}

/// Reject heterogeneous models on backends that haven't opted in via
/// `Capability::HeterogeneousAttention`. The error fires *before* the
/// first decode dispatch so the caller sees a precise unsupported-backend
/// message rather than corrupted KV state at the first non-uniform layer.
pub(crate) fn ensure_attention_supported(
    weights: &ModelWeights,
    backend: &dyn ComputeBackend,
) -> Result<(), GenerateError> {
    if has_heterogeneous_attention(weights) && !backend.supports(Capability::HeterogeneousAttention)
    {
        return Err(GenerateError::unsupported_backend(format!(
            "{} model has heterogeneous attention geometry but the active backend ({}) does not advertise Capability::HeterogeneousAttention",
            weights.arch.family(),
            backend.name(),
        )));
    }
    Ok(())
}

pub(super) struct GpuDecodeSetup<'a> {
    pub layers: Vec<FullPipelineLayer<'a>>,
    pub hidden: usize,
    pub intermediate: usize,
}

pub(super) fn build_gpu_decode_setup<'a>(
    weights: &'a ModelWeights,
    index: &'a larql_vindex::VectorIndex,
    backend: &dyn ComputeBackend,
    layer_range: Range<usize>,
    constrained: bool,
) -> Result<GpuDecodeSetup<'a>, GenerateError> {
    let hidden = weights.hidden_size;
    let gate_index: &dyn larql_vindex::GateIndex = index;

    let (q4_ffn, ffn_is_q4k) = if let Some(mmap) = gate_index.interleaved_q4k_mmap_ref() {
        (Some(mmap), true)
    } else {
        (gate_index.interleaved_q4_mmap_ref(), false)
    };

    if !backend.has_q4() || q4_ffn.is_none() {
        return Err(GenerateError::unsupported_backend(format!(
            "{}GPU generation requires backend Q4 support and interleaved Q4 FFN weights",
            if constrained { "constrained " } else { "" }
        )));
    }
    ensure_attention_supported(weights, backend)?;

    let first_layer = layer_range.start;
    let intermediate = gate_index.num_features(first_layer);
    let has_q4k = index.attn_q4k_layer_data(first_layer).is_some();
    let has_q8 = index.attn_q8_layer_data(first_layer).is_some();
    if intermediate == 0 || (!has_q4k && !has_q8) {
        return Err(GenerateError::missing_weights(format!(
            "{}GPU generation requires non-empty FFN features and Q4/Q8 attention weights",
            if constrained { "constrained " } else { "" }
        )));
    }

    let ffn_format = if ffn_is_q4k {
        larql_compute::QuantFormat::Q4_K
    } else {
        larql_compute::QuantFormat::Q4_0
    };
    let q4_ffn_per_matrix = ffn_format
        .packed_matrix_bytes(intermediate, hidden)
        .ok_or_else(|| {
            GenerateError::missing_weights("Q4 interleaved FFN format has invalid packed geometry")
        })?;
    let layers = crate::layer_graph::pipeline_layer::build_pipeline_layers(
        weights,
        index,
        0..weights.num_layers,
        q4_ffn.expect("checked above"),
        q4_ffn_per_matrix,
        ffn_format,
    );

    Ok(GpuDecodeSetup {
        layers,
        hidden,
        intermediate,
    })
}

pub(super) fn ensure_prompt_fits(seq_len: usize) -> Result<(), GenerateError> {
    if seq_len > DEFAULT_GPU_KV_CACHE_MAX_SEQ {
        return Err(GenerateError::prompt_too_long(
            seq_len,
            DEFAULT_GPU_KV_CACHE_MAX_SEQ,
        ));
    }
    Ok(())
}

pub(super) fn reset_and_preallocate_kv_cache(weights: &ModelWeights, backend: &dyn ComputeBackend) {
    backend.reset_kv_cache();
    let kv_shapes = kv_cache_shapes_for_arch(weights);
    backend.preallocate_kv_cache_per_layer(&kv_shapes, DEFAULT_GPU_KV_CACHE_MAX_SEQ);
}

#[allow(clippy::too_many_arguments)]
pub(super) fn prefill_q4_prompt(
    backend: &dyn ComputeBackend,
    layers: &[FullPipelineLayer<'_>],
    x: &[f32],
    hidden: usize,
    intermediate: usize,
    seq_len: usize,
    qk_norm: bool,
    softcap: f32,
    failure_reason: &'static str,
) -> Result<Vec<f32>, GenerateError> {
    backend
        .prefill_q4(layers, x, hidden, intermediate, seq_len, qk_norm, softcap)
        .ok_or_else(|| GenerateError::prefill_failed(failure_reason))
}
