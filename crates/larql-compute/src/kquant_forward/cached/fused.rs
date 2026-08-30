//! The fused prefill / decode-step variants and their state-dump forms.
//!
//! Split out of `cached.rs`; [`super`] composes the pieces.

#[allow(unused_imports)]
use super::*;
use larql_models::ModelWeights;
use ndarray::Array2;

/// True when the whole model can run on the direct-matvec decode path.
/// Metal-fused multi-token prefill: run the prompt through all layers
/// via the backend's fused `prefill_kquant` kernel, populating the
/// backend's internal K/V cache for subsequent decode steps.
///
/// Returns `None` for CPU backends (no fused `prefill_kquant` impl) and
/// for vindex shapes the fused pipeline can't handle. Refactored to
/// take `&dyn KvIndex` (ADR-0022 Step 7).
pub fn fused_prefill(
    weights: &ModelWeights,
    index: &dyn crate::KvIndex,
    token_ids: &[u32],
    backend: &dyn crate::ComputeBackend,
) -> Option<Array2<f32>> {
    if !backend.supports_quant(crate::QuantFormat::Q4_K) {
        return None;
    }
    let (q4_ffn_mmap, ffn_is_q4k) = if let Some(m) = index.interleaved_kquant_mmap_ref() {
        (m, true)
    } else {
        (index.interleaved_q4_mmap_ref()?, false)
    };
    index.attn_kquant_layer_data(0)?;

    let arch = &*weights.arch;
    let hidden = weights.hidden_size;
    let num_layers = weights.num_layers;
    let intermediate = index.num_features(0);
    if intermediate == 0 {
        return None;
    }

    let ffn_format = if ffn_is_q4k {
        crate::QuantFormat::Q4_K
    } else {
        crate::QuantFormat::Q4_0
    };
    let q4_ffn_per_matrix = ffn_format.packed_matrix_bytes(intermediate, hidden)?;

    let layers = crate::pipeline_layer::build_pipeline_layers(
        weights,
        index,
        0..num_layers,
        q4_ffn_mmap,
        q4_ffn_per_matrix,
        ffn_format,
    );

    let h_embed = crate::forward::embed_tokens_pub(weights, token_ids);
    let x: Vec<f32> = h_embed.as_slice().unwrap_or(&[]).to_vec();

    let seq_len = token_ids.len();
    let softcap = arch.attn_logit_softcapping().unwrap_or(0.0);
    let qk_norm = arch.attn_q_norm_key(0).is_some();

    backend.reset_kv_cache();
    {
        let kv_shapes: Vec<(usize, usize)> = (0..num_layers)
            .map(|l| (arch.num_kv_heads_for_layer(l), arch.head_dim_for_layer(l)))
            .collect();
        backend.preallocate_kv_cache_per_layer(
            &kv_shapes,
            crate::pipeline_layer::DEFAULT_GPU_KV_CACHE_MAX_SEQ,
        );
    }

    let h_vec =
        backend.prefill_kquant(&layers, &x, hidden, intermediate, seq_len, qk_norm, softcap)?;

    let h_2d = Array2::from_shape_vec((seq_len, hidden), h_vec).ok()?;
    let last = h_2d.shape()[0] - 1;
    Some(h_2d.slice(ndarray::s![last..=last, ..]).to_owned())
}

/// Metal-fused single-token decode: run one token through all layers
/// via the backend's fused `decode_token` kernel, using the K/V cache
/// populated by a prior [`fused_prefill`] call on the same backend.
pub fn fused_decode_step(
    weights: &ModelWeights,
    index: &dyn crate::KvIndex,
    token_id: u32,
    backend: &dyn crate::ComputeBackend,
) -> Option<Array2<f32>> {
    fused_decode_step_inner(
        weights,
        index,
        token_id,
        backend,
        None,
        crate::StateDumpMask::Full,
    )
}

/// Variant of [`fused_decode_step`] that also captures per-layer state
/// via the backend's `decode_token_with_state_dump`.
pub fn fused_decode_step_with_state(
    weights: &ModelWeights,
    index: &dyn crate::KvIndex,
    token_id: u32,
    backend: &dyn crate::ComputeBackend,
    state: &mut crate::DecodeStateDump,
) -> Option<Array2<f32>> {
    fused_decode_step_inner(
        weights,
        index,
        token_id,
        backend,
        Some(state),
        crate::StateDumpMask::Full,
    )
}

/// Mask-aware variant of [`fused_decode_step_with_state`]. Lets engines
/// that treat K/V as derivative state request
/// [`crate::StateDumpMask::HOnly`] to skip the K/V staging + readback.
pub fn fused_decode_step_with_state_masked(
    weights: &ModelWeights,
    index: &dyn crate::KvIndex,
    token_id: u32,
    backend: &dyn crate::ComputeBackend,
    state: &mut crate::DecodeStateDump,
    mask: crate::StateDumpMask,
) -> Option<Array2<f32>> {
    fused_decode_step_inner(weights, index, token_id, backend, Some(state), mask)
}

pub(crate) fn fused_decode_step_inner(
    weights: &ModelWeights,
    index: &dyn crate::KvIndex,
    token_id: u32,
    backend: &dyn crate::ComputeBackend,
    state: Option<&mut crate::DecodeStateDump>,
    mask: crate::StateDumpMask,
) -> Option<Array2<f32>> {
    let (q4_ffn_mmap, ffn_is_q4k) = if let Some(m) = index.interleaved_kquant_mmap_ref() {
        (m, true)
    } else {
        (index.interleaved_q4_mmap_ref()?, false)
    };

    let hidden = weights.hidden_size;
    let num_layers = weights.num_layers;
    let intermediate = index.num_features(0);

    let ffn_format = if ffn_is_q4k {
        crate::QuantFormat::Q4_K
    } else {
        crate::QuantFormat::Q4_0
    };
    let q4_ffn_per_matrix = ffn_format.packed_matrix_bytes(intermediate, hidden)?;

    let layers = crate::pipeline_layer::build_pipeline_layers(
        weights,
        index,
        0..num_layers,
        q4_ffn_mmap,
        q4_ffn_per_matrix,
        ffn_format,
    );

    let h_tok = crate::forward::embed_tokens_pub(weights, &[token_id]);
    let x_dec: Vec<f32> = h_tok.row(0).to_vec();

    let h_vec = backend.decode_token_with_state_dump_masked(
        &layers,
        &x_dec,
        hidden,
        intermediate,
        state,
        mask,
    )?;
    Array2::from_shape_vec((1, hidden), h_vec).ok()
}

/// Same gating as [`supports_cached_decode`] plus a per-layer format
/// check. Used by the bench labeler and as the cpu.rs routing key.
pub fn supports_direct_matvec_decode(weights: &ModelWeights, index: &dyn crate::KvIndex) -> bool {
    if !supports_cached_decode(weights) {
        return false;
    }
    for layer in 0..weights.num_layers {
        if !layer_supports_direct_matvec(index, layer) {
            return false;
        }
    }
    true
}

pub(crate) fn vec_to_2d_row(v: Vec<f32>) -> Array2<f32> {
    let n = v.len();
    Array2::from_shape_vec((1, n), v).expect("matvec output shape")
}
