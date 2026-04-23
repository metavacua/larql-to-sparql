//! Shared FullPipelineLayer construction from ModelWeights + VectorIndex.
//!
//! Single source of truth for extracting per-layer architecture parameters
//! from larql-models and wiring them into larql-compute's FullPipelineLayer.
//! Both GPU and CPU paths use this — no duplicated param extraction.

use larql_compute::{QuantWeight, QuantFormat, FullPipelineLayer, MoeLayerWeights};
use crate::model::ModelWeights;

/// Extract per-layer architecture parameters into a FullPipelineLayer.
///
/// This is the single construction site for all per-layer params:
/// head_dim, num_q/kv_heads, rope_base, attn_scale, rotary_dim,
/// sliding_window, norm offsets, activation, FFN type, V-norm, layer scalar.
///
/// The attention weights (wq/wk/wv/wo) and FFN weights (gate/up/down)
/// must be provided separately since they come from different sources
/// (Q4_K from vindex, Q8 from vindex, f32 from model weights).
#[allow(clippy::too_many_arguments)]
pub fn build_arch_params<'a>(
    weights: &'a ModelWeights,
    layer: usize,
    wq: QuantWeight<'a>,
    wk: QuantWeight<'a>,
    wv: QuantWeight<'a>,
    wo: QuantWeight<'a>,
    gate: QuantWeight<'a>,
    up: QuantWeight<'a>,
    down: QuantWeight<'a>,
) -> FullPipelineLayer<'a> {
    let arch = &*weights.arch;
    let layer_hd = arch.head_dim_for_layer(layer);
    let layer_nq = arch.num_q_heads_for_layer(layer);
    let layer_nkv = arch.num_kv_heads_for_layer(layer);
    let rotary_frac = arch.rotary_fraction_for_layer(layer);
    let rotary_dim = if rotary_frac >= 1.0 { 0 } else { (layer_hd as f64 * rotary_frac) as usize };
    let sw = if arch.is_sliding_window_layer(layer) {
        arch.sliding_window_size().unwrap_or(0)
    } else {
        0
    };
    let layer_scalar = arch.layer_scalar_key(layer)
        .and_then(|k| weights.vectors.get(&k))
        .and_then(|v| v.first().copied())
        .unwrap_or(0.0);

    FullPipelineLayer {
        wq, wk, wv, wo,
        gate, up, down,
        input_norm: weights.vectors.get(&arch.input_layernorm_key(layer))
            .map(|v| v.as_slice()).unwrap_or(&[]),
        post_attn_norm: weights.vectors.get(&arch.post_attention_layernorm_key(layer))
            .map(|v| v.as_slice()).unwrap_or(&[]),
        pre_ffn_norm: arch.pre_feedforward_layernorm_key(layer)
            .and_then(|k| weights.vectors.get(&k)).map(|v| v.as_slice()),
        post_ffn_norm: arch.post_feedforward_layernorm_key(layer)
            .and_then(|k| weights.vectors.get(&k)).map(|v| v.as_slice()),
        norm_offset: arch.norm_weight_offset(),
        has_post_norms: arch.has_post_norms(),
        activation: match arch.activation() {
            larql_models::Activation::GeluTanh => larql_compute::Activation::GeluTanh,
            _ => larql_compute::Activation::Silu,
        },
        qk_norm_offset: arch.qk_norm_weight_offset(),
        eps: arch.norm_eps(),
        norm_type: match arch.norm_type() {
            larql_models::NormType::LayerNorm => larql_compute::NormType::LayerNorm,
            _ => larql_compute::NormType::RmsNorm,
        },
        ffn_type: match arch.ffn_type() {
            larql_models::FfnType::Standard => larql_compute::FfnType::Standard,
            _ => larql_compute::FfnType::Gated,
        },
        attn_scale: arch.attention_scale_for_layer(layer) as f32,
        head_dim: layer_hd,
        num_q_heads: layer_nq,
        num_kv_heads: layer_nkv,
        rope_base: arch.rope_base_for_layer(layer) as f32,
        rotary_dim,
        sliding_window: sw,
        has_v_norm: arch.has_v_norm(),
        layer_scalar,
        input_norm_bias: None,
        post_attn_norm_bias: None,
        q_norm_weight: arch.attn_q_norm_key(layer)
            .and_then(|k| weights.vectors.get(&k)).map(|v| v.as_slice()),
        k_norm_weight: arch.attn_k_norm_key(layer)
            .and_then(|k| weights.vectors.get(&k)).map(|v| v.as_slice()),
        ffn_up_bias: arch.ffn_up_bias_key(layer)
            .and_then(|k| weights.vectors.get(&k)).map(|v| v.as_slice()),
        ffn_down_bias: arch.ffn_down_bias_key(layer)
            .and_then(|k| weights.vectors.get(&k)).map(|v| v.as_slice()),

        moe: build_moe_weights(weights, arch, layer),
        moe_combined_output_norm: arch.moe_has_combined_output_norm(),
        moe_outer_post_norm: arch.moe_post_outer_norm_key(layer)
            .and_then(|k| weights.vectors.get(&k)).map(|v| v.as_slice()),
    }
}

fn build_moe_weights<'a>(
    weights: &'a ModelWeights,
    arch: &dyn larql_models::ModelArchitecture,
    layer: usize,
) -> Option<MoeLayerWeights<'a>> {
    if !arch.is_hybrid_moe() { return None; }

    let gate_up_key = arch.packed_experts_gate_up_key(layer)?;
    let down_key = arch.packed_experts_down_key(layer)?;
    let router_key = arch.moe_router_key(layer)?;

    let experts_gate_up = weights.get_packed_bytes(&gate_up_key)?;
    let experts_down = weights.get_packed_bytes(&down_key)?;
    let router_proj = weights.vectors.get(&router_key)?.as_slice();

    let router_scale = arch.moe_router_scale_key(layer)
        .and_then(|k| weights.vectors.get(&k))
        .map(|v| v.as_slice())
        .unwrap_or(&[]);
    let router_per_expert_scale = arch.moe_router_per_expert_scale_key(layer)
        .and_then(|k| weights.vectors.get(&k))
        .map(|v| v.as_slice())
        .unwrap_or(&[]);
    let pre_experts_norm = arch.moe_pre_experts_norm_key(layer)
        .and_then(|k| weights.vectors.get(&k))
        .map(|v| v.as_slice())
        .unwrap_or(&[]);
    let post_ffn1_norm = arch.moe_post_ffn1_norm_key(layer)
        .and_then(|k| weights.vectors.get(&k))
        .map(|v| v.as_slice())
        .unwrap_or(&[]);
    let post_experts_norm = arch.moe_post_experts_norm_key(layer)
        .and_then(|k| weights.vectors.get(&k))
        .map(|v| v.as_slice())
        .unwrap_or(&[]);
    let router_norm = arch.moe_router_norm_key(layer)
        .and_then(|k| weights.vectors.get(&k))
        .map(|v| v.as_slice())
        .unwrap_or(&[]);
    let router_norm_parameter_free = arch.moe_router_norm_parameter_free();
    let router_input_scalar = arch.moe_router_input_scalar().unwrap_or(1.0);

    let activation = match arch.activation() {
        larql_models::Activation::GeluTanh => larql_compute::Activation::GeluTanh,
        _ => larql_compute::Activation::Silu,
    };

    Some(MoeLayerWeights {
        experts_gate_up,
        experts_down,
        router_proj,
        router_scale,
        router_per_expert_scale,
        router_norm,
        router_norm_parameter_free,
        router_input_scalar,
        pre_experts_norm,
        post_ffn1_norm,
        post_experts_norm,
        num_experts: arch.num_experts(),
        top_k: arch.num_experts_per_token(),
        intermediate_size: arch.moe_intermediate_size(),
        activation,
    })
}

/// Helper: resolve attention weights from vindex (Q4_K preferred, Q8 fallback).
pub fn resolve_attn_weights<'a>(
    index: &'a larql_vindex::VectorIndex,
    layer: usize,
) -> Option<(QuantWeight<'a>, QuantWeight<'a>, QuantWeight<'a>, QuantWeight<'a>)> {
    fn to_format(s: &str) -> QuantFormat {
        match s { "Q6_K" => QuantFormat::Q6_K, _ => QuantFormat::Q4_K }
    }

    if let Some([q, k, v, o]) = index.attn_q4k_layer_data(layer) {
        Some((
            QuantWeight { data: q.0, scales: None, format: to_format(q.1) },
            QuantWeight { data: k.0, scales: None, format: to_format(k.1) },
            QuantWeight { data: v.0, scales: None, format: to_format(v.1) },
            QuantWeight { data: o.0, scales: None, format: to_format(o.1) },
        ))
    } else if let Some([q, k, v, o]) = index.attn_q8_layer_data(layer) {
        Some((
            QuantWeight { data: q.0, scales: Some(q.1), format: QuantFormat::Q8_0 },
            QuantWeight { data: k.0, scales: Some(k.1), format: QuantFormat::Q8_0 },
            QuantWeight { data: v.0, scales: Some(v.1), format: QuantFormat::Q8_0 },
            QuantWeight { data: o.0, scales: Some(o.1), format: QuantFormat::Q8_0 },
        ))
    } else {
        None
    }
}

/// Helper: resolve FFN weights from vindex interleaved mmap.
///
/// Prefers the per-matrix manifest when available (emitted by the streaming
/// `--quant q4k` writer: gate/up Q4_K, down Q6_K — non-uniform stride). Falls
/// back to the legacy uniform-stride layout produced by `build_q4k_weights.rs`
/// when the manifest is absent so older vindexes still work.
pub fn resolve_ffn_weights<'a>(
    index: &'a larql_vindex::VectorIndex,
    layer: usize,
    q4_ffn_mmap: &'a [u8],
    q4_ffn_per_matrix: usize,
    ffn_format: QuantFormat,
) -> (QuantWeight<'a>, QuantWeight<'a>, QuantWeight<'a>) {
    fn str_to_format(s: &str, fallback: QuantFormat) -> QuantFormat {
        match s {
            "Q6_K" => QuantFormat::Q6_K,
            "Q4_K" => QuantFormat::Q4_K,
            "Q4_0" => QuantFormat::Q4_0,
            _ => fallback,
        }
    }

    if let Some([gate, up, down]) = index.interleaved_q4k_layer_data(layer) {
        return (
            QuantWeight { data: gate.0, scales: None, format: str_to_format(gate.1, ffn_format) },
            QuantWeight { data: up.0,   scales: None, format: str_to_format(up.1,   ffn_format) },
            QuantWeight { data: down.0, scales: None, format: str_to_format(down.1, ffn_format) },
        );
    }

    let q4_ffn_per_layer = q4_ffn_per_matrix * 3;
    let fs = layer * q4_ffn_per_layer;
    (
        QuantWeight { data: &q4_ffn_mmap[fs..fs + q4_ffn_per_matrix], scales: None, format: ffn_format },
        QuantWeight { data: &q4_ffn_mmap[fs + q4_ffn_per_matrix..fs + 2 * q4_ffn_per_matrix], scales: None, format: ffn_format },
        QuantWeight { data: &q4_ffn_mmap[fs + 2 * q4_ffn_per_matrix..fs + 3 * q4_ffn_per_matrix], scales: None, format: ffn_format },
    )
}

/// Build a complete Vec<FullPipelineLayer> for a range of layers.
/// Single source of truth — used by both GPU decode and GPU prefill paths.
#[allow(clippy::too_many_arguments)]
pub fn build_pipeline_layers<'a>(
    weights: &'a ModelWeights,
    index: &'a larql_vindex::VectorIndex,
    layer_range: std::ops::Range<usize>,
    q4_ffn_mmap: &'a [u8],
    q4_ffn_per_matrix: usize,
    ffn_format: QuantFormat,
) -> Vec<FullPipelineLayer<'a>> {
    layer_range.map(|layer| {
        let (wq, wk, wv, wo) = resolve_attn_weights(index, layer)
            .expect("No attention weights available for layer");
        let (gate, up, down) = resolve_ffn_weights(index, layer, q4_ffn_mmap, q4_ffn_per_matrix, ffn_format);
        build_arch_params(weights, layer, wq, wk, wv, wo, gate, up, down)
    }).collect()
}
