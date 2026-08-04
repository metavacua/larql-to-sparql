//! Shared FullPipelineLayer construction from ModelWeights + VectorIndex.
//!
//! Single source of truth for extracting per-layer architecture parameters
//! from larql-models and wiring them into larql-compute's FullPipelineLayer.
//! Both GPU and CPU paths use this — no duplicated param extraction.
//!
//! Per-layer override resolution (env vars vs arch defaults) lives in
//! [`crate::forward_overrides`]; this module consumes those helpers.

use crate::forward_overrides::effective_rope_base_for_layer;

/// Wire value the kernel's attention spec uses for "no sliding window —
/// attend the whole context". Named because it sits in the same field as
/// a real window width, where a bare `0` reads as an empty window.
const NO_ATTENTION_WINDOW: usize = 0;

use crate::{
    FullPipelineLayer, MoeLayerWeights, MoeRoutingPolicy, MoeWeightLayout, QuantFormat, QuantWeight,
};
use larql_models::ModelWeights;

pub const DEFAULT_GPU_KV_CACHE_MAX_SEQ: usize = 4096;

pub fn kv_cache_shapes_for_arch(weights: &ModelWeights) -> Vec<(usize, usize)> {
    let arch = &*weights.arch;
    (0..weights.num_layers)
        .map(|layer| {
            (
                arch.num_kv_heads_for_layer(layer),
                arch.head_dim_for_layer(layer),
            )
        })
        .collect()
}

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
    let rotary_dim = if rotary_frac >= 1.0 {
        0
    } else {
        (layer_hd as f64 * rotary_frac) as usize
    };
    // Resolved through the shared rule so the Metal spec and the CPU
    // attention path cannot answer this differently. `0` is the wire
    // sentinel for "attend everything" in the kernel's spec struct, which
    // is what `None` means here.
    let sw = crate::forward_overrides::effective_attention_window_for_layer(arch, layer)
        .unwrap_or(NO_ATTENTION_WINDOW);
    let layer_scalar = arch
        .layer_scalar_key(layer)
        .and_then(|k| weights.vectors.get(&k))
        .and_then(|v| v.first().copied())
        .unwrap_or(0.0);

    let attn_sinks = arch
        .attn_sinks_key(layer)
        .and_then(|k| weights.vectors.get(&k))
        .map(|v| v.as_slice());

    FullPipelineLayer {
        attn_sinks,
        wq,
        wk,
        wv,
        wo,
        gate,
        up,
        down,
        input_norm: weights
            .vectors
            .get(&arch.input_layernorm_key(layer))
            .map(|v| v.as_slice())
            .unwrap_or(&[]),
        post_attn_norm: weights
            .vectors
            .get(&arch.post_attention_layernorm_key(layer))
            .map(|v| v.as_slice())
            .unwrap_or(&[]),
        pre_ffn_norm: arch
            .pre_feedforward_layernorm_key(layer)
            .and_then(|k| weights.vectors.get(&k))
            .map(|v| v.as_slice()),
        post_ffn_norm: arch
            .post_feedforward_layernorm_key(layer)
            .and_then(|k| weights.vectors.get(&k))
            .map(|v| v.as_slice()),
        norm_offset: arch.norm_weight_offset(),
        has_post_norms: arch.has_post_norms(),
        activation: match arch.activation() {
            larql_models::Activation::GeluTanh => crate::Activation::GeluTanh,
            _ => crate::Activation::Silu,
        },
        qk_norm_offset: arch.qk_norm_weight_offset(),
        eps: arch.norm_eps(),
        norm_type: match arch.norm_type() {
            larql_models::NormType::LayerNorm => crate::NormType::LayerNorm,
            _ => crate::NormType::RmsNorm,
        },
        ffn_type: match arch.ffn_type() {
            larql_models::FfnType::Standard => crate::FfnType::Standard,
            _ => crate::FfnType::Gated,
        },
        // Granite-family `attention_multiplier` (1/64 on 3B, 1/128 on
        // 8B/30B) *replaces* `1/sqrt(head_dim)` — it is the trained-time
        // attention score scale, not a factor multiplied on top of
        // sqrt-scaling. The F32 attention path at
        // `attention/gpu.rs::scale` follows the same convention; the
        // Metal Q4K decode kernels (`decode/encode_attn.rs`,
        // `ops/full_layer.rs`) read this `attn_scale` directly with no
        // further adjustment, so failing to fold the multiplier in here
        // leaves Granite's 0.015625 / 0.0078125 trained scale unused and
        // the model degenerates to repeating high-frequency tokens
        // ("ikea ikea ikea…") because every attention distribution
        // peaks too sharply.
        attn_scale: if arch.attention_multiplier() != 1.0 {
            arch.attention_multiplier()
        } else {
            arch.attention_scale_for_layer(layer) as f32
        },
        head_dim: layer_hd,
        num_q_heads: layer_nq,
        num_kv_heads: layer_nkv,
        rope_base: effective_rope_base_for_layer(arch, layer) as f32,
        rotary_dim,
        sliding_window: sw,
        has_v_norm: arch.has_v_norm(),
        layer_scalar,
        // LayerNorm `β`. `None` for RMSNorm architectures, which have no
        // bias term. These were hardcoded `None` until 2026-07-31, so every
        // vindex-backed and Metal path silently dropped the shift for GPT-2
        // and StarCoder2 even though the `layer_norm` shader implements it.
        input_norm_bias: arch
            .input_layernorm_bias_key(layer)
            .and_then(|k| weights.vectors.get(&k))
            .map(|v| v.as_slice()),
        post_attn_norm_bias: arch
            .post_attention_layernorm_bias_key(layer)
            .and_then(|k| weights.vectors.get(&k))
            .map(|v| v.as_slice()),
        q_norm_weight: arch
            .attn_q_norm_key(layer)
            .and_then(|k| weights.vectors.get(&k))
            .map(|v| v.as_slice()),
        k_norm_weight: arch
            .attn_k_norm_key(layer)
            .and_then(|k| weights.vectors.get(&k))
            .map(|v| v.as_slice()),
        ffn_up_bias: arch
            .ffn_up_bias_key(layer)
            .and_then(|k| weights.vectors.get(&k))
            .map(|v| v.as_slice()),
        ffn_down_bias: arch
            .ffn_down_bias_key(layer)
            .and_then(|k| weights.vectors.get(&k))
            .map(|v| v.as_slice()),

        moe: build_moe_weights(weights, arch, layer),
        ffn_is_remote: false,
        moe_combined_output_norm: arch.moe_has_combined_output_norm(),
        moe_outer_post_norm: arch
            .moe_post_outer_norm_key(layer)
            .and_then(|k| weights.vectors.get(&k))
            .map(|v| v.as_slice()),
        ple_input_gate: arch
            .per_layer_input_gate_key(layer)
            .and_then(|k| weights.tensors.get(&k))
            .and_then(|t| t.as_slice()),
        ple_projection: arch
            .per_layer_projection_key(layer)
            .and_then(|k| weights.tensors.get(&k))
            .and_then(|t| t.as_slice()),
        ple_post_norm: arch
            .post_per_layer_input_norm_key(layer)
            .and_then(|k| weights.vectors.get(&k))
            .map(|v| v.as_slice()),
        kv_shared_source: arch.kv_shared_source_layer(layer),
        // Granite-style residual scaling: HF `modeling_granite.py` does
        // `hidden_states = residual + self.residual_multiplier * hidden_states`
        // after both attention and FFN. Trait getter returns 1.0 for
        // every non-Granite arch, so this default is bit-identical for
        // them and the Metal `residual_add` shader's `b_scale` binding
        // is a no-op (multiply by 1.0).
        residual_multiplier: arch.residual_multiplier(),
    }
}

pub fn build_moe_weights<'a>(
    weights: &'a ModelWeights,
    arch: &dyn larql_models::ModelArchitecture,
    layer: usize,
) -> Option<MoeLayerWeights<'a>> {
    // Pure MoE (GraniteMoE, OLMoE) builds identically to hybrid — the expert
    // store, router and per-expert byte tables are the same. Hybrid differs
    // only in having a parallel dense slab, which lives outside this struct.
    if !(arch.is_moe() || arch.is_hybrid_moe()) {
        return None;
    }
    let router_key = arch.moe_router_key(layer)?;
    let router_proj = weights.vectors.get(&router_key)?.as_slice();

    // Build per-expert byte tables. Per-layer Q4_K reads each expert from
    // its own offset-table entry; legacy BF16 slices the monolith by stride.
    let num_experts = arch.num_experts();
    let moe_inter = arch.moe_intermediate_size();
    let hidden = weights.hidden_size;
    let (experts_gate_up, experts_down, expert_data_format): (Vec<&[u8]>, Vec<&[u8]>, _) =
        if weights.has_per_layer_ffn() {
            let mut gu_table = Vec::with_capacity(num_experts);
            let mut dn_table = Vec::with_capacity(num_experts);
            for e in 0..num_experts {
                let (gu, dn) = weights.get_layer_entry_bytes(layer, e)?;
                gu_table.push(gu);
                dn_table.push(dn);
            }
            (gu_table, dn_table, crate::QuantFormat::Q4_K)
        } else {
            // Legacy BF16 monolithic blob: split into per-expert strides.
            let gate_up_key = arch.packed_experts_gate_up_key(layer)?;
            let down_key = arch.packed_experts_down_key(layer)?;
            let gu_all = weights.get_packed_bytes(&gate_up_key)?;
            let dn_all = weights.get_packed_bytes(&down_key)?;
            let gu_stride = 2 * moe_inter * hidden * 2; // BF16 = 2 bytes
            let dn_stride = hidden * moe_inter * 2;
            let gu_table: Vec<&[u8]> = (0..num_experts)
                .map(|e| &gu_all[e * gu_stride..(e + 1) * gu_stride])
                .collect();
            let dn_table: Vec<&[u8]> = (0..num_experts)
                .map(|e| &dn_all[e * dn_stride..(e + 1) * dn_stride])
                .collect();
            (gu_table, dn_table, crate::QuantFormat::BF16)
        };

    let router_scale = arch
        .moe_router_scale_key(layer)
        .and_then(|k| weights.vectors.get(&k))
        .map(|v| v.as_slice())
        .unwrap_or(&[]);
    let router_per_expert_scale = arch
        .moe_router_per_expert_scale_key(layer)
        .and_then(|k| weights.vectors.get(&k))
        .map(|v| v.as_slice())
        .unwrap_or(&[]);
    let pre_experts_norm = arch
        .moe_pre_experts_norm_key(layer)
        .and_then(|k| weights.vectors.get(&k))
        .map(|v| v.as_slice())
        .unwrap_or(&[]);
    let post_ffn1_norm = arch
        .moe_post_ffn1_norm_key(layer)
        .and_then(|k| weights.vectors.get(&k))
        .map(|v| v.as_slice())
        .unwrap_or(&[]);
    let post_experts_norm = arch
        .moe_post_experts_norm_key(layer)
        .and_then(|k| weights.vectors.get(&k))
        .map(|v| v.as_slice())
        .unwrap_or(&[]);
    let router_norm = arch
        .moe_router_norm_key(layer)
        .and_then(|k| weights.vectors.get(&k))
        .map(|v| v.as_slice())
        .unwrap_or(&[]);
    let router_norm_parameter_free = arch.moe_router_norm_parameter_free();
    let router_input_scalar = arch.moe_router_input_scalar().unwrap_or(1.0);

    let activation = match arch.activation() {
        larql_models::Activation::GeluTanh => crate::Activation::GeluTanh,
        _ => crate::Activation::Silu,
    };

    Some(MoeLayerWeights {
        experts_gate_up,
        experts_down,
        routing_policy: moe_routing_policy(arch.moe_router_type()),
        weight_layout: MoeWeightLayout::default(),
        expert_data_format,
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

/// Registry tag → `compute::QuantFormat` for the attention surface.
/// Explicit so a typo or new tag fails loudly rather than silently
/// aliasing to Q4_K. Lifted from inside `resolve_attn_weights` so the
/// mapping is unit-testable in isolation.
fn attn_str_to_format(s: &str) -> QuantFormat {
    // Source the tag→format mapping from `from_registry_tag` (single source
    // of truth) but keep the attention surface's supported-subset guard:
    // only Q4_K / Q6_K have a q8k attention matvec, so anything else still
    // fails loudly rather than silently aliasing.
    match QuantFormat::from_registry_tag(s) {
        Some(f @ (QuantFormat::Q4_K | QuantFormat::Q6_K)) => f,
        _ => panic!("resolve_attn_weights: registry tag {s:?} has no compute::QuantFormat mapping"),
    }
}

/// Helper: resolve attention weights from vindex (Q4_K preferred, Q8 fallback).
pub fn resolve_attn_weights<'a>(
    index: &'a dyn crate::KvIndex,
    layer: usize,
) -> Option<(
    QuantWeight<'a>,
    QuantWeight<'a>,
    QuantWeight<'a>,
    QuantWeight<'a>,
)> {
    let to_format = attn_str_to_format;

    if let Some([q, k, v, o]) = index.attn_kquant_layer_data(layer) {
        Some((
            QuantWeight {
                data: q.0,
                scales: None,
                format: to_format(q.1),
            },
            QuantWeight {
                data: k.0,
                scales: None,
                format: to_format(k.1),
            },
            QuantWeight {
                data: v.0,
                scales: None,
                format: to_format(v.1),
            },
            QuantWeight {
                data: o.0,
                scales: None,
                format: to_format(o.1),
            },
        ))
    } else if let Some([q, k, v, o]) = index.attn_q8_layer_data(layer) {
        Some((
            QuantWeight {
                data: q.0,
                scales: Some(q.1),
                format: QuantFormat::Q8_0,
            },
            QuantWeight {
                data: k.0,
                scales: Some(k.1),
                format: QuantFormat::Q8_0,
            },
            QuantWeight {
                data: v.0,
                scales: Some(v.1),
                format: QuantFormat::Q8_0,
            },
            QuantWeight {
                data: o.0,
                scales: Some(o.1),
                format: QuantFormat::Q8_0,
            },
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
/// Registry tag → `compute::QuantFormat` for the FFN surface, with an
/// explicit `fallback` for the legacy uniform-stride writer that
/// didn't emit per-matrix tags. Lifted from inside `resolve_ffn_weights`
/// so the mapping is unit-testable in isolation.
fn ffn_str_to_format(s: &str, fallback: QuantFormat) -> QuantFormat {
    // Empty tag = legacy uniform-stride writer (no per-matrix tags) → fallback.
    if s.is_empty() {
        return fallback;
    }
    // Mapping from `from_registry_tag` (single source of truth); the FFN
    // surface supports Q4_K / Q6_K / Q4_0, anything else fails loudly.
    match QuantFormat::from_registry_tag(s) {
        Some(f @ (QuantFormat::Q4_K | QuantFormat::Q6_K | QuantFormat::Q4_0)) => f,
        _ => panic!("resolve_ffn_weights: registry tag {s:?} has no compute::QuantFormat mapping"),
    }
}

pub fn resolve_ffn_weights<'a>(
    index: &'a dyn crate::KvIndex,
    layer: usize,
    q4_ffn_mmap: &'a [u8],
    q4_ffn_per_matrix: usize,
    ffn_format: QuantFormat,
) -> (QuantWeight<'a>, QuantWeight<'a>, QuantWeight<'a>) {
    let str_to_format = ffn_str_to_format;

    if let Some([gate, up, down]) = index.interleaved_kquant_layer_data(layer) {
        return (
            QuantWeight {
                data: gate.0,
                scales: None,
                format: str_to_format(gate.1, ffn_format),
            },
            QuantWeight {
                data: up.0,
                scales: None,
                format: str_to_format(up.1, ffn_format),
            },
            QuantWeight {
                data: down.0,
                scales: None,
                format: str_to_format(down.1, ffn_format),
            },
        );
    }

    // Pure MoE vindexes (e.g. Gemma-4 26B A4B) have no dense FFN tensor, so
    // `interleaved_q4k.bin` is 0 bytes. The interleaved-kquant branch above is
    // also absent for these (no per-matrix manifest entries), so we fall through
    // here with an empty `q4_ffn_mmap`. Return empty `QuantWeight` stubs —
    // `patch_pipeline_layers_for_remote_moe` (below) overwrites the per-layer
    // dense weights for MoE layers, and `moe_fn` supersedes the dense FFN path
    // during decode, so the empty slices are never read.
    if q4_ffn_mmap.is_empty() {
        let empty = QuantWeight {
            data: &[],
            scales: None,
            format: ffn_format,
        };
        return (empty, empty, empty);
    }

    let q4_ffn_per_layer = q4_ffn_per_matrix * 3;
    let fs = layer * q4_ffn_per_layer;
    (
        QuantWeight {
            data: &q4_ffn_mmap[fs..fs + q4_ffn_per_matrix],
            scales: None,
            format: ffn_format,
        },
        QuantWeight {
            data: &q4_ffn_mmap[fs + q4_ffn_per_matrix..fs + 2 * q4_ffn_per_matrix],
            scales: None,
            format: ffn_format,
        },
        QuantWeight {
            data: &q4_ffn_mmap[fs + 2 * q4_ffn_per_matrix..fs + 3 * q4_ffn_per_matrix],
            scales: None,
            format: ffn_format,
        },
    )
}

/// Build a complete Vec<FullPipelineLayer> for a range of layers.
/// Single source of truth — used by both GPU decode and GPU prefill paths.
#[allow(clippy::too_many_arguments)]
pub fn build_pipeline_layers<'a>(
    weights: &'a ModelWeights,
    index: &'a dyn crate::KvIndex,
    layer_range: std::ops::Range<usize>,
    q4_ffn_mmap: &'a [u8],
    q4_ffn_per_matrix: usize,
    ffn_format: QuantFormat,
) -> Vec<FullPipelineLayer<'a>> {
    layer_range
        .map(|layer| {
            let (wq, wk, wv, wo) = resolve_attn_weights(index, layer)
                .expect("No attention weights available for layer");
            let (gate, up, down) =
                resolve_ffn_weights(index, layer, q4_ffn_mmap, q4_ffn_per_matrix, ffn_format);
            build_arch_params(weights, layer, wq, wk, wv, wo, gate, up, down)
        })
        .collect()
}

/// For `--ffn URL` (remote dense FFN) deployments: all FFN work is delegated
/// to a remote server via `moe_fn` on every layer. This function sets
/// `ffn_is_remote = true` on all layers, which causes the Metal decode loop
/// to skip the local GPU FFN dispatches and route all FFN output through the
/// `moe_fn` callback instead.
///
/// No MoE stub injection is needed: the `has_moe` check in `setup.rs` now
/// also fires on `ffn_is_remote`, so the interleave path is taken for every
/// layer even without `layer.moe` being set.
pub fn patch_pipeline_layers_for_remote_ffn(layers: &mut [FullPipelineLayer<'_>]) {
    for layer in layers.iter_mut() {
        layer.ffn_is_remote = true;
    }
}

/// For `--moe-shards` (remote expert) deployments: the client vindex has no
/// per-layer expert bytes, so `build_moe_weights` returns `None` for every
/// layer, `has_moe = false`, and the Metal decode never calls `moe_fn`.
///
/// This function patches that by injecting a stub `MoeLayerWeights` for every
/// MoE-capable layer whose `moe` field is still `None`.  The stub has empty
/// expert slices — they are never read when `moe_fn` is `Some` (the remote
/// dispatch closure supersedes local `cpu_moe_forward`).  Norm weights are
/// populated from `weights.vectors` (loaded from `norms.bin` in the client
/// slice) so post-MoE normalisation remains correct.
pub fn patch_pipeline_layers_for_remote_moe<'a>(
    layers: &mut [FullPipelineLayer<'a>],
    weights: &'a ModelWeights,
) {
    let arch = &*weights.arch;
    if !arch.is_hybrid_moe() {
        return;
    }
    for (i, layer) in layers.iter_mut().enumerate() {
        if layer.moe.is_some() {
            continue;
        }
        if arch.moe_router_key(i).is_none() {
            continue;
        }
        layer.moe = Some(build_moe_stub(weights, arch, i));
    }
}

fn build_moe_stub<'a>(
    weights: &'a ModelWeights,
    arch: &dyn larql_models::ModelArchitecture,
    layer: usize,
) -> MoeLayerWeights<'a> {
    let sl = |k: Option<String>| -> &'a [f32] {
        k.and_then(|k| weights.vectors.get(&k))
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    };
    // expert_data_format is never read when moe_fn fires (remote path); match
    // what build_moe_weights would use so any fallback cpu_moe_forward still
    // decodes correctly if it ever runs.
    let expert_data_format = if weights.has_per_layer_ffn() {
        QuantFormat::Q4_K
    } else {
        QuantFormat::BF16
    };
    MoeLayerWeights {
        experts_gate_up: vec![],
        experts_down: vec![],
        routing_policy: moe_routing_policy(arch.moe_router_type()),
        weight_layout: MoeWeightLayout::default(),
        expert_data_format,
        router_proj: &[],
        router_scale: sl(arch.moe_router_scale_key(layer)),
        router_per_expert_scale: sl(arch.moe_router_per_expert_scale_key(layer)),
        router_norm: sl(arch.moe_router_norm_key(layer)),
        router_norm_parameter_free: arch.moe_router_norm_parameter_free(),
        router_input_scalar: arch.moe_router_input_scalar().unwrap_or(1.0),
        pre_experts_norm: sl(arch.moe_pre_experts_norm_key(layer)),
        post_ffn1_norm: sl(arch.moe_post_ffn1_norm_key(layer)),
        post_experts_norm: sl(arch.moe_post_experts_norm_key(layer)),
        num_experts: arch.num_experts(),
        top_k: arch.num_experts_per_token(),
        intermediate_size: arch.moe_intermediate_size(),
        activation: match arch.activation() {
            larql_models::Activation::GeluTanh => crate::Activation::GeluTanh,
            _ => crate::Activation::Silu,
        },
    }
}

fn moe_routing_policy(router_type: &str) -> MoeRoutingPolicy {
    match router_type {
        "gemma4_top_k_softmax" => MoeRoutingPolicy::gemma4_hybrid(),
        _ => MoeRoutingPolicy::top_k_softmax(),
    }
}

#[cfg(test)]
mod tests {
    //! Coverage for the simple per-arch helpers (kv shapes, format
    //! parsing, routing policy). The big MoE branches in
    //! `build_moe_weights` need a Gemma 4 MoE fixture and live in the
    //! `larql-inference` integration tests where that fixture is
    //! reachable.
    use super::*;
    use larql_models::test_fixtures::make_test_weights;

    #[test]
    fn kv_cache_shapes_for_arch_returns_one_pair_per_layer() {
        let weights = make_test_weights();
        let shapes = kv_cache_shapes_for_arch(&weights);
        assert_eq!(shapes.len(), weights.num_layers);
        for (num_kv, head_dim) in &shapes {
            assert!(*num_kv > 0);
            assert!(*head_dim > 0);
        }
    }

    #[test]
    fn attn_str_to_format_maps_known_tags() {
        assert_eq!(attn_str_to_format("Q4_K"), QuantFormat::Q4_K);
        assert_eq!(attn_str_to_format("Q6_K"), QuantFormat::Q6_K);
    }

    #[test]
    #[should_panic(expected = "no compute::QuantFormat mapping")]
    fn attn_str_to_format_panics_on_unknown_tag() {
        let _ = attn_str_to_format("Q42_X");
    }

    #[test]
    fn ffn_str_to_format_maps_known_tags() {
        assert_eq!(
            ffn_str_to_format("Q4_K", QuantFormat::Q4_K),
            QuantFormat::Q4_K
        );
        assert_eq!(
            ffn_str_to_format("Q6_K", QuantFormat::Q4_K),
            QuantFormat::Q6_K
        );
        assert_eq!(
            ffn_str_to_format("Q4_0", QuantFormat::Q4_K),
            QuantFormat::Q4_0
        );
        // Empty tag falls through to the caller's fallback.
        assert_eq!(ffn_str_to_format("", QuantFormat::Q4_0), QuantFormat::Q4_0);
        assert_eq!(ffn_str_to_format("", QuantFormat::Q4_K), QuantFormat::Q4_K);
    }

    #[test]
    #[should_panic(expected = "no compute::QuantFormat mapping")]
    fn ffn_str_to_format_panics_on_unknown_tag() {
        let _ = ffn_str_to_format("unknown", QuantFormat::Q4_K);
    }

    #[test]
    fn moe_routing_policy_maps_gemma4_tag() {
        // Gemma 4 hybrid tag → Gemma 4 routing.
        let _ = moe_routing_policy("gemma4_top_k_softmax");
        // Unknown tag → top-K softmax default.
        let _ = moe_routing_policy("unknown");
    }

    /// `resolve_attn_weights` falls through to the Q8 branch when the
    /// index returns Q8 data instead of Q4_K.
    #[test]
    fn resolve_attn_weights_uses_q8_branch_when_index_returns_q8() {
        struct Q8Idx {
            bytes: Vec<u8>,
            scales: Vec<f32>,
        }
        impl crate::KvIndex for Q8Idx {
            fn attn_q8_layer_data(&self, _l: usize) -> Option<[(&[u8], &[f32]); 4]> {
                Some([
                    (self.bytes.as_slice(), self.scales.as_slice()),
                    (self.bytes.as_slice(), self.scales.as_slice()),
                    (self.bytes.as_slice(), self.scales.as_slice()),
                    (self.bytes.as_slice(), self.scales.as_slice()),
                ])
            }
        }
        let idx = Q8Idx {
            bytes: vec![0u8; 16],
            scales: vec![1.0f32; 4],
        };
        let result = resolve_attn_weights(&idx, 0);
        let (q, _k, _v, _o) = result.expect("Q8 fallback returns Some");
        assert_eq!(q.format, QuantFormat::Q8_0);
    }

    /// `build_arch_params` rotary_dim branch fires when `rotary_fraction`
    /// is < 1.0 (partial-rotary archs like StarCoder2).
    #[test]
    fn build_arch_params_handles_partial_rotary_fraction() {
        let weights = larql_models::test_fixtures::make_starcoder2_test_weights();
        let dummy = crate::QuantWeight {
            data: &[],
            scales: None,
            format: QuantFormat::Q4_K,
        };
        // The partial-rotary branch is shape-dependent on the arch; what
        // we want is just to ensure no panic on a non-full-rotary arch.
        let layer = build_arch_params(&weights, 0, dummy, dummy, dummy, dummy, dummy, dummy, dummy);
        let _ = layer.rotary_dim;
    }

    /// `build_arch_params` on Llama2-style (Silu activation) fixture —
    /// covers the Silu fallback branch in the activation match.
    #[test]
    fn build_arch_params_handles_silu_activation() {
        let weights = make_test_weights();
        let dummy = crate::QuantWeight {
            data: &[],
            scales: None,
            format: QuantFormat::Q4_K,
        };
        let layer = build_arch_params(&weights, 0, dummy, dummy, dummy, dummy, dummy, dummy, dummy);
        assert!(matches!(layer.activation, crate::Activation::Silu));
    }

    /// `build_arch_params` on Starcoder2-style fixture covers the
    /// LayerNorm branch and the Standard (non-gated) FFN type.
    #[test]
    fn build_arch_params_handles_layernorm_and_standard_ffn() {
        let weights = larql_models::test_fixtures::make_starcoder2_test_weights();
        let dummy = crate::QuantWeight {
            data: &[],
            scales: None,
            format: QuantFormat::Q4_K,
        };
        let layer = build_arch_params(&weights, 0, dummy, dummy, dummy, dummy, dummy, dummy, dummy);
        assert!(matches!(layer.norm_type, crate::NormType::LayerNorm));
        assert!(matches!(layer.ffn_type, crate::FfnType::Standard));
    }

    /// `build_moe_weights` happy path on the Gemma 4 hybrid-MoE fixture
    /// — exercises the per-layer FFN router + packed expert slicing,
    /// the BF16-stride math, and the routing-policy assignment.
    #[test]
    fn build_moe_weights_succeeds_on_hybrid_moe_fixture() {
        let weights = larql_models::test_fixtures::make_test_gemma4_moe_weights();
        assert!(weights.arch.is_hybrid_moe());
        let arch = &*weights.arch;
        for layer in 0..weights.num_layers {
            let result = build_moe_weights(&weights, arch, layer);
            assert!(
                result.is_some(),
                "MoE weights should resolve for layer {layer} on Gemma 4 hybrid-MoE"
            );
        }
    }

    /// `build_moe_weights` returns None on a non-MoE arch — covers the
    /// `arch.moe_router_key(layer)?` short-circuit.
    #[test]
    fn build_moe_weights_returns_none_on_non_moe_arch() {
        let weights = make_test_weights();
        assert!(!weights.arch.is_hybrid_moe());
        assert!(build_moe_weights(&weights, &*weights.arch, 0).is_none());
    }

    /// `patch_pipeline_layers_for_remote_moe` injects MoE stubs on
    /// MoE-capable layers when the local moe slot is still None.
    #[test]
    fn patch_pipeline_layers_for_remote_moe_injects_stubs() {
        let weights = larql_models::test_fixtures::make_test_gemma4_moe_weights();
        // Build pipeline layers with no MoE locally — simulates the
        // remote-MoE client deployment.
        let dummy = crate::QuantWeight {
            data: &[],
            scales: None,
            format: QuantFormat::Q4_K,
        };
        let mut layers: Vec<crate::FullPipelineLayer<'_>> = (0..weights.num_layers)
            .map(|_| crate::FullPipelineLayer {
                wq: dummy,
                wk: dummy,
                wv: dummy,
                wo: dummy,
                gate: dummy,
                up: dummy,
                down: dummy,
                ..crate::FullPipelineLayer::default()
            })
            .collect();
        // Pre-patch: every layer has moe = None.
        for l in &layers {
            assert!(l.moe.is_none());
        }
        patch_pipeline_layers_for_remote_moe(&mut layers, &weights);
        // Post-patch: every MoE-capable layer has Some moe stub.
        let mut any_patched = false;
        for l in &layers {
            if l.moe.is_some() {
                any_patched = true;
            }
        }
        assert!(any_patched, "patch must inject at least one MoE stub");
    }

    #[test]
    fn patch_pipeline_layers_for_remote_ffn_sets_remote_flag() {
        // Build a 1-layer pipeline and patch to remote FFN.
        let layer = crate::FullPipelineLayer::default();
        let mut layers = vec![layer];
        assert!(!layers[0].ffn_is_remote);
        patch_pipeline_layers_for_remote_ffn(&mut layers);
        for l in &layers {
            assert!(l.ffn_is_remote, "patch should set ffn_is_remote = true");
        }
    }

    /// Pure-MoE vindexes (e.g. Gemma-4 26B A4B) ship a zero-byte
    /// `interleaved_q4k.bin` because there is no dense FFN. Before the
    /// `is_empty()` guard, `resolve_ffn_weights` would panic on the first
    /// `q4_ffn_mmap[fs..fs + q4_ffn_per_matrix]` slice. The guard returns
    /// empty `QuantWeight` stubs — `patch_pipeline_layers_for_remote_moe`
    /// overwrites the per-layer MoE weights afterward and the dense FFN
    /// path is bypassed entirely by `moe_fn` during decode, so the empty
    /// slices are never read.
    #[test]
    fn resolve_ffn_weights_returns_empty_stubs_when_q4_ffn_mmap_is_empty() {
        struct EmptyIdx;
        impl crate::KvIndex for EmptyIdx {}
        let idx = EmptyIdx;
        let empty_mmap: &[u8] = &[];
        // q4_ffn_per_matrix is irrelevant on this path — what we're pinning
        // is "no slice happens against the empty mmap" (i.e. no panic).
        let (gate, up, down) =
            resolve_ffn_weights(&idx, 7, empty_mmap, 1_115_136, QuantFormat::Q4_K);
        assert!(gate.data.is_empty());
        assert!(up.data.is_empty());
        assert!(down.data.is_empty());
        assert_eq!(gate.format, QuantFormat::Q4_K);
        assert_eq!(up.format, QuantFormat::Q4_K);
        assert_eq!(down.format, QuantFormat::Q4_K);
    }
}
