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
    FullPipelineLayer, MoeExpertScales, MoeFusedRowLayout, MoeLayerWeights, MoeRoutingPolicy,
    MoeWeightLayout, QuantFormat, QuantWeight,
};
use larql_models::ModelWeights;

pub const DEFAULT_GPU_KV_CACHE_MAX_SEQ: usize = 4096;

/// Occupancy multiple of the window at which KV compaction runs.
///
/// Compaction memmoves the surviving rows, so running it every step would
/// cost O(window) per token; letting occupancy reach this multiple makes
/// it O(1) amortised. The attention span clamp means the extra resident
/// rows are never read, so this is pure amortisation slack and carries no
/// semantic meaning.
///
/// It is therefore the difference between what a sliding layer can
/// *reach* (`W`) and what it must be able to *hold* (`SLACK * W`). A ring
/// buffer would let capacity approach `W` without changing any policy.
pub const KV_COMPACTION_SLACK: usize = 2;

/// Physical rows a layer's KV cache must be able to hold.
///
/// A windowed layer leaves prefill with `W` rows (see
/// `full_pipeline/kv_copy.rs`), climbs to `SLACK * W` during decode and is
/// compacted back. It can therefore never need more than `SLACK * W`, and
/// allocating `default_capacity` for it is waste.
///
/// `sliding_window == 0` is the unbounded sentinel — global layers keep
/// the full default, because nothing bounds what they may read.
pub fn kv_capacity_for_window(sliding_window: usize, default_capacity: usize) -> usize {
    if sliding_window == 0 {
        default_capacity
    } else {
        sliding_window
            .saturating_mul(KV_COMPACTION_SLACK)
            .min(default_capacity)
    }
}

/// Per-layer KV capacities for a model, parallel to
/// [`kv_cache_shapes_for_arch`].
///
/// **Presumes compaction.** A window-derived capacity is only sound on a
/// path that compacts the cache back below `window x slack`; the GPU
/// decode path does not (it appends at `current_len` and attends absolute
/// rows), and sizing its sliding layers with this function ran them off
/// the end of their buffers (#229). That path now allocates every layer at
/// the full default; use this only where `compact_kv_to_window` runs.
///
/// Deliberately a separate slice rather than a third tuple element:
/// `num_kv_heads` and `head_dim` describe the *representation* of a row,
/// while capacity is *residency policy*. Widening the shape tuple would
/// put both behind one type across five crates — see
/// `docs/kv-residency-contract.md`.
pub fn kv_capacities_for_arch(weights: &ModelWeights, default_capacity: usize) -> Vec<usize> {
    let arch = &*weights.arch;
    (0..weights.num_layers)
        .map(|layer| {
            let window =
                crate::forward_overrides::effective_attention_window_for_layer(arch, layer)
                    .unwrap_or(0);
            kv_capacity_for_window(window, default_capacity)
        })
        .collect()
}

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

    // Attention projection biases (GPT-OSS: all four; most archs: none).
    // Resolved here, at the arch boundary, exactly like the sinks above —
    // the CPU reference path reads the same keys, so leaving these unset
    // makes the GPU forward silently biasless (docs/k3-funnel.md §4.6).
    let bias_vec = |key: Option<String>| {
        key.and_then(|k| weights.vectors.get(&k))
            .map(|v| v.as_slice())
    };
    let attn_q_bias = bias_vec(arch.attn_q_bias_key(layer));
    let attn_k_bias = bias_vec(arch.attn_k_bias_key(layer));
    let attn_v_bias = bias_vec(arch.attn_v_bias_key(layer));
    let attn_o_bias = bias_vec(arch.attn_o_bias_key(layer));

    // 0.0 = disabled. Set here, at the arch boundary, so every decode
    // consumer sees the model's real cap instead of the field default
    // silently answering for capped architectures.
    let attn_softcap = arch.attn_logit_softcapping().unwrap_or(0.0);

    FullPipelineLayer {
        attn_sinks,
        attn_q_bias,
        attn_k_bias,
        attn_v_bias,
        attn_o_bias,
        attn_softcap,
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
        activation: arch.activation().into(),
        qk_norm_offset: arch.qk_norm_weight_offset(),
        eps: arch.norm_eps(),
        norm_type: arch.norm_type().into(),
        ffn_type: arch.ffn_type().into(),
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
        // Same resolver the CPU rope path uses, so the GPU cannot be handed a
        // different notion of this layer's frequencies.
        rope_freq: crate::attention::rope::rope_freq_plan(
            layer_hd,
            rotary_frac,
            effective_rope_base_for_layer(arch, layer),
            crate::forward_overrides::effective_rope_position_divisor_for_layer(arch, layer),
            crate::forward_overrides::effective_rope_freq_scaling(arch),
        ),
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
            QuantWeight::new(to_format(q.1), q.0, crate::QuantAux::None),
            QuantWeight::new(to_format(k.1), k.0, crate::QuantAux::None),
            QuantWeight::new(to_format(v.1), v.0, crate::QuantAux::None),
            QuantWeight::new(to_format(o.1), o.0, crate::QuantAux::None),
        ))
    } else if let Some([q, k, v, o]) = index.attn_q8_layer_data(layer) {
        Some((
            QuantWeight::new(QuantFormat::Q8_0, q.0, crate::QuantAux::ExternalScales(q.1)),
            QuantWeight::new(QuantFormat::Q8_0, k.0, crate::QuantAux::ExternalScales(k.1)),
            QuantWeight::new(QuantFormat::Q8_0, v.0, crate::QuantAux::ExternalScales(v.1)),
            QuantWeight::new(QuantFormat::Q8_0, o.0, crate::QuantAux::ExternalScales(o.1)),
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
            QuantWeight::new(
                str_to_format(gate.1, ffn_format),
                gate.0,
                crate::QuantAux::None,
            ),
            QuantWeight::new(str_to_format(up.1, ffn_format), up.0, crate::QuantAux::None),
            QuantWeight::new(
                str_to_format(down.1, ffn_format),
                down.0,
                crate::QuantAux::None,
            ),
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
        let empty = QuantWeight::new(ffn_format, &[], crate::QuantAux::None);
        return (empty, empty, empty);
    }

    let q4_ffn_per_layer = q4_ffn_per_matrix * 3;
    let fs = layer * q4_ffn_per_layer;
    (
        QuantWeight::new(
            ffn_format,
            &q4_ffn_mmap[fs..fs + q4_ffn_per_matrix],
            crate::QuantAux::None,
        ),
        QuantWeight::new(
            ffn_format,
            &q4_ffn_mmap[fs + q4_ffn_per_matrix..fs + 2 * q4_ffn_per_matrix],
            crate::QuantAux::None,
        ),
        QuantWeight::new(
            ffn_format,
            &q4_ffn_mmap[fs + 2 * q4_ffn_per_matrix..fs + 3 * q4_ffn_per_matrix],
            crate::QuantAux::None,
        ),
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
mod moe_build;
pub use moe_build::{build_moe_weights, patch_pipeline_layers_for_remote_moe};

#[cfg(test)]
mod tests;
