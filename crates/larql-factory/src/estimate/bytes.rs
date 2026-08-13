//! Coarse per-component byte-count estimate from model dimensions.
//!
//! There's no pre-extraction size-prediction formula anywhere in the
//! codebase to reuse — the closest precedent is
//! `larql_vindex::config::index::VindexConfig::estimate_resident_bytes`,
//! which estimates *post-extraction* RAM for dense f16/f32 vindexes
//! (embed = vocab·hidden·elem; attention ≈ 4·hidden²·elem/layer; FFN =
//! 3·hidden·intermediate·elem/layer) and explicitly bails to zero for
//! any quantised format. This module follows the same arithmetic shape,
//! extended to MoE (expert count multiplies the FFN term) and to a
//! quantised FFN term — approximate by construction, not byte-exact.
//! Good enough for "should this be a $3 build or a $150 one", not for
//! reconciling against a real manifest.

#[cfg(target_arch = "wasm32")]
use crate::prelude::*;

use larql_vindex_spec::StorageDtype;

use super::dims::ModelDims;

/// Informal Q4_K rate — no formal constant exists anywhere in the
/// codebase; this mirrors the "~4.5 bits/elem" comment in
/// `larql_vindex::config::index` (`VindexConfig::estimate_resident_bytes`).
const Q4K_APPROX_BITS_PER_ELEMENT: f64 = 4.5;
const BITS_PER_BYTE: f64 = 8.0;

/// Attention's Q/K/V/O projections, each roughly `hidden × hidden` —
/// coarse (ignores GQA's smaller K/V heads).
const ATTN_PROJECTIONS_PER_LAYER: f64 = 4.0;

/// Dense FFN's gate/up/down projections.
const FFN_PROJECTIONS_PER_LAYER: f64 = 3.0;

/// Estimated bytes per coarse component, before any preset weighting.
#[derive(Clone, Copy, Debug, Default)]
pub struct ComponentBytes {
    /// Embedding table (`vocab_size × hidden_size`).
    pub embed: u64,
    /// Attention projections across every layer.
    pub attn: u64,
    /// FFN weights across every layer (all experts, for MoE).
    pub ffn: u64,
    /// Output projection (`lm_head`) — modelled as the same shape as
    /// `embed`; safe-overestimate if the model ties embeddings.
    pub lm_head: u64,
    /// Router weights (MoE only).
    pub router: u64,
}

fn dtype_bytes(dtype: StorageDtype) -> f64 {
    match dtype {
        StorageDtype::F32 => 4.0,
        StorageDtype::F16 => 2.0,
    }
}

/// Estimate per-component bytes for `dims` at `dtype`, with `down_q4k`
/// applying the informal Q4_K ratio to the FFN component only (the
/// recipe option this mirrors is specifically about the down
/// projection's quant format).
pub fn estimate_component_bytes(
    dims: &ModelDims,
    dtype: StorageDtype,
    down_q4k: bool,
) -> ComponentBytes {
    let elem_bytes = dtype_bytes(dtype);
    let quant_scale = if down_q4k {
        Q4K_APPROX_BITS_PER_ELEMENT / (elem_bytes * BITS_PER_BYTE)
    } else {
        1.0
    };

    let embed = (dims.vocab_size as f64 * dims.hidden_size as f64 * elem_bytes) as u64;
    let lm_head = embed;

    let attn = (ATTN_PROJECTIONS_PER_LAYER
        * (dims.hidden_size as f64).powi(2)
        * elem_bytes
        * dims.num_layers as f64) as u64;

    let expert_multiplier = dims.num_experts.unwrap_or(1) as f64;
    let ffn = (FFN_PROJECTIONS_PER_LAYER
        * dims.hidden_size as f64
        * dims.intermediate_size as f64
        * elem_bytes
        * expert_multiplier
        * dims.num_layers as f64
        * quant_scale) as u64;

    let router = dims
        .num_experts
        .map(|n| (n as f64 * dims.hidden_size as f64 * elem_bytes) as u64)
        .unwrap_or(0);

    ComponentBytes {
        embed,
        attn,
        ffn,
        lm_head,
        router,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dense_dims() -> ModelDims {
        ModelDims {
            num_layers: 34,
            hidden_size: 2560,
            intermediate_size: 10240,
            vocab_size: 262_208,
            num_experts: None,
        }
    }

    #[test]
    fn embed_and_lm_head_match_vocab_times_hidden() {
        let c = estimate_component_bytes(&dense_dims(), StorageDtype::F16, false);
        let expected = 262_208 * 2560 * 2;
        assert_eq!(c.embed, expected as u64);
        assert_eq!(c.lm_head, expected as u64);
    }

    #[test]
    fn f32_doubles_every_component_vs_f16() {
        let f16 = estimate_component_bytes(&dense_dims(), StorageDtype::F16, false);
        let f32 = estimate_component_bytes(&dense_dims(), StorageDtype::F32, false);
        assert_eq!(f32.embed, f16.embed * 2);
        assert_eq!(f32.attn, f16.attn * 2);
        assert_eq!(f32.ffn, f16.ffn * 2);
    }

    #[test]
    fn dense_model_has_no_router_bytes() {
        let c = estimate_component_bytes(&dense_dims(), StorageDtype::F16, false);
        assert_eq!(c.router, 0);
    }

    #[test]
    fn moe_model_scales_ffn_by_expert_count_and_has_router_bytes() {
        let mut moe = dense_dims();
        moe.num_experts = Some(8);
        let dense = estimate_component_bytes(&dense_dims(), StorageDtype::F16, false);
        let moe_est = estimate_component_bytes(&moe, StorageDtype::F16, false);
        assert_eq!(moe_est.ffn, dense.ffn * 8);
        assert!(moe_est.router > 0);
    }

    #[test]
    fn down_q4k_shrinks_ffn_but_not_attention_or_embed() {
        let full = estimate_component_bytes(&dense_dims(), StorageDtype::F16, false);
        let quantised = estimate_component_bytes(&dense_dims(), StorageDtype::F16, true);
        assert!(quantised.ffn < full.ffn);
        assert_eq!(quantised.attn, full.attn);
        assert_eq!(quantised.embed, full.embed);
    }

    #[test]
    fn zero_vocab_size_zeroes_embed_and_lm_head_only() {
        let mut dims = dense_dims();
        dims.vocab_size = 0;
        let c = estimate_component_bytes(&dims, StorageDtype::F16, false);
        assert_eq!(c.embed, 0);
        assert_eq!(c.lm_head, 0);
        assert!(c.attn > 0);
        assert!(c.ffn > 0);
    }
}
