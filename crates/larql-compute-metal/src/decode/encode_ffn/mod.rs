//! Step 6 of the decode pipeline: format-aware FFN dispatch.
//!
//! Three production paths on the same `(gate, up, down)` triplet:
//!   - **Q4_KF** — llama.cpp-exact kernel; fused gate+up; `act_buf` then
//!     down via `quant_matvec` (mixed-quant aware).
//!   - **Q4_K** — our kernel; fused gate+up; down via `quant_matvec`
//!     (Gemma 3 4B ships Q6_K down even when gate/up are Q4_K).
//!   - **Q4_0** (legacy) — Q8-input matvec for gate/up; `q4.f32_matvec`
//!     for down.
//!
//! Used to live inline in `decode_token_with_moe_fn`; pulled out here
//! so `decode/mod.rs` stays readable. Behaviour is byte-identical to
//! the original block.
//!
//! All buffer + pipeline references are held in `FfnBufs` and
//! `FfnDims` so the encoder method has a manageable signature.

use larql_compute::FullPipelineLayer;

/// Max `inter_padded` for which the fused Q4_K GEGLU+down kernel is
/// known to be NaN-free.
///
/// Set after Gemma 4 31B (`inter = 21504`) hit a data-dependent NaN at
/// layer 11 despite clean gate/up inputs and finite weight scales (see
/// the block doc on the dispatch site below). 16384 covers Gemma 3 4B
/// (`inter = 10240`), Gemma 4 26B-A4B (`inter = 2112`), Llama 2 7B
/// (`inter = 11008`), Mistral 7B (`inter = 14336`); larger intermediate
/// sizes fall through to the separated GEGLU + matvec path until the
/// fused-kernel NaN root cause is found.
pub(super) const MAX_FUSED_GEGLU_DOWN_INTER: usize = 16384;

mod per_format;
mod phases;
mod step;

pub(crate) struct FfnBufs<'a> {
    // Weights for this layer
    pub gate_w: &'a metal::Buffer,
    pub up_w: &'a metal::Buffer,
    pub down_w: &'a metal::Buffer,
    // Inputs
    pub ffn_norm_out: &'a metal::Buffer, // f32 input (Q4_K / Q4_KF paths)
    pub ffn_q8: &'a metal::Buffer,       // Q8 input bytes (Q4_0 path)
    pub ffn_q8s: &'a metal::Buffer,      // Q8 input scales (Q4_0 path)
    // Scratch (gate output reused even on non-gated paths)
    pub gate_out_scratch: &'a metal::Buffer,
    pub up_out: &'a metal::Buffer,
    pub act_buf: &'a metal::Buffer,
    // Output
    pub down_out: &'a metal::Buffer,
}

#[derive(Copy, Clone)]
pub(crate) struct FfnDims {
    pub hidden: usize,
    pub inter: usize,
    /// `inter` rounded up to the next multiple of 256 — used by the Q4K
    /// down dispatch when storage is per-row-padded super-blocks.
    pub inter_padded: usize,
}

/// Validate a layer's FFN formats and return the format that drives
/// route selection (capability audit F3).
///
/// Called at decode entry, before any command buffer or encoder is
/// created, so a refusal panic unwinds cleanly instead of tripping
/// Metal's "encoder released without endEncoding" abort. Rules:
/// - a gated layer's gate and up must share a format — no fused kernel
///   decodes a mixed pair, and reinterpretation is silent corruption;
/// - a non-gated layer routes on `up` (its actual first matvec; `gate`
///   may be a default-constructed placeholder);
/// - only Q4_K / Q4_KF / Q6_K / Q4_0 have decode FFN paths.
pub(crate) fn validate_ffn_formats(layer: &FullPipelineLayer) -> larql_compute::QuantFormat {
    use larql_compute::QuantFormat;
    if layer.is_gated() {
        assert_eq!(
            layer.gate.format(),
            layer.up.format(),
            "mixed gate/up FFN formats have no fused Metal kernel; \
             gate={:?} up={:?}",
            layer.gate.format(),
            layer.up.format(),
        );
    }
    let route_fmt = if layer.is_gated() {
        layer.gate.format()
    } else {
        layer.up.format()
    };
    match route_fmt {
        QuantFormat::Q4_K | QuantFormat::Q4_KF | QuantFormat::Q6_K | QuantFormat::Q4_0 => route_fmt,
        other => panic!(
            "FFN has no Metal decode path for {other:?} gate/up weights; \
             supported: Q4_K, Q4_KF, Q6_K, Q4_0"
        ),
    }
}
