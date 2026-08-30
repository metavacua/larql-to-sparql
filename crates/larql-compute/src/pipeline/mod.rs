//! Pipeline layer types — per-layer architecture parameters for the compute pipeline.
//!
//! These types carry all model-specific behavior per-layer:
//! norm type, activation, attention geometry, RoPE, FFN type, etc.
//! The compute backends read these fields per-layer — no hardcoded
//! model assumptions in the execution path.

mod enums;
mod layer;
mod moe;
mod quant_format;
mod weights;

#[cfg(test)]
mod tests;

/// Default RoPE base frequency (Llama, Gemma sliding-window layers).
pub const ROPE_BASE_DEFAULT: f32 = 10_000.0;

/// Long-context RoPE base frequency (Gemma global-attention layers).
pub const ROPE_BASE_GLOBAL: f32 = 1_000_000.0;

/// Default RMSNorm epsilon. Prevents division by zero in normalization.
pub const RMSNORM_EPSILON_DEFAULT: f32 = 1e-6;

pub use enums::{
    Activation, FfnType, MoeDownPaddingPolicy, MoeExpertScalePolicy, MoeInputSource,
    MoePostExpertNormPolicy, MoeRouterNormPolicy, MoeTopKWeightPolicy, NormType,
    PositionEncodingType,
};
pub use layer::FullPipelineLayer;
pub use moe::{
    stored_gate_up_cols, ExpertBankOverride, ExpertMlp, MoeExpertScales, MoeFusedRowLayout,
    MoeGateRule, MoeLayerWeights, MoeRoutingPolicy, MoeSpec, MoeWeightLayout,
};
pub use quant_format::{
    ExternalScaleKind, QuantAux, QuantFormat, QuantWeight, ScaleStorage, Q4_KF_BLOCK_BYTES,
};
pub use weights::{
    AttentionSpec, AttentionWeights, FfnSpec, FfnWeights, LayerNorms, LayerWeights, PleSpec,
    RemoteFfnSpec,
};
