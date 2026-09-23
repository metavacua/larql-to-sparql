use super::enums::{Activation, FfnType, NormType, PositionEncodingType};
use super::quant_format::QuantWeight;

/// Attention projection weights for one layer.
#[derive(Clone, Copy)]
pub struct AttentionWeights<'a> {
    pub wq: QuantWeight<'a>,
    pub wk: QuantWeight<'a>,
    pub wv: QuantWeight<'a>,
    pub wo: QuantWeight<'a>,
}

/// Dense FFN projection weights for one layer.
#[derive(Clone, Copy)]
pub struct FfnWeights<'a> {
    /// Gate projection. Used only when [`FfnSpec::ffn_type`] is [`FfnType::Gated`].
    pub gate: QuantWeight<'a>,
    pub up: QuantWeight<'a>,
    pub down: QuantWeight<'a>,
}

/// Grouped weight view for one layer.
#[derive(Clone, Copy)]
pub struct LayerWeights<'a> {
    pub attention: AttentionWeights<'a>,
    pub ffn: FfnWeights<'a>,
}

/// Norm weights, biases, and scalar norm behavior for one layer.
#[derive(Clone, Copy)]
pub struct LayerNorms<'a> {
    pub input_norm: &'a [f32],
    pub post_attn_norm: &'a [f32],
    pub pre_ffn_norm: Option<&'a [f32]>,
    pub post_ffn_norm: Option<&'a [f32]>,
    pub input_norm_bias: Option<&'a [f32]>,
    pub post_attn_norm_bias: Option<&'a [f32]>,
    pub norm_offset: f32,
    pub qk_norm_offset: f32,
    pub eps: f32,
    pub has_post_norms: bool,
    pub norm_type: NormType,
}

/// Per-layer attention geometry and position-encoding behavior.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AttentionSpec {
    pub attn_scale: f32,
    pub head_dim: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub rope_base: f32,
    pub rotary_dim: usize,
    pub sliding_window: usize,
    pub has_v_norm: bool,
    pub q_norm_enabled: bool,
    pub k_norm_enabled: bool,
    pub position_encoding: PositionEncodingType,
}

/// Dense FFN architecture behavior for one layer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FfnSpec {
    pub ffn_type: FfnType,
    pub activation: Activation,
}

/// Remote FFN dispatch behavior for one layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RemoteFfnSpec {
    pub is_remote: bool,
}

/// Per-Layer Embeddings (Gemma 4 E2B) per-layer weights view.
///
/// Returned by [`FullPipelineLayer::ple_spec`] only when all three required
/// weights are present.  All slices are row-major f32.
#[derive(Clone, Copy)]
pub struct PleSpec<'a> {
    /// `[ple_dim, hidden]` — projects hidden state down to `ple_dim` for gating.
    pub input_gate: &'a [f32],
    /// `[hidden, ple_dim]` — projects gated PLE signal back up to hidden state.
    pub projection: &'a [f32],
    /// `[hidden]` — RMSNorm applied to the projection output before residual add.
    pub post_norm: &'a [f32],
}
