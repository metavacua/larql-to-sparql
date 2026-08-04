/// Norm type for layer normalization.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NormType {
    /// RMSNorm — Llama, Gemma, Qwen, most modern models.
    RmsNorm,
    /// Standard LayerNorm (mean-subtraction + variance normalization) — StarCoder2, GPT-2.
    LayerNorm,
}

/// FFN type: gated (gate+up→GEGLU→down) vs standard (up→activation→down).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FfnType {
    /// Gated: SiLU(x @ gate.T) * (x @ up.T) @ down.T — Llama, Gemma, Mistral.
    Gated,
    /// Standard: activation(x @ up.T) @ down.T — StarCoder2, GPT-2.
    Standard,
}

/// Activation function for FFN.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Activation {
    /// SiLU / Swish — Llama, Mistral, Qwen.
    Silu,
    /// GELU with tanh approximation — Gemma, StarCoder2.
    GeluTanh,
    /// Exact GELU (erf-based) — used in some GPT-2 variants.
    GeluExact,
    /// ReLU — legacy models (GPT-J, etc.).
    ReLU,
}

/// Positional encoding strategy for attention.
///
/// Most transformer models use RoPE. Non-RoPE variants (ALiBi, absolute,
/// none) are tracked here so future backends can guard on the type rather
/// than assuming RoPE is always present.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PositionEncodingType {
    /// Rotary position embedding. `base` and `rotary_dim` live in
    /// [`FullPipelineLayer`] so the encoding is still fully per-layer.
    RoPE,
    /// Attention with Linear Biases (no learned embeddings).
    ALiBi,
    /// Fixed absolute sinusoidal or learned embeddings (injected at
    /// the embedding layer, not per-attention-head).
    Absolute,
    /// No position encoding (e.g. some cross-attention blocks).
    None,
}

/// Hybrid MoE (Mixture-of-Experts) weights for one layer.
///
/// Gemma 4 26B A4B runs a dense MLP and an expert block in parallel per layer,
/// summing their outputs. This struct carries the expert-block tensors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoeInputSource {
    /// Use the residual stream exactly as passed into the MoE block.
    Residual,
    /// Use `rms_norm(residual, pre_experts_norm)` as the stage input.
    PreExpertsNorm,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoeRouterNormPolicy {
    /// Do not apply a router-specific norm.
    None,
    /// Apply `router_norm` when present; otherwise leave the router input unchanged.
    Learned,
    /// Apply parameter-free RMSNorm regardless of learned router weights.
    ParameterFree,
    /// Prefer learned `router_norm`; otherwise use parameter-free RMSNorm when enabled.
    LearnedOrParameterFree,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoeTopKWeightPolicy {
    /// Keep selected weights as the original softmax probabilities.
    RawSoftmax,
    /// Renormalize selected top-k weights so they sum to 1 before scaling.
    RenormalizedSoftmax,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoeExpertScalePolicy {
    /// Ignore `router_per_expert_scale`.
    None,
    /// Multiply selected weights by `router_per_expert_scale[expert_id]` when present.
    PerExpert,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoePostExpertNormPolicy {
    /// Return the weighted expert sum directly.
    None,
    /// Apply `post_experts_norm` via RMSNorm when the tensor is present.
    RmsNorm,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoeDownPaddingPolicy {
    /// Expert down matrices use `intermediate_size` columns.
    None,
    /// Expert down matrices are padded to the quant format's block width.
    QuantBlock,
}

// ── Backward compatibility: convert old-style bool to new enums ──

impl From<bool> for Activation {
    /// `true` = GeluTanh (Gemma), `false` = Silu (Llama).
    fn from(use_gelu_tanh: bool) -> Self {
        if use_gelu_tanh {
            Activation::GeluTanh
        } else {
            Activation::Silu
        }
    }
}
