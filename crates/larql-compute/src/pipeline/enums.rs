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

// ── model → compute translation ─────────────────────────────────────
//
// These three `From` impls are the ONE definition of how an
// architecture's declared behaviour crosses into the compute layer.
// They exist because the previous form — a `match` with a `_ =>`
// wildcard, re-spelled at each of the three `pipeline_layer.rs`
// construction sites — is the silent-default shape that this programme
// has now found seven times (ROADMAP H1's `moe_router_type`, H5a's
// `lm_head` tie, three in `docs/k3-funnel.md` §4.7.8, `layer_types` in
// §4.9.1, and this one).
//
// The `Activation` case was not merely stylistic. The wildcard mapped
// **`Relu` and `Gelu` both onto `Silu`**, and it did so *upstream* of
// `larql-compute-metal`'s `assert_metal_activation_supported`, whose own
// doc comment says it exists to "fail loud rather than silently routing
// GeluExact / ReLU layers to SiLU (the prior behaviour, which produced
// wrong logits with no signal)". That guard is tested, correct, and was
// unreachable: nothing could ever hand it a `GeluExact` or `ReLU`
// because this translation had already collapsed them. The fix was
// applied at the consumer and missed at the producer.
//
// No in-tree architecture returns `Gelu` or `Relu` today (every
// override returns `GeluTanh`; the trait default is `Silu`), so this is
// behaviour-preserving for every model currently served. It converts a
// latent silent-wrong into the loud refusal the Metal layer already
// implements.

impl Activation {
    /// Which of the two implemented gate/up FFN kernel families this
    /// activation dispatches to on the CPU MoE expert paths:
    /// `true` = gelu-tanh, `false` = SiLU.
    ///
    /// Compute-side twin of
    /// [`larql_models::Activation::uses_gelu_tanh_gate_up`], and for the
    /// same reason: the mapping was copy-pasted across five MoE expert
    /// loops (`moe/expert/f32.rs` ×3, `moe/expert/q4k.rs`,
    /// `moe/forward.rs`), each with a `_ =>` wildcard that dropped every
    /// non-`GeluTanh` variant into SiLU. Once the model→compute
    /// translation above stopped collapsing `Gelu`/`Relu` into `Silu`,
    /// those wildcards would have become the *new* place the same defect
    /// lived. Exhaustive here so a new variant fails to compile instead.
    ///
    /// - [`Activation::GeluExact`] is served by the tanh approximation —
    ///   the same deliberate, documented approximation the models-side
    ///   helper makes; no exact-GELU gate/up kernel exists.
    /// - [`Activation::ReLU`] has no gate/up kernel and panics loudly
    ///   rather than silently computing SiLU numerics.
    pub fn gate_up_is_gelu_tanh(self) -> bool {
        match self {
            Self::GeluTanh | Self::GeluExact => true,
            Self::Silu => false,
            Self::ReLU => panic!(
                "Activation::ReLU has no gate/up FFN kernel on the CPU MoE expert paths \
                 (only gelu-tanh and SiLU are implemented)"
            ),
        }
    }
}

impl From<larql_models::Activation> for Activation {
    fn from(a: larql_models::Activation) -> Self {
        // Exhaustive on purpose: a new `larql_models::Activation`
        // variant must fail to compile here rather than land in SiLU.
        match a {
            larql_models::Activation::Silu => Self::Silu,
            larql_models::Activation::GeluTanh => Self::GeluTanh,
            larql_models::Activation::Gelu => Self::GeluExact,
            larql_models::Activation::Relu => Self::ReLU,
        }
    }
}

impl From<larql_models::NormType> for NormType {
    fn from(n: larql_models::NormType) -> Self {
        match n {
            larql_models::NormType::RmsNorm => Self::RmsNorm,
            larql_models::NormType::LayerNorm => Self::LayerNorm,
        }
    }
}

impl From<larql_models::FfnType> for FfnType {
    fn from(f: larql_models::FfnType) -> Self {
        match f {
            larql_models::FfnType::Gated => Self::Gated,
            larql_models::FfnType::Standard => Self::Standard,
        }
    }
}

#[cfg(test)]
mod translation_tests {
    use super::*;

    /// Every model-side activation must reach a *distinct* compute-side
    /// variant. The predecessor mapping sent three of the four to
    /// `Silu`, which is precisely what made the defect invisible.
    #[test]
    fn every_activation_crosses_to_a_distinct_variant() {
        let pairs = [
            (larql_models::Activation::Silu, Activation::Silu),
            (larql_models::Activation::GeluTanh, Activation::GeluTanh),
            (larql_models::Activation::Gelu, Activation::GeluExact),
            (larql_models::Activation::Relu, Activation::ReLU),
        ];
        for (from, want) in pairs {
            assert_eq!(Activation::from(from), want, "{from:?} mistranslated");
        }
        // No two inputs collapse onto the same output.
        let outs: Vec<Activation> = pairs.iter().map(|(f, _)| Activation::from(*f)).collect();
        for i in 0..outs.len() {
            for j in (i + 1)..outs.len() {
                assert_ne!(
                    outs[i], outs[j],
                    "two model activations collapsed onto one compute variant"
                );
            }
        }
    }

    /// `Relu` and `Gelu` must NOT arrive as `Silu`. Stated separately
    /// from the table above because this is the regression, not a
    /// property of the table.
    #[test]
    fn relu_and_gelu_do_not_silently_become_silu() {
        assert_ne!(
            Activation::from(larql_models::Activation::Relu),
            Activation::Silu
        );
        assert_ne!(
            Activation::from(larql_models::Activation::Gelu),
            Activation::Silu
        );
    }

    #[test]
    fn norm_and_ffn_types_cross_faithfully() {
        assert_eq!(
            NormType::from(larql_models::NormType::RmsNorm),
            NormType::RmsNorm
        );
        assert_eq!(
            NormType::from(larql_models::NormType::LayerNorm),
            NormType::LayerNorm
        );
        assert_eq!(FfnType::from(larql_models::FfnType::Gated), FfnType::Gated);
        assert_eq!(
            FfnType::from(larql_models::FfnType::Standard),
            FfnType::Standard
        );
    }
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
