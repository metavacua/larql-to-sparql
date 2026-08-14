//! Mixture-of-experts storage format and the two policies that decide
//! how an expert block computes: its gate shape and its router normalisation.

/// How expert weights are stored in a MoE model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExpertFormat {
    /// Per-expert separate tensors (Mixtral, DeepSeek).
    /// Keys: `experts.{id}.w1.weight`, `experts.{id}.w2.weight`, etc.
    PerExpert,
    /// Packed MXFP4 (GPT-OSS/OpenAI).
    /// All experts fused into one tensor with block quantization.
    /// Keys: `experts.gate_up_proj_blocks`, `experts.gate_up_proj_scales`, etc.
    PackedMxfp4,
    /// Packed BF16/F16 stacked tensors (Gemma 4 26B A4B).
    /// All experts fused into one tensor per projection, no quantization scales.
    /// Keys: `experts.gate_up_proj` [num_experts, 2*moe_intermediate, hidden],
    ///        `experts.down_proj`   [num_experts, hidden, moe_intermediate].
    PackedBF16,
}

/// How an expert's fused gate/up projection becomes the down projection's
/// input.
///
/// Most MoE models are a plain gated FFN. GPT-OSS is not, and the difference
/// is not cosmetic: it clamps both halves, scales the sigmoid argument, and
/// adds one to the up branch. Modelling that as "SiLU with extra steps" is how
/// a forward pass ends up plausibly wrong — hence an explicit policy rather
/// than an [`Activation`] variant.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExpertGatePolicy {
    /// `activation(gate) * up` — Mixtral, Gemma 4, OLMoE, GraniteMoE.
    Gated,
    /// GPT-OSS's clamped GLU, from `GptOssExperts._apply_gate`:
    ///
    /// ```text
    /// g   = gate.clamp(max = limit)          // upper bound only
    /// u   = up.clamp(-limit, limit)          // symmetric
    /// glu = g * sigmoid(alpha * g)
    /// out = (u + 1) * glu
    /// ```
    ClampedGlu {
        /// Clamp bound (`swiglu_limit`, 7.0 on the released checkpoints).
        limit: f32,
        /// Multiplier on the sigmoid argument (1.702 in the reference).
        alpha: f32,
    },
}

/// How a router's top-k weights are normalised.
///
/// There are only two observable behaviours, and the difference is whether the
/// selected weights sum to 1. Getting it wrong rescales the entire expert
/// branch, which is a large error that still produces coherent-looking output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertRoutingPolicy {
    /// Softmax over **all** experts, then keep the top-k probabilities as they
    /// are. They sum to *less* than 1 — by whatever mass the unselected
    /// experts hold. Mixtral and OLMoE with `norm_topk_prob: false`.
    SoftmaxThenSelect,
    /// The selected weights are normalised to sum to 1.
    ///
    /// Two routes arrive here and they are algebraically identical, which is
    /// why one variant covers both: renormalising the top-k probabilities
    /// (`norm_topk_prob: true`, Gemma 4), or softmaxing over just the selected
    /// logits (GPT-OSS) —
    /// `softmax(l)_i / Σ_{j∈topk} softmax(l)_j = exp(l_i) / Σ_{j∈topk} exp(l_j)`.
    NormalisedOverSelected,
}
