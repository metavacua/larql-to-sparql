use super::enums::{
    Activation, MoeDownPaddingPolicy, MoeExpertScalePolicy, MoeInputSource,
    MoePostExpertNormPolicy, MoeRouterNormPolicy, MoeTopKWeightPolicy,
};
use super::quant_format::QuantFormat;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MoeWeightLayout {
    pub down_padding: MoeDownPaddingPolicy,
}

impl MoeWeightLayout {
    pub const fn unpadded() -> Self {
        Self {
            down_padding: MoeDownPaddingPolicy::None,
        }
    }

    pub const fn quant_block_padded_down() -> Self {
        Self {
            down_padding: MoeDownPaddingPolicy::QuantBlock,
        }
    }

    pub fn down_cols(self, intermediate_size: usize, format: QuantFormat) -> usize {
        match self.down_padding {
            MoeDownPaddingPolicy::None => intermediate_size,
            MoeDownPaddingPolicy::QuantBlock => format
                .packed_block_layout()
                .map(|(block_elems, _)| intermediate_size.div_ceil(block_elems) * block_elems)
                .unwrap_or(intermediate_size),
        }
    }
}

impl Default for MoeWeightLayout {
    fn default() -> Self {
        Self::quant_block_padded_down()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MoeRoutingPolicy {
    pub expert_input: MoeInputSource,
    pub router_input: MoeInputSource,
    pub router_norm: MoeRouterNormPolicy,
    pub selected_weight: MoeTopKWeightPolicy,
    pub expert_scale: MoeExpertScalePolicy,
    pub post_expert_norm: MoePostExpertNormPolicy,
}

impl MoeRoutingPolicy {
    /// Gemma 4 A4B hybrid-MoE behavior validated by local CPU/Metal parity:
    /// route and run experts from the pre-experts-normalized residual, apply
    /// router RMSNorm/scale, renormalize selected top-k probabilities, apply
    /// learned per-expert scales, then post-normalize the expert branch.
    pub const fn gemma4_hybrid() -> Self {
        Self {
            expert_input: MoeInputSource::PreExpertsNorm,
            router_input: MoeInputSource::PreExpertsNorm,
            router_norm: MoeRouterNormPolicy::LearnedOrParameterFree,
            selected_weight: MoeTopKWeightPolicy::RenormalizedSoftmax,
            expert_scale: MoeExpertScalePolicy::PerExpert,
            post_expert_norm: MoePostExpertNormPolicy::RmsNorm,
        }
    }

    /// Conventional sparse-MoE router behavior: route on the provided input,
    /// keep top-k probabilities as softmax weights, and do not apply Gemma 4
    /// branch-specific scales or post norms.
    pub const fn top_k_softmax() -> Self {
        Self {
            expert_input: MoeInputSource::Residual,
            router_input: MoeInputSource::Residual,
            router_norm: MoeRouterNormPolicy::None,
            selected_weight: MoeTopKWeightPolicy::RawSoftmax,
            expert_scale: MoeExpertScalePolicy::None,
            post_expert_norm: MoePostExpertNormPolicy::None,
        }
    }
}

impl Default for MoeRoutingPolicy {
    fn default() -> Self {
        Self::gemma4_hybrid()
    }
}

pub struct MoeLayerWeights<'a> {
    /// Per-expert gate+up weight bytes (`experts_gate_up[e]` is expert `e`'s
    /// gate+up slice). Bytes are interpreted under `expert_data_format`.
    /// Built from `layers/{L}/{e}/gate_up` mmap ranges (per-layer Q4_K) or
    /// from `[num_experts, 2*inter, hidden]` strides (legacy BF16 monolith).
    pub experts_gate_up: Vec<&'a [u8]>,
    /// Per-expert down weight bytes (`experts_down[e]` is expert `e`'s down).
    pub experts_down: Vec<&'a [u8]>,
    /// Explicit routing behavior for this layer/model family.
    pub routing_policy: MoeRoutingPolicy,
    /// Explicit byte layout for expert tensors.
    pub weight_layout: MoeWeightLayout,
    /// Format of the per-expert byte slices. `Q4_K` = per-layer Q4_K files;
    /// `BF16` = legacy monolith. Both flow through the same per-expert tables.
    pub expert_data_format: QuantFormat,
    /// Router linear projection weight [num_experts, hidden_size].
    pub router_proj: &'a [f32],
    /// Router learned input-scale [hidden_size].
    pub router_scale: &'a [f32],
    /// Router per-expert output-scale [num_experts].
    pub router_per_expert_scale: &'a [f32],
    /// Router's own RMS-norm weight applied to the router input before projection.
    /// Empty slice → fall back to parameter-free RMSNorm (if the flag below
    /// is set) or to `pre_experts_norm`.
    pub router_norm: &'a [f32],
    /// Parameter-free router RMSNorm: apply `x / sqrt(mean(x²) + eps)` on
    /// the router input when `router_norm` is empty. HF Gemma 4 sets this
    /// true (`Gemma4RMSNorm(with_scale=False)` — no learned weight on disk).
    pub router_norm_parameter_free: bool,
    /// Scalar multiplier on the router input after the norm and `router_scale`.
    /// HF Gemma 4: `hidden_size^-0.5`. Use `1.0` to disable.
    pub router_input_scalar: f32,
    /// Pre-norm applied to the expert matmuls' input (not the router's). [hidden_size].
    pub pre_experts_norm: &'a [f32],
    /// Post-norm for dense FFN output (replaces plain post_ffn_norm). [hidden_size].
    pub post_ffn1_norm: &'a [f32],
    /// Post-norm for expert block output. [hidden_size].
    pub post_experts_norm: &'a [f32],
    /// Total number of routed experts.
    pub num_experts: usize,
    /// Experts activated per token (top-K).
    pub top_k: usize,
    /// Per-expert intermediate (hidden) dimension.
    pub intermediate_size: usize,
    /// Activation function for expert MLPs. Gemma 4 uses GeluTanh; Mixtral/others use Silu.
    pub activation: Activation,
}

impl MoeLayerWeights<'_> {
    pub fn inter_padded(&self) -> usize {
        self.weight_layout
            .down_cols(self.intermediate_size, self.expert_data_format)
    }
}

/// Hybrid MoE behavior for one layer. The expert tensors remain in
/// [`MoeLayerWeights`]; this view captures how the dense and expert branches
/// are combined.
#[derive(Clone, Copy)]
pub struct MoeSpec<'layer, 'data> {
    pub weights: Option<&'layer MoeLayerWeights<'data>>,
    pub combined_output_norm: bool,
    pub outer_post_norm: Option<&'data [f32]>,
}
