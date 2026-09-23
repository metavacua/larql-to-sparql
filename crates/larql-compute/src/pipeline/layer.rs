use super::enums::{Activation, FfnType, NormType, PositionEncodingType};
use super::moe::{MoeLayerWeights, MoeSpec};
use super::quant_format::QuantWeight;
use super::weights::{
    AttentionSpec, AttentionWeights, FfnSpec, FfnWeights, LayerNorms, LayerWeights, PleSpec,
    RemoteFfnSpec,
};
use super::{RMSNORM_EPSILON_DEFAULT, ROPE_BASE_DEFAULT};

/// Per-layer quantized weights for the full pipeline.
///
/// Carries all architecture-specific behavior per-layer — no model
/// type strings or hardcoded constants in the compute path.
/// Supports Q4_K/Q6_K (Ollama strategy) or Q8_0 (higher precision fallback).
pub struct FullPipelineLayer<'a> {
    // ── Attention weights ──
    pub wq: QuantWeight<'a>,
    pub wk: QuantWeight<'a>,
    pub wv: QuantWeight<'a>,
    pub wo: QuantWeight<'a>,

    // ── FFN weights ──
    /// Gate projection (only used when ffn_type == Gated).
    pub gate: QuantWeight<'a>,
    pub up: QuantWeight<'a>,
    pub down: QuantWeight<'a>,

    // ── Norm weights (f32 vectors, hidden_size elements) ──
    pub input_norm: &'a [f32],
    pub post_attn_norm: &'a [f32],
    pub pre_ffn_norm: Option<&'a [f32]>,
    pub post_ffn_norm: Option<&'a [f32]>,
    /// Norm bias (only for LayerNorm). None for RMSNorm.
    /// Per-query-head attention sinks (GPT-OSS): learned logits that
    /// compete in the attention softmax and are then discarded, so the
    /// emitted weights sum to less than one. `None` for every other
    /// architecture. See `docs/k3-funnel.md` §4.6.
    pub attn_sinks: Option<&'a [f32]>,
    pub input_norm_bias: Option<&'a [f32]>,
    pub post_attn_norm_bias: Option<&'a [f32]>,

    // ── Per-layer architecture parameters ──
    /// Norm weight offset: 0.0 (Llama, Gemma 4), 1.0 (Gemma 2/3).
    pub norm_offset: f32,
    /// QK norm weight offset: 0.0 (Llama, Gemma 4), 1.0 (Gemma 2/3).
    pub qk_norm_offset: f32,
    /// RMSNorm epsilon. Default: 1e-6.
    pub eps: f32,
    /// Whether this model uses post-norms (4 norms per layer: Gemma 2/3/4).
    pub has_post_norms: bool,
    /// Norm type: RMSNorm (default) or LayerNorm (StarCoder2).
    pub norm_type: NormType,
    /// FFN type: Gated (default) or Standard (StarCoder2).
    pub ffn_type: FfnType,
    /// Activation function for the FFN.
    pub activation: Activation,
    /// Attention scale for this layer. Default: 1/sqrt(head_dim).
    /// Gemma 4 (with QK-norm): 1.0.
    pub attn_scale: f32,
    /// Head dimension for this layer. Gemma 4: 256 (sliding) or 512 (global).
    pub head_dim: usize,
    /// Number of Q heads for this layer.
    pub num_q_heads: usize,
    /// Number of KV heads for this layer.
    pub num_kv_heads: usize,
    /// RoPE base frequency for this layer. Gemma 3/4: 10k (sliding) or 1M (global).
    pub rope_base: f32,
    /// Dimensions to apply RoPE to. 0 = full head_dim. Gemma 4 global: head_dim * 0.25.
    pub rotary_dim: usize,
    /// Sliding window size. 0 = full attention (no window).
    pub sliding_window: usize,
    /// Whether to apply parameter-free V-norm (Gemma 4).
    pub has_v_norm: bool,
    /// Per-layer scalar multiplier. 0.0 = disabled (no scaling). Gemma 4: learned scalar.
    pub layer_scalar: f32,
    /// QK-norm weight for Q heads (Gemma 3 / Gemma 4). Length = head_dim.
    /// Applied per-head as RMS-norm before RoPE. `None` means skip QK-norm.
    pub q_norm_weight: Option<&'a [f32]>,
    /// QK-norm weight for K heads. Same shape as `q_norm_weight`.
    pub k_norm_weight: Option<&'a [f32]>,
    /// FFN bias on up projection (StarCoder2). None = no bias.
    pub ffn_up_bias: Option<&'a [f32]>,
    /// FFN bias on down projection (StarCoder2). None = no bias.
    pub ffn_down_bias: Option<&'a [f32]>,

    /// Hybrid MoE block (Gemma 4 26B A4B: dense MLP + expert block, outputs summed).
    /// None for all dense models.
    pub moe: Option<MoeLayerWeights<'a>>,

    /// When true, the local FFN (gate/up/down) is skipped and the FFN
    /// contribution is provided externally via `moe_fn`. Used by
    /// `generate_with_remote_ffn` where ALL FFN goes to a remote server.
    /// Default: false.
    pub ffn_is_remote: bool,

    /// When true, a final RMS norm is applied to the combined (dense + expert)
    /// output before the residual add. Gemma 4 26B A4B: true. Other models:
    /// false (use `layer_scalar` instead).
    pub moe_combined_output_norm: bool,

    /// Outer post-FFN norm weight applied to `(h1 + h2)` before the residual
    /// add. When present and `moe_combined_output_norm` is true, this weight
    /// is used instead of `post_ffn_norm` for the combined norm.
    /// HF Gemma 4: `layers.N.post_feedforward_layernorm.weight` (un-suffixed,
    /// distinct from the `_1` dense-branch norm stored in `post_ffn_norm`).
    /// `None` → fall back to `post_ffn_norm` (legacy behavior).
    pub moe_outer_post_norm: Option<&'a [f32]>,

    // ── Per-Layer Embeddings (Gemma 4 E2B) ──
    /// Per-layer input gate matrix `[ple_dim, hidden]` row-major, f32.
    /// `None` for non-PLE archs.
    pub ple_input_gate: Option<&'a [f32]>,
    /// Per-layer output projection matrix `[hidden, ple_dim]` row-major, f32.
    /// `None` for non-PLE archs.
    pub ple_projection: Option<&'a [f32]>,
    /// Post-PLE RMSNorm weight `[hidden]`, f32. `None` for non-PLE archs.
    pub ple_post_norm: Option<&'a [f32]>,

    /// KV-cache sharing source: when `Some(src)`, this layer reuses K/V from
    /// layer `src`'s cache instead of computing its own. Gemma 4 E2B's last
    /// 20 of 35 layers point to the last non-shared sliding (or global) layer
    /// of the same attention type. `None` for non-shared layers — the
    /// production case for every model except E2B and similar future
    /// KV-shared archs.
    pub kv_shared_source: Option<usize>,

    /// Granite-style residual-stream multiplier. The residual add inside
    /// each transformer block becomes `h += residual_multiplier * x` where
    /// `x` is the attention output (post-W_O) or FFN output (post-down).
    /// Granite 4.1: 0.22 on 3B/8B, 0.175 on 30B (μP scaling). Every other
    /// model: 1.0 (identity — keep the residual add unchanged).
    ///
    /// Plumbed through `encode_post_attn` / `encode_post_ffn` →
    /// `encode_residual_add` → the `residual_add` Metal shader's `b_scale`
    /// binding. The shader treats 1.0 as a no-op so non-Granite paths are
    /// bit-identical to the pre-Granite implementation.
    pub residual_multiplier: f32,
}

impl<'a> FullPipelineLayer<'a> {
    /// Group the layer's quantized attention and FFN weights.
    pub fn weights(&self) -> LayerWeights<'a> {
        LayerWeights {
            attention: AttentionWeights {
                wq: self.wq,
                wk: self.wk,
                wv: self.wv,
                wo: self.wo,
            },
            ffn: FfnWeights {
                gate: self.gate,
                up: self.up,
                down: self.down,
            },
        }
    }

    /// Group the layer's norm weights, biases, and norm scalar behavior.
    pub fn norms(&self) -> LayerNorms<'a> {
        LayerNorms {
            input_norm: self.input_norm,
            post_attn_norm: self.post_attn_norm,
            pre_ffn_norm: self.pre_ffn_norm,
            post_ffn_norm: self.post_ffn_norm,
            input_norm_bias: self.input_norm_bias,
            post_attn_norm_bias: self.post_attn_norm_bias,
            norm_offset: self.norm_offset,
            qk_norm_offset: self.qk_norm_offset,
            eps: self.eps,
            has_post_norms: self.has_post_norms,
            norm_type: self.norm_type,
        }
    }

    /// Return the layer's attention shape, RoPE, and attention-normalization behavior.
    pub fn attention_spec(&self) -> AttentionSpec {
        AttentionSpec {
            attn_scale: self.attn_scale,
            head_dim: self.head_dim,
            num_q_heads: self.num_q_heads,
            num_kv_heads: self.num_kv_heads,
            rope_base: self.rope_base,
            rotary_dim: self.rotary_dim,
            sliding_window: self.sliding_window,
            has_v_norm: self.has_v_norm,
            q_norm_enabled: self.q_norm_weight.is_some(),
            k_norm_enabled: self.k_norm_weight.is_some(),
            position_encoding: PositionEncodingType::RoPE,
        }
    }

    /// Return the layer's dense FFN architecture behavior.
    pub fn ffn_spec(&self) -> FfnSpec {
        FfnSpec {
            ffn_type: self.ffn_type,
            activation: self.activation,
        }
    }

    /// Return the layer's hybrid-MoE behavior.
    pub fn moe_spec(&self) -> MoeSpec<'_, 'a> {
        MoeSpec {
            weights: self.moe.as_ref(),
            combined_output_norm: self.moe_combined_output_norm,
            outer_post_norm: self.moe_outer_post_norm,
        }
    }

    /// Return the layer's remote-FFN dispatch behavior.
    pub fn remote_ffn_spec(&self) -> RemoteFfnSpec {
        RemoteFfnSpec {
            is_remote: self.ffn_is_remote,
        }
    }

    /// Whether this layer uses gated FFN (gate + up → GEGLU → down).
    pub fn is_gated(&self) -> bool {
        self.ffn_type == FfnType::Gated
    }

    /// Whether this layer has a hybrid MoE block alongside the dense FFN.
    /// When true, the forward pass runs both branches and sums their outputs.
    pub fn is_hybrid_moe(&self) -> bool {
        self.moe.is_some()
    }

    /// Per-Layer Embeddings spec for this layer, if active. Returns `None`
    /// unless all three required weights are present (gate, projection, post-norm).
    pub fn ple_spec(&self) -> Option<PleSpec<'a>> {
        Some(PleSpec {
            input_gate: self.ple_input_gate?,
            projection: self.ple_projection?,
            post_norm: self.ple_post_norm?,
        })
    }
}

// ── Defaults ──
//
// `Default` for the leaf types (`QuantWeight`, `FullPipelineLayer`, …) lets
// tests construct minimal instances with `..Default::default()` instead of
// spelling out all 30+ fields. The roadmap's "FullPipelineLayer 63 pub
// fields" cleanup tracks a fuller restructure into LayerWeights /
// LayerNorms / LayerArchParams sub-structs; that's deferred until the
// MoE refactor settles. In the meantime `Default` collapses the test
// boilerplate without rippling through 30 caller files.

impl Default for FullPipelineLayer<'_> {
    fn default() -> Self {
        let qw = QuantWeight::default();
        Self {
            attn_sinks: None,
            wq: qw,
            wk: qw,
            wv: qw,
            wo: qw,
            gate: qw,
            up: qw,
            down: qw,
            input_norm: &[],
            post_attn_norm: &[],
            pre_ffn_norm: None,
            post_ffn_norm: None,
            input_norm_bias: None,
            post_attn_norm_bias: None,
            norm_offset: 0.0,
            qk_norm_offset: 0.0,
            eps: RMSNORM_EPSILON_DEFAULT,
            has_post_norms: false,
            norm_type: NormType::RmsNorm,
            ffn_type: FfnType::Gated,
            activation: Activation::Silu,
            attn_scale: 1.0,
            head_dim: 0,
            num_q_heads: 0,
            num_kv_heads: 0,
            rope_base: ROPE_BASE_DEFAULT,
            rotary_dim: 0,
            sliding_window: 0,
            has_v_norm: false,
            layer_scalar: 0.0,
            q_norm_weight: None,
            k_norm_weight: None,
            ffn_up_bias: None,
            ffn_down_bias: None,
            moe: None,
            moe_combined_output_norm: false,
            moe_outer_post_norm: None,
            ffn_is_remote: false,
            ple_input_gate: None,
            ple_projection: None,
            ple_post_norm: None,
            kv_shared_source: None,
            residual_multiplier: 1.0,
        }
    }
}
