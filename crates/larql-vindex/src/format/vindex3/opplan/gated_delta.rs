//! Gated DeltaNet: an attention-class operator whose continuation state is
//! a fixed-size recurrence rather than a growing KV cache.
//!
//! Qwen3.8-27B declares 48 of its 64 layers `linear_attention` and the
//! other 16 `full_attention`, on a `full_attention_interval: 4` cadence.
//! The linear layers carry an operand set that shares nothing with
//! [`AttentionOp`](super::AttentionOp) beyond the residual it reads and
//! writes: there is no per-position key/value to retain, no span to mask,
//! and no softmax. What persists between positions is one dense state
//! tensor per layer, whose size does not depend on sequence length.
//!
//! That is the whole reason this is a separate op rather than another
//! [`AttentionSpan`](super::super::graph::policy::AttentionSpan) variant.
//! `AttentionSpan` answers "how far back does this layer's softmax
//! attend", and every consumer reads it that way — a KV planner uses it to
//! decide which positions are architecturally dead. A DeltaNet layer has
//! no answer to that question: nothing it retains is indexed by position
//! at all. Spelling it as a span would hand those consumers a number that
//! looks like liveness information and is not.

use serde::Serialize;

use super::OperandRef;

/// The recurrent state's declared element type, from the checkpoint's
/// `mamba_ssm_dtype`.
///
/// Carried as the declaration rather than resolved to the container's
/// bulk dtype on purpose: Qwen3.8 declares `float32` here against a model
/// whose own default dtype is `bfloat16`. That is the checkpoint stating
/// that the recurrence is precision-sensitive in a way its bulk weights
/// are not — error in a state that feeds itself forward compounds across
/// the whole sequence, where a one-shot weight rounding does not. An
/// executor that quietly ran this state at the bulk dtype would be running
/// a different model, so the declaration is recorded and left for the
/// executor to honour or refuse.
pub use larql_models::inventory::report::RecurrentStateDtype as StateDtype;

/// One layer's Gated DeltaNet operator.
///
/// Every field is transcribed from the container's own operand roles and
/// execution surface. The geometry names follow the checkpoint's
/// declarations (`linear_num_key_heads` and friends) rather than the
/// softmax vocabulary, because the two do not line up: the key and value
/// sides carry *different head counts* here (16 and 48), which no softmax
/// layer in this engine does.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GatedDeltaOp {
    /// `linear_num_key_heads` — heads on the query/key side (16).
    pub num_key_heads: usize,
    /// `linear_num_value_heads` — heads on the value side (48). Larger
    /// than [`Self::num_key_heads`] by design, not a GQA-style sharing
    /// ratio: the value side is the axis the recurrent state is blocked
    /// along, and each value head carries its own decay and write gate.
    pub num_value_heads: usize,
    /// `linear_key_head_dim` (128).
    pub key_head_dim: usize,
    /// `linear_value_head_dim` (128).
    pub value_head_dim: usize,
    /// `linear_conv_kernel_dim` (4) — the depthwise causal convolution
    /// applied across the fused q|k|v channels before the recurrence.
    pub conv_kernel: usize,
    /// `mamba_ssm_dtype`. See [`StateDtype`].
    ///
    /// `None` when the checkpoint declares none, or spells one this build
    /// does not represent — a fact, never a licence to fall back to the
    /// model's bulk dtype.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub state_dtype: Option<StateDtype>,

    /// Fused query|key|value projection, `[2·Hk·Dk + Hv·Dv, hidden]`.
    /// Fused in the checkpoint and left fused here — splitting it would
    /// invent three operands the container does not carry.
    pub in_proj_qkv: OperandRef,
    /// Per-value-head decay projection, `[Hv, hidden]`.
    pub in_proj_a: OperandRef,
    /// Per-value-head write-strength projection, `[Hv, hidden]`.
    pub in_proj_b: OperandRef,
    /// Output-gate projection, `[Hv·Dv, hidden]`.
    pub in_proj_z: OperandRef,
    /// Depthwise causal conv over the fused q|k|v channels,
    /// `[2·Hk·Dk + Hv·Dv, 1, conv_kernel]`.
    pub conv1d: OperandRef,
    /// Per-value-head log decay, `[Hv]`.
    pub a_log: OperandRef,
    /// Per-value-head timestep bias, `[Hv]`.
    pub dt_bias: OperandRef,
    /// Gated RMSNorm weight over one value head's width, `[Dv]`.
    pub norm: OperandRef,
    /// Output projection, `[hidden, Hv·Dv]`.
    pub out_proj: OperandRef,
}

impl GatedDeltaOp {
    /// Channels the fused q|k|v projection emits: query and key at the key
    /// geometry, value at the value geometry.
    ///
    /// Derived rather than stored so it cannot drift from the head counts
    /// beside it — this is the number [`Self::conv1d`] is depthwise over,
    /// and a container whose fused projection disagrees with its own
    /// declared geometry is exactly what closure should catch.
    pub fn qkv_channels(&self) -> usize {
        self.num_key_heads * self.key_head_dim * 2 + self.num_value_heads * self.value_head_dim
    }

    /// Elements in this layer's recurrent state: one `Dk × Dv` matrix per
    /// value head.
    ///
    /// The number that makes this operator a different runtime problem
    /// from softmax attention: it is constant in sequence length. A
    /// planner sizing continuation storage for a DeltaNet layer needs this
    /// once, not per position.
    pub fn state_elements(&self) -> usize {
        self.num_value_heads * self.key_head_dim * self.value_head_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn operand() -> OperandRef {
        OperandRef {
            object: "target.decoder_stack".into(),
            tensor: "tiny".into(),
            dtype: "F32".into(),
            shape: vec![1],
        }
    }

    /// Qwen3.8's own linear-layer geometry (16 key heads, 48 value heads,
    /// 128-wide on both), so the number this pins is the real per-layer
    /// state size a continuation-storage planner would size against —
    /// not an arbitrary tiny fixture.
    #[test]
    fn state_elements_is_value_heads_times_key_dim_times_value_dim() {
        let op = GatedDeltaOp {
            num_key_heads: 16,
            num_value_heads: 48,
            key_head_dim: 128,
            value_head_dim: 128,
            conv_kernel: 4,
            state_dtype: Some(StateDtype::Float32),
            in_proj_qkv: operand(),
            in_proj_a: operand(),
            in_proj_b: operand(),
            in_proj_z: operand(),
            conv1d: operand(),
            a_log: operand(),
            dt_bias: operand(),
            norm: operand(),
            out_proj: operand(),
        };
        // Constant in sequence length, by design — the whole point of
        // this being state_elements() rather than a per-position size.
        assert_eq!(op.state_elements(), 48 * 128 * 128);
    }
}
