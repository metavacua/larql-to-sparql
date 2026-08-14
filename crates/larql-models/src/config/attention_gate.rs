//! Attention output gating — a generic attention primitive, judged per
//! model from its reference implementation, never inferred from shapes.
//!
//! The spec describes *what generic attention operation the gate operand
//! implements* — source, projection target, nonlinearity, combination and
//! placement — in vocabulary that covers any gated-attention architecture
//! without naming one. The first judged instance (from the upstream
//! Transformers source):
//!
//! ```text
//! gate = sigmoid(gate_proj(attention_input))   # normalized layer input
//! attn_output = attn_output * gate             # after head aggregation
//! attn_output = o_proj(attn_output)            # before output projection
//! ```
//!
//! Every enum here is intentionally single-variant today: each variant
//! set grows only when a new judged instance actually differs, and an
//! unjudged difference must fail closed rather than reuse the nearest
//! variant.

use serde::{Deserialize, Serialize};

/// The fully judged semantics of one attention output gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttentionGateSpec {
    pub source: GateSource,
    pub activation: GateActivation,
    pub combine: GateCombine,
    pub placement: GatePlacement,
}

/// What the gate projection reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GateSource {
    /// The attention block's input hidden state — already normalised,
    /// because the decoder layer norms before calling attention.
    AttentionInput,
}

/// The gate nonlinearity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GateActivation {
    Sigmoid,
}

/// How the gate combines with the attention output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GateCombine {
    ElementwiseMultiply,
}

/// Where the gate applies in the attention pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GatePlacement {
    /// After head outputs are aggregated/flattened, before `o_proj`.
    AfterAggregationBeforeOutputProjection,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_serialises_snake_case_and_round_trips() {
        let spec = AttentionGateSpec {
            source: GateSource::AttentionInput,
            activation: GateActivation::Sigmoid,
            combine: GateCombine::ElementwiseMultiply,
            placement: GatePlacement::AfterAggregationBeforeOutputProjection,
        };
        let json = serde_json::to_string(&spec).unwrap();
        assert!(json.contains("\"sigmoid\""));
        assert!(json.contains("\"attention_input\""));
        let back: AttentionGateSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(back, spec);
    }
}
