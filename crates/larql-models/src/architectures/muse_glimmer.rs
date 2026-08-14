//! Muse-Glimmer target architecture (`muse_glimmer` / `muse_glimmer_text`).
//!
//! Every topology and scalar fact resolves through the config-driven
//! trait defaults — the checkpoint declares them and the generic parser
//! already consumes them (hybrid `layer_types`, per-layer
//! `layer_rope_theta` with NoPE zeros, `qk_scale_factor`,
//! `output_multiplier`, `post_norm_eps`). What this family adds is only
//! what no config key or tensor can state: the **judged execution
//! semantics** read from the upstream reference implementation.
//!
//! - Attention output gate: `sigmoid(gate_proj(attention_input))`
//!   multiplied into the aggregated head output before `o_proj`. The
//!   gate projection is `Linear(hidden → q_heads × head_dim)`, matching
//!   the `[q_heads × head_dim, hidden]` operand the stack ships.
//! - Parameter-free QK norm: Q and K are RMS-normalised with **no
//!   learned weights** (the stack ships no `q_norm`/`k_norm` tensors),
//!   with the declared `qk_scale_factor` applied to normalised Q — the
//!   canonical `head_dim^-0.5` stays a separate score-time multiply,
//!   which is why resolution records query and score scales separately.
//!
//! The assistant (`muse_glimmer_assistant`) is deliberately **not** this
//! family: its layers carry weighted QK norms and no gate, and nobody
//! has judged assistant-specific semantics — it stays on the generic
//! path, which its operand estate closes under.

use crate::config::{
    AttentionGateSpec, GateActivation, GateCombine, GatePlacement, GateSource, ModelArchitecture,
    ModelConfig, ParameterFreeQkNorm,
};

pub struct MuseGlimmerArch {
    config: ModelConfig,
}

impl MuseGlimmerArch {
    pub fn from_config(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArchitecture for MuseGlimmerArch {
    fn family(&self) -> &str {
        "muse_glimmer"
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn attention_output_gate(&self) -> Option<AttentionGateSpec> {
        Some(AttentionGateSpec {
            source: GateSource::AttentionInput,
            activation: GateActivation::Sigmoid,
            combine: GateCombine::ElementwiseMultiply,
            placement: GatePlacement::AfterAggregationBeforeOutputProjection,
        })
    }

    fn parameter_free_qk_norm(&self) -> ParameterFreeQkNorm {
        ParameterFreeQkNorm { q: true, k: true }
    }
}
