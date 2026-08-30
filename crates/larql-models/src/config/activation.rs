//! FFN activation functions and the gated/standard FFN shape.

use serde::{Deserialize, Serialize};

/// Activation function used in the FFN.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Activation {
    /// SiLU / Swish (Gemma, Llama)
    Silu,
    /// GELU (GPT-2, BERT)
    Gelu,
    /// GELU with tanh approximation
    GeluTanh,
    /// ReLU
    Relu,
}

/// HF `hidden_act` / `hidden_activation` spellings, one row per variant.
/// The single definition of the name↔variant mapping — parsers and
/// inventory classification both read this table.
const HF_ACTIVATION_NAMES: &[(&str, Activation)] = &[
    ("silu", Activation::Silu),
    ("swish", Activation::Silu),
    ("gelu", Activation::Gelu),
    ("gelu_new", Activation::GeluTanh),
    ("gelu_pytorch_tanh", Activation::GeluTanh),
    ("relu", Activation::Relu),
];

impl Activation {
    /// Map an HF activation name to a variant. `None` for a spelling this
    /// build has never judged — callers must not guess a default for an
    /// unrecognised name.
    pub fn from_hf_name(name: &str) -> Option<Self> {
        HF_ACTIVATION_NAMES
            .iter()
            .find(|(hf_name, _)| name.eq_ignore_ascii_case(hf_name))
            .map(|&(_, activation)| activation)
    }

    /// Which of the two implemented gate/up FFN kernel families this
    /// activation dispatches to on the CPU walk / kquant paths:
    /// `true` = gelu-tanh, `false` = SiLU.
    ///
    /// This is the ONE definition of that mapping — the 2026-07-30
    /// vindex/walk-FFN review (§4) found it copy-pasted across eight
    /// walk backends, where a new `Activation` variant would silently
    /// land in the SiLU arm. The match is deliberately exhaustive (no
    /// wildcard): adding a variant fails compilation here instead.
    ///
    /// - [`Activation::Gelu`] (exact GELU) is served by the tanh
    ///   approximation — a deliberate, documented approximation on
    ///   these paths (no exact-GELU kernel exists; no in-tree
    ///   architecture currently returns `Gelu`).
    /// - [`Activation::Relu`] has NO gate/up kernel; it panics loudly
    ///   rather than silently computing SiLU numerics. No in-tree
    ///   architecture returns `Relu`.
    pub fn uses_gelu_tanh_gate_up(self) -> bool {
        match self {
            Activation::GeluTanh | Activation::Gelu => true,
            Activation::Silu => false,
            Activation::Relu => panic!(
                "Activation::Relu has no gate/up FFN kernel on the walk/kquant paths \
                 (only gelu-tanh and SiLU are implemented)"
            ),
        }
    }
}

/// Whether the FFN uses a gated architecture.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FfnType {
    /// Gated: SiLU(x @ gate.T) * (x @ up.T) @ down.T (Gemma, Llama)
    Gated,
    /// Standard: activation(x @ up.T) @ down.T (GPT-2)
    Standard,
}
