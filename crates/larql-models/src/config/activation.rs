//! FFN activation functions and the gated/standard FFN shape.

/// Activation function used in the FFN.
#[derive(Debug, Clone, Copy, PartialEq)]
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

impl Activation {
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
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FfnType {
    /// Gated: SiLU(x @ gate.T) * (x @ up.T) @ down.T (Gemma, Llama)
    Gated,
    /// Standard: activation(x @ up.T) @ down.T (GPT-2)
    Standard,
}
