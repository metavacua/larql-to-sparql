//! What an execution is given.
//!
//! # Gemma forced this
//!
//! The first draft passed one vector to both the router and the experts,
//! because fixture A and every synthetic case share one input. Gemma does not:
//! `moe_router_input` applies a router-specific RMS norm, a learned
//! element-wise scale and a scalar multiplier on top of the vector the experts
//! receive. Routing therefore scores a *different vector* from the one the
//! selected experts consume.
//!
//! ```text
//! residual
//!   → pre_experts_norm ─────────────→ expert input
//!                       └→ router_norm × router_scale × scalar → router input
//! ```
//!
//! A single-input model cannot express that. It would have had to score on the
//! expert input, which selects different experts — a wrong answer with no
//! shape error anywhere, exactly the class of fault this programme keeps
//! finding.
//!
//! # Why both are supplied rather than derived
//!
//! Norms, scales and scalars belong to the surrounding block, not to the MoE
//! operation — the same reason `execute` returns a delta instead of an updated
//! residual. Deriving the router input here would mean re-modelling the
//! incumbent's routing-policy enums inside VINDEX3, giving two descriptions of
//! one behaviour and somewhere for them to disagree.
//!
//! So the block computes both vectors, as it already does, and hands them
//! over. Models where the two coincide use [`MoeInputs::shared`].

/// The vectors one MoE execution reads.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MoeInputs<'a> {
    /// The vector the experts consume. Routed-input transforms apply to this.
    pub bank: &'a [f32],
    /// The vector the router scores, in the space the router expects.
    ///
    /// Equal to `bank` whenever a model routes on the same vector its experts
    /// see. Gemma does not; Mixtral-style models do.
    pub router: &'a [f32],
}

impl<'a> MoeInputs<'a> {
    /// One vector for both — the common case, and fixture A's.
    pub fn shared(input: &'a [f32]) -> Self {
        Self {
            bank: input,
            router: input,
        }
    }

    /// Distinct vectors, as Gemma's hybrid MoE requires.
    pub fn split(bank: &'a [f32], router: &'a [f32]) -> Self {
        Self { bank, router }
    }

    /// Whether routing and expert compute read the same vector.
    ///
    /// Not a pointer comparison: two separately-computed vectors that happen to
    /// be equal *are* shared for every purpose that matters here, and a model
    /// whose router norm is the identity produces exactly that.
    pub fn is_shared(&self) -> bool {
        self.bank == self.router
    }

    pub fn describe(&self) -> String {
        if self.is_shared() {
            format!("shared input, width {}", self.bank.len())
        } else {
            format!(
                "split inputs, bank width {} / router width {}",
                self.bank.len(),
                self.router.len()
            )
        }
    }
}
