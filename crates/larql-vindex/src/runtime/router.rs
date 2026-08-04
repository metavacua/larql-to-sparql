//! The bound router — scores, selects and weights.
//!
//! The user-facing sketch of a bound operation carried the router as a bare
//! tensor. A tensor alone cannot express the decision, though: top-k depth,
//! whether selected probabilities are renormalised, and whether learned
//! per-expert scales apply are all part of *what routing means* for a given
//! model, and all three change which experts run and how much each
//! contributes.
//!
//! They are bound properties rather than execution-time inference, and they
//! reuse `larql-compute`'s policy vocabulary so that the reference path and
//! the incumbent path are describing the same thing in the same words.

use larql_compute::{MoeExpertScalePolicy, MoeTopKWeightPolicy};

use super::error::ExecutionError;
use super::tensor::BoundTensor;

/// One expert selected for a token, with the weight it contributes at.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SelectedExpert {
    pub expert_id: u32,
    /// Weight after every policy has been applied — what the reduction uses.
    pub weight: f32,
    /// Probability before renormalisation and per-expert scaling. Kept because
    /// a routing disagreement shows up here first, while the final weight can
    /// still coincide after renormalisation hides it.
    pub raw_score: f32,
}

/// Which kernel computes the router's scores.
///
/// A **bound** choice, not an inferred one — the same discipline as every
/// other decision in this object. Binding the incumbent kernel is what makes
/// score parity a statement about the production code path; reimplementing a
/// BLAS-shaped loop here and calling the result "parity" would prove only that
/// two similar loops agree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RouterKernel {
    /// Index-order f32 accumulation. Plain, allocation-light, and the oracle
    /// every other kernel is checked against.
    #[default]
    Reference,
    /// `larql-compute`'s own scoring: BLAS `sgemv` via ndarray, then its
    /// softmax. Requires the weight to be contiguous row-major f32.
    Incumbent,
}

impl RouterKernel {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Reference => "reference",
            Self::Incumbent => "incumbent",
        }
    }
}

/// Per-expert output scaling, with its operand.
///
/// One field, not two. The previous shape — a policy enum beside an
/// `Option<BoundTensor>` — permitted two invalid states:
///
/// ```text
/// PerExpert + None        policy demands a scale that was never bound
/// None + Some(scales)     an operand nothing will read
/// ```
///
/// The first is not hypothetical. Gemma's routing policy is `PerExpert`, the
/// Gemma harness bound `None`, and execution silently declined to scale —
/// producing bit-identical router *scores* and normalised weights 7e-4 apart.
/// The parity ladder localised it, but the type had allowed the operation to
/// be built at all.
///
/// Coupling them makes both states unrepresentable rather than merely
/// rejected, which is the difference between a check someone can forget to
/// call and a mistake that does not compile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BoundExpertScaling<'a> {
    /// Selected weights are used as the routing policy produced them.
    None,
    /// Each selected weight is multiplied by `scales[expert_id]`.
    ///
    /// Applied *after* renormalisation, so the selected weights need not sum
    /// to one afterwards. Before it, renormalisation would divide the learned
    /// scale straight back out.
    PerExpert { scales: BoundTensor<'a> },
}

impl BoundExpertScaling<'_> {
    /// The scale operand, if one is bound.
    pub fn scales(&self) -> Option<&BoundTensor<'_>> {
        match self {
            Self::None => None,
            Self::PerExpert { scales } => Some(scales),
        }
    }

    /// The equivalent incumbent policy, for reporting against a
    /// `MoeLayerWeights`.
    pub const fn policy(&self) -> MoeExpertScalePolicy {
        match self {
            Self::None => MoeExpertScalePolicy::None,
            Self::PerExpert { .. } => MoeExpertScalePolicy::PerExpert,
        }
    }

    pub const fn name(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::PerExpert { .. } => "per_expert",
        }
    }
}

/// Scoring and selection for one bank.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundRouter<'a> {
    /// `[num_experts, hidden]`.
    pub weight: BoundTensor<'a>,
    pub top_k: usize,
    pub selected_weight: MoeTopKWeightPolicy,
    /// Per-expert output scaling, together with the operand it needs.
    pub scaling: BoundExpertScaling<'a>,
    /// Which kernel scores. Defaults to the reference.
    pub kernel: RouterKernel,
}

impl BoundRouter<'_> {
    /// Experts this router can address.
    pub fn population(&self) -> usize {
        self.weight.rows()
    }

    /// Input width the router contracts over.
    pub fn hidden_dim(&self) -> usize {
        self.weight.cols()
    }

    /// Check the router's operands against the population it addresses.
    ///
    /// `population` is the router's own **addressable** population, never a
    /// resident shard's. The per-expert scale is routing semantics: expert 90's
    /// learned scale is part of what routing means whether or not expert 90 is
    /// resident here, so a shard must carry the whole vector. Truncating it to
    /// the resident subset would make two shards of one model weight the same
    /// expert differently.
    pub fn validate(&self, population: usize, hidden: usize) -> Result<(), ExecutionError> {
        self.weight.require_matrix(population, hidden)?;
        if let Some(scales) = self.scaling.scales() {
            scales.require_vector(population)?;
            // A non-finite scale multiplies a valid routing weight into a NaN
            // that then propagates through the reduction into the residual
            // stream. Caught here, at bind time, rather than as a mysterious
            // token later.
            let values = scales.to_vec()?;
            if let Some((expert, value)) = values
                .iter()
                .position(|v| !v.is_finite())
                .map(|i| (i, values[i]))
            {
                return Err(ExecutionError::NonFiniteExpertScale {
                    expert,
                    value,
                    operand: scales.describe(),
                });
            }
        }
        Ok(())
    }

    pub fn describe(&self) -> String {
        format!(
            "router {} — top-{} of {}",
            self.weight.describe(),
            self.top_k,
            self.population()
        )
    }
}
