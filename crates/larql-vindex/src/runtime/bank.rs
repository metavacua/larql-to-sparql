//! A bound expert bank — the population an operation can route into.
//!
//! The bank owns the expert list and the shapes they share. Selection happens
//! upstream in the router; the bank's job is to hand back the expert a
//! selected id names, and to refuse when the id has no expert behind it.
//!
//! That refusal matters more than it looks. A router and a bank that disagree
//! about the population is a binding fault, and the natural coding of it —
//! skip the expert, carry on — produces a token that is quietly missing a
//! fraction of its FFN contribution and looks entirely reasonable.

use crate::format::capability::coordinate::BankCoordinate;
use larql_compute::Activation;

use super::error::ExecutionError;
use super::expert_kernel::ExpertKernel;
use super::projection::BoundProjection;
use super::tensor::BoundTensor;

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// One expert: its gated projection and its down projection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundExpert<'a> {
    /// Id within the bank's population, as the router names it.
    pub expert_id: u32,
    pub projection: BoundProjection<'a>,
    /// `[hidden, intermediate]` — contracts the intermediate axis away.
    pub down: BoundTensor<'a>,
}

impl BoundExpert<'_> {
    pub fn validate(&self, intermediate: usize, hidden: usize) -> Result<(), ExecutionError> {
        self.projection.validate(intermediate, hidden)?;
        self.down.require_matrix(hidden, intermediate)
    }
}

/// A population of experts sharing one shape and one activation.
///
/// `PartialEq` only: `Activation` is not `Eq`, which is the correct choice for
/// a type that names float behaviour rather than a discrete tag.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundBankOperation<'a> {
    /// Which bank this is. Diagnostics only — execution never branches on it.
    pub bank: BankCoordinate,
    pub experts: Vec<BoundExpert<'a>>,
    /// Intermediate width of every expert in the bank.
    pub intermediate_dim: usize,
    /// Input/output width — the residual width for a direct bank, the latent
    /// width for a bank sitting behind routed-input/output transforms.
    pub hidden_dim: usize,
    pub activation: Activation,
    /// Which kernel runs this bank's experts.
    ///
    /// A bank property rather than an expert one, because the kernel's
    /// per-token state is shared across the bank: the incumbent quantises the
    /// bank input to Q8_K once and every selected expert reads that one
    /// activation. Two experts of one bank on two kernels would not be a
    /// binding this runtime can express, and that is the correct restriction.
    pub kernel: ExpertKernel,
}

impl<'a> BoundBankOperation<'a> {
    pub fn population(&self) -> usize {
        self.experts.len()
    }

    /// The expert an id names, or a refusal naming where it was looked for.
    ///
    /// Deliberately not `Option`. A selected-but-absent expert is never a
    /// condition to handle locally: skipping it, or renormalising the
    /// surviving weights around it, produces a token quietly missing part of
    /// its FFN contribution and looks entirely reasonable. The request has to
    /// reach the caller intact so that placement can satisfy it.
    ///
    /// `addressable` is the router's population, which is what makes the
    /// report actionable — "expert 90 of 128, and this bank holds 8" says
    /// *fetch it*, where "expert 90, bank holds 8" reads like corruption.
    pub fn expert(
        &self,
        expert_id: u32,
        addressable: usize,
    ) -> Result<&BoundExpert<'a>, ExecutionError> {
        self.experts
            .iter()
            .find(|e| e.expert_id == expert_id)
            .ok_or_else(|| ExecutionError::SelectedExpertNotResident {
                expert: expert_id,
                bank: self.bank.describe(),
                resident: self.population(),
                population: addressable,
            })
    }

    /// Whether this bank holds the expert an id names.
    pub fn holds(&self, expert_id: u32) -> bool {
        self.experts.iter().any(|e| e.expert_id == expert_id)
    }

    /// Check every expert against the bank's declared shape, and against what
    /// the bound kernel can take.
    ///
    /// Both, in that order. The shape check is what the operation means; the
    /// kernel check is what this particular implementation of it can be handed.
    /// A correctly-shaped expert the bound kernel cannot read is a binding
    /// fault, and finding it here rather than mid-token is the whole reason
    /// the kernel is a bound property.
    pub fn validate(&self) -> Result<(), ExecutionError> {
        for expert in &self.experts {
            expert.validate(self.intermediate_dim, self.hidden_dim)?;
            self.kernel.validate_operands(
                expert,
                self.intermediate_dim,
                self.hidden_dim,
                self.activation,
            )?;
        }
        Ok(())
    }

    pub fn describe(&self) -> String {
        format!(
            "{} — {} experts, {}×{}, {} kernel",
            self.bank.describe(),
            self.population(),
            self.intermediate_dim,
            self.hidden_dim,
            self.kernel.name()
        )
    }
}
