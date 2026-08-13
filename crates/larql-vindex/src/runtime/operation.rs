//! The bound MoE operation — the object the token path executes.
//!
//! # What is deliberately absent
//!
//! No manifest, no profile, no capability traversal, no authority fold, no
//! variant candidates, no kernel registry. Every one of those questions was
//! answered during binding, and none of them is reachable from here.
//!
//! That is not tidiness. Resolution work that stays reachable from the
//! execution object gets called from the execution path eventually — usually
//! as a cache lookup that looks harmless — and then per-token cost depends on
//! catalogue size. Making the object incapable of resolution is what keeps the
//! load path complex and the token path boring.
//!
//! # Shape
//!
//! ```text
//! router       scores the population and selects
//! transforms   optional latent projections either side of the banks
//! banks        the expert populations
//! reduction    how selected outputs combine
//! ```

use super::axis::Axis;
use super::bank::BoundBankOperation;
use super::error::ExecutionError;
use super::reduction::BoundReduction;
use super::router::BoundRouter;
use super::transform::{BoundTransform, TransformStage};

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// An immutable, fully-bound routed MoE operation for one layer.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundMoeOperation<'a> {
    pub router: BoundRouter<'a>,
    /// Latent projections. Empty for a direct MoE, where the bank reads and
    /// writes the residual width directly.
    pub transforms: Vec<BoundTransform<'a>>,
    /// Expert populations the router selects into.
    ///
    /// A Vec because K3-shaped layers pair a routed bank with a shared one.
    /// Today every bank here is routed by `router` and their reduced outputs
    /// sum; unrouted shared banks arrive with the Mini-K3 rung.
    pub banks: Vec<BoundBankOperation<'a>>,
    pub reduction: BoundReduction,
    /// Residual width — what this operation reads and what it returns.
    pub residual_dim: usize,
}

impl BoundMoeOperation<'_> {
    /// Width the banks operate at: the latent width when a routed-input
    /// transform is bound, the residual width otherwise.
    pub fn bank_input_dim(&self) -> usize {
        // The *last* routed-input transform, since a stage may chain several
        // and it is the final one that fixes the width the banks see.
        self.transforms
            .iter()
            .rfind(|t| t.stage == TransformStage::RoutedInput)
            .map_or(self.residual_dim, |t| t.out_dim())
    }

    /// Whether this operation projects into a separate routed space.
    pub fn is_latent(&self) -> bool {
        !self.transforms.is_empty()
    }

    /// Check every operand against the shape the others imply.
    ///
    /// Belongs to the load path. Binding calls it once; execution does not,
    /// because re-validating per token is exactly the resolution creep this
    /// object exists to prevent.
    ///
    /// # A bank need not hold the whole population
    ///
    /// An earlier version required `router.population() == bank.population()`.
    /// That is wrong, and `bank.rs` already said so: a **sharded** bank holds a
    /// subset, and expert 40 may be the first one a shard carries. Requiring
    /// equality would have refused every sharded operation — including the
    /// expert-server slices this format exists to serve.
    ///
    /// So the router is validated against its own addressable population, and
    /// the bank is checked to lie inside it. An expert id the router cannot
    /// address is the genuine fault, because nothing would ever select it.
    pub fn validate(&self) -> Result<(), ExecutionError> {
        let bank_input = self.bank_input_dim();
        self.router.validate(self.router.population(), bank_input)?;
        for bank in &self.banks {
            if bank.hidden_dim != bank_input {
                return Err(ExecutionError::DimensionMismatch {
                    operand: bank.describe(),
                    axis: Axis::InputWidth,
                    expected: bank_input,
                    found: bank.hidden_dim,
                });
            }
            bank.validate()?;
            for expert in &bank.experts {
                if expert.expert_id as usize >= self.router.population() {
                    return Err(ExecutionError::ExpertOutOfRange {
                        expert: expert.expert_id,
                        population: self.router.population(),
                    });
                }
            }
        }
        Ok(())
    }

    /// Whether the bound banks hold the router's whole addressable population.
    ///
    /// False for a shard. Distinct from validity: a shard is a legitimate
    /// operation, it simply cannot serve every routing outcome, and a caller
    /// that needs completeness must ask rather than assume.
    pub fn holds_full_population(&self) -> bool {
        self.banks
            .iter()
            .any(|b| b.population() == self.router.population())
    }

    pub fn describe(&self) -> String {
        let banks: Vec<String> = self
            .banks
            .iter()
            .map(BoundBankOperation::describe)
            .collect();
        format!(
            "{} → [{}] → {} ({})",
            self.router.describe(),
            banks.join("; "),
            self.reduction.name(),
            if self.is_latent() { "latent" } else { "direct" }
        )
    }
}
