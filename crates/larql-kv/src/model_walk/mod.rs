//! The model-walk layer: **the graph decides which semantic path is
//! valid; the KV engine materialises that path.**
//!
//! A peer of [`crate::engines`], not a part of one. The dependency runs
//! one way — walk → engine — so the semantic-promotion engine cannot
//! invent, extend or reinterpret a walk (gate W8). Nothing under
//! `engines/` references this module.
//!
//! ```text
//! ModelGraph          what relationships exist
//!     ↓ WalkPlanner   choose one executable path
//! ModelWalkPlan       a typed, immutable step list
//!     ↓ WalkValidator whole-plan legality, once, up front
//! ValidatedWalkPlan   the only thing an executor accepts
//!     ↓ executor      physical steps against SemanticPromotionEngine
//! WalkTrace           what was visited, proposed and emitted
//! ```
//!
//! Phase A is observe-only: the executor begins the operation, plans the
//! promotion, checks the qualification against the live regime, and
//! records what *would* happen. It hides nothing. The same validated
//! walk is what `EnforceResidentRollback` will later execute for real,
//! which is what makes "the enforced run performs exactly the actions
//! the observe run predicted" a testable claim.
//!
//! There is no search here yet. The planner enumerates operands, keeps
//! those with a self-consistent certificate for the operation, and ranks
//! by the widest boundary the evidence covers. Cost models and A* wait
//! until alternative access paths exist to choose between.

pub mod cost;
pub mod enforce;
pub mod executor;
pub mod graph;
pub mod plan;
pub mod planner;
pub mod validate;

#[cfg(test)]
mod tests;

pub use cost::{
    pareto_frontier, rank, CostError, CostEstimate, CostObservation, QualifiedWalkCandidate,
    RankingPolicy, ResidencyState, WalkCost,
};
pub use enforce::{EnforceWalkExecutor, WalkTokens};
pub use executor::{
    AuthoritySnapshot, ModelWalkExecutor, ObserveWalkExecutor, PhysicalEffect, WalkAction,
    WalkExecutionError, WalkTrace,
};
pub use graph::{render_boundary, ModelEdge, ModelGraph, ModelNode, Relation};
pub use plan::{ExpertPrefetchHint, ModelWalkPlan, WalkPlanId, WalkStep};
pub use planner::{WalkPlanError, WalkPlanner, WalkStrategy};
pub use validate::{DefaultWalkValidator, ValidatedWalkPlan, WalkValidationError, WalkValidator};
