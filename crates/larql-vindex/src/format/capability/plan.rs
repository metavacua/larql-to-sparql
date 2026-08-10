//! Operation plans and execution routes (spec §9.2, §10, §11).
//!
//! # Authority belongs to a route, not to an operation
//!
//! An operation can often run more than one way, and the ways need not be
//! equally faithful. WALK may read a direct `gate` region *or* stride the gate
//! half of a fused one; those are different bytes and may carry different
//! fidelities. Attaching one authority to the abstract operation forces a bad
//! choice:
//!
//! - report the stronger one, and kernel binding may later pick the weaker
//!   route, making the report a lie;
//! - report the weaker one, and an exact route is understated;
//! - pick a route during inference, and binding is constrained before it has
//!   seen the kernel registry.
//!
//! So authority is folded per route, and the operation reports every route it
//! admits. The fidelity of an *execution* is the fidelity of the route that
//! gets bound — which has not happened yet at this stage.
//!
//! # Choice groups, not a Cartesian product
//!
//! Thirty layers each admitting two alternatives is 2³⁰ whole-model routes.
//! Enumerating them is not a representation, it is a denial of service. A plan
//! is therefore *fixed requirements plus independent choice groups*: binding
//! picks one alternative per group, and the executed authority is folded over
//! the fixed regions plus whatever it picked.
//!
//! Before binding, the honest statement about such a plan is a **range**, not
//! a value.

use super::authority::{derive_authority, AuthorityInputs, DerivedAuthority, Fidelity};
use super::component::{ComponentCoordinate, SelectedComponent};
use super::coordinate::BankCoordinate;
use crate::format::moe_manifest::programme::RoleAlternative;

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// A region an operation would read, with the fidelity that feeds the fold.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedRegion {
    pub coordinate: super::coordinate::RegionCoordinate,
    pub fidelity: Fidelity,
}

impl PlannedRegion {
    pub fn new(coordinate: super::coordinate::RegionCoordinate, fidelity: Fidelity) -> Self {
        Self {
            coordinate,
            fidelity,
        }
    }

    pub fn as_component(&self) -> SelectedComponent {
        SelectedComponent::region(self.coordinate.clone(), self.fidelity)
    }
}

/// One way to satisfy a bank within a plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QualifiedAlternative {
    /// Bank-region layout, empty for a component choice.
    pub alternative: RoleAlternative,
    pub regions: Vec<PlannedRegion>,
    /// Manifest tensors, for a choice between component candidates such as a
    /// dedicated versus tied LM head.
    pub components: Vec<SelectedComponent>,
}

impl QualifiedAlternative {
    /// Weakest fidelity across everything this alternative reads.
    pub fn weakest_fidelity(&self) -> Option<Fidelity> {
        self.regions
            .iter()
            .map(|r| r.fidelity)
            .chain(self.components.iter().map(|c| c.fidelity()))
            .min()
    }
}

/// A bank whose satisfying alternative binding will choose.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanChoice {
    pub bank: BankCoordinate,
    /// Every alternative that works, in declaration order. Never collapsed —
    /// the kernel registry may support one at a higher maturity than another.
    pub alternatives: Vec<QualifiedAlternative>,
}

impl PlanChoice {
    /// The alternative whose weakest region is strongest — the best fidelity
    /// binding *could* achieve here.
    ///
    /// **Introspection and planning only.** This must never be used as a
    /// binding tie-break: when two alternatives carry equal authority, kernel
    /// binding has to stay free to choose on support and performance grounds.
    /// An authority-oriented tie-break here would silently decide a question
    /// that belongs to the kernel registry, and would do so on a criterion
    /// that cannot distinguish the candidates anyway.
    pub fn best_alternative(&self) -> Option<&QualifiedAlternative> {
        self.alternatives
            .iter()
            .max_by_key(|a| a.weakest_fidelity().unwrap_or(Fidelity::AnalysisOnly))
    }

    /// The alternative whose weakest region is weakest — the floor binding
    /// could land on.
    pub fn worst_alternative(&self) -> Option<&QualifiedAlternative> {
        self.alternatives
            .iter()
            .min_by_key(|a| a.weakest_fidelity().unwrap_or(Fidelity::AnalysisOnly))
    }
}

/// What an operation would read, as fixed regions plus open choices.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct OperationPlan {
    /// Components every execution of this plan reads, whatever binding
    /// chooses. Bank regions *and* manifest-addressed tensors — a decode path
    /// needs embeddings, norms, routers and transforms, none of which are
    /// bank regions.
    pub fixed_components: Vec<SelectedComponent>,
    /// Independent choice groups. Empty for a plan with a single shape.
    pub choices: Vec<PlanChoice>,
}

impl OperationPlan {
    /// A plan with no open choices.
    pub fn fixed(components: Vec<SelectedComponent>) -> Self {
        Self {
            fixed_components: components,
            choices: Vec::new(),
        }
    }

    /// A plan whose fixed side is bank regions only — the WALK shape.
    pub fn fixed_regions(regions: Vec<PlannedRegion>) -> Self {
        Self::fixed(regions.iter().map(PlannedRegion::as_component).collect())
    }

    pub fn is_empty(&self) -> bool {
        self.fixed_components.is_empty() && self.choices.is_empty()
    }

    /// Whether binding still has a decision to make.
    pub fn has_open_choices(&self) -> bool {
        self.choices.iter().any(|c| c.alternatives.len() > 1)
    }

    /// Every coordinate this plan could read, for diagnostics.
    pub fn all_coordinates(&self) -> Vec<ComponentCoordinate> {
        let mut out: Vec<ComponentCoordinate> = self
            .fixed_components
            .iter()
            .map(|c| c.coordinate())
            .collect();
        for choice in &self.choices {
            for alt in &choice.alternatives {
                out.extend(
                    alt.regions
                        .iter()
                        .map(|r| ComponentCoordinate::BankRegion(r.coordinate.clone())),
                );
            }
        }
        out.sort();
        out.dedup();
        out
    }

    fn fold(
        &self,
        pick: impl Fn(&PlanChoice) -> Option<&QualifiedAlternative>,
    ) -> DerivedAuthority {
        let mut fidelities: Vec<Fidelity> =
            self.fixed_components.iter().map(|c| c.fidelity()).collect();
        for choice in &self.choices {
            if let Some(alt) = pick(choice) {
                fidelities.extend(alt.regions.iter().map(|r| r.fidelity));
                fidelities.extend(alt.components.iter().map(|c| c.fidelity()));
            }
        }
        DerivedAuthority::of(&AuthorityInputs {
            selected_fidelities: fidelities,
            execution_complete: true,
            structural: None,
        })
    }

    /// The strongest authority any binding of this plan could achieve.
    ///
    /// **Achievable, not achieved.** No route has been bound, so this is a
    /// ceiling — quoting it as the fidelity of an execution would be a claim
    /// about a decision that has not been made.
    pub fn best_achievable_authority(&self) -> DerivedAuthority {
        self.fold(PlanChoice::best_alternative)
    }

    /// The weakest authority any binding of this plan could land on.
    pub fn worst_achievable_authority(&self) -> DerivedAuthority {
        self.fold(PlanChoice::worst_alternative)
    }
}

/// One way to run an operation, with the fidelity that route would carry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QualifiedOperationRoute {
    pub plan: OperationPlan,
}

impl QualifiedOperationRoute {
    pub fn new(plan: OperationPlan) -> Self {
        Self { plan }
    }

    /// Ceiling for this route. Equals the floor when the plan has no open
    /// choices, which is the common single-shape case.
    pub fn best_achievable_authority(&self) -> DerivedAuthority {
        self.plan.best_achievable_authority()
    }

    pub fn worst_achievable_authority(&self) -> DerivedAuthority {
        self.plan.worst_achievable_authority()
    }

    /// Whether this route's fidelity is already settled — no open choices, so
    /// ceiling and floor coincide and binding cannot move it.
    pub fn authority_is_settled(&self) -> bool {
        self.best_achievable_authority().level == self.worst_achievable_authority().level
    }
}

/// Convenience: fold a bare fidelity list, used where a route has no choices.
pub fn authority_of(fidelities: &[Fidelity]) -> Fidelity {
    derive_authority(&AuthorityInputs {
        selected_fidelities: fidelities.to_vec(),
        execution_complete: true,
        structural: None,
    })
}
