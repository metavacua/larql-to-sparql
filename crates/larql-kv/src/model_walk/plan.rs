//! The typed executable plan.
//!
//! A plan is a sequence of [`WalkStep`]s chosen by the planner and
//! frozen. The KV engine executes physical steps; it never invents,
//! extends or reinterprets one (gate W8). `expert_hints` is the seam to
//! the weight side and stays empty for this milestone.

use crate::semantic_promotion::{
    AnswerBoundary, AuthorityId, OperationId, QualificationId, RecordId, RetirementScope,
};

use super::graph::render_boundary;

/// Deterministic plan identity: same operation and same primary record
/// give the same id, so two planning runs are comparable (gate W0).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct WalkPlanId(u128);

/// Bit width of each packed field in a [`WalkPlanId`].
const WALK_ID_FIELD_BITS: u32 = 32;

impl WalkPlanId {
    pub fn derived(operation: OperationId, record: RecordId) -> Self {
        let field = |raw: u128| raw & ((1u128 << WALK_ID_FIELD_BITS) - 1);
        Self((field(operation.raw()) << WALK_ID_FIELD_BITS) | field(record.raw()))
    }

    pub fn raw(self) -> u128 {
        self.0
    }
}

/// A prefetch hint for the weight side. Carries stable owner references,
/// never raw expert indices — see spec §5.1.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExpertPrefetchHint {
    /// Stable bank references. Typed as opaque bytes here so the walk
    /// layer does not pull `larql-vindex` into its dependency set before
    /// Phase H decides the shape.
    pub owners: Vec<String>,
    pub predicted_ttl_tokens: u32,
}

/// One step of a walk.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WalkStep {
    ResolveAuthority {
        authority: AuthorityId,
    },
    SelectRecord {
        record: RecordId,
    },
    BeginOperation {
        operation: OperationId,
    },
    PreparePromotion {
        source: AuthorityId,
        record: RecordId,
    },
    QualifyPromotion {
        qualification: QualificationId,
        requested_scope: RetirementScope,
    },
    CommitPromotion,
    Generate {
        boundary: AnswerBoundary,
    },
    FinishOperation,
    RestoreAuthority,
    ReplayAuthority {
        authority: AuthorityId,
    },
}

impl WalkStep {
    /// Whether executing this step would change what attention reads.
    /// Observe mode records these rather than performing them.
    pub fn mutates_visibility(self) -> bool {
        matches!(
            self,
            Self::PreparePromotion { .. }
                | Self::CommitPromotion
                | Self::RestoreAuthority
                | Self::ReplayAuthority { .. }
        )
    }

    /// Stable one-line rendering. The W0 fingerprint is the
    /// concatenation of these, so it must not embed anything that
    /// varies between runs.
    pub fn render(self) -> String {
        match self {
            Self::ResolveAuthority { authority } => {
                format!("resolve_authority({})", authority.raw())
            }
            Self::SelectRecord { record } => format!("select_record({})", record.raw()),
            Self::BeginOperation { operation } => format!("begin_operation({})", operation.raw()),
            Self::PreparePromotion { source, record } => {
                format!("prepare_promotion({},{})", source.raw(), record.raw())
            }
            Self::QualifyPromotion {
                qualification,
                requested_scope,
            } => format!(
                "qualify_promotion({},{})",
                qualification.raw(),
                render_scope(requested_scope)
            ),
            Self::CommitPromotion => "commit_promotion".into(),
            Self::Generate { boundary } => format!("generate({})", render_boundary(boundary)),
            Self::FinishOperation => "finish_operation".into(),
            Self::RestoreAuthority => "restore_authority".into(),
            Self::ReplayAuthority { authority } => format!("replay_authority({})", authority.raw()),
        }
    }
}

fn render_scope(scope: RetirementScope) -> String {
    match scope {
        RetirementScope::AnswerScoped { operation, through } => {
            format!(
                "answer_scoped({},{})",
                operation.raw(),
                render_boundary(through)
            )
        }
        RetirementScope::GeneralState { certificate } => {
            format!("general_state({})", certificate.raw())
        }
    }
}

/// An immutable plan. Only a validated one reaches an executor.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelWalkPlan {
    pub id: WalkPlanId,
    pub operation: OperationId,
    pub steps: Vec<WalkStep>,
    /// Taken when the primary path fails validation or execution.
    pub fallback: Option<Box<ModelWalkPlan>>,
    pub expert_hints: Vec<ExpertPrefetchHint>,
}

impl ModelWalkPlan {
    /// Canonical rendering used by the determinism gate. Includes the
    /// fallback chain, since two plans that differ only in fallback are
    /// different plans.
    pub fn fingerprint(&self) -> String {
        let mut out = format!("walk({}) op({})", self.id.raw(), self.operation.raw());
        for step in &self.steps {
            out.push('\n');
            out.push_str(&step.render());
        }
        if let Some(fallback) = &self.fallback {
            out.push_str("\nfallback:\n");
            out.push_str(&fallback.fingerprint());
        }
        out
    }

    /// The boundary this walk generates through, if it generates.
    pub fn generated_boundary(&self) -> Option<AnswerBoundary> {
        self.steps.iter().find_map(|s| match s {
            WalkStep::Generate { boundary } => Some(*boundary),
            _ => None,
        })
    }

    /// The scope this walk requests, if it promotes.
    pub fn requested_scope(&self) -> Option<RetirementScope> {
        self.steps.iter().find_map(|s| match s {
            WalkStep::QualifyPromotion {
                requested_scope, ..
            } => Some(*requested_scope),
            _ => None,
        })
    }

    /// The record this walk promotes, if it promotes.
    pub fn promoted_record(&self) -> Option<RecordId> {
        self.steps.iter().find_map(|s| match s {
            WalkStep::PreparePromotion { record, .. } => Some(*record),
            _ => None,
        })
    }

    /// The source this walk retires, if it retires one.
    pub fn retired_source(&self) -> Option<AuthorityId> {
        self.steps.iter().find_map(|s| match s {
            WalkStep::PreparePromotion { source, .. } => Some(*source),
            _ => None,
        })
    }

    /// Tokens this walk's exclusion removes, if it promotes.
    pub fn exclusion_token_count(&self) -> u64 {
        self.steps
            .iter()
            .find_map(|s| match s {
                WalkStep::PreparePromotion { .. } => Some(1),
                _ => None,
            })
            .unwrap_or(0)
    }

    /// Whether the walk has a route back to source authority — a
    /// restore or a replay step.
    pub fn has_recovery_step(&self) -> bool {
        self.steps.iter().any(|s| {
            matches!(
                s,
                WalkStep::RestoreAuthority | WalkStep::ReplayAuthority { .. }
            )
        })
    }
}
