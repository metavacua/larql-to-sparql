//! The relationship graph a walk is planned over.
//!
//! The graph stores what is *available*: which records assert which
//! sources, which records are operands of which operation, which
//! certificates exist. It does not choose. A [`ModelWalkPlan`](super::plan::ModelWalkPlan)
//! is one executable path through it.
//!
//! Deliberately distinct from the engine's own
//! [`InMemoryAuthorityGraph`](crate::semantic_promotion::graph::InMemoryAuthorityGraph):
//! that one is the engine's registration bookkeeping, this one is the
//! planner's view. Keeping them separate is what stops the KV engine
//! from becoming the planner.

use crate::collections::{BTreeMap, BTreeSet};

use crate::semantic_promotion::{
    AnswerBoundary, AuthorityId, CheckpointId, OperationId, OperationSpec,
    QualificationCertificate, QualificationId, RecordId, RetirementScope,
};

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// A node. Ordered so iteration — and therefore planning — is
/// deterministic (gate W0).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum ModelNode {
    SourceAuthority(AuthorityId),
    CanonicalFact(RecordId),
    DerivedClaim(RecordId),
    Operation(OperationId),
    AnswerBoundary(AnswerBoundary),
    ReplayCheckpoint(CheckpointId),
}

impl ModelNode {
    /// The record this node names, if it is a record node.
    pub fn record(self) -> Option<RecordId> {
        match self {
            Self::CanonicalFact(r) | Self::DerivedClaim(r) => Some(r),
            _ => None,
        }
    }

    /// The authority this node names, if it is a source node.
    pub fn authority(self) -> Option<AuthorityId> {
        match self {
            Self::SourceAuthority(a) => Some(a),
            _ => None,
        }
    }

    pub fn is_derived_claim(self) -> bool {
        matches!(self, Self::DerivedClaim(_))
    }

    /// Stable rendering, used by the walk trace and the W0 fingerprint.
    pub fn render(self) -> String {
        match self {
            Self::SourceAuthority(a) => format!("source({})", a.raw()),
            Self::CanonicalFact(r) => format!("canonical({})", r.raw()),
            Self::DerivedClaim(r) => format!("derived({})", r.raw()),
            Self::Operation(o) => format!("operation({})", o.raw()),
            Self::AnswerBoundary(b) => format!("boundary({})", render_boundary(b)),
            Self::ReplayCheckpoint(c) => format!("checkpoint({})", c.raw()),
        }
    }
}

/// Stable rendering of a boundary, shared by nodes, steps and traces.
pub fn render_boundary(boundary: AnswerBoundary) -> String {
    match boundary {
        AnswerBoundary::FirstTermination => "first_termination".into(),
        AnswerBoundary::ExactPayloadLength(n) => format!("payload({n})"),
        AnswerBoundary::ExternalCommit => "external_commit".into(),
    }
}

/// An edge.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum ModelEdge {
    /// A canonical fact asserts what a source span said.
    Asserts,
    /// A derived result was computed from a record or source.
    DerivedFrom,
    /// A record is an operand of an operation.
    OperandOf,
    /// An operation produces an answer boundary.
    Produces,
    /// A record replaces a source, for one operation, under a scope.
    ReplacesFor {
        qualification: QualificationId,
        scope: RetirementScope,
    },
    /// A derived result discharges an operation.
    Discharges,
    /// A source can be rebuilt from a checkpoint.
    ReplaysFrom,
}

impl ModelEdge {
    pub fn render(self) -> &'static str {
        match self {
            Self::Asserts => "asserts",
            Self::DerivedFrom => "derived_from",
            Self::OperandOf => "operand_of",
            Self::Produces => "produces",
            Self::ReplacesFor { .. } => "replaces_for",
            Self::Discharges => "discharges",
            Self::ReplaysFrom => "replays_from",
        }
    }
}

/// A directed, labelled edge.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Relation {
    pub from: ModelNode,
    pub edge: ModelEdge,
    pub to: ModelNode,
}

/// The planner's view of a session.
#[derive(Debug, Default, Clone)]
pub struct ModelGraph {
    nodes: BTreeSet<ModelNode>,
    /// Ordered, so planning over the same inputs is reproducible.
    relations: BTreeSet<Relation>,
    certificates: BTreeMap<QualificationId, QualificationCertificate>,
    /// Operation specs, so the planner can read a signature and the
    /// executor can open the operation without the caller threading it
    /// through separately.
    operations: BTreeMap<OperationId, OperationSpec>,
    /// How many tokens each record renders to. The planner needs it to
    /// cost promoting the record — a replacement that is not compact is
    /// not a saving.
    record_tokens: BTreeMap<RecordId, u64>,
    /// Span length per source authority, in tokens. What an exclusion
    /// actually removes — distinct from the answer's payload length.
    source_spans: BTreeMap<AuthorityId, u64>,
}

impl ModelGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_node(&mut self, node: ModelNode) -> &mut Self {
        self.nodes.insert(node);
        self
    }

    /// Add a relation, inserting both endpoints.
    pub fn relate(&mut self, from: ModelNode, edge: ModelEdge, to: ModelNode) -> &mut Self {
        self.nodes.insert(from);
        self.nodes.insert(to);
        self.relations.insert(Relation { from, edge, to });
        self
    }

    /// Register an operation, inserting its node.
    pub fn add_operation(&mut self, spec: OperationSpec) -> &mut Self {
        self.nodes.insert(ModelNode::Operation(spec.id));
        self.operations.insert(spec.id, spec);
        self
    }

    pub fn operation_spec(&self, operation: OperationId) -> Option<&OperationSpec> {
        self.operations.get(&operation)
    }

    /// Declare a record's rendered token count.
    pub fn set_record_tokens(&mut self, record: RecordId, tokens: u64) -> &mut Self {
        self.record_tokens.insert(record, tokens);
        self
    }

    /// A record's rendered token count, if declared.
    pub fn record_tokens(&self, record: RecordId) -> Option<u64> {
        self.record_tokens.get(&record).copied()
    }

    /// Logical length of a source span, read off its `SourceSpan` node's
    /// declared extent.
    pub fn source_span_tokens(&self, authority: AuthorityId) -> Option<u64> {
        self.source_spans.get(&authority).copied()
    }

    /// Declare a source authority's span length in tokens.
    pub fn set_source_span_tokens(&mut self, authority: AuthorityId, tokens: u64) -> &mut Self {
        self.source_spans.insert(authority, tokens);
        self
    }

    pub fn add_certificate(&mut self, certificate: QualificationCertificate) -> &mut Self {
        self.certificates.insert(certificate.id, certificate);
        self
    }

    pub fn contains(&self, node: ModelNode) -> bool {
        self.nodes.contains(&node)
    }

    pub fn has_relation(&self, from: ModelNode, edge: ModelEdge, to: ModelNode) -> bool {
        self.relations.contains(&Relation { from, edge, to })
    }

    pub fn certificate(&self, id: QualificationId) -> Option<&QualificationCertificate> {
        self.certificates.get(&id)
    }

    /// Every certificate id, in deterministic order.
    pub fn certificate_ids(&self) -> Vec<QualificationId> {
        self.certificates.keys().copied().collect()
    }

    pub fn nodes(&self) -> impl Iterator<Item = ModelNode> + '_ {
        self.nodes.iter().copied()
    }

    pub fn relations(&self) -> impl Iterator<Item = Relation> + '_ {
        self.relations.iter().copied()
    }

    /// Records that are operands of `operation`, in deterministic order.
    pub fn operands_of(&self, operation: OperationId) -> Vec<ModelNode> {
        let target = ModelNode::Operation(operation);
        self.relations
            .iter()
            .filter(|r| r.edge == ModelEdge::OperandOf && r.to == target)
            .map(|r| r.from)
            .collect()
    }

    /// Source authorities a record reaches through `Asserts` or
    /// `DerivedFrom`, transitively through intermediate records.
    pub fn provenance_of(&self, record: ModelNode) -> Vec<AuthorityId> {
        let mut seen = BTreeSet::new();
        let mut found = BTreeSet::new();
        let mut frontier = vec![record];
        while let Some(node) = frontier.pop() {
            if !seen.insert(node) {
                continue;
            }
            for r in self.relations.iter().filter(|r| {
                r.from == node && matches!(r.edge, ModelEdge::Asserts | ModelEdge::DerivedFrom)
            }) {
                match r.to.authority() {
                    Some(a) => {
                        found.insert(a);
                    }
                    // An intermediate record — keep walking back.
                    None => frontier.push(r.to),
                }
            }
        }
        found.into_iter().collect()
    }

    /// Whether `authority` has a durable route back.
    pub fn replay_route_for(&self, authority: AuthorityId) -> Option<CheckpointId> {
        let from = ModelNode::SourceAuthority(authority);
        self.relations
            .iter()
            .find(|r| r.from == from && r.edge == ModelEdge::ReplaysFrom)
            .and_then(|r| match r.to {
                ModelNode::ReplayCheckpoint(c) => Some(c),
                _ => None,
            })
    }

    /// Certificates that apply to `record`, in deterministic id order.
    pub fn certificates_for(&self, record: RecordId) -> Vec<&QualificationCertificate> {
        self.certificates
            .values()
            .filter(|c| c.record == record)
            .collect()
    }
}
