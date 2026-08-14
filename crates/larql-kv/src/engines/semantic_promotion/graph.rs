//! The authority graph (spec §10).
//!
//! The graph is the durable semantic truth of a session; KV pages are
//! materialisations of its nodes. Retirement is recorded here as an edge
//! — `ReplacesFor { operation, scope, qualification }` — never as a
//! property of a record, because "this record replaces that source" is
//! only ever true *for an operation, under a scope, on evidence*.

use std::collections::HashMap;

use super::authority::{SourceAuthority, SourceAuthorityRef};
use super::error::PromotionError;
use super::ids::{AuthorityId, CheckpointId, IdAllocator, OperationId, QualificationId, RecordId};
use super::operation::OperationSpec;
use super::record::SemanticRecord;
use super::scope::RetirementScope;

/// A node in the graph.
#[derive(Clone, Debug, PartialEq)]
pub enum AuthorityNode {
    SourceSpan(SourceAuthority),
    CanonicalRecord(RecordId),
    DerivedClaim(RecordId),
    ReplayCheckpoint(CheckpointId),
}

/// An edge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AuthorityRelation {
    Summarises,
    DerivedFrom,
    Replays,
    ReplacesFor {
        operation: OperationId,
        scope: RetirementScope,
        qualification: QualificationId,
    },
}

/// What is authoritative for one operation right now.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuthorityResolution {
    /// Records the operation will read.
    pub records: Vec<RecordId>,
    /// Sources that must stay visible: nothing has qualified to replace
    /// them for this operation.
    pub required_authorities: Vec<AuthorityId>,
    /// Sources a committed replacement covers for this operation.
    pub replaced_authorities: Vec<AuthorityId>,
}

/// Minimum graph operations (spec §10).
pub trait AuthorityGraph {
    fn add_source(&mut self, source: SourceAuthority) -> Result<AuthorityId, PromotionError>;
    fn add_record(&mut self, record: SemanticRecord) -> Result<RecordId, PromotionError>;
    fn link_provenance(
        &mut self,
        record: RecordId,
        sources: &[AuthorityId],
    ) -> Result<(), PromotionError>;
    fn resolve_for_operation(
        &self,
        operation: &OperationSpec,
    ) -> Result<AuthorityResolution, PromotionError>;
}

/// In-memory graph. One per session.
#[derive(Debug, Default)]
pub struct InMemoryAuthorityGraph {
    ids: IdAllocator,
    sources: HashMap<AuthorityId, SourceAuthority>,
    records: HashMap<RecordId, SemanticRecord>,
    /// `record -> sources it summarises or was derived from`.
    provenance: HashMap<RecordId, Vec<AuthorityId>>,
    /// Committed replacement edges, keyed by the source they cover.
    replacements: HashMap<AuthorityId, Vec<(RecordId, AuthorityRelation)>>,
    session_epoch: u64,
}

impl InMemoryAuthorityGraph {
    pub fn new(session_epoch: u64) -> Self {
        Self {
            session_epoch,
            ..Self::default()
        }
    }

    pub fn session_epoch(&self) -> u64 {
        self.session_epoch
    }

    /// Stable reference to a registered source.
    pub fn source_ref(&self, id: AuthorityId) -> Result<SourceAuthorityRef, PromotionError> {
        let source = self
            .sources
            .get(&id)
            .ok_or(PromotionError::UnknownAuthority(id))?;
        Ok(SourceAuthorityRef::new(id, source))
    }

    pub fn source(&self, id: AuthorityId) -> Result<&SourceAuthority, PromotionError> {
        self.sources
            .get(&id)
            .ok_or(PromotionError::UnknownAuthority(id))
    }

    pub fn record(&self, id: RecordId) -> Result<&SemanticRecord, PromotionError> {
        self.records
            .get(&id)
            .ok_or(PromotionError::UnknownRecord(id))
    }

    /// Mint ids for the promotion protocol. Kept on the graph so a
    /// session has one id source.
    pub fn ids_mut(&mut self) -> &mut IdAllocator {
        &mut self.ids
    }

    /// Record a committed replacement edge.
    pub fn add_replacement(
        &mut self,
        source: AuthorityId,
        record: RecordId,
        relation: AuthorityRelation,
    ) -> Result<(), PromotionError> {
        if !self.sources.contains_key(&source) {
            return Err(PromotionError::UnknownAuthority(source));
        }
        if !self.records.contains_key(&record) {
            return Err(PromotionError::UnknownRecord(record));
        }
        if !matches!(relation, AuthorityRelation::ReplacesFor { .. }) {
            return Err(PromotionError::InvariantViolation(
                "add_replacement requires a ReplacesFor relation",
            ));
        }
        self.replacements
            .entry(source)
            .or_default()
            .push((record, relation));
        Ok(())
    }

    /// Drop replacement edges belonging to `operation` — answer-scoped
    /// expiry (invariant I6).
    pub fn expire_replacements_for(&mut self, operation: OperationId) -> usize {
        let mut removed = 0;
        for edges in self.replacements.values_mut() {
            let before = edges.len();
            edges.retain(|(_, rel)| !replaces_for_operation(rel, operation));
            removed += before - edges.len();
        }
        self.replacements.retain(|_, e| !e.is_empty());
        removed
    }

    /// Whether any committed edge replaces `source` for `operation`.
    fn is_replaced_for(&self, source: AuthorityId, operation: OperationId) -> bool {
        self.replacements.get(&source).is_some_and(|edges| {
            edges
                .iter()
                .any(|(_, r)| replaces_for_operation(r, operation))
        })
    }
}

fn replaces_for_operation(relation: &AuthorityRelation, operation: OperationId) -> bool {
    matches!(
        relation,
        AuthorityRelation::ReplacesFor { operation: op, .. } if *op == operation
    )
}

impl AuthorityGraph for InMemoryAuthorityGraph {
    fn add_source(&mut self, source: SourceAuthority) -> Result<AuthorityId, PromotionError> {
        source
            .validate()
            .map_err(PromotionError::InvariantViolation)?;
        if source.session_epoch != self.session_epoch {
            return Err(PromotionError::StaleAuthorityEpoch);
        }
        let id = self.ids.next_authority();
        self.sources.insert(id, source);
        Ok(id)
    }

    fn add_record(&mut self, record: SemanticRecord) -> Result<RecordId, PromotionError> {
        record
            .validate()
            .map_err(PromotionError::InvariantViolation)?;
        for source in record.provenance() {
            if !self.sources.contains_key(source) {
                return Err(PromotionError::UnknownAuthority(*source));
            }
        }
        let id = self.ids.next_record();
        let provenance = record.provenance().to_vec();
        self.records.insert(id, record);
        self.provenance.insert(id, provenance);
        Ok(id)
    }

    fn link_provenance(
        &mut self,
        record: RecordId,
        sources: &[AuthorityId],
    ) -> Result<(), PromotionError> {
        if !self.records.contains_key(&record) {
            return Err(PromotionError::UnknownRecord(record));
        }
        for s in sources {
            if !self.sources.contains_key(s) {
                return Err(PromotionError::UnknownAuthority(*s));
            }
        }
        let entry = self.provenance.entry(record).or_default();
        for s in sources {
            if !entry.contains(s) {
                entry.push(*s);
            }
        }
        Ok(())
    }

    fn resolve_for_operation(
        &self,
        operation: &OperationSpec,
    ) -> Result<AuthorityResolution, PromotionError> {
        operation
            .validate()
            .map_err(PromotionError::InvariantViolation)?;

        let mut required = Vec::new();
        let mut replaced = Vec::new();
        for record in &operation.operands {
            if !self.records.contains_key(record) {
                return Err(PromotionError::UnknownRecord(*record));
            }
            for source in self
                .provenance
                .get(record)
                .map(Vec::as_slice)
                .unwrap_or(&[])
            {
                // An operation the class cannot authorise retirement for
                // always needs its sources, whatever edges exist.
                let replaced_here = operation.class.can_authorise_retirement()
                    && self.is_replaced_for(*source, operation.id);
                let bucket = if replaced_here {
                    &mut replaced
                } else {
                    &mut required
                };
                if !bucket.contains(source) {
                    bucket.push(*source);
                }
            }
        }
        Ok(AuthorityResolution {
            records: operation.operands.clone(),
            required_authorities: required,
            replaced_authorities: replaced,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engines::semantic_promotion::operation::{
        AnswerBoundary, OperationClass, OperationSignature,
    };
    use crate::engines::semantic_promotion::record::{CanonicalFact, SemanticAtom, SemanticValue};

    const EPOCH: u64 = 1;

    fn graph_with_source() -> (InMemoryAuthorityGraph, AuthorityId) {
        let mut g = InMemoryAuthorityGraph::new(EPOCH);
        let id = g
            .add_source(SourceAuthority::from_tokens(EPOCH, 16_389, &[1, 2, 3, 4]))
            .unwrap();
        (g, id)
    }

    fn fact(provenance: Vec<AuthorityId>) -> SemanticRecord {
        SemanticRecord::CanonicalFact(CanonicalFact {
            id: RecordId::from_counter(0),
            subject: SemanticAtom::new("Meridian"),
            relation: SemanticAtom::new("code"),
            value: SemanticValue::new("integer", "7431"),
            provenance,
            schema_version: 1,
        })
    }

    fn spec(id: OperationId, class: OperationClass, operands: Vec<RecordId>) -> OperationSpec {
        OperationSpec {
            id,
            class,
            operands,
            answer_boundary: AnswerBoundary::ExactPayloadLength(4),
            signature: OperationSignature::new("identity", ["operand"], "integer", 1),
        }
    }

    fn replaces(operation: OperationId) -> AuthorityRelation {
        AuthorityRelation::ReplacesFor {
            operation,
            scope: RetirementScope::AnswerScoped {
                operation,
                through: AnswerBoundary::ExactPayloadLength(4),
            },
            qualification: QualificationId::from_counter(1),
        }
    }

    #[test]
    fn a_source_from_another_epoch_is_refused() {
        let mut g = InMemoryAuthorityGraph::new(EPOCH);
        assert_eq!(
            g.add_source(SourceAuthority::from_tokens(2, 0, &[1]))
                .unwrap_err(),
            PromotionError::StaleAuthorityEpoch
        );
        assert_eq!(g.session_epoch(), EPOCH);
    }

    #[test]
    fn a_record_naming_an_unregistered_source_is_refused() {
        let mut g = InMemoryAuthorityGraph::new(EPOCH);
        assert!(matches!(
            g.add_record(fact(vec![AuthorityId::from_counter(99)])),
            Err(PromotionError::UnknownAuthority(_))
        ));
    }

    #[test]
    fn sources_remain_required_until_a_replacement_is_committed() {
        let (mut g, source) = graph_with_source();
        let record = g.add_record(fact(vec![source])).unwrap();
        let op = OperationId::from_counter(1);

        let before = g
            .resolve_for_operation(&spec(op, OperationClass::Read, vec![record]))
            .unwrap();
        assert_eq!(before.required_authorities, vec![source]);
        assert!(before.replaced_authorities.is_empty());

        g.add_replacement(source, record, replaces(op)).unwrap();
        let after = g
            .resolve_for_operation(&spec(op, OperationClass::Read, vec![record]))
            .unwrap();
        assert!(after.required_authorities.is_empty());
        assert_eq!(after.replaced_authorities, vec![source]);
        assert_eq!(after.records, vec![record]);
    }

    #[test]
    fn a_replacement_for_one_operation_does_not_cover_another() {
        let (mut g, source) = graph_with_source();
        let record = g.add_record(fact(vec![source])).unwrap();
        let op1 = OperationId::from_counter(1);
        let op2 = OperationId::from_counter(2);
        g.add_replacement(source, record, replaces(op1)).unwrap();

        let other = g
            .resolve_for_operation(&spec(op2, OperationClass::Read, vec![record]))
            .unwrap();
        assert_eq!(other.required_authorities, vec![source]);
    }

    #[test]
    fn an_unknown_operation_class_never_retires_a_source() {
        let (mut g, source) = graph_with_source();
        let record = g.add_record(fact(vec![source])).unwrap();
        let op = OperationId::from_counter(1);
        g.add_replacement(source, record, replaces(op)).unwrap();

        let resolved = g
            .resolve_for_operation(&spec(op, OperationClass::Unknown, vec![record]))
            .unwrap();
        assert_eq!(resolved.required_authorities, vec![source]);
        assert!(resolved.replaced_authorities.is_empty());
    }

    #[test]
    fn expiry_drops_only_the_named_operations_edges() {
        let (mut g, source) = graph_with_source();
        let record = g.add_record(fact(vec![source])).unwrap();
        let op1 = OperationId::from_counter(1);
        let op2 = OperationId::from_counter(2);
        g.add_replacement(source, record, replaces(op1)).unwrap();
        g.add_replacement(source, record, replaces(op2)).unwrap();

        assert_eq!(g.expire_replacements_for(op1), 1);
        assert!(g
            .resolve_for_operation(&spec(op1, OperationClass::Read, vec![record]))
            .unwrap()
            .required_authorities
            .contains(&source));
        assert!(g
            .resolve_for_operation(&spec(op2, OperationClass::Read, vec![record]))
            .unwrap()
            .replaced_authorities
            .contains(&source));
    }

    #[test]
    fn replacement_edges_must_name_known_nodes_and_be_replacement_edges() {
        let (mut g, source) = graph_with_source();
        let record = g.add_record(fact(vec![source])).unwrap();
        let op = OperationId::from_counter(1);

        assert!(matches!(
            g.add_replacement(AuthorityId::from_counter(99), record, replaces(op)),
            Err(PromotionError::UnknownAuthority(_))
        ));
        assert!(matches!(
            g.add_replacement(source, RecordId::from_counter(99), replaces(op)),
            Err(PromotionError::UnknownRecord(_))
        ));
        assert!(matches!(
            g.add_replacement(source, record, AuthorityRelation::Summarises),
            Err(PromotionError::InvariantViolation(_))
        ));
    }

    #[test]
    fn provenance_links_are_idempotent_and_validated() {
        let (mut g, source) = graph_with_source();
        let record = g.add_record(fact(vec![source])).unwrap();
        g.link_provenance(record, &[source]).unwrap();
        g.link_provenance(record, &[source]).unwrap();
        let resolved = g
            .resolve_for_operation(&spec(
                OperationId::from_counter(1),
                OperationClass::Read,
                vec![record],
            ))
            .unwrap();
        assert_eq!(resolved.required_authorities, vec![source]);

        assert!(matches!(
            g.link_provenance(RecordId::from_counter(99), &[source]),
            Err(PromotionError::UnknownRecord(_))
        ));
        assert!(matches!(
            g.link_provenance(record, &[AuthorityId::from_counter(99)]),
            Err(PromotionError::UnknownAuthority(_))
        ));
    }

    #[test]
    fn lookups_report_unknown_ids() {
        let (g, source) = graph_with_source();
        assert!(g.source(source).is_ok());
        assert!(g.source_ref(source).is_ok());
        assert!(matches!(
            g.source(AuthorityId::from_counter(99)),
            Err(PromotionError::UnknownAuthority(_))
        ));
        assert!(matches!(
            g.record(RecordId::from_counter(99)),
            Err(PromotionError::UnknownRecord(_))
        ));
    }

    #[test]
    fn resolving_an_operation_with_an_unknown_operand_fails() {
        let (g, _) = graph_with_source();
        assert!(matches!(
            g.resolve_for_operation(&spec(
                OperationId::from_counter(1),
                OperationClass::Read,
                vec![RecordId::from_counter(99)],
            )),
            Err(PromotionError::UnknownRecord(_))
        ));
    }
}
