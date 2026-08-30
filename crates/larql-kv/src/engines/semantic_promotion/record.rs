//! Typed semantic records (spec §7).
//!
//! Two kinds, and the distinction is causal rather than cosmetic. A
//! [`CanonicalFact`] restates what the source said. A [`DerivedClaim`]
//! states the *answer to an operation over* what the source said — it is
//! discharged by construction, so the model has nothing left to compute.
//!
//! Measured (EXP-25, 65K distance, forced-reference scoring): the derived
//! form carried `transform_inc` on the full reachable trajectory where the
//! canonical form carried only its payload — and the same derived form
//! *failed outright* on `transform_rev`, answering `7431` where `1347` was
//! required. Rendering a result in explicit prose therefore buys nothing
//! by itself; §9's certificate is what licenses its use, per operation.
//!
//! **`discharged` is a property of the record, not of how the model reads
//! it.** EXP-25CF's paired arms (4K, both histories) found the discharged
//! result being treated as a fresh *operand*: injecting "the code plus one
//! is 7432" produced 7433, and "the code reversed is 1347" produced 7431 —
//! the operation ran again, identically in both the source-conditioned and
//! the counterfactual history. So the type below states an intent the model
//! may decline to honour, and that is exactly why nothing here grants a
//! derived record capability on the strength of its own flag. Only a
//! certificate for the specific operation does.
//!
//! Vocabulary is caller-supplied throughout. There are no built-in
//! relations, subjects or value types in this module — the engine
//! compares atoms, it never interprets them.

use serde::{Deserialize, Serialize};

use super::ids::{AuthorityId, RecordId};
use super::operation::OperationSignature;

/// An opaque semantic token — a subject or a relation name.
#[derive(Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct SemanticAtom(pub String);

impl SemanticAtom {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// A record's value, with the caller's own type name attached.
///
/// `type_name` is compared, never parsed: it exists so a certificate
/// keyed on one result type cannot be reused for another.
#[derive(Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct SemanticValue {
    pub type_name: String,
    pub rendered: String,
}

impl SemanticValue {
    pub fn new(type_name: impl Into<String>, rendered: impl Into<String>) -> Self {
        Self {
            type_name: type_name.into(),
            rendered: rendered.into(),
        }
    }
}

/// A restatement of source content: subject–relation–value.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CanonicalFact {
    pub id: RecordId,
    pub subject: SemanticAtom,
    pub relation: SemanticAtom,
    pub value: SemanticValue,
    /// Source authorities this fact restates. Never empty — a fact with
    /// no provenance cannot be followed back on replay (invariant I7).
    pub provenance: Vec<AuthorityId>,
    pub schema_version: u32,
}

/// An assertion that an operation has already been computed.
///
/// **What it claims, not what the model does.** `declared_discharged`
/// says the *producer* computed the result; whether the model consumes
/// it as an answer is a separate, empirical question answered by a
/// certificate's [`ConsumptionMode`](super::qualification::ConsumptionMode).
/// The wire form still rejects `discharged: false`, because a record that
/// does not even claim to be discharged is not this type.
#[derive(Clone, Debug, Eq, PartialEq, Deserialize)]
#[serde(try_from = "DerivedClaimWire")]
pub struct DerivedClaim {
    pub id: RecordId,
    /// The operation this result discharges — the same structural type
    /// an [`OperationSpec`](super::operation::OperationSpec) carries, so
    /// "does this record answer that operation" is a comparison rather
    /// than an interpretation.
    pub operation: OperationSignature,
    pub operands: Vec<RecordId>,
    pub result: SemanticValue,
    pub provenance: Vec<AuthorityId>,
    pub schema_version: u32,
}

impl DerivedClaim {
    /// What the record *asserts*. Always `true` — a claim that does not
    /// claim discharge is not this type.
    ///
    /// Deliberately not named `is_discharged`: that read as a statement
    /// about the interaction, and EXP-25CF showed the model re-applying
    /// the operation to exactly such a record in both histories. Only
    /// `ConsumptionMode::AnswerWithoutReapplication` on a certificate
    /// licenses bypassing the computation.
    pub fn declares_discharged(&self) -> bool {
        true
    }
}

/// Manual, so the wire form always carries `discharged: true`.
///
/// A derived `Serialize` would omit the flag, and the deserialiser
/// defaults it to `false` and rejects — a struct that cannot round-trip
/// through its own encoding.
impl Serialize for DerivedClaim {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut st = s.serialize_struct("DerivedClaim", 7)?;
        st.serialize_field("id", &self.id)?;
        st.serialize_field("operation", &self.operation)?;
        st.serialize_field("operands", &self.operands)?;
        st.serialize_field("result", &self.result)?;
        st.serialize_field("provenance", &self.provenance)?;
        st.serialize_field("schema_version", &self.schema_version)?;
        st.serialize_field("discharged", &true)?;
        st.end()
    }
}

/// On-wire shape of a [`DerivedClaim`], carrying the explicit
/// `discharged` flag described in spec §7.
#[derive(Clone, Debug, Deserialize)]
pub struct DerivedClaimWire {
    pub id: RecordId,
    pub operation: OperationSignature,
    pub operands: Vec<RecordId>,
    pub result: SemanticValue,
    pub provenance: Vec<AuthorityId>,
    pub schema_version: u32,
    #[serde(default)]
    pub discharged: bool,
}

impl TryFrom<DerivedClaimWire> for DerivedClaim {
    type Error = &'static str;

    fn try_from(w: DerivedClaimWire) -> Result<Self, Self::Error> {
        if !w.discharged {
            return Err("derived claim with discharged=false is not representable");
        }
        Ok(Self {
            id: w.id,
            operation: w.operation,
            operands: w.operands,
            result: w.result,
            provenance: w.provenance,
            schema_version: w.schema_version,
        })
    }
}

/// Either kind of record.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum SemanticRecord {
    CanonicalFact(CanonicalFact),
    DerivedClaim(DerivedClaim),
}

impl SemanticRecord {
    pub fn id(&self) -> RecordId {
        match self {
            Self::CanonicalFact(f) => f.id,
            Self::DerivedClaim(d) => d.id,
        }
    }

    pub fn provenance(&self) -> &[AuthorityId] {
        match self {
            Self::CanonicalFact(f) => &f.provenance,
            Self::DerivedClaim(d) => &d.provenance,
        }
    }

    pub fn schema_version(&self) -> u32 {
        match self {
            Self::CanonicalFact(f) => f.schema_version,
            Self::DerivedClaim(d) => d.schema_version,
        }
    }

    /// The operation this record *claims* to have computed, if any.
    ///
    /// `None` for a canonical fact. Naming matters: this is the record's
    /// assertion, and a certificate is what turns it into a licence.
    pub fn claims_computed(&self) -> Option<&OperationSignature> {
        match self {
            Self::CanonicalFact(_) => None,
            Self::DerivedClaim(d) => Some(&d.operation),
        }
    }

    /// Digest of the record's exact content.
    ///
    /// What a certificate is keyed on. Excludes the record's own id, which
    /// is session-local — two sessions registering the same fact must
    /// produce the same digest, or a certificate could never transfer.
    pub fn content_digest(&self) -> [u8; 32] {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(b"larql.semantic_promotion.record.v1");
        match self {
            Self::CanonicalFact(f) => {
                h.update(b"canonical");
                h.update(f.subject.as_str().as_bytes());
                h.update([0]);
                h.update(f.relation.as_str().as_bytes());
                h.update([0]);
                h.update(f.value.type_name.as_bytes());
                h.update([0]);
                h.update(f.value.rendered.as_bytes());
            }
            Self::DerivedClaim(d) => {
                h.update(b"derived");
                h.update(d.operation.operator.as_bytes());
                h.update([0]);
                h.update(d.operation.result_type.as_bytes());
                h.update([0]);
                h.update(d.result.type_name.as_bytes());
                h.update([0]);
                h.update(d.result.rendered.as_bytes());
            }
        }
        h.update(self.schema_version().to_le_bytes());
        h.finalize().into()
    }

    /// Structural self-check run at registration.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.provenance().is_empty() {
            return Err("record has no provenance; it cannot be followed back on replay");
        }
        if self.provenance().iter().any(|a| !a.is_well_formed()) {
            return Err("record provenance contains a malformed authority id");
        }
        if !self.id().is_well_formed() {
            return Err("record carries a malformed record id");
        }
        if let Self::DerivedClaim(d) = self {
            if d.operands.len() != d.operation.arity() {
                return Err("derived result operand count does not match its operation signature");
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn authority() -> AuthorityId {
        AuthorityId::from_counter(1)
    }

    fn fact() -> CanonicalFact {
        CanonicalFact {
            id: RecordId::from_counter(1),
            subject: SemanticAtom::new("Meridian"),
            relation: SemanticAtom::new("code"),
            value: SemanticValue::new("integer", "7431"),
            provenance: vec![authority()],
            schema_version: 1,
        }
    }

    fn derived() -> DerivedClaim {
        DerivedClaim {
            id: RecordId::from_counter(2),
            operation: OperationSignature::new("increment", ["operand"], "integer", 1),
            operands: vec![RecordId::from_counter(1)],
            result: SemanticValue::new("integer", "7432"),
            provenance: vec![authority()],
            schema_version: 1,
        }
    }

    #[test]
    fn canonical_fact_discharges_nothing() {
        assert!(SemanticRecord::CanonicalFact(fact())
            .claims_computed()
            .is_none());
    }

    #[test]
    fn a_derived_claim_names_the_operation_it_claims_to_have_computed() {
        let rec = SemanticRecord::DerivedClaim(derived());
        assert_eq!(
            rec.claims_computed().map(|s| s.operator.as_str()),
            Some("increment")
        );
        assert!(derived().declares_discharged());
    }

    /// Re-encode `derived()` and edit one field, so these gates cannot
    /// drift out of step with the real wire format.
    fn wire_json(edit: impl FnOnce(&mut serde_json::Map<String, serde_json::Value>)) -> String {
        let mut value = serde_json::to_value(derived()).unwrap();
        edit(value.as_object_mut().unwrap());
        value.to_string()
    }

    #[test]
    fn a_derived_claim_round_trips_through_its_own_wire_form() {
        // The encoder must emit `discharged: true`, or the struct cannot
        // read back what it just wrote.
        let json = serde_json::to_string(&derived()).unwrap();
        assert!(json.contains("\"discharged\":true"), "{json}");
        assert_eq!(
            serde_json::from_str::<DerivedClaim>(&json).unwrap(),
            derived()
        );
    }

    #[test]
    fn wire_form_rejects_undischarged_derived_claims() {
        let json = wire_json(|m| {
            m.insert("discharged".into(), serde_json::Value::Bool(false));
        });
        let err = serde_json::from_str::<DerivedClaim>(&json).unwrap_err();
        assert!(err.to_string().contains("discharged=false"), "{err}");
    }

    #[test]
    fn wire_form_defaults_to_undischarged_and_is_rejected() {
        // A producer that simply omits the flag must not be read as a
        // discharged result.
        let json = wire_json(|m| {
            m.remove("discharged");
        });
        assert!(serde_json::from_str::<DerivedClaim>(&json).is_err());
    }

    #[test]
    fn validate_rejects_a_record_with_no_provenance() {
        let mut f = fact();
        f.provenance.clear();
        assert!(SemanticRecord::CanonicalFact(f).validate().is_err());
    }

    #[test]
    fn validate_rejects_derived_operand_arity_mismatch() {
        let mut d = derived();
        d.operands.clear();
        assert!(SemanticRecord::DerivedClaim(d).validate().is_err());
    }

    #[test]
    fn validate_accepts_well_formed_records() {
        assert!(SemanticRecord::CanonicalFact(fact()).validate().is_ok());
        assert!(SemanticRecord::DerivedClaim(derived()).validate().is_ok());
    }

    #[test]
    fn accessors_agree_across_both_variants() {
        let f = SemanticRecord::CanonicalFact(fact());
        let d = SemanticRecord::DerivedClaim(derived());
        assert_eq!(f.id(), RecordId::from_counter(1));
        assert_eq!(d.id(), RecordId::from_counter(2));
        assert_eq!(f.provenance(), &[authority()]);
        assert_eq!(d.provenance(), &[authority()]);
        assert_eq!(f.schema_version(), 1);
        assert_eq!(d.schema_version(), 1);
    }

    /// The digest is content-derived and **id-independent**: two sessions
    /// registering the same fact must key the same certificate, and record
    /// ids are session-local.
    #[test]
    fn the_content_digest_ignores_the_session_local_id() {
        let mut a = fact();
        let mut b = fact();
        a.id = RecordId::from_counter(1);
        b.id = RecordId::from_counter(999);
        assert_eq!(
            SemanticRecord::CanonicalFact(a).content_digest(),
            SemanticRecord::CanonicalFact(b).content_digest()
        );
    }

    #[test]
    fn the_content_digest_separates_different_values_and_kinds() {
        let base = SemanticRecord::CanonicalFact(fact());
        let mut other = fact();
        other.value = SemanticValue::new("integer", "5824");
        assert_ne!(
            base.content_digest(),
            SemanticRecord::CanonicalFact(other).content_digest(),
            "a different value must not share a certificate"
        );
        assert_ne!(
            base.content_digest(),
            SemanticRecord::DerivedClaim(derived()).content_digest(),
            "a canonical fact and a derived claim are different records"
        );
    }

    #[test]
    fn atom_and_value_constructors_round_trip() {
        assert_eq!(SemanticAtom::new("x").as_str(), "x");
        let v = SemanticValue::new("integer", "7431");
        assert_eq!(v.type_name, "integer");
        assert_eq!(v.rendered, "7431");
    }
}
