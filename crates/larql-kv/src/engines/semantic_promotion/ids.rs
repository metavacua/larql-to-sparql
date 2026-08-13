//! Stable logical identities (spec §6, invariant I5).
//!
//! Every object the promotion protocol names gets an opaque `u128`
//! identity that is **never** a physical KV offset, cache slot or vector
//! index. Physical storage is free to compact, evict or grow; a stored
//! [`AuthorityId`] must still resolve afterwards.
//!
//! Ids are minted by [`IdAllocator`], a deterministic per-domain counter.
//! Determinism is deliberate: the parity and promotion gates compare two
//! engine runs token-for-token, and a random id would make an otherwise
//! bit-exact trace differ in its `Debug` output. Uniqueness is required
//! only within one engine instance.

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// Bit position at which the domain tag is stored in a minted id.
///
/// The low 96 bits hold the per-domain counter, the high 32 bits the
/// domain tag. That keeps ids from different domains disjoint, so a
/// `RecordId` transmuted into an `AuthorityId` by a caller bug fails
/// lookup instead of silently aliasing a real authority.
const DOMAIN_TAG_SHIFT: u32 = 96;

/// Domain tags. Distinct non-zero values; 0 is reserved so that a
/// default-constructed / zeroed id is never a valid minted id.
const TAG_AUTHORITY: u128 = 1;
const TAG_RECORD: u128 = 2;
const TAG_OPERATION: u128 = 3;
const TAG_PROMOTION: u128 = 4;
const TAG_QUALIFICATION: u128 = 5;
const TAG_CHECKPOINT: u128 = 6;

macro_rules! stable_id {
    ($(#[$meta:meta])* $name:ident, $tag:expr) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
        pub struct $name(u128);

        // Ids are 128-bit and JSON has no 128-bit integer, so the wire
        // form is a decimal string. Serialising as a bare number would
        // round-trip only for ids below `u64::MAX` — which, with the
        // domain tag in the high bits, is none of them.
        impl serde::Serialize for $name {
            fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
                s.serialize_str(&self.0.to_string())
            }
        }

        impl<'de> serde::Deserialize<'de> for $name {
            fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
                let raw = String::deserialize(d)?;
                raw.parse::<u128>().map(Self).map_err(serde::de::Error::custom)
            }
        }

        impl $name {
            /// The domain tag every id of this type carries.
            pub const DOMAIN_TAG: u128 = $tag;

            /// Build from a raw counter value. Test/deserialisation
            /// helper — production code mints through [`IdAllocator`].
            pub fn from_counter(counter: u128) -> Self {
                debug_assert!(
                    counter < (1u128 << DOMAIN_TAG_SHIFT),
                    "counter overflows the 96-bit id counter field"
                );
                Self((Self::DOMAIN_TAG << DOMAIN_TAG_SHIFT) | counter)
            }

            /// The raw `u128`, for serialisation only.
            pub fn raw(self) -> u128 {
                self.0
            }

            /// `true` when the raw value carries this domain's tag.
            /// Used by the graph to reject cross-domain lookups.
            pub fn is_well_formed(self) -> bool {
                (self.0 >> DOMAIN_TAG_SHIFT) == Self::DOMAIN_TAG
            }
        }
    };
}

stable_id!(
    /// A source authority: a span of context that carries original
    /// truth. Stable across compaction (spec §6).
    AuthorityId,
    TAG_AUTHORITY
);
stable_id!(
    /// A typed semantic record — a [`CanonicalFact`](super::record::CanonicalFact)
    /// or [`DerivedClaim`](super::record::DerivedClaim).
    RecordId,
    TAG_RECORD
);
stable_id!(
    /// An unresolved operation. Retirement is always relative to one.
    OperationId,
    TAG_OPERATION
);
stable_id!(
    /// One promotion attempt through the plan → prepare → qualify →
    /// commit lifecycle.
    PromotionId,
    TAG_PROMOTION
);
stable_id!(
    /// A qualification decision (certificate or shadow comparison).
    QualificationId,
    TAG_QUALIFICATION
);
stable_id!(
    /// A replay checkpoint from which a source span can be rebuilt.
    CheckpointId,
    TAG_CHECKPOINT
);

/// Width of each counter field packed into a derived [`PromotionId`].
/// Three of them fill the 96-bit counter space exactly.
const DERIVED_FIELD_BITS: u32 = 32;

impl PromotionId {
    /// Derive a promotion id from the triple that identifies the
    /// promotion: which operation, which source, which record.
    ///
    /// Deterministic and injective, so `plan_promotion` can stay `&self`
    /// — planning the same request twice yields the same id, and two
    /// promotions of one source for different operations do not collide.
    pub fn derived(operation: OperationId, source: AuthorityId, record: RecordId) -> Self {
        let field = |raw: u128| -> u128 {
            let counter = raw & ((1u128 << DOMAIN_TAG_SHIFT) - 1);
            debug_assert!(
                counter < (1u128 << DERIVED_FIELD_BITS),
                "id counter overflows a derived promotion id field"
            );
            counter & ((1u128 << DERIVED_FIELD_BITS) - 1)
        };
        Self::from_counter(
            (field(operation.raw()) << (2 * DERIVED_FIELD_BITS))
                | (field(source.raw()) << DERIVED_FIELD_BITS)
                | field(record.raw()),
        )
    }
}

/// Deterministic per-domain id source, one per engine instance.
#[derive(Debug, Default, Clone)]
pub struct IdAllocator {
    authority: u128,
    record: u128,
    operation: u128,
    promotion: u128,
    qualification: u128,
    checkpoint: u128,
}

impl IdAllocator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn next_authority(&mut self) -> AuthorityId {
        self.authority += 1;
        AuthorityId::from_counter(self.authority)
    }

    pub fn next_record(&mut self) -> RecordId {
        self.record += 1;
        RecordId::from_counter(self.record)
    }

    pub fn next_operation(&mut self) -> OperationId {
        self.operation += 1;
        OperationId::from_counter(self.operation)
    }

    pub fn next_promotion(&mut self) -> PromotionId {
        self.promotion += 1;
        PromotionId::from_counter(self.promotion)
    }

    pub fn next_qualification(&mut self) -> QualificationId {
        self.qualification += 1;
        QualificationId::from_counter(self.qualification)
    }

    pub fn next_checkpoint(&mut self) -> CheckpointId {
        self.checkpoint += 1;
        CheckpointId::from_counter(self.checkpoint)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocator_is_monotone_per_domain() {
        let mut ids = IdAllocator::new();
        let a1 = ids.next_authority();
        let a2 = ids.next_authority();
        assert_ne!(a1, a2);
        assert!(a1 < a2);
    }

    #[test]
    fn domains_do_not_alias() {
        // First id of every domain has counter 1; only the tag differs.
        let mut ids = IdAllocator::new();
        let raws = [
            ids.next_authority().raw(),
            ids.next_record().raw(),
            ids.next_operation().raw(),
            ids.next_promotion().raw(),
            ids.next_qualification().raw(),
            ids.next_checkpoint().raw(),
        ];
        for (i, a) in raws.iter().enumerate() {
            for b in raws.iter().skip(i + 1) {
                assert_ne!(a, b, "domain tags must keep first-counter ids disjoint");
            }
        }
    }

    #[test]
    fn allocation_is_deterministic_across_instances() {
        let mut a = IdAllocator::new();
        let mut b = IdAllocator::new();
        for _ in 0..8 {
            assert_eq!(a.next_record(), b.next_record());
        }
    }

    #[test]
    fn zero_is_not_a_valid_id_in_any_domain() {
        assert!(!AuthorityId(0).is_well_formed());
        assert!(!RecordId(0).is_well_formed());
        assert!(!OperationId(0).is_well_formed());
        assert!(!PromotionId(0).is_well_formed());
        assert!(!QualificationId(0).is_well_formed());
        assert!(!CheckpointId(0).is_well_formed());
    }

    #[test]
    fn minted_ids_are_well_formed() {
        let mut ids = IdAllocator::new();
        assert!(ids.next_authority().is_well_formed());
        assert!(ids.next_record().is_well_formed());
        assert!(ids.next_operation().is_well_formed());
        assert!(ids.next_promotion().is_well_formed());
        assert!(ids.next_qualification().is_well_formed());
        assert!(ids.next_checkpoint().is_well_formed());
    }

    #[test]
    fn cross_domain_raw_reuse_is_detectable() {
        let mut ids = IdAllocator::new();
        let record = ids.next_record();
        // A caller bug that funnels a RecordId's raw bits into an
        // AuthorityId must not produce a well-formed authority.
        assert!(!AuthorityId(record.raw()).is_well_formed());
    }

    #[test]
    fn from_counter_round_trips_through_raw() {
        let id = RecordId::from_counter(42);
        assert_eq!(RecordId::from_counter(42).raw(), id.raw());
        assert!(id.is_well_formed());
    }

    #[test]
    fn ids_round_trip_through_json_above_the_64_bit_range() {
        // Every tagged id exceeds u64::MAX, so a numeric wire form would
        // fail here.
        let id = RecordId::from_counter(42);
        assert!(id.raw() > u128::from(u64::MAX));
        let json = serde_json::to_string(&id).unwrap();
        assert_eq!(json, format!("\"{}\"", id.raw()));
        assert_eq!(serde_json::from_str::<RecordId>(&json).unwrap(), id);
    }

    #[test]
    fn derived_promotion_ids_are_deterministic_and_injective() {
        let op = |n| OperationId::from_counter(n);
        let src = |n| AuthorityId::from_counter(n);
        let rec = |n| RecordId::from_counter(n);

        // Same triple → same id.
        assert_eq!(
            PromotionId::derived(op(1), src(2), rec(3)),
            PromotionId::derived(op(1), src(2), rec(3))
        );

        // Varying any one component changes the id — in particular two
        // promotions of the SAME source for different operations.
        let base = PromotionId::derived(op(1), src(2), rec(3));
        for other in [
            PromotionId::derived(op(9), src(2), rec(3)),
            PromotionId::derived(op(1), src(9), rec(3)),
            PromotionId::derived(op(1), src(2), rec(9)),
        ] {
            assert_ne!(base, other);
        }
        assert!(base.is_well_formed());
    }

    #[test]
    fn derived_ids_do_not_alias_across_field_boundaries() {
        // Packing must not let a large record counter bleed into the
        // source field.
        let a = PromotionId::derived(
            OperationId::from_counter(0),
            AuthorityId::from_counter(1),
            RecordId::from_counter(0),
        );
        let b = PromotionId::derived(
            OperationId::from_counter(0),
            AuthorityId::from_counter(0),
            RecordId::from_counter(u128::from(u32::MAX)),
        );
        assert_ne!(a, b);
    }

    #[test]
    fn a_malformed_id_string_is_a_deserialisation_error() {
        assert!(serde_json::from_str::<RecordId>("\"not-a-number\"").is_err());
        assert!(serde_json::from_str::<RecordId>("42").is_err());
    }
}
