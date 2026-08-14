//! Corroborating material carried alongside a promoted value.
//!
//! EXP-36 found two facts in one corpus that behave completely
//! differently against an identical style of terse record: one resists
//! promotion entirely until its source is excluded, the other is
//! overridden unaided. The corpus contains a candidate explanation, and
//! it is not depth and not answer type — it is whether the value the
//! record asserts already appears somewhere **in a type-correct role**:
//!
//! ```text
//! fact 1  vault code  7431 -> 5824    5824 appears, but as a SECTOR    resists
//! fact 2  region      North -> South  South appears, as a REGION       overridden
//! fact 3  supervisor  Ilex -> Corvin  Corvin appears, as a SUPERVISOR  (EXP-38)
//! fact 4  clearance   8 -> 3          3 appears, as a CLEARANCE LEVEL  (EXP-38)
//! ```
//!
//! [`SupportOrigin`] is the field the architecture turns on, and EXP-38
//! measures it directly. If support only works when it was
//! [`Prefilled`](SupportOrigin::Prefilled), a caller cannot manufacture
//! authority mid-walk and explicit de-authorisation stays on the
//! critical path. If [`SuppliedAtPromotion`](SupportOrigin::SuppliedAtPromotion)
//! works, a graph walk becomes `RESOLVE -> construct support -> PROMOTE
//! -> EXECUTE` with no attention intervention anywhere in it.
//!
//! Nothing here asserts which. The type exists so the distinction is
//! recorded rather than assumed, and so a promotion that carried no
//! support at all is distinguishable from one whose support was ignored.

use serde::{Deserialize, Serialize};

use super::ids::AuthorityId;
use super::record::{SemanticAtom, SemanticValue};

/// Where the supporting material came from, relative to the walk.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum SupportOrigin {
    /// Already in the context when the walk began. The caller did not
    /// put it there and cannot rely on it being present for an
    /// arbitrary value.
    Prefilled { authority: AuthorityId },
    /// Manufactured by the caller and injected with the promotion. The
    /// form a graph walk can actually produce on demand.
    SuppliedAtPromotion,
}

impl SupportOrigin {
    /// Whether a caller can produce this kind of support for a value of
    /// its choosing.
    pub fn is_manufacturable(self) -> bool {
        matches!(self, Self::SuppliedAtPromotion)
    }
}

/// Material establishing that a value occupies a particular role.
///
/// `role` is an opaque caller atom, never parsed and never matched
/// against a fixed vocabulary — the engine has no notion of "region" or
/// "clearance level", only of whether two atoms are equal.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TypedSupport {
    /// The role the material establishes, compared and never
    /// interpreted.
    pub role: SemanticAtom,
    /// The value the material places in that role.
    pub value: SemanticValue,
    pub origin: SupportOrigin,
}

impl TypedSupport {
    pub fn new(role: SemanticAtom, value: SemanticValue, origin: SupportOrigin) -> Self {
        Self {
            role,
            value,
            origin,
        }
    }

    /// Whether this material supports `value` in `role`.
    ///
    /// Both must match. Supporting the right value in the wrong role is
    /// exactly fact 1's situation — `5824` is present in the corpus, as
    /// a sector number rather than as a vault code — and that fact is
    /// the one that resists promotion.
    pub fn establishes(&self, role: &SemanticAtom, value: &SemanticValue) -> bool {
        &self.role == role && &self.value == value
    }
}

/// Whether any of `support` establishes `value` in `role`.
pub fn establishes_any(
    support: &[TypedSupport],
    role: &SemanticAtom,
    value: &SemanticValue,
) -> bool {
    support.iter().any(|s| s.establishes(role, value))
}

/// The support in `support` that a caller could have manufactured.
pub fn manufacturable(support: &[TypedSupport]) -> impl Iterator<Item = &TypedSupport> {
    support.iter().filter(|s| s.origin.is_manufacturable())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn region() -> SemanticAtom {
        SemanticAtom::new("region")
    }

    fn sector() -> SemanticAtom {
        SemanticAtom::new("sector")
    }

    fn south() -> SemanticValue {
        SemanticValue::new("region_name", "South")
    }

    fn north() -> SemanticValue {
        SemanticValue::new("region_name", "North")
    }

    fn supplied() -> TypedSupport {
        TypedSupport::new(region(), south(), SupportOrigin::SuppliedAtPromotion)
    }

    fn prefilled() -> TypedSupport {
        TypedSupport::new(
            region(),
            south(),
            SupportOrigin::Prefilled {
                authority: AuthorityId::from_counter(3),
            },
        )
    }

    #[test]
    fn support_matches_only_its_own_role_and_value() {
        let s = supplied();
        assert!(s.establishes(&region(), &south()));
        assert!(!s.establishes(&region(), &north()));
        // The fact-1 case: right value, wrong role.
        assert!(!s.establishes(&sector(), &south()));
    }

    #[test]
    fn only_supplied_support_is_manufacturable() {
        assert!(SupportOrigin::SuppliedAtPromotion.is_manufacturable());
        assert!(!SupportOrigin::Prefilled {
            authority: AuthorityId::from_counter(1),
        }
        .is_manufacturable());
    }

    #[test]
    fn establishes_any_scans_the_whole_set() {
        let set = vec![prefilled(), supplied()];
        assert!(establishes_any(&set, &region(), &south()));
        assert!(!establishes_any(&set, &region(), &north()));
        assert!(!establishes_any(&[], &region(), &south()));
    }

    #[test]
    fn manufacturable_keeps_only_what_a_caller_could_build() {
        let set = vec![prefilled(), supplied()];
        let got: Vec<_> = manufacturable(&set).collect();
        assert_eq!(got, vec![&supplied()]);
        assert_eq!(manufacturable(&[]).count(), 0);
    }

    #[test]
    fn support_round_trips_through_json() {
        for s in [prefilled(), supplied()] {
            let j = serde_json::to_string(&s).expect("serialise");
            let back: TypedSupport = serde_json::from_str(&j).expect("deserialise");
            assert_eq!(s, back);
        }
    }
}
