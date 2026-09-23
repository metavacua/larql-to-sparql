//! Region roles — what a stored region *is* within an expert (spec §6.5).
//!
//! Unknown roles are preserved, not rejected. A file carrying a role this
//! binary has never heard of is still a valid file: §6.5 makes presence of an
//! unregistered role harmless, and absence of a *required* role a capability
//! failure for one programme rather than a corrupt container. Fail-closed
//! belongs at capability-check time (§11), not at parse time — a browse-only
//! reader must not choke on a `down` region it will never touch.

use super::consts::PAIR_ID_UNPAIRED;

/// A registered region role, or an unrecognised tag preserved verbatim.
///
/// `Ord` follows the registry tag, so ordering is the spec's role order rather
/// than declaration order in this enum — capability reports sort by coordinate
/// and a stable, spec-derived order keeps their output diffable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegionRole {
    Gate,
    Up,
    GateUpFused,
    Down,
    Bias,
    Scales,
    LatentIn,
    LatentOut,
    /// A tag this binary does not recognise. Round-trips unchanged.
    Unknown(u16),
}

impl PartialOrd for RegionRole {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RegionRole {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.as_u16().cmp(&other.as_u16())
    }
}

/// First tag in the vendor/experimental space (spec §6.5).
pub const VENDOR_ROLE_BASE: u16 = 256;

impl RegionRole {
    pub fn from_u16(tag: u16) -> Self {
        match tag {
            0 => Self::Gate,
            1 => Self::Up,
            2 => Self::GateUpFused,
            3 => Self::Down,
            4 => Self::Bias,
            5 => Self::Scales,
            6 => Self::LatentIn,
            7 => Self::LatentOut,
            other => Self::Unknown(other),
        }
    }

    pub fn as_u16(self) -> u16 {
        match self {
            Self::Gate => 0,
            Self::Up => 1,
            Self::GateUpFused => 2,
            Self::Down => 3,
            Self::Bias => 4,
            Self::Scales => 5,
            Self::LatentIn => 6,
            Self::LatentOut => 7,
            Self::Unknown(tag) => tag,
        }
    }

    /// Whether this role carries gate rows, and so is walkable by gate KNN
    /// (spec §15.1). `GateUpFused` qualifies only when the bank's browse mode
    /// permits striding — that is a bank-level question, not a role-level one,
    /// so this answers the role half alone.
    pub fn carries_gate_rows(self) -> bool {
        matches!(self, Self::Gate | Self::GateUpFused)
    }

    /// Whether this role is in the vendor/experimental space.
    pub fn is_vendor(self) -> bool {
        self.as_u16() >= VENDOR_ROLE_BASE
    }

    /// Human-readable name for diagnostics. Unregistered tags render as
    /// `role_<n>` so an error message never claims a name it does not know.
    pub fn name(self) -> String {
        match self {
            Self::Gate => "gate".into(),
            Self::Up => "up".into(),
            Self::GateUpFused => "gate_up_fused".into(),
            Self::Down => "down".into(),
            Self::Bias => "bias".into(),
            Self::Scales => "scales".into(),
            Self::LatentIn => "latent_in".into(),
            Self::LatentOut => "latent_out".into(),
            Self::Unknown(tag) => format!("role_{tag}"),
        }
    }
}

/// Whether a `pair_id` names a partner schema.
pub fn is_paired(pair_id: u16) -> bool {
    pair_id != PAIR_ID_UNPAIRED
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registered_roles_round_trip() {
        for tag in 0u16..=7 {
            assert_eq!(RegionRole::from_u16(tag).as_u16(), tag);
        }
    }

    #[test]
    fn unknown_role_is_preserved_not_rejected() {
        let role = RegionRole::from_u16(9_000);
        assert_eq!(role, RegionRole::Unknown(9_000));
        assert_eq!(role.as_u16(), 9_000);
    }

    #[test]
    fn reserved_registered_span_parses_as_unknown() {
        // 8..255 is reserved-registered: not yet named here, still legal.
        assert_eq!(RegionRole::from_u16(8), RegionRole::Unknown(8));
        assert!(!RegionRole::Unknown(8).is_vendor());
    }

    #[test]
    fn vendor_space_starts_at_256() {
        assert!(!RegionRole::from_u16(VENDOR_ROLE_BASE - 1).is_vendor());
        assert!(RegionRole::from_u16(VENDOR_ROLE_BASE).is_vendor());
        assert!(!RegionRole::Gate.is_vendor());
    }

    #[test]
    fn only_gate_bearing_roles_are_walkable() {
        assert!(RegionRole::Gate.carries_gate_rows());
        assert!(RegionRole::GateUpFused.carries_gate_rows());
        assert!(!RegionRole::Up.carries_gate_rows());
        assert!(!RegionRole::Down.carries_gate_rows());
        assert!(!RegionRole::Scales.carries_gate_rows());
        assert!(!RegionRole::Unknown(300).carries_gate_rows());
    }

    #[test]
    fn names_are_stable_for_registered_roles() {
        assert_eq!(RegionRole::Gate.name(), "gate");
        assert_eq!(RegionRole::GateUpFused.name(), "gate_up_fused");
        assert_eq!(RegionRole::LatentOut.name(), "latent_out");
    }

    #[test]
    fn every_registered_role_has_a_distinct_name() {
        let names: Vec<String> = (0u16..=7).map(|t| RegionRole::from_u16(t).name()).collect();
        assert_eq!(
            names,
            vec![
                "gate",
                "up",
                "gate_up_fused",
                "down",
                "bias",
                "scales",
                "latent_in",
                "latent_out"
            ]
        );
    }

    #[test]
    fn unknown_role_name_admits_it_is_unknown() {
        assert_eq!(RegionRole::Unknown(42).name(), "role_42");
    }

    #[test]
    fn roles_order_by_registry_tag_not_declaration() {
        // Sorting a capability report must be stable and spec-derived, so the
        // order follows §6.5's tag numbering.
        let mut v = vec![
            RegionRole::Down,
            RegionRole::Gate,
            RegionRole::Unknown(300),
            RegionRole::Up,
        ];
        v.sort();
        assert_eq!(
            v,
            vec![
                RegionRole::Gate,
                RegionRole::Up,
                RegionRole::Down,
                RegionRole::Unknown(300),
            ]
        );
    }

    #[test]
    fn an_unknown_role_sorts_after_every_registered_one() {
        assert!(RegionRole::Unknown(256) > RegionRole::LatentOut);
    }

    #[test]
    fn unpaired_sentinel_reads_as_unpaired() {
        assert!(!is_paired(PAIR_ID_UNPAIRED));
        assert!(is_paired(0));
        assert!(is_paired(3));
    }
}
