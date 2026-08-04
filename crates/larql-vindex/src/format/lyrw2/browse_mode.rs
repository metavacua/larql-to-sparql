//! Per-bank browse eligibility (spec §15.2).
//!
//! A browse-enabled index requires gate rows readable without decoding `up`.
//! Two storage shapes satisfy that — decomposed `gate` + `up` regions, or a
//! fused region whose packing permits striding into the gate half. The choice
//! is recorded per bank at extraction time so a loader never has to infer it
//! from the region list, and serving-only banks can fuse freely.

use super::consts::BANK_FLAG_BROWSE_MASK;
use super::region_format::Packing;

/// How gate rows may be read from this bank.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BrowseMode {
    /// Not browse-enabled. Gate KNN must refuse this bank.
    #[default]
    None,
    /// Gate rows live in their own region — a direct read.
    Direct,
    /// Gate rows live in a fused region and are reached by striding.
    Strided,
}

impl BrowseMode {
    pub fn from_flags(flags: u16) -> Self {
        match flags & BANK_FLAG_BROWSE_MASK {
            1 => Self::Direct,
            2 => Self::Strided,
            // 0 and the unassigned 3 both mean "not browse-enabled". An
            // unknown mode must never read as browsable.
            _ => Self::None,
        }
    }

    pub fn to_flag_bits(self) -> u16 {
        match self {
            Self::None => 0,
            Self::Direct => 1,
            Self::Strided => 2,
        }
    }

    /// Whether gate KNN may walk this bank at all.
    pub fn is_browsable(self) -> bool {
        !matches!(self, Self::None)
    }

    /// Whether `packing` can actually deliver what this mode promises.
    ///
    /// Striding into the gate half of a fused region is only sound for
    /// row-major layouts; interleaved quantised blocks generally cannot be
    /// strided without decoding the `up` rows too, which is the whole cost
    /// browse was avoiding.
    pub fn is_satisfiable_by(self, packing: Packing) -> bool {
        match self {
            Self::None => true,
            Self::Direct => true,
            Self::Strided => packing.permits_strided_gate_read(),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Direct => "direct",
            Self::Strided => "strided",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modes_round_trip_through_flag_bits() {
        for mode in [BrowseMode::None, BrowseMode::Direct, BrowseMode::Strided] {
            assert_eq!(BrowseMode::from_flags(mode.to_flag_bits()), mode);
        }
    }

    #[test]
    fn unassigned_mode_reads_as_not_browsable() {
        // Bit pattern 3 is unassigned; it must fail closed, not browse.
        assert_eq!(BrowseMode::from_flags(3), BrowseMode::None);
    }

    #[test]
    fn bits_above_the_mask_are_ignored() {
        let with_noise = BrowseMode::Direct.to_flag_bits() | 0xFFFC;
        assert_eq!(BrowseMode::from_flags(with_noise), BrowseMode::Direct);
    }

    #[test]
    fn default_is_not_browsable() {
        assert_eq!(BrowseMode::default(), BrowseMode::None);
        assert!(!BrowseMode::default().is_browsable());
    }

    #[test]
    fn direct_and_strided_are_browsable() {
        assert!(BrowseMode::Direct.is_browsable());
        assert!(BrowseMode::Strided.is_browsable());
    }

    #[test]
    fn strided_needs_a_row_major_packing() {
        assert!(BrowseMode::Strided.is_satisfiable_by(Packing::RowMajor));
        assert!(!BrowseMode::Strided.is_satisfiable_by(Packing::BlocksWithScalesInline));
        assert!(!BrowseMode::Strided.is_satisfiable_by(Packing::BlocksValues));
    }

    #[test]
    fn direct_and_none_accept_any_packing() {
        for packing in [
            Packing::RowMajor,
            Packing::BlocksWithScalesInline,
            Packing::BlocksValues,
        ] {
            assert!(BrowseMode::Direct.is_satisfiable_by(packing));
            assert!(BrowseMode::None.is_satisfiable_by(packing));
        }
    }

    #[test]
    fn names_are_stable() {
        assert_eq!(BrowseMode::None.name(), "none");
        assert_eq!(BrowseMode::Direct.name(), "direct");
        assert_eq!(BrowseMode::Strided.name(), "strided");
    }
}
