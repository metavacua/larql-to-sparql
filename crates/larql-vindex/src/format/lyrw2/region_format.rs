//! Region encodings and packings (spec §6.4).
//!
//! Like roles, unrecognised tags are preserved rather than rejected — a reader
//! that only walks gate regions must not fail because a `down` region uses a
//! codec it cannot decode. Refusal happens when a kernel is asked to execute
//! the region (spec §10/§11), which is where the honest diagnosis lives.

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// Numeric encoding of a region's payload bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegionFormat {
    F32,
    F16,
    BF16,
    Q4_0,
    Q4K,
    Q6K,
    Q8_0,
    Fp4Larql,
    Mxfp4,
    Nvfp4,
    Mxfp8,
    /// A codec this binary does not recognise. Round-trips unchanged.
    Unknown(u16),
}

impl RegionFormat {
    pub fn from_u16(tag: u16) -> Self {
        match tag {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::BF16,
            3 => Self::Q4_0,
            4 => Self::Q4K,
            5 => Self::Q6K,
            6 => Self::Q8_0,
            7 => Self::Fp4Larql,
            8 => Self::Mxfp4,
            9 => Self::Nvfp4,
            10 => Self::Mxfp8,
            other => Self::Unknown(other),
        }
    }

    pub fn as_u16(self) -> u16 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::BF16 => 2,
            Self::Q4_0 => 3,
            Self::Q4K => 4,
            Self::Q6K => 5,
            Self::Q8_0 => 6,
            Self::Fp4Larql => 7,
            Self::Mxfp4 => 8,
            Self::Nvfp4 => 9,
            Self::Mxfp8 => 10,
            Self::Unknown(tag) => tag,
        }
    }

    /// Whether rows can be reinterpreted in place with no dequantisation —
    /// the zero-copy gate-KNN fast path (spec §15.1).
    pub fn is_zero_copy_walkable(self) -> bool {
        matches!(self, Self::F32 | Self::F16)
    }

    /// A borrowed name for the registered codecs; `None` for [`Self::Unknown`],
    /// whose name carries its tag and so cannot be static.
    ///
    /// Exists because a kernel refusal names the format it *wanted*, and a
    /// refusal is a `&'static str` field — the alternative was a second,
    /// hand-written spelling of every codec name at each call site.
    /// [`Self::name`] delegates here so the two cannot drift.
    pub const fn registered_name(self) -> Option<&'static str> {
        Some(match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::BF16 => "bf16",
            Self::Q4_0 => "q4_0",
            Self::Q4K => "q4_k",
            Self::Q6K => "q6_k",
            Self::Q8_0 => "q8_0",
            Self::Fp4Larql => "fp4_larql",
            Self::Mxfp4 => "mxfp4",
            Self::Nvfp4 => "nvfp4",
            Self::Mxfp8 => "mxfp8",
            Self::Unknown(_) => return None,
        })
    }

    pub fn name(self) -> String {
        match self.registered_name() {
            Some(name) => name.into(),
            None => format!("format_{}", self.as_u16()),
        }
    }
}

/// How a region's bytes are laid out.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Packing {
    /// Dense rows, no separate scale stream.
    RowMajor,
    /// Quantised blocks with their scales interleaved inline.
    BlocksWithScalesInline,
    /// Block values only; scales live in a partner region named by `pair_id`.
    BlocksValues,
    /// Block scales only; values live in a partner region named by `pair_id`.
    BlocksScales,
    Unknown(u16),
}

impl Packing {
    pub fn from_u16(tag: u16) -> Self {
        match tag {
            0 => Self::RowMajor,
            1 => Self::BlocksWithScalesInline,
            2 => Self::BlocksValues,
            3 => Self::BlocksScales,
            other => Self::Unknown(other),
        }
    }

    pub fn as_u16(self) -> u16 {
        match self {
            Self::RowMajor => 0,
            Self::BlocksWithScalesInline => 1,
            Self::BlocksValues => 2,
            Self::BlocksScales => 3,
            Self::Unknown(tag) => tag,
        }
    }

    /// Whether this packing *requires* a partner region via `pair_id`.
    pub fn requires_pair(self) -> bool {
        matches!(self, Self::BlocksValues | Self::BlocksScales)
    }

    /// Whether gate rows can be reached by striding, without decoding the
    /// interleaved `up` half of a fused region (spec §15.2).
    pub fn permits_strided_gate_read(self) -> bool {
        matches!(self, Self::RowMajor)
    }

    pub fn name(self) -> String {
        match self {
            Self::RowMajor => "row_major".into(),
            Self::BlocksWithScalesInline => "blocks_with_scales_inline".into(),
            Self::BlocksValues => "blocks_values".into(),
            Self::BlocksScales => "blocks_scales".into(),
            Self::Unknown(tag) => format!("packing_{tag}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registered_formats_round_trip() {
        for tag in 0u16..=10 {
            assert_eq!(RegionFormat::from_u16(tag).as_u16(), tag);
        }
    }

    #[test]
    fn unknown_format_is_preserved_not_rejected() {
        let f = RegionFormat::from_u16(4_242);
        assert_eq!(f, RegionFormat::Unknown(4_242));
        assert_eq!(f.as_u16(), 4_242);
        assert_eq!(f.name(), "format_4242");
    }

    #[test]
    fn only_f16_and_f32_are_zero_copy_walkable() {
        assert!(RegionFormat::F16.is_zero_copy_walkable());
        assert!(RegionFormat::F32.is_zero_copy_walkable());
        // bf16 needs a widening pass, so it is not the v1 zero-copy path.
        assert!(!RegionFormat::BF16.is_zero_copy_walkable());
        assert!(!RegionFormat::Q6K.is_zero_copy_walkable());
        assert!(!RegionFormat::Mxfp4.is_zero_copy_walkable());
    }

    #[test]
    fn native_low_bit_codecs_are_registered() {
        assert_eq!(RegionFormat::Mxfp4.name(), "mxfp4");
        assert_eq!(RegionFormat::Nvfp4.name(), "nvfp4");
        assert_eq!(RegionFormat::Mxfp8.name(), "mxfp8");
    }

    #[test]
    fn every_registered_format_has_a_distinct_name() {
        let names: Vec<String> = (0u16..=10)
            .map(|t| RegionFormat::from_u16(t).name())
            .collect();
        assert_eq!(
            names,
            vec![
                "f32",
                "f16",
                "bf16",
                "q4_0",
                "q4_k",
                "q6_k",
                "q8_0",
                "fp4_larql",
                "mxfp4",
                "nvfp4",
                "mxfp8"
            ]
        );
    }

    #[test]
    fn no_registered_format_is_zero_copy_beyond_f16_and_f32() {
        let walkable: Vec<u16> = (0u16..=10)
            .filter(|t| RegionFormat::from_u16(*t).is_zero_copy_walkable())
            .collect();
        assert_eq!(walkable, vec![0, 1]);
    }

    #[test]
    fn registered_packings_round_trip() {
        for tag in 0u16..=3 {
            assert_eq!(Packing::from_u16(tag).as_u16(), tag);
        }
    }

    #[test]
    fn unknown_packing_is_preserved() {
        assert_eq!(Packing::from_u16(77), Packing::Unknown(77));
        assert_eq!(Packing::Unknown(77).name(), "packing_77");
    }

    #[test]
    fn split_packings_require_a_partner() {
        assert!(Packing::BlocksValues.requires_pair());
        assert!(Packing::BlocksScales.requires_pair());
        assert!(!Packing::RowMajor.requires_pair());
        assert!(!Packing::BlocksWithScalesInline.requires_pair());
        assert!(!Packing::Unknown(9).requires_pair());
    }

    #[test]
    fn every_registered_packing_has_a_distinct_name() {
        let names: Vec<String> = (0u16..=3).map(|t| Packing::from_u16(t).name()).collect();
        assert_eq!(
            names,
            vec![
                "row_major",
                "blocks_with_scales_inline",
                "blocks_values",
                "blocks_scales"
            ]
        );
    }

    #[test]
    fn only_row_major_permits_strided_gate_reads() {
        assert!(Packing::RowMajor.permits_strided_gate_read());
        assert!(!Packing::BlocksWithScalesInline.permits_strided_gate_read());
        assert!(!Packing::BlocksValues.permits_strided_gate_read());
    }
}
