//! Tests for `addressing` — how a codec's bytes map to its elements.
//!
//! The distinction these pin is not cosmetic. Sizing a blocked region with a
//! scalar stride under-counts by roughly 4× for Q4_K, which binds a region that
//! looks long enough and reads past its end on the last row.

use crate::format::lyrw2::region_format::RegionFormat;

use crate::runtime::addressing::{Addressing, BlockOperand};
use crate::runtime::consts::{BF16_BYTES, F16_BYTES, F32_BYTES};

use super::support::{Q4K_BLOCK_BYTES, Q4K_BLOCK_ELEMS};

/// Every codec the format registry names, so a new one cannot be added without
/// this file having an opinion about it.
const REGISTERED: [RegionFormat; 11] = [
    RegionFormat::F32,
    RegionFormat::F16,
    RegionFormat::BF16,
    RegionFormat::Q4_0,
    RegionFormat::Q4K,
    RegionFormat::Q6K,
    RegionFormat::Q8_0,
    RegionFormat::Fp4Larql,
    RegionFormat::Mxfp4,
    RegionFormat::Nvfp4,
    RegionFormat::Mxfp8,
];

/// A padded `down`-shaped operand: 704 live columns stored in 768.
const PADDED_STORAGE_COLS: usize = 768;
const PADDED_ROLE_COLS: usize = 704;

// ── Which codecs have a layout ─────────────────────────────────────────────

#[test]
fn directly_addressed_codecs_report_their_element_width() {
    assert_eq!(
        Addressing::of(RegionFormat::F32),
        Some(Addressing::Scalar { bytes: F32_BYTES })
    );
    assert_eq!(
        Addressing::of(RegionFormat::F16),
        Some(Addressing::Scalar { bytes: F16_BYTES })
    );
    assert_eq!(
        Addressing::of(RegionFormat::BF16),
        Some(Addressing::Scalar { bytes: BF16_BYTES })
    );
}

#[test]
fn q4k_reports_the_registrys_block_geometry() {
    assert_eq!(
        Addressing::of(RegionFormat::Q4K),
        Some(Addressing::Blocked {
            elements: Q4K_BLOCK_ELEMS,
            bytes: Q4K_BLOCK_BYTES,
        })
    );
}

#[test]
fn the_k_quants_are_blocked_and_the_rest_have_no_geometry() {
    for format in [RegionFormat::Q4_0, RegionFormat::Q4K, RegionFormat::Q6K] {
        assert!(
            matches!(Addressing::of(format), Some(Addressing::Blocked { .. })),
            "{format:?} should be blocked"
        );
    }
    // Guessing a layout for these would read plausible bytes at the wrong
    // offsets and produce a well-shaped tensor of noise, so they are refused.
    for format in [
        RegionFormat::Q8_0,
        RegionFormat::Fp4Larql,
        RegionFormat::Mxfp4,
        RegionFormat::Nvfp4,
        RegionFormat::Mxfp8,
    ] {
        assert_eq!(Addressing::of(format), None, "{format:?} has no geometry");
    }
}

#[test]
fn an_unrecognised_codec_has_no_addressing() {
    assert_eq!(Addressing::of(RegionFormat::Unknown(4_242)), None);
}

#[test]
fn the_codec_sweep_still_covers_the_registry() {
    // A guard on the guard. If a codec joins the registry and not `REGISTERED`,
    // the sweep above narrows silently and stops being a sweep.
    for (tag, expected) in REGISTERED.iter().enumerate() {
        assert_eq!(
            RegionFormat::from_u16(tag as u16),
            *expected,
            "REGISTERED is out of step with the format registry at tag {tag}"
        );
    }
    assert!(matches!(
        RegionFormat::from_u16(REGISTERED.len() as u16),
        RegionFormat::Unknown(_)
    ));
}

// ── Sizing ─────────────────────────────────────────────────────────────────

#[test]
fn a_scalar_region_is_its_element_count_times_its_width() {
    let f32s = Addressing::Scalar { bytes: F32_BYTES };
    assert_eq!(f32s.region_bytes(0), 0);
    assert_eq!(f32s.region_bytes(3), 3 * F32_BYTES);
}

#[test]
fn a_blocked_region_rounds_up_to_a_whole_block() {
    let q4k = Addressing::of(RegionFormat::Q4K).expect("q4_k has geometry");
    assert_eq!(q4k.region_bytes(0), 0);
    // One element still costs a whole super-block.
    assert_eq!(q4k.region_bytes(1), Q4K_BLOCK_BYTES);
    assert_eq!(q4k.region_bytes(Q4K_BLOCK_ELEMS), Q4K_BLOCK_BYTES);
    assert_eq!(q4k.region_bytes(Q4K_BLOCK_ELEMS + 1), 2 * Q4K_BLOCK_BYTES);
}

#[test]
fn blocked_sizing_is_not_the_scalar_answer() {
    // The mistake this type exists to prevent: 256 Q4_K elements are 144 bytes,
    // not 256 × anything.
    let q4k = Addressing::of(RegionFormat::Q4K).expect("q4_k has geometry");
    let elements = 4 * Q4K_BLOCK_ELEMS;
    assert_eq!(q4k.region_bytes(elements), 4 * Q4K_BLOCK_BYTES);
    assert!(q4k.region_bytes(elements) < elements);
}

// ── The block descriptors ──────────────────────────────────────────────────

#[test]
fn only_a_blocked_addressing_reports_block_geometry() {
    let q4k = Addressing::of(RegionFormat::Q4K).expect("q4_k has geometry");
    assert_eq!(q4k.block_elements(), Some(Q4K_BLOCK_ELEMS));
    assert_eq!(q4k.block_bytes(), Some(Q4K_BLOCK_BYTES));

    let f32s = Addressing::Scalar { bytes: F32_BYTES };
    assert_eq!(f32s.block_elements(), None);
    assert_eq!(f32s.block_bytes(), None);
}

// ── The handover type ──────────────────────────────────────────────────────

#[test]
fn a_block_operand_reports_the_padding_between_its_two_extents() {
    let bytes = [0u8; Q4K_BLOCK_BYTES];
    let operand = BlockOperand {
        bytes: &bytes,
        rows: 1,
        storage_cols: PADDED_STORAGE_COLS,
        role_cols: PADDED_ROLE_COLS,
        row_bytes: Q4K_BLOCK_BYTES,
    };
    assert_eq!(
        operand.padding_cols(),
        PADDED_STORAGE_COLS - PADDED_ROLE_COLS
    );
}

#[test]
fn an_unpadded_block_operand_has_no_padding() {
    let bytes = [0u8; Q4K_BLOCK_BYTES];
    let operand = BlockOperand {
        bytes: &bytes,
        rows: 1,
        storage_cols: Q4K_BLOCK_ELEMS,
        role_cols: Q4K_BLOCK_ELEMS,
        row_bytes: Q4K_BLOCK_BYTES,
    };
    assert_eq!(operand.padding_cols(), 0);
}
