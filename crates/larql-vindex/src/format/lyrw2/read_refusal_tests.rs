//! Refusal tests for `read` — what a LYRW v2 reader must NOT accept.
//!
//! Each test takes a known-good file and corrupts exactly one field, then
//! requires the matching named error. The class being defended against is an
//! offset that is still *inside* the file: it does not bounds-fail, it hands
//! back plausible bytes from the wrong place. Every case here must refuse
//! rather than return.

use super::consts::{BANK_DESCRIPTOR_BYTES, HEADER_BYTES};
use super::error::Lyrw2Error;
use super::read::Lyrw2Reader;
use super::region_role::RegionRole;
use super::test_fixtures::{single_segment_file, slot_offset, BANK_ID};

/// Byte offset of the version field within the header.
const VERSION_FIELD: usize = 4;
/// Byte offset of the first segment descriptor's `bank_id`.
const FIRST_SEGMENT_BANK_ID: usize = HEADER_BYTES + BANK_DESCRIPTOR_BYTES;

#[test]
fn a_v1_file_is_refused_by_generation_not_by_parse_error() {
    let mut bytes = single_segment_file("v1-generation.weights");
    bytes[VERSION_FIELD..VERSION_FIELD + 4].copy_from_slice(&1u32.to_le_bytes());
    let err = Lyrw2Reader::parse(&bytes).unwrap_err();
    assert!(matches!(err, Lyrw2Error::WrongGeneration { found: 1, .. }));
    assert!(err.to_string().contains("VINDEX2"), "{err}");
}

#[test]
fn a_future_generation_is_refused_rather_than_guessed_at() {
    let mut bytes = single_segment_file("lyrw3-generation.weights");
    bytes[VERSION_FIELD..VERSION_FIELD + 4].copy_from_slice(&3u32.to_le_bytes());
    let err = Lyrw2Reader::parse(&bytes).unwrap_err();
    assert!(matches!(err, Lyrw2Error::WrongGeneration { found: 3, .. }));
    // LYRW format_version 3 belongs to a VINDEX4 container, not VINDEX3 —
    // the layer sequence trails the container sequence by one.
    assert!(err.to_string().contains("VINDEX4"), "{err}");
}

#[test]
fn a_foreign_file_is_refused_by_magic() {
    let mut bytes = single_segment_file("foreign.weights");
    bytes[0..4].copy_from_slice(&0x4B4E_5546u32.to_le_bytes());
    assert!(matches!(
        Lyrw2Reader::parse(&bytes),
        Err(Lyrw2Error::NotLyrw { .. })
    ));
}

#[test]
fn a_truncated_bank_table_is_named_in_the_error() {
    let bytes = single_segment_file("trunc-bank.weights");
    let short = &bytes[..HEADER_BYTES + 6];
    let err = Lyrw2Reader::parse(short).unwrap_err();
    assert!(
        matches!(err, Lyrw2Error::Truncated { what, .. } if what.contains("bank")),
        "{err}"
    );
}

#[test]
fn a_truncated_segment_table_is_named_in_the_error() {
    let bytes = single_segment_file("trunc-segment.weights");
    let short = &bytes[..FIRST_SEGMENT_BANK_ID + 4];
    let err = Lyrw2Reader::parse(short).unwrap_err();
    assert!(
        matches!(err, Lyrw2Error::Truncated { what, .. } if what.contains("segment")),
        "{err}"
    );
}

#[test]
fn a_truncated_schema_table_is_named_in_the_error() {
    let bytes = single_segment_file("trunc-schema.weights");
    // Header + one bank + one segment = 60 bytes; the schema table starts there.
    let short = &bytes[..70];
    let err = Lyrw2Reader::parse(short).unwrap_err();
    assert!(
        matches!(err, Lyrw2Error::Truncated { what, .. } if what.contains("schema")),
        "{err}"
    );
}

#[test]
fn a_segment_naming_an_unknown_bank_is_refused_at_parse() {
    let mut bytes = single_segment_file("bad-segment.weights");
    bytes[FIRST_SEGMENT_BANK_ID..FIRST_SEGMENT_BANK_ID + 2].copy_from_slice(&9u16.to_le_bytes());
    assert!(matches!(
        Lyrw2Reader::parse(&bytes),
        Err(Lyrw2Error::SegmentNamesUnknownBank { bank_id: 9, .. })
    ));
}

#[test]
fn a_region_offset_past_the_file_is_refused_not_returned() {
    // The headline failure mode: an offset that is still a plausible number.
    let mut bytes = single_segment_file("oob-offset.weights");
    let at = slot_offset(0, 0);
    let past_end = (bytes.len() as u64) + 1_024;
    bytes[at..at + 8].copy_from_slice(&past_end.to_le_bytes());

    let reader = Lyrw2Reader::parse(&bytes).unwrap();
    let err = reader.resolve(BANK_ID, 0, RegionRole::Gate).unwrap_err();
    assert!(
        matches!(&err, Lyrw2Error::RegionOutOfBounds { role, .. } if role == "gate"),
        "{err}"
    );
}

#[test]
fn a_length_overrunning_the_file_is_refused() {
    let mut bytes = single_segment_file("oob-length.weights");
    let at = slot_offset(0, 1);
    let too_long = bytes.len() as u64;
    bytes[at + 8..at + 16].copy_from_slice(&too_long.to_le_bytes());

    let reader = Lyrw2Reader::parse(&bytes).unwrap();
    let err = reader.resolve(BANK_ID, 0, RegionRole::Down).unwrap_err();
    assert!(
        matches!(&err, Lyrw2Error::RegionOutOfBounds { role, .. } if role == "down"),
        "{err}"
    );
}

#[test]
fn an_out_of_bounds_error_reports_the_file_length_it_checked_against() {
    let mut bytes = single_segment_file("oob-detail.weights");
    let at = slot_offset(1, 0);
    let past_end = (bytes.len() as u64) * 2;
    bytes[at..at + 8].copy_from_slice(&past_end.to_le_bytes());

    let reader = Lyrw2Reader::parse(&bytes).unwrap();
    let err = reader.resolve(BANK_ID, 1, RegionRole::Gate).unwrap_err();
    let text = err.to_string();
    assert!(text.contains("entry 1"), "{text}");
    assert!(text.contains(&bytes.len().to_string()), "{text}");
}
