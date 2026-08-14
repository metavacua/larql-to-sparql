//! Colocated tests for `plan` — the pre-write consistency checks.
//!
//! Every rule here is one the reader later relies on, so each test corrupts a
//! valid plan in exactly one way and requires the matching named refusal. The
//! point is that a malformed plan fails at the API, before a file exists —
//! not when someone eventually tries to serve it.

use super::bank::{BankDescriptor, BankKind};
use super::browse_mode::BrowseMode;
use super::consts::PAIR_ID_UNPAIRED;
use super::error::Lyrw2Error;
use super::plan::Lyrw2Plan;
use super::region_format::{Packing, RegionFormat};
use super::region_role::RegionRole;
use super::region_schema::RegionSchema;
use super::segment::SegmentDescriptor;

const EXPERTS: u32 = 8;
const HIDDEN: u32 = 16;
const INTERMEDIATE: u32 = 32;

fn bank() -> BankDescriptor {
    BankDescriptor {
        bank_id: 0,
        kind: BankKind::Routed,
        num_entries: EXPERTS,
        input_dim: HIDDEN,
        intermediate_dim: INTERMEDIATE,
        output_dim: HIDDEN,
        region_schema_count: 2,
        browse: BrowseMode::Direct,
    }
}

fn schemas() -> Vec<RegionSchema> {
    vec![
        RegionSchema::unpaired(
            0,
            RegionRole::Gate,
            RegionFormat::F16,
            Packing::RowMajor,
            INTERMEDIATE,
            HIDDEN,
        ),
        RegionSchema::unpaired(
            1,
            RegionRole::Down,
            RegionFormat::F16,
            Packing::RowMajor,
            HIDDEN,
            INTERMEDIATE,
        ),
    ]
}

fn plan() -> Lyrw2Plan {
    Lyrw2Plan::single_segment(3, bank(), schemas())
}

#[test]
fn single_segment_plan_validates() {
    assert_eq!(plan().validate(), Ok(()));
}

#[test]
fn single_segment_covers_every_entry() {
    let p = plan();
    assert_eq!(p.segments.len(), 1);
    assert_eq!(p.segments[0].entry_count, EXPERTS);
    assert_eq!(p.segments[0].first_entry, 0);
}

#[test]
fn bank_lookup_finds_by_id_not_position() {
    let mut p = plan();
    p.banks[0].bank_id = 9;
    assert_eq!(p.bank_ordinal(9), Some(0));
    assert_eq!(p.bank_ordinal(0), None);
    assert!(p.bank(9).is_some());
}

#[test]
fn header_bank_count_must_match_the_bank_list() {
    let mut p = plan();
    p.header.num_banks = 2;
    assert!(matches!(
        p.validate(),
        Err(Lyrw2Error::BankSchemaCountMismatch { .. })
    ));
}

#[test]
fn schema_list_must_be_parallel_to_the_bank_list() {
    let mut p = plan();
    p.schemas.push(Vec::new());
    assert!(matches!(
        p.validate(),
        Err(Lyrw2Error::BankSchemaCountMismatch { .. })
    ));
}

#[test]
fn declared_schema_count_must_match_the_schemas_supplied() {
    let mut p = plan();
    p.banks[0].region_schema_count = 3;
    assert!(matches!(
        p.validate(),
        Err(Lyrw2Error::BankSchemaCountMismatch {
            bank_id: 0,
            declared: 3,
            found: 2
        })
    ));
}

#[test]
fn schema_index_must_equal_its_position() {
    let mut p = plan();
    p.schemas[0][1].schema_index = 7;
    assert!(matches!(
        p.validate(),
        Err(Lyrw2Error::BankSchemaCountMismatch { .. })
    ));
}

#[test]
fn a_split_region_without_a_partner_is_refused() {
    let mut p = plan();
    p.schemas[0][0].packing = Packing::BlocksValues;
    assert!(matches!(
        p.validate(),
        Err(Lyrw2Error::InconsistentPairing { pair_id, .. }) if pair_id == PAIR_ID_UNPAIRED
    ));
}

#[test]
fn an_inline_region_naming_a_partner_is_refused() {
    let mut p = plan();
    p.schemas[0][0].pair_id = 1;
    assert!(matches!(
        p.validate(),
        Err(Lyrw2Error::InconsistentPairing {
            schema_index: 0,
            ..
        })
    ));
}

#[test]
fn a_matched_values_scales_pair_validates() {
    let mut p = plan();
    p.schemas[0][0].packing = Packing::BlocksValues;
    p.schemas[0][0].pair_id = 1;
    p.schemas[0][1].packing = Packing::BlocksScales;
    p.schemas[0][1].pair_id = 0;
    assert_eq!(p.validate(), Ok(()));
}

#[test]
fn a_segment_naming_an_absent_bank_is_refused() {
    let mut p = plan();
    p.segments[0].bank_id = 5;
    assert!(matches!(
        p.validate(),
        Err(Lyrw2Error::SegmentNamesUnknownBank { bank_id: 5, .. })
    ));
}

#[test]
fn a_segment_past_the_banks_entries_is_refused() {
    let mut p = plan();
    p.segments[0].entry_count = EXPERTS + 1;
    assert!(matches!(
        p.validate(),
        Err(Lyrw2Error::SegmentOutOfBounds { .. })
    ));
}

#[test]
fn two_segments_splitting_a_bank_validate() {
    let half = EXPERTS / 2;
    let mut p = plan();
    p.header.num_segments = 2;
    p.header = p.header.with_multi_segment(true);
    p.segments = vec![
        SegmentDescriptor {
            bank_id: 0,
            segment_index: 0,
            first_entry: 0,
            entry_count: half,
        },
        SegmentDescriptor {
            bank_id: 0,
            segment_index: 1,
            first_entry: half,
            entry_count: half,
        },
    ];
    assert_eq!(p.validate(), Ok(()));
}
