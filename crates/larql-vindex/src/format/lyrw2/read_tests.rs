//! Colocated tests for `read` — the LYRW v2 table parser and region resolver.
//!
//! Verified by ROUND-TRIP against the streaming writer: build a file, parse it
//! back, and require every declared region to resolve to the bytes written.
//! Refusal paths live in `read_refusal_tests`.

use super::bank::{BankDescriptor, BankKind};
use super::browse_mode::BrowseMode;
use super::header::Lyrw2Header;
use super::plan::Lyrw2Plan;
use super::read::Lyrw2Reader;
use super::region_format::RegionFormat;
use super::region_role::RegionRole;
use super::segment::SegmentDescriptor;
use super::test_fixtures::{
    bank, down_pattern, gate_pattern, plan, schemas, single_segment_file, temp_path, write_file,
    BANK_ID, DOWN_BYTES, ENTRIES, HIDDEN, INTERMEDIATE, LOGICAL_LAYER,
};

#[test]
fn header_survives_the_round_trip() {
    let bytes = single_segment_file("header.weights");
    let reader = Lyrw2Reader::parse(&bytes).unwrap();
    assert_eq!(reader.header().logical_layer, LOGICAL_LAYER);
    assert_eq!(reader.header().num_banks, 1);
    assert_eq!(reader.header().num_segments, 1);
}

#[test]
fn bank_geometry_survives_the_round_trip() {
    let bytes = single_segment_file("bank.weights");
    let reader = Lyrw2Reader::parse(&bytes).unwrap();
    assert_eq!(reader.banks(), &[bank(ENTRIES)]);
}

#[test]
fn segments_survive_the_round_trip() {
    let bytes = single_segment_file("segments.weights");
    let reader = Lyrw2Reader::parse(&bytes).unwrap();
    assert_eq!(reader.segments().len(), 1);
    assert_eq!(reader.segments()[0].entry_count, ENTRIES);
}

#[test]
fn schemas_survive_the_round_trip_including_mixed_formats() {
    let bytes = single_segment_file("schemas.weights");
    let reader = Lyrw2Reader::parse(&bytes).unwrap();
    let got = reader.schemas_for(BANK_ID).unwrap();
    assert_eq!(got, schemas().as_slice());
    // The point of per-region tags: two roles, two different codecs, one file.
    assert_eq!(got[0].format, RegionFormat::F16);
    assert_eq!(got[1].format, RegionFormat::Q6K);
}

#[test]
fn every_region_resolves_to_the_bytes_written() {
    let bytes = single_segment_file("payload.weights");
    let reader = Lyrw2Reader::parse(&bytes).unwrap();
    for e in 0..ENTRIES {
        let gate = reader
            .region_bytes(BANK_ID, e, RegionRole::Gate)
            .unwrap()
            .unwrap();
        let down = reader
            .region_bytes(BANK_ID, e, RegionRole::Down)
            .unwrap()
            .unwrap();
        assert_eq!(gate, gate_pattern(e), "gate of entry {e}");
        assert_eq!(down, down_pattern(e), "down of entry {e}");
    }
}

#[test]
fn resolved_regions_carry_their_schema() {
    let bytes = single_segment_file("resolved.weights");
    let reader = Lyrw2Reader::parse(&bytes).unwrap();
    let region = reader
        .resolve(BANK_ID, 2, RegionRole::Down)
        .unwrap()
        .unwrap();
    assert_eq!(region.length, DOWN_BYTES as u64);
    assert_eq!(region.schema.format, RegionFormat::Q6K);
    assert_eq!(region.schema.role, RegionRole::Down);
}

#[test]
fn browse_reads_only_gate_bytes() {
    // The §15.1 economics claim as a byte-range assertion: no gate region's
    // span overlaps any down region's, so a gate-only sweep faults no down page.
    let bytes = single_segment_file("browse.weights");
    let reader = Lyrw2Reader::parse(&bytes).unwrap();
    let mut gate_spans = Vec::new();
    let mut down_spans = Vec::new();
    for e in 0..ENTRIES {
        let g = reader
            .resolve(BANK_ID, e, RegionRole::Gate)
            .unwrap()
            .unwrap();
        let d = reader
            .resolve(BANK_ID, e, RegionRole::Down)
            .unwrap()
            .unwrap();
        gate_spans.push((g.offset, g.offset + g.length));
        down_spans.push((d.offset, d.offset + d.length));
    }
    for (gs, ge) in &gate_spans {
        for (ds, de) in &down_spans {
            assert!(
                ge <= ds || gs >= de,
                "gate {gs}..{ge} overlaps down {ds}..{de}"
            );
        }
    }
}

#[test]
fn an_absent_role_resolves_to_none_not_an_error() {
    // §6.5: absence of a role is a capability question, not a corrupt file.
    let bytes = single_segment_file("absent-role.weights");
    let reader = Lyrw2Reader::parse(&bytes).unwrap();
    assert_eq!(reader.resolve(BANK_ID, 0, RegionRole::Up).unwrap(), None);
    assert_eq!(reader.resolve(BANK_ID, 0, RegionRole::Bias).unwrap(), None);
}

#[test]
fn an_absent_bank_resolves_to_none() {
    let bytes = single_segment_file("absent-bank.weights");
    let reader = Lyrw2Reader::parse(&bytes).unwrap();
    assert_eq!(reader.resolve(99, 0, RegionRole::Gate).unwrap(), None);
    assert!(reader.schemas_for(99).is_none());
}

#[test]
fn an_entry_past_the_segment_resolves_to_none() {
    let bytes = single_segment_file("past-end.weights");
    let reader = Lyrw2Reader::parse(&bytes).unwrap();
    assert_eq!(
        reader.resolve(BANK_ID, ENTRIES, RegionRole::Gate).unwrap(),
        None
    );
}

#[test]
fn a_split_layer_resolves_only_its_own_entries() {
    let half = ENTRIES / 2;
    let mut p = plan();
    p.header = Lyrw2Header::new(LOGICAL_LAYER, 1, 1).with_multi_segment(true);
    p.segments = vec![SegmentDescriptor {
        bank_id: BANK_ID,
        segment_index: 1,
        first_entry: half,
        entry_count: half,
    }];
    let bytes = write_file(&temp_path("split-seg1.weights"), p, half);
    let reader = Lyrw2Reader::parse(&bytes).unwrap();

    assert!(reader.header().is_multi_segment());
    // Entries 0..half live in the sibling segment file.
    assert_eq!(reader.resolve(BANK_ID, 0, RegionRole::Gate).unwrap(), None);
    // Entry `half` is this segment's local entry 0, written with byte 0.
    let gate = reader
        .region_bytes(BANK_ID, half, RegionRole::Gate)
        .unwrap()
        .unwrap();
    assert_eq!(gate, gate_pattern(0));
}

#[test]
fn a_dense_layer_is_the_single_entry_case() {
    let dense = BankDescriptor {
        bank_id: BANK_ID,
        kind: BankKind::Dense,
        num_entries: 1,
        input_dim: HIDDEN,
        intermediate_dim: INTERMEDIATE,
        output_dim: HIDDEN,
        region_schema_count: 2,
        browse: BrowseMode::None,
    };
    let p = Lyrw2Plan::single_segment(0, dense, schemas());
    let bytes = write_file(&temp_path("dense.weights"), p, 1);
    let reader = Lyrw2Reader::parse(&bytes).unwrap();
    assert_eq!(reader.banks()[0].kind, BankKind::Dense);
    assert!(reader
        .region_bytes(BANK_ID, 0, RegionRole::Gate)
        .unwrap()
        .is_some());
}

#[test]
fn layout_is_recovered_from_the_parsed_tables() {
    let bytes = single_segment_file("layout.weights");
    let reader = Lyrw2Reader::parse(&bytes).unwrap();
    let layout = reader.layout();
    assert!(layout.payload_start > layout.entry_table);
    assert_eq!(layout.schema_count(0), Some(2));
}
