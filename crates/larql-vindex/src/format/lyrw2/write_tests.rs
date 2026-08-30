//! Colocated tests for `write` — the streaming LYRW v2 writer.
//!
//! The writer is exercised end to end (create, stream regions, backpatch) and
//! then pinned on its refusal paths: an invalid plan, a short file, an overrun,
//! and a deliberate abandon. Byte-level assertions resolve slots through
//! `Lyrw2Layout` rather than hard-coding offsets, so a layout change surfaces
//! as a layout test failure rather than as silently relocated expectations.

use super::consts::REGION_ALIGNMENT;
use super::layout::Lyrw2Layout;
use super::test_fixtures::{
    down_pattern, gate_pattern, plan, single_segment_file, slot, temp_path, DOWN_BYTES, ENTRIES,
    GATE_BYTES, SCHEMAS,
};
use super::write::{Lyrw2Writer, RegionCursor};

#[test]
fn a_complete_file_writes_and_closes() {
    let path = temp_path("w-complete.weights");
    let mut w = Lyrw2Writer::create(&path, plan()).unwrap();
    for e in 0..ENTRIES {
        w.write_region(&gate_pattern(e)).unwrap();
        w.write_region(&down_pattern(e)).unwrap();
    }
    assert!(w.finish().is_ok());
    assert!(path.exists());
}

#[test]
fn cursor_starts_at_the_first_region() {
    let path = temp_path("cursor-start.weights");
    let w = Lyrw2Writer::create(&path, plan()).unwrap();
    assert_eq!(
        w.cursor(),
        RegionCursor {
            segment_ordinal: 0,
            local_entry: 0,
            schema_index: 0
        }
    );
    assert!(!w.is_complete());
    assert_eq!(w.regions_written(), 0);
    w.abandon();
}

#[test]
fn cursor_walks_schemas_then_entries() {
    let path = temp_path("cursor-walk.weights");
    let mut w = Lyrw2Writer::create(&path, plan()).unwrap();
    w.write_region(&[0u8; GATE_BYTES]).unwrap();
    assert_eq!(w.cursor().schema_index, 1);
    assert_eq!(w.cursor().local_entry, 0);
    w.write_region(&[0u8; DOWN_BYTES]).unwrap();
    assert_eq!(w.cursor().schema_index, 0);
    assert_eq!(w.cursor().local_entry, 1);
    w.abandon();
}

#[test]
fn writer_reports_completion_after_the_last_region() {
    let path = temp_path("complete-flag.weights");
    let mut w = Lyrw2Writer::create(&path, plan()).unwrap();
    for _ in 0..ENTRIES {
        w.write_region(&[0u8; GATE_BYTES]).unwrap();
        w.write_region(&[0u8; DOWN_BYTES]).unwrap();
    }
    assert!(w.is_complete());
    assert_eq!(w.regions_written(), (ENTRIES * u32::from(SCHEMAS)) as usize);
    w.finish().unwrap();
}

#[test]
fn finishing_early_is_refused() {
    let path = temp_path("short.weights");
    let mut w = Lyrw2Writer::create(&path, plan()).unwrap();
    w.write_region(&[0u8; GATE_BYTES]).unwrap();
    let err = w.finish().unwrap_err();
    assert!(err.to_string().contains("plan declares more"), "{err}");
}

#[test]
fn writing_past_the_plan_is_refused() {
    let path = temp_path("overrun.weights");
    let mut w = Lyrw2Writer::create(&path, plan()).unwrap();
    for _ in 0..ENTRIES {
        w.write_region(&[0u8; GATE_BYTES]).unwrap();
        w.write_region(&[0u8; DOWN_BYTES]).unwrap();
    }
    let err = w.write_region(&[0u8; 4]).unwrap_err();
    assert!(err.to_string().contains("after every declared"), "{err}");
    w.finish().unwrap();
}

#[test]
fn an_invalid_plan_is_refused_before_the_file_is_touched() {
    let path = temp_path("invalid-plan.weights");
    let _ = std::fs::remove_file(&path);
    let mut bad = plan();
    bad.banks[0].region_schema_count = 5;
    assert!(Lyrw2Writer::create(&path, bad).is_err());
    assert!(!path.exists());
}

#[test]
fn abandoning_a_partial_file_is_not_a_forgotten_finish() {
    let path = temp_path("abandoned.weights");
    let mut w = Lyrw2Writer::create(&path, plan()).unwrap();
    w.write_region(&[0u8; GATE_BYTES]).unwrap();
    w.abandon();
    assert!(path.exists());
}

#[test]
fn retained_state_is_one_slot_per_region_not_the_payload() {
    let path = temp_path("streaming.weights");
    let mut w = Lyrw2Writer::create(&path, plan()).unwrap();
    w.write_region(&[7u8; GATE_BYTES]).unwrap();
    assert_eq!(w.regions_written(), 1);
    assert_eq!(w.region_length(0), Some(GATE_BYTES as u64));
    assert_eq!(w.region_length(1), None);
    w.abandon();
}

#[test]
fn every_region_lands_on_the_alignment_boundary() {
    let bytes = single_segment_file("w-aligned.weights");
    for entry in 0..ENTRIES {
        for schema in 0..SCHEMAS {
            let (offset, _) = slot(&bytes, entry, schema);
            assert_eq!(
                offset % REGION_ALIGNMENT,
                0,
                "entry {entry} schema {schema}"
            );
        }
    }
}

#[test]
fn the_entry_table_records_the_lengths_written() {
    let bytes = single_segment_file("w-lengths.weights");
    assert_eq!(slot(&bytes, 0, 0).1, GATE_BYTES as u64);
    assert_eq!(slot(&bytes, 0, 1).1, DOWN_BYTES as u64);
}

#[test]
fn payload_bytes_are_readable_at_the_recorded_offsets() {
    let bytes = single_segment_file("w-payload.weights");
    let (offset, _) = slot(&bytes, 2, 0);
    let start = offset as usize;
    // Entry 2's gate region was filled with the byte 2.
    assert_eq!(&bytes[start..start + GATE_BYTES], &[2u8; GATE_BYTES]);
}

#[test]
fn regions_never_overlap() {
    let bytes = single_segment_file("w-disjoint.weights");
    let mut spans: Vec<(u64, u64)> = Vec::new();
    for entry in 0..ENTRIES {
        for schema in 0..SCHEMAS {
            let (offset, length) = slot(&bytes, entry, schema);
            spans.push((offset, offset + length));
        }
    }
    spans.sort_unstable();
    for pair in spans.windows(2) {
        assert!(
            pair[0].1 <= pair[1].0,
            "{:?} overlaps {:?}",
            pair[0],
            pair[1]
        );
    }
}

#[test]
fn tables_precede_the_payload() {
    let bytes = single_segment_file("w-ordering.weights");
    let layout = Lyrw2Layout::of(&plan());
    assert!(bytes.len() as u64 > layout.payload_start);
    assert!(slot(&bytes, 0, 0).0 >= layout.payload_start);
}
