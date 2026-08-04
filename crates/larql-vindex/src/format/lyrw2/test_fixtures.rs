//! Shared fixtures for the LYRW v2 container tests.
//!
//! One routed bank with two roles carrying two *different* codecs — the whole
//! point of per-region format tags — written with a recognisable byte pattern
//! so a test can tell entry 2's gate from entry 3's down by inspection.

use std::path::{Path, PathBuf};

use super::bank::{BankDescriptor, BankKind};
use super::browse_mode::BrowseMode;
use super::layout::Lyrw2Layout;
use super::plan::Lyrw2Plan;
use super::region_format::{Packing, RegionFormat};
use super::region_role::RegionRole;
use super::region_schema::RegionSchema;
use super::write::Lyrw2Writer;

pub(super) const BANK_ID: u16 = 0;
pub(super) const ENTRIES: u32 = 4;
pub(super) const SCHEMAS: u16 = 2;
pub(super) const HIDDEN: u32 = 4;
pub(super) const INTERMEDIATE: u32 = 8;
pub(super) const GATE_BYTES: usize = 40;
pub(super) const DOWN_BYTES: usize = 24;
pub(super) const LOGICAL_LAYER: u32 = 7;

/// Gate bytes for entry `e` are all `e`; down bytes are all `0x80 | e`.
pub(super) fn gate_pattern(entry: u32) -> Vec<u8> {
    vec![entry as u8; GATE_BYTES]
}

pub(super) fn down_pattern(entry: u32) -> Vec<u8> {
    vec![0x80 | entry as u8; DOWN_BYTES]
}

pub(super) fn bank(num_entries: u32) -> BankDescriptor {
    BankDescriptor {
        bank_id: BANK_ID,
        kind: BankKind::Routed,
        num_entries,
        input_dim: HIDDEN,
        intermediate_dim: INTERMEDIATE,
        output_dim: HIDDEN,
        region_schema_count: SCHEMAS,
        browse: BrowseMode::Direct,
    }
}

/// Gate at f16 row-major (browse-walkable), down at Q6_K blocks — two roles,
/// two codecs, one file.
pub(super) fn schemas() -> Vec<RegionSchema> {
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
            RegionFormat::Q6K,
            Packing::BlocksWithScalesInline,
            HIDDEN,
            INTERMEDIATE,
        ),
    ]
}

pub(super) fn plan() -> Lyrw2Plan {
    Lyrw2Plan::single_segment(LOGICAL_LAYER, bank(ENTRIES), schemas())
}

pub(super) fn temp_path(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join("lyrw2-tests");
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

/// Write `entries` entries through the streaming writer and return the bytes.
pub(super) fn write_file(path: &Path, plan: Lyrw2Plan, entries: u32) -> Vec<u8> {
    let mut w = Lyrw2Writer::create(path, plan).unwrap();
    for e in 0..entries {
        w.write_region(&gate_pattern(e)).unwrap();
        w.write_region(&down_pattern(e)).unwrap();
    }
    w.finish().unwrap();
    std::fs::read(path).unwrap()
}

/// A complete single-segment file covering every entry.
pub(super) fn single_segment_file(name: &str) -> Vec<u8> {
    write_file(&temp_path(name), plan(), ENTRIES)
}

/// The `(offset, length)` the writer recorded for one region, read back out of
/// the entry table through the layout rather than a hard-coded offset.
pub(super) fn slot(bytes: &[u8], entry: u32, schema: u16) -> (u64, u64) {
    let at = slot_offset(entry, schema);
    let offset = u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap());
    let length = u64::from_le_bytes(bytes[at + 8..at + 16].try_into().unwrap());
    (offset, length)
}

/// Byte offset of one entry-table slot in the default fixture file.
pub(super) fn slot_offset(entry: u32, schema: u16) -> usize {
    Lyrw2Layout::of(&plan())
        .entry_slot(0, entry, schema)
        .unwrap() as usize
}
