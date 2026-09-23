//! Where every table sits in a LYRW v2 segment file.
//!
//! All offset arithmetic lives here, in one place, computed from a validated
//! plan. That is deliberate: the failure this format has to design against is
//! a table read at the wrong stride, which does not bounds-fail — it lands on
//! offsets that are still inside the file and returns bytes from the wrong
//! expert. One arithmetic surface with its own tests is cheaper than that bug.

use super::consts::{
    align_up, BANK_DESCRIPTOR_BYTES, ENTRY_REGION_BYTES, HEADER_BYTES, REGION_SCHEMA_BYTES,
    SEGMENT_DESCRIPTOR_BYTES,
};
use super::plan::Lyrw2Plan;

/// Byte offsets of each table, plus where the payload starts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Lyrw2Layout {
    pub bank_table: u64,
    pub segment_table: u64,
    pub schema_table: u64,
    pub entry_table: u64,
    /// Start of each segment's entry table, parallel to `plan.segments`.
    pub segment_entry_tables: Vec<u64>,
    /// Region slots per entry for each segment, parallel to `plan.segments`.
    segment_schema_counts: Vec<usize>,
    /// Total bytes across every segment's entry table.
    entry_table_total: u64,
    pub payload_start: u64,
}

impl Lyrw2Layout {
    /// Compute the layout for a plan. The plan must already have validated —
    /// in particular every segment must name a declared bank, which this
    /// relies on when sizing per-segment entry tables.
    pub fn of(plan: &Lyrw2Plan) -> Self {
        let bank_table = HEADER_BYTES as u64;
        let segment_table = bank_table + (plan.banks.len() * BANK_DESCRIPTOR_BYTES) as u64;
        let schema_table = segment_table + (plan.segments.len() * SEGMENT_DESCRIPTOR_BYTES) as u64;

        let schema_bytes: u64 = plan
            .schemas
            .iter()
            .map(|s| (s.len() * REGION_SCHEMA_BYTES) as u64)
            .sum();
        let entry_table = schema_table + schema_bytes;

        let mut segment_entry_tables = Vec::with_capacity(plan.segments.len());
        let mut segment_schema_counts = Vec::with_capacity(plan.segments.len());
        let mut cursor = entry_table;
        for segment in &plan.segments {
            let schema_count = plan
                .bank(segment.bank_id)
                .map(|b| b.regions_per_entry())
                .unwrap_or(0);
            segment_entry_tables.push(cursor);
            segment_schema_counts.push(schema_count);
            cursor += entry_table_bytes(segment.entry_count, schema_count);
        }

        Self {
            bank_table,
            segment_table,
            schema_table,
            entry_table,
            segment_entry_tables,
            segment_schema_counts,
            entry_table_total: cursor - entry_table,
            payload_start: align_up(cursor),
        }
    }

    /// Total bytes occupied by every entry table in this file.
    pub fn entry_table_bytes(&self) -> u64 {
        self.entry_table_total
    }

    /// Byte offset of one entry-table slot.
    ///
    /// `segment_ordinal` indexes `plan.segments`; `local_entry` is the entry's
    /// position *within that segment*, not its logical index.
    pub fn entry_slot(
        &self,
        segment_ordinal: usize,
        local_entry: u32,
        schema_index: u16,
    ) -> Option<u64> {
        let table = *self.segment_entry_tables.get(segment_ordinal)?;
        let schema_count = *self.segment_schema_counts.get(segment_ordinal)?;
        if usize::from(schema_index) >= schema_count {
            return None;
        }
        let slot = u64::from(local_entry) * schema_count as u64 + u64::from(schema_index);
        Some(table + slot * ENTRY_REGION_BYTES as u64)
    }

    /// Region slots per entry for a segment.
    pub fn schema_count(&self, segment_ordinal: usize) -> Option<usize> {
        self.segment_schema_counts.get(segment_ordinal).copied()
    }
}

/// Bytes one segment's entry table occupies.
pub fn entry_table_bytes(entry_count: u32, schema_count: usize) -> u64 {
    u64::from(entry_count) * schema_count as u64 * ENTRY_REGION_BYTES as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::lyrw2::bank::{BankDescriptor, BankKind};
    use crate::format::lyrw2::browse_mode::BrowseMode;
    use crate::format::lyrw2::consts::REGION_ALIGNMENT;
    use crate::format::lyrw2::region_format::{Packing, RegionFormat};
    use crate::format::lyrw2::region_role::RegionRole;
    use crate::format::lyrw2::region_schema::RegionSchema;

    const ENTRIES: u32 = 4;
    const SCHEMAS: usize = 2;

    fn plan() -> Lyrw2Plan {
        let bank = BankDescriptor {
            bank_id: 0,
            kind: BankKind::Routed,
            num_entries: ENTRIES,
            input_dim: 8,
            intermediate_dim: 8,
            output_dim: 8,
            region_schema_count: SCHEMAS as u16,
            browse: BrowseMode::Direct,
        };
        let schemas = vec![
            RegionSchema::unpaired(
                0,
                RegionRole::Gate,
                RegionFormat::F16,
                Packing::RowMajor,
                8,
                8,
            ),
            RegionSchema::unpaired(
                1,
                RegionRole::Down,
                RegionFormat::F16,
                Packing::RowMajor,
                8,
                8,
            ),
        ];
        Lyrw2Plan::single_segment(0, bank, schemas)
    }

    #[test]
    fn tables_follow_the_header_in_declaration_order() {
        let layout = Lyrw2Layout::of(&plan());
        assert_eq!(layout.bank_table, HEADER_BYTES as u64);
        assert_eq!(layout.segment_table, 24 + 24);
        assert_eq!(layout.schema_table, 24 + 24 + 12);
        assert_eq!(layout.entry_table, 24 + 24 + 12 + 2 * 20);
    }

    #[test]
    fn payload_starts_on_the_region_alignment() {
        let layout = Lyrw2Layout::of(&plan());
        assert_eq!(layout.payload_start % REGION_ALIGNMENT, 0);
        assert!(layout.payload_start >= layout.entry_table);
    }

    #[test]
    fn entry_table_is_sized_by_entries_times_schemas() {
        assert_eq!(
            entry_table_bytes(ENTRIES, SCHEMAS),
            u64::from(ENTRIES) * SCHEMAS as u64 * ENTRY_REGION_BYTES as u64
        );
    }

    #[test]
    fn empty_table_is_zero_bytes() {
        assert_eq!(entry_table_bytes(0, SCHEMAS), 0);
        assert_eq!(entry_table_bytes(ENTRIES, 0), 0);
    }

    #[test]
    fn entry_slots_are_contiguous_within_an_entry() {
        let layout = Lyrw2Layout::of(&plan());
        let first = layout.entry_slot(0, 0, 0).unwrap();
        let second = layout.entry_slot(0, 0, 1).unwrap();
        assert_eq!(second - first, ENTRY_REGION_BYTES as u64);
    }

    #[test]
    fn entries_are_strided_by_their_schema_count() {
        let layout = Lyrw2Layout::of(&plan());
        let entry0 = layout.entry_slot(0, 0, 0).unwrap();
        let entry1 = layout.entry_slot(0, 1, 0).unwrap();
        assert_eq!(entry1 - entry0, (SCHEMAS * ENTRY_REGION_BYTES) as u64);
    }

    #[test]
    fn first_slot_sits_at_the_entry_table_base() {
        let layout = Lyrw2Layout::of(&plan());
        assert_eq!(layout.entry_slot(0, 0, 0), Some(layout.entry_table));
    }

    #[test]
    fn a_schema_index_past_the_bank_declaration_has_no_slot() {
        let layout = Lyrw2Layout::of(&plan());
        assert_eq!(layout.entry_slot(0, 0, SCHEMAS as u16), None);
    }

    #[test]
    fn an_absent_segment_has_no_slot() {
        let layout = Lyrw2Layout::of(&plan());
        assert_eq!(layout.entry_slot(1, 0, 0), None);
        assert_eq!(layout.schema_count(1), None);
    }

    #[test]
    fn schema_count_is_reported_per_segment() {
        let layout = Lyrw2Layout::of(&plan());
        assert_eq!(layout.schema_count(0), Some(SCHEMAS));
    }

    #[test]
    fn total_entry_table_bytes_covers_every_segment() {
        let layout = Lyrw2Layout::of(&plan());
        assert_eq!(
            layout.entry_table_bytes(),
            entry_table_bytes(ENTRIES, SCHEMAS)
        );
    }

    #[test]
    fn payload_start_clears_the_whole_entry_table() {
        let layout = Lyrw2Layout::of(&plan());
        assert!(layout.payload_start >= layout.entry_table + layout.entry_table_bytes());
    }

    #[test]
    fn two_segments_get_disjoint_entry_tables() {
        let mut p = plan();
        p.header.num_segments = 2;
        p.segments = vec![
            crate::format::lyrw2::segment::SegmentDescriptor {
                bank_id: 0,
                segment_index: 0,
                first_entry: 0,
                entry_count: 2,
            },
            crate::format::lyrw2::segment::SegmentDescriptor {
                bank_id: 0,
                segment_index: 1,
                first_entry: 2,
                entry_count: 2,
            },
        ];
        let layout = Lyrw2Layout::of(&p);
        let gap = layout.segment_entry_tables[1] - layout.segment_entry_tables[0];
        assert_eq!(gap, entry_table_bytes(2, SCHEMAS));
    }
}
