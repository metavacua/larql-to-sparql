//! LYRW v2 wire constants — magic, sizes, alignment and sentinels.
//!
//! Every width here is fixed by the format spec (§6.1–6.4). Nothing in this
//! module may change without a `FORMAT_VERSION` bump, because a reader that
//! disagrees with a writer about a field width does not fail — it parses the
//! next field from the wrong offset and returns plausible garbage.

/// File magic, shared with LYRW v1 so a v1 reader recognises the file and can
/// reject it on the version field rather than on a parse error (spec §6.6).
pub const MAGIC: u32 = u32::from_le_bytes(*b"LYRW");

/// Container generation this module reads and writes.
pub const FORMAT_VERSION: u32 = 2;

/// Every payload region begins on this boundary, measured from the start of
/// the containing segment file.
pub const REGION_ALIGNMENT: u64 = 64;

/// `magic`, `format_version`, `logical_layer`, `num_banks`, `num_segments`,
/// `flags`, `reserved`.
pub const HEADER_BYTES: usize = 24;

/// `bank_id`, `bank_kind`, `num_entries`, `input_dim`, `intermediate_dim`,
/// `output_dim`, `region_schema_count`, `flags`.
pub const BANK_DESCRIPTOR_BYTES: usize = 24;

/// `bank_id`, `segment_index`, `first_entry`, `entry_count`.
pub const SEGMENT_DESCRIPTOR_BYTES: usize = 12;

/// `schema_index`, `role`, `format`, `packing`, `pair_id`, `reserved`,
/// `rows`, `cols`.
pub const REGION_SCHEMA_BYTES: usize = 20;

/// One entry-table slot: `offset`, `length`.
pub const ENTRY_REGION_BYTES: usize = 16;

/// `pair_id` value meaning "this region is not one half of a values/scales
/// pair". Chosen as the maximum `u16` so a zeroed field is a *valid* pair id
/// rather than an accidental sentinel.
pub const PAIR_ID_UNPAIRED: u16 = 0xFFFF;

/// Header `flags` bit 0 — this file is one segment of a multi-segment logical
/// layer (spec §6.1).
pub const HEADER_FLAG_MULTI_SEGMENT: u32 = 1 << 0;

/// Bank `flags` bits 0–1 — how gate rows may be read for browse (spec §15.2).
pub const BANK_FLAG_BROWSE_MASK: u16 = 0b11;

/// Round `offset` up to the next `REGION_ALIGNMENT` boundary.
pub fn align_up(offset: u64) -> u64 {
    offset.div_ceil(REGION_ALIGNMENT) * REGION_ALIGNMENT
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn magic_is_the_ascii_tag() {
        assert_eq!(MAGIC.to_le_bytes(), *b"LYRW");
    }

    #[test]
    fn format_version_is_two() {
        assert_eq!(FORMAT_VERSION, 2);
    }

    #[test]
    fn align_up_leaves_aligned_offsets_alone() {
        assert_eq!(align_up(0), 0);
        assert_eq!(align_up(REGION_ALIGNMENT), REGION_ALIGNMENT);
        assert_eq!(align_up(REGION_ALIGNMENT * 3), REGION_ALIGNMENT * 3);
    }

    #[test]
    fn align_up_rounds_unaligned_offsets_forward() {
        assert_eq!(align_up(1), REGION_ALIGNMENT);
        assert_eq!(align_up(REGION_ALIGNMENT - 1), REGION_ALIGNMENT);
        assert_eq!(align_up(REGION_ALIGNMENT + 1), REGION_ALIGNMENT * 2);
    }

    #[test]
    fn unpaired_sentinel_is_not_zero() {
        // A zeroed pair_id must mean "paired with schema 0", not "unpaired",
        // otherwise a partially written table reads as valid.
        assert_ne!(PAIR_ID_UNPAIRED, 0);
    }

    #[test]
    fn descriptor_widths_are_multiples_of_four() {
        // Keeps every u32 field naturally aligned when tables are read as a
        // contiguous slice.
        for width in [
            HEADER_BYTES,
            BANK_DESCRIPTOR_BYTES,
            SEGMENT_DESCRIPTOR_BYTES,
            REGION_SCHEMA_BYTES,
            ENTRY_REGION_BYTES,
        ] {
            assert_eq!(width % 4, 0, "width {width} is not 4-byte aligned");
        }
    }
}
