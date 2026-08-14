//! Per-bank region schema (spec §6.4).
//!
//! Expert banks are homogeneous: every entry in a bank shares one region
//! layout, so the schema is declared once per bank and each entry stores only
//! offsets and lengths. That is what makes per-expert codec variation — which
//! no grouped kernel supports — unrepresentable by construction rather than by
//! convention, and what makes parsing O(schemas) instead of O(entries × regions).

use super::consts::{PAIR_ID_UNPAIRED, REGION_SCHEMA_BYTES};
use super::region_format::{Packing, RegionFormat};
use super::region_role::RegionRole;
use super::wire::{push_u16, push_u32, read_u16, read_u32};

/// One region's declared shape and encoding, shared by every entry in a bank.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegionSchema {
    pub schema_index: u16,
    pub role: RegionRole,
    pub format: RegionFormat,
    pub packing: Packing,
    /// Links a `BlocksValues` schema to its `BlocksScales` partner, and back.
    /// `PAIR_ID_UNPAIRED` when the region stands alone.
    pub pair_id: u16,
    pub rows: u32,
    pub cols: u32,
}

impl RegionSchema {
    /// A standalone region with no values/scales partner.
    pub fn unpaired(
        schema_index: u16,
        role: RegionRole,
        format: RegionFormat,
        packing: Packing,
        rows: u32,
        cols: u32,
    ) -> Self {
        Self {
            schema_index,
            role,
            format,
            packing,
            pair_id: PAIR_ID_UNPAIRED,
            rows,
            cols,
        }
    }

    pub fn encode(&self, out: &mut Vec<u8>) {
        push_u16(out, self.schema_index);
        push_u16(out, self.role.as_u16());
        push_u16(out, self.format.as_u16());
        push_u16(out, self.packing.as_u16());
        push_u16(out, self.pair_id);
        push_u16(out, 0); // reserved
        push_u32(out, self.rows);
        push_u32(out, self.cols);
    }

    pub fn decode(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < REGION_SCHEMA_BYTES {
            return None;
        }
        Some(Self {
            schema_index: read_u16(bytes, 0)?,
            role: RegionRole::from_u16(read_u16(bytes, 2)?),
            format: RegionFormat::from_u16(read_u16(bytes, 4)?),
            packing: Packing::from_u16(read_u16(bytes, 6)?),
            pair_id: read_u16(bytes, 8)?,
            // bytes 10..12 reserved
            rows: read_u32(bytes, 12)?,
            cols: read_u32(bytes, 16)?,
        })
    }

    /// Whether this schema declares a partner but names none, or names one
    /// while declaring a packing that has no partner. Both are writer bugs
    /// that would otherwise surface as a silently half-decoded region.
    pub fn pairing_is_consistent(&self) -> bool {
        self.packing.requires_pair() == (self.pair_id != PAIR_ID_UNPAIRED)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> RegionSchema {
        RegionSchema::unpaired(
            1,
            RegionRole::Down,
            RegionFormat::Q6K,
            Packing::BlocksWithScalesInline,
            3_584,
            3_072,
        )
    }

    #[test]
    fn schema_round_trips_through_bytes() {
        let schema = sample();
        let mut buf = Vec::new();
        schema.encode(&mut buf);
        assert_eq!(buf.len(), REGION_SCHEMA_BYTES);
        assert_eq!(RegionSchema::decode(&buf), Some(schema));
    }

    #[test]
    fn paired_schema_round_trips() {
        let schema = RegionSchema {
            schema_index: 2,
            role: RegionRole::Scales,
            format: RegionFormat::Mxfp4,
            packing: Packing::BlocksScales,
            pair_id: 1,
            rows: 64,
            cols: 8,
        };
        let mut buf = Vec::new();
        schema.encode(&mut buf);
        assert_eq!(RegionSchema::decode(&buf), Some(schema));
    }

    #[test]
    fn unknown_tags_survive_a_round_trip() {
        let schema = RegionSchema {
            schema_index: 7,
            role: RegionRole::Unknown(900),
            format: RegionFormat::Unknown(901),
            packing: Packing::Unknown(902),
            pair_id: PAIR_ID_UNPAIRED,
            rows: 1,
            cols: 1,
        };
        let mut buf = Vec::new();
        schema.encode(&mut buf);
        assert_eq!(RegionSchema::decode(&buf), Some(schema));
    }

    #[test]
    fn short_record_decodes_to_none() {
        let mut buf = Vec::new();
        sample().encode(&mut buf);
        buf.pop();
        assert_eq!(RegionSchema::decode(&buf), None);
    }

    #[test]
    fn unpaired_helper_sets_the_sentinel() {
        assert_eq!(sample().pair_id, PAIR_ID_UNPAIRED);
    }

    #[test]
    fn reserved_field_is_written_as_zero() {
        let mut buf = Vec::new();
        sample().encode(&mut buf);
        assert_eq!(read_u16(&buf, 10), Some(0));
    }

    #[test]
    fn pairing_consistency_accepts_matched_declarations() {
        assert!(sample().pairing_is_consistent());
        let paired = RegionSchema {
            pair_id: 3,
            packing: Packing::BlocksValues,
            ..sample()
        };
        assert!(paired.pairing_is_consistent());
    }

    #[test]
    fn pairing_consistency_rejects_a_split_region_with_no_partner() {
        let orphan = RegionSchema {
            packing: Packing::BlocksValues,
            ..sample()
        };
        assert!(!orphan.pairing_is_consistent());
    }

    #[test]
    fn pairing_consistency_rejects_a_partner_on_an_inline_region() {
        let spurious = RegionSchema {
            pair_id: 0,
            ..sample()
        };
        assert!(!spurious.pairing_is_consistent());
    }
}
