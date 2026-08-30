//! Per-bank region schema (spec §6.4).
//!
//! Expert banks are homogeneous: every entry in a bank shares one region
//! layout, so the schema is declared once per bank and each entry stores only
//! offsets and lengths. That is what makes per-expert codec variation — which
//! no grouped kernel supports — unrepresentable by construction rather than by
//! convention, and what makes parsing O(schemas) instead of O(entries × regions).

use super::consts::{PAIR_ID_UNPAIRED, REGION_SCHEMA_BYTES};
use super::region_format::{Packing, RegionFormat};
use super::region_layout::RegionLayout;
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
    /// How this region's rows are arranged **as stored** — see
    /// [`RegionLayout`]. Occupies the u16 that was reserved at bytes
    /// 10..12, so `REGION_SCHEMA_BYTES` is unchanged at 20.
    ///
    /// The byte count not changing is *not* why this is compatible. The
    /// meaning of a nonzero value there did change, and a reader that
    /// ignores it can mix a fused operand's two branches — so admission is
    /// gated by the index schema version, not by the wire size.
    pub layout: RegionLayout,
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
            layout: RegionLayout::Unspecified,
            rows,
            cols,
        }
    }

    /// Whether this region declares a partner.
    ///
    /// The question a consumer asks *before* reaching for one:
    /// `resolve_paired` treats a partner lookup on an unpaired region as a
    /// caller mistake and errors, deliberately, so an inline-scale bank must
    /// be recognised here rather than by attempting the lookup and reading
    /// the error as an answer. Error-as-absence would also swallow
    /// `MissingPartner`, which is a corrupt bank and not an inline one.
    pub fn is_paired(&self) -> bool {
        self.pair_id != PAIR_ID_UNPAIRED
    }

    pub fn encode(&self, out: &mut Vec<u8>) {
        push_u16(out, self.schema_index);
        push_u16(out, self.role.as_u16());
        push_u16(out, self.format.as_u16());
        push_u16(out, self.packing.as_u16());
        push_u16(out, self.pair_id);
        push_u16(out, self.layout.as_u16());
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
            layout: RegionLayout::from_u16(read_u16(bytes, 10)?),
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

    /// Whether this region's role is one whose row arrangement is a real
    /// choice — i.e. a *fused* operand carrying two branches in one matrix.
    ///
    /// `Gate` and `Up` stored separately have no arrangement to declare;
    /// neither does `Down`, `Bias` or a scale stream. Keeping this a
    /// property of the role rather than a writer convention is what makes
    /// "layout on a non-fused region" checkable instead of merely unusual.
    pub fn role_has_row_arrangement(&self) -> bool {
        matches!(self.role, RegionRole::GateUpFused)
    }

    /// Whether the layout declaration is consistent with the role.
    ///
    /// Two directions, both real:
    /// - a non-fused region declaring an arrangement is describing
    ///   something it does not have, and a consumer that believed it would
    ///   act on a fiction;
    /// - a fused region that declares nothing cannot be interpreted at all
    ///   without inferring, which is the failure this field exists to stop.
    ///
    /// The second is permitted only for containers written before the
    /// field existed, which is why the check is separate from
    /// [`Self::layout_is_declared_where_required`].
    pub fn layout_is_consistent_with_role(&self) -> bool {
        self.role_has_row_arrangement() || !self.layout.is_declared()
    }

    /// Whether a *newly written* schema declares an arrangement wherever
    /// one is meaningful. Legacy containers fail this and are read under
    /// an explicit legacy rule instead.
    pub fn layout_is_declared_where_required(&self) -> bool {
        !self.role_has_row_arrangement() || self.layout.is_declared()
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
            layout: RegionLayout::Unspecified,
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
            layout: RegionLayout::Unknown(903),
            rows: 1,
            cols: 1,
        };
        let mut buf = Vec::new();
        schema.encode(&mut buf);
        assert_eq!(RegionSchema::decode(&buf), Some(schema));
    }

    // ── the stored-layout declaration ────────────────────────────────

    /// The field occupies what was the reserved u16, so the wire size is
    /// unchanged at 20. That is a compatibility *hazard*, not a
    /// compatibility guarantee — an old reader parses the record fine and
    /// ignores a value that changes what the bytes mean. Admission is
    /// gated by the index schema version instead (`V3_CURRENT_SCHEMA`).
    #[test]
    fn layout_round_trips_in_the_formerly_reserved_field() {
        for layout in [
            RegionLayout::Unspecified,
            RegionLayout::ContiguousHalves,
            RegionLayout::Interleaved,
            RegionLayout::Unknown(904),
        ] {
            let schema = RegionSchema { layout, ..sample() };
            let mut buf = Vec::new();
            schema.encode(&mut buf);
            assert_eq!(buf.len(), REGION_SCHEMA_BYTES, "wire size must not change");
            assert_eq!(RegionSchema::decode(&buf), Some(schema), "{layout:?}");
            // And it really is the old reserved slot.
            assert_eq!(
                u16::from_le_bytes([buf[10], buf[11]]),
                layout.as_u16(),
                "layout must occupy bytes 10..12"
            );
        }
    }

    /// A container written before the field existed has zeros there, which
    /// must read as "did not say" — never as a concrete arrangement.
    #[test]
    fn a_legacy_reserved_zero_reads_as_undeclared() {
        let mut buf = Vec::new();
        sample().encode(&mut buf);
        buf[10] = 0;
        buf[11] = 0;
        let decoded = RegionSchema::decode(&buf).unwrap();
        assert_eq!(decoded.layout, RegionLayout::Unspecified);
        assert!(!decoded.layout.is_declared());
    }

    /// Only a fused operand has two branches to arrange. Declaring a
    /// layout on anything else describes something the region does not
    /// have.
    #[test]
    fn only_fused_roles_carry_a_row_arrangement() {
        let fused = RegionSchema {
            role: RegionRole::GateUpFused,
            layout: RegionLayout::Interleaved,
            ..sample()
        };
        assert!(fused.role_has_row_arrangement());
        assert!(fused.layout_is_consistent_with_role());
        assert!(fused.layout_is_declared_where_required());

        for role in [RegionRole::Down, RegionRole::Scales, RegionRole::Bias] {
            let declared = RegionSchema {
                role,
                layout: RegionLayout::Interleaved,
                ..sample()
            };
            assert!(!declared.role_has_row_arrangement(), "{role:?}");
            assert!(
                !declared.layout_is_consistent_with_role(),
                "{role:?} must not declare an arrangement"
            );

            let quiet = RegionSchema {
                role,
                layout: RegionLayout::Unspecified,
                ..sample()
            };
            assert!(quiet.layout_is_consistent_with_role(), "{role:?}");
            assert!(quiet.layout_is_declared_where_required(), "{role:?}");
        }
    }

    /// A fused region that declares nothing is readable (legacy) but is
    /// not something a new writer may emit — the two checks are separate
    /// on purpose.
    #[test]
    fn an_undeclared_fused_region_is_legacy_readable_but_not_writable() {
        let legacy = RegionSchema {
            role: RegionRole::GateUpFused,
            layout: RegionLayout::Unspecified,
            ..sample()
        };
        assert!(legacy.layout_is_consistent_with_role());
        assert!(!legacy.layout_is_declared_where_required());
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
