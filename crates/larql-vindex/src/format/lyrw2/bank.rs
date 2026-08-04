//! Bank descriptors (spec §6.2).
//!
//! A bank is one homogeneous population of entries: an expert bank, a shared
//! bank, or a dense layer expressed as the degenerate `num_entries = 1` case.
//! `input_dim`/`output_dim` are the *entry's own* operand dims — for a latent
//! expert bank these are the latent width, not the residual width.
//!
//! The binary carries no programme identity. LYRW describes storage only; the
//! MoE manifest binds `bank_id → programme`. Two authorities for the same fact
//! is a disagreement waiting to happen.

use super::browse_mode::BrowseMode;
use super::consts::BANK_DESCRIPTOR_BYTES;
use super::wire::{push_u16, push_u32, read_u16, read_u32};

/// What kind of population a bank holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BankKind {
    Dense,
    Routed,
    Shared,
    Unknown(u16),
}

impl BankKind {
    pub fn from_u16(tag: u16) -> Self {
        match tag {
            0 => Self::Dense,
            1 => Self::Routed,
            2 => Self::Shared,
            other => Self::Unknown(other),
        }
    }

    pub fn as_u16(self) -> u16 {
        match self {
            Self::Dense => 0,
            Self::Routed => 1,
            Self::Shared => 2,
            Self::Unknown(tag) => tag,
        }
    }

    pub fn name(self) -> String {
        match self {
            Self::Dense => "dense".into(),
            Self::Routed => "routed".into(),
            Self::Shared => "shared".into(),
            Self::Unknown(tag) => format!("bank_kind_{tag}"),
        }
    }
}

/// One bank's declared geometry and region-schema count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BankDescriptor {
    pub bank_id: u16,
    pub kind: BankKind,
    pub num_entries: u32,
    pub input_dim: u32,
    pub intermediate_dim: u32,
    pub output_dim: u32,
    pub region_schema_count: u16,
    pub browse: BrowseMode,
}

impl BankDescriptor {
    /// Field order is normative (spec §6.2 draft-2): the two `u16` counts
    /// precede the four `u32` dims, so every `u32` sits on a 4-byte boundary
    /// within the 24-byte record.
    pub fn encode(&self, out: &mut Vec<u8>) {
        push_u16(out, self.bank_id);
        push_u16(out, self.kind.as_u16());
        push_u16(out, self.region_schema_count);
        push_u16(out, self.browse.to_flag_bits());
        push_u32(out, self.num_entries);
        push_u32(out, self.input_dim);
        push_u32(out, self.intermediate_dim);
        push_u32(out, self.output_dim);
    }

    pub fn decode(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < BANK_DESCRIPTOR_BYTES {
            return None;
        }
        Some(Self {
            bank_id: read_u16(bytes, 0)?,
            kind: BankKind::from_u16(read_u16(bytes, 2)?),
            region_schema_count: read_u16(bytes, 4)?,
            browse: BrowseMode::from_flags(read_u16(bytes, 6)?),
            num_entries: read_u32(bytes, 8)?,
            input_dim: read_u32(bytes, 12)?,
            intermediate_dim: read_u32(bytes, 16)?,
            output_dim: read_u32(bytes, 20)?,
        })
    }

    /// Slots this bank contributes to a segment's entry table, per entry.
    pub fn regions_per_entry(&self) -> usize {
        self.region_schema_count as usize
    }

    /// Walkable features this bank contributes to the flattened feature space
    /// (spec §15.1): one per intermediate row, per entry, across all entries —
    /// gate KNN selects across every expert with no router involved.
    pub fn walkable_feature_count(&self) -> u64 {
        u64::from(self.num_entries) * u64::from(self.intermediate_dim)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn routed_bank() -> BankDescriptor {
        BankDescriptor {
            bank_id: 0,
            kind: BankKind::Routed,
            num_entries: 896,
            input_dim: 3_584,
            intermediate_dim: 3_072,
            output_dim: 3_584,
            region_schema_count: 2,
            browse: BrowseMode::Direct,
        }
    }

    #[test]
    fn bank_kinds_round_trip() {
        for tag in 0u16..=2 {
            assert_eq!(BankKind::from_u16(tag).as_u16(), tag);
        }
    }

    #[test]
    fn unknown_bank_kind_is_preserved() {
        assert_eq!(BankKind::from_u16(55), BankKind::Unknown(55));
        assert_eq!(BankKind::Unknown(55).name(), "bank_kind_55");
    }

    #[test]
    fn registered_kind_names_are_stable() {
        assert_eq!(BankKind::Dense.name(), "dense");
        assert_eq!(BankKind::Routed.name(), "routed");
        assert_eq!(BankKind::Shared.name(), "shared");
    }

    #[test]
    fn descriptor_round_trips_through_bytes() {
        let bank = routed_bank();
        let mut buf = Vec::new();
        bank.encode(&mut buf);
        assert_eq!(buf.len(), BANK_DESCRIPTOR_BYTES);
        assert_eq!(BankDescriptor::decode(&buf), Some(bank));
    }

    #[test]
    fn dense_layer_is_the_single_entry_case() {
        let dense = BankDescriptor {
            bank_id: 3,
            kind: BankKind::Dense,
            num_entries: 1,
            input_dim: 2_304,
            intermediate_dim: 9_216,
            output_dim: 2_304,
            region_schema_count: 3,
            browse: BrowseMode::None,
        };
        let mut buf = Vec::new();
        dense.encode(&mut buf);
        assert_eq!(BankDescriptor::decode(&buf), Some(dense));
    }

    #[test]
    fn short_record_decodes_to_none() {
        let mut buf = Vec::new();
        routed_bank().encode(&mut buf);
        buf.pop();
        assert_eq!(BankDescriptor::decode(&buf), None);
    }

    #[test]
    fn regions_per_entry_follows_the_schema_count() {
        assert_eq!(routed_bank().regions_per_entry(), 2);
    }

    #[test]
    fn walkable_features_span_every_entry() {
        // 896 experts x 3072 intermediate rows — gate KNN sees all of them.
        assert_eq!(routed_bank().walkable_feature_count(), 2_752_512);
    }

    #[test]
    fn field_order_matches_the_normative_record() {
        // Spec §6.2 draft-2: bank_id, bank_kind, region_schema_count, flags,
        // then the four u32 dims. Pinned so a reordering is a test failure
        // rather than a silently mis-parsed sibling file.
        let mut buf = Vec::new();
        routed_bank().encode(&mut buf);
        assert_eq!(read_u16(&buf, 0), Some(0), "bank_id");
        assert_eq!(read_u16(&buf, 2), Some(1), "bank_kind = routed");
        assert_eq!(read_u16(&buf, 4), Some(2), "region_schema_count");
        assert_eq!(read_u16(&buf, 6), Some(1), "flags = browse direct");
        assert_eq!(read_u32(&buf, 8), Some(896), "num_entries");
        assert_eq!(read_u32(&buf, 12), Some(3_584), "input_dim");
        assert_eq!(read_u32(&buf, 16), Some(3_072), "intermediate_dim");
        assert_eq!(read_u32(&buf, 20), Some(3_584), "output_dim");
    }

    #[test]
    fn every_u32_field_is_four_byte_aligned() {
        for at in [8usize, 12, 16, 20] {
            assert_eq!(at % 4, 0, "u32 field at {at} is misaligned");
        }
    }

    #[test]
    fn walkable_feature_count_does_not_overflow_u32_products() {
        let big = BankDescriptor {
            num_entries: u32::MAX,
            intermediate_dim: 4,
            ..routed_bank()
        };
        assert_eq!(big.walkable_feature_count(), u64::from(u32::MAX) * 4);
    }
}
