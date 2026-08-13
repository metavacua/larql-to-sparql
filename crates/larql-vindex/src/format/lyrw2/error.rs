//! LYRW v2 container diagnostics.
//!
//! Every variant names the thing that was wrong and the thing that was
//! expected. That is a V2-0 acceptance requirement, not politeness: the
//! failure mode this format has to design against is a reader that parses a
//! table at the wrong stride, lands on offsets that are still inside the file,
//! and hands back plausible bytes from the wrong expert. A wrong answer is
//! worse than a refusal, so every check here fails closed and says why.

use super::consts::FORMAT_VERSION;

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Lyrw2Error {
    #[error("not a LYRW file: magic was {found:#010x}, expected {expected:#010x}")]
    NotLyrw { found: u32, expected: u32 },

    #[error(
        "LYRW format_version {found} requires the VINDEX{required_generation} loader; \
         this reader implements version {supported}"
    )]
    WrongGeneration {
        found: u32,
        supported: u32,
        required_generation: u32,
    },

    #[error("truncated {what}: need {need} bytes at offset {at}, file has {have}")]
    Truncated {
        what: &'static str,
        at: usize,
        need: usize,
        have: usize,
    },

    #[error("bank {bank_id} declares {declared} region schemas but {found} were readable")]
    BankSchemaCountMismatch {
        bank_id: u16,
        declared: usize,
        found: usize,
    },

    #[error("segment {segment_index} names bank {bank_id}, which this file does not declare")]
    SegmentNamesUnknownBank { segment_index: u16, bank_id: u16 },

    #[error(
        "segment {segment_index} of bank {bank_id} covers entries {first_entry}..{end_entry}, \
         past the bank's {num_entries} entries"
    )]
    SegmentOutOfBounds {
        bank_id: u16,
        segment_index: u16,
        first_entry: u32,
        end_entry: u64,
        num_entries: u32,
    },

    #[error(
        "bank {bank_id} schema {schema_index} ({packing}) declares pair_id {pair_id}, \
         which is inconsistent with its packing"
    )]
    InconsistentPairing {
        bank_id: u16,
        schema_index: u16,
        packing: String,
        pair_id: u16,
    },

    #[error(
        "bank {bank_id} region {role} of entry {entry} spans {offset}..{end}, \
         past the file's {file_len} bytes"
    )]
    RegionOutOfBounds {
        bank_id: u16,
        entry: u32,
        role: String,
        offset: u64,
        end: u64,
        file_len: u64,
    },
}

impl Lyrw2Error {
    /// Build the generation error, deriving which loader the caller needs from
    /// the version actually found. A LYRW v1 file must produce "requires the
    /// VINDEX2 loader", never a parse error.
    pub fn wrong_generation(found: u32) -> Self {
        Self::WrongGeneration {
            found,
            supported: FORMAT_VERSION,
            required_generation: container_generation_for(found),
        }
    }
}

/// LYRW `format_version` N is carried by container generation N + 1.
///
/// The two sequences are offset because LYRW started at 1 while `index.json`
/// was already at 2. Reporting the LYRW number alone would name a version no
/// CLI flag or directory is labelled with, so the error translates it into the
/// loader the caller actually has to reach for.
const fn container_generation_for(lyrw_format_version: u32) -> u32 {
    lyrw_format_version + 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::lyrw2::consts::MAGIC;

    #[test]
    fn not_lyrw_reports_both_magics() {
        let e = Lyrw2Error::NotLyrw {
            found: 0xDEAD_BEEF,
            expected: MAGIC,
        };
        let s = e.to_string();
        assert!(s.contains("deadbeef"), "{s}");
        assert!(s.contains("not a LYRW file"), "{s}");
    }

    #[test]
    fn wrong_generation_names_the_loader_the_caller_needs() {
        let s = Lyrw2Error::wrong_generation(1).to_string();
        assert!(s.contains("VINDEX2"), "{s}");
        assert!(s.contains("version 2"), "{s}");
    }

    #[test]
    fn wrong_generation_names_a_future_loader_too() {
        // LYRW format_version 3 would be carried by a VINDEX4 container —
        // the offset applies forwards as well as backwards.
        let s = Lyrw2Error::wrong_generation(3).to_string();
        assert!(s.contains("VINDEX4"), "{s}");
    }

    #[test]
    fn the_generation_offset_holds_across_the_range() {
        for lyrw in 1u32..=4 {
            assert_eq!(container_generation_for(lyrw), lyrw + 1);
        }
    }

    #[test]
    fn truncation_reports_what_where_and_how_short() {
        let s = Lyrw2Error::Truncated {
            what: "bank descriptor table",
            at: 24,
            need: 48,
            have: 30,
        }
        .to_string();
        assert!(s.contains("bank descriptor table"), "{s}");
        assert!(s.contains("48"), "{s}");
        assert!(s.contains("30"), "{s}");
    }

    #[test]
    fn schema_count_mismatch_names_the_bank() {
        let s = Lyrw2Error::BankSchemaCountMismatch {
            bank_id: 7,
            declared: 3,
            found: 2,
        }
        .to_string();
        assert!(s.contains("bank 7"), "{s}");
        assert!(s.contains('3'), "{s}");
    }

    #[test]
    fn segment_naming_an_unknown_bank_is_reported() {
        let s = Lyrw2Error::SegmentNamesUnknownBank {
            segment_index: 1,
            bank_id: 9,
        }
        .to_string();
        assert!(s.contains("segment 1"), "{s}");
        assert!(s.contains("bank 9"), "{s}");
    }

    #[test]
    fn segment_out_of_bounds_shows_the_overrun() {
        let s = Lyrw2Error::SegmentOutOfBounds {
            bank_id: 0,
            segment_index: 1,
            first_entry: 448,
            end_entry: 1_024,
            num_entries: 896,
        }
        .to_string();
        assert!(s.contains("448"), "{s}");
        assert!(s.contains("896"), "{s}");
    }

    #[test]
    fn inconsistent_pairing_names_bank_schema_and_packing() {
        let s = Lyrw2Error::InconsistentPairing {
            bank_id: 2,
            schema_index: 1,
            packing: "blocks_values".into(),
            pair_id: 0xFFFF,
        }
        .to_string();
        assert!(s.contains("bank 2"), "{s}");
        assert!(s.contains("blocks_values"), "{s}");
    }

    #[test]
    fn region_out_of_bounds_names_the_role_and_entry() {
        let s = Lyrw2Error::RegionOutOfBounds {
            bank_id: 0,
            entry: 511,
            role: "down".into(),
            offset: 1_000,
            end: 2_000,
            file_len: 1_500,
        }
        .to_string();
        assert!(s.contains("entry 511"), "{s}");
        assert!(s.contains("down"), "{s}");
        assert!(s.contains("1500"), "{s}");
    }
}
