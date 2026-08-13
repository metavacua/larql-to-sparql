//! A validated description of one segment file, before any byte is written.
//!
//! Everything a LYRW v2 file declares is known before the payload exists:
//! which banks, which entries, which region schemas. Fixing that up front is
//! what lets the writer stream — it can emit the tables, reserve the entry
//! table, and then append one entry at a time without ever holding the whole
//! layer in memory.

use super::bank::BankDescriptor;
use super::error::Lyrw2Error;
use super::header::Lyrw2Header;
use super::region_schema::RegionSchema;
use super::segment::SegmentDescriptor;

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// The complete declaration of one segment file's tables.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Lyrw2Plan {
    pub header: Lyrw2Header,
    pub banks: Vec<BankDescriptor>,
    pub segments: Vec<SegmentDescriptor>,
    /// Region schemas per bank, parallel to `banks`.
    pub schemas: Vec<Vec<RegionSchema>>,
}

impl Lyrw2Plan {
    /// Build a plan for a single-bank, single-segment file covering every
    /// entry — the common dense and small-MoE case.
    pub fn single_segment(
        logical_layer: u32,
        bank: BankDescriptor,
        schemas: Vec<RegionSchema>,
    ) -> Self {
        let segment = SegmentDescriptor {
            bank_id: bank.bank_id,
            segment_index: 0,
            first_entry: 0,
            entry_count: bank.num_entries,
        };
        Self {
            header: Lyrw2Header::new(logical_layer, 1, 1),
            banks: vec![bank],
            segments: vec![segment],
            schemas: vec![schemas],
        }
    }

    /// Position of `bank_id` within `banks`.
    pub fn bank_ordinal(&self, bank_id: u16) -> Option<usize> {
        self.banks.iter().position(|b| b.bank_id == bank_id)
    }

    pub fn bank(&self, bank_id: u16) -> Option<&BankDescriptor> {
        self.bank_ordinal(bank_id).map(|i| &self.banks[i])
    }

    /// Check every internal consistency rule the reader will later rely on.
    ///
    /// This runs before writing, so a malformed plan fails at the API rather
    /// than producing a file that only fails when someone tries to serve it.
    pub fn validate(&self) -> Result<(), Lyrw2Error> {
        self.validate_declared_counts()?;
        self.validate_schemas()?;
        self.validate_segments()
    }

    fn validate_declared_counts(&self) -> Result<(), Lyrw2Error> {
        if self.header.num_banks as usize != self.banks.len() {
            return Err(Lyrw2Error::BankSchemaCountMismatch {
                bank_id: 0,
                declared: self.header.num_banks as usize,
                found: self.banks.len(),
            });
        }
        if self.schemas.len() != self.banks.len() {
            return Err(Lyrw2Error::BankSchemaCountMismatch {
                bank_id: 0,
                declared: self.banks.len(),
                found: self.schemas.len(),
            });
        }
        Ok(())
    }

    fn validate_schemas(&self) -> Result<(), Lyrw2Error> {
        for (bank, schemas) in self.banks.iter().zip(&self.schemas) {
            if bank.regions_per_entry() != schemas.len() {
                return Err(Lyrw2Error::BankSchemaCountMismatch {
                    bank_id: bank.bank_id,
                    declared: bank.regions_per_entry(),
                    found: schemas.len(),
                });
            }
            for (position, schema) in schemas.iter().enumerate() {
                if !schema.pairing_is_consistent() {
                    return Err(Lyrw2Error::InconsistentPairing {
                        bank_id: bank.bank_id,
                        schema_index: schema.schema_index,
                        packing: schema.packing.name(),
                        pair_id: schema.pair_id,
                    });
                }
                if schema.schema_index as usize != position {
                    return Err(Lyrw2Error::BankSchemaCountMismatch {
                        bank_id: bank.bank_id,
                        declared: position,
                        found: schema.schema_index as usize,
                    });
                }
            }
        }
        Ok(())
    }

    fn validate_segments(&self) -> Result<(), Lyrw2Error> {
        for segment in &self.segments {
            let bank = self.bank(segment.bank_id).ok_or({
                Lyrw2Error::SegmentNamesUnknownBank {
                    segment_index: segment.segment_index,
                    bank_id: segment.bank_id,
                }
            })?;
            if segment.end_entry() > u64::from(bank.num_entries) {
                return Err(Lyrw2Error::SegmentOutOfBounds {
                    bank_id: bank.bank_id,
                    segment_index: segment.segment_index,
                    first_entry: segment.first_entry,
                    end_entry: segment.end_entry(),
                    num_entries: bank.num_entries,
                });
            }
        }
        Ok(())
    }
}
