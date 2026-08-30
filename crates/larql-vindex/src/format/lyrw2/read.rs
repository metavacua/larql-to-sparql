//! LYRW v2 reader — parses tables and resolves region byte ranges.
//!
//! The reader never copies payload bytes. It resolves a `(bank, entry, role)`
//! request to an offset and length, and the caller decides whether to mmap,
//! stride, or dequantise. Untouched `up`/`down` pages then cost nothing, which
//! is what lets a browse sweep over a full-fat index read only gate bytes.
//!
//! Every bounds check here fails closed with the bank, entry and role named.
//! An offset that is merely *inside the file* is not evidence of correctness —
//! it is the exact shape of the mis-strided-table bug this format guards against.

use super::bank::BankDescriptor;
use super::consts::PAIR_ID_UNPAIRED;
use super::consts::{
    BANK_DESCRIPTOR_BYTES, HEADER_BYTES, REGION_SCHEMA_BYTES, SEGMENT_DESCRIPTOR_BYTES,
};
use super::error::Lyrw2Error;
use super::header::Lyrw2Header;
use super::layout::Lyrw2Layout;
use super::plan::Lyrw2Plan;
use super::region_role::RegionRole;
use super::region_schema::RegionSchema;
use super::segment::SegmentDescriptor;
use super::wire::read_u64;

/// A resolved region: where its bytes are, and what they mean.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolvedRegion {
    pub offset: u64,
    pub length: u64,
    pub schema: RegionSchema,
}

/// Parsed view over one LYRW v2 segment file's bytes.
#[derive(Debug, Clone)]
pub struct Lyrw2Reader<'a> {
    bytes: &'a [u8],
    plan: Lyrw2Plan,
    layout: Lyrw2Layout,
}

impl<'a> Lyrw2Reader<'a> {
    /// Parse every table. Payload bytes are not touched.
    pub fn parse(bytes: &'a [u8]) -> Result<Self, Lyrw2Error> {
        let header = Lyrw2Header::decode(bytes)?;

        let banks = Self::read_banks(bytes, &header)?;
        let segments = Self::read_segments(bytes, &header)?;
        let schemas = Self::read_schemas(bytes, &header, &banks)?;

        let plan = Lyrw2Plan {
            header,
            banks,
            segments,
            schemas,
        };
        plan.validate()?;
        let layout = Lyrw2Layout::of(&plan);
        Ok(Self {
            bytes,
            plan,
            layout,
        })
    }

    fn read_banks(bytes: &[u8], header: &Lyrw2Header) -> Result<Vec<BankDescriptor>, Lyrw2Error> {
        let count = usize::from(header.num_banks);
        let mut banks = Vec::with_capacity(count);
        for i in 0..count {
            let at = HEADER_BYTES + i * BANK_DESCRIPTOR_BYTES;
            let slice = bytes
                .get(at..at + BANK_DESCRIPTOR_BYTES)
                .ok_or(Lyrw2Error::Truncated {
                    what: "bank descriptor table",
                    at,
                    need: BANK_DESCRIPTOR_BYTES,
                    have: bytes.len().saturating_sub(at),
                })?;
            banks.push(BankDescriptor::decode(slice).ok_or(Lyrw2Error::Truncated {
                what: "bank descriptor",
                at,
                need: BANK_DESCRIPTOR_BYTES,
                have: slice.len(),
            })?);
        }
        Ok(banks)
    }

    fn read_segments(
        bytes: &[u8],
        header: &Lyrw2Header,
    ) -> Result<Vec<SegmentDescriptor>, Lyrw2Error> {
        let base = HEADER_BYTES + usize::from(header.num_banks) * BANK_DESCRIPTOR_BYTES;
        let count = usize::from(header.num_segments);
        let mut segments = Vec::with_capacity(count);
        for i in 0..count {
            let at = base + i * SEGMENT_DESCRIPTOR_BYTES;
            let slice =
                bytes
                    .get(at..at + SEGMENT_DESCRIPTOR_BYTES)
                    .ok_or(Lyrw2Error::Truncated {
                        what: "segment descriptor table",
                        at,
                        need: SEGMENT_DESCRIPTOR_BYTES,
                        have: bytes.len().saturating_sub(at),
                    })?;
            segments.push(
                SegmentDescriptor::decode(slice).ok_or(Lyrw2Error::Truncated {
                    what: "segment descriptor",
                    at,
                    need: SEGMENT_DESCRIPTOR_BYTES,
                    have: slice.len(),
                })?,
            );
        }
        Ok(segments)
    }

    fn read_schemas(
        bytes: &[u8],
        header: &Lyrw2Header,
        banks: &[BankDescriptor],
    ) -> Result<Vec<Vec<RegionSchema>>, Lyrw2Error> {
        let mut at = HEADER_BYTES
            + usize::from(header.num_banks) * BANK_DESCRIPTOR_BYTES
            + usize::from(header.num_segments) * SEGMENT_DESCRIPTOR_BYTES;
        let mut all = Vec::with_capacity(banks.len());
        for bank in banks {
            let mut schemas = Vec::with_capacity(bank.regions_per_entry());
            for _ in 0..bank.regions_per_entry() {
                let slice =
                    bytes
                        .get(at..at + REGION_SCHEMA_BYTES)
                        .ok_or(Lyrw2Error::Truncated {
                            what: "region schema table",
                            at,
                            need: REGION_SCHEMA_BYTES,
                            have: bytes.len().saturating_sub(at),
                        })?;
                schemas.push(RegionSchema::decode(slice).ok_or(Lyrw2Error::Truncated {
                    what: "region schema",
                    at,
                    need: REGION_SCHEMA_BYTES,
                    have: slice.len(),
                })?);
                at += REGION_SCHEMA_BYTES;
            }
            all.push(schemas);
        }
        Ok(all)
    }

    pub fn header(&self) -> &Lyrw2Header {
        &self.plan.header
    }

    pub fn banks(&self) -> &[BankDescriptor] {
        &self.plan.banks
    }

    pub fn segments(&self) -> &[SegmentDescriptor] {
        &self.plan.segments
    }

    pub fn schemas_for(&self, bank_id: u16) -> Option<&[RegionSchema]> {
        self.plan
            .bank_ordinal(bank_id)
            .map(|i| self.plan.schemas[i].as_slice())
    }

    pub fn layout(&self) -> &Lyrw2Layout {
        &self.layout
    }

    /// Ordinal of the segment holding `logical_entry` of `bank_id`.
    fn segment_holding(&self, bank_id: u16, logical_entry: u32) -> Option<(usize, u32)> {
        self.plan.segments.iter().enumerate().find_map(|(i, s)| {
            (s.bank_id == bank_id)
                .then(|| s.local_index(logical_entry))
                .flatten()
                .map(|local| (i, local))
        })
    }

    /// Resolve one region's byte range, or say precisely why it is absent.
    ///
    /// `logical_entry` is the entry's index within the *bank*, not within this
    /// file — a caller holding one segment of a split layer passes the same
    /// index it would for a single-file layer, and gets `None` if the entry
    /// lives in a sibling segment.
    pub fn resolve(
        &self,
        bank_id: u16,
        logical_entry: u32,
        role: RegionRole,
    ) -> Result<Option<ResolvedRegion>, Lyrw2Error> {
        let Some(schema) = self.schema_for_role(bank_id, role)? else {
            return Ok(None);
        };
        self.resolve_schema(bank_id, logical_entry, schema)
    }

    /// The unique schema carrying `role`, or an error if more than one does.
    ///
    /// **Refuses rather than picking.** Before split-scale banks every role in
    /// a bank was unique, so `find` was total; a bank with two `Scales`
    /// regions makes it a silent default that hands back gate/up's exponents
    /// when down's were meant — bytes from the wrong region, which is the
    /// exact failure this format's diagnostics exist to prevent. Callers that
    /// need a duplicated role must say *which* by naming its pair
    /// ([`Self::resolve_paired`]).
    fn schema_for_role(
        &self,
        bank_id: u16,
        role: RegionRole,
    ) -> Result<Option<RegionSchema>, Lyrw2Error> {
        let Some(schemas) = self.schemas_for(bank_id) else {
            return Ok(None);
        };
        let mut matching = schemas.iter().copied().filter(|s| s.role == role);
        let Some(first) = matching.next() else {
            return Ok(None);
        };
        let count = 1 + matching.count();
        if count > 1 {
            return Err(Lyrw2Error::AmbiguousRole {
                bank_id,
                role: role.name(),
                count,
            });
        }
        Ok(Some(first))
    }

    /// Resolve the region that `role`'s pairing binds to it.
    ///
    /// `pair_id` is the authority, not ordinal position. A caller asks for
    /// "the `Scales` partnered with `Down`" rather than "the second `Scales`",
    /// so a writer free to emit regions in any order cannot silently
    /// re-associate a payload with the wrong exponents.
    ///
    /// ```text
    /// GateUpFused ── pair_id 1 ── Scales
    /// Down        ── pair_id 2 ── Scales
    /// ```
    ///
    /// `source_role` must itself be unique in the bank; the partner must be
    /// the unique region with the same `pair_id` and `partner_role`.
    pub fn resolve_paired(
        &self,
        bank_id: u16,
        logical_entry: u32,
        source_role: RegionRole,
        partner_role: RegionRole,
    ) -> Result<Option<ResolvedRegion>, Lyrw2Error> {
        let Some(source) = self.schema_for_role(bank_id, source_role)? else {
            return Ok(None);
        };
        if source.pair_id == PAIR_ID_UNPAIRED {
            return Err(Lyrw2Error::NotPaired {
                bank_id,
                role: source_role.name(),
                partner_role: partner_role.name(),
            });
        }
        // `schemas_for` already answered above, so this cannot be None.
        let schemas = self.schemas_for(bank_id).unwrap_or(&[]);
        let partners: Vec<RegionSchema> = schemas
            .iter()
            .copied()
            .filter(|s| s.role == partner_role && s.pair_id == source.pair_id)
            .collect();
        match partners.len() {
            0 => Err(Lyrw2Error::MissingPartner {
                bank_id,
                role: source_role.name(),
                pair_id: source.pair_id,
                partner_role: partner_role.name(),
            }),
            1 => self.resolve_schema(bank_id, logical_entry, partners[0]),
            count => Err(Lyrw2Error::AmbiguousPartner {
                bank_id,
                partner_role: partner_role.name(),
                pair_id: source.pair_id,
                count,
            }),
        }
    }

    /// Resolve a region by its **schema index**, bypassing role lookup.
    ///
    /// For callers that already hold the schema table and want to visit
    /// every declared region — a verifier, or a writer checking its own
    /// output. Role-agnostic by construction, so a bank with duplicated
    /// roles is enumerable without the ambiguity that role lookup refuses.
    pub fn resolve_by_schema_index(
        &self,
        bank_id: u16,
        logical_entry: u32,
        schema_index: u16,
    ) -> Result<Option<ResolvedRegion>, Lyrw2Error> {
        let Some(schemas) = self.schemas_for(bank_id) else {
            return Ok(None);
        };
        let Some(schema) = schemas
            .iter()
            .copied()
            .find(|s| s.schema_index == schema_index)
        else {
            return Ok(None);
        };
        self.resolve_schema(bank_id, logical_entry, schema)
    }

    /// Byte range of one already-selected schema's region.
    ///
    /// Split out so role lookup and pair lookup share one offset/bounds path
    /// — two copies of this arithmetic is how a reader ends up checking
    /// bounds on one route and not the other.
    fn resolve_schema(
        &self,
        bank_id: u16,
        logical_entry: u32,
        schema: RegionSchema,
    ) -> Result<Option<ResolvedRegion>, Lyrw2Error> {
        let role = schema.role;
        let Some((segment_ordinal, local_entry)) = self.segment_holding(bank_id, logical_entry)
        else {
            return Ok(None);
        };
        let Some(slot) = self
            .layout
            .entry_slot(segment_ordinal, local_entry, schema.schema_index)
        else {
            return Ok(None);
        };

        let slot = slot as usize;
        let offset = read_u64(self.bytes, slot).ok_or(Lyrw2Error::Truncated {
            what: "entry table",
            at: slot,
            need: 16,
            have: self.bytes.len().saturating_sub(slot),
        })?;
        let length = read_u64(self.bytes, slot + 8).ok_or(Lyrw2Error::Truncated {
            what: "entry table",
            at: slot + 8,
            need: 8,
            have: self.bytes.len().saturating_sub(slot + 8),
        })?;

        let end = offset.saturating_add(length);
        if end > self.bytes.len() as u64 {
            return Err(Lyrw2Error::RegionOutOfBounds {
                bank_id,
                entry: logical_entry,
                role: role.name(),
                offset,
                end,
                file_len: self.bytes.len() as u64,
            });
        }
        Ok(Some(ResolvedRegion {
            offset,
            length,
            schema,
        }))
    }

    /// Borrow one region's payload bytes.
    pub fn region_bytes(
        &self,
        bank_id: u16,
        logical_entry: u32,
        role: RegionRole,
    ) -> Result<Option<&'a [u8]>, Lyrw2Error> {
        let Some(region) = self.resolve(bank_id, logical_entry, role)? else {
            return Ok(None);
        };
        Ok(Some(self.borrow(&region)))
    }

    /// Borrow the bytes of the region `source_role`'s pairing binds to it.
    ///
    /// The addressing a split-scale bank needs: "down's exponents", not
    /// "the second `Scales` region". See [`Self::resolve_paired`].
    pub fn paired_region_bytes(
        &self,
        bank_id: u16,
        logical_entry: u32,
        source_role: RegionRole,
        partner_role: RegionRole,
    ) -> Result<Option<&'a [u8]>, Lyrw2Error> {
        let Some(region) =
            self.resolve_paired(bank_id, logical_entry, source_role, partner_role)?
        else {
            return Ok(None);
        };
        Ok(Some(self.borrow(&region)))
    }

    fn borrow(&self, region: &ResolvedRegion) -> &'a [u8] {
        let start = region.offset as usize;
        let end = start + region.length as usize;
        &self.bytes[start..end]
    }
}
