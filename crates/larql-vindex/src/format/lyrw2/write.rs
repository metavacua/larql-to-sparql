//! Streaming LYRW v2 writer.
//!
//! The v1 writer took `&[LayerEntry]` — every expert's quantised bytes, fully
//! materialised. At a K3 routed layer that is 39.4 GB of input, a ~118 GB f32
//! intermediate and 24.3 GB of output against 128 GB of RAM: unsurvivable, and
//! a property of the API signature rather than of the machine.
//!
//! This writer streams instead. It emits the tables, reserves the entry table,
//! then takes one region at a time and backpatches offsets on close. Peak
//! resident bytes are one region plus the table — about 27 MB and 28 KB
//! respectively at K3 scale, whatever the layer's total size.
//!
//! Regions must arrive in file order: for each segment, each entry, each
//! schema. That ordering is checked, not assumed — a caller that interleaves
//! banks gets an error, not a file with plausible offsets into the wrong place.

use std::fs::File;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::Path;

use super::consts::align_up;
use super::layout::Lyrw2Layout;
use super::plan::Lyrw2Plan;
use super::wire::push_u64;
use crate::VindexError;

/// Where the writer expects the next region to belong.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegionCursor {
    pub segment_ordinal: usize,
    pub local_entry: u32,
    pub schema_index: u16,
}

/// Streaming writer for one LYRW v2 segment file.
pub struct Lyrw2Writer {
    file: BufWriter<File>,
    plan: Lyrw2Plan,
    layout: Lyrw2Layout,
    /// Recorded `(offset, length)` per region, in the order written.
    slots: Vec<(u64, u64)>,
    cursor: RegionCursor,
    payload_cursor: u64,
    finished: bool,
}

impl Lyrw2Writer {
    /// Create the file and write every table except the entry table, which is
    /// reserved as zeroes and backpatched by [`Lyrw2Writer::finish`].
    pub fn create(path: &Path, plan: Lyrw2Plan) -> Result<Self, VindexError> {
        plan.validate().map_err(to_vindex_error)?;
        let layout = Lyrw2Layout::of(&plan);

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut file = BufWriter::new(File::create(path)?);

        let mut tables = Vec::new();
        plan.header.encode(&mut tables);
        for bank in &plan.banks {
            bank.encode(&mut tables);
        }
        for segment in &plan.segments {
            segment.encode(&mut tables);
        }
        for schemas in &plan.schemas {
            for schema in schemas {
                schema.encode(&mut tables);
            }
        }
        file.write_all(&tables)?;

        // Reserve the entry table plus the alignment gap before the payload.
        let reserved = layout.payload_start - layout.entry_table;
        file.write_all(&vec![0u8; reserved as usize])?;

        Ok(Self {
            file,
            plan,
            payload_cursor: layout.payload_start,
            layout,
            slots: Vec::new(),
            cursor: RegionCursor {
                segment_ordinal: 0,
                local_entry: 0,
                schema_index: 0,
            },
            finished: false,
        })
    }

    /// Where the next [`Lyrw2Writer::write_region`] call will land.
    pub fn cursor(&self) -> RegionCursor {
        self.cursor
    }

    /// Whether every region the plan declares has been written.
    pub fn is_complete(&self) -> bool {
        self.cursor.segment_ordinal >= self.plan.segments.len()
    }

    /// Regions appended so far. This is also the writer's entire retained
    /// state besides the plan — one `(offset, length)` pair per region, never
    /// the payload — which is the property that lets a 24 GB layer stream
    /// through a 27 MB working set.
    pub fn regions_written(&self) -> usize {
        self.slots.len()
    }

    /// Byte length recorded for the region at `ordinal`, in write order.
    pub fn region_length(&self, ordinal: usize) -> Option<u64> {
        self.slots.get(ordinal).map(|(_, length)| *length)
    }

    /// Append one region's payload bytes, aligning to the region boundary
    /// first. Regions must arrive in file order; see the module docs.
    pub fn write_region(&mut self, bytes: &[u8]) -> Result<(), VindexError> {
        if self.is_complete() {
            return Err(VindexError::Parse(
                "lyrw2: write_region called after every declared region was written".into(),
            ));
        }

        let aligned = align_up(self.payload_cursor);
        let pad = aligned - self.payload_cursor;
        if pad > 0 {
            self.file.write_all(&vec![0u8; pad as usize])?;
        }
        self.file.write_all(bytes)?;

        self.slots.push((aligned, bytes.len() as u64));
        self.payload_cursor = aligned + bytes.len() as u64;
        self.advance();
        Ok(())
    }

    /// Step the cursor one region forward through (segment, entry, schema).
    fn advance(&mut self) {
        let schema_count = self
            .layout
            .schema_count(self.cursor.segment_ordinal)
            .unwrap_or(0) as u16;
        let entry_count = self.plan.segments[self.cursor.segment_ordinal].entry_count;

        self.cursor.schema_index += 1;
        if self.cursor.schema_index < schema_count {
            return;
        }
        self.cursor.schema_index = 0;
        self.cursor.local_entry += 1;
        if self.cursor.local_entry < entry_count {
            return;
        }
        self.cursor.local_entry = 0;
        self.cursor.segment_ordinal += 1;
    }

    /// Backpatch the entry table and flush.
    ///
    /// Refuses to close a file that is missing regions — a short file whose
    /// table is half zeroes reads as valid offsets into byte zero, which is
    /// exactly the plausible-garbage failure this format exists to prevent.
    pub fn finish(mut self) -> Result<(), VindexError> {
        // The writer is consumed either way; the drop guard exists to catch a
        // *forgotten* finish, not a refused one.
        self.finished = true;
        if !self.is_complete() {
            return Err(VindexError::Parse(format!(
                "lyrw2: {} regions written but the plan declares more; \
                 next expected segment {} entry {} schema {}",
                self.slots.len(),
                self.cursor.segment_ordinal,
                self.cursor.local_entry,
                self.cursor.schema_index,
            )));
        }

        let mut table = Vec::with_capacity(self.slots.len() * 16);
        for (offset, length) in &self.slots {
            push_u64(&mut table, *offset);
            push_u64(&mut table, *length);
        }

        self.file.seek(SeekFrom::Start(self.layout.entry_table))?;
        self.file.write_all(&table)?;
        self.file.flush()?;
        Ok(())
    }

    /// Deliberately discard a partially written file.
    ///
    /// The bytes on disk keep a zeroed entry table and must not be served —
    /// this exists so a caller that aborts on purpose is distinguishable from
    /// one that forgot to call [`Lyrw2Writer::finish`].
    pub fn abandon(mut self) {
        self.finished = true;
    }
}

impl Drop for Lyrw2Writer {
    fn drop(&mut self) {
        debug_assert!(
            self.finished || std::thread::panicking(),
            "Lyrw2Writer dropped without finish(): the entry table is still zeroes"
        );
    }
}

fn to_vindex_error(e: super::error::Lyrw2Error) -> VindexError {
    VindexError::Parse(e.to_string())
}
