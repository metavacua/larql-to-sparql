//! Segment descriptors (spec §6.3).
//!
//! A segment is a contiguous slice of one bank's entries stored in one file.
//! Single-file layers have exactly one segment covering `[0, num_entries)`.
//! Multi-segment layers repeat the header in every segment file; `index.json`
//! lists the files per logical layer so the loader never globs a directory.
//!
//! Segmentation is a physical storage parameter. It is invisible to model
//! semantics: the logical layer remains the stable unit, and which file an
//! expert lives in is not something a manifest or a kernel ever asks.

use super::consts::SEGMENT_DESCRIPTOR_BYTES;
use super::wire::{push_u16, push_u32, read_u16, read_u32};

/// The slice of a bank's entries carried by one file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SegmentDescriptor {
    pub bank_id: u16,
    pub segment_index: u16,
    pub first_entry: u32,
    pub entry_count: u32,
}

impl SegmentDescriptor {
    pub fn encode(&self, out: &mut Vec<u8>) {
        push_u16(out, self.bank_id);
        push_u16(out, self.segment_index);
        push_u32(out, self.first_entry);
        push_u32(out, self.entry_count);
    }

    pub fn decode(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < SEGMENT_DESCRIPTOR_BYTES {
            return None;
        }
        Some(Self {
            bank_id: read_u16(bytes, 0)?,
            segment_index: read_u16(bytes, 2)?,
            first_entry: read_u32(bytes, 4)?,
            entry_count: read_u32(bytes, 8)?,
        })
    }

    /// One past the last logical entry index this segment carries.
    pub fn end_entry(&self) -> u64 {
        u64::from(self.first_entry) + u64::from(self.entry_count)
    }

    /// Whether `logical_entry` falls inside this segment.
    pub fn contains(&self, logical_entry: u32) -> bool {
        let entry = u64::from(logical_entry);
        entry >= u64::from(self.first_entry) && entry < self.end_entry()
    }

    /// Position of `logical_entry` within this segment's entry table, or
    /// `None` if the entry lives in a different segment.
    pub fn local_index(&self, logical_entry: u32) -> Option<u32> {
        self.contains(logical_entry)
            .then(|| logical_entry - self.first_entry)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// K3 exact-Q6_K prior: two segments of 448 experts per routed layer.
    fn second_half_of_k3_layer() -> SegmentDescriptor {
        SegmentDescriptor {
            bank_id: 0,
            segment_index: 1,
            first_entry: 448,
            entry_count: 448,
        }
    }

    #[test]
    fn descriptor_round_trips_through_bytes() {
        let seg = second_half_of_k3_layer();
        let mut buf = Vec::new();
        seg.encode(&mut buf);
        assert_eq!(buf.len(), SEGMENT_DESCRIPTOR_BYTES);
        assert_eq!(SegmentDescriptor::decode(&buf), Some(seg));
    }

    #[test]
    fn short_record_decodes_to_none() {
        let mut buf = Vec::new();
        second_half_of_k3_layer().encode(&mut buf);
        buf.pop();
        assert_eq!(SegmentDescriptor::decode(&buf), None);
    }

    #[test]
    fn end_entry_is_exclusive() {
        assert_eq!(second_half_of_k3_layer().end_entry(), 896);
    }

    #[test]
    fn contains_covers_the_half_open_range() {
        let seg = second_half_of_k3_layer();
        assert!(!seg.contains(447));
        assert!(seg.contains(448));
        assert!(seg.contains(895));
        assert!(!seg.contains(896));
    }

    #[test]
    fn local_index_rebases_onto_the_segment() {
        let seg = second_half_of_k3_layer();
        assert_eq!(seg.local_index(448), Some(0));
        assert_eq!(seg.local_index(895), Some(447));
    }

    #[test]
    fn local_index_refuses_entries_from_another_segment() {
        let seg = second_half_of_k3_layer();
        assert_eq!(seg.local_index(0), None);
        assert_eq!(seg.local_index(447), None);
        assert_eq!(seg.local_index(896), None);
    }

    #[test]
    fn single_file_layer_covers_every_entry() {
        let seg = SegmentDescriptor {
            bank_id: 0,
            segment_index: 0,
            first_entry: 0,
            entry_count: 256,
        };
        assert!(seg.contains(0));
        assert!(seg.contains(255));
        assert!(!seg.contains(256));
        assert_eq!(seg.local_index(17), Some(17));
    }

    #[test]
    fn end_entry_does_not_overflow_at_u32_bounds() {
        let seg = SegmentDescriptor {
            bank_id: 0,
            segment_index: 0,
            first_entry: u32::MAX,
            entry_count: u32::MAX,
        };
        assert_eq!(seg.end_entry(), u64::from(u32::MAX) * 2);
    }

    #[test]
    fn empty_segment_contains_nothing() {
        let seg = SegmentDescriptor {
            bank_id: 0,
            segment_index: 0,
            first_entry: 10,
            entry_count: 0,
        };
        assert!(!seg.contains(10));
        assert_eq!(seg.local_index(10), None);
    }
}
