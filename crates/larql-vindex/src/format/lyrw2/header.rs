//! LYRW v2 file header (spec §6.1).
//!
//! The magic is shared with v1 deliberately. It buys forensics, not
//! compatibility: a v1 reader recognises the file, reaches the version field,
//! and refuses precisely — "requires the VINDEX3 loader" — instead of parsing
//! a v2 table at the v1 stride and returning bytes from the wrong expert.

use super::consts::{FORMAT_VERSION, HEADER_BYTES, HEADER_FLAG_MULTI_SEGMENT, MAGIC};
use super::error::Lyrw2Error;
use super::wire::{push_u16, push_u32, read_u16, read_u32};

/// Fixed-size prelude describing what tables follow.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Lyrw2Header {
    pub logical_layer: u32,
    pub num_banks: u16,
    pub num_segments: u16,
    pub flags: u32,
}

impl Lyrw2Header {
    pub fn new(logical_layer: u32, num_banks: u16, num_segments: u16) -> Self {
        Self {
            logical_layer,
            num_banks,
            num_segments,
            flags: 0,
        }
    }

    /// Mark this file as one segment of a multi-segment logical layer.
    pub fn with_multi_segment(mut self, multi: bool) -> Self {
        if multi {
            self.flags |= HEADER_FLAG_MULTI_SEGMENT;
        } else {
            self.flags &= !HEADER_FLAG_MULTI_SEGMENT;
        }
        self
    }

    /// Whether the logical layer spans more files than this one.
    pub fn is_multi_segment(&self) -> bool {
        self.flags & HEADER_FLAG_MULTI_SEGMENT != 0
    }

    pub fn encode(&self, out: &mut Vec<u8>) {
        push_u32(out, MAGIC);
        push_u32(out, FORMAT_VERSION);
        push_u32(out, self.logical_layer);
        push_u16(out, self.num_banks);
        push_u16(out, self.num_segments);
        push_u32(out, self.flags);
        push_u32(out, 0); // reserved
    }

    /// Parse the header, refusing anything that is not a v2 LYRW file.
    ///
    /// The two refusals are ordered: magic first, so a wholly unrelated file
    /// is not reported as a version problem, then version, so a sibling
    /// generation is named rather than mis-parsed.
    pub fn decode(bytes: &[u8]) -> Result<Self, Lyrw2Error> {
        let need = HEADER_BYTES;
        if bytes.len() < need {
            return Err(Lyrw2Error::Truncated {
                what: "header",
                at: 0,
                need,
                have: bytes.len(),
            });
        }
        let magic = read_u32(bytes, 0).ok_or(Lyrw2Error::Truncated {
            what: "header magic",
            at: 0,
            need,
            have: bytes.len(),
        })?;
        if magic != MAGIC {
            return Err(Lyrw2Error::NotLyrw {
                found: magic,
                expected: MAGIC,
            });
        }
        let version = read_u32(bytes, 4).ok_or(Lyrw2Error::Truncated {
            what: "header format_version",
            at: 4,
            need,
            have: bytes.len(),
        })?;
        if version != FORMAT_VERSION {
            return Err(Lyrw2Error::wrong_generation(version));
        }
        Ok(Self {
            logical_layer: read_u32(bytes, 8).unwrap_or_default(),
            num_banks: read_u16(bytes, 12).unwrap_or_default(),
            num_segments: read_u16(bytes, 14).unwrap_or_default(),
            flags: read_u32(bytes, 16).unwrap_or_default(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Lyrw2Header {
        Lyrw2Header::new(37, 2, 1)
    }

    #[test]
    fn header_round_trips_through_bytes() {
        let header = sample();
        let mut buf = Vec::new();
        header.encode(&mut buf);
        assert_eq!(buf.len(), HEADER_BYTES);
        assert_eq!(Lyrw2Header::decode(&buf), Ok(header));
    }

    #[test]
    fn multi_segment_flag_round_trips() {
        let header = sample().with_multi_segment(true);
        assert!(header.is_multi_segment());
        let mut buf = Vec::new();
        header.encode(&mut buf);
        assert_eq!(Lyrw2Header::decode(&buf), Ok(header));
    }

    #[test]
    fn multi_segment_flag_can_be_cleared() {
        let header = sample().with_multi_segment(true).with_multi_segment(false);
        assert!(!header.is_multi_segment());
        assert_eq!(header.flags, 0);
    }

    #[test]
    fn writer_stamps_the_current_version() {
        let mut buf = Vec::new();
        sample().encode(&mut buf);
        assert_eq!(read_u32(&buf, 4), Some(FORMAT_VERSION));
    }

    #[test]
    fn reserved_field_is_written_as_zero() {
        let mut buf = Vec::new();
        sample().encode(&mut buf);
        assert_eq!(read_u32(&buf, 20), Some(0));
    }

    #[test]
    fn foreign_magic_is_reported_as_not_lyrw() {
        let mut buf = Vec::new();
        sample().encode(&mut buf);
        buf[0..4].copy_from_slice(&0x1234_5678u32.to_le_bytes());
        assert!(matches!(
            Lyrw2Header::decode(&buf),
            Err(Lyrw2Error::NotLyrw { .. })
        ));
    }

    #[test]
    fn a_v1_file_is_refused_by_generation_not_by_parse_error() {
        // The exact scenario §6.6 designs for: same magic, older version.
        let mut buf = Vec::new();
        sample().encode(&mut buf);
        buf[4..8].copy_from_slice(&1u32.to_le_bytes());
        let err = Lyrw2Header::decode(&buf).unwrap_err();
        assert!(matches!(err, Lyrw2Error::WrongGeneration { found: 1, .. }));
        assert!(err.to_string().contains("VINDEX2"), "{err}");
    }

    #[test]
    fn a_future_version_is_refused_rather_than_guessed_at() {
        let mut buf = Vec::new();
        sample().encode(&mut buf);
        buf[4..8].copy_from_slice(&3u32.to_le_bytes());
        assert!(matches!(
            Lyrw2Header::decode(&buf),
            Err(Lyrw2Error::WrongGeneration { found: 3, .. })
        ));
    }

    #[test]
    fn short_header_is_truncation_not_a_magic_failure() {
        let mut buf = Vec::new();
        sample().encode(&mut buf);
        buf.truncate(HEADER_BYTES - 1);
        assert!(matches!(
            Lyrw2Header::decode(&buf),
            Err(Lyrw2Error::Truncated { what: "header", .. })
        ));
    }

    #[test]
    fn empty_input_is_truncation() {
        assert!(matches!(
            Lyrw2Header::decode(&[]),
            Err(Lyrw2Error::Truncated { .. })
        ));
    }
}
