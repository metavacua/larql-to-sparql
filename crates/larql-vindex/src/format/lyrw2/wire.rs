//! Little-endian field access over fixed-width descriptor slices.
//!
//! Every descriptor in LYRW v2 is a fixed-size record read from a known
//! offset, so the whole wire layer needs exactly two readers and two writers.
//! Keeping them here means no descriptor module hand-rolls a `try_into`, which
//! is where off-by-two field offsets come from.

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// Read a `u16` at `at` within `bytes`, or `None` if the record is short.
pub fn read_u16(bytes: &[u8], at: usize) -> Option<u16> {
    let end = at.checked_add(2)?;
    let slice = bytes.get(at..end)?;
    Some(u16::from_le_bytes([slice[0], slice[1]]))
}

/// Read a `u32` at `at` within `bytes`, or `None` if the record is short.
pub fn read_u32(bytes: &[u8], at: usize) -> Option<u32> {
    let end = at.checked_add(4)?;
    let slice = bytes.get(at..end)?;
    Some(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

/// Read a `u64` at `at` within `bytes`, or `None` if the record is short.
pub fn read_u64(bytes: &[u8], at: usize) -> Option<u64> {
    let end = at.checked_add(8)?;
    let slice = bytes.get(at..end)?;
    let mut buf = [0u8; 8];
    buf.copy_from_slice(slice);
    Some(u64::from_le_bytes(buf))
}

pub fn push_u16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_le_bytes());
}

pub fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

pub fn push_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn u16_round_trips_at_offset() {
        let mut buf = vec![0xAA, 0xBB];
        push_u16(&mut buf, 0x1234);
        assert_eq!(read_u16(&buf, 2), Some(0x1234));
    }

    #[test]
    fn u32_round_trips_at_offset() {
        let mut buf = vec![0u8; 3];
        push_u32(&mut buf, 0xDEAD_BEEF);
        assert_eq!(read_u32(&buf, 3), Some(0xDEAD_BEEF));
    }

    #[test]
    fn u64_round_trips_at_offset() {
        let mut buf = vec![0u8; 1];
        push_u64(&mut buf, u64::MAX - 7);
        assert_eq!(read_u64(&buf, 1), Some(u64::MAX - 7));
    }

    #[test]
    fn short_records_return_none_rather_than_panicking() {
        let buf = [0u8; 3];
        assert_eq!(read_u16(&buf, 2), None);
        assert_eq!(read_u32(&buf, 0), None);
        assert_eq!(read_u64(&buf, 0), None);
    }

    #[test]
    fn offset_overflow_returns_none() {
        let buf = [0u8; 8];
        assert_eq!(read_u16(&buf, usize::MAX), None);
        assert_eq!(read_u32(&buf, usize::MAX), None);
        assert_eq!(read_u64(&buf, usize::MAX), None);
    }

    #[test]
    fn encoding_is_little_endian() {
        let mut buf = Vec::new();
        push_u32(&mut buf, 1);
        assert_eq!(buf, vec![1, 0, 0, 0]);
    }
}
