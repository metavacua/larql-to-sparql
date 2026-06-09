//! Shared little-endian wire primitives for the DSv4 vindex blob formats
//! (`dsv4-vindex-extraction`).
//!
//! The attention (`dsv4_attn.bin`), HCA/indexer (`dsv4_hca.bin`), and mHC
//! (`dsv4_mhc.bin`) writers all serialize the same building blocks — raw
//! quantized tensors ([`RawExpertTensor`] passthrough) and f32 vectors —
//! behind a 4-byte magic + u16 version header, with bounds-checked reads.
//! Centralized here so a fix lands once and every format inherits it.

use super::dsv4_gguf_reader::RawExpertTensor;

/// Error decoding any DSv4 vindex blob. Variant shapes are stable across
/// the attention/HCA/mHC formats (each aliases this type).
#[derive(Debug, PartialEq, Eq)]
pub enum DsV4VindexWireError {
    /// Leading magic bytes didn't match the format's expected tag.
    BadMagic([u8; 4]),
    /// Version field is newer/older than this reader understands.
    UnsupportedVersion(u16),
    /// The buffer ended mid-field.
    Truncated {
        what: &'static str,
        need: usize,
        have: usize,
    },
}

impl std::fmt::Display for DsV4VindexWireError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DsV4VindexWireError::BadMagic(m) => write!(f, "bad magic {m:?}"),
            DsV4VindexWireError::UnsupportedVersion(v) => write!(f, "unsupported version {v}"),
            DsV4VindexWireError::Truncated { what, need, have } => {
                write!(f, "truncated reading {what}: need {need}, have {have}")
            }
        }
    }
}
impl std::error::Error for DsV4VindexWireError {}

// ── encode ──────────────────────────────────────────────────────────────

/// `magic[4] | version:u16`.
pub(crate) fn put_header(out: &mut Vec<u8>, magic: [u8; 4], version: u16) {
    out.extend_from_slice(&magic);
    out.extend_from_slice(&version.to_le_bytes());
}

pub(crate) fn put_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

/// `tensor_type:u32 | rows:u32 | cols:u32 | byte_len:u64 | bytes`.
pub(crate) fn put_raw(out: &mut Vec<u8>, t: &RawExpertTensor) {
    put_u32(out, t.tensor_type);
    put_u32(out, t.rows as u32);
    put_u32(out, t.cols as u32);
    out.extend_from_slice(&(t.bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(&t.bytes);
}

/// `len:u32 | len*f32`.
pub(crate) fn put_f32_vec(out: &mut Vec<u8>, v: &[f32]) {
    put_u32(out, v.len() as u32);
    for &x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
}

/// `present:u8 | [f32_vec]`.
pub(crate) fn put_opt_f32_vec(out: &mut Vec<u8>, v: &Option<Vec<f32>>) {
    match v {
        Some(s) => {
            out.push(1);
            put_f32_vec(out, s);
        }
        None => out.push(0),
    }
}

/// `len:u32 | len*i32`.
pub(crate) fn put_i32_vec(out: &mut Vec<u8>, v: &[i32]) {
    put_u32(out, v.len() as u32);
    for &x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
}

/// `present:u8 | [i32_vec]`.
pub(crate) fn put_opt_i32_vec(out: &mut Vec<u8>, v: &Option<Vec<i32>>) {
    match v {
        Some(s) => {
            out.push(1);
            put_i32_vec(out, s);
        }
        None => out.push(0),
    }
}

// ── decode ──────────────────────────────────────────────────────────────

pub(crate) struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    pub(crate) fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn take(&mut self, n: usize, what: &'static str) -> Result<&'a [u8], DsV4VindexWireError> {
        let have = self.buf.len().saturating_sub(self.pos);
        if n > have {
            return Err(DsV4VindexWireError::Truncated {
                what,
                need: n,
                have,
            });
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    /// Read + validate the `magic[4] | version:u16` header.
    pub(crate) fn read_header(
        &mut self,
        expected_magic: [u8; 4],
        expected_version: u16,
    ) -> Result<(), DsV4VindexWireError> {
        let s = self.take(4, "magic")?;
        let magic = [s[0], s[1], s[2], s[3]];
        if magic != expected_magic {
            return Err(DsV4VindexWireError::BadMagic(magic));
        }
        let version = self.read_u16("version")?;
        if version != expected_version {
            return Err(DsV4VindexWireError::UnsupportedVersion(version));
        }
        Ok(())
    }

    pub(crate) fn read_u8(&mut self, what: &'static str) -> Result<u8, DsV4VindexWireError> {
        Ok(self.take(1, what)?[0])
    }
    pub(crate) fn read_u16(&mut self, what: &'static str) -> Result<u16, DsV4VindexWireError> {
        let s = self.take(2, what)?;
        Ok(u16::from_le_bytes([s[0], s[1]]))
    }
    pub(crate) fn read_u32(&mut self, what: &'static str) -> Result<u32, DsV4VindexWireError> {
        let s = self.take(4, what)?;
        Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }
    fn read_u64(&mut self, what: &'static str) -> Result<u64, DsV4VindexWireError> {
        let s = self.take(8, what)?;
        let mut b = [0u8; 8];
        b.copy_from_slice(s);
        Ok(u64::from_le_bytes(b))
    }

    pub(crate) fn read_raw(
        &mut self,
        what: &'static str,
    ) -> Result<RawExpertTensor, DsV4VindexWireError> {
        let tensor_type = self.read_u32(what)?;
        let rows = self.read_u32(what)? as usize;
        let cols = self.read_u32(what)? as usize;
        let byte_len = self.read_u64(what)? as usize;
        let bytes = self.take(byte_len, what)?.to_vec();
        Ok(RawExpertTensor {
            bytes,
            tensor_type,
            rows,
            cols,
        })
    }

    pub(crate) fn read_f32_vec(
        &mut self,
        what: &'static str,
    ) -> Result<Vec<f32>, DsV4VindexWireError> {
        let n = self.read_u32(what)? as usize;
        let s = self.take(n * 4, what)?;
        let mut v = Vec::with_capacity(n);
        for chunk in s.chunks_exact(4) {
            v.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        Ok(v)
    }

    pub(crate) fn read_opt_f32_vec(
        &mut self,
        what: &'static str,
    ) -> Result<Option<Vec<f32>>, DsV4VindexWireError> {
        if self.read_u8(what)? != 0 {
            Ok(Some(self.read_f32_vec(what)?))
        } else {
            Ok(None)
        }
    }

    pub(crate) fn read_i32_vec(
        &mut self,
        what: &'static str,
    ) -> Result<Vec<i32>, DsV4VindexWireError> {
        let n = self.read_u32(what)? as usize;
        let s = self.take(n * 4, what)?;
        let mut v = Vec::with_capacity(n);
        for chunk in s.chunks_exact(4) {
            v.push(i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        Ok(v)
    }

    pub(crate) fn read_opt_i32_vec(
        &mut self,
        what: &'static str,
    ) -> Result<Option<Vec<i32>>, DsV4VindexWireError> {
        if self.read_u8(what)? != 0 {
            Ok(Some(self.read_i32_vec(what)?))
        } else {
            Ok(None)
        }
    }
}
