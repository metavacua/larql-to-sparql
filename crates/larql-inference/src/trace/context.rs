//! Tiered context store — boundary residuals + critical layer deltas.
//!
//! Three tiers of stored data per window boundary:
//!
//!   Tier 1: Boundary residual only (10 KB/window)
//!           → needs forward pass replay for reconstruction
//!
//!   Tier 2: + FFN deltas at critical layers (50 KB/window)
//!           → partial reconstruction, no replay for knowledge queries
//!
//!   Tier 3: + attention deltas at critical layers (70 KB/window)
//!           → full reconstruction, zero replay cost
//!
//! For 370K tokens (Apollo 11 transcript):
//!   Tier 1: 725 windows × 10 KB  = 7 MB     (needs replay)
//!   Tier 2: 725 windows × 50 KB  = 36 MB    (partial)
//!   Tier 3: 725 windows × 70 KB  = 50 MB    (full, no replay)
//!   KV cache:                     = 56,000 MB
//!   Compression: 1,100× at Tier 3
//!
//! File layout:
//!   Header (128 bytes):    magic, version, hidden_size, n_layers, window_size,
//!                          critical_layers[], tier, n_boundaries, ...
//!   Boundary index:        n_boundaries × ContextEntry (24 bytes each)
//!   Vector data:           contiguous, variable per tier
//!
//! Mmap'd, append-only, zero-copy reads.

use std::fs::{File, OpenOptions};
use std::io::{self, Write, Seek, SeekFrom};
use std::path::Path;

use memmap2::Mmap;

const MAGIC: [u8; 4] = *b"CTXT";
const VERSION: u32 = 1;
const HEADER_SIZE: usize = 128;
const ENTRY_SIZE: usize = 24;
const MAX_CRITICAL_LAYERS: usize = 8;

/// Storage tier.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum ContextTier {
    /// Boundary residual only (10 KB/window for gemma3-4b).
    Residual = 1,
    /// + FFN deltas at critical layers (~50 KB/window).
    FfnDeltas = 2,
    /// + attention deltas at critical layers (~70 KB/window).
    Full = 3,
}

impl ContextTier {
    fn from_u8(v: u8) -> Self {
        match v {
            2 => Self::FfnDeltas,
            3 => Self::Full,
            _ => Self::Residual,
        }
    }

    /// Number of vectors stored per boundary at this tier.
    fn vectors_per_boundary(&self, n_critical: usize) -> usize {
        match self {
            Self::Residual => 1,                         // just boundary residual
            Self::FfnDeltas => 1 + n_critical,           // + ffn_delta per critical layer
            Self::Full => 1 + 2 * n_critical,            // + attn_delta + ffn_delta per critical layer
        }
    }
}

/// File header.
#[repr(C)]
#[derive(Clone, Copy)]
struct ContextHeader {
    magic: [u8; 4],
    version: u32,
    hidden_size: u32,
    n_layers: u32,
    window_size: u32,
    tier: u8,
    n_critical: u8,
    _pad: [u8; 2],
    critical_layers: [u8; MAX_CRITICAL_LAYERS],  // layer indices
    n_boundaries: u32,
    total_tokens: u32,
    _reserved: [u8; 88],
}

impl ContextHeader {
    fn bytes_per_boundary(&self) -> usize {
        let tier = ContextTier::from_u8(self.tier);
        let n_vecs = tier.vectors_per_boundary(self.n_critical as usize);
        n_vecs * self.hidden_size as usize * 4
    }

    fn to_bytes(self) -> [u8; HEADER_SIZE] {
        unsafe { std::mem::transmute(self) }
    }

    fn from_bytes(bytes: &[u8; HEADER_SIZE]) -> Self {
        unsafe { std::mem::transmute(*bytes) }
    }

    fn critical_layer_list(&self) -> Vec<usize> {
        (0..self.n_critical as usize)
            .map(|i| self.critical_layers[i] as usize)
            .collect()
    }
}

/// Index entry.
#[repr(C)]
#[derive(Clone, Copy)]
struct ContextEntry {
    token_offset: u32,
    window_tokens: u32,
    data_offset: u64,  // byte offset to this boundary's vectors
    _reserved: u64,
}

impl ContextEntry {
    fn to_bytes(self) -> [u8; ENTRY_SIZE] {
        unsafe { std::mem::transmute(self) }
    }
    fn from_bytes(bytes: &[u8; ENTRY_SIZE]) -> Self {
        unsafe { std::mem::transmute(*bytes) }
    }
}

/// Read-only mmap'd context store.
pub struct ContextStore {
    mmap: Mmap,
    header: ContextHeader,
}

impl ContextStore {
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        if mmap.len() < HEADER_SIZE {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "file too small"));
        }

        let mut header_bytes = [0u8; HEADER_SIZE];
        header_bytes.copy_from_slice(&mmap[..HEADER_SIZE]);
        let header = ContextHeader::from_bytes(&header_bytes);

        if header.magic != MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "bad magic"));
        }

        #[cfg(unix)]
        unsafe {
            libc::madvise(mmap.as_ptr() as *mut libc::c_void, mmap.len(), libc::MADV_RANDOM);
        }

        Ok(Self { mmap, header })
    }

    pub fn n_boundaries(&self) -> usize { self.header.n_boundaries as usize }
    pub fn total_tokens(&self) -> usize { self.header.total_tokens as usize }
    pub fn hidden_size(&self) -> usize { self.header.hidden_size as usize }
    pub fn window_size(&self) -> usize { self.header.window_size as usize }
    pub fn tier(&self) -> ContextTier { ContextTier::from_u8(self.header.tier) }
    pub fn critical_layers(&self) -> Vec<usize> { self.header.critical_layer_list() }
    pub fn bytes_per_boundary(&self) -> usize { self.header.bytes_per_boundary() }

    fn entry(&self, i: usize) -> Option<ContextEntry> {
        if i >= self.header.n_boundaries as usize { return None; }
        let offset = HEADER_SIZE + i * ENTRY_SIZE;
        if offset + ENTRY_SIZE > self.mmap.len() { return None; }
        let mut bytes = [0u8; ENTRY_SIZE];
        bytes.copy_from_slice(&self.mmap[offset..offset + ENTRY_SIZE]);
        Some(ContextEntry::from_bytes(&bytes))
    }

    fn read_vec_at(&self, byte_offset: usize) -> Option<&[f32]> {
        let hidden = self.header.hidden_size as usize;
        let end = byte_offset + hidden * 4;
        if end > self.mmap.len() { return None; }
        Some(unsafe {
            std::slice::from_raw_parts(
                self.mmap[byte_offset..].as_ptr() as *const f32, hidden,
            )
        })
    }

    /// Read the boundary residual for window i.
    pub fn residual(&self, i: usize) -> Option<&[f32]> {
        let entry = self.entry(i)?;
        self.read_vec_at(entry.data_offset as usize)
    }

    /// Read FFN delta at critical layer index `cl_idx` for boundary `i`.
    /// Only available at Tier 2+.
    pub fn ffn_delta(&self, i: usize, cl_idx: usize) -> Option<&[f32]> {
        if self.header.tier < ContextTier::FfnDeltas as u8 { return None; }
        if cl_idx >= self.header.n_critical as usize { return None; }
        let entry = self.entry(i)?;
        let hidden = self.header.hidden_size as usize;
        // Layout: [residual, ffn_0, ffn_1, ..., ffn_n, attn_0, attn_1, ...]
        let offset = entry.data_offset as usize + (1 + cl_idx) * hidden * 4;
        self.read_vec_at(offset)
    }

    /// Read attention delta at critical layer index `cl_idx` for boundary `i`.
    /// Only available at Tier 3.
    pub fn attn_delta(&self, i: usize, cl_idx: usize) -> Option<&[f32]> {
        if self.header.tier < ContextTier::Full as u8 { return None; }
        let n_crit = self.header.n_critical as usize;
        if cl_idx >= n_crit { return None; }
        let entry = self.entry(i)?;
        let hidden = self.header.hidden_size as usize;
        // attn deltas come after all ffn deltas
        let offset = entry.data_offset as usize + (1 + n_crit + cl_idx) * hidden * 4;
        self.read_vec_at(offset)
    }

    /// Get token range for boundary i.
    pub fn token_range(&self, i: usize) -> Option<(usize, usize)> {
        let entry = self.entry(i)?;
        Some((entry.token_offset as usize, entry.token_offset as usize + entry.window_tokens as usize))
    }

    /// Find boundary containing a token offset.
    pub fn boundary_for_token(&self, token: usize) -> Option<usize> {
        for i in 0..self.header.n_boundaries as usize {
            if let Some((start, end)) = self.token_range(i) {
                if token >= start && token < end { return Some(i); }
            }
        }
        None
    }

    pub fn file_size(&self) -> usize { self.mmap.len() }
    pub fn data_size(&self) -> usize {
        self.header.n_boundaries as usize * self.header.bytes_per_boundary()
    }
}

/// Writable context store.
pub struct ContextWriter {
    file: File,
    header: ContextHeader,
    path: std::path::PathBuf,
    max_boundaries: usize,
}

impl ContextWriter {
    /// Create a new context store.
    pub fn create(
        path: &Path,
        hidden_size: usize,
        n_layers: usize,
        window_size: usize,
        tier: ContextTier,
        critical_layers: &[usize],
        max_boundaries: usize,
    ) -> io::Result<Self> {
        let n_critical = critical_layers.len().min(MAX_CRITICAL_LAYERS);
        let mut cl = [0u8; MAX_CRITICAL_LAYERS];
        for (i, &l) in critical_layers.iter().take(n_critical).enumerate() {
            cl[i] = l as u8;
        }

        let header = ContextHeader {
            magic: MAGIC,
            version: VERSION,
            hidden_size: hidden_size as u32,
            n_layers: n_layers as u32,
            window_size: window_size as u32,
            tier: tier as u8,
            n_critical: n_critical as u8,
            _pad: [0; 2],
            critical_layers: cl,
            n_boundaries: 0,
            total_tokens: 0,
            _reserved: [0; 88],
        };

        let mut file = OpenOptions::new()
            .read(true).write(true).create(true).truncate(true)
            .open(path)?;
        file.write_all(&header.to_bytes())?;
        // Pre-allocate index
        file.write_all(&vec![0u8; max_boundaries * ENTRY_SIZE])?;
        file.flush()?;

        Ok(Self { file, header, path: path.to_path_buf(), max_boundaries })
    }

    /// Append a boundary with its vectors.
    ///
    /// `residual`: the boundary residual (always required)
    /// `ffn_deltas`: FFN deltas at critical layers (Tier 2+)
    /// `attn_deltas`: attention deltas at critical layers (Tier 3)
    pub fn append(
        &mut self,
        token_offset: usize,
        window_tokens: usize,
        residual: &[f32],
        ffn_deltas: &[Vec<f32>],
        attn_deltas: &[Vec<f32>],
    ) -> io::Result<()> {
        let hidden = self.header.hidden_size as usize;
        let n_crit = self.header.n_critical as usize;
        let tier = ContextTier::from_u8(self.header.tier);
        let idx = self.header.n_boundaries as usize;

        if idx >= self.max_boundaries {
            return Err(io::Error::other("index full"));
        }
        if residual.len() != hidden {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "residual size mismatch"));
        }

        // Write data
        self.file.seek(SeekFrom::End(0))?;
        let data_pos = self.file.stream_position()?;

        // Always write residual
        write_f32_slice(&mut self.file, residual)?;

        // Tier 2+: write FFN deltas
        if tier as u8 >= ContextTier::FfnDeltas as u8 {
            for i in 0..n_crit {
                let delta = ffn_deltas.get(i)
                    .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput,
                        format!("missing ffn_delta for critical layer {}", i)))?;
                if delta.len() != hidden {
                    return Err(io::Error::new(io::ErrorKind::InvalidInput, "ffn_delta size mismatch"));
                }
                write_f32_slice(&mut self.file, delta)?;
            }
        }

        // Tier 3: write attention deltas
        if tier == ContextTier::Full {
            for i in 0..n_crit {
                let delta = attn_deltas.get(i)
                    .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput,
                        format!("missing attn_delta for critical layer {}", i)))?;
                if delta.len() != hidden {
                    return Err(io::Error::new(io::ErrorKind::InvalidInput, "attn_delta size mismatch"));
                }
                write_f32_slice(&mut self.file, delta)?;
            }
        }

        // Write index entry
        let entry = ContextEntry {
            token_offset: token_offset as u32,
            window_tokens: window_tokens as u32,
            data_offset: data_pos,
            _reserved: 0,
        };
        let entry_offset = HEADER_SIZE + idx * ENTRY_SIZE;
        self.file.seek(SeekFrom::Start(entry_offset as u64))?;
        self.file.write_all(&entry.to_bytes())?;

        // Update header
        self.header.n_boundaries += 1;
        self.header.total_tokens = (token_offset + window_tokens) as u32;
        self.file.seek(SeekFrom::Start(0))?;
        self.file.write_all(&self.header.to_bytes())?;
        self.file.flush()?;

        Ok(())
    }

    pub fn n_boundaries(&self) -> usize { self.header.n_boundaries as usize }
    pub fn total_tokens(&self) -> usize { self.header.total_tokens as usize }

    pub fn finish(mut self) -> io::Result<std::path::PathBuf> {
        self.file.flush()?;
        Ok(self.path)
    }
}

fn write_f32_slice(file: &mut File, data: &[f32]) -> io::Result<()> {
    let bytes = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
    };
    file.write_all(bytes)
}
