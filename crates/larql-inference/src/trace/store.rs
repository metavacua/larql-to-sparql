//! Mmap'd trace store — append-only file for growing context graphs.
//!
//! File layout:
//!   Header (64 bytes): magic, version, hidden_size, n_layers, n_tokens, ...
//!   Token chains:      contiguous, fixed-size per token
//!
//! Each token chain = (n_layers + 1) × 3 vectors × hidden_size × 4 bytes
//!   - 3 vectors per layer-waypoint: residual, attn_delta, ffn_delta
//!   - (n_layers + 1) because layer -1 (embedding) is included
//!
//! For gemma3-4b (34 layers, 2560 hidden):
//!   Chain size = 35 × 3 × 2560 × 4 = 1,075,200 bytes (~1.05 MB per token)
//!
//! Append-only: new tokens extend the file. Old chains are frozen.
//! Mmap'd: OS pages in/out on demand. RSS ≈ active chain only.

use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::Path;

use memmap2::Mmap;

use super::types::TraceNode;

const MAGIC: [u8; 4] = *b"TRAC";
const VERSION: u32 = 1;
const HEADER_SIZE: usize = 64;

/// Fixed header at the start of the trace file.
#[repr(C)]
#[derive(Clone, Copy)]
struct TraceHeader {
    magic: [u8; 4],
    version: u32,
    hidden_size: u32,
    n_layers: u32,      // transformer layers (not counting embedding)
    n_tokens: u32,      // number of complete token chains
    _reserved: [u8; 44],
}

impl TraceHeader {
    fn chain_size(&self) -> usize {
        let waypoints = self.n_layers as usize + 1; // +1 for embedding
        let vectors_per_waypoint = 3; // residual, attn_delta, ffn_delta
        waypoints * vectors_per_waypoint * self.hidden_size as usize * 4
    }

    fn to_bytes(self) -> [u8; HEADER_SIZE] {
        unsafe { std::mem::transmute(self) }
    }

    fn from_bytes(bytes: &[u8; HEADER_SIZE]) -> Self {
        unsafe { std::mem::transmute(*bytes) }
    }
}

/// Read-only mmap'd trace store.
pub struct TraceStore {
    mmap: Mmap,
    header: TraceHeader,
}

impl TraceStore {
    /// Open an existing trace file for reading.
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        if mmap.len() < HEADER_SIZE {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "file too small"));
        }

        let mut header_bytes = [0u8; HEADER_SIZE];
        header_bytes.copy_from_slice(&mmap[..HEADER_SIZE]);
        let header = TraceHeader::from_bytes(&header_bytes);

        if header.magic != MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "bad magic"));
        }
        if header.version != VERSION {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "unsupported version"));
        }

        // Advise OS: random access (attention reads arbitrary token chains)
        #[cfg(unix)]
        unsafe {
            libc::madvise(
                mmap.as_ptr() as *mut libc::c_void,
                mmap.len(),
                libc::MADV_RANDOM,
            );
        }

        Ok(Self { mmap, header })
    }

    pub fn n_tokens(&self) -> usize { self.header.n_tokens as usize }
    pub fn n_layers(&self) -> usize { self.header.n_layers as usize }
    pub fn hidden_size(&self) -> usize { self.header.hidden_size as usize }

    /// Read a specific vector from the store.
    /// Returns a slice into mmap'd memory — zero-copy.
    ///
    /// `token`: token position (0-indexed)
    /// `layer`: layer index (0 = embedding, 1..=n_layers = transformer layers)
    /// `component`: 0 = residual, 1 = attn_delta, 2 = ffn_delta
    pub fn read_vector(&self, token: usize, layer: usize, component: usize) -> Option<&[f32]> {
        if token >= self.header.n_tokens as usize { return None; }
        let n_waypoints = self.header.n_layers as usize + 1;
        if layer >= n_waypoints { return None; }
        if component >= 3 { return None; }

        let hidden = self.header.hidden_size as usize;
        let chain_offset = HEADER_SIZE + token * self.header.chain_size();
        let waypoint_offset = layer * 3 * hidden * 4;
        let vec_offset = component * hidden * 4;
        let start = chain_offset + waypoint_offset + vec_offset;
        let end = start + hidden * 4;

        if end > self.mmap.len() { return None; }

        let slice = &self.mmap[start..end];
        let floats = unsafe {
            std::slice::from_raw_parts(slice.as_ptr() as *const f32, hidden)
        };
        Some(floats)
    }

    /// Read the residual at (token, layer). Layer 0 = embedding.
    pub fn residual(&self, token: usize, layer: usize) -> Option<&[f32]> {
        self.read_vector(token, layer, 0)
    }

    /// Read the attention delta at (token, layer).
    pub fn attn_delta(&self, token: usize, layer: usize) -> Option<&[f32]> {
        self.read_vector(token, layer, 1)
    }

    /// Read the FFN delta at (token, layer).
    pub fn ffn_delta(&self, token: usize, layer: usize) -> Option<&[f32]> {
        self.read_vector(token, layer, 2)
    }

    /// Read all 3 vectors for a waypoint as a TraceNode.
    pub fn node(&self, token: usize, layer: usize) -> Option<TraceNode> {
        let residual = self.read_vector(token, layer, 0)?.to_vec();
        let attn_delta = self.read_vector(token, layer, 1)?.to_vec();
        let ffn_delta = self.read_vector(token, layer, 2)?.to_vec();
        Some(TraceNode {
            layer: layer as i32 - 1, // convert: store layer 0 = embedding = layer -1
            position: token,
            residual, attn_delta, ffn_delta,
        })
    }
}

/// Writable trace store — append-only.
pub struct TraceWriter {
    file: File,
    header: TraceHeader,
    path: std::path::PathBuf,
}

impl TraceWriter {
    /// Create a new trace file.
    pub fn create(path: &Path, hidden_size: usize, n_layers: usize) -> io::Result<Self> {
        let header = TraceHeader {
            magic: MAGIC,
            version: VERSION,
            hidden_size: hidden_size as u32,
            n_layers: n_layers as u32,
            n_tokens: 0,
            _reserved: [0; 44],
        };

        let mut file = OpenOptions::new()
            .read(true).write(true).create(true).truncate(true)
            .open(path)?;
        file.write_all(&header.to_bytes())?;
        file.flush()?;

        Ok(Self { file, header, path: path.to_path_buf() })
    }

    /// Open an existing trace file for appending.
    pub fn open(path: &Path) -> io::Result<Self> {
        let mut file = OpenOptions::new().read(true).write(true).open(path)?;

        let mut header_bytes = [0u8; HEADER_SIZE];
        file.read_exact(&mut header_bytes)?;
        let header = TraceHeader::from_bytes(&header_bytes);

        if header.magic != MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "bad magic"));
        }

        // Seek to end for appending
        file.seek(io::SeekFrom::End(0))?;

        Ok(Self { file, header, path: path.to_path_buf() })
    }

    /// Append a complete token chain (all layers) to the store.
    ///
    /// `nodes` must contain (n_layers + 1) nodes for this token, ordered by layer
    /// (-1, 0, 1, ..., n_layers-1).
    pub fn append_chain(&mut self, nodes: &[TraceNode]) -> io::Result<()> {
        let expected = self.header.n_layers as usize + 1;
        if nodes.len() != expected {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("expected {} nodes, got {}", expected, nodes.len()),
            ));
        }

        let hidden = self.header.hidden_size as usize;

        // Write vectors in order: for each waypoint, [residual, attn_delta, ffn_delta]
        for node in nodes {
            if node.residual.len() != hidden || node.attn_delta.len() != hidden || node.ffn_delta.len() != hidden {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("vector size mismatch: expected {}", hidden),
                ));
            }
            let r_bytes = unsafe {
                std::slice::from_raw_parts(node.residual.as_ptr() as *const u8, hidden * 4)
            };
            let a_bytes = unsafe {
                std::slice::from_raw_parts(node.attn_delta.as_ptr() as *const u8, hidden * 4)
            };
            let f_bytes = unsafe {
                std::slice::from_raw_parts(node.ffn_delta.as_ptr() as *const u8, hidden * 4)
            };
            self.file.write_all(r_bytes)?;
            self.file.write_all(a_bytes)?;
            self.file.write_all(f_bytes)?;
        }

        // Update token count in header
        self.header.n_tokens += 1;
        self.file.seek(io::SeekFrom::Start(0))?;
        self.file.write_all(&self.header.to_bytes())?;
        self.file.seek(io::SeekFrom::End(0))?;
        self.file.flush()?;

        Ok(())
    }

    /// Write a full ResidualTrace (all positions) to the store.
    pub fn write_trace(&mut self, trace: &super::types::ResidualTrace) -> io::Result<usize> {
        let n_positions = trace.tokens.len();
        let n_waypoints = self.header.n_layers as usize + 1;

        let mut written = 0;
        for pos in 0..n_positions {
            // Collect nodes for this position, ordered by layer
            let mut chain: Vec<&TraceNode> = trace.nodes.iter()
                .filter(|n| n.position == pos)
                .collect();
            chain.sort_by_key(|n| n.layer);

            if chain.len() != n_waypoints {
                continue; // skip positions without full chains
            }

            let owned: Vec<TraceNode> = chain.into_iter().cloned().collect();
            self.append_chain(&owned)?;
            written += 1;
        }

        Ok(written)
    }

    /// Finish writing — flush and return the path.
    pub fn finish(mut self) -> io::Result<std::path::PathBuf> {
        self.file.flush()?;
        Ok(self.path)
    }

    pub fn n_tokens(&self) -> usize { self.header.n_tokens as usize }
}

// Need Seek for TraceWriter
use std::io::Seek;
