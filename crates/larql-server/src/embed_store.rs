//! f16-at-rest embedding store with L1 f32 cache.
//!
//! On disk, `embeddings.bin` is stored as f16 (half-precision), which is
//! 1.34 GB for Gemma 3 4B vs 2.69 GB for the f32 copy that
//! `load_vindex_embeddings` builds on the heap.  `EmbedStoreF16` keeps the
//! raw mmap alive and decodes individual rows on demand, cutting embed-server
//! RSS from ~2.9 GB to ~1.5 GB (ADR-0008 §Optimization).
//!
//! An L1 hot-vocab cache (default 5 000 entries, ~50 MB) absorbs the Zipf
//! tail: the first N distinct token IDs accessed are cached as f32 forever.
//! Once the cap is reached, subsequent cache misses decode fresh from the mmap
//! on every call — still only 1–2 µs, negligible vs network overhead.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use memmap2::Mmap;

pub struct EmbedStoreF16 {
    mmap: Arc<Mmap>,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub embed_scale: f32,
    /// f16 bytes per token row (hidden_size × 2).
    row_bytes: usize,
    /// L1: populated on first access, capped at `l1_cap` entries.
    l1: Mutex<HashMap<u32, Vec<f32>>>,
    l1_cap: usize,
}

impl EmbedStoreF16 {
    /// Open `{dir}/embeddings.bin` as a read-only mmap.
    ///
    /// Validates the file size matches `vocab_size × hidden_size × 2` bytes.
    /// Returns an error if the file is missing or wrong size (e.g. f32 format
    /// — fall back to `load_vindex_embeddings` in that case).
    pub fn open(
        dir: &Path,
        embed_scale: f32,
        vocab_size: usize,
        hidden_size: usize,
        l1_cap: usize,
    ) -> Result<Self, String> {
        let path = dir.join("embeddings.bin");
        let file = std::fs::File::open(&path)
            .map_err(|e| format!("open {}: {e}", path.display()))?;
        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|e| format!("mmap {}: {e}", path.display()))?;
        let expected_f16 = vocab_size * hidden_size * 2;
        if mmap.len() != expected_f16 {
            return Err(format!(
                "embeddings.bin size {} != expected f16 size {} — not an f16 file",
                mmap.len(),
                expected_f16
            ));
        }
        Ok(Self {
            mmap: Arc::new(mmap),
            vocab_size,
            hidden_size,
            embed_scale,
            row_bytes: hidden_size * 2,
            l1: Mutex::new(HashMap::new()),
            l1_cap,
        })
    }

    /// Look up one token row, returning a scaled f32 vector.
    /// Checks L1 first; populates L1 on miss if below cap.
    pub fn lookup(&self, token_id: u32) -> Result<Vec<f32>, String> {
        let tid = token_id as usize;
        if tid >= self.vocab_size {
            return Err(format!(
                "token_id {token_id} out of range (vocab={})",
                self.vocab_size
            ));
        }

        // L1 hit — no decode needed.
        {
            let cache = self.l1.lock().unwrap();
            if let Some(row) = cache.get(&token_id) {
                return Ok(row.clone());
            }
        }

        // Decode from f16 mmap.
        let offset = tid * self.row_bytes;
        let raw = &self.mmap[offset..offset + self.row_bytes];
        let scale = self.embed_scale;
        let row: Vec<f32> = raw
            .chunks_exact(2)
            .map(|b| {
                let bits = u16::from_le_bytes([b[0], b[1]]);
                f16_to_f32(bits) * scale
            })
            .collect();

        // Populate L1 if there's room.
        {
            let mut cache = self.l1.lock().unwrap();
            if cache.len() < self.l1_cap {
                cache.insert(token_id, row.clone());
            }
        }
        Ok(row)
    }

    /// L1 cache hit count (for /v1/stats).
    pub fn l1_len(&self) -> usize {
        self.l1.lock().unwrap().len()
    }
}

/// IEEE 754 half-precision → f32.
/// Matches `larql_models::quant::half::f16_to_f32` but inlined here to avoid
/// a dependency on larql-models from this thin crate.
#[inline(always)]
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits as u32) & 0x8000) << 16;         // bit 31
    let exp16 = (bits >> 10) & 0x1F;                   // 5-bit exponent
    let mant16 = (bits as u32) & 0x03FF;               // 10-bit mantissa

    let (exp32, mant32) = if exp16 == 0 {
        if mant16 == 0 {
            // ±zero
            (0u32, 0u32)
        } else {
            // Subnormal: normalise by shifting mantissa.
            let mut m = mant16;
            let mut e = 127u32 - 14; // = 113
            while m & 0x0400 == 0 {
                m <<= 1;
                e -= 1;
            }
            (e, (m & 0x03FF) << 13)
        }
    } else if exp16 == 31 {
        // Inf or NaN.
        (0xFFu32, mant16 << 13)
    } else {
        // Normal.
        (exp16 as u32 + 127 - 15, mant16 << 13)
    };

    f32::from_bits(sign | (exp32 << 23) | mant32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_to_f32_zero() {
        assert_eq!(f16_to_f32(0), 0.0);
    }

    #[test]
    fn f16_to_f32_one() {
        // f16 1.0 = 0x3C00
        assert!((f16_to_f32(0x3C00) - 1.0).abs() < 1e-4);
    }

    #[test]
    fn f16_to_f32_neg_two() {
        // f16 -2.0 = 0xC000
        assert!((f16_to_f32(0xC000) - (-2.0)).abs() < 1e-4);
    }

    #[test]
    fn f16_to_f32_roundtrip_approx() {
        // Encode 3.14 as f16 (manually: sign=0, exp=16+127-15=128 → f16 exp=16,
        // mantissa truncated). Just check we're in the right ballpark.
        // 3.14 in f16 = 0x4248
        let got = f16_to_f32(0x4248);
        assert!((got - 3.140625).abs() < 0.01, "got {got}");
    }
}
