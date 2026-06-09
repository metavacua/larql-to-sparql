//! DeepSeek V4 KV-cache serialization wire format (Full-SWA).
//!
//! `dsv4-ondisk-prefix-cache` P1/P3. Serializes a per-layer
//! [`DsV4LayerCache`] to a versioned little-endian binary blob for the
//! on-disk prefix cache. The format is a **lossless, complete**
//! round-trip of the cache state (the "Full-SWA" strategy): the `raw`
//! sliding-window KV, the `compressed` CSA/HCA KV (+ the indexer's
//! compressed KV), the partial-chunk `pending_cur` buffer, the
//! `compress_ratio`, and both compressor overlap states. Because the
//! deserialized cache equals the post-prefill cache bit-for-bit, a
//! prefix hit can simply **continue prefilling** at position `H` via the
//! existing cached forward — no recompute (the storage-leaner Zero-SWA
//! strategy, which drops `raw`/`pending_cur` and recomputes the SWA
//! tail, is a later optimization).
//!
//! Hand-rolled LE encoding (no serde/bincode), matching the codebase's
//! existing GGUF / kv-snapshot readers. All reads are bounds-checked:
//! malformed input yields a [`KvPersistError`], never a panic.

use ndarray::{Array1, Array2, ArrayView2};

use super::dsv4_compressor_prefill::CompressorOverlapState;
use super::dsv4_kv_cache::{DsV4LayerCache, DsV4LayerHcaCache, DsV4LayerKvCache};

/// 4-byte magic identifying a DSv4 KV-cache blob ("D4KV").
const MAGIC: [u8; 4] = *b"D4KV";
/// Wire format version. v2 = Full-SWA (complete cache, incl. `raw` +
/// `pending_cur`). v1 (Zero-SWA, compressed-only) is not read back.
const VERSION: u16 = 2;

/// Layer-cache variant tag in the blob header.
const TAG_NO_COMPRESS: u8 = 0;
const TAG_HCA: u8 = 1;

/// Error from deserializing a KV-cache blob.
#[derive(Debug, PartialEq, Eq)]
pub enum KvPersistError {
    /// The leading 4 bytes weren't the expected magic.
    BadMagic([u8; 4]),
    /// The version field isn't one this build understands.
    UnsupportedVersion(u16),
    /// An unknown layer-cache variant tag.
    UnknownTag(u8),
    /// The blob ended before a field could be fully read.
    Truncated {
        /// What was being read when the buffer ran out.
        what: &'static str,
        /// Bytes needed past the cursor.
        need: usize,
        /// Bytes actually remaining.
        have: usize,
    },
}

impl std::fmt::Display for KvPersistError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KvPersistError::BadMagic(m) => write!(f, "bad magic {m:?} (expected {MAGIC:?})"),
            KvPersistError::UnsupportedVersion(v) => write!(f, "unsupported version {v}"),
            KvPersistError::UnknownTag(t) => write!(f, "unknown layer-cache tag {t}"),
            KvPersistError::Truncated { what, need, have } => {
                write!(
                    f,
                    "truncated reading {what}: need {need} bytes, have {have}"
                )
            }
        }
    }
}

impl std::error::Error for KvPersistError {}

/// Serialize a per-layer cache to a complete (Full-SWA) blob.
///
/// Round-trips every field — `raw`, `compressed`, `pending_cur`,
/// `compress_ratio`, both overlap states, and the indexer compressed
/// cache — so `deserialize_layer_cache` reproduces the cache exactly.
pub fn serialize_layer_cache(cache: &DsV4LayerCache) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    match cache {
        DsV4LayerCache::NoCompress(kv) => {
            out.push(TAG_NO_COMPRESS);
            put_kv_section(&mut out, kv);
        }
        DsV4LayerCache::Hca(hca) => {
            out.push(TAG_HCA);
            put_u32(&mut out, hca.compress_ratio as u32);
            put_kv_section(&mut out, &hca.raw);
            put_kv_section(&mut out, &hca.compressed);
            put_pending(&mut out, &hca.pending_cur);
            match &hca.index_compressed {
                Some(idx) => {
                    out.push(1);
                    put_kv_section(&mut out, idx);
                }
                None => out.push(0),
            }
            put_overlap(&mut out, &hca.overlap_state);
            put_overlap(&mut out, &hca.indexer_overlap_state);
        }
    }
    out
}

/// Deserialize a per-layer cache from a Full-SWA blob — a complete
/// reconstruction equal to the serialized cache.
pub fn deserialize_layer_cache(bytes: &[u8]) -> Result<DsV4LayerCache, KvPersistError> {
    let mut c = Cursor::new(bytes);
    let magic = c.take_array4("magic")?;
    if magic != MAGIC {
        return Err(KvPersistError::BadMagic(magic));
    }
    let version = c.read_u16("version")?;
    if version != VERSION {
        return Err(KvPersistError::UnsupportedVersion(version));
    }
    let tag = c.read_u8("tag")?;
    match tag {
        TAG_NO_COMPRESS => Ok(DsV4LayerCache::NoCompress(c.read_kv_section("nc.kv")?)),
        TAG_HCA => {
            let compress_ratio = c.read_u32("hca.compress_ratio")? as usize;
            let raw = c.read_kv_section("hca.raw")?;
            let compressed = c.read_kv_section("hca.compressed")?;
            let pending_cur = c.read_pending("hca.pending")?;
            let index_compressed = if c.read_u8("hca.has_indexer")? != 0 {
                Some(c.read_kv_section("hca.index_compressed")?)
            } else {
                None
            };
            let overlap_state = c.read_overlap("hca.overlap")?;
            let indexer_overlap_state = c.read_overlap("hca.indexer_overlap")?;
            Ok(DsV4LayerCache::Hca(DsV4LayerHcaCache {
                raw,
                compressed,
                compress_ratio,
                pending_cur,
                overlap_state,
                index_compressed,
                indexer_overlap_state,
            }))
        }
        other => Err(KvPersistError::UnknownTag(other)),
    }
}

// ── encode helpers ──────────────────────────────────────────────────────

fn put_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn put_f32_rows(out: &mut Vec<u8>, rows: ArrayView2<f32>) {
    for v in rows.iter() {
        out.extend_from_slice(&v.to_le_bytes());
    }
}

/// `max_seq_len | head_dim | current_len | current_len*head_dim f32`.
fn put_kv_section(out: &mut Vec<u8>, kv: &DsV4LayerKvCache) {
    put_u32(out, kv.max_seq_len() as u32);
    put_u32(out, kv.head_dim() as u32);
    put_u32(out, kv.current_len() as u32);
    put_f32_rows(out, kv.view_current());
}

/// `count | width | count*width f32`. `pending_cur` rows are all `n_embd`
/// wide; `width = 0` when empty.
fn put_pending(out: &mut Vec<u8>, pending: &[Array1<f32>]) {
    let width = pending.first().map(|r| r.len()).unwrap_or(0);
    put_u32(out, pending.len() as u32);
    put_u32(out, width as u32);
    for row in pending {
        for &v in row.iter() {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
}

/// `optArray2(kv_prev_last) | optArray2(score_prev_last)`.
fn put_overlap(out: &mut Vec<u8>, o: &CompressorOverlapState) {
    put_opt_array2(out, o.kv_prev_last.as_ref());
    put_opt_array2(out, o.score_prev_last.as_ref());
}

/// `present:u8 [rows:u32 cols:u32 rows*cols f32]`.
fn put_opt_array2(out: &mut Vec<u8>, a: Option<&Array2<f32>>) {
    match a {
        Some(arr) => {
            out.push(1);
            put_u32(out, arr.shape()[0] as u32);
            put_u32(out, arr.shape()[1] as u32);
            put_f32_rows(out, arr.view());
        }
        None => out.push(0),
    }
}

fn kv_with_capacity(max_seq_len: usize, head_dim: usize) -> DsV4LayerKvCache {
    DsV4LayerKvCache::with_capacity(max_seq_len.max(1), head_dim.max(1))
}

// ── decode helpers ──────────────────────────────────────────────────────

/// Bounds-checked little-endian reader over a byte slice.
struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn take(&mut self, n: usize, what: &'static str) -> Result<&'a [u8], KvPersistError> {
        let have = self.buf.len() - self.pos.min(self.buf.len());
        if self.pos + n > self.buf.len() {
            return Err(KvPersistError::Truncated {
                what,
                need: n,
                have,
            });
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    fn take_array4(&mut self, what: &'static str) -> Result<[u8; 4], KvPersistError> {
        let s = self.take(4, what)?;
        Ok([s[0], s[1], s[2], s[3]])
    }

    fn read_u8(&mut self, what: &'static str) -> Result<u8, KvPersistError> {
        Ok(self.take(1, what)?[0])
    }

    fn read_u16(&mut self, what: &'static str) -> Result<u16, KvPersistError> {
        let s = self.take(2, what)?;
        Ok(u16::from_le_bytes([s[0], s[1]]))
    }

    fn read_u32(&mut self, what: &'static str) -> Result<u32, KvPersistError> {
        let s = self.take(4, what)?;
        Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }

    fn read_f32_rows(
        &mut self,
        rows: usize,
        cols: usize,
        what: &'static str,
    ) -> Result<Array2<f32>, KvPersistError> {
        let n = rows * cols;
        let s = self.take(n * 4, what)?;
        let mut data = Vec::with_capacity(n);
        for chunk in s.chunks_exact(4) {
            data.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        Ok(Array2::from_shape_vec((rows, cols), data).expect("rows*cols matches read length"))
    }

    fn read_kv_section(&mut self, what: &'static str) -> Result<DsV4LayerKvCache, KvPersistError> {
        let max_seq_len = self.read_u32(what)? as usize;
        let head_dim = self.read_u32(what)? as usize;
        let current_len = self.read_u32(what)? as usize;
        let mut kv = kv_with_capacity(max_seq_len, head_dim);
        if current_len > 0 {
            let rows = self.read_f32_rows(current_len, head_dim, what)?;
            kv.append(rows.view());
        }
        Ok(kv)
    }

    fn read_pending(&mut self, what: &'static str) -> Result<Vec<Array1<f32>>, KvPersistError> {
        let count = self.read_u32(what)? as usize;
        let width = self.read_u32(what)? as usize;
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            let s = self.take(width * 4, what)?;
            let mut row = Vec::with_capacity(width);
            for chunk in s.chunks_exact(4) {
                row.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            out.push(Array1::from(row));
        }
        Ok(out)
    }

    fn read_overlap(
        &mut self,
        what: &'static str,
    ) -> Result<CompressorOverlapState, KvPersistError> {
        let kv_prev_last = self.read_opt_array2(what)?;
        let score_prev_last = self.read_opt_array2(what)?;
        Ok(CompressorOverlapState {
            kv_prev_last,
            score_prev_last,
        })
    }

    fn read_opt_array2(
        &mut self,
        what: &'static str,
    ) -> Result<Option<Array2<f32>>, KvPersistError> {
        if self.read_u8(what)? == 0 {
            return Ok(None);
        }
        let rows = self.read_u32(what)? as usize;
        let cols = self.read_u32(what)? as usize;
        Ok(Some(self.read_f32_rows(rows, cols, what)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kv_with_rows(max_seq_len: usize, head_dim: usize, n: usize) -> DsV4LayerKvCache {
        let mut kv = DsV4LayerKvCache::with_capacity(max_seq_len, head_dim);
        if n > 0 {
            let rows =
                Array2::<f32>::from_shape_fn((n, head_dim), |(r, d)| (r * 31 + d) as f32 * 0.1);
            kv.append(rows.view());
        }
        kv
    }

    fn assert_kv_eq(a: &DsV4LayerKvCache, b: &DsV4LayerKvCache) {
        assert_eq!(a.max_seq_len(), b.max_seq_len());
        assert_eq!(a.head_dim(), b.head_dim());
        assert_eq!(a.current_len(), b.current_len());
        assert_eq!(a.view_current(), b.view_current(), "kv rows differ");
    }

    /// Full HCA cache (raw + compressed + pending + overlap) round-trips
    /// bit-exactly — the Full-SWA lossless contract.
    #[test]
    fn hca_full_cache_round_trips_losslessly() {
        let mut hca = DsV4LayerHcaCache::with_capacity(64, 8, 4);
        hca.raw.append(
            Array2::<f32>::from_shape_fn((10, 8), |(r, d)| (r * 3 + d) as f32 * 0.7).view(),
        );
        hca.compressed
            .append(Array2::<f32>::from_shape_fn((3, 8), |(r, d)| (r + d) as f32 * 0.5).view());
        hca.pending_cur
            .push(Array1::<f32>::from_shape_fn(16, |i| i as f32 * 0.25));
        hca.pending_cur.push(Array1::<f32>::from_elem(16, -2.0));

        let blob = serialize_layer_cache(&DsV4LayerCache::Hca(hca.clone()));
        let DsV4LayerCache::Hca(got) = deserialize_layer_cache(&blob).unwrap() else {
            panic!("expected Hca variant");
        };

        assert_eq!(got.compress_ratio, 4);
        assert_kv_eq(&got.raw, &hca.raw); // raw now PRESERVED (Full-SWA)
        assert_kv_eq(&got.compressed, &hca.compressed);
        assert_eq!(got.pending_cur.len(), 2);
        assert_eq!(got.pending_cur[0], hca.pending_cur[0]);
        assert_eq!(got.pending_cur[1], hca.pending_cur[1]);
        assert!(got.index_compressed.is_none());
    }

    /// Indexer-variant cache: indexer compressed + both overlap states +
    /// raw all round-trip.
    #[test]
    fn hca_with_indexer_and_overlap_round_trips() {
        let mut hca = DsV4LayerHcaCache::with_capacity_and_indexer(64, 8, 4, 5);
        hca.raw
            .append(Array2::<f32>::from_shape_fn((6, 8), |(r, d)| (r + d) as f32).view());
        hca.compressed
            .append(Array2::<f32>::from_shape_fn((2, 8), |(r, d)| (r * 7 + d) as f32).view());
        hca.index_compressed
            .as_mut()
            .unwrap()
            .append(Array2::<f32>::from_shape_fn((2, 5), |(r, d)| (r + d * 2) as f32).view());
        hca.overlap_state = CompressorOverlapState {
            kv_prev_last: Some(Array2::<f32>::from_shape_fn((4, 8), |(r, d)| {
                (r + d) as f32 * 0.3
            })),
            score_prev_last: Some(Array2::<f32>::from_elem((4, 8), -1.0)),
        };
        hca.indexer_overlap_state = CompressorOverlapState {
            kv_prev_last: Some(Array2::<f32>::from_elem((4, 5), 2.0)),
            score_prev_last: None,
        };

        let blob = serialize_layer_cache(&DsV4LayerCache::Hca(hca.clone()));
        let DsV4LayerCache::Hca(got) = deserialize_layer_cache(&blob).unwrap() else {
            panic!("expected Hca");
        };

        assert_kv_eq(&got.raw, &hca.raw);
        assert_kv_eq(&got.compressed, &hca.compressed);
        assert_kv_eq(
            got.index_compressed.as_ref().unwrap(),
            hca.index_compressed.as_ref().unwrap(),
        );
        assert_eq!(
            got.overlap_state.kv_prev_last,
            hca.overlap_state.kv_prev_last
        );
        assert_eq!(
            got.overlap_state.score_prev_last,
            hca.overlap_state.score_prev_last
        );
        assert_eq!(
            got.indexer_overlap_state.kv_prev_last,
            hca.indexer_overlap_state.kv_prev_last
        );
        assert_eq!(got.indexer_overlap_state.score_prev_last, None);
    }

    /// NoCompress (pure-SWA) layer: its `raw` rows round-trip in full
    /// (Full-SWA — those rows ARE the layer's cache).
    #[test]
    fn no_compress_layer_round_trips_full() {
        let kv = kv_with_rows(128, 512, 17);
        let blob = serialize_layer_cache(&DsV4LayerCache::NoCompress(kv.clone()));
        let DsV4LayerCache::NoCompress(got) = deserialize_layer_cache(&blob).unwrap() else {
            panic!("expected NoCompress");
        };
        assert_kv_eq(&got, &kv);
    }

    /// Empty HCA cache (no chunk yet) round-trips.
    #[test]
    fn empty_cache_round_trips() {
        let hca = DsV4LayerHcaCache::with_capacity(64, 8, 4);
        let blob = serialize_layer_cache(&DsV4LayerCache::Hca(hca));
        let DsV4LayerCache::Hca(got) = deserialize_layer_cache(&blob).unwrap() else {
            panic!("expected Hca");
        };
        assert_eq!(got.raw.current_len(), 0);
        assert_eq!(got.compressed.current_len(), 0);
        assert!(got.pending_cur.is_empty());
        assert!(got.index_compressed.is_none());
    }

    /// Bad magic → typed error, not panic.
    #[test]
    fn bad_magic_is_typed_error() {
        let mut blob = serialize_layer_cache(&DsV4LayerCache::NoCompress(kv_with_rows(8, 4, 0)));
        blob[0] = b'X';
        match deserialize_layer_cache(&blob) {
            Err(KvPersistError::BadMagic(_)) => {}
            Err(e) => panic!("expected BadMagic, got {e:?}"),
            Ok(_) => panic!("expected BadMagic, got Ok"),
        }
    }

    /// Unknown version → typed error.
    #[test]
    fn unsupported_version_is_typed_error() {
        let mut blob = serialize_layer_cache(&DsV4LayerCache::NoCompress(kv_with_rows(8, 4, 0)));
        blob[4] = 0xFF;
        blob[5] = 0xFF;
        match deserialize_layer_cache(&blob) {
            Err(KvPersistError::UnsupportedVersion(0xFFFF)) => {}
            Err(e) => panic!("expected UnsupportedVersion, got {e:?}"),
            Ok(_) => panic!("expected UnsupportedVersion, got Ok"),
        }
    }

    /// Unknown tag → typed error.
    #[test]
    fn unknown_tag_is_typed_error() {
        let mut blob = serialize_layer_cache(&DsV4LayerCache::NoCompress(kv_with_rows(8, 4, 0)));
        blob[6] = 0x9; // tag byte (after 4 magic + 2 version)
        match deserialize_layer_cache(&blob) {
            Err(KvPersistError::UnknownTag(0x9)) => {}
            Err(e) => panic!("expected UnknownTag, got {e:?}"),
            Ok(_) => panic!("expected UnknownTag, got Ok"),
        }
    }

    /// Truncated blob → typed error, never a panic.
    #[test]
    fn truncated_blob_is_typed_error() {
        let mut hca = DsV4LayerHcaCache::with_capacity(64, 8, 4);
        hca.raw.append(Array2::<f32>::ones((4, 8)).view());
        hca.compressed.append(Array2::<f32>::ones((1, 8)).view());
        let blob = serialize_layer_cache(&DsV4LayerCache::Hca(hca));
        for cut in 0..blob.len() {
            match deserialize_layer_cache(&blob[..cut]) {
                Err(_) => {}
                Ok(_) => panic!("truncation at {cut} should error"),
            }
        }
    }
}
