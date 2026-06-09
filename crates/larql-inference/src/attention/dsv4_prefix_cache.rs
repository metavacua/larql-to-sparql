//! DeepSeek V4 prefix-keyed on-disk KV cache (Zero-SWA store).
//!
//! `dsv4-ondisk-prefix-cache` P2. A content-addressed store that maps a
//! hash of a prompt token-id prefix — taken at `PREFIX_BLOCK_TOKENS`
//! (`lcm(m, m') = 128`) boundaries and salted by a model identifier — to
//! the per-layer serialized compressed-KV blobs from [`super::
//! dsv4_kv_persist`]. On a new request, [`DsV4PrefixCache::
//! get_longest_prefix`] returns the longest cached block-prefix so the
//! caller can reuse it (the Zero-SWA reconstruct is P3).
//!
//! Layout: `<root>/<model_id>/<prefix_hash>/{tokens.bin, layer_{i}.kvz}`.
//! - `tokens.bin` stores the exact token-id prefix, so a candidate hash
//!   hit is **verified** against the query tokens before use — hash
//!   collisions can never return the wrong cache.
//! - Writes are atomic (populate a `.tmp.*` dir, then `rename`).
//! - A size cap evicts least-recently-used prefixes.
//!
//! The hash is a stable FNV-1a (not the std `RandomState`, whose seed
//! varies per process) so the same prefix maps to the same directory
//! across runs — the whole point of an on-disk cache.

use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::SystemTime;

use super::dsv4_kv_cache::DsV4LayerCache;
use super::dsv4_kv_persist::{deserialize_layer_cache, serialize_layer_cache, KvPersistError};

/// Prefix-cache block granularity in tokens: `lcm(m=4, m'=128) = 128`
/// for DSv4-Flash — the point at which both the CSA (`m`) and HCA (`m'`)
/// compressed streams land on a completed chunk (no `pending_cur` carry).
pub const PREFIX_BLOCK_TOKENS: usize = 128;

/// Error from the prefix cache.
#[derive(Debug)]
pub enum PrefixCacheError {
    /// `put` was called with a token count that isn't a positive
    /// multiple of [`PREFIX_BLOCK_TOKENS`].
    NotBlockAligned(usize),
    /// A stored blob failed to deserialize.
    Persist(KvPersistError),
    /// Underlying filesystem error.
    Io(io::Error),
}

impl std::fmt::Display for PrefixCacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PrefixCacheError::NotBlockAligned(n) => write!(
                f,
                "token count {n} is not a positive multiple of {PREFIX_BLOCK_TOKENS}"
            ),
            PrefixCacheError::Persist(e) => write!(f, "blob deserialize failed: {e}"),
            PrefixCacheError::Io(e) => write!(f, "io error: {e}"),
        }
    }
}

impl std::error::Error for PrefixCacheError {}

impl From<io::Error> for PrefixCacheError {
    fn from(e: io::Error) -> Self {
        PrefixCacheError::Io(e)
    }
}
impl From<KvPersistError> for PrefixCacheError {
    fn from(e: KvPersistError) -> Self {
        PrefixCacheError::Persist(e)
    }
}

/// In-memory index entry for one stored prefix.
#[derive(Clone)]
struct PrefixEntry {
    /// Token-id prefix length (a multiple of [`PREFIX_BLOCK_TOKENS`]).
    token_count: usize,
    /// Total on-disk size of this prefix's blobs.
    size_bytes: u64,
    /// Last access (for LRU eviction). Updated on hit.
    last_used: SystemTime,
}

/// A content-addressed, prefix-keyed on-disk KV cache for one model.
pub struct DsV4PrefixCache {
    /// `<root>/<model_id>` — all of this cache's prefix dirs live here.
    dir: PathBuf,
    /// Salt folded into every prefix hash (the model identifier).
    model_id: String,
    /// Eviction cap on total on-disk bytes.
    max_bytes: u64,
    /// hex(hash) → entry. Mirrors the on-disk dirs.
    entries: HashMap<String, PrefixEntry>,
    /// Sum of `entries[*].size_bytes`.
    total_bytes: u64,
}

impl DsV4PrefixCache {
    /// Open (creating if needed) the cache for `model_id` under `root`,
    /// with an on-disk size cap of `max_bytes`. Rebuilds the in-memory
    /// index by scanning existing prefix dirs.
    pub fn open(
        root: impl AsRef<Path>,
        model_id: &str,
        max_bytes: u64,
    ) -> Result<Self, PrefixCacheError> {
        let dir = root.as_ref().join(sanitize_segment(model_id));
        fs::create_dir_all(&dir)?;
        let mut cache = Self {
            dir,
            model_id: model_id.to_string(),
            max_bytes,
            entries: HashMap::new(),
            total_bytes: 0,
        };
        cache.rebuild_index()?;
        Ok(cache)
    }

    /// Number of prefixes currently indexed.
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    /// True if no prefixes are cached.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    /// Total on-disk bytes currently tracked.
    pub fn total_bytes(&self) -> u64 {
        self.total_bytes
    }

    /// Store the per-layer caches for a block-aligned token prefix.
    ///
    /// `token_ids.len()` must be a positive multiple of
    /// [`PREFIX_BLOCK_TOKENS`]. Re-storing an existing prefix just
    /// refreshes its LRU timestamp. Writes are atomic; eviction runs
    /// afterward to honor the size cap.
    pub fn put(
        &mut self,
        token_ids: &[u32],
        layers: &[DsV4LayerCache],
    ) -> Result<(), PrefixCacheError> {
        let n = token_ids.len();
        if n == 0 || n % PREFIX_BLOCK_TOKENS != 0 {
            return Err(PrefixCacheError::NotBlockAligned(n));
        }
        let key = hex64(fnv1a(self.model_id.as_bytes(), token_ids));
        let final_dir = self.dir.join(&key);

        if self.entries.contains_key(&key) {
            // Already stored — just touch the LRU timestamp.
            self.touch(&key)?;
            return Ok(());
        }

        // Populate a temp dir, then atomically rename into place.
        let tmp = self.dir.join(format!(".tmp.{key}.{}", next_nonce()));
        // Clean any stale tmp from a crashed prior run.
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp)?;
        let mut size_bytes: u64 = 0;
        size_bytes += write_atomic_inner(&tmp.join("tokens.bin"), &encode_tokens(token_ids))?;
        for (i, layer) in layers.iter().enumerate() {
            let blob = serialize_layer_cache(layer);
            size_bytes += write_atomic_inner(&tmp.join(format!("layer_{i}.kvz")), &blob)?;
        }
        // If a concurrent writer already produced final_dir, drop ours.
        if final_dir.exists() {
            let _ = fs::remove_dir_all(&tmp);
        } else {
            fs::rename(&tmp, &final_dir)?;
        }

        self.entries.insert(
            key,
            PrefixEntry {
                token_count: n,
                size_bytes,
                last_used: SystemTime::now(),
            },
        );
        self.total_bytes += size_bytes;
        self.evict_to_cap()?;
        Ok(())
    }

    /// Return the longest cached block-prefix of `token_ids`, if any:
    /// `(hit_len, per-layer caches)`. `hit_len` is a multiple of
    /// [`PREFIX_BLOCK_TOKENS`]. A hit refreshes the prefix's LRU
    /// timestamp.
    #[allow(clippy::type_complexity)]
    pub fn get_longest_prefix(
        &mut self,
        token_ids: &[u32],
    ) -> Result<Option<(usize, Vec<DsV4LayerCache>)>, PrefixCacheError> {
        let n_blocks = token_ids.len() / PREFIX_BLOCK_TOKENS;
        if n_blocks == 0 {
            return Ok(None);
        }
        // Incremental hashes at every block boundary (index k-1 = first
        // k*BLOCK tokens). One O(n) pass.
        let hashes = block_prefix_hashes(self.model_id.as_bytes(), token_ids, n_blocks);
        // Longest first.
        for k in (1..=n_blocks).rev() {
            let hit_len = k * PREFIX_BLOCK_TOKENS;
            let key = hex64(hashes[k - 1]);
            let Some(entry) = self.entries.get(&key) else {
                continue;
            };
            if entry.token_count != hit_len {
                continue; // hash present but for a different length — skip
            }
            let pdir = self.dir.join(&key);
            // Verify the stored tokens match (defeats any hash collision).
            let stored = match read_tokens(&pdir.join("tokens.bin")) {
                Ok(t) => t,
                Err(_) => continue, // missing/corrupt → treat as miss
            };
            if stored != token_ids[..hit_len] {
                continue;
            }
            // Load + deserialize every layer blob.
            let mut layers = Vec::new();
            let mut i = 0;
            loop {
                let lp = pdir.join(format!("layer_{i}.kvz"));
                if !lp.exists() {
                    break;
                }
                let bytes = fs::read(&lp)?;
                layers.push(deserialize_layer_cache(&bytes)?);
                i += 1;
            }
            self.touch(&key)?;
            return Ok(Some((hit_len, layers)));
        }
        Ok(None)
    }

    // ── internals ───────────────────────────────────────────────────────

    fn rebuild_index(&mut self) -> Result<(), PrefixCacheError> {
        self.entries.clear();
        self.total_bytes = 0;
        for entry in fs::read_dir(&self.dir)? {
            let entry = entry?;
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().into_owned();
            if !path.is_dir() || name.starts_with(".tmp.") {
                // Sweep leftover temp dirs from crashed writes.
                if name.starts_with(".tmp.") {
                    let _ = fs::remove_dir_all(&path);
                }
                continue;
            }
            let tokens_path = path.join("tokens.bin");
            let token_count = match read_tokens(&tokens_path) {
                Ok(t) => t.len(),
                Err(_) => continue, // not a valid prefix dir
            };
            let (size_bytes, last_used) = dir_size_and_mtime(&path)?;
            self.entries.insert(
                name,
                PrefixEntry {
                    token_count,
                    size_bytes,
                    last_used,
                },
            );
            self.total_bytes += size_bytes;
        }
        Ok(())
    }

    fn touch(&mut self, key: &str) -> Result<(), PrefixCacheError> {
        if let Some(e) = self.entries.get_mut(key) {
            e.last_used = SystemTime::now();
        }
        Ok(())
    }

    fn evict_to_cap(&mut self) -> Result<(), PrefixCacheError> {
        if self.total_bytes <= self.max_bytes {
            return Ok(());
        }
        // Oldest first.
        let mut by_age: Vec<(String, SystemTime, u64)> = self
            .entries
            .iter()
            .map(|(k, e)| (k.clone(), e.last_used, e.size_bytes))
            .collect();
        by_age.sort_by_key(|(_, t, _)| *t);
        for (key, _, size) in by_age {
            if self.total_bytes <= self.max_bytes {
                break;
            }
            let _ = fs::remove_dir_all(self.dir.join(&key));
            self.entries.remove(&key);
            self.total_bytes = self.total_bytes.saturating_sub(size);
        }
        Ok(())
    }
}

// ── free helpers ────────────────────────────────────────────────────────

/// FNV-1a 64-bit over `salt` then `token_ids` (LE u32 bytes). Stable
/// across processes (unlike std `RandomState`).
fn fnv1a(salt: &[u8], token_ids: &[u32]) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut h = OFFSET;
    for &b in salt {
        h ^= b as u64;
        h = h.wrapping_mul(PRIME);
    }
    for &t in token_ids {
        for b in t.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(PRIME);
        }
    }
    h
}

/// FNV-1a snapshots at each `k*BLOCK`-token boundary for `k in 1..=n_blocks`.
/// Returns a vec of length `n_blocks` (index `k-1`).
fn block_prefix_hashes(salt: &[u8], token_ids: &[u32], n_blocks: usize) -> Vec<u64> {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut h = OFFSET;
    for &b in salt {
        h ^= b as u64;
        h = h.wrapping_mul(PRIME);
    }
    let mut out = Vec::with_capacity(n_blocks);
    for (idx, &t) in token_ids.iter().enumerate() {
        for b in t.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(PRIME);
        }
        if (idx + 1) % PREFIX_BLOCK_TOKENS == 0 {
            out.push(h);
            if out.len() == n_blocks {
                break;
            }
        }
    }
    out
}

fn hex64(h: u64) -> String {
    format!("{h:016x}")
}

/// Filesystem-safe directory segment: keep `[A-Za-z0-9._-]`, replace
/// anything else with `_`.
fn sanitize_segment(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// `tokens.bin` = `u32 n_tokens | n_tokens * u32 LE`.
fn encode_tokens(token_ids: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 + token_ids.len() * 4);
    out.extend_from_slice(&(token_ids.len() as u32).to_le_bytes());
    for &t in token_ids {
        out.extend_from_slice(&t.to_le_bytes());
    }
    out
}

fn read_tokens(path: &Path) -> io::Result<Vec<u32>> {
    let bytes = fs::read(path)?;
    if bytes.len() < 4 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "tokens.bin too short",
        ));
    }
    let n = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
    if bytes.len() != 4 + n * 4 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "tokens.bin length mismatch",
        ));
    }
    let mut out = Vec::with_capacity(n);
    for chunk in bytes[4..].chunks_exact(4) {
        out.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

/// Write `data` to `path` and return its byte length. (Used inside a
/// temp dir that is itself atomically renamed into place.)
fn write_atomic_inner(path: &Path, data: &[u8]) -> io::Result<u64> {
    fs::write(path, data)?;
    Ok(data.len() as u64)
}

fn dir_size_and_mtime(dir: &Path) -> io::Result<(u64, SystemTime)> {
    let mut size = 0u64;
    let mut mtime = SystemTime::UNIX_EPOCH;
    for e in fs::read_dir(dir)? {
        let e = e?;
        let m = e.metadata()?;
        size += m.len();
        if let Ok(t) = m.modified() {
            if t > mtime {
                mtime = t;
            }
        }
    }
    Ok((size, mtime))
}

fn next_nonce() -> u64 {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let c = COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    nanos ^ (c.wrapping_mul(0x9e37_79b9_7f4a_7c15))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::dsv4_kv_cache::DsV4LayerHcaCache;
    use ndarray::Array2;
    use tempfile::TempDir;

    /// One Hca layer with `n` compressed rows of distinct values.
    fn hca_layer(cr: usize, n: usize, seed: f32) -> DsV4LayerCache {
        let mut hca = DsV4LayerHcaCache::with_capacity(256, 8, cr);
        if n > 0 {
            hca.compressed.append(
                Array2::<f32>::from_shape_fn((n, 8), |(r, d)| (r * 13 + d) as f32 + seed).view(),
            );
        }
        DsV4LayerCache::Hca(hca)
    }

    fn block(n_blocks: usize, base: u32) -> Vec<u32> {
        (0..n_blocks * PREFIX_BLOCK_TOKENS)
            .map(|i| base + i as u32)
            .collect()
    }

    fn compressed_rows(c: &DsV4LayerCache) -> Array2<f32> {
        match c {
            DsV4LayerCache::Hca(h) => h.compressed.view_current().to_owned(),
            DsV4LayerCache::NoCompress(_) => Array2::zeros((0, 0)),
        }
    }

    /// put then get-longest-prefix returns the same compressed caches.
    #[test]
    fn put_then_get_longest_prefix_round_trips() {
        let tmp = TempDir::new().unwrap();
        let mut cache = DsV4PrefixCache::open(tmp.path(), "model-A", 1 << 30).unwrap();

        let toks = block(2, 1000); // 256 tokens
        let layers = vec![hca_layer(4, 3, 0.5), hca_layer(128, 2, 1.5)];
        cache.put(&toks, &layers).unwrap();

        // Query a superset (the cached 256-token prefix + more).
        let mut query = toks.clone();
        query.extend(0..50);
        let (hit_len, got) = cache.get_longest_prefix(&query).unwrap().unwrap();
        assert_eq!(hit_len, 256);
        assert_eq!(got.len(), 2);
        assert_eq!(compressed_rows(&got[0]), compressed_rows(&layers[0]));
        assert_eq!(compressed_rows(&got[1]), compressed_rows(&layers[1]));
    }

    /// Longest of several stored prefixes wins.
    #[test]
    fn longest_prefix_wins() {
        let tmp = TempDir::new().unwrap();
        let mut cache = DsV4PrefixCache::open(tmp.path(), "m", 1 << 30).unwrap();
        let short = block(1, 7); // 128
        let long = block(3, 7); // 384, superset of `short`
        cache.put(&short, &[hca_layer(4, 1, 0.0)]).unwrap();
        cache.put(&long, &[hca_layer(4, 3, 9.0)]).unwrap();

        let (hit, got) = cache.get_longest_prefix(&block(3, 7)).unwrap().unwrap();
        assert_eq!(hit, 384, "should pick the 3-block prefix, not the 1-block");
        assert_eq!(compressed_rows(&got[0]).shape(), &[3, 8]);
    }

    /// A sequence sharing no cached block-prefix misses.
    #[test]
    fn no_shared_prefix_misses() {
        let tmp = TempDir::new().unwrap();
        let mut cache = DsV4PrefixCache::open(tmp.path(), "m", 1 << 30).unwrap();
        cache.put(&block(1, 100), &[hca_layer(4, 1, 0.0)]).unwrap();
        // Different tokens → different hash → miss.
        assert!(cache.get_longest_prefix(&block(1, 555)).unwrap().is_none());
        // Shorter than a block → miss.
        assert!(cache.get_longest_prefix(&[1, 2, 3]).unwrap().is_none());
    }

    /// Non-block-aligned put is rejected.
    #[test]
    fn put_rejects_unaligned() {
        let tmp = TempDir::new().unwrap();
        let mut cache = DsV4PrefixCache::open(tmp.path(), "m", 1 << 30).unwrap();
        let toks: Vec<u32> = (0..130).collect(); // not a multiple of 128
        match cache.put(&toks, &[hca_layer(4, 1, 0.0)]) {
            Err(PrefixCacheError::NotBlockAligned(130)) => {}
            other => panic!("expected NotBlockAligned, got {other:?}"),
        }
    }

    /// Size cap evicts least-recently-used; survivors still load.
    #[test]
    fn size_cap_evicts_lru() {
        let tmp = TempDir::new().unwrap();
        // Cap that holds ~1 prefix's worth; each put adds another.
        // Put A, then B (bigger), then access A to make B the LRU, then
        // a put that forces eviction of B.
        let a = block(1, 1);
        let b = block(1, 2);
        let c = block(1, 3);
        let one = vec![hca_layer(4, 2, 0.0)];
        // First measure one prefix's size with a generous cap.
        let mut probe = DsV4PrefixCache::open(tmp.path(), "probe", 1 << 30).unwrap();
        probe.put(&a, &one).unwrap();
        let per = probe.total_bytes();
        drop(probe);

        let tmp2 = TempDir::new().unwrap();
        let cap = per * 2 + per / 2; // room for 2 prefixes, not 3
        let mut cache = DsV4PrefixCache::open(tmp2.path(), "m", cap).unwrap();
        cache.put(&a, &one).unwrap();
        cache.put(&b, &one).unwrap();
        // Touch A so B becomes the LRU victim.
        cache.get_longest_prefix(&a).unwrap().unwrap();
        cache.put(&c, &one).unwrap(); // exceeds cap → evict LRU (B)

        assert!(cache.total_bytes() <= cap);
        assert!(
            cache.get_longest_prefix(&a).unwrap().is_some(),
            "A survives (recently used)"
        );
        assert!(
            cache.get_longest_prefix(&c).unwrap().is_some(),
            "C survives (just written)"
        );
        assert!(
            cache.get_longest_prefix(&b).unwrap().is_none(),
            "B evicted (LRU)"
        );
    }

    /// A fresh `open` on a populated dir rebuilds the index (persistence
    /// across instances).
    #[test]
    fn reopen_rebuilds_index() {
        let tmp = TempDir::new().unwrap();
        let toks = block(2, 42);
        {
            let mut cache = DsV4PrefixCache::open(tmp.path(), "m", 1 << 30).unwrap();
            cache.put(&toks, &[hca_layer(4, 3, 7.0)]).unwrap();
        }
        // New instance, same dir.
        let mut reopened = DsV4PrefixCache::open(tmp.path(), "m", 1 << 30).unwrap();
        assert_eq!(reopened.len(), 1);
        let (hit, got) = reopened.get_longest_prefix(&toks).unwrap().unwrap();
        assert_eq!(hit, 256);
        assert_eq!(compressed_rows(&got[0]).shape(), &[3, 8]);
    }

    /// Different model_id → isolated namespace (no cross-model hits).
    #[test]
    fn model_id_isolates() {
        let tmp = TempDir::new().unwrap();
        let toks = block(1, 5);
        let mut a = DsV4PrefixCache::open(tmp.path(), "model-A", 1 << 30).unwrap();
        a.put(&toks, &[hca_layer(4, 1, 0.0)]).unwrap();
        let mut b = DsV4PrefixCache::open(tmp.path(), "model-B", 1 << 30).unwrap();
        assert!(b.get_longest_prefix(&toks).unwrap().is_none());
    }
}
