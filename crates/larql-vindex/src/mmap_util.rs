//! Optimized mmap helpers for vindex file loading.
//!
//! Two access patterns:
//! - `mmap_optimized`: MADV_SEQUENTIAL + MADV_WILLNEED — for files that must
//!   be fully resident (embeddings, norms, attn weights). Prefaults pages at
//!   load time so the first query doesn't stall on page faults.
//! - `mmap_demand_paged`: MADV_RANDOM — for large sparse files (gate vectors,
//!   feature payloads). Pages fault in only when accessed, keeping RSS low at
//!   load time. Gate KNN touches all pages during a linear scan but only a
//!   logarithmic subset when HNSW is active.

/// Create an mmap with SEQUENTIAL + WILLNEED hints — prefaults all pages.
///
/// Use for files that will be read fully on every forward pass (embeddings,
/// norms, attention weights). Not suitable for large sparse files where only
/// a fraction of pages are touched per token.
///
/// # Safety
///
/// The caller must ensure the file is not modified or truncated while the
/// mmap is alive.
pub unsafe fn mmap_optimized(file: &std::fs::File) -> Result<memmap2::Mmap, std::io::Error> {
    let mmap = memmap2::Mmap::map(file)?;
    advise_sequential(&mmap);
    Ok(mmap)
}

/// Create an mmap with RANDOM hint — no prefaulting, demand-paged only.
///
/// Use for large sparse files (gate_vectors.bin, interleaved_kquant.bin) where
/// RSS should reflect only the pages actually touched during inference, not
/// the full file size. Pages fault in on first access and are evictable under
/// memory pressure without any explicit unmap.
///
/// On non-unix targets the madvise hint is a no-op — the page-cache
/// policy is OS-managed there.
///
/// # Safety
///
/// The caller must ensure the file is not modified or truncated while the
/// mmap is alive.
pub unsafe fn mmap_demand_paged(file: &std::fs::File) -> Result<memmap2::Mmap, std::io::Error> {
    let mmap = memmap2::Mmap::map(file)?;
    #[cfg(unix)]
    {
        let ptr = mmap.as_ptr() as *mut libc::c_void;
        let len = mmap.len();
        unsafe {
            libc::madvise(ptr, len, libc::MADV_RANDOM);
        }
    }
    Ok(mmap)
}

/// Best-effort `MADV_WILLNEED` over a region already resolved to a raw
/// pointer + length (e.g. a per-layer sub-range of a larger mmap). Centralises
/// the `libc` reference so callers stay platform-agnostic — unix-only, a no-op
/// on Windows (no `madvise`; the OS manages readahead).
///
/// # Safety
///
/// `ptr`/`len` must describe a region of a live mmap that outlives the call.
#[cfg_attr(not(unix), allow(unused_variables))]
pub fn advise_willneed(ptr: *const u8, len: usize) {
    #[cfg(unix)]
    unsafe {
        libc::madvise(ptr as *mut libc::c_void, len, libc::MADV_WILLNEED);
    }
}

/// Anonymous read-write mapping of `len` bytes whose result survives the
/// `make_read_only()` conversion on every platform, including `len == 0`.
///
/// A zero-length `MmapMut::map_anon(0)` is portable on its own — memmap2
/// clamps the *underlying* Windows mapping to one byte. What is **not**
/// portable is what the vindex load path does next. memmap2 records the
/// requested `len` (0), not the clamped one, so a later `make_read_only()`
/// reaches `VirtualProtect(ptr, 0, PAGE_READONLY, …)`; Windows rejects a
/// zero `dwSize` with `ERROR_INVALID_PARAMETER` — os error 87, "The
/// parameter is incorrect." Unix has no equivalent step, so the same code
/// passes on macOS and Linux and fails only on Windows.
///
/// (The early-out in memmap2's `virtual_protect` does not help here: it
/// triggers on the sentinel pointer used by *file-backed* empty maps, which
/// an anonymous mapping never has.)
///
/// Empty is a legitimate state for several vindex buffers — an attention-only
/// client slice (`larql slice --preset client`) carries no gate data at all,
/// and a layer range can own no features. Requesting one byte keeps `len`
/// non-zero so the protection change is valid; every consumer bounds-checks
/// against the mapping length before reading, and the buffer is described as
/// empty by its slices, so the extra byte is never addressed.
pub fn map_anon_min_one(len: usize) -> Result<memmap2::MmapMut, std::io::Error> {
    memmap2::MmapMut::map_anon(len.max(1))
}

/// Apply sequential + willneed hints to an existing mmap. Unix-only;
/// on Windows the function is a no-op (the OS handles readahead).
#[cfg_attr(not(unix), allow(unused_variables))]
pub fn advise_sequential(mmap: &memmap2::Mmap) {
    #[cfg(unix)]
    {
        let ptr = mmap.as_ptr() as *mut libc::c_void;
        let len = mmap.len();
        unsafe {
            // Sequential: tell OS we stream linearly (enables aggressive readahead)
            libc::madvise(ptr, len, libc::MADV_SEQUENTIAL);
            // Willneed: prefault pages into cache (background page-in)
            libc::madvise(ptr, len, libc::MADV_WILLNEED);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pinned on every platform so the non-zero-length contract can't be
    /// quietly dropped by a refactor that "simplifies" back to a bare
    /// `map_anon`. The Windows failure this guards is in
    /// `make_read_only` (see below), not here — this case only asserts
    /// the length that keeps that conversion legal.
    #[test]
    fn map_anon_min_one_accepts_zero_length() {
        let m = map_anon_min_one(0).expect("zero-length anon mapping must succeed");
        assert!(!m.is_empty(), "must map at least one addressable byte");
    }

    #[test]
    fn map_anon_min_one_preserves_requested_length() {
        let m = map_anon_min_one(4096).expect("anon mapping");
        assert_eq!(m.len(), 4096, "non-zero lengths pass through unchanged");
    }

    /// A dense-only / client-slice gate buffer is read-only downstream —
    /// the `make_read_only` conversion the load path performs must work on
    /// the rounded-up mapping too.
    #[test]
    fn map_anon_min_one_zero_length_converts_to_read_only() {
        let ro = map_anon_min_one(0)
            .expect("anon mapping")
            .make_read_only()
            .expect("read-only conversion");
        assert!(!ro.is_empty());
    }
}
