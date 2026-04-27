//! Optimized mmap helpers for vindex file loading.
//!
//! Two access patterns:
//! - `mmap_optimized`: MADV_SEQUENTIAL + MADV_WILLNEED on Unix; basic mmap on Windows.
//!   For files that will be read fully on every forward pass (embeddings, norms, attn weights).
//!   Prefaults pages on Unix; on Windows relies on OS readahead heuristics.
//! - `mmap_demand_paged`: MADV_RANDOM on Unix; basic mmap on Windows.
//!   For large sparse files (gate vectors, feature payloads). Pages fault on access only.
//!
//! **Platform differences**:
//! - Unix (Linux, macOS): Kernel madvise() hints guide page cache behavior.
//! - Windows: NTFS/ReFS provide implicit readahead; no equivalent to MADV_RANDOM.
//!   Prefaulting has no benefit; pages fault on demand as accessed.

/// Create an mmap with optimized hints for sequential access patterns.
///
/// **Unix**: Applies MADV_SEQUENTIAL + MADV_WILLNEED to prefault pages.
/// **Windows**: Maps normally; OS handles readahead implicitly.
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

/// Create an mmap with hints for random/sparse access patterns.
///
/// **Unix**: Applies MADV_RANDOM to prevent speculative prefaulting.
/// **Windows**: Maps normally; pages fault on demand as accessed.
///
/// Use for large sparse files (gate_vectors.bin, interleaved_q4k.bin) where
/// RSS should reflect only the pages actually touched during inference, not
/// the full file size. Pages fault in on first access and are evictable under
/// memory pressure.
///
/// # Safety
///
/// The caller must ensure the file is not modified or truncated while the
/// mmap is alive.
#[cfg(unix)]
pub unsafe fn mmap_demand_paged(file: &std::fs::File) -> Result<memmap2::Mmap, std::io::Error> {
    let mmap = memmap2::Mmap::map(file)?;
    let ptr = mmap.as_ptr() as *mut libc::c_void;
    let len = mmap.len();
    unsafe {
        libc::madvise(ptr, len, libc::MADV_RANDOM);
    }
    Ok(mmap)
}

/// Windows implementation: no madvise equivalent, rely on OS demand paging.
///
/// Windows NTFS/ReFS automatically fault pages on access. No prefaulting benefit.
/// This is semantically identical to Unix MADV_RANDOM in practice.
#[cfg(target_os = "windows")]
pub unsafe fn mmap_demand_paged(file: &std::fs::File) -> Result<memmap2::Mmap, std::io::Error> {
    memmap2::Mmap::map(file)
}

/// Apply sequential + willneed hints to an existing mmap.
///
/// **Unix**: Calls madvise() with MADV_SEQUENTIAL + MADV_WILLNEED.
/// **Windows**: No-op; OS handles readahead implicitly on sequential access.
///
/// Call after Mmap::map() to optimize access patterns.
#[cfg(unix)]
pub fn advise_sequential(mmap: &memmap2::Mmap) {
    let ptr = mmap.as_ptr() as *mut libc::c_void;
    let len = mmap.len();
    unsafe {
        // Sequential: tell OS we stream linearly (enables aggressive readahead)
        libc::madvise(ptr, len, libc::MADV_SEQUENTIAL);
        // Willneed: prefault pages into cache (background page-in)
        libc::madvise(ptr, len, libc::MADV_WILLNEED);
    }
}

/// Windows implementation: no-op.
///
/// Windows kernel uses heuristics to detect sequential access patterns
/// automatically (e.g., via ReadFile() patterns). Explicit hints not available.
#[cfg(target_os = "windows")]
pub fn advise_sequential(_mmap: &memmap2::Mmap) {
    // No-op on Windows: kernel readahead is implicit.
}
