//! Optimized mmap helpers for vindex file loading.
//!
//! Applies OS hints (madvise) to improve memory-mapped I/O performance:
//! - MADV_SEQUENTIAL: enables aggressive readahead for streaming access
//! - MADV_WILLNEED: prefaults pages into the page cache
//!
//! On M3 Max with 400 GB/s theoretical bandwidth, these hints can
//! improve effective throughput from ~50 GB/s to closer to peak.

/// Create an mmap with optimized access hints for streaming reads.
///
/// Safe to call on any file. The advisory hints are best-effort —
/// the OS may ignore them, but on macOS/Linux they significantly
/// improve page cache behavior for large sequential reads.
///
/// # Safety
///
/// The caller must ensure the file is not modified or truncated while the
/// mmap is alive. This is the standard memmap2 safety contract — the mmap
/// returns a `&[u8]` view into the file's pages, which become invalid if
/// the file changes on disk.
pub unsafe fn mmap_optimized(file: &std::fs::File) -> Result<memmap2::Mmap, std::io::Error> {
    let mmap = memmap2::Mmap::map(file)?;
    advise_sequential(&mmap);
    Ok(mmap)
}

/// Apply sequential + willneed hints to an existing mmap.
/// Call after Mmap::map() to optimize access patterns.
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
