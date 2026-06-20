//! CPU and memory resource reservation — enforced internally at process start.
//!
//! # CPU reservation
//! `apply_cpu_reservation()` caps the rayon global thread pool to
//! `available_parallelism() - 1` when more than one logical CPU is present,
//! reserving exactly one thread slot for the OS.  The SpinPool sizes itself
//! from `rayon::current_num_threads()` on first use, so calling this before
//! any rayon work suffices for both pools.
//!
//! Tokio async runtime threads are NOT managed here; binary entry points must
//! pass the returned worker count to `tokio::runtime::Builder::worker_threads`.
//!
//! # Memory cap
//! The shared `CapAllocator`, `TrackedMmap`, and `TrackedMmapMut` types live in
//! `larql_models::resource` (lowest dep in the tree).  This module re-exports
//! them and provides `detect_physical_memory_bytes()` which reads physical RAM
//! once via `libc::sysconf` (no allocation, no file parse, no env dependency).

pub use larql_models::resource::{
    charge_mmap, release_mmap, set_memory_cap, CapAllocator, TrackedMmap, TrackedMmapMut,
};

/// Read physical RAM in bytes via sysconf.  Returns 0 if unavailable.
///
/// Uses `_SC_PHYS_PAGES * _SC_PAGESIZE` — two syscalls, no allocation,
/// no file parsing, no env dependency.  Platform support: Linux and macOS.
pub fn detect_physical_memory_bytes() -> usize {
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    {
        // SAFETY: sysconf is always safe to call.
        let pages = unsafe { libc::sysconf(libc::_SC_PHYS_PAGES) };
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if pages > 0 && page_size > 0 {
            return (pages as usize).saturating_mul(page_size as usize);
        }
    }
    0
}

/// Reserve one logical CPU for the OS and apply the reservation to the rayon
/// global thread pool.  Returns the number of worker threads that larql will
/// use (`available_parallelism() - 1`, minimum 1).
///
/// Must be called before any rayon work or the SpinPool is first accessed.
/// Subsequent `rayon::ThreadPoolBuilder::build_global()` calls (e.g., from
/// `bench/run.rs`) are silently ignored by rayon — the first call wins.
///
/// On a single-CPU host, returns 1 and does NOT call `build_global` — rayon's
/// own default of 1 thread is already correct.
pub fn apply_cpu_reservation() -> usize {
    let n = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let workers = if n > 1 { n - 1 } else { 1 };
    if n > 1 {
        // Ignore the error — it only fires if another thread already
        // initialized the pool (impossible at this call point in main).
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build_global();
    }
    workers
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_physical_memory_bytes_nonzero_on_supported_platforms() {
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        {
            let phys = detect_physical_memory_bytes();
            assert!(phys > 0, "expected non-zero physical memory on this platform");
            assert!(phys >= 1024 * 1024, "expected at least 1 MiB");
        }
        // On unsupported platforms the function returns 0 gracefully.
    }

    #[test]
    fn apply_cpu_reservation_returns_at_least_one() {
        let workers = apply_cpu_reservation();
        assert!(workers >= 1, "must reserve at least one worker thread");
    }

    #[test]
    fn apply_cpu_reservation_caps_below_available_parallelism() {
        let n = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let workers = apply_cpu_reservation();
        if n > 1 {
            assert!(
                workers < n,
                "on a multi-core host, worker count must be < available_parallelism"
            );
        }
    }
}
