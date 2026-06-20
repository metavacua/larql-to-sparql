//! Internal resource-guard state for OOM prevention.
//!
//! Provides `CapAllocator` (a `GlobalAlloc` wrapper) and `TrackedMmap`/
//! `TrackedMmapMut` RAII wrappers that keep `MMAP_BYTES` accurate.
//!
//! Binaries call `set_memory_cap(bytes)` once at startup (before any
//! allocation) to arm the guard.  The cap is derived from physical RAM
//! via `sysconf`; see `larql_compute::resource_guard::detect_physical_memory_bytes`.
//!
//! Invariants:
//! - `MEMORY_CAP == 0` → guard is disarmed (binary omitted the startup call;
//!   every allocation passes through to `System` unconditionally).
//! - `HEAP_BYTES + size > cap` → `alloc` returns null ptr → Rust calls
//!   `handle_alloc_error` → controlled process abort (not kernel OOM-kill).
//! - mmap-backed bytes are charged by `TrackedMmap`/`TrackedMmapMut` and
//!   counted in the cap check alongside heap bytes.

use memmap2::{Mmap, MmapMut};
use std::alloc::{GlobalAlloc, Layout, System};
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicUsize, Ordering};

static MEMORY_CAP: AtomicUsize = AtomicUsize::new(0);
static HEAP_BYTES: AtomicUsize = AtomicUsize::new(0);
static MMAP_BYTES: AtomicUsize = AtomicUsize::new(0);

/// Arm the allocator guard. Must be called once at process start before
/// any allocation occurs. `bytes` is the maximum allowed combined heap +
/// mmap consumption; passing 0 leaves the guard disarmed.
pub fn set_memory_cap(bytes: usize) {
    MEMORY_CAP.store(bytes, Ordering::Relaxed);
}

/// Charge `bytes` of mmap usage against the combined cap.
/// Call immediately after a successful `Mmap::map` at the mmap site.
/// Use `TrackedMmap` / `TrackedMmapMut` to get RAII release automatically.
pub fn charge_mmap(bytes: usize) {
    MMAP_BYTES.fetch_add(bytes, Ordering::Relaxed);
}

/// Release `bytes` of mmap usage.  Called from `TrackedMmap` / `TrackedMmapMut`
/// `Drop` impls; also callable directly for permanent mmaps (no-release needed).
pub fn release_mmap(bytes: usize) {
    MMAP_BYTES.fetch_sub(bytes, Ordering::Relaxed);
}

/// Heap allocator that enforces `MEMORY_CAP`.
///
/// Declared with `#[global_allocator]` in each binary crate entry point.
/// Must be a unit struct — the actual state lives in module-level statics.
pub struct CapAllocator;

unsafe impl GlobalAlloc for CapAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let sz = layout.size();
        // Always track — dealloc always decrements, so alloc must always increment.
        let prev = HEAP_BYTES.fetch_add(sz, Ordering::Relaxed);
        let cap = MEMORY_CAP.load(Ordering::Relaxed);
        if cap > 0 && prev + sz + MMAP_BYTES.load(Ordering::Relaxed) > cap {
            // Over cap: undo and return null → triggers handle_alloc_error.
            HEAP_BYTES.fetch_sub(sz, Ordering::Relaxed);
            return core::ptr::null_mut();
        }
        let ptr = System.alloc(layout);
        if ptr.is_null() {
            // System alloc failed: undo the optimistic add.
            HEAP_BYTES.fetch_sub(sz, Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        HEAP_BYTES.fetch_sub(layout.size(), Ordering::Relaxed);
        System.dealloc(ptr, layout);
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let sz = layout.size();
        let prev = HEAP_BYTES.fetch_add(sz, Ordering::Relaxed);
        let cap = MEMORY_CAP.load(Ordering::Relaxed);
        if cap > 0 && prev + sz + MMAP_BYTES.load(Ordering::Relaxed) > cap {
            HEAP_BYTES.fetch_sub(sz, Ordering::Relaxed);
            return core::ptr::null_mut();
        }
        let ptr = System.alloc_zeroed(layout);
        if ptr.is_null() {
            HEAP_BYTES.fetch_sub(sz, Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if new_size > layout.size() {
            let delta = new_size - layout.size();
            let prev = HEAP_BYTES.fetch_add(delta, Ordering::Relaxed);
            let cap = MEMORY_CAP.load(Ordering::Relaxed);
            if cap > 0 && prev + delta + MMAP_BYTES.load(Ordering::Relaxed) > cap {
                HEAP_BYTES.fetch_sub(delta, Ordering::Relaxed);
                return core::ptr::null_mut();
            }
            let new_ptr = System.realloc(ptr, layout, new_size);
            if new_ptr.is_null() {
                HEAP_BYTES.fetch_sub(delta, Ordering::Relaxed);
            }
            new_ptr
        } else {
            // Shrinking: update counter first, then realloc.
            HEAP_BYTES.fetch_sub(layout.size() - new_size, Ordering::Relaxed);
            System.realloc(ptr, layout, new_size)
        }
    }
}

/// RAII wrapper for `memmap2::Mmap` that charges/releases `MMAP_BYTES`.
pub struct TrackedMmap(Mmap);

impl TrackedMmap {
    /// Map a file read-only.  Charges `MMAP_BYTES` on success; releases on drop.
    ///
    /// # Safety
    /// Same as `memmap2::Mmap::map`: the file must not be modified while
    /// the mapping is live.
    pub unsafe fn map(file: &std::fs::File) -> std::io::Result<Self> {
        let m = Mmap::map(file)?;
        charge_mmap(m.len());
        Ok(Self(m))
    }

    /// Wrap an existing `Mmap` and charge its byte count.
    pub fn from_raw(m: Mmap) -> Self {
        charge_mmap(m.len());
        Self(m)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl Deref for TrackedMmap {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        &self.0
    }
}

impl AsRef<[u8]> for TrackedMmap {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl Drop for TrackedMmap {
    fn drop(&mut self) {
        release_mmap(self.0.len());
    }
}

/// RAII wrapper for `memmap2::MmapMut` (anonymous or file-backed) that
/// charges/releases `MMAP_BYTES`.
pub struct TrackedMmapMut(MmapMut);

impl TrackedMmapMut {
    /// Create an anonymous (RAM-backed) mapping of `length` bytes.
    pub fn map_anon(length: usize) -> std::io::Result<Self> {
        let m = MmapMut::map_anon(length)?;
        charge_mmap(m.len());
        Ok(Self(m))
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Convert to a read-only `TrackedMmap`, preserving the charge.
    pub fn make_read_only(self) -> std::io::Result<TrackedMmap> {
        let _len = self.0.len();
        // Consume self without releasing (we're transferring the charge).
        let inner = unsafe { std::ptr::read(&self.0) };
        std::mem::forget(self);
        let ro = inner.make_read_only()?;
        // The charge was already recorded; TrackedMmap::from_raw would double-charge,
        // so bypass it and wrap directly.
        Ok(TrackedMmap(ro))
        // Note: len bytes remain charged under MMAP_BYTES via the returned TrackedMmap.
    }
}

impl Deref for TrackedMmapMut {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        &self.0
    }
}

impl DerefMut for TrackedMmapMut {
    fn deref_mut(&mut self) -> &mut [u8] {
        &mut self.0
    }
}

impl AsRef<[u8]> for TrackedMmapMut {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl Drop for TrackedMmapMut {
    fn drop(&mut self) {
        release_mmap(self.0.len());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_and_read_cap() {
        // Use a high value so we don't interfere with ongoing allocations.
        set_memory_cap(usize::MAX / 2);
        assert_eq!(MEMORY_CAP.load(Ordering::Relaxed), usize::MAX / 2);
        // Reset to 0 so the disarmed fast-path is restored for other tests.
        set_memory_cap(0);
    }

    #[test]
    fn charge_and_release_mmap() {
        let before = MMAP_BYTES.load(Ordering::Relaxed);
        charge_mmap(1024);
        assert_eq!(MMAP_BYTES.load(Ordering::Relaxed), before + 1024);
        release_mmap(1024);
        assert_eq!(MMAP_BYTES.load(Ordering::Relaxed), before);
    }

    #[test]
    fn tracked_mmap_mut_anon_charges_and_releases() {
        let before = MMAP_BYTES.load(Ordering::Relaxed);
        {
            let m = TrackedMmapMut::map_anon(4096).expect("anon mmap");
            assert_eq!(MMAP_BYTES.load(Ordering::Relaxed), before + m.len());
        }
        assert_eq!(MMAP_BYTES.load(Ordering::Relaxed), before);
    }

    #[test]
    fn tracked_mmap_mut_make_read_only_preserves_charge() {
        let before = MMAP_BYTES.load(Ordering::Relaxed);
        let mut m = TrackedMmapMut::map_anon(4096).expect("anon mmap");
        m[0] = 42;
        let len = m.len();
        let ro = m.make_read_only().expect("make_read_only");
        // Charge should still be live (now under ro).
        assert_eq!(MMAP_BYTES.load(Ordering::Relaxed), before + len);
        assert_eq!(ro[0], 42);
        drop(ro);
        assert_eq!(MMAP_BYTES.load(Ordering::Relaxed), before);
    }
}
