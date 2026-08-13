//! Registered weight regions — zero-copy Metal buffers over whole mmaps.
//!
//! The per-expert MoE decode previously memcpy'd every selected expert's
//! bytes into staging buffers each layer each token (GPT-OSS: top-4 ×
//! ~22 MB × 24 layers ≈ 2.1 GB of CPU memcpy per token — the dominant
//! decode cost once attention moved to the GPU). Registering each
//! layer-weights mmap ONCE as a `newBufferWithBytesNoCopy` buffer lets
//! the dispatch bind an expert as a byte offset into the shared buffer:
//! no staging, no duplication, no per-token CPU bandwidth.
//!
//! A region must be page-aligned at its base — true of every mmap by
//! construction — and lives for the process, the same contract
//! [`BufferCache::get_bytes`] already imposes on cached weight slices.

use metal::{Buffer, MTLResourceOptions};

use super::{BufferCache, PAGE_SIZE};

/// One registered region: `[start, start + len)` in host memory, plus the
/// no-copy Metal buffer that aliases it.
pub(super) struct Region {
    pub start: usize,
    pub len: usize,
    pub buf: Buffer,
}

impl BufferCache {
    /// Register `region` for zero-copy sub-slice resolution.
    ///
    /// Returns `false` (and registers nothing) when the base pointer is
    /// not page-aligned — the caller falls back to staged copies. Calling
    /// again with an already-registered base is a cheap no-op.
    ///
    /// The Metal buffer length is `region.len` rounded UP to the page
    /// size: `newBufferWithBytesNoCopy` requires a page-multiple length,
    /// and an mmap maps whole pages, so the bytes between `len` and the
    /// page boundary are readable (zero-filled by the kernel) even though
    /// they are past the slice. That tail is never bound at an offset —
    /// resolution is bounded by `len` — it only pads the allocation.
    ///
    /// # Contract
    ///
    /// `region` must point into an allocation that is page-granular
    /// (mmap-backed), never moves, and outlives this cache — the same
    /// stability contract as [`Self::get_bytes`].
    pub fn register_region(&self, region: &[u8]) -> bool {
        let start = region.as_ptr() as usize;
        if !start.is_multiple_of(PAGE_SIZE) || region.is_empty() {
            return false;
        }
        let mut regions = self.regions.lock().unwrap();
        if regions.iter().any(|r| r.start == start) {
            return true;
        }
        let rounded = region.len().div_ceil(PAGE_SIZE) * PAGE_SIZE;
        let buf = self.device.new_buffer_with_bytes_no_copy(
            region.as_ptr() as *mut std::ffi::c_void,
            rounded as u64,
            MTLResourceOptions::StorageModeShared,
            None,
        );
        regions.push(Region {
            start,
            len: region.len(),
            buf,
        });
        true
    }

    /// Resolve `sub` to `(buffer, byte_offset)` if it lies wholly inside a
    /// registered region. `None` → the caller stages a copy instead.
    pub fn resolve_region(&self, sub: &[u8]) -> Option<(Buffer, u64)> {
        if sub.is_empty() {
            return None;
        }
        let p = sub.as_ptr() as usize;
        let regions = self.regions.lock().unwrap();
        for r in regions.iter() {
            if p >= r.start && p + sub.len() <= r.start + r.len {
                return Some((r.buf.clone(), (p - r.start) as u64));
            }
        }
        None
    }

    /// Number of registered regions (test/diagnostic surface).
    pub fn region_count(&self) -> usize {
        self.regions.lock().unwrap().len()
    }
}

#[cfg(test)]
mod tests {
    use super::super::BufferCache;
    use metal::Device;

    fn dev() -> Device {
        Device::system_default().expect("Metal device available on test host")
    }

    /// Page-aligned by construction — the production contract's shape.
    fn anon_region(len: usize) -> memmap2::Mmap {
        let mut m = memmap2::MmapMut::map_anon(len).expect("anon mmap");
        for (i, b) in m.iter_mut().enumerate() {
            *b = (i % 251) as u8;
        }
        m.make_read_only().expect("read-only")
    }

    /// Sub-slices anywhere inside a registered region resolve to the
    /// region's buffer at the right byte offset; slices outside miss.
    #[test]
    fn resolve_returns_offset_within_registered_region() {
        let cache = BufferCache::new(&dev());
        let region = anon_region(3 * super::PAGE_SIZE / 2); // non-page-multiple len
        assert!(cache.register_region(&region[..]));

        let sub = &region[4096..4096 + 512];
        let (buf, off) = cache.resolve_region(sub).expect("inside must resolve");
        assert_eq!(off, 4096);
        // The buffer aliases the region: bytes at the offset match.
        let p = buf.contents() as *const u8;
        let via_buf = unsafe { std::slice::from_raw_parts(p.add(off as usize), sub.len()) };
        assert_eq!(via_buf, sub);

        let other = anon_region(super::PAGE_SIZE);
        assert!(
            cache.resolve_region(&other[..64]).is_none(),
            "unregistered allocation must miss"
        );
    }

    /// A sub-slice extending past the region's logical end must miss —
    /// the rounded-up buffer tail is allocation padding, never data.
    #[test]
    fn resolve_rejects_slices_past_the_logical_end() {
        let cache = BufferCache::new(&dev());
        let len = super::PAGE_SIZE + 100; // logical end mid-page
        let region = anon_region(len);
        assert!(cache.register_region(&region[..]));
        assert!(cache.resolve_region(&region[len - 50..len]).is_some());
        // Reconstruct a slice crossing the logical end via raw parts —
        // the mmap maps the whole final page, so this is readable memory
        // that is nonetheless OUTSIDE the registered data.
        let past = unsafe { std::slice::from_raw_parts(region.as_ptr().add(len - 10), 20) };
        assert!(cache.resolve_region(past).is_none());
    }

    /// Re-registering the same base is a no-op; misaligned bases and
    /// empty regions refuse.
    #[test]
    fn register_dedupes_and_rejects_unusable_regions() {
        let cache = BufferCache::new(&dev());
        let region = anon_region(super::PAGE_SIZE);
        assert!(cache.register_region(&region[..]));
        assert!(cache.register_region(&region[..]), "re-register is a no-op");
        assert_eq!(cache.region_count(), 1);

        // Interior pointer — not page-aligned.
        assert!(!cache.register_region(&region[8..]));
        assert!(!cache.register_region(&region[..0]));
        assert_eq!(cache.region_count(), 1);
    }
}
