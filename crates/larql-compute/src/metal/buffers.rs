//! Metal GPU buffer management — caching, zero-copy mmap, transient allocation.
//!
//! Weight buffers (mmap'd, constant) are cached by pointer address.
//! Transient buffers (Q8 input, activation, output) are allocated fresh each call.
//! Page-aligned mmap data uses newBufferWithBytesNoCopy (zero-copy GPU access).

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Mutex;

use metal::*;

/// Cache key: (pointer address, byte length) of the source data.
type CacheKey = (usize, usize);

/// Apple Silicon page size (16KB).
const PAGE_SIZE: usize = 16384;

/// Buffer cache for Metal GPU buffers.
/// Weight matrices from mmap'd files have stable addresses — their GPU buffers
/// are created once and reused for all subsequent calls.
pub struct BufferCache {
    device: Device,
    cache: Mutex<HashMap<CacheKey, Buffer>>,
}

impl BufferCache {
    pub fn new(device: &Device) -> Self {
        Self {
            device: device.clone(),
            cache: Mutex::new(HashMap::new()),
        }
    }

    /// Get or create a cached GPU buffer for f32 data.
    /// Uses zero-copy for page-aligned mmap data, copies otherwise.
    /// Empty slices (absent optional weight arrays such as Q4_K scale vectors)
    /// return a minimal 4-byte stub — Metal rejects zero-length allocations.
    pub fn get_f32(&self, data: &[f32]) -> Buffer {
        if data.is_empty() {
            // All empty-slice calls share the same stub key so the stub is
            // allocated once and reused.
            let stub_key: CacheKey = (0, 0);
            let mut cache = self.cache.lock().unwrap();
            if let Some(buf) = cache.get(&stub_key) { return buf.clone(); }
            let buf = self.device.new_buffer(4, MTLResourceOptions::StorageModeShared);
            cache.insert(stub_key, buf.clone());
            return buf;
        }

        let key: CacheKey = (data.as_ptr() as usize, data.len());
        let mut cache = self.cache.lock().unwrap();
        if let Some(buf) = cache.get(&key) { return buf.clone(); }

        let bytes = data.len() * 4;
        let ptr = data.as_ptr() as *const c_void;

        let buf = if Self::is_page_aligned(ptr, bytes) {
            self.device.new_buffer_with_bytes_no_copy(
                ptr as *mut c_void, bytes as u64,
                MTLResourceOptions::StorageModeShared, None,
            )
        } else {
            self.device.new_buffer_with_data(
                ptr, bytes as u64, MTLResourceOptions::StorageModeShared,
            )
        };

        cache.insert(key, buf.clone());
        buf
    }

    /// Get or create a cached GPU buffer for raw byte data (Q4 packed weights).
    /// Uses zero-copy for page-aligned mmap data.
    /// Empty slices return a minimal 4-byte stub (see `get_f32` for rationale).
    pub fn get_bytes(&self, data: &[u8]) -> Buffer {
        if data.is_empty() {
            let stub_key: CacheKey = (1, 0);
            let mut cache = self.cache.lock().unwrap();
            if let Some(buf) = cache.get(&stub_key) { return buf.clone(); }
            let buf = self.device.new_buffer(4, MTLResourceOptions::StorageModeShared);
            cache.insert(stub_key, buf.clone());
            return buf;
        }

        let key: CacheKey = (data.as_ptr() as usize, data.len());
        let mut cache = self.cache.lock().unwrap();
        if let Some(buf) = cache.get(&key) { return buf.clone(); }

        let ptr = data.as_ptr() as *const c_void;
        let bytes = data.len();

        let buf = if Self::is_page_aligned(ptr, bytes) {
            self.device.new_buffer_with_bytes_no_copy(
                ptr as *mut c_void, bytes as u64,
                MTLResourceOptions::StorageModeShared, None,
            )
        } else {
            self.device.new_buffer_with_data(
                ptr, bytes as u64, MTLResourceOptions::StorageModeShared,
            )
        };

        cache.insert(key, buf.clone());
        buf
    }

    /// Create a transient buffer (NOT cached — contents change each call).
    /// Used for Q8 input vectors, activations, and output buffers.
    pub fn transient_from_i8(&self, data: &[i8]) -> Buffer {
        self.device.new_buffer_with_data(
            data.as_ptr() as *const c_void,
            data.len() as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    /// Create a transient buffer from f32 data.
    pub fn transient_from_f32(&self, data: &[f32]) -> Buffer {
        self.device.new_buffer_with_data(
            data.as_ptr() as *const c_void,
            (data.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }


    /// Create an empty output buffer of given byte size.
    pub fn output(&self, bytes: u64) -> Buffer {
        self.device.new_buffer(bytes, MTLResourceOptions::StorageModeShared)
    }

    /// Number of cached buffers (for diagnostics).
    pub fn len(&self) -> usize {
        self.cache.lock().unwrap().len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn is_page_aligned(ptr: *const c_void, bytes: usize) -> bool {
        (ptr as usize).is_multiple_of(PAGE_SIZE) && bytes.is_multiple_of(PAGE_SIZE)
    }
}

/// Read `len` f32 values from a completed Metal buffer.
///
/// # Safety (encapsulated)
/// The caller must ensure the buffer has been committed and completed
/// (i.e., `cmd.wait_until_completed()` has returned) before calling this.
/// The pointer is valid for the buffer's lifetime — we immediately copy
/// into a new Vec, so no dangling reference is possible.
///
/// # Panics
/// Panics if the buffer's contents pointer is null or the buffer is
/// smaller than `len * sizeof(f32)` bytes.
pub fn read_buffer_f32(buf: &metal::Buffer, len: usize) -> Vec<f32> {
    let ptr = buf.contents() as *const f32;
    assert!(!ptr.is_null(), "Metal buffer contents pointer is null");
    assert!(
        buf.length() as usize >= len * std::mem::size_of::<f32>(),
        "Metal buffer too small: {} bytes, need {}",
        buf.length(),
        len * std::mem::size_of::<f32>(),
    );
    // SAFETY: ptr is non-null, buffer is large enough, and command buffer
    // has completed (caller invariant). Data is immediately copied to Vec.
    unsafe { std::slice::from_raw_parts(ptr, len).to_vec() }
}
