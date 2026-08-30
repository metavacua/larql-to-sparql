//! Bounds guard on the router → descriptor indirection (#229).
//!
//! `moe_descriptor_gather` reads `descs[selected_ids[slot]]`; `descs` has
//! `num_experts` entries and the ids come off the GPU router with nothing
//! between them and that read. An id past the end is a GPU address fault,
//! and a faulted command buffer returns from its wait like a finished one
//! (see `cb_status`). The kernel now clamps such an id to expert 0 and
//! bumps this counter; the host reads the counter after every decode step
//! and refuses the step if it moved. That turns a silent page fault into a
//! named, counted, refused event — and tells us whether the router is the
//! source of the fault at all.

use metal::{Buffer, Device, MTLResourceOptions};
use std::sync::atomic::{AtomicU32, Ordering};

/// One `u32` in shared memory, incremented by the gather kernel.
pub struct RouteGuard {
    pub counter: Buffer,
    /// Value last observed by the host, so a step reads a delta.
    last_seen: AtomicU32,
}

impl RouteGuard {
    pub fn new(device: &Device) -> Self {
        let counter = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
        // A fresh Metal buffer is zeroed, but say so rather than rely on it.
        // SAFETY: 4-byte shared buffer, no GPU work in flight yet.
        unsafe { std::ptr::write(counter.contents() as *mut u32, 0) };
        Self {
            counter,
            last_seen: AtomicU32::new(0),
        }
    }

    /// Current device-side total. Valid after the command buffers that
    /// could have bumped it have completed.
    pub fn total(&self) -> u32 {
        // SAFETY: shared 4-byte buffer owned by `self`; a torn read is
        // impossible for an aligned u32 and the value is only advisory.
        unsafe { std::ptr::read_volatile(self.counter.contents() as *const u32) }
    }

    /// Ids refused since the previous call. Zero on a healthy step.
    pub fn take_new_refusals(&self) -> u32 {
        let now = self.total();
        let before = self.last_seen.swap(now, Ordering::Relaxed);
        now.wrapping_sub(before)
    }
}

#[cfg(test)]
mod tests;
