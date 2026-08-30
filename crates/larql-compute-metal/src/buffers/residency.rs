//! Explicit Metal residency for the registered weight regions.
//!
//! Each decode command buffer references the WHOLE `layer_NN.weights` mmap as
//! one `newBufferWithBytesNoCopy` allocation (~697 MB for gpt-oss) even though
//! the four selected experts read only ~87 MB of it — the grouped kernel's
//! `single_base` requirement forces one base allocation. With implicit
//! residency, Metal does its resource-preparation bookkeeping per command
//! buffer, so a 24-layer token pays it 24 times over ~697 MB each.
//!
//! `MTLResidencySet` (macOS 15+) exists to declare that work once:
//! `requestResidency` performs the preparation ahead of any commit, and
//! attaching the set to the **command queue** applies it to every command
//! buffer from that queue automatically. Apple documents queue attachment as
//! the efficient form for resources needed for the application's lifetime.
//!
//! `metal-rs` 0.29 has no binding for any of this, so the selectors are called
//! directly — the same technique `decode/gpu_timing.rs` uses for
//! `GPUStartTime`.
//!
//! **This changes nothing numerical.** Same buffers, same offsets, same
//! kernels, same command-buffer count, same routes. It is an A/B on resource
//! preparation alone, which is what makes it a usable control.

use metal::foreign_types::ForeignType;
use metal::{Buffer, CommandQueue, Device};
use objc::runtime::Object;
use objc::{class, msg_send, sel, sel_impl};

/// Which residency arm to run. The ladder separates *declaring* residency
/// from *pulling the preparation forward* — if B wins and C adds nothing, the
/// cost was the per-command-buffer declaration; if only C wins, the cost was
/// the preparation itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResidencyArm {
    /// A — current behaviour: implicit, per-command-buffer residency.
    Implicit,
    /// B — build a residency set over every registered region and attach it
    /// to the command queue. No explicit `requestResidency` call.
    QueueSet,
    /// C — B, plus `requestResidency` so the preparation happens before the
    /// first timed command buffer rather than lazily.
    QueueSetRequested,
}

impl ResidencyArm {
    /// `LARQL_RESIDENCY_SET` — unset/`0` = A, `1` = B, `2` = C.
    pub fn from_env() -> Self {
        match larql_compute::options::env_usize(ENV_RESIDENCY_SET) {
            Some(1) => Self::QueueSet,
            Some(2) => Self::QueueSetRequested,
            _ => Self::Implicit,
        }
    }
}

pub const ENV_RESIDENCY_SET: &str = "LARQL_RESIDENCY_SET";

/// An `MTLResidencySet` holding the registered weight allocations.
///
/// Owns a `+1` reference; released on drop. `Send`/`Sync` are not derived —
/// the set is built once on the thread that registers regions and then only
/// read by Metal, so it is kept behind the same mutex as the regions.
pub struct ResidencySet {
    inner: *mut Object,
    committed: bool,
}

// SAFETY: `inner` is an `MTLResidencySet`, an Objective-C object whose
// lifetime is managed by retain/release. It is built once during model setup
// and thereafter only read by Metal itself (via the command queue it is
// attached to); no Rust code mutates it across threads. `MetalBackend` is
// `Send + Sync` and holds this through `BufferCache`, so the raw pointer needs
// these to match what every other `metal-rs` handle in the backend already
// asserts about Metal objects.
unsafe impl Send for ResidencySet {}
unsafe impl Sync for ResidencySet {}

impl ResidencySet {
    /// Build an empty residency set on `device`. `None` when the runtime
    /// predates `MTLResidencySet` (pre-macOS 15) or the descriptor class is
    /// unavailable — the caller silently keeps implicit residency, which is
    /// correct, just slower.
    pub fn new(device: &Device) -> Option<Self> {
        unsafe {
            let dev: *mut Object = device.as_ptr() as *mut Object;
            let responds: bool =
                msg_send![dev, respondsToSelector: sel!(newResidencySetWithDescriptor:error:)];
            if !responds {
                return None;
            }
            let desc_cls = class!(MTLResidencySetDescriptor);
            let desc: *mut Object = msg_send![desc_cls, alloc];
            let desc: *mut Object = msg_send![desc, init];
            if desc.is_null() {
                return None;
            }
            let mut err: *mut Object = std::ptr::null_mut();
            let set: *mut Object =
                msg_send![dev, newResidencySetWithDescriptor: desc error: &mut err];
            let _: () = msg_send![desc, release];
            if set.is_null() {
                return None;
            }
            Some(Self {
                inner: set,
                committed: false,
            })
        }
    }

    /// Add one allocation. Takes effect only after [`Self::commit`].
    pub fn add_buffer(&self, buf: &Buffer) {
        unsafe {
            let alloc: *mut Object = buf.as_ptr() as *mut Object;
            let _: () = msg_send![self.inner, addAllocation: alloc];
        }
    }

    /// Apply pending additions.
    pub fn commit(&mut self) {
        unsafe {
            let _: () = msg_send![self.inner, commit];
        }
        self.committed = true;
    }

    /// Ask Metal to make the committed allocations resident now, rather than
    /// as a side effect of the first command buffer that references them.
    pub fn request_residency(&self) {
        unsafe {
            let _: () = msg_send![self.inner, requestResidency];
        }
    }

    /// Attach to a command queue so every command buffer from that queue
    /// inherits the set — the form Apple recommends over per-command-buffer
    /// attachment for long-lived resources.
    pub fn add_to_queue(&self, queue: &CommandQueue) -> bool {
        unsafe {
            let q: *mut Object = queue.as_ptr() as *mut Object;
            let responds: bool = msg_send![q, respondsToSelector: sel!(addResidencySet:)];
            if !responds {
                return false;
            }
            let _: () = msg_send![q, addResidencySet: self.inner];
            true
        }
    }

    pub fn is_committed(&self) -> bool {
        self.committed
    }
}

impl Drop for ResidencySet {
    fn drop(&mut self) {
        unsafe {
            let _: () = msg_send![self.inner, release];
        }
    }
}

#[cfg(test)]
pub(in crate::buffers) mod tests;
