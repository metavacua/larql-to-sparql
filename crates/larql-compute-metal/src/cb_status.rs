//! Command-buffer completion status — the check every `wait_until_completed`
//! site was missing.
//!
//! `waitUntilCompleted` returns for a *failed* buffer just as it does for a
//! finished one: `status == Error`, immediately. Once the GPU has faulted,
//! later buffers on the queue may be dropped outright ("ignored for causing
//! prior/excessive GPU errors"), which also completes instantly. Nothing
//! downstream can tell — the output buffers simply hold whatever was there
//! before — so a caller that only waits will read stale or garbage results
//! at full speed. See #229 and `docs/kv-attention-scaling.md`, "The fault is
//! on main and predates seqpar": impossible ~0.5 ms/token decode steps,
//! token ids past the vocabulary, EOS at token 1.
//!
//! This module names the condition. It does not yet change control flow —
//! that is the fix, and it needs the evidence this produces first.

use metal::foreign_types::ForeignTypeRef;
use metal::{CommandBufferRef, MTLCommandBufferStatus};
use objc::runtime::Object;
use objc::{msg_send, sel, sel_impl};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Buffers observed in a non-`Completed` state since process start.
static NON_COMPLETED: AtomicUsize = AtomicUsize::new(0);

/// How many command buffers this process has seen finish in any state other
/// than `Completed`. Zero on a healthy process.
pub fn non_completed_count() -> usize {
    NON_COMPLETED.load(Ordering::Relaxed)
}

/// Wait for `cmd`, then inspect it. This is the only sanctioned way to
/// wait on a command buffer in this crate: `wait_until_completed` alone
/// cannot distinguish a finished buffer from a failed or ignored one, and
/// a test pins that no production site calls it directly. Callers with a
/// result channel propagate the `Err`; callers without one at least leave
/// the line in the log.
pub fn wait_checked(cmd: &CommandBufferRef, site: &'static str) -> Result<(), String> {
    cmd.wait_until_completed();
    check_completed(cmd, site)
}

/// Inspect `cmd` after `wait_until_completed`. Returns `Ok(())` for
/// `Completed`; otherwise records the event, prints one line naming the
/// site, the status and Metal's own error description, and returns that
/// description so a caller can decide what to do with a poisoned step.
pub fn check_completed(cmd: &CommandBufferRef, site: &'static str) -> Result<(), String> {
    let status = cmd.status();
    if status == MTLCommandBufferStatus::Completed {
        return Ok(());
    }
    let n = NON_COMPLETED.fetch_add(1, Ordering::Relaxed) + 1;
    let desc = error_description(cmd).unwrap_or_else(|| "<no NSError>".to_string());
    let msg = format!("command buffer at {site} finished with status {status:?} (#{n}): {desc}");
    eprintln!("[metal] {msg}");
    Err(msg)
}

/// UTF-8 contents of an `NSString*`, or `None` for nil.
unsafe fn ns_string(ns: *mut Object) -> Option<String> {
    if ns.is_null() {
        return None;
    }
    let utf8: *const std::os::raw::c_char = msg_send![ns, UTF8String];
    if utf8.is_null() {
        return None;
    }
    Some(
        std::ffi::CStr::from_ptr(utf8)
            .to_string_lossy()
            .into_owned(),
    )
}

/// `-[MTLCommandBuffer error].localizedDescription`, or `None` when the
/// buffer carries no NSError (e.g. `NotEnqueued`, or an ignored buffer on
/// some OS versions).
fn error_description(cmd: &CommandBufferRef) -> Option<String> {
    // SAFETY: `cmd` is a live MTLCommandBuffer; `error` returns an
    // autoreleased NSError* or nil, and `localizedDescription` an
    // autoreleased NSString*. Both are read, never retained or released.
    unsafe {
        let obj: *mut Object = cmd.as_ptr() as *mut Object;
        let err: *mut Object = msg_send![obj, error];
        if err.is_null() {
            return None;
        }
        ns_string(msg_send![err, localizedDescription])
    }
}

#[cfg(test)]
mod tests;
