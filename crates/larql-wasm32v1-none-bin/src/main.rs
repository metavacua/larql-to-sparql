// On wasm32 targets there is no OS and no std — use no_std + alloc.
// On native targets (cargo check / dev builds) this is a normal std binary.
#![cfg_attr(target_arch = "wasm32", no_std)]
#![cfg_attr(target_arch = "wasm32", no_main)]

#[cfg(target_arch = "wasm32")]
extern crate alloc;

// dlmalloc provides the global allocator on wasm32 (no libc malloc).
#[cfg(target_arch = "wasm32")]
#[global_allocator]
static ALLOC: dlmalloc::GlobalDlmalloc = dlmalloc::GlobalDlmalloc;

// Required lang item for no_std binaries; never actually called via wasmtime
// because the smoke test doesn't panic.
#[cfg(target_arch = "wasm32")]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}

use larql_wasm32v1_none_lib::{gate, linalg};

// Native entry point (std binary).
#[cfg(not(target_arch = "wasm32"))]
fn main() {
    run();
}

// wasm32 entry point — exported so wasmtime can call it directly.
#[cfg(target_arch = "wasm32")]
#[no_mangle]
pub extern "C" fn _start() {
    run();
}

fn run() {
    // ── linalg (no heap required, slice-based) ───────────────────────────────
    let a = [1.0_f32, 0.0, 0.0];
    let b = [0.0_f32, 1.0, 0.0];
    let _ = linalg::dot(&a, &b);
    let _ = linalg::norm(&a);
    let _ = linalg::cosine(&a, &b);

    // ── gate (empty index; no actual heap allocation for zero-capacity vecs) ──
    let idx = gate::index::GateIndex::empty();
    let _ = gate::knn::gate_knn(&idx, 0, &a, 1);
}
