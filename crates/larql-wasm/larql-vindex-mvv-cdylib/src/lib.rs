//! Certification harness for the Minimal-Viable Vindex.
//!
//! This crate is **not** part of the portable kernel — it is verification
//! scaffolding. It exports the MVV *pure* gate-KNN kernel as `extern "C"` so the
//! workspace produces a standalone `wasm32v1-none` module the wasm-safe
//! certifier can parse. The smoke export builds a tiny in-memory index and runs
//! `gate_knn`, pulling only the pure kernel into the reachable closure (no
//! `serde_json` reader), so the certifier checks the floor.
//!
//! The `#[global_allocator]` + `#[panic_handler]` below are the leaf-artifact
//! runtime items a cdylib must supply on no_std wasm32v1-none. They are
//! explicitly NOT pure-kernel code; dlmalloc is itself import-free and
//! `call_indirect`-free, so it does not perturb the certification result.
#![cfg_attr(target_arch = "wasm32", no_std)]

extern crate alloc;
use alloc::vec::Vec;

#[cfg(target_arch = "wasm32")]
#[global_allocator]
static ALLOC: dlmalloc::GlobalDlmalloc = dlmalloc::GlobalDlmalloc;

#[cfg(target_arch = "wasm32")]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}

use larql_vindex::index::mvv::{MvvDescriptor, MvvIndex, MvvLayer, StorageDtype};

/// Smoke export: build a tiny f32 MVV (2 features, hidden 2) and run the pure
/// gate-KNN kernel. Returns the top feature index, or a sentinel on error.
/// Existence + reachability is the point — it links the kernel into the module.
#[no_mangle]
pub extern "C" fn mvv_smoke() -> u32 {
    // blob = [[1,0],[0,1]] as 4 little-endian f32 = 16 bytes.
    let mut blob: Vec<u8> = Vec::new();
    for x in [1.0f32, 0.0, 0.0, 1.0] {
        blob.extend_from_slice(&x.to_le_bytes());
    }
    let desc = MvvDescriptor {
        version: 2,
        num_layers: 1,
        hidden_size: 2,
        dtype: StorageDtype::F32,
        layers: alloc::vec![MvvLayer {
            layer: 0,
            num_features: 2,
            offset: 0,
            length: 16,
        }],
    };
    match MvvIndex::new(desc, blob) {
        Ok(idx) => match idx.gate_knn(0, &[0.0, 1.0], 1) {
            Ok(v) => v.first().map(|(f, _)| *f as u32).unwrap_or(u32::MAX),
            Err(_) => u32::MAX - 1,
        },
        Err(_) => u32::MAX - 2,
    }
}
