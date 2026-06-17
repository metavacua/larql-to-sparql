#![no_std]
extern crate alloc;

use alloc::{string::String, vec::Vec};

/// Provides tensor weight data to the kernel without exposing the loading mechanism.
/// Native: reads from mmap'd safetensors/gguf. Browser: from ArrayBuffer.
pub trait WeightProvider {
    fn tensor_f32(&self, name: &str) -> Option<&[f32]>;
    fn tensor_u8(&self, name: &str) -> Option<&[u8]>;
}

/// Provides read/write access to key-value storage.
/// Native: disk/memmap KV. Browser: IndexedDB.
pub trait KvStore {
    fn get(&self, key: &[u8]) -> Option<Vec<u8>>;
    fn set(&mut self, key: &[u8], value: &[u8]);
}

/// Routes WASM expert computation.
/// Native: wasmi runner. Browser: WebAssembly.instantiate.
pub trait ExpertDispatch {
    fn run(&self, expert_id: &str, op: &str, args: &[f32]) -> Vec<f32>;
}

/// Provides raw HTTP responses as bytes.
/// Native: reqwest blocking. Browser: fetch (via JS promise bridge).
pub trait HttpFetch {
    fn get_bytes(&self, url: &str) -> Result<Vec<u8>, String>;
}
