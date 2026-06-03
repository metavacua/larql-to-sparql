//! GPU-portable kernel surface for LARQL compute.
//!
//! This crate is a focused fork of `larql-compute-wasm32v1-none`, containing
//! only the ¬L∧¬M-portable kernels: no recursion, no unbounded heap allocation,
//! no C FFI, no host imports. It compiles to:
//!
//! - `wasm32v1-none` (the original wasm target — baseline)
//! - `wasm32-wasip1` (WASI Preview 1; tests run under wasmtime)
//! - `nvptx64-nvidia-cuda` (CUDA PTX; compile-only on standard runners)
//! - `spirv-unknown-vulkan1.2` (SPIR-V; requires rust-gpu toolchain)
//!
//! ## What is included
//!
//! Only the auto-parallelizable set from `crates/larql-compute/docs/compute-inventory.md`:
//! pointwise operations with static workgroup geometry and no barrier instructions.
//! Every function in this crate is a direct port of the corresponding function in
//! `larql-compute-wasm32v1-none/src/cpu/ops/`.
//!
//! ## What is excluded
//!
//! Everything that requires std: BLAS (`ndarray`), C FFI (Q4 kernel), rayon MoE
//! dispatch, serde, `larql-models`, and the `ComputeBackend` trait abstraction.
//! Those live in `larql-compute-wasm32v1-none` (for wasm32v1-none) and the root
//! workspace `larql-compute` (for native). This crate does NOT replicate them.
//!
//! ## Feature parity verification
//!
//! The tests in this crate run against the same inputs as the corresponding tests
//! in `larql-compute-wasm32v1-none::cpu::ops::geglu`. If results diverge, there
//! is a bug in the GPU port.

// no_std for targets without a standard library:
//   target_os = "none"  → wasm32v1-none (MVP WASM, no OS)
//   target_arch = "nvptx64"  → CUDA PTX (no host std)
//   target_arch = "spirv"    → SPIR-V shader (no host std)
//
// wasm32-wasip1 (target_os = "wasi") is NOT listed — it has WASI-provided
// std so tests can actually execute there under wasmtime.
#![cfg_attr(
    any(target_os = "none", target_arch = "nvptx64", target_arch = "spirv"),
    no_std
)]

pub mod portable;

pub use portable::{geglu_silu, silu};
