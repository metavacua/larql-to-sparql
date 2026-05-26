//! WASI-portable Vindex layer.
//!
//! This crate contains the portions of `larql-vindex` that compile for
//! `wasm32-wasip1` (and native). It depends on `larql-vindex-wasm32uu`
//! for the pure in-memory compute core and adds:
//!
//! - `std::fs` / `std::io` weight loading via `safetensors` (pure Rust)
//! - Config, index, engine, quant, trie, cache — everything that does not
//!   require C build scripts, OS mmap, OS sockets, or OS threads.
//!
//! Not included here (goes in `larql-vindex-os`):
//! - `extract/` — tokenizers (onig C backend), hf-hub, reqwest
//! - `format/huggingface/` — hf-hub, reqwest (OS sockets)
//! - `format/load.rs` — memmap2 (OS mmap)
//! - `mmap_util.rs` — memmap2 + libc
//! - `clustering/probe.rs` — tokenizers
//! - Rayon-parallelised compute paths — OS threads
//!
//! # Dependency chain
//! `larql-vindex-wasm32uu` → `larql-vindex-wasi` → `larql-vindex-os`
