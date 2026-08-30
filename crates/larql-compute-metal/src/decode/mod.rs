// Clippy's `needless_option_as_deref` lint flags `state_dump.as_deref_mut()`
// inside the per-layer loops here. The lint is wrong in this context: the
// loop reuses `state_dump` across iterations, and pattern-matching the
// `Option<&mut DecodeStateDump>` directly would move it on the first
// iteration. `as_deref_mut()` re-borrows each iteration without moving.
#![allow(clippy::needless_option_as_deref)]

use super::*;

mod diag;
mod encode_attn;
mod encode_ffn;
mod encode_ple;
mod encode_post_ffn;
mod encode_qkv;
mod entry;
pub mod gpu_timing;
mod head;
mod kv_setup;
mod moe_combine;
mod moe_interleave;
pub mod profile;
mod setup;
mod token;

pub use head::HeadRequest;
pub(crate) use moe_interleave::InlineMoeCtx;
pub use profile::ProfileTimings;

pub(crate) const DEFAULT_KV_CACHE_MAX_SEQ: usize = 4096;

#[cfg(test)]
mod tests;
