//! Decode-step attention — GQA for a single new token against a
//! growing KV cache.
//!
//! Prefill does full O(seq²) attention and returns K/V per layer. Decode
//! runs one token at a time with O(cached_len) attention: Q for the new
//! token attends against [K_cache | K_new] and [V_cache | V_new], with
//! no causal mask needed (the new query is at the end and can see every
//! cached position + itself).
//!
//! The per-layer K/V cache type ([`larql_kv::KvCache`]) and the
//! generation loops that drive prefill→decode now live in `larql-kv`
//! (the canonical engine state shape) — see `larql-kv/src/cache.rs`
//! and `larql-kv/src/generation.rs`.

mod dispatch;
mod gqa_step;
mod inplace;
mod q4k_direct;
#[cfg(test)]
mod tests;

pub use dispatch::{
    q4k_direct_attn_enabled, run_attention_block_decode_step_auto,
    run_attention_block_decode_step_backend,
};
pub use gqa_step::{
    gqa_attention_decode_step, gqa_attention_decode_step_windowed, run_attention_block_decode_step,
};
pub use inplace::{
    run_attention_block_decode_step_auto_inplace,
    run_attention_block_decode_step_q4k_direct_inplace,
};
pub use q4k_direct::{
    decode_step_attend_q4k_direct, decode_step_project_q4k_direct,
    run_attention_block_decode_step_q4k_direct, Q4kDecodeProj,
};
