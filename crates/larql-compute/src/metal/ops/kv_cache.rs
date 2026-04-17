//! KV cache management and cached attention dispatch.
//!
//! Per-layer Metal buffers for cached K/V vectors. Grows with generation.
//! At decode time: append new K/V, then attend Q against full cache.

use std::ffi::c_void;
use metal::*;

use crate::metal::buffers::BufferCache;

/// KV cache for one layer — pre-allocated Metal buffers.
pub struct LayerKVCache {
    pub k_cache: Buffer,  // [max_seq, num_kv_heads, head_dim] f32
    pub v_cache: Buffer,  // same
    pub current_len: usize,
    pub max_seq: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl LayerKVCache {
    /// Create empty KV cache for one layer.
    pub fn new(bufs: &BufferCache, max_seq: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        let size = (max_seq * num_kv_heads * head_dim * 4) as u64;
        Self {
            k_cache: bufs.output(size),
            v_cache: bufs.output(size),
            current_len: 0,
            max_seq,
            num_kv_heads,
            head_dim,
        }
    }

    /// Reset cache (for new prompt).
    pub fn clear(&mut self) {
        self.current_len = 0;
    }
}

/// Full KV cache for all layers.
pub struct KVCache {
    pub layers: Vec<LayerKVCache>,
}

impl KVCache {
    /// Allocate a KV cache with uniform per-layer dims — the Llama / Mistral
    /// / Gemma 3 case where every layer shares num_kv_heads and head_dim.
    pub fn new(bufs: &BufferCache, num_layers: usize, max_seq: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        let layers = (0..num_layers)
            .map(|_| LayerKVCache::new(bufs, max_seq, num_kv_heads, head_dim))
            .collect();
        Self { layers }
    }

    /// Allocate with per-layer shapes — Gemma 4 31B alternates sliding
    /// (num_kv=16, head_dim=256) with global (num_kv=4, head_dim=512) layers,
    /// so a single uniform allocation would either over-size globals or
    /// under-size slidings and produce wrong attention reads.
    ///
    /// `shapes[i]` is `(num_kv_heads_i, head_dim_i)` for layer i.
    pub fn new_per_layer(bufs: &BufferCache, shapes: &[(usize, usize)], max_seq: usize) -> Self {
        let layers = shapes
            .iter()
            .map(|&(num_kv, hd)| LayerKVCache::new(bufs, max_seq, num_kv, hd))
            .collect();
        Self { layers }
    }

    pub fn clear(&mut self) {
        for layer in &mut self.layers { layer.clear(); }
    }

    pub fn current_len(&self) -> usize {
        self.layers.first().map(|l| l.current_len).unwrap_or(0)
    }
}

/// Encode KV append dispatch into an existing encoder.
/// The encoder is NOT ended — caller continues adding dispatches.
#[allow(clippy::too_many_arguments)]
pub fn encode_kv_append(
    enc: &ComputeCommandEncoderRef,
    cache: &LayerKVCache,
    append_pipeline: &ComputePipelineState,
    new_k: &Buffer,
    new_v: &Buffer,
) {
    let pos = cache.current_len as u32;
    let num_kv = cache.num_kv_heads as u32;
    let hd = cache.head_dim as u32;
    let total = cache.num_kv_heads * cache.head_dim;

    enc.set_compute_pipeline_state(append_pipeline);
    enc.set_buffer(0, Some(new_k), 0);
    enc.set_buffer(1, Some(new_v), 0);
    enc.set_buffer(2, Some(&cache.k_cache), 0);
    enc.set_buffer(3, Some(&cache.v_cache), 0);
    enc.set_bytes(4, 4, &pos as *const u32 as *const c_void);
    enc.set_bytes(5, 4, &num_kv as *const u32 as *const c_void);
    enc.set_bytes(6, 4, &hd as *const u32 as *const c_void);
    enc.dispatch_threads(
        MTLSize::new(total as u64, 1, 1),
        MTLSize::new(256.min(total as u64), 1, 1),
    );
}

/// Encode KV attend dispatch into an existing encoder.
/// The encoder is NOT ended — caller continues adding dispatches.
#[allow(clippy::too_many_arguments)]
pub fn encode_kv_attend(
    enc: &ComputeCommandEncoderRef,
    cache: &LayerKVCache,
    attend_pipeline: &ComputePipelineState,
    q: &Buffer,
    out: &Buffer,
    num_q_heads: usize,
    scale: f32,
    window_size: u32,
) {
    let t_val = (cache.current_len + 1) as u32;
    let hd = cache.head_dim as u32;
    let num_q_val = num_q_heads as u32;
    let num_kv = cache.num_kv_heads as u32;

    enc.set_compute_pipeline_state(attend_pipeline);
    enc.set_buffer(0, Some(q), 0);
    enc.set_buffer(1, Some(&cache.k_cache), 0);
    enc.set_buffer(2, Some(&cache.v_cache), 0);
    enc.set_buffer(3, Some(out), 0);
    enc.set_bytes(4, 4, &t_val as *const u32 as *const c_void);
    enc.set_bytes(5, 4, &hd as *const u32 as *const c_void);
    enc.set_bytes(6, 4, &num_q_val as *const u32 as *const c_void);
    enc.set_bytes(7, 4, &num_kv as *const u32 as *const c_void);
    enc.set_bytes(8, 4, &scale as *const f32 as *const c_void);
    enc.set_bytes(9, 4, &window_size as *const u32 as *const c_void);
    enc.dispatch_thread_groups(
        MTLSize::new(num_q_heads as u64, 1, 1),
        MTLSize::new(256.min(cache.head_dim as u64), 1, 1),
    );
}

/// Append new K/V to cache and run attention in one command buffer.
/// Returns attention output [num_q_heads, head_dim].
/// Legacy API — creates its own encoders. For merged pipelines, use
/// encode_kv_append + encode_kv_attend directly.
#[allow(clippy::too_many_arguments)]
pub fn append_and_attend(
    cmd: &CommandBufferRef,
    cache: &mut LayerKVCache,
    append_pipeline: &ComputePipelineState,
    attend_pipeline: &ComputePipelineState,
    new_k: &Buffer,
    new_v: &Buffer,
    q: &Buffer,
    out: &Buffer,
    num_q_heads: usize,
    scale: f32,
) {
    // Append in its own encoder
    {
        let enc = cmd.new_compute_command_encoder();
        encode_kv_append(enc, cache, append_pipeline, new_k, new_v);
        enc.end_encoding();
    }

    // Attend in its own encoder (reads from cache written by append)
    {
        let enc = cmd.new_compute_command_encoder();
        encode_kv_attend(enc, cache, attend_pipeline, q, out, num_q_heads, scale, 0);
        enc.end_encoding();
    }

    cache.current_len += 1;
}
