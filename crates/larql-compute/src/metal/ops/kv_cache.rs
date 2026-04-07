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
    pub fn new(bufs: &BufferCache, num_layers: usize, max_seq: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        let layers = (0..num_layers)
            .map(|_| LayerKVCache::new(bufs, max_seq, num_kv_heads, head_dim))
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

/// Append new K/V to cache and run attention in one command buffer.
/// Returns attention output [num_q_heads, head_dim].
#[allow(clippy::too_many_arguments)]
pub fn append_and_attend(
    cmd: &CommandBufferRef,
    cache: &mut LayerKVCache,
    append_pipeline: &ComputePipelineState,
    attend_pipeline: &ComputePipelineState,
    new_k: &Buffer,      // [num_kv_heads, head_dim]
    new_v: &Buffer,      // [num_kv_heads, head_dim]
    q: &Buffer,          // [num_q_heads, head_dim]
    out: &Buffer,        // [num_q_heads, head_dim]
    num_q_heads: usize,
    scale: f32,
) {
    let pos = cache.current_len as u32;
    let num_kv = cache.num_kv_heads as u32;
    let hd = cache.head_dim as u32;
    let total = cache.num_kv_heads * cache.head_dim;

    // Append new K/V to cache
    {
        let enc = cmd.new_compute_command_encoder();
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
        enc.end_encoding();
    }

    // Attend: Q against full cache [0..pos+1]
    let t_val = (cache.current_len + 1) as u32;
    let num_q_val = num_q_heads as u32;
    {
        let enc = cmd.new_compute_command_encoder();
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
        let window_size: u32 = 0; // 0 = full attention (no sliding window)
        enc.set_bytes(9, 4, &window_size as *const u32 as *const c_void);
        // One threadgroup per head
        enc.dispatch_thread_groups(
            MTLSize::new(num_q_heads as u64, 1, 1),
            MTLSize::new(256.min(cache.head_dim as u64), 1, 1),
        );
        enc.end_encoding();
    }

    cache.current_len += 1;
}
