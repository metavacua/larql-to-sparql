use super::*;

impl MetalBackend {
    /// Create a KV cache for decode mode with uniform per-layer dims.
    pub fn create_kv_cache(&self, num_layers: usize, max_seq: usize, num_kv_heads: usize, head_dim: usize) -> ops::kv_cache::KVCache {
        ops::kv_cache::KVCache::new(&self.bufs, num_layers, max_seq, num_kv_heads, head_dim)
    }

    /// Create a KV cache with per-layer shapes for models with asymmetric
    /// attention geometry (Gemma 4 31B sliding=16×256 / global=4×512).
    /// `shapes[i] = (num_kv_heads_i, head_dim_i)` for layer i.
    pub fn create_kv_cache_per_layer(&self, shapes: &[(usize, usize)], max_seq: usize) -> ops::kv_cache::KVCache {
        ops::kv_cache::KVCache::new_per_layer(&self.bufs, shapes, max_seq)
    }

    /// Decode one token through all layers with KV cache.
    ///
    /// **Single command buffer**, one encoder per layer, no explicit barriers
    /// (Apple Silicon serialises compute dispatches within an encoder).
    ///
    /// Per-layer pipeline (~10 dispatches):
    ///   1. Input norm
    ///   2. Fused QKV projection (Q4_K or Q4_KF)
    ///   3. Batched RoPE (all Q heads), batched RoPE (all K heads)
    ///   4. Batched V-norm (optional, Gemma 4)
    ///   5. KV cache append + KV attend
    ///   6. O projection
    ///   7. Residual + norm (f32 for Q4_K/Q4_KF, +Q8 for Q4_0)
    ///   8. FFN: fused gate+up (Q4_K) or separate gate/up (Q4_KF) + GEGLU + down
    ///   9. Post-FFN residual + optional layer scalar
    ///
    /// Format-aware FFN routing:
    ///   - Q4_KF: llama.cpp-exact kernel (q4kf_proj) with register-cached input
    ///   - Q4_K:  fused gate+up kernel + q4k_matvec (uint4, 8 rows/TG, nr0=2)
    ///   - Q4_0:  legacy Q8-input path
    #[allow(clippy::too_many_arguments)]
    pub fn decode_token(
        &self,
        kv_cache: &mut ops::kv_cache::KVCache,
        layers: &[crate::FullPipelineLayer],
        x: &[f32],
        hidden: usize,
        inter: usize,
        q_dim: usize,
        kv_dim: usize,
        _num_q_heads: usize,
        _num_kv_heads: usize,
        _head_dim: usize,
        _rope_base: f32,
    ) -> Vec<f32> {
        let num_layers = layers.len();
        let hidden_val = hidden as u32;
        let inter_val = inter as u32;

        // Scratch buffers are reused across all layers within the encoder.
        // When attention geometry varies layer to layer (Gemma 4 sliding=8192
        // vs global=16384 q_dim) we must size each scratch to the MAX across
        // layers; the outer scalar `q_dim` / `kv_dim` only reflect the first
        // layer's shape. Taking the per-layer max means a global layer's
        // 16384-wide Q output won't overflow a buffer sized for 8192.
        let max_q_dim = layers
            .iter()
            .map(|l| l.num_q_heads * l.head_dim)
            .max()
            .unwrap_or(q_dim);
        let max_kv_dim = layers
            .iter()
            .map(|l| l.num_kv_heads * l.head_dim)
            .max()
            .unwrap_or(kv_dim);

        // Pre-cache weight buffers
        let wq_bufs: Vec<_> = layers.iter().map(|l| self.bufs.get_bytes(l.wq.data)).collect();
        let wk_bufs: Vec<_> = layers.iter().map(|l| self.bufs.get_bytes(l.wk.data)).collect();
        let wv_bufs: Vec<_> = layers.iter().map(|l| self.bufs.get_bytes(l.wv.data)).collect();
        let wo_bufs: Vec<_> = layers.iter().map(|l| self.bufs.get_bytes(l.wo.data)).collect();
        // Stable across decode calls → cache by slice identity. Skips ~136
        // per-token Metal-buffer allocations for scales/norms on 34-layer
        // Gemma 3. `get_f32` hits the cache from the second decode onward.
        let wq_scale_bufs: Vec<_> = layers.iter().map(|l| self.bufs.get_f32(l.wq.scales.unwrap_or(&[]))).collect();
        let wk_scale_bufs: Vec<_> = layers.iter().map(|l| self.bufs.get_f32(l.wk.scales.unwrap_or(&[]))).collect();
        let wv_scale_bufs: Vec<_> = layers.iter().map(|l| self.bufs.get_f32(l.wv.scales.unwrap_or(&[]))).collect();
        let wo_scale_bufs: Vec<_> = layers.iter().map(|l| self.bufs.get_f32(l.wo.scales.unwrap_or(&[]))).collect();
        let gate_bufs: Vec<_> = layers.iter().map(|l| self.bufs.get_bytes(l.gate.data)).collect();
        let up_bufs: Vec<_> = layers.iter().map(|l| self.bufs.get_bytes(l.up.data)).collect();
        let down_bufs: Vec<_> = layers.iter().map(|l| self.bufs.get_bytes(l.down.data)).collect();
        let input_norm_bufs: Vec<_> = layers.iter().map(|l| self.bufs.get_f32(l.input_norm)).collect();
        let post_attn_norm_bufs: Vec<_> = layers.iter().map(|l| self.bufs.get_f32(l.post_attn_norm)).collect();

        // Two h buffers for ping-pong: even layers write to h_a, odd to h_b.
        let h_init = self.bufs.transient_from_f32(x);
        let h_a = self.bufs.output((hidden * 4) as u64);
        let h_b = self.bufs.output((hidden * 4) as u64);
        let mut h_buf = &h_init;

        // Pre-allocate scratch buffers reused across layers.
        // GPU processes layers sequentially within one cmd buffer, so
        // these buffers are never read and written simultaneously.
        let q_out = self.bufs.output((max_q_dim * 4) as u64);
        let k_out = self.bufs.output((max_kv_dim * 4) as u64);
        let v_out = self.bufs.output((max_kv_dim * 4) as u64);
        let norm_f32_buf = self.bufs.output((hidden * 4) as u64);
        let attn_out_buf = self.bufs.output((max_q_dim * 4) as u64);
        let o_out_buf = self.bufs.output((hidden * 4) as u64);
        let h_post_attn = self.bufs.output((hidden * 4) as u64);
        let ffn_norm_out = self.bufs.output((hidden * 4) as u64);
        let ffn_q8 = self.bufs.output(hidden as u64);
        let ffn_q8s = self.bufs.output((hidden / 32 * 4) as u64);
        let up_out = self.bufs.output((inter * 4) as u64);
        let act_buf = self.bufs.output((inter * 4) as u64);
        let down_out = self.bufs.output((hidden * 4) as u64);
        let gate_out_scratch = self.bufs.output((inter * 4) as u64);
        // new_h is ping-ponged via h_a/h_b above
        let normed_scratch = self.bufs.output((hidden * 4) as u64);
        let o_q8_scratch = self.bufs.output(max_q_dim as u64);
        let o_q8s_scratch = self.bufs.output((max_q_dim / 32 * 4) as u64);
        let scaled_scratch = self.bufs.output((hidden * 4) as u64);

        // Single command buffer + single encoder for ALL layers.
        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();

        // Diagnostic: run only up to (and including) the specified layer,
        // then dump intermediates and exit. Pinpoints which sub-stage in
        // which layer first produces NaN on real-vindex decode.
        let diag_stop_layer: Option<usize> = std::env::var("LARQL_DECODE_DIAG_LAYER")
            .ok().and_then(|v| v.parse::<usize>().ok());

        for l in 0..num_layers {
            let layer = &layers[l];
            let norm_offset = layer.norm_offset;
            let eps = layer.eps;
            let scale = layer.attn_scale;
            let layer_head_dim = layer.head_dim;
            let layer_num_q_heads = layer.num_q_heads;
            let layer_num_kv_heads = layer.num_kv_heads;
            let layer_rope_base = layer.rope_base;
            let layer_rotary_dim = if layer.rotary_dim > 0 { layer.rotary_dim } else { layer_head_dim };
            let uses_q4k = layer.wq.format == crate::QuantFormat::Q4_K
                || layer.wq.format == crate::QuantFormat::Q6_K
                || layer.wq.format == crate::QuantFormat::Q4_KF;
            let layer_q_dim = layer_num_q_heads * layer_head_dim;
            let _layer_kv_dim = layer_num_kv_heads * layer_head_dim;
            let window_size = layer.sliding_window as u32;

            // ── Step 1: Input norm + Q/K/V projection ──
            // Dispatches per-projection to handle mixed formats (Q4_K Q/K + Q6_K V).
            if uses_q4k {
                use crate::metal::ops::full_pipeline::encode_rms_norm;
                // Dispatch 1: norm
                if layer.norm_type == crate::NormType::LayerNorm {
                    let len_val = hidden as u32;
                    if let Some(bias) = layer.input_norm_bias {
                        let bias_buf = self.bufs.get_f32(bias);
                        enc.set_compute_pipeline_state(&self.layer_norm_pipeline);
                        enc.set_buffer(0, Some(&h_buf), 0);
                        enc.set_buffer(1, Some(&input_norm_bufs[l]), 0);
                        enc.set_buffer(2, Some(&bias_buf), 0);
                        enc.set_buffer(3, Some(&norm_f32_buf), 0);
                        enc.set_bytes(4, 4, &len_val as *const u32 as *const std::ffi::c_void);
                        enc.set_bytes(5, 4, &eps as *const f32 as *const std::ffi::c_void);
                        enc.set_bytes(6, 4, &norm_offset as *const f32 as *const std::ffi::c_void);
                    } else {
                        enc.set_compute_pipeline_state(&self.layer_norm_no_bias_pipeline);
                        enc.set_buffer(0, Some(&h_buf), 0);
                        enc.set_buffer(1, Some(&input_norm_bufs[l]), 0);
                        enc.set_buffer(2, Some(&norm_f32_buf), 0);
                        enc.set_bytes(3, 4, &len_val as *const u32 as *const std::ffi::c_void);
                        enc.set_bytes(4, 4, &eps as *const f32 as *const std::ffi::c_void);
                        enc.set_bytes(5, 4, &norm_offset as *const f32 as *const std::ffi::c_void);
                    }
                    enc.dispatch_threads(
                        MTLSize::new(hidden as u64, 1, 1),
                        MTLSize::new(256.min(hidden as u64), 1, 1),
                    );
                } else {
                    encode_rms_norm(enc, &self.rms_norm_pipeline,
                        &h_buf, &input_norm_bufs[l], &norm_f32_buf,
                        hidden, eps, norm_offset);
                }

                // Dispatch 2+: QKV projections. Three paths in priority order:
                //
                //  (i)  Uniform Q4_K / Q4_KF Q/K/V — single fused shader.
                //  (ii) Q4_K Q/K + Q6_K V (Gemma 3 / 4 Ollama convention) —
                //       dedicated mixed-quant fused shader. Replaces the
                //       per-projection fallback that costs 2 extra dispatches
                //       per layer × 34 layers ≈ 4 ms / token.
                //  (iii) Anything else — per-projection fallback.
                let uniform_q4k = layer.wq.format == layer.wk.format
                    && layer.wk.format == layer.wv.format
                    && layer.wq.format != crate::QuantFormat::Q6_K;
                let mixed_q4k_q6k_v = layer.wq.format == crate::QuantFormat::Q4_K
                    && layer.wk.format == crate::QuantFormat::Q4_K
                    && layer.wv.format == crate::QuantFormat::Q6_K;

                if uniform_q4k {
                    let fused_pipe = if layer.wq.format == crate::QuantFormat::Q4_KF {
                        &self.q4kf_qkv_proj_pipeline
                    } else {
                        &self.q4k_qkv_proj_pipeline
                    };
                    crate::metal::stages::qkv_proj::encode_fused_f32(
                        enc, fused_pipe,
                        &wq_bufs[l], &wk_bufs[l], &wv_bufs[l],
                        &norm_f32_buf, 0,
                        &q_out, 0, &k_out, 0, &v_out, 0,
                        q_dim, kv_dim, hidden,
                    );
                } else if mixed_q4k_q6k_v {
                    // Fused Q4K Q/K + Q6K V — one dispatch for all three.
                    use crate::metal::shaders::q4k_q6k_qkv_proj as sh;
                    let total_rows = (q_dim + kv_dim + kv_dim) as u64;
                    let num_tgs = total_rows.div_ceil(sh::ROWS_PER_TG);
                    let q_rows_u = q_dim as u32;
                    let k_rows_u = kv_dim as u32;
                    let v_rows_u = kv_dim as u32;
                    let k_u = hidden as u32;
                    enc.set_compute_pipeline_state(&self.q4k_q6k_qkv_proj_pipeline);
                    enc.set_buffer(0, Some(&wq_bufs[l]), 0);
                    enc.set_buffer(1, Some(&wk_bufs[l]), 0);
                    enc.set_buffer(2, Some(&wv_bufs[l]), 0);
                    enc.set_buffer(3, Some(&norm_f32_buf), 0);
                    enc.set_buffer(4, Some(&q_out), 0);
                    enc.set_buffer(5, Some(&k_out), 0);
                    enc.set_buffer(6, Some(&v_out), 0);
                    enc.set_bytes(7, 4, &q_rows_u as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(8, 4, &k_rows_u as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(9, 4, &v_rows_u as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(10, 4, &k_u as *const u32 as *const std::ffi::c_void);
                    enc.dispatch_thread_groups(
                        MTLSize::new(num_tgs, 1, 1),
                        MTLSize::new(sh::THREADS_PER_TG, 1, 1),
                    );
                } else {
                    // Mixed-but-unsupported (e.g. Q4_KF + Q6_K, or Q4_0 legacy):
                    // per-projection dispatch through the format-aware helper.
                    use crate::metal::stages::qkv_proj::{self, Proj};
                    use crate::metal::stages::quant_matvec::Pipelines;
                    let pipes = Pipelines {
                        q4kf_proj: Some(&self.q4kf_proj_pipeline),
                        q4k_matvec_fallback: &self.q4k_matvec_pipeline,
                        q6k_matvec: &self.q6k_matvec_pipeline,
                        q4_matvec: &self.q4.matvec,
                    };
                    qkv_proj::encode_per_proj(
                        enc, &pipes,
                        &norm_f32_buf, 0,
                        // Q8 bufs unused for f32-input formats — pass the
                        // norm buffer as a harmless placeholder.
                        &norm_f32_buf, 0, &norm_f32_buf, 0,
                        [
                            Proj { format: layer.wq.format, w_buf: &wq_bufs[l], out_buf: &q_out, out_off: 0, rows: q_dim },
                            Proj { format: layer.wk.format, w_buf: &wk_bufs[l], out_buf: &k_out, out_off: 0, rows: kv_dim },
                            Proj { format: layer.wv.format, w_buf: &wv_bufs[l], out_buf: &v_out, out_off: 0, rows: kv_dim },
                        ],
                        hidden,
                    );
                }
            } else {
                // Q8 path: norm+Q8 → Q8 QKV (reuse ffn_q8/q8s scratch)
                let q8_buf = &ffn_q8;
                let q8s_buf = &ffn_q8s;

                enc.set_compute_pipeline_state(&self.rms_norm_q8_pipeline);
                enc.set_buffer(0, Some(&h_buf), 0);
                enc.set_buffer(1, Some(&input_norm_bufs[l]), 0);
                enc.set_buffer(2, Some(&q8_buf), 0);
                enc.set_buffer(3, Some(&q8s_buf), 0);
                enc.set_bytes(4, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(5, 4, &eps as *const f32 as *const std::ffi::c_void);
                enc.set_bytes(6, 4, &norm_offset as *const f32 as *const std::ffi::c_void);
                enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256.min(hidden as u64), 1, 1));

                let total_rows = (q_dim + kv_dim + kv_dim) as u32;
                let q_rows = q_dim as u32;
                let k_rows = kv_dim as u32;
                let v_rows = kv_dim as u32;
                let k_val = hidden as u32;
                enc.set_compute_pipeline_state(&self.q8_qkv_proj_pipeline);
                enc.set_buffer(0, Some(&wq_bufs[l]), 0);
                enc.set_buffer(1, Some(&wk_bufs[l]), 0);
                enc.set_buffer(2, Some(&wv_bufs[l]), 0);
                enc.set_buffer(3, Some(&q8_buf), 0);
                enc.set_buffer(4, Some(&wq_scale_bufs[l]), 0);
                enc.set_buffer(5, Some(&wk_scale_bufs[l]), 0);
                enc.set_buffer(6, Some(&wv_scale_bufs[l]), 0);
                enc.set_buffer(7, Some(&q8s_buf), 0);
                enc.set_buffer(8, Some(&q_out), 0);
                enc.set_buffer(9, Some(&k_out), 0);
                enc.set_buffer(10, Some(&v_out), 0);
                enc.set_bytes(11, 4, &q_rows as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(12, 4, &k_rows as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(13, 4, &v_rows as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(14, 4, &k_val as *const u32 as *const std::ffi::c_void);
                enc.dispatch_thread_groups(
                    MTLSize::new((total_rows as u64).div_ceil(8), 1, 1),
                    MTLSize::new(256, 1, 1),
                );
            }

            // ── Step 1.5: QK-norm on Q and K (Gemma 3 / Gemma 4) ──
            //
            // Per-head RMS-norm with learned weight, applied to the raw
            // projection output before RoPE. Without this the Q/K vectors
            // on Gemma 3/4 are unscaled — attention dot products overflow
            // and softmax collapses to NaN by layer 0.
            //
            // Formula (matches CPU `rms_norm_heads_eps`):
            //   out[h, d] = (x[h, d] / sqrt(mean(x_head²) + eps))
            //             * (qk_norm_offset + weight[d])
            //
            // The qk_norm_offset is 0.0 on Gemma 4 and 1.0 on Gemma 2/3.
            // Passed as `offset` to the shader so `offset + weight[d]` does
            // the right thing for both families.
            if let (Some(q_w), Some(k_w)) = (layer.q_norm_weight, layer.k_norm_weight) {
                let hd_val = layer_head_dim as u32;
                let qk_off = layer.qk_norm_offset;
                let eps = layer.eps;
                // One threadgroup per head; threads per tg = min(head_dim, 512)
                // rounded up to a power of two for the tree reduction.
                let mut tg_w: usize = 1;
                while tg_w < layer_head_dim && tg_w < 512 { tg_w <<= 1; }

                // Q heads
                let q_w_buf = self.bufs.get_f32(q_w);
                let nq_val = layer_num_q_heads as u32;
                enc.set_compute_pipeline_state(&self.qk_norm_pipeline);
                enc.set_buffer(0, Some(&q_out), 0);
                enc.set_buffer(1, Some(&q_out), 0);
                enc.set_buffer(2, Some(&q_w_buf), 0);
                enc.set_bytes(3, 4, &hd_val as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(4, 4, &nq_val as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(5, 4, &eps as *const f32 as *const std::ffi::c_void);
                enc.set_bytes(6, 4, &qk_off as *const f32 as *const std::ffi::c_void);
                enc.dispatch_thread_groups(
                    MTLSize::new(layer_num_q_heads as u64, 1, 1),
                    MTLSize::new(tg_w as u64, 1, 1),
                );

                // K heads
                let k_w_buf = self.bufs.get_f32(k_w);
                let nkv_val = layer_num_kv_heads as u32;
                enc.set_buffer(0, Some(&k_out), 0);
                enc.set_buffer(1, Some(&k_out), 0);
                enc.set_buffer(2, Some(&k_w_buf), 0);
                enc.set_bytes(4, 4, &nkv_val as *const u32 as *const std::ffi::c_void);
                enc.dispatch_thread_groups(
                    MTLSize::new(layer_num_kv_heads as u64, 1, 1),
                    MTLSize::new(tg_w as u64, 1, 1),
                );
            }

            // ── Step 2: RoPE on Q and K heads (batched — one dispatch each) ──
            {
                let pos = kv_cache.layers[l].current_len as u32;
                let hd = layer_head_dim as u32;
                let rdim = layer_rotary_dim as u32;
                let rope_pairs = (layer_rotary_dim / 2) as u64;
                let num_q = layer_num_q_heads as u32;
                let num_kv = layer_num_kv_heads as u32;

                // Q heads — all in one dispatch
                enc.set_compute_pipeline_state(&self.rope_at_pos_batched_pipeline);
                enc.set_buffer(0, Some(&q_out), 0);
                enc.set_bytes(1, 4, &hd as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(2, 4, &layer_rope_base as *const f32 as *const std::ffi::c_void);
                enc.set_bytes(3, 4, &pos as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(4, 4, &rdim as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(5, 4, &num_q as *const u32 as *const std::ffi::c_void);
                enc.dispatch_threads(
                    MTLSize::new(rope_pairs, layer_num_q_heads as u64, 1),
                    MTLSize::new(rope_pairs.min(256), 1, 1),
                );

                // K heads — all in one dispatch
                enc.set_buffer(0, Some(&k_out), 0);
                enc.set_bytes(5, 4, &num_kv as *const u32 as *const std::ffi::c_void);
                enc.dispatch_threads(
                    MTLSize::new(rope_pairs, layer_num_kv_heads as u64, 1),
                    MTLSize::new(rope_pairs.min(256), 1, 1),
                );
            }

            // ── Step 3: V-norm batched (optional, Gemma 4) ──
            if layer.has_v_norm {
                let hd_val = layer_head_dim as u32;
                let num_kv = layer_num_kv_heads as u32;
                enc.set_compute_pipeline_state(&self.v_norm_batched_pipeline);
                enc.set_buffer(0, Some(&v_out), 0);
                enc.set_buffer(1, Some(&v_out), 0);
                enc.set_bytes(2, 4, &hd_val as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(3, 4, &eps as *const f32 as *const std::ffi::c_void);
                enc.set_bytes(4, 4, &num_kv as *const u32 as *const std::ffi::c_void);
                enc.dispatch_threads(
                    MTLSize::new(layer_head_dim as u64, layer_num_kv_heads as u64, 1),
                    MTLSize::new((layer_head_dim as u64).min(256), 1, 1),
                );
            }

            // No explicit barriers — Apple Silicon executes compute dispatches
            // within a single encoder in submission order. Verified by tests.

            let attn_out = &attn_out_buf;
            ops::kv_cache::encode_kv_append(
                enc, &kv_cache.layers[l],
                &self.kv_append_pipeline, &k_out, &v_out,
            );
            ops::kv_cache::encode_kv_attend(
                enc, &kv_cache.layers[l],
                &self.kv_attend_pipeline, &q_out, &attn_out,
                layer_num_q_heads, scale, window_size,
            );
            kv_cache.layers[l].current_len += 1;


            // Scratch buffers pre-allocated above — reused each layer.
            let new_h = if l % 2 == 0 { &h_a } else { &h_b };
            if uses_q4k {
                // Q4_K / Q4_KF / Q6_K O-projection via the stage helper.
                use crate::metal::stages::quant_matvec::Pipelines;
                let pipes = Pipelines {
                    q4kf_proj: Some(&self.q4kf_proj_pipeline),
                    q4k_matvec_fallback: &self.q4k_proj_pipeline,
                    q6k_matvec: &self.q6k_matvec_pipeline,
                    q4_matvec: &self.q4.matvec,
                };
                crate::metal::stages::o_proj::encode(
                    enc, &pipes, &self.q8_quant_pipeline,
                    layer.wo.format,
                    &wo_bufs[l],
                    &attn_out, 0,
                    &o_q8_scratch, 0, &o_q8s_scratch, 0,
                    &o_out_buf, 0,
                    layer_q_dim, hidden,
                );
            } else {
                // Q8 legacy path: decode-specific `q8_matvec` shader (not in
                // stages::quant_matvec which uses `q4_matvec` for Q4_0/Q8_0
                // with a different buffer layout). Inline.
                let o_q8 = &o_q8_scratch;
                let o_q8s = &o_q8s_scratch;
                let dim_val = layer_q_dim as u32;
                let blocks = (layer_q_dim / 32) as u32;
                enc.set_compute_pipeline_state(&self.q8_quant_pipeline);
                enc.set_buffer(0, Some(&attn_out), 0);
                enc.set_buffer(1, Some(&o_q8), 0);
                enc.set_buffer(2, Some(&o_q8s), 0);
                enc.set_bytes(3, 4, &dim_val as *const u32 as *const std::ffi::c_void);
                enc.dispatch_threads(MTLSize::new(blocks as u64, 1, 1), MTLSize::new(256.min(blocks as u64), 1, 1));

                let o_rows = hidden as u32;
                let o_k = layer_q_dim as u32;
                enc.set_compute_pipeline_state(&self.q8_matvec_pipeline);
                enc.set_buffer(0, Some(&wo_bufs[l]), 0);
                enc.set_buffer(1, Some(&o_q8), 0);
                enc.set_buffer(2, Some(&wo_scale_bufs[l]), 0);
                enc.set_buffer(3, Some(&o_q8s), 0);
                enc.set_buffer(4, Some(&o_out_buf), 0);
                enc.set_bytes(5, 4, &o_rows as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(6, 4, &o_k as *const u32 as *const std::ffi::c_void);
                enc.dispatch_thread_groups(
                    MTLSize::new((hidden as u64).div_ceil(8), 1, 1),
                    MTLSize::new(256, 1, 1),
                );
            }

            // ── Step 5: Residual + norm (format-aware: Q4_K skips Q8 quantize) ──
            let ffn_uses_q4k = layer.gate.format == crate::QuantFormat::Q4_K
                || layer.gate.format == crate::QuantFormat::Q4_KF
                || layer.gate.format == crate::QuantFormat::Q6_K;
            // ffn_norm_out pre-allocated above

            let has_post_norms = layer.has_post_norms;
            if has_post_norms {
                let normed_o = &normed_scratch;
                {
                    use crate::metal::ops::full_pipeline::encode_rms_norm;
                    encode_rms_norm(enc, &self.rms_norm_pipeline,
                        &o_out_buf, &post_attn_norm_bufs[l], &normed_o, hidden, eps, norm_offset);
                }
                let pre_ffn_buf = if let Some(pfn) = layer.pre_ffn_norm {
                    self.bufs.get_f32(pfn)
                } else {
                    post_attn_norm_bufs[l].clone()
                };
                if ffn_uses_q4k {
                    // Q4_K path: residual+norm → f32 output (no Q8)
                    enc.set_compute_pipeline_state(&self.residual_norm_pipeline);
                    enc.set_buffer(0, Some(&h_buf), 0);
                    enc.set_buffer(1, Some(&normed_o), 0);
                    enc.set_buffer(2, Some(&pre_ffn_buf), 0);
                    enc.set_buffer(3, Some(&ffn_norm_out), 0);
                    enc.set_bytes(4, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(5, 4, &eps as *const f32 as *const std::ffi::c_void);
                    enc.set_bytes(6, 4, &norm_offset as *const f32 as *const std::ffi::c_void);
                    enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256.min(hidden as u64), 1, 1));
                    // h_post_attn = h + normed_o (residual_norm also writes this to buffer 3? No — residual_norm only outputs normed.
                    // We need the pre-norm residual for the post-FFN add. Use residual_add separately.
                    use crate::metal::ops::full_pipeline::encode_residual_add;
                    encode_residual_add(enc, &self.residual_add_pipeline,
                        &h_buf, &normed_o, &h_post_attn, hidden);
                } else {
                    enc.set_compute_pipeline_state(&self.residual_norm_q8_pipeline);
                    enc.set_buffer(0, Some(&h_buf), 0);
                    enc.set_buffer(1, Some(&normed_o), 0);
                    enc.set_buffer(2, Some(&pre_ffn_buf), 0);
                    enc.set_buffer(3, Some(&ffn_q8), 0);
                    enc.set_buffer(4, Some(&ffn_q8s), 0);
                    enc.set_buffer(5, Some(&h_post_attn), 0);
                    enc.set_bytes(6, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(7, 4, &eps as *const f32 as *const std::ffi::c_void);
                    enc.set_bytes(8, 4, &norm_offset as *const f32 as *const std::ffi::c_void);
                    enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256.min(hidden as u64), 1, 1));
                }
            } else if ffn_uses_q4k {
                // Q4_K path: residual+norm → f32 output (no Q8)
                enc.set_compute_pipeline_state(&self.residual_norm_pipeline);
                enc.set_buffer(0, Some(&h_buf), 0);
                enc.set_buffer(1, Some(&o_out_buf), 0);
                enc.set_buffer(2, Some(&post_attn_norm_bufs[l]), 0);
                enc.set_buffer(3, Some(&ffn_norm_out), 0);
                enc.set_bytes(4, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(5, 4, &eps as *const f32 as *const std::ffi::c_void);
                enc.set_bytes(6, 4, &norm_offset as *const f32 as *const std::ffi::c_void);
                enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256.min(hidden as u64), 1, 1));
                // h_post_attn = h + o (pre-norm residual for post-FFN add)
                use crate::metal::ops::full_pipeline::encode_residual_add;
                encode_residual_add(enc, &self.residual_add_pipeline,
                    &h_buf, &o_out_buf, &h_post_attn, hidden);
            } else {
                enc.set_compute_pipeline_state(&self.residual_norm_q8_pipeline);
                enc.set_buffer(0, Some(&h_buf), 0);
                enc.set_buffer(1, Some(&o_out_buf), 0);
                enc.set_buffer(2, Some(&post_attn_norm_bufs[l]), 0);
                enc.set_buffer(3, Some(&ffn_q8), 0);
                enc.set_buffer(4, Some(&ffn_q8s), 0);
                enc.set_buffer(5, Some(&h_post_attn), 0);
                enc.set_bytes(6, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(7, 4, &eps as *const f32 as *const std::ffi::c_void);
                enc.set_bytes(8, 4, &norm_offset as *const f32 as *const std::ffi::c_void);
                enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256.min(hidden as u64), 1, 1));
            }

            // ── Step 6: FFN (format-aware: Q4_KF uses llama.cpp kernel, Q4_K uses our kernel, Q4_0 uses Q8) ──
            {
                let ffn_is_q4kf = layer.gate.format == crate::QuantFormat::Q4_KF;

                if ffn_is_q4kf {
                    // Q4_KF (GGUF) FFN path: llama.cpp-exact kernel
                    use crate::metal::shaders::q4kf_qkv_proj as q4kf;
                    use crate::metal::shaders::q4kf_ffn_gate_up as q4kf_gu;
                    let n_tgs_down = (hidden as u64).div_ceil(q4kf::ROWS_PER_TG);

                    if layer.is_gated() {
                        let gate_out = &gate_out_scratch;
                        // Fused gate+up: one dispatch, shared input (llama.cpp inner loop)
                        let n_tgs_per_mat = (inter as u64).div_ceil(q4kf_gu::ROWS_PER_TG);
                        enc.set_compute_pipeline_state(&self.q4kf_ffn_gate_up_pipeline);
                        enc.set_buffer(0, Some(&gate_bufs[l]), 0);
                        enc.set_buffer(1, Some(&up_bufs[l]), 0);
                        enc.set_buffer(2, Some(&ffn_norm_out), 0);
                        enc.set_buffer(3, Some(&gate_out), 0);
                        enc.set_buffer(4, Some(&up_out), 0);
                        enc.set_bytes(5, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                        enc.set_bytes(6, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                        enc.dispatch_thread_groups(
                            MTLSize::new(n_tgs_per_mat * 2, 1, 1),
                            MTLSize::new(q4kf_gu::THREADS_PER_TG, 1, 1),
                        );
                        // GEGLU
                        let geglu = match layer.activation {
                            crate::Activation::GeluTanh => &self.geglu_gelu_tanh_pipeline,
                            _ => &self.geglu_pipeline,
                        };
                        enc.set_compute_pipeline_state(geglu);
                        enc.set_buffer(0, Some(&gate_out), 0);
                        enc.set_buffer(1, Some(&up_out), 0);
                        enc.set_buffer(2, Some(&act_buf), 0);
                        enc.set_bytes(3, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                        enc.dispatch_threads(MTLSize::new(inter as u64, 1, 1), MTLSize::new(256, 1, 1));
                        // Down — format-aware. Mixed Q4_KF gate/up + Q6_K
                        // down ships on some vindexes; route through the
                        // format-matching shader.
                        use crate::metal::stages::quant_matvec::{self as qmv, Pipelines};
                        let pipes = Pipelines {
                            q4kf_proj: Some(&self.q4kf_proj_pipeline),
                            q4k_matvec_fallback: &self.q4k_matvec_pipeline,
                            q6k_matvec: &self.q6k_matvec_pipeline,
                            q4_matvec: &self.q4.matvec,
                        };
                        qmv::encode(
                            enc, layer.down.format, &down_bufs[l],
                            &act_buf, 0,
                            &act_buf, 0, &act_buf, 0,
                            &down_out, 0,
                            &pipes,
                            hidden, inter,
                        );
                        let _ = n_tgs_down;
                    } else {
                        let n_tgs_up = (inter as u64).div_ceil(q4kf::ROWS_PER_TG);
                        enc.set_compute_pipeline_state(&self.q4kf_proj_pipeline);
                        enc.set_buffer(0, Some(&up_bufs[l]), 0);
                        enc.set_buffer(1, Some(&ffn_norm_out), 0);
                        enc.set_buffer(2, Some(&up_out), 0);
                        enc.set_bytes(3, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                        enc.set_bytes(4, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                        enc.dispatch_thread_groups(MTLSize::new(n_tgs_up, 1, 1), MTLSize::new(q4kf::THREADS_PER_TG, 1, 1));
                        let activation_pipeline = match layer.activation {
                            crate::Activation::GeluTanh => &self.gelu_tanh_pipeline,
                            _ => &self.silu_pipeline,
                        };
                        enc.set_compute_pipeline_state(activation_pipeline);
                        enc.set_buffer(0, Some(&up_out), 0);
                        enc.set_buffer(1, Some(&act_buf), 0);
                        enc.set_bytes(2, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                        enc.dispatch_threads(MTLSize::new(inter as u64, 1, 1), MTLSize::new(256, 1, 1));
                        enc.set_compute_pipeline_state(&self.q4kf_proj_pipeline);
                        enc.set_buffer(0, Some(&down_bufs[l]), 0);
                        enc.set_buffer(1, Some(&act_buf), 0);
                        enc.set_buffer(2, Some(&down_out), 0);
                        enc.set_bytes(3, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                        enc.set_bytes(4, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                        enc.dispatch_thread_groups(MTLSize::new(n_tgs_down, 1, 1), MTLSize::new(q4kf::THREADS_PER_TG, 1, 1));
                    }
                } else if ffn_uses_q4k {
                    // Q4_K FFN path: f32 input → Q4_K matvec
                    use crate::metal::shaders::q4k_matvec as q4k;
                    use crate::metal::shaders::q4k_ffn_gate_up as q4k_gu;
                    let n_tgs_down = (hidden as u64).div_ceil(q4k::ROWS_PER_TG);

                    if layer.is_gated() {
                        let gate_out = &gate_out_scratch;
                        // Fused gate+up: one dispatch, reads input once
                        let n_tgs_per_mat = (inter as u64).div_ceil(q4k_gu::ROWS_PER_TG);
                        enc.set_compute_pipeline_state(&self.q4k_ffn_gate_up_pipeline);
                        enc.set_buffer(0, Some(&gate_bufs[l]), 0);
                        enc.set_buffer(1, Some(&up_bufs[l]), 0);
                        enc.set_buffer(2, Some(&ffn_norm_out), 0);
                        enc.set_buffer(3, Some(&gate_out), 0);
                        enc.set_buffer(4, Some(&up_out), 0);
                        enc.set_bytes(5, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                        enc.set_bytes(6, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                        enc.dispatch_thread_groups(
                            MTLSize::new(n_tgs_per_mat * 2, 1, 1),
                            MTLSize::new(q4k_gu::THREADS_PER_TG, 1, 1),
                        );
                        // GEGLU activation
                        let geglu = match layer.activation {
                            crate::Activation::GeluTanh => &self.geglu_gelu_tanh_pipeline,
                            _ => &self.geglu_pipeline,
                        };
                        enc.set_compute_pipeline_state(geglu);
                        enc.set_buffer(0, Some(&gate_out), 0);
                        enc.set_buffer(1, Some(&up_out), 0);
                        enc.set_buffer(2, Some(&act_buf), 0);
                        enc.set_bytes(3, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                        enc.dispatch_threads(MTLSize::new(inter as u64, 1, 1), MTLSize::new(256, 1, 1));
                        // Down projection — format-aware. Gemma 3 4B ships
                        // Q6_K down even when gate/up are Q4_K. Route through
                        // the format-matching shader so we don't decode Q6_K
                        // bytes as if they were Q4_K (→ NaN).
                        use crate::metal::stages::quant_matvec::{self as qmv, Pipelines};
                        let pipes = Pipelines {
                            q4kf_proj: Some(&self.q4kf_proj_pipeline),
                            q4k_matvec_fallback: &self.q4k_matvec_pipeline,
                            q6k_matvec: &self.q6k_matvec_pipeline,
                            q4_matvec: &self.q4.matvec,
                        };
                        qmv::encode(
                            enc, layer.down.format, &down_bufs[l],
                            &act_buf, 0,
                            &act_buf, 0, &act_buf, 0, // Q8 unused for f32 input
                            &down_out, 0,
                            &pipes,
                            hidden, inter,
                        );
                        let _ = n_tgs_down;
                    } else {
                        let n_tgs_up = (inter as u64).div_ceil(q4k::ROWS_PER_TG);
                        enc.set_compute_pipeline_state(&self.q4k_matvec_pipeline);
                        enc.set_buffer(0, Some(&up_bufs[l]), 0);
                        enc.set_buffer(1, Some(&ffn_norm_out), 0);
                        enc.set_buffer(2, Some(&up_out), 0);
                        enc.set_bytes(3, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                        enc.set_bytes(4, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                        enc.dispatch_thread_groups(MTLSize::new(n_tgs_up, 1, 1), MTLSize::new(q4k::THREADS_PER_TG, 1, 1));
                        let activation_pipeline = match layer.activation {
                            crate::Activation::GeluTanh => &self.gelu_tanh_pipeline,
                            _ => &self.silu_pipeline,
                        };
                        enc.set_compute_pipeline_state(activation_pipeline);
                        enc.set_buffer(0, Some(&up_out), 0);
                        enc.set_buffer(1, Some(&act_buf), 0);
                        enc.set_bytes(2, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                        enc.dispatch_threads(MTLSize::new(inter as u64, 1, 1), MTLSize::new(256, 1, 1));
                        enc.set_compute_pipeline_state(&self.q4k_matvec_pipeline);
                        enc.set_buffer(0, Some(&down_bufs[l]), 0);
                        enc.set_buffer(1, Some(&act_buf), 0);
                        enc.set_buffer(2, Some(&down_out), 0);
                        enc.set_bytes(3, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                        enc.set_bytes(4, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                        enc.dispatch_thread_groups(MTLSize::new(n_tgs_down, 1, 1), MTLSize::new(q4k::THREADS_PER_TG, 1, 1));
                    }
                } else {
                    // Q4_0 FFN path: Q8 input → Q4_0 matvec (legacy)
                    use crate::metal::shaders::q4_matvec as q4mv;
                    let n_tgs_ffn = (inter as u64).div_ceil(q4mv::ROWS_PER_TG);

                    if layer.is_gated() {
                        let gate_out = &gate_out_scratch;
                        enc.set_compute_pipeline_state(&self.q4.matvec);
                        enc.set_buffer(0, Some(&gate_bufs[l]), 0);
                        enc.set_buffer(1, Some(&ffn_q8), 0);
                        enc.set_buffer(2, Some(&ffn_q8s), 0);
                        enc.set_buffer(3, Some(&gate_out), 0);
                        enc.set_bytes(4, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                        enc.set_bytes(5, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                        enc.dispatch_thread_groups(MTLSize::new(n_tgs_ffn, 1, 1), MTLSize::new(q4mv::THREADS_PER_TG, 1, 1));
                        enc.set_buffer(0, Some(&up_bufs[l]), 0);
                        enc.set_buffer(3, Some(&up_out), 0);
                        enc.dispatch_thread_groups(MTLSize::new(n_tgs_ffn, 1, 1), MTLSize::new(q4mv::THREADS_PER_TG, 1, 1));
                        let geglu = match layer.activation {
                            crate::Activation::GeluTanh => &self.geglu_gelu_tanh_pipeline,
                            _ => &self.geglu_pipeline,
                        };
                        enc.set_compute_pipeline_state(geglu);
                        enc.set_buffer(0, Some(&gate_out), 0);
                        enc.set_buffer(1, Some(&up_out), 0);
                        enc.set_buffer(2, Some(&act_buf), 0);
                        enc.set_bytes(3, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                        enc.dispatch_threads(MTLSize::new(inter as u64, 1, 1), MTLSize::new(256, 1, 1));
                    } else {
                        enc.set_compute_pipeline_state(&self.q4.matvec);
                        enc.set_buffer(0, Some(&up_bufs[l]), 0);
                        enc.set_buffer(1, Some(&ffn_q8), 0);
                        enc.set_buffer(2, Some(&ffn_q8s), 0);
                        enc.set_buffer(3, Some(&up_out), 0);
                        enc.set_bytes(4, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                        enc.set_bytes(5, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                        enc.dispatch_thread_groups(MTLSize::new(n_tgs_ffn, 1, 1), MTLSize::new(q4mv::THREADS_PER_TG, 1, 1));
                        let activation_pipeline = match layer.activation {
                            crate::Activation::GeluTanh => &self.gelu_tanh_pipeline,
                            _ => &self.silu_pipeline,
                        };
                        enc.set_compute_pipeline_state(activation_pipeline);
                        enc.set_buffer(0, Some(&up_out), 0);
                        enc.set_buffer(1, Some(&act_buf), 0);
                        enc.set_bytes(2, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                        enc.dispatch_threads(MTLSize::new(inter as u64, 1, 1), MTLSize::new(256, 1, 1));
                    }

                    enc.set_compute_pipeline_state(&self.q4.f32_matvec);
                    enc.set_buffer(0, Some(&down_bufs[l]), 0);
                    enc.set_buffer(1, Some(&act_buf), 0);
                    enc.set_buffer(2, Some(&down_out), 0);
                    enc.set_bytes(3, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(4, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                    enc.dispatch_threads(MTLSize::new(hidden as u64, 1, 1), MTLSize::new(256, 1, 1));
                }
            }

            // ── Step 7: Post-FFN residual ──
            if has_post_norms {
                if let Some(post_ffn) = layer.post_ffn_norm {
                    let post_ffn_buf = self.bufs.get_f32(post_ffn);
                    let normed_ffn = &normed_scratch;
                    use crate::metal::ops::full_pipeline::encode_rms_norm;
                    encode_rms_norm(enc, &self.rms_norm_pipeline,
                        &down_out, &post_ffn_buf, &normed_ffn, hidden, eps, norm_offset);
                    use crate::metal::ops::full_pipeline::encode_residual_add;
                    encode_residual_add(enc, &self.residual_add_pipeline,
                        &h_post_attn, &normed_ffn, &new_h, hidden);
                } else {
                    use crate::metal::ops::full_pipeline::encode_residual_add;
                    encode_residual_add(enc, &self.residual_add_pipeline,
                        &h_post_attn, &down_out, &new_h, hidden);
                }
            } else {
                let len_val = hidden as u32;
                enc.set_compute_pipeline_state(&self.residual_add_pipeline);
                enc.set_buffer(0, Some(&h_post_attn), 0);
                enc.set_buffer(1, Some(&down_out), 0);
                enc.set_buffer(2, Some(&new_h), 0);
                enc.set_bytes(3, 4, &len_val as *const u32 as *const std::ffi::c_void);
                enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256.min(hidden as u64), 1, 1));
            }

            // ── Step 8: Optional layer scalar ──
            //
            // Note: stages::layer_scalar scales in-place (src == dst). Decode's
            // original code wrote to a separate `scaled_scratch` buffer then
            // pointed `h_buf` at it. In-place on `new_h` produces the same
            // downstream residual read since nothing else aliases `new_h`
            // between this write and the next layer's input norm read.
            if layer.layer_scalar != 0.0 {
                crate::metal::stages::layer_scalar::encode(
                    enc, &self.scale_vector_pipeline,
                    new_h, 1, hidden, layer.layer_scalar,
                );
            }
            h_buf = new_h;
            let _ = &scaled_scratch; // keep binding alive; no longer needed

            // Diagnostic early-exit after layer `l`. Commits what we have,
            // reads the per-sub-stage buffers, and reports NaN counts.
            if diag_stop_layer == Some(l) {
                enc.end_encoding();
                cmd.commit();
                cmd.wait_until_completed();
                let stat = |name: &str, buf: &metal::Buffer, n: usize| {
                    let ptr = buf.contents() as *const f32;
                    if ptr.is_null() { eprintln!("[diag L{l}] {name}: null contents"); return; }
                    let s = unsafe { std::slice::from_raw_parts(ptr, n) };
                    let nan = s.iter().filter(|v| v.is_nan()).count();
                    let inf = s.iter().filter(|v| v.is_infinite()).count();
                    let maxabs = s.iter().map(|v| v.abs()).filter(|v| v.is_finite()).fold(0.0f32, f32::max);
                    eprintln!("[diag L{l}] {name}: len={n} nan={nan} inf={inf} max_abs={maxabs:.3e}");
                };
                stat("norm_f32_buf", &norm_f32_buf, hidden);
                stat("q_out",        &q_out,        layer_q_dim);
                stat("k_out",        &k_out,        layer_num_kv_heads * layer_head_dim);
                stat("v_out",        &v_out,        layer_num_kv_heads * layer_head_dim);
                stat("attn_out_buf", &attn_out_buf, layer_q_dim);
                stat("o_out_buf",    &o_out_buf,    hidden);
                stat("h_post_attn",  &h_post_attn,  hidden);
                stat("ffn_norm_out", &ffn_norm_out, hidden);
                stat("gate_out_scratch", &gate_out_scratch, inter);
                stat("up_out",       &up_out,       inter);
                stat("act_buf",      &act_buf,      inter);
                stat("down_out",     &down_out,     hidden);
                stat("new_h (h_out)", new_h,        hidden);
                return super::buffers::read_buffer_f32(new_h, hidden);
            }
        }

        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        super::buffers::read_buffer_f32(&h_buf, hidden)
    }
}
