use super::*;

impl MetalBackend {
    /// Create a KV cache for decode mode.
    pub fn create_kv_cache(&self, num_layers: usize, max_seq: usize, num_kv_heads: usize, head_dim: usize) -> ops::kv_cache::KVCache {
        ops::kv_cache::KVCache::new(&self.bufs, num_layers, max_seq, num_kv_heads, head_dim)
    }

    /// Decode one token through all layers with KV cache.
    ///
    /// **Single command buffer, single encoder per layer** with memory barriers:
    ///   1. Input norm → QKV projection → RoPE → V-norm
    ///   2. ── memory barrier (Q/K/V outputs) ──
    ///   3. KV cache append
    ///   4. ── memory barrier (K/V caches) ──
    ///   5. KV attend
    ///   6. ── memory barrier (attention output) ──
    ///   7. O projection
    ///   8. ── memory barrier (O output) ──
    ///   9. Residual+norm+Q8 → FFN → post-FFN residual → layer scalar
    ///
    /// One cmd buffer + one encoder per layer eliminates dispatch overhead
    /// from encoder creation/teardown (was: 4 encoders per layer).
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

        // Pre-cache weight buffers
        let wq_bufs: Vec<_> = layers.iter().map(|l| self.bufs.get_bytes(l.wq.data)).collect();
        let wk_bufs: Vec<_> = layers.iter().map(|l| self.bufs.get_bytes(l.wk.data)).collect();
        let wv_bufs: Vec<_> = layers.iter().map(|l| self.bufs.get_bytes(l.wv.data)).collect();
        let wo_bufs: Vec<_> = layers.iter().map(|l| self.bufs.get_bytes(l.wo.data)).collect();
        let wq_scale_bufs: Vec<_> = layers.iter().map(|l| self.bufs.transient_from_f32(l.wq.scales.unwrap_or(&[]))).collect();
        let wk_scale_bufs: Vec<_> = layers.iter().map(|l| self.bufs.transient_from_f32(l.wk.scales.unwrap_or(&[]))).collect();
        let wv_scale_bufs: Vec<_> = layers.iter().map(|l| self.bufs.transient_from_f32(l.wv.scales.unwrap_or(&[]))).collect();
        let wo_scale_bufs: Vec<_> = layers.iter().map(|l| self.bufs.transient_from_f32(l.wo.scales.unwrap_or(&[]))).collect();
        let gate_bufs: Vec<_> = layers.iter().map(|l| self.bufs.get_bytes(l.gate.data)).collect();
        let up_bufs: Vec<_> = layers.iter().map(|l| self.bufs.get_bytes(l.up.data)).collect();
        let down_bufs: Vec<_> = layers.iter().map(|l| self.bufs.get_bytes(l.down.data)).collect();
        let input_norm_bufs: Vec<_> = layers.iter().map(|l| self.bufs.transient_from_f32(l.input_norm)).collect();
        let post_attn_norm_bufs: Vec<_> = layers.iter().map(|l| self.bufs.transient_from_f32(l.post_attn_norm)).collect();

        let mut h_buf = self.bufs.transient_from_f32(x);

        // Single command buffer for all layers.
        let cmd = self.queue.new_command_buffer();

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

            let q_out = self.bufs.output((q_dim * 4) as u64);
            let k_out = self.bufs.output((kv_dim * 4) as u64);
            let v_out = self.bufs.output((kv_dim * 4) as u64);

            let enc = cmd.new_compute_command_encoder();

            // ── Step 1: Input norm + Q/K/V projection ──
            // Dispatches per-projection to handle mixed formats (Q4_K Q/K + Q6_K V).
            if uses_q4k {
                use crate::metal::ops::full_pipeline::encode_rms_norm;
                let norm_f32_buf = self.bufs.output((hidden * 4) as u64);

                // Dispatch 1: norm
                if layer.norm_type == crate::NormType::LayerNorm {
                    let len_val = hidden as u32;
                    if let Some(bias) = layer.input_norm_bias {
                        let bias_buf = self.bufs.transient_from_f32(bias);
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

                // Dispatch 2+: Per-projection matvec (handles mixed Q4_K/Q6_K formats)
                // Each projection dispatched with its format-specific shader.
                let all_same_format = layer.wq.format == layer.wk.format && layer.wk.format == layer.wv.format;
                if all_same_format && layer.wq.format != crate::QuantFormat::Q6_K {
                    // Fused QKV: all same Q4_K/Q4_KF format
                    let total_rows = (q_dim + kv_dim + kv_dim) as u32;
                    let q_rows_val = q_dim as u32;
                    let k_rows_val = kv_dim as u32;
                    let v_rows_val = kv_dim as u32;
                    let k_val = hidden as u32;
                    // Use correct ROWS_PER_TG for the selected pipeline
                    let (qkv_pipeline, rows_per_tg) = if layer.wq.format == crate::QuantFormat::Q4_KF {
                        (&self.q4kf_qkv_proj_pipeline, crate::metal::shaders::q4kf_qkv_proj::ROWS_PER_TG)
                    } else {
                        (&self.q4k_qkv_proj_pipeline, crate::metal::shaders::q4k_qkv_proj::ROWS_PER_TG)
                    };
                    let num_tgs = (total_rows as u64).div_ceil(rows_per_tg);
                    enc.set_compute_pipeline_state(qkv_pipeline);
                    enc.set_buffer(0, Some(&wq_bufs[l]), 0);
                    enc.set_buffer(1, Some(&wk_bufs[l]), 0);
                    enc.set_buffer(2, Some(&wv_bufs[l]), 0);
                    enc.set_buffer(3, Some(&norm_f32_buf), 0);
                    enc.set_buffer(4, Some(&q_out), 0);
                    enc.set_buffer(5, Some(&k_out), 0);
                    enc.set_buffer(6, Some(&v_out), 0);
                    enc.set_bytes(7, 4, &q_rows_val as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(8, 4, &k_rows_val as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(9, 4, &v_rows_val as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(10, 4, &k_val as *const u32 as *const std::ffi::c_void);
                    let threads_per_tg = if layer.wq.format == crate::QuantFormat::Q4_KF {
                        crate::metal::shaders::q4kf_qkv_proj::THREADS_PER_TG
                    } else {
                        crate::metal::shaders::q4k_qkv_proj::THREADS_PER_TG
                    };
                    enc.dispatch_thread_groups(
                        MTLSize::new(num_tgs, 1, 1),
                        MTLSize::new(threads_per_tg, 1, 1),
                    );
                } else {
                    // Mixed formats: dispatch each projection separately.
                    // This handles Q4_K Q/K + Q6_K V (Ollama strategy).
                    let k_val = hidden as u32;

                    // Helper: dispatch one projection with format-appropriate shader
                    fn encode_single_proj(
                        enc: &metal::ComputeCommandEncoderRef,
                        w_buf: &metal::Buffer, x_buf: &metal::Buffer, out_buf: &metal::Buffer,
                        rows: usize, k: u32, format: crate::QuantFormat,
                        q4k_pipeline: &metal::ComputePipelineState,
                        q4kf_pipeline: &metal::ComputePipelineState,
                        q6k_pipeline: &metal::ComputePipelineState,
                    ) {
                        match format {
                            crate::QuantFormat::Q6_K => {
                                use crate::metal::shaders::q6k_matvec as q6k;
                                let n = rows as u32;
                                let num_tgs = (rows as u64).div_ceil(q6k::ROWS_PER_TG);
                                enc.set_compute_pipeline_state(q6k_pipeline);
                                enc.set_buffer(0, Some(w_buf), 0);
                                enc.set_buffer(1, Some(x_buf), 0);
                                enc.set_buffer(2, Some(out_buf), 0);
                                enc.set_bytes(3, 4, &n as *const u32 as *const std::ffi::c_void);
                                enc.set_bytes(4, 4, &k as *const u32 as *const std::ffi::c_void);
                                enc.dispatch_thread_groups(
                                    MTLSize::new(num_tgs, 1, 1),
                                    MTLSize::new(q6k::THREADS_PER_TG, 1, 1),
                                );
                            }
                            crate::QuantFormat::Q4_KF => {
                                use crate::metal::shaders::q4kf_qkv_proj as proj_sh;
                                let n = rows as u32;
                                let num_tgs = (rows as u64).div_ceil(proj_sh::ROWS_PER_TG);
                                enc.set_compute_pipeline_state(q4kf_pipeline);
                                enc.set_buffer(0, Some(w_buf), 0);
                                enc.set_buffer(1, Some(x_buf), 0);
                                enc.set_buffer(2, Some(out_buf), 0);
                                enc.set_bytes(3, 4, &n as *const u32 as *const std::ffi::c_void);
                                enc.set_bytes(4, 4, &k as *const u32 as *const std::ffi::c_void);
                                enc.dispatch_thread_groups(
                                    MTLSize::new(num_tgs, 1, 1),
                                    MTLSize::new(proj_sh::THREADS_PER_TG, 1, 1),
                                );
                            }
                            _ => {
                                // Q4_K standard
                                use crate::metal::shaders::q4k_matvec as q4k;
                                let n = rows as u32;
                                let num_tgs = (rows as u64).div_ceil(q4k::ROWS_PER_TG);
                                enc.set_compute_pipeline_state(q4k_pipeline);
                                enc.set_buffer(0, Some(w_buf), 0);
                                enc.set_buffer(1, Some(x_buf), 0);
                                enc.set_buffer(2, Some(out_buf), 0);
                                enc.set_bytes(3, 4, &n as *const u32 as *const std::ffi::c_void);
                                enc.set_bytes(4, 4, &k as *const u32 as *const std::ffi::c_void);
                                enc.dispatch_thread_groups(
                                    MTLSize::new(num_tgs, 1, 1),
                                    MTLSize::new(q4k::THREADS_PER_TG, 1, 1),
                                );
                            }
                        }
                    }

                    encode_single_proj(enc, &wq_bufs[l], &norm_f32_buf, &q_out,
                        q_dim, k_val, layer.wq.format,
                        &self.q4k_matvec_pipeline, &self.q4kf_proj_pipeline, &self.q6k_matvec_pipeline);
                    encode_single_proj(enc, &wk_bufs[l], &norm_f32_buf, &k_out,
                        kv_dim, k_val, layer.wk.format,
                        &self.q4k_matvec_pipeline, &self.q4kf_proj_pipeline, &self.q6k_matvec_pipeline);
                    encode_single_proj(enc, &wv_bufs[l], &norm_f32_buf, &v_out,
                        kv_dim, k_val, layer.wv.format,
                        &self.q4k_matvec_pipeline, &self.q4kf_proj_pipeline, &self.q6k_matvec_pipeline);
                }
            } else {
                // Q8 path: norm+Q8 → Q8 QKV
                let q8_buf = self.bufs.output(hidden as u64);
                let q8s_buf = self.bufs.output((hidden / 32 * 4) as u64);

                enc.set_compute_pipeline_state(&self.rms_norm_q8_pipeline);
                enc.set_buffer(0, Some(&h_buf), 0);
                enc.set_buffer(1, Some(&input_norm_bufs[l]), 0);
                enc.set_buffer(2, Some(&q8_buf), 0);
                enc.set_buffer(3, Some(&q8s_buf), 0);
                enc.set_bytes(4, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(5, 4, &eps as *const f32 as *const std::ffi::c_void);
                enc.set_bytes(6, 4, &norm_offset as *const f32 as *const std::ffi::c_void);
                enc.dispatch_threads(MTLSize::new(hidden as u64, 1, 1), MTLSize::new(256.min(hidden as u64), 1, 1));

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

            let attn_out = self.bufs.output((layer_q_dim * 4) as u64);
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


            let h_post_attn = self.bufs.output((hidden * 4) as u64);
            let ffn_q8 = self.bufs.output(hidden as u64);
            let ffn_q8s = self.bufs.output((hidden / 32 * 4) as u64);
            let up_out = self.bufs.output((inter * 4) as u64);
            let act_buf = self.bufs.output((inter * 4) as u64);
            let down_out = self.bufs.output((hidden * 4) as u64);
            let new_h = self.bufs.output((hidden * 4) as u64);
            let o_out_buf = self.bufs.output((hidden * 4) as u64);
            {
                if uses_q4k {
                    use crate::metal::shaders::q4kf_qkv_proj as proj_sh;
                    let o_rows = hidden as u32;
                    let o_k = layer_q_dim as u32;
                    let num_tgs = (hidden as u64).div_ceil(proj_sh::ROWS_PER_TG);
                    let o_pipeline = if layer.wo.format == crate::QuantFormat::Q4_KF {
                        &self.q4kf_proj_pipeline
                    } else {
                        &self.q4k_proj_pipeline
                    };
                    enc.set_compute_pipeline_state(o_pipeline);
                    enc.set_buffer(0, Some(&wo_bufs[l]), 0);
                    enc.set_buffer(1, Some(&attn_out), 0);
                    enc.set_buffer(2, Some(&o_out_buf), 0);
                    enc.set_bytes(3, 4, &o_rows as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(4, 4, &o_k as *const u32 as *const std::ffi::c_void);
                    enc.dispatch_thread_groups(
                        MTLSize::new(num_tgs, 1, 1),
                        MTLSize::new(proj_sh::THREADS_PER_TG, 1, 1),
                    );
                } else {
                    let o_q8 = self.bufs.output(layer_q_dim as u64);
                    let o_q8s = self.bufs.output((layer_q_dim / 32 * 4) as u64);
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
            }

            // ── Step 5: Residual + norm (format-aware: Q4_K skips Q8 quantize) ──
            let ffn_uses_q4k = layer.gate.format == crate::QuantFormat::Q4_K
                || layer.gate.format == crate::QuantFormat::Q4_KF
                || layer.gate.format == crate::QuantFormat::Q6_K;
            let ffn_norm_out = self.bufs.output((hidden * 4) as u64);

            let has_post_norms = layer.has_post_norms;
            if has_post_norms {
                let normed_o = self.bufs.output((hidden * 4) as u64);
                {
                    use crate::metal::ops::full_pipeline::encode_rms_norm;
                    encode_rms_norm(enc, &self.rms_norm_pipeline,
                        &o_out_buf, &post_attn_norm_bufs[l], &normed_o, hidden, eps, norm_offset);
                }
                let pre_ffn_buf = if let Some(pfn) = layer.pre_ffn_norm {
                    self.bufs.transient_from_f32(pfn)
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
                    enc.dispatch_threads(MTLSize::new(hidden as u64, 1, 1), MTLSize::new(256.min(hidden as u64), 1, 1));
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
                    enc.dispatch_threads(MTLSize::new(hidden as u64, 1, 1), MTLSize::new(256.min(hidden as u64), 1, 1));
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
                enc.dispatch_threads(MTLSize::new(hidden as u64, 1, 1), MTLSize::new(256.min(hidden as u64), 1, 1));
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
                enc.dispatch_threads(MTLSize::new(hidden as u64, 1, 1), MTLSize::new(256.min(hidden as u64), 1, 1));
            }

            // ── Step 6: FFN (format-aware: Q4_KF uses llama.cpp kernel, Q4_K uses our kernel, Q4_0 uses Q8) ──
            {
                let ffn_is_q4kf = layer.gate.format == crate::QuantFormat::Q4_KF;

                if ffn_is_q4kf {
                    // Q4_KF (GGUF) FFN path: llama.cpp-exact kernel with register-cached input
                    use crate::metal::shaders::q4kf_qkv_proj as q4kf;
                    let n_tgs_gate = (inter as u64).div_ceil(q4kf::ROWS_PER_TG);
                    let n_tgs_down = (hidden as u64).div_ceil(q4kf::ROWS_PER_TG);

                    if layer.is_gated() {
                        let gate_out = self.bufs.output((inter * 4) as u64);
                        // Gate
                        enc.set_compute_pipeline_state(&self.q4kf_proj_pipeline);
                        enc.set_buffer(0, Some(&gate_bufs[l]), 0);
                        enc.set_buffer(1, Some(&ffn_norm_out), 0);
                        enc.set_buffer(2, Some(&gate_out), 0);
                        enc.set_bytes(3, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                        enc.set_bytes(4, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                        enc.dispatch_thread_groups(MTLSize::new(n_tgs_gate, 1, 1), MTLSize::new(q4kf::THREADS_PER_TG, 1, 1));
                        // Up
                        enc.set_buffer(0, Some(&up_bufs[l]), 0);
                        enc.set_buffer(2, Some(&up_out), 0);
                        enc.dispatch_thread_groups(MTLSize::new(n_tgs_gate, 1, 1), MTLSize::new(q4kf::THREADS_PER_TG, 1, 1));
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
                        // Down
                        enc.set_compute_pipeline_state(&self.q4kf_proj_pipeline);
                        enc.set_buffer(0, Some(&down_bufs[l]), 0);
                        enc.set_buffer(1, Some(&act_buf), 0);
                        enc.set_buffer(2, Some(&down_out), 0);
                        enc.set_bytes(3, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                        enc.set_bytes(4, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                        enc.dispatch_thread_groups(MTLSize::new(n_tgs_down, 1, 1), MTLSize::new(q4kf::THREADS_PER_TG, 1, 1));
                    } else {
                        enc.set_compute_pipeline_state(&self.q4kf_proj_pipeline);
                        enc.set_buffer(0, Some(&up_bufs[l]), 0);
                        enc.set_buffer(1, Some(&ffn_norm_out), 0);
                        enc.set_buffer(2, Some(&up_out), 0);
                        enc.set_bytes(3, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                        enc.set_bytes(4, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                        let n_tgs_up = (inter as u64).div_ceil(q4kf::ROWS_PER_TG);
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
                        let gate_out = self.bufs.output((inter * 4) as u64);
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
                        // Down projection (Q4_K, f32 input from GEGLU)
                        enc.set_compute_pipeline_state(&self.q4k_matvec_pipeline);
                        enc.set_buffer(0, Some(&down_bufs[l]), 0);
                        enc.set_buffer(1, Some(&act_buf), 0);
                        enc.set_buffer(2, Some(&down_out), 0);
                        enc.set_bytes(3, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                        enc.set_bytes(4, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                        enc.dispatch_thread_groups(MTLSize::new(n_tgs_down, 1, 1), MTLSize::new(q4k::THREADS_PER_TG, 1, 1));
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
                        let gate_out = self.bufs.output((inter * 4) as u64);
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
                    let post_ffn_buf = self.bufs.transient_from_f32(post_ffn);
                    let normed_ffn = self.bufs.output((hidden * 4) as u64);
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
                enc.dispatch_threads(MTLSize::new(hidden as u64, 1, 1), MTLSize::new(256.min(hidden as u64), 1, 1));
            }

            // ── Step 8: Optional layer scalar ──
            if layer.layer_scalar != 0.0 {
                let scaled = self.bufs.output((hidden * 4) as u64);
                let scalar_val = layer.layer_scalar;
                enc.set_compute_pipeline_state(&self.scale_vector_pipeline);
                enc.set_buffer(0, Some(&new_h), 0);
                enc.set_buffer(1, Some(&scaled), 0);
                enc.set_bytes(2, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(3, 4, &scalar_val as *const f32 as *const std::ffi::c_void);
                enc.dispatch_threads(MTLSize::new(hidden as u64, 1, 1), MTLSize::new(256.min(hidden as u64), 1, 1));
                enc.end_encoding();
                h_buf = scaled;
            } else {
                enc.end_encoding();
                h_buf = new_h;
            }

        }

        cmd.commit();
        cmd.wait_until_completed();

        super::buffers::read_buffer_f32(&h_buf, hidden)
    }
}
