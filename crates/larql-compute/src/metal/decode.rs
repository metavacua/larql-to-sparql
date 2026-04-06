use super::*;

impl MetalBackend {
    /// Create a KV cache for decode mode.
    pub fn create_kv_cache(&self, num_layers: usize, max_seq: usize, num_kv_heads: usize, head_dim: usize) -> ops::kv_cache::KVCache {
        ops::kv_cache::KVCache::new(&self.bufs, num_layers, max_seq, num_kv_heads, head_dim)
    }

    /// Decode one token through all layers with KV cache.
    /// Q8 attention + KV cache append/attend + Q4 FFN, one command buffer.
    /// Returns the updated hidden state after all layers.
    pub fn decode_token(
        &self,
        kv_cache: &mut ops::kv_cache::KVCache,
        layers: &[crate::FullPipelineLayer],
        x: &[f32],
        hidden: usize,
        inter: usize,
        q_dim: usize,
        kv_dim: usize,
        num_q_heads: usize,
        _num_kv_heads: usize,
        head_dim: usize,
        _rope_base: f32,
    ) -> Vec<f32> {
        let num_layers = layers.len();
        let hidden_val = hidden as u32;
        let inter_val = inter as u32;
        let eps = 1e-6f32;

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

        // Initial hidden state
        let mut h_buf = self.bufs.transient_from_f32(x);
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        let cmd = self.queue.new_command_buffer();

        for l in 0..num_layers {
            let norm_offset = layers[l].norm_offset;
            let _has_post_norms = layers[l].has_post_norms;

            // 1. RMS norm + Q8 quantize (fused)
            let q8_buf = self.bufs.output(hidden as u64);
            let q8s_buf = self.bufs.output((hidden / 32 * 4) as u64);
            {
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&self.rms_norm_q8_pipeline);
                enc.set_buffer(0, Some(&h_buf), 0);
                enc.set_buffer(1, Some(&input_norm_bufs[l]), 0);
                enc.set_buffer(2, Some(&q8_buf), 0);
                enc.set_buffer(3, Some(&q8s_buf), 0);
                enc.set_bytes(4, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(5, 4, &eps as *const f32 as *const std::ffi::c_void);
                enc.set_bytes(6, 4, &norm_offset as *const f32 as *const std::ffi::c_void);
                enc.dispatch_threads(MTLSize::new(hidden as u64, 1, 1), MTLSize::new(256.min(hidden as u64), 1, 1));
                enc.end_encoding();
            }

            // 2. Fused Q+K+V projection (one dispatch)
            let q_out = self.bufs.output((q_dim * 4) as u64);
            let k_out = self.bufs.output((kv_dim * 4) as u64);
            let v_out = self.bufs.output((kv_dim * 4) as u64);
            {
                let total_rows = (q_dim + kv_dim + kv_dim) as u32;
                let q_rows = q_dim as u32;
                let k_rows = kv_dim as u32;
                let v_rows = kv_dim as u32;
                let k_val = hidden as u32;
                let enc = cmd.new_compute_command_encoder();
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
                    MTLSize::new(((total_rows as u64) + 7) / 8, 1, 1),
                    MTLSize::new(256, 1, 1),
                );
                enc.end_encoding();
            }

            // 3. KV cache: append K/V, attend Q against cache
            let attn_out = self.bufs.output((q_dim * 4) as u64);
            ops::kv_cache::append_and_attend(
                cmd, &mut kv_cache.layers[l],
                &self.kv_append_pipeline, &self.kv_attend_pipeline,
                &k_out, &v_out, &q_out, &attn_out,
                num_q_heads, scale,
            );

            // 4. Q8 quantize attention output + O projection
            let o_q8 = self.bufs.output(q_dim as u64);
            let o_q8s = self.bufs.output((q_dim / 32 * 4) as u64);
            {
                let dim_val = q_dim as u32;
                let blocks = (q_dim / 32) as u32;
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&self.q8_quant_pipeline);
                enc.set_buffer(0, Some(&attn_out), 0);
                enc.set_buffer(1, Some(&o_q8), 0);
                enc.set_buffer(2, Some(&o_q8s), 0);
                enc.set_bytes(3, 4, &dim_val as *const u32 as *const std::ffi::c_void);
                enc.dispatch_threads(MTLSize::new(blocks as u64, 1, 1), MTLSize::new(256.min(blocks as u64), 1, 1));
                enc.end_encoding();
            }
            let o_out = self.bufs.output((hidden * 4) as u64);
            {
                let o_rows = hidden as u32;
                let o_k = q_dim as u32;
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&self.q8_matvec_pipeline);
                enc.set_buffer(0, Some(&wo_bufs[l]), 0);
                enc.set_buffer(1, Some(&o_q8), 0);
                enc.set_buffer(2, Some(&wo_scale_bufs[l]), 0);
                enc.set_buffer(3, Some(&o_q8s), 0);
                enc.set_buffer(4, Some(&o_out), 0);
                enc.set_bytes(5, 4, &o_rows as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(6, 4, &o_k as *const u32 as *const std::ffi::c_void);
                enc.dispatch_thread_groups(
                    MTLSize::new(((hidden as u64) + 7) / 8, 1, 1),
                    MTLSize::new(256, 1, 1),
                );
                enc.end_encoding();
            }

            // 5. Residual + pre-FFN norm + Q8 (fused)
            let h_post_attn = self.bufs.output((hidden * 4) as u64);
            let ffn_q8 = self.bufs.output(hidden as u64);
            let ffn_q8s = self.bufs.output((hidden / 32 * 4) as u64);
            {
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&self.residual_norm_q8_pipeline);
                enc.set_buffer(0, Some(&h_buf), 0);
                enc.set_buffer(1, Some(&o_out), 0);
                enc.set_buffer(2, Some(&post_attn_norm_bufs[l]), 0);
                enc.set_buffer(3, Some(&ffn_q8), 0);
                enc.set_buffer(4, Some(&ffn_q8s), 0);
                enc.set_buffer(5, Some(&h_post_attn), 0);
                enc.set_bytes(6, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(7, 4, &eps as *const f32 as *const std::ffi::c_void);
                enc.set_bytes(8, 4, &norm_offset as *const f32 as *const std::ffi::c_void);
                enc.dispatch_threads(MTLSize::new(hidden as u64, 1, 1), MTLSize::new(256.min(hidden as u64), 1, 1));
                enc.end_encoding();
            }

            // 6. Q4 FFN: gate+up (one encoder) → GEGLU → down
            let gate_out = self.bufs.output((inter * 4) as u64);
            let up_out = self.bufs.output((inter * 4) as u64);
            let act_buf = self.bufs.output((inter * 4) as u64);
            let down_out = self.bufs.output((hidden * 4) as u64);
            {
                let enc = cmd.new_compute_command_encoder();
                use crate::metal::shaders::q4_matvec as q4mv;
                let n_tgs_gate = ((inter as u64) + q4mv::ROWS_PER_TG - 1) / q4mv::ROWS_PER_TG;
                // Gate
                enc.set_compute_pipeline_state(&self.q4.matvec);
                enc.set_buffer(0, Some(&gate_bufs[l]), 0);
                enc.set_buffer(1, Some(&ffn_q8), 0);
                enc.set_buffer(2, Some(&ffn_q8s), 0);
                enc.set_buffer(3, Some(&gate_out), 0);
                enc.set_bytes(4, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(5, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                enc.dispatch_thread_groups(MTLSize::new(n_tgs_gate, 1, 1), MTLSize::new(q4mv::THREADS_PER_TG, 1, 1));
                // Up
                enc.set_buffer(0, Some(&up_bufs[l]), 0);
                enc.set_buffer(3, Some(&up_out), 0);
                enc.dispatch_thread_groups(MTLSize::new(n_tgs_gate, 1, 1), MTLSize::new(q4mv::THREADS_PER_TG, 1, 1));
                enc.end_encoding();
            }
            {
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&self.geglu_pipeline);
                enc.set_buffer(0, Some(&gate_out), 0);
                enc.set_buffer(1, Some(&up_out), 0);
                enc.set_buffer(2, Some(&act_buf), 0);
                enc.set_bytes(3, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                enc.dispatch_threads(MTLSize::new(inter as u64, 1, 1), MTLSize::new(256, 1, 1));
                enc.end_encoding();
            }
            {
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&self.q4.f32_matvec);
                enc.set_buffer(0, Some(&down_bufs[l]), 0);
                enc.set_buffer(1, Some(&act_buf), 0);
                enc.set_buffer(2, Some(&down_out), 0);
                enc.set_bytes(3, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(4, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                enc.dispatch_threads(MTLSize::new(hidden as u64, 1, 1), MTLSize::new(256, 1, 1));
                enc.end_encoding();
            }

            // 7. Post-FFN residual add → h for next layer
            let new_h = self.bufs.output((hidden * 4) as u64);
            {
                let len_val = hidden as u32;
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&self.residual_add_pipeline);
                enc.set_buffer(0, Some(&h_post_attn), 0);
                enc.set_buffer(1, Some(&down_out), 0);
                enc.set_buffer(2, Some(&new_h), 0);
                enc.set_bytes(3, 4, &len_val as *const u32 as *const std::ffi::c_void);
                enc.dispatch_threads(MTLSize::new(hidden as u64, 1, 1), MTLSize::new(256.min(hidden as u64), 1, 1));
                enc.end_encoding();
            }
            h_buf = new_h;
        }

        cmd.commit();
        cmd.wait_until_completed();

        let ptr = h_buf.contents() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, hidden).to_vec() }
    }
}
