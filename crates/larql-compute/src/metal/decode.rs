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
        num_kv_heads: usize,
        head_dim: usize,
        rope_base: f32,
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
            let uses_q4k = layers[l].wq.format == crate::QuantFormat::Q4_K
                || layers[l].wq.format == crate::QuantFormat::Q6_K
                || layers[l].wq.format == crate::QuantFormat::Q4_KF;

            // 1+2. Input norm + QKV projection (format-dependent)
            let q_out = self.bufs.output((q_dim * 4) as u64);
            let k_out = self.bufs.output((kv_dim * 4) as u64);
            let v_out = self.bufs.output((kv_dim * 4) as u64);

            if uses_q4k {
                // ── Q4_K path: norm + QKV in ONE encoder (2 dispatches, no barrier needed) ──
                let norm_f32_buf = self.bufs.output((hidden * 4) as u64);
                {
                    use crate::metal::ops::full_pipeline::encode_rms_norm;
                    use crate::metal::shaders::q4kf_qkv_proj as qkv_sh;
                    let total_rows = (q_dim + kv_dim + kv_dim) as u32;
                    let q_rows_val = q_dim as u32;
                    let k_rows_val = kv_dim as u32;
                    let v_rows_val = kv_dim as u32;
                    let k_val = hidden as u32;
                    let num_tgs = ((total_rows as u64) + qkv_sh::ROWS_PER_TG - 1) / qkv_sh::ROWS_PER_TG;

                    let enc = cmd.new_compute_command_encoder();
                    // Dispatch 1: rms_norm
                    encode_rms_norm(enc, &self.rms_norm_pipeline,
                        &h_buf, &input_norm_bufs[l], &norm_f32_buf,
                        hidden, eps, norm_offset);
                    // Dispatch 2: Q4_K QKV (depends on norm output)
                    enc.set_compute_pipeline_state(&self.q4kf_qkv_proj_pipeline);
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
                    enc.dispatch_thread_groups(
                        MTLSize::new(num_tgs, 1, 1),
                        MTLSize::new(qkv_sh::THREADS_PER_TG, 1, 1),
                    );
                    enc.end_encoding();
                }
            } else {
                // ── Q8 path: fused rms_norm+Q8 → fused Q8 QKV ──
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
            }

            // 2b. Apply RoPE to Q and K at the correct absolute position
            // Uses rope_at_pos shader: one dispatch per head, explicit position parameter
            {
                let pos = kv_cache.layers[l].current_len as u32;
                let hd = head_dim as u32;
                let hdim = (head_dim / 2) as u64;

                let enc = cmd.new_compute_command_encoder();
                // RoPE on each Q head
                for qh in 0..num_q_heads {
                    let offset = (qh * head_dim * 4) as u64;
                    enc.set_compute_pipeline_state(&self.rope_at_pos_pipeline);
                    enc.set_buffer(0, Some(&q_out), offset);
                    enc.set_bytes(1, 4, &hd as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(2, 4, &rope_base as *const f32 as *const std::ffi::c_void);
                    enc.set_bytes(3, 4, &pos as *const u32 as *const std::ffi::c_void);
                    enc.dispatch_threads(MTLSize::new(hdim, 1, 1), MTLSize::new(hdim.min(256), 1, 1));
                }
                // RoPE on each KV head
                for kvh in 0..num_kv_heads {
                    let offset = (kvh * head_dim * 4) as u64;
                    enc.set_compute_pipeline_state(&self.rope_at_pos_pipeline);
                    enc.set_buffer(0, Some(&k_out), offset);
                    enc.set_bytes(1, 4, &hd as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(2, 4, &rope_base as *const f32 as *const std::ffi::c_void);
                    enc.set_bytes(3, 4, &pos as *const u32 as *const std::ffi::c_void);
                    enc.dispatch_threads(MTLSize::new(hdim, 1, 1), MTLSize::new(hdim.min(256), 1, 1));
                }
                enc.end_encoding();
            }

            // 3. KV cache: append RoPE'd K/V, attend RoPE'd Q against cache
            let attn_out = self.bufs.output((q_dim * 4) as u64);
            ops::kv_cache::append_and_attend(
                cmd, &mut kv_cache.layers[l],
                &self.kv_append_pipeline, &self.kv_attend_pipeline,
                &k_out, &v_out, &q_out, &attn_out,
                num_q_heads, scale,
            );

            // 4. O projection (format-dependent)
            let o_out = self.bufs.output((hidden * 4) as u64);
            if uses_q4k {
                use crate::metal::shaders::q4kf_qkv_proj as proj_sh;
                let o_rows = hidden as u32;
                let o_k = q_dim as u32;
                let num_tgs = ((hidden as u64) + proj_sh::ROWS_PER_TG - 1) / proj_sh::ROWS_PER_TG;
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&self.q4kf_proj_pipeline);
                enc.set_buffer(0, Some(&wo_bufs[l]), 0);
                enc.set_buffer(1, Some(&attn_out), 0);
                enc.set_buffer(2, Some(&o_out), 0);
                enc.set_bytes(3, 4, &o_rows as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(4, 4, &o_k as *const u32 as *const std::ffi::c_void);
                enc.dispatch_thread_groups(
                    MTLSize::new(num_tgs, 1, 1),
                    MTLSize::new(proj_sh::THREADS_PER_TG, 1, 1),
                );
                enc.end_encoding();
            } else {
                // Q8 O projection: Q8 quantize attention → Q8 matvec
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
            }

            // 5. Post-attention residual + pre-FFN norm + Q8
            let h_post_attn = self.bufs.output((hidden * 4) as u64);
            let ffn_q8 = self.bufs.output(hidden as u64);
            let ffn_q8s = self.bufs.output((hidden / 32 * 4) as u64);
            let has_post_norms = layers[l].has_post_norms;
            if has_post_norms {
                // Post-norm: norm(O) → residual_add(h, normed) → pre_ffn_norm → Q8
                let normed_o = self.bufs.output((hidden * 4) as u64);
                {
                    use crate::metal::ops::full_pipeline::encode_rms_norm;
                    let enc = cmd.new_compute_command_encoder();
                    encode_rms_norm(enc, &self.rms_norm_pipeline,
                        &o_out, &post_attn_norm_bufs[l], &normed_o, hidden, eps, norm_offset);
                    enc.end_encoding();
                }
                // residual_add(h, normed_o) + pre_ffn_norm + Q8
                let pre_ffn_buf = if let Some(pfn) = layers[l].pre_ffn_norm {
                    self.bufs.transient_from_f32(pfn)
                } else {
                    post_attn_norm_bufs[l].clone()
                };
                {
                    let enc = cmd.new_compute_command_encoder();
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
                    enc.end_encoding();
                }
            } else {
                // Standard: FUSED residual_add(h, O) + post_attn_norm + Q8
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

            // 6. Q4 FFN + residual: gate+up → GEGLU → down → residual (one encoder)
            let gate_out = self.bufs.output((inter * 4) as u64);
            let up_out = self.bufs.output((inter * 4) as u64);
            let act_buf = self.bufs.output((inter * 4) as u64);
            let down_out = self.bufs.output((hidden * 4) as u64);
            let new_h = self.bufs.output((hidden * 4) as u64);
            {
                // Gate + Up in one encoder (independent dispatches)
                let enc = cmd.new_compute_command_encoder();
                use crate::metal::shaders::q4_matvec as q4mv;
                let n_tgs_gate = ((inter as u64) + q4mv::ROWS_PER_TG - 1) / q4mv::ROWS_PER_TG;
                enc.set_compute_pipeline_state(&self.q4.matvec);
                enc.set_buffer(0, Some(&gate_bufs[l]), 0);
                enc.set_buffer(1, Some(&ffn_q8), 0);
                enc.set_buffer(2, Some(&ffn_q8s), 0);
                enc.set_buffer(3, Some(&gate_out), 0);
                enc.set_bytes(4, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(5, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                enc.dispatch_thread_groups(MTLSize::new(n_tgs_gate, 1, 1), MTLSize::new(q4mv::THREADS_PER_TG, 1, 1));
                enc.set_buffer(0, Some(&up_bufs[l]), 0);
                enc.set_buffer(3, Some(&up_out), 0);
                enc.dispatch_thread_groups(MTLSize::new(n_tgs_gate, 1, 1), MTLSize::new(q4mv::THREADS_PER_TG, 1, 1));
                // GEGLU in same encoder — select activation based on model architecture
                let geglu = if layers[l].use_gelu_tanh { &self.geglu_gelu_tanh_pipeline } else { &self.geglu_pipeline };
                enc.set_compute_pipeline_state(geglu);
                enc.set_buffer(0, Some(&gate_out), 0);
                enc.set_buffer(1, Some(&up_out), 0);
                enc.set_buffer(2, Some(&act_buf), 0);
                enc.set_bytes(3, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                enc.dispatch_threads(MTLSize::new(inter as u64, 1, 1), MTLSize::new(256, 1, 1));
                // Down in same encoder (depends on GEGLU)
                enc.set_compute_pipeline_state(&self.q4.f32_matvec);
                enc.set_buffer(0, Some(&down_bufs[l]), 0);
                enc.set_buffer(1, Some(&act_buf), 0);
                enc.set_buffer(2, Some(&down_out), 0);
                enc.set_bytes(3, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(4, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                enc.dispatch_threads(MTLSize::new(hidden as u64, 1, 1), MTLSize::new(256, 1, 1));
                enc.end_encoding();
            }
            // 7. Post-FFN: norm (if post_norms) + residual add
            if has_post_norms {
                if let Some(post_ffn) = layers[l].post_ffn_norm {
                    let post_ffn_buf = self.bufs.transient_from_f32(post_ffn);
                    let normed_ffn = self.bufs.output((hidden * 4) as u64);
                    {
                        use crate::metal::ops::full_pipeline::encode_rms_norm;
                        let enc = cmd.new_compute_command_encoder();
                        encode_rms_norm(enc, &self.rms_norm_pipeline,
                            &down_out, &post_ffn_buf, &normed_ffn, hidden, eps, norm_offset);
                        enc.end_encoding();
                    }
                    {
                        use crate::metal::ops::full_pipeline::encode_residual_add;
                        let enc = cmd.new_compute_command_encoder();
                        encode_residual_add(enc, &self.residual_add_pipeline,
                            &h_post_attn, &normed_ffn, &new_h, hidden);
                        enc.end_encoding();
                    }
                } else {
                    use crate::metal::ops::full_pipeline::encode_residual_add;
                    let enc = cmd.new_compute_command_encoder();
                    encode_residual_add(enc, &self.residual_add_pipeline,
                        &h_post_attn, &down_out, &new_h, hidden);
                    enc.end_encoding();
                }
            } else {
                let enc = cmd.new_compute_command_encoder();
                let len_val = hidden as u32;
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

        super::buffers::read_buffer_f32(&h_buf, hidden)
    }
}
