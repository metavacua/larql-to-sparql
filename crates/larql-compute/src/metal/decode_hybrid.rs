//! Hybrid decode — GPU attention only, returns hidden state for CPU FFN.
//!
//! Unlike `decode_token` which runs attention+FFN on GPU in one command buffer,
//! this runs ONLY the attention portion per layer:
//!   norm → QKV → RoPE → V-norm → KV cache → attend → O proj → residual+norm
//!
//! The caller then runs FFN on CPU (e.g., vindex walk) and feeds the result
//! back for the next layer.
//!
//! This enables the hybrid pipeline: GPU attention + vindex walk FFN,
//! replacing 13.6ms of GPU FFN with ~1ms/layer of mmap'd sparse accumulation.

use super::*;

impl MetalBackend {
    /// Run ONE layer of attention on GPU with KV cache, return post-attention hidden state.
    ///
    /// Steps: input norm → QKV projection → RoPE → V-norm → KV append → KV attend →
    /// O projection → post-attention residual + post-attn norm.
    ///
    /// Does NOT run FFN — caller handles that (typically via vindex walk).
    /// Returns the post-attention, pre-FFN-norm hidden state as f32 vector.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_attention_layer(
        &self,
        kv_cache: &mut ops::kv_cache::KVCache,
        layer: &crate::FullPipelineLayer,
        layer_idx: usize,
        x: &[f32],
        hidden: usize,
        q_dim: usize,
        kv_dim: usize,
    ) -> Vec<f32> {
        let hidden_val = hidden as u32;
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
        let window_size = layer.sliding_window as u32;

        // Pre-cache weight buffers for this layer
        let wq_buf = self.bufs.get_bytes(layer.wq.data);
        let wk_buf = self.bufs.get_bytes(layer.wk.data);
        let wv_buf = self.bufs.get_bytes(layer.wv.data);
        let wo_buf = self.bufs.get_bytes(layer.wo.data);
        let wq_scale_buf = self.bufs.transient_from_f32(layer.wq.scales.unwrap_or(&[]));
        let wk_scale_buf = self.bufs.transient_from_f32(layer.wk.scales.unwrap_or(&[]));
        let wv_scale_buf = self.bufs.transient_from_f32(layer.wv.scales.unwrap_or(&[]));
        let wo_scale_buf = self.bufs.transient_from_f32(layer.wo.scales.unwrap_or(&[]));
        let input_norm_buf = self.bufs.transient_from_f32(layer.input_norm);
        let post_attn_norm_buf = self.bufs.transient_from_f32(layer.post_attn_norm);

        let h_buf = self.bufs.transient_from_f32(x);

        let cmd = self.queue.new_command_buffer();

        // ═══════════════════════════════════════════════════════════
        // ENCODER A: Input norm → QKV projection → RoPE → V-norm
        // ═══════════════════════════════════════════════════════════
        let q_out = self.bufs.output((q_dim * 4) as u64);
        let k_out = self.bufs.output((kv_dim * 4) as u64);
        let v_out = self.bufs.output((kv_dim * 4) as u64);

        let enc_a = cmd.new_compute_command_encoder();

        if uses_q4k {
            use crate::metal::ops::full_pipeline::encode_rms_norm;
            use crate::metal::shaders::q4kf_qkv_proj as qkv_sh;
            let norm_f32_buf = self.bufs.output((hidden * 4) as u64);
            let total_rows = (q_dim + kv_dim + kv_dim) as u32;
            let q_rows_val = q_dim as u32;
            let k_rows_val = kv_dim as u32;
            let v_rows_val = kv_dim as u32;
            let k_val = hidden as u32;
            let num_tgs = (total_rows as u64).div_ceil(qkv_sh::ROWS_PER_TG);

            encode_rms_norm(enc_a, &self.rms_norm_pipeline,
                &h_buf, &input_norm_buf, &norm_f32_buf,
                hidden, eps, norm_offset);

            let qkv_pipeline = if layer.wq.format == crate::QuantFormat::Q4_KF {
                &self.q4kf_qkv_proj_pipeline
            } else {
                &self.q4k_qkv_proj_pipeline
            };
            enc_a.set_compute_pipeline_state(qkv_pipeline);
            enc_a.set_buffer(0, Some(&wq_buf), 0);
            enc_a.set_buffer(1, Some(&wk_buf), 0);
            enc_a.set_buffer(2, Some(&wv_buf), 0);
            enc_a.set_buffer(3, Some(&norm_f32_buf), 0);
            enc_a.set_buffer(4, Some(&q_out), 0);
            enc_a.set_buffer(5, Some(&k_out), 0);
            enc_a.set_buffer(6, Some(&v_out), 0);
            enc_a.set_bytes(7, 4, &q_rows_val as *const u32 as *const std::ffi::c_void);
            enc_a.set_bytes(8, 4, &k_rows_val as *const u32 as *const std::ffi::c_void);
            enc_a.set_bytes(9, 4, &v_rows_val as *const u32 as *const std::ffi::c_void);
            enc_a.set_bytes(10, 4, &k_val as *const u32 as *const std::ffi::c_void);
            enc_a.dispatch_thread_groups(
                MTLSize::new(num_tgs, 1, 1),
                MTLSize::new(qkv_sh::THREADS_PER_TG, 1, 1),
            );
        } else {
            // Q8 path
            let q8_buf = self.bufs.output(hidden as u64);
            let q8s_buf = self.bufs.output((hidden / 32 * 4) as u64);

            enc_a.set_compute_pipeline_state(&self.rms_norm_q8_pipeline);
            enc_a.set_buffer(0, Some(&h_buf), 0);
            enc_a.set_buffer(1, Some(&input_norm_buf), 0);
            enc_a.set_buffer(2, Some(&q8_buf), 0);
            enc_a.set_buffer(3, Some(&q8s_buf), 0);
            enc_a.set_bytes(4, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
            enc_a.set_bytes(5, 4, &eps as *const f32 as *const std::ffi::c_void);
            enc_a.set_bytes(6, 4, &norm_offset as *const f32 as *const std::ffi::c_void);
            enc_a.dispatch_threads(MTLSize::new(hidden as u64, 1, 1), MTLSize::new(256.min(hidden as u64), 1, 1));

            let total_rows = (q_dim + kv_dim + kv_dim) as u32;
            enc_a.set_compute_pipeline_state(&self.q8_qkv_proj_pipeline);
            enc_a.set_buffer(0, Some(&wq_buf), 0);
            enc_a.set_buffer(1, Some(&wk_buf), 0);
            enc_a.set_buffer(2, Some(&wv_buf), 0);
            enc_a.set_buffer(3, Some(&q8_buf), 0);
            enc_a.set_buffer(4, Some(&wq_scale_buf), 0);
            enc_a.set_buffer(5, Some(&wk_scale_buf), 0);
            enc_a.set_buffer(6, Some(&wv_scale_buf), 0);
            enc_a.set_buffer(7, Some(&q8s_buf), 0);
            enc_a.set_buffer(8, Some(&q_out), 0);
            enc_a.set_buffer(9, Some(&k_out), 0);
            enc_a.set_buffer(10, Some(&v_out), 0);
            enc_a.set_bytes(11, 4, &(q_dim as u32) as *const u32 as *const std::ffi::c_void);
            enc_a.set_bytes(12, 4, &(kv_dim as u32) as *const u32 as *const std::ffi::c_void);
            enc_a.set_bytes(13, 4, &(kv_dim as u32) as *const u32 as *const std::ffi::c_void);
            enc_a.set_bytes(14, 4, &(hidden as u32) as *const u32 as *const std::ffi::c_void);
            enc_a.dispatch_thread_groups(
                MTLSize::new((total_rows as u64).div_ceil(8), 1, 1),
                MTLSize::new(256, 1, 1),
            );
        }

        // RoPE
        {
            let pos = kv_cache.layers[layer_idx].current_len as u32;
            let hd = layer_head_dim as u32;
            let rdim = layer_rotary_dim as u32;
            let rope_pairs = (layer_rotary_dim / 2) as u64;

            for qh in 0..layer_num_q_heads {
                let offset = (qh * layer_head_dim * 4) as u64;
                enc_a.set_compute_pipeline_state(&self.rope_at_pos_pipeline);
                enc_a.set_buffer(0, Some(&q_out), offset);
                enc_a.set_bytes(1, 4, &hd as *const u32 as *const std::ffi::c_void);
                enc_a.set_bytes(2, 4, &layer_rope_base as *const f32 as *const std::ffi::c_void);
                enc_a.set_bytes(3, 4, &pos as *const u32 as *const std::ffi::c_void);
                enc_a.set_bytes(4, 4, &rdim as *const u32 as *const std::ffi::c_void);
                enc_a.dispatch_threads(MTLSize::new(rope_pairs, 1, 1), MTLSize::new(rope_pairs.min(256), 1, 1));
            }
            for kvh in 0..layer_num_kv_heads {
                let offset = (kvh * layer_head_dim * 4) as u64;
                enc_a.set_compute_pipeline_state(&self.rope_at_pos_pipeline);
                enc_a.set_buffer(0, Some(&k_out), offset);
                enc_a.set_bytes(1, 4, &hd as *const u32 as *const std::ffi::c_void);
                enc_a.set_bytes(2, 4, &layer_rope_base as *const f32 as *const std::ffi::c_void);
                enc_a.set_bytes(3, 4, &pos as *const u32 as *const std::ffi::c_void);
                enc_a.set_bytes(4, 4, &rdim as *const u32 as *const std::ffi::c_void);
                enc_a.dispatch_threads(MTLSize::new(rope_pairs, 1, 1), MTLSize::new(rope_pairs.min(256), 1, 1));
            }
        }

        // V-norm
        if layer.has_v_norm {
            for kvh in 0..layer_num_kv_heads {
                let offset = (kvh * layer_head_dim * 4) as u64;
                let hd_val = layer_head_dim as u32;
                enc_a.set_compute_pipeline_state(&self.v_norm_pipeline);
                enc_a.set_buffer(0, Some(&v_out), offset);
                enc_a.set_buffer(1, Some(&v_out), offset);
                enc_a.set_bytes(2, 4, &hd_val as *const u32 as *const std::ffi::c_void);
                enc_a.set_bytes(3, 4, &eps as *const f32 as *const std::ffi::c_void);
                enc_a.dispatch_threads(
                    MTLSize::new(layer_head_dim as u64, 1, 1),
                    MTLSize::new((layer_head_dim as u64).min(256), 1, 1),
                );
            }
        }

        enc_a.end_encoding();

        // ═══════════════════════════════════════════════════════════
        // ENCODER B: KV cache append + attend
        // ═══════════════════════════════════════════════════════════
        let attn_out = self.bufs.output((layer_q_dim * 4) as u64);
        {
            let enc_b = cmd.new_compute_command_encoder();
            ops::kv_cache::encode_kv_append(
                enc_b, &kv_cache.layers[layer_idx],
                &self.kv_append_pipeline, &k_out, &v_out,
            );
            ops::kv_cache::encode_kv_attend(
                enc_b, &kv_cache.layers[layer_idx],
                &self.kv_attend_pipeline, &q_out, &attn_out,
                layer_num_q_heads, scale, window_size,
            );
            enc_b.end_encoding();
        }
        kv_cache.layers[layer_idx].current_len += 1;

        // ═══════════════════════════════════════════════════════════
        // ENCODER C: O projection → residual add (post-attention)
        // Returns h + O(attention_output) — ready for FFN.
        // ═══════════════════════════════════════════════════════════
        let h_post_attn = self.bufs.output((hidden * 4) as u64);

        let enc_c = cmd.new_compute_command_encoder();

        // O projection
        if uses_q4k {
            use crate::metal::shaders::q4kf_qkv_proj as proj_sh;
            let o_rows = hidden as u32;
            let o_k = layer_q_dim as u32;
            let num_tgs = (hidden as u64).div_ceil(proj_sh::ROWS_PER_TG);
            let o_out = self.bufs.output((hidden * 4) as u64);
            let o_pipeline = if layer.wo.format == crate::QuantFormat::Q4_KF {
                &self.q4kf_proj_pipeline
            } else {
                &self.q4k_proj_pipeline
            };
            enc_c.set_compute_pipeline_state(o_pipeline);
            enc_c.set_buffer(0, Some(&wo_buf), 0);
            enc_c.set_buffer(1, Some(&attn_out), 0);
            enc_c.set_buffer(2, Some(&o_out), 0);
            enc_c.set_bytes(3, 4, &o_rows as *const u32 as *const std::ffi::c_void);
            enc_c.set_bytes(4, 4, &o_k as *const u32 as *const std::ffi::c_void);
            enc_c.dispatch_thread_groups(
                MTLSize::new(num_tgs, 1, 1),
                MTLSize::new(proj_sh::THREADS_PER_TG, 1, 1),
            );

            // Residual add: h_post_attn = h + O_out
            if layer.has_post_norms {
                // Post-norm: norm(O) then add
                let normed_o = self.bufs.output((hidden * 4) as u64);
                use crate::metal::ops::full_pipeline::encode_rms_norm;
                encode_rms_norm(enc_c, &self.rms_norm_pipeline,
                    &o_out, &post_attn_norm_buf, &normed_o, hidden, eps, norm_offset);
                use crate::metal::ops::full_pipeline::encode_residual_add;
                encode_residual_add(enc_c, &self.residual_add_pipeline,
                    &h_buf, &normed_o, &h_post_attn, hidden);
            } else {
                // Standard: add O directly
                use crate::metal::ops::full_pipeline::encode_residual_add;
                encode_residual_add(enc_c, &self.residual_add_pipeline,
                    &h_buf, &o_out, &h_post_attn, hidden);
            }
        } else {
            // Q8 path: quantize attention → Q8 O proj → residual
            let o_q8 = self.bufs.output(layer_q_dim as u64);
            let o_q8s = self.bufs.output((layer_q_dim / 32 * 4) as u64);
            let o_out = self.bufs.output((hidden * 4) as u64);

            let dim_val = layer_q_dim as u32;
            let blocks = (layer_q_dim / 32) as u32;
            enc_c.set_compute_pipeline_state(&self.q8_quant_pipeline);
            enc_c.set_buffer(0, Some(&attn_out), 0);
            enc_c.set_buffer(1, Some(&o_q8), 0);
            enc_c.set_buffer(2, Some(&o_q8s), 0);
            enc_c.set_bytes(3, 4, &dim_val as *const u32 as *const std::ffi::c_void);
            enc_c.dispatch_threads(MTLSize::new(blocks as u64, 1, 1), MTLSize::new(256.min(blocks as u64), 1, 1));

            let o_rows = hidden as u32;
            let o_k = layer_q_dim as u32;
            enc_c.set_compute_pipeline_state(&self.q8_matvec_pipeline);
            enc_c.set_buffer(0, Some(&wo_buf), 0);
            enc_c.set_buffer(1, Some(&o_q8), 0);
            enc_c.set_buffer(2, Some(&wo_scale_buf), 0);
            enc_c.set_buffer(3, Some(&o_q8s), 0);
            enc_c.set_buffer(4, Some(&o_out), 0);
            enc_c.set_bytes(5, 4, &o_rows as *const u32 as *const std::ffi::c_void);
            enc_c.set_bytes(6, 4, &o_k as *const u32 as *const std::ffi::c_void);
            enc_c.dispatch_thread_groups(
                MTLSize::new((hidden as u64).div_ceil(8), 1, 1),
                MTLSize::new(256, 1, 1),
            );

            // Residual
            if layer.has_post_norms {
                let normed_o = self.bufs.output((hidden * 4) as u64);
                use crate::metal::ops::full_pipeline::encode_rms_norm;
                encode_rms_norm(enc_c, &self.rms_norm_pipeline,
                    &o_out, &post_attn_norm_buf, &normed_o, hidden, eps, norm_offset);
                use crate::metal::ops::full_pipeline::encode_residual_add;
                encode_residual_add(enc_c, &self.residual_add_pipeline,
                    &h_buf, &normed_o, &h_post_attn, hidden);
            } else {
                use crate::metal::ops::full_pipeline::encode_residual_add;
                encode_residual_add(enc_c, &self.residual_add_pipeline,
                    &h_buf, &o_out, &h_post_attn, hidden);
            }
        }

        enc_c.end_encoding();

        cmd.commit();
        cmd.wait_until_completed();

        super::buffers::read_buffer_f32(&h_post_attn, hidden)
    }
}
