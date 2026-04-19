//! Split-profiling variant of `decode_token`: 3 command buffers per layer.
//! Activated by `LARQL_PROFILE_SPLIT=1` via `generate`.
use super::*;

impl MetalBackend {
    /// Profile variant: splits each layer into 3 command buffers (attn /
    /// gate+up+GEGLU / down+residual) and times each stage separately.
    /// Activated by `LARQL_PROFILE_SPLIT=1`; only called for one decode step.
    /// Returns `(result, attn_ms, gate_up_ms, down_ms)` accumulated across all
    /// layers (divide by num_layers for per-layer averages).
    #[allow(clippy::too_many_arguments)]
    pub fn decode_token_split_profile(
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
    ) -> (Vec<f32>, f64, f64, f64) {
        let num_layers = layers.len();
        let hidden_val = hidden as u32;
        let inter_val = inter as u32;

        let max_q_dim = layers.iter().map(|l| l.num_q_heads * l.head_dim).max().unwrap_or(q_dim);
        let max_kv_dim = layers.iter().map(|l| l.num_kv_heads * l.head_dim).max().unwrap_or(kv_dim);

        let wq_bufs: Vec<_> = layers.iter().map(|l| self.bufs.get_bytes(l.wq.data)).collect();
        let wk_bufs: Vec<_> = layers.iter().map(|l| self.bufs.get_bytes(l.wk.data)).collect();
        let wv_bufs: Vec<_> = layers.iter().map(|l| self.bufs.get_bytes(l.wv.data)).collect();
        let wo_bufs: Vec<_> = layers.iter().map(|l| self.bufs.get_bytes(l.wo.data)).collect();
        let wq_scale_bufs: Vec<_> = layers.iter().map(|l| self.bufs.get_f32(l.wq.scales.unwrap_or(&[]))).collect();
        let wk_scale_bufs: Vec<_> = layers.iter().map(|l| self.bufs.get_f32(l.wk.scales.unwrap_or(&[]))).collect();
        let wv_scale_bufs: Vec<_> = layers.iter().map(|l| self.bufs.get_f32(l.wv.scales.unwrap_or(&[]))).collect();
        let wo_scale_bufs: Vec<_> = layers.iter().map(|l| self.bufs.get_f32(l.wo.scales.unwrap_or(&[]))).collect();
        let gate_bufs: Vec<_> = layers.iter().map(|l| self.bufs.get_bytes(l.gate.data)).collect();
        let up_bufs: Vec<_> = layers.iter().map(|l| self.bufs.get_bytes(l.up.data)).collect();
        let down_bufs: Vec<_> = layers.iter().map(|l| self.bufs.get_bytes(l.down.data)).collect();
        let input_norm_bufs: Vec<_> = layers.iter().map(|l| self.bufs.get_f32(l.input_norm)).collect();
        let post_attn_norm_bufs: Vec<_> = layers.iter().map(|l| self.bufs.get_f32(l.post_attn_norm)).collect();

        let h_init = self.bufs.transient_from_f32(x);
        let h_a = self.bufs.output((hidden * 4) as u64);
        let h_b = self.bufs.output((hidden * 4) as u64);
        let mut h_buf = &h_init;

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
        let normed_scratch = self.bufs.output((hidden * 4) as u64);
        let o_q8_scratch = self.bufs.output(max_q_dim as u64);
        let o_q8s_scratch = self.bufs.output((max_q_dim / 32 * 4) as u64);
        let scaled_scratch = self.bufs.output((hidden * 4) as u64);

        let mut t_attn = 0.0f64;
        let mut t_gate_up = 0.0f64;
        let mut t_down = 0.0f64;

        macro_rules! timed_cmd {
            ($acc:expr, $enc:ident, $body:block) => {{
                let _cmd = self.queue.new_command_buffer();
                {
                    let $enc = _cmd.new_compute_command_encoder();
                    $body
                    $enc.end_encoding();
                }
                let _t0 = std::time::Instant::now();
                _cmd.commit();
                _cmd.wait_until_completed();
                $acc += _t0.elapsed().as_secs_f64() * 1000.0;
            }};
        }

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
            let window_size = layer.sliding_window as u32;
            let new_h = if l % 2 == 0 { &h_a } else { &h_b };

            // ── Attn cmd: norm → QKV → QK-norm → RoPE → V-norm → KV-attend → O-proj → post-attn residual+norm ──
            timed_cmd!(t_attn, enc, {
                use crate::metal::ops::full_pipeline::encode_rms_norm;

                // Input norm
                if uses_q4k {
                    let uniform_q4k = layer.wq.format == layer.wk.format
                        && layer.wk.format == layer.wv.format
                        && layer.wq.format != crate::QuantFormat::Q6_K;
                    let mixed_q4k_q6k_v = layer.wq.format == crate::QuantFormat::Q4_K
                        && layer.wk.format == crate::QuantFormat::Q4_K
                        && layer.wv.format == crate::QuantFormat::Q6_K;

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
                        enc.dispatch_threads(MTLSize::new(hidden as u64, 1, 1), MTLSize::new(256.min(hidden as u64), 1, 1));
                    } else {
                        encode_rms_norm(enc, &self.rms_norm_pipeline, &h_buf, &input_norm_bufs[l], &norm_f32_buf, hidden, eps, norm_offset);
                    }

                    // QKV
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
                        use crate::metal::shaders::q4k_q6k_qkv_proj as sh;
                        let total_rows = (q_dim + kv_dim + kv_dim) as u64;
                        let num_tgs = total_rows.div_ceil(sh::ROWS_PER_TG);
                        let (q_rows_u, k_rows_u, v_rows_u, k_u) = (q_dim as u32, kv_dim as u32, kv_dim as u32, hidden as u32);
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
                        enc.dispatch_thread_groups(MTLSize::new(num_tgs, 1, 1), MTLSize::new(sh::THREADS_PER_TG, 1, 1));
                    } else {
                        use crate::metal::stages::qkv_proj::{self, Proj};
                        use crate::metal::stages::quant_matvec::Pipelines;
                        let pipes = Pipelines {
                            q4kf_proj: Some(&self.q4kf_proj_pipeline),
                            q4k_matvec_fallback: &self.q4k_matvec_pipeline,
                            q6k_matvec: &self.q6k_matvec_pipeline,
                            q4_matvec: &self.q4.matvec,
                        };
                        qkv_proj::encode_per_proj(
                            enc, &pipes, &norm_f32_buf, 0, &norm_f32_buf, 0, &norm_f32_buf, 0,
                            [
                                Proj { format: layer.wq.format, w_buf: &wq_bufs[l], out_buf: &q_out, out_off: 0, rows: q_dim },
                                Proj { format: layer.wk.format, w_buf: &wk_bufs[l], out_buf: &k_out, out_off: 0, rows: kv_dim },
                                Proj { format: layer.wv.format, w_buf: &wv_bufs[l], out_buf: &v_out, out_off: 0, rows: kv_dim },
                            ],
                            hidden,
                        );
                    }
                } else {
                    let (q8_buf, q8s_buf) = (&ffn_q8, &ffn_q8s);
                    enc.set_compute_pipeline_state(&self.rms_norm_q8_pipeline);
                    enc.set_buffer(0, Some(&h_buf), 0);
                    enc.set_buffer(1, Some(&input_norm_bufs[l]), 0);
                    enc.set_buffer(2, Some(&q8_buf), 0);
                    enc.set_buffer(3, Some(&q8s_buf), 0);
                    enc.set_bytes(4, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(5, 4, &eps as *const f32 as *const std::ffi::c_void);
                    enc.set_bytes(6, 4, &norm_offset as *const f32 as *const std::ffi::c_void);
                    enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256.min(hidden as u64), 1, 1));
                    let (total_rows, q_rows, k_rows, v_rows, k_val) = (
                        (q_dim + kv_dim + kv_dim) as u32, q_dim as u32, kv_dim as u32, kv_dim as u32, hidden as u32,
                    );
                    enc.set_compute_pipeline_state(&self.q8_qkv_proj_pipeline);
                    enc.set_buffer(0, Some(&wq_bufs[l]), 0); enc.set_buffer(1, Some(&wk_bufs[l]), 0);
                    enc.set_buffer(2, Some(&wv_bufs[l]), 0); enc.set_buffer(3, Some(&q8_buf), 0);
                    enc.set_buffer(4, Some(&wq_scale_bufs[l]), 0); enc.set_buffer(5, Some(&wk_scale_bufs[l]), 0);
                    enc.set_buffer(6, Some(&wv_scale_bufs[l]), 0); enc.set_buffer(7, Some(&q8s_buf), 0);
                    enc.set_buffer(8, Some(&q_out), 0); enc.set_buffer(9, Some(&k_out), 0);
                    enc.set_buffer(10, Some(&v_out), 0);
                    enc.set_bytes(11, 4, &q_rows as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(12, 4, &k_rows as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(13, 4, &v_rows as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(14, 4, &k_val as *const u32 as *const std::ffi::c_void);
                    enc.dispatch_thread_groups(MTLSize::new((total_rows as u64).div_ceil(8), 1, 1), MTLSize::new(256, 1, 1));
                }

                // QK-norm
                if let (Some(q_w), Some(k_w)) = (layer.q_norm_weight, layer.k_norm_weight) {
                    let hd_val = layer_head_dim as u32;
                    let qk_off = layer.qk_norm_offset;
                    let mut tg_w: usize = 1;
                    while tg_w < layer_head_dim && tg_w < 512 { tg_w <<= 1; }
                    let q_w_buf = self.bufs.get_f32(q_w);
                    let nq_val = layer_num_q_heads as u32;
                    enc.set_compute_pipeline_state(&self.qk_norm_pipeline);
                    enc.set_buffer(0, Some(&q_out), 0); enc.set_buffer(1, Some(&q_out), 0);
                    enc.set_buffer(2, Some(&q_w_buf), 0);
                    enc.set_bytes(3, 4, &hd_val as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(4, 4, &nq_val as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(5, 4, &eps as *const f32 as *const std::ffi::c_void);
                    enc.set_bytes(6, 4, &qk_off as *const f32 as *const std::ffi::c_void);
                    enc.dispatch_thread_groups(MTLSize::new(layer_num_q_heads as u64, 1, 1), MTLSize::new(tg_w as u64, 1, 1));
                    let k_w_buf = self.bufs.get_f32(k_w);
                    let nkv_val = layer_num_kv_heads as u32;
                    enc.set_buffer(0, Some(&k_out), 0); enc.set_buffer(1, Some(&k_out), 0);
                    enc.set_buffer(2, Some(&k_w_buf), 0);
                    enc.set_bytes(4, 4, &nkv_val as *const u32 as *const std::ffi::c_void);
                    enc.dispatch_thread_groups(MTLSize::new(layer_num_kv_heads as u64, 1, 1), MTLSize::new(tg_w as u64, 1, 1));
                }

                // RoPE
                {
                    let pos = kv_cache.layers[l].current_len as u32;
                    let hd = layer_head_dim as u32;
                    let rdim = layer_rotary_dim as u32;
                    let rope_pairs = (layer_rotary_dim / 2) as u64;
                    let (num_q, num_kv) = (layer_num_q_heads as u32, layer_num_kv_heads as u32);
                    enc.set_compute_pipeline_state(&self.rope_at_pos_batched_pipeline);
                    enc.set_buffer(0, Some(&q_out), 0);
                    enc.set_bytes(1, 4, &hd as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(2, 4, &layer_rope_base as *const f32 as *const std::ffi::c_void);
                    enc.set_bytes(3, 4, &pos as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(4, 4, &rdim as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(5, 4, &num_q as *const u32 as *const std::ffi::c_void);
                    enc.dispatch_threads(MTLSize::new(rope_pairs, layer_num_q_heads as u64, 1), MTLSize::new(rope_pairs.min(256), 1, 1));
                    enc.set_buffer(0, Some(&k_out), 0);
                    enc.set_bytes(5, 4, &num_kv as *const u32 as *const std::ffi::c_void);
                    enc.dispatch_threads(MTLSize::new(rope_pairs, layer_num_kv_heads as u64, 1), MTLSize::new(rope_pairs.min(256), 1, 1));
                }

                // V-norm (optional)
                if layer.has_v_norm {
                    let hd_val = layer_head_dim as u32;
                    let num_kv = layer_num_kv_heads as u32;
                    enc.set_compute_pipeline_state(&self.v_norm_batched_pipeline);
                    enc.set_buffer(0, Some(&v_out), 0); enc.set_buffer(1, Some(&v_out), 0);
                    enc.set_bytes(2, 4, &hd_val as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(3, 4, &eps as *const f32 as *const std::ffi::c_void);
                    enc.set_bytes(4, 4, &num_kv as *const u32 as *const std::ffi::c_void);
                    enc.dispatch_threads(MTLSize::new(layer_head_dim as u64, layer_num_kv_heads as u64, 1), MTLSize::new((layer_head_dim as u64).min(256), 1, 1));
                }

                // KV-cache + attend
                ops::kv_cache::encode_kv_append(enc, &kv_cache.layers[l], &self.kv_append_pipeline, &k_out, &v_out);
                ops::kv_cache::encode_kv_attend(enc, &kv_cache.layers[l], &self.kv_attend_pipeline, &q_out, &attn_out_buf, layer_num_q_heads, scale, window_size);

                // O-projection
                let _ffn_uses_q4k = layer.gate.format == crate::QuantFormat::Q4_K
                    || layer.gate.format == crate::QuantFormat::Q4_KF
                    || layer.gate.format == crate::QuantFormat::Q6_K;
                if uses_q4k {
                    use crate::metal::stages::quant_matvec::Pipelines;
                    let pipes = Pipelines {
                        q4kf_proj: Some(&self.q4kf_proj_pipeline),
                        q4k_matvec_fallback: &self.q4k_proj_pipeline,
                        q6k_matvec: &self.q6k_matvec_pipeline,
                        q4_matvec: &self.q4.matvec,
                    };
                    crate::metal::stages::o_proj::encode(enc, &pipes, &self.q8_quant_pipeline, layer.wo.format, &wo_bufs[l], &attn_out_buf, 0, &o_q8_scratch, 0, &o_q8s_scratch, 0, &o_out_buf, 0, layer_q_dim, hidden);
                } else {
                    let (dim_val, blocks) = (layer_q_dim as u32, (layer_q_dim / 32) as u32);
                    enc.set_compute_pipeline_state(&self.q8_quant_pipeline);
                    enc.set_buffer(0, Some(&attn_out_buf), 0); enc.set_buffer(1, Some(&o_q8_scratch), 0);
                    enc.set_buffer(2, Some(&o_q8s_scratch), 0);
                    enc.set_bytes(3, 4, &dim_val as *const u32 as *const std::ffi::c_void);
                    enc.dispatch_threads(MTLSize::new(blocks as u64, 1, 1), MTLSize::new(256.min(blocks as u64), 1, 1));
                    let (o_rows, o_k) = (hidden as u32, layer_q_dim as u32);
                    enc.set_compute_pipeline_state(&self.q8_matvec_pipeline);
                    enc.set_buffer(0, Some(&wo_bufs[l]), 0); enc.set_buffer(1, Some(&o_q8_scratch), 0);
                    enc.set_buffer(2, Some(&wo_scale_bufs[l]), 0); enc.set_buffer(3, Some(&o_q8s_scratch), 0);
                    enc.set_buffer(4, Some(&o_out_buf), 0);
                    enc.set_bytes(5, 4, &o_rows as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(6, 4, &o_k as *const u32 as *const std::ffi::c_void);
                    enc.dispatch_thread_groups(MTLSize::new((hidden as u64).div_ceil(8), 1, 1), MTLSize::new(256, 1, 1));
                }

                // Post-attn residual + FFN norm
                let has_post_norms = layer.has_post_norms;
                let ffn_uses_q4k = layer.gate.format == crate::QuantFormat::Q4_K
                    || layer.gate.format == crate::QuantFormat::Q4_KF
                    || layer.gate.format == crate::QuantFormat::Q6_K;
                if has_post_norms {
                    let normed_o = &normed_scratch;
                    encode_rms_norm(enc, &self.rms_norm_pipeline, &o_out_buf, &post_attn_norm_bufs[l], &normed_o, hidden, eps, norm_offset);
                    let pre_ffn_buf = if let Some(pfn) = layer.pre_ffn_norm {
                        self.bufs.get_f32(pfn)
                    } else { post_attn_norm_bufs[l].clone() };
                    if ffn_uses_q4k {
                        enc.set_compute_pipeline_state(&self.residual_norm_pipeline);
                        enc.set_buffer(0, Some(&h_buf), 0); enc.set_buffer(1, Some(&normed_o), 0);
                        enc.set_buffer(2, Some(&pre_ffn_buf), 0); enc.set_buffer(3, Some(&ffn_norm_out), 0);
                        enc.set_bytes(4, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                        enc.set_bytes(5, 4, &eps as *const f32 as *const std::ffi::c_void);
                        enc.set_bytes(6, 4, &norm_offset as *const f32 as *const std::ffi::c_void);
                        enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256.min(hidden as u64), 1, 1));
                        use crate::metal::ops::full_pipeline::encode_residual_add;
                        encode_residual_add(enc, &self.residual_add_pipeline, &h_buf, &normed_o, &h_post_attn, hidden);
                    } else {
                        enc.set_compute_pipeline_state(&self.residual_norm_q8_pipeline);
                        enc.set_buffer(0, Some(&h_buf), 0); enc.set_buffer(1, Some(&normed_o), 0);
                        enc.set_buffer(2, Some(&pre_ffn_buf), 0); enc.set_buffer(3, Some(&ffn_q8), 0);
                        enc.set_buffer(4, Some(&ffn_q8s), 0); enc.set_buffer(5, Some(&h_post_attn), 0);
                        enc.set_bytes(6, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                        enc.set_bytes(7, 4, &eps as *const f32 as *const std::ffi::c_void);
                        enc.set_bytes(8, 4, &norm_offset as *const f32 as *const std::ffi::c_void);
                        enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256.min(hidden as u64), 1, 1));
                    }
                } else if ffn_uses_q4k {
                    enc.set_compute_pipeline_state(&self.residual_norm_pipeline);
                    enc.set_buffer(0, Some(&h_buf), 0); enc.set_buffer(1, Some(&o_out_buf), 0);
                    enc.set_buffer(2, Some(&post_attn_norm_bufs[l]), 0); enc.set_buffer(3, Some(&ffn_norm_out), 0);
                    enc.set_bytes(4, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(5, 4, &eps as *const f32 as *const std::ffi::c_void);
                    enc.set_bytes(6, 4, &norm_offset as *const f32 as *const std::ffi::c_void);
                    enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256.min(hidden as u64), 1, 1));
                    use crate::metal::ops::full_pipeline::encode_residual_add;
                    encode_residual_add(enc, &self.residual_add_pipeline, &h_buf, &o_out_buf, &h_post_attn, hidden);
                } else {
                    enc.set_compute_pipeline_state(&self.residual_norm_q8_pipeline);
                    enc.set_buffer(0, Some(&h_buf), 0); enc.set_buffer(1, Some(&o_out_buf), 0);
                    enc.set_buffer(2, Some(&post_attn_norm_bufs[l]), 0); enc.set_buffer(3, Some(&ffn_q8), 0);
                    enc.set_buffer(4, Some(&ffn_q8s), 0); enc.set_buffer(5, Some(&h_post_attn), 0);
                    enc.set_bytes(6, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(7, 4, &eps as *const f32 as *const std::ffi::c_void);
                    enc.set_bytes(8, 4, &norm_offset as *const f32 as *const std::ffi::c_void);
                    enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256.min(hidden as u64), 1, 1));
                }
            });
            kv_cache.layers[l].current_len += 1;

            // ── Gate+up+GEGLU cmd ──
            let ffn_is_q4kf = layer.gate.format == crate::QuantFormat::Q4_KF;
            let ffn_uses_q4k = layer.gate.format == crate::QuantFormat::Q4_K
                || layer.gate.format == crate::QuantFormat::Q4_KF
                || layer.gate.format == crate::QuantFormat::Q6_K;

            timed_cmd!(t_gate_up, enc, {
                if ffn_is_q4kf {
                    if layer.is_gated() {
                        use crate::metal::shaders::q4kf_ffn_gate_up as q4kf_gu;
                        let n_tgs_per_mat = (inter as u64).div_ceil(q4kf_gu::ROWS_PER_TG);
                        enc.set_compute_pipeline_state(&self.q4kf_ffn_gate_up_pipeline);
                        enc.set_buffer(0, Some(&gate_bufs[l]), 0); enc.set_buffer(1, Some(&up_bufs[l]), 0);
                        enc.set_buffer(2, Some(&ffn_norm_out), 0); enc.set_buffer(3, Some(&gate_out_scratch), 0);
                        enc.set_buffer(4, Some(&up_out), 0);
                        enc.set_bytes(5, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                        enc.set_bytes(6, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                        enc.dispatch_thread_groups(MTLSize::new(n_tgs_per_mat * 2, 1, 1), MTLSize::new(q4kf_gu::THREADS_PER_TG, 1, 1));
                        let geglu = match layer.activation { crate::Activation::GeluTanh => &self.geglu_gelu_tanh_pipeline, _ => &self.geglu_pipeline };
                        enc.set_compute_pipeline_state(geglu);
                        enc.set_buffer(0, Some(&gate_out_scratch), 0); enc.set_buffer(1, Some(&up_out), 0); enc.set_buffer(2, Some(&act_buf), 0);
                        enc.set_bytes(3, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                        enc.dispatch_threads(MTLSize::new(inter as u64, 1, 1), MTLSize::new(256, 1, 1));
                    } else {
                        use crate::metal::shaders::q4kf_qkv_proj as q4kf;
                        let n_tgs_up = (inter as u64).div_ceil(q4kf::ROWS_PER_TG);
                        enc.set_compute_pipeline_state(&self.q4kf_proj_pipeline);
                        enc.set_buffer(0, Some(&up_bufs[l]), 0); enc.set_buffer(1, Some(&ffn_norm_out), 0); enc.set_buffer(2, Some(&up_out), 0);
                        enc.set_bytes(3, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                        enc.set_bytes(4, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                        enc.dispatch_thread_groups(MTLSize::new(n_tgs_up, 1, 1), MTLSize::new(q4kf::THREADS_PER_TG, 1, 1));
                        let act_pipe = match layer.activation { crate::Activation::GeluTanh => &self.gelu_tanh_pipeline, _ => &self.silu_pipeline };
                        enc.set_compute_pipeline_state(act_pipe);
                        enc.set_buffer(0, Some(&up_out), 0); enc.set_buffer(1, Some(&act_buf), 0);
                        enc.set_bytes(2, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                        enc.dispatch_threads(MTLSize::new(inter as u64, 1, 1), MTLSize::new(256, 1, 1));
                    }
                } else if ffn_uses_q4k {
                    if layer.is_gated() {
                        use crate::metal::shaders::q4k_matvec as q4k;
                        use crate::metal::shaders::q4k_ffn_gate_up as q4k_gu;
                        let n_tgs_per_mat = (inter as u64).div_ceil(q4k_gu::ROWS_PER_TG);
                        enc.set_compute_pipeline_state(&self.q4k_ffn_gate_up_pipeline);
                        enc.set_buffer(0, Some(&gate_bufs[l]), 0); enc.set_buffer(1, Some(&up_bufs[l]), 0);
                        enc.set_buffer(2, Some(&ffn_norm_out), 0); enc.set_buffer(3, Some(&gate_out_scratch), 0);
                        enc.set_buffer(4, Some(&up_out), 0);
                        enc.set_bytes(5, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                        enc.set_bytes(6, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                        enc.dispatch_thread_groups(MTLSize::new(n_tgs_per_mat * 2, 1, 1), MTLSize::new(q4k_gu::THREADS_PER_TG, 1, 1));
                        let geglu = match layer.activation { crate::Activation::GeluTanh => &self.geglu_gelu_tanh_pipeline, _ => &self.geglu_pipeline };
                        enc.set_compute_pipeline_state(geglu);
                        enc.set_buffer(0, Some(&gate_out_scratch), 0); enc.set_buffer(1, Some(&up_out), 0); enc.set_buffer(2, Some(&act_buf), 0);
                        enc.set_bytes(3, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                        enc.dispatch_threads(MTLSize::new(inter as u64, 1, 1), MTLSize::new(256, 1, 1));
                        let _ = q4k::ROWS_PER_TG; // suppress unused import warning
                    } else {
                        use crate::metal::shaders::q4k_matvec as q4k;
                        let n_tgs_up = (inter as u64).div_ceil(q4k::ROWS_PER_TG);
                        enc.set_compute_pipeline_state(&self.q4k_matvec_pipeline);
                        enc.set_buffer(0, Some(&up_bufs[l]), 0); enc.set_buffer(1, Some(&ffn_norm_out), 0); enc.set_buffer(2, Some(&up_out), 0);
                        enc.set_bytes(3, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                        enc.set_bytes(4, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                        enc.dispatch_thread_groups(MTLSize::new(n_tgs_up, 1, 1), MTLSize::new(q4k::THREADS_PER_TG, 1, 1));
                        let act_pipe = match layer.activation { crate::Activation::GeluTanh => &self.gelu_tanh_pipeline, _ => &self.silu_pipeline };
                        enc.set_compute_pipeline_state(act_pipe);
                        enc.set_buffer(0, Some(&up_out), 0); enc.set_buffer(1, Some(&act_buf), 0);
                        enc.set_bytes(2, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                        enc.dispatch_threads(MTLSize::new(inter as u64, 1, 1), MTLSize::new(256, 1, 1));
                    }
                } else {
                    use crate::metal::shaders::q4_matvec as q4mv;
                    let n_tgs_ffn = (inter as u64).div_ceil(q4mv::ROWS_PER_TG);
                    if layer.is_gated() {
                        enc.set_compute_pipeline_state(&self.q4.matvec);
                        enc.set_buffer(0, Some(&gate_bufs[l]), 0); enc.set_buffer(1, Some(&ffn_q8), 0);
                        enc.set_buffer(2, Some(&ffn_q8s), 0); enc.set_buffer(3, Some(&gate_out_scratch), 0);
                        enc.set_bytes(4, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                        enc.set_bytes(5, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                        enc.dispatch_thread_groups(MTLSize::new(n_tgs_ffn, 1, 1), MTLSize::new(q4mv::THREADS_PER_TG, 1, 1));
                        enc.set_buffer(0, Some(&up_bufs[l]), 0); enc.set_buffer(3, Some(&up_out), 0);
                        enc.dispatch_thread_groups(MTLSize::new(n_tgs_ffn, 1, 1), MTLSize::new(q4mv::THREADS_PER_TG, 1, 1));
                        let geglu = match layer.activation { crate::Activation::GeluTanh => &self.geglu_gelu_tanh_pipeline, _ => &self.geglu_pipeline };
                        enc.set_compute_pipeline_state(geglu);
                        enc.set_buffer(0, Some(&gate_out_scratch), 0); enc.set_buffer(1, Some(&up_out), 0); enc.set_buffer(2, Some(&act_buf), 0);
                        enc.set_bytes(3, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                        enc.dispatch_threads(MTLSize::new(inter as u64, 1, 1), MTLSize::new(256, 1, 1));
                    } else {
                        enc.set_compute_pipeline_state(&self.q4.matvec);
                        enc.set_buffer(0, Some(&up_bufs[l]), 0); enc.set_buffer(1, Some(&ffn_q8), 0);
                        enc.set_buffer(2, Some(&ffn_q8s), 0); enc.set_buffer(3, Some(&up_out), 0);
                        enc.set_bytes(4, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                        enc.set_bytes(5, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                        enc.dispatch_thread_groups(MTLSize::new(n_tgs_ffn, 1, 1), MTLSize::new(q4mv::THREADS_PER_TG, 1, 1));
                        let act_pipe = match layer.activation { crate::Activation::GeluTanh => &self.gelu_tanh_pipeline, _ => &self.silu_pipeline };
                        enc.set_compute_pipeline_state(act_pipe);
                        enc.set_buffer(0, Some(&up_out), 0); enc.set_buffer(1, Some(&act_buf), 0);
                        enc.set_bytes(2, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                        enc.dispatch_threads(MTLSize::new(inter as u64, 1, 1), MTLSize::new(256, 1, 1));
                    }
                }
            });

            // ── Down + post-FFN residual + layer scalar cmd ──
            timed_cmd!(t_down, enc, {
                if ffn_is_q4kf {
                    if layer.is_gated() {
                        use crate::metal::stages::quant_matvec::{self as qmv, Pipelines};
                        let pipes = Pipelines {
                            q4kf_proj: Some(&self.q4kf_proj_pipeline),
                            q4k_matvec_fallback: &self.q4k_matvec_pipeline,
                            q6k_matvec: &self.q6k_matvec_pipeline,
                            q4_matvec: &self.q4.matvec,
                        };
                        qmv::encode(enc, layer.down.format, &down_bufs[l], &act_buf, 0, &act_buf, 0, &act_buf, 0, &down_out, 0, &pipes, hidden, inter);
                    } else {
                        use crate::metal::shaders::q4kf_qkv_proj as q4kf;
                        let n_tgs_down = (hidden as u64).div_ceil(q4kf::ROWS_PER_TG);
                        enc.set_compute_pipeline_state(&self.q4kf_proj_pipeline);
                        enc.set_buffer(0, Some(&down_bufs[l]), 0); enc.set_buffer(1, Some(&act_buf), 0); enc.set_buffer(2, Some(&down_out), 0);
                        enc.set_bytes(3, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                        enc.set_bytes(4, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                        enc.dispatch_thread_groups(MTLSize::new(n_tgs_down, 1, 1), MTLSize::new(q4kf::THREADS_PER_TG, 1, 1));
                    }
                } else if ffn_uses_q4k {
                    if layer.is_gated() {
                        use crate::metal::stages::quant_matvec::{self as qmv, Pipelines};
                        let pipes = Pipelines {
                            q4kf_proj: Some(&self.q4kf_proj_pipeline),
                            q4k_matvec_fallback: &self.q4k_matvec_pipeline,
                            q6k_matvec: &self.q6k_matvec_pipeline,
                            q4_matvec: &self.q4.matvec,
                        };
                        qmv::encode(enc, layer.down.format, &down_bufs[l], &act_buf, 0, &act_buf, 0, &act_buf, 0, &down_out, 0, &pipes, hidden, inter);
                    } else {
                        use crate::metal::shaders::q4k_matvec as q4k;
                        let n_tgs_down = (hidden as u64).div_ceil(q4k::ROWS_PER_TG);
                        enc.set_compute_pipeline_state(&self.q4k_matvec_pipeline);
                        enc.set_buffer(0, Some(&down_bufs[l]), 0); enc.set_buffer(1, Some(&act_buf), 0); enc.set_buffer(2, Some(&down_out), 0);
                        enc.set_bytes(3, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                        enc.set_bytes(4, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                        enc.dispatch_thread_groups(MTLSize::new(n_tgs_down, 1, 1), MTLSize::new(q4k::THREADS_PER_TG, 1, 1));
                    }
                } else {
                    enc.set_compute_pipeline_state(&self.q4.f32_matvec);
                    enc.set_buffer(0, Some(&down_bufs[l]), 0); enc.set_buffer(1, Some(&act_buf), 0); enc.set_buffer(2, Some(&down_out), 0);
                    enc.set_bytes(3, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(4, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                    enc.dispatch_threads(MTLSize::new(hidden as u64, 1, 1), MTLSize::new(256, 1, 1));
                }

                // Post-FFN residual
                let has_post_norms = layer.has_post_norms;
                if has_post_norms {
                    if let Some(post_ffn) = layer.post_ffn_norm {
                        let post_ffn_buf = self.bufs.get_f32(post_ffn);
                        let normed_ffn = &normed_scratch;
                        use crate::metal::ops::full_pipeline::encode_rms_norm;
                        encode_rms_norm(enc, &self.rms_norm_pipeline, &down_out, &post_ffn_buf, normed_ffn, hidden, eps, norm_offset);
                        use crate::metal::ops::full_pipeline::encode_residual_add;
                        encode_residual_add(enc, &self.residual_add_pipeline, &h_post_attn, normed_ffn, new_h, hidden);
                    } else {
                        use crate::metal::ops::full_pipeline::encode_residual_add;
                        encode_residual_add(enc, &self.residual_add_pipeline, &h_post_attn, &down_out, new_h, hidden);
                    }
                } else {
                    let len_val = hidden as u32;
                    enc.set_compute_pipeline_state(&self.residual_add_pipeline);
                    enc.set_buffer(0, Some(&h_post_attn), 0); enc.set_buffer(1, Some(&down_out), 0); enc.set_buffer(2, Some(new_h), 0);
                    enc.set_bytes(3, 4, &len_val as *const u32 as *const std::ffi::c_void);
                    enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256.min(hidden as u64), 1, 1));
                }

                // Layer scalar
                if layer.layer_scalar != 0.0 {
                    crate::metal::stages::layer_scalar::encode(enc, &self.scale_vector_pipeline, new_h, 1, hidden, layer.layer_scalar);
                }
                let _ = &scaled_scratch;
            });

            h_buf = new_h;
        }

        let result = super::buffers::read_buffer_f32(&h_buf, hidden);
        let total = t_attn + t_gate_up + t_down;
        let pct = |v: f64| if total > 0.0 { v / total * 100.0 } else { 0.0 };
        eprintln!(
            "[profile-split] {:>2} layers: attn={:.2}ms ({:.0}%)  gate+up={:.2}ms ({:.0}%)  down={:.2}ms ({:.0}%)  total={:.2}ms",
            num_layers, t_attn, pct(t_attn), t_gate_up, pct(t_gate_up), t_down, pct(t_down), total,
        );
        eprintln!(
            "[profile-split] per-layer: attn={:.3}ms  gate+up={:.3}ms  down={:.3}ms",
            t_attn / num_layers as f64, t_gate_up / num_layers as f64, t_down / num_layers as f64,
        );
        (result, t_attn, t_gate_up, t_down)
    }
}
