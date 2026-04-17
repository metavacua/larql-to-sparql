use super::*;

// ── ComputeBackend trait implementation ──

impl ComputeBackend for MetalBackend {
    fn matmul(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
        self.f32_ops.matmul(&self.queue, &self.bufs, a, b, self.flop_threshold.load(Ordering::Relaxed))
    }

    fn matmul_transb(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
        self.f32_ops.matmul_transb(&self.queue, &self.bufs, a, b, self.flop_threshold.load(Ordering::Relaxed))
    }

    fn matmul_batch(&self, ops: &[MatMulOp]) -> Vec<Array2<f32>> {
        ops.iter().map(|op| {
            if op.transpose_b { self.matmul_transb(op.a.view(), op.b.view()) }
            else { self.matmul(op.a.view(), op.b.view()) }
        }).collect()
    }

    fn q4_matvec(
        &self, q4_data: &[u8], q8_x: &[i8], q8_scales: &[f32],
        num_rows: usize, hidden: usize,
    ) -> Option<Vec<f32>> {
        Some(self.q4_matvec_direct(q4_data, q8_x, q8_scales, num_rows, hidden))
    }

    fn q4_vecmat(
        &self, activation: &[f32], q4_data: &[u8],
        intermediate: usize, hidden: usize,
    ) -> Option<Vec<f32>> {
        Some(self.q4_vecmat_direct(activation, q4_data, intermediate, hidden))
    }

    fn q4_matvec_pair_batch(
        &self, gate_q4: &[u8], up_q4: &[u8],
        x_matrix: &[f32], seq_len: usize,
        num_rows: usize, hidden: usize,
    ) -> Option<(Vec<Vec<f32>>, Vec<Vec<f32>>)> {
        Some(self.q4_matvec_pair_batch_direct(gate_q4, up_q4, x_matrix, seq_len, num_rows, hidden))
    }

    fn full_pipeline_q4(
        &self,
        layers: &[crate::FullPipelineLayer<'_>],
        x: &[f32],
        hidden: usize, inter: usize,
        q_dim: usize, kv_dim: usize,
        seq_len: usize,
        num_q_heads: usize, num_kv_heads: usize, head_dim: usize,
        rope_base: f32, use_qk_norm: bool, softcap: f32,
    ) -> Option<Vec<f32>> {
        let geglu = if layers.first().is_some_and(|l| l.activation == crate::Activation::GeluTanh) {
            &self.geglu_gelu_tanh_pipeline
        } else {
            &self.geglu_pipeline
        };
        Some(ops::full_pipeline::dispatch_full_pipeline(
            &self.queue, &self.bufs, &self.q4,
            geglu,
            &self.geglu_gelu_tanh_pipeline,
            &self.silu_pipeline,
            &self.gelu_tanh_pipeline,
            &self.q8_quant_pipeline,
            Some(&self.fused_attn_pipeline),
            &self.q8_matvec_pipeline,
            &self.q8_qkv_proj_pipeline,
            &self.q4k_matvec_pipeline, &self.q6k_matvec_pipeline,
            &self.rms_norm_pipeline, &self.residual_add_pipeline,
            &self.rms_norm_q8_pipeline, &self.residual_norm_q8_pipeline,
            Some(&self.q4k_qkv_proj_pipeline), Some(&self.q4k_proj_pipeline),
            None, None, // no rope_at_pos or KV cache for standard full_pipeline_q4
            layers, x, hidden, inter, q_dim, kv_dim,
            seq_len, num_q_heads, num_kv_heads, head_dim,
            rope_base, use_qk_norm, softcap,
        ))
    }

    fn multi_layer_q4_ffn(
        &self,
        layers_q4: &[(&[u8], &[u8], &[u8])],
        x: &[f32],
        inter: usize,
        hidden: usize,
    ) -> Option<Vec<f32>> {
        Some(MetalBackend::multi_layer_q4_ffn(self, layers_q4, x, inter, hidden))
    }

    fn q4k_matvec(
        &self, q4k_data: &[u8], x: &[f32], num_rows: usize, hidden: usize,
    ) -> Option<Vec<f32>> {
        use crate::metal::shaders::q4k_matvec as q4k;
        let buf_w = self.bufs.get_bytes(q4k_data);
        let buf_x = self.bufs.transient_from_f32(x);
        let buf_out = self.bufs.output((num_rows * 4) as u64);
        let n = num_rows as u32;
        let k = hidden as u32;
        let num_tgs = (num_rows as u64).div_ceil(q4k::ROWS_PER_TG);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.q4k_matvec_pipeline);
        enc.set_buffer(0, Some(&buf_w), 0);
        enc.set_buffer(1, Some(&buf_x), 0);
        enc.set_buffer(2, Some(&buf_out), 0);
        enc.set_bytes(3, 4, &n as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(4, 4, &k as *const u32 as *const std::ffi::c_void);
        enc.dispatch_thread_groups(
            metal::MTLSize::new(num_tgs, 1, 1),
            metal::MTLSize::new(q4k::THREADS_PER_TG, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        Some(super::buffers::read_buffer_f32(&buf_out, num_rows))
    }

    fn q6k_matvec(
        &self, q6k_data: &[u8], x: &[f32], num_rows: usize, hidden: usize,
    ) -> Option<Vec<f32>> {
        use crate::metal::shaders::q6k_matvec as q6k;
        let buf_w = self.bufs.get_bytes(q6k_data);
        let buf_x = self.bufs.transient_from_f32(x);
        let buf_out = self.bufs.output((num_rows * 4) as u64);
        let n = num_rows as u32;
        let k = hidden as u32;
        let num_tgs = (num_rows as u64).div_ceil(q6k::ROWS_PER_TG);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.q6k_matvec_pipeline);
        enc.set_buffer(0, Some(&buf_w), 0);
        enc.set_buffer(1, Some(&buf_x), 0);
        enc.set_buffer(2, Some(&buf_out), 0);
        enc.set_bytes(3, 4, &n as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(4, 4, &k as *const u32 as *const std::ffi::c_void);
        enc.dispatch_thread_groups(
            metal::MTLSize::new(num_tgs, 1, 1),
            metal::MTLSize::new(q6k::THREADS_PER_TG, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        Some(super::buffers::read_buffer_f32(&buf_out, num_rows))
    }

    fn prefill_q4(
        &self,
        layers: &[crate::FullPipelineLayer<'_>],
        x: &[f32],
        hidden: usize, inter: usize,
        q_dim: usize, kv_dim: usize,
        seq_len: usize,
        num_q_heads: usize, num_kv_heads: usize, head_dim: usize,
        rope_base: f32, use_qk_norm: bool, softcap: f32,
    ) -> Option<Vec<f32>> {
        // Use full_pipeline with KV cache population via separate RoPE + skip_rope=1
        let num_layers = layers.len();
        let mut cache_guard = self.kv_cache.lock().unwrap();
        if cache_guard.is_none() {
            *cache_guard = Some(self.create_kv_cache(num_layers, 4096, num_kv_heads, head_dim));
        }
        let kv = cache_guard.as_mut().unwrap();
        while kv.layers.len() < num_layers {
            kv.layers.push(ops::kv_cache::LayerKVCache::new(&self.bufs, 4096, num_kv_heads, head_dim));
        }
        let geglu = if layers.first().is_some_and(|l| l.activation == crate::Activation::GeluTanh) {
            &self.geglu_gelu_tanh_pipeline
        } else {
            &self.geglu_pipeline
        };
        Some(ops::full_pipeline::dispatch_full_pipeline(
            &self.queue, &self.bufs, &self.q4,
            geglu,
            &self.geglu_gelu_tanh_pipeline,
            &self.silu_pipeline,
            &self.gelu_tanh_pipeline,
            &self.q8_quant_pipeline,
            Some(&self.fused_attn_pipeline),
            &self.q8_matvec_pipeline,
            &self.q8_qkv_proj_pipeline,
            &self.q4k_matvec_pipeline, &self.q6k_matvec_pipeline,
            &self.rms_norm_pipeline, &self.residual_add_pipeline,
            &self.rms_norm_q8_pipeline, &self.residual_norm_q8_pipeline,
            Some(&self.q4k_qkv_proj_pipeline), Some(&self.q4k_proj_pipeline),
            Some(&self.rope_at_pos_pipeline), Some(kv),
            layers, x, hidden, inter, q_dim, kv_dim,
            seq_len, num_q_heads, num_kv_heads, head_dim,
            rope_base, use_qk_norm, softcap,
        ))
    }

    fn has_kv_cache(&self) -> bool { true }

    fn populate_kv_layer(
        &self, layer: usize,
        k_data: &[f32], v_data: &[f32],
        seq_len: usize, num_kv_heads: usize, head_dim: usize,
    ) {
        let mut cache_guard = self.kv_cache.lock().unwrap();
        // Ensure KV cache exists with enough layers
        if cache_guard.is_none() {
            *cache_guard = Some(self.create_kv_cache(layer + 1, 4096, num_kv_heads, head_dim));
        }
        let kv = cache_guard.as_mut().unwrap();
        // Extend if needed
        while kv.layers.len() <= layer {
            kv.layers.push(ops::kv_cache::LayerKVCache::new(&self.bufs, 4096, num_kv_heads, head_dim));
        }

        let lc = &mut kv.layers[layer];
        // Write K/V data directly to Metal buffers
        let total = seq_len * num_kv_heads * head_dim;
        let k_ptr = lc.k_cache.contents() as *mut f32;
        let v_ptr = lc.v_cache.contents() as *mut f32;
        // SAFETY: k_ptr/v_ptr point to pre-allocated Metal buffers sized for max_seq * kv_dim.
        // k_data/v_data are borrow-checked &[f32] params. Copy size is bounded by min(total, src.len()).
        unsafe {
            std::ptr::copy_nonoverlapping(k_data.as_ptr(), k_ptr, total.min(k_data.len()));
            std::ptr::copy_nonoverlapping(v_data.as_ptr(), v_ptr, total.min(v_data.len()));
        }
        lc.current_len = seq_len;
    }

    fn reset_kv_cache(&self) {
        let mut cache_guard = self.kv_cache.lock().unwrap();
        *cache_guard = None; // drop entirely so next decode_token re-creates with correct layer count
    }

    fn decode_token(
        &self,
        layers: &[crate::FullPipelineLayer<'_>],
        x: &[f32],
        hidden: usize, inter: usize,
        q_dim: usize, kv_dim: usize,
        num_q_heads: usize, num_kv_heads: usize, head_dim: usize,
        rope_base: f32,
    ) -> Option<Vec<f32>> {
        let num_layers = layers.len();
        // Lazily initialize KV cache
        let mut cache_guard = self.kv_cache.lock().unwrap();
        if cache_guard.is_none() {
            *cache_guard = Some(self.create_kv_cache(num_layers, 4096, num_kv_heads, head_dim));
        }
        let kv = cache_guard.as_mut().unwrap();
        Some(MetalBackend::decode_token(self, kv, layers, x, hidden, inter, q_dim, kv_dim,
            num_q_heads, num_kv_heads, head_dim, rope_base))
    }

    fn has_q4(&self) -> bool { true }

    fn preallocate_kv_cache_per_layer(
        &self, shapes: &[(usize, usize)], max_seq: usize,
    ) {
        // Replace any existing cache — callers invoke this once per model
        // load, before the first decode dispatch. If we kept an old cache
        // sized with the wrong per-layer dims the first decode would read
        // off the end of a global-layer buffer.
        let mut cache_guard = self.kv_cache.lock().unwrap();
        *cache_guard = Some(self.create_kv_cache_per_layer(shapes, max_seq));
    }

    fn name(&self) -> &str { "metal (GPU)" }

    fn device_info(&self) -> String {
        format!("Metal GPU, FLOP threshold: {}", self.flop_threshold())
    }
}
