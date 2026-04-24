use super::*;

impl MetalBackend {
    /// Full pipeline: attention + FFN for all layers in ONE command buffer.
    /// No CPU-GPU round-trips between layers.
    /// This is the old benchmark entry point — uses dummy norms (no residual correctness).
    pub fn full_pipeline(
        &self,
        layers: &[ops::full_pipeline::LayerWeights],
        x: &[f32],
        hidden: usize, inter: usize,
        q_dim: usize, kv_dim: usize,
    ) -> Vec<f32> {
        // Convert old LayerWeights to new FullPipelineLayer with dummy norms
        let dummy_norm = vec![1.0f32; hidden];
        // Convert old LayerWeights (Q4 attention) to new FullPipelineLayer (Q8 attention)
        // For backward compat: treat Q4 data as Q8 (wrong but benchmark-only path)
        let _dummy_scales = vec![1.0f32; hidden * hidden / 32]; // oversized, reserved for Q8 path
        let full_layers: Vec<crate::FullPipelineLayer> = layers.iter().map(|l| {
            crate::FullPipelineLayer {
                wq: crate::QuantWeight { data: l.wq_q4, scales: None, format: crate::QuantFormat::Q4_0 },
                wk: crate::QuantWeight { data: l.wk_q4, scales: None, format: crate::QuantFormat::Q4_0 },
                wv: crate::QuantWeight { data: l.wv_q4, scales: None, format: crate::QuantFormat::Q4_0 },
                wo: crate::QuantWeight { data: l.wo_q4, scales: None, format: crate::QuantFormat::Q4_0 },
                gate: crate::QuantWeight { data: l.gate_q4, scales: None, format: crate::QuantFormat::Q4_0 },
                up: crate::QuantWeight { data: l.up_q4, scales: None, format: crate::QuantFormat::Q4_0 },
                down: crate::QuantWeight { data: l.down_t_q4, scales: None, format: crate::QuantFormat::Q4_0 },
                input_norm: &dummy_norm, post_attn_norm: &dummy_norm,
                pre_ffn_norm: None, post_ffn_norm: None,
                norm_offset: 0.0, has_post_norms: false,
                activation: crate::Activation::Silu,
                qk_norm_offset: 0.0,
                eps: 1e-6,
                norm_type: crate::NormType::RmsNorm,
                ffn_type: crate::FfnType::Gated,
                attn_scale: 0.0,
                head_dim: 0,
                num_q_heads: 0,
                num_kv_heads: 0,
                rope_base: 10000.0,
                rotary_dim: 0,
                sliding_window: 0,
                has_v_norm: false,
                layer_scalar: 0.0,
                input_norm_bias: None,
                post_attn_norm_bias: None,
                q_norm_weight: None,
                k_norm_weight: None,
                ffn_up_bias: None,
                ffn_down_bias: None,
                moe: None, moe_combined_output_norm: false, moe_outer_post_norm: None,
            }
        }).collect();
        ops::full_pipeline::dispatch_full_pipeline(
            &self.queue, &self.bufs, &self.q4,
            &self.geglu_pipeline,
            &self.geglu_gelu_tanh_pipeline,
            &self.silu_pipeline,
            &self.gelu_tanh_pipeline,
            &self.q8_quant_pipeline,
            None,
            &self.q8_matvec_pipeline,
            &self.q8_qkv_proj_pipeline,
            &self.q4k_matvec_pipeline, &self.q6k_matvec_pipeline,
            &self.rms_norm_pipeline, &self.residual_add_pipeline,
            &self.rms_norm_q8_pipeline, &self.residual_norm_q8_pipeline,
            None,       // no q4k_qkv_proj (legacy 148-byte)
            None, None, // no q4kf_qkv_proj / q4kf_proj (legacy benchmark path)
            None,       // no rope_at_pos
            None,       // no qk_norm
            None,       // no scale_vector (no layer_scalar)
            None,       // no KV cache
            &full_layers, x, hidden, inter, q_dim, kv_dim,
            1, 0, 0, 0, 0.0, false, 0.0,
        )
    }

    /// Multi-layer Q4 FFN in ONE command buffer.
    /// gate → up → GEGLU → down → Q8 quantize → next layer.
    /// All on GPU, no CPU return between layers.
    pub fn multi_layer_q4_ffn(
        &self,
        layers_q4: &[(&[u8], &[u8], &[u8])], // [(gate, up, down_t)]
        x: &[f32],
        inter: usize,
        hidden: usize,
    ) -> Vec<f32> {
        ops::q4_batched::multi_layer_ffn(
            &self.queue, &self.bufs, &self.q4,
            &self.geglu_pipeline, &self.q8_quant_pipeline,
            layers_q4, x, inter, hidden,
        )
    }
}
