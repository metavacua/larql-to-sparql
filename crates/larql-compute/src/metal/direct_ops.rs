use super::*;

impl MetalBackend {
    // ── Direct Q4 ops (for benchmarking outside the trait) ──

    pub fn q4_matvec_direct(
        &self,
        q4_data: &[u8],
        q8_x: &[i8],
        q8_scales: &[f32],
        num_rows: usize,
        hidden: usize,
    ) -> Vec<f32> {
        ops::q4_matvec::dispatch(
            &self.queue,
            &self.bufs,
            &self.q4.matvec,
            q4_data,
            q8_x,
            q8_scales,
            num_rows,
            hidden,
        )
    }

    pub fn q4_vecmat_direct(
        &self,
        activation: &[f32],
        q4_data: &[u8],
        intermediate: usize,
        hidden: usize,
    ) -> Vec<f32> {
        ops::q4_vecmat::dispatch(
            &self.queue,
            &self.bufs,
            &self.q4.vecmat,
            activation,
            q4_data,
            intermediate,
            hidden,
        )
    }

    /// Q4 × f32 matvec (for transposed down projection).
    pub fn q4_f32_matvec_direct(
        &self,
        q4_data: &[u8],
        x: &[f32],
        num_rows: usize,
        hidden: usize,
    ) -> Vec<f32> {
        ops::q4_f32_matvec::dispatch(
            &self.queue,
            &self.bufs,
            &self.q4.f32_matvec,
            q4_data,
            x,
            num_rows,
            hidden,
        )
    }

    /// Full layer pipeline: attention + FFN in one Metal command buffer.
    #[allow(clippy::too_many_arguments)]
    pub fn full_layer_direct(
        &self,
        w_q: &[f32],
        w_k: &[f32],
        w_v: &[f32],
        w_o: &[f32],
        gate_q4: &[u8],
        up_q4: &[u8],
        down_t_q4: &[u8],
        x: &[f32],
        seq_len: usize,
        hidden: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        inter: usize,
        attn_scale: f32,
    ) -> Vec<f32> {
        ops::full_layer::dispatch(
            &self.queue,
            &self.bufs,
            &self.f32_ops.transb_pipeline,
            &self.attention.causal_attn_pipeline,
            &self.q4,
            w_q,
            w_k,
            w_v,
            w_o,
            gate_q4,
            up_q4,
            down_t_q4,
            x,
            seq_len,
            hidden,
            num_q_heads,
            num_kv_heads,
            head_dim,
            inter,
            attn_scale,
        )
    }

    pub fn q4_matvec_pair_batch_direct(
        &self,
        gate_q4: &[u8],
        up_q4: &[u8],
        x_matrix: &[f32],
        seq_len: usize,
        num_rows: usize,
        hidden: usize,
    ) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        ops::q4_batched::pair_batch(
            &self.queue,
            &self.bufs,
            &self.q4,
            gate_q4,
            up_q4,
            x_matrix,
            seq_len,
            num_rows,
            hidden,
        )
    }
}
