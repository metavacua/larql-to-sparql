//! Metal shader registry — one file per shader, compiled as one library.
//!
//! Each shader module exports a `SHADER` constant with the MSL source.
//! `all_shaders()` concatenates them with the common header for compilation.

pub mod common;
pub mod sgemm;
pub mod sgemm_transb;
pub mod q4_matvec;
pub mod q4_vecmat;
pub mod q4_f32_matvec;
pub mod geglu;
pub mod quantize_q8;
pub mod causal_attention;
pub mod q4_matvec_v2;
pub mod q4_matvec_v3;
pub mod q4_matvec_v4;
pub mod q4_matvec_v5;
pub mod q8_matvec;
pub mod kv_attention;
pub mod q4_sparse_matvec;
pub mod residual_inject;
pub mod rope;
pub mod fused_attention;
pub mod fused_ops;
pub mod q8_attn_proj;
pub mod q4k_matvec;
pub mod q4k_qkv_proj;
pub mod q4kf_ffn_gate_up;
pub mod q4kf_qkv_proj;
pub mod q4k_ffn_gate_up;
pub mod q4k_geglu_down;
pub mod q6k_matvec;
pub mod activation;
pub mod layer_norm;
pub mod v_norm;
pub mod qk_norm;
pub mod turboquant_encode;
pub mod turboquant_decode;
pub mod graph_walk_knn;

/// Concatenate all shaders into one MSL source string for compilation.
pub fn all_shaders() -> String {
    let mut src = String::with_capacity(32768);
    src.push_str(common::HEADER);
    // f32 matmul
    src.push_str(sgemm::SHADER);
    src.push_str(sgemm_transb::SHADER);
    // Q4 dense matvec variants
    src.push_str(q4_matvec::SHADER);
    src.push_str(q4_matvec_v2::SHADER);
    src.push_str(q4_matvec_v3::SHADER);
    src.push_str(q4_matvec_v4::SHADER);
    src.push_str(q4_matvec_v5::SHADER);
    // Q4 other
    src.push_str(q4_vecmat::SHADER);
    src.push_str(q4_f32_matvec::SHADER);
    src.push_str(q4_sparse_matvec::SHADER);
    // Q8
    src.push_str(q8_matvec::SHADER);
    // Element-wise
    src.push_str(geglu::SHADER);
    src.push_str(quantize_q8::SHADER);
    src.push_str(residual_inject::SHADER);
    // Attention
    src.push_str(causal_attention::SHADER);
    src.push_str(kv_attention::SHADER);
    src.push_str(rope::SHADER);
    src.push_str(fused_attention::SHADER);
    src.push_str(fused_ops::SHADER);
    src.push_str(q8_attn_proj::SHADER);
    src.push_str(q4k_matvec::SHADER);
    src.push_str(q4k_qkv_proj::SHADER);
    src.push_str(q4kf_qkv_proj::SHADER);
    src.push_str(q4k_ffn_gate_up::SHADER);
    src.push_str(q4k_geglu_down::SHADER);
    src.push_str(q4kf_ffn_gate_up::SHADER);
    src.push_str(q6k_matvec::SHADER);
    // Standalone activations (non-gated FFN)
    src.push_str(activation::SHADER);
    // LayerNorm (StarCoder2, GPT-2)
    src.push_str(layer_norm::SHADER);
    // V-norm (parameter-free, Gemma 4)
    src.push_str(v_norm::SHADER);
    // QK-norm (learned-weight per-head RMS, Gemma 3/4)
    src.push_str(qk_norm::SHADER);
    // TurboQuant (KV cache compression)
    src.push_str(turboquant_encode::SHADER);
    src.push_str(turboquant_decode::SHADER);
    // Graph walk KNN
    src.push_str(graph_walk_knn::SHADER);
    src
}
