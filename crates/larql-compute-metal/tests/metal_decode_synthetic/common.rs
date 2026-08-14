//! Shared fixture for the synthetic decode test modules: dims sized
//! for Q4_K super-blocks, deterministic synth weights, the layer
//! builder, and the env-mutation lock every module must hold.

pub use larql_compute::{
    Activation, ComputeBackend, DecodeBackend, FfnType, FullPipelineLayer, NormType, QuantFormat,
    QuantWeight,
};

/// Process-wide guard for tests that mutate env vars read by the decode
/// hot path (e.g. `LARQL_FUSED_PRELAYER_NORM`, `LARQL_QKV_FUSED`). Cargo
/// runs tests inside a binary in parallel by default; without this lock
/// a parallel `decode_token` test races with the env-toggling test and
/// observes the var in either state. Hold the guard for the entire
/// duration of any backend creation + decode that depends on the env.
pub static ENV_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Synthetic dims chosen to be Q4_K-compatible (multiples of 256) and
/// small enough for a fast test. Q4_K super-blocks are 256 elements.
pub const HIDDEN: usize = 256;
pub const INTER: usize = 512;
pub const HEAD_DIM: usize = 64;
pub const NUM_Q_HEADS: usize = 2;
pub const NUM_KV_HEADS: usize = 1;
pub const Q_DIM: usize = NUM_Q_HEADS * HEAD_DIM; // 128
pub const KV_DIM: usize = NUM_KV_HEADS * HEAD_DIM; // 64

pub fn synth_input(len: usize, seed: f32) -> Vec<f32> {
    (0..len)
        .map(|i| ((i as f32 * 0.013 + seed).sin() + 0.1 * ((i >> 4) as f32).cos()) * 0.5)
        .collect()
}

pub fn synth_weight_f32(len: usize, seed: f32) -> Vec<f32> {
    (0..len)
        .map(|i| ((i as f32 * 0.001 + seed).sin() + 0.2 * ((i >> 8) as f32).cos()) * 0.3)
        .collect()
}

// Test helper: 7 quant tensor slices + 1 norm slice = 8 args. Mirrors
// the production `FullPipelineLayer` constructor surface; collapsing to
// a single struct just for the test would obscure the per-tensor
// fixture-builder pattern callers actually want.
#[allow(clippy::too_many_arguments)]
pub fn build_synth_layer<'a>(
    wq_data: &'a [u8],
    wk_data: &'a [u8],
    wv_data: &'a [u8],
    wo_data: &'a [u8],
    gate_data: &'a [u8],
    up_data: &'a [u8],
    down_data: &'a [u8],
    norm_w: &'a [f32],
) -> FullPipelineLayer<'a> {
    FullPipelineLayer {
        attn_sinks: None,
        attn_q_bias: None,
        attn_k_bias: None,
        attn_v_bias: None,
        attn_o_bias: None,
        attn_softcap: 0.0,
        wq: QuantWeight::new(QuantFormat::Q4_K, wq_data, larql_compute::QuantAux::None),
        wk: QuantWeight::new(QuantFormat::Q4_K, wk_data, larql_compute::QuantAux::None),
        wv: QuantWeight::new(QuantFormat::Q4_K, wv_data, larql_compute::QuantAux::None),
        wo: QuantWeight::new(QuantFormat::Q4_K, wo_data, larql_compute::QuantAux::None),
        gate: QuantWeight::new(QuantFormat::Q4_0, gate_data, larql_compute::QuantAux::None),
        up: QuantWeight::new(QuantFormat::Q4_0, up_data, larql_compute::QuantAux::None),
        down: QuantWeight::new(QuantFormat::Q4_0, down_data, larql_compute::QuantAux::None),
        input_norm: norm_w,
        post_attn_norm: norm_w,
        pre_ffn_norm: None,
        post_ffn_norm: None,
        norm_offset: 0.0,
        has_post_norms: false, // Llama-style (non-Gemma); enables D-RMS-FUSE path
        activation: Activation::Silu,
        qk_norm_offset: 0.0,
        eps: 1e-6,
        norm_type: NormType::RmsNorm,
        ffn_type: FfnType::Gated,
        attn_scale: 1.0 / (HEAD_DIM as f32).sqrt(),
        head_dim: HEAD_DIM,
        num_q_heads: NUM_Q_HEADS,
        num_kv_heads: NUM_KV_HEADS,
        rope_base: 10_000.0,
        rotary_dim: 0,
        rope_freq: larql_compute::attention::rope::RopeFreqPlan::unscaled(
            HEAD_DIM,
            0_usize,
            10_000.0_f64,
        ),
        sliding_window: 0,
        has_v_norm: false,
        layer_scalar: 0.0,
        input_norm_bias: None,
        post_attn_norm_bias: None,
        q_norm_weight: None,
        k_norm_weight: None,
        ffn_up_bias: None,
        ffn_down_bias: None,
        moe: None,
        ffn_is_remote: false,
        moe_combined_output_norm: false,
        moe_outer_post_norm: None,
        kv_shared_source: None,
        residual_multiplier: 1.0,
        ple_input_gate: None,
        ple_projection: None,
        ple_post_norm: None,
    }
}

// ─── decode/mod.rs diagnostic-env-var coverage ───
//
// These tests drive the env-gated diagnostic branches in
// `decode/mod.rs` (NaN inspector, residual dump, decode-diag-layer
// early-stop, L0 dump) that production decode never enters but the
// per-file 90 % coverage policy requires.  Each test is gated on
// ENV_TEST_LOCK because env vars are process-global.

pub fn decode_one_token_with_env(
    vars: &[(&str, Option<&str>)],
    extra_fn: impl FnOnce(&larql_compute_metal::MetalBackend),
) {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_q4_k};
    let metal = match larql_compute_metal::MetalBackend::new() {
        Some(m) => m,
        None => return,
    };
    let saved: Vec<_> = vars
        .iter()
        .map(|(n, _)| (*n, std::env::var_os(n)))
        .collect();
    for (n, v) in vars {
        match v {
            Some(s) => unsafe { std::env::set_var(n, s) },
            None => unsafe { std::env::remove_var(n) },
        }
    }
    let wq = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.1));
    let wk = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.2));
    let wv = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.3));
    let wo = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let gate = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.5));
    let up = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down = quantize_q4_0(&synth_weight_f32(HIDDEN * INTER, 0.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();
    let layer = build_synth_layer(&wq, &wk, &wv, &wo, &gate, &up, &down, &norm_w);
    let x = synth_input(HIDDEN, 0.9);
    let mut kv = metal.create_kv_cache(1, 64, NUM_KV_HEADS, HEAD_DIM);
    let out = larql_compute_metal::MetalBackend::decode_token(
        &metal,
        &mut kv,
        &[layer],
        &x,
        HIDDEN,
        INTER,
        Q_DIM,
        KV_DIM,
        NUM_Q_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        10_000.0,
    );
    assert_eq!(out.len(), HIDDEN);
    extra_fn(&metal);
    for (n, v) in saved {
        match v {
            Some(s) => unsafe { std::env::set_var(n, s) },
            None => unsafe { std::env::remove_var(n) },
        }
    }
}
