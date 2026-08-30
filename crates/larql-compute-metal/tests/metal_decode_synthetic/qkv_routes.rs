//! QKV format-route selection: uniform Q4_KF, mixed Q4_K/Q6_K, Q4_0 norm+QKV.

#[allow(unused_imports)]
use crate::common::*;

/// Q4_KF QKV format decode — drives `decode/encode_qkv.rs` lines
/// 230-237 (`UniformQ4Kf` arm of the format route).
#[test]
fn decode_token_with_q4kf_qkv_drives_uniform_q4kf_path() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_q4_k};
    let wq = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.1));
    let wk = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.2));
    let wv = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.3));
    let wo = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let gate = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.5));
    let up = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down = quantize_q4_0(&synth_weight_f32(HIDDEN * INTER, 0.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();
    let mut layer = build_synth_layer(&wq, &wk, &wv, &wo, &gate, &up, &down, &norm_w);
    layer.wq = layer.wq.with_format(QuantFormat::Q4_KF);
    layer.wk = layer.wk.with_format(QuantFormat::Q4_KF);
    layer.wv = layer.wv.with_format(QuantFormat::Q4_KF);

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
}

/// Mixed Q4_K + Q6_K_V format — drives the `MixedQ4kQ6kV` arm
/// (`decode/encode_qkv.rs` lines 258-284, Gemma 4 convention).
#[test]
fn decode_token_with_mixed_q4k_q6k_v_drives_mixed_path() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_q4_k, quantize_q6_k};
    let wq = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.1));
    let wk = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.2));
    let wv = quantize_q6_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.3));
    let wo = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let gate = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.5));
    let up = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down = quantize_q4_0(&synth_weight_f32(HIDDEN * INTER, 0.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();
    let mut layer = build_synth_layer(&wq, &wk, &wv, &wo, &gate, &up, &down, &norm_w);
    layer.wv = layer.wv.with_format(QuantFormat::Q6_K);

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
}

/// Q4_0 QKV format decode — drives `decode/encode_qkv.rs` line 130
/// (the Q4_0 norm+qkv chain), which my Q4_K tests don't reach.
#[test]
fn decode_token_with_q4_0_qkv_drives_q4_0_norm_qkv_path() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    use larql_compute::cpu::ops::q4_common::quantize_q4_0;
    let wq = quantize_q4_0(&synth_weight_f32(Q_DIM * HIDDEN, 0.1));
    let wk = quantize_q4_0(&synth_weight_f32(KV_DIM * HIDDEN, 0.2));
    let wv = quantize_q4_0(&synth_weight_f32(KV_DIM * HIDDEN, 0.3));
    let wo = quantize_q4_0(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let gate = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.5));
    let up = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down = quantize_q4_0(&synth_weight_f32(HIDDEN * INTER, 0.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();
    let mut layer = build_synth_layer(&wq, &wk, &wv, &wo, &gate, &up, &down, &norm_w);
    layer.wq = layer.wq.with_format(QuantFormat::Q4_0);
    layer.wk = layer.wk.with_format(QuantFormat::Q4_0);
    layer.wv = layer.wv.with_format(QuantFormat::Q4_0);
    layer.wo = layer.wo.with_format(QuantFormat::Q4_0);

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
}

/// Prefill with the production Gemma 3/4 attention shape — Q4_K Q/K +
/// Q6_K V — now runs the fused mixed kernel (slice 1 of the capability
/// selector). Before the QKV plan, prefill's `all_same_format` gate
/// could not reach `q4k_q6k_qkv_proj` at all and this triple silently
/// degraded to three per-projection dispatches. Same assertion set as
/// `prefill_q4_seq4_synthetic_smoke`.
#[test]
fn prefill_mixed_q4k_q6k_v_seq4_runs_fused_mixed_kernel() {
    // Serialised with every other GPU-touching test in this binary. The
    // three `decode_token_*` tests above already took this lock; the
    // prefill trio never did, and they are exactly the tests that went
    // non-finite about one run in two under `cargo test`. The lock's name
    // predates its job: it is now "one GPU test at a time". What makes two
    // concurrent backends interfere is a separate open question — this
    // makes the suite honest, it does not answer it.
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_q4_k, quantize_q6_k};

    let wq_data = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 3.1));
    let wk_data = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 3.2));
    let wv_data = quantize_q6_k(&synth_weight_f32(KV_DIM * HIDDEN, 3.3));
    let wo_data = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 3.4));
    let gate_data = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 3.5));
    let up_data = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 3.6));
    let down_data = quantize_q4_0(&synth_weight_f32(HIDDEN * INTER, 3.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.0008)).collect();

    let mut layer = build_synth_layer(
        &wq_data, &wk_data, &wv_data, &wo_data, &gate_data, &up_data, &down_data, &norm_w,
    );
    layer.wv = QuantWeight::new(QuantFormat::Q6_K, &wv_data, larql_compute::QuantAux::None);

    let seq_len = 4usize;
    let x: Vec<f32> = (0..seq_len * HIDDEN)
        .map(|i| ((i as f32 * 0.011 + 3.9).sin()) * 0.4)
        .collect();

    let result = (&metal as &dyn ComputeBackend)
        .as_any()
        .downcast_ref::<larql_compute_metal::MetalBackend>()
        .unwrap()
        .prefill_kquant(&[layer], &x, HIDDEN, INTER, seq_len, false, 0.0);

    let Some(result) = result else {
        panic!("prefill_kquant returned None for the mixed Q4_K/Q6_K-V triple");
    };
    assert_eq!(result.len(), seq_len * HIDDEN);
    assert_eq!(result.iter().filter(|v| v.is_nan()).count(), 0);
    assert_eq!(result.iter().filter(|v| v.is_infinite()).count(), 0);
    let max_abs = result.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
    assert!(max_abs > 0.0, "mixed prefill output is all-zero");
    assert!(
        max_abs < 1e6,
        "mixed prefill magnitude {max_abs} unreasonable"
    );
}

/// Uniform Q6_K attention routes PerProjection at prefill (no fused
/// uniform-Q6_K kernel exists). This coverage previously came from the
/// mixed Gemma triple as a side effect; once the plan routes that
/// triple to the fused mixed kernel, the per-projection prefill arm
/// needs its own exercise.
#[test]
fn prefill_uniform_q6k_seq2_runs_per_projection() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_q6_k};

    let wq_data = quantize_q6_k(&synth_weight_f32(Q_DIM * HIDDEN, 3.1));
    let wk_data = quantize_q6_k(&synth_weight_f32(KV_DIM * HIDDEN, 3.2));
    let wv_data = quantize_q6_k(&synth_weight_f32(KV_DIM * HIDDEN, 3.3));
    let wo_data = quantize_q6_k(&synth_weight_f32(HIDDEN * Q_DIM, 3.4));
    let gate_data = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 3.5));
    let up_data = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 3.6));
    let down_data = quantize_q4_0(&synth_weight_f32(HIDDEN * INTER, 3.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.0008)).collect();

    let mut layer = build_synth_layer(
        &wq_data, &wk_data, &wv_data, &wo_data, &gate_data, &up_data, &down_data, &norm_w,
    );
    layer.wq = QuantWeight::new(QuantFormat::Q6_K, &wq_data, larql_compute::QuantAux::None);
    layer.wk = QuantWeight::new(QuantFormat::Q6_K, &wk_data, larql_compute::QuantAux::None);
    layer.wv = QuantWeight::new(QuantFormat::Q6_K, &wv_data, larql_compute::QuantAux::None);
    layer.wo = QuantWeight::new(QuantFormat::Q6_K, &wo_data, larql_compute::QuantAux::None);

    let seq_len = 2usize;
    let x: Vec<f32> = (0..seq_len * HIDDEN)
        .map(|i| ((i as f32 * 0.011 + 3.9).sin()) * 0.4)
        .collect();
    let result = (&metal as &dyn ComputeBackend)
        .as_any()
        .downcast_ref::<larql_compute_metal::MetalBackend>()
        .unwrap()
        .prefill_kquant(&[layer], &x, HIDDEN, INTER, seq_len, false, 0.0);
    let Some(result) = result else {
        panic!("prefill_kquant returned None for uniform Q6_K");
    };
    assert_eq!(result.len(), seq_len * HIDDEN);
    assert!(result.iter().all(|v| v.is_finite()));
    assert!(result.iter().any(|v| v.abs() > 1e-6));
}

/// Uniform Q4_KF attention at prefill drives the Q4_KF fused arm.
/// The bytes are standard 144-byte GGUF Q4_K — the Q4_KF tag selects
/// the llama.cpp-exact kernel, not a different layout — so the retag
/// from Q4_K is an inline-to-inline `with_format`.
#[test]
fn prefill_uniform_q4kf_seq2_runs_fused_kernel() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_q4_k};

    let wq_data = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 3.1));
    let wk_data = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 3.2));
    let wv_data = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 3.3));
    let wo_data = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 3.4));
    let gate_data = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 3.5));
    let up_data = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 3.6));
    let down_data = quantize_q4_0(&synth_weight_f32(HIDDEN * INTER, 3.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.0008)).collect();

    let mut layer = build_synth_layer(
        &wq_data, &wk_data, &wv_data, &wo_data, &gate_data, &up_data, &down_data, &norm_w,
    );
    layer.wq = layer.wq.with_format(QuantFormat::Q4_KF);
    layer.wk = layer.wk.with_format(QuantFormat::Q4_KF);
    layer.wv = layer.wv.with_format(QuantFormat::Q4_KF);

    let seq_len = 2usize;
    let x: Vec<f32> = (0..seq_len * HIDDEN)
        .map(|i| ((i as f32 * 0.011 + 3.9).sin()) * 0.4)
        .collect();
    let result = (&metal as &dyn ComputeBackend)
        .as_any()
        .downcast_ref::<larql_compute_metal::MetalBackend>()
        .unwrap()
        .prefill_kquant(&[layer], &x, HIDDEN, INTER, seq_len, false, 0.0);
    let Some(result) = result else {
        panic!("prefill_kquant returned None for uniform Q4_KF");
    };
    assert_eq!(result.len(), seq_len * HIDDEN);
    assert!(result.iter().all(|v| v.is_finite()));
}
