//! PLE wiring, Q4_KF FFN variants, env-selected FFN pipelines, layer-scalar dispatch.

#[allow(unused_imports)]
use crate::common::*;

/// Decode with PLE weights wired on the layer drives
/// `encode_per_layer_embed` (covers `decode/encode_ple.rs` end-to-end).
#[test]
fn decode_token_with_ple_weights_drives_per_layer_embed() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_q4_k};

    let ple_dim = 32usize;
    let num_layers = 1usize;
    let positions = 1usize;
    let ple_inputs: Vec<f32> = (0..positions * num_layers * ple_dim)
        .map(|i| (i as f32) * 0.01)
        .collect();
    metal.prepare_ple_inputs(&ple_inputs, num_layers, ple_dim);

    let wq = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.1));
    let wk = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.2));
    let wv = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.3));
    let wo = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let gate = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.5));
    let up = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down = quantize_q4_0(&synth_weight_f32(HIDDEN * INTER, 0.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();

    // PLE weights: input_gate is [ple_dim × hidden], projection is
    // [hidden × ple_dim], post_norm is [hidden].
    let ple_input_gate: Vec<f32> = (0..ple_dim * HIDDEN).map(|i| (i as f32) * 0.0001).collect();
    let ple_projection: Vec<f32> = (0..HIDDEN * ple_dim).map(|i| (i as f32) * 0.0001).collect();
    let ple_post_norm: Vec<f32> = vec![1.0f32; HIDDEN];

    let mut layer = build_synth_layer(&wq, &wk, &wv, &wo, &gate, &up, &down, &norm_w);
    layer.ple_input_gate = Some(&ple_input_gate);
    layer.ple_projection = Some(&ple_projection);
    layer.ple_post_norm = Some(&ple_post_norm);

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
    metal.clear_ple_inputs();
}

/// Decode with `DECODE_DEBUG=1` drives the `log_decode_entry`
/// diagnostic block in `decode/diag.rs` lines 24-39.
#[test]
fn decode_token_with_decode_debug_env_logs_diagnostic_entry() {
    decode_one_token_with_env(&[("DECODE_DEBUG", Some("1"))], |_| {});
}

/// Decode with Q4_KF FFN weights — drives `decode/encode_ffn.rs`
/// lines 86-105 (Q4_KF FFN branch) + the gated Q4_KF path 110-145.
#[test]
fn decode_token_with_q4kf_ffn_drives_q4kf_paths() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    use larql_compute::cpu::ops::q4_common::quantize_q4_k;
    // Q4_KF reads the same 144-byte super-block layout as Q4_K but
    // with pre-baked half-scales.  Synthetic Q4_K bytes pass through
    // the kernel (it just reads bytes) — output won't be numerically
    // meaningful, but the dispatch covers the branch.
    let wq = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.1));
    let wk = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.2));
    let wv = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.3));
    let wo = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let gate = quantize_q4_k(&synth_weight_f32(INTER * HIDDEN, 0.5));
    let up = quantize_q4_k(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down = quantize_q4_k(&synth_weight_f32(HIDDEN * INTER, 0.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();
    let mut layer = build_synth_layer(&wq, &wk, &wv, &wo, &gate, &up, &down, &norm_w);
    layer.gate = layer.gate.with_format(QuantFormat::Q4_KF);
    layer.up = layer.up.with_format(QuantFormat::Q4_KF);
    layer.down = layer.down.with_format(QuantFormat::Q4_KF);

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

/// Decode with `LARQL_GATE_UP_COOP=1` opts in to the cooperative
/// scale-loading variant — drives `decode/encode_ffn.rs` lines
/// 239-247 (use_coop branch).
#[test]
fn decode_token_with_gate_up_coop_env_drives_coop_pipeline() {
    decode_one_token_with_env(&[("LARQL_GATE_UP_COOP", Some("1"))], |_| {});
}

/// Decode with `LARQL_GATE_UP_8SG=0` + `LARQL_F16_ACC=1` opts to the
/// 4sg+f16-acc variant — drives `decode/encode_ffn.rs` lines 247-253.
#[test]
fn decode_token_with_4sg_f16_acc_env_drives_f16acc_pipeline() {
    decode_one_token_with_env(
        &[
            ("LARQL_GATE_UP_8SG", Some("0")),
            ("LARQL_F16_ACC", Some("1")),
        ],
        |_| {},
    );
}

/// Decode with `LARQL_GATE_UP_8SG=0` (no f16) — drives the plain 4sg
/// variant at lines 253-258.
#[test]
fn decode_token_with_4sg_env_drives_4sg_pipeline() {
    decode_one_token_with_env(&[("LARQL_GATE_UP_8SG", Some("0"))], |_| {});
}

/// Decode with Q4_KF + Standard (non-gated) FFN — drives
/// `decode/encode_ffn.rs` lines 146-180 (Q4_KF non-gated arm).
#[test]
fn decode_token_with_q4kf_standard_ffn_drives_non_gated_q4kf() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    use larql_compute::cpu::ops::q4_common::quantize_q4_k;
    let wq = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.1));
    let wk = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.2));
    let wv = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.3));
    let wo = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let up = quantize_q4_k(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down = quantize_q4_k(&synth_weight_f32(HIDDEN * INTER, 0.7));
    // gate is unused in Standard FFN — use a stub.
    let gate = vec![0u8; 256];
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();
    let mut layer = build_synth_layer(&wq, &wk, &wv, &wo, &gate, &up, &down, &norm_w);
    layer.gate = layer.gate.with_format(QuantFormat::Q4_KF);
    layer.up = layer.up.with_format(QuantFormat::Q4_KF);
    layer.down = layer.down.with_format(QuantFormat::Q4_KF);
    layer.ffn_type = FfnType::Standard;

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

/// Decode with `LARQL_DECODE_DUMP_LAYERS=<dir>` drives the per-layer
/// dump path in `decode/mod.rs` lines 621-672.
#[test]
fn decode_token_with_decode_dump_layers_env() {
    let tmp = std::env::temp_dir().join("larql-compute-metal-decode-dump-test");
    let _ = std::fs::create_dir_all(&tmp);
    let path = tmp.to_str().unwrap().to_string();
    decode_one_token_with_env(
        &[(
            "LARQL_DECODE_DUMP_LAYERS",
            Some(Box::leak(path.into_boxed_str())),
        )],
        |_| {
            let _ = std::fs::remove_dir_all(&tmp);
        },
    );
}

/// Decode with `layer_scalar != 0.0` drives the post-FFN scale_vector
/// dispatch (`decode/mod.rs` lines 593-602, non-MoE branch).
#[test]
fn decode_token_with_layer_scalar_drives_scale_vector_dispatch() {
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
    layer.layer_scalar = 0.5; // non-zero → scale_vector dispatch runs

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
