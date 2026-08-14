//! Env-gated diagnostic branches of `decode/mod.rs` plus qk-norm and layer-norm path routing.

#[allow(unused_imports)]
use crate::common::*;

/// `LARQL_DEBUG_NAN_LAYERS=1` forces a per-layer commit+wait + NaN
/// histogram print.  Covers `decode/mod.rs` lines 528-543.
#[test]
fn decode_token_with_debug_nan_layers_env() {
    decode_one_token_with_env(&[("LARQL_DEBUG_NAN_LAYERS", Some("1"))], |_| {});
}

/// `LARQL_DUMP_L0=<dir>` enables the L0 residual dump on the first
/// layer. Covers the dump_l0_dir guard at line 279.
#[test]
fn decode_token_with_dump_l0_env() {
    let tmp = std::env::temp_dir().join("larql-compute-metal-dump-l0-test");
    let _ = std::fs::create_dir_all(&tmp);
    let path = tmp.to_str().unwrap().to_string();
    decode_one_token_with_env(
        &[("LARQL_DUMP_L0", Some(Box::leak(path.into_boxed_str())))],
        |_| {},
    );
}

/// `LARQL_DECODE_DIAG_LAYER=0` stops decode after layer 0 and dumps
/// stage buffers.  Covers the diag_stop_layer paths around line 254.
#[test]
fn decode_token_with_decode_diag_layer_env() {
    decode_one_token_with_env(&[("LARQL_DECODE_DIAG_LAYER", Some("0"))], |_| {});
}

/// `LARQL_DUMP_RESIDUALS=<path>` enables the residual-dump capture
/// (`super::buffers::read_buffer_f32` per layer).  Covers lines
/// 273-277 + the dump-write tail.
#[test]
fn decode_token_with_dump_residuals_env() {
    let tmp = std::env::temp_dir().join("larql-compute-metal-residual-dump.bin");
    let path = tmp.to_str().unwrap().to_string();
    decode_one_token_with_env(
        &[(
            "LARQL_DUMP_RESIDUALS",
            Some(Box::leak(path.into_boxed_str())),
        )],
        |_| {
            let _ = std::fs::remove_file(&tmp);
        },
    );
}

/// Decode with QK-norm weights wired (Gemma-style layer).  Drives the
/// fused-attention path in `decode/encode_attn.rs` lines 172-217
/// (q_norm_enabled && k_norm_enabled && !has_v_norm).
#[test]
fn decode_token_with_qk_norm_drives_fused_attention_path() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let metal = match larql_compute_metal::MetalBackend::new() {
        Some(m) => m,
        None => return,
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
    let head_dim_norm: Vec<f32> = (0..HEAD_DIM).map(|i| 1.0 + (i as f32 * 0.002)).collect();

    let mut layer = build_synth_layer(&wq, &wk, &wv, &wo, &gate, &up, &down, &norm_w);
    // Enable Gemma-style: has_post_norms + QK-norm weights.
    layer.has_post_norms = true;
    layer.post_attn_norm = &norm_w;
    layer.pre_ffn_norm = Some(&norm_w);
    layer.post_ffn_norm = Some(&norm_w);
    layer.q_norm_weight = Some(&head_dim_norm);
    layer.k_norm_weight = Some(&head_dim_norm);

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
    assert!(out.iter().all(|v| v.is_finite()));
}

/// Decode with `LARQL_FUSED_ATTN=0` opts out of attn_fused, forcing
/// the unfused QK-norm + RoPE + attend path.  Covers the unfused
/// branches around `decode/encode_attn.rs` lines 219+.
#[test]
fn decode_token_with_unfused_attn_env_drives_separate_qkn_rope_path() {
    decode_one_token_with_env(&[("LARQL_FUSED_ATTN", Some("0"))], |_| {});
}

/// Decode with `LARQL_FUSED_KV_APPEND_ATTEND=0` opts out of the fused
/// kv_append_attend kernel — separate append + attend dispatches.
#[test]
fn decode_token_with_unfused_kv_append_attend_env() {
    decode_one_token_with_env(&[("LARQL_FUSED_KV_APPEND_ATTEND", Some("0"))], |_| {});
}

/// `LARQL_PROFILE_SPLIT=1` drives the paired-commit per-stage timing
/// path — closes the encoder between attention CB and FFN CB so each
/// stage is recorded separately.  Covers `decode/mod.rs` lines
/// 396-402, 450-475 (gate_up CB split → down CB split).
#[test]
fn decode_token_with_profile_split_env() {
    decode_one_token_with_env(&[("LARQL_PROFILE_SPLIT", Some("1"))], |m| {
        // Reading the timing back covers the `take_last_split_timings`
        // path (`decode/profile.rs::take_last_split_timings`).
        let _ = larql_compute_metal::take_last_split_timings();
        let _ = m;
    });
}

/// `LARQL_FUSED_POST_FFN_NORM=0` opts out of the fused post-FFN
/// kernel — covers the unfused rms_norm + residual_add chain inside
/// `encode_post_ffn_residual` (already 91%) and the gated branch in
/// `decode/mod.rs` that picks `use_fused_post_ffn`.
#[test]
fn decode_token_with_unfused_post_ffn_norm_env() {
    decode_one_token_with_env(&[("LARQL_FUSED_POST_FFN_NORM", Some("0"))], |_| {});
}

/// LayerNorm decode (no bias) — drives `decode/encode_qkv.rs` lines
/// 167-175 (layer_norm_no_bias dispatch path).
#[test]
fn decode_token_with_layer_norm_no_bias_drives_layer_norm_path() {
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
    layer.norm_type = NormType::LayerNorm;

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

/// LayerNorm + bias decode — drives `decode/encode_qkv.rs` lines
/// 156-166 (layer_norm + bias dispatch path).
#[test]
fn decode_token_with_layer_norm_and_bias_drives_layer_norm_bias_path() {
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
    let bias: Vec<f32> = (0..HIDDEN).map(|i| (i as f32) * 0.001).collect();
    let mut layer = build_synth_layer(&wq, &wk, &wv, &wo, &gate, &up, &down, &norm_w);
    layer.norm_type = NormType::LayerNorm;
    layer.input_norm_bias = Some(&bias);

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
