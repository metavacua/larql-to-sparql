//! encode_attn branch coverage via explicit BackendOptions; kv-shared, post-norm fallbacks, per-projection and fused-Q8 QKV routes.

#[allow(unused_imports)]
use crate::common::*;

// ─────────────────────────────────────────────────────────────────
// encode_attn.rs branch coverage via explicit BackendOptions.
// `MetalBackend::new()` snapshots env into `DecodeFlags` at startup,
// so env-set-after-new tests can't toggle decode-path branches.
// `with_options(...)` bypasses env entirely — direct flag injection.
// ─────────────────────────────────────────────────────────────────

// Build a QK-norm-enabled (Gemma-style) layer for `fused_attn` and
// `fused_qk_norm_rope` branch coverage. Mirrors `build_synth_layer`'s
// 8-arg fixture-builder shape but with two extra slices (norm + head-
// dim norm).
#[allow(clippy::too_many_arguments)]
fn synth_qk_norm_layer<'a>(
    wq: &'a [u8],
    wk: &'a [u8],
    wv: &'a [u8],
    wo: &'a [u8],
    gate: &'a [u8],
    up: &'a [u8],
    down: &'a [u8],
    norm_w: &'a [f32],
    head_dim_norm: &'a [f32],
) -> FullPipelineLayer<'a> {
    let mut layer = build_synth_layer(wq, wk, wv, wo, gate, up, down, norm_w);
    layer.has_post_norms = true;
    layer.post_attn_norm = norm_w;
    layer.pre_ffn_norm = Some(norm_w);
    layer.post_ffn_norm = Some(norm_w);
    layer.q_norm_weight = Some(head_dim_norm);
    layer.k_norm_weight = Some(head_dim_norm);
    layer
}

/// `BackendOptions { fused_attn: true }` drives the triple-fused
/// `attn_fused` path in `decode/encode_attn.rs` lines 172-229.
/// Gated on `q_norm_enabled && k_norm_enabled && !has_v_norm &&
/// kv_shared_source.is_none() && head_dim <= MAX_HEAD_DIM_SINGLE_SG
/// && attn_span <= SHORT_ATTENTION_SPAN`. All hold for the synth
/// fixture (HEAD_DIM=64 ≤ 256, t_val=1 ≤ 1024).
#[test]
fn decode_token_with_fused_attn_options_drives_attn_fused_path() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut opts = larql_compute_metal::BackendOptions::default();
    opts.decode_flags.fused_attn = true;
    let Some(metal) = larql_compute_metal::MetalBackend::with_options(opts) else {
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
    let head_dim_norm: Vec<f32> = (0..HEAD_DIM).map(|i| 1.0 + (i as f32 * 0.002)).collect();
    let layer = synth_qk_norm_layer(
        &wq,
        &wk,
        &wv,
        &wo,
        &gate,
        &up,
        &down,
        &norm_w,
        &head_dim_norm,
    );
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

/// `BackendOptions { fused_qk_norm_rope: false }` with QK-norm
/// weights drives the legacy non-fused `qk_norm_qk_pipeline` +
/// separate batched-RoPE path in `decode/encode_attn.rs` lines
/// 267-316.
#[test]
fn decode_token_with_unfused_qkn_rope_options_drives_legacy_qkn_path() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut opts = larql_compute_metal::BackendOptions::default();
    opts.decode_flags.fused_qk_norm_rope = false;
    let Some(metal) = larql_compute_metal::MetalBackend::with_options(opts) else {
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
    let head_dim_norm: Vec<f32> = (0..HEAD_DIM).map(|i| 1.0 + (i as f32 * 0.002)).collect();
    let layer = synth_qk_norm_layer(
        &wq,
        &wk,
        &wv,
        &wo,
        &gate,
        &up,
        &down,
        &norm_w,
        &head_dim_norm,
    );
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

/// `BackendOptions { fused_kv_append_attend: false }` drives the
/// unfused `encode_kv_append` + `encode_kv_attend` path in
/// `decode/encode_attn.rs` lines 410-428 (and the `current_len`
/// bump at line 433 in the non-shared, non-fused-attn branch).
#[test]
fn decode_token_with_unfused_kv_aa_options_drives_unfused_append_attend() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut opts = larql_compute_metal::BackendOptions::default();
    opts.decode_flags.fused_kv_append_attend = false;
    let Some(metal) = larql_compute_metal::MetalBackend::with_options(opts) else {
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
    assert!(out.iter().all(|v| v.is_finite()));
}

/// Two-layer decode with layer[1].kv_shared_source = Some(0)
/// drives the shared-cache branch in `decode/encode_attn.rs`:
/// lines 131-141 (source-pinned pos/t_val), 373-409 (attend against
/// source's cache, skip own append).
#[test]
fn decode_token_with_kv_shared_source_drives_shared_layer_branch() {
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
    let layer0 = build_synth_layer(&wq, &wk, &wv, &wo, &gate, &up, &down, &norm_w);
    let mut layer1 = build_synth_layer(&wq, &wk, &wv, &wo, &gate, &up, &down, &norm_w);
    // Gemma 4 E2B style: layer 1 reads K/V from layer 0's cache.
    layer1.kv_shared_source = Some(0);
    let x = synth_input(HIDDEN, 0.9);
    let mut kv = metal.create_kv_cache(2, 64, NUM_KV_HEADS, HEAD_DIM);
    let out = larql_compute_metal::MetalBackend::decode_token(
        &metal,
        &mut kv,
        &[layer0, layer1],
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

/// `has_post_norms = true` with `pre_ffn_norm = None` drives the
/// `bufs.post_attn_norm.clone()` fallback at `decode/encode_attn.rs`
/// line 505 (when the layer doesn't carry a separate pre-FFN norm).
#[test]
fn decode_token_with_post_norms_no_pre_ffn_norm_drives_fallback() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    use larql_compute::cpu::ops::q4_common::quantize_q4_k;
    let wq = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.1));
    let wk = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.2));
    let wv = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.3));
    let wo = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let gate = quantize_q4_k(&synth_weight_f32(INTER * HIDDEN, 0.5));
    let up = quantize_q4_k(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down = quantize_q4_k(&synth_weight_f32(HIDDEN * INTER, 0.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();
    let mut layer = build_synth_layer(&wq, &wk, &wv, &wo, &gate, &up, &down, &norm_w);
    layer.has_post_norms = true;
    layer.post_attn_norm = &norm_w;
    layer.pre_ffn_norm = None; // <- triggers fallback at L505
    layer.post_ffn_norm = Some(&norm_w);
    // Q4_K FFN gate so `ffn_uses_kquant` is true → exercises the
    // fused-post-attn + residual_norm_store path that consults
    // `pre_ffn_buf`.
    layer.gate = layer.gate.with_format(QuantFormat::Q4_K);
    layer.up = layer.up.with_format(QuantFormat::Q4_K);
    layer.down = layer.down.with_format(QuantFormat::Q4_K);
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

/// `BackendOptions { fused_post_attn_norm: false }` with
/// `has_post_norms = true` and a Q4_K FFN drives the un-triple-fused
/// post-attn-norm path in `decode/encode_attn.rs` lines 528-557:
/// separate `encode_rms_norm` + `residual_norm_store_pipeline` chain.
#[test]
fn decode_token_with_unfused_post_attn_norm_options_drives_split_norm() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut opts = larql_compute_metal::BackendOptions::default();
    opts.decode_flags.fused_post_attn_norm = false;
    let Some(metal) = larql_compute_metal::MetalBackend::with_options(opts) else {
        return;
    };
    use larql_compute::cpu::ops::q4_common::quantize_q4_k;
    let wq = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.1));
    let wk = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.2));
    let wv = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.3));
    let wo = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let gate = quantize_q4_k(&synth_weight_f32(INTER * HIDDEN, 0.5));
    let up = quantize_q4_k(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down = quantize_q4_k(&synth_weight_f32(HIDDEN * INTER, 0.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();
    let mut layer = build_synth_layer(&wq, &wk, &wv, &wo, &gate, &up, &down, &norm_w);
    layer.has_post_norms = true;
    layer.post_attn_norm = &norm_w;
    layer.pre_ffn_norm = Some(&norm_w);
    layer.post_ffn_norm = Some(&norm_w);
    layer.gate = layer.gate.with_format(QuantFormat::Q4_K);
    layer.up = layer.up.with_format(QuantFormat::Q4_K);
    layer.down = layer.down.with_format(QuantFormat::Q4_K);
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

/// Decode with Q4_KF Q + Q4_KF K + Q6_K V drives the `PerProjection`
/// route in `decode/encode_qkv.rs` lines 285-334 (mixed format
/// outside the table).
#[test]
fn decode_token_with_mixed_q4kf_q6k_v_drives_per_projection() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_q4_k, quantize_q6_k};
    metal.reset_kv_cache();
    let wq = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.1));
    let wk = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.2));
    let wv = quantize_q6_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.3));
    let wo = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let gate = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.5));
    let up = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down = quantize_q4_0(&synth_weight_f32(HIDDEN * INTER, 0.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();
    let mut layer = build_synth_layer(&wq, &wk, &wv, &wo, &gate, &up, &down, &norm_w);
    // Q4_KF Q + Q4_KF K + Q6_K V — not a UniformQ4Kf, not a
    // MixedQ4kQ6kV (that pattern wants Q4_K Q + K).  Falls into the
    // PerProjection table-miss bucket.
    layer.wq = layer.wq.with_format(QuantFormat::Q4_KF);
    layer.wk = layer.wk.with_format(QuantFormat::Q4_KF);
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

/// Decode with Q8_0 QKV weights drives the fused Q8 attention path
/// in `decode/encode_qkv.rs` lines 378-411 (encode_q4_0_norm_and_qkv's
/// Q8_0 branch) and `ops/full_pipeline/stages.rs` lines 204-227.
#[test]
fn decode_token_with_q8_0_qkv_drives_fused_q8_qkv_path() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    use larql_compute::cpu::ops::q4_common::quantize_q4_0;
    metal.reset_kv_cache();

    // Q8_0 in larql's split representation: int8 rows plus a separate
    // f32 scale per 32-element block (`ScaleStorage::External(PerBlockF32)`;
    // the `q8_qkv_proj` shader indexes `Ws + row * blocks`). This fixture
    // used to build the weights as Q4_K and flip `format` to Q8_0 by
    // field assignment — leaving the fused path a scale buffer that did
    // not exist. `with_format` now refuses that retag, so the weights are
    // constructed as what they claim to be.
    let blocks = HIDDEN / larql_models::quant::ggml::LEGACY_BLOCK_ELEMS;
    let q8_rows = |rows: usize, seed: f32| -> (Vec<u8>, Vec<f32>) {
        let data: Vec<u8> = (0..rows * HIDDEN)
            .map(|i| (((i as f32) * 0.013 + seed).sin() * 90.0) as i8 as u8)
            .collect();
        let scales: Vec<f32> = (0..rows * blocks)
            .map(|i| 0.01 + (i % 7) as f32 * 0.001)
            .collect();
        (data, scales)
    };
    let (wq, wq_s) = q8_rows(Q_DIM, 0.1);
    let (wk, wk_s) = q8_rows(KV_DIM, 0.2);
    let (wv, wv_s) = q8_rows(KV_DIM, 0.3);
    let wo = quantize_q4_0(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let gate = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.5));
    let up = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down = quantize_q4_0(&synth_weight_f32(HIDDEN * INTER, 0.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();
    let mut layer = build_synth_layer(&wq, &wk, &wv, &wo, &gate, &up, &down, &norm_w);
    use larql_compute::QuantAux::ExternalScales;
    layer.wq = QuantWeight::new(QuantFormat::Q8_0, &wq, ExternalScales(&wq_s));
    layer.wk = QuantWeight::new(QuantFormat::Q8_0, &wk, ExternalScales(&wk_s));
    layer.wv = QuantWeight::new(QuantFormat::Q8_0, &wv, ExternalScales(&wv_s));
    // wo stays Q4_0 (an inline format the O projection genuinely
    // supports); only QKV needs Q8_0 for the fused path.
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
