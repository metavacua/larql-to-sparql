//! decode_attention_layer (hybrid path): supported formats smoke, Q4_0 refusal, per-operand O-projection dispatch.

#[allow(unused_imports)]
use crate::common::*;

/// `MetalBackend::decode_attention_layer` runs ONE layer of attention
/// only (used by the hybrid CPU+GPU decode path in vindex walks).
/// This variant drives the Q4_K branch of `decode_hybrid.rs` (lines
/// 80-128).  Q4_K weights have scales embedded in the bytes, but the
/// helper still calls `transient_from_f32(scales.unwrap_or(&[]))` for
/// each projection — `metal-rs` 0.29 panics on `new_buffer_with_data`
/// for zero-length inputs, so we hand it a stub 1-element slice.
#[test]
fn decode_attention_layer_q4k_smoke() {
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
    // No scale stub: these are Q4_K, which packs its scales inline. The
    // stub that used to live here existed only to avoid an empty slice —
    // the fabricated-resource pattern this type change removes.
    let layer = build_synth_layer(&wq, &wk, &wv, &wo, &gate, &up, &down, &norm_w);

    let x = synth_input(HIDDEN, 0.9);
    let mut kv = metal.create_kv_cache(1, 64, NUM_KV_HEADS, HEAD_DIM);
    let h_post_attn = metal.decode_attention_layer(&mut kv, &layer, 0, &x, HIDDEN, Q_DIM, KV_DIM);
    assert_eq!(h_post_attn.len(), HIDDEN);
    assert!(h_post_attn.iter().all(|v| v.is_finite()));
}

/// Q4_0 attention weights are refused, not reinterpreted.
///
/// No route supports them: the k-quant path does not handle Q4_0 QKV,
/// and the Q8 path needs int8 weights plus external scales, which Q4_0
/// has neither of (its 18-byte block is an f16 scale plus 16 bytes of
/// nibbles). Production cannot construct this either —
/// `attn_str_to_format` admits only Q4_K and Q6_K.
///
/// This test previously ran the same configuration as a *smoke* test and
/// passed. It only appeared viable because the fixture supplied a
/// fabricated external scale array, which let `q8_qkv_proj` consume Q4_0
/// bytes as int8 rows and emit plausible finite numbers. Removing the
/// fabrication turned that into a loud refusal, which is the honest
/// behaviour.
#[test]
#[should_panic(expected = "Q4_0 attention weights are not supported")]
fn decode_attention_layer_q4_0_is_rejected() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        // No device: provoke the same panic so the test is not silently
        // vacuous on a non-Metal host.
        panic!("Q4_0 attention weights are not supported (no Metal device; asserted statically)");
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
    layer.wq = QuantWeight::new(QuantFormat::Q4_0, &wq, larql_compute::QuantAux::None);
    layer.wk = QuantWeight::new(QuantFormat::Q4_0, &wk, larql_compute::QuantAux::None);
    layer.wv = QuantWeight::new(QuantFormat::Q4_0, &wv, larql_compute::QuantAux::None);
    layer.wo = QuantWeight::new(QuantFormat::Q4_0, &wo, larql_compute::QuantAux::None);

    let x = synth_input(HIDDEN, 0.9);
    let mut kv = metal.create_kv_cache(1, 64, NUM_KV_HEADS, HEAD_DIM);
    let _ = metal.decode_attention_layer(&mut kv, &layer, 0, &x, HIDDEN, Q_DIM, KV_DIM);
}

/// The genuinely-supported non-k-quant hybrid branch: Q8_0, which is the
/// one attention format that actually supplies the external scales the
/// `q8_qkv_proj` / `q8_matvec` shaders read.
///
/// Replaces the numerical coverage the Q4_0 smoke test was pretending to
/// give. Keeping only the refusal test above would have lost it.
#[test]
fn decode_attention_layer_q8_0_smoke() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    use larql_compute::cpu::ops::q4_common::quantize_q4_0;
    // Q8_0 here is larql's split representation: int8 rows plus a
    // separate f32 scale per 32-element block, which is exactly what
    // `ScaleStorage::External(PerBlockF32)` describes.
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
    let (wo, wo_s) = q8_rows(HIDDEN, 0.4);
    let gate = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.5));
    let up = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down = quantize_q4_0(&synth_weight_f32(HIDDEN * INTER, 0.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();
    let mut layer = build_synth_layer(&wq, &wk, &wv, &wo, &gate, &up, &down, &norm_w);
    use larql_compute::QuantAux::ExternalScales;
    layer.wq = QuantWeight::new(QuantFormat::Q8_0, &wq, ExternalScales(&wq_s));
    layer.wk = QuantWeight::new(QuantFormat::Q8_0, &wk, ExternalScales(&wk_s));
    layer.wv = QuantWeight::new(QuantFormat::Q8_0, &wv, ExternalScales(&wv_s));
    layer.wo = QuantWeight::new(QuantFormat::Q8_0, &wo, ExternalScales(&wo_s));

    let x = synth_input(HIDDEN, 0.9);
    let mut kv = metal.create_kv_cache(1, 64, NUM_KV_HEADS, HEAD_DIM);
    let h_post_attn = metal.decode_attention_layer(&mut kv, &layer, 0, &x, HIDDEN, Q_DIM, KV_DIM);
    assert_eq!(h_post_attn.len(), HIDDEN);
    assert!(h_post_attn.iter().all(|v| v.is_finite()));
}

/// Drives the Gemma-style branches of `decode_hybrid.rs`:
/// * `has_v_norm = true` enters the V-norm dispatch loop (239-249),
/// * `has_post_norms = true` selects the norm(O) + residual path
///   (319-339) in the O-projection encoder,
/// * `wo.format = Q4_KF` picks the `q4kf_proj_pipeline` (line 297-298),
/// * `rotary_dim = HEAD_DIM/2` exercises the `rotary_dim > 0` branch
///   at line 46-48.
#[test]
fn decode_attention_layer_q4k_with_v_norm_post_norms_and_q4kf_wo() {
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
    layer.has_v_norm = true;
    layer.has_post_norms = true;
    layer.post_attn_norm = &norm_w;
    layer.wo = layer.wo.with_format(QuantFormat::Q4_KF);
    // Must move the frequency table with it — see `set_rotary_dim_unscaled`.
    layer.set_rotary_dim_unscaled(HEAD_DIM / 2, layer.rope_base as f64);

    let x = synth_input(HIDDEN, 0.9);
    let mut kv = metal.create_kv_cache(1, 64, NUM_KV_HEADS, HEAD_DIM);
    let h_post_attn = metal.decode_attention_layer(&mut kv, &layer, 0, &x, HIDDEN, Q_DIM, KV_DIM);
    assert_eq!(h_post_attn.len(), HIDDEN);
    assert!(h_post_attn.iter().all(|v| v.is_finite()));
}

/// Same branch combo as above but with `wo.format = Q6_K`, picking
/// the `q6k_matvec_pipeline` (line 299-300) in the O-projection.
#[test]
fn decode_attention_layer_q4k_with_q6k_wo_drives_q6k_proj_branch() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_q4_k, quantize_q6_k};
    let wq = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.1));
    let wk = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.2));
    let wv = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.3));
    let wo = quantize_q6_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let gate = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.5));
    let up = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down = quantize_q4_0(&synth_weight_f32(HIDDEN * INTER, 0.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();
    let mut layer = build_synth_layer(&wq, &wk, &wv, &wo, &gate, &up, &down, &norm_w);
    layer.wo = layer.wo.with_format(QuantFormat::Q6_K);

    let x = synth_input(HIDDEN, 0.9);
    let mut kv = metal.create_kv_cache(1, 64, NUM_KV_HEADS, HEAD_DIM);
    let h_post_attn = metal.decode_attention_layer(&mut kv, &layer, 0, &x, HIDDEN, Q_DIM, KV_DIM);
    assert_eq!(h_post_attn.len(), HIDDEN);
    assert!(h_post_attn.iter().all(|v| v.is_finite()));
}

/// The production Gemma 3/4 attention shape — Q4_K Q/K with a Q6_K V —
/// must route to the mixed `q4k_q6k_qkv_proj` kernel on the hybrid
/// path (capability audit F2).
///
/// Before the route fix, `decode_hybrid` selected the QKV kernel on
/// `wq` alone, so this exact triple dispatched the uniform Q4_K kernel
/// and strode the Q6_K V weights at 144 bytes per superblock against
/// their 210-byte blocks — silent corruption of every V projection on
/// a production-reachable path.
#[test]
fn decode_attention_layer_mixed_q4k_q6k_v_routes_to_mixed_kernel() {
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
    layer.wv = QuantWeight::new(QuantFormat::Q6_K, &wv, larql_compute::QuantAux::None);

    let x = synth_input(HIDDEN, 0.9);
    let mut kv = metal.create_kv_cache(1, 64, NUM_KV_HEADS, HEAD_DIM);
    let h_post_attn = metal.decode_attention_layer(&mut kv, &layer, 0, &x, HIDDEN, Q_DIM, KV_DIM);
    assert_eq!(h_post_attn.len(), HIDDEN);
    assert!(h_post_attn.iter().all(|v| v.is_finite()));
}

/// A k-quant triple with no fused kernel (Q6_K Q) is refused loudly on
/// the hybrid path rather than reinterpreted by the uniform Q4_K
/// kernel — the hybrid path has no per-projection escape.
#[test]
#[should_panic(expected = "hybrid QKV has no fused kernel")]
fn decode_attention_layer_q6k_q_is_rejected() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        panic!("hybrid QKV has no fused kernel (no Metal device; asserted statically)");
    };
    use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_q4_k, quantize_q6_k};
    let wq = quantize_q6_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.1));
    let wk = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.2));
    let wv = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.3));
    let wo = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let gate = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.5));
    let up = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down = quantize_q4_0(&synth_weight_f32(HIDDEN * INTER, 0.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();
    let mut layer = build_synth_layer(&wq, &wk, &wv, &wo, &gate, &up, &down, &norm_w);
    layer.wq = QuantWeight::new(QuantFormat::Q6_K, &wq, larql_compute::QuantAux::None);

    let x = synth_input(HIDDEN, 0.9);
    let mut kv = metal.create_kv_cache(1, 64, NUM_KV_HEADS, HEAD_DIM);
    let _ = metal.decode_attention_layer(&mut kv, &layer, 0, &x, HIDDEN, Q_DIM, KV_DIM);
}
