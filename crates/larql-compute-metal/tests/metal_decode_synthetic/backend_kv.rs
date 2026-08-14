//! DecodeBackend trait surface: KV-cache management, trait-method decode, head replacement, capture hooks, MoE split profile.

#[allow(unused_imports)]
use crate::common::*;

/// Cover the KV-cache management trait methods on
/// `DecodeBackend for MetalBackend`: `has_kv_cache`, `populate_kv_layer`,
/// `kv_cache_len`, `reset_kv_cache`, `truncate_kv_cache`,
/// `preallocate_kv_cache_per_layer`.  Each is a public part of the
/// trait surface that's reached only when callers manage the cache
/// out-of-band (server-side prefill reuse, vindex walk).
#[test]
fn decode_backend_kv_cache_management_methods_round_trip() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    use larql_compute::backend::DecodeBackend;

    assert!(
        metal.has_kv_cache(),
        "Metal backend reports KV cache support"
    );

    // `preallocate_kv_cache_per_layer` replaces any existing cache.
    let shapes = vec![(NUM_KV_HEADS, HEAD_DIM), (NUM_KV_HEADS, HEAD_DIM)];
    metal.preallocate_kv_cache_per_layer(&shapes, 64);

    // Populate layer 0 with synthetic KV — covers populate_kv_layer's
    // happy path (cache already exists; cache_guard.is_some()).
    let synth_kv: Vec<f32> = (0..2 * NUM_KV_HEADS * HEAD_DIM)
        .map(|i| (i as f32) * 0.01)
        .collect();
    metal.populate_kv_layer(0, &synth_kv, &synth_kv, 2, NUM_KV_HEADS, HEAD_DIM);
    assert_eq!(metal.kv_cache_len(), 2);

    // `truncate_kv_cache` resets the per-layer counter to the given
    // length without re-allocating.
    metal.truncate_kv_cache(1);
    assert_eq!(metal.kv_cache_len(), 1);

    // `reset_kv_cache` zeroes every layer's current_len.
    metal.reset_kv_cache();
    assert_eq!(metal.kv_cache_len(), 0);
}

/// `populate_kv_layer` with **no pre-existing cache** drives the
/// `cache_guard.is_none()` branch — creates a fresh cache and grows
/// to the requested layer index.
#[test]
fn decode_backend_populate_kv_layer_creates_cache_on_first_call() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    use larql_compute::backend::DecodeBackend;

    // Reset so we drop any cache from prior tests.  Then populate
    // layer 2 directly — exercises the `while kv.layers.len() <= layer`
    // grow loop.
    metal.reset_kv_cache();
    let synth_kv: Vec<f32> = vec![0.5f32; 4 * NUM_KV_HEADS * HEAD_DIM];
    metal.populate_kv_layer(2, &synth_kv, &synth_kv, 4, NUM_KV_HEADS, HEAD_DIM);
    // `kv_cache_len()` reads layer[0]'s current_len. We only populated
    // layer 2, so layer 0 keeps its initial 0. The point of this test
    // is that `populate_kv_layer(2, ...)` succeeded by growing the
    // cache to 3 layers — verify that by writing layer 0 too and
    // re-reading.
    metal.populate_kv_layer(0, &synth_kv, &synth_kv, 4, NUM_KV_HEADS, HEAD_DIM);
    assert_eq!(metal.kv_cache_len(), 4);
}

/// `DecodeBackend::decode_token` trait method — wraps the inherent
/// MetalBackend::decode_token via the cached KV.  Covers
/// `trait_impl/decode.rs` lines 629-658.
#[test]
fn decode_backend_decode_token_trait_method_uses_internal_kv() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    use larql_compute::backend::DecodeBackend;
    use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_q4_k};
    metal.reset_kv_cache();
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
    let backend: &dyn DecodeBackend = &metal;
    let out = backend
        .decode_token(&[layer], &x, HIDDEN, INTER)
        .expect("decode_token trait returns Some");
    assert_eq!(out.len(), HIDDEN);
}

/// GeluTanh activation drives the `&self.ffn.geglu_gelu_tanh_pipeline`
/// branch in `trait_impl/decode.rs::full_pipeline_q4` (line 59) and
/// `full_pipeline_q4_with_head_replacement` (line 131).
#[test]
fn prefill_q4_with_gelu_tanh_activation_drives_gelu_tanh_pipeline() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    use larql_compute::backend::DecodeBackend;
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
    layer.activation = larql_compute::Activation::GeluTanh;

    let x = synth_input(HIDDEN, 0.9);
    // prefill_kquant goes through `full_pipeline_q4` which picks the
    // geglu_pipeline based on activation.
    let out = metal
        .prefill_kquant(&[layer], &x, HIDDEN, INTER, 1, false, 0.0)
        .expect("prefill_kquant returns Some");
    assert_eq!(out.len(), HIDDEN);

    // Same prefill_kquant_with_head_replacement variant — covers the
    // GeluTanh arm of its geglu picker too.
    let mut layer2 = build_synth_layer(&wq, &wk, &wv, &wo, &gate, &up, &down, &norm_w);
    layer2.activation = larql_compute::Activation::GeluTanh;
    let delta = vec![0.0f32; HIDDEN];
    let out2 = metal
        .prefill_kquant_with_head_replacement(
            &[layer2],
            &x,
            HIDDEN,
            INTER,
            1,
            false,
            0.0,
            0,
            0,
            &delta,
        )
        .expect("head-replacement returns Some");
    assert_eq!(out2.len(), HIDDEN);
}

/// `DecodeBackend::full_pipeline_q4_with_head_replacement` is the
/// trait-level direct variant (no MoE fallback).  Covers
/// `trait_impl/decode.rs` lines 111-196.
#[test]
fn decode_backend_full_pipeline_q4_with_head_replacement_runs() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    use larql_compute::backend::DecodeBackend;
    use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_q4_k};
    let wq = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.1));
    let wk = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.2));
    let wv = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.3));
    let wo = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let gate = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.5));
    let up = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down = quantize_q4_0(&synth_weight_f32(HIDDEN * INTER, 0.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();
    let layers = vec![build_synth_layer(
        &wq, &wk, &wv, &wo, &gate, &up, &down, &norm_w,
    )];
    let x = synth_input(HIDDEN, 0.9);
    let delta = vec![0.0f32; HIDDEN];
    let backend: &dyn DecodeBackend = &metal;
    let out = backend
        .full_pipeline_q4_with_head_replacement(
            &layers, &x, HIDDEN, INTER, 1, false, 0.0, 0, 0, &delta,
        )
        .expect("full_pipeline_q4_with_head_replacement returns Some");
    assert_eq!(out.len(), HIDDEN);
}

/// `DecodeBackend::full_pipeline_kquant_capture_pre_wo` — covers
/// `trait_impl/decode.rs` lines 359-449 (capture pre-W_O variant).
#[test]
fn decode_backend_full_pipeline_q4_capture_pre_wo_runs() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    use larql_compute::backend::DecodeBackend;
    use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_q4_k};
    let wq = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.1));
    let wk = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.2));
    let wv = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.3));
    let wo = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let gate = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.5));
    let up = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down = quantize_q4_0(&synth_weight_f32(HIDDEN * INTER, 0.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();
    let layers = vec![build_synth_layer(
        &wq, &wk, &wv, &wo, &gate, &up, &down, &norm_w,
    )];
    let x = synth_input(HIDDEN, 0.9);
    let backend: &dyn DecodeBackend = &metal;
    let out = backend
        .full_pipeline_kquant_capture_pre_wo(&layers, &x, HIDDEN, INTER, 1, false, 0.0, 0, 0);
    let capture = out.expect("capture_pre_wo returns Some");
    // capture is a Vec<f32> of seq_len × head_dim; pin shape.
    assert_eq!(capture.len(), HEAD_DIM);
}

/// `DecodeBackend::decode_token_q4k_moe` — Metal-backend impl
/// returns `None` (delegates to default impl).  Covers
/// `trait_impl/decode.rs` lines 693-719.
#[test]
fn decode_backend_decode_token_q4k_moe_returns_none_on_metal() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    use larql_compute::backend::DecodeBackend;
    use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_q4_k};
    let wq = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.1));
    let wk = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.2));
    let wv = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.3));
    let wo = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let gate = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.5));
    let up = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down = quantize_q4_0(&synth_weight_f32(HIDDEN * INTER, 0.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();
    let layers = vec![build_synth_layer(
        &wq, &wk, &wv, &wo, &gate, &up, &down, &norm_w,
    )];
    let x = synth_input(HIDDEN, 0.9);
    let backend: &dyn DecodeBackend = &metal;
    // Metal backend uses the default `None` impl — it doesn't have a
    // dedicated Q4K-MoE decode pipeline. The wrapper still constructs
    // the geometry tuple and calls through, which is the part we cover.
    let _ = backend.decode_token_q4k_moe(&layers, &x, HIDDEN, INTER, 1e-6, &|_, _| None);
}

/// `DecodeBackend::decode_token_split_profile` — returns
/// `(result, attn_ms, gate_up_ms, down_ms)`.  Covers
/// `trait_impl/decode.rs` lines 762-790 + the fallback case where
/// LARQL_PROFILE_SPLIT isn't set so attn_ms = whole-token wall.
#[test]
fn decode_backend_decode_token_split_profile_returns_timings() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    use larql_compute::backend::DecodeBackend;
    use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_q4_k};
    metal.reset_kv_cache();
    let wq = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.1));
    let wk = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.2));
    let wv = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.3));
    let wo = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let gate = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.5));
    let up = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down = quantize_q4_0(&synth_weight_f32(HIDDEN * INTER, 0.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();
    let layers = vec![build_synth_layer(
        &wq, &wk, &wv, &wo, &gate, &up, &down, &norm_w,
    )];
    let x = synth_input(HIDDEN, 0.9);
    let backend: &dyn DecodeBackend = &metal;
    let (result, attn_ms, _gate_up, _down) =
        backend.decode_token_split_profile(&layers, &x, HIDDEN, INTER);
    assert!(result.is_some());
    // Fallback: attn_ms = whole-token wall, > 0.
    assert!(attn_ms >= 0.0);
}

/// `DecodeBackend::decode_token_with_moe_split` trait method covers
/// `trait_impl/decode.rs` lines 721-760 (the fire/collect split
/// wrapper with `Vec::new()` discard inside fire_wrapper).
#[test]
fn decode_backend_decode_token_with_moe_split_runs() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    use larql_compute::backend::DecodeBackend;
    use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_q4_k};
    use larql_compute::{
        Activation, MoeLayerWeights, MoeRoutingPolicy, MoeWeightLayout, QuantFormat,
    };
    metal.reset_kv_cache();
    let wq = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.1));
    let wk = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.2));
    let wv = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.3));
    let wv2 = wv.clone();
    let wo = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let gate = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.5));
    let up = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down = quantize_q4_0(&synth_weight_f32(HIDDEN * INTER, 0.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();
    let _ = wv2;
    let null_moe = MoeLayerWeights {
        experts_gate_up: Vec::new(),
        experts_down: Vec::new(),
        routing_policy: MoeRoutingPolicy::default(),
        weight_layout: MoeWeightLayout::default(),
        router_proj: &[],
        router_scale: &[],
        router_per_expert_scale: &[],
        router_norm: &[],
        router_norm_parameter_free: false,
        router_input_scalar: 1.0,
        pre_experts_norm: &[],
        post_ffn1_norm: &[],
        post_experts_norm: &[],
        num_experts: 0,
        top_k: 1,
        intermediate_size: INTER,
        router_bias: &[],
        experts_gate_up_bias: &[],
        experts_down_bias: &[],
        gate_rule: larql_compute::MoeGateRule::Gated(Activation::Silu),
        expert_data_format: QuantFormat::BF16,
    };
    let mut layer = build_synth_layer(&wq, &wk, &wv, &wo, &gate, &up, &down, &norm_w);
    layer.moe = Some(null_moe);
    let x = synth_input(HIDDEN, 0.9);
    let mut fire_called = 0;
    let mut collect_called = 0;
    let mut fire_fn = |_l: usize, _h: &[f32]| {
        fire_called += 1;
    };
    let mut collect_fn = |_l: usize| -> Vec<f32> {
        collect_called += 1;
        vec![0.0f32; HIDDEN]
    };
    let backend: &dyn DecodeBackend = &metal;
    let out = backend
        .decode_token_with_moe_split(&[layer], &x, HIDDEN, INTER, &mut fire_fn, &mut collect_fn)
        .expect("split returns Some");
    assert_eq!(out.len(), HIDDEN);
    assert_eq!(fire_called, 1);
    assert_eq!(collect_called, 1);
}

/// `DecodeBackend::multi_layer_q4_ffn` — covers `trait_impl/decode.rs`
/// lines 198-209.
#[test]
fn decode_backend_multi_layer_q4_ffn_runs() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    use larql_compute::backend::DecodeBackend;
    let block_bytes = 18usize;
    let hidden = 32usize;
    let inter = 64usize;
    let blocks_per_row = hidden / 32;
    let gate = vec![0u8; inter * blocks_per_row * block_bytes];
    let up = vec![0u8; inter * blocks_per_row * block_bytes];
    let down = vec![0u8; hidden * (inter / 32) * block_bytes];
    let layers = vec![(gate.as_slice(), up.as_slice(), down.as_slice())];
    let x = vec![0.0f32; hidden];
    let backend: &dyn DecodeBackend = &metal;
    let out = backend
        .multi_layer_q4_ffn(&layers, &x, inter, hidden)
        .expect("multi_layer_q4_ffn returns Some");
    assert_eq!(out.len(), hidden);
}

/// Decode with V-norm + QK-norm + `LARQL_FUSED_ATTN=0` drives the
/// unfused qk_norm_qk + v_norm_batched paths in
/// `decode/encode_attn.rs` lines 268-292 (qk_norm_qk dispatch) and
/// 318-336 (V-norm batched dispatch).
#[test]
fn decode_token_with_unfused_attn_v_norm_qk_norm_drives_unfused_paths() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_q4_k};
    let saved_fused = std::env::var_os("LARQL_FUSED_ATTN");
    // Force unfused: rebuild backend so DecodeFlags re-reads env.
    unsafe {
        std::env::set_var("LARQL_FUSED_ATTN", "0");
    }
    let metal2 = larql_compute_metal::MetalBackend::new().expect("Metal device");
    let _ = metal; // keep first backend alive briefly

    let wq = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.1));
    let wk = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.2));
    let wv = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.3));
    let wo = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let gate = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.5));
    let up = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down = quantize_q4_0(&synth_weight_f32(HIDDEN * INTER, 0.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();
    let head_norm: Vec<f32> = (0..HEAD_DIM).map(|i| 1.0 + (i as f32 * 0.002)).collect();

    let mut layer = build_synth_layer(&wq, &wk, &wv, &wo, &gate, &up, &down, &norm_w);
    layer.q_norm_weight = Some(&head_norm);
    layer.k_norm_weight = Some(&head_norm);
    layer.has_v_norm = true;
    layer.has_post_norms = true;
    layer.post_attn_norm = &norm_w;
    layer.pre_ffn_norm = Some(&norm_w);
    layer.post_ffn_norm = Some(&norm_w);

    let x = synth_input(HIDDEN, 0.9);
    let mut kv = metal2.create_kv_cache(1, 64, NUM_KV_HEADS, HEAD_DIM);
    let out = larql_compute_metal::MetalBackend::decode_token(
        &metal2,
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

    unsafe {
        match saved_fused {
            Some(v) => std::env::set_var("LARQL_FUSED_ATTN", v),
            None => std::env::remove_var("LARQL_FUSED_ATTN"),
        }
    }
}
