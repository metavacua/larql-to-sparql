//! MoE decode paths: local fallback, dumps, remote FFN, interleave and split modes.

#[allow(unused_imports)]
use crate::common::*;

/// MoE layer with NO moe_fn callback — drives the local
/// `cpu_moe_forward` fallback path in `decode/moe_interleave.rs`
/// (lines 161-166).
#[test]
fn decode_token_with_moe_layer_no_callback_drives_local_fallback() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_q4_k};
    use larql_compute::{
        Activation, MoeLayerWeights, MoeRoutingPolicy, MoeWeightLayout, QuantFormat,
    };

    let wq = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.1));
    let wk = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.2));
    let wv = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.3));
    let wo = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let gate = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.5));
    let up = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down = quantize_q4_0(&synth_weight_f32(HIDDEN * INTER, 0.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();
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
    let mut kv = metal.create_kv_cache(1, 64, NUM_KV_HEADS, HEAD_DIM);
    // Plain decode_token → no moe_fn → local cpu_moe_forward fallback.
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

/// MoE layer + `LARQL_DUMP_L0` env — drives the
/// `moe_interleave::dump_l0_moe_intermediates` path
/// (`decode/moe_interleave.rs` lines 199-213).
#[test]
fn decode_token_with_moe_and_dump_l0_drives_moe_intermediate_dump() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_q4_k};
    use larql_compute::{
        Activation, MoeLayerWeights, MoeRoutingPolicy, MoeWeightLayout, QuantFormat,
    };

    let tmp = std::env::temp_dir().join("larql-cm-moe-dump-l0-test");
    let _ = std::fs::create_dir_all(&tmp);
    let path = tmp.to_str().unwrap().to_string();
    let path_static: &'static str = Box::leak(path.into_boxed_str());
    let saved = std::env::var_os("LARQL_DUMP_L0");
    unsafe {
        std::env::set_var("LARQL_DUMP_L0", path_static);
    }

    let wq = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.1));
    let wk = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.2));
    let wv = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.3));
    let wo = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let gate = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.5));
    let up = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down = quantize_q4_0(&synth_weight_f32(HIDDEN * INTER, 0.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();
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

    unsafe {
        match saved {
            Some(v) => std::env::set_var("LARQL_DUMP_L0", v),
            None => std::env::remove_var("LARQL_DUMP_L0"),
        }
    }
    let _ = std::fs::remove_dir_all(&tmp);
}

/// MoE layer + `LARQL_DUMP_RESIDUALS` env on a 2-layer model —
/// drives the residual_dump record_layer call in
/// `moe_interleave.rs` lines 218-223 + next-layer-cmd-reset at
/// 226-228.
#[test]
fn decode_token_with_moe_and_dump_residuals_drives_record_layer() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_q4_k};
    use larql_compute::{
        Activation, MoeLayerWeights, MoeRoutingPolicy, MoeWeightLayout, QuantFormat,
    };

    let tmp = std::env::temp_dir().join("larql-cm-moe-residual-dump.bin");
    let path = tmp.to_str().unwrap().to_string();
    let path_static: &'static str = Box::leak(path.into_boxed_str());
    let saved = std::env::var_os("LARQL_DUMP_RESIDUALS");
    unsafe {
        std::env::set_var("LARQL_DUMP_RESIDUALS", path_static);
    }

    let wq = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.1));
    let wk = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.2));
    let wv = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.3));
    let wo = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let gate = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.5));
    let up = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down = quantize_q4_0(&synth_weight_f32(HIDDEN * INTER, 0.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();
    let make_null_moe = || MoeLayerWeights {
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
    let mut layer0 = build_synth_layer(&wq, &wk, &wv, &wo, &gate, &up, &down, &norm_w);
    layer0.moe = Some(make_null_moe());
    let mut layer1 = build_synth_layer(&wq, &wk, &wv, &wo, &gate, &up, &down, &norm_w);
    layer1.moe = Some(make_null_moe());

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

    unsafe {
        match saved {
            Some(v) => std::env::set_var("LARQL_DUMP_RESIDUALS", v),
            None => std::env::remove_var("LARQL_DUMP_RESIDUALS"),
        }
    }
    let _ = std::fs::remove_file(&tmp);
}

/// MoE layer with `ffn_is_remote = true` — drives the remote-FFN
/// branch in `moe_interleave.rs` lines 180-187 + the same line at
/// the dense-encode skip in `decode/mod.rs`.
#[test]
fn decode_token_with_moe_remote_ffn_drives_remote_path() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    use larql_compute::backend::DecodeBackend;
    use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_q4_k};
    use larql_compute::{
        Activation, MoeLayerWeights, MoeRoutingPolicy, MoeWeightLayout, QuantFormat,
    };

    let wq = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.1));
    let wk = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.2));
    let wv = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.3));
    let wo = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let gate = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.5));
    let up = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down = quantize_q4_0(&synth_weight_f32(HIDDEN * INTER, 0.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();
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
    layer.ffn_is_remote = true;

    let x = synth_input(HIDDEN, 0.9);
    let mut moe_fn = |_l: usize, h: &[f32]| -> Vec<f32> { vec![0.0f32; h.len()] };
    let out = metal.decode_token_with_moe(&[layer], &x, HIDDEN, INTER, &mut moe_fn);
    assert_eq!(out.expect("decode returns Some").len(), HIDDEN);
}

/// MoE-interleave decode: layer with `moe.is_some()` + a moe_fn
/// callback drives the CPU-side MoE interleave block
/// (`decode/mod.rs` lines 550-589 + `handle_moe_interleave`).
#[test]
fn decode_token_with_moe_fn_drives_interleave_path() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    use larql_compute::backend::DecodeBackend;
    use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_q4_k};
    use larql_compute::{
        Activation, MoeLayerWeights, MoeRoutingPolicy, MoeWeightLayout, QuantFormat,
    };

    let wq = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.1));
    let wk = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.2));
    let wv = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.3));
    let wo = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let gate = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.5));
    let up = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down = quantize_q4_0(&synth_weight_f32(HIDDEN * INTER, 0.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();
    // Null MoE: num_experts=0 makes cpu_moe_forward bail before doing
    // expert work, but the moe_fn callback is still invoked which is
    // what we need for coverage.
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
    let mut moe_call_count = 0usize;
    let mut moe_fn = |_l: usize, h: &[f32]| -> Vec<f32> {
        moe_call_count += 1;
        vec![0.0f32; h.len()]
    };
    let out = metal.decode_token_with_moe(&[layer], &x, HIDDEN, INTER, &mut moe_fn);
    assert_eq!(out.expect("decode returns Some").len(), HIDDEN);
    assert_eq!(
        moe_call_count, 1,
        "moe_fn must be called once per MoE layer"
    );
}

/// MoE split-fire variant — both `moe_fn` (fire) and `moe_collect_fn`
/// callbacks drive the split-mode path (`split_mode = true` at
/// `decode/mod.rs:254`).
#[test]
fn decode_token_with_moe_split_fn_drives_split_mode_path() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    use larql_compute::backend::DecodeBackend;
    use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_q4_k};
    use larql_compute::{
        Activation, MoeLayerWeights, MoeRoutingPolicy, MoeWeightLayout, QuantFormat,
    };

    let wq = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.1));
    let wk = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.2));
    let wv = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.3));
    let wo = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let gate = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.5));
    let up = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down = quantize_q4_0(&synth_weight_f32(HIDDEN * INTER, 0.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();
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
    let mut fired = 0usize;
    let mut collected = 0usize;
    let mut moe_fire = |_l: usize, _h: &[f32]| {
        fired += 1;
    };
    let mut moe_collect = |_l: usize| -> Vec<f32> {
        collected += 1;
        vec![0.0f32; HIDDEN]
    };
    let out = metal.decode_token_with_moe_split(
        &[layer],
        &x,
        HIDDEN,
        INTER,
        &mut moe_fire,
        &mut moe_collect,
    );
    assert_eq!(out.expect("decode returns Some").len(), HIDDEN);
    assert_eq!(fired, 1);
    assert_eq!(collected, 1);
}

/// PURE-MoE layer: `moe` present and NO dense FFN weights extracted (the
/// GPT-OSS shape). The dense branch must be skipped — encoding it over
/// empty slices poisoned `new_h` before the expert add — and the combine
/// must be `new_h = h_post_attn + moe_out` exactly. Pinned by the
/// difference between two moe_fn arms: constant-c minus zeros must be c
/// in every element, and both outputs finite.
#[test]
fn pure_moe_layer_skips_dense_ffn_and_adds_expert_output_directly() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    use larql_compute::cpu::ops::q4_common::quantize_q4_k;
    use larql_compute::{
        Activation, MoeLayerWeights, MoeRoutingPolicy, MoeWeightLayout, QuantFormat,
    };

    let wq = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.1));
    let wk = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.2));
    let wv = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.3));
    let wo = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();
    let null_moe = || MoeLayerWeights {
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

    const EXPERT_CONST: f32 = 0.75;
    let mut outs: Vec<Vec<f32>> = Vec::new();
    for c in [0.0f32, EXPERT_CONST] {
        // Empty dense weights: `FullPipelineLayer::default()` quant slices
        // are zero-length — `has_dense_ffn()` is false by construction.
        let mut layer = larql_compute::FullPipelineLayer {
            wq: larql_compute::QuantWeight::new(
                QuantFormat::Q4_K,
                &wq,
                larql_compute::QuantAux::None,
            ),
            wk: larql_compute::QuantWeight::new(
                QuantFormat::Q4_K,
                &wk,
                larql_compute::QuantAux::None,
            ),
            wv: larql_compute::QuantWeight::new(
                QuantFormat::Q4_K,
                &wv,
                larql_compute::QuantAux::None,
            ),
            wo: larql_compute::QuantWeight::new(
                QuantFormat::Q4_K,
                &wo,
                larql_compute::QuantAux::None,
            ),
            input_norm: &norm_w,
            post_attn_norm: &norm_w,
            attn_scale: 1.0 / (HEAD_DIM as f32).sqrt(),
            head_dim: HEAD_DIM,
            num_q_heads: NUM_Q_HEADS,
            num_kv_heads: NUM_KV_HEADS,
            rope_freq: larql_compute::attention::rope::RopeFreqPlan::unscaled(
                HEAD_DIM,
                0_usize,
                10_000.0_f64,
            ),
            ..larql_compute::FullPipelineLayer::default()
        };
        layer.moe = Some(null_moe());
        assert!(!layer.has_dense_ffn(), "fixture must be the pure-MoE shape");

        let x = synth_input(HIDDEN, 0.9);
        let mut kv = metal.create_kv_cache(1, 64, NUM_KV_HEADS, HEAD_DIM);
        let mut moe_fn = move |_l: usize, h: &[f32]| -> Vec<f32> { vec![c; h.len()] };
        let out = metal.decode_token_with_moe_fn(
            &mut kv,
            std::slice::from_ref(&layer),
            &x,
            HIDDEN,
            INTER,
            Q_DIM,
            KV_DIM,
            NUM_Q_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            10_000.0,
            Some(&mut moe_fn),
        );
        assert!(
            out.iter().all(|v| v.is_finite()),
            "pure-MoE decode produced non-finite output (dense branch ran on empty weights?)"
        );
        outs.push(out);
    }
    for (i, (a, b)) in outs[0].iter().zip(outs[1].iter()).enumerate() {
        let delta = b - a;
        assert!(
            (delta - EXPERT_CONST).abs() < 1e-4,
            "combine is not h_post_attn + moe_out at element {i}: delta={delta}"
        );
    }
}
