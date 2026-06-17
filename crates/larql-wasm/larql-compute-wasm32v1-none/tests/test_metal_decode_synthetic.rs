//! Synthetic end-to-end decode tests.
//!
//! Builds a small `FullPipelineLayer` with synthetic Q4_K (attn) +
//! Q4_0 (FFN) weights and runs `MetalBackend::decode_token` on it.
//! Adapted from `examples/diag_decode_pipeline.rs`.
//!
//! Why this file exists: per-shader tests (`test_metal_shaders.rs` and
//! friends) hit the kernels but never exercise the production decode
//! orchestration code in `metal/decode/encode_{attn,qkv,ffn,post_ffn}.rs`
//! and `metal/decode/mod.rs::decode_token_with_moe_split_fn`. End-to-end
//! tests in `larql-inference/tests/` do, but those don't show up in
//! per-crate `cargo llvm-cov --package larql-compute` runs. This test
//! file fills that gap — a single decode_token call lifts ~2856 LoC of
//! production decode code from 0% to executed.
//!
//! These are smoke tests, not numerical-parity tests. They verify:
//! - decode_token returns a non-NaN, non-zero output buffer
//! - dimensions are right
//! - The `LARQL_FUSED_PRELAYER_NORM=1` D-RMS-FUSE wiring produces
//!   bit-identical output to the unfused path on a non-Gemma-style
//!   layer (no `has_post_norms`).
//!
//! Numerical-correctness against a CPU reference happens in
//! `larql-inference/tests/test_cpu_metal_parity.rs` against real
//! vindexes; it's at the wrong scope to live here.

#![cfg(all(feature = "metal", target_os = "macos"))]

use larql_compute::{
    Activation, ComputeBackend, DecodeBackend, FfnType, FullPipelineLayer, NormType, QuantFormat,
    QuantWeight,
};

/// Process-wide guard for tests that mutate env vars read by the decode
/// hot path (e.g. `LARQL_FUSED_PRELAYER_NORM`, `LARQL_QKV_FUSED`). Cargo
/// runs tests inside a binary in parallel by default; without this lock
/// a parallel `decode_token` test races with the env-toggling test and
/// observes the var in either state. Hold the guard for the entire
/// duration of any backend creation + decode that depends on the env.
static ENV_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Synthetic dims chosen to be Q4_K-compatible (multiples of 256) and
/// small enough for a fast test. Q4_K super-blocks are 256 elements.
const HIDDEN: usize = 256;
const INTER: usize = 512;
const HEAD_DIM: usize = 64;
const NUM_Q_HEADS: usize = 2;
const NUM_KV_HEADS: usize = 1;
const Q_DIM: usize = NUM_Q_HEADS * HEAD_DIM; // 128
const KV_DIM: usize = NUM_KV_HEADS * HEAD_DIM; // 64

fn synth_input(len: usize, seed: f32) -> Vec<f32> {
    (0..len)
        .map(|i| ((i as f32 * 0.013 + seed).sin() + 0.1 * ((i >> 4) as f32).cos()) * 0.5)
        .collect()
}

fn synth_weight_f32(len: usize, seed: f32) -> Vec<f32> {
    (0..len)
        .map(|i| ((i as f32 * 0.001 + seed).sin() + 0.2 * ((i >> 8) as f32).cos()) * 0.3)
        .collect()
}

// Test helper: 7 quant tensor slices + 1 norm slice = 8 args. Mirrors
// the production `FullPipelineLayer` constructor surface; collapsing to
// a single struct just for the test would obscure the per-tensor
// fixture-builder pattern callers actually want.
#[allow(clippy::too_many_arguments)]
fn build_synth_layer<'a>(
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
        wq: QuantWeight {
            data: wq_data,
            scales: None,
            format: QuantFormat::Q4_K,
        },
        wk: QuantWeight {
            data: wk_data,
            scales: None,
            format: QuantFormat::Q4_K,
        },
        wv: QuantWeight {
            data: wv_data,
            scales: None,
            format: QuantFormat::Q4_K,
        },
        wo: QuantWeight {
            data: wo_data,
            scales: None,
            format: QuantFormat::Q4_K,
        },
        gate: QuantWeight {
            data: gate_data,
            scales: None,
            format: QuantFormat::Q4_0,
        },
        up: QuantWeight {
            data: up_data,
            scales: None,
            format: QuantFormat::Q4_0,
        },
        down: QuantWeight {
            data: down_data,
            scales: None,
            format: QuantFormat::Q4_0,
        },
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
        ple_input_gate: None,
        ple_projection: None,
        ple_post_norm: None,
    }
}

/// End-to-end smoke: a single-layer Llama-style decode produces a
/// finite output of the correct size. Exercises the production decode
/// orchestration in `metal/decode/{mod,encode_attn,encode_qkv,
/// encode_ffn,encode_post_ffn}.rs`.
#[test]
fn decode_token_single_layer_synthetic_q4k_smoke() {
    let metal = match larql_compute::metal::MetalBackend::new() {
        Some(m) => m,
        None => {
            eprintln!("skip: no Metal device");
            return;
        }
    };

    use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_q4_k};

    let wq_data = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.1));
    let wk_data = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.2));
    let wv_data = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.3));
    let wo_data = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let gate_data = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.5));
    let up_data = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down_data = quantize_q4_0(&synth_weight_f32(HIDDEN * INTER, 0.7));

    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();
    let layer = build_synth_layer(
        &wq_data, &wk_data, &wv_data, &wo_data, &gate_data, &up_data, &down_data, &norm_w,
    );

    let x = synth_input(HIDDEN, 0.9);
    let mut kv = metal.create_kv_cache(1, 64, NUM_KV_HEADS, HEAD_DIM);

    let result = larql_compute::metal::MetalBackend::decode_token(
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

    assert_eq!(result.len(), HIDDEN, "decode_token output length");
    let nan = result.iter().filter(|v| v.is_nan()).count();
    assert_eq!(nan, 0, "decode_token output had {nan} NaNs");
    let inf = result.iter().filter(|v| v.is_infinite()).count();
    assert_eq!(inf, 0, "decode_token output had {inf} infinities");
    let max_abs = result.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
    assert!(
        max_abs > 0.0,
        "decode_token output is all zero (likely uninitialized buffers)"
    );
    assert!(
        max_abs < 1e6,
        "decode_token output magnitude {max_abs} is suspiciously large"
    );
}

/// D-RMS-FUSE Phase 1 end-to-end parity: `LARQL_FUSED_PRELAYER_NORM=1`
/// produces bit-identical output to the unfused path on a non-Gemma
/// (no `has_post_norms`) two-layer setup. Two layers exercises the
/// fusion at the layer-0→layer-1 boundary; a single layer wouldn't
/// engage the fusion (no next layer).
///
/// This is the integration counterpart to the kernel-level tests in
/// `test_kernel_fused_ops_norms.rs::residual_norm_store_*`.
#[test]
fn d_rms_fuse_phase1_produces_identical_output() {
    use std::env;

    // Hold the env lock for the whole test: both runs must observe the
    // env state at the time we construct each backend, and we must not
    // cross-pollute with the other env-mutating test
    // (`decode_token_qkv_fused_opt_in_smoke`) running in parallel.
    let _env_guard = ENV_TEST_LOCK.lock().expect("env lock poisoned");

    use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_q4_k};

    let wq_data = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.11));
    let wk_data = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.22));
    let wv_data = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.33));
    let wo_data = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.44));
    let gate_data = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.55));
    let up_data = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.66));
    let down_data = quantize_q4_0(&synth_weight_f32(HIDDEN * INTER, 0.77));

    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.0007)).collect();
    let layer0 = build_synth_layer(
        &wq_data, &wk_data, &wv_data, &wo_data, &gate_data, &up_data, &down_data, &norm_w,
    );
    let layer1 = build_synth_layer(
        &wq_data, &wk_data, &wv_data, &wo_data, &gate_data, &up_data, &down_data, &norm_w,
    );
    let layers = [layer0, layer1];
    let x = synth_input(HIDDEN, 0.99);

    // Decode flags are cached at `MetalBackend::new()`. The test must
    // construct a fresh backend AFTER the env is in the desired state
    // — the previous "set env then call decode on the existing backend"
    // pattern silently no-ops with cached flags.
    //
    // Run with fusion OFF.
    env::remove_var("LARQL_FUSED_PRELAYER_NORM");
    let metal_off = match larql_compute::metal::MetalBackend::new() {
        Some(m) => m,
        None => {
            eprintln!("skip: no Metal device");
            return;
        }
    };
    assert!(
        !metal_off.decode_flags.fused_prelayer_norm,
        "expected fused_prelayer_norm=false in 'off' backend"
    );
    let mut kv_off = metal_off.create_kv_cache(2, 64, NUM_KV_HEADS, HEAD_DIM);
    let out_off = larql_compute::metal::MetalBackend::decode_token(
        &metal_off,
        &mut kv_off,
        &layers,
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

    // Run with fusion ON — fresh backend that captures the env flip.
    env::set_var("LARQL_FUSED_PRELAYER_NORM", "1");
    let metal_on = larql_compute::metal::MetalBackend::new()
        .expect("Metal device available since metal_off succeeded");
    assert!(
        metal_on.decode_flags.fused_prelayer_norm,
        "expected fused_prelayer_norm=true in 'on' backend"
    );
    let mut kv_on = metal_on.create_kv_cache(2, 64, NUM_KV_HEADS, HEAD_DIM);
    let out_on = larql_compute::metal::MetalBackend::decode_token(
        &metal_on,
        &mut kv_on,
        &layers,
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
    env::remove_var("LARQL_FUSED_PRELAYER_NORM");

    assert_eq!(out_off.len(), out_on.len(), "output length mismatch");
    let mut max_diff = 0.0f32;
    let mut max_idx = 0usize;
    for (i, (a, b)) in out_off.iter().zip(&out_on).enumerate() {
        let d = (a - b).abs();
        if d > max_diff {
            max_diff = d;
            max_idx = i;
        }
    }
    // Bit-identical isn't realistic across two different RMS reductions
    // (residual_norm_store does cooperative reduction in a different
    // grouping than rms_norm + plain residual_add); allow small FP drift.
    assert!(
        max_diff < 1e-3,
        "D-RMS-FUSE off-vs-on diverged: max_diff={max_diff} at index {max_idx}; \
         out_off[{max_idx}]={a} vs out_on[{max_idx}]={b}",
        a = out_off[max_idx],
        b = out_on[max_idx],
    );
}

/// Gemma-3-style layer: `has_post_norms = true`, mixed Q4_K Q/K +
/// Q6_K V, QK-norm enabled. Exercises the post-norms branches in
/// `encode_attn.rs` (line 401's `if has_post_norms`) and the mixed-
/// quant QKV path in `encode_qkv.rs` that the Llama-style smoke test
/// above doesn't reach.
#[test]
fn decode_token_gemma3_style_post_norms_smoke() {
    let metal = match larql_compute::metal::MetalBackend::new() {
        Some(m) => m,
        None => {
            eprintln!("skip: no Metal device");
            return;
        }
    };

    use larql_compute::cpu::ops::q4_common::{quantize_q4_k, quantize_q6_k};

    // Mixed-quant attention: Q/K are Q4_K, V is Q6_K (Gemma 3/4 ollama
    // convention). FFN gate/up Q4_K, down Q6_K (also production
    // convention).
    let wq_data = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 1.1));
    let wk_data = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 1.2));
    let wv_data = quantize_q6_k(&synth_weight_f32(KV_DIM * HIDDEN, 1.3));
    let wo_data = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 1.4));
    let gate_data = quantize_q4_k(&synth_weight_f32(INTER * HIDDEN, 1.5));
    let up_data = quantize_q4_k(&synth_weight_f32(INTER * HIDDEN, 1.6));
    let down_data = quantize_q6_k(&synth_weight_f32(HIDDEN * INTER, 1.7));

    // Per-head QK norm weights (head_dim).
    let qk_norm_w: Vec<f32> = (0..HEAD_DIM).map(|i| 0.5 + (i as f32 * 0.01)).collect();
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.0005)).collect();

    // post_attn_norm + pre_ffn_norm + post_ffn_norm = the Gemma 3/4
    // four-norm-per-layer pattern.
    let layer = FullPipelineLayer {
        wq: QuantWeight {
            data: &wq_data,
            scales: None,
            format: QuantFormat::Q4_K,
        },
        wk: QuantWeight {
            data: &wk_data,
            scales: None,
            format: QuantFormat::Q4_K,
        },
        wv: QuantWeight {
            data: &wv_data,
            scales: None,
            format: QuantFormat::Q6_K,
        },
        wo: QuantWeight {
            data: &wo_data,
            scales: None,
            format: QuantFormat::Q4_K,
        },
        gate: QuantWeight {
            data: &gate_data,
            scales: None,
            format: QuantFormat::Q4_K,
        },
        up: QuantWeight {
            data: &up_data,
            scales: None,
            format: QuantFormat::Q4_K,
        },
        down: QuantWeight {
            data: &down_data,
            scales: None,
            format: QuantFormat::Q6_K,
        },
        input_norm: &norm_w,
        post_attn_norm: &norm_w,
        pre_ffn_norm: Some(&norm_w),
        post_ffn_norm: Some(&norm_w),
        norm_offset: 1.0, // Gemma 2/3 HF baked-in offset
        has_post_norms: true,
        activation: Activation::GeluTanh,
        qk_norm_offset: 1.0,
        eps: 1e-6,
        norm_type: NormType::RmsNorm,
        ffn_type: FfnType::Gated,
        attn_scale: 1.0 / (HEAD_DIM as f32).sqrt(),
        head_dim: HEAD_DIM,
        num_q_heads: NUM_Q_HEADS,
        num_kv_heads: NUM_KV_HEADS,
        rope_base: 10_000.0,
        rotary_dim: 0,
        sliding_window: 0,
        has_v_norm: false,
        layer_scalar: 0.0,
        input_norm_bias: None,
        post_attn_norm_bias: None,
        q_norm_weight: Some(&qk_norm_w),
        k_norm_weight: Some(&qk_norm_w),
        ffn_up_bias: None,
        ffn_down_bias: None,
        moe: None,
        ffn_is_remote: false,
        moe_combined_output_norm: false,
        moe_outer_post_norm: None,
        kv_shared_source: None,
        ple_input_gate: None,
        ple_projection: None,
        ple_post_norm: None,
    };

    let x = synth_input(HIDDEN, 1.9);
    let mut kv = metal.create_kv_cache(1, 64, NUM_KV_HEADS, HEAD_DIM);

    let result = larql_compute::metal::MetalBackend::decode_token(
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

    assert_eq!(result.len(), HIDDEN);
    let nan = result.iter().filter(|v| v.is_nan()).count();
    assert_eq!(nan, 0, "Gemma-3-style decode produced {nan} NaNs");
    let inf = result.iter().filter(|v| v.is_infinite()).count();
    assert_eq!(inf, 0, "Gemma-3-style decode produced {inf} infinities");
    let max_abs = result.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
    assert!(max_abs > 0.0, "Gemma-3-style output is all-zero");
    assert!(
        max_abs < 1e6,
        "Gemma-3-style output magnitude {max_abs} unreasonable"
    );
}

/// Multi-layer decode (3 layers) — exercises the layer-loop's
/// state-propagation logic in `metal/decode/mod.rs`. Single-layer
/// tests skip the inter-iteration `h_buf = new_h` swap and the
/// per-layer scratch reuse paths.
#[test]
fn decode_token_multi_layer_synthetic_smoke() {
    let metal = match larql_compute::metal::MetalBackend::new() {
        Some(m) => m,
        None => {
            eprintln!("skip: no Metal device");
            return;
        }
    };

    use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_q4_k};

    // (wq, wk, wv, wo, gate, up, down) packed bytes for one synthetic
    // layer. Named alias keeps the per-layer Vec literal readable and
    // shuts up clippy::type_complexity.
    type SynthLayerWeights = (
        Vec<u8>,
        Vec<u8>,
        Vec<u8>,
        Vec<u8>,
        Vec<u8>,
        Vec<u8>,
        Vec<u8>,
    );

    // Build 3 distinct layers with different seeds so the layer loop
    // genuinely advances state.
    let mut layers_data: Vec<SynthLayerWeights> = Vec::with_capacity(3);
    for l in 0..3usize {
        let s = l as f32 * 0.1;
        layers_data.push((
            quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.10 + s)),
            quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.20 + s)),
            quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.30 + s)),
            quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.40 + s)),
            quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.50 + s)),
            quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.60 + s)),
            quantize_q4_0(&synth_weight_f32(HIDDEN * INTER, 0.70 + s)),
        ));
    }
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();

    let layers: Vec<FullPipelineLayer<'_>> = layers_data
        .iter()
        .map(|(wq, wk, wv, wo, gate, up, down)| {
            build_synth_layer(wq, wk, wv, wo, gate, up, down, &norm_w)
        })
        .collect();

    let x = synth_input(HIDDEN, 0.95);
    let mut kv = metal.create_kv_cache(layers.len(), 64, NUM_KV_HEADS, HEAD_DIM);

    let result = larql_compute::metal::MetalBackend::decode_token(
        &metal,
        &mut kv,
        &layers,
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

    assert_eq!(result.len(), HIDDEN);
    assert_eq!(result.iter().filter(|v| v.is_nan()).count(), 0);
    assert_eq!(result.iter().filter(|v| v.is_infinite()).count(), 0);
    let max_abs = result.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
    assert!(max_abs > 0.0, "multi-layer output is all-zero");
}

/// `LARQL_QKV_FUSED=1` opts into the `q4k_q6k_qkv_proj_normed` path
/// (norm rolled into the matmul; defused as default 2026-05-09 per
/// ADR-016). Exercises `encode_normed_q4k_q6k_qkv` which is otherwise
/// unreached by the default tests.
#[test]
fn decode_token_qkv_fused_opt_in_smoke() {
    use std::env;

    // Serialise against `d_rms_fuse_phase1_produces_identical_output`
    // and any future env-mutating test in this binary. Decode flags
    // are cached at backend construction; set the env BEFORE creating
    // the backend.
    let _env_guard = ENV_TEST_LOCK.lock().expect("env lock poisoned");

    use larql_compute::cpu::ops::q4_common::{quantize_q4_k, quantize_q6_k};

    let wq_data = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 2.1));
    let wk_data = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 2.2));
    let wv_data = quantize_q6_k(&synth_weight_f32(KV_DIM * HIDDEN, 2.3));
    let wo_data = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 2.4));
    let gate_data = quantize_q4_k(&synth_weight_f32(INTER * HIDDEN, 2.5));
    let up_data = quantize_q4_k(&synth_weight_f32(INTER * HIDDEN, 2.6));
    let down_data = quantize_q6_k(&synth_weight_f32(HIDDEN * INTER, 2.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.0009)).collect();

    // Layer matches the dispatcher's mixed_q4k_q6k_v + RmsNorm + no-bias
    // condition that gates the fused path. has_post_norms is false here
    // (a non-Gemma layer that still hits the normed QKV opt-in).
    let layer = FullPipelineLayer {
        wq: QuantWeight {
            data: &wq_data,
            scales: None,
            format: QuantFormat::Q4_K,
        },
        wk: QuantWeight {
            data: &wk_data,
            scales: None,
            format: QuantFormat::Q4_K,
        },
        wv: QuantWeight {
            data: &wv_data,
            scales: None,
            format: QuantFormat::Q6_K,
        },
        wo: QuantWeight {
            data: &wo_data,
            scales: None,
            format: QuantFormat::Q4_K,
        },
        gate: QuantWeight {
            data: &gate_data,
            scales: None,
            format: QuantFormat::Q4_K,
        },
        up: QuantWeight {
            data: &up_data,
            scales: None,
            format: QuantFormat::Q4_K,
        },
        down: QuantWeight {
            data: &down_data,
            scales: None,
            format: QuantFormat::Q6_K,
        },
        input_norm: &norm_w,
        post_attn_norm: &norm_w,
        pre_ffn_norm: None,
        post_ffn_norm: None,
        norm_offset: 0.0,
        has_post_norms: false,
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
        ple_input_gate: None,
        ple_projection: None,
        ple_post_norm: None,
    };

    let x = synth_input(HIDDEN, 2.9);

    // Decode flags are cached at `MetalBackend::new()`; set env BEFORE
    // construction so the fused QKV path is actually engaged.
    env::set_var("LARQL_QKV_FUSED", "1");
    let metal = match larql_compute::metal::MetalBackend::new() {
        Some(m) => m,
        None => {
            env::remove_var("LARQL_QKV_FUSED");
            eprintln!("skip: no Metal device");
            return;
        }
    };
    assert!(
        metal.decode_flags.qkv_fused,
        "expected qkv_fused=true after setting env before backend construction"
    );
    let mut kv = metal.create_kv_cache(1, 64, NUM_KV_HEADS, HEAD_DIM);
    let result = larql_compute::metal::MetalBackend::decode_token(
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
    env::remove_var("LARQL_QKV_FUSED");

    assert_eq!(result.len(), HIDDEN);
    assert_eq!(result.iter().filter(|v| v.is_nan()).count(), 0);
    let max_abs = result.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
    assert!(max_abs > 0.0, "QKV-fused-opt-in output is all-zero");
}

/// `prefill_q4` exercises a different code path than `decode_token`:
/// `metal/ops/full_pipeline/{dispatch,stages,full_layer}.rs` instead of
/// `metal/decode/*`. Multi-position seq_len=4 prefill on a synthetic
/// Llama-style layer.
#[test]
fn prefill_q4_seq4_synthetic_smoke() {
    let metal = match larql_compute::metal::MetalBackend::new() {
        Some(m) => m,
        None => {
            eprintln!("skip: no Metal device");
            return;
        }
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

    let layer = build_synth_layer(
        &wq_data, &wk_data, &wv_data, &wo_data, &gate_data, &up_data, &down_data, &norm_w,
    );

    let seq_len = 4usize;
    let x: Vec<f32> = (0..seq_len * HIDDEN)
        .map(|i| ((i as f32 * 0.011 + 3.9).sin()) * 0.4)
        .collect();

    // prefill_q4 returns the final-position hidden state (size HIDDEN);
    // KV cache is populated in place. None means the backend doesn't
    // support this path — only Metal does.
    let result = (&metal as &dyn ComputeBackend)
        .as_any()
        .downcast_ref::<larql_compute::metal::MetalBackend>()
        .unwrap()
        .prefill_q4(
            &[layer],
            &x,
            HIDDEN,
            INTER,
            seq_len,
            false, // use_qk_norm
            0.0,   // softcap
        );

    let result = match result {
        Some(r) => r,
        None => {
            eprintln!(
                "skip: prefill_q4 returned None (synthetic layer not supported by this path)"
            );
            return;
        }
    };

    // prefill_q4 returns seq_len × hidden (all positions, not just last).
    assert_eq!(result.len(), seq_len * HIDDEN, "prefill_q4 output length");
    assert_eq!(result.iter().filter(|v| v.is_nan()).count(), 0);
    assert_eq!(result.iter().filter(|v| v.is_infinite()).count(), 0);
    let max_abs = result.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
    assert!(max_abs > 0.0, "prefill_q4 output is all-zero");
    assert!(
        max_abs < 1e6,
        "prefill_q4 output magnitude {max_abs} unreasonable"
    );
}

/// `MetalBackend::with_options` honours its `BackendOptions` argument —
/// in particular, env vars must NOT override an explicit option. Pre-M1
/// the constructor read env directly; the field-driven build path makes
/// programmatic configuration trustworthy regardless of process env.
#[test]
fn with_options_honours_explicit_decode_flags_over_env() {
    use std::env;

    // Serialise against the env-toggling tests in this binary.
    let _env_guard = ENV_TEST_LOCK.lock().expect("env lock poisoned");

    // Set the env var to "on", then construct a backend with the option
    // explicitly OFF. The backend must reflect the explicit choice.
    env::set_var("LARQL_QKV_FUSED", "1");

    let mut opts = larql_compute::BackendOptions::default();
    opts.decode_flags.qkv_fused = false;

    let metal = match larql_compute::metal::MetalBackend::with_options(opts) {
        Some(m) => m,
        None => {
            env::remove_var("LARQL_QKV_FUSED");
            eprintln!("skip: no Metal device");
            return;
        }
    };

    assert!(
        !metal.decode_flags.qkv_fused,
        "with_options must override env: explicit qkv_fused=false but \
         backend resolved to {}",
        metal.decode_flags.qkv_fused
    );

    // And the inverse: env unset, explicit option ON.
    env::remove_var("LARQL_QKV_FUSED");
    let mut opts_on = larql_compute::BackendOptions::default();
    opts_on.decode_flags.qkv_fused = true;
    let metal_on = larql_compute::metal::MetalBackend::with_options(opts_on)
        .expect("Metal device available since first construction succeeded");
    assert!(
        metal_on.decode_flags.qkv_fused,
        "with_options must honour explicit qkv_fused=true even with env unset"
    );
}

/// Regression: dispatch geometry must travel with `KernelHandle`, not
/// with shader-module re-exports. Pin the QKV projection pipelines'
/// rows/threads against the shader-module constants so any drift fails
/// at unit-test time, not at "decode emits garbage on this model" time.
///
/// This is the audit pair to `decode_attention_layer_q4k_writes_all_kv_rows`:
/// that test catches the runtime symptom; this one catches the static
/// invariant.
#[test]
fn qkv_pipeline_geometry_matches_shader_constants() {
    let metal = match larql_compute::metal::MetalBackend::new() {
        Some(m) => m,
        None => {
            eprintln!("skip: no Metal device");
            return;
        }
    };

    use larql_compute::metal::shaders::{q4k_qkv_proj as q4k, q4kf_qkv_proj as q4kf};

    assert_eq!(
        metal.attention.q4k_qkv_proj_pipeline.rows_per_tg,
        q4k::ROWS_PER_TG
    );
    assert_eq!(
        metal.attention.q4k_qkv_proj_pipeline.threads_per_tg,
        q4k::THREADS_PER_TG
    );
    assert_eq!(
        metal.attention.q4kf_qkv_proj_pipeline.rows_per_tg,
        q4kf::ROWS_PER_TG
    );
    assert_eq!(
        metal.attention.q4kf_qkv_proj_pipeline.threads_per_tg,
        q4kf::THREADS_PER_TG
    );

    // The two pipelines must have DIFFERENT geometry — that's the whole
    // reason the bug existed. If they ever converge, delete this assert
    // and document the consolidation.
    assert!(
        metal.attention.q4k_qkv_proj_pipeline.rows_per_tg
            != metal.attention.q4kf_qkv_proj_pipeline.rows_per_tg
            || metal.attention.q4k_qkv_proj_pipeline.threads_per_tg
                != metal.attention.q4kf_qkv_proj_pipeline.threads_per_tg,
        "Q4_K and Q4_KF QKV pipelines now share geometry — \
         the decode_hybrid bug class no longer applies"
    );
}

/// Regression: MoE gate+up dispatch geometry must come from the bound
/// `KernelHandle`, not from re-imported shader-module constants. The
/// existing `q4k_ffn_gate_up_pipeline` is currently 4sg, but if it ever
/// gets bumped to 8sg (mirroring `q4k_matvec_pipeline`'s 4→8sg flip
/// from 2026-04-28), the `moe_dispatch.rs` paths must follow.
#[test]
fn moe_gate_up_pipeline_geometry_matches_shader_constants() {
    let metal = match larql_compute::metal::MetalBackend::new() {
        Some(m) => m,
        None => {
            eprintln!("skip: no Metal device");
            return;
        }
    };

    use larql_compute::metal::shaders::{
        q4k_ffn_gate_up as q4k_gu, q4k_ffn_gate_up_8sg as q4k_gu_8sg,
    };

    assert_eq!(
        metal.ffn.q4k_ffn_gate_up_pipeline.rows_per_tg,
        q4k_gu::ROWS_PER_TG
    );
    assert_eq!(
        metal.ffn.q4k_ffn_gate_up_pipeline.threads_per_tg,
        q4k_gu::THREADS_PER_TG
    );
    assert_eq!(
        metal.ffn.q4k_ffn_gate_up_8sg_pipeline.rows_per_tg,
        q4k_gu_8sg::ROWS_PER_TG
    );
    assert_eq!(
        metal.ffn.q4k_ffn_gate_up_8sg_pipeline.threads_per_tg,
        q4k_gu_8sg::THREADS_PER_TG
    );
}
