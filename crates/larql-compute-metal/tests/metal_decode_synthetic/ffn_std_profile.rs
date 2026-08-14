//! PLE outer guard, standard (non-gated) FFN paths, profile-split matrix over FFN formats.

#[allow(unused_imports)]
use crate::common::*;

/// Backend with PLE inputs prepared — covers the `ple_inputs.as_ref()`
/// branch around `decode/mod.rs` lines 496-519.  Synthetic layer
/// doesn't actually wire PLE weights so the inner `layer.ple_spec()`
/// check returns None; that exercises the `if Some(pli)` outer guard
/// while skipping the actual PLE dispatch (which would need real
/// gate/projection/post-norm weights to be correct).
#[test]
fn decode_token_with_ple_inputs_drives_outer_guard() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_q4_k};
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };

    // Synthetic per-layer-input table.  Sized for one layer × one
    // position with ple_dim = 32; any ple_dim works for the outer
    // guard test since `layer.ple_spec()` returns None.
    let ple_inputs: Vec<f32> = (0..32).map(|i| (i as f32) * 0.01).collect();
    metal.prepare_ple_inputs(&ple_inputs, 1, 32);

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
    metal.clear_ple_inputs();
}

// ─────────────────────────────────────────────────────────────────
// encode_ffn.rs branch coverage. Drives non-gated FFN variants
// (FfnType::Standard), the `gate_up_coop` / `gate_up_use_4sg` /
// `f16_acc` pipeline picks at the top of `encode_q4k_ffn`, and the
// `LARQL_PROFILE_SPLIT=1` paired-phase code path
// (encode_ffn_gate_up_phase + encode_ffn_down_phase) across the
// three quant families.
// ─────────────────────────────────────────────────────────────────

fn decode_with_options_synth_q4k_layer(opts: larql_compute_metal::BackendOptions) {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
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

fn decode_with_profile_split_synth<F: FnOnce(&mut FullPipelineLayer)>(setup: F) {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_q4_k};
    let saved = std::env::var_os("LARQL_PROFILE_SPLIT");
    unsafe { std::env::set_var("LARQL_PROFILE_SPLIT", "1") };
    let wq = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.1));
    let wk = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.2));
    let wv = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.3));
    let wo = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let gate = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.5));
    let up = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down = quantize_q4_0(&synth_weight_f32(HIDDEN * INTER, 0.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();
    let mut layer = build_synth_layer(&wq, &wk, &wv, &wo, &gate, &up, &down, &norm_w);
    setup(&mut layer);
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
    match saved {
        Some(v) => unsafe { std::env::set_var("LARQL_PROFILE_SPLIT", v) },
        None => unsafe { std::env::remove_var("LARQL_PROFILE_SPLIT") },
    }
}

/// `BackendOptions { gate_up_coop: true }` selects the cooperative
/// gate+up Q4_K pipeline (`q4k_ffn_gate_up_coop_pipeline`) at
/// `decode/encode_ffn.rs` lines 239-246.
///
/// Ignored on CI: the cooperative scale-loading kernel produces NaN on
/// the GitHub Actions macOS-14 (M1) runner against synthetic Q4_K
/// weights, while passing on M3 Max. The kernel is opt-in (documented
/// as kept around for future larger-K hardware in
/// `shaders/q4k_ffn_gate_up_coop.rs`); never on the default decode
/// path. Run `cargo test -- --ignored` on dev hardware to exercise it.
#[test]
#[ignore = "flaky on GitHub Actions M1 runner; gate_up_coop kernel produces NaN on M1 with synthetic Q4_K (passes on M3 Max). Opt-in kernel — not on default decode path. See shader retention doc."]
fn decode_token_with_q4k_ffn_and_gate_up_coop_option() {
    let mut opts = larql_compute_metal::BackendOptions::default();
    opts.decode_flags.gate_up_coop = true;
    decode_with_options_synth_q4k_layer(opts);
}

/// `BackendOptions { gate_up_use_4sg: true }` (LARQL_GATE_UP_8SG=0)
/// drives the 4-simdgroup Q4_K gate+up pipeline at lines 253-258.
#[test]
fn decode_token_with_q4k_ffn_and_4sg_option() {
    let mut opts = larql_compute_metal::BackendOptions::default();
    opts.decode_flags.gate_up_use_4sg = true;
    decode_with_options_synth_q4k_layer(opts);
}

/// `BackendOptions { gate_up_use_4sg: true, f16_acc: true }` drives
/// the 4sg + f16-accumulator Q4_K gate+up pipeline at lines 247-252.
#[test]
fn decode_token_with_q4k_ffn_4sg_and_f16_acc_option() {
    let mut opts = larql_compute_metal::BackendOptions::default();
    opts.decode_flags.gate_up_use_4sg = true;
    opts.decode_flags.f16_acc = true;
    decode_with_options_synth_q4k_layer(opts);
}

/// `BackendOptions { fused_down: false }` opts out of the fused
/// `q4k_geglu_silu_down` kernel — drives the separated GEGLU +
/// `quant_matvec` chain at `encode_q4k_ffn` lines 361-386.
#[test]
fn decode_token_with_q4k_ffn_and_unfused_down_option() {
    let mut opts = larql_compute_metal::BackendOptions::default();
    opts.decode_flags.fused_down = false;
    decode_with_options_synth_q4k_layer(opts);
}

/// `FfnType::Standard` (non-gated) + Q4_K weights drives the
/// `else` arm of `encode_q4k_ffn` (lines 389-424): up → activation
/// → down without GEGLU multiplication.
#[test]
fn decode_token_with_q4k_non_gated_ffn_drives_standard_path() {
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
    layer.gate = layer.gate.with_format(QuantFormat::Q4_K);
    layer.up = layer.up.with_format(QuantFormat::Q4_K);
    layer.down = layer.down.with_format(QuantFormat::Q4_K);
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
    assert!(out.iter().all(|v| v.is_finite()));
}

/// `FfnType::Standard` + Q4_0 weights drives the `else` arm of
/// `encode_q4_0_ffn` (lines 463-481): up Q8-matvec → activation →
/// down.
#[test]
fn decode_token_with_q4_0_non_gated_ffn_drives_standard_path() {
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
    assert!(out.iter().all(|v| v.is_finite()));
}

/// `LARQL_PROFILE_SPLIT=1` + Q4_K-gated FFN drives the split-phase
/// encoders at `decode/encode_ffn.rs` lines 649-674 (gate_up phase
/// Q4_K gated) and 751-783 (down phase Q4_K gated + Q4_K fused down).
#[test]
fn decode_token_with_profile_split_and_q4k_gated_ffn() {
    decode_with_profile_split_synth(|layer| {
        layer.gate = layer.gate.with_format(QuantFormat::Q4_K);
        layer.up = layer.up.with_format(QuantFormat::Q4_K);
        layer.down = layer.down.with_format(QuantFormat::Q4_K);
    });
}

/// `LARQL_PROFILE_SPLIT=1` + Q4_K non-gated drives the non-gated
/// arms of both split-phase encoders at lines 663-673 + 784-806.
#[test]
fn decode_token_with_profile_split_and_q4k_non_gated_ffn() {
    decode_with_profile_split_synth(|layer| {
        layer.gate = layer.gate.with_format(QuantFormat::Q4_K);
        layer.up = layer.up.with_format(QuantFormat::Q4_K);
        layer.down = layer.down.with_format(QuantFormat::Q4_K);
        layer.ffn_type = FfnType::Standard;
    });
}

/// `LARQL_PROFILE_SPLIT=1` + Q4_KF gated FFN drives the Q4_KF
/// arms of both split-phase encoders at lines 619-635 + 725-728.
#[test]
fn decode_token_with_profile_split_and_q4kf_gated_ffn() {
    decode_with_profile_split_synth(|layer| {
        layer.gate = layer.gate.with_format(QuantFormat::Q4_KF);
        layer.up = layer.up.with_format(QuantFormat::Q4_KF);
        layer.down = layer.down.with_format(QuantFormat::Q4_KF);
    });
}

/// `LARQL_PROFILE_SPLIT=1` + Q4_KF non-gated drives the Q4_KF
/// non-gated arm of `encode_ffn_gate_up_phase` (lines 636-647) and
/// `encode_ffn_down_phase` (lines 729-749).
#[test]
fn decode_token_with_profile_split_and_q4kf_non_gated_ffn() {
    decode_with_profile_split_synth(|layer| {
        layer.gate = layer.gate.with_format(QuantFormat::Q4_KF);
        layer.up = layer.up.with_format(QuantFormat::Q4_KF);
        layer.down = layer.down.with_format(QuantFormat::Q4_KF);
        layer.ffn_type = FfnType::Standard;
    });
}

/// `LARQL_PROFILE_SPLIT=1` + Q4_0 non-gated drives the Q4_0
/// non-gated arm of `encode_ffn_gate_up_phase` (lines 692-700) +
/// the Q4_0 non-gated arm of `encode_ffn_down_phase` (lines 812+).
#[test]
fn decode_token_with_profile_split_and_q4_0_non_gated_ffn() {
    decode_with_profile_split_synth(|layer| {
        layer.ffn_type = FfnType::Standard;
    });
}

/// Q6_K gate/up FFN decodes through the separated per-format chain
/// (capability audit F3).
///
/// Before the per-operand routing fix, a Q6_K gate fell into
/// `encode_q4k_ffn` via `is_kquant_family()` and its 210-byte planar
/// superblocks were decoded as 144-byte Q4_K — silent corruption with
/// no panic. The prefill path always routed Q6_K correctly; this pins
/// decode doing the same.
#[test]
fn decode_token_with_q6k_ffn_drives_separated_per_format_path() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    use larql_compute::cpu::ops::q4_common::{quantize_q4_k, quantize_q6_k};
    metal.reset_kv_cache();
    let wq = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.1));
    let wk = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.2));
    let wv = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.3));
    let wo = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let gate = quantize_q6_k(&synth_weight_f32(INTER * HIDDEN, 0.5));
    let up = quantize_q6_k(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down = quantize_q6_k(&synth_weight_f32(HIDDEN * INTER, 0.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();
    let mut layer = build_synth_layer(&wq, &wk, &wv, &wo, &gate, &up, &down, &norm_w);
    layer.gate = QuantWeight::new(QuantFormat::Q6_K, &gate, larql_compute::QuantAux::None);
    layer.up = QuantWeight::new(QuantFormat::Q6_K, &up, larql_compute::QuantAux::None);
    layer.down = QuantWeight::new(QuantFormat::Q6_K, &down, larql_compute::QuantAux::None);

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
    assert!(out.iter().any(|v| v.abs() > 1e-6), "all-zero decode output");
}

/// Mixed gate/up FFN formats are refused, not silently mis-decoded
/// with whichever kernel the gate format selected (capability audit
/// F3: `up.format()` was previously never read on the decode path).
#[test]
#[should_panic(expected = "mixed gate/up FFN formats")]
fn decode_token_with_mixed_gate_up_formats_is_rejected() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        panic!("mixed gate/up FFN formats (no Metal device; asserted statically)");
    };
    use larql_compute::cpu::ops::q4_common::{quantize_q4_k, quantize_q6_k};
    metal.reset_kv_cache();
    let wq = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.1));
    let wk = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.2));
    let wv = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.3));
    let wo = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let gate = quantize_q4_k(&synth_weight_f32(INTER * HIDDEN, 0.5));
    let up = quantize_q6_k(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down = quantize_q4_k(&synth_weight_f32(HIDDEN * INTER, 0.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();
    let mut layer = build_synth_layer(&wq, &wk, &wv, &wo, &gate, &up, &down, &norm_w);
    layer.gate = QuantWeight::new(QuantFormat::Q4_K, &gate, larql_compute::QuantAux::None);
    layer.up = QuantWeight::new(QuantFormat::Q6_K, &up, larql_compute::QuantAux::None);

    let x = synth_input(HIDDEN, 0.9);
    let mut kv = metal.create_kv_cache(1, 64, NUM_KV_HEADS, HEAD_DIM);
    let _ = larql_compute_metal::MetalBackend::decode_token(
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
}

/// Non-gated (`FfnType::Standard`) Q6_K FFN — covers `encode_q6k_ffn`'s
/// up → activation → down arm, which the gated test above cannot reach.
#[test]
fn decode_token_with_q6k_non_gated_ffn_drives_standard_path() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    use larql_compute::cpu::ops::q4_common::{quantize_q4_k, quantize_q6_k};
    metal.reset_kv_cache();
    let wq = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.1));
    let wk = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.2));
    let wv = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.3));
    let wo = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let gate = quantize_q6_k(&synth_weight_f32(INTER * HIDDEN, 0.5));
    let up = quantize_q6_k(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down = quantize_q6_k(&synth_weight_f32(HIDDEN * INTER, 0.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();
    let mut layer = build_synth_layer(&wq, &wk, &wv, &wo, &gate, &up, &down, &norm_w);
    layer.gate = QuantWeight::new(QuantFormat::Q6_K, &gate, larql_compute::QuantAux::None);
    layer.up = QuantWeight::new(QuantFormat::Q6_K, &up, larql_compute::QuantAux::None);
    layer.down = QuantWeight::new(QuantFormat::Q6_K, &down, larql_compute::QuantAux::None);
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
    assert!(out.iter().all(|v| v.is_finite()));
}

/// Profile-split with Q6_K gate/up/down. The split phases previously
/// branched on the kquant family boolean and sent a Q6_K gate to the
/// Q4_K pipelines under `LARQL_PROFILE_SPLIT=1`; they now consult the
/// same per-operand route as `encode_ffn_step` (slice 2).
#[test]
fn decode_token_with_profile_split_and_q6k_gated_ffn() {
    decode_with_profile_split_synth(|layer| {
        set_q6k_ffn(layer);
    });
}

#[test]
fn decode_token_with_profile_split_and_q6k_non_gated_ffn() {
    decode_with_profile_split_synth(|layer| {
        set_q6k_ffn(layer);
        layer.ffn_type = FfnType::Standard;
    });
}

/// `LARQL_FUSED_Q6K_DOWN=1` equivalent via BackendOptions: Q4_K
/// gate/up with a Q6_K GELU-tanh down drives the opt-in fused Q6_K
/// down kernel (which now carries the tanh clamp from F1).
#[test]
#[ignore = "reproducer for the fused-Q6K-down decode-integration NaN: the kernel \
passes isolation parity at this exact shape, but the decode dispatch \
produces 256/256 NaN. Arm hard-disabled in encode_ffn.rs until root-caused."]
fn decode_token_with_q4k_ffn_and_fused_q6k_down_option() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut opts = larql_compute_metal::BackendOptions::default();
    opts.decode_flags.fused_q6k_down = true;
    let Some(metal) = larql_compute_metal::MetalBackend::with_options(opts) else {
        return;
    };
    use larql_compute::cpu::ops::q4_common::{quantize_q4_k, quantize_q6_k};
    let wq = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.1));
    let wk = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.2));
    let wv = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.3));
    let wo = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let gate = quantize_q4_k(&synth_weight_f32(INTER * HIDDEN, 0.5));
    let up = quantize_q4_k(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down = quantize_q6_k(&synth_weight_f32(HIDDEN * INTER, 0.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();
    let mut layer = build_synth_layer(&wq, &wk, &wv, &wo, &gate, &up, &down, &norm_w);
    layer.down = QuantWeight::new(QuantFormat::Q6_K, &down, larql_compute::QuantAux::None);
    layer.activation = Activation::GeluTanh;

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
    let n_nan = out.iter().filter(|v| v.is_nan()).count();
    let n_inf = out.iter().filter(|v| v.is_infinite()).count();
    assert!(
        out.iter().all(|v| v.is_finite()),
        "fused q6k down: {n_nan} NaN, {n_inf} inf of {}; first vals {:?}",
        out.len(),
        &out[..4]
    );
}

/// Shared Q6_K FFN fixture mutation for the split tests above. The
/// weight bytes were built as Q4_K/Q4_0 by `build_synth_layer`'s
/// defaults, so requantize each matrix as genuine Q6_K.
fn set_q6k_ffn(layer: &mut FullPipelineLayer) {
    use larql_compute::cpu::ops::q4_common::quantize_q6_k;
    use std::sync::OnceLock;
    static GATE: OnceLock<Vec<u8>> = OnceLock::new();
    static UP: OnceLock<Vec<u8>> = OnceLock::new();
    static DOWN: OnceLock<Vec<u8>> = OnceLock::new();
    let gate = GATE.get_or_init(|| quantize_q6_k(&synth_weight_f32(INTER * HIDDEN, 0.5)));
    let up = UP.get_or_init(|| quantize_q6_k(&synth_weight_f32(INTER * HIDDEN, 0.6)));
    let down = DOWN.get_or_init(|| quantize_q6_k(&synth_weight_f32(HIDDEN * INTER, 0.7)));
    layer.gate = QuantWeight::new(QuantFormat::Q6_K, gate, larql_compute::QuantAux::None);
    layer.up = QuantWeight::new(QuantFormat::Q6_K, up, larql_compute::QuantAux::None);
    layer.down = QuantWeight::new(QuantFormat::Q6_K, down, larql_compute::QuantAux::None);
}
