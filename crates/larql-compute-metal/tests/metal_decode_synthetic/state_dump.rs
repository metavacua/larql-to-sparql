//! W10 state-dump masked-fn tests (D-STATE-MASK).

#[allow(unused_imports)]
use crate::common::*;

// ── W10 state-dump masked-fn tests (D-STATE-MASK) ──────────────────────────
//
// `decode_token_with_state_dump_*_fn` are the W10 capture surface used by
// the kv-engines (markov_residual / codec / turbo_quant) to read per-layer
// `h_in` + new K/V back without an extra forward pass. The masked variants
// let engines that treat K/V as derivative state skip the K/V staging
// (`HOnly`) or skip *both* staging and h_in readback (`None`). These
// smoke tests exercise the wrapper + the three Mask branches inside
// `decode_token_with_moe_split_fn` so the W10-added LOC are covered.

/// Build the same Q4_K-attention + Q4_0-FFN fixture the standard smoke
/// test uses, run a decode step through the requested mask, and return
/// the output buffer + populated `DecodeStateDump`.
fn run_state_dump_decode(
    mask: larql_compute::StateDumpMask,
) -> Option<(Vec<f32>, larql_compute::DecodeStateDump)> {
    let metal = larql_compute_metal::MetalBackend::new()?;
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
    let mut state = larql_compute::DecodeStateDump::with_capacity(1);

    let result = metal.decode_token_with_state_dump_masked_fn(
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
        &mut state,
        mask,
    );
    Some((result, state))
}

#[test]
fn decode_token_with_state_dump_full_captures_h_and_kv_per_layer() {
    // Hold the env-test lock since `decode_token_*` reads
    // `LARQL_FUSED_PRELAYER_NORM` / `LARQL_QKV_FUSED` and we don't
    // want a concurrent test toggling them mid-flight.
    let _g = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some((out, state)) = run_state_dump_decode(larql_compute::StateDumpMask::Full) else {
        eprintln!("skip: no Metal device");
        return;
    };
    assert_eq!(out.len(), HIDDEN);
    assert!(out.iter().all(|v| v.is_finite()));
    // Full mask populates all three per-layer vectors.
    assert_eq!(state.h_in_per_layer.len(), 1);
    assert_eq!(state.k_new_per_layer.len(), 1);
    assert_eq!(state.v_new_per_layer.len(), 1);
    assert_eq!(state.h_in_per_layer[0].len(), HIDDEN);
    assert_eq!(state.k_new_per_layer[0].len(), KV_DIM);
    assert_eq!(state.v_new_per_layer[0].len(), KV_DIM);
    assert!(state.is_complete_under(1, larql_compute::StateDumpMask::Full));
}

#[test]
fn decode_token_with_state_dump_h_only_skips_kv_readback() {
    let _g = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some((out, state)) = run_state_dump_decode(larql_compute::StateDumpMask::HOnly) else {
        eprintln!("skip: no Metal device");
        return;
    };
    assert_eq!(out.len(), HIDDEN);
    assert!(out.iter().all(|v| v.is_finite()));
    // HOnly mask: h_in populated, K/V vecs stay empty.
    assert_eq!(state.h_in_per_layer.len(), 1);
    assert_eq!(state.k_new_per_layer.len(), 0);
    assert_eq!(state.v_new_per_layer.len(), 0);
    assert!(state.is_complete_under(1, larql_compute::StateDumpMask::HOnly));
}

#[test]
fn decode_token_with_state_dump_none_skips_all_readbacks() {
    let _g = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some((out, state)) = run_state_dump_decode(larql_compute::StateDumpMask::None) else {
        eprintln!("skip: no Metal device");
        return;
    };
    assert_eq!(out.len(), HIDDEN);
    assert!(out.iter().all(|v| v.is_finite()));
    // None mask: state stays empty; Metal kv-cache is the source of truth.
    assert_eq!(state.h_in_per_layer.len(), 0);
    assert_eq!(state.k_new_per_layer.len(), 0);
    assert_eq!(state.v_new_per_layer.len(), 0);
    assert!(state.is_complete_under(1, larql_compute::StateDumpMask::None));
}

#[test]
fn decode_token_with_state_dump_unmasked_wrapper_defaults_to_full() {
    // Calls the no-mask wrapper (`decode_token_with_state_dump_fn`).
    // Wrapper-side coverage: the 14-arg shim that forwards with
    // `StateDumpMask::Full`. Numerical equivalence with the masked
    // variant above is the contract.
    let _g = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let metal = match larql_compute_metal::MetalBackend::new() {
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
    let mut state = larql_compute::DecodeStateDump::with_capacity(1);

    let out = metal.decode_token_with_state_dump_fn(
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
        &mut state,
    );
    assert_eq!(out.len(), HIDDEN);
    assert!(out.iter().all(|v| v.is_finite()));
    assert!(state.is_complete_for(1));
}
