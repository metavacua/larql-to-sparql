//! The `DecodeBackend` trait surface, driven through `&dyn ComputeBackend`.
//!
//! The rest of this directory calls `MetalBackend::…` inherent methods.
//! That leaves the *trait* impls — the form every real caller in
//! `larql-inference` actually uses — reachable only through an end-to-end
//! run in another crate, so a signature that compiles but dispatches to the
//! wrong inherent method would not be caught here.
//!
//! These are smoke tests in the same spirit as their neighbours: shape,
//! finiteness, and that the trait method reaches the implementation rather
//! than the default. Numerical parity lives at the vindex scope.

#[allow(unused_imports)]
use crate::common::*;

use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_q4_k};
use larql_compute::{ComputeBackend, StateDumpMask};

/// The synthetic layer every test here shares.
struct Weights {
    wq: Vec<u8>,
    wk: Vec<u8>,
    wv: Vec<u8>,
    wo: Vec<u8>,
    gate: Vec<u8>,
    up: Vec<u8>,
    down: Vec<u8>,
    norm_w: Vec<f32>,
}

fn weights() -> Weights {
    Weights {
        wq: quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.1)),
        wk: quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.2)),
        wv: quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.3)),
        wo: quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.4)),
        gate: quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.5)),
        up: quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.6)),
        down: quantize_q4_0(&synth_weight_f32(HIDDEN * INTER, 0.7)),
        norm_w: (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect(),
    }
}

/// `decode_token` through the trait. The KV cache is preallocated through
/// the trait too, because the two are used together and a decode against an
/// unallocated cache is the shape that used to panic.
#[test]
fn trait_decode_token_runs_a_preallocated_layer() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    let w = weights();
    let layer = build_synth_layer(
        &w.wq, &w.wk, &w.wv, &w.wo, &w.gate, &w.up, &w.down, &w.norm_w,
    );
    let backend: &dyn ComputeBackend = &metal;

    backend.reset_kv_cache();
    backend.preallocate_kv_cache_per_layer_with_capacity(&[(NUM_KV_HEADS, HEAD_DIM)], &[64]);

    let x = synth_input(HIDDEN, 0.9);
    let out = backend
        .decode_token(std::slice::from_ref(&layer), &x, HIDDEN, INTER)
        .expect("the trait method must reach the Metal decode, not the default None");
    assert_eq!(out.len(), HIDDEN);
    assert!(out.iter().all(|v| v.is_finite()));
    assert!(
        out.iter().any(|v| *v != 0.0),
        "an all-zero result would pass every shape assertion while computing nothing"
    );
}

/// The state-dump variants: same compute path, plus per-layer capture.
/// `Full` captures h and K/V; `HOnly` skips the K/V readback; `None`
/// captures nothing. The masks are what let an engine pay only for what it
/// reads, so each is asserted to actually differ in what it populates.
#[test]
fn trait_decode_token_with_state_dump_honours_each_mask() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    let w = weights();
    let layer = build_synth_layer(
        &w.wq, &w.wk, &w.wv, &w.wo, &w.gate, &w.up, &w.down, &w.norm_w,
    );
    let backend: &dyn ComputeBackend = &metal;
    let x = synth_input(HIDDEN, 0.9);

    // Unmasked wrapper: the historical entry point, equivalent to Full.
    backend.reset_kv_cache();
    backend.preallocate_kv_cache_per_layer_with_capacity(&[(NUM_KV_HEADS, HEAD_DIM)], &[64]);
    let mut full = larql_compute::DecodeStateDump::default();
    let out = backend
        .decode_token_with_state_dump(
            std::slice::from_ref(&layer),
            &x,
            HIDDEN,
            INTER,
            Some(&mut full),
        )
        .expect("state-dump decode reaches the Metal impl");
    assert_eq!(out.len(), HIDDEN);
    assert!(
        !full.h_in_per_layer.is_empty(),
        "Full captures the residual"
    );
    assert!(
        !full.k_new_per_layer.is_empty(),
        "Full captures K/V as well as h"
    );

    // HOnly: h still captured, K/V readback skipped.
    backend.reset_kv_cache();
    backend.preallocate_kv_cache_per_layer_with_capacity(&[(NUM_KV_HEADS, HEAD_DIM)], &[64]);
    let mut h_only = larql_compute::DecodeStateDump::default();
    backend
        .decode_token_with_state_dump_masked(
            std::slice::from_ref(&layer),
            &x,
            HIDDEN,
            INTER,
            Some(&mut h_only),
            StateDumpMask::HOnly,
        )
        .expect("masked state-dump decode reaches the Metal impl");
    assert!(
        !h_only.h_in_per_layer.is_empty(),
        "HOnly still captures the residual"
    );
    assert!(
        h_only.k_new_per_layer.is_empty(),
        "HOnly must skip the K/V readback — that is the whole point of the mask"
    );

    // None: nothing captured at all.
    backend.reset_kv_cache();
    backend.preallocate_kv_cache_per_layer_with_capacity(&[(NUM_KV_HEADS, HEAD_DIM)], &[64]);
    let mut none = larql_compute::DecodeStateDump::default();
    backend
        .decode_token_with_state_dump_masked(
            std::slice::from_ref(&layer),
            &x,
            HIDDEN,
            INTER,
            Some(&mut none),
            StateDumpMask::None,
        )
        .expect("masked state-dump decode reaches the Metal impl");
    assert!(none.h_in_per_layer.is_empty() && none.k_new_per_layer.is_empty());
}

/// The split fire/collect variant. Both callbacks must be invoked for a MoE
/// layer; the fire's return is discarded and the collect's is used, which is
/// the contract a remote expert backend relies on.
#[test]
fn trait_decode_token_with_moe_split_invokes_both_callbacks() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    let w = weights();
    let mut layer = build_synth_layer(
        &w.wq, &w.wk, &w.wv, &w.wo, &w.gate, &w.up, &w.down, &w.norm_w,
    );
    layer.moe = Some(null_moe_layer());
    let backend: &dyn ComputeBackend = &metal;

    backend.reset_kv_cache();
    backend.preallocate_kv_cache_per_layer_with_capacity(&[(NUM_KV_HEADS, HEAD_DIM)], &[64]);

    let mut fired = 0usize;
    let mut collected = 0usize;
    let x = synth_input(HIDDEN, 0.9);
    let out = {
        let mut fire = |_l: usize, _h: &[f32]| {
            fired += 1;
        };
        let mut collect = |_l: usize| -> Vec<f32> {
            collected += 1;
            vec![0.0f32; HIDDEN]
        };
        backend.decode_token_with_moe_split(
            std::slice::from_ref(&layer),
            &x,
            HIDDEN,
            INTER,
            &mut fire,
            &mut collect,
        )
    }
    .expect("split decode reaches the Metal impl");

    assert_eq!(out.len(), HIDDEN);
    assert_eq!(fired, 1, "the fire callback runs once per MoE layer");
    assert_eq!(collected, 1, "the collect callback runs once per MoE layer");
}

/// TOKEN-B1 rung 2's trait entry: the fused decode head reached through
/// `&dyn ComputeBackend`, which is how `decode_loop` calls it.
///
/// Both directions matter. A plan the backend can serve must come back as
/// a top-K — that is the fast path actually firing. A plan it cannot serve
/// must come back `None` so the caller runs the unfused head; a `Some`
/// there would be a distribution computed under the wrong normaliser.
#[test]
fn trait_decode_token_q4k_moe_head_serves_a_valid_plan_and_refuses_an_unservable_one() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    let w = weights();
    // A DENSE layer deliberately. On the legacy MoE interleave path
    // `handle_moe_interleave` commits at the last layer's boundary and does
    // not re-open an encoder, so there is no open buffer for the head to
    // ride and it correctly answers `None`. The head rides the merged-CB
    // GPU-route arm in production; here a dense layer gives the same
    // "encoder still open at the end of the token" shape without needing a
    // routed expert bank.
    let layer = build_synth_layer(
        &w.wq, &w.wk, &w.wv, &w.wo, &w.gate, &w.up, &w.down, &w.norm_w,
    );
    let backend: &dyn ComputeBackend = &metal;

    const VOCAB: usize = 512;
    let final_norm = vec![1.0f32; HIDDEN];
    let lm_head = quantize_q4_k(&synth_weight_f32(VOCAB * HIDDEN, 0.8));
    let plan = larql_compute::DecodeHeadPlan {
        norm_type: larql_compute::NormType::RmsNorm,
        final_norm_weight: &final_norm,
        norm_eps: 1e-6,
        norm_offset: 0.0,
        lm_head_q4k: &lm_head,
        vocab: VOCAB,
        cols: HIDDEN,
        top_k: 5,
    };
    let x = synth_input(HIDDEN, 0.9);

    backend.reset_kv_cache();
    backend.preallocate_kv_cache_per_layer_with_capacity(&[(NUM_KV_HEADS, HEAD_DIM)], &[64]);
    let hits = backend
        .decode_token_q4k_moe_head(
            std::slice::from_ref(&layer),
            &x,
            HIDDEN,
            INTER,
            1e-6,
            &|_l, _e| None,
            &plan,
        )
        .expect("a servable plan must produce a top-K through the trait");
    assert_eq!(hits.len(), 5);
    assert!(
        hits.windows(2).all(|p| p[0].1 >= p[1].1),
        "top-K comes back descending: {hits:?}"
    );
    assert!(hits
        .iter()
        .all(|(id, s)| (*id as usize) < VOCAB && s.is_finite()));

    // A norm the Metal head does not implement: must refuse, not guess.
    let unservable = larql_compute::DecodeHeadPlan {
        norm_type: larql_compute::NormType::LayerNorm,
        final_norm_weight: &final_norm,
        norm_eps: 1e-6,
        norm_offset: 0.0,
        lm_head_q4k: &lm_head,
        vocab: VOCAB,
        cols: HIDDEN,
        top_k: 5,
    };
    backend.reset_kv_cache();
    backend.preallocate_kv_cache_per_layer_with_capacity(&[(NUM_KV_HEADS, HEAD_DIM)], &[64]);
    assert!(
        backend
            .decode_token_q4k_moe_head(
                std::slice::from_ref(&layer),
                &x,
                HIDDEN,
                INTER,
                1e-6,
                &|_l, _e| None,
                &unservable,
            )
            .is_none(),
        "LayerNorm is not implemented by the fused head; it must refuse"
    );
}
