//! Engine-family PLE + layer_scalar parity vs the legacy oracle.
//!
//! Every engine's per-layer forward loop must run the same per-layer
//! sequence as the legacy `kv_prefill_run` / `kv_decode_step_run`
//! reference: attention → FFN → `apply_per_layer_embedding` →
//! `apply_layer_scalar`. The synthetic E2B-like fixture carries
//! seeded non-zero PLE tensors and a non-identity `layer_scalar`
//! (0.75), so an engine loop that drops either step diverges
//! bit-visibly at prefill AND at every decode step — these tests fail
//! on any regression of that sequence, not just on missing tensors.
//!
//! BLAS on Windows has non-deterministic reduction order across
//! successive matmul calls, so bit-for-bit parity between two forward
//! paths doesn't hold there; gate on `not(windows)` like
//! `dispatch_parity.rs`.

#![cfg(not(windows))]

use larql_compute::CpuBackend;
use larql_inference::ffn::BackendFfn;
use larql_inference::forward::NoopHook;
use larql_inference::test_utils::make_synthetic_e2b_like_weights;
use larql_kv::generation::{kv_decode_step_run, kv_prefill_run};
use larql_kv::EngineKind;
use ndarray::Array2;

/// Prompt driven through every engine. Token ids < the fixture's
/// vocab_size (32).
const PLE_PARITY_PROMPT: [u32; 3] = [0, 1, 2];

/// Decode steps per test — several steps deep because the issue-#98
/// signature was "first token fine, later tokens garbage".
const PLE_PARITY_DECODE_STEPS: usize = 3;

/// First decode token id; steps use consecutive ids after it.
const FIRST_DECODE_TOKEN: u32 = 3;

/// Window for the windowed engines, large enough that nothing is
/// evicted/archived during prompt + decode steps — keeps the engine
/// exactly equivalent to the unwindowed legacy reference.
const NO_EVICTION_WINDOW: usize = 64;

/// The boundary-per-layer engine spec must carry the fixture's layer
/// count (its `layers=` param defaults to Gemma 3 4B's 34).
const E2B_FIXTURE_NUM_LAYERS: usize = 4;

fn assert_bits_eq(a: &Array2<f32>, b: &Array2<f32>, ctx: &str) {
    assert_eq!(a.shape(), b.shape(), "{ctx}: shape mismatch");
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "{ctx}: element {i} differs ({x} vs {y})"
        );
    }
}

/// Legacy reference forward on the E2B fixture: prefill hidden plus
/// `PLE_PARITY_DECODE_STEPS` decode-step hiddens.
fn legacy_reference(weights: &larql_inference::ModelWeights) -> (Array2<f32>, Vec<Array2<f32>>) {
    let ffn = BackendFfn {
        weights,
        backend: &CpuBackend,
    };
    let (h_prefill, mut cache) = kv_prefill_run(
        larql_inference::WeightsView::dense(weights),
        &ffn,
        &PLE_PARITY_PROMPT,
        None,
        Some(&CpuBackend),
        &mut NoopHook,
    )
    .expect("legacy PLE prefill");
    let mut decode_hiddens = Vec::with_capacity(PLE_PARITY_DECODE_STEPS);
    for step in 0..PLE_PARITY_DECODE_STEPS {
        let token = FIRST_DECODE_TOKEN + step as u32;
        let h = kv_decode_step_run(
            weights,
            &ffn,
            &mut cache,
            token,
            Some(&CpuBackend),
            &mut NoopHook,
        )
        .expect("legacy PLE decode");
        decode_hiddens.push(h);
    }
    (h_prefill, decode_hiddens)
}

/// Drive `spec` through prefill + decode on the E2B fixture and return
/// the same (prefill hidden, decode hiddens) shape as the reference.
fn engine_forward(
    spec: &str,
    weights: &larql_inference::ModelWeights,
) -> (Array2<f32>, Vec<Array2<f32>>) {
    let ffn = BackendFfn {
        weights,
        backend: &CpuBackend,
    };
    let kind = EngineKind::from_name(spec)
        .unwrap_or_else(|| panic!("spec {spec:?} no longer parses — update this test"));
    let mut engine = kind.build(larql_inference::cpu_engine_backend());
    let h_prefill = engine
        .prefill(weights, &ffn, &PLE_PARITY_PROMPT)
        .unwrap_or_else(|e| panic!("{spec}: prefill failed: {e:?}"));
    let mut decode_hiddens = Vec::with_capacity(PLE_PARITY_DECODE_STEPS);
    for step in 0..PLE_PARITY_DECODE_STEPS {
        let token = FIRST_DECODE_TOKEN + step as u32;
        let h = engine
            .decode_step(weights, &ffn, token)
            .unwrap_or_else(|e| panic!("{spec}: decode step {step} failed: {e:?}"));
        decode_hiddens.push(h);
    }
    (h_prefill, decode_hiddens)
}

/// Bit-exact engine families: their prefill and decode paths run the
/// identical attention + FFN kernels as the legacy loops, so parity is
/// exact once the PLE + layer_scalar tail is applied.
#[test]
fn exact_engines_match_legacy_on_ple_arch() {
    let weights = make_synthetic_e2b_like_weights();
    let (h_ref, decode_ref) = legacy_reference(&weights);

    let specs = [
        "markov-rs".to_string(),
        "markov-rs-codec".to_string(),
        format!("boundary-per-layer:layers={E2B_FIXTURE_NUM_LAYERS}"),
    ];
    for spec in &specs {
        let (h_eng, decode_eng) = engine_forward(spec, &weights);
        assert_bits_eq(&h_eng, &h_ref, &format!("{spec}: PLE prefill"));
        for (step, (e, r)) in decode_eng.iter().zip(decode_ref.iter()).enumerate() {
            assert_bits_eq(e, r, &format!("{spec}: PLE decode step {step}"));
        }
    }
}

/// unlimited-context processes the prompt one token at a time (its
/// window-extend loop is decode-shaped), so its matmuls are single-row
/// where the legacy prefill batches all rows — BLAS accumulation order
/// legitimately differs by a few ulp (observed ≤ 2.4e-7 relative L2 on
/// this fixture). Dropping PLE / layer_scalar deviates by ≥ 0.76
/// relative L2 (measured with the tail removed), five orders of
/// magnitude beyond this bound, so the tolerance still pins the bug
/// class hard.
const UNLIMITED_CONTEXT_ACCUM_ORDER_REL_TOL: f32 = 1e-5;

#[test]
fn windowed_checkpoint_matches_legacy_within_accum_order_on_ple_arch() {
    let weights = make_synthetic_e2b_like_weights();
    let (h_ref, decode_ref) = legacy_reference(&weights);
    let spec = format!("unlimited-context:window={NO_EVICTION_WINDOW}");
    let (h_eng, decode_eng) = engine_forward(&spec, &weights);

    let check = |e: &Array2<f32>, r: &Array2<f32>, ctx: &str| {
        assert_eq!(e.shape(), r.shape(), "{ctx}: shape mismatch");
        let rel = rel_l2(e, r);
        assert!(
            rel <= UNLIMITED_CONTEXT_ACCUM_ORDER_REL_TOL,
            "{ctx}: relative L2 deviation {rel} exceeds accumulation-order \
             tolerance {UNLIMITED_CONTEXT_ACCUM_ORDER_REL_TOL}"
        );
    };
    check(&h_eng, &h_ref, &format!("{spec}: PLE prefill"));
    for (step, (e, r)) in decode_eng.iter().zip(decode_ref.iter()).enumerate() {
        check(e, r, &format!("{spec}: PLE decode step {step}"));
    }
}

/// turbo-quant prefill runs attention on the exact f32 hidden state
/// (compression happens after the K/V capture), so its prefill is
/// bit-exact against the legacy reference on this fixture.
#[test]
fn turbo_quant_prefill_matches_legacy_on_ple_arch() {
    let weights = make_synthetic_e2b_like_weights();
    let (h_ref, _) = legacy_reference(&weights);
    let (h_eng, _) = engine_forward("turbo-quant", &weights);
    assert_bits_eq(&h_eng, &h_ref, "turbo-quant: PLE prefill");
}

/// turbo-quant decode attends against WHT + Lloyd-Max roundtripped
/// prior K/V — lossy by contract (engine ladder: per-row cos≈0.9954 at
/// 4-bit) — so decode parity is necessarily tolerance-based, not
/// bit-exact. On this tiny fixture (head_dim 4) the codec-only
/// deviation measures ≤ 0.125 relative L2 across 3 decode steps, while
/// dropping the PLE + layer_scalar tail measures ≥ 0.73 (tail removed
/// and re-measured) — the tolerance sits between the two, so it admits
/// the codec and rejects the bug class.
const TURBO_QUANT_DECODE_REL_TOL: f32 = 0.25;

#[test]
fn turbo_quant_decode_tracks_legacy_within_codec_tolerance_on_ple_arch() {
    let weights = make_synthetic_e2b_like_weights();
    let (_, decode_ref) = legacy_reference(&weights);
    let (_, decode_eng) = engine_forward("turbo-quant", &weights);

    for (step, (e, r)) in decode_eng.iter().zip(decode_ref.iter()).enumerate() {
        assert_eq!(e.shape(), r.shape(), "turbo-quant decode step {step}");
        let rel = rel_l2(e, r);
        assert!(
            rel <= TURBO_QUANT_DECODE_REL_TOL,
            "turbo-quant decode step {step}: relative L2 deviation {rel} \
             exceeds codec tolerance {TURBO_QUANT_DECODE_REL_TOL}"
        );
    }
}

/// Relative L2 deviation of `a` against reference `b`.
fn rel_l2(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
    let ref_norm = b.iter().map(|v| v * v).sum::<f32>().sqrt();
    let err_norm = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt();
    err_norm / ref_norm.max(f32::MIN_POSITIVE)
}
