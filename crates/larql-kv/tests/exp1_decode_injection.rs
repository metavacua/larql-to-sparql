//! Experiment Q1 — is `decode_step` provenance-agnostic?
//!
//! Design question (from the kv-cache-vs-residual-stream research residual):
//! the persistent-context agent loop wants to inject externally-produced
//! tokens (a larql command's OUTPUT) into a live decode state and continue.
//! It can only do that if `decode_step` treats an injected token exactly like
//! a model-sampled one — i.e. feeding a token via `decode_step` produces the
//! same state as having that token in the prompt. `decode_step` cannot know a
//! token's provenance; this test measures whether that ignorance is *safe*.
//!
//! Test: the hidden state at the final position after
//!   prefill([t0,t1,t2,t3])
//! must be bit-identical to the state after
//!   prefill([t0,t1]) + decode_step(t2) + decode_step(t3).
//! Bit-identity ⇒ injection ≡ in-prompt ⇒ `decode_step` is provenance-agnostic
//! ⇒ larql already has the "extend the model's running context with new
//! tokens" primitive, with no re-prefill.
//!
//! Synthetic weights (`make_test_weights`) — this is a *mechanism* property,
//! not an architecture property, so no model download is needed and the
//! result is deterministic. Gated `not(windows)` for the same BLAS
//! reduction-order determinism reason as `dispatch_parity.rs`.
//!
//! The CI job's verdict IS the experiment result:
//!   green (test passes) ⇒ Q1 = ACCEPT (injection is sound)
//!   red   (test fails)  ⇒ Q1 = REJECT (injection diverges from in-prompt)

#![cfg(not(windows))]

use larql_compute::CpuBackend;
use larql_inference::ffn::WeightFfn;
use larql_inference::forward::NoopHook;
use larql_inference::test_utils::make_test_weights;
use larql_kv::generation::{kv_decode_step_run, kv_prefill_run};

#[test]
fn decode_step_injection_matches_in_prompt_bit_for_bit() {
    let weights = make_test_weights();
    let backend = CpuBackend;
    let ffn = WeightFfn { weights: &weights };

    // Path FULL — all four tokens in the prompt.
    let full = [0u32, 1, 2, 3];
    let (h_full, _cache_full) = kv_prefill_run(
        larql_inference::WeightsView::dense(&weights),
        &ffn,
        &full,
        None,
        Some(&backend),
        &mut NoopHook,
    )
    .expect("full prefill");
    // Hidden at the final position (the one that predicts the next token).
    let full_last = h_full.row(h_full.nrows() - 1).to_owned();

    // Path INJECT — prefill the first two, then feed the last two through
    // `decode_step` as if they were freshly produced (injected) tokens.
    let (_h_pre, mut cache) = kv_prefill_run(
        larql_inference::WeightsView::dense(&weights),
        &ffn,
        &[0u32, 1],
        None,
        Some(&backend),
        &mut NoopHook,
    )
    .expect("prefix prefill");
    let _h2 = kv_decode_step_run(&weights, &ffn, &mut cache, 2, Some(&backend), &mut NoopHook)
        .expect("decode injected t2");
    let h3 = kv_decode_step_run(&weights, &ffn, &mut cache, 3, Some(&backend), &mut NoopHook)
        .expect("decode injected t3");
    let inject_last = h3.row(h3.nrows() - 1).to_owned();

    assert_eq!(
        full_last, inject_last,
        "decode_step injection must equal in-prompt state bit-for-bit \
         (⇒ decode_step is provenance-agnostic; token injection is sound)"
    );
}
