//! Regression guard for a cross-backend divergence on the Gemma-3 arch.
//!
//! **History, because the diagnosis was wrong twice before it was right.**
//! Metal's batched prefill used to disagree with both the CPU and Metal's
//! own iterative path by 23-43% on the Gemma-3 fixture, growing with
//! depth, absent at one token, and not reproducing on a real Gemma 3 4B.
//!
//! It was first attributed to per-layer sliding-window attention. Wrong:
//! the fixture resolves to no window on either backend. It was then
//! attributed to a `head_dim` shape assumption, on the reasoning that
//! shape was the only surviving difference from the real model. Also
//! wrong, and that one was never tested — a sweep across
//! `head_dim ∈ {32, 64, 128, 256, 512}` diverged at *every* shape,
//! including the real model's 256.
//!
//! The cause was the fixture: it declared Gemma-3's QK-norm keys and
//! never populated the weights. Every consumer that resolves the weight
//! got `None`, and the two backends did not agree on what a
//! declared-but-absent QK-norm weight means. Real checkpoints always
//! carry those weights, which is why nothing shipped was ever affected.
//!
//! The fixture now supplies them (`larql-models::test_fixtures`), and
//! this file pins the agreement so neither half can regress: not the
//! fixture back to an impossible architecture, and not a backend into
//! disagreeing about the stage.

use larql_inference::ffn::NullFfn;
use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};

use super::support::{build, relative_l2, PREFILL_TOL};

/// Longest prompt swept. The divergence appeared from two positions on,
/// so anything past one token exercises it.
const MAX_PROMPT_LEN: usize = 6;

/// The fixture must actually carry the architecture it claims. A
/// Gemma-3 fixture without QK-norm weights is not a Gemma-3 fixture, and
/// its absence is what let two backends disagree unnoticed.
#[test]
fn the_gemma3_fixture_carries_the_qk_norm_weights_it_declares() {
    let weights = make_test_q4k_weights();
    let arch = &*weights.arch;
    for layer in 0..weights.num_layers {
        for (label, key) in [
            ("q_norm", arch.attn_q_norm_key(layer)),
            ("k_norm", arch.attn_k_norm_key(layer)),
        ] {
            let Some(key) = key else {
                panic!("layer {layer}: Gemma-3 must declare a {label} key");
            };
            assert!(
                weights.vectors.contains_key(&key),
                "layer {layer}: {label} is declared as {key:?} but absent — the \
                 fixture is claiming an architecture it does not carry"
            );
        }
    }
}

/// Metal's batched prefill and the CPU must agree on the Gemma-3 arch at
/// every prompt length. This is the comparison that used to fail.
#[test]
fn metal_batched_prefill_agrees_with_cpu_on_gemma3_arch() {
    let weights = make_test_q4k_weights();
    let index = make_test_q4k_vindex(&weights);

    for n in 1..=MAX_PROMPT_LEN {
        let prompt: Vec<u32> = (0..n as u32).collect();
        let mut gpu = build("standard", true);
        let mut cpu = build("standard", false);
        let hg = gpu
            .prefill_quant(
                &weights,
                &NullFfn,
                &index,
                &prompt,
                &*larql_compute::default_backend(),
            )
            .expect("gpu prefill");
        let hc = cpu
            .prefill_quant(
                &weights,
                &NullFfn,
                &index,
                &prompt,
                &*larql_compute::cpu_backend(),
            )
            .expect("cpu prefill");
        let rel = relative_l2(&hg, &hc);
        assert!(
            rel < PREFILL_TOL,
            "prompt len {n}: backends diverged by relative L2 {rel:.3e} — this is \
             the regression this file exists to catch (it used to read ~4e-1)"
        );
    }
}
