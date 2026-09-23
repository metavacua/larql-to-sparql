//! The GPU path must produce the same hidden state as the CPU path.

use larql_inference::ffn::NullFfn;
use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights_silu};

use super::support::{all_specs, assert_close, build, COMPOUNDING_FACTOR, DECODE_TOL, PREFILL_TOL};

/// Prompt used for the cross-backend comparison.
const PROMPT: [u32; 4] = [0, 1, 2, 3];
/// First decode token id; decode sweeps `DECODE_FIRST_TOKEN..+DECODE_STEPS`.
const DECODE_FIRST_TOKEN: u32 = 4;
/// Enough steps for compounding drift to show if it exists.
const DECODE_STEPS: u32 = 10;

/// The property that matters: routing an engine through the default
/// (GPU-capable) backend must not change what the model says.
///
/// Tolerance rather than bit-equality: the coarse pipeline fuses
/// operations the per-layer CPU path runs separately, so reassociation
/// is expected. A real divergence — wrong K/V rows, a shifted RoPE
/// position, a dropped window — moves the state far more than this.
///
/// Uses the `tinymodel` fixture for the sweep across engines, and
/// [`super::gemma3_prefill_gap`] covers the Gemma-3 arch separately. The
/// Gemma fixture was excluded here while a cross-backend divergence was
/// open on it; that turned out to be the fixture declaring QK-norm
/// without supplying the weights, is fixed, and is now pinned by a
/// regression test rather than worked around.
#[test]
fn gpu_and_cpu_backends_agree_on_hidden_state() {
    let weights = make_test_q4k_weights_silu();
    let index = make_test_q4k_vindex(&weights);

    for spec in all_specs() {
        let mut gpu = build(spec, true);
        let mut cpu = build(spec, false);

        let h_gpu = gpu
            .prefill_quant(
                &weights,
                &NullFfn,
                &index,
                &PROMPT,
                &*larql_compute::default_backend(),
            )
            .unwrap_or_else(|e| panic!("{spec}: gpu prefill_quant failed: {e}"));
        let h_cpu = cpu
            .prefill_quant(
                &weights,
                &NullFfn,
                &index,
                &PROMPT,
                &*larql_compute::cpu_backend(),
            )
            .unwrap_or_else(|e| panic!("{spec}: cpu prefill_quant failed: {e}"));

        assert_eq!(
            h_gpu.shape(),
            h_cpu.shape(),
            "{spec}: prefill shape differs"
        );
        assert_close(&h_gpu, &h_cpu, PREFILL_TOL, spec, "prefill");

        // Decode divergence must stay BOUNDED and must not compound. A
        // fixed per-step difference is two Q4K kernels rounding
        // differently; a growing one means the two caches are drifting
        // apart, which is the failure that matters and which a
        // single-step check cannot see.
        let mut rels = Vec::new();
        for (step, tok) in (DECODE_FIRST_TOKEN..DECODE_FIRST_TOKEN + DECODE_STEPS).enumerate() {
            let d_gpu = gpu
                .decode_step_quant(
                    &weights,
                    &NullFfn,
                    &index,
                    tok,
                    &*larql_compute::default_backend(),
                )
                .unwrap_or_else(|e| panic!("{spec}: gpu decode {step} failed: {e}"));
            let d_cpu = cpu
                .decode_step_quant(
                    &weights,
                    &NullFfn,
                    &index,
                    tok,
                    &*larql_compute::cpu_backend(),
                )
                .unwrap_or_else(|e| panic!("{spec}: cpu decode {step} failed: {e}"));
            rels.push(assert_close(
                &d_gpu,
                &d_cpu,
                DECODE_TOL,
                spec,
                &format!("decode step {step}"),
            ));
        }

        let first = rels[0].max(f32::MIN_POSITIVE);
        let last = *rels.last().expect("decode ran at least one step");
        assert!(
            last <= first * COMPOUNDING_FACTOR,
            "{spec}: cross-backend divergence is compounding across decode \
             steps ({first:.2e} → {last:.2e}); the two K/V caches are drifting \
             apart rather than just rounding differently"
        );
    }
}
