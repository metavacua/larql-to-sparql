//! Tests for the cached CPU decode path.
//!
//! ## What was actually missing
//!
//! The sibling `kquant_forward/mod.rs` suite already drives most of this
//! file's public surface, so "zero tests" (ROADMAP H3) understates one
//! thing and overstates another. Every one of those tests asserts a
//! *shape* — `h.shape() == [1, hidden]`, "the call must complete without
//! panic", "whether or not the direct path engaged, the function
//! shouldn't panic". None asserts a *value*.
//!
//! That is exactly the gap the §4.10 RoPE defect fell through. Metal
//! decode roped from `rope_base` alone and honoured no scaling family
//! while prefill honoured all of them; the outputs stayed the right
//! shape the whole time. A shape-only suite cannot see it. The tests
//! here are therefore built around **agreement**, not execution:
//!
//! 1. `prefill(N)` last row vs `prefill(N-1)` + `decode(token N)` — the
//!    CPU-side analogue of ROADMAP M5, on a fixture whose last layer
//!    carries a position divisor of 8.
//! 2. A fixture-adequacy guard, so the agreement test cannot pass
//!    vacuously on a fixture where scaled RoPE never fires (a fixture
//!    too small to distinguish the candidate behaviours is an *absent*
//!    test, not a weak one).
//! 3. A discriminating control: the same decode at the wrong absolute
//!    position must miss the bar by orders of magnitude, so we know the
//!    bar can fail.
//!
//! Plus the two branches the mod.rs suite leaves uncovered: the
//! padded-intermediate refusal in `layer_supports_direct_matvec`, and
//! the guard clauses of `matvec_q4k_or_q6k_q8k`.

// Quantisation helpers these tests drive directly.
use crate::cpu::ops::q4k_q8k_dot::{quantize_x_to_q8k_into, Q8KActivation};

use super::*;
use crate::test_fixtures::{make_q4k_fixture_index, Q4kFixtureIndex};
use larql_models::test_fixtures::{make_test_q4k_weights, make_test_q4k_weights_rope_scaled};

/// Prompt used by the agreement tests. Eight positions over a six-layer
/// fixture puts the split (7 prefilled + 1 decoded) well inside the
/// model and past the layer-5 global layer.
const AGREEMENT_TOKENS: [u32; 8] = [3, 11, 27, 5, 61, 2, 44, 19];

/// Relative L2 error between two equal-shaped rows, as a fraction of the
/// reference's norm. The parity statistic for a kernel-route swap: the
/// object is the decode step's output residual and the operation
/// replaces the kernel that produces it, so the question is how far the
/// whole row moved, not how any one element rounded.
fn rel_l2(actual: &Array2<f32>, reference: &Array2<f32>) -> f64 {
    assert_eq!(actual.shape(), reference.shape(), "shape mismatch");
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (a, r) in actual.iter().zip(reference.iter()) {
        let d = (*a as f64) - (*r as f64);
        num += d * d;
        den += (*r as f64) * (*r as f64);
    }
    if den == 0.0 {
        return num.sqrt();
    }
    (num / den).sqrt()
}

/// Last row of a `[seq_len, hidden]` prefill result.
fn last_row(h: &Array2<f32>) -> Array2<f32> {
    let n = h.shape()[0];
    h.slice(ndarray::s![n - 1..n, ..]).to_owned()
}

// ── Fixture adequacy ────────────────────────────────────────────────
//
// Runs first because everything below depends on it. If the rope-scaled
// fixture's layers all carried the same position divisor, the agreement
// tests would pass on a model that never exercises the branch they exist
// to guard.

/// The rope-scaled Q4_K fixture must actually contain a layer whose
/// position divisor differs from the rest — otherwise the agreement
/// tests below are guarding a path the fixture never takes.
#[test]
fn rope_scaled_fixture_contains_a_scaled_global_layer() {
    let weights = make_test_q4k_weights_rope_scaled();
    let arch = &*weights.arch;

    let divisors: Vec<f64> = (0..weights.num_layers)
        .map(|l| crate::forward_overrides::effective_rope_position_divisor_for_layer(arch, l))
        .collect();

    // Gemma 3's 5:1 local:global pattern over six layers puts the single
    // full-attention layer last, and `rope_scaling.factor = 8` applies
    // only there.
    assert_eq!(
        divisors[5], 8.0,
        "layer 5 should be a global layer carrying the linear factor; got {divisors:?}"
    );
    assert_eq!(
        divisors[0], 1.0,
        "sliding layers must not take the factor; got {divisors:?}"
    );
    assert!(
        arch.is_sliding_window_layer(0) && !arch.is_sliding_window_layer(5),
        "fixture must mix sliding and global layers"
    );
}

// ── Prefill/decode agreement — the M5 analogue ──────────────────────

/// `prefill(N)` and `prefill(N-1) + decode(token N)` must produce the
/// same final residual.
///
/// This is the assertion that would have caught §4.10 on the CPU side.
/// The two routes rope independently: prefill rotates every position in
/// one host-side pass, the decode step rotates a single position from
/// `abs_position`. A divisor, base or scaling-family mismatch between
/// them changes the layer-5 rotation angle by 8× and shows up here.
///
/// The bar is calibrated, not guessed. The two routes are *not* the same
/// kernel — prefill projects the FFN straight from Q4_K bytes against
/// f32 activations while the direct decode step quantises its activation
/// to Q8_K first — so they disagree at the activation-quantisation
/// floor, ~1e-3 relative. `decode_at_wrong_position_misses_the_bar`
/// below measures what a position error costs by comparison.
#[test]
fn direct_decode_step_agrees_with_prefill_on_rope_scaled_fixture() {
    let weights = make_test_q4k_weights_rope_scaled();
    let idx = make_q4k_fixture_index(&weights);
    let backend = crate::CpuBackend;

    let split = AGREEMENT_TOKENS.len() - 1;
    let (h_full, _, _) = predict_kquant_prefill(&weights, &AGREEMENT_TOKENS, &idx);
    let reference = last_row(&h_full);

    let (_, mut cache, _) = predict_kquant_prefill(&weights, &AGREEMENT_TOKENS[..split], &idx);
    let decoded = predict_kquant_decode_step_direct(
        &weights,
        AGREEMENT_TOKENS[split],
        &idx,
        &backend,
        &mut cache,
        split,
    )
    .expect("direct decode step engages on the Q4_K fixture");

    let err = rel_l2(&decoded, &reference);
    assert!(
        err < DIRECT_DECODE_AGREEMENT_BAR,
        "direct decode diverged from prefill: relative L2 {err:.3e} \
         exceeds {DIRECT_DECODE_AGREEMENT_BAR:.0e} — check the RoPE plan \
         (position divisor, base, scaling family) before the kernel"
    );
}

/// Same agreement for the staged decode step, which dequantises to f32
/// per layer instead of running the direct matvec. Both decode routes
/// have to track prefill; only one of them was covered by any assertion
/// before this file.
#[test]
fn staged_decode_step_agrees_with_prefill_on_rope_scaled_fixture() {
    let weights = make_test_q4k_weights_rope_scaled();
    let idx = make_q4k_fixture_index(&weights);

    let split = AGREEMENT_TOKENS.len() - 1;
    let (h_full, _, _) = predict_kquant_prefill(&weights, &AGREEMENT_TOKENS, &idx);
    let reference = last_row(&h_full);

    let (_, mut cache, _) = predict_kquant_prefill(&weights, &AGREEMENT_TOKENS[..split], &idx);
    let (decoded, _) =
        predict_kquant_decode_step(&weights, AGREEMENT_TOKENS[split], &idx, &mut cache, split)
            .expect("staged decode step succeeds with a populated cache");

    let err = rel_l2(&decoded, &reference);
    assert!(
        err < STAGED_DECODE_AGREEMENT_BAR,
        "staged decode diverged from prefill: relative L2 {err:.3e} \
         exceeds {STAGED_DECODE_AGREEMENT_BAR:.0e}"
    );
}

/// Agreement has to survive *repeated* decode, not just the first step.
/// A per-step position error that a single step hides (position 7 vs 7)
/// compounds once the cache carries rows written at the wrong angle.
#[test]
fn repeated_direct_decode_agrees_with_prefill() {
    let weights = make_test_q4k_weights_rope_scaled();
    let idx = make_q4k_fixture_index(&weights);
    let backend = crate::CpuBackend;

    const PREFILLED: usize = 4;
    let (h_full, _, _) = predict_kquant_prefill(&weights, &AGREEMENT_TOKENS, &idx);
    let reference = last_row(&h_full);

    let (_, mut cache, _) = predict_kquant_prefill(&weights, &AGREEMENT_TOKENS[..PREFILLED], &idx);
    let mut decoded = None;
    for (offset, &token) in AGREEMENT_TOKENS[PREFILLED..].iter().enumerate() {
        decoded = Some(
            predict_kquant_decode_step_direct(
                &weights,
                token,
                &idx,
                &backend,
                &mut cache,
                PREFILLED + offset,
            )
            .expect("direct decode step engages on the Q4_K fixture"),
        );
    }

    let err = rel_l2(&decoded.expect("at least one decode step"), &reference);
    assert!(
        err < DIRECT_DECODE_AGREEMENT_BAR,
        "four chained decode steps diverged from prefill: relative L2 \
         {err:.3e} exceeds {DIRECT_DECODE_AGREEMENT_BAR:.0e}"
    );
}

/// Discriminating control for both bars, run at the **weakest** defect
/// there is.
///
/// A RoPE-plan defect is precisely "this position was rotated by the
/// wrong angle", so decoding the correct token at the wrong absolute
/// position is the closest in-process proxy for it — the divisor comes
/// from a `OnceLock`-backed env override and cannot be flipped
/// mid-process.
///
/// The perturbation here is off-by-**one**, not off-by-everything.
/// Decoding at position 0 instead of 7 is a much louder signal (2.8e-1)
/// and proves less: a bar that only catches the loudest possible error
/// is not a bar. One position is the smallest wrong answer the RoPE plan
/// can give, so clearing it is the sensitivity claim these tests are
/// entitled to make. Both routes must fail it.
#[test]
fn an_off_by_one_position_misses_both_bars() {
    let weights = make_test_q4k_weights_rope_scaled();
    let idx = make_q4k_fixture_index(&weights);
    let backend = crate::CpuBackend;

    let split = AGREEMENT_TOKENS.len() - 1;
    let (h_full, _, _) = predict_kquant_prefill(&weights, &AGREEMENT_TOKENS, &idx);
    let reference = last_row(&h_full);

    let (_, mut cache, _) = predict_kquant_prefill(&weights, &AGREEMENT_TOKENS[..split], &idx);
    let direct = predict_kquant_decode_step_direct(
        &weights,
        AGREEMENT_TOKENS[split],
        &idx,
        &backend,
        &mut cache,
        split - 1,
    )
    .expect("direct decode step engages on the Q4_K fixture");
    let direct_err = rel_l2(&direct, &reference);
    assert!(
        direct_err > DIRECT_DECODE_AGREEMENT_BAR * 2.0,
        "a one-position RoPE error must clear the direct bar with margin, \
         but relative L2 was {direct_err:.3e} against a bar of \
         {DIRECT_DECODE_AGREEMENT_BAR:.0e} — the direct route can no longer \
         distinguish a rotation-angle defect from its own quantisation floor"
    );

    let (_, mut cache, _) = predict_kquant_prefill(&weights, &AGREEMENT_TOKENS[..split], &idx);
    let (staged, _) = predict_kquant_decode_step(
        &weights,
        AGREEMENT_TOKENS[split],
        &idx,
        &mut cache,
        split - 1,
    )
    .expect("staged decode step succeeds with a populated cache");
    let staged_err = rel_l2(&staged, &reference);
    assert!(
        staged_err > STAGED_DECODE_AGREEMENT_BAR * 2.0,
        "a one-position RoPE error must clear the staged bar with margin, \
         but relative L2 was {staged_err:.3e}"
    );
}

// ── Calibration ─────────────────────────────────────────────────────
//
// Measured on this fixture pair, relative L2 of the final residual row
// (larql-compute, debug, 2026-08-06):
//
// | route  | fixture     | agrees with prefill | off-by-1 position | margin |
// |--------|-------------|---------------------|-------------------|--------|
// | staged | rope-scaled | 3.65e-7             | 1.59e-1           | 4.4e5x |
// | staged | plain       | 3.40e-7             | 2.02e-1           | 5.9e5x |
// | direct | rope-scaled | 2.45e-2             | 1.61e-1           | 6.6x   |
// | direct | plain       | 2.01e-2             | 2.05e-1           | 10.2x  |
//
// Two things follow, and both are worth knowing before trusting these
// tests.
//
// **The 2.4e-2 on the direct route is the kernel, not the RoPE plan.**
// The plain fixture carries no rope_scaling at all and disagrees by the
// same 2.0e-2, so the residual is the Q8_K activation quantisation the
// direct route applies and prefill does not —
// `direct_decode_residual_is_kernel_not_rope_plan` pins that attribution
// so a future RoPE regression cannot be waved off as quantisation noise.
//
// **The direct route is a blunt instrument and the staged one is sharp.**
// Staged decode tracks prefill to 3.7e-7 and a one-position error costs
// 1.6e-1, so its bar has five orders of headroom. The direct route —
// the production path — has 6.6x. It will catch a wrong position,
// divisor, base or scaling family, because all four move the residual by
// >1.6e-1. It would *not* catch a rope defect worth less than ~2.5x the
// quantisation floor. The sharp CPU-side parity gate is the staged test;
// the direct test is the one that covers the code that actually ships.
//
// Bars are set at the geometric mean of floor and weakest defect, so
// each has comparable headroom in both directions.

/// Agreement bar for the direct-matvec decode route (the production
/// path). Floor 2.45e-2, weakest defect 1.61e-1; sqrt of the product is
/// 6.3e-2.
const DIRECT_DECODE_AGREEMENT_BAR: f64 = 6e-2;

/// Agreement bar for the staged (dequant-to-f32) decode route. Floor
/// 3.7e-7, weakest defect 1.6e-1 — five orders apart, so this bar sits
/// well clear of both rather than at the geometric mean.
const STAGED_DECODE_AGREEMENT_BAR: f64 = 1e-5;

/// Attribution for the direct route's 2.4e-2 floor.
///
/// A fixture with no `rope_scaling` whatsoever must show the same
/// disagreement as the rope-scaled one. If it did not, the residual
/// would be evidence about the RoPE plan and the bar above would be
/// hiding a real defect behind a "quantisation noise" label.
#[test]
fn direct_decode_residual_is_kernel_not_rope_plan() {
    let split = AGREEMENT_TOKENS.len() - 1;
    let mut errs = Vec::new();
    for weights in [make_test_q4k_weights_rope_scaled(), make_test_q4k_weights()] {
        let idx = make_q4k_fixture_index(&weights);
        let backend = crate::CpuBackend;
        let (h_full, _, _) = predict_kquant_prefill(&weights, &AGREEMENT_TOKENS, &idx);
        let reference = last_row(&h_full);
        let (_, mut cache, _) = predict_kquant_prefill(&weights, &AGREEMENT_TOKENS[..split], &idx);
        let decoded = predict_kquant_decode_step_direct(
            &weights,
            AGREEMENT_TOKENS[split],
            &idx,
            &backend,
            &mut cache,
            split,
        )
        .expect("direct decode step engages on the Q4_K fixture");
        errs.push(rel_l2(&decoded, &reference));
    }

    let (scaled, plain) = (errs[0], errs[1]);
    assert!(
        (scaled / plain) < 4.0 && (plain / scaled) < 4.0,
        "the direct route's disagreement with prefill should be the same \
         order with and without rope_scaling (it is the Q8_K activation \
         floor), but scaled={scaled:.3e} vs unscaled={plain:.3e} — if these \
         have separated, the residual is about the RoPE plan and the \
         agreement bar is masking it"
    );
}

// ── Capability predicates ───────────────────────────────────────────

/// `KvIndex` wrapper that reports a caller-chosen `num_features` while
/// delegating every byte accessor to a real fixture index. Isolates the
/// intermediate-width branch of `layer_supports_direct_matvec` from the
/// format branches, which the mod.rs suite already covers.
struct RestatedIntermediate<'a> {
    inner: &'a Q4kFixtureIndex,
    intermediate: usize,
}

impl crate::KvIndex for RestatedIntermediate<'_> {
    fn num_features(&self, _layer: usize) -> usize {
        self.intermediate
    }
    fn attn_kquant_layer_data(&self, layer: usize) -> Option<[(&[u8], &str); 4]> {
        self.inner.attn_kquant_layer_data(layer)
    }
    fn interleaved_kquant_layer_data(
        &self,
        layer: usize,
    ) -> Option<[(&[u8], &str); crate::kv_index::FFN_COMPONENTS_PER_LAYER]> {
        self.inner.interleaved_kquant_layer_data(layer)
    }
    fn interleaved_kquant_mmap_ref(&self) -> Option<&[u8]> {
        self.inner.interleaved_kquant_mmap_ref()
    }
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }
}

/// The padded-down-projection refusal. When the FFN's intermediate width
/// is not a 256-multiple the direct matvec would reject `cols` and
/// silently zero its output, so the layer must decline the direct path
/// and let the dequant fallback run.
#[test]
fn layer_supports_direct_matvec_refuses_padded_intermediate() {
    let weights = make_test_q4k_weights();
    let fixture = make_q4k_fixture_index(&weights);

    let padded = RestatedIntermediate {
        inner: &fixture,
        // One short of the 256-multiple the fixture really has.
        intermediate: weights.intermediate_size - 1,
    };
    assert!(
        !layer_supports_direct_matvec(&padded, 0),
        "a non-256-multiple intermediate must decline the direct path"
    );
    assert!(
        !supports_direct_matvec_decode(&weights, &padded),
        "the whole-model gate must inherit the per-layer refusal"
    );
}

/// Positive twin of the test above: the same wrapper reporting the
/// fixture's real width qualifies, so the refusal is attributable to the
/// width and not to the wrapper.
#[test]
fn layer_supports_direct_matvec_accepts_aligned_intermediate() {
    let weights = make_test_q4k_weights();
    let fixture = make_q4k_fixture_index(&weights);

    let aligned = RestatedIntermediate {
        inner: &fixture,
        intermediate: weights.intermediate_size,
    };
    assert!(layer_supports_direct_matvec(&aligned, 0));
    assert!(supports_direct_matvec_decode(&weights, &aligned));
}

/// The Q4_K fixture qualifies for the direct path outright. The sibling
/// `supports_direct_matvec_decode_inspects_fixture` in mod.rs binds the
/// result to `let _: bool` and asserts nothing — the same shape as the
/// H1 predecessor test that called its subject twice and checked no
/// output. Pin the value here so a routing regression is visible.
#[test]
fn supports_direct_matvec_decode_accepts_the_q4k_fixture() {
    let weights = make_test_q4k_weights();
    let idx = make_q4k_fixture_index(&weights);
    assert!(supports_cached_decode(&weights));
    assert!(supports_direct_matvec_decode(&weights, &idx));
}

// ── `matvec_q4k_or_q6k_q8k` guard clauses ───────────────────────────

/// Q8_K-quantise `x` for the matvec helpers below.
fn q8k(x: &[f32]) -> Q8KActivation {
    let mut a = Q8KActivation::with_capacity(x.len());
    quantize_x_to_q8k_into(&mut a, x);
    a
}

#[test]
fn matvec_q4k_returns_zero_row_for_empty_shapes() {
    let x = q8k(&[0.0f32; 256]);
    assert_eq!(matvec_q4k_or_q6k_q8k(&[], "Q4_K", &x, 0, 256), Some(vec![]));
    assert_eq!(
        matvec_q4k_or_q6k_q8k(&[], "Q4_K", &x, 4, 0),
        Some(vec![0.0; 4])
    );
}

#[test]
fn matvec_q4k_declines_unaligned_columns() {
    let x = q8k(&[0.0f32; 256]);
    // 255 is not a super-block multiple; the kernel would read a partial
    // block, so the helper must decline rather than truncate.
    assert!(matvec_q4k_or_q6k_q8k(&[0u8; 144], "Q4_K", &x, 1, 255).is_none());
}

#[test]
fn matvec_q4k_declines_formats_without_a_q8k_kernel() {
    let x = q8k(&[0.0f32; 256]);
    for tag in ["Q8_0", "BF16", "F32", "I2_S", "not-a-format"] {
        assert!(
            matvec_q4k_or_q6k_q8k(&[0u8; 4096], tag, &x, 1, 256).is_none(),
            "{tag} has no Q8_K matvec kernel and must decline"
        );
    }
}

#[test]
fn matvec_q4k_declines_truncated_byte_slices() {
    let x = q8k(&[0.0f32; 256]);
    // Two rows claimed, one row's worth of bytes present.
    assert!(matvec_q4k_or_q6k_q8k(&[0u8; 144], "Q4_K", &x, 2, 256).is_none());
}

/// The kernel itself, against a dequantise-then-dot reference on the
/// same bytes. Bounds how much of the residual in the agreement tests
/// above belongs to the matvec rather than to the RoPE plan.
#[test]
fn matvec_q4k_matches_dequantised_reference() {
    use crate::cpu::ops::q4_common::{dequantize_q4_k, quantize_q4_k};

    const ROWS: usize = 8;
    const COLS: usize = 256;
    let w: Vec<f32> = (0..ROWS * COLS)
        .map(|i| ((i % 37) as f32 - 18.0) / 64.0)
        .collect();
    let x: Vec<f32> = (0..COLS).map(|i| ((i % 11) as f32 - 5.0) / 32.0).collect();

    let bytes = quantize_q4_k(&w);
    let out = matvec_q4k_or_q6k_q8k(&bytes, "Q4_K", &q8k(&x), ROWS, COLS)
        .expect("Q4_K × Q8_K matvec is available");

    // Reference: dequantise the same bytes and dot in f32, so the only
    // difference is the Q8_K activation quantisation the kernel applies.
    let w_deq = dequantize_q4_k(&bytes, ROWS * COLS);
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for r in 0..ROWS {
        let expected: f64 = (0..COLS)
            .map(|c| w_deq[r * COLS + c] as f64 * x[c] as f64)
            .sum();
        num += (out[r] as f64 - expected).powi(2);
        den += expected * expected;
    }
    let err = (num / den).sqrt();
    assert!(
        err < 1e-2,
        "Q4_K × Q8_K matvec drifted from the dequantised reference: {err:.3e}"
    );
}

// ── `vec_to_2d_row` ─────────────────────────────────────────────────

#[test]
fn vec_to_2d_row_produces_a_single_row() {
    let row = vec_to_2d_row(vec![1.0, 2.0, 3.0]);
    assert_eq!(row.shape(), &[1, 3]);
    assert_eq!(row[[0, 2]], 3.0);
}

#[test]
fn vec_to_2d_row_handles_the_empty_vector() {
    assert_eq!(vec_to_2d_row(Vec::new()).shape(), &[1, 0]);
}
