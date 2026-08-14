//! **R8-policy — R8a–R8d.** Mixed sliding/global retention over a cache
//! with a positional hole in it.
//!
//! R8-storage proved the row map stays congruent. These prove the map is
//! *used correctly*: that a window means logical age on the layers that
//! declare one, and means nothing at all on the layers that do not.
//!
//! The fixture is `make_gemma3_narrow_window_test_weights` — six layers
//! on Gemma-3's 5-sliding-then-1-global repeat, with a declared window of
//! 4 tokens. Small enough that every retained row can be named.

use larql_inference::kv_engine::KvEngine;
use larql_inference::larql_models::test_fixtures::make_gemma3_narrow_window_test_weights;
use larql_inference::model::ModelWeights;

use crate::engines::semantic_promotion::checkpoint::BoundaryCheckpoint;
use crate::engines::semantic_promotion::exclusion::LayerAttention;
use crate::engines::semantic_promotion::ids::CheckpointId;
use crate::engines::semantic_promotion::window_policy::apply_window_policy;
use crate::engines::standard::StandardEngine;

const PREFIX: [u32; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
/// The fixture's declared sliding window.
const WINDOW: u64 = 4;

fn ffn(weights: &ModelWeights) -> larql_compute::ffn::WeightFfn<'_> {
    larql_compute::ffn::WeightFfn { weights }
}

/// An engine with a contiguous eight-row cache and **no engine-level
/// window**, so nothing has been clipped uniformly and the policy is the
/// only thing that removes a row.
fn prefilled(weights: &ModelWeights) -> StandardEngine {
    let mut e = StandardEngine::with_backend(None, larql_inference::cpu_engine_backend());
    e.prefill(weights, &ffn(weights), &PREFIX).expect("prefill");
    e
}

/// Wind the cache forward by `gap` positions without decoding them, so
/// the resident rows sit `gap` positions behind the next token.
fn open_a_hole(weights: &ModelWeights, e: &mut StandardEngine, gap: usize) -> usize {
    let kv = e.per_layer_kv_mut().expect("per-layer access");
    let checkpoint =
        BoundaryCheckpoint::capture(CheckpointId::from_counter(9), kv).expect("capture");
    let target = checkpoint.logical_next_position + gap;
    checkpoint
        .restore_for_offset_replay(kv, target)
        .expect("offset restore");
    e.decode_step(weights, &ffn(weights), 20).expect("decode");
    target
}

fn positions(e: &mut StandardEngine, layer: usize) -> Vec<u64> {
    e.per_layer_kv_mut()
        .unwrap()
        .row_positions()
        .unwrap()
        .layer(layer)
        .unwrap()
        .as_slice()
        .to_vec()
}

/// The fixture's classification, read off the architecture rather than
/// assumed — if the repeat pattern changes, these gates follow it.
fn classification(weights: &ModelWeights) -> (LayerAttention, usize, usize) {
    let attention = LayerAttention::from_weights(weights);
    let sliding = attention
        .is_sliding
        .iter()
        .position(|&s| s)
        .expect("fixture has a sliding layer");
    let global = attention
        .is_sliding
        .iter()
        .position(|&s| !s)
        .expect("fixture has a global layer");
    (attention, sliding, global)
}

/// **R8a — gap > window.** Every pre-hole row is outside the window by
/// thousands of positions, so a sliding layer must retain none of them.
///
/// This is the R7b failure stated as a requirement: the physical clip
/// keeps four rows here, and four is the wrong answer.
#[test]
fn r8a_no_pre_hole_row_survives_on_a_sliding_layer() {
    let weights = make_gemma3_narrow_window_test_weights();
    let (attention, sliding, _) = classification(&weights);
    let mut e = prefilled(&weights);
    let far = open_a_hole(&weights, &mut e, 4_096);

    let outcome = apply_window_policy(e.per_layer_kv_mut().unwrap(), &attention).expect("policy");
    assert_eq!(
        positions(&mut e, sliding),
        vec![far as u64],
        "only the post-hole row is inside a {WINDOW}-position window"
    );
    assert_eq!(outcome.rows_dropped[sliding], PREFIX.len());
}

/// **R8b — gap < window.** With the hole smaller than the window, the
/// answer is neither "keep everything" nor "keep nothing": exactly the
/// rows still in logical range survive.
///
/// A gap of 2 leaves the cache at positions 0..8 plus 10, with the next
/// token at 11. A 4-position window reaches back to 7, so exactly one
/// pre-hole row (7) survives alongside the post-hole row — not the four
/// physical rows a row-counting window would keep, and not none.
#[test]
fn r8b_only_rows_still_in_logical_range_survive() {
    const GAP: usize = 2;
    let weights = make_gemma3_narrow_window_test_weights();
    let (attention, sliding, _) = classification(&weights);
    let mut e = prefilled(&weights);
    let next = open_a_hole(&weights, &mut e, GAP) + 1;

    apply_window_policy(e.per_layer_kv_mut().unwrap(), &attention).expect("policy");
    let kept = positions(&mut e, sliding);
    let oldest = next as u64 - WINDOW;
    assert!(
        kept.iter().all(|&p| p >= oldest),
        "kept {kept:?}, all must be >= {oldest}"
    );
    assert_eq!(
        kept,
        vec![7, 10],
        "the boundary row survives and the one before it does not"
    );
}

/// **R8c — the global layer.** The same hole, the same window, the same
/// call: a global layer keeps every pre-hole row.
///
/// Run against R8a this is the whole point of the milestone — one
/// operation, opposite outcomes, decided by the architecture rather than
/// by a single engine-level number.
#[test]
fn r8c_a_global_layer_keeps_pre_hole_rows_across_the_same_hole() {
    let weights = make_gemma3_narrow_window_test_weights();
    let (attention, sliding, global) = classification(&weights);
    let mut e = prefilled(&weights);
    let far = open_a_hole(&weights, &mut e, 4_096);

    let outcome = apply_window_policy(e.per_layer_kv_mut().unwrap(), &attention).expect("policy");
    let mut expected: Vec<u64> = (0..PREFIX.len() as u64).collect();
    expected.push(far as u64);
    assert_eq!(
        positions(&mut e, global),
        expected,
        "nothing may be dropped"
    );
    assert_eq!(outcome.rows_dropped[global], 0);
    assert_ne!(
        positions(&mut e, sliding),
        positions(&mut e, global),
        "the two layer classes must end up genuinely different"
    );
    assert_eq!(outcome.sliding_layers + outcome.global_layers, 6);
}

/// **R8d — the parity gate.** Under the mixed policy, a cache reached by
/// *decoding a gap and excising it* must be indistinguishable from one
/// reached by *offset replay*.
///
/// This is R7's hole oracle re-run with the policy applied, and it is
/// what upgrades "heterogeneous extents are mechanically valid" to
/// "those extents are the ones the architecture asks for": the two
/// routes produce different physical row histories, so if the policy
/// selected on anything other than logical position they would diverge.
#[test]
fn r8d_decoded_then_excised_matches_offset_replay_under_mixed_policy() {
    let weights = make_gemma3_narrow_window_test_weights();
    let (attention, _, _) = classification(&weights);
    let backend = ffn(&weights);

    for gap in [2usize, 4, 6, 16] {
        // Reference: decode the gap, then take those rows back out.
        let mut reference = prefilled(&weights);
        let start = reference
            .per_layer_kv_mut()
            .unwrap()
            .logical_next_position();
        let target = start + gap;
        for i in 0..gap {
            reference
                .decode_step(&weights, &backend, (i % 30) as u32 + 1)
                .expect("gap decode");
        }
        let kv = reference.per_layer_kv_mut().unwrap();
        for l in 0..kv.layer_count().unwrap() {
            kv.excise_kv_rows(l, start..target).expect("gap excision");
        }
        reference
            .decode_step(&weights, &backend, 20)
            .expect("post-gap decode");

        // Candidate: never decode the gap at all.
        let mut candidate = prefilled(&weights);
        let far = open_a_hole(&weights, &mut candidate, gap);
        assert_eq!(
            far, target,
            "gap {gap}: both routes must reach one position"
        );

        let a = apply_window_policy(reference.per_layer_kv_mut().unwrap(), &attention)
            .expect("reference policy");
        let b = apply_window_policy(candidate.per_layer_kv_mut().unwrap(), &attention)
            .expect("candidate policy");
        assert_eq!(a, b, "gap {gap}: the policy itself must act identically");

        for l in 0..attention.layer_count() {
            assert_eq!(
                positions(&mut reference, l),
                positions(&mut candidate, l),
                "gap {gap} layer {l}: retained positions differ"
            );
            let (ka, va) = reference
                .per_layer_kv_mut()
                .unwrap()
                .read_layer_kv(l)
                .unwrap();
            let (kb, vb) = candidate
                .per_layer_kv_mut()
                .unwrap()
                .read_layer_kv(l)
                .unwrap();
            assert_eq!(ka, kb, "gap {gap} layer {l}: K differs");
            assert_eq!(va, vb, "gap {gap} layer {l}: V differs");
        }
    }
}

/// A contiguous cache is the case production is already in. The policy
/// must reduce to the ordinary window there, or routing decode through
/// it later would change unperforated behaviour.
#[test]
fn on_a_contiguous_cache_the_policy_is_an_ordinary_window() {
    let weights = make_gemma3_narrow_window_test_weights();
    let (attention, sliding, global) = classification(&weights);
    let mut e = prefilled(&weights);

    apply_window_policy(e.per_layer_kv_mut().unwrap(), &attention).expect("policy");
    assert_eq!(positions(&mut e, sliding), vec![4, 5, 6, 7]);
    assert_eq!(positions(&mut e, global), vec![0, 1, 2, 3, 4, 5, 6, 7]);
}

/// A sliding architecture with no declared window has no bound to apply.
/// Keeping everything and reporting a clip would be the silent-wrong
/// answer, so it refuses.
#[test]
fn a_sliding_architecture_without_a_declared_window_is_refused() {
    let weights = make_gemma3_narrow_window_test_weights();
    let mut e = prefilled(&weights);
    let undeclared = LayerAttention {
        is_sliding: vec![true, true, true, true, true, false],
        sliding_window: None,
    };
    assert!(apply_window_policy(e.per_layer_kv_mut().unwrap(), &undeclared).is_err());
    // Refused before anything moved.
    assert_eq!(
        positions(&mut e, 0),
        (0..PREFIX.len() as u64).collect::<Vec<_>>()
    );
}

/// A classification that does not describe this cache is refused rather
/// than applied to whichever layers happen to line up.
#[test]
fn a_layer_count_mismatch_is_refused() {
    let weights = make_gemma3_narrow_window_test_weights();
    let mut e = prefilled(&weights);
    let wrong = LayerAttention {
        is_sliding: vec![true, false],
        sliding_window: Some(WINDOW as usize),
    };
    assert!(apply_window_policy(e.per_layer_kv_mut().unwrap(), &wrong).is_err());
    assert_eq!(
        positions(&mut e, 0),
        (0..PREFIX.len() as u64).collect::<Vec<_>>()
    );
}
