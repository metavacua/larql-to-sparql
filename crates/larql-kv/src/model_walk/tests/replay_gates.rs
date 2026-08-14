//! **R0–R5** — durable boundary replay, and whether it is deterministic
//! enough to carry a counterfactual.
//!
//! Every branch of the counterfactual-absence experiment is built by
//! restoring one checkpoint and replaying a suffix. If two replays of
//! the *same* checkpoint under the *same* visibility can differ, then a
//! difference between branches proves nothing. These gates establish the
//! null case before the interesting one is attempted.
//!
//! The geometry the experiment needs:
//!
//! ```text
//! [ prefix ][ chain ][ filler ][ query ]
//!           ^ checkpoint
//! ```
//!
//! - **reference** replays the suffix unchanged;
//! - **counterfactual** replays decoys over the chain's positions, so no
//!   replayed position ever attends the chain and every later position
//!   keeps its original index.

use larql_inference::hidden_to_raw_logits;
use larql_inference::larql_models::test_fixtures::make_gemma3_narrow_window_test_weights;
use larql_inference::model::ModelWeights;
use ndarray::Array2;

use crate::engines::standard::StandardEngine;
use crate::semantic_promotion::{BoundaryCheckpoint, CheckpointId, SuffixReplay, SuffixVisibility};
use crate::KvEngine;

/// Everything before the planted chain.
const PREFIX: [u32; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
/// The suffix, from the checkpoint onward: chain, then filler, then a
/// query token. Every id stays inside the fixture's 32-token vocab.
const SUFFIX: [u32; 8] = [20, 21, 22, 23, 24, 25, 26, 27];
/// Which suffix positions the chain occupies.
const CHAIN: std::ops::Range<usize> = 0..4;
/// Same length, unrelated content.
const DECOY: [u32; 4] = [10, 11, 12, 13];

fn engine() -> StandardEngine {
    StandardEngine::with_backend(None, larql_inference::cpu_engine_backend())
}

fn ffn(weights: &ModelWeights) -> larql_compute::ffn::WeightFfn<'_> {
    larql_compute::ffn::WeightFfn { weights }
}

/// Prefill the prefix and capture a checkpoint at that boundary.
fn checkpointed(weights: &ModelWeights) -> (StandardEngine, BoundaryCheckpoint) {
    let mut e = engine();
    e.prefill(weights, &ffn(weights), &PREFIX).unwrap();
    let checkpoint =
        BoundaryCheckpoint::capture(CheckpointId::from_counter(1), e.per_layer_kv_mut().unwrap())
            .expect("capture");
    (e, checkpoint)
}

/// Restore and replay a branch, returning per-step logits.
fn replay(
    weights: &ModelWeights,
    e: &mut StandardEngine,
    checkpoint: &BoundaryCheckpoint,
    branch: &SuffixReplay,
) -> Vec<Vec<f32>> {
    checkpoint
        .restore(e.per_layer_kv_mut().unwrap())
        .expect("restore");
    let backend = ffn(weights);
    branch
        .tokens()
        .expect("branch tokens")
        .into_iter()
        .map(|t| hidden_to_raw_logits(weights, &e.decode_step(weights, &backend, t).unwrap()))
        .collect()
}

fn branch(visibility: SuffixVisibility) -> SuffixReplay {
    SuffixReplay {
        checkpoint: CheckpointId::from_counter(1),
        original_suffix: SUFFIX.to_vec(),
        visibility,
    }
}

fn counterfactual() -> SuffixReplay {
    branch(SuffixVisibility::ChainReplacedByDecoy {
        chain: CHAIN,
        decoy: DECOY.to_vec(),
    })
}

fn layers(e: &mut StandardEngine) -> Vec<(Array2<f32>, Array2<f32>)> {
    let kv = e.per_layer_kv_mut().unwrap();
    let n = kv.layer_count().unwrap();
    (0..n).map(|l| kv.read_layer_kv(l).unwrap()).collect()
}

fn assert_bit_identical(label: &str, a: &[f32], b: &[f32]) {
    let diverged = a
        .iter()
        .zip(b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count();
    assert_eq!(
        diverged,
        0,
        "{label}: {diverged} of {} logits differ",
        a.len()
    );
}

/// **R0** — a checkpoint captures the boundary it claims to.
#[test]
fn r0_a_checkpoint_records_the_logical_boundary() {
    let weights = make_gemma3_narrow_window_test_weights();
    let (_, checkpoint) = checkpointed(&weights);
    assert_eq!(checkpoint.logical_next_position, PREFIX.len());
    assert!(checkpoint.rows().iter().all(|&r| r == PREFIX.len()));
    assert!(checkpoint.retained_bytes() > 0);
}

/// **R1, the null case.** Two replays of the same checkpoint under the
/// same visibility are bit-identical, in logits and in K/V.
///
/// Without this, any difference between counterfactual branches could be
/// replay noise rather than the missing evidence.
#[test]
fn r1_replay_is_deterministic() {
    let weights = make_gemma3_narrow_window_test_weights();
    let (mut e, checkpoint) = checkpointed(&weights);
    let reference = branch(SuffixVisibility::Original);

    let first = replay(&weights, &mut e, &checkpoint, &reference);
    let first_kv = layers(&mut e);
    let second = replay(&weights, &mut e, &checkpoint, &reference);
    let second_kv = layers(&mut e);

    for (i, (a, b)) in first.iter().zip(second.iter()).enumerate() {
        assert_bit_identical(&format!("r1 step {i}"), a, b);
    }
    for (layer, ((ka, va), (kb, vb))) in first_kv.iter().zip(second_kv.iter()).enumerate() {
        assert_eq!(ka, kb, "layer {layer} K differs across identical replays");
        assert_eq!(va, vb, "layer {layer} V differs across identical replays");
    }
}

/// **R2** — replaying the original suffix from a checkpoint reproduces
/// what an uninterrupted prefill+decode would have produced. Restore is
/// not merely self-consistent; it lands on the real trajectory.
#[test]
fn r2_replay_reproduces_the_uninterrupted_run() {
    let weights = make_gemma3_narrow_window_test_weights();
    let backend = ffn(&weights);

    let mut straight = engine();
    straight.prefill(&weights, &backend, &PREFIX).unwrap();
    let expected: Vec<_> = SUFFIX
        .iter()
        .map(|&t| {
            hidden_to_raw_logits(
                &weights,
                &straight.decode_step(&weights, &backend, t).unwrap(),
            )
        })
        .collect();

    let (mut e, checkpoint) = checkpointed(&weights);
    // Decode something else first, so the restore has real work to undo.
    for t in [28, 29, 30] {
        e.decode_step(&weights, &backend, t).unwrap();
    }
    let actual = replay(
        &weights,
        &mut e,
        &checkpoint,
        &branch(SuffixVisibility::Original),
    );

    for (i, (a, b)) in expected.iter().zip(actual.iter()).enumerate() {
        assert_bit_identical(&format!("r2 step {i}"), a, b);
    }
}

/// **R3** — the counterfactual branch keeps every position aligned with
/// the reference. Same suffix length, same logical positions, different
/// content only inside the chain's range.
#[test]
fn r3_the_counterfactual_branch_is_positionally_aligned() {
    let reference = branch(SuffixVisibility::Original).tokens().unwrap();
    let absent = counterfactual().tokens().unwrap();

    assert_eq!(reference.len(), absent.len());
    for (i, (a, b)) in reference.iter().zip(absent.iter()).enumerate() {
        if CHAIN.contains(&i) {
            assert_ne!(a, b, "position {i} should carry a decoy");
        } else {
            assert_eq!(a, b, "position {i} must be untouched");
        }
    }

    // ...and the engine agrees on where it ends up.
    let weights = make_gemma3_narrow_window_test_weights();
    let (mut e, checkpoint) = checkpointed(&weights);
    replay(&weights, &mut e, &checkpoint, &counterfactual());
    let after_absent = e.per_layer_kv_mut().unwrap().logical_next_position();
    replay(
        &weights,
        &mut e,
        &checkpoint,
        &branch(SuffixVisibility::Original),
    );
    let after_reference = e.per_layer_kv_mut().unwrap().logical_next_position();
    assert_eq!(after_absent, after_reference);
    assert_eq!(after_reference, PREFIX.len() + SUFFIX.len());
}

/// **R4, the control.** The counterfactual branch must actually differ
/// from the reference. If replacing the chain with decoys changed
/// nothing, the branch would not be counterfactual and every downstream
/// comparison would be vacuous.
#[test]
fn r4_the_counterfactual_branch_diverges_from_the_reference() {
    let weights = make_gemma3_narrow_window_test_weights();
    let (mut e, checkpoint) = checkpointed(&weights);

    let reference = replay(
        &weights,
        &mut e,
        &checkpoint,
        &branch(SuffixVisibility::Original),
    );
    let absent = replay(&weights, &mut e, &checkpoint, &counterfactual());

    // The query token is the last position — the one an experiment reads.
    let last = reference.len() - 1;
    let differing = reference[last]
        .iter()
        .zip(&absent[last])
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert!(
        differing > 0,
        "removing the chain from the replayed suffix must change the \
         query's logits; if it does not, the branch is not counterfactual"
    );
}

/// **R6** — an offset replay actually moves the sequence's phase, and
/// replaying at the checkpoint's own position is indistinguishable from a
/// plain restore.
///
/// This is the protocol the positional-offset experiment needs: restore a
/// checkpoint taken *before* the sequence, set the new start, then
/// regenerate every token whose phase should move. Shifting the position
/// and reusing the old rows would leave the cache disagreeing with its own
/// geometry, which nothing observable would flag.
#[test]
fn r6_offset_replay_moves_the_phase_and_is_a_no_op_at_the_original_offset() {
    let weights = make_gemma3_narrow_window_test_weights();
    let (mut e, checkpoint) = checkpointed(&weights);
    let backend = ffn(&weights);
    let seq: Vec<u32> = SUFFIX[..4].to_vec();

    let run_at = |e: &mut StandardEngine, start: Option<usize>| -> Vec<Vec<f32>> {
        match start {
            Some(p) => checkpoint
                .restore_for_offset_replay(e.per_layer_kv_mut().unwrap(), p)
                .expect("offset restore"),
            None => checkpoint
                .restore(e.per_layer_kv_mut().unwrap())
                .expect("restore"),
        }
        seq.iter()
            .map(|&t| {
                hidden_to_raw_logits(&weights, &e.decode_step(&weights, &backend, t).unwrap())
            })
            .collect()
    };

    let plain = run_at(&mut e, None);
    let same_offset = run_at(&mut e, Some(checkpoint.logical_next_position));
    for (i, (a, b)) in plain.iter().zip(same_offset.iter()).enumerate() {
        assert_bit_identical(&format!("r6 same-offset step {i}"), a, b);
    }

    // A different start position must change the result — otherwise the
    // offset is not reaching the RoPE phase and the experiment it exists
    // for could not measure anything.
    let shifted = run_at(&mut e, Some(checkpoint.logical_next_position + 4_096));
    let differing = plain[0]
        .iter()
        .zip(&shifted[0])
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert!(
        differing > 0,
        "replaying at a different absolute position must change the logits"
    );

    // The checkpoint's own rows are untouched by the offset — only what
    // follows moves. Resident rows are history plus the replayed sequence.
    let extent = e.per_layer_kv_mut().unwrap().extent().unwrap();
    assert!(extent
        .per_layer_resident_rows
        .iter()
        .all(|&r| r == PREFIX.len() + seq.len()));
}

/// **R7 — the hole oracle.** A sparse-position cache built by offset
/// replay must be indistinguishable from one built by decoding the gap and
/// then removing it.
///
/// Two routes to the same logical state:
///
/// ```text
/// reference   restore at P → decode gap P..Q → excise the gap rows → decode S at Q
/// candidate   restore_for_offset_replay(Q)                          → decode S at Q
/// ```
///
/// Not tautological: the reference reaches its state through
/// `read_kv_to_host` → rebuild, which could perturb bytes, and its engine
/// position was advanced by decoding rather than set. If the two agree
/// bit-exactly on logits *and* K/V, then gap tokens leave no trace and the
/// hole is genuinely invisible.
#[test]
fn r7_a_replayed_hole_is_indistinguishable_from_a_decoded_then_excised_one() {
    let weights = make_gemma3_narrow_window_test_weights();
    let backend = ffn(&weights);
    let seq: Vec<u32> = SUFFIX[..4].to_vec();

    // Gap sizes around and well past the architecture's declared 4-token
    // window, so a window-sensitive bug would show at some of them.
    for gap in [2usize, 4, 6, 16] {
        let (mut e, checkpoint) = checkpointed(&weights);
        let start = checkpoint.logical_next_position;
        let target = start + gap;

        // Reference: decode the gap, then take those rows back out.
        checkpoint.restore(e.per_layer_kv_mut().unwrap()).unwrap();
        for i in 0..gap {
            e.decode_step(&weights, &backend, (i % 30) as u32 + 1)
                .unwrap();
        }
        let kv = e.per_layer_kv_mut().unwrap();
        let layers = kv.layer_count().unwrap();
        for l in 0..layers {
            kv.excise_kv_rows(l, start..target).expect("gap excision");
        }
        assert_eq!(
            e.per_layer_kv_mut().unwrap().logical_next_position(),
            target,
            "gap {gap}: excision must not move the logical position"
        );
        let reference: Vec<_> = seq
            .iter()
            .map(|&t| {
                hidden_to_raw_logits(&weights, &e.decode_step(&weights, &backend, t).unwrap())
            })
            .collect();
        let reference_kv = layers_of(&mut e);

        // Candidate: never decode the gap at all.
        let mut c = engine();
        c.prefill(&weights, &backend, &PREFIX).unwrap();
        checkpoint
            .restore_for_offset_replay(c.per_layer_kv_mut().unwrap(), target)
            .unwrap();
        let candidate: Vec<_> = seq
            .iter()
            .map(|&t| {
                hidden_to_raw_logits(&weights, &c.decode_step(&weights, &backend, t).unwrap())
            })
            .collect();
        let candidate_kv = layers_of(&mut c);

        for (i, (a, b)) in reference.iter().zip(candidate.iter()).enumerate() {
            assert_bit_identical(&format!("r7 gap {gap} step {i}"), a, b);
        }
        for (l, ((ka, va), (kb, vb))) in reference_kv.iter().zip(candidate_kv.iter()).enumerate() {
            assert_eq!(ka, kb, "gap {gap} layer {l}: K differs");
            assert_eq!(va, vb, "gap {gap} layer {l}: V differs");
        }
        assert_eq!(
            e.per_layer_kv_mut().unwrap().logical_next_position(),
            c.per_layer_kv_mut().unwrap().logical_next_position()
        );
    }
}

/// **R7b — the limitation this path actually has.**
///
/// The interleave hazard (global layers keeping an ancient row while
/// sliding layers drop it) *cannot* manifest here: the CPU per-layer decode
/// path in `larql_inference::kv_dispatch::helpers` calls `attention_step`
/// with no per-layer window and then `clip_kv(handle, w)` using the
/// **engine-level** window uniformly. It never consults
/// `arch.is_sliding_window_layer`. So every layer attends over everything
/// resident, and a Gemma interleave is a *policy* distinction the promotion
/// wrapper applies, not something attention enforces.
///
/// The real hazard is one level down and this gate pins it: `clip_kv` keeps
/// the tail *N physical rows*. Across a positional hole, "last N physical"
/// is not "last N logical" — pre-hole rows that are thousands of positions
/// old survive a window they are far outside. Recorded as a limitation, not
/// a pass: a windowed engine over a sparse cache does not implement
/// logical-age semantics.
#[test]
fn r7b_a_windowed_engine_selects_by_physical_row_not_logical_age() {
    const WINDOW: usize = 4;
    let weights = make_gemma3_narrow_window_test_weights();
    let backend = ffn(&weights);

    let mut e = StandardEngine::with_backend(Some(WINDOW), larql_inference::cpu_engine_backend());
    e.prefill(&weights, &backend, &PREFIX).unwrap();
    let checkpoint =
        BoundaryCheckpoint::capture(CheckpointId::from_counter(2), e.per_layer_kv_mut().unwrap())
            .unwrap();
    let ancient = checkpoint.per_layer[0].k.clone();

    // Jump far past the window, then decode one token.
    let far = checkpoint.logical_next_position + 4_096;
    checkpoint
        .restore_for_offset_replay(e.per_layer_kv_mut().unwrap(), far)
        .unwrap();
    e.decode_step(&weights, &backend, 20).unwrap();

    let (k, _) = e.per_layer_kv_mut().unwrap().read_layer_kv(0).unwrap();
    assert_eq!(k.nrows(), WINDOW, "clip_kv holds the window");

    // At least one surviving row is byte-identical to a checkpoint row —
    // i.e. a row from ~4096 positions ago is still inside a 4-token window,
    // because the window counts rows rather than positions.
    let survivors = (0..k.nrows())
        .filter(|&i| (0..ancient.nrows()).any(|j| k.row(i).to_vec() == ancient.row(j).to_vec()))
        .count();
    assert!(
        survivors > 0,
        "expected pre-hole rows to survive a physical window; if this now \
         fails, windowing has become logical-age aware and the limitation \
         note above is stale"
    );
}

/// Full K/V for every layer.
fn layers_of(e: &mut StandardEngine) -> Vec<(Array2<f32>, Array2<f32>)> {
    let kv = e.per_layer_kv_mut().unwrap();
    let n = kv.layer_count().unwrap();
    (0..n).map(|l| kv.read_layer_kv(l).unwrap()).collect()
}

/// **R5** — a checkpoint restores a *shorter* cache, discarding
/// everything decoded after it. Splice cannot do this; replace can, and
/// that is why the primitive was added.
#[test]
fn r5_restore_shortens_the_cache_back_to_the_boundary() {
    let weights = make_gemma3_narrow_window_test_weights();
    let backend = ffn(&weights);
    let (mut e, checkpoint) = checkpointed(&weights);

    for &t in &SUFFIX {
        e.decode_step(&weights, &backend, t).unwrap();
    }
    let grown = e.per_layer_kv_mut().unwrap().extent().unwrap();
    assert_eq!(grown.logical_next_position, PREFIX.len() + SUFFIX.len());

    checkpoint.restore(e.per_layer_kv_mut().unwrap()).unwrap();
    let restored = e.per_layer_kv_mut().unwrap().extent().unwrap();
    assert_eq!(restored.logical_next_position, PREFIX.len());
    assert!(restored
        .per_layer_resident_rows
        .iter()
        .all(|&r| r == PREFIX.len()));
    assert!(restored.is_uniform());
}
