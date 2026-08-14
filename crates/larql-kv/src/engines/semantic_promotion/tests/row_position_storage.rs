//! **R8-storage.** The row-position map stays congruent with the cache
//! through every mutation the engine performs.
//!
//! This is deliberately the *narrow* half of R8. It proves that after
//! append, excise, splice, checkpoint capture/restore and offset replay,
//! the map still describes the rows that are actually resident — one
//! position per row, ascending, and the positions the rows were written
//! at. It proves nothing about which rows a *policy* should keep; the
//! sliding/global gates (R8a–R8d) are R8-policy and live elsewhere.
//!
//! Why the split matters: a passing storage gate licenses "heterogeneous
//! layer extents remain mechanically valid", not "these extents match
//! Gemma's production visibility". The tested path never consults the
//! architectural per-layer window classification, so it cannot support
//! the second claim, and the milestone is named so nothing borrows it.
//!
//! The one behavioural gate here is `clip_layer_to_logical_window`,
//! included because it is a *mechanism* over the map — select by logical
//! age, prune physically — rather than a decision about which layers
//! deserve it.

use larql_inference::kv_engine::{KvEngine, PerLayerKvAccess};
use larql_inference::test_utils::make_test_weights;

use super::fixtures::bare_standard;
use crate::engines::semantic_promotion::checkpoint::BoundaryCheckpoint;
use crate::engines::semantic_promotion::ids::CheckpointId;

/// Prompt long enough to excise a span from and still have rows either
/// side of the hole.
const PROMPT: [u32; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
const LAYER: usize = 0;

fn engine_after_prefill() -> crate::engines::standard::StandardEngine {
    let weights = make_test_weights();
    let ffn = larql_compute::ffn::WeightFfn { weights: &weights };
    let mut e = bare_standard();
    e.prefill(&weights, &ffn, &PROMPT).expect("prefill");
    e
}

fn decode(e: &mut crate::engines::standard::StandardEngine, token: u32) {
    let weights = make_test_weights();
    let ffn = larql_compute::ffn::WeightFfn { weights: &weights };
    e.decode_step(&weights, &ffn, token).expect("decode step");
}

/// A windowed engine, whose dispatch clips inside every prefill and
/// decode step. The map has to follow a drop it did not perform.
fn windowed_engine_after_prefill(window: usize) -> crate::engines::standard::StandardEngine {
    let weights = make_test_weights();
    let ffn = larql_compute::ffn::WeightFfn { weights: &weights };
    let mut e = crate::engines::standard::StandardEngine::with_backend(
        Some(window),
        larql_inference::cpu_engine_backend(),
    );
    e.prefill(&weights, &ffn, &PROMPT).expect("prefill");
    e
}

/// Every layer's map must have exactly one ascending position per
/// resident row. The check the engine itself cannot skip.
fn assert_congruent(e: &dyn PerLayerKvAccess, note: &str) {
    let layers = e.layer_count().expect("per-layer access");
    let rows: Vec<usize> = (0..layers)
        .map(|l| e.resident_rows(l).expect("resident rows"))
        .collect();
    e.row_positions()
        .expect("row positions")
        .check(&rows)
        .unwrap_or_else(|err| panic!("{note}: map disagrees with cache: {err}"));
}

#[test]
fn prefill_gives_one_ascending_position_per_row() {
    let mut e = engine_after_prefill();
    assert_congruent(e.per_layer_kv_mut().unwrap(), "after prefill");
    let kv = e.per_layer_kv_mut().unwrap();
    let map = kv.row_positions().unwrap().layer(LAYER).unwrap();
    assert_eq!(map.as_slice(), &[0, 1, 2, 3, 4, 5, 6, 7]);
    assert!(map.is_contiguous());
    assert!(map.holes().is_empty());
}

#[test]
fn every_decode_append_records_its_absolute_position() {
    let mut e = engine_after_prefill();
    decode(&mut e, 9);
    decode(&mut e, 10);
    assert_congruent(e.per_layer_kv_mut().unwrap(), "after decode");
    let kv = e.per_layer_kv_mut().unwrap();
    assert_eq!(kv.logical_next_position(), PROMPT.len() + 2);
    assert_eq!(
        kv.row_positions().unwrap().layer(LAYER).unwrap().as_slice(),
        &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    );
}

/// A windowed prefill drops its oldest rows before the map is ever
/// built, so the run must start where the surviving rows start — not at
/// zero, which is what a naive `0..len` would give.
#[test]
fn a_windowed_prefill_records_the_tail_it_actually_kept() {
    const WINDOW: usize = 4;
    let mut e = windowed_engine_after_prefill(WINDOW);
    assert_congruent(e.per_layer_kv_mut().unwrap(), "after windowed prefill");
    let kv = e.per_layer_kv_mut().unwrap();
    assert_eq!(kv.resident_rows(LAYER).unwrap(), WINDOW);
    assert_eq!(
        kv.row_positions().unwrap().layer(LAYER).unwrap().as_slice(),
        &[4, 5, 6, 7],
        "the window kept the last four positions, not the first four"
    );
}

/// **The clip the engine does not perform.** `clip_kv` runs inside
/// dispatch, below the engine, and drops the oldest row on every step.
/// The map has to mirror that drop or it will claim rows the cache no
/// longer holds — and on a contiguous cache mirroring it is also the
/// right answer, since physical tail and logical tail coincide.
#[test]
fn a_windowed_decode_mirrors_the_dispatch_clip() {
    const WINDOW: usize = 4;
    let mut e = windowed_engine_after_prefill(WINDOW);
    decode(&mut e, 9);
    decode(&mut e, 10);
    assert_congruent(e.per_layer_kv_mut().unwrap(), "after windowed decode");
    let kv = e.per_layer_kv_mut().unwrap();
    assert_eq!(kv.resident_rows(LAYER).unwrap(), WINDOW);
    assert_eq!(kv.logical_next_position(), PROMPT.len() + 2);
    assert_eq!(
        kv.row_positions().unwrap().layer(LAYER).unwrap().as_slice(),
        &[6, 7, 8, 9],
        "two appends, two evictions, the map following the rows"
    );
}

/// **R8e, the round trip.** Excise a span, confirm the hole is visible
/// and correctly addressed, splice it back, confirm the map is whole.
#[test]
fn excise_then_splice_restores_the_positions_exactly() {
    let mut e = engine_after_prefill();
    let kv = e.per_layer_kv_mut().unwrap();
    let before = kv.row_positions().unwrap().layer(LAYER).unwrap().clone();

    let excised = kv.excise_kv_rows(LAYER, 2..5).expect("excision");
    assert_eq!(excised.logical_positions(), &[2, 3, 4]);
    assert_eq!(excised.len(), 3, "positions and rows leave together");
    assert_congruent(kv, "after excise");

    let map = kv.row_positions().unwrap().layer(LAYER).unwrap();
    assert_eq!(map.as_slice(), &[0, 1, 5, 6, 7]);
    assert_eq!(map.holes(), vec![2..5]);
    // The lookup that matters: position 5 now lives at physical row 2,
    // and the excised positions resolve to no row rather than to
    // whatever happens to sit at that index.
    assert_eq!(map.row_of(5), Some(2));
    assert_eq!(map.row_of(3), None);

    assert!(kv.splice_kv_rows(LAYER, 2, &excised), "splice");
    assert_congruent(kv, "after splice");
    assert_eq!(
        kv.row_positions().unwrap().layer(LAYER).unwrap(),
        &before,
        "round trip must be exact, not merely well-formed"
    );
}

/// A splice at the wrong offset would leave positions out of order. It
/// must be refused *before* the handle is rebuilt, or the cache ends up
/// in a state neither caller asked for.
#[test]
fn a_misplaced_splice_is_refused_without_touching_the_cache() {
    let mut e = engine_after_prefill();
    let kv = e.per_layer_kv_mut().unwrap();
    let excised = kv.excise_kv_rows(LAYER, 2..5).expect("excision");
    let after_excise = kv.row_positions().unwrap().layer(LAYER).unwrap().clone();
    let rows_after_excise = kv.resident_rows(LAYER).unwrap();

    // Putting positions 2..5 back at the end would give 0,1,5,6,7,2,3,4.
    assert!(!kv.splice_kv_rows(LAYER, 5, &excised), "must refuse");
    assert_eq!(kv.resident_rows(LAYER).unwrap(), rows_after_excise);
    assert_eq!(
        kv.row_positions().unwrap().layer(LAYER).unwrap(),
        &after_excise,
        "a refused splice leaves the map untouched"
    );
    assert_congruent(kv, "after refused splice");
}

/// Only the layers named lose rows. This is the ragged shape B10 covers,
/// stated over the map: heterogeneous extents, each internally valid.
#[test]
fn excision_on_one_layer_leaves_the_map_ragged_but_congruent() {
    let mut e = engine_after_prefill();
    let kv = e.per_layer_kv_mut().unwrap();
    let layers = kv.layer_count().unwrap();
    assert!(layers > 1, "fixture needs more than one layer to be ragged");

    kv.excise_kv_rows(layers - 1, 2..5).expect("excision");
    assert_congruent(kv, "after single-layer excise");
    let map = kv.row_positions().unwrap();
    assert!(!map.is_uniform(), "one layer is short — that is the point");
    assert!(map.layer(0).unwrap().is_contiguous());
    assert_eq!(map.layer(layers - 1).unwrap().holes(), vec![2..5]);
}

#[test]
fn a_checkpoint_carries_positions_and_restores_them_with_its_rows() {
    let mut e = engine_after_prefill();
    let kv = e.per_layer_kv_mut().unwrap();
    kv.excise_kv_rows(LAYER, 2..5).expect("excision");
    let sparse = kv.row_positions().unwrap().clone();

    let checkpoint =
        BoundaryCheckpoint::capture(CheckpointId::from_counter(1), kv).expect("capture");
    assert_eq!(
        checkpoint.per_layer[LAYER].positions,
        vec![0, 1, 5, 6, 7],
        "the checkpoint records the sparse addressing, not 0..n"
    );

    // Decode past the checkpoint, then wind back.
    decode(&mut e, 9);
    decode(&mut e, 10);
    let kv = e.per_layer_kv_mut().unwrap();
    assert_ne!(kv.row_positions().unwrap(), &sparse);

    checkpoint.restore(kv).expect("restore");
    assert_congruent(kv, "after checkpoint restore");
    assert_eq!(
        kv.row_positions().unwrap(),
        &sparse,
        "restoring rows without their positions would re-address them"
    );
    assert_eq!(kv.logical_next_position(), PROMPT.len());
}

/// Offset replay: the checkpoint's rows keep the phases they were
/// written at, the next token lands far ahead, and the map records the
/// gap rather than pretending the run is contiguous.
#[test]
fn offset_replay_leaves_a_hole_the_map_can_describe() {
    let mut e = engine_after_prefill();
    let kv = e.per_layer_kv_mut().unwrap();
    let checkpoint =
        BoundaryCheckpoint::capture(CheckpointId::from_counter(2), kv).expect("capture");
    let far = checkpoint.logical_next_position + 4_096;
    checkpoint
        .restore_for_offset_replay(kv, far)
        .expect("offset restore");
    assert_eq!(kv.logical_next_position(), far);

    decode(&mut e, 9);
    let kv = e.per_layer_kv_mut().unwrap();
    assert_congruent(kv, "after offset replay");
    let map = kv.row_positions().unwrap().layer(LAYER).unwrap();
    assert_eq!(map.max_position(), Some(far as u64));
    assert_eq!(map.holes(), vec![8..far as u64]);
}

/// Moving the next position *behind* a resident row would leave the
/// cache holding a position it claims has not happened yet.
#[test]
fn the_next_position_cannot_be_moved_behind_a_resident_row() {
    let mut e = engine_after_prefill();
    let kv = e.per_layer_kv_mut().unwrap();
    assert!(!kv.set_logical_next_position(4), "must refuse");
    assert_eq!(kv.logical_next_position(), PROMPT.len());
    assert!(kv.set_logical_next_position(PROMPT.len() + 100));
}

/// **The R7b divergence, closed.** A physical-tail clip keeps the last
/// N rows; a logical clip keeps rows inside N positions. Across an
/// offset-replay hole those differ, and this is the one that is right.
#[test]
fn a_logical_clip_drops_pre_hole_rows_a_physical_clip_would_keep() {
    const WINDOW: u64 = 4;
    let mut e = engine_after_prefill();
    let kv = e.per_layer_kv_mut().unwrap();
    let checkpoint =
        BoundaryCheckpoint::capture(CheckpointId::from_counter(3), kv).expect("capture");
    let far = checkpoint.logical_next_position + 4_096;
    checkpoint
        .restore_for_offset_replay(kv, far)
        .expect("offset restore");
    decode(&mut e, 9);

    let kv = e.per_layer_kv_mut().unwrap();
    // Nine rows spanning four thousand positions: 0..8 then `far`.
    assert_eq!(kv.resident_rows(LAYER).unwrap(), PROMPT.len() + 1);

    let dropped = kv
        .clip_layer_to_logical_window(LAYER, WINDOW)
        .expect("logical clip");
    assert_eq!(dropped, PROMPT.len(), "every pre-hole row is out of range");
    assert_congruent(kv, "after logical clip");
    assert_eq!(
        kv.row_positions().unwrap().layer(LAYER).unwrap().as_slice(),
        &[far as u64],
        "only the post-hole row is actually recent"
    );
}

/// On a contiguous cache the logical clip and the physical one agree —
/// so wiring it in cannot change unperforated behaviour.
#[test]
fn on_a_contiguous_cache_a_logical_clip_matches_the_physical_one() {
    const WINDOW: u64 = 4;
    let mut e = engine_after_prefill();
    let kv = e.per_layer_kv_mut().unwrap();
    let dropped = kv
        .clip_layer_to_logical_window(LAYER, WINDOW)
        .expect("logical clip");
    assert_eq!(dropped, PROMPT.len() - WINDOW as usize);
    assert_eq!(
        kv.row_positions().unwrap().layer(LAYER).unwrap().as_slice(),
        &[4, 5, 6, 7],
        "the tail four rows are also the four most recent positions"
    );
    assert_congruent(kv, "after contiguous logical clip");

    // Already inside the window: nothing moves.
    assert_eq!(kv.clip_layer_to_logical_window(LAYER, WINDOW), Some(0));
}
