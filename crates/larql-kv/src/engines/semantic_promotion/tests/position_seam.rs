//! The seam Phase B depends on: **logical sequence position must be
//! independent of the number of physically retained K/V rows.**
//!
//! Host-side span exclusion works by removing rows from the live cache
//! and continuing to decode. That is equivalent to an index-gather mask
//! only if nothing downstream re-derives position, phase or geometry
//! from the resident row count. These gates pin that property on the CPU
//! per-layer path so a later change to `KvDispatch` cannot quietly break
//! it — the failure mode is a shifted RoPE phase on the first promoted
//! append, which would invalidate the promotion rather than error.
//!
//! Scope: `larql-compute`'s CPU dispatch, tested here because this engine
//! is what depends on the property. See spec §7 (Phase B) for the three
//! places the property does *not* currently hold.

use larql_inference::larql_models::WeightsView;
use larql_inference::test_utils::make_test_weights;
use ndarray::Array2;

/// `make_test_weights` fixture geometry: 1 kv-head × head_dim 8.
const KV_DIM: usize = 8;
const HIDDEN: usize = 16;
const LAYER: usize = 0;
/// Rows planted before the exclusion, and the half-open span removed.
const TOTAL_ROWS: usize = 6;
const EXCLUDED: std::ops::Range<usize> = 2..4;

/// Deterministic, distinguishable K/V row for logical position `p`.
fn row(p: usize, offset: f32) -> Vec<f32> {
    (0..KV_DIM)
        .map(|i| (p as f32 + 1.0) * 0.125 + (i as f32) * 0.03125 + offset)
        .collect()
}

fn query() -> Array2<f32> {
    Array2::from_shape_fn((1, HIDDEN), |(_, i)| (i as f32) * 0.0625 - 0.5)
}

/// Build a handle holding exactly the rows for `positions`, in order.
fn handle_with(
    backend: &dyn larql_inference::EngineBackend,
    positions: &[usize],
) -> larql_compute::kv_dispatch::KvHandle {
    let mut h = backend.alloc_kv_buffer(LAYER, TOTAL_ROWS, KV_DIM);
    for &p in positions {
        backend.append_kv(&mut h, &row(p, 0.0), &row(p, 0.5), p);
    }
    h
}

/// The Phase B claim in miniature: a cache that had rows appended and
/// then physically removed must attend identically to one where those
/// rows were never appended at all.
///
/// If this holds, "drop the rows" and "gather-mask the span" are the same
/// operation, and Phase B needs no kernel change for the exclusion itself.
#[test]
fn removing_rows_is_equivalent_to_never_appending_them() {
    let weights = make_test_weights();
    let backend = larql_inference::cpu_engine_backend();
    let kept: Vec<usize> = (0..TOTAL_ROWS).filter(|p| !EXCLUDED.contains(p)).collect();

    // Arm A: append everything, read back, rebuild without the span —
    // the host-side compaction Phase B would perform.
    let full = handle_with(backend.as_ref(), &(0..TOTAL_ROWS).collect::<Vec<_>>());
    let (k_all, v_all) = backend
        .read_kv_to_host(&full)
        .expect("CPU backend must support host readback");
    assert_eq!(k_all.nrows(), TOTAL_ROWS);
    let mut compacted = backend.alloc_kv_buffer(LAYER, TOTAL_ROWS, KV_DIM);
    for &p in &kept {
        // Rows are re-appended from the READ-BACK cache, not regenerated,
        // so this exercises the real bytes rather than the fixture.
        backend.append_kv(
            &mut compacted,
            k_all.row(p).to_slice().expect("contiguous row"),
            v_all.row(p).to_slice().expect("contiguous row"),
            p,
        );
    }

    // Arm B: the same surviving rows, never having seen the span.
    let mut oracle = handle_with(backend.as_ref(), &kept);

    let q = query();
    let a = backend
        .attention_step(
            WeightsView::dense(&weights),
            &q,
            &mut compacted,
            LAYER,
            TOTAL_ROWS,
            None,
        )
        .expect("CPU attention_step is implemented");
    let b = backend
        .attention_step(
            WeightsView::dense(&weights),
            &q,
            &mut oracle,
            LAYER,
            TOTAL_ROWS,
            None,
        )
        .expect("CPU attention_step is implemented");

    assert_eq!(
        a, b,
        "physically removing a span must equal never having appended it"
    );
}

/// `abs_position` is a real independent input, not something derived
/// from the resident row count. Two calls over the *same* cache at
/// different absolute positions must differ — otherwise the parameter is
/// being ignored and compaction would silently reset the RoPE phase.
#[test]
fn absolute_position_is_an_input_not_a_function_of_resident_rows() {
    let weights = make_test_weights();
    let backend = larql_inference::cpu_engine_backend();
    let q = query();

    let mut at_resident_count = handle_with(backend.as_ref(), &[0, 1, 4, 5]);
    let mut at_true_position = handle_with(backend.as_ref(), &[0, 1, 4, 5]);

    let derived = backend
        .attention_step(
            WeightsView::dense(&weights),
            &q,
            &mut at_resident_count,
            LAYER,
            // What a buggy implementation would use: the row count.
            4,
            None,
        )
        .expect("CPU attention_step is implemented");
    let correct = backend
        .attention_step(
            WeightsView::dense(&weights),
            &q,
            &mut at_true_position,
            LAYER,
            // What Phase B must use: the untouched logical position.
            TOTAL_ROWS,
            None,
        )
        .expect("CPU attention_step is implemented");

    assert_ne!(
        derived, correct,
        "abs_position must affect the result; if these agree, a compacted \
         cache would silently decode at the wrong RoPE phase"
    );
}

/// Appending after a compaction must not renumber the surviving rows:
/// the promoted record lands at the original next absolute position,
/// while the cache is `EXCLUDED.len()` rows shorter.
#[test]
fn appending_after_compaction_uses_the_untouched_logical_position() {
    let backend = larql_inference::cpu_engine_backend();
    let kept: Vec<usize> = (0..TOTAL_ROWS).filter(|p| !EXCLUDED.contains(p)).collect();
    let mut compacted = handle_with(backend.as_ref(), &kept);

    assert_eq!(
        compacted.cached_len(),
        TOTAL_ROWS - EXCLUDED.len(),
        "resident rows shrink by the excluded span"
    );

    // The promoted record's first token is logical position TOTAL_ROWS,
    // NOT `cached_len()`. Phase B must carry this itself: the engine's
    // own `abs_position` is the only source of truth here, because the
    // handle no longer knows how many positions have passed.
    let next_logical = TOTAL_ROWS;
    assert_ne!(next_logical, compacted.cached_len());
    backend.append_kv(
        &mut compacted,
        &row(next_logical, 0.0),
        &row(next_logical, 0.5),
        next_logical,
    );
    assert_eq!(compacted.cached_len(), TOTAL_ROWS - EXCLUDED.len() + 1);
}
