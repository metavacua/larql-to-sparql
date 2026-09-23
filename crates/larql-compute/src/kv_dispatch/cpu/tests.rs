//! Coverage tests for the CPU `KvDispatch` impl.
//!
//! Exercises:
//! - `CpuKvHandle` accessors (`cached_len`, `kv_dim`, `backend_name`,
//!   `as_any{,_mut}`).
//! - `CpuResidualHandle` accessors.
//! - `CpuQ4kCacheHandle` accessors against a manually-constructed
//!   cache (no Q4K vindex needed).
//! - Wrong-backend-handle panic paths via the dispatch-time downcast
//!   helpers (`cpu_handle*`, `cpu_residual`).
//! - The simple buffer ops (`alloc_kv_buffer`, `append_kv`, `clip_kv`,
//!   `read_kv_to_host`).
//!
//! End-to-end attention dispatch + Q4K decode paths are covered by the
//! integration tests on the inference engines (StandardEngine uses
//! this `KvDispatch` impl through the trait).
use super::handles::try_cpu_q4k_cache_mut;
use super::*;
use crate::attention::decode::q4k_direct_attn_enabled;
use crate::kv_dispatch::{
    KvDispatch, KvHandle, KvHandleInner, ResidualHandle, ResidualHandleInner,
};
use crate::CpuBackend;
use ndarray::Array2;

fn backend() -> CpuBackend {
    CpuBackend
}

#[test]
fn cpu_kv_handle_accessors_reflect_state() {
    let b = backend();
    let mut h = b.alloc_kv_buffer(0, 8, 4);
    // Empty handle: cached_len=0, kv_dim from alloc.
    assert_eq!(h.cached_len(), 0);
    assert_eq!(h.kv_dim(), 4);
    assert_eq!(h.backend_name(), "cpu");
    // After append: cached_len reflects the rows.
    b.append_kv(&mut h, &[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0], 0);
    b.append_kv(&mut h, &[9.0; 4], &[10.0; 4], 1);
    assert_eq!(h.cached_len(), 2);
    assert_eq!(h.kv_dim(), 4);
}

#[test]
fn cpu_kv_handle_as_any_round_trip() {
    let b = backend();
    let mut h = b.alloc_kv_buffer(0, 4, 4);
    // immutable downcast through KvHandle's as_inner
    {
        let inner: &dyn KvHandleInner = h.as_inner();
        let any: &dyn std::any::Any = inner.as_any();
        assert!(any.downcast_ref::<CpuKvHandle>().is_some());
    }
    // mutable downcast
    {
        let inner_mut: &mut dyn KvHandleInner = h.as_inner_mut();
        let any_mut: &mut dyn std::any::Any = inner_mut.as_any_mut();
        assert!(any_mut.downcast_mut::<CpuKvHandle>().is_some());
    }
}

#[test]
fn cpu_residual_handle_shape_and_backend_name() {
    let b = backend();
    let res = Array2::<f32>::zeros((2, 8));
    let h = b.upload_boundary_residual(&res).expect("upload");
    assert_eq!(h.shape(), (2, 8));
    assert_eq!(ResidualHandleInner::backend_name(&*h.inner), "cpu");
    // as_any round-trip
    let any: &dyn std::any::Any = ResidualHandleInner::as_any(&*h.inner);
    assert!(any.downcast_ref::<CpuResidualHandle>().is_some());
}

#[test]
fn clip_kv_truncates_to_window_size() {
    let b = backend();
    let mut h = b.alloc_kv_buffer(0, 8, 2);
    for i in 0..4u32 {
        let f = i as f32;
        b.append_kv(&mut h, &[f, f], &[f, f], i as usize);
    }
    assert_eq!(h.cached_len(), 4);
    b.clip_kv(&mut h, 2);
    assert_eq!(h.cached_len(), 2);
    let (k, _) = b.read_kv_to_host(&h).unwrap();
    // After clip-to-2, the newest two rows (indices 2 and 3) remain.
    assert_eq!(k[[0, 0]], 2.0);
    assert_eq!(k[[1, 0]], 3.0);
}

#[test]
fn clip_kv_below_window_is_a_no_op() {
    let b = backend();
    let mut h = b.alloc_kv_buffer(0, 4, 2);
    b.append_kv(&mut h, &[1.0, 2.0], &[3.0, 4.0], 0);
    b.append_kv(&mut h, &[5.0, 6.0], &[7.0, 8.0], 1);
    b.clip_kv(&mut h, 10);
    // Window size > rows → unchanged.
    assert_eq!(h.cached_len(), 2);
}

#[test]
fn clip_kv_with_no_state_is_a_no_op() {
    let b = backend();
    let mut h = b.alloc_kv_buffer(0, 4, 2);
    // No append yet → state is None.
    b.clip_kv(&mut h, 2);
    assert_eq!(h.cached_len(), 0);
}

// ── truncate_kv ─────────────────────────────────────────────────────────────
//
// The inverse of an append: rewinds a partially-applied decode step. The
// contract that matters is *which* rows survive — `clip_kv` keeps the tail,
// this keeps the head — because a rewind that kept the wrong end would
// restore the row count while corrupting the cache, and a length assertion
// alone would not notice.

#[test]
fn truncate_kv_keeps_the_head_where_clip_keeps_the_tail() {
    let b = backend();
    let mut h = b.alloc_kv_buffer(0, 8, 2);
    for i in 0..4u32 {
        let f = i as f32;
        b.append_kv(&mut h, &[f, f], &[f, f], i as usize);
    }
    assert!(b.truncate_kv(&mut h, 2));
    assert_eq!(h.cached_len(), 2);
    let (k, v) = b.read_kv_to_host(&h).unwrap();
    // Rows 0 and 1 — the opposite end from `clip_kv_truncates_to_window_size`.
    assert_eq!(k[[0, 0]], 0.0);
    assert_eq!(k[[1, 0]], 1.0);
    assert_eq!(v[[0, 0]], 0.0);
    assert_eq!(v[[1, 0]], 1.0);
}

#[test]
fn truncate_kv_to_the_current_length_is_a_no_op() {
    let b = backend();
    let mut h = b.alloc_kv_buffer(0, 4, 2);
    b.append_kv(&mut h, &[1.0, 2.0], &[3.0, 4.0], 0);
    assert!(b.truncate_kv(&mut h, 1));
    assert_eq!(h.cached_len(), 1);
    let (k, _) = b.read_kv_to_host(&h).unwrap();
    assert_eq!(k[[0, 0]], 1.0);
}

#[test]
fn truncate_kv_to_zero_empties_the_cache() {
    let b = backend();
    let mut h = b.alloc_kv_buffer(0, 4, 2);
    b.append_kv(&mut h, &[1.0, 2.0], &[3.0, 4.0], 0);
    assert!(b.truncate_kv(&mut h, 0));
    assert_eq!(h.cached_len(), 0);
}

#[test]
fn truncate_kv_beyond_the_current_length_refuses() {
    // Asking for more rows than exist is asking this to invent them. It must
    // answer `false` rather than silently leave the handle short, because the
    // caller reads `true` as "the cache is exactly what you asked for".
    let b = backend();
    let mut h = b.alloc_kv_buffer(0, 8, 2);
    b.append_kv(&mut h, &[1.0, 2.0], &[3.0, 4.0], 0);
    assert!(!b.truncate_kv(&mut h, 5));
    assert_eq!(h.cached_len(), 1, "a refused rewind must change nothing");
}

#[test]
fn read_kv_to_host_returns_none_for_empty_handle() {
    let b = backend();
    let h = b.alloc_kv_buffer(0, 4, 2);
    assert!(b.read_kv_to_host(&h).is_none());
}

#[test]
fn cpu_q4k_cache_handle_inner_methods_on_empty_cache() {
    // Build a CpuQ4kCacheHandle with an entirely-empty cache —
    // `cached_len` / `kv_dim` short-circuit to 0 via the
    // `.next().unwrap_or(0)` branch.
    let handle = CpuQ4kCacheHandle {
        cache: vec![None; 4],
    };
    assert_eq!(handle.cached_len(), 0);
    assert_eq!(handle.kv_dim(), 0);
    assert_eq!(handle.backend_name(), "cpu-q4k");
}

#[test]
fn cpu_q4k_cache_handle_inner_methods_with_populated_layer() {
    // Populate one layer slot with `(K, V)` Array2s; the inner
    // methods read the first populated layer's shape.
    let k = Array2::<f32>::zeros((3, 16));
    let v = Array2::<f32>::zeros((3, 16));
    let handle = CpuQ4kCacheHandle {
        cache: vec![None, Some((k, v))],
    };
    assert_eq!(handle.cached_len(), 3);
    assert_eq!(handle.kv_dim(), 16);
}

#[test]
fn cpu_q4k_cache_handle_as_any_round_trip() {
    let mut handle = CpuQ4kCacheHandle {
        cache: vec![None; 2],
    };
    let any: &dyn std::any::Any = handle.as_any();
    assert!(any.downcast_ref::<CpuQ4kCacheHandle>().is_some());
    let any_mut: &mut dyn std::any::Any = handle.as_any_mut();
    assert!(any_mut.downcast_mut::<CpuQ4kCacheHandle>().is_some());
}

/// Wrong-backend handle panics — the dispatch-time downcast helper
/// for `CpuKvHandle` rejects a `CpuQ4kCacheHandle` because the
/// concrete handle type doesn't match. Pinning the panic message
/// surface keeps the misuse error informative.
#[test]
#[should_panic(expected = "foreign handle")]
fn cpu_handle_mut_panics_on_wrong_handle_type() {
    let b = backend();
    let mut h = KvHandle::new(CpuQ4kCacheHandle { cache: vec![None] });
    // Trying to use the Q4K cache handle on the simple-append path
    // must panic — the downcast in `cpu_handle_mut` fails.
    b.append_kv(&mut h, &[0.0; 4], &[0.0; 4], 0);
}

/// The coarse-decode downcast declines gracefully: a per-layer
/// `CpuKvHandle` (or any non-Q4K-cache handle) yields `None`, not a
/// panic — the engine interprets `None` as "coarse-decode this handle
/// elsewhere" and takes its per-layer path. Pinned because the
/// panicking predecessor aborted 1-layer models whose per-layer prefill
/// produced exactly one handle.
#[test]
fn try_cpu_q4k_cache_mut_returns_none_on_wrong_handle_type() {
    let mut h = KvHandle::new(CpuKvHandle::new(0, 4));
    assert!(
        try_cpu_q4k_cache_mut(&mut h).is_none(),
        "wrong handle type must decline, not panic"
    );
    // The matching type still downcasts.
    let mut q4k = KvHandle::new(CpuQ4kCacheHandle { cache: vec![None] });
    assert!(try_cpu_q4k_cache_mut(&mut q4k).is_some());
}

/// `coarse_decode_step` with a per-layer handle returns `None` (the
/// graceful decline surfaced through the trait method engines call).
#[test]
fn coarse_decode_step_declines_gracefully_on_per_layer_handle() {
    use crate::test_fixtures::make_q4k_fixture_index;
    use larql_models::test_fixtures::make_test_q4k_weights;
    let b = backend();
    let weights = make_test_q4k_weights();
    let idx = make_q4k_fixture_index(&weights);
    // A per-layer handle — the shape a per-layer prefill stores.
    let mut h = KvHandle::new(CpuKvHandle::new(0, 4));
    let result = b.coarse_decode_step(&weights, 0u32, Some(&idx), &mut h, 0);
    assert!(
        result.is_none(),
        "coarse decode on a per-layer handle must return None, not panic"
    );
}

/// Immutable downcast helper (`cpu_handle`) panics on the wrong
/// inner type. Reached through `read_kv_to_host`, which takes
/// `&KvHandle` not `&mut`.
#[test]
#[should_panic(expected = "foreign handle")]
fn cpu_handle_panics_on_wrong_handle_type_via_read_kv_to_host() {
    let b = backend();
    let h = KvHandle::new(CpuQ4kCacheHandle {
        cache: vec![None; 1],
    });
    let _ = b.read_kv_to_host(&h);
}

/// `cpu_residual` (immutable) panics on the wrong residual-handle
/// type. Reached through `forward_from_layer`, which takes
/// `&ResidualHandle`.
#[test]
#[should_panic(expected = "foreign residual")]
fn cpu_residual_panics_on_wrong_handle_type() {
    let b = backend();
    let weights = larql_models::test_fixtures::make_test_weights();
    // Build a stub ResidualHandle whose inner isn't CpuResidualHandle.
    struct OtherResidual;
    impl ResidualHandleInner for OtherResidual {
        fn shape(&self) -> (usize, usize) {
            (1, 4)
        }
        fn backend_name(&self) -> &'static str {
            "other"
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }
    // Exercise the stub's own trait methods so they aren't dead — the
    // `shape` accessor is otherwise never called (the panic fires in
    // `cpu_residual` before `forward_from_layer` reads the shape).
    assert_eq!(ResidualHandleInner::shape(&OtherResidual), (1, 4));
    let h = ResidualHandle::new(OtherResidual);
    let _ = b.forward_from_layer(larql_models::WeightsView::dense(&weights), 0, &h, &[0u32]);
}

// ── Coarse default early-return branches ─────────────────────────

use larql_models::test_fixtures::make_test_weights;

/// `coarse_prefill_with_state` returns None on an empty token list.
#[test]
fn coarse_prefill_with_state_returns_none_on_empty_tokens() {
    let b = backend();
    let weights = make_test_weights();
    let result = b.coarse_prefill_with_state(&weights, &[], None, None);
    assert!(result.is_none());
}

/// `coarse_prefill_with_state` returns None when no index is provided.
#[test]
fn coarse_prefill_with_state_returns_none_without_index() {
    let b = backend();
    let weights = make_test_weights();
    let result = b.coarse_prefill_with_state(&weights, &[0u32, 1], None, None);
    assert!(result.is_none());
}

/// `coarse_prefill` delegates to `coarse_prefill_with_state(_, _, _, None)`
/// — same observable behaviour on the empty-token path.
#[test]
fn coarse_prefill_delegates_to_with_state_variant() {
    let b = backend();
    let weights = make_test_weights();
    let result = b.coarse_prefill(&weights, &[], None);
    assert!(result.is_none());
}

/// `coarse_prefill_with_state` happy path on a Q4K-backed fixture
/// — drives `predict_kquant_prefill_with_state` end-to-end and
/// returns the last hidden + a `MetalCoarseHandle`-equivalent on
/// CPU (`CpuQ4kCacheHandle`).
#[test]
fn coarse_prefill_with_state_returns_hidden_and_q4k_cache_handle() {
    use crate::test_fixtures::make_q4k_fixture_index;
    use larql_models::test_fixtures::make_test_q4k_weights;
    let b = backend();
    let weights = make_test_q4k_weights();
    let idx = make_q4k_fixture_index(&weights);
    let mut state = crate::PerLayerDecodeState::with_capacity(weights.num_layers);
    let result = b.coarse_prefill_with_state(&weights, &[0u32, 1, 2], Some(&idx), Some(&mut state));
    let (h, handle) = result.expect("Q4K prefill succeeds");
    assert_eq!(h.shape(), &[1, weights.hidden_size]);
    // Handle width is reported through the inner trait.
    let _ = handle.backend_name();
    // State captured for every layer.
    assert!(state.is_complete_for(weights.num_layers));
}

/// `coarse_decode_step` happy path: prefill first, then decode one
/// more token against the populated Q4K cache handle.
#[test]
fn coarse_decode_step_succeeds_after_prefill() {
    use crate::test_fixtures::make_q4k_fixture_index;
    use larql_models::test_fixtures::make_test_q4k_weights;
    let b = backend();
    let weights = make_test_q4k_weights();
    let idx = make_q4k_fixture_index(&weights);
    let (_h, mut handle) = b
        .coarse_prefill(&weights, &[0u32, 1, 2], Some(&idx))
        .expect("prefill seeds the handle");
    let result = b.coarse_decode_step(&weights, 4u32, Some(&idx), &mut handle, 3);
    let h = result.expect("decode step succeeds with populated handle");
    assert_eq!(h.shape(), &[1, weights.hidden_size]);
}

/// `coarse_decode_step_with_state` happy path: prefill then decode one
/// step with per-layer state capture. The Q4K fixture supports the
/// direct-matvec decode, so this drives the
/// `predict_kquant_decode_step_direct_with_state` arm and the captured
/// state is complete for every layer.
#[test]
fn coarse_decode_step_with_state_captures_per_layer_state() {
    use crate::test_fixtures::make_q4k_fixture_index;
    use larql_models::test_fixtures::make_test_q4k_weights;
    let b = backend();
    let weights = make_test_q4k_weights();
    let idx = make_q4k_fixture_index(&weights);
    let (_h, mut handle) = b
        .coarse_prefill(&weights, &[0u32, 1, 2], Some(&idx))
        .expect("prefill seeds the handle");
    let mut state = crate::PerLayerDecodeState::with_capacity(weights.num_layers);
    let result = b.coarse_decode_step_with_state(
        &weights,
        4u32,
        Some(&idx),
        &mut handle,
        3,
        Some(&mut state),
    );
    let h = result.expect("decode-step-with-state succeeds on direct-matvec fixture");
    assert_eq!(h.shape(), &[1, weights.hidden_size]);
}

/// `coarse_decode_step_with_state` returns None when no index is
/// provided — covers the `let index = index?;` early-return.
#[test]
fn coarse_decode_step_with_state_returns_none_without_index() {
    let b = backend();
    let weights = make_test_weights();
    let mut handle = KvHandle::new(CpuQ4kCacheHandle { cache: vec![None] });
    let result = b.coarse_decode_step_with_state(&weights, 0u32, None, &mut handle, 0, None);
    assert!(result.is_none());
}

/// `coarse_decode_step` returns None when no index is provided —
/// covers the `let index = index?;` early-return in the non-state
/// variant.
#[test]
fn coarse_decode_step_returns_none_without_index() {
    let b = backend();
    let weights = make_test_weights();
    let mut handle = KvHandle::new(CpuQ4kCacheHandle { cache: vec![None] });
    let result = b.coarse_decode_step(&weights, 0u32, None, &mut handle, 0);
    assert!(result.is_none());
}

/// `read_kv_row_at` on the coarse Q4K handle returns the row at the
/// requested ABSOLUTE position, value-equal to the per-position K/V the
/// state dump captured — the contract `WindowedCheckpointEngine`'s
/// boundary-checkpoint readback depends on.
#[test]
fn read_kv_row_at_returns_absolute_rows_matching_state_dump() {
    use crate::test_fixtures::make_q4k_fixture_index;
    use larql_models::test_fixtures::make_test_q4k_weights;
    let b = backend();
    let weights = make_test_q4k_weights();
    let idx = make_q4k_fixture_index(&weights);
    const PROMPT: [u32; 3] = [0, 1, 2];
    let mut state = crate::PerLayerDecodeState::with_capacity(weights.num_layers);
    let (_h, handle) = b
        .coarse_prefill_with_state(&weights, &PROMPT, Some(&idx), Some(&mut state))
        .expect("Q4K prefill succeeds");
    let k_dump: Vec<Array2<f32>> = state
        .k_new_per_layer
        .into_iter()
        .map(|h| h.into_array())
        .collect();
    let v_dump: Vec<Array2<f32>> = state
        .v_new_per_layer
        .into_iter()
        .map(|h| h.into_array())
        .collect();
    for layer in 0..weights.num_layers {
        for pos in 0..PROMPT.len() {
            let (k_row, v_row) = b
                .read_kv_row_at(&handle, layer, pos)
                .expect("in-range row readback");
            assert_eq!(
                k_row,
                k_dump[layer].row(pos).to_vec(),
                "layer {layer} pos {pos}: K row != state-dump row"
            );
            assert_eq!(
                v_row,
                v_dump[layer].row(pos).to_vec(),
                "layer {layer} pos {pos}: V row != state-dump row"
            );
        }
        // Out-of-range position → None, not garbage.
        assert!(b.read_kv_row_at(&handle, layer, PROMPT.len()).is_none());
    }
    // Out-of-range layer → None.
    assert!(b.read_kv_row_at(&handle, weights.num_layers, 0).is_none());
}

/// `read_kv_row_at` returns None for the per-layer `CpuKvHandle` shape
/// (no cross-layer cache to index into).
#[test]
fn read_kv_row_at_returns_none_for_per_layer_handle() {
    let b = backend();
    let mut h = b.alloc_kv_buffer(0, 4, 4);
    b.append_kv(&mut h, &[1.0; 4], &[2.0; 4], 0);
    assert!(b.read_kv_row_at(&h, 0, 0).is_none());
}

/// `coarse_prefill_with_state` returns None when the arch can't run
/// the cached-decode path (hybrid-MoE), even with an index + tokens —
/// covers the `!supports_cached_decode(weights)` early-return.
#[test]
fn coarse_prefill_with_state_returns_none_on_unsupported_arch() {
    use crate::test_fixtures::make_q4k_fixture_index;
    let b = backend();
    let weights = larql_models::test_fixtures::make_test_gemma4_moe_weights();
    assert!(
        !crate::kquant_forward::supports_cached_decode(&weights),
        "MoE arch must not support cached decode"
    );
    // A real index is supplied so the `index?` gate passes and the
    // `supports_cached_decode` gate is the one that bites.
    let idx = make_q4k_fixture_index(&weights);
    let result = b.coarse_prefill_with_state(&weights, &[0u32, 1], Some(&idx), None);
    assert!(result.is_none());
}

// ── Q4K-direct attention_step path (env-gated) ──────────────────────
//
// `attention_step`'s opt-in Q4K-direct branch fires only under
// `LARQL_Q4K_DIRECT_ATTN=1`. The flag is read once through a
// process-wide `OnceLock`, so this single test owns that
// initialisation (no other test in this binary reads the flag). It
// sets the var before the first `attention_step` call, which both
// seeds the `OnceLock` (covering `q4k_direct_attn_enabled`) and drives
// the `Some(idx)` q4k-direct dispatch arm.
#[test]
fn attention_step_q4k_direct_branch_fires_when_env_enabled() {
    use crate::test_fixtures::make_q4k_fixture_index;
    use larql_models::test_fixtures::make_test_q4k_weights;

    // Thread-local override (NOT `set_var`, which races concurrent `getenv`
    // → SIGSEGV); cleared on drop, so it can't leak to a sibling test.
    let _guard = crate::options::FastPathGuard::set(&[(crate::options::ENV_Q4K_DIRECT_ATTN, true)]);
    assert!(q4k_direct_attn_enabled());

    let b = backend();
    let weights = make_test_q4k_weights();
    let idx = make_q4k_fixture_index(&weights);
    let mut handle = b.alloc_kv_buffer(0, 8, weights.num_kv_heads * weights.head_dim);
    let query = Array2::from_elem((1, weights.hidden_size), 0.1f32);

    // Step 1 (empty prior KV) through the q4k-direct branch.
    let h1 = b
        .attention_step(
            larql_models::WeightsView::dense(&weights),
            &query,
            &mut handle,
            0,
            0,
            Some(&idx),
        )
        .expect("q4k-direct attention_step returns Some on the Q4K fixture");
    assert_eq!(h1.shape(), &[1, weights.hidden_size]);
    assert_eq!(handle.cached_len(), 1, "KV grew by 1");

    // Step 2 (prior KV present) keeps using the q4k-direct branch.
    let h2 = b
        .attention_step(
            larql_models::WeightsView::dense(&weights),
            &query,
            &mut handle,
            0,
            1,
            Some(&idx),
        )
        .expect("second q4k-direct attention_step succeeds");
    assert_eq!(h2.shape(), &[1, weights.hidden_size]);
    assert_eq!(handle.cached_len(), 2, "KV grew to 2");
}

/// The q4k-direct branch falls back to the f32 path per layer when the
/// index lacks Q4K attn bytes. With the flag enabled but an empty
/// index, `run_attention_block_decode_step_q4k_direct` returns None and
/// `attention_step` takes the `.or_else(f32_path)` fallback — which
/// also returns None here because `make_test_weights` has no f32 attn
/// tensors for the q4k fixture's dimensions… so we use the Q4K weights
/// (which DO carry f32 attn tensors) and an empty index.
#[test]
fn attention_step_q4k_direct_falls_back_to_f32_on_empty_index() {
    use larql_models::test_fixtures::make_test_q4k_weights;

    // Enable the q4k-direct gate on this thread (override, not `set_var`).
    let _guard = crate::options::FastPathGuard::set(&[(crate::options::ENV_Q4K_DIRECT_ATTN, true)]);

    struct EmptyIdx;
    impl crate::KvIndex for EmptyIdx {}

    let b = backend();
    let weights = make_test_q4k_weights();
    let mut handle = b.alloc_kv_buffer(0, 8, weights.num_kv_heads * weights.head_dim);
    let query = Array2::from_elem((1, weights.hidden_size), 0.1f32);

    // EmptyIdx → q4k-direct returns None → `.or_else(f32_path)` runs the
    // f32-BLAS path on `weights.tensors` (the fixture carries f32 attn
    // tensors), which succeeds.
    let h = b
        .attention_step(
            larql_models::WeightsView::dense(&weights),
            &query,
            &mut handle,
            0,
            0,
            Some(&EmptyIdx),
        )
        .expect("f32 fallback succeeds when the index lacks Q4K attn bytes");
    assert_eq!(h.shape(), &[1, weights.hidden_size]);
    assert_eq!(handle.cached_len(), 1);
}
