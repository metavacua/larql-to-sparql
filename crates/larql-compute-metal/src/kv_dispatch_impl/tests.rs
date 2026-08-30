//! `tests` for [`super`].
//!
//! Split out of `kv_dispatch_impl.rs` to keep the implementation file within
//! the repo's per-file size budget.

//! Coverage tests for the CPU-delegation `KvDispatch` scaffold.
//!
//! Each method on `MetalBackend` forwards to `CpuBackend` at Step 4;
//! the assertions here drive the delegation paths and (where the
//! result is observable) confirm shape parity with the direct CPU
//! call. The coarse Q4_K fused methods (`coarse_prefill*`,
//! `coarse_decode_step*`) need a real Q4_K vindex fixture and are
//! covered end-to-end in `tests/test_metal_decode_synthetic.rs`.
use super::*;
use larql_models::test_fixtures::make_test_weights;

fn backend() -> MetalBackend {
    MetalBackend::new().expect("Metal device available on test host")
}

#[test]
fn alloc_kv_buffer_delegates_to_cpu() {
    let m = backend();
    let h = m.alloc_kv_buffer(
        /*layer=*/ 0, /*max_tokens=*/ 8, /*kv_dim=*/ 32,
    );
    assert_eq!(h.cached_len(), 0);
    assert_eq!(h.kv_dim(), 32);
}

#[test]
fn append_and_read_kv_round_trips_through_cpu() {
    let m = backend();
    let mut h = m.alloc_kv_buffer(0, 4, 4);
    m.append_kv(&mut h, &[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0], 0);
    m.append_kv(
        &mut h,
        &[9.0, 10.0, 11.0, 12.0],
        &[13.0, 14.0, 15.0, 16.0],
        1,
    );
    let (k, v) = m.read_kv_to_host(&h).expect("read after append");
    assert_eq!(k.shape(), &[2, 4]);
    assert_eq!(v.shape(), &[2, 4]);
    assert_eq!(k[[0, 0]], 1.0);
    assert_eq!(v[[1, 3]], 16.0);
}

#[test]
fn clip_kv_truncates_to_window() {
    let m = backend();
    let mut h = m.alloc_kv_buffer(0, 8, 2);
    for i in 0..4u32 {
        let f = i as f32;
        m.append_kv(&mut h, &[f, f], &[f, f], i as usize);
    }
    m.clip_kv(&mut h, 2);
    let (k, _) = m.read_kv_to_host(&h).expect("read after clip");
    assert_eq!(k.shape(), &[2, 2], "clip to window=2 keeps newest 2 rows");
}

#[test]
fn attention_step_delegates_through_cpu() {
    let weights = make_test_weights();
    let m = backend();
    let tokens = vec![0u32, 1, 2];
    let h_in = larql_compute::forward::embed_tokens_pub(&weights, &tokens);
    let (_, mut kv) = m
        .attention_prefill(
            larql_models::WeightsView::dense(&weights),
            &h_in,
            0,
            None,
            None,
        )
        .expect("prefill");
    let h_new = larql_compute::forward::embed_tokens_pub(&weights, &[3u32]);
    let h = m
        .attention_step(
            larql_models::WeightsView::dense(&weights),
            &h_new,
            &mut kv,
            0,
            tokens.len(),
            None,
        )
        .expect("attention_step");
    assert_eq!(h.shape(), &[1, weights.hidden_size]);
}

#[test]
fn attention_step_windowed_delegates_through_cpu() {
    let weights = make_test_weights();
    let m = backend();
    let tokens = vec![0u32, 1, 2];
    let h_in = larql_compute::forward::embed_tokens_pub(&weights, &tokens);
    let (_, mut kv) = m
        .attention_prefill(
            larql_models::WeightsView::dense(&weights),
            &h_in,
            0,
            None,
            None,
        )
        .expect("prefill");
    let h_new = larql_compute::forward::embed_tokens_pub(&weights, &[3u32]);
    let h = m
        .attention_step_windowed(
            larql_models::WeightsView::dense(&weights),
            &h_new,
            &mut kv,
            0,
            tokens.len(),
            64,
            None,
        )
        .expect("windowed attention_step");
    assert_eq!(h.shape(), &[1, weights.hidden_size]);
}

#[test]
fn attention_prefill_delegates_through_cpu() {
    let weights = make_test_weights();
    let m = backend();
    let tokens = vec![0u32, 1, 2];
    let h_in = larql_compute::forward::embed_tokens_pub(&weights, &tokens);
    let (h, kv) = m
        .attention_prefill(
            larql_models::WeightsView::dense(&weights),
            &h_in,
            0,
            None,
            None,
        )
        .expect("prefill");
    assert_eq!(h.shape(), &[tokens.len(), weights.hidden_size]);
    assert_eq!(kv.cached_len(), tokens.len());
}

#[test]
fn recompute_kv_from_residuals_delegates_through_cpu() {
    // `CpuBackend` doesn't override the trait default (it's a Metal-
    // shaped intent, MarkovResidual-only), so delegation returns the
    // default `None`. The point of the test is to drive the Metal
    // dispatch into CpuBackend and confirm it surfaces the same
    // `None` — exercising the delegation pathway.
    let weights = make_test_weights();
    let m = backend();
    let cpu = larql_compute::CpuBackend;
    let residuals =
        Array2::from_shape_vec((3, weights.hidden_size), vec![0.0; 3 * weights.hidden_size])
            .unwrap();
    let m_result =
        m.recompute_kv_from_residuals(larql_models::WeightsView::dense(&weights), &residuals, 0);
    let cpu_result =
        cpu.recompute_kv_from_residuals(larql_models::WeightsView::dense(&weights), &residuals, 0);
    assert_eq!(
        m_result.is_some(),
        cpu_result.is_some(),
        "Metal delegation must match CpuBackend"
    );
}

#[test]
fn upload_boundary_residual_delegates_through_cpu() {
    let weights = make_test_weights();
    let m = backend();
    let residual =
        Array2::from_shape_vec((1, weights.hidden_size), vec![0.0; weights.hidden_size]).unwrap();
    let handle = m.upload_boundary_residual(&residual).expect("upload");
    let _ = handle;
}

#[test]
fn forward_from_layer_delegates_through_cpu() {
    let weights = make_test_weights();
    let m = backend();
    let residual =
        Array2::from_shape_vec((1, weights.hidden_size), vec![0.0; weights.hidden_size]).unwrap();
    let handle = m.upload_boundary_residual(&residual).expect("upload");
    let h = m
        .forward_from_layer(
            larql_models::WeightsView::dense(&weights),
            1,
            &handle,
            &[0u32, 1, 2],
        )
        .expect("forward_from_layer");
    assert_eq!(h.ncols(), weights.hidden_size);
}

#[test]
fn residual_norm_store_delegates_through_cpu() {
    let m = backend();
    let cpu = larql_compute::CpuBackend;
    let x = Array2::from_shape_vec((2, 4), (0..8).map(|i| i as f32).collect()).unwrap();
    let res = Array2::from_shape_vec((2, 4), (0..8).map(|i| -(i as f32)).collect()).unwrap();
    let norm = vec![1.0; 4];
    let h_m = m.residual_norm_store(&x, &res, &norm);
    let h_c = cpu.residual_norm_store(&x, &res, &norm);
    assert_eq!(h_m, h_c, "Metal delegation must bit-match CpuBackend");
}

#[test]
fn read_kv_row_at_returns_none_when_cache_empty() {
    let m = backend();
    let sentinel = KvHandle::new(MetalCoarseHandle);
    assert!(m.read_kv_row_at(&sentinel, 0, 0).is_none());
}

// ── MetalCoarseHandle inner impl ──────────────────────────────────

#[test]
fn metal_coarse_handle_reports_sentinel_values() {
    let mut h = MetalCoarseHandle;
    assert_eq!(KvHandleInner::cached_len(&h), 0);
    assert_eq!(KvHandleInner::kv_dim(&h), 0);
    assert_eq!(KvHandleInner::backend_name(&h), "metal-coarse");
    let any: &dyn std::any::Any = KvHandleInner::as_any(&h);
    assert!(any.downcast_ref::<MetalCoarseHandle>().is_some());
    let any_mut: &mut dyn std::any::Any = KvHandleInner::as_any_mut(&mut h);
    assert!(any_mut.downcast_mut::<MetalCoarseHandle>().is_some());
}

#[test]
fn coarse_decode_step_without_index_returns_none() {
    let weights = make_test_weights();
    let m = backend();
    let mut handle = KvHandle::new(MetalCoarseHandle);
    let result = m.coarse_decode_step(&weights, 0u32, None, &mut handle, 0);
    assert!(result.is_none());
}

/// Drives `MetalBackend::coarse_prefill` end-to-end against the Q4_K
/// fixture — runs through `fused_prefill` and exits via
/// `prefill_kquant` on the real Metal kernel. This is the test that
/// makes the file's coverage jump from 60% → 90%+.
#[test]
fn coarse_prefill_with_q4k_fixture_returns_hidden_and_handle() {
    use larql_compute::test_fixtures::make_q4k_fixture_index;
    use larql_models::test_fixtures::make_test_q4k_weights;
    let m = backend();
    let weights = make_test_q4k_weights();
    let idx = make_q4k_fixture_index(&weights);
    let result = m.coarse_prefill(&weights, &[0u32, 1, 2], Some(&idx));
    let (h, _handle) = result.expect("Metal Q4K prefill succeeds");
    assert_eq!(h.shape(), &[1, weights.hidden_size]);
}

/// `coarse_prefill_with_state` happy path on Metal with Q4_K.
#[test]
fn coarse_prefill_with_state_drives_metal_decode_loop() {
    use larql_compute::test_fixtures::make_q4k_fixture_index;
    use larql_models::test_fixtures::make_test_q4k_weights;
    let m = backend();
    let weights = make_test_q4k_weights();
    let idx = make_q4k_fixture_index(&weights);
    let mut state = larql_compute::PerLayerDecodeState::with_capacity(weights.num_layers);
    let result = m.coarse_prefill_with_state(&weights, &[0u32, 1, 2], Some(&idx), Some(&mut state));
    let (h, _handle) = result.expect("Metal Q4K prefill-with-state succeeds");
    assert_eq!(h.shape(), &[1, weights.hidden_size]);
    assert!(state.is_complete_for(weights.num_layers));
}

/// `coarse_decode_step` end-to-end on Metal with the Q4_K fixture.
#[test]
fn coarse_decode_step_with_q4k_fixture_returns_hidden() {
    use larql_compute::test_fixtures::make_q4k_fixture_index;
    use larql_models::test_fixtures::make_test_q4k_weights;
    let m = backend();
    let weights = make_test_q4k_weights();
    let idx = make_q4k_fixture_index(&weights);
    // Seed the KV cache via prefill.
    let (_h, mut handle) = m
        .coarse_prefill(&weights, &[0u32, 1, 2], Some(&idx))
        .expect("prefill seeds the cache");
    let result = m.coarse_decode_step(&weights, 4u32, Some(&idx), &mut handle, 3);
    let h = result.expect("Metal Q4K decode step returns Some");
    assert_eq!(h.shape(), &[1, weights.hidden_size]);
}

/// `coarse_decode_step_with_state_masked` over all 3 mask variants
/// against Metal + the Q4_K fixture. Drives the masked-state-dump
/// bridging logic in `kv_dispatch_impl`.
#[test]
fn coarse_decode_step_with_state_masked_over_all_mask_variants() {
    use larql_compute::test_fixtures::make_q4k_fixture_index;
    use larql_models::test_fixtures::make_test_q4k_weights;
    let m = backend();
    let weights = make_test_q4k_weights();
    let idx = make_q4k_fixture_index(&weights);
    let (_h, mut handle) = m
        .coarse_prefill(&weights, &[0u32, 1, 2], Some(&idx))
        .expect("prefill seeds the cache");
    for mask in [
        larql_compute::StateDumpMask::Full,
        larql_compute::StateDumpMask::HOnly,
        larql_compute::StateDumpMask::None,
    ] {
        let mut state = larql_compute::PerLayerDecodeState::with_capacity(weights.num_layers);
        let result = m.coarse_decode_step_with_state_masked(
            &weights,
            5u32,
            Some(&idx),
            &mut handle,
            4,
            Some(&mut state),
            mask,
        );
        assert!(
            result.is_some(),
            "Metal decode-step-with-state-masked should return Some under {mask:?}"
        );
    }
}

#[test]
fn coarse_decode_step_with_state_masked_without_index_returns_none() {
    let weights = make_test_weights();
    let m = backend();
    let mut handle = KvHandle::new(MetalCoarseHandle);
    let result = m.coarse_decode_step_with_state_masked(
        &weights,
        0u32,
        None,
        &mut handle,
        0,
        None,
        larql_compute::StateDumpMask::Full,
    );
    assert!(result.is_none());
}
