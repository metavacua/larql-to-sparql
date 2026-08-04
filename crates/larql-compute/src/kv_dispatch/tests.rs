//! Trait-default contract tests. `KvDispatch` has no supertraits, so
//! a stub backend that overrides nothing exercises every default
//! body. These tests document the documented "implement-me" contract:
//! every default either returns `None` (engines treat it as
//! "backend doesn't support this intent, fall back") or panics with
//! a `not implemented for this backend` message.
//!
//! The `attention_step_windowed` and `residual_norm_store` defaults
//! have meaningful decompositions; tests check the actual decomposed
//! behaviour, not just panic semantics.
//!
//! Coverage role: this module's lines are dominated by trait-default
//! bodies. Without these tests, those bodies are unreachable from
//! the rest of the crate because every concrete backend
//! (`CpuBackend`, `MetalBackend`) overrides the methods that don't
//! `unimplemented!()`.

use super::*;
use ndarray::Array2;

// ── Stub backend with all-default `KvDispatch` ───────────────────

struct StubKvBackend;
impl KvDispatch for StubKvBackend {}

struct StubKvInner {
    len: usize,
    dim: usize,
}
impl KvHandleInner for StubKvInner {
    fn cached_len(&self) -> usize {
        self.len
    }
    fn kv_dim(&self) -> usize {
        self.dim
    }
    fn backend_name(&self) -> &'static str {
        "stub"
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

struct StubResidualInner {
    shape: (usize, usize),
}
impl ResidualHandleInner for StubResidualInner {
    fn shape(&self) -> (usize, usize) {
        self.shape
    }
    fn backend_name(&self) -> &'static str {
        "stub"
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

struct StubCodec;
impl CompressionCodec for StubCodec {
    fn encode(&self, vec: &[f32]) -> Vec<u8> {
        vec.iter().flat_map(|f| f.to_le_bytes()).collect()
    }
    fn decode(&self, bytes: &[u8], dim: usize) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .take(dim)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()
    }
    fn name(&self) -> &str {
        "stub"
    }
}

fn stub_kv_handle(len: usize, dim: usize) -> KvHandle {
    KvHandle::new(StubKvInner { len, dim })
}

fn stub_residual_handle(rows: usize, cols: usize) -> ResidualHandle {
    ResidualHandle::new(StubResidualInner {
        shape: (rows, cols),
    })
}

// ── KvHandle / ResidualHandle accessor coverage ──────────────────

#[test]
fn kv_handle_accessors_through_stub_inner() {
    let mut handle = stub_kv_handle(5, 64);
    assert_eq!(handle.cached_len(), 5);
    assert_eq!(handle.kv_dim(), 64);
    assert_eq!(handle.backend_name(), "stub");
    // `as_inner` + `as_inner_mut` paths.
    let _: &dyn KvHandleInner = handle.as_inner();
    let _: &mut dyn KvHandleInner = handle.as_inner_mut();
}

#[test]
fn residual_handle_accessors_through_stub_inner() {
    let handle = stub_residual_handle(3, 4);
    assert_eq!(handle.shape(), (3, 4));
    assert_eq!(handle.backend_name(), "stub");
    let _: &dyn ResidualHandleInner = handle.as_inner();
}

#[test]
fn stub_codec_round_trips() {
    let codec = StubCodec;
    assert_eq!(codec.name(), "stub");
    let bytes = codec.encode(&[1.5_f32, 2.25, -0.5]);
    let back = codec.decode(&bytes, 3);
    assert_eq!(back, vec![1.5, 2.25, -0.5]);
}

// ── KvDispatch default bodies — None returns ─────────────────────

#[test]
fn default_read_kv_to_host_returns_none() {
    let backend = StubKvBackend;
    let handle = stub_kv_handle(0, 64);
    assert!(backend.read_kv_to_host(&handle).is_none());
}

#[test]
fn default_attention_step_returns_none() {
    let backend = StubKvBackend;
    let weights = larql_models::test_fixtures::make_test_weights();
    let mut handle = stub_kv_handle(0, weights.hidden_size);
    let query = Array2::zeros((1, weights.hidden_size));
    assert!(backend
        .attention_step(
            larql_models::WeightsView::dense(&weights),
            &query,
            &mut handle,
            0,
            0,
            None
        )
        .is_none());
}

#[test]
fn default_attention_step_windowed_propagates_none() {
    // Default decomposes into `attention_step` (returns None) then
    // `clip_kv`. The `?` on None short-circuits before clip_kv would
    // panic. Tests the default-body's None-propagation branch.
    let backend = StubKvBackend;
    let weights = larql_models::test_fixtures::make_test_weights();
    let mut handle = stub_kv_handle(0, weights.hidden_size);
    let query = Array2::zeros((1, weights.hidden_size));
    assert!(backend
        .attention_step_windowed(
            larql_models::WeightsView::dense(&weights),
            &query,
            &mut handle,
            0,
            0,
            4,
            None
        )
        .is_none());
}

#[test]
fn default_attention_prefill_returns_none() {
    let backend = StubKvBackend;
    let weights = larql_models::test_fixtures::make_test_weights();
    let tokens = Array2::zeros((2, weights.hidden_size));
    assert!(backend
        .attention_prefill(
            larql_models::WeightsView::dense(&weights),
            &tokens,
            0,
            None,
            None
        )
        .is_none());
}

#[test]
fn default_recompute_kv_from_residuals_returns_none() {
    let backend = StubKvBackend;
    let weights = larql_models::test_fixtures::make_test_weights();
    let residuals = Array2::zeros((1, weights.hidden_size));
    assert!(backend
        .recompute_kv_from_residuals(larql_models::WeightsView::dense(&weights), &residuals, 0)
        .is_none());
}

#[test]
fn default_upload_boundary_residual_returns_none() {
    let backend = StubKvBackend;
    let residual = Array2::zeros((1, 8));
    assert!(backend.upload_boundary_residual(&residual).is_none());
}

#[test]
fn default_forward_from_layer_returns_none() {
    let backend = StubKvBackend;
    let weights = larql_models::test_fixtures::make_test_weights();
    let residuals = stub_residual_handle(1, weights.hidden_size);
    assert!(backend
        .forward_from_layer(
            larql_models::WeightsView::dense(&weights),
            0,
            &residuals,
            &[0u32]
        )
        .is_none());
}

// ── KvDispatch default bodies — `unimplemented!()` panics ────────

#[test]
#[should_panic(expected = "alloc_kv_buffer not implemented")]
fn default_alloc_kv_buffer_panics() {
    let backend = StubKvBackend;
    let _ = backend.alloc_kv_buffer(0, 32, 64);
}

#[test]
#[should_panic(expected = "append_kv not implemented")]
fn default_append_kv_panics() {
    let backend = StubKvBackend;
    let mut handle = stub_kv_handle(0, 4);
    backend.append_kv(&mut handle, &[0.0; 4], &[0.0; 4], 0);
}

#[test]
#[should_panic(expected = "clip_kv not implemented")]
fn default_clip_kv_panics() {
    let backend = StubKvBackend;
    let mut handle = stub_kv_handle(0, 4);
    backend.clip_kv(&mut handle, 2);
}

#[test]
fn default_truncate_kv_declines_instead_of_panicking() {
    // Unlike its siblings above, this default must *return* rather than
    // panic: "this backend cannot rewind" is a state the caller handles by
    // invalidating the cache, not a programming error. A panic here would
    // turn an unported backend into a crash on every refused decode step.
    let backend = StubKvBackend;
    let mut handle = stub_kv_handle(0, 4);
    assert!(!backend.truncate_kv(&mut handle, 0));
}

#[test]
#[should_panic(expected = "compressed_kv_append not implemented")]
fn default_compressed_kv_append_panics() {
    let backend = StubKvBackend;
    let mut handle = stub_kv_handle(0, 4);
    let k = Array2::zeros((1, 4));
    let v = Array2::zeros((1, 4));
    let codec = StubCodec;
    backend.compressed_kv_append(&mut handle, &k, &v, &codec);
}

// ── Real default decomposition: residual_norm_store ──────────────

#[test]
fn default_residual_norm_store_decomposes_add_plus_rmsnorm() {
    // The trait's default body implements `residual_add` followed by
    // a per-row rmsnorm with eps=1e-6. Test against a hand-computed
    // expected output so the decomposition body is exercised end-to-end.
    let backend = StubKvBackend;
    let x = Array2::from_shape_vec((1, 4), vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let residual = Array2::from_shape_vec((1, 4), vec![0.5_f32, 0.5, 0.5, 0.5]).unwrap();
    let norm_weights = vec![1.0_f32; 4];

    let out = backend.residual_norm_store(&x, &residual, &norm_weights);

    // Hand-computed: added = [1.5, 2.5, 3.5, 4.5]
    // mean_sq = (1.5² + 2.5² + 3.5² + 4.5²) / 4 = (2.25+6.25+12.25+20.25)/4 = 10.25
    // scale = 1.0 / sqrt(10.25 + 1e-6) ≈ 0.31234752...
    let added = [1.5_f32, 2.5, 3.5, 4.5];
    let mean_sq: f32 = added.iter().map(|v| v * v).sum::<f32>() / 4.0;
    let scale = (mean_sq + 1e-6).sqrt().recip();
    for j in 0..4 {
        let expected = added[j] * scale;
        assert!(
            (out[[0, j]] - expected).abs() < 1e-5,
            "col {j}: out={} expected={}",
            out[[0, j]],
            expected
        );
    }
}

#[test]
fn default_residual_norm_store_applies_per_column_weights() {
    // Same shape but non-uniform norm_weights — verifies the per-column
    // multiplication branch.
    let backend = StubKvBackend;
    let x = Array2::from_shape_vec((1, 2), vec![1.0_f32, 1.0]).unwrap();
    let residual = Array2::from_shape_vec((1, 2), vec![0.0_f32, 0.0]).unwrap();
    let norm_weights = vec![2.0_f32, 0.5_f32];
    let out = backend.residual_norm_store(&x, &residual, &norm_weights);
    // mean_sq = (1 + 1) / 2 = 1.0, scale ≈ 1 / sqrt(1+1e-6)
    let scale = (1.0_f32 + 1e-6).sqrt().recip();
    assert!((out[[0, 0]] - scale * 2.0).abs() < 1e-5);
    assert!((out[[0, 1]] - scale * 0.5).abs() < 1e-5);
}

// ── EngineBackend blanket impl ───────────────────────────────────

#[test]
fn engine_backend_as_compute_returns_self() {
    // Any type implementing ComputeBackend + KvDispatch auto-implements
    // EngineBackend. Use CpuBackend, then call `as_compute` to exercise
    // the blanket impl's body (one line: `self`).
    let backend = crate::CpuBackend;
    let as_engine: &dyn EngineBackend = &backend;
    let _: &dyn crate::ComputeBackend = as_engine.as_compute();
}

// ── Coarse-fused defaults ─────────────────────────────────────────
//
// The 4 coarse_* methods + `read_kv_row_at` are trait defaults that
// return None / fall through to the simpler-coarse variant. Engines
// probe via Option-return; pinning the default `None` shape keeps
// the contract documented and the lines covered.

use larql_models::test_fixtures::make_test_weights;

#[test]
fn default_coarse_prefill_returns_none() {
    let weights = make_test_weights();
    let backend = StubKvBackend;
    let result = backend.coarse_prefill(&weights, &[0u32, 1], None);
    assert!(result.is_none());
}

#[test]
fn default_coarse_decode_step_returns_none() {
    let weights = make_test_weights();
    let backend = StubKvBackend;
    let mut handle = KvHandle::new(StubKvInner {
        len: 0,
        dim: weights.hidden_size,
    });
    let result = backend.coarse_decode_step(&weights, 0, None, &mut handle, 0);
    assert!(result.is_none());
}

#[test]
fn default_coarse_prefill_with_state_delegates_to_coarse_prefill() {
    let weights = make_test_weights();
    let backend = StubKvBackend;
    let mut state = PerLayerDecodeState::with_capacity(weights.num_layers);
    let result = backend.coarse_prefill_with_state(&weights, &[0u32, 1], None, Some(&mut state));
    assert!(result.is_none());
}

#[test]
fn default_coarse_decode_step_with_state_delegates() {
    let weights = make_test_weights();
    let backend = StubKvBackend;
    let mut handle = KvHandle::new(StubKvInner {
        len: 0,
        dim: weights.hidden_size,
    });
    let mut state = PerLayerDecodeState::with_capacity(weights.num_layers);
    let result =
        backend.coarse_decode_step_with_state(&weights, 0, None, &mut handle, 0, Some(&mut state));
    assert!(result.is_none());
}

#[test]
fn default_coarse_decode_step_with_state_masked_delegates() {
    let weights = make_test_weights();
    let backend = StubKvBackend;
    let mut handle = KvHandle::new(StubKvInner {
        len: 0,
        dim: weights.hidden_size,
    });
    for mask in [
        crate::StateDumpMask::Full,
        crate::StateDumpMask::HOnly,
        crate::StateDumpMask::None,
    ] {
        let mut state = PerLayerDecodeState::with_capacity(weights.num_layers);
        let result = backend.coarse_decode_step_with_state_masked(
            &weights,
            0,
            None,
            &mut handle,
            0,
            Some(&mut state),
            mask,
        );
        assert!(result.is_none(), "mask {mask:?} should produce None");
    }
}

#[test]
fn default_read_kv_row_at_returns_none() {
    let backend = StubKvBackend;
    let handle = KvHandle::new(StubKvInner { len: 0, dim: 4 });
    assert!(backend.read_kv_row_at(&handle, 0, 0).is_none());
}

#[test]
fn default_backend_resident_kv_bytes_is_zero() {
    // Default 0 = every byte is reachable through a handle; a backend
    // that hides K/V internally must override or its memory accounting
    // double-counts nothing and under-reports everything.
    let backend = StubKvBackend;
    assert_eq!(backend.backend_resident_kv_bytes(), 0);
}

#[test]
fn default_per_layer_is_host_delegated_is_false() {
    // Default false = "a backend is assumed to mean what it says".
    let backend = StubKvBackend;
    assert!(!backend.per_layer_is_host_delegated());
}

// ── Windowed coarse defaults fail closed ─────────────────────────
//
// The windowed variants' defaults promise: `window: None` delegates to
// the window-less method, `Some(_)` answers None even when the window-
// less path would succeed — "this backend cannot bound what you asked
// me to bound". A stub whose window-less methods DO succeed proves the
// Some(_) arm declines without consulting them.

struct CoarseCapableStub;
impl KvDispatch for CoarseCapableStub {
    fn coarse_prefill(
        &self,
        _weights: &larql_models::ModelWeights,
        _token_ids: &[u32],
        _index: Option<&dyn crate::KvIndex>,
    ) -> Option<(Array2<f32>, KvHandle)> {
        Some((Array2::zeros((1, 4)), stub_kv_handle(1, 4)))
    }
    fn coarse_decode_step(
        &self,
        _weights: &larql_models::ModelWeights,
        _token_id: u32,
        _index: Option<&dyn crate::KvIndex>,
        _handle: &mut KvHandle,
        _abs_position: usize,
    ) -> Option<Array2<f32>> {
        Some(Array2::zeros((1, 4)))
    }
}

#[test]
fn default_coarse_prefill_windowed_delegates_when_unwindowed() {
    let weights = make_test_weights();
    let backend = CoarseCapableStub;
    let result = backend.coarse_prefill_windowed(&weights, &[0u32, 1], None, None);
    assert!(
        result.is_some(),
        "window: None must delegate to coarse_prefill"
    );
}

#[test]
fn default_coarse_prefill_windowed_fails_closed_on_real_window() {
    let weights = make_test_weights();
    let backend = CoarseCapableStub;
    let result = backend.coarse_prefill_windowed(&weights, &[0u32, 1], None, Some(4));
    assert!(
        result.is_none(),
        "a real window must decline even though coarse_prefill succeeds"
    );
}

#[test]
fn default_coarse_decode_step_windowed_delegates_when_unwindowed() {
    let weights = make_test_weights();
    let backend = CoarseCapableStub;
    let mut handle = stub_kv_handle(1, 4);
    let result = backend.coarse_decode_step_windowed(&weights, 0, None, &mut handle, 1, None);
    assert!(
        result.is_some(),
        "window: None must delegate to coarse_decode_step"
    );
}

#[test]
fn default_coarse_decode_step_windowed_fails_closed_on_real_window() {
    let weights = make_test_weights();
    let backend = CoarseCapableStub;
    let mut handle = stub_kv_handle(1, 4);
    let result = backend.coarse_decode_step_windowed(&weights, 0, None, &mut handle, 1, Some(4));
    assert!(
        result.is_none(),
        "a real window must decline even though coarse_decode_step succeeds"
    );
}

// ── attention_step_windowed happy path ───────────────────────────
//
// The None-propagation branch is pinned above; this stub drives the
// success branch of the default body: attention_step returns a hidden
// state, then the default MUST clip the cache to the window before
// returning it.

struct AttnCapableStub {
    clipped_to: std::sync::atomic::AtomicUsize,
}
impl KvDispatch for AttnCapableStub {
    fn attention_step(
        &self,
        _weights: larql_models::WeightsView,
        query: &Array2<f32>,
        _kv: &mut KvHandle,
        _layer: usize,
        _abs_position: usize,
        _index: Option<&dyn crate::KvIndex>,
    ) -> Option<Array2<f32>> {
        Some(query.clone())
    }
    fn clip_kv(&self, _handle: &mut KvHandle, window_size: usize) {
        self.clipped_to
            .store(window_size, std::sync::atomic::Ordering::SeqCst);
    }
}

#[test]
fn default_attention_step_windowed_clips_after_successful_step() {
    let backend = AttnCapableStub {
        clipped_to: std::sync::atomic::AtomicUsize::new(0),
    };
    let weights = make_test_weights();
    let mut handle = stub_kv_handle(0, weights.hidden_size);
    let query = Array2::from_elem((1, weights.hidden_size), 1.0_f32);
    let out = backend
        .attention_step_windowed(
            larql_models::WeightsView::dense(&weights),
            &query,
            &mut handle,
            0,
            0,
            7,
            None,
        )
        .expect("successful attention_step must propagate through the windowed default");
    assert_eq!(out, query, "default returns attention_step's hidden state");
    assert_eq!(
        backend.clipped_to.load(std::sync::atomic::Ordering::SeqCst),
        7,
        "default must clip_kv to the requested window after the step"
    );
}

// ── Inner handle as_any / as_any_mut surface ─────────────────────

#[test]
fn stub_kv_inner_exposes_as_any_round_trip() {
    let mut inner = StubKvInner { len: 3, dim: 4 };
    // immutable side
    {
        let any: &dyn std::any::Any = inner.as_any();
        assert!(any.downcast_ref::<StubKvInner>().is_some());
    }
    // mutable side
    {
        let any_mut: &mut dyn std::any::Any = inner.as_any_mut();
        assert!(any_mut.downcast_mut::<StubKvInner>().is_some());
    }
}

#[test]
fn stub_residual_inner_exposes_as_any() {
    let inner = StubResidualInner { shape: (2, 4) };
    let any: &dyn std::any::Any = inner.as_any();
    assert!(any.downcast_ref::<StubResidualInner>().is_some());
    assert_eq!(inner.shape(), (2, 4));
    assert_eq!(inner.backend_name(), "stub");
}
