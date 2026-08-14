//! The wrapper's `KvEngine` surface.
//!
//! `SemanticPromotionEngine` implements every entry point on `KvEngine`,
//! but the promotion suites only ever drive the two the fixtures use.
//! The rest are the ones that matter for a wrapper: each must reach the
//! base engine, and each *prefill* must additionally restart the session
//! epoch and re-read the architecture's layer classification while each
//! *decode* advances the token position by one.
//!
//! Getting that wrong is not a crash. A prefill variant that skipped
//! `on_prefill` would leave the wrapper describing the previous prompt —
//! stale authority references, a token position that never moved, and a
//! `layer_attention` read off whichever model was prefilled last. All of
//! that reads as a working engine right up until a promotion consults it.
//!
//! So each variant is asserted on the *observable consequence* of the
//! bookkeeping, not merely on the call being forwarded.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use larql_inference::ffn::{FfnBackend, NullFfn};
use larql_inference::kv_engine::{EngineError, EngineInfo};
use larql_inference::model::ModelWeights;
use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
use ndarray::Array2;

use crate::engines::semantic_promotion::config::SemanticPromotionConfig;
use crate::engines::semantic_promotion::engine::SemanticPromotionEngine;
use crate::KvEngine;

const HIDDEN: usize = 4;

/// A prefill invocation that returns its hidden state, so the variant
/// loop can assert on the result as well as the bookkeeping.
type Call = Box<dyn Fn(&mut SemanticPromotionEngine) -> Result<Array2<f32>, EngineError>>;
/// A prefill invocation whose borrows are built inside the closure.
type Drive = Box<dyn Fn(&mut SemanticPromotionEngine)>;

/// Records which entry points were reached, so delegation can be
/// asserted without the base engine doing any real work.
#[derive(Default)]
struct Calls {
    prefill: AtomicUsize,
    decode: AtomicUsize,
    names: std::sync::Mutex<Vec<&'static str>>,
}

impl Calls {
    fn note(&self, what: &'static str, is_prefill: bool) {
        if is_prefill {
            self.prefill.fetch_add(1, Ordering::SeqCst);
        } else {
            self.decode.fetch_add(1, Ordering::SeqCst);
        }
        self.names.lock().unwrap().push(what);
    }
    fn last(&self) -> Option<&'static str> {
        self.names.lock().unwrap().last().copied()
    }
}

/// A base engine that reaches none of the real machinery. Returns a
/// well-shaped hidden state so the wrapper's success path runs.
struct RecordingEngine {
    calls: Arc<Calls>,
    rows: usize,
    fail: bool,
}

impl RecordingEngine {
    fn out(&self) -> Result<Array2<f32>, EngineError> {
        if self.fail {
            return Err(EngineError::BackendFailure {
                details: "test stub: fail set".into(),
            });
        }
        Ok(Array2::zeros((self.rows, HIDDEN)))
    }
}

impl KvEngine for RecordingEngine {
    fn name(&self) -> &str {
        "recording"
    }
    fn info(&self) -> EngineInfo {
        EngineInfo {
            name: "recording".into(),
            description: "test fixture".into(),
            backend: "cpu".into(),
            config: "base-config".into(),
        }
    }
    fn prefill(
        &mut self,
        _w: &ModelWeights,
        _f: &dyn FfnBackend,
        ids: &[u32],
    ) -> Result<Array2<f32>, EngineError> {
        self.rows = ids.len();
        self.calls.note("prefill", true);
        self.out()
    }
    fn decode_step(
        &mut self,
        _w: &ModelWeights,
        _f: &dyn FfnBackend,
        _t: u32,
    ) -> Result<Array2<f32>, EngineError> {
        self.calls.note("decode_step", false);
        self.out()
    }
    fn supports_multimodal(&self) -> bool {
        true
    }
    fn prefill_from_hidden(
        &mut self,
        _w: &ModelWeights,
        _f: &dyn FfnBackend,
        h: &Array2<f32>,
    ) -> Result<Array2<f32>, EngineError> {
        self.rows = h.nrows();
        self.calls.note("prefill_from_hidden", true);
        self.out()
    }
    fn memory_bytes(&self) -> usize {
        1234
    }
    fn window_tokens(&self) -> usize {
        77
    }
    fn cold_bytes(&self) -> usize {
        99
    }
    fn prefill_quant(
        &mut self,
        _w: &ModelWeights,
        _f: &dyn FfnBackend,
        _i: &larql_vindex::VectorIndex,
        ids: &[u32],
        _b: &dyn larql_compute::ComputeBackend,
    ) -> Result<Array2<f32>, EngineError> {
        self.rows = ids.len();
        self.calls.note("prefill_quant", true);
        self.out()
    }
    fn decode_step_quant(
        &mut self,
        _w: &ModelWeights,
        _f: &dyn FfnBackend,
        _i: &larql_vindex::VectorIndex,
        _t: u32,
        _b: &dyn larql_compute::ComputeBackend,
    ) -> Result<Array2<f32>, EngineError> {
        self.calls.note("decode_step_quant", false);
        self.out()
    }
    fn prefill_resident(
        &mut self,
        _w: &ModelWeights,
        _f: &dyn FfnBackend,
        _i: &larql_vindex::VectorIndex,
        ids: &[u32],
    ) -> Result<Array2<f32>, EngineError> {
        self.rows = ids.len();
        self.calls.note("prefill_resident", true);
        self.out()
    }
    fn decode_step_resident(
        &mut self,
        _w: &ModelWeights,
        _f: &dyn FfnBackend,
        _i: &larql_vindex::VectorIndex,
        _t: u32,
    ) -> Result<Array2<f32>, EngineError> {
        self.calls.note("decode_step_resident", false);
        self.out()
    }
    fn prefill_via_executor(
        &mut self,
        _w: &ModelWeights,
        _e: &dyn larql_inference::layer_executor::LayerExecutor,
        _f: &dyn FfnBackend,
        ids: &[u32],
    ) -> Result<Array2<f32>, EngineError> {
        self.rows = ids.len();
        self.calls.note("prefill_via_executor", true);
        self.out()
    }
    fn decode_step_via_executor(
        &mut self,
        _w: &ModelWeights,
        _e: &dyn larql_inference::layer_executor::LayerExecutor,
        _f: &dyn FfnBackend,
        _t: u32,
    ) -> Result<Array2<f32>, EngineError> {
        self.calls.note("decode_step_via_executor", false);
        self.out()
    }
    fn prefill_quant_via_executor(
        &mut self,
        _w: &ModelWeights,
        _e: &dyn larql_inference::layer_executor::LayerExecutor,
        _f: &dyn FfnBackend,
        _i: &larql_vindex::VectorIndex,
        ids: &[u32],
    ) -> Result<Array2<f32>, EngineError> {
        self.rows = ids.len();
        self.calls.note("prefill_quant_via_executor", true);
        self.out()
    }
    fn decode_step_quant_via_executor(
        &mut self,
        _w: &ModelWeights,
        _e: &dyn larql_inference::layer_executor::LayerExecutor,
        _f: &dyn FfnBackend,
        _i: &larql_vindex::VectorIndex,
        _t: u32,
    ) -> Result<Array2<f32>, EngineError> {
        self.calls.note("decode_step_quant_via_executor", false);
        self.out()
    }
}

/// A `LayerExecutor` that is only ever passed through — the recording
/// base engine never calls back into it.
struct PassthroughExecutor(Box<dyn larql_compute::ComputeBackend>);

impl larql_inference::layer_executor::LayerExecutor for PassthroughExecutor {
    fn backend(&self) -> &dyn larql_compute::ComputeBackend {
        self.0.as_ref()
    }
    fn dispatch_kind(&self) -> larql_inference::layer_executor::ExecutorDispatchKind {
        larql_inference::layer_executor::ExecutorDispatchKind::Fused
    }
    fn name(&self) -> &str {
        "passthrough"
    }
}

fn wrap(calls: Arc<Calls>, fail: bool) -> SemanticPromotionEngine {
    SemanticPromotionEngine::new(
        Box::new(RecordingEngine {
            calls,
            rows: 1,
            fail,
        }),
        SemanticPromotionConfig::default(),
    )
    .expect("observe mode is admissible with ffn-only capabilities")
}

fn engine() -> (SemanticPromotionEngine, Arc<Calls>) {
    let calls = Arc::new(Calls::default());
    (wrap(Arc::clone(&calls), false), calls)
}

// ── identity and reporting ───────────────────────────────────────────────

/// The wrapper names itself in terms of what it wraps, and its `info`
/// carries the base engine's own config alongside its own.
#[test]
fn name_and_info_are_composed_from_the_base_engine() {
    let (e, _) = engine();
    assert_eq!(e.name(), "semantic-promotion(recording)");

    let info = e.info();
    assert_eq!(info.name, "semantic-promotion(recording)");
    assert_eq!(info.backend, "cpu");
    assert!(
        info.description.contains("recording"),
        "info should name the wrapped engine: {}",
        info.description
    );
    assert!(
        info.config.contains("base[base-config]"),
        "info should carry the base config: {}",
        info.config
    );
}

/// Observe mode advertises bit-identical decode. That string is the
/// wrapper's own claim about itself, and the promotion suites rely on
/// it, so it must track the mode rather than being fixed text.
#[test]
fn info_describes_observe_mode_as_bit_identical() {
    let (e, _) = engine();
    assert!(
        e.info().description.contains("bit-identical"),
        "observe mode must say decode is unchanged: {}",
        e.info().description
    );
    assert!(e.session().is_passthrough());
}

/// Pure reporting is forwarded rather than answered by the wrapper —
/// the wrapper holds no cache of its own.
#[test]
fn size_and_capability_reporting_is_forwarded() {
    let (e, _) = engine();
    assert_eq!(e.memory_bytes(), 1234);
    assert_eq!(e.window_tokens(), 77);
    assert_eq!(e.cold_bytes(), 99);
    assert!(e.supports_multimodal());
    // `stage_summary` has no stub value; it must still reach the base.
    assert!(e.stage_summary().is_none());
}

#[test]
fn debug_reports_the_wrappers_own_policy_state() {
    let (e, _) = engine();
    let rendered = format!("{e:?}");
    assert!(rendered.contains("SemanticPromotionEngine"), "{rendered}");
    assert!(
        rendered.contains("semantic-promotion(recording)"),
        "{rendered}"
    );
    assert!(rendered.contains("session_epoch"), "{rendered}");
}

#[test]
fn accessors_expose_the_wrapped_engine_and_its_own_config() {
    let (mut e, _) = engine();
    assert_eq!(e.base().name(), "recording");
    assert_eq!(e.base_mut().name(), "recording");
    assert_eq!(e.config().mode, SemanticPromotionConfig::default().mode);
    assert!(e.capabilities().caller_supplied_ffn_backend);
    assert_eq!(e.metrics().promotions_committed, 0);
    assert!(e.journal().is_none());
    assert!(e.layer_attention().is_none(), "nothing prefilled yet");
    // `session_mut` is the planner's handle onto the same session.
    assert_eq!(e.session_mut().token_position(), 0);
}

// ── prefill variants ─────────────────────────────────────────────────────

/// Every prefill entry point must restart the session epoch, mirror the
/// prompt length into the token position, and read the architecture's
/// layer classification. A variant that skipped this would leave the
/// wrapper describing the previous prompt.
#[test]
fn every_prefill_variant_restarts_the_session_and_reads_the_architecture() {
    let weights = make_test_q4k_weights();
    let index = make_test_q4k_vindex(&weights);
    let backend = larql_compute::cpu_backend();
    let executor = PassthroughExecutor(larql_compute::cpu_backend());
    let ids = [1u32, 2, 3];

    // (label, invocation)
    let variants: Vec<(&str, Call)> = vec![
        (
            "prefill",
            Box::new(|e: &mut SemanticPromotionEngine| {
                let w = make_test_q4k_weights();
                e.prefill(&w, &NullFfn, &[1, 2, 3])
            }),
        ),
        (
            "prefill_from_hidden",
            Box::new(|e: &mut SemanticPromotionEngine| {
                let w = make_test_q4k_weights();
                let h = Array2::zeros((3, w.hidden_size));
                e.prefill_from_hidden(&w, &NullFfn, &h)
            }),
        ),
    ];

    for (label, call) in variants {
        let (mut e, calls) = engine();
        let epoch_before = e.session_epoch();
        call(&mut e).unwrap_or_else(|err| panic!("{label} should succeed: {err:?}"));

        assert_eq!(calls.prefill.load(Ordering::SeqCst), 1, "{label} delegated");
        assert_eq!(calls.last(), Some(label), "{label} reached its own entry");
        assert_eq!(
            e.session_epoch(),
            epoch_before + 1,
            "{label} must start a new session epoch"
        );
        assert_eq!(
            e.session().token_position(),
            ids.len() as u64,
            "{label} must mirror the prompt length"
        );
        assert!(
            e.layer_attention().is_some(),
            "{label} must read the layer classification"
        );
    }

    // The index/backend/executor-bearing variants, driven directly so
    // their borrows stay simple.
    let cases: Vec<(&str, Drive)> = vec![
        (
            "prefill_quant",
            Box::new(|e: &mut SemanticPromotionEngine| {
                let w = make_test_q4k_weights();
                let i = make_test_q4k_vindex(&w);
                let b = larql_compute::cpu_backend();
                e.prefill_quant(&w, &NullFfn, &i, &[1, 2, 3], &*b)
                    .expect("prefill_quant");
            }),
        ),
        (
            "prefill_resident",
            Box::new(|e: &mut SemanticPromotionEngine| {
                let w = make_test_q4k_weights();
                let i = make_test_q4k_vindex(&w);
                e.prefill_resident(&w, &NullFfn, &i, &[1, 2, 3])
                    .expect("prefill_resident");
            }),
        ),
    ];
    for (label, call) in cases {
        let (mut e, calls) = engine();
        call(&mut e);
        assert_eq!(calls.last(), Some(label));
        assert_eq!(e.session_epoch(), 2, "{label} must start a new epoch");
        assert_eq!(e.session().token_position(), 3, "{label} mirrors length");
        assert!(e.layer_attention().is_some(), "{label} reads architecture");
    }

    // Executor-bearing prefills.
    for label in ["prefill_via_executor", "prefill_quant_via_executor"] {
        let (mut e, calls) = engine();
        let out = if label == "prefill_via_executor" {
            e.prefill_via_executor(&weights, &executor, &NullFfn, &ids)
        } else {
            e.prefill_quant_via_executor(&weights, &executor, &NullFfn, &index, &ids)
        };
        out.unwrap_or_else(|err| panic!("{label}: {err:?}"));
        assert_eq!(calls.last(), Some(label));
        assert_eq!(e.session_epoch(), 2, "{label} must start a new epoch");
        assert_eq!(e.session().token_position(), 3, "{label} mirrors length");
        assert!(e.layer_attention().is_some(), "{label} reads architecture");
    }
    let _ = backend;
}

// ── decode variants ──────────────────────────────────────────────────────

/// Every decode entry point advances the token position by exactly one.
/// The position is what scopes an answer boundary, so a variant that
/// forgot to advance would let a scope outlive the answer it was
/// qualified for.
#[test]
fn every_decode_variant_advances_the_token_position_by_one() {
    let weights = make_test_q4k_weights();
    let index = make_test_q4k_vindex(&weights);
    let backend = larql_compute::cpu_backend();
    let executor = PassthroughExecutor(larql_compute::cpu_backend());

    for label in [
        "decode_step",
        "decode_step_quant",
        "decode_step_resident",
        "decode_step_via_executor",
        "decode_step_quant_via_executor",
    ] {
        let (mut e, calls) = engine();
        e.prefill(&weights, &NullFfn, &[1, 2, 3]).expect("prefill");
        let before = e.session().token_position();

        let out = match label {
            "decode_step" => e.decode_step(&weights, &NullFfn, 9),
            "decode_step_quant" => e.decode_step_quant(&weights, &NullFfn, &index, 9, &*backend),
            "decode_step_resident" => e.decode_step_resident(&weights, &NullFfn, &index, 9),
            "decode_step_via_executor" => {
                e.decode_step_via_executor(&weights, &executor, &NullFfn, 9)
            }
            _ => e.decode_step_quant_via_executor(&weights, &executor, &NullFfn, &index, 9),
        };
        out.unwrap_or_else(|err| panic!("{label}: {err:?}"));

        assert_eq!(calls.last(), Some(label), "{label} reached its own entry");
        assert_eq!(calls.decode.load(Ordering::SeqCst), 1, "{label} delegated");
        assert_eq!(
            e.session().token_position(),
            before + 1,
            "{label} must advance the position by exactly one"
        );
    }
}

// ── failure ──────────────────────────────────────────────────────────────

/// A failed prefill must not restart the session. Bookkeeping runs only
/// on success, so a base engine that refused leaves the wrapper
/// describing the state that is actually still resident.
#[test]
fn a_failed_prefill_does_not_restart_the_session() {
    let calls = Arc::new(Calls::default());
    let mut e = wrap(Arc::clone(&calls), true);
    let weights = make_test_q4k_weights();

    let epoch_before = e.session_epoch();
    assert!(e.prefill(&weights, &NullFfn, &[1, 2, 3]).is_err());

    assert_eq!(e.session_epoch(), epoch_before, "epoch must not advance");
    assert_eq!(e.session().token_position(), 0, "position must not move");
    assert!(
        e.layer_attention().is_none(),
        "a refused prefill must not publish a layer classification"
    );
}

/// Likewise a failed decode leaves the position where it was — an
/// advanced position after a refused step would misplace every
/// subsequent boundary.
#[test]
fn a_failed_decode_does_not_advance_the_position() {
    let calls = Arc::new(Calls::default());
    let mut e = wrap(Arc::clone(&calls), true);
    let weights = make_test_q4k_weights();

    assert!(e.decode_step(&weights, &NullFfn, 9).is_err());
    assert_eq!(e.session().token_position(), 0);
}
