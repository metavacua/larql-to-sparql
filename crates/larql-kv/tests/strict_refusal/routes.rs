//! Expert routes for the suite: one that refuses, one that executes.
//!
//! Both are `MoeExpertBackend`s, the seam `MoeFfn` dispatches experts through.
//! Keeping them here rather than in each test keeps the tests about policy.

use larql_execution::RefusalKind;
use larql_inference::ffn::{MoeBackendError, MoeExpertBackend};
use larql_inference::model::ModelWeights;
use larql_vindex::runtime::ExecutionError;
use ndarray::Array2;

/// Fixture values for the concrete errors below. Meaningless as numbers — they
/// exist so each refusal is a well-formed instance of its variant, and so a
/// failure message names something traceable rather than a bare literal.
const ABSENT_EXPERT: u32 = 7;
const OUT_OF_RANGE_EXPERT: u32 = 99;
const EXPERT_POPULATION: usize = 8;
const RESIDENT_EXPERTS: usize = 0;
const BANK_NAME: &str = "strict-refusal-test-bank";
const UNSERVED_FORMAT: &str = "mxfp4";
const UNSERVED_OPERAND: &str = "experts.up_proj";

/// A route that refuses every layer with a chosen classification.
///
/// One concrete error per [`RefusalKind`], picked so the mapping under test is
/// the real one in `ExecutionError::refusal()` rather than a fixture asserting
/// its own answer: a selected-but-absent expert is `Residency`, an
/// unimplemented decoder is `Unsupported`, and an expert outside the router's
/// addressable population is a `BindingDefect`.
pub struct RefusingRoute {
    pub kind: RefusalKind,
}

impl RefusingRoute {
    pub fn new(kind: RefusalKind) -> Self {
        Self { kind }
    }

    fn error(&self) -> ExecutionError {
        match self.kind {
            RefusalKind::Residency => ExecutionError::SelectedExpertNotResident {
                expert: ABSENT_EXPERT,
                bank: BANK_NAME.into(),
                resident: RESIDENT_EXPERTS,
                population: EXPERT_POPULATION,
            },
            RefusalKind::Unsupported => ExecutionError::UnsupportedFormat {
                format: UNSERVED_FORMAT.into(),
                operand: UNSERVED_OPERAND.into(),
            },
            RefusalKind::BindingDefect => ExecutionError::ExpertOutOfRange {
                expert: OUT_OF_RANGE_EXPERT,
                population: EXPERT_POPULATION,
            },
        }
    }
}

impl MoeExpertBackend for RefusingRoute {
    fn forward_moe_seq(
        &self,
        _weights: &ModelWeights,
        _layer: usize,
        _h: &Array2<f32>,
        _norm_offset: f32,
        _eps: f32,
    ) -> Result<Array2<f32>, MoeBackendError> {
        Err(MoeBackendError::Bound(self.error()))
    }

    fn name(&self) -> &'static str {
        "refusing-route"
    }
}

/// A route that executes.
///
/// A zero expert contribution is legitimate — the trait documents it as the
/// answer for a layer with no expert weights to route into. What matters is
/// that it returns `Ok`, so nothing is recorded and a strict policy stays
/// silent.
pub struct ExecutingRoute;

impl MoeExpertBackend for ExecutingRoute {
    fn forward_moe_seq(
        &self,
        _weights: &ModelWeights,
        _layer: usize,
        h: &Array2<f32>,
        _norm_offset: f32,
        _eps: f32,
    ) -> Result<Array2<f32>, MoeBackendError> {
        Ok(Array2::zeros(h.raw_dim()))
    }

    fn name(&self) -> &'static str {
        "executing-route"
    }
}

/// A route that serves one full pass over the layers, then refuses.
///
/// Models the case the other routes cannot: a shard that goes away *mid
/// stream*, after the engine has already committed something irreversible.
/// `unlimited-context` archives a window and saves its boundary checkpoint
/// when the window fills, so "refuse on the first token" and "refuse on the
/// second" are different questions — only the second can find the engine
/// holding a stream it cannot complete.
///
/// A pass is counted by visits to layer zero, because that is the only chunk
/// boundary visible from inside an `FfnBackend`.
pub struct RefuseAfterFirstPass {
    passes: std::cell::Cell<usize>,
    inner: RefusingRoute,
}

impl RefuseAfterFirstPass {
    pub fn new(kind: RefusalKind) -> Self {
        Self {
            passes: std::cell::Cell::new(0),
            inner: RefusingRoute::new(kind),
        }
    }
}

impl MoeExpertBackend for RefuseAfterFirstPass {
    fn forward_moe_seq(
        &self,
        weights: &ModelWeights,
        layer: usize,
        h: &Array2<f32>,
        norm_offset: f32,
        eps: f32,
    ) -> Result<Array2<f32>, MoeBackendError> {
        if layer == 0 {
            self.passes.set(self.passes.get() + 1);
        }
        if self.passes.get() <= 1 {
            return ExecutingRoute.forward_moe_seq(weights, layer, h, norm_offset, eps);
        }
        self.inner
            .forward_moe_seq(weights, layer, h, norm_offset, eps)
    }

    fn name(&self) -> &'static str {
        "refuse-after-first-pass"
    }
}
