//! `WalkFfn` constructors and builder methods.
//!
//! Everything that *creates or configures* a `WalkFfn` lives here; the
//! struct definition, routing ladder, and trace accessors stay in
//! `mod.rs`. All constructors funnel through [`WalkFfn::from_config`].

use super::timings::PhaseTimingsHandle;
use super::WalkFfn;
use crate::model::ModelWeights;
use crate::vindex::l1_cache::FfnL1Cache;
use crate::vindex::walk_config::WalkFfnConfig;
use larql_compute::prelude::*;
use larql_vindex::GateIndex;

impl<'a> WalkFfn<'a> {
    pub fn from_config(
        weights: &'a ModelWeights,
        index: &'a dyn GateIndex,
        config: WalkFfnConfig,
    ) -> Self {
        let num_layers = weights.num_layers;
        Self {
            weights,
            index,
            config,
            backend: None,
            trace_residuals: std::cell::RefCell::new(Vec::new()),
            record_trace: false,
            runtime_trace: std::cell::RefCell::new(Vec::new()),
            last_path: std::cell::Cell::new(super::trace::PATH_UNROUTED),
            l1_cache: None,
            dispatch_trace: std::cell::RefCell::new(None),
            phase_timings: None,
            down_norms_cache: std::cell::RefCell::new(vec![None; num_layers]),
            up_norms_cache: std::cell::RefCell::new(vec![None; num_layers]),
            selector_fallbacks: std::cell::Cell::new(0),
            shortlist_declines: std::cell::Cell::new(0),
        }
    }

    // ── Legacy constructors (stable public API) ──

    pub fn new(weights: &'a ModelWeights, index: &'a dyn GateIndex, top_k: usize) -> Self {
        let config = if top_k == usize::MAX {
            WalkFfnConfig::dense(weights.num_layers)
        } else {
            WalkFfnConfig::sparse(weights.num_layers, top_k)
        };
        Self::from_config(weights, index, config)
    }

    pub fn new_unlimited(weights: &'a ModelWeights, index: &'a dyn GateIndex) -> Self {
        Self::from_config(weights, index, WalkFfnConfig::dense(weights.num_layers))
    }

    pub fn new_with_backend(
        weights: &'a ModelWeights,
        index: &'a dyn GateIndex,
        top_k: usize,
        backend: &'a dyn ComputeBackend,
    ) -> Self {
        Self::new(weights, index, top_k).with_backend(backend)
    }

    pub fn new_unlimited_with_backend(
        weights: &'a ModelWeights,
        index: &'a dyn GateIndex,
        backend: &'a dyn ComputeBackend,
    ) -> Self {
        Self::new_unlimited(weights, index).with_backend(backend)
    }

    pub fn new_with_trace(
        weights: &'a ModelWeights,
        index: &'a dyn GateIndex,
        top_k: usize,
    ) -> Self {
        Self::new(weights, index, top_k).with_trace()
    }

    pub fn new_unlimited_with_trace(weights: &'a ModelWeights, index: &'a dyn GateIndex) -> Self {
        Self::new_unlimited(weights, index).with_trace()
    }

    // ── Builder methods ──

    /// Attach a phase-timing sink. Records cache_fetch / scan / reduce
    /// timings inside `sparse:parallel_q4k_down` via atomic adds.
    pub fn with_phase_timings(mut self, handle: std::sync::Arc<PhaseTimingsHandle>) -> Self {
        self.phase_timings = Some(handle);
        self
    }

    pub fn with_backend(mut self, backend: &'a dyn ComputeBackend) -> Self {
        self.backend = Some(backend);
        self
    }

    pub fn with_trace(mut self) -> Self {
        self.record_trace = true;
        self
    }

    pub fn with_l1_cache(mut self, num_layers: usize) -> Self {
        self.l1_cache = Some(FfnL1Cache::new(num_layers));
        self
    }

    /// Enable the dispatch trace. Each walk path records its name to
    /// this buffer on exit. Use [`WalkFfn::take_dispatch_trace`] to
    /// retrieve.
    pub fn with_dispatch_trace(self) -> Self {
        *self.dispatch_trace.borrow_mut() = Some(Vec::new());
        self
    }
}
