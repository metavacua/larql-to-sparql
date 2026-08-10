//! Opt-in FFN activation observation — the `forward`/`forward_observed`
//! split (2026-07-30 vindex/WalkFFN review, item 15) plus the runtime
//! trace fields it carries (item 17).
//!
//! Activations used to be a mandatory second return value on every
//! [`super::FfnBackend`] forward, which most paths didn't honestly
//! produce: sparse walk paths zero-filled a dense `seq_len ×
//! intermediate` buffer they mostly didn't write, cache hits fabricated
//! zeros, and ordinary generation paid the allocation just to discard
//! it. [`FfnActivations`] makes observation opt-in and honest:
//!
//! - [`FfnActivations::Dense`] — every feature was computed (dense
//!   paths, where the activation matrix is an intrinsic intermediate).
//! - [`FfnActivations::Sparse`] — only the listed features were
//!   computed; every unlisted feature contributed exactly zero to the
//!   executed output, so densifying is faithful to the execution.
//! - [`FfnActivations::Absent`] — the executed path observed nothing
//!   (cache hit, remote backend, MoE). Never zeros: consumers can tell
//!   "not observed" from "observed as zero".
//!
//! Runtime-trace extension (item 17): sparse recorders that know their
//! projection scores at the accumulate site record them via
//! [`SparseActivations::record_scored`], and per-position kernel labels
//! via [`SparseActivations::set_kernel`] — so a trace built from the
//! observation describes the EXECUTED computation, not a re-run.
//! Recorders that don't score use plain [`SparseActivations::record`]
//! and the score fields stay honestly `None`.

use ndarray::Array2;

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// Reason for the trait-default [`super::FfnBackend::forward_observed`]:
/// the backend ran, but it does not implement activation observation.
pub const REASON_UNOBSERVED_BACKEND: &str =
    "backend does not observe activations (forward_observed not overridden)";

/// Honest record of the pre-down activations an executed FFN path
/// actually computed. See the module doc for variant semantics.
#[derive(Debug, Clone, PartialEq)]
pub enum FfnActivations {
    /// Every feature computed: `[seq_len, num_features]`.
    Dense(Array2<f32>),
    /// Only the recorded `(feature, activation)` pairs were computed.
    Sparse(SparseActivations),
    /// The executed path produced no observation. `reason` names the
    /// path that declined (cache hit, remote, unobserved backend).
    Absent { reason: &'static str },
}

impl FfnActivations {
    /// The trait-default observation for backends that don't observe.
    pub fn unobserved() -> Self {
        FfnActivations::Absent {
            reason: REASON_UNOBSERVED_BACKEND,
        }
    }

    /// True when the executed path produced no observation.
    pub fn is_absent(&self) -> bool {
        matches!(self, FfnActivations::Absent { .. })
    }

    /// The decline reason, when absent.
    pub fn absent_reason(&self) -> Option<&'static str> {
        match self {
            FfnActivations::Absent { reason } => Some(reason),
            _ => None,
        }
    }

    /// Materialise as a dense `[seq_len, num_features]` matrix.
    ///
    /// `Sparse` densification is faithful to the execution: features the
    /// path never selected contributed exactly zero to its output.
    /// `Absent` returns `None` — there is nothing honest to materialise.
    pub fn into_dense(self) -> Option<Array2<f32>> {
        match self {
            FfnActivations::Dense(a) => Some(a),
            FfnActivations::Sparse(s) => Some(s.to_dense()),
            FfnActivations::Absent { .. } => None,
        }
    }
}

/// One computed feature in a sparse observation. Score fields are
/// `Some` only when the recorder knew them at the accumulate site
/// (walk kernels record via [`SparseActivations::record_scored`]);
/// `None` means "the executed path did not compute this score" —
/// never a fabricated zero.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SparseEntry {
    /// Feature index within the layer.
    pub feature: usize,
    /// Scalar pre-down activation, e.g. `φ(g·x)·(u·x)` for gated FFNs.
    pub activation: f32,
    /// Gate projection score `g·x` (or the selection score on paths
    /// where selection IS the gate projection).
    pub gate_score: Option<f32>,
    /// Up projection score `u·x`. `None` on non-gated paths where the
    /// single projection is recorded as `gate_score`.
    pub up_score: Option<f32>,
}

/// Per-position computed-feature records from a sparse FFN path.
/// One entry per feature the path actually computed — no zero-fill.
/// Entry order is the executed visit order (= selection rank on
/// ranked routes, route order on precomputed routes).
#[derive(Debug, Clone, PartialEq)]
pub struct SparseActivations {
    num_features: usize,
    /// `positions[s]` = entries recorded for sequence position `s`.
    positions: Vec<Vec<SparseEntry>>,
    /// `kernels[s]` = label of the kernel that computed position `s`'s
    /// entries, when the recorder labels per-position sub-paths.
    kernels: Vec<Option<&'static str>>,
}

impl SparseActivations {
    /// Empty observation for `seq_len` positions over a layer with
    /// `num_features` features.
    pub fn new(seq_len: usize, num_features: usize) -> Self {
        Self {
            num_features,
            positions: vec![Vec::new(); seq_len],
            kernels: vec![None; seq_len],
        }
    }

    /// Record one computed activation without projection scores.
    /// Panics on an out-of-range position or feature — a mis-indexed
    /// observation is a bug, not data.
    pub fn record(&mut self, position: usize, feature: usize, activation: f32) {
        self.record_scored(position, feature, activation, None, None);
    }

    /// Record one computed activation with the projection scores the
    /// executed kernel used to compute it. Same panic contract as
    /// [`SparseActivations::record`].
    pub fn record_scored(
        &mut self,
        position: usize,
        feature: usize,
        activation: f32,
        gate_score: Option<f32>,
        up_score: Option<f32>,
    ) {
        assert!(
            feature < self.num_features,
            "SparseActivations::record: feature {feature} >= num_features {}",
            self.num_features
        );
        self.positions[position].push(SparseEntry {
            feature,
            activation,
            gate_score,
            up_score,
        });
    }

    /// Label the kernel that computed one position's entries. Panics on
    /// an out-of-range position (same contract as `record`).
    pub fn set_kernel(&mut self, position: usize, kernel: &'static str) {
        self.kernels[position] = Some(kernel);
    }

    /// The kernel label for one position, when the recorder set one.
    pub fn kernel(&self, position: usize) -> Option<&'static str> {
        self.kernels[position]
    }

    pub fn seq_len(&self) -> usize {
        self.positions.len()
    }

    pub fn num_features(&self) -> usize {
        self.num_features
    }

    /// Entries recorded for one position, in executed visit order.
    pub fn position(&self, position: usize) -> &[SparseEntry] {
        &self.positions[position]
    }

    /// Densify to `[seq_len, num_features]`. Repeated records of the
    /// same feature accumulate — matching how repeated contributions
    /// would have summed into the executed output.
    pub fn to_dense(&self) -> Array2<f32> {
        let mut dense = Array2::<f32>::zeros((self.positions.len(), self.num_features));
        for (s, entries) in self.positions.iter().enumerate() {
            for e in entries {
                dense[[s, e.feature]] += e.activation;
            }
        }
        dense
    }
}
