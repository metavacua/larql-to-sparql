//! Observation-mode plumbing for the walk paths — the
//! `forward`/`forward_observed` split (2026-07-30 review, item 15).
//!
//! Every walk path takes an [`Observe`] mode. In `Skip` mode (the hot
//! `forward` entry) no path allocates or fills any activation buffer —
//! the sparse `seq_len × intermediate` zero-fill this module's
//! introduction removed simply never exists. In `Record` mode
//! (`forward_observed`) sparse paths push exactly the `(feature,
//! activation)` pairs they compute into a
//! [`SparseActivations`](larql_compute::ffn::SparseActivations) record,
//! and dense paths hand over the activation matrix they had to compute
//! anyway.

use larql_compute::ffn::FfnActivations;
use ndarray::Array2;

/// Whether a walk invocation records activations.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Observe {
    /// Hot path: never touch an activation buffer.
    Skip,
    /// Observed path: emit an honest [`FfnActivations`].
    Record,
}

impl Observe {
    pub(super) fn recording(self) -> bool {
        self == Observe::Record
    }

    /// Wrap a dense path's intrinsic activation matrix: kept when
    /// recording, dropped otherwise. Dense paths need this matrix to
    /// compute their output, so neither mode allocates anything extra.
    pub(super) fn dense(self, activation: Array2<f32>) -> Option<FfnActivations> {
        self.recording().then(|| FfnActivations::Dense(activation))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recording_maps_modes() {
        assert!(Observe::Record.recording());
        assert!(!Observe::Skip.recording());
    }

    #[test]
    fn dense_keeps_activation_only_when_recording() {
        let act = Array2::<f32>::from_elem((1, 2), 3.0);
        assert_eq!(
            Observe::Record.dense(act.clone()),
            Some(FfnActivations::Dense(act))
        );
        assert_eq!(Observe::Skip.dense(Array2::zeros((1, 2))), None);
    }
}
