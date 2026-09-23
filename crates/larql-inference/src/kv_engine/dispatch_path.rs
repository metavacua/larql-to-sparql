//! Which dispatch shape an engine actually took for the current sequence.
//!
//! An engine's *name* does not determine how its forward runs. The same
//! `standard` engine takes a fused whole-model kernel when it is
//! unwindowed, and a generic per-layer loop when it is not — and on
//! `MetalBackend` the per-layer loop is delegated to the host, so a row
//! labelled `[metal (GPU)]` can be doing all of its attention and FFN on
//! the CPU. That difference is worth ~15 ms/token on qwen3-0.6b and is
//! invisible in every other field an engine reports.
//!
//! Engines record the shape when they choose it (at prefill, since decode
//! must follow the recorded mode) and report it here so a benchmark table
//! can say which one it measured instead of leaving the reader to infer
//! it from the timings.

/// The dispatch shape a prefill committed the sequence to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DispatchPath {
    /// One whole-model handle through the backend's fused pipeline
    /// (`coarse_prefill` / `coarse_decode_step`). The K/V lives in the
    /// backend, not in the engine, and the engine's own state policy is
    /// not engaged.
    Coarse,
    /// The generic per-layer loop: one handle per layer, attention and
    /// FFN dispatched a layer at a time. Every engine-specific K/V
    /// policy runs here.
    PerLayer,
}

impl DispatchPath {
    /// Short tag for diagnostics and bench rows.
    pub fn as_str(self) -> &'static str {
        match self {
            DispatchPath::Coarse => "coarse",
            DispatchPath::PerLayer => "per-layer",
        }
    }

    /// Render with where the compute lands. `host_delegated` is the
    /// backend's answer to [`crate::kv_dispatch::KvDispatch::per_layer_is_host_delegated`]
    /// — true when this backend implements the per-layer surface by
    /// forwarding to the CPU, which makes a "GPU" row a CPU measurement.
    pub fn describe(self, host_delegated: bool) -> &'static str {
        match (self, host_delegated) {
            (DispatchPath::Coarse, _) => "coarse",
            (DispatchPath::PerLayer, false) => "per-layer",
            (DispatchPath::PerLayer, true) => "per-layer→host",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn as_str_names_both_shapes() {
        assert_eq!(DispatchPath::Coarse.as_str(), "coarse");
        assert_eq!(DispatchPath::PerLayer.as_str(), "per-layer");
    }

    #[test]
    fn describe_flags_host_delegation_only_for_per_layer() {
        // The whole point: a per-layer path on a host-delegating backend
        // must be visibly distinct from one running native kernels.
        assert_eq!(DispatchPath::PerLayer.describe(true), "per-layer→host");
        assert_eq!(DispatchPath::PerLayer.describe(false), "per-layer");
    }

    #[test]
    fn describe_ignores_host_delegation_for_coarse() {
        // Coarse never touches the per-layer surface, so the backend's
        // delegation answer is irrelevant to it.
        assert_eq!(DispatchPath::Coarse.describe(true), "coarse");
        assert_eq!(DispatchPath::Coarse.describe(false), "coarse");
    }

    #[test]
    fn dispatch_path_is_comparable_and_copy() {
        let a = DispatchPath::Coarse;
        let b = a;
        assert_eq!(a, b);
        assert_ne!(DispatchPath::Coarse, DispatchPath::PerLayer);
        assert_eq!(format!("{:?}", DispatchPath::PerLayer), "PerLayer");
    }
}
