//! `RemoteMoeFfn` — an [`FfnBackend`] adapter that lets the KvEngine layer
//! drive CPU remote-MoE decode with a real KV cache.
//!
//! The engine owns attention (and its KV cache) and calls
//! [`FfnBackend::forward_moe_full_layer`] per MoE layer; this adapter
//! computes that layer's MoE FFN block via
//! [`moe_ffn_block_cpu`](crate::vindex::kquant_forward::moe_ffn_block_cpu)
//! — dense `h1` locally + experts `h2` dispatched to the remote shards
//! through [`RemoteMoeBackend`]. This is the engine-routed counterpart to
//! the standalone full-recompute `generate_kquant_cpu_remote` path that
//! closed #146; see the larql-kv "MoE-aware KV engines (C1)" roadmap item.

// FfnBackend's only uses are the two `impl FfnBackend for {RemoteMoeFfn,
// MoeFfn}` blocks below, both native-only. ModelWeights/Array2/WeightFfn
// are used only inside those same native-gated structs/impls (the
// #[cfg(test)] module below has its own local `use ndarray::Array2;`).
#[cfg(not(target_arch = "wasm32"))]
use crate::ffn::WeightFfn;
#[cfg(not(target_arch = "wasm32"))]
use larql_compute::ffn::FfnBackend;
#[cfg(not(target_arch = "wasm32"))]
use larql_models::ModelWeights;
#[cfg(not(target_arch = "wasm32"))]
use ndarray::Array2;

// `RemoteMoeBackend` (backend.rs) is tokio+rayon+tonic gRPC/HTTP transport,
// native-only -- see mod.rs. `RemoteMoeFfn` below wraps it concretely, so
// that struct/impl/const are gated the same way. The generalised `MoeFfn`
// (trait-object based via `MoeExpertBackend`) is *also* native-only despite
// being expert-route-agnostic: its `forward_moe_full_layer` impl calls
// `moe_ffn_block_cpu` (vindex, native-only) directly for the dense `h1`
// half -- see the import comment below.
#[cfg(not(target_arch = "wasm32"))]
use super::RemoteMoeBackend;
// `moe_ffn_block_cpu` lives in `vindex`, which is native-only (the whole
// module is `#[cfg(not(target_arch = "wasm32"))]`-gated in lib.rs) --
// `MoeFfn` below is the only user, in its `forward_moe_full_layer` impl, so
// that struct/impl (and `RefusalRecorder`, its sole helper) are gated the
// same way, alongside this import.
#[cfg(not(target_arch = "wasm32"))]
use crate::vindex::moe_ffn_block_cpu;

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// `FfnBackend` for CPU remote-MoE decode through a `KvEngine`.
///
/// The dense `h1` contribution runs through [`WeightFfn`] (f32 dense FFN over
/// `weights.tensors` — the caller pre-dequantizes the client's Q4K FFN), and
/// the expert `h2` contribution dispatches to the remote shards via
/// `forward_moe_seq`.
///
/// (A `WalkFfn`-based Q4K-direct `h1` was tried 2026-05-29 and reverted: its
/// dense mode runs the per-position sparse-walk machinery → ~8.5× slower than
/// f32 BLAS. The genuine Q4K-direct dense kernel is
/// `kquant_ffn_forward_layer_q8k`; see the bottleneck-diagnosis follow-up.)
///
/// PLE is **not** applied on this path (`moe_ffn_block_cpu` is called with
/// `ple_input = None`), so callers must route Per-Layer-Embedding
/// architectures (Gemma 4 E-series) through the full-recompute path
/// instead. Non-PLE MoE models (Gemma 4 26B-A4B, 31B-MoE) are unaffected.
#[cfg(not(target_arch = "wasm32"))]
pub struct RemoteMoeFfn<'a> {
    pub weights: &'a ModelWeights,
    pub remote: &'a RemoteMoeBackend,
}

#[cfg(not(target_arch = "wasm32"))]
impl<'a> FfnBackend for RemoteMoeFfn<'a> {
    fn forward(&self, layer: usize, x: &Array2<f32>) -> Array2<f32> {
        self.general().forward(layer, x)
    }

    fn forward_observed(
        &self,
        layer: usize,
        x: &Array2<f32>,
    ) -> (Array2<f32>, crate::ffn::FfnActivations) {
        self.general().forward_observed(layer, x)
    }

    fn name(&self) -> &str {
        REMOTE_MOE_FFN_NAME
    }

    fn forward_moe_full_layer(
        &self,
        layer: usize,
        h_post_attn: &Array2<f32>,
    ) -> Result<Option<Array2<f32>>, larql_execution::BoxRefusal> {
        self.general().forward_moe_full_layer(layer, h_post_attn)
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl<'a> RemoteMoeFfn<'a> {
    /// This adapter as the general one. Kept as a delegation rather than a type
    /// alias so the `{ weights, remote }` literal its callers use still
    /// type-checks — the name is load-bearing at one CLI call site and in two
    /// roadmaps.
    fn general(&self) -> MoeFfn<'a> {
        // Best-effort, because that is what this adapter has always been and
        // its callers depend on: a shard that cannot be reached degrades rather
        // than stops.
        MoeFfn::best_effort(self.weights, self.remote)
    }
}

/// The name `RemoteMoeFfn` reports. Unchanged from before the generalisation:
/// it appears in engine diagnostics and is matched on in at least one test.
#[cfg(not(target_arch = "wasm32"))]
const REMOTE_MOE_FFN_NAME: &str = "remote-moe";

/// `FfnBackend` for CPU MoE decode through a `KvEngine`, over **any** expert
/// route.
///
/// The engine owns attention and its KV cache and calls
/// [`FfnBackend::forward_moe_full_layer`] per MoE layer; this computes that
/// layer's block — dense `h1` locally, experts `h2` through whichever
/// [`MoeExpertBackend`] is installed.
///
/// The remote-specific adapter above predates this and now delegates to it. The
/// generalisation is what lets a VINDEX3 bound route reach the decode path at
/// all: the block loop's seam became a trait in the composition rung, but the
/// *engine's* adapter still named one concrete backend, so the bound route was
/// reachable only from the full-recompute path.
///
/// PLE is **not** applied here (`moe_ffn_block_cpu` is called with
/// `ple_input = None`), so Per-Layer-Embedding architectures must go through
/// the full-recompute path instead.
#[cfg(not(target_arch = "wasm32"))]
pub struct MoeFfn<'a> {
    pub weights: &'a ModelWeights,
    pub moe: &'a dyn crate::ffn::MoeExpertBackend,
    /// Whether a refusing route is a diagnosis or a fallback.
    policy: RefusalPolicy,
    /// The first refusal seen, with the layer that produced it.
    ///
    /// A cell rather than a return value because `FfnBackend` has no error
    /// channel — `forward_moe_full_layer` returns `Option<Array2<f32>>`, and
    /// widening it would reach every engine. The refusal is therefore caught on
    /// the way *in*, before `moe_ffn_block_cpu` handles it.
    refusal: core::cell::RefCell<Option<RecordedRefusal>>,
}

/// What a refusing expert route means to the caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RefusalPolicy {
    /// Log it, contribute zeros, return the dense half. The historical
    /// behaviour, and the right one for best-effort remote inference where a
    /// degraded continuation beats no continuation.
    #[default]
    BestEffort,
    /// Record it, so the caller can refuse the output.
    ///
    /// A selected-but-absent expert must not silently become an FFN-skipping
    /// approximation. Under this policy the block still returns its dense half —
    /// the trait cannot say otherwise — but [`MoeFfn::refusal`] is set, and a
    /// caller that accepts logits without checking it is accepting a
    /// numerically wrong continuation that looks plausible.
    Strict,
}

/// A refusal, with where it happened.
///
/// An error in its own right, so it can cross `FfnBackend`'s boundary as a
/// `BoxRefusal` while keeping both levels: the response category an engine
/// switches on, and the concrete message the route produced.
#[derive(Debug, Clone)]
pub struct RecordedRefusal {
    pub layer: usize,
    /// The refusal's own classification, preserved rather than flattened —
    /// `Residency` means fetch the operand, and is not a defect.
    pub kind: larql_execution::RefusalKind,
    pub message: String,
}

impl core::fmt::Display for RecordedRefusal {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "layer {}: {} ({})", self.layer, self.message, self.kind)
    }
}

impl core::error::Error for RecordedRefusal {}

impl larql_execution::ExecutionRefusal for RecordedRefusal {
    fn kind(&self) -> larql_execution::RefusalKind {
        self.kind
    }
}

/// Wraps the route so a refusal is recorded before the block loop swallows it.
///
/// Only used by [`MoeFfn::forward_moe_full_layer`], which is native-only
/// (calls `moe_ffn_block_cpu`) -- gated the same way.
#[cfg(not(target_arch = "wasm32"))]
struct RefusalRecorder<'a> {
    inner: &'a dyn crate::ffn::MoeExpertBackend,
    sink: &'a core::cell::RefCell<Option<RecordedRefusal>>,
}

#[cfg(not(target_arch = "wasm32"))]
impl crate::ffn::MoeExpertBackend for RefusalRecorder<'_> {
    fn forward_moe_seq(
        &self,
        weights: &ModelWeights,
        layer: usize,
        h: &Array2<f32>,
        norm_offset: f32,
        eps: f32,
    ) -> Result<Array2<f32>, crate::ffn::MoeBackendError> {
        let out = self
            .inner
            .forward_moe_seq(weights, layer, h, norm_offset, eps);
        if let Err(err) = &out {
            let kind = match err {
                crate::ffn::MoeBackendError::Bound(inner) => inner.refusal(),
                // A remote dispatch failure is the operand not being here.
                crate::ffn::MoeBackendError::Remote(_) => larql_execution::RefusalKind::Residency,
                // So is a routed region missing from its container — same
                // diagnosis, different distance: the bytes this layer needs
                // are not reachable, rather than reachable and rejected.
                crate::ffn::MoeBackendError::Container(_) => {
                    larql_execution::RefusalKind::Residency
                }
            };
            // First only: the earliest refusal is the diagnosis, and later ones
            // are usually the same cause repeating per layer.
            let mut sink = self.sink.borrow_mut();
            if sink.is_none() {
                *sink = Some(RecordedRefusal {
                    layer,
                    kind,
                    message: err.to_string(),
                });
            }
        }
        out
    }

    fn name(&self) -> &'static str {
        self.inner.name()
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl<'a> MoeFfn<'a> {
    /// Best-effort: a refusing route contributes zeros and the dense half is
    /// returned. Historical behaviour, unchanged.
    pub fn best_effort(
        weights: &'a ModelWeights,
        moe: &'a dyn crate::ffn::MoeExpertBackend,
    ) -> Self {
        Self {
            weights,
            moe,
            policy: RefusalPolicy::BestEffort,
            refusal: core::cell::RefCell::new(None),
        }
    }

    /// Strict: a refusal is recorded and the caller must check [`Self::refusal`]
    /// before accepting the output.
    ///
    /// Strictness is in the constructor rather than a field a caller can forget
    /// to set, and the check is a method rather than a log line someone has to
    /// read.
    pub fn strict(weights: &'a ModelWeights, moe: &'a dyn crate::ffn::MoeExpertBackend) -> Self {
        Self {
            weights,
            moe,
            policy: RefusalPolicy::Strict,
            refusal: core::cell::RefCell::new(None),
        }
    }

    pub fn policy(&self) -> RefusalPolicy {
        self.policy
    }

    /// The first refusal seen, if any. Always recorded; only *meaningful* as a
    /// gate under [`RefusalPolicy::Strict`], where the caller is contracted to
    /// consult it.
    pub fn refusal(&self) -> Option<RecordedRefusal> {
        self.refusal.borrow().clone()
    }

    /// Whether every layer so far executed its expert route.
    pub fn all_experts_executed(&self) -> bool {
        self.refusal.borrow().is_none()
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl FfnBackend for MoeFfn<'_> {
    fn forward(&self, layer: usize, x: &Array2<f32>) -> Array2<f32> {
        WeightFfn {
            weights: self.weights,
        }
        .forward(layer, x)
    }

    fn forward_observed(
        &self,
        layer: usize,
        x: &Array2<f32>,
    ) -> (Array2<f32>, crate::ffn::FfnActivations) {
        WeightFfn {
            weights: self.weights,
        }
        .forward_observed(layer, x)
    }

    fn name(&self) -> &str {
        self.moe.name()
    }

    fn forward_moe_full_layer(
        &self,
        layer: usize,
        h_post_attn: &Array2<f32>,
    ) -> Result<Option<Array2<f32>>, larql_execution::BoxRefusal> {
        let recorder = RefusalRecorder {
            inner: self.moe,
            sink: &self.refusal,
        };
        let out = moe_ffn_block_cpu(
            self.weights,
            h_post_attn,
            layer,
            &WeightFfn {
                weights: self.weights,
            },
            None,
            // Recording, because this adapter has its own `RefusalPolicy` one
            // level up: it needs the refusal *captured* so `Strict` can turn it
            // into an error and `BestEffort` can degrade having said so.
            // Making the block itself fatal here would bypass that decision.
            Some(crate::ffn::MoeRoute::recording(&recorder)),
        )
        // Unreachable: `recording` never propagates out of the block.
        .expect("RecordRefusal never returns Err");
        // `moe_ffn_block_cpu` has already logged the refusal and left the
        // expert contribution at zero, so `out` is the dense half wearing the
        // shape of an answer. Under `Strict` it must not escape — which is the
        // difference between this and the audited contract it replaces, where
        // the caller had to remember to ask.
        //
        // The dense half is computed and discarded on that path. It is an error
        // path, and paying for it buys the guarantee that no caller can consume
        // a partial layer.
        match self.refusal.borrow().as_ref() {
            Some(recorded) if self.policy == RefusalPolicy::Strict => {
                Err(Box::new(recorded.clone()))
            }
            // Best-effort keeps its historical behaviour: degrade, having said
            // so. Callers of the remote adapter depend on a shard outage
            // degrading rather than stopping.
            _ => Ok(Some(out)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::make_test_gemma4_moe_weights;
    use ndarray::Array2;

    /// `forward_moe_full_layer` runs the MoE FFN block: dense `h1` + experts
    /// via the (disconnected → zero) remote + combine. Asserts a full, finite
    /// layer output of the right shape.
    #[test]
    fn forward_moe_full_layer_returns_finite_combined_output() {
        let weights = make_test_gemma4_moe_weights();
        let remote = RemoteMoeBackend::new_disconnected();
        let ffn = RemoteMoeFfn {
            weights: &weights,
            remote: &remote,
        };
        let h_post_attn = Array2::<f32>::from_elem((2, weights.hidden_size), 0.1);
        let out = ffn
            .forward_moe_full_layer(0, &h_post_attn)
            .expect("executes")
            .expect("RemoteMoeFfn always returns Some");
        assert_eq!(out.shape(), &[2, weights.hidden_size]);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    /// `forward` / `forward_with_activation` run the dense FFN fallback;
    /// `name` is stable.
    #[test]
    fn dense_fallbacks_and_name() {
        let weights = make_test_gemma4_moe_weights();
        let remote = RemoteMoeBackend::new_disconnected();
        let ffn = RemoteMoeFfn {
            weights: &weights,
            remote: &remote,
        };
        assert_eq!(ffn.name(), "remote-moe");
        let x = Array2::<f32>::from_elem((2, weights.hidden_size), 0.1);
        let dense = ffn.forward(0, &x);
        assert_eq!(dense.shape()[0], 2);
        assert!(dense.iter().all(|v| v.is_finite()));
        let (out, obs) = ffn.forward_observed(0, &x);
        assert_eq!(out.shape()[0], 2);
        let act = obs.into_dense().expect("dense fallback observes densely");
        assert_eq!(act.shape()[0], 2);
    }

    /// The generalisation must not have changed the remote adapter. Same
    /// weights, same input, same output — the delegation relocates the call and
    /// nothing else, which is the same property the block-loop seam had to
    /// prove in the composition rung.
    #[test]
    fn the_remote_adapter_still_equals_the_general_one() {
        let weights = make_test_gemma4_moe_weights();
        let remote = RemoteMoeBackend::new_disconnected();
        let specific = RemoteMoeFfn {
            weights: &weights,
            remote: &remote,
        };
        let general = MoeFfn::best_effort(&weights, &remote);
        let h = Array2::<f32>::from_elem((2, weights.hidden_size), 0.1);
        assert_eq!(
            specific
                .forward_moe_full_layer(0, &h)
                .expect("executes")
                .expect("produces a layer"),
            general
                .forward_moe_full_layer(0, &h)
                .expect("executes")
                .expect("produces a layer")
        );
        assert_eq!(specific.forward(0, &h), general.forward(0, &h));
        // The name is the one thing that deliberately differs: the remote
        // adapter keeps its historical name for diagnostics, the general one
        // reports whichever route it carries.
        assert_eq!(specific.name(), REMOTE_MOE_FFN_NAME);
        assert_eq!(general.name(), crate::ffn::MoeExpertBackend::name(&remote));
    }

    /// The point of the generalisation: a VINDEX3 bound route can now drive the
    /// engine's FFN seam, which the remote-typed adapter made impossible.
    #[test]
    fn a_bound_route_can_drive_the_engine_adapter() {
        let weights = make_test_gemma4_moe_weights();
        // The fixture stores BF16 experts, so the reference pairing is the one
        // that executes; the production Q4_K pairing would refuse them.
        let bound = crate::ffn::BoundMoeBackend::reference();
        let ffn = MoeFfn::best_effort(&weights, &bound);
        let h = Array2::<f32>::from_elem((2, weights.hidden_size), 0.1);
        let out = ffn
            .forward_moe_full_layer(0, &h)
            .expect("executes")
            .expect("the adapter always returns Some");
        assert_eq!(out.shape(), &[2, weights.hidden_size]);
        assert!(out.iter().all(|v| v.is_finite()));
        assert!(out.iter().any(|v| v.abs() > f32::EPSILON));
    }

    /// A refused route must not silently contribute zeros through the adapter.
    /// `moe_ffn_block_cpu` logs and leaves `h2` zero on error, so this pins that
    /// the dense half still flows — the failure is visible as a *different*
    /// output rather than as a plausible one.
    #[test]
    fn a_refusing_route_still_returns_the_dense_half() {
        let weights = make_test_gemma4_moe_weights();
        // Q4_K kernels against a BF16 store: refused at bind.
        let refusing = crate::ffn::BoundMoeBackend::production();
        let executing = crate::ffn::BoundMoeBackend::reference();
        let h = Array2::<f32>::from_elem((2, weights.hidden_size), 0.1);
        let refused = MoeFfn::best_effort(&weights, &refusing)
            .forward_moe_full_layer(0, &h)
            .expect("executes")
            .expect("returns Some even when the route refused");
        let executed = MoeFfn::best_effort(&weights, &executing)
            .forward_moe_full_layer(0, &h)
            .expect("executes")
            .expect("produces a layer");
        assert_ne!(
            refused, executed,
            "a refused expert route must not produce the same output as one that ran"
        );
    }

    // ── Strictness ─────────────────────────────────────────────────────────

    #[test]
    fn a_strict_adapter_records_a_refusal_that_best_effort_would_swallow() {
        // The behaviour that must not become the MAP-3A contract. Best-effort
        // logs, contributes zeros and returns the dense half — a plausible but
        // numerically wrong continuation. Strict records it so the caller can
        // refuse the output.
        let weights = make_test_gemma4_moe_weights();
        // Q4_K kernels against the fixture's BF16 store: refused at bind.
        let refusing = crate::ffn::BoundMoeBackend::production();
        let h = Array2::<f32>::from_elem((2, weights.hidden_size), 0.1);

        let lenient = MoeFfn::best_effort(&weights, &refusing);
        let _ = lenient.forward_moe_full_layer(0, &h);

        let strict = MoeFfn::strict(&weights, &refusing);
        let _ = strict.forward_moe_full_layer(0, &h);

        assert_eq!(strict.policy(), RefusalPolicy::Strict);
        assert_eq!(lenient.policy(), RefusalPolicy::BestEffort);
        let recorded = strict.refusal().expect("a refusal must be recorded");
        assert_eq!(recorded.layer, 0);
        assert_eq!(
            recorded.kind,
            larql_vindex::runtime::RefusalKind::Unsupported
        );
        assert!(!strict.all_experts_executed());
    }

    #[test]
    fn an_executing_route_records_nothing() {
        // The counter-case. If a refusal were recorded for a route that ran,
        // the gate would reject every output and be quietly useless.
        let weights = make_test_gemma4_moe_weights();
        let executing = crate::ffn::BoundMoeBackend::reference();
        let strict = MoeFfn::strict(&weights, &executing);
        let h = Array2::<f32>::from_elem((2, weights.hidden_size), 0.1);
        let _ = strict.forward_moe_full_layer(0, &h);
        assert!(strict.refusal().is_none());
        assert!(strict.all_experts_executed());
    }

    #[test]
    fn the_refusal_keeps_its_classification_rather_than_a_flattened_message() {
        // `Residency` means fetch the operand and is not a defect; `Unsupported`
        // means write a kernel. A gate that only had a string could not tell a
        // sharded deployment from a broken one.
        let weights = make_test_gemma4_moe_weights();
        let remote = RemoteMoeBackend::new_disconnected();
        let strict = MoeFfn::strict(&weights, &remote);
        let h = Array2::<f32>::from_elem((2, weights.hidden_size), 0.1);
        let _ = strict.forward_moe_full_layer(0, &h);
        if let Some(recorded) = strict.refusal() {
            assert_eq!(
                recorded.kind,
                larql_execution::RefusalKind::Residency,
                "an unreachable shard is the operand not being here"
            );
            assert!(!recorded.message.is_empty());
        }
    }

    #[test]
    fn only_the_first_refusal_is_kept() {
        // Later layers usually repeat the same cause, so the earliest one is
        // the diagnosis. A sink that overwrote would report the last layer
        // rather than the first — the same first-versus-causal confusion the
        // divergence report exists to avoid.
        let weights = make_test_gemma4_moe_weights();
        let refusing = crate::ffn::BoundMoeBackend::production();
        let strict = MoeFfn::strict(&weights, &refusing);
        let h = Array2::<f32>::from_elem((2, weights.hidden_size), 0.1);
        let _ = strict.forward_moe_full_layer(0, &h);
        let _ = strict.forward_moe_full_layer(1, &h);
        assert_eq!(strict.refusal().expect("recorded").layer, 0);
    }

    #[test]
    fn the_remote_adapter_stays_best_effort() {
        // Its callers depend on degrading rather than stopping when a shard is
        // unreachable, so the generalisation must not have made it strict.
        let weights = make_test_gemma4_moe_weights();
        let remote = RemoteMoeBackend::new_disconnected();
        let specific = RemoteMoeFfn {
            weights: &weights,
            remote: &remote,
        };
        assert_eq!(specific.general().policy(), RefusalPolicy::BestEffort);
    }
}
