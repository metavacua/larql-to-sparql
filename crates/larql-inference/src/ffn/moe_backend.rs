//! Who computes a hybrid-MoE layer's expert contribution.
//!
//! The block loop — attention, KV, the dense slab, the norms, PLE, the layer
//! scalar — is shared by every route. What varies is one step:
//!
//! ```text
//! h_post_attn
//!   ├── dense FFN slab        shared
//!   └── expert contribution   this trait
//!         ├── in-process      cpu_moe_forward over MoeLayerWeights (the default)
//!         ├── remote          RemoteMoeBackend, experts fetched from shards
//!         └── bound           a VINDEX3 BoundMoeOperation
//! ```
//!
//! # Why a trait rather than a second parameter
//!
//! `moe_ffn_block_cpu` previously took `Option<&RemoteMoeBackend>` — a
//! concrete type, so the only way to add a third route was another optional
//! parameter and a rule that at most one may be `Some`. That rule is not
//! expressible in the type, which means it is a rule someone eventually
//! breaks, and the failure mode is a layer whose expert contribution comes
//! from a route nobody chose.
//!
//! One `Option<&dyn MoeExpertBackend>` makes "exactly one route, or the
//! in-process default" the only representable state.
//!
//! # The backend derives its own operands
//!
//! The caller passes `weights` and a layer, not a prepared router. The remote
//! backend needs the router only, because it fetches experts from shards; the
//! bound backend needs the whole layer. Preparing the union at the call site
//! would make the block loop know what each route reads, which is exactly the
//! coupling this trait removes.

use larql_models::ModelWeights;
use ndarray::Array2;

use super::moe_remote::RemoteMoeError;
#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// Why a backend could not produce an expert contribution.
///
/// Two variants rather than a string, so a refusal keeps the structure the
/// route gave it. In particular a bound route's [`ExecutionError`] carries
/// `refusal()`, which distinguishes an operand that lives elsewhere from a
/// binding that is wrong — a distinction that a formatted message destroys.
///
/// No `PartialEq`: [`RemoteMoeError`] does not claim it, and neither should a
/// type that wraps it.
///
/// [`ExecutionError`]: larql_vindex::runtime::ExecutionError
#[derive(Debug, thiserror::Error)]
pub enum MoeBackendError {
    #[error("remote expert dispatch failed: {0}")]
    Remote(#[from] RemoteMoeError),
    #[error("bound expert execution failed: {0}")]
    Bound(#[from] larql_vindex::runtime::ExecutionError),
    /// A routed operand could not be sourced from its VINDEX3 container.
    ///
    /// Distinct from [`Self::Bound`]: that is the executor rejecting operands
    /// it was given, this is not having them. Collapsing the two would report
    /// a missing bank as a kernel failure and send the reader to the wrong
    /// half of the system. Reaching this at generation time is itself a bug —
    /// composition is validated when the container is opened — so the message
    /// names the layer and expert rather than assuming a retry.
    #[error("routed container operand unavailable: {0}")]
    Container(String),
}

/// What an operation does when a MoE route refuses.
///
/// This is a property of the **operation**, not of the backend. The same
/// backend is legitimately used both to generate (where a failed layer must
/// abort) and to probe (where a refusal is the measurement). A backend cannot
/// distinguish those callers, so asking it would bake operation policy into an
/// operand provider — and would imply a false taxonomy in which some backends'
/// failures are tolerable. They are not: a remote shard failing mid-generation
/// and contributing zero is exactly as invalid as a missing container region.
///
/// The default for anything that produces a continuation, a score, a parity
/// number or a benchmark is [`Self::Fatal`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoeFailurePolicy {
    /// Abort the operation. No token, score or measurement is produced from a
    /// forward pass in which any expert contribution was not computed.
    Fatal,
    /// Record the refusal and continue with an incomplete result.
    ///
    /// Analysis-only. A caller selecting this is declaring that it *intends*
    /// to inspect partial execution and will report the incompleteness; it may
    /// not present the outcome as an ordinary model continuation.
    RecordRefusal,
}

/// A route plus the policy the calling operation applies to its refusals.
///
/// Paired rather than passed separately so a caller cannot supply a backend
/// and forget to state what a failure means — the omission that let a failed
/// layer contribute silent zeros for as long as this branch has existed.
#[derive(Clone, Copy)]
pub struct MoeRoute<'a> {
    pub backend: &'a dyn MoeExpertBackend,
    pub policy: MoeFailurePolicy,
}

impl<'a> MoeRoute<'a> {
    /// A route whose failures abort the operation. The correct choice for
    /// generation, scoring, parity and benchmarking.
    pub fn fatal(backend: &'a dyn MoeExpertBackend) -> Self {
        Self {
            backend,
            policy: MoeFailurePolicy::Fatal,
        }
    }

    /// A route whose failures are recorded and tolerated. Analysis only.
    pub fn recording(backend: &'a dyn MoeExpertBackend) -> Self {
        Self {
            backend,
            policy: MoeFailurePolicy::RecordRefusal,
        }
    }
}

/// A route that computes one hybrid-MoE layer's expert contribution.
pub trait MoeExpertBackend {
    /// Expert contribution for every position of `h`, shaped `[seq_len, hidden]`.
    ///
    /// Returns zeros — not an error — when the layer has no expert weights to
    /// route into. That is the in-process path's behaviour and a backend that
    /// diverged from it would change the model rather than the route.
    fn forward_moe_seq(
        &self,
        weights: &ModelWeights,
        layer: usize,
        h: &Array2<f32>,
        norm_offset: f32,
        eps: f32,
    ) -> Result<Array2<f32>, MoeBackendError>;

    /// Which route this is, for diagnostics. Never branched on.
    fn name(&self) -> &'static str;
}

/// The in-process route, made explicit as a backend.
///
/// Byte-identical to the block loop's default branch: the same
/// `cpu_moe_forward` call with the same arguments, position by position. It
/// exists for two reasons.
///
/// First, so a comparison can put both routes behind one interface and record
/// them the same way — otherwise the incumbent has no seam to observe and the
/// harness ends up reconstructing boundaries it cannot see.
///
/// Second, so the seam itself is falsifiable. Running with this backend must
/// equal running with no backend at all; if it does not, the trait changed the
/// model rather than merely relocating the call, and every comparison built on
/// it is measuring the wrong thing.
#[derive(Debug, Clone, Copy, Default)]
pub struct InProcessMoeBackend;

impl MoeExpertBackend for InProcessMoeBackend {
    fn forward_moe_seq(
        &self,
        weights: &ModelWeights,
        layer: usize,
        h: &Array2<f32>,
        norm_offset: f32,
        eps: f32,
    ) -> Result<Array2<f32>, MoeBackendError> {
        let seq_len = h.nrows();
        let hidden = h.ncols();
        let arch = &*weights.arch;
        let mut out = Array2::<f32>::zeros((seq_len, hidden));
        let Some(moe) = larql_compute::pipeline_layer::build_moe_weights(weights, arch, layer)
        else {
            return Ok(out);
        };
        // The same layer tag the default branch sets, so the within-expert
        // probe behaves identically whichever way the call arrives.
        larql_compute::cpu::ops::moe::set_current_layer(layer);
        for pos in 0..seq_len {
            let row: Vec<f32> = h.row(pos).to_vec();
            let moe_out =
                larql_compute::cpu::ops::moe::cpu_moe_forward(&row, &moe, norm_offset, eps);
            for (dst, src) in out.row_mut(pos).iter_mut().zip(moe_out.iter()) {
                *dst = *src;
            }
        }
        Ok(out)
    }

    fn name(&self) -> &'static str {
        "in-process"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{make_test_gemma4_moe_weights, GEMMA4_MOE_HIDDEN};
    use larql_compute::cpu::ops::moe::cpu_moe_forward;

    const EPS: f32 = 1e-6;
    const NORM_OFFSET: f32 = 0.0;
    const MOE_LAYER: usize = 0;
    /// Past the fixture's two layers, so `build_moe_weights` finds nothing.
    const ABSENT_LAYER: usize = 99;

    fn residual(rows: usize) -> Array2<f32> {
        Array2::from_shape_fn((rows, GEMMA4_MOE_HIDDEN), |(r, c)| {
            ((r * 7 + c) % 11) as f32 * 0.1 - 0.5
        })
    }

    #[test]
    fn the_in_process_backend_is_the_default_branch_relocated() {
        // The property the seam rests on. If this diverged, routing the
        // incumbent through the trait would be measuring the trait rather than
        // relocating a call — and every comparison built on it would be wrong.
        let weights = make_test_gemma4_moe_weights();
        let h = residual(3);
        let via_trait = InProcessMoeBackend
            .forward_moe_seq(&weights, MOE_LAYER, &h, NORM_OFFSET, EPS)
            .expect("the fixture executes");

        let arch = &*weights.arch;
        let moe = larql_compute::pipeline_layer::build_moe_weights(&weights, arch, MOE_LAYER)
            .expect("the fixture has an MoE layer 0");
        for pos in 0..h.nrows() {
            let row: Vec<f32> = h.row(pos).to_vec();
            let expected = cpu_moe_forward(&row, &moe, NORM_OFFSET, EPS);
            assert_eq!(
                via_trait.row(pos).to_vec(),
                expected,
                "position {pos} diverged"
            );
        }
    }

    #[test]
    fn the_in_process_backend_produces_a_non_zero_contribution() {
        // Guards the guard: an all-zero agreement would be two failures
        // agreeing rather than a parity result.
        let weights = make_test_gemma4_moe_weights();
        let out = InProcessMoeBackend
            .forward_moe_seq(&weights, MOE_LAYER, &residual(1), NORM_OFFSET, EPS)
            .expect("executes");
        assert!(out.iter().any(|v| v.abs() > f32::EPSILON));
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn a_layer_with_no_expert_weights_contributes_zeros_rather_than_erroring() {
        // The block loop adds this contribution unconditionally, so a backend
        // that errored here would change the model rather than the route.
        let weights = make_test_gemma4_moe_weights();
        let h = residual(2);
        let out = InProcessMoeBackend
            .forward_moe_seq(&weights, ABSENT_LAYER, &h, NORM_OFFSET, EPS)
            .expect("an absent MoE layer is not an error");
        assert_eq!(out.shape(), h.shape());
        assert!(out.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn every_route_names_itself_distinctly() {
        // The name reaches an operator-facing dispatch-error line, so two
        // routes sharing one would make the message unactionable.
        let names = [
            InProcessMoeBackend.name(),
            crate::ffn::BoundMoeBackend::production().name(),
        ];
        assert_ne!(names[0], names[1]);
        assert!(!names[0].is_empty());
    }

    #[test]
    fn a_backend_error_names_the_route_that_refused() {
        // The refusal keeps the structure the route gave it — a bound route's
        // `ExecutionError` still carries `refusal()` after wrapping.
        let inner = larql_vindex::runtime::ExecutionError::SelectedExpertNotResident {
            expert: 90,
            bank: "layer 5 bank 0".into(),
            resident: 8,
            population: 128,
        };
        let wrapped = MoeBackendError::from(inner.clone());
        let text = wrapped.to_string();
        assert!(text.contains("bound"), "{text}");
        assert!(text.contains("not resident"), "{text}");
        let MoeBackendError::Bound(recovered) = wrapped else {
            panic!("wrapping lost the variant");
        };
        assert_eq!(
            recovered.refusal(),
            larql_vindex::runtime::RefusalKind::Residency
        );
    }
}
