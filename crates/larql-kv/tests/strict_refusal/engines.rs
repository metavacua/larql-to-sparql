//! Every engine in [`EngineKind`], and what a strict refusal does to it.
//!
//! The gate in [`crate::gate`] sweeps `StandardEngine`'s entry points because
//! that is where the dispatch ring terminates. This module sweeps the other
//! axis — the engine — because five of them reach their FFN through
//! `larql_kv::engines::layer_ffn_or_moe` instead, and that copy used to log a
//! refusal and return the dense half.
//!
//! ## Two populations, and the difference is not a detail
//!
//! ```text
//! RoutesExperts   dispatches experts through the FFN hook → must refuse a
//!                 refusing route, and serve an executing one
//! NoExpertSeam    has no FfnBackend seam in its forward at all → must refuse
//!                 the *architecture*, before any route is consulted
//! ```
//!
//! The second population exists because "cannot dispatch experts" is not the
//! same failure as "an expert was missing". An engine whose forward has no
//! hook runs the dense half of every layer and returns an apparently valid
//! answer — a *different model* wearing the same answer shape, which nothing
//! downstream can detect. So those engines refuse the model up front, with
//! `RefusalKind::Unsupported`: the operands are fine, this executor cannot
//! serve them, pick another.
//!
//! `apollo` is the only member. Its forward (`forward_from_layer` /
//! `forward_raw_logits`) lives in `larql-compute` *below* the `FfnBackend`
//! seam and builds its own dense `ViewFfn`, so no caller-supplied backend can
//! reach it; giving it real dispatch is a change to the forward, not to the
//! engine. `no-cache` was in this population until
//! `larql_kv::generation::kv_prefill_run` — which is also the oracle the
//! dispatch ring is compared against — gained the hook.

use larql_inference::ffn::MoeFfn;
use larql_inference::kv_engine::EngineError;
use larql_inference::model::ModelWeights;
use larql_kv::markov_residual_codec::ColdResidualCodec;
use larql_kv::EngineKind;
use ndarray::Array2;

use crate::entry::{NEXT_TOKEN, PROMPT};
use crate::routes::ExecutingRoute;

/// Sliding window for the windowed variants: wide enough that a
/// three-token prompt plus one decode step never reaches the limit.
///
/// Deliberately *not* at capacity. A windowed cache that must evict to make
/// room cannot be rewound at all — append-then-drop leaves the row count
/// unchanged while the oldest row is gone — and that case has its own pin in
/// [`crate::outcomes`]. Mixing it in here would make the whole sweep assert
/// the eviction rule instead of the refusal rule.
const WINDOW: usize = 8;
/// `boundary-kv` chunks its prefill; one chunk per prompt token here.
const CHUNK_TOKENS: usize = 1;
const SEQUENCE_ID: &str = "strict-refusal-gate";
/// Apollo's injection parameters. Values are immaterial — the engine is in
/// the table to be *excluded* by evidence, not to be exercised.
const APOLLO_LAYER: usize = 1;
const APOLLO_COEFFICIENT: f32 = 1.0;
const APOLLO_TOP_K: usize = 1;

/// Whether an engine dispatches experts at all.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Coverage {
    /// Reaches `FfnBackend::forward_moe_full_layer`, so a strict route can
    /// refuse through it and the refusal must terminate the operation.
    RoutesExperts,
    /// Has no `FfnBackend` seam in its forward, so it cannot dispatch
    /// experts at all. Must refuse a hybrid-MoE architecture outright rather
    /// than answer it densely — see this module's header.
    NoExpertSeam,
}

pub struct EngineUnderTest {
    pub label: &'static str,
    pub coverage: Coverage,
    build: fn(&ModelWeights) -> EngineKind,
}

impl EngineUnderTest {
    pub fn engine(&self, weights: &ModelWeights) -> larql_kv::AnyEngine {
        (self.build)(weights).build(larql_inference::cpu_engine_backend())
    }
}

/// Every `EngineKind` variant, exactly once.
///
/// Written as a table rather than a list of hand-rolled tests so a new
/// variant that nobody classified shows up as a missing row here — the
/// arity assertion at the end of each sweep is what makes that fail loudly.
pub const ALL: [EngineUnderTest; 9] = [
    EngineUnderTest {
        label: "standard",
        coverage: Coverage::RoutesExperts,
        build: |_| EngineKind::Standard { window_size: None },
    },
    EngineUnderTest {
        label: "standard:windowed",
        coverage: Coverage::RoutesExperts,
        build: |_| EngineKind::Standard {
            window_size: Some(WINDOW),
        },
    },
    EngineUnderTest {
        label: "markov-rs",
        coverage: Coverage::RoutesExperts,
        build: |_| EngineKind::MarkovResidual { window_size: None },
    },
    EngineUnderTest {
        label: "markov-rs-codec",
        coverage: Coverage::RoutesExperts,
        build: |_| EngineKind::MarkovResidualCodec {
            window_size: None,
            codec: ColdResidualCodec::Bf16,
        },
    },
    EngineUnderTest {
        label: "turbo-quant",
        coverage: Coverage::RoutesExperts,
        build: |_| EngineKind::TurboQuant { bits: 4 },
    },
    EngineUnderTest {
        label: "unlimited-context",
        coverage: Coverage::RoutesExperts,
        build: |_| EngineKind::WindowedCheckpoint {
            window_size: WINDOW,
        },
    },
    EngineUnderTest {
        label: "boundary-per-layer",
        coverage: Coverage::RoutesExperts,
        build: |w| EngineKind::BoundaryPerLayer {
            window_size: None,
            num_layers: w.num_layers,
        },
    },
    EngineUnderTest {
        label: "boundary-kv",
        coverage: Coverage::RoutesExperts,
        build: |_| EngineKind::BoundaryKv {
            window_size: None,
            chunk_tokens: CHUNK_TOKENS,
            sequence_id: SEQUENCE_ID.to_string(),
        },
    },
    EngineUnderTest {
        label: "no-cache",
        coverage: Coverage::RoutesExperts,
        build: |_| EngineKind::NoCache,
    },
];

/// Apollo, kept out of [`ALL`] because it cannot be driven without a boundary
/// store — every other question the sweep asks would be answered by its state
/// checks rather than by its routing. The architecture refusal below fires
/// *before* those checks, which is the whole point: it costs nothing and
/// mutates nothing.
pub const APOLLO: EngineUnderTest = EngineUnderTest {
    label: "apollo",
    coverage: Coverage::NoExpertSeam,
    build: |_| EngineKind::Apollo {
        injection_layer: APOLLO_LAYER,
        inject_coefficient: APOLLO_COEFFICIENT,
        top_k: APOLLO_TOP_K,
        bos_token_id: None,
    },
};

/// Which operation to drive. Both, because prefill and decode reach the FFN
/// through different bodies in most of these engines.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    Prefill,
    Decode,
}

impl Op {
    pub const ALL: [Self; 2] = [Self::Prefill, Self::Decode];
}

/// Drive `op` on a fresh engine with `ffn`.
///
/// A decode prefills first with a route that executes, so a refusal observed
/// afterwards can only have come from the decode step.
pub fn drive(
    under_test: &EngineUnderTest,
    op: Op,
    weights: &ModelWeights,
    ffn: &dyn larql_inference::ffn::FfnBackend,
) -> Result<Array2<f32>, EngineError> {
    let mut engine = under_test.engine(weights);
    if op == Op::Decode {
        let clean = MoeFfn::strict(weights, &ExecutingRoute);
        engine.prefill(weights, &clean, &PROMPT)?;
        return engine.decode_step(weights, ffn, NEXT_TOKEN);
    }
    engine.prefill(weights, ffn, &PROMPT)
}
