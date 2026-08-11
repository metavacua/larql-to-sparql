//! FFN backend selection, per-layer routing policy, and live router
//! construction.
//!
//! This module is the FFN-axis counterpart to
//! [`larql_kv::EngineKind`] — a typed dispatcher description that
//! parses from a CLI spec language, validates against a model's layer
//! count, and binds to runtime model handles to produce a live
//! `&dyn FfnBackend` per layer.
//!
//! # Module home
//!
//! The FFN axis is structurally orthogonal to the KV axis. KV engine
//! selection lives in `larql-kv`; FFN backend selection lives here in
//! `larql-inference` (alongside the `FfnBackend` trait impls
//! [`WeightFfn`], [`WalkFfn`], etc. that the policy constructs).
//!
//! # Spec syntax
//!
//! Single uniform backend:
//!
//! ```text
//! dense
//! walk:k=100
//! walk:k=None                   # explicit "no sparsity" form
//! remote-walk:endpoint=http://shard:8080
//! null                          # debug passthrough
//! ```
//!
//! Per-layer routing — backend wrapped in `{...}`, routing predicate
//! after `@`, multiple bindings joined by `;`:
//!
//! ```text
//! {walk:k=100}@layers=14-27;{dense}@otherwise
//! {walk:k=100}@layers=14-27;{dense}@layers=0-13,28-33
//! {walk:k=None}@all
//! ```
//!
//! Predicates today:
//!
//! - `layers=N-M[,N-M,...]` — half-open layer ranges. `14-27` covers
//!   L14 through L27 inclusive (becomes `14..28` internally).
//! - `all` — every layer; sugar for "no predicate."
//! - `otherwise` — every layer not covered by an earlier binding.
//!   Required when other bindings don't form a partition.
//!
//! The predicate slot is intentionally extensible — future predicates
//! like `confidence>0.9` or `dispatcher=<kind>` slot in without
//! changing the outer `{ffn}@pred` shape.
//!
//! # Usage flow
//!
//! Three steps from CLI spec to live dispatcher:
//!
//! ```text
//! FfnLayerPolicy::from_spec(spec)?     // parse: model-independent
//!   .validate_for(num_layers)?         // validate: needs layer count
//!   .build_router(weights, index)?     // bind: needs model handles
//! ```
//!
//! [`ValidatedFfnLayerPolicy`] is the type-system enforcement of
//! "validate before build" — its constructor is non-public so the
//! only way to obtain one is via [`FfnLayerPolicy::validate_for`].
//! [`BoundFfnRouter`] owns its backend instances so callers don't
//! have to manage backend lifetimes alongside the router's. Design
//! rationale: [`docs/ffn-build-router.md`](../../docs/ffn-build-router.md).
//!
//! # What this module does NOT do (v0 scope)
//!
//! - **No CLI flag wiring.** `larql bench --ffn <spec>` and
//!   `larql accuracy --ffn <spec>` come in a follow-up slice.
//! - **No `Sparse`, `LayerSharded`, `RemoteMoe` variants.** Each needs
//!   construction parameters that don't have an obvious CLI spelling
//!   today. Add when a use case forces the design.
//! - **No `RemoteWalk` build path** (v0 of `build_router`). The enum
//!   carries the variant so the parser shape is stable, but
//!   `build_router` errors on it for now — wiring requires threading
//!   the `RemoteWalkBackend` connection pool and is deferred.
//! - **No backend deduplication.** `build_router` constructs one
//!   backend instance per layer. For cheap backends (`Dense` /
//!   `Null`) this is trivial. For `Walk { k }`, each instance carries
//!   its own caches, but those caches are lazily populated per-layer
//!   so duplication overhead is bounded. Dedup is a v1 optimization
//!   if profiling surfaces a hotspot.

pub mod backend_kind;
pub mod policy;
// Live `WalkFfn`/`VectorIndex` binding -- native-only.
#[cfg(not(target_arch = "wasm32"))]
pub mod router;
pub mod routing;
pub mod validated;

pub use backend_kind::FfnBackendKind;
pub use policy::{FfnLayerPolicy, PolicyParseError, PolicyValidationError};
#[cfg(not(target_arch = "wasm32"))]
pub use router::{BoundFfnRouter, BuildError};
pub use routing::RoutingPredicate;
pub use validated::ValidatedFfnLayerPolicy;
