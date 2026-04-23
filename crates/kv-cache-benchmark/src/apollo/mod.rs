//! Tier 3 — Apollo v12 architecture (SCAFFOLD, not end-to-end).
//!
//! This is the Rust port target for the Python/MLX Apollo 11 demo system.
//! It sits above Tier 2's RS-boundary mechanism and adds:
//!
//! 1. **Sparse single-vector boundary at `crystal_layer`** (10 KB per window
//!    on Gemma 3 4B) rather than the per-layer K,V checkpoint Tier 2 uses.
//! 2. **Routing index** (~120 KB on Apollo): maps query keywords → window IDs,
//!    lets `replay_window` target the right window without scanning.
//! 3. **`vec_inject` retrieval index** + per-fact entries with
//!    `(token_id, coefficient, window_id, position_in_window, fact_id)`.
//! 4. **Injection at `injection_layer`** with a `inject_coefficient` ≈ 10×
//!    natural: retrieved fact token embeddings are additively injected at
//!    the residual stream to amplify them past the sparse-boundary
//!    reconstruction noise.
//!
//! Total store on Apollo 11 (176 windows × 512 tokens = 90K tokens):
//!   boundaries 1.76 MB + token archive ~350 KB + routing ~120 KB
//!   + vec_inject entries ~60 KB ≈ **2.8 MB total**
//!   vs ~56 GB standard KV cache.
//!
//! ## Correctness target (not bit-exact — task accuracy)
//!
//! Unlike Tiers 1/2, Apollo is not aiming for bit-exact KV reproduction
//! against joint forward. The correctness target is: for queries that can
//! be answered by a single retrievable fact from the `vec_inject` index,
//! produce the same top-1 token (and ideally same logit distribution
//! within KL < 0.01) as running the full document in context.
//!
//! ## Status
//!
//! **Scaffold only.** Types and the public API surface are defined, but
//! none of the end-to-end functions are implemented. Porting targets:
//!
//! | Python reference | Rust target |
//! |---|---|
//! | `chuk-mlx/.../vec_inject/_primitives.py::retrieve` | `engine::ApolloEngine::retrieve` |
//! | `chuk-mlx/.../vec_inject/_primitives.py::inject_at_layer` | `engine::ApolloEngine::inject` |
//! | `apollo-demo/apollo11_store/` format | `store::ApolloStore` load/save |
//! | Routing index (tf-idf + keyword index) | `routing::RoutingIndex` |
//!
//! All unimplemented functions return `Err(NotImplemented)` or panic on
//! a `todo!()` so it's impossible to accidentally depend on scaffolded
//! behaviour. The intent is that subsequent work fills these in without
//! having to re-design the module layout.

pub mod entry;
pub mod npy;
pub mod routing;
pub mod store;
pub mod engine;

pub use entry::{VecInjectEntry, InjectionConfig};
pub use routing::{RoutingIndex, RoutingQuery};
pub use store::{ApolloStore, StoreManifest};
pub use engine::{ApolloEngine, ApolloError, GenerationTrace, QueryTrace};
