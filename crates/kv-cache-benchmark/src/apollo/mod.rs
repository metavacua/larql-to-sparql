//! Tier 3 — Apollo v12 architecture (end-to-end on Gemma 3 4B).
//!
//! Rust port of the Python/MLX Apollo 11 demo. Sits above Tier 2's
//! `UnlimitedContextEngine` and trades per-window K/V checkpoints for a
//! single-vector boundary plus retrieval-driven injection:
//!
//! 1. **Sparse single-vector boundary at `crystal_layer`** (10 KB per window
//!    on Gemma 3 4B) rather than the per-layer K,V checkpoint Tier 2 uses.
//! 2. **Routing index** (~120 KB on Apollo 11): maps query keywords → window
//!    IDs, so retrieval targets the right window without scanning.
//! 3. **`vec_inject` retrieval index** + per-fact entries with
//!    `(token_id, coefficient, window_id, position_in_window, fact_id)`.
//! 4. **Injection at `injection_layer`** (L30 on Gemma 3 4B, coefficient
//!    ≈ 10× natural): retrieved fact token embeddings are additively
//!    injected at the residual stream to amplify them past the
//!    sparse-boundary reconstruction noise.
//!
//! Total store on Apollo 11 (176 windows × 512 tokens = 90K tokens):
//! boundaries 1.76 MB + token archive ~350 KB + routing ~120 KB +
//! vec_inject entries ~60 KB ≈ **2.8 MB total** vs ~56 GB standard KV cache.
//!
//! ## Correctness target (not bit-exact — task accuracy)
//!
//! Unlike Tiers 1/2, Apollo is not aiming for bit-exact KV reproduction
//! against joint forward. The correctness target is: for queries that can
//! be answered by a single retrievable fact from the `vec_inject` index,
//! produce the same top-1 token (and ideally same logit distribution
//! within KL < 0.01) as running the full document in context.
//!
//! ## Implementation status
//!
//! Four end-to-end query entry points land on real apollo11_store +
//! Gemma 3 4B (see `engine::ApolloEngine`): `query_greedy`,
//! `query_greedy_compressed`, `query_generate_uncompressed`,
//! `query_generate_compressed`. The "compressed" variants forward the
//! 10 KB boundary + query (~9 context tokens) and exercise the actual
//! compression claim; the "uncompressed" variants forward the window
//! tokens directly and are higher-fidelity but not compressed. Integration
//! tests in `tests/test_apollo_*.rs` are `#[ignore]`-gated on model
//! weights being present.
//!
//! Known simplification vs the Python reference: injection happens at the
//! last-token position only; Python injects at each entry's
//! `position_in_window`. See `engine.rs` module docs for the full list.
//!
//! ## Reference
//!
//! - `chuk-mlx/src/chuk_lazarus/inference/context/research/unlimited_engine.py`
//! - `chuk-mlx/.../vec_inject/_primitives.py`
//! - `apollo-demo/apollo11_store/` (store format reference)

pub mod entry;
pub mod npy;
pub mod routing;
pub mod store;
pub mod engine;

pub use entry::{VecInjectEntry, InjectionConfig};
pub use routing::{RoutingIndex, RoutingQuery};
pub use store::{ApolloStore, StoreManifest};
pub use engine::{ApolloEngine, ApolloError, GenerationTrace, QueryTrace};
