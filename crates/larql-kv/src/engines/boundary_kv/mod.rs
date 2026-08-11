//! BoundaryKvEngine — `Standard` in-session, `larql-boundary` frames at chunk boundaries.
//!
//! Specification: [`crates/larql-inference/docs/specs/boundary-kv-engine.md`].
//!
//! The engine behaves as `StandardEngine` for prefill and decode; the additive
//! capability is per-chunk emission of [`larql_boundary::BoundaryFrame`] objects
//! into a [`BoundaryArchive`]. These frames are a transport / save-restore
//! format — they are not consulted during in-session decode, and the in-session
//! correctness contract is unchanged from `Standard` (bit-identical at the same
//! quantisation tier).
//!
//! See the spec for the contract and `BOUNDARY_REF_PROTOCOL.md` for the wire
//! format.

pub mod archive;
pub mod engine;
pub(crate) mod gate;
pub mod identity;

pub use archive::{ArchiveError, BoundaryArchive};
// `InMemoryArchive` holds `Mutex<HashMap<...>>` — no core/alloc
// equivalent under wasm32v1-none.
#[cfg(not(target_arch = "wasm32"))]
pub use archive::InMemoryArchive;
// `BoundaryKvEngineConfig` is pure data; `BoundaryKvEngine` wraps
// `StandardEngine` (WHOLESALE_NATIVE) and `impl KvEngine for
// BoundaryKvEngine` (KvEngine is not(wasm32)-gated upstream).
pub use engine::BoundaryKvEngineConfig;
#[cfg(not(target_arch = "wasm32"))]
pub use engine::BoundaryKvEngine;
pub use identity::BoundaryModelIdentity;
