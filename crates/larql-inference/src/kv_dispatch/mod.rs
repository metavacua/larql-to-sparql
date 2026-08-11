//! `KvDispatch` engine-facing intent surface — moved to
//! `larql_compute::kv_dispatch` (ADR-0022 Step 3d). This shim
//! preserves `crate::kv_dispatch::*` paths used by engines + helpers.

pub use larql_compute::kv_dispatch::*;
// helpers.rs stays in inference because it depends on
// `AsyncComputeBackend` (still inference-side until Step 4). Every
// function takes `Option<&larql_vindex::VectorIndex>` plus real
// `Instant::now()` calls -- native-only.
#[cfg(not(target_arch = "wasm32"))]
pub mod helpers;
