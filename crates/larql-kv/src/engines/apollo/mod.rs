// `impl RetrievalEngine for ApolloEngine` (RetrievalEngine is
// not(wasm32)-gated upstream); its default prefill_quant/decode_step_quant
// methods also take `&larql_vindex::VectorIndex` unconditionally.
#[cfg(not(target_arch = "wasm32"))]
pub mod engine;
pub mod entry;
pub mod npy;
pub mod routing;
pub mod store;

#[cfg(not(target_arch = "wasm32"))]
pub use engine::{ApolloEngine, ApolloError, QueryTrace};
pub use entry::{InjectionConfig, VecInjectEntry};
pub use routing::{RoutingBuildError, RoutingIndex, MAX_ROUTABLE_WINDOWS};
// `ApolloStore` (struct + total_bytes()/hidden_size()) is portable;
// `StoreLoadError` embeds `zip::result::ZipError`/`std::io::Error`
// directly in its variants and cannot exist on wasm32 even as a type.
pub use store::ApolloStore;
#[cfg(not(target_arch = "wasm32"))]
pub use store::StoreLoadError;
