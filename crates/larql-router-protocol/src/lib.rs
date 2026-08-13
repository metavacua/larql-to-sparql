// See crates/larql-core/src/lib.rs for the pattern-2 rationale. Unlike
// larql-models' detect/ module, there is no pure-logic subset here to
// preserve with a surgical split -- every public item is a
// tonic::include_proto!-generated gRPC type or a re-export of one, so
// the whole crate is wholesale-excluded on wasm32 (same shape as
// larql-models' connectors/encoders/loading/speech modules). All
// dependencies (tonic, prost, tokio, ...) are native-only for the same
// reason; see Cargo.toml.
#![cfg_attr(target_arch = "wasm32", no_std)]

#[cfg(not(target_arch = "wasm32"))]
pub mod proto {
    tonic::include_proto!("larql.grid.v1");
}

#[cfg(not(target_arch = "wasm32"))]
pub mod expert_proto {
    tonic::include_proto!("larql.expert.v1");
}

#[cfg(not(target_arch = "wasm32"))]
pub mod shard_proto {
    tonic::include_proto!("larql.shard.v1");
}

#[cfg(all(not(target_arch = "wasm32"), feature = "quic"))]
pub mod transport;

#[cfg(not(target_arch = "wasm32"))]
pub use expert_proto::expert_service_client::ExpertServiceClient;
#[cfg(not(target_arch = "wasm32"))]
pub use expert_proto::expert_service_server::{ExpertService, ExpertServiceServer};
#[cfg(not(target_arch = "wasm32"))]
pub use expert_proto::{
    ExpertBatchItem, ExpertBatchRequest, ExpertBatchResponse, ExpertBatchResult, ExpertLayerInput,
    ExpertLayerOutput,
};
#[cfg(not(target_arch = "wasm32"))]
pub use proto::grid_service_client::GridServiceClient;
#[cfg(not(target_arch = "wasm32"))]
pub use proto::grid_service_server::{GridService, GridServiceServer};
#[cfg(not(target_arch = "wasm32"))]
pub use proto::router_message::Payload as RouterPayload;
#[cfg(not(target_arch = "wasm32"))]
pub use proto::server_message::Payload as ServerPayload;
#[cfg(not(target_arch = "wasm32"))]
pub use proto::*;
#[cfg(not(target_arch = "wasm32"))]
pub use shard_proto::shard_service_client::ShardServiceClient;
#[cfg(not(target_arch = "wasm32"))]
pub use shard_proto::shard_service_server::{ShardService, ShardServiceServer};
#[cfg(not(target_arch = "wasm32"))]
pub use shard_proto::{ShardQuery, ShardResult};
