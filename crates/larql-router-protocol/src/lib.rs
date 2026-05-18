#[cfg(not(target_arch = "wasm32"))]
pub mod proto {
    tonic::include_proto!("larql.grid.v1");
}

#[cfg(not(target_arch = "wasm32"))]
pub mod expert_proto {
    tonic::include_proto!("larql.expert.v1");
}

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
