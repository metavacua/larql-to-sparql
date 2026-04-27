pub mod proto {
    tonic::include_proto!("larql.grid.v1");
}

pub use proto::grid_service_client::GridServiceClient;
pub use proto::grid_service_server::{GridService, GridServiceServer};
pub use proto::server_message::Payload as ServerPayload;
pub use proto::router_message::Payload as RouterPayload;
pub use proto::*;
