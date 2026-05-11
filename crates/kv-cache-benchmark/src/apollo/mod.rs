//! Apollo — re-exported from `larql_kv::apollo`.
//!
//! The implementation now lives in larql-inference. This module re-exports
//! all public types so existing benchmark code continues to compile unchanged.

pub use larql_kv::apollo::routing::RoutingQuery;
pub use larql_kv::apollo::store::{ApolloStore, StoreManifest};
pub use larql_kv::apollo::{
    ApolloEngine, ApolloError, InjectionConfig, QueryTrace, RoutingIndex, VecInjectEntry,
};

// Sub-modules re-exported in case tests import from them directly.
pub use larql_kv::apollo::entry;
pub use larql_kv::apollo::routing;
pub use larql_kv::apollo::store;
