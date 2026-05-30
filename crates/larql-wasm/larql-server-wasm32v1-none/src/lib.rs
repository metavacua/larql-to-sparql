//! larql-server library — shared between the binary and integration tests.
#![cfg_attr(target_arch = "wasm32", no_std)]
// tonic::Status is a fat error type (176 bytes). It's our external contract
// for all gRPC handlers, so flipping to Box<Status> is not worth the churn.
#![allow(clippy::result_large_err)]
#[macro_use]
extern crate alloc;
#[allow(unused_imports)]
use alloc::{boxed::Box, string::{String, ToString}, vec::Vec, vec, format, borrow::{Cow, ToOwned}, rc::Rc, sync::Arc, collections::{BTreeMap, BTreeSet, VecDeque, BinaryHeap}};
#[cfg(target_arch = "wasm32")]
#[allow(unused_imports)]
use hashbrown::{HashMap, HashSet};
#[cfg(not(target_arch = "wasm32"))]
#[allow(unused_imports)]
use std::collections::{HashMap, HashSet};
#[cfg(target_arch = "wasm32")]
#[allow(unused_imports)]
use larql_wasm_math::FloatExt as _;
// larql-server is the native HTTP/gRPC layer (axum/tonic/tokio/clap/utoipa).
// On wasm32v1-none there is no server runtime, so the HTTP/gRPC/state surface
// is gated. Portable leaves (pure data/codec) remain ungated below.
#[cfg(not(target_arch = "wasm32"))]
pub mod announce;
#[cfg(not(target_arch = "wasm32"))]
pub mod auth;
#[cfg(not(target_arch = "wasm32"))]
pub mod band_utils;
#[cfg(not(target_arch = "wasm32"))]
pub mod bootstrap;
#[cfg(not(target_arch = "wasm32"))]
pub mod cache;
#[cfg(not(target_arch = "wasm32"))]
pub mod embed_store;
#[cfg(not(target_arch = "wasm32"))]
pub mod env_flags;
#[cfg(not(target_arch = "wasm32"))]
pub mod error;
pub mod etag;
#[cfg(not(target_arch = "wasm32"))]
pub mod ffn_l2_cache;
#[cfg(not(target_arch = "wasm32"))]
pub mod grpc;
#[cfg(not(target_arch = "wasm32"))]
pub mod grpc_expert;
pub mod http;
#[cfg(not(target_arch = "wasm32"))]
pub mod metrics;
#[cfg(not(target_arch = "wasm32"))]
pub mod openapi;
#[cfg(not(target_arch = "wasm32"))]
pub mod ratelimit;
#[cfg(not(target_arch = "wasm32"))]
pub mod routes;
#[cfg(not(target_arch = "wasm32"))]
pub mod session;
#[cfg(not(target_arch = "wasm32"))]
pub mod shard_loader;
#[cfg(not(target_arch = "wasm32"))]
pub mod state;
#[cfg(not(target_arch = "wasm32"))]
pub mod wire;
