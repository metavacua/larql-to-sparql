//! larql-server library — shared between the binary and integration tests.
#![cfg_attr(target_arch = "wasm32", no_std)]
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
#[macro_use]
extern crate alloc;
// tonic::Status is a fat error type (176 bytes). It's our external contract
// for all gRPC handlers, so flipping to Box<Status> is not worth the churn.
#![allow(clippy::result_large_err)]
pub mod announce;
pub mod auth;
pub mod band_utils;
pub mod bootstrap;
pub mod cache;
pub mod embed_store;
pub mod env_flags;
pub mod error;
pub mod etag;
pub mod ffn_l2_cache;
pub mod grpc;
pub mod grpc_expert;
pub mod http;
pub mod metrics;
pub mod openapi;
pub mod ratelimit;
pub mod routes;
pub mod session;
pub mod shard_loader;
pub mod state;
pub mod wire;
