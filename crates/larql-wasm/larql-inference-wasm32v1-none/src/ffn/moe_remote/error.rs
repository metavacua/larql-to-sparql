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
// ── Public error type ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum RemoteMoeError {
    /// Could not reach the shard server (connection refused, DNS failure, etc.).
    Unreachable { url: String, cause: String },
    /// The server responded with a non-2xx status.
    ServerError { status: u16, body: String },
    /// Response body could not be parsed.
    BadResponse(String),
    /// No shard owns a required expert ID.
    NoShard { expert_id: usize },
    /// HTTP client construction failed.
    Client(String),
}

impl core::fmt::Display for RemoteMoeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Unreachable { url, cause } => {
                write!(f, "expert shard unreachable: {url} ({cause})")
            }
            Self::ServerError { status, body } => {
                write!(f, "expert shard returned {status}: {body}")
            }
            Self::BadResponse(msg) => write!(f, "bad expert response: {msg}"),
            Self::NoShard { expert_id } => write!(f, "no shard owns expert {expert_id}"),
            Self::Client(msg) => write!(f, "HTTP client error: {msg}"),
        }
    }
}

impl core::error::Error for RemoteMoeError {}
