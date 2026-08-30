//! Bounded in-memory store backing the Responses API's `store` /
//! `previous_response_id` semantics.
//!
//! Every stored entry carries (a) the flattened conversation up to and
//! including the generated output — replayed ahead of a follow-up
//! request's `input` — and (b) the full response envelope JSON, served
//! verbatim by `GET /v1/responses/{id}`.
//!
//! The store is deliberately process-local and bounded: larql-server
//! is a model server, not a database. When the cap is hit the oldest
//! entry is evicted FIFO; clients that outlive the window get a 404
//! and are expected to resend the conversation inline (the same
//! contract as `store: false`).

use std::collections::HashMap;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

/// Maximum retained responses before FIFO eviction. At the default
/// 256-token completions this bounds the store to a few MB.
pub const MAX_STORED_RESPONSES: usize = 1024;

/// One flattened conversation turn. Tool traffic is stored in its
/// rendered-text form so replay needs no tool-shape awareness.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StoredMessage {
    pub role: String,
    pub content: String,
}

/// One persisted response.
#[derive(Clone, Debug)]
pub struct StoredResponse {
    pub id: String,
    /// Model that produced it — follow-ups must resolve to the same
    /// runtime, so this is replayed as the default model id.
    pub model_id: String,
    /// Full conversation: prior context + this request's input + the
    /// generated assistant turn.
    pub conversation: Vec<StoredMessage>,
    /// The response envelope exactly as returned to the client.
    pub envelope: serde_json::Value,
}

#[derive(Default)]
struct StoreInner {
    by_id: HashMap<String, Arc<StoredResponse>>,
    /// Insertion order for FIFO eviction.
    order: VecDeque<String>,
}

/// Thread-safe bounded response store. Cheap to clone via `AppState`'s
/// `Arc`; all methods take `&self`.
#[derive(Default)]
pub struct ResponseStore {
    inner: Mutex<StoreInner>,
}

impl ResponseStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert (or replace) a response, evicting the oldest entry when
    /// the store is at capacity.
    pub fn insert(&self, response: StoredResponse) {
        let mut inner = self.lock();
        let id = response.id.clone();
        if inner.by_id.insert(id.clone(), Arc::new(response)).is_none() {
            inner.order.push_back(id);
        }
        while inner.by_id.len() > MAX_STORED_RESPONSES {
            let Some(oldest) = inner.order.pop_front() else {
                break;
            };
            inner.by_id.remove(&oldest);
        }
    }

    pub fn get(&self, id: &str) -> Option<Arc<StoredResponse>> {
        self.lock().by_id.get(id).cloned()
    }

    /// Remove one response; returns whether it existed.
    pub fn remove(&self, id: &str) -> bool {
        let mut inner = self.lock();
        let existed = inner.by_id.remove(id).is_some();
        if existed {
            inner.order.retain(|entry| entry != id);
        }
        existed
    }

    pub fn len(&self) -> usize {
        self.lock().by_id.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, StoreInner> {
        // A poisoned lock only means a panic mid-insert; the map is
        // still structurally sound, so recover rather than wedge every
        // subsequent Responses request.
        self.inner.lock().unwrap_or_else(|p| p.into_inner())
    }
}

#[cfg(test)]
mod tests;
