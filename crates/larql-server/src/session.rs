//! Per-session PatchedVindex management.
//!
//! Each session gets its own PatchedVindex overlay. The base vindex is shared
//! (readonly). Patches applied via the session API are isolated to that session.
//!
//! Sessions are identified by a `X-Session-Id` header. If no header is present,
//! patches go to the global (shared) PatchedVindex.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use larql_vindex::PatchedVindex;
use tokio::sync::RwLock;

use crate::state::LoadedModel;

/// Per-session state — an isolated PatchedVindex overlay.
pub struct SessionState {
    pub patched: PatchedVindex,
    last_accessed: Instant,
}

impl SessionState {
    pub fn new(base: larql_vindex::VectorIndex, now: Instant) -> Self {
        Self {
            patched: PatchedVindex::new(base),
            last_accessed: now,
        }
    }

    pub fn touch(&mut self, now: Instant) {
        self.last_accessed = now;
    }
}

/// Manages per-session PatchedVindex instances.
#[allow(dead_code)]
pub struct SessionManager {
    sessions: RwLock<HashMap<String, SessionState>>,
    ttl: Duration,
}

impl SessionManager {
    pub fn new(ttl_secs: u64) -> Self {
        Self {
            sessions: RwLock::new(HashMap::new()),
            ttl: Duration::from_secs(if ttl_secs == 0 { 3600 } else { ttl_secs }),
        }
    }

    /// Get or create a session's PatchedVindex.
    #[allow(dead_code)]
    pub async fn get_or_create(
        &self,
        session_id: &str,
        model: &Arc<LoadedModel>,
    ) -> PatchedVindex {
        let mut sessions = self.sessions.write().await;

        // Evict expired sessions opportunistically (max 10 per call).
        let now = Instant::now();
        let expired: Vec<String> = sessions
            .iter()
            .filter(|(_, s)| now.duration_since(s.last_accessed) > self.ttl)
            .take(10)
            .map(|(k, _)| k.clone())
            .collect();
        for k in expired {
            sessions.remove(&k);
        }

        if let Some(session) = sessions.get_mut(session_id) {
            session.last_accessed = now;
            // Clone the base and replay patches for isolation.
            let base = model.patched.read().await;
            let mut cloned = PatchedVindex::new(base.base().clone());
            for patch in &session.patched.patches {
                cloned.apply_patch(patch.clone());
            }
            return cloned;
        }

        // New session — start from the global patched state.
        let base = model.patched.read().await;
        let patched = PatchedVindex::new(base.base().clone());
        sessions.insert(
            session_id.to_string(),
            SessionState {
                patched: PatchedVindex::new(base.base().clone()),
                last_accessed: now,
            },
        );
        patched
    }

    /// Apply a patch to a session (not global).
    pub async fn apply_patch(
        &self,
        session_id: &str,
        model: &Arc<LoadedModel>,
        patch: larql_vindex::VindexPatch,
    ) -> (usize, usize) {
        let mut sessions = self.sessions.write().await;
        let now = Instant::now();

        let session = sessions
            .entry(session_id.to_string())
            .or_insert_with(|| {
                // We need the base — block briefly.
                let base = model.patched.blocking_read();
                SessionState {
                    patched: PatchedVindex::new(base.base().clone()),
                    last_accessed: now,
                }
            });

        session.last_accessed = now;
        let op_count = patch.operations.len();
        session.patched.apply_patch(patch);
        (op_count, session.patched.num_patches())
    }

    /// List patches for a session.
    pub async fn list_patches(&self, session_id: &str) -> Vec<serde_json::Value> {
        let sessions = self.sessions.read().await;
        match sessions.get(session_id) {
            Some(session) => session
                .patched
                .patches
                .iter()
                .map(|p| {
                    serde_json::json!({
                        "name": p.description.as_deref().unwrap_or("unnamed"),
                        "operations": p.operations.len(),
                        "base_model": p.base_model,
                    })
                })
                .collect(),
            None => vec![],
        }
    }

    /// Remove a patch from a session.
    pub async fn remove_patch(
        &self,
        session_id: &str,
        name: &str,
    ) -> Result<usize, String> {
        let mut sessions = self.sessions.write().await;
        let session = sessions
            .get_mut(session_id)
            .ok_or_else(|| format!("session '{}' not found", session_id))?;

        let idx = session
            .patched
            .patches
            .iter()
            .position(|p| p.description.as_deref().unwrap_or("unnamed") == name)
            .ok_or_else(|| format!("patch '{}' not found in session", name))?;

        session.patched.remove_patch(idx);
        Ok(session.patched.num_patches())
    }

    /// Blocking write access to sessions map (for use in spawn_blocking).
    pub fn sessions_blocking_write(&self) -> tokio::sync::RwLockWriteGuard<'_, HashMap<String, SessionState>> {
        self.sessions.blocking_write()
    }

    /// Number of active sessions.
    #[allow(dead_code)]
    pub async fn session_count(&self) -> usize {
        self.sessions.read().await.len()
    }
}
