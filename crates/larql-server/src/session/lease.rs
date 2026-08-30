//! The lock-free half of a session: identity, liveness, idle clock.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::clock::SessionClock;

/// The shared identity record for one session.
///
/// A session has two halves. The heavy half — the patch overlay — lives in
/// the [`SessionManager`](super::SessionManager)'s map behind an async
/// `RwLock`. This is the light half: an `Arc` a request can carry around
/// and consult from any context, including the blocking generation thread,
/// with no lock and no `.await`.
///
/// # Why the liveness flag exists
///
/// The N1 continuation cache retains a KV state *after* generation
/// finishes. That opens a resurrection race against session deletion:
///
/// ```text
/// request takes the previous turn's KV  →  DELETE /v1/sessions/{id}
///   →  request finishes  →  request inserts the new turn's KV
/// ```
///
/// A delete implemented purely as "remove from the map and purge the
/// cache" loses that race: the late insert re-populates a cache entry for
/// a session the operator was told is gone. So deletion (and TTL
/// eviction) *kills the lease*, and
/// [`ResponseKvCache::insert`](crate::response_kv::ResponseKvCache::insert)
/// checks [`Self::is_alive`] **while holding the cache lock**. Ordering
/// then makes the race unwinnable in either direction:
///
/// - if the insert reads `alive == true`, the killer had not yet stored
///   `false`, so its purge — which needs the same cache lock — is ordered
///   *after* the insert and removes the entry;
/// - otherwise the insert reads `false` and never happens at all.
///
/// The flag is [`Ordering::SeqCst`] on both sides so the store cannot be
/// reordered past the killer's lock acquisition.
///
/// # Why the counters live here
///
/// `resumptions` / `reused_tokens_total` are recorded by the same blocking
/// generation path, which cannot await the session map. Atomics on the
/// lease keep them lock-free and keep the per-session numbers on the same
/// record as the per-session identity.
pub struct SessionLease {
    id: String,
    model_id: String,
    clock: SessionClock,
    created_millis: u64,
    last_used_millis: AtomicU64,
    alive: AtomicBool,
    resumptions: AtomicU64,
    reused_tokens_total: AtomicU64,
}

impl SessionLease {
    /// Mint a live lease for `id`, bound to the runtime `model_id`, born
    /// at `at` on `clock`.
    pub(super) fn new(id: &str, model_id: &str, clock: SessionClock, at: Instant) -> Arc<Self> {
        let millis = clock.millis_at(at);
        Arc::new(Self {
            id: id.to_string(),
            model_id: model_id.to_string(),
            clock,
            created_millis: millis,
            last_used_millis: AtomicU64::new(millis),
            alive: AtomicBool::new(true),
            resumptions: AtomicU64::new(0),
            reused_tokens_total: AtomicU64::new(0),
        })
    }

    /// The session id this lease speaks for.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// The runtime binding the session was created against. The patch
    /// overlay is derived from that model's base index and a KV
    /// continuation is only meaningful under its weights, so this is
    /// creation-time provenance, not a mutable pointer.
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Whether the session still exists. See the type doc for the ordering
    /// contract this participates in.
    pub fn is_alive(&self) -> bool {
        self.alive.load(Ordering::SeqCst)
    }

    /// Retire the lease. Called on explicit deletion and on TTL eviction —
    /// both mean "this session is gone", and both must stop late
    /// continuation inserts.
    pub(super) fn kill(&self) {
        self.alive.store(false, Ordering::SeqCst);
    }

    /// Refresh the idle clock to now.
    pub fn touch(&self) {
        self.touch_at(Instant::now());
    }

    /// [`Self::touch`] with an explicit instant, so tests can advance time
    /// without sleeping.
    pub fn touch_at(&self, at: Instant) {
        self.last_used_millis
            .store(self.clock.millis_at(at), Ordering::Relaxed);
    }

    /// How long the session has been idle as of `at`.
    pub fn idle_at(&self, at: Instant) -> Duration {
        let now = self.clock.millis_at(at);
        Duration::from_millis(now.saturating_sub(self.last_used_millis.load(Ordering::Relaxed)))
    }

    /// Creation time, unix seconds.
    pub fn created_at_unix(&self) -> u64 {
        self.clock.unix_at(self.created_millis)
    }

    /// Last access, unix seconds.
    pub fn last_used_at_unix(&self) -> u64 {
        self.clock
            .unix_at(self.last_used_millis.load(Ordering::Relaxed))
    }

    /// When the session expires if left idle from its last access.
    pub fn expires_at_unix(&self, ttl: Duration) -> u64 {
        self.last_used_at_unix().saturating_add(ttl.as_secs())
    }

    /// Record that a chained generation on this session actually resumed
    /// from resident KV, serving `reused_tokens` prompt positions for free.
    pub fn record_resumption(&self, reused_tokens: usize) {
        self.resumptions.fetch_add(1, Ordering::Relaxed);
        self.reused_tokens_total
            .fetch_add(reused_tokens as u64, Ordering::Relaxed);
    }

    /// Generations on this session where resumption engaged.
    pub fn resumptions(&self) -> u64 {
        self.resumptions.load(Ordering::Relaxed)
    }

    /// Prompt tokens this session has served from resumed KV.
    pub fn reused_tokens_total(&self) -> u64 {
        self.reused_tokens_total.load(Ordering::Relaxed)
    }
}
