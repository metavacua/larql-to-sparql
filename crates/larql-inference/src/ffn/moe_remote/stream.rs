use super::error::RemoteMoeError;

/// Receiver end of a shard stream's per-collect result channel.
/// Inner item is `(h2, server_compute_ms)`.
type ShardStreamResultRx =
    std::sync::Mutex<std::sync::mpsc::Receiver<Result<(Vec<f32>, f32), RemoteMoeError>>>;

// ── InflightMoe — handle returned by forward_moe_stream_fire ─────────────────
//
// Carries the post-norm context across the fire/collect boundary so callers do
// not need to retain the `MoeRouterWeights` borrow while GPU work runs in
// between.  `n_streams == 0` signals the trivial case (empty hidden / zero
// experts / no shards) where `collect` returns zeros without waiting.

/// Opaque handle for a fire-and-collect MoE round trip on a stream.
pub struct InflightMoe {
    pub(super) layer: usize,
    pub(super) hidden: usize,
    pub(super) active_stream_indices: Vec<usize>,
    pub(super) post_experts_norm: Vec<f32>,
    pub(super) norm_offset: f32,
    pub(super) eps: f32,
}

// ── ShardStream — async-native dispatch without block_on ─────────────────────
//
// Architecture: one async tokio task per shard manages the gRPC stream.
// The sync Metal decode thread communicates via std::sync::mpsc channels:
//
//   Metal thread               tokio async task
//   ────────────────────────   ──────────────────────────────────
//   work_tx.send(input)  ───▶  work_rx.recv().await
//                              gRPC stream: send + await response
//   result_rx.recv()     ◀───  result_tx.send(decoded_h2)
//
// `work_tx.send` is non-blocking (UnboundedSender — returns immediately).
// `result_rx.recv` uses a condvar/futex — ~0.1ms overhead vs ~1.45ms
// for `Runtime::block_on` on macOS.  The gRPC itself runs as proper async
// inside the tokio task without any scheduling penalty.

/// A live gRPC bidirectional stream to one shard.
///
/// The async gRPC work runs in a dedicated tokio task.  The sync Metal decode
/// thread fires inputs via `fire()` (non-blocking) and collects results via
/// `collect()` (condvar wait, ~0.1ms overhead).
pub struct ShardStream {
    /// Non-blocking input channel: Metal thread → tokio task.
    pub(super) work_tx: tokio::sync::mpsc::UnboundedSender<larql_router_protocol::ExpertLayerInput>,
    /// Blocking result channel: tokio task → Metal thread.
    /// Each item is `(h2, server_compute_ms)` — `compute_ms` is `0.0` when the
    /// server isn't recording timing.
    ///
    /// `std::sync::mpsc::Receiver` is `!Sync` (only `Send`); wrapping in
    /// `Mutex` makes `ShardStream: Sync`, which the parallel
    /// `forward_moe_stream_collect_with_timing` requires to spawn one
    /// `std::thread::scope` thread per shard. The mutex is contended only if
    /// two threads ever called `collect()` on the same stream concurrently —
    /// which the API contract forbids — so the lock is uncontended in
    /// practice and adds only the futex check cost.
    pub(super) result_rx: ShardStreamResultRx,
    /// Keep the runtime alive so the tokio task keeps running.
    pub(super) _runtime: std::sync::Arc<tokio::runtime::Runtime>,
}

impl ShardStream {
    /// Fire: push input to the async task, return immediately.
    /// Pair with `collect()` to retrieve the result.
    pub fn fire(
        &self,
        input: larql_router_protocol::ExpertLayerInput,
    ) -> Result<(), RemoteMoeError> {
        self.work_tx
            .send(input)
            .map_err(|_| RemoteMoeError::BadResponse("shard stream closed".into()))
    }

    /// Collect: condvar-wait for the async task's result (~0.1ms).
    /// No tokio block_on — just a futex wake when the result arrives.
    /// Discards `compute_ms` — use [`Self::collect_with_timing`] to keep it.
    pub fn collect(&self) -> Result<Vec<f32>, RemoteMoeError> {
        self.collect_with_timing().map(|(h2, _)| h2)
    }

    /// Collect with the server's `compute_ms` value attached. `compute_ms` is
    /// `0.0` when the server isn't recording timing (`LARQL_MOE_TIMING` unset).
    pub fn collect_with_timing(&self) -> Result<(Vec<f32>, f32), RemoteMoeError> {
        let rx = self.result_rx.lock().expect("result_rx mutex poisoned");
        rx.recv().unwrap_or(Err(RemoteMoeError::BadResponse(
            "shard result channel closed".into(),
        )))
    }

    /// Convenience: fire then collect.
    pub fn send_recv(
        &self,
        input: larql_router_protocol::ExpertLayerInput,
    ) -> Result<Vec<f32>, RemoteMoeError> {
        self.fire(input)?;
        self.collect()
    }
}
