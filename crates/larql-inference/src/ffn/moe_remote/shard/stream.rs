//! gRPC bidirectional-stream setup for [`super::Shard`].
//!
//! [`Shard::open_stream`] spawns a dedicated tokio task per shard that
//! forwards work between two channels — sync mpsc on the Metal-thread
//! side, async tokio mpsc on the gRPC side. The sync caller never calls
//! `block_on` per-layer; it just `send`s and `recv`s.

use prost::Message;

use super::super::error::RemoteMoeError;
use super::super::metrics;
use super::super::stream::ShardStream;
use super::{Shard, ShardTransport};

impl Shard {
    /// Open a bidirectional gRPC stream for one decode step.
    ///
    /// Spawns a dedicated async tokio task that:
    ///   1. Reads work inputs from `work_rx` (async channel — no thread wakeup)
    ///   2. Sends them on the gRPC stream via `await` (no block_on)
    ///   3. Awaits the server's response (async)
    ///   4. Puts the decoded result in `result_tx` (sync mpsc — condvar wakeup)
    ///
    /// The sync Metal thread communicates via `work_tx.send` (non-blocking) and
    /// `result_rx.recv()` (condvar, ~0.1ms) — no tokio Runtime::block_on anywhere.
    pub(in super::super) fn open_stream(&self) -> Result<ShardStream, RemoteMoeError> {
        match &self.transport {
            ShardTransport::Grpc(grpc) => {
                let rt = std::sync::Arc::clone(&grpc.runtime);
                let mut client = grpc.client.clone();

                // Work channel: Metal thread → async task (non-blocking send)
                let (work_tx, mut work_rx) = tokio::sync::mpsc::unbounded_channel::<
                    larql_router_protocol::ExpertLayerInput,
                >();

                // Result channel: async task → Metal thread (condvar recv).
                // The f32 carries `compute_ms` from the server (0.0 when the
                // server isn't recording timing) so the client can decompose
                // its wall-clock collect time into network vs server compute.
                let (result_tx, result_rx) =
                    std::sync::mpsc::channel::<Result<(Vec<f32>, f32), RemoteMoeError>>();
                let shard_url = self.config.url.clone();

                // Open the gRPC stream + spawn the dispatch task in one block_on.
                // This is the ONLY block_on — one-time stream setup, not per-layer.
                rt.block_on(async {
                    // Channel for feeding the gRPC request stream.
                    let (grpc_input_tx, mut grpc_input_rx) = tokio::sync::mpsc::unbounded_channel::<
                        larql_router_protocol::ExpertLayerInput,
                    >();

                    let req_stream = async_stream::stream! {
                        while let Some(msg) = grpc_input_rx.recv().await { yield msg; }
                    };
                    let mut grpc_output = client
                        .expert_stream(tonic::Request::new(req_stream))
                        .await
                        .map(|r| r.into_inner())
                        .map_err(|e| RemoteMoeError::ServerError {
                            status: e.code() as u16,
                            body: e.message().to_string(),
                        })?;

                    // Spawn the async dispatch loop.
                    tokio::spawn(async move {
                        use futures::StreamExt;
                        while let Some(input) = work_rx.recv().await {
                            let request_bytes = input.encoded_len();
                            let active_experts = input.expert_ids.len();
                            // Forward input to gRPC stream.
                            if grpc_input_tx.send(input).is_err() {
                                break;
                            }
                            // Await server response (pure async, no block_on).
                            let result = match grpc_output.next().await {
                                Some(Ok(out)) => {
                                    let response_bytes = out.encoded_len();
                                    if out.h2.len() % 4 != 0 {
                                        Err(RemoteMoeError::BadResponse("h2 unaligned".into()))
                                    } else {
                                        let h2: Vec<f32> = out
                                            .h2
                                            .chunks_exact(4)
                                            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                                            .collect();
                                        metrics::record_call(
                                            &shard_url,
                                            request_bytes,
                                            response_bytes,
                                            active_experts,
                                        );
                                        Ok((h2, out.compute_ms))
                                    }
                                }
                                Some(Err(e)) => Err(RemoteMoeError::ServerError {
                                    status: e.code() as u16,
                                    body: e.message().to_string(),
                                }),
                                None => Err(RemoteMoeError::BadResponse("stream ended".into())),
                            };
                            // Wake the Metal thread via condvar (much cheaper than block_on).
                            if result_tx.send(result).is_err() {
                                break;
                            }
                        }
                    });

                    Ok::<(), RemoteMoeError>(())
                })?;

                Ok(ShardStream {
                    work_tx,
                    result_rx: std::sync::Mutex::new(result_rx),
                    _runtime: rt,
                })
            }
            ShardTransport::Http(_) | ShardTransport::Uds(_) => Err(RemoteMoeError::Client(
                "open_stream requires grpc:// shards".into(),
            )),
        }
    }
}
