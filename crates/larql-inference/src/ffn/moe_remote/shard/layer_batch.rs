//! `Shard::call_layer_batch` — one residual + K (expert_id, weight) pairs.
//!
//! Eliminates the K-1 redundant residual copies on the wire and the K-1
//! redundant `pre_experts_norm` + Q8_K quantisations on the server (the
//! server applies them once and shares across the K experts).
//!
//! Wire format: f32 by default, opt-in f16 via `LARQL_MOE_WIRE_F16=1`
//! (lifted into [`super::super::runtime::RemoteMoeRuntime`]).

use super::super::error::RemoteMoeError;
use super::super::metrics;
use super::super::wire::{
    decode_layer_batch_response, decode_layer_batch_response_f16, encode_layer_batch_request,
    encode_layer_batch_request_f16, ExpertCallItem, LAYER_BATCH_CONTENT_TYPE,
    LAYER_BATCH_F16_CONTENT_TYPE, LAYER_BATCH_F16_PATH, LAYER_BATCH_PATH,
};
#[cfg(unix)]
use super::uds_call;
use super::{Shard, ShardTransport};

impl Shard {
    /// Send a layer-batch request: ONE residual + K (expert_id, weight) pairs.
    /// Returns the router-weighted sum across the K experts owned by this
    /// shard.
    ///
    /// HTTP-only for now (gRPC variant TODO).  Falls back to `call_batch` if
    /// the shard transport is gRPC.
    pub(in super::super) fn call_layer_batch(
        &self,
        layer: usize,
        residual: &[f32],
        expert_ids: &[u32],
        expert_weights: &[f32],
    ) -> Result<Vec<f32>, RemoteMoeError> {
        match &self.transport {
            ShardTransport::Grpc(_) => {
                // TODO: gRPC variant.  For now, encode-and-fall-back to
                // call_batch with K identical residuals.
                let items: Vec<ExpertCallItem> = expert_ids
                    .iter()
                    .map(|&eid| ExpertCallItem {
                        layer,
                        expert_id: eid as usize,
                        residual: residual.to_vec(),
                    })
                    .collect();
                let results = self.call_batch(&items)?;
                // Apply weights and sum on the client (mirrors the server's
                // run_experts_cpu_batch behaviour for the http path).
                let hidden = residual.len();
                let mut out = vec![0.0f32; hidden];
                for (i, item) in results.iter().enumerate() {
                    let w = expert_weights[i];
                    for (a, &v) in out.iter_mut().zip(item.output.iter()) {
                        *a += w * v;
                    }
                }
                Ok(out)
            }
            ShardTransport::Http(client) => {
                // Per-stage client-side timing + wire-format selection.
                // Driven from the process-wide `RemoteMoeRuntime`; default
                // is f32 wire on (loopback / same-host grids — TCP buffer
                // costs dominate, f16 conversion CPU cancels the wire-byte
                // saving).  Set `LARQL_MOE_WIRE_F16=1` for LAN deployments
                // where the 5 KB/call wire saving matters more than the
                // 9 µs/call f32↔f16 conversion CPU.  Bench (M3 Max loopback,
                // 2026-05-01): f16 was 0.5-1% slower (within noise) on a
                // 100-token poem; expected to invert on >100 µs RTT links.
                let runtime = super::super::runtime::RemoteMoeRuntime::get();
                let timing = runtime.http_timing;
                let use_f16 = runtime.wire_f16;

                let url = if use_f16 {
                    format!("{}{LAYER_BATCH_F16_PATH}", self.config.url)
                } else {
                    format!("{}{LAYER_BATCH_PATH}", self.config.url)
                };
                let ct = if use_f16 {
                    LAYER_BATCH_F16_CONTENT_TYPE
                } else {
                    LAYER_BATCH_CONTENT_TYPE
                };

                let t_encode_in = std::time::Instant::now();
                let body = if use_f16 {
                    encode_layer_batch_request_f16(layer, residual, expert_ids, expert_weights)
                } else {
                    encode_layer_batch_request(layer, residual, expert_ids, expert_weights)
                };
                let request_bytes = body.len();
                let t_encode = t_encode_in.elapsed();

                let t_send_in = std::time::Instant::now();
                let resp = client
                    .post(&url)
                    .header("Content-Type", ct)
                    .header("Accept", ct)
                    .body(body)
                    .send()
                    .map_err(|e| RemoteMoeError::Unreachable {
                        url: url.clone(),
                        cause: e.to_string(),
                    })?;
                let t_send = t_send_in.elapsed();

                if !resp.status().is_success() {
                    return Err(RemoteMoeError::ServerError {
                        status: resp.status().as_u16(),
                        body: resp.text().unwrap_or_default(),
                    });
                }

                let t_recv_in = std::time::Instant::now();
                let bytes = resp
                    .bytes()
                    .map_err(|e| RemoteMoeError::BadResponse(e.to_string()))?;
                let t_recv = t_recv_in.elapsed();
                metrics::record_call(
                    &self.config.url,
                    request_bytes,
                    bytes.len(),
                    expert_ids.len(),
                );

                let t_decode_in = std::time::Instant::now();
                let out = if use_f16 {
                    decode_layer_batch_response_f16(&bytes)
                } else {
                    decode_layer_batch_response(&bytes)
                }
                .ok_or_else(|| {
                    RemoteMoeError::BadResponse("layer-batch response truncated".into())
                });
                let t_decode = t_decode_in.elapsed();

                if timing {
                    eprintln!(
                        "[shard.call_layer_batch] layer={layer} K={} wire={} \
                         encode={:.0}us send_total={:.0}us recv_body={:.0}us decode={:.0}us",
                        expert_ids.len(),
                        if use_f16 { "f16" } else { "f32" },
                        t_encode.as_secs_f64() * 1e6,
                        t_send.as_secs_f64() * 1e6,
                        t_recv.as_secs_f64() * 1e6,
                        t_decode.as_secs_f64() * 1e6,
                    );
                }

                out
            }
            #[cfg(unix)]
            ShardTransport::Uds(uds) => {
                // Manual HTTP/1.1 over UnixStream — same wire format as
                // the TCP `Http` variant, just no TCP stack.  The server
                // is the same axum router on a `UnixListener`; from the
                // handler's perspective it can't tell.
                let runtime = super::super::runtime::RemoteMoeRuntime::get();
                let timing = runtime.http_timing;
                let use_f16 = runtime.wire_f16;

                let path = if use_f16 {
                    LAYER_BATCH_F16_PATH
                } else {
                    LAYER_BATCH_PATH
                };
                let ct = if use_f16 {
                    LAYER_BATCH_F16_CONTENT_TYPE
                } else {
                    LAYER_BATCH_CONTENT_TYPE
                };

                let t_encode_in = std::time::Instant::now();
                let body = if use_f16 {
                    encode_layer_batch_request_f16(layer, residual, expert_ids, expert_weights)
                } else {
                    encode_layer_batch_request(layer, residual, expert_ids, expert_weights)
                };
                let request_bytes = body.len();
                let t_encode = t_encode_in.elapsed();

                let t_send_in = std::time::Instant::now();
                let resp_bytes = uds_call(uds, path, ct, &body)?;
                let t_send = t_send_in.elapsed();
                metrics::record_call(
                    &self.config.url,
                    request_bytes,
                    resp_bytes.len(),
                    expert_ids.len(),
                );

                let t_decode_in = std::time::Instant::now();
                let out = if use_f16 {
                    decode_layer_batch_response_f16(&resp_bytes)
                } else {
                    decode_layer_batch_response(&resp_bytes)
                }
                .ok_or_else(|| {
                    RemoteMoeError::BadResponse("layer-batch response truncated (uds)".into())
                });
                let t_decode = t_decode_in.elapsed();

                if timing {
                    eprintln!(
                        "[shard.call_layer_batch] layer={layer} K={} wire={} \
                         transport=uds encode={:.0}us send_total={:.0}us decode={:.0}us",
                        expert_ids.len(),
                        if use_f16 { "f16" } else { "f32" },
                        t_encode.as_secs_f64() * 1e6,
                        t_send.as_secs_f64() * 1e6,
                        t_decode.as_secs_f64() * 1e6,
                    );
                }
                out
            }
        }
    }
}
