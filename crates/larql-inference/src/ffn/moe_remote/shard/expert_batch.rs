//! `Shard::call_batch` — single-expert batch dispatch.
//!
//! HTTP and UDS use the binary `application/x-larql-expert` wire body;
//! gRPC uses the `ExpertBatchRequest` proto over the persistent HTTP/2
//! channel.

use prost::Message;

use super::super::error::RemoteMoeError;
use super::super::metrics;
use super::super::wire::{
    decode_expert_response, encode_expert_request, ExpertCallItem, ExpertResultItem,
    EXPERT_BATCH_PATH, EXPERT_BINARY_CONTENT_TYPE,
};
#[cfg(unix)]
use super::uds_call;
use super::{Shard, ShardTransport};

impl Shard {
    /// Send a batch of expert calls to this shard.
    ///
    /// Dispatches via gRPC (persistent HTTP/2) when the shard URL starts with
    /// `grpc://`, otherwise falls back to binary HTTP.
    pub(in super::super) fn call_batch(
        &self,
        requests: &[ExpertCallItem],
    ) -> Result<Vec<ExpertResultItem>, RemoteMoeError> {
        match &self.transport {
            ShardTransport::Grpc(grpc) => {
                // Build protobuf items — raw bytes for residuals avoids varint overhead.
                let items: Vec<larql_router_protocol::ExpertBatchItem> = requests
                    .iter()
                    .map(|r| larql_router_protocol::ExpertBatchItem {
                        layer: r.layer as u32,
                        expert_id: r.expert_id as u32,
                        residual: r.residual.iter().flat_map(|v| v.to_le_bytes()).collect(),
                    })
                    .collect();

                let grpc_req = larql_router_protocol::ExpertBatchRequest { items };
                let request_bytes = grpc_req.encoded_len();
                // Block on the async gRPC call from this sync context.
                let mut client = grpc.client.clone();
                let t_call = std::time::Instant::now();
                let resp = grpc
                    .runtime
                    .block_on(client.expert_batch(tonic::Request::new(grpc_req)))
                    .map_err(|e| RemoteMoeError::ServerError {
                        status: e.code() as u16,
                        body: e.message().to_string(),
                    })?
                    .into_inner();
                let response_bytes = resp.encoded_len();
                metrics::record_call(
                    &self.config.url,
                    request_bytes,
                    response_bytes,
                    requests.len(),
                );

                eprintln!(
                    "[call_batch/grpc] n={} block_on={:.1}ms",
                    requests.len(),
                    t_call.elapsed().as_secs_f64() * 1000.0
                );
                // Decode proto results back to ExpertResultItem.
                resp.results
                    .into_iter()
                    .map(|r| {
                        if r.output.len() % 4 != 0 {
                            return Err(RemoteMoeError::BadResponse(
                                "output bytes not divisible by 4".into(),
                            ));
                        }
                        let output: Vec<f32> = r
                            .output
                            .chunks_exact(4)
                            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                            .collect();
                        Ok(ExpertResultItem {
                            layer: r.layer as usize,
                            expert_id: r.expert_id as usize,
                            output,
                        })
                    })
                    .collect()
            }

            ShardTransport::Http(client) => {
                // Binary HTTP fallback (application/x-larql-expert).
                let url = format!("{}{EXPERT_BATCH_PATH}", self.config.url);
                let body = encode_expert_request(requests);
                let request_bytes = body.len();
                let resp = client
                    .post(&url)
                    .header("Content-Type", EXPERT_BINARY_CONTENT_TYPE)
                    .header("Accept", EXPERT_BINARY_CONTENT_TYPE)
                    .body(body)
                    .send()
                    .map_err(|e| RemoteMoeError::Unreachable {
                        url: url.clone(),
                        cause: e.to_string(),
                    })?;

                if !resp.status().is_success() {
                    return Err(RemoteMoeError::ServerError {
                        status: resp.status().as_u16(),
                        body: resp.text().unwrap_or_default(),
                    });
                }

                let bytes = resp
                    .bytes()
                    .map_err(|e| RemoteMoeError::BadResponse(e.to_string()))?;
                metrics::record_call(&self.config.url, request_bytes, bytes.len(), requests.len());
                decode_expert_response(&bytes)
                    .ok_or_else(|| RemoteMoeError::BadResponse("binary response truncated".into()))
            }
            #[cfg(unix)]
            ShardTransport::Uds(uds) => {
                // Same wire body as the HTTP path; UDS framing is identical
                // to TCP HTTP/1.1 — only the transport differs.
                let body = encode_expert_request(requests);
                let request_bytes = body.len();
                let resp_bytes =
                    uds_call(uds, EXPERT_BATCH_PATH, EXPERT_BINARY_CONTENT_TYPE, &body)?;
                metrics::record_call(
                    &self.config.url,
                    request_bytes,
                    resp_bytes.len(),
                    requests.len(),
                );
                decode_expert_response(&resp_bytes).ok_or_else(|| {
                    RemoteMoeError::BadResponse("UDS expert/batch response truncated".into())
                })
            }
        }
    }
}
