//! Multi-layer batch dispatch — send every layer's routing decisions in one
//! request, receive all `h2` values back. HTTP and UDS only (gRPC has its
//! own bidirectional stream path; multi-layer-batch over gRPC isn't needed
//! because the stream already amortises per-call setup).
//!
//! The sequential server-side loop eliminates rayon oversubscription;
//! each task gets the full thread pool.

use super::super::error::RemoteMoeError;
use super::super::metrics;
use super::super::multi_layer_wire::{
    decode_multi_layer_response, encode_multi_layer_request, encode_multi_layer_request_q8k,
    MultiLayerResult, MultiLayerTask, MultiLayerTaskQ8K, MULTI_LAYER_BATCH_CONTENT_TYPE,
    MULTI_LAYER_BATCH_PATH, MULTI_LAYER_BATCH_Q8K_CONTENT_TYPE, MULTI_LAYER_BATCH_Q8K_PATH,
};
#[cfg(unix)]
use super::uds_call;
use super::{Shard, ShardTransport};

impl Shard {
    /// Send all layers' routing decisions in one request, receive all h2 values.
    ///
    /// HTTP and UDS only.
    pub(in super::super) fn call_multi_layer_batch(
        &self,
        tasks: &[MultiLayerTask],
    ) -> Result<Vec<MultiLayerResult>, RemoteMoeError> {
        let body = encode_multi_layer_request(tasks);
        let request_bytes = body.len();
        let active_experts: usize = tasks.iter().map(|t| t.expert_ids.len()).sum();
        match &self.transport {
            ShardTransport::Http(client) => {
                let url = format!("{}{MULTI_LAYER_BATCH_PATH}", self.config.url);
                let resp = client
                    .post(&url)
                    .header("Content-Type", MULTI_LAYER_BATCH_CONTENT_TYPE)
                    .header("Accept", MULTI_LAYER_BATCH_CONTENT_TYPE)
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
                metrics::record_call(&self.config.url, request_bytes, bytes.len(), active_experts);
                decode_multi_layer_response(&bytes).ok_or_else(|| {
                    RemoteMoeError::BadResponse("multi-layer-batch response truncated".into())
                })
            }
            #[cfg(unix)]
            ShardTransport::Uds(uds) => {
                let resp_bytes = uds_call(
                    uds,
                    MULTI_LAYER_BATCH_PATH,
                    MULTI_LAYER_BATCH_CONTENT_TYPE,
                    &body,
                )?;
                metrics::record_call(
                    &self.config.url,
                    request_bytes,
                    resp_bytes.len(),
                    active_experts,
                );
                decode_multi_layer_response(&resp_bytes).ok_or_else(|| {
                    RemoteMoeError::BadResponse("UDS multi-layer-batch response truncated".into())
                })
            }
            ShardTransport::Grpc(_) => Err(RemoteMoeError::Client(
                "call_multi_layer_batch unavailable for gRPC shards".into(),
            )),
        }
    }

    /// Q8K-prenormed variant: client sends pre-quantised h_norm instead of
    /// the raw residual.  4× smaller upload; server skips pre_experts_norm
    /// + Q8K quantisation and calls the matvec directly.
    pub(in super::super) fn call_multi_layer_batch_q8k(
        &self,
        tasks: &[MultiLayerTaskQ8K],
    ) -> Result<Vec<MultiLayerResult>, RemoteMoeError> {
        let body = encode_multi_layer_request_q8k(tasks);
        let request_bytes = body.len();
        let active_experts: usize = tasks.iter().map(|t| t.expert_ids.len()).sum();
        match &self.transport {
            ShardTransport::Http(client) => {
                let url = format!("{}{MULTI_LAYER_BATCH_Q8K_PATH}", self.config.url);
                let resp = client
                    .post(&url)
                    .header("Content-Type", MULTI_LAYER_BATCH_Q8K_CONTENT_TYPE)
                    .header("Accept", MULTI_LAYER_BATCH_CONTENT_TYPE)
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
                metrics::record_call(&self.config.url, request_bytes, bytes.len(), active_experts);
                decode_multi_layer_response(&bytes).ok_or_else(|| {
                    RemoteMoeError::BadResponse("multi-layer-batch-q8k response truncated".into())
                })
            }
            #[cfg(unix)]
            ShardTransport::Uds(uds) => {
                let resp_bytes = uds_call(
                    uds,
                    MULTI_LAYER_BATCH_Q8K_PATH,
                    MULTI_LAYER_BATCH_Q8K_CONTENT_TYPE,
                    &body,
                )?;
                metrics::record_call(
                    &self.config.url,
                    request_bytes,
                    resp_bytes.len(),
                    active_experts,
                );
                decode_multi_layer_response(&resp_bytes).ok_or_else(|| {
                    RemoteMoeError::BadResponse(
                        "UDS multi-layer-batch-q8k response truncated".into(),
                    )
                })
            }
            ShardTransport::Grpc(_) => Err(RemoteMoeError::Client(
                "call_multi_layer_batch_q8k unavailable for gRPC shards".into(),
            )),
        }
    }
}
