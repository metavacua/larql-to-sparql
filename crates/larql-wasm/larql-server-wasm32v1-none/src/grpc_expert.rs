//! gRPC `ExpertService` — unary batch + bidirectional streaming.

use std::pin::Pin;
use std::sync::Arc;
use std::time::Instant;

use futures::StreamExt;
use tonic::{Request, Response, Status, Streaming};

use larql_router_protocol::{
    ExpertBatchItem, ExpertBatchRequest, ExpertBatchResponse, ExpertBatchResult, ExpertLayerInput,
    ExpertLayerOutput, ExpertService,
};

use crate::env_flags;
use crate::state::AppState;

pub struct ExpertGrpcService {
    pub state: Arc<AppState>,
}

/// Process one batch item: decode residual bytes, dispatch to the per-expert
/// runner, and pack the f32 output back as little-endian bytes.  Pulled out so
/// `expert_batch` can switch between `par_iter` (small N) and `iter()` (large
/// N) without duplicating the per-item logic.
fn process_batch_item(
    state: &Arc<AppState>,
    item: &ExpertBatchItem,
) -> Result<ExpertBatchResult, Status> {
    let layer = item.layer as usize;
    let expert_id = item.expert_id as usize;
    if !item.residual.len().is_multiple_of(4) {
        return Err(Status::invalid_argument("residual not 4-byte aligned"));
    }
    let residual: Vec<f32> = item
        .residual
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let output = crate::routes::expert::run_expert(state, layer, expert_id, &residual)
        .map_err(|e| Status::internal(e.to_string()))?;
    Ok(ExpertBatchResult {
        layer: item.layer,
        expert_id: item.expert_id,
        output: output.iter().flat_map(|v| v.to_le_bytes()).collect(),
    })
}

type StreamOutput =
    Pin<Box<dyn futures::Stream<Item = Result<ExpertLayerOutput, Status>> + Send + 'static>>;

#[tonic::async_trait]
impl ExpertService for ExpertGrpcService {
    // ── Unary batch ──────────────────────────────────────────────────────────

    async fn expert_batch(
        &self,
        request: Request<ExpertBatchRequest>,
    ) -> Result<Response<ExpertBatchResponse>, Status> {
        self.state.bump_requests();
        let start = Instant::now();
        let req = request.into_inner();
        let state = Arc::clone(&self.state);
        let n_items = req.items.len();

        // Compute strategy: each `run_expert` already drives BLAS sgemv
        // (Accelerate on macOS / OpenBLAS on Linux), which is internally
        // multi-threaded.  Wrapping that in an outer `par_iter` over many
        // items creates thread oversubscription — the diagnostic measured
        // batch (120 items in `par_iter`) at ~400ms vs streaming (4 items in
        // `par_iter` × 30 sequential layer calls) at ~220ms.
        //
        // The right shape is one rayon task per CHUNK, with each chunk
        // processed serially inside.  That gives the outer level exactly
        // `min(n, n_cores)` work-stealing tasks (≤ core count, no
        // oversubscription) while letting BLAS use whatever threading it
        // wants on each call.  `LARQL_MOE_BATCH_MODE` lets the operator
        // override the auto-pick: `par`, `serial`, or `chunked` (default).
        let items = req.items;
        let timing_enabled = env_flags::moe_timing_enabled();
        let mode_override = env_flags::moe_batch_mode();
        let n_cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(8);
        let mode =
            mode_override
                .as_deref()
                .unwrap_or(if n_items <= n_cores { "par" } else { "chunked" });
        let results: Vec<ExpertBatchResult> = tokio::task::block_in_place(|| {
            use rayon::prelude::*;
            let t0 = Instant::now();
            let res = match mode {
                "par" => items
                    .par_iter()
                    .map(|item| process_batch_item(&state, item))
                    .collect::<Result<Vec<_>, Status>>(),
                "serial" => items
                    .iter()
                    .map(|item| process_batch_item(&state, item))
                    .collect::<Result<Vec<_>, Status>>(),
                _ => {
                    // chunked: ceil(n / n_cores) items per chunk, processed
                    // serially within each rayon task.
                    let chunk_size = n_items.div_ceil(n_cores).max(1);
                    items
                        .par_chunks(chunk_size)
                        .map(|chunk| -> Result<Vec<_>, Status> {
                            chunk
                                .iter()
                                .map(|item| process_batch_item(&state, item))
                                .collect()
                        })
                        .collect::<Result<Vec<Vec<_>>, Status>>()
                        .map(|chunks| chunks.into_iter().flatten().collect())
                }
            };
            if timing_enabled {
                eprintln!(
                    "[expert_batch grpc] n={n_items} mode={mode} cores={n_cores} \
                    elapsed={:.1}ms",
                    t0.elapsed().as_secs_f64() * 1000.0
                );
            }
            res
        })?;

        let latency_ms = start.elapsed().as_secs_f32() * 1000.0;
        Ok(Response::new(ExpertBatchResponse {
            results,
            latency_ms,
        }))
    }

    // ── Bidirectional streaming ──────────────────────────────────────────────

    type ExpertStreamStream = StreamOutput;

    async fn expert_stream(
        &self,
        request: Request<Streaming<ExpertLayerInput>>,
    ) -> Result<Response<Self::ExpertStreamStream>, Status> {
        self.state.bump_requests();
        let state = Arc::clone(&self.state);
        let mut in_stream = request.into_inner();

        let timing_enabled = env_flags::moe_timing_enabled();
        let out = async_stream::try_stream! {
            while let Some(msg) = in_stream.next().await {
                let input = msg?;
                let layer = input.layer as usize;
                if !input.residual.len().is_multiple_of(4) {
                    Err(Status::invalid_argument("residual not 4-byte aligned"))?;
                }
                let residual: Vec<f32> = input.residual.chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect();
                // post_experts_norm / norm_offset / eps are reserved for a
                // future server-side post-norm path; ignored today (the
                // client applies post-experts norm itself in
                // `RemoteMoeBackend::forward_moe_stream_collect`).  Keep
                // the wire fields in `ExpertLayerInput` for forward-compat;
                // discard them here.
                let _ = (&input.post_experts_norm, input.norm_offset, input.eps);
                let expert_ids: Vec<usize> =
                    input.expert_ids.iter().map(|&e| e as usize).collect();
                let expert_weights: Vec<f32> = input.expert_weights.clone();
                let state2 = Arc::clone(&state);
                let hidden = residual.len();
                let n_experts_active = expert_ids.len();

                let t_compute = Instant::now();
                // Path selection: when `metal-experts` feature is on AND a
                // Metal backend is available, dispatch the layer's selected
                // experts to GPU as one MoE call (q4k_ffn_gate_up + GELU +
                // K × q4k_matvec).  Falls through to the per-expert rayon
                // CPU path otherwise — preserves identical wire output.
                let path_used: &str;
                #[cfg(all(feature = "metal-experts", target_os = "macos"))]
                let metal_h2 = tokio::task::block_in_place(|| -> Result<Option<Vec<f32>>, Status> {
                    crate::routes::expert::run_experts_metal_batch(
                        &state2, layer, &residual, &expert_ids, &expert_weights,
                    )
                    .map_err(|e| Status::internal(e.to_string()))
                })?;
                #[cfg(not(all(feature = "metal-experts", target_os = "macos")))]
                let metal_h2: Option<Vec<f32>> = None;

                let h2 = if let Some(h2_metal) = metal_h2 {
                    path_used = "metal";
                    h2_metal
                } else if env_flags::use_legacy_cpu() {
                    // Legacy reference path — per-expert run_expert with
                    // its own pre_norm pass.  Kept as a correctness oracle
                    // while we debug whether the pooled `run_experts_cpu_batch`
                    // produces identical output.
                    path_used = "cpu-legacy";
                    tokio::task::block_in_place(|| -> Result<Vec<f32>, Status> {
                        use rayon::prelude::*;
                        let partial: Vec<(Vec<f32>, f32)> = expert_ids
                            .par_iter()
                            .zip(expert_weights.par_iter())
                            .filter(|(_, &w)| w != 0.0)
                            .filter_map(|(&eid, &w)| {
                                crate::routes::expert::run_expert(&state2, layer, eid, &residual)
                                    .ok()
                                    .map(|out| (out, w))
                            })
                            .collect();
                        let mut out = vec![0.0f32; hidden];
                        for (expert_out, weight) in partial {
                            for (acc, &v) in out.iter_mut().zip(expert_out.iter()) {
                                *acc += weight * v;
                            }
                        }
                        Ok(out)
                    })?
                } else {
                    path_used = "cpu";
                    tokio::task::block_in_place(|| -> Result<Vec<f32>, Status> {
                        crate::routes::expert::run_experts_cpu_batch(
                            &state2, layer, &residual, &expert_ids, &expert_weights,
                        )
                        .map_err(|e| Status::internal(e.to_string()))
                    })?
                };
                let compute_ms = t_compute.elapsed().as_secs_f32() * 1000.0;
                if timing_enabled {
                    eprintln!(
                        "[expert_stream] layer={layer} experts={n_experts_active} \
                         path={path_used} compute={compute_ms:.2}ms"
                    );
                }

                yield ExpertLayerOutput {
                    layer: input.layer,
                    h2: h2.iter().flat_map(|v| v.to_le_bytes()).collect(),
                    compute_ms,
                };
            }
        };

        Ok(Response::new(Box::pin(out)))
    }
}
