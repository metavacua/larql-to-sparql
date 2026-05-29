use std::sync::{Arc, RwLock};

use rayon::prelude::*;

use super::config::ShardConfig;
use super::error::RemoteMoeError;
use super::metrics;
use super::multi_layer_wire::{MultiLayerResult, MultiLayerTask, MultiLayerTaskQ8K};
use super::router::{rms_norm, MoeRouterWeights};
use super::shard::{Shard, ShardTransport};
use super::stream::{InflightMoe, ShardStream};
use super::wire::{ExpertCallItem, ExpertResultItem};
use larql_compute::cpu::ops::moe::quantize_x_to_q8k;

/// Per-shard call list element: (position, expert_id, residual).
type ShardCallItem = (usize, usize, Vec<f32>);
/// Output of `forward_layer_moe`: (output rows, optional per-expert (logit, weight)).
type LayerMoeResult = (Vec<f32>, Vec<(f32, f32)>);

// ── RemoteMoeBackend ───────────────────────────────────────────────────────

/// Remote MoE expert backend. Thread-safe — all methods take `&self`.
///
/// The shard map is stored behind an `RwLock` so `reshard()` can replace it
/// without interrupting in-flight `forward_moe` calls on other threads.
pub struct RemoteMoeBackend {
    pub(super) shards: Arc<RwLock<Vec<Shard>>>,
}

impl RemoteMoeBackend {
    /// Build with no shards and no health check. Tests only — the backend
    /// will return errors on any actual dispatch attempt.
    #[cfg(test)]
    pub fn new_disconnected() -> Self {
        Self {
            shards: Arc::new(RwLock::new(vec![])),
        }
    }

    /// Build from a shard list. Performs a health check on each shard.
    pub fn connect(configs: Vec<ShardConfig>) -> Result<Self, RemoteMoeError> {
        let shards: Result<Vec<Shard>, _> = configs.into_iter().map(Shard::connect).collect();
        Ok(Self {
            shards: Arc::new(RwLock::new(shards?)),
        })
    }

    /// Replace the shard map live (no model reload, no inference interruption).
    ///
    /// Reconnects to new shards, then atomically swaps the map.
    /// In-flight requests against old shards complete normally.
    pub fn reshard(&self, configs: Vec<ShardConfig>) -> Result<(), RemoteMoeError> {
        let new_shards: Result<Vec<Shard>, _> = configs.into_iter().map(Shard::connect).collect();
        *self.shards.write().unwrap() = new_shards?;
        Ok(())
    }

    /// Returns true if all shards use gRPC transport (`grpc://` URLs).
    /// When true, `open_streams` is available and `forward_moe_stream` can be used.
    pub fn has_grpc_shards(&self) -> bool {
        let shards = self.shards.read().unwrap();
        !shards.is_empty()
            && shards
                .iter()
                .all(|s| matches!(s.transport, ShardTransport::Grpc(_)))
    }

    /// Latency-stats probe: test-call each shard with a zero-length batch and
    /// return `(url, rtt_ms)` per shard. Non-fatal — returns partial results.
    pub fn probe_latency(&self) -> Vec<(String, f64)> {
        let shards = self.shards.read().unwrap();
        shards
            .par_iter()
            .map(|shard| {
                let t = std::time::Instant::now();
                let _ = shard.call_batch(&[]);
                let rtt_ms = t.elapsed().as_secs_f64() * 1000.0;
                (shard.config.url.clone(), rtt_ms)
            })
            .collect()
    }

    /// Run one MoE layer forward pass with experts dispatched remotely.
    ///
    /// Steps:
    ///   1. Router runs locally on `h` using `router`.
    ///   2. Selected experts are grouped by owning shard.
    ///   3. One `POST /v1/expert/batch` per shard (parallel).
    ///   4. Weighted outputs are summed; post-experts norm applied.
    ///
    /// Returns the expert-block contribution (same shape as `h`).
    pub fn forward_moe(
        &self,
        layer: usize,
        h: &[f32],
        router: &MoeRouterWeights<'_>,
        norm_offset: f32,
        eps: f32,
    ) -> Result<Vec<f32>, RemoteMoeError> {
        let hidden = h.len();
        if hidden == 0 || router.num_experts == 0 || router.top_k == 0 {
            return Ok(vec![0.0f32; hidden]);
        }

        // 1. Route locally.
        let (_h_norm, expert_indices, expert_weights) = router.route(h, norm_offset, eps);

        // 2. Build per-shard (expert_id, weight) lists.  The new
        //    layer-batch wire format ships ONE residual per shard plus K
        //    (expert_id, weight) pairs — saves the K-1 redundant residual
        //    copies that the legacy `call_batch` path forced.
        let shards = self.shards.read().unwrap();
        let mut shard_calls: Vec<(usize, Vec<u32>, Vec<f32>)> = (0..shards.len())
            .map(|i| (i, Vec::new(), Vec::new()))
            .collect();

        for (&expert_id, &weight) in expert_indices.iter().zip(expert_weights.iter()) {
            let shard_idx = shards
                .iter()
                .position(|s| s.owns_unit(layer, expert_id))
                .ok_or(RemoteMoeError::NoShard { expert_id })?;
            shard_calls[shard_idx].1.push(expert_id as u32);
            shard_calls[shard_idx].2.push(weight);
        }

        // 3. Parallel dispatch — one layer-batch call per shard that has
        //    work.  Each shard returns its own router-weighted partial sum;
        //    the client just sums shard partials (no per-expert weighting
        //    needed because the server already applied the weights).
        let shard_timing = metrics::shard_timing_enabled();
        let layer_start = std::time::Instant::now();
        let non_empty: Vec<(usize, &Vec<u32>, &Vec<f32>)> = shard_calls
            .iter()
            .filter(|(_, ids, _)| !ids.is_empty())
            .map(|(si, ids, ws)| (*si, ids, ws))
            .collect();
        if metrics::enabled() {
            for (si, ids, _) in &shard_calls {
                if ids.is_empty() {
                    metrics::record_skip(&shards[*si].config.url);
                }
            }
        }

        let results_per_shard: Vec<Result<Vec<f32>, RemoteMoeError>> = non_empty
            .par_iter()
            .map(|(si, ids, ws)| {
                let shard_url = &shards[*si].config.url;
                let issue_us = layer_start.elapsed().as_secs_f64() * 1e6;
                if shard_timing {
                    eprintln!(
                        "[moe-shard-timing] transport=http layer={layer} shard={shard_url} K={} issue_us={issue_us:.0}",
                        ids.len(),
                    );
                }
                let call_start = std::time::Instant::now();
                let result = shards[*si].call_layer_batch(layer, h, ids, ws);
                if shard_timing {
                    let done_us = layer_start.elapsed().as_secs_f64() * 1e6;
                    let wall_us = call_start.elapsed().as_secs_f64() * 1e6;
                    eprintln!(
                        "[moe-shard-timing] transport=http layer={layer} shard={shard_url} K={} done_us={done_us:.0} wall_us={wall_us:.0}",
                        ids.len(),
                    );
                }
                result
            })
            .collect();

        // 4. Sum shard partials into the layer's combined expert output.
        let mut out = vec![0.0f32; hidden];
        for result in results_per_shard {
            let shard_out = result?;
            if shard_out.len() != hidden {
                return Err(RemoteMoeError::BadResponse(format!(
                    "shard returned {} floats, expected {hidden}",
                    shard_out.len()
                )));
            }
            for (acc, &v) in out.iter_mut().zip(shard_out.iter()) {
                *acc += v;
            }
        }

        // 5. Post-experts norm.
        Ok(rms_norm(&out, router.post_experts_norm, eps, norm_offset))
    }

    /// Batch MoE forward for a full sequence of positions in one shot.
    ///
    /// Runs the router on every row of `h`, then issues **one** HTTP batch
    /// call per shard per layer (instead of one call per position). For a
    /// prefill of N positions this reduces dispatch from `N × shards` calls
    /// to `shards` calls — 18× fewer round trips for an 18-token context.
    ///
    /// Results are stitched back into an `[N, hidden]` output array by
    /// sequential index: the server returns items in request order, so we
    /// can match result[i] → request[i] without a position tag in the
    /// wire format.
    pub fn forward_moe_seq(
        &self,
        layer: usize,
        h: &ndarray::Array2<f32>,
        router: &MoeRouterWeights<'_>,
        norm_offset: f32,
        eps: f32,
    ) -> Result<ndarray::Array2<f32>, RemoteMoeError> {
        let seq_len = h.nrows();
        let hidden = h.ncols();
        if hidden == 0 || router.num_experts == 0 || router.top_k == 0 {
            return Ok(ndarray::Array2::zeros((seq_len, hidden)));
        }

        // 1. Route every position locally.
        // routing[pos] = (expert_indices, expert_weights)
        let mut routing: Vec<(Vec<usize>, Vec<f32>)> = Vec::with_capacity(seq_len);
        for pos in 0..seq_len {
            let row: Vec<f32> = h.row(pos).to_vec();
            let (_, idx, wts) = router.route(&row, norm_offset, eps);
            routing.push((idx, wts));
        }

        // 2. Build per-shard call lists preserving (pos, local_idx) so we
        //    can reconstruct the output ordering.
        //    shard_items[si] = Vec<(pos, expert_id, residual)>
        let shards = self.shards.read().unwrap();
        let mut shard_items: Vec<Vec<ShardCallItem>> =
            (0..shards.len()).map(|_| Vec::new()).collect();

        for (pos, route) in routing.iter().enumerate().take(seq_len) {
            let row: Vec<f32> = h.row(pos).to_vec();
            for &expert_id in &route.0 {
                let si = shards
                    .iter()
                    .position(|s| s.owns_unit(layer, expert_id))
                    .ok_or(RemoteMoeError::NoShard { expert_id })?;
                shard_items[si].push((pos, expert_id, row.clone()));
            }
        }

        // 3. One batch call per shard that has work (parallel).
        let non_empty: Vec<(usize, &Vec<ShardCallItem>)> = shard_items
            .iter()
            .enumerate()
            .filter(|(_, items)| !items.is_empty())
            .collect();

        let dispatch_results: Vec<(usize, Result<Vec<ExpertResultItem>, RemoteMoeError>)> =
            non_empty
                .par_iter()
                .map(|(si, items)| {
                    let calls: Vec<ExpertCallItem> = items
                        .iter()
                        .map(|(_, expert_id, residual)| ExpertCallItem {
                            layer,
                            expert_id: *expert_id,
                            residual: residual.clone(),
                        })
                        .collect();
                    (*si, shards[*si].call_batch(&calls))
                })
                .collect();

        // 4. Reassemble: for each shard, result[i] corresponds to
        //    shard_items[si][i].  Accumulate weighted sums per position.
        let mut out = ndarray::Array2::<f32>::zeros((seq_len, hidden));

        for (si, result) in dispatch_results {
            let items = &shard_items[si];
            let results = result?;
            if results.len() != items.len() {
                return Err(RemoteMoeError::BadResponse(format!(
                    "shard returned {} results for {} requests at layer {layer}",
                    results.len(),
                    items.len()
                )));
            }
            for ((pos, expert_id, _), item) in items.iter().zip(results.iter()) {
                if item.output.len() != hidden {
                    return Err(RemoteMoeError::BadResponse(format!(
                        "expert {expert_id} at pos {pos} returned {} floats, expected {hidden}",
                        item.output.len()
                    )));
                }
                // Find the weight for this expert at this position.
                let weight = routing[*pos]
                    .0
                    .iter()
                    .zip(routing[*pos].1.iter())
                    .find(|(&eid, _)| eid == *expert_id)
                    .map(|(_, &w)| w)
                    .unwrap_or(0.0);

                let mut row = out.row_mut(*pos);
                for (acc, &val) in row.iter_mut().zip(item.output.iter()) {
                    *acc += weight * val;
                }
            }
        }

        // 5. Post-experts norm per position.
        if !router.post_experts_norm.is_empty() {
            for pos in 0..seq_len {
                let row_vec: Vec<f32> = out.row(pos).to_vec();
                let normed = rms_norm(&row_vec, router.post_experts_norm, eps, norm_offset);
                for (dst, src) in out.row_mut(pos).iter_mut().zip(normed.iter()) {
                    *dst = *src;
                }
            }
        }

        Ok(out)
    }

    /// Open one gRPC streaming channel per shard for a decode step.
    ///
    /// Returns a `Vec<ShardStream>`, one per shard in the internal shard map.
    /// Each stream stays open until dropped; the caller sends one
    /// `ExpertLayerInput` per MoE layer and receives one `ExpertLayerOutput`.
    ///
    /// Use in `generate_with_remote_moe`:
    ///   ```ignore
    ///   let mut streams = backend.open_streams()?;
    ///   // inside moe_fn for each layer:
    ///   let h2 = backend.forward_moe_stream(layer, h_post_attn, &router, &mut streams, norm_offset, eps)?;
    ///   // streams are dropped (and gRPC streams closed) at end of decode step.
    ///   ```
    pub fn open_streams(&self) -> Result<Vec<ShardStream>, RemoteMoeError> {
        let shards = self.shards.read().unwrap();
        shards.iter().map(|shard| shard.open_stream()).collect()
    }

    /// Run one MoE layer via the already-open per-shard streams.
    ///
    /// Eliminates the per-call connection overhead of `forward_moe` — the
    /// gRPC streams stay alive for the entire decode step (30 layers) so
    /// each layer only pays the cost of sending/receiving one proto frame
    /// over an existing HTTP/2 connection (~0.5ms vs ~12ms per layer).
    pub fn forward_moe_stream(
        &self,
        layer: usize,
        h: &[f32],
        router: &MoeRouterWeights<'_>,
        streams: &mut [ShardStream],
        norm_offset: f32,
        eps: f32,
    ) -> Result<Vec<f32>, RemoteMoeError> {
        let inflight = self.forward_moe_stream_fire(layer, h, router, streams, norm_offset, eps)?;
        self.forward_moe_stream_collect(streams, inflight)
    }

    /// Fire half of `forward_moe_stream`: route locally, push one input per
    /// shard onto its async dispatch task, and return immediately.
    ///
    /// Pair with [`Self::forward_moe_stream_collect`] to retrieve the result.
    /// The [`InflightMoe`] handle carries the post-norm context so the caller
    /// does not need to keep the [`MoeRouterWeights`] borrow alive across the
    /// fire/collect boundary.
    ///
    /// Used by the GPU/MoE overlap path: the metal decode loop fires the MoE
    /// call as soon as `h_post_attn` is ready, encodes dense FFN on a fresh
    /// command buffer, and then collects — letting GPU dense FFN run in
    /// parallel with the remote round trip.
    pub fn forward_moe_stream_fire(
        &self,
        layer: usize,
        h: &[f32],
        router: &MoeRouterWeights<'_>,
        streams: &[ShardStream],
        norm_offset: f32,
        eps: f32,
    ) -> Result<InflightMoe, RemoteMoeError> {
        let hidden = h.len();
        if hidden == 0 || router.num_experts == 0 || router.top_k == 0 || streams.is_empty() {
            return Ok(InflightMoe {
                layer,
                hidden,
                active_stream_indices: Vec::new(),
                post_experts_norm: Vec::new(),
                norm_offset,
                eps,
            });
        }

        // 1. Route locally.
        let (_h_norm, expert_indices, expert_weights) = router.route(h, norm_offset, eps);

        // 2. Encode residual bytes once. The client applies post-experts norm
        // after collecting all shard outputs, so the gRPC request must not
        // carry that hidden-sized tensor per shard/layer.
        let residual_bytes: Vec<u8> = h.iter().flat_map(|v| v.to_le_bytes()).collect();

        // 3. Distribute expert_ids/weights across shards.
        let shards_guard = self.shards.read().unwrap();
        let num_shards = shards_guard.len();
        let shard_urls: Vec<String> = shards_guard.iter().map(|s| s.config.url.clone()).collect();
        let mut shard_eids: Vec<Vec<u32>> = vec![Vec::new(); num_shards];
        let mut shard_ewts: Vec<Vec<f32>> = vec![Vec::new(); num_shards];
        for (&eid, &w) in expert_indices.iter().zip(expert_weights.iter()) {
            let si = shards_guard
                .iter()
                .position(|s| s.owns_unit(layer, eid))
                .ok_or(RemoteMoeError::NoShard { expert_id: eid })?;
            shard_eids[si].push(eid as u32);
            shard_ewts[si].push(w);
        }
        drop(shards_guard);
        let active_stream_indices: Vec<usize> = shard_eids
            .iter()
            .enumerate()
            .filter_map(|(si, ids)| (!ids.is_empty()).then_some(si))
            .collect();
        if metrics::enabled() {
            for (si, url) in shard_urls.iter().enumerate() {
                if shard_eids[si].is_empty() {
                    metrics::record_skip(url);
                }
            }
        }
        if active_stream_indices.is_empty() {
            return Ok(InflightMoe {
                layer,
                hidden,
                active_stream_indices,
                post_experts_norm: router.post_experts_norm.to_vec(),
                norm_offset,
                eps,
            });
        }
        if active_stream_indices.iter().any(|&si| si >= streams.len()) {
            return Err(RemoteMoeError::BadResponse(format!(
                "stream map has {} streams for {num_shards} shards",
                streams.len()
            )));
        }

        // 4. Fire one input per stream in parallel.
        //
        // Each fire is `tokio::sync::mpsc::UnboundedSender::send` (non-blocking
        // channel push, ~1µs) plus building the `ExpertLayerInput` struct,
        // which clones `residual_bytes` (~hidden × 4 = 11 KB) per shard.
        // Rayon's thread pool is already initialised across the inference path
        // and amortises scheduling to single-µs overhead per task, so parallel
        // fire wins even at N=2 and scales linearly with shard count.
        //
        // Single-shard fast path skips the rayon overhead — same shape as
        // the parallel-collect path.
        let shard_timing = metrics::shard_timing_enabled();
        let layer_start = std::time::Instant::now();
        if active_stream_indices.len() == 1 {
            let si = active_stream_indices[0];
            let input = larql_router_protocol::ExpertLayerInput {
                layer: layer as u32,
                expert_ids: shard_eids[si].clone(),
                expert_weights: shard_ewts[si].clone(),
                residual: residual_bytes.clone(),
                post_experts_norm: Vec::new(),
                norm_offset,
                eps,
            };
            if shard_timing {
                let issue_us = layer_start.elapsed().as_secs_f64() * 1e6;
                eprintln!(
                    "[moe-shard-timing] transport=grpc layer={layer} shard={} K={} fire_us={issue_us:.0}",
                    shard_urls[si],
                    shard_eids[si].len(),
                );
            }
            streams[si].fire(input)?;
        } else {
            let residual_ref: &[u8] = &residual_bytes;
            active_stream_indices
                .par_iter()
                .try_for_each(|&si| -> Result<(), RemoteMoeError> {
                    let issue_us = layer_start.elapsed().as_secs_f64() * 1e6;
                    if shard_timing {
                        eprintln!(
                            "[moe-shard-timing] transport=grpc layer={layer} shard={} K={} fire_us={issue_us:.0}",
                            shard_urls[si],
                            shard_eids[si].len(),
                        );
                    }
                    let input = larql_router_protocol::ExpertLayerInput {
                        layer: layer as u32,
                        expert_ids: shard_eids[si].clone(),
                        expert_weights: shard_ewts[si].clone(),
                        residual: residual_ref.to_vec(),
                        post_experts_norm: Vec::new(),
                        norm_offset,
                        eps,
                    };
                    streams[si].fire(input)
                })?;
        }

        Ok(InflightMoe {
            layer,
            hidden,
            active_stream_indices,
            post_experts_norm: router.post_experts_norm.to_vec(),
            norm_offset,
            eps,
        })
    }

    /// Collect half of `forward_moe_stream`: condvar-wait one partial weighted
    /// sum per shard, accumulate, and apply the post-experts RMS norm.
    ///
    /// Each shard returns the raw weighted sum of its own experts (without
    /// post-norm) so the caller can sum across shards and norm the combined
    /// output once — `rms_norm(a) + rms_norm(b) ≠ rms_norm(a + b)`.
    pub fn forward_moe_stream_collect(
        &self,
        streams: &[ShardStream],
        inflight: InflightMoe,
    ) -> Result<Vec<f32>, RemoteMoeError> {
        self.forward_moe_stream_collect_with_timing(streams, inflight)
            .map(|(h2, _)| h2)
    }

    /// Same as [`Self::forward_moe_stream_collect`] but also returns
    /// per-shard `(wall_collect_ms, server_compute_ms)` for diagnostics.
    /// The `wall_collect_ms` is the wall-clock time the caller waited
    /// for that shard's response (network + server compute + decode);
    /// `server_compute_ms` is what the server reported (when timing is
    /// enabled there).  `network_ms ≈ wall_collect_ms − server_compute_ms`.
    pub fn forward_moe_stream_collect_with_timing(
        &self,
        streams: &[ShardStream],
        inflight: InflightMoe,
    ) -> Result<LayerMoeResult, RemoteMoeError> {
        let InflightMoe {
            layer,
            hidden,
            active_stream_indices,
            post_experts_norm,
            norm_offset,
            eps,
        } = inflight;
        let n_streams = active_stream_indices.len();

        if hidden == 0 || n_streams == 0 {
            return Ok((vec![0.0f32; hidden], Vec::new()));
        }

        // Parallel collect across shards: spawn one OS thread per stream and
        // join them all. Each thread blocks on its shard's `result_rx` condvar
        // independently, so the per-layer collect wall time is `max(per_shard)`
        // not `sum(per_shard)`. The win scales linearly with shard count and
        // is the load-bearing primitive for multi-shard remote topologies
        // (Kimi K2.6 / DeepSeek V4 class deployments) — see roadmap F-COLLECT.
        //
        // Single-shard runs hit the `n_streams == 1` shortcut to skip the
        // thread::scope overhead (~50µs/layer) — measurable on a single-shard
        // colocated bench where parallel and sequential are equivalent anyway.
        let shard_timing = metrics::shard_timing_enabled();
        let collect_start = std::time::Instant::now();
        type CollectResult = (usize, f32, Result<(Vec<f32>, f32), RemoteMoeError>);
        let results: Vec<CollectResult> = if n_streams == 1 {
            let si = active_stream_indices[0];
            let t0 = std::time::Instant::now();
            let res = streams[si].collect_with_timing();
            let wall_ms = t0.elapsed().as_secs_f32() * 1000.0;
            vec![(si, wall_ms, res)]
        } else {
            std::thread::scope(|s| {
                let handles: Vec<_> = streams
                    .iter()
                    .enumerate()
                    .filter_map(|(si, stream)| {
                        active_stream_indices.contains(&si).then_some((si, stream))
                    })
                    .map(|(si, stream)| {
                        s.spawn(move || -> CollectResult {
                            let t0 = std::time::Instant::now();
                            let res = stream.collect_with_timing();
                            let wall_ms = t0.elapsed().as_secs_f32() * 1000.0;
                            (si, wall_ms, res)
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|h| h.join().expect("collect thread panicked"))
                    .collect()
            })
        };

        let mut out = vec![0.0f32; hidden];
        let mut per_shard: Vec<(f32, f32)> = Vec::with_capacity(n_streams);
        for (si, wall_ms, res) in results {
            let (partial, server_compute_ms) = res?;
            if shard_timing {
                let done_us = collect_start.elapsed().as_secs_f64() * 1e6;
                eprintln!(
                    "[moe-shard-timing] transport=grpc layer={layer} shard_index={si} collect_done_us={done_us:.0} wall_us={:.0} server_compute_us={:.0}",
                    wall_ms as f64 * 1000.0,
                    server_compute_ms as f64 * 1000.0,
                );
            }
            per_shard.push((wall_ms, server_compute_ms));
            if partial.len() == hidden {
                for (acc, v) in out.iter_mut().zip(partial.iter()) {
                    *acc += v;
                }
            }
        }

        let normed = rms_norm(&out, &post_experts_norm, eps, norm_offset);
        Ok((normed, per_shard))
    }

    /// Pre-dispatch: route ALL layers at once, fire ONE batch call per shard
    /// (parallel), return h2 per layer.
    ///
    /// # Why faster than streaming
    ///
    /// `forward_moe` / `forward_moe_stream` make N sequential round-trips (one
    /// per layer). `forward_moe_predispatch` collapses them into ONE call per
    /// shard regardless of layer count.  The trade-off: each layer's expert
    /// input is computed from `h_post_attn` captured WITHOUT prior layers'
    /// expert contributions (pass-1 approximation), so the returned h2 values
    /// are slightly wrong for layers > 0.  In practice the error is small
    /// enough that the model still produces the correct top-1 token.
    ///
    /// # Usage
    ///
    /// 1. Run Metal with `moe_fn = |l, h| { capture[l] = h.to_vec(); zeros }`.
    /// 2. Call `forward_moe_predispatch(&captures, routers, ...)` — ONE async call.
    /// 3. Run Metal again with `moe_fn = |l, _h| { h2_per_layer[l].clone() }`.
    pub fn forward_moe_predispatch(
        &self,
        // h_post_attn captured per layer in the SKIP_MOE pass
        h_per_layer: &[Vec<f32>],
        // router weights for each layer (same length as h_per_layer)
        routers: &[MoeRouterWeights<'_>],
        norm_offset: f32,
        eps: f32,
    ) -> Result<Vec<Vec<f32>>, RemoteMoeError> {
        let num_layers = h_per_layer.len().min(routers.len());
        if num_layers == 0 {
            return Ok(vec![]);
        }
        let hidden = h_per_layer[0].len();
        let t0 = std::time::Instant::now();

        // Route each layer locally, build one dispatch task per (layer, shard).
        // One task = one call_layer_batch request to the server's
        // /v1/experts/layer-batch endpoint (efficient Q8_K path, weighted sum
        // returned).  This replaces the old call_batch path which hit
        // /v1/expert/batch (legacy per-item f32 path, ~7× slower per expert).
        struct LayerTask {
            layer: usize,
            shard_idx: usize,
            expert_ids: Vec<u32>,
            expert_weights: Vec<f32>,
        }

        let mut tasks: Vec<LayerTask> = Vec::with_capacity(num_layers);
        // h_norm per layer — captured during routing (first return value of route()).
        // Already computed, zero extra cost.  Used to build Q8K-prenormed wire tasks
        // that cut upload 4× vs sending the raw f32 residual.
        let mut h_norm_per_layer: Vec<Option<larql_compute::Q8KActivation>> =
            (0..num_layers).map(|_| None).collect();
        {
            let shards = self.shards.read().unwrap();
            let num_shards = shards.len();
            let all_http = !shards.is_empty() && shards.iter().all(|s| !s.is_grpc());
            for l in 0..num_layers {
                let (h_norm, expert_indices, expert_weights) =
                    routers[l].route(&h_per_layer[l], norm_offset, eps);
                if expert_indices.is_empty() {
                    continue;
                }
                // Capture Q8K-quantised h_norm for the multi-layer fast path.
                if all_http && h_norm.len() % 256 == 0 {
                    h_norm_per_layer[l] = Some(quantize_x_to_q8k(&h_norm));
                }
                let mut shard_ids: Vec<Vec<u32>> = vec![Vec::new(); num_shards];
                let mut shard_wts: Vec<Vec<f32>> = vec![Vec::new(); num_shards];
                for (&eid, &w) in expert_indices.iter().zip(expert_weights.iter()) {
                    // Skip experts not owned by any shard (partial deployment).
                    if let Some(si) = shards.iter().position(|s| s.owns_unit(l, eid)) {
                        shard_ids[si].push(eid as u32);
                        shard_wts[si].push(w);
                    }
                }
                for si in 0..num_shards {
                    if !shard_ids[si].is_empty() {
                        tasks.push(LayerTask {
                            layer: l,
                            shard_idx: si,
                            expert_ids: std::mem::take(&mut shard_ids[si]),
                            expert_weights: std::mem::take(&mut shard_wts[si]),
                        });
                    }
                }
            }
        } // shards lock released
        let t_route = t0.elapsed().as_secs_f64() * 1000.0;

        // ── Fast path: one multi-layer request per shard ────────────────────────
        //
        // When all shards are HTTP/UDS, collapse the 30 per-layer calls into
        // one request per shard.  The server processes layers sequentially so
        // rayon runs at full utilisation (no oversubscription), cutting server
        // compute from ~180 ms to ~30 ms and network from 30 × RTT to 1 × RTT.
        {
            let shards_guard = self.shards.read().unwrap();
            // Use `is_grpc()` helper to avoid naming the private UdsState type.
            let all_http = !shards_guard.is_empty() && shards_guard.iter().all(|s| !s.is_grpc());
            drop(shards_guard);

            if all_http {
                // Group tasks by shard — use Q8K if all h_norms were captured,
                // otherwise fall back to f32 residual.
                // Q8K wire: 4× smaller upload (client pre-quantises h_norm).
                // Disable with LARQL_DISABLE_Q8K_WIRE=1 for debugging.
                let q8k_enabled = super::runtime::RemoteMoeRuntime::get().q8k_enabled();
                let use_q8k = q8k_enabled
                    && h_norm_per_layer.iter().enumerate().all(|(l, q)| {
                        let has_task = tasks.iter().any(|t| t.layer == l);
                        !has_task || q.is_some()
                    });
                let shards_guard = self.shards.read().unwrap();
                let num_shards = shards_guard.len();
                let shard_results: Vec<(usize, Result<Vec<MultiLayerResult>, RemoteMoeError>)> =
                    if use_q8k {
                        let mut per_shard: Vec<Vec<MultiLayerTaskQ8K>> =
                            (0..num_shards).map(|_| Vec::new()).collect();
                        for task in &tasks {
                            if let Some(q8k) = &h_norm_per_layer[task.layer] {
                                per_shard[task.shard_idx].push(MultiLayerTaskQ8K {
                                    layer: task.layer,
                                    hidden,
                                    qs: q8k.qs.clone(),
                                    d: q8k.d.clone(),
                                    sums: q8k.sums.clone(),
                                    expert_ids: task.expert_ids.clone(),
                                    weights: task.expert_weights.clone(),
                                });
                            }
                        }
                        per_shard
                            .par_iter()
                            .enumerate()
                            .filter(|(si, t)| {
                                if t.is_empty() {
                                    metrics::record_skip(&shards_guard[*si].config.url);
                                    false
                                } else {
                                    true
                                }
                            })
                            .map(|(si, t)| (si, shards_guard[si].call_multi_layer_batch_q8k(t)))
                            .collect()
                    } else {
                        let mut per_shard: Vec<Vec<MultiLayerTask>> =
                            (0..num_shards).map(|_| Vec::new()).collect();
                        for task in &tasks {
                            per_shard[task.shard_idx].push(MultiLayerTask {
                                layer: task.layer,
                                residual: h_per_layer[task.layer].clone(),
                                expert_ids: task.expert_ids.clone(),
                                weights: task.expert_weights.clone(),
                            });
                        }
                        per_shard
                            .par_iter()
                            .enumerate()
                            .filter(|(si, t)| {
                                if t.is_empty() {
                                    metrics::record_skip(&shards_guard[*si].config.url);
                                    false
                                } else {
                                    true
                                }
                            })
                            .map(|(si, t)| (si, shards_guard[si].call_multi_layer_batch(t)))
                            .collect()
                    };
                drop(shards_guard);

                let t_dispatch = t0.elapsed().as_secs_f64() * 1000.0;
                let mut h2_per_layer: Vec<Vec<f32>> = vec![vec![0.0f32; hidden]; num_layers];
                for (_, result) in shard_results {
                    // Err: partial deployment — contribute zeros.
                    if let Ok(results) = result {
                        for r in results {
                            if r.h2.len() == hidden {
                                for (acc, &v) in h2_per_layer[r.layer].iter_mut().zip(r.h2.iter()) {
                                    *acc += v;
                                }
                            }
                        }
                    }
                }
                let t_accum = t0.elapsed().as_secs_f64() * 1000.0;
                if super::runtime::RemoteMoeRuntime::get().verbose {
                    eprintln!(
                        "[predispatch/multi] route={:.1}ms dispatch={:.1}ms accum={:.1}ms  shards={} wire={}",
                        t_route,
                        t_dispatch - t_route,
                        t_accum - t_dispatch,
                        num_shards,
                        if use_q8k { "q8k" } else { "f32" },
                    );
                }
                // Post-experts norm (caller expects it applied).
                for (l, h2) in h2_per_layer.iter_mut().enumerate() {
                    if !routers[l].post_experts_norm.is_empty() {
                        *h2 = rms_norm(h2, routers[l].post_experts_norm, eps, norm_offset);
                    }
                }
                return Ok(h2_per_layer);
            }
        }

        // ── Fallback: 30 parallel per-layer calls (gRPC shards) ─────────────────
        let shards = self.shards.read().unwrap();
        let task_results: Vec<(usize, Result<Vec<f32>, RemoteMoeError>)> = tasks
            .par_iter()
            .map(|task| {
                let result = shards[task.shard_idx].call_layer_batch(
                    task.layer,
                    &h_per_layer[task.layer],
                    &task.expert_ids,
                    &task.expert_weights,
                );
                (task.layer, result)
            })
            .collect();
        drop(shards);
        let t_dispatch = t0.elapsed().as_secs_f64() * 1000.0;

        // Accumulate per-layer partial sums.
        let mut h2_per_layer: Vec<Vec<f32>> = vec![vec![0.0f32; hidden]; num_layers];
        for (layer, result) in task_results {
            match result {
                Ok(partial) if partial.len() == hidden => {
                    for (acc, &v) in h2_per_layer[layer].iter_mut().zip(partial.iter()) {
                        *acc += v;
                    }
                }
                Ok(_) => {}
                Err(_) => {} // partial shard deployment — contribute zeros
            }
        }

        let t_accum = t0.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "[predispatch] route={:.1}ms dispatch={:.1}ms accum={:.1}ms  tasks={}",
            t_route,
            t_dispatch - t_route,
            t_accum - t_dispatch,
            tasks.len(),
        );

        // Apply post-experts norm per layer.
        for (l, h2) in h2_per_layer.iter_mut().enumerate() {
            if !routers[l].post_experts_norm.is_empty() {
                *h2 = rms_norm(h2, routers[l].post_experts_norm, eps, norm_offset);
            }
        }

        Ok(h2_per_layer)
    }
}
