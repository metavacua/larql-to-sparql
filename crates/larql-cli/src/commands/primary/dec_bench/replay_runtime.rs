//! I/O-bound replay driver: sweeps batch × wire × dispatch against a live
//! expert server using frames built from a capture pool. Coverage-excluded
//! (needs a running server); every pure decision lives in `replay.rs`.
//!
//! Dispatch parity: `streaming` = sequential per-layer POSTs (step time =
//! Σ layer RTTs, mirroring `generate_with_remote_ffn`'s per-layer round
//! trips); `batch` = all layers fired in parallel via `std::thread::scope`
//! (mirroring `LayerShardedBackend::forward_predispatch_all`).
//!
//! Endpoint seam: each sweep point resolves to an [`Endpoint`]
//! (walk-ffn[-q8k] dense arms, experts multi-layer[-q8k] routed arms) that
//! owns path, content types, frame build, and response decode — all through
//! the production codecs. The routed arms replay the pool's captured
//! per-row `(expert_id, weight)` routing; the multi-layer wire IS the
//! production decode path for MoE (`backend.rs` fast path), unlike the
//! dense arms where per-layer-distinct residuals never share a frame.

use super::args::ReplayArgs;
use super::capture_format::CapturePool;
use super::output::{CaptureSummary, DecBenchJsonResult};
use super::pulse;
use super::replay::{
    build_frame, check_experts_response, derive_transmit_us, expand_sweep, gather_frame_inputs,
    moe_weight_stats, parse_batch_list, parse_layer_range, routed_denominators_for_point,
    routed_layer_subset, summarize, weight_bytes_per_token, wire_label_for_content_type,
    DispatchMode, Endpoint, EndpointKind, FrameInputs, MoeWeightStats, RequestSample,
    RoutedDenominators, ServeLatencySource, SweepPoint, SweepPointStats, WireSpec,
};

use larql_inference::ffn::remote::STATS_PATH;
use larql_inference::ffn::Q4K_Q8K_SUPERBLOCK_ELEMS;

pub(super) fn run_replay(args: &ReplayArgs) -> Result<(), Box<dyn std::error::Error>> {
    let pool = CapturePool::open(&args.capture)?;

    let endpoint_kind = EndpointKind::parse(&args.endpoint)?;
    let batches = parse_batch_list(&args.batch)?;
    if let Some(&max_b) = batches.iter().max() {
        if max_b > pool.num_prompts() {
            return Err(format!(
                "max batch {max_b} exceeds pool prompt count {} — capture more prompts",
                pool.num_prompts()
            )
            .into());
        }
    }
    let wires = WireSpec::parse_list(&args.wire)?;
    let dispatches = DispatchMode::parse_list(&args.dispatch)?;
    // Resolves each wire arm onto the endpoint family — f16/i8 or an
    // asymmetric in/return pair with `--endpoint experts` fails loudly
    // here, before any request is sent.
    let sweep = expand_sweep(&batches, &wires, &dispatches, endpoint_kind)?;

    if endpoint_kind == EndpointKind::Experts {
        if !pool.has_routing() {
            return Err(format!(
                "capture pool {} has no routing sidecars — the experts endpoints replay \
                 captured per-row routing; recapture with `larql dec-bench capture --routing`",
                args.capture.display()
            )
            .into());
        }
        if wires.iter().any(|w| w.is_q8k())
            && pool.manifest.hidden_size % Q4K_Q8K_SUPERBLOCK_ELEMS != 0
        {
            return Err(format!(
                "pool hidden {} is not a multiple of {Q4K_Q8K_SUPERBLOCK_ELEMS} — the \
                 multi-layer q8k wire quantises in {Q4K_Q8K_SUPERBLOCK_ELEMS}-element \
                 super-blocks (drop the q8k arm)",
                pool.manifest.hidden_size
            )
            .into());
        }
    }

    let steps = match args.steps {
        Some(s) => {
            if s == 0 || s > pool.manifest.steps {
                return Err(format!(
                    "--steps {s} out of range (pool has {} steps)",
                    pool.manifest.steps
                )
                .into());
            }
            s
        }
        None => pool.manifest.steps,
    };

    let base_url = args.ffn.trim_end_matches('/').to_string();
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(args.timeout_secs))
        .build()?;

    let stats_before = fetch_stats(&client, &base_url)?;
    let server_layers = stats_before
        .get("layers")
        .and_then(|v| v.as_u64())
        .unwrap_or(pool.manifest.num_layers as u64) as usize;
    if server_layers != pool.manifest.num_layers {
        eprintln!(
            "[dec-bench] WARNING: server reports {server_layers} layers, pool captured {} — \
             replaying the pool's layer space",
            pool.manifest.num_layers
        );
    }
    let mut layers = parse_layer_range(args.layers.as_deref(), pool.manifest.num_layers)?;
    if endpoint_kind == EndpointKind::Experts {
        let moe_layers = routed_layer_subset(&pool, &layers)?;
        let excluded = layers.len() - moe_layers.len();
        if moe_layers.is_empty() {
            return Err(
                "no MoE layer in the requested range carries captured routing — nothing to \
                 replay on the experts endpoints"
                    .into(),
            );
        }
        if excluded > 0 {
            eprintln!(
                "[dec-bench] excluding {excluded} non-MoE layer(s) from the routed sweep \
                 (and its denominator); {} layers remain",
                moe_layers.len()
            );
        }
        layers = moe_layers;
    }

    // Movement-ratio denominators. Dense endpoints: per_layer_dense_bytes
    // summed over the replayed layers (constant per point). Routed
    // endpoints: per_expert_bytes × captured routing, batch-dependent →
    // computed per point below; a null per_expert_bytes is a loud error.
    let (dense_weight_bytes_tok, weight_missing) = if endpoint_kind == EndpointKind::WalkFfn {
        match weight_bytes_per_token(&stats_before, &layers) {
            Some((bytes, missing)) => (Some(bytes), missing),
            None => {
                eprintln!(
                    "[dec-bench] server /v1/stats has no ffn_weights block — \
                     dec/movement_ratio will be omitted"
                );
                (None, 0)
            }
        }
    } else {
        (None, 0)
    };
    let moe_stats: Option<MoeWeightStats> = if endpoint_kind == EndpointKind::Experts {
        let ms = moe_weight_stats(&stats_before)?;
        if let (Some(server_tk), Some(pool_tk)) = (ms.top_k, pool.routing_top_k()) {
            if server_tk as usize != pool_tk {
                eprintln!(
                    "[dec-bench] WARNING: server top_k {server_tk} != pool routing top_k \
                     {pool_tk} — pool captured from a different arch?"
                );
            }
        }
        Some(ms)
    } else {
        None
    };

    eprintln!(
        "[dec-bench] replaying {} points on {}: batch {:?} × wire {:?} × dispatch {:?}, \
         {} layers × {} steps × {} repeats",
        sweep.len(),
        endpoint_kind.label(),
        batches,
        wires.iter().map(|w| w.label()).collect::<Vec<_>>(),
        dispatches.iter().map(|d| d.label()).collect::<Vec<_>>(),
        layers.len(),
        steps,
        args.repeats,
    );

    let mut points = Vec::with_capacity(sweep.len());
    let mut pulse_lines = Vec::with_capacity(sweep.len());
    for (idx, point) in sweep.iter().enumerate() {
        let routed_denoms: Option<RoutedDenominators> = match &moe_stats {
            Some(ms) => Some(routed_denominators_for_point(
                &pool,
                &layers,
                steps,
                point.batch,
                ms.per_expert_bytes,
            )?),
            None => None,
        };
        let stats = run_point(&client, &base_url, &pool, point, &layers, steps, args)?;
        let summary = summarize(
            point,
            &stats,
            dense_weight_bytes_tok,
            routed_denoms.as_ref(),
        );
        // Accept negotiation exists only on the walk-ffn endpoint; the
        // other endpoints serve a fixed CT. Only the RETURN direction can
        // fall back — the inbound direction is client-authored.
        if point.endpoint == Endpoint::WalkFfn
            && summary.served_wire_out != vec![summary.wire_out.clone()]
        {
            eprintln!(
                "[dec-bench] NOTE: {} arm's return direction was served as {:?} (Accept \
                 fallback — is LARQL_I8_WIRE set on the server for i8 returns?) — \
                 bandwidth numbers belong to the served format",
                summary.wire_format, summary.served_wire_out
            );
        }
        eprintln!(
            "[dec-bench] point {}/{}: endpoint={} batch={} wire={} dispatch={} → \
             step p50 {:.2} ms, p99 {:.2} ms, {:.1} tok/s, {:.0} B/tok{}",
            idx + 1,
            sweep.len(),
            summary.endpoint,
            summary.batch,
            summary.wire_format,
            summary.dispatch_mode,
            summary.step_ms_p50,
            summary.step_ms_p99,
            summary.tok_s,
            summary.payload_bytes_tok,
            summary
                .movement_ratio
                .map(|r| format!(", movement {r:.2e}"))
                .unwrap_or_default(),
        );
        pulse_lines.push(pulse::pulse_line(
            idx,
            &summary,
            args.net_rtt_ms,
            args.net_gbps,
            args.pulse_per_layer,
        ));
        points.push(summary);
    }

    let stats_after = fetch_stats(&client, &base_url)?;

    if let Some(path) = &args.pulse_file {
        std::fs::write(path, pulse::to_jsonl(&pulse_lines))
            .map_err(|e| format!("write pulse file {}: {e}", path.display()))?;
        eprintln!("[dec-bench] pulse → {}", path.display());
    }

    let record = DecBenchJsonResult {
        timestamp: {
            let secs = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            format!("{secs}")
        },
        endpoint: endpoint_kind.label().into(),
        ffn_url: base_url.clone(),
        capture: CaptureSummary::from(&pool.manifest),
        stats_before,
        stats_after,
        replayed_layer_count: layers.len(),
        layers,
        steps,
        repeats: args.repeats,
        warmup_passes: args.warmup_passes,
        weight_bytes_missing_layers: weight_missing,
        client_rayon_threads: rayon::current_num_threads(),
        net_rtt_ms: args.net_rtt_ms,
        net_gbps: args.net_gbps,
        points,
    };
    let json = serde_json::to_string_pretty(&record)?;
    match &args.output_file {
        Some(path) => {
            std::fs::write(path, &json)
                .map_err(|e| format!("write output file {}: {e}", path.display()))?;
            eprintln!("[dec-bench] run record → {}", path.display());
        }
        None => println!("{json}"),
    }
    Ok(())
}

fn fetch_stats(
    client: &reqwest::blocking::Client,
    base_url: &str,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let resp = client.get(format!("{base_url}{STATS_PATH}")).send()?;
    if !resp.status().is_success() {
        return Err(format!("/v1/stats returned {}", resp.status()).into());
    }
    Ok(resp.json()?)
}

/// Run one sweep point: warmup, then `repeats × steps` full passes over
/// `layers`.
fn run_point(
    client: &reqwest::blocking::Client,
    base_url: &str,
    pool: &CapturePool,
    point: &SweepPoint,
    layers: &[usize],
    steps: usize,
    args: &ReplayArgs,
) -> Result<SweepPointStats, Box<dyn std::error::Error>> {
    // Warmup: full passes over ALL replayed layers, discarded. Per-layer
    // warming matters — the server's FFN weights are mmap-backed, and a
    // cold layer's first touch pays page-fault cost that would otherwise
    // land in the first measured step's p99 (observed: 4s p99 on an
    // otherwise ~120ms point). Routed points always get ≥1 pass: the
    // post-warmup corruption guard below needs at least one response.
    let warmup_passes = if point.endpoint.requires_routing() {
        args.warmup_passes.max(1)
    } else {
        args.warmup_passes
    };
    let mut warm_any_nonzero = false;
    for _ in 0..warmup_passes {
        for &layer in layers {
            let inputs = gather_frame_inputs(pool, point.endpoint, layer, 0, point.batch)?;
            let s = send_one(client, base_url, point, &inputs, pool)?;
            warm_any_nonzero |= s.any_nonzero;
        }
    }
    // A routed replay whose responses are ALL zero is corrupt, not slow —
    // the batch-handler failure class where unresolvable/wrong experts
    // produce partial or empty sums (audit §1a). Refuse to measure it.
    if point.endpoint.requires_routing() && !warm_any_nonzero {
        return Err(format!(
            "routed warmup on {} returned all-zero responses across {} layers — replay is \
             corrupt (wrong expert ids, shard ownership mismatch, or unresolvable experts); \
             refusing to record timings",
            point.endpoint.label(),
            layers.len()
        )
        .into());
    }

    let mut stats = SweepPointStats::default();
    for _repeat in 0..args.repeats.max(1) {
        for step in 0..steps {
            match point.dispatch {
                DispatchMode::Streaming => {
                    let mut step_ms = 0.0f64;
                    for &layer in layers {
                        let inputs =
                            gather_frame_inputs(pool, point.endpoint, layer, step, point.batch)?;
                        let s = send_one(client, base_url, point, &inputs, pool)?;
                        step_ms += s.client_ms;
                        stats.samples.push(s);
                    }
                    stats.step_ms.push(step_ms);
                }
                DispatchMode::Batch => {
                    // Pre-fetch each layer's pool inputs (rows + routing)
                    // outside the timed window. Frame ENCODE stays
                    // deliberately INSIDE it (send_one encodes before
                    // posting): the fan-out wall time includes per-request
                    // encode as client-side cost, exactly as production
                    // clients pay it — predispatch also encodes each shard
                    // frame per request from already-materialised residuals.
                    let mut frames: Vec<FrameInputs> = Vec::with_capacity(layers.len());
                    for &layer in layers {
                        frames.push(gather_frame_inputs(
                            pool,
                            point.endpoint,
                            layer,
                            step,
                            point.batch,
                        )?);
                    }
                    let t0 = std::time::Instant::now();
                    let results: Vec<Result<RequestSample, String>> = std::thread::scope(|scope| {
                        let handles: Vec<_> = frames
                            .iter()
                            .map(|inputs| {
                                scope.spawn(move || {
                                    send_one(client, base_url, point, inputs, pool)
                                        .map_err(|e| e.to_string())
                                })
                            })
                            .collect();
                        handles
                            .into_iter()
                            .map(|h| {
                                h.join()
                                    .unwrap_or_else(|_| Err("replay worker panicked".into()))
                            })
                            .collect()
                    });
                    let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
                    for r in results {
                        stats
                            .samples
                            .push(r.map_err(|e| -> Box<dyn std::error::Error> { e.into() })?);
                    }
                    stats.step_ms.push(wall_ms);
                }
            }
        }
    }
    Ok(stats)
}

/// Send one B-row request for one (step, layer); validate the decoded shape
/// through the same codecs the production client uses. Frame encode happens
/// before the timer starts (streaming's `client_ms` = transport + server,
/// matching the pre-seam behaviour); response decode happens after it stops.
/// Encode and decode are timed separately (`encode_us`/`client_decode_us`),
/// and every request opts into the serve_us timing extension via the
/// `x-larql-timing: 1` header — the replay driver IS the new client
/// (dec-funnel §3 DEC-1A two-scoreboard schema).
fn send_one(
    client: &reqwest::blocking::Client,
    base_url: &str,
    point: &SweepPoint,
    inputs: &FrameInputs,
    pool: &CapturePool,
) -> Result<RequestSample, Box<dyn std::error::Error>> {
    use larql_inference::ffn::remote::{split_timing_trailer, TIMING_HEADER, TIMING_HEADER_VALUE};

    let hidden = pool.manifest.hidden_size;
    let batch = point.batch;
    let endpoint = point.endpoint;
    let layer = inputs.layer;

    let t_enc = std::time::Instant::now();
    let body = build_frame(endpoint, point.wire, inputs, batch, hidden)?;
    let encode_us = t_enc.elapsed().as_secs_f64() * 1e6;
    let bytes_sent = body.len() as u64;

    // Inbound direction: the request Content-Type declares how the frame
    // is encoded; the echo of its label is the run record's served_wire_in
    // (the client is authoritative for this direction).
    let request_ct = endpoint.request_content_type(point.wire);
    let served_wire_in = wire_label_for_content_type(request_ct);

    let mut req = client
        .post(format!("{base_url}{}", endpoint.path()))
        .header(reqwest::header::CONTENT_TYPE, request_ct)
        .header(TIMING_HEADER, TIMING_HEADER_VALUE)
        .body(body);
    if let Some(accept) = endpoint.accept(point.wire) {
        req = req.header(reqwest::header::ACCEPT, accept);
    }
    let t0 = std::time::Instant::now();
    let resp = req.send()?;
    let status = resp.status();
    let resp_ct = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let resp_body = resp.bytes()?;
    let client_ms = t0.elapsed().as_secs_f64() * 1000.0;
    if !status.is_success() {
        return Err(format!(
            "{} replay layer {layer} batch {batch}: server returned {status}",
            endpoint.label()
        )
        .into());
    }

    // Response decode — outside the timed send→receive window. Trailer
    // endpoints strip the opt-in serve_us trailer BEFORE the production
    // decoder runs, so its exact length checks see the pre-extension
    // payload; a pre-extension server yields `trailer_us = None`.
    let t_dec = std::time::Instant::now();
    let (payload, trailer_us) = match endpoint.serve_latency_source() {
        ServeLatencySource::TimingTrailer => split_timing_trailer(&resp_body),
        ServeLatencySource::EmbeddedMs => (&resp_body[..], None),
    };

    let (server_ms, serve_us, served_wire_out, any_nonzero) = match endpoint {
        Endpoint::WalkFfn => {
            let resp_ct = resp_ct.unwrap_or_else(|| larql_inference::BINARY_CT.to_string());
            let (resp_layer, server_ms, floats) =
                larql_inference::decode_single_response(&resp_ct, payload, hidden)?;
            if resp_layer != layer || floats.len() != batch * hidden {
                return Err(format!(
                    "walk-ffn replay layer {layer}: expected {batch}×{hidden} floats for \
                     layer {layer}, got {} floats for layer {resp_layer}",
                    floats.len()
                )
                .into());
            }
            // The f32 walk-ffn response embeds its latency in the header
            // (audit §2); serve_us is its µs twin.
            (
                Some(server_ms),
                Some(server_ms * 1000.0),
                wire_label_for_content_type(&resp_ct),
                floats.iter().any(|&v| v != 0.0),
            )
        }
        Endpoint::WalkFfnQ8k => {
            let entries = larql_inference::decode_q8k_batch_response_entries(payload)?;
            if entries.len() != batch
                || entries
                    .iter()
                    .any(|(l, v)| *l != layer || v.len() != hidden)
            {
                return Err(format!(
                    "q8k replay layer {layer}: expected {batch} entries × hidden {hidden}, \
                     got {} entries",
                    entries.len()
                )
                .into());
            }
            let any_nonzero = entries.iter().any(|(_, v)| v.iter().any(|&x| x != 0.0));
            (None, trailer_us, "q8k".into(), any_nonzero)
        }
        Endpoint::ExpertsMultiLayer | Endpoint::ExpertsMultiLayerQ8k => {
            let any_nonzero = check_experts_response(payload, layer, batch, hidden)?;
            let served_ct = resp_ct.unwrap_or_else(|| {
                larql_inference::ffn::moe_remote::MULTI_LAYER_BATCH_CONTENT_TYPE.to_string()
            });
            (
                None,
                trailer_us,
                wire_label_for_content_type(&served_ct),
                any_nonzero,
            )
        }
    };
    let client_decode_us = t_dec.elapsed().as_secs_f64() * 1e6;
    let (transmit_us, transmit_clamped) = derive_transmit_us(client_ms, serve_us);

    Ok(RequestSample {
        layer,
        client_ms,
        server_ms,
        serve_us,
        encode_us,
        client_decode_us,
        transmit_us,
        transmit_clamped,
        bytes_sent,
        bytes_recv: resp_body.len() as u64,
        served_wire_in,
        served_wire_out,
        any_nonzero,
    })
}
