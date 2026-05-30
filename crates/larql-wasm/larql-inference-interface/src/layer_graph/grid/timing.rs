use crate::ffn::moe_remote::{MoeRouterWeights, RemoteMoeError, ShardStream};
use crate::ffn::RemoteMoeBackend;

#[derive(Clone, Debug)]
pub(super) struct LayerTiming {
    pub total_ms: f32,
    pub route_fire_ms: f32,
    pub collect_ms: f32,
    /// One entry per shard: `(wall_collect_ms, server_compute_ms)`.
    pub per_shard: Vec<(f32, f32)>,
}

fn shard_compute_max(t: &LayerTiming) -> f32 {
    t.per_shard.iter().map(|(_, c)| *c).fold(0.0, f32::max)
}

pub(super) fn print_token_breakdown(label: &str, tok_idx: usize, timings: &[LayerTiming]) {
    if timings.is_empty() {
        return;
    }
    let n = timings.len();
    let total: f32 = timings.iter().map(|t| t.total_ms).sum();
    let route: f32 = timings.iter().map(|t| t.route_fire_ms).sum();
    let collect: f32 = timings.iter().map(|t| t.collect_ms).sum();
    let server_max: f32 = timings.iter().map(shard_compute_max).sum();
    let network = (collect - server_max).max(0.0);
    eprintln!(
        "[moe-timing] {label} tok={tok_idx} layers={n} \
         moe_total={total:.1}ms (route+fire={route:.1}ms collect={collect:.1}ms \
         | server_compute≈{server_max:.1}ms network≈{network:.1}ms)"
    );
}

pub(super) fn print_run_summary(label: &str, per_token: &[Vec<LayerTiming>]) {
    if per_token.is_empty() {
        return;
    }
    let n_tokens = per_token.len();
    let layers_per_tok = per_token.iter().map(|v| v.len()).max().unwrap_or(0);

    let mut tot_total = 0.0f32;
    let mut tot_route = 0.0f32;
    let mut tot_collect = 0.0f32;
    let mut tot_server = 0.0f32;
    for tok in per_token {
        tot_total += tok.iter().map(|t| t.total_ms).sum::<f32>();
        tot_route += tok.iter().map(|t| t.route_fire_ms).sum::<f32>();
        tot_collect += tok.iter().map(|t| t.collect_ms).sum::<f32>();
        tot_server += tok.iter().map(shard_compute_max).sum::<f32>();
    }
    let avg_total = tot_total / n_tokens as f32;
    let avg_route = tot_route / n_tokens as f32;
    let avg_collect = tot_collect / n_tokens as f32;
    let avg_server = tot_server / n_tokens as f32;
    let avg_net = (avg_collect - avg_server).max(0.0);

    eprintln!(
        "[moe-timing] {label} SUMMARY ({n_tokens} tokens, {layers_per_tok} MoE layers/token)"
    );
    eprintln!(
        "[moe-timing]   per-token avg: moe_total={avg_total:.1}ms \
         (route+fire={avg_route:.1}ms collect={avg_collect:.1}ms \
         | server_compute≈{avg_server:.1}ms network≈{avg_net:.1}ms)"
    );
    if layers_per_tok > 0 {
        let avg_per_layer_total = avg_total / layers_per_tok as f32;
        let avg_per_layer_collect = avg_collect / layers_per_tok as f32;
        let avg_per_layer_server = avg_server / layers_per_tok as f32;
        let avg_per_layer_net = (avg_per_layer_collect - avg_per_layer_server).max(0.0);
        eprintln!(
            "[moe-timing]   per-layer avg: total={avg_per_layer_total:.2}ms \
             collect={avg_per_layer_collect:.2}ms \
             (server≈{avg_per_layer_server:.2}ms net≈{avg_per_layer_net:.2}ms)"
        );
    }
    if avg_total > 0.0 {
        let collect_pct = 100.0 * avg_collect / avg_total;
        let server_pct = 100.0 * avg_server / avg_total;
        let net_pct = 100.0 * avg_net / avg_total;
        let route_pct = 100.0 * avg_route / avg_total;
        eprintln!(
            "[moe-timing]   bottleneck: collect={collect_pct:.0}% \
             (of which server≈{server_pct:.0}%, network≈{net_pct:.0}%) \
             route+fire={route_pct:.0}%"
        );
    }
}

/// Inner MoE call with optional timing capture.
#[allow(clippy::too_many_arguments)]
pub(super) fn moe_call_timed(
    remote: &RemoteMoeBackend,
    layer: usize,
    h_post_attn: &[f32],
    router: &MoeRouterWeights<'_>,
    streams: &mut [ShardStream],
    norm_offset: f32,
    eps: f32,
    timing: Option<&mut Vec<LayerTiming>>,
) -> Result<Vec<f32>, RemoteMoeError> {
    if streams.is_empty() {
        return remote.forward_moe(layer, h_post_attn, router, norm_offset, eps);
    }
    let Some(timing) = timing else {
        return remote.forward_moe_stream(layer, h_post_attn, router, streams, norm_offset, eps);
    };
    let t_total = std::time::Instant::now();
    let t_fire = std::time::Instant::now();
    let inflight =
        remote.forward_moe_stream_fire(layer, h_post_attn, router, streams, norm_offset, eps)?;
    let route_fire_ms = t_fire.elapsed().as_secs_f32() * 1000.0;
    let t_collect = std::time::Instant::now();
    let (h2, per_shard) = remote.forward_moe_stream_collect_with_timing(streams, inflight)?;
    let collect_ms = t_collect.elapsed().as_secs_f32() * 1000.0;
    let total_ms = t_total.elapsed().as_secs_f32() * 1000.0;
    timing.push(LayerTiming {
        total_ms,
        route_fire_ms,
        collect_ms,
        per_shard,
    });
    Ok(h2)
}
