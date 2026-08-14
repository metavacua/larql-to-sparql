//! MAP-4 bridge: depth diagnostics, computed BLIND to the routing-lead-time
//! result, to test whether the excess-over-popularity dips at L20/L27
//! (`map4_bridge_routing_lead_time.rs`) align with independently-defined
//! structural transitions rather than being unexplained noise.
//!
//! Every signal here comes from data already on disk (the same 8 cached
//! `LARQL_CPU_DUMP_LAYERS` dumps) and is defined WITHOUT reference to the
//! routing-excess curve, so the comparison at the end is a real test, not
//! a fit to the dips:
//!
//! - **write-delta reorientation**: cosine between consecutive layers' total
//!   per-layer residual writes (`h_full(L) - h_full(L-1)`). A drop means
//!   layer L+1 moves the residual in a substantially different direction
//!   than layer L just did — a candidate "coordinate rewrite" boundary.
//! - **write-delta magnitude ratio**: how much bigger/smaller one layer's
//!   write is than the previous layer's — a spike either way is a candidate
//!   transition.
//! - **FFN-gain proxy**: `||h_full(L) - h_post_attn(L)|| / ||h_post_attn(L)||`
//!   — this layer's own FFN contribution, relative to its input magnitude.
//! - **router entropy / top-8 margin**: computed from the FULL per-expert
//!   softmax (not just the top-8 IDs) on the TRUE (non-stale) residual —
//!   properties of the routing decision itself, not of the stale-residual
//!   probe.
//! - **popularity concentration**: top-1 expert's share of all top-8 slots
//!   selected across the 8 prompts at that layer.
//! - **attention layer-type**: `arch.is_sliding_window_layer(l)`, real
//!   architecture data, independent of anything computed here.
//!
//! Usage: same dumps as map4_bridge_routing_lead_time.rs.
//! ```text
//! cargo run --release -p larql-inference --example map4_bridge_depth_diagnostics -- \
//!   --vindex <path> --dump-prefix /tmp/dump_p
//! ```

use std::collections::HashMap;

use larql_compute::cpu::ops::moe::{
    moe_expert_input, moe_route_from_router_input, moe_router_input,
};
use larql_compute::forward::dump_config::{cpu_layer_h_post_attn_path, cpu_layer_path};
use larql_compute::pipeline_layer::build_moe_weights;
use larql_compute::MoeLayerWeights;
use larql_models::ModelWeights;

const NUM_PROMPTS: usize = 8;

fn arg(name: &str) -> Option<String> {
    std::env::args()
        .collect::<Vec<_>>()
        .iter()
        .position(|a| a == name)
        .and_then(|i| std::env::args().nth(i + 1))
}

fn last_row(path: &str, width: usize) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let values: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert!(
        values.len().is_multiple_of(width) && !values.is_empty(),
        "{path}: {} values not a whole number of {width}-wide rows",
        values.len()
    );
    let rows = values.len() / width;
    values[(rows - 1) * width..].to_vec()
}

fn sub(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| x - y).collect()
}

fn norm(a: &[f32]) -> f32 {
    a.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let (na, nb) = (norm(a), norm(b));
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>() / (na * nb)
}

fn discover_moe_layers(
    weights: &ModelWeights,
    arch: &dyn larql_models::ModelArchitecture,
) -> Vec<usize> {
    (0..weights.num_layers)
        .filter(|&l| build_moe_weights(weights, arch, l).is_some())
        .collect()
}

/// Full per-expert router probability distribution (not just top-8) —
/// `moe_route_from_router_input` discards everything but the top-k, so this
/// reimplements the same matmul+softmax to keep the whole vector for
/// entropy/margin.
fn router_full_probs(router_in: &[f32], moe: &MoeLayerWeights<'_>) -> Vec<f32> {
    let hidden = router_in.len();
    let mut logits: Vec<f32> = (0..moe.num_experts)
        .map(|e| {
            let row = &moe.router_proj[e * hidden..(e + 1) * hidden];
            row.iter().zip(router_in).map(|(w, x)| w * x).sum()
        })
        .collect();
    let max = logits.iter().copied().fold(f32::MIN, f32::max);
    let mut sum = 0.0f32;
    for v in &mut logits {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in &mut logits {
        *v /= sum;
    }
    logits
}

fn entropy_bits(probs: &[f32]) -> f64 {
    probs
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -(p as f64) * (p as f64).log2())
        .sum()
}

fn margin_at_k(probs: &[f32], k: usize) -> f32 {
    let mut sorted = probs.to_vec();
    sorted.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap());
    sorted[k - 1] - sorted[k]
}

fn main() {
    println!("=== MAP-4 bridge: depth diagnostics (blind to the routing-excess result) ===");
    let vindex = match arg("--vindex") {
        Some(v) => v,
        None => {
            println!("skipped — set --vindex <path> and --dump-prefix <dir prefix>");
            return;
        }
    };
    let dump_prefix = arg("--dump-prefix").expect("set --dump-prefix <dir prefix>");

    let mut cb = larql_vindex::SilentLoadCallbacks;
    let weights = larql_vindex::load_model_weights_kquant(std::path::Path::new(&vindex), &mut cb)
        .expect("load real vindex weights");
    let arch = &*weights.arch;
    let hidden = weights.hidden_size;
    let norm_offset = arch.norm_weight_offset();
    let eps = arch.norm_eps();

    let layers = discover_moe_layers(&weights, arch);
    println!("layers: {layers:?}\n");

    let mut moe_at: HashMap<usize, MoeLayerWeights> = HashMap::new();
    for &l in &layers {
        moe_at.insert(
            l,
            build_moe_weights(&weights, arch, l).expect("must be MoE"),
        );
    }

    struct LayerStats {
        ffn_gain: f64,
        entropy_bits: f64,
        margin: f64,
        pop_top1_share: f64,
        pop_distinct_experts: usize,
        sliding_window: bool,
    }
    let mut stats: HashMap<usize, LayerStats> = HashMap::new();
    // write_delta[(layer, prompt)] = this layer's total residual write (h_full(L) - h_full(L-1), or - h_embed for L=0).
    let mut write_delta: HashMap<(usize, usize), Vec<f32>> = HashMap::new();

    for &l in &layers {
        let moe = &moe_at[&l];
        let mut ffn_gains = Vec::with_capacity(NUM_PROMPTS);
        let mut entropies = Vec::with_capacity(NUM_PROMPTS);
        let mut margins = Vec::with_capacity(NUM_PROMPTS);
        let mut expert_counts: HashMap<usize, usize> = HashMap::new();
        for i in 0..NUM_PROMPTS {
            let dump_dir = format!("{dump_prefix}{i}");
            let h_post_attn = last_row(&cpu_layer_h_post_attn_path(&dump_dir, l), hidden);
            let h_full = last_row(&cpu_layer_path(&dump_dir, l), hidden);
            let h_prev_full = if l == 0 {
                last_row(&format!("{dump_dir}/cpu_h_embed.f32"), hidden)
            } else {
                last_row(&cpu_layer_path(&dump_dir, l - 1), hidden)
            };
            write_delta.insert((l, i), sub(&h_full, &h_prev_full));

            let ffn_delta = sub(&h_full, &h_post_attn);
            let denom = norm(&h_post_attn);
            ffn_gains.push(if denom > 0.0 {
                (norm(&ffn_delta) / denom) as f64
            } else {
                0.0
            });

            let expert_input = moe_expert_input(&h_post_attn, moe, norm_offset, eps);
            let router_in = moe_router_input(&h_post_attn, &expert_input, moe, norm_offset, eps);
            let full_probs = router_full_probs(&router_in, moe);
            entropies.push(entropy_bits(&full_probs));
            margins.push(margin_at_k(&full_probs, moe.top_k) as f64);

            let (ids, _weights) = moe_route_from_router_input(&router_in, moe);
            for id in ids {
                *expert_counts.entry(id).or_insert(0) += 1;
            }
        }
        let top1_count = expert_counts.values().copied().max().unwrap_or(0);
        stats.insert(
            l,
            LayerStats {
                ffn_gain: ffn_gains.iter().sum::<f64>() / ffn_gains.len() as f64,
                entropy_bits: entropies.iter().sum::<f64>() / entropies.len() as f64,
                margin: margins.iter().sum::<f64>() / margins.len() as f64,
                // top1_count's natural ceiling is NUM_PROMPTS (an expert selected by
                // every prompt), not the raw slot count -- dividing by the latter
                // would peg this near a meaningless fixed ratio whenever most layers
                // have a near-universal expert, which they do here.
                pop_top1_share: top1_count as f64 / NUM_PROMPTS as f64,
                pop_distinct_experts: expert_counts.len(),
                sliding_window: arch.is_sliding_window_layer(l),
            },
        );
    }

    // reorientation[l] = mean cosine(write_delta(l), write_delta(l+1)) across prompts — candidate transition AT l+1 when low.
    let mut reorientation: HashMap<usize, f64> = HashMap::new();
    let mut magnitude_ratio: HashMap<usize, f64> = HashMap::new();
    for &l in &layers {
        let Some(&l_next) = layers.iter().find(|&&x| x == l + 1) else {
            continue;
        };
        let mut cosines = Vec::with_capacity(NUM_PROMPTS);
        let mut ratios = Vec::with_capacity(NUM_PROMPTS);
        for i in 0..NUM_PROMPTS {
            let a = &write_delta[&(l, i)];
            let b = &write_delta[&(l_next, i)];
            cosines.push(cosine(a, b) as f64);
            let (na, nb) = (norm(a), norm(b));
            ratios.push(if na > 0.0 { (nb / na) as f64 } else { 0.0 });
        }
        reorientation.insert(l_next, cosines.iter().sum::<f64>() / cosines.len() as f64);
        magnitude_ratio.insert(l_next, ratios.iter().sum::<f64>() / ratios.len() as f64);
    }

    println!("== per-layer diagnostics (all computed blind to the routing-excess result) ==");
    println!(
        "{:>4} {:>10} {:>9} {:>8} {:>8} {:>12} {:>9} {:>9}",
        "L", "reorient", "mag_rat", "ffn_gn", "entropy", "margin", "pop_top1", "distinct"
    );
    for &l in &layers {
        let s = &stats[&l];
        let reo = reorientation
            .get(&l)
            .map(|v| format!("{v:.3}"))
            .unwrap_or_else(|| "  n/a".into());
        let mag = magnitude_ratio
            .get(&l)
            .map(|v| format!("{v:.3}"))
            .unwrap_or_else(|| "  n/a".into());
        println!(
            "{l:>4} {reo:>10} {mag:>9} {:>8.3} {:>8.3} {:>12.5} {:>9.3} {:>9}  sliding={}",
            s.ffn_gain,
            s.entropy_bits,
            s.margin,
            s.pop_top1_share,
            s.pop_distinct_experts,
            s.sliding_window
        );
    }

    // rank layers by each diagnostic (lowest reorientation = most "rewritten", lowest margin = least confident, etc.)
    let dip_layers = [20usize, 27usize];
    let by_reorientation: Vec<usize> = {
        let mut v: Vec<usize> = reorientation.keys().copied().collect();
        v.sort_by(|a, b| reorientation[a].partial_cmp(&reorientation[b]).unwrap());
        v
    };
    let by_margin: Vec<usize> = {
        let mut v: Vec<usize> = layers.clone();
        v.sort_by(|a, b| stats[a].margin.partial_cmp(&stats[b].margin).unwrap());
        v
    };
    let by_entropy_desc: Vec<usize> = {
        let mut v: Vec<usize> = layers.clone();
        v.sort_by(|a, b| {
            stats[b]
                .entropy_bits
                .partial_cmp(&stats[a].entropy_bits)
                .unwrap()
        });
        v
    };
    let by_mag_ratio_extremity: Vec<usize> = {
        let mut v: Vec<usize> = magnitude_ratio.keys().copied().collect();
        v.sort_by(|a, b| {
            (magnitude_ratio[b] - 1.0)
                .abs()
                .partial_cmp(&(magnitude_ratio[a] - 1.0).abs())
                .unwrap()
        });
        v
    };
    let by_distinct_experts_asc: Vec<usize> = {
        let mut v: Vec<usize> = layers.clone();
        v.sort_by_key(|l| stats[l].pop_distinct_experts);
        v
    };

    println!("\n== does the routing-excess dip set {{{:?}}} appear among the top-5 candidate transition layers, by each INDEPENDENT diagnostic? ==", dip_layers);
    println!(
        "  lowest write-delta reorientation (most 'rewritten'): {:?}",
        &by_reorientation[..5.min(by_reorientation.len())]
    );
    println!(
        "  most extreme write-delta magnitude jump:             {:?}",
        &by_mag_ratio_extremity[..5.min(by_mag_ratio_extremity.len())]
    );
    println!(
        "  lowest router top-8 margin (least confident cutoff): {:?}",
        &by_margin[..5.min(by_margin.len())]
    );
    println!(
        "  highest router entropy (most diffuse routing):       {:?}",
        &by_entropy_desc[..5.min(by_entropy_desc.len())]
    );
    println!(
        "  fewest distinct experts selected (most reused set):  {:?}",
        &by_distinct_experts_asc[..5.min(by_distinct_experts_asc.len())]
    );
    for &d in &dip_layers {
        println!(
            "  L{d}: in reorientation top-5={}, mag-jump top-5={}, low-margin top-5={}, high-entropy top-5={}, low-distinct top-5={}, sliding_window={}",
            by_reorientation[..5.min(by_reorientation.len())].contains(&d),
            by_mag_ratio_extremity[..5.min(by_mag_ratio_extremity.len())].contains(&d),
            by_margin[..5.min(by_margin.len())].contains(&d),
            by_entropy_desc[..5.min(by_entropy_desc.len())].contains(&d),
            by_distinct_experts_asc[..5.min(by_distinct_experts_asc.len())].contains(&d),
            stats[&d].sliding_window,
        );
    }
    println!(
        "\nA credible zone-boundary story needs {{{:?}}} to stand out on at least one of these INDEPENDENT signals — reorientation and magnitude-jump are the most direct \
        tests of 'the intervening layer performs a substantial re-encoding' since they don't reference the router at all.",
        dip_layers
    );
}
